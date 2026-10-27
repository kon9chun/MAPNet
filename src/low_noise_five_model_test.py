"""低噪声等比例混合增强（sigma=0,.02,.04,.06,.08,.1 各1/6）：四模型基础上加入纯MLP。

纯MLP（10->128->128->128->9600，SiLU）本轮 seed42 随机初始化直接训练；
POD16 / FNO / DeepONet / CNN 直接复用 low_noise_four_model_20261022 中同协议、
已核验的 clean_best 权重，不重复训练。测试扰动直接读取该实验保存的固定噪声，
保证五模型面对的输入逐位相同。本脚本不生成HTML报告。

运行：python src/low_noise_five_model_test.py --run
中断：同命令加 --resume（仅MLP有逐轮完整状态）。
"""
import argparse
import copy
import gc
import hashlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import spatial_decoder_clean_test as m
import four_model_benchmark as f
import low_noise_four_model_test as base

ROOT = Path(__file__).resolve().parent
SRC4 = ROOT / 'low_noise_four_model_20261022'
OUT = ROOT / 'low_noise_five_model_20261022'
NAMES = ('pod16_clean', 'fno2d', 'deeponet', 'cnn', 'mlp')
REUSED = ('pod16_clean', 'fno2d', 'deeponet', 'cnn')
NEW = ('mlp',)
LABELS = {'pod16_clean': '本文：均值分解+POD16', 'fno2d': 'FNO',
          'deeponet': 'DeepONet', 'cnn': 'CNN空间解码器', 'mlp': '纯MLP（直接输出9600维）'}
H, W, D, BATCH = 40, 240, 9600, 256
SIGMAS = (0., .02, .04, .06, .08, .1)
TEST_SIGMAS = (0., .02, .04, .06, .08, .1, .2, .5)
SEEDS = (31001, 31002, 31003)
SELECT = 'minimum_clean_validation_physical_fullfield_MSE'
CONFIG = copy.deepcopy(base.CONFIG)
CONFIG['mlp'] = {'encoder': [10, 128], 'hidden_layers': 3, 'hidden_width': 128,
                 'output': 9600, 'activation': 'SiLU', 'decode': 'direct_full_field',
                 'uses_pod': False, 'coordinate_channels': False,
                 'residual_basis_source': 'shared train residual mean/scale'}
LIMITS = [
    '仅seed42单次训练；31001/31002/31003是测试噪声重复，不是训练重复，不构成显著性结论。',
    '旧随机测试子集已被多次使用；8192/1000/2000为固定子集pilot，不是新盲测，不证明跨工况或全量训练泛化。',
    '五模型参数量、FLOP与训练时间不同；比较共享数据、输入信息、损失、优化上限与选模规则，不是等参数消融或各自官方最优实现。',
    'POD基底仅由同8192训练残差拟合，是POD独有的训练拟合先验；随机化SVD不等于完整SVD。',
    '只按干净验证全场物理MSE选模，本身偏向干净输入精度；不按含噪验证或测试指标选权重。',
    'sigma各档按1/6概率抽样，是随机抽样而非每批精确等量；标签始终干净。',
    '合成独立高斯不代表真实传感器误差；0.2/0.5 K为训练范围外的外推压力测试。',
    '训练网络/目标FP32、物理重建与指标FP64，禁用TF32/AMP；CUDA算子不承诺跨设备位级复现。',
    'CNN为“10维特征编码+上采样卷积解码”，不是以规则网格温度场为输入的图像CNN。',
    'POD16/FNO/DeepONet/CNN四个基线权重来自同协议的前一轮四模型实验，本轮未重训；MLP为本轮新训练。',
    'GPU推断计时含当前b、10特征、插值/坐标、网络及完整物理场，不含主机传输与磁盘IO。',
    '创新性判断需要独立文献检索与对比，本实验只提供精度证据，不能自行认定新颖性。',
]


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def canonical_hash(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def save_pt(path, obj):
    tmp = Path(path).with_name(Path(path).name + '.tmp')
    torch.save(obj, tmp)
    m.os.replace(tmp, path)


def cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def real_params(net):
    return sum(p.numel() * (2 if p.is_complex() else 1) for p in net.parameters())


class Net(base.Net):
    """在前四类结构基础上增加纯MLP：10->128->128->128->9600，无POD解码。"""

    def __init__(self, kind, mean, scale, norm, basis):
        if kind != 'mlp':
            super().__init__(kind, mean, scale, norm, basis)
            return
        nn.Module.__init__(self)
        self.kind = self.label = kind
        self.norm = copy.deepcopy(norm)
        for name, value in [('residual_mean', mean), ('residual_scale', scale),
                            ('sensor_scale', norm['residual_sensor_scale']),
                            ('b_mean', norm['b_mean']), ('b_std', norm['b_std'])]:
            self.register_buffer(name, torch.as_tensor(value, dtype=torch.float64).clone())
        self.layers = nn.Sequential(nn.Linear(10, 128), nn.SiLU(),
                                    nn.Linear(128, 128), nn.SiLU(),
                                    nn.Linear(128, 128), nn.SiLU(),
                                    nn.Linear(128, D))

    def normalized_residual(self, x):
        if self.kind == 'mlp':
            return self.layers(x)
        return super().normalized_residual(x)


def construct(ck):
    kind = ck['kind']
    require(kind in CONFIG, f'未知kind={kind}')
    if 'structure_config' in ck:
        require(ck['structure_config'] == CONFIG[kind], f'{kind}结构配置不匹配')
    return Net(kind, ck['residual_mean'], ck['residual_scale'], ck['normalization'], ck['basis'])


def load_model(path):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    net = construct(ck).cuda()
    net.load_state_dict(ck['state_dict'], strict=True)
    net.eval()
    return net, ck


def load_predictor(path):
    net, ck = load_model(path)
    batch = ck['inference_batch_size']

    def predict(sensors):
        array = np.asarray(sensors, dtype=np.float64)
        require(array.ndim == 2 and array.shape[1] == 9 and np.isfinite(array).all(), '输入应为(N,9)')
        with torch.inference_mode():
            ss = m.tensor(array, torch.float64)
            return torch.cat([net(ss[i:i + batch]) for i in range(0, len(ss), batch)]
                             ).cpu().numpy().reshape(-1, H, W)
    return predict


def best_path(name):
    return (OUT if name in NEW else SRC4) / f'{name}_clean_best.pt'


def noise_batch(clean, rng):
    sigmas = m.tensor(SIGMAS, torch.float64)
    category = torch.randint(0, len(SIGMAS), (len(clean),), generator=rng, device='cuda')
    noise = torch.randn(clean.shape, generator=rng, device='cuda',
                        dtype=torch.float64) * sigmas[category, None]
    return clean + noise, category, noise


def checkpoint_base(model, data, phash):
    return {'format_version': 3, 'kind': model.kind, 'structure_config': CONFIG[model.kind],
            'state_dict': cpu_state(model), 'residual_mean': data['mean'],
            'residual_scale': data['scale'], 'normalization': data['norm'],
            'basis': data['basis'], 'batch_size': BATCH, 'inference_batch_size': BATCH,
            'microbatch_size': BATCH, 'indices_sha256': data['metadata']['indices_sha256'],
            'training_indices_sha256': data['metadata']['indices_sha256']['train'],
            'script_sha256': m.sha(__file__), 'protocol_hash': phash, 'train_seed': 42,
            'augmentation': {'sigmas_K': list(SIGMAS), 'probability': 1 / len(SIGMAS),
                             'seed': 420022, 'clean_labels': True},
            'no_clean_truth_required_for_inference': True,
            'prediction_formula': 'b_current + residual_mean + residual_scale * normalized_residual'}


def save_best(state, kind):
    best = {k: v for k, v in state.items()
            if k not in ['optimizer', 'scheduler', 'rng', 'noise_rng_state', 'history',
                         'best_model_state', 'best_score', 'best_epoch', 'stale',
                         'elapsed_seconds', 'peak_allocated_bytes']}
    best.update(state_dict=state['best_model_state'], epoch=state['best_epoch'],
                validation_clean_MSE_K2=state['best_score'], selection=SELECT)
    save_pt(OUT / f'{kind}_clean_best.pt', best)


def backward_batch(model, noisy, truth):
    x, b = model.features(noisy)
    target = ((truth - b[:, None] - model.residual_mean) / model.residual_scale).float()
    pred = model.normalized_residual(x)
    loss = (pred - target).square().mean()
    require(bool(torch.isfinite(loss)), '训练损失非有限')
    loss.backward()
    return float(loss.detach())


def train(kind, data, phash, resume):
    require(kind in NEW, '仅新模型训练')
    random.seed(42)
    m.seed_all()
    model = Net(kind, data['mean'], data['scale'], data['norm'], data['basis']).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=.5,
                                                       patience=10, min_lr=1e-6)
    rng = torch.Generator(device='cuda').manual_seed(420022)
    history, best, best_epoch, stale, elapsed, peak = [], math.inf, 0, 0, 0.0, 0
    best_state = None
    lastpath = OUT / f'{kind}_last.pt'
    if lastpath.exists():
        require(resume, '已有训练状态，必须显式--resume')
        last = torch.load(lastpath, map_location='cpu', weights_only=False)
        require(last['protocol_hash'] == phash and last['kind'] == kind
                and last['structure_config'] == CONFIG[kind], '恢复协议/结构不匹配')
        model.load_state_dict(last['state_dict'])
        opt.load_state_dict(last['optimizer'])
        sched.load_state_dict(last['scheduler'])
        rng.set_state(last['noise_rng_state'])
        torch.set_rng_state(last['rng']['torch'])
        torch.cuda.set_rng_state_all(last['rng']['cuda'])
        np.random.set_state(last['rng']['numpy'])
        random.setstate(last['rng']['python'])
        history, best, best_epoch, stale = last['history'], last['best_score'], last['best_epoch'], last['stale']
        require(last['epoch'] == len(history), '恢复history/epoch不一致')
        elapsed, peak, best_state = last['elapsed_seconds'], last['peak_allocated_bytes'], last['best_model_state']
        m.dump(OUT / f'{kind}_history.json', history)
        del last
        m.log(f'{kind} 从完整epoch={len(history)}恢复')
    m.sync()
    torch.cuda.reset_peak_memory_stats()
    if stale < 40:
        for epoch in range(len(history) + 1, 301):
            m.status('training', model=kind, epoch=epoch, best_epoch=best_epoch)
            m.sync()
            start = time.perf_counter()
            order = torch.randperm(8192, generator=torch.Generator().manual_seed(4200000 + epoch)).cuda()
            audit = hashlib.sha256(order.cpu().numpy().tobytes())
            model.train()
            total, lr_used = 0.0, opt.param_groups[0]['lr']
            for i in range(0, 8192, BATCH):
                ix = order[i:i + BATCH]
                noisy, category, noise = noise_batch(data['sensors']['train'][ix], rng)
                audit.update(category.cpu().numpy().tobytes())
                audit.update(noise.cpu().numpy().tobytes())
                opt.zero_grad(set_to_none=True)
                total += backward_batch(model, noisy, data['fields']['train'][ix]) * len(ix)
                opt.step()
            score = m.validation(model, data, BATCH)
            require(math.isfinite(score), '干净验证分数非有限')
            sched.step(score)
            improved = score < best
            if improved:
                best, best_epoch, stale = score, epoch, 0
                best_state = cpu_state(model)
            else:
                stale += 1
            m.sync()
            seconds = time.perf_counter() - start
            elapsed += seconds
            peak = max(peak, torch.cuda.max_memory_allocated())
            history.append({'epoch': epoch, 'train_MSE_K2': total / 8192 * data['scale'] ** 2,
                            'validation_clean_MSE_K2': score, 'lr_used': lr_used,
                            'lr_next': opt.param_groups[0]['lr'], 'stale': stale,
                            'seconds_synchronized': seconds, 'microbatch_size': BATCH,
                            'train_samples': 8192, 'optimizer_steps': 32,
                            'augmentation_order_sha256': audit.hexdigest(),
                            'augmentation_rng_end_sha256': m.arrsha(rng.get_state().numpy()),
                            'cuda_peak_allocated_bytes': peak})
            state = checkpoint_base(model, data, phash)
            state.update(epoch=epoch, validation_clean_MSE_K2=score,
                         selection='last_epoch_not_for_test', optimizer=opt.state_dict(),
                         scheduler=sched.state_dict(), noise_rng_state=rng.get_state(),
                         rng={'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all(),
                              'numpy': np.random.get_state(), 'python': random.getstate()},
                         history=history, best_score=best, best_epoch=best_epoch, stale=stale,
                         best_model_state=best_state, elapsed_seconds=elapsed,
                         peak_allocated_bytes=peak)
            m.sync()
            save_pt(lastpath, state)
            if improved:
                save_best(state, kind)
            m.dump(OUT / f'{kind}_history.json', history)
            del state
            if epoch == 1 or epoch % 10 == 0 or stale >= 40:
                m.log(f'{kind} epoch={epoch} valRMSE={math.sqrt(score):.8f}K '
                      f'best={best_epoch} stale={stale} lr={lr_used:g}')
            if stale >= 40:
                break
    require(bool(history) and best_state is not None, '缺少训练历史或最佳权重')
    info = {'parameters': real_params(model),
            'parameter_tensor_elements': sum(p.numel() for p in model.parameters()),
            'parameters_convention': 'complex64按两个实标量，固定缓冲不计参数',
            'epochs': len(history), 'optimizer_steps': len(history) * 32, 'batch_size': BATCH,
            'microbatch_size': BATCH, 'best_epoch': best_epoch,
            'best_validation_clean_MSE_K2': best, 'best_validation_clean_RMSE_K': math.sqrt(best),
            'train_seconds_synchronized': elapsed,
            'time_scope': '累计每epoch GPU同步包围训练、验证、调度和best-state CPU快照；不含checkpoint文件IO/启动准备/中断停机',
            'cuda_peak_allocated_bytes': peak, 'final_lr': opt.param_groups[0]['lr'],
            'stopping_reason': 'patience40' if stale >= 40 else 'max_epochs300',
            'selection': SELECT, 'initialized_from': 'random seed42',
            'pretrained_network_loaded': False}
    m.dump(OUT / f'{kind}_training_summary.json', info)
    base.write_csv(OUT / f'{kind}_history.csv', history)
    del model, opt, sched, best_state
    gc.collect()
    torch.cuda.empty_cache()
    return info


def verify_models(data, training, phash):
    old_protocol = read_json(SRC4 / 'protocol.json')
    old_phash = old_protocol['protocol_hash']
    old_training = read_json(SRC4 / 'training_summary.json')
    result = {}
    for name in NAMES:
        path = best_path(name)
        require(path.exists(), f'缺少权重{path}')
        net, ck = load_model(path)
        source_dir = OUT if name in NEW else SRC4
        hist = read_json(source_dir / f'{name}_history.json')
        selected = min(hist, key=lambda r: r['validation_clean_MSE_K2'])
        expect = training[name]['best_epoch'] if name in NEW else old_training[name]['best_epoch']
        require(ck['epoch'] == selected['epoch'] == expect, f'{name}最佳轮不是clean验证argmin')
        require(ck['validation_clean_MSE_K2'] == selected['validation_clean_MSE_K2'], f'{name}最佳分数不符')
        require(ck['protocol_hash'] == (phash if name in NEW else old_phash), f'{name}协议哈希不符')
        require(ck['kind'] == name and ck['structure_config'] == CONFIG[name], f'{name}结构不符')
        require(ck['indices_sha256'] == data['metadata']['indices_sha256'], f'{name}索引哈希不符')
        score = m.validation(net, data, ck['inference_batch_size'])
        require(abs(score - ck['validation_clean_MSE_K2']) <= max(1e-12, abs(score) * 1e-9),
                f'{name}重载验证分数不符')
        ss = data['sensors']['validation'][:2]
        with torch.inference_mode():
            expected = net(ss).cpu().numpy().reshape(-1, H, W)
        require(np.allclose(load_predictor(path)(ss.cpu().numpy()), expected, rtol=0, atol=1e-10),
                f'{name}自足预测器核验失败')
        last = torch.load(source_dir / f'{name}_last.pt', map_location='cpu', weights_only=False)
        require(last['epoch'] == len(hist), f'{name} last不完整')
        require(all(k in last for k in ['optimizer', 'scheduler', 'rng', 'noise_rng_state',
                                        'best_model_state']), f'{name} last缺少恢复状态')
        result[name] = {'best_checkpoint': str(path), 'best_checkpoint_sha256': m.sha(path),
                        'trained_this_run': name in NEW,
                        'protocol_hash': ck['protocol_hash'], 'kind': ck['kind'],
                        'epoch': ck['epoch'], 'reloaded_validation_MSE_K2': score,
                        'standalone_predictor_verified': True, 'last_epoch': last['epoch']}
        del net, ck, last
        gc.collect()
        torch.cuda.empty_cache()
    audits = {name: read_json((OUT if name in NEW else SRC4) / f'{name}_history.json') for name in NAMES}
    overlap = min(len(h) for h in audits.values())
    for epoch in range(overlap):
        rows = [h[epoch] for h in audits.values()]
        require(len({r['augmentation_order_sha256'] for r in rows}) == 1 and
                len({r['augmentation_rng_end_sha256'] for r in rows}) == 1,
                '模型间同epoch噪声/shuffle不一致')
    m.dump(OUT / 'checkpoint_verification.json', result)
    m.dump(OUT / 'noise_sequence_overlap_check.json',
           {'compared_epochs': overlap, 'identical_across_models': True,
            'reused_models': list(REUSED), 'new_models': list(NEW),
            'note': 'MLP与四个复用基线在重叠epoch上逐轮噪声类别、噪声张量与shuffle哈希完全一致'})
    return result


def metric_array(pred, truth):
    err = pred - truth
    return torch.stack((err.square().mean(1), err.abs().mean(1),
                        (pred.max(1).values - truth.max(1).values).abs(),
                        err.abs().max(1).values, err.mean(1)), 1).cpu().numpy()


def aggregate_array(array):
    mse, mae, peak, maximum, bias = array.T
    return {'MSE_K2': float(mse.mean()),
            'pooled_global_field_RMSE_K': math.sqrt(float(mse.mean())),
            'MAE_K': float(mae.mean()), 'peak_MAE_K': float(peak.mean()),
            'maxpoint_K': float(maximum.max()), 'mean_sample_RMSE_K': float(np.sqrt(mse).mean()),
            'signed_bias_K': float(bias.mean()),
            'sample_RMSE_p95_K': float(np.percentile(np.sqrt(mse), 95))}


def evaluate(data, verification):
    frozen = {'checkpoints': verification, 'no_test_tuning': True, 'structure_config': CONFIG,
              'reused_from': str(SRC4)}
    path = OUT / 'all_models_frozen_before_test.json'
    if path.exists():
        require(read_json(path) == frozen, '冻结记录变化，禁止重选模型')
    else:
        m.dump(path, frozen)
    m.status('testing', checkpoints_frozen=True)
    ids = data['ids']['test_reserved']
    with np.load(SRC4 / 'test_fixed_noise.npz', allow_pickle=False) as z:
        require(np.array_equal(z['indices'], ids), '测试索引与该实验不一致')
        noises = {k: z[k] for k in z.files if k != 'indices'}
    cases = []
    for sigma in TEST_SIGMAS:
        if sigma == 0.:
            cases.append(('clean', None))
        else:
            for seed in SEEDS:
                cases.append((f'noise_{sigma:g}', seed))
    regenerated = all(np.array_equal(noises[f'noise_{s:g}__{sd}'],
                                     np.random.default_rng(sd).normal(size=(2000, 9)) * s)
                      for s in TEST_SIGMAS[1:] for sd in SEEDS)
    m.dump(OUT / 'test_noise_audit.json',
           {'source': str(SRC4 / 'test_fixed_noise.npz'), 'cases': len(cases),
            'regenerated_noise_matches_stored': bool(regenerated),
            'stored_noise_used_for_all_models': True})
    yy = np.load(m.SOURCE / 'ydata.npy', mmap_mode='r', allow_pickle=False)
    sp = np.load(m.SOURCE / 'sp.npy', mmap_mode='r', allow_pickle=False)
    truth = m.tensor(np.asarray(yy[ids], dtype=np.float64).reshape(-1, D), torch.float64)
    sensors = m.tensor(np.asarray(sp[ids], dtype=np.float64), torch.float64)
    rows, aggregates, timings = [], [], {}
    columns = ['MSE_K2', 'MAE_K', 'peak_abs_error_K', 'maxpoint_abs_error_K', 'signed_bias_K']
    for name in NAMES:
        net, ck = load_model(best_path(name))
        batch = ck['inference_batch_size']
        sample = {}
        for scene, seed in cases:
            key = f'{scene}__{seed}'
            noisy = sensors + m.tensor(noises[key], torch.float64)
            collected = []
            with torch.inference_mode():
                for i in range(0, 2000, batch):
                    pred = net(noisy[i:i + batch])
                    require(bool(torch.isfinite(pred).all()), f'{name}预测非有限')
                    collected.append(metric_array(pred, truth[i:i + batch]))
            array = np.concatenate(collected)
            require(array.shape == (2000, 5), '测试样本数错误')
            sample[key] = array
            rows.append({'model': name, 'scenario': scene, 'noise_seed': seed, 'n_test': 2000,
                         **aggregate_array(array)})
        np.savez_compressed(OUT / f'{name}_per_sample_errors.npz', indices=ids,
                            columns=np.array(columns), **sample)
        for scene in dict.fromkeys(scene for scene, _ in cases):
            keys = [f'{ss}__{sd}' for ss, sd in cases if ss == scene]
            rr = [r for r in rows if r['model'] == name and r['scenario'] == scene]
            aggregates.append({'model': name, 'scenario': scene, 'n_unique_test': 2000,
                               'noise_repetitions': len(keys),
                               **aggregate_array(np.concatenate([sample[k] for k in keys])),
                               'noise_seed_RMSE_std_K': float(np.std(
                                   [r['pooled_global_field_RMSE_K'] for r in rr], ddof=1))
                               if len(rr) > 1 else 0.0})
        timings[name] = {**m.timing(net, sensors, batch), 'inference_batch_size': batch}
        m.dump(OUT / 'test_metrics_per_seed.json', rows)
        m.dump(OUT / 'inference_timings.json', timings)
        m.log(f'{name} 完成{len(cases)}个测试情形，每例2000样本')
        del net, ck, sample
        gc.collect()
        torch.cuda.empty_cache()
    base.write_csv(OUT / 'test_metrics_per_seed.csv', rows)
    base.write_csv(OUT / 'test_metrics_aggregate.csv', aggregates)
    return rows, aggregates, timings


def rankings(aggregate):
    index = {(r['model'], r['scenario']): r for r in aggregate}
    scenes = list(dict.fromkeys(r['scenario'] for r in aggregate))
    table = []
    for scene in scenes:
        ordered = sorted(NAMES, key=lambda name: index[(name, scene)]['pooled_global_field_RMSE_K'])
        for rank, name in enumerate(ordered, 1):
            table.append({'scenario': scene, 'rank': rank, 'model': name,
                          'RMSE_K': index[(name, scene)]['pooled_global_field_RMSE_K']})
    trained = [f'noise_{s:g}' for s in TEST_SIGMAS[1:] if s <= .1]
    overall = []
    for name in NAMES:
        mean_mse = float(np.mean([index[(name, s)]['MSE_K2'] for s in trained]))
        overall.append({'model': name, 'training_range_equal_scenario_RMSE_K': math.sqrt(mean_mse),
                        'scope': 'sigma0.02-0.10四种情形等权MSE再开根，描述性汇总，不参与选模'})
    overall.sort(key=lambda r: r['training_range_equal_scenario_RMSE_K'])
    for rank, row in enumerate(overall, 1):
        row['rank'] = rank
    return table, overall


def protocol(data, phash):
    return {'script_sha256': m.sha(__file__), 'prepare_module_sha256': m.sha(m.__file__),
            'architecture': CONFIG, 'architecture_frozen_before_test': True,
            'reused_checkpoint_source': str(SRC4),
            'reused_checkpoint_protocol_hash': read_json(SRC4 / 'protocol.json')['protocol_hash'],
            'indices_sha256': data['metadata']['indices_sha256'], 'batch_size': BATCH,
            'train_n': 8192, 'validation_n': 1000, 'test_n': 2000, 'train_seed': 42,
            'augmentation': {'sigmas_K': list(SIGMAS), 'probability': 1 / len(SIGMAS),
                             'seed': 420022, 'clean_labels': True,
                             'distribution': 'per-sample per-channel independent Gaussian'},
            'shuffle': 'CPU randperm8192 seed4200000+epoch, then CUDA',
            'optimizer': 'AdamW lr1e-3 wd1e-4, max300, patience40, plateau .5/10/1e-6',
            'selection': SELECT, 'test_sigmas_K': list(TEST_SIGMAS),
            'test_noise_seeds': list(SEEDS), 'no_test_tuning': True,
            'previously_exposed_test': True, 'protocol_hash': phash}


def run(resume):
    require(torch.cuda.is_available(), '必须使用CUDA，禁止CPU替代')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(False)
    m.seed_all()
    data = m.prepare()
    del data['target'], data['x']
    derived = canonical_hash({'config': CONFIG, 'sigmas': SIGMAS,
                              'indices': data['metadata']['indices_sha256'],
                              'script': m.sha(__file__), 'prepare': m.sha(m.__file__)})
    if OUT.exists():
        require(resume, '结果目录已存在；拒绝覆盖，请用--resume')
        saved = read_json(OUT / 'protocol.json')
        require(saved['protocol_hash'] == derived, '脚本或配置变化，拒绝恢复')
        if (OUT / 'summary.json').exists() and read_json(OUT / 'summary.json').get('completed'):
            print('实验已完成；不重复训练或测试。', flush=True)
            return
    else:
        require(not resume, '--resume要求结果目录已存在')
        OUT.mkdir()
        (OUT / 'run.log').write_text('', encoding='utf-8')
        phash = derived
        m.OUT, m.START = OUT, time.perf_counter()
        m.dump(OUT / 'protocol.json', {'protocol_hash': phash, 'protocol': protocol(data, phash)})
        np.savez(OUT / 'subset_indices.npz', **data['ids'])
        np.savez(OUT / 'shared_preprocessing.npz', basis=data['basis'],
                 residual_mean=data['mean'], residual_scale=data['scale'],
                 train_indices=data['ids']['train'])
    phash = derived
    m.OUT, m.START = OUT, time.perf_counter()
    m.log(f'五模型实验开始；复用{list(REUSED)}，新训练{list(NEW)}')
    m.dump(OUT / 'metadata.json', data['metadata'])
    m.status('preparing', protocol_hash=phash)
    training_path = OUT / 'training_summary.json'
    training = read_json(training_path) if training_path.exists() else {}
    for name in REUSED:
        training.setdefault(name, {'reused_from': str(SRC4),
                                   **read_json(SRC4 / 'training_summary.json')[name]})
    m.dump(training_path, training)
    for kind in NEW:
        training[kind] = train(kind, data, phash, resume)
        m.dump(training_path, training)
    verification = verify_models(data, training, phash)
    rows, aggregate, timings = evaluate(data, verification)
    scenario_rank, overall_rank = rankings(aggregate)
    summary = {'completed': True, 'protocol_hash': phash, 'protocol': protocol(data, phash),
               'metadata': data['metadata'], 'training': training,
               'checkpoint_verification': verification, 'main_aggregate': aggregate,
               'per_seed_metrics': rows, 'timings': timings,
               'scenario_ranking': scenario_rank, 'overall_ranking': overall_rank,
               'limitations': LIMITS,
               'elapsed_seconds_total': time.perf_counter() - m.START}
    m.status('reporting', checkpoints_frozen=True)
    m.dump(OUT / 'summary.json', summary)
    m.dump(OUT / 'artifact_manifest.json',
           {p.relative_to(OUT).as_posix(): {'size': p.stat().st_size, 'sha256': m.sha(p)}
            for p in OUT.rglob('*') if p.is_file()
            and p.name not in ['artifact_manifest.json', 'run.log', 'status.json']})
    m.status('completed', completed=True, checkpoints_verified=True,
             summary_sha256=m.sha(OUT / 'summary.json'),
             elapsed_seconds_total=summary['elapsed_seconds_total'])
    index = {(r['model'], r['scenario']): r for r in aggregate}
    print('\n全场RMSE(K)')
    header = 'sigma  ' + '  '.join(f'{n:>12}' for n in NAMES)
    print(header)
    for sigma in TEST_SIGMAS:
        scene = 'clean' if sigma == 0 else f'noise_{sigma:g}'
        print(f'{sigma:<6g}' + '  '.join(f'{index[(n, scene)]["pooled_global_field_RMSE_K"]:12.6f}'
                                         for n in NAMES))
    print('\n0.02-0.10等权:', json.dumps(overall_rank, ensure_ascii=False))
    print('\n训练:', json.dumps(training, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    require(args.run or args.resume, '需--run或--resume')
    run(args.resume)
