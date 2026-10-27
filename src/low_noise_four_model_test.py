"""低噪声等比例混合增强（sigma=0,.02,.04,.06,.08,.1 各1/6）：POD16 vs FNO vs DeepONet vs CNN。

四模型全部 seed42 随机初始化直接训练，不加载任何旧网络权重。
运行：python src/low_noise_four_model_test.py --run
中断：同命令加 --resume（每轮原子保存完整状态）。
"""
import argparse
import copy
import gc
import hashlib
import html
import json
import math
import random
import sys
import time
from pathlib import Path

import spatial_decoder_clean_test as m
import four_model_benchmark as f

np, torch, nn, F, plt = m.np, m.torch, m.nn, m.F, m.plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'low_noise_four_model_20261022'
NAMES = ('pod16_clean', 'fno2d', 'deeponet', 'cnn')
TRAIN_ORDER = ('pod16_clean', 'deeponet', 'cnn', 'fno2d')
LABELS = {'pod16_clean': '本文：均值分解+POD16', 'fno2d': 'FNO',
          'deeponet': 'DeepONet', 'cnn': 'CNN空间解码器'}
# 作图统一用英文，避免Matplotlib缺中文字体导致图内乱码。
PLOT_LABELS = {'pod16_clean': 'Ours (mean-decomp + POD16)', 'fno2d': 'FNO',
               'deeponet': 'DeepONet', 'cnn': 'CNN decoder'}
H, W, D, BATCH = 40, 240, 9600, 256
SIGMAS = (0., .02, .04, .06, .08, .1)
TEST_SIGMAS = (0., .02, .04, .06, .08, .1, .2, .5)
SEEDS = (31001, 31002, 31003)
SELECT = 'minimum_clean_validation_physical_fullfield_MSE'
CHANNELS = [64, 32, 16, 8, 1]
POSITIONS = [[0, 0], [0, 120], [0, 239], [19, 0], [19, 120], [19, 239],
             [39, 0], [39, 120], [39, 239]]
CONFIG = {
    'pod16_clean': {'encoder': '10->128 SiLU + 3 residual(128->128 SiLU->128)',
                    'head': [128, 16], 'decode': 'sqrt(9600)*basis.T'},
    'fno2d': {'width': 12, 'modes': [6, 12], 'spectral_layers': 4,
              'input_channels': ['bilinear_relative_sensor', 'y', 'x', 'standardized_b'],
              'positions_row_col': POSITIONS, 'lift': [4, 12], 'project': [12, 32, 1],
              'activation': 'GELU', 'padding': 0},
    'deeponet': {'branch': [10, 128, 128, 64], 'trunk': [2, 128, 128, 64],
                 'activation': 'SiLU', 'dot_divisor': 8.0, 'scalar_bias': True,
                 'coords': 'y,x in [0,1]'},
    'cnn': {'encoder': '10->128 SiLU + 3 residual(128->128 SiLU->128)',
            'head': [128, 64 * 5 * 30], 'grid': [5, 30], 'channels': CHANNELS,
            'upsample': [[10, 60], [20, 120], [40, 240]], 'kernel': 3, 'padding': 1,
            'activation': 'SiLU', 'coordinate_channels': False, 'uses_pod': False},
}
LIMITS = [
    '仅seed42单次训练；31001/31002/31003是测试噪声重复，不是训练重复，不构成显著性结论。',
    '旧随机测试子集已多次暴露；8192/1000/2000为固定子集pilot，不是新盲测，也不证明跨工况、跨电池或全量训练泛化。',
    '四模型参数、FLOP和计算时间不同；比较共享数据、输入信息、损失、优化上限与选模规则，不是等参数消融或各模型官方最优实现。',
    'POD基底仅由同8192训练残差拟合，是POD独有的训练拟合先验；随机化SVD不等于完整SVD。',
    '只按干净验证全场物理MSE选模，本身偏向干净输入精度；不按含噪验证或测试指标选权重。',
    'sigma各档按1/6概率抽样，是随机抽样而非每批精确等量；标签始终干净。',
    '合成独立高斯不代表真实传感器误差；0.2/0.5 K为训练范围外的外推压力测试。',
    '训练网络/目标FP32、物理重建与指标FP64，禁用TF32/AMP；CUDA卷积与稀疏算子不承诺跨设备位级复现。',
    'CNN为“10维特征编码+上采样卷积解码”，不是以规则网格温度场为输入的图像CNN；无显式坐标通道。',
    'GPU推断计时含当前b、10特征、插值/坐标、网络及完整物理场，不含主机传输与磁盘IO；吞吐不等于在线单样本延迟。',
    '创新性判断需要独立文献检索与对比，本实验只提供精度证据，不能自行认定新颖性。',
]


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def canonical_hash(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def save_pt(path, obj):
    tmp = path.with_name(path.name + '.tmp')
    torch.save(obj, tmp)
    m.os.replace(tmp, path)


def real_params(net):
    return sum(p.numel() * (2 if p.is_complex() else 1) for p in net.parameters())


class Net(nn.Module):
    """统一封装：features 生成10维特征与当前b，forward 加回 b+μ+scale·残差。"""
    features = m.FieldModel.features
    forward = m.FieldModel.forward

    def __init__(self, kind, mean, scale, norm, basis):
        super().__init__()
        self.kind, self.label = kind, kind
        self.norm = copy.deepcopy(norm)
        for name, value in [('residual_mean', mean), ('residual_scale', scale),
                            ('sensor_scale', norm['residual_sensor_scale']),
                            ('b_mean', norm['b_mean']), ('b_std', norm['b_std'])]:
            self.register_buffer(name, torch.as_tensor(value, dtype=torch.float64).clone())
        if kind in ('pod16_clean', 'cnn'):
            self.encoder = nn.Sequential(nn.Linear(10, 128), nn.SiLU(),
                                         *[m.ResidualBlock() for _ in range(3)])
            self.register_buffer('basis', torch.as_tensor(basis, dtype=torch.float64).clone())
            self.register_buffer('pod_decode32',
                                 (torch.as_tensor(basis.T).float() * math.sqrt(D)).clone())
            if kind == 'pod16_clean':
                self.head = nn.Linear(128, 16)
            else:
                self.head = nn.Linear(128, CHANNELS[0] * 5 * 30)
                self.convs = nn.ModuleList([nn.Conv2d(a, b, 3, padding=1)
                                            for a, b in zip(CHANNELS[:-1], CHANNELS[1:])])
        elif kind == 'fno2d':
            self.register_buffer('interp', f.interpolation_matrix(np.array(POSITIONS)))
            self.register_buffer('coords', f.coordinates().T.reshape(1, 2, H, W))
            self.lift = nn.Conv2d(4, 12, 1)
            self.spectral = nn.ModuleList([f.SpectralConv2d(12, (6, 12)) for _ in range(4)])
            self.local = nn.ModuleList([nn.Conv2d(12, 12, 1) for _ in range(4)])
            self.project = nn.Sequential(nn.Conv2d(12, 32, 1), nn.GELU(), nn.Conv2d(32, 1, 1))
        elif kind == 'deeponet':
            self.branch = nn.Sequential(nn.Linear(10, 128), nn.SiLU(), nn.Linear(128, 128),
                                        nn.SiLU(), nn.Linear(128, 64))
            self.trunk = nn.Sequential(nn.Linear(2, 128), nn.SiLU(), nn.Linear(128, 128),
                                       nn.SiLU(), nn.Linear(128, 64))
            self.bias = nn.Parameter(torch.zeros(()))
            self.register_buffer('coords', f.coordinates())
        else:
            raise ValueError(kind)

    def normalized_residual(self, x):
        if self.kind == 'pod16_clean':
            return self.head(self.encoder(x)) @ self.pod_decode32
        if self.kind == 'cnn':
            z = self.head(self.encoder(x)).reshape(-1, CHANNELS[0], 5, 30)
            for i, size in enumerate([(10, 60), (20, 120), (40, 240)]):
                z = F.silu(self.convs[i](F.interpolate(z, size=size, mode='bilinear',
                                                       align_corners=False)))
            return self.convs[3](z).flatten(1)
        if self.kind == 'deeponet':
            return self.branch(x) @ self.trunk(self.coords).T / 8.0 + self.bias
        field = (x[:, :9] @ self.interp).reshape(-1, 1, H, W)
        absolute_b = x[:, 9, None, None, None].expand(-1, 1, H, W)
        z = self.lift(torch.cat((field, self.coords.expand(len(x), -1, -1, -1), absolute_b), 1))
        for spectral, local in zip(self.spectral, self.local):
            z = F.gelu(spectral(z) + local(z))
        return self.project(z).flatten(1)


def construct(ck):
    kind = ck['kind']
    require(kind in CONFIG, f'未知kind={kind}')
    if 'structure_config' in ck:
        require(ck['structure_config'] == CONFIG[kind], '结构配置不匹配')
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
            return torch.cat([net(ss[i:i+batch]) for i in range(0, len(ss), batch)]
                             ).cpu().numpy().reshape(-1, H, W)
    return predict


def noise_batch(clean, rng):
    sigmas = m.tensor(SIGMAS, torch.float64)
    category = torch.randint(0, len(SIGMAS), (len(clean),), generator=rng, device='cuda')
    noise = torch.randn(clean.shape, generator=rng, device='cuda',
                        dtype=torch.float64) * sigmas[category, None]
    return clean + noise, category, noise


def new_model(kind, data):
    random.seed(42)
    m.seed_all()
    return Net(kind, data['mean'], data['scale'], data['norm'], data['basis']).cuda()


def checkpoint_base(model, data, phash, micro):
    return {'format_version': 3, 'kind': model.kind, 'structure_config': CONFIG[model.kind],
            'state_dict': cpu_state(model), 'residual_mean': data['mean'], 'residual_scale': data['scale'],
            'normalization': data['norm'], 'basis': data['basis'], 'batch_size': BATCH,
            'inference_batch_size': 32 if model.kind == 'fno2d' else BATCH, 'microbatch_size': micro,
            'indices_sha256': data['metadata']['indices_sha256'],
            'training_indices_sha256': data['metadata']['indices_sha256']['train'],
            'script_sha256': m.sha(__file__), 'protocol_hash': phash, 'train_seed': 42,
            'augmentation': {'sigmas_K': list(SIGMAS), 'probability': 1 / len(SIGMAS),
                             'seed': 420022, 'clean_labels': True},
            'no_clean_truth_required_for_inference': True,
            'prediction_formula': 'b_current + residual_mean + residual_scale * normalized_residual'}


def rebuild_best(last, kind):
    keys = ['optimizer', 'scheduler', 'rng', 'noise_rng_state', 'history', 'best_model_state',
            'best_score', 'best_epoch', 'stale', 'elapsed_seconds', 'peak_allocated_bytes']
    best = {k: v for k, v in last.items() if k not in keys}
    best.update(state_dict=last['best_model_state'], epoch=last['best_epoch'],
                validation_clean_MSE_K2=last['best_score'], selection=SELECT)
    path = OUT / f'{kind}_clean_best.pt'
    if path.exists():
        old = torch.load(path, map_location='cpu', weights_only=False)
        same = (old.get('epoch') == best['epoch'] and old.get('protocol_hash') == best['protocol_hash']
                and old.get('kind') == best['kind'] and old['state_dict'].keys() == best['state_dict'].keys()
                and all(torch.equal(old['state_dict'][k], v) for k, v in best['state_dict'].items()))
        if same:
            return
    save_pt(path, best)


def backward_batch(model, noisy, truth, micro):
    total = 0.0
    for i in range(0, len(noisy), micro):
        ss, tt = noisy[i:i + micro], truth[i:i + micro]
        x, b = model.features(ss)
        target = ((tt - b[:, None] - model.residual_mean) / model.residual_scale).float()
        pred = model.normalized_residual(x)
        loss = (pred - target).square().mean()
        require(bool(torch.isfinite(loss)), '训练损失非有限')
        weight = len(ss) / len(noisy)
        (loss * weight).backward()
        total += float(loss.detach()) * weight
    return total


def train(kind, data, phash, resume):
    model = new_model(kind, data)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=.5,
                                                       patience=10, min_lr=1e-6)
    rng = torch.Generator(device='cuda').manual_seed(420022)
    history, best, best_epoch, stale, elapsed, micro, peak = [], math.inf, 0, 0, 0.0, BATCH, 0
    best_state = None
    lastpath = OUT / f'{kind}_last.pt'
    if lastpath.exists():
        require(resume, '已有训练，必须显式--resume')
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
        elapsed, micro, peak = last['elapsed_seconds'], last['microbatch_size'], last['peak_allocated_bytes']
        best_state = last['best_model_state']
        rebuild_best(last, kind)
        m.dump(OUT / f'{kind}_history.json', history)
        del last
        m.log(f'{kind} 从完整epoch={len(history)}恢复，microbatch={micro}')
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
            total, lr_used, count = 0.0, opt.param_groups[0]['lr'], 0
            for i in range(0, 8192, BATCH):
                ix = order[i:i + BATCH]
                noisy, category, noise = noise_batch(data['sensors']['train'][ix], rng)
                audit.update(category.cpu().numpy().tobytes())
                audit.update(noise.cpu().numpy().tobytes())
                opt.zero_grad(set_to_none=True)
                nvidia_oom = False
                try:
                    value = backward_batch(model, noisy, data['fields']['train'][ix], micro)
                except torch.cuda.OutOfMemoryError:
                    if kind != 'fno2d' or micro == 32:
                        raise
                    nvidia_oom = True
                if nvidia_oom:
                    opt.zero_grad(set_to_none=True)
                    gc.collect()
                    torch.cuda.empty_cache()
                    micro = 32
                    m.log('FNO CUDA OOM：改microbatch32累积，逻辑batch仍256；噪声不重抽')
                    value = backward_batch(model, noisy, data['fields']['train'][ix], micro)
                opt.step()
                total += value * len(ix)
                count += len(ix)
            require(count == 8192, '禁止缩减训练样本')
            score = m.validation(model, data, 32 if kind == 'fno2d' else BATCH)
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
            row = {'epoch': epoch, 'train_MSE_K2': total / 8192 * data['scale'] ** 2,
                   'validation_clean_MSE_K2': score, 'lr_used': lr_used,
                   'lr_next': opt.param_groups[0]['lr'], 'stale': stale,
                   'seconds_synchronized': seconds, 'microbatch_size': micro,
                   'train_samples': count, 'optimizer_steps': 32,
                   'augmentation_order_sha256': audit.hexdigest(),
                   'augmentation_rng_end_sha256': m.arrsha(rng.get_state().numpy()),
                   'cuda_peak_allocated_bytes': peak}
            history.append(row)
            state = checkpoint_base(model, data, phash, micro)
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
                rebuild_best(state, kind)
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
            'microbatch_size': micro, 'best_epoch': best_epoch,
            'best_validation_clean_MSE_K2': best, 'best_validation_clean_RMSE_K': math.sqrt(best),
            'train_seconds_synchronized': elapsed,
            'time_scope': '累计每epoch GPU同步包围训练、验证、调度和best-state CPU快照；不含checkpoint文件IO/启动准备/中断停机',
            'cuda_peak_allocated_bytes': peak, 'final_lr': opt.param_groups[0]['lr'],
            'stopping_reason': 'patience40' if stale >= 40 else 'max_epochs300',
            'selection': SELECT, 'initialized_from': 'random seed42',
            'pretrained_network_loaded': False}
    m.dump(OUT / f'{kind}_training_summary.json', info)
    write_csv(OUT / f'{kind}_history.csv', history)
    del model, opt, sched, best_state
    gc.collect()
    torch.cuda.empty_cache()
    return info


def verify_models(data, training, phash):
    result = {}
    for name in NAMES:
        path = OUT / f'{name}_clean_best.pt'
        net, ck = load_model(path)
        hist = read_json(OUT / f'{name}_history.json')
        selected = min(hist, key=lambda r: r['validation_clean_MSE_K2'])
        require(ck['epoch'] == selected['epoch'] == training[name]['best_epoch'], '最佳轮不是clean验证argmin')
        require(ck['validation_clean_MSE_K2'] == selected['validation_clean_MSE_K2'], '最佳分数不符')
        require(ck['protocol_hash'] == phash and ck['indices_sha256'] == data['metadata']['indices_sha256'],
                'checkpoint协议/索引不符')
        score = m.validation(net, data, 32 if name == 'fno2d' else BATCH)
        require(abs(score - ck['validation_clean_MSE_K2']) <= max(1e-12, abs(score) * 1e-9),
                '重载验证分数不符')
        ss = data['sensors']['validation'][:2]
        with torch.inference_mode():
            expected = net(ss).cpu().numpy().reshape(-1, H, W)
        require(np.allclose(load_predictor(path)(ss.cpu().numpy()), expected, rtol=0, atol=1e-10),
                '自足预测器核验失败')
        last = torch.load(OUT / f'{name}_last.pt', map_location='cpu', weights_only=False)
        require(last['epoch'] == len(hist) and last['protocol_hash'] == phash, 'last不完整')
        require(all(k in last for k in ['optimizer', 'scheduler', 'rng', 'noise_rng_state',
                                        'best_model_state']), 'last缺少恢复状态')
        result[name] = {'best_checkpoint_sha256': m.sha(path), 'kind': ck['kind'],
                        'epoch': ck['epoch'], 'reloaded_validation_MSE_K2': score,
                        'standalone_predictor_verified': True, 'last_epoch': last['epoch']}
        del net, ck, last
        gc.collect()
        torch.cuda.empty_cache()
    audits = {name: read_json(OUT / f'{name}_history.json') for name in NAMES}
    for epoch in range(max(len(h) for h in audits.values())):
        rows = [h[epoch] for h in audits.values() if epoch < len(h)]
        require(len({r['augmentation_order_sha256'] for r in rows}) == 1 and
                len({r['augmentation_rng_end_sha256'] for r in rows}) == 1,
                '模型间同epoch噪声/shuffle不一致')
    m.dump(OUT / 'checkpoint_verification.json', result)
    return result


def fixed_noise(ids):
    noises, cases = {}, []
    for sigma in TEST_SIGMAS:
        if sigma == 0.:
            cases.append(('clean', None))
            noises['clean__None'] = np.zeros((2000, 9))
            continue
        for seed in SEEDS:
            scene = f'noise_{sigma:g}'
            cases.append((scene, seed))
            noises[f'{scene}__{seed}'] = np.random.default_rng(seed).normal(size=(2000, 9)) * sigma
    np.savez_compressed(OUT / 'test_fixed_noise.npz', indices=ids, **noises)
    return cases, noises


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
    frozen = {'checkpoints': verification, 'no_test_tuning': True, 'structure_config': CONFIG}
    path = OUT / 'all_models_frozen_before_test.json'
    if path.exists():
        require(read_json(path) == frozen, '冻结记录变化，禁止重选模型')
    else:
        m.dump(path, frozen)
    m.status('testing', checkpoints_frozen=True)
    ids = data['ids']['test_reserved']
    cases, noises = fixed_noise(ids)
    yy = np.load(m.SOURCE / 'ydata.npy', mmap_mode='r', allow_pickle=False)
    sp = np.load(m.SOURCE / 'sp.npy', mmap_mode='r', allow_pickle=False)
    truth = m.tensor(np.asarray(yy[ids], dtype=np.float64).reshape(-1, D), torch.float64)
    sensors = m.tensor(np.asarray(sp[ids], dtype=np.float64), torch.float64)
    rows, aggregates, timings = [], [], {}
    plot = {'truth': truth[0].cpu().numpy().reshape(H, W), 'test_index': np.array(ids[0])}
    columns = ['MSE_K2', 'MAE_K', 'peak_abs_error_K', 'maxpoint_abs_error_K', 'signed_bias_K']
    for name in NAMES:
        net, ck = load_model(OUT / f'{name}_clean_best.pt')
        batch = ck['inference_batch_size']
        sample, csvrows = {}, []
        for scene, seed in cases:
            key = f'{scene}__{seed}'
            noisy = sensors + m.tensor(noises[key], torch.float64)
            collected = []
            with torch.inference_mode():
                for i in range(0, 2000, batch):
                    pred = net(noisy[i:i + batch])
                    require(bool(torch.isfinite(pred).all()), '预测非有限')
                    collected.append(metric_array(pred, truth[i:i + batch]))
                    if i == 0 and scene in ('clean', 'noise_0.06', 'noise_0.1') and seed in (None, 31001):
                        plot[f'{name}_{scene}'] = pred[0].cpu().numpy().reshape(H, W)
            array = np.concatenate(collected)
            require(array.shape == (2000, 5), '测试样本数错误')
            sample[key] = array
            rows.append({'model': name, 'scenario': scene, 'noise_seed': seed, 'n_test': 2000,
                         **aggregate_array(array)})
            for idx, values in zip(ids, array):
                csvrows.append({'model': name, 'scenario': scene, 'noise_seed': seed,
                                'global_index': int(idx),
                                **{col: float(v) for col, v in zip(columns, values)}})
        np.savez_compressed(OUT / f'{name}_per_sample_errors.npz', indices=ids,
                            columns=np.array(columns), **sample)
        write_csv(OUT / f'{name}_per_sample_errors.csv', csvrows)
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
        del net, ck, sample, csvrows
        gc.collect()
        torch.cuda.empty_cache()
    np.savez_compressed(OUT / 'same_sample_plot_data.npz', **plot)
    write_csv(OUT / 'test_metrics_per_seed.csv', rows)
    write_csv(OUT / 'test_metrics_aggregate.csv', aggregates)
    return rows, aggregates, timings, plot


def write_csv(path, rows):
    if not rows:
        return
    import csv
    with Path(path).open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


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


def report(summary, plot):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9})
    figures = []
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for name in NAMES:
        hist = read_json(OUT / f'{name}_history.json')
        ep = [r['epoch'] for r in hist]
        axes[0].plot(ep, [math.sqrt(r['train_MSE_K2']) for r in hist], label=PLOT_LABELS[name])
        axes[1].plot(ep, [math.sqrt(r['validation_clean_MSE_K2']) for r in hist], label=PLOT_LABELS[name])
        axes[2].plot(ep, [r['lr_used'] for r in hist], label=PLOT_LABELS[name])
    for ax, title in zip(axes, ['Augmented train RMSE (K)', 'Clean validation RMSE (K)',
                                'Learning rate used']):
        ax.set(xlabel='Epoch', title=title, yscale='log')
        ax.legend(fontsize=7, prop={'size': 7})
        ax.grid(alpha=.25)
    fig.tight_layout()
    figures.append(('四模型完整训练曲线、干净验证曲线与学习率；仅干净验证MSE参与选模', m.image64(fig)))
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for name in NAMES:
        rr = [(0. if r['scenario'] == 'clean' else float(r['scenario'][6:]),
               r['pooled_global_field_RMSE_K']) for r in summary['main_aggregate']
              if r['model'] == name]
        rr.sort()
        ax.plot([x for x, _ in rr], [y for _, y in rr], marker='o', label=PLOT_LABELS[name])
    ax.axvspan(.0, .1, color='#dfeaf7', alpha=.6, zorder=0)
    ax.set(xlabel='Sensor Gaussian sigma (K)', ylabel='Pooled full-field RMSE (K)',
           title='0-0.1 K is the training range; 0.2/0.5 K are extrapolation')
    ax.grid(alpha=.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    figures.append(('不同sigma的全场误差；浅色区域为训练覆盖范围，右侧为外推压力测试', m.image64(fig)))
    for scene, title in [('clean', 'clean'), ('noise_0.06', 'sigma=0.06 K'),
                         ('noise_0.1', 'sigma=0.10 K')]:
        truth = plot['truth']
        preds = [plot[f'{name}_{scene}'] for name in NAMES]
        vmin = min(float(a.min()) for a in [truth] + preds)
        vmax = max(float(a.max()) for a in [truth] + preds)
        limit = max(1e-8, max(float(np.abs(a - truth).max()) for a in preds))
        fig, axes = plt.subplots(4, 3, figsize=(16, 10))
        for row, name in enumerate(NAMES):
            for col, array in enumerate([truth, preds[row], preds[row] - truth]):
                im = axes[row, col].imshow(array, aspect='auto',
                                           cmap='coolwarm' if col == 2 else 'inferno',
                                           vmin=-limit if col == 2 else vmin,
                                           vmax=limit if col == 2 else vmax)
                axes[row, col].set_title(PLOT_LABELS[name] + [' | truth K', ' | prediction K',
                                                              ' | error K'][col], fontsize=9)
                fig.colorbar(im, ax=axes[row, col], fraction=.025)
        fig.tight_layout()
        figures.append((f'固定test_indices[0]={int(plot["test_index"])}，{title}；'
                        f'噪声图固定seed31001；四模型共享温度与误差色限', m.image64(fig)))
    images = ''.join(f'<figure><img src="data:image/png;base64,{img}" '
                     f'alt="{html.escape(cap)}"><figcaption>{html.escape(cap)}</figcaption></figure>'
                     for cap, img in figures)
    labels = {'model': '模型', 'scenario': '情形', 'noise_repetitions': '噪声重复',
              'pooled_global_field_RMSE_K': '全场RMSE/K', 'MAE_K': '全场MAE/K',
              'peak_MAE_K': '峰温MAE/K', 'maxpoint_K': '最大点绝对误差/K',
              'noise_seed_RMSE_std_K': '重复RMSE标准差/K', 'rank': '排名', 'RMSE_K': '全场RMSE/K',
              'parameters': '实标量参数', 'epochs': '实际轮数', 'best_epoch': '最佳轮',
              'best_validation_clean_RMSE_K': '干净验证RMSE/K',
              'train_seconds_synchronized': '同步训练秒',
              'training_range_equal_scenario_RMSE_K': '0.02-0.10等权RMSE/K',
              'sample_RMSE_p95_K': '单样本RMSE P95/K', 'signed_bias_K': '平均偏差/K'}

    def table(rows, cols, mapping=None):
        text = m.table(rows, cols)
        for key in cols:
            text = text.replace('<th>' + html.escape(key) + '</th>',
                                '<th>' + labels.get(key, key) + '</th>')
        if mapping:
            for key, value in mapping.items():
                text = text.replace('<td>' + html.escape(key) + '</td>',
                                    '<td>' + html.escape(value) + '</td>')
        return text

    mapping = {name: LABELS[name] for name in NAMES}
    main = table(summary['main_aggregate'],
                 ['model', 'scenario', 'noise_repetitions', 'pooled_global_field_RMSE_K',
                  'MAE_K', 'peak_MAE_K', 'maxpoint_K', 'noise_seed_RMSE_std_K'], mapping)
    ranking = table(summary['scenario_ranking'], ['scenario', 'rank', 'model', 'RMSE_K'], mapping)
    overall = table(summary['overall_ranking'], ['rank', 'model',
                                                 'training_range_equal_scenario_RMSE_K'], mapping)
    trainrows = [{'model': name, **info} for name, info in summary['training'].items()]
    train_table = table(trainrows, ['model', 'parameters', 'epochs', 'best_epoch',
                                    'best_validation_clean_RMSE_K', 'microbatch_size',
                                    'train_seconds_synchronized'], mapping)
    timingrows = [{'model': name, 'batch_size': v['inference_batch_size'],
                   'single_median_us': v['1']['median_us_per_sample'],
                   'batch2000_seconds': v['2000']['median_seconds'],
                   'batch2000_us_per_sample': v['2000']['median_us_per_sample']}
                  for name, v in summary['timings'].items()]
    timing_table = m.table(timingrows, ['model', 'batch_size', 'single_median_us',
                                        'batch2000_seconds', 'batch2000_us_per_sample'])
    for name in NAMES:
        timing_table = timing_table.replace('<td>' + name + '</td>',
                                            '<td>' + html.escape(LABELS[name]) + '</td>')
    index = {(r['model'], r['scenario']): r for r in summary['main_aggregate']}
    lines = []
    for sigma in TEST_SIGMAS:
        scene = 'clean' if sigma == 0 else f'noise_{sigma:g}'
        best = min(NAMES, key=lambda n: index[(n, scene)]['pooled_global_field_RMSE_K'])
        lines.append(f'<tr><td>{sigma:g}</td>' + ''.join(
            f'<td{" style=\'font-weight:600\'" if n == best else ""}>'
            f'{index[(n, scene)]["pooled_global_field_RMSE_K"]:.6f}</td>' for n in NAMES) +
            f'<td>{html.escape(LABELS[best])}</td></tr>')
    head = ''.join(f'<th>{html.escape(LABELS[n])}</th>' for n in NAMES)
    sigma_table = f'<div class="scroll"><table><tr><th>σ/K</th>{head}<th>该σ最优</th></tr>{"".join(lines)}</table></div>'
    limits = ''.join('<li>' + html.escape(s) + '</li>' for s in LIMITS)
    style = ('body{font:15px/1.7 system-ui,"Microsoft YaHei",sans-serif;background:#f4f7fb;'
             'color:#20334b;max-width:1480px;margin:25px auto;padding:0 20px}'
             'section,figure{background:white;border:1px solid #dce3ec;border-radius:10px;'
             'padding:20px;margin:20px 0}table{border-collapse:collapse;width:100%;font-size:12px}'
             'td,th{border:1px solid #dce3ec;padding:8px;white-space:nowrap;text-align:right}'
             'th{background:#eaf2fb}.scroll{overflow:auto}img{width:100%;height:auto}'
             'pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}')
    raw = html.escape(json.dumps(summary, ensure_ascii=False, indent=2))
    protocol_text = html.escape(json.dumps(summary['protocol'], ensure_ascii=False, indent=2))
    text = f'''<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>低噪声等比例混合增强：POD16 vs FNO vs DeepONet vs CNN</title><style>{style}</style></head><body>
<h1>σ∈[0,0.10] 等比例混合增强的四模型精度对照</h1>
<p>真实 CUDA · 全部随机初始化直接训练 · 同 8192/1000/2000 · 同 10 维输入信息 · 同噪声序列 · 干净验证选模</p>
<section><h2>1. 不同 σ 下的全场 RMSE</h2>{sigma_table}
<p>σ=0 为干净输入，其余每个 σ 用 31001/31002/31003 三个独立噪声种子、每例 2000 样本，pooled RMSE=sqrt(全部样本与种子场MSE均值)。0.20/0.50 K 在训练范围之外，仅作外推压力测试。</p>
<p>训练覆盖范围内（0.02–0.10 K）等权描述性排名：</p>{overall}</section>
<section><h2>2. 完整指标（含峰温与最大点误差）</h2>{main}
<p>峰温MAE=mean(|max预测−max真值|)，不是热点位置误差；maxpoint 为全部样本与种子的最大格点绝对误差；平均偏差为带符号全场均值误差。</p></section>
<section><h2>3. 各 σ 排名</h2>{ranking}</section>
<section><h2>4. 训练协议与结构</h2>
<p>训练噪声为 σ=0/0.02/0.04/0.06/0.08/0.10 K 六档等概率（各 1/6），每个训练样本按其九点读数逐通道叠加独立高斯噪声，标签始终为干净温度场。独立 CUDA Generator seed420022 连续跨轮，逻辑 batch 256；每轮 CPU randperm 种子 4200000+epoch。四模型逐轮噪声与顺序哈希已交叉核对一致。</p>
<p>四模型共享均值分解：b=九点均值，10 维特征为九点相对温差除固定训练尺度，加绝对 b 的固定训练标准化；输出统一为 b_current+μ+scale·标准化残差。训练目标为 (干净场−含噪b−μ)/scale，绝不使用干净 b 重建含噪输入。</p>
<p>POD16：10→128 + 三残差块→16→固定训练基底；FNO：九点相对温差双线性映射 40×240，拼 y、x 与标准化 b 共四通道，width12、modes(6,12)、四谱层；DeepONet：branch 10→128→128→64、trunk(y,x)→128→128→64，点积/8；CNN：同 POD 的 128 维编码器，10→128 + 三残差块→64×5×30→双线性上采样 10×60/20×120/40×240 + 3×3 卷积 64→32→16→8→1，前向不使用 POD 基底，也无显式坐标通道。</p>
<p>统一预算：最多 300 轮、连续 40 轮干净验证无改善即停、AdamW lr1e−3 wd1e−4、ReduceLROnPlateau factor0.5 patience10 min1e−6、batch256；FNO 仅 OOM 时降 microbatch32 并累积，逻辑批量与样本数不变。结构在任何测试前冻结。</p>
<details><summary>完整冻结协议与哈希</summary><pre>{protocol_text}</pre></details></section>
<section><h2>5. 参数、训练与推断成本</h2>{train_table}
<p>复参数按两个实标量计，固定基底/均值/坐标不计参数。训练时间前后 GPU 同步，含训练、验证、调度与最佳权重 CPU 快照，不含文件 IO。</p>{timing_table}
<p>推断为同 GPU 实测完整物理重建路径，前后同步，不含主机传输与磁盘 IO。</p></section>
<section><h2>6. 曲线、σ 扫描与固定首样本场图</h2>{images}</section>
<section><h2>7. 必须保留的限制</h2><ul>{limits}</ul>
<p>所有 best checkpoint 自足包含预处理、结构配置、state_dict 与索引哈希；每轮 last 原子保存优化器、调度器、Python/NumPy/Torch/CUDA 与独立噪声 RNG、完整 history 与最佳权重，支持真正断点恢复。逐样本误差、完整 history、summary、status 与测试扰动均已保留，HTML 图片与 JSON 全部内嵌。</p></section>
<details><summary>完整结果 JSON</summary><pre>{raw}</pre></details></body></html>'''
    (OUT / 'result_summary.html').write_text(text, encoding='utf-8')


def protocol(data, phash):
    return {'script_sha256': m.sha(__file__), 'prepare_module_sha256': m.sha(m.__file__),
            'architecture': CONFIG, 'architecture_frozen_before_test': True,
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
    keep = ['created_local', 'python', 'python_version', 'torch', 'cuda', 'gpu',
            'gpu_total_bytes', 'platform', 'device_asserted', 'no_dependencies_installed',
            'indices_sha256', 'source_files', 'source_metadata_sha256', 'source_summary_sha256',
            'basis_source', 'basis_file_sha256', 'basis_array_sha256', 'residual_mean_sha256',
            'basis_matches_old_selfcontained_checkpoint', 'basis_train_indices_equal',
            'basis_orthogonality_max_abs', 'recomputed_train_mean_max_abs_K', 'normalization',
            'residual_scale_K', 'residual_scale_fit', 'precision']
    data['metadata'] = {k: data['metadata'][k] for k in keep}
    derived = canonical_hash({'config': CONFIG, 'sigmas': SIGMAS,
                              'indices': data['metadata']['indices_sha256'],
                              'script': m.sha(__file__), 'prepare': m.sha(m.__file__)})
    phash = derived
    settings = protocol(data, phash)
    if OUT.exists():
        require(resume, '结果目录已存在；拒绝覆盖，请用--resume')
        saved = read_json(OUT / 'protocol.json')
        old = saved['protocol']
        # 只允许脚本修复；核心训练协议必须逐项一致，且沿用原protocol_hash以续用已有checkpoint。
        for key in ['architecture', 'indices_sha256', 'batch_size', 'train_n', 'validation_n',
                    'test_n', 'train_seed', 'augmentation', 'shuffle', 'optimizer', 'selection',
                    'test_sigmas_K']:
            require(old[key] == settings[key], f'恢复时核心协议变化：{key}')
        phash = saved['protocol_hash']
        if old['script_sha256'] != settings['script_sha256']:
            saved.setdefault('script_updates', []).append({
                'time_local': time.strftime('%Y-%m-%d %H:%M:%S'),
                'from': old['script_sha256'], 'to': settings['script_sha256'],
                'reason': '修复CSV写出函数的AttributeError；噪声档位、结构、预算、选模与测试协议均未改变',
            })
            settings['script_sha256'] = saved['protocol']['script_sha256'] = m.sha(__file__)
            saved['protocol']['protocol_hash'] = phash
            m.dump(OUT / 'protocol.json', saved)
        settings['protocol_hash'] = phash
        if (OUT / 'summary.json').exists() and read_json(OUT / 'summary.json').get('completed'):
            print('实验已完成且协议一致；不重复训练或测试。', flush=True)
            return
    else:
        require(not resume, '--resume要求结果目录已存在')
        OUT.mkdir()
        m.dump(OUT / 'protocol.json', {'protocol_hash': phash, 'protocol': settings})
        np.savez(OUT / 'subset_indices.npz', **data['ids'])
        np.savez(OUT / 'shared_preprocessing.npz', basis=data['basis'],
                 residual_mean=data['mean'], residual_scale=data['scale'],
                 train_indices=data['ids']['train'])
        (OUT / 'run.log').write_text('', encoding='utf-8')
    require(phash == settings['protocol_hash'], 'protocol_hash不一致')
    m.OUT, m.START = OUT, time.perf_counter()
    data['metadata'].update(protocol_hash=phash, script_sha256=m.sha(__file__))
    m.dump(OUT / 'metadata.json', data['metadata'])
    m.status('preparing', protocol_hash=phash)
    training_path = OUT / 'training_summary.json'
    training = read_json(training_path) if training_path.exists() else {}
    for kind in TRAIN_ORDER:
        training[kind] = train(kind, data, phash, resume)
        m.dump(training_path, training)
    verification = verify_models(data, training, phash)
    rows, aggregate, timings, plot = evaluate(data, verification)
    scenario_rank, overall_rank = rankings(aggregate)
    summary = {'completed': True, 'protocol_hash': phash, 'protocol': settings,
               'metadata': data['metadata'], 'training': training,
               'checkpoint_verification': verification, 'main_aggregate': aggregate,
               'per_seed_metrics': rows, 'timings': timings,
               'scenario_ranking': scenario_rank, 'overall_ranking': overall_rank,
               'limitations': LIMITS,
               'elapsed_seconds_before_report': time.perf_counter() - m.START}
    m.status('reporting', checkpoints_frozen=True)
    report(summary, plot)
    summary['elapsed_seconds_total'] = time.perf_counter() - m.START
    m.dump(OUT / 'summary.json', summary)
    m.dump(OUT / 'artifact_manifest.json',
           {p.relative_to(OUT).as_posix(): {'size': p.stat().st_size, 'sha256': m.sha(p)}
            for p in OUT.rglob('*') if p.is_file()
            and p.name not in ['artifact_manifest.json', 'run.log', 'status.json']})
    m.status('completed', completed=True, checkpoints_verified=True,
             report_path=str(OUT / 'result_summary.html'),
             summary_sha256=m.sha(OUT / 'summary.json'),
             elapsed_seconds_total=summary['elapsed_seconds_total'])
    print(json.dumps({'completed': True, 'training': training, 'main_aggregate': aggregate},
                     ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    require(args.run or args.resume, '需--run或--resume')
    run(args.resume)
