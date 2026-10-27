"""仅干净训练：均值分解 ResMLP + POD16 与可学习卷积空间解码器的配对 CUDA 实验。"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from pathlib import Path
import argparse
import base64
import gc
import hashlib
import html
import io
import json
import math
import platform
import shutil
import sys
import time
import traceback
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import paths

ROOT = Path(__file__).resolve().parent
OLD = ROOT / 'mean_decomposition_20261021'
OUT = ROOT / 'spatial_decoder_clean_20261022'
SOURCE = paths.DATA
H, W, D, RANK = 40, 240, 9600, 16
NAMES = ['pod16_clean', 'spatial_clean']
SEEDS = [31001, 31002, 31003]
START = time.perf_counter()


def dump(path, obj):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(tmp, path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def arrsha(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def log(text):
    line = f'[{time.perf_counter()-START:.2f}s] {text}'
    print(line, flush=True)
    with (OUT / 'run.log').open('a', encoding='utf-8') as f:
        f.write(line + '\n')


def status(stage, **kwargs):
    dump(OUT / 'status.json', {'stage': stage, 'time_local': time.strftime('%Y-%m-%d %H:%M:%S'), **kwargs})


def sync():
    torch.cuda.synchronize()


def tensor(x, dtype=torch.float32):
    return torch.as_tensor(x, dtype=dtype, device='cuda')


def seed_all():
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)


class ResidualBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 128))

    def forward(self, x):
        return F.silu(x + self.net(x))


class FieldModel(nn.Module):
    def __init__(self, kind, channels, basis, mean, scale, norm):
        super().__init__()
        self.kind, self.channels, self.norm = kind, channels, norm
        self.encoder = nn.Sequential(nn.Linear(10, 128), nn.SiLU(), *[ResidualBlock() for _ in range(3)])
        self.register_buffer('basis', torch.as_tensor(basis, dtype=torch.float64).clone())
        self.register_buffer('residual_mean', torch.as_tensor(mean, dtype=torch.float64).clone())
        self.register_buffer('residual_scale', torch.tensor(scale, dtype=torch.float64))
        self.register_buffer('sensor_scale', torch.tensor(norm['residual_sensor_scale'], dtype=torch.float64))
        self.register_buffer('b_mean', torch.tensor(norm['b_mean'], dtype=torch.float64))
        self.register_buffer('b_std', torch.tensor(norm['b_std'], dtype=torch.float64))
        self.register_buffer('pod_decode32', (torch.as_tensor(basis.T).float() * math.sqrt(D)).clone())
        if kind == 'pod16_clean':
            self.head = nn.Linear(128, RANK)
        else:
            self.head = nn.Linear(128, channels[0] * 5 * 30)
            self.convs = nn.ModuleList([nn.Conv2d(a, b, 3, padding=1) for a, b in zip(channels[:-1], channels[1:])])

    def features(self, sensor):
        sensor = sensor.double()
        b = sensor.mean(1)
        x = torch.cat([(sensor-b[:, None])/self.sensor_scale, ((b-self.b_mean)/self.b_std)[:, None]], 1)
        return x.float(), b

    def normalized_residual(self, x):
        z = self.head(self.encoder(x))
        if self.kind == 'pod16_clean':
            return z @ self.pod_decode32
        z = z.reshape(-1, self.channels[0], 5, 30)
        for i, size in enumerate([(10, 60), (20, 120), (40, 240)]):
            z = F.silu(self.convs[i](F.interpolate(z, size=size, mode='bilinear', align_corners=False)))
        return self.convs[3](z).flatten(1)

    def forward(self, sensor):
        x, b = self.features(sensor)
        # 推断完整路径：当前读数均值 + 训练残差均值 + 固定训练尺度乘解码输出。
        return b[:, None] + self.residual_mean + self.residual_scale * self.normalized_residual(x).double()


def load_predictor(checkpoint_path):
    assert torch.cuda.is_available(), '禁止退回CPU'
    ck = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model = FieldModel(ck['kind'], ck['channels'], ck['basis'], ck['residual_mean'], ck['residual_scale'], ck['normalization']).cuda()
    model.load_state_dict(ck['state_dict'])
    model.eval()

    def predict(sensor):
        with torch.inference_mode():
            s = tensor(np.asarray(sensor), torch.float64)
            return torch.cat([model(s[i:i+ck['batch_size']]) for i in range(0, len(s), ck['batch_size'])]).cpu().numpy().reshape(-1, H, W)
    return predict


def new_model(kind, channels, data):
    seed_all()
    return FieldModel(kind, channels, data['basis'], data['mean'], data['scale'], data['norm']).cuda()


def prepare():
    assert torch.cuda.is_available(), '本实验必须CUDA，禁止CPU替代'
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # CUDA双线性插值反向非确定性；固定种子、顺序和共享初始化，不声称位级可复现。
    torch.use_deterministic_algorithms(False)
    oldmeta = json.loads((OLD / 'metadata.json').read_text(encoding='utf-8'))
    oldsummary = json.loads((OLD / 'summary.json').read_text(encoding='utf-8'))
    assert oldsummary['completed'] and oldsummary['metadata']['indices_sha256'] == oldmeta['indices_sha256']
    with np.load(OLD / 'subset_indices.npz', allow_pickle=False) as z:
        ids = {k: z[k].copy() for k in ['train', 'validation', 'test_reserved']}
    for k, n in [('train', 8192), ('validation', 1000), ('test_reserved', 2000)]:
        assert len(ids[k]) == n and len(np.unique(ids[k])) == n
        assert arrsha(ids[k]) == oldmeta['indices_sha256'][k]
    assert not set(ids['train']) & set(ids['validation'])
    assert not set(ids['train']) & set(ids['test_reserved'])
    assert not set(ids['validation']) & set(ids['test_reserved'])
    with np.load(OLD / 'mean_pod16.npz', allow_pickle=False) as p:
        assert np.array_equal(ids['train'], p['train_indices'])
        basis, mean = p['basis'].copy(), p['mean'].copy()
    oldck = torch.load(OLD / 'mean_clean_clean_best.pt', map_location='cpu', weights_only=False)
    assert oldck['training_indices_sha256'] == arrsha(ids['train'])
    assert np.array_equal(basis, oldck['basis']) and np.array_equal(mean, oldck['field_mean'])
    sourceinfo = {}
    for f in ['ydata.npy', 'sp.npy']:
        path = SOURCE / f
        expected = next(v for k, v in oldmeta['source_files'].items() if k.replace('\\', '/').endswith('/'+f))
        actual = sha(path)
        assert actual == expected['sha256']
        sourceinfo[str(path)] = {'sha256': actual, 'size': path.stat().st_size}
    y = np.load(SOURCE / 'ydata.npy', mmap_mode='r', allow_pickle=False)
    s = np.load(SOURCE / 'sp.npy', mmap_mode='r', allow_pickle=False)
    assert y.shape == (93340, H, W) and s.shape == (93340, 9)
    sensors, fields, target, x = {}, {}, {}, {}
    # 测试真值在两个checkpoint固定后才装载；预基准仅用训练数据。
    for k in ['train', 'validation']:
        sensors[k] = tensor(np.asarray(s[ids[k]], dtype=np.float64), torch.float64)
        fields[k] = tensor(np.asarray(y[ids[k]], dtype=np.float64).reshape(-1, D), torch.float64)
    st, ft = sensors['train'], fields['train']
    bt = st.mean(1)
    norm = dict(oldmeta['normalization'])
    fitted = {'residual_sensor_scale': float((st-bt[:, None]).square().mean().sqrt()),
              'b_mean': float(bt.mean()), 'b_std': float(bt.std(correction=0))}
    for k, v in fitted.items():
        assert abs(v-norm[k]) < 1e-10
    recomputed_mean = (ft-bt[:, None]).mean(0)
    mean_error = float((recomputed_mean-tensor(mean, torch.float64)).abs().max())
    assert mean_error < 1e-10
    centered = ft-bt[:, None]-recomputed_mean
    scale = float(centered.square().mean().sqrt())
    assert scale > 0
    for k in sensors:
        b = sensors[k].mean(1)
        target[k] = ((fields[k]-b[:, None]-recomputed_mean)/scale).float()
        x[k] = torch.cat([(sensors[k]-b[:, None])/norm['residual_sensor_scale'], ((b-norm['b_mean'])/norm['b_std'])[:, None]], 1).float()
    orth = float(np.max(np.abs(basis.T@basis-np.eye(RANK))))
    assert orth < 1e-10
    metadata = {'created_local': time.strftime('%Y-%m-%d %H:%M:%S'), 'script_sha256': sha(__file__),
        'python': sys.executable, 'python_version': sys.version, 'torch': torch.__version__, 'cuda': torch.version.cuda,
        'gpu': torch.cuda.get_device_name(), 'gpu_total_bytes': torch.cuda.get_device_properties(0).total_memory,
        'platform': platform.platform(), 'device_asserted': 'cuda', 'no_dependencies_installed': True,
        'indices_sha256': {k: arrsha(v) for k, v in ids.items()}, 'source_files': sourceinfo,
        'source_metadata_sha256': sha(OLD/'metadata.json'), 'source_summary_sha256': sha(OLD/'summary.json'),
        'basis_source': str(OLD/'mean_pod16.npz'), 'basis_file_sha256': sha(OLD/'mean_pod16.npz'),
        'basis_array_sha256': arrsha(basis), 'residual_mean_sha256': arrsha(mean),
        'basis_matches_old_selfcontained_checkpoint': True, 'basis_train_indices_equal': True,
        'basis_orthogonality_max_abs': orth, 'recomputed_train_mean_max_abs_K': mean_error,
        'normalization': norm, 'residual_scale_K': scale, 'residual_scale_fit': 'train8192 centered residual RMS; same scalar both models',
        'train_seed': 42, 'training_repetitions': 1, 'test_noise_seeds': SEEDS,
        'train_noise_augmentation': False, 'dropout': False, 'coordinate_channels': False,
        'coordinate_explanation': '无显式坐标输入；MLP展开为5x30网格隐含固定位置，卷积含零填充；不是FNO，也没有传感器场栅格输入。',
        'spatial_no_pod_truth_inputs': True, 'encoder': '10->128 SiLU + 3*(128->128 SiLU->128 residual+SiLU)',
        'input_features': '9个(sp-b)/固定训练共同尺度，加(b-训练b均值)/训练b标准差；b来自当前九点读数',
        'output_formula': 'b_current + shared_train_residual_mean + shared_train_residual_scale * normalized_decoder_output',
        'pod_output': '16维非白化输出@ (sqrt(9600)*basis.T)，随后乘共同residual_scale',
        'loss': '两模型显式完整场normalized residual MSE，统一乘scale²等于物理K²；没有逐模态白化',
        'selection': '仅干净validation1000物理单位全场MSE最低；两模型从头重训',
        'optimizer': {'name': 'AdamW', 'lr': .001, 'weight_decay': .0001, 'max_epochs': 300, 'patience': 40,
            'scheduler': {'name': 'ReduceLROnPlateau', 'factor': .5, 'patience': 10, 'min_lr': 1e-6}},
        'precision': '网络与训练target float32；物理输出/验证/测试指标float64；TF32禁用，无AMP',
        'determinism': '固定seed、共享encoder初始化、相同每轮shuffle；cuDNN deterministic；bilinear CUDA backward可能非确定，不承诺位级复现',
        'prior_test_exposure': True, 'new_blind_test': False, 'no_test_tuning': True,
        'paired_main_comparison_only_new_models': True, 'random_split_not_cross_condition': True,
        'benchmark_policy': '先channels64,32,16,8,1；若训练单步>0.15秒仅按速度减为32,16,8,4,1；batch256 OOM才两模型统一128；不按测试或验证效果选结构'}
    return {'ids': ids, 'basis': basis, 'mean': mean, 'scale': scale, 'norm': norm,
            'sensors': sensors, 'fields': fields, 'target': target, 'x': x, 'metadata': metadata}


def probe(kind, channels, batch, data):
    model = new_model(kind, channels, data)
    opt = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    xx, yy = data['x']['train'][:batch], data['target']['train'][:batch]
    torch.cuda.reset_peak_memory_stats()
    times = []
    for step in range(7):
        sync(); t = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        out = model.normalized_residual(xx)
        loss = (out-yy).square().mean()
        loss.backward(); opt.step()
        sync()
        if step >= 2:
            times.append(time.perf_counter()-t)
    result = {'kind': kind, 'channels': channels, 'batch': batch, 'median_train_step_seconds': float(np.median(times)),
              'step_seconds': times, 'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'parameters': sum(p.numel() for p in model.parameters()), 'finite_loss': bool(torch.isfinite(loss))}
    assert result['finite_loss']
    del model, opt, out, loss
    gc.collect(); torch.cuda.empty_cache()
    log(f'GPU预基准 {result}')
    return result


def benchmark(data):
    probes, batch, channels = [], 256, [64, 32, 16, 8, 1]
    try:
        p = probe('spatial_clean', channels, batch, data)
    except torch.cuda.OutOfMemoryError:
        gc.collect(); torch.cuda.empty_cache()
        batch = 128
        p = probe('spatial_clean', channels, batch, data)
    probes.append(p)
    if p['median_train_step_seconds'] > .15:
        channels = [32, 16, 8, 4, 1]
        probes.append(probe('spatial_clean', channels, batch, data))
    probes.append(probe('pod16_clean', channels, batch, data))
    result = {'completed': True, 'batch_size': batch, 'channels': channels, 'probes': probes,
              'architecture_frozen_before_training_and_test': True, 'selection_reason': '仅预先声明的CUDA时间/显存规则',
              'script_sha256': sha(__file__)}
    dump(OUT/'benchmark.json', result)
    return result


def sanity(data, bench):
    result = {'coordinate_channels': False, 'fullfield_training_both_models': True, 'loss_equivalence': {}, 'gradient': {}}
    models = [new_model(n, bench['channels'], data) for n in NAMES]
    assert all(torch.equal(a, b) for a, b in zip(models[0].encoder.parameters(), models[1].encoder.parameters()))
    result['initial_encoder_exact_equal'] = True
    scale = data['scale']
    for model in models:
        xx, yy = data['x']['train'][:4], data['target']['train'][:4]
        zz = model.normalized_residual(xx)
        physical_residual = data['fields']['train'][:4]-data['sensors']['train'][:4].mean(1)[:, None]-model.residual_mean
        normalized_loss = (zz-yy).square().mean()
        physical_loss = (zz.double()*scale-physical_residual).square().mean()
        delta = float((normalized_loss.double()*scale**2-physical_loss).abs())
        gn = torch.autograd.grad(normalized_loss*scale**2, zz, retain_graph=True)[0]
        gp = torch.autograd.grad(physical_loss, zz, retain_graph=True)[0]
        grad_abs = float((gn-gp).abs().max())
        assert delta < 1e-6 and torch.allclose(gn, gp, rtol=2e-5, atol=1e-8)
        normalized_loss.backward()
        grads = [p.grad for p in model.parameters()]
        assert all(g is not None and bool(torch.isfinite(g).all()) for g in grads)
        gradnorm = float(torch.stack([g.square().sum() for g in grads]).sum().sqrt())
        assert gradnorm > 0
        result['loss_equivalence'][model.kind] = {'normalized_times_scale2_vs_physical_abs_K2': delta, 'gradient_max_abs': grad_abs,
                                                'physical_loss_K2': float(physical_loss.detach())}
        result['gradient'][model.kind] = {'all_trainable_parameters_have_finite_gradient': True, 'gradient_L2_norm': gradnorm}
    # POD额外核验：完整场损失 = 系数平方误差 + 不可表示的投影残差，梯度相同。
    basis = tensor(data['basis'], torch.float64)
    target = (data['fields']['train'][:8]-data['sensors']['train'][:8].mean(1)[:, None]-tensor(data['mean'], torch.float64))/scale
    a = target@basis/math.sqrt(D)
    residual = target-a@(basis.T*math.sqrt(D))
    c = a.detach().clone().add(.01).requires_grad_(True)
    full = (c@(basis.T*math.sqrt(D))-target).square().mean()
    exact = (c-a).square().sum(1).mean()+residual.square().mean()
    g1 = torch.autograd.grad(full, c, retain_graph=True)[0]
    g2 = torch.autograd.grad(exact, c)[0]
    assert abs(float(full-exact)) < 1e-10 and float((g1-g2).abs().max()) < 1e-10
    result['pod_projection_equivalence'] = {'loss_abs_normalized': abs(float(full-exact)), 'gradient_max_abs': float((g1-g2).abs().max()),
        'projection_residual_MSE_normalized': float(residual.square().mean()), 'projection_residual_included': True}
    del models
    gc.collect(); torch.cuda.empty_cache()
    dump(OUT/'sanity_checks.json', result)
    log(f'损失与梯度门禁通过：{result}')
    return result


def checkpoint(path, model, data, bench, epoch, score, selection, opt=None, sched=None):
    ck = {'format_version': 1, 'kind': model.kind, 'channels': bench['channels'], 'batch_size': bench['batch_size'],
          'state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()},
          'basis': data['basis'], 'residual_mean': data['mean'], 'residual_scale': data['scale'], 'normalization': data['norm'],
          'epoch': epoch, 'validation_clean_MSE_K2': score, 'selection': selection, 'train_seed': 42,
          'training_indices_sha256': data['metadata']['indices_sha256']['train'], 'script_sha256': sha(__file__),
          'no_clean_truth_required_for_inference': True, 'prediction_formula': data['metadata']['output_formula']}
    if opt is not None:
        ck.update(optimizer=opt.state_dict(), scheduler=sched.state_dict(), torch_rng_state=torch.get_rng_state(),
                  cuda_rng_states=torch.cuda.get_rng_state_all())
    tmp = path.with_suffix('.pt.tmp')
    torch.save(ck, tmp); os.replace(tmp, path)


def validation(model, data, batch):
    total = torch.zeros((), device='cuda', dtype=torch.float64)
    with torch.inference_mode():
        for i in range(0, 1000, batch):
            pred = model(data['sensors']['validation'][i:i+batch])
            total += (pred-data['fields']['validation'][i:i+batch]).square().mean(1).sum()
    return float(total/1000)


def train(kind, data, bench):
    model = new_model(kind, bench['channels'], data)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=.5, patience=10, min_lr=1e-6)
    batch, best, stale, best_epoch = bench['batch_size'], math.inf, 0, 0
    history = []
    sync(); begin = time.perf_counter(); torch.cuda.reset_peak_memory_stats()
    for epoch in range(1, 301):
        status('training', model=kind, epoch=epoch, best_epoch=best_epoch)
        sync(); et = time.perf_counter()
        order = torch.randperm(8192, generator=torch.Generator().manual_seed(42*100000+epoch)).cuda()
        model.train(); losses = torch.zeros((), device='cuda')
        lr_used = opt.param_groups[0]['lr']
        for i in range(0, 8192, batch):
            ix = order[i:i+batch]
            opt.zero_grad(set_to_none=True)
            prediction = model.normalized_residual(data['x']['train'][ix])
            loss = (prediction-data['target']['train'][ix]).square().mean()
            loss.backward(); opt.step()
            losses += loss.detach()*len(ix)
        assert bool(torch.isfinite(losses)), '训练损失非有限'
        model.eval(); score = validation(model, data, batch)
        assert math.isfinite(score)
        sched.step(score)
        if score < best:
            best, best_epoch, stale = score, epoch, 0
            checkpoint(OUT/f'{kind}_clean_best.pt', model, data, bench, epoch, score, 'minimum_clean_validation_physical_fullfield_MSE')
        else:
            stale += 1
        sync()
        row = {'epoch': epoch, 'train_MSE_K2': float(losses/8192)*data['scale']**2,
               'validation_clean_MSE_K2': score, 'lr_used': lr_used, 'lr_next': opt.param_groups[0]['lr'],
               'seconds_synchronized': time.perf_counter()-et, 'stale': stale,
               'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated()}
        history.append(row); dump(OUT/f'{kind}_history.json', history)
        if epoch == 1 or epoch % 10 == 0 or stale >= 40:
            log(f'{kind} epoch={epoch} trainRMSE={math.sqrt(row["train_MSE_K2"]):.9f} valRMSE={math.sqrt(score):.9f} bestEpoch={best_epoch} lr={lr_used:g}->{row["lr_next"]:g} stale={stale}')
        if stale >= 40:
            break
    sync()
    checkpoint(OUT/f'{kind}_last.pt', model, data, bench, epoch, score, 'last_epoch_not_for_test', opt, sched)
    info = {'parameters': sum(p.numel() for p in model.parameters()), 'encoder_parameters': sum(p.numel() for p in model.encoder.parameters()),
            'epochs': epoch, 'optimizer_steps': epoch*math.ceil(8192/batch), 'batch_size': batch, 'best_epoch': best_epoch,
            'best_validation_clean_MSE_K2': best, 'best_validation_clean_RMSE_K': math.sqrt(best),
            'final_lr': opt.param_groups[0]['lr'], 'lr_schedule': [{'epoch': h['epoch'], 'lr_used': h['lr_used'], 'lr_next': h['lr_next']} for h in history],
            'train_seconds_synchronized': time.perf_counter()-begin, 'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated(),
            'stopping_reason': 'patience40' if stale >= 40 else 'max_epochs300', 'selection': 'minimum_clean_validation_physical_fullfield_MSE'}
    del model, opt, sched
    gc.collect(); torch.cuda.empty_cache()
    log(f'{kind} 完成：epochs={epoch} best={best_epoch} validationRMSE={math.sqrt(best):.9f} seconds={info["train_seconds_synchronized"]:.3f}')
    return info


def load_model(kind):
    ck = torch.load(OUT/f'{kind}_clean_best.pt', map_location='cpu', weights_only=False)
    model = FieldModel(kind, ck['channels'], ck['basis'], ck['residual_mean'], ck['residual_scale'], ck['normalization']).cuda()
    model.load_state_dict(ck['state_dict']); model.eval()
    return model, ck


def timing(model, sensors, batch):
    result = {}
    for size, repetitions, warmups in [(2000, 7, 5), (1, 30, 10)]:
        ss = sensors[:size]
        with torch.inference_mode():
            for _ in range(warmups):
                for i in range(0, size, batch):
                    model(ss[i:i+batch])
            times = []
            for _ in range(repetitions):
                sync(); t = time.perf_counter()
                for i in range(0, size, batch):
                    model(ss[i:i+batch])
                sync(); times.append(time.perf_counter()-t)
        result[str(size)] = {'repetitions': repetitions, 'seconds_synchronized': times,
                            'median_seconds': float(np.median(times)), 'median_us_per_sample': float(np.median(times)/size*1e6)}
    result['scope'] = 'GPU-resident九点float64读数->10feature->encoder->POD/卷积decode->共同mean/scale与当前b加回->完整float64场；含输出计算，不含CPU传输/文件IO'
    return result


def test(data, bench):
    ids = data['ids']['test_reserved']
    y = np.load(SOURCE/'ydata.npy', mmap_mode='r', allow_pickle=False)
    s = np.load(SOURCE/'sp.npy', mmap_mode='r', allow_pickle=False)
    truth = tensor(np.asarray(y[ids], dtype=np.float64).reshape(-1, D), torch.float64)
    sensors = tensor(np.asarray(s[ids], dtype=np.float64), torch.float64)
    noises = {'clean__None': np.zeros((2000, 9))}
    cases = [('clean', None)]
    for sigma in [.05, .1, .2, .5, 1.]:
        for seed in SEEDS:
            scene = f'noise_{sigma}'
            cases.append((scene, seed)); noises[f'{scene}__{seed}'] = np.random.default_rng(seed).normal(size=(2000, 9))*sigma
    for scene in ['common_bias_0.2', 'ch0_bias_0.5']:
        noise = np.full((2000, 9), .2) if scene.startswith('common') else np.zeros((2000, 9))
        if scene.startswith('ch0'):
            noise[:, 0] = .5
        cases.append((scene, None)); noises[f'{scene}__None'] = noise
    np.savez_compressed(OUT/'test_fixed_noise.npz', indices=ids, **noises)
    oldnoise = np.load(OLD/'test_fixed_noise.npz', allow_pickle=False)
    paired_old = all(np.array_equal(noises[f'{scene}__{seed}'], oldnoise[scene if seed is None else f'{scene}__{seed}']) for scene, seed in cases)
    assert paired_old
    rows, sample, examples, timings = [], {}, {}, {}
    batch = bench['batch_size']
    for kind in NAMES:
        model, ck = load_model(kind)
        assert ck['selection'] == 'minimum_clean_validation_physical_fullfield_MSE'
        examples[kind], sample[kind] = {}, {}
        for scene, seed in cases:
            key = f'{scene}__{seed}'
            noisy = sensors+tensor(noises[key], torch.float64)
            mse, mae, peak, maxerr = [], [], [], 0.
            with torch.inference_mode():
                for i in range(0, 2000, batch):
                    pred = model(noisy[i:i+batch]); tt = truth[i:i+batch]
                    assert bool(torch.isfinite(pred).all())
                    err = pred-tt
                    mse.append(err.square().mean(1).cpu().numpy())
                    mae.append(err.abs().mean(1).cpu().numpy())
                    peak.append((pred.max(1).values-tt.max(1).values).abs().cpu().numpy())
                    maxerr = max(maxerr, float(err.abs().max()))
                    if i == 0 and (scene == 'clean' or (scene == 'noise_0.2' and seed == 31001)):
                        examples[kind][scene] = pred[0].cpu().numpy().reshape(H, W)
            mm = np.concatenate(mse); sample[kind][key] = mm
            rows.append({'model': kind, 'scenario': scene, 'noise_seed': seed, 'n_test': 2000,
                         'MSE_K2': float(mm.mean()), 'global_field_RMSE_K': float(np.sqrt(mm.mean())),
                         'MAE_K': float(np.concatenate(mae).mean()), 'peak_MAE_K': float(np.concatenate(peak).mean()),
                         'maxpoint_K': maxerr, 'mean_sample_RMSE_K': float(np.sqrt(mm).mean())})
        np.savez_compressed(OUT/f'{kind}_per_sample_mse.npz', **sample[kind])
        timings[kind] = timing(model, sensors, batch)
        log(f'{kind} 全部2000测试完成 cleanRMSE={rows[-18]["global_field_RMSE_K"]:.9f} K')
        dump(OUT/'test_metrics_per_seed.json', rows)
        del model; gc.collect(); torch.cuda.empty_cache()
    aggregated, paired = [], []
    for scene in dict.fromkeys(s for s, _ in cases):
        for kind in NAMES:
            rr = [r for r in rows if r['scenario'] == scene and r['model'] == kind]
            rmses = [r['global_field_RMSE_K'] for r in rr]
            aggregated.append({'model': kind, 'scenario': scene, 'noise_repetitions': len(rr),
                'pooled_global_field_RMSE_K': math.sqrt(float(np.mean([r['MSE_K2'] for r in rr]))),
                'noise_seed_RMSE_std_K': float(np.std(rmses, ddof=1)) if len(rr)>1 else 0.,
                'MAE_K': float(np.mean([r['MAE_K'] for r in rr])), 'peak_MAE_K': float(np.mean([r['peak_MAE_K'] for r in rr])),
                'maxpoint_K': max(r['maxpoint_K'] for r in rr),
                'mean_sample_RMSE_K': float(np.mean([r['mean_sample_RMSE_K'] for r in rr]))})
        keys = [f'{ss}__{seed}' for ss, seed in cases if ss == scene]
        diff = np.concatenate([sample[NAMES[1]][k]-sample[NAMES[0]][k] for k in keys])
        paired.append({'scenario': scene, 'mean_sample_MSE_difference_spatial_minus_pod_K2': float(diff.mean()),
                       'fraction_samples_spatial_lower_MSE': float((diff<0).mean())})
    plot = {'truth': truth[0].cpu().numpy().reshape(H, W), 'test_index': np.array(ids[0]),
            'noisy_sensor_sigma02_seed31001': (sensors[0]+tensor(noises['noise_0.2__31001'][0], torch.float64)).cpu().numpy()}
    for kind in NAMES:
        for scene, arr in examples[kind].items():
            plot[f'{kind}_{scene}'] = arr
    np.savez_compressed(OUT/'same_sample_plot_data.npz', **plot)
    return rows, aggregated, paired, timings, plot, paired_old


def image64(fig):
    buf = io.BytesIO(); fig.savefig(buf, format='png', dpi=140, bbox_inches='tight'); plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def table(rows, cols):
    def cell(x):
        return f'{x:.9g}' if isinstance(x, float) else html.escape(str(x))
    return '<div class="scroll"><table><tr>'+''.join('<th>'+html.escape(k)+'</th>' for k in cols)+'</tr>'+''.join('<tr>'+''.join('<td>'+cell(r.get(k, ''))+'</td>' for k in cols)+'</tr>' for r in rows)+'</table></div>'


def report(summary, plot):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9})
    figs = []
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    for kind in NAMES:
        hist = json.loads((OUT/f'{kind}_history.json').read_text(encoding='utf-8'))
        ep = [r['epoch'] for r in hist]
        for ax, key in zip(axes[:2], ['train_MSE_K2', 'validation_clean_MSE_K2']):
            ax.plot(ep, [math.sqrt(r[key]) for r in hist], label=kind)
        axes[2].plot(ep, [r['lr_used'] for r in hist], label=kind)
    for ax, title in zip(axes, ['Clean training RMSE (K)', 'Clean validation RMSE (K) - selection', 'Learning rate used']):
        ax.set(xlabel='Epoch', title=title, yscale='log'); ax.grid(alpha=.25); ax.legend(fontsize=7)
    figs.append(('训练/干净验证曲线及每轮实际学习率', image64(fig)))
    fig, ax = plt.subplots(figsize=(8, 3.5))
    for kind in NAMES:
        rr = [r for r in summary['main_aggregate'] if r['model'] == kind and (r['scenario']=='clean' or r['scenario'].startswith('noise_'))]
        ax.plot([0 if r['scenario']=='clean' else float(r['scenario'][6:]) for r in rr], [r['pooled_global_field_RMSE_K'] for r in rr], marker='o', label=kind)
    ax.set(xlabel='Gaussian sensor sigma (K)', ylabel='Pooled global field RMSE (K)', title='Clean-only training: degradation, not model selection'); ax.legend(); ax.grid(alpha=.25)
    figs.append(('测试噪声退化，不参与训练或选模', image64(fig)))
    for scene in ['clean', 'noise_0.2']:
        truth = plot['truth']; preds = [plot[f'{kind}_{scene}'] for kind in NAMES]
        # 同一场图、同一温度范围；两个误差图共享对称色限。
        vmin = min(float(a.min()) for a in [truth]+preds); vmax = max(float(a.max()) for a in [truth]+preds)
        lim = max(float(np.abs(a-truth).max()) for a in preds)
        fig, axes = plt.subplots(2, 3, figsize=(14, 6))
        for row, kind in enumerate(NAMES):
            for col, arr in enumerate([truth, preds[row], preds[row]-truth]):
                im = axes[row, col].imshow(arr, aspect='auto', cmap='coolwarm' if col==2 else 'inferno', vmin=-lim if col==2 else vmin, vmax=lim if col==2 else vmax)
                axes[row, col].set_title(kind+[' | truth K', ' | prediction K', ' | error K'][col]); fig.colorbar(im, ax=axes[row, col], fraction=.025)
        fig.tight_layout(); figs.append((f'固定首个测试样本 {int(plot["test_index"])}，{scene}；高斯噪声种子31001；共享温度与误差色限', image64(fig)))
    index = {(r['model'], r['scenario']): r for r in summary['main_aggregate']}
    conclusions = []
    for scene in ['clean', 'noise_0.2', 'noise_1.0', 'common_bias_0.2', 'ch0_bias_0.5']:
        a = index[(NAMES[0], scene)]['pooled_global_field_RMSE_K']; b = index[(NAMES[1], scene)]['pooled_global_field_RMSE_K']
        conclusions.append(f'<li>{scene}：POD {a:.9f} K，空间decoder {b:.9f} K；空间decoder相对变化 {(b/a-1)*100:+.3f}%（负值更好）。</li>')
    images = ''.join(f'<figure><img src="data:image/png;base64,{im}" alt="配对实验图"><figcaption>{html.escape(cap)}</figcaption></figure>' for cap, im in figs)
    trainrows = [{'model': k, **v} for k, v in summary['training'].items()]
    limits = ''.join('<li>'+html.escape(s)+'</li>' for s in summary['limitations'])
    raw = html.escape(json.dumps(summary, ensure_ascii=False, indent=2))
    text = f'''<!DOCTYPE html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>MLP+POD 与 MLP+空间decoder：clean配对GPU实验</title><style>body{{font:15px/1.65 system-ui,sans-serif;max-width:1440px;margin:30px auto;padding:0 22px;background:#f4f7fb;color:#182d44}}section,figure{{background:white;padding:20px;border:1px solid #d8e1ec;border-radius:10px;margin:20px 0}}h1,h2{{color:#154b7a}}table{{border-collapse:collapse;font-size:12px;width:100%}}td,th{{border:1px solid #d8e1ec;padding:8px;white-space:nowrap;text-align:right}}th{{background:#e9f1fa}}.scroll{{overflow:auto}}img{{max-width:100%;height:auto}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}}</style>
<h1>MLP + POD16 vs MLP + 可学习空间decoder</h1><p>真实CUDA实测 · 仅干净训练，无augmentation/dropout噪声 · 同8192/1000/2000索引 · 本轮两模型重新训练 · 仅clean验证选模 · seed42单次训练</p>
<section><h2>1. 本轮实测结论</h2><ul>{''.join(conclusions)}</ul><p>只比较本轮两新模型。不把上一轮mixed-selected指标差异称为结构收益。空间decoder参数更多，不是严格等参数比较，也没有统计显著性结论。</p></section>
<section><h2>2. 完整2000样本配对测试</h2>{table(summary['main_aggregate'], ['model','scenario','noise_repetitions','pooled_global_field_RMSE_K','MAE_K','peak_MAE_K','maxpoint_K','mean_sample_RMSE_K','noise_seed_RMSE_std_K'])}<p>pooledglobalfieldRMSE = sqrt(mean(MSE over samples and noise seeds))，不是三次RMSE算术均值。MAE为全部场点绝对误差均值；peakMAE为每样本预测场最大温度与真实最大温度之差的绝对值均值；maxpoint为全部样本和种子的最大格点绝对误差。mean_sample_RMSE是先对每样本开根再平均，不能与globalRMSE混淆。三噪声种子重复不是三次训练。</p>{table(summary['paired_comparisons'], ['scenario','mean_sample_MSE_difference_spatial_minus_pod_K2','fraction_samples_spatial_lower_MSE'])}</section>
<section><h2>3. 严格配对协议</h2><p>输入均为同10特征：[九点温差/固定训练尺度，九点均值b的固定标准化]，同128宽、3残差块SiLU encoder且初始权重逐项相等。每轮共享shuffle，训练仅干净读数。b只来自当前输入：训练为clean，测试从含噪读数计算，绝不使用真值归一化或clean b推断。</p><p>共享训练残差均值 μR=mean(Ttrain−bclean) 与 residual_scale=RMS(Ttrain−bclean−μR)；两模型输出都是 b_current+μR+scale·decoder(x)。POD16基底严格复用同8192 clean residual拟合基底并核验indices、文件/数组SHA与旧checkpoint一致，仅使用其基底/统计，没有复用网络权重。POD输出16维无模态白化，展开矩阵为sqrt(9600)Uᵀ。空间版本只从encoder的128维特征，经MLP展开{summary['benchmark']['channels'][0]}×5×30，双线性上采样10×60、20×120、40×240与3×3卷积，channels={summary['benchmark']['channels']}，中间SiLU，末层线性。空间前向不使用POD基底或真值场；checkpoint中公共basis缓冲仅为统一封装，不参与空间计算。</p><p>无显式coordinate channels；MLP展开的固定网格与卷积边界隐含位置，不是坐标无关模型。这是bilinear卷积decoder，不是FNO输入或频域算子。两模型都显式训练完整9600点残差标准化MSE；乘共同scale²等于物理K²，保留POD投影残差。实际选模使用float64物理全场clean验证MSE。损失等价及梯度均数值核验。</p><p>相同budget：max300，early-stop patience40，AdamW lr1e−3 wd1e−4，ReduceLROnPlateau factor0.5 patience10 minlr1e−6，batch={summary['benchmark']['batch_size']}。实际轮数/调度取决各自clean验证，规则相同而非强行等训练步数。无增强、无dropout、无test tuning。TF32禁用，网络FP32、物理重建FP64；CUDA bilinear反向可能非确定，不声称位级可复现。</p><pre>{html.escape(json.dumps(summary['sanity'], ensure_ascii=False, indent=2))}</pre></section>
<section><h2>4. 成本与GPU延迟</h2>{table(trainrows, ['model','parameters','encoder_parameters','epochs','optimizer_steps','best_epoch','best_validation_clean_RMSE_K','final_lr','train_seconds_synchronized','cuda_peak_allocated_bytes','stopping_reason'])}<p>同GPU，所有计时前后CUDA同步。以下完整路径含feature、均值、固定缩放、POD或空间decode；2000批量吞吐和单样本延迟分别报告。不与旧CPU时延比较，不含CPU到GPU传输和磁盘IO。</p><pre>{html.escape(json.dumps(summary['timings'], ensure_ascii=False, indent=2))}</pre><h3>测试前GPU预基准</h3><pre>{html.escape(json.dumps(summary['benchmark'], ensure_ascii=False, indent=2))}</pre></section>
<section><h2>5. 曲线及同样本同色图</h2><p>固定取test_indices[0]，不按重建效果挑样本。sigma0.2 seed31001读数：{html.escape(str(plot['noisy_sensor_sigma02_seed31001'].tolist()))}</p>{images}</section>
<section><h2>6. 限制与文件</h2><ul>{limits}</ul><p>结果绝对目录：{html.escape(str(OUT))}。保留两模型clean-best自足checkpoint、末轮含optimizer/scheduler/RNG checkpoint、完整每轮history、metadata/summary/status/manifest、共享索引、测试扰动、逐种子/逐样本MSE、图数据和脚本副本。HTML全部图片及JSON内嵌，无外链依赖。</p></section><details><summary>完整结果JSON</summary><pre>{raw}</pre></details></html>'''
    (OUT/'result_summary.html').write_text(text, encoding='utf-8')


def verify(data, training, bench):
    result = {}
    for kind in NAMES:
        hist = json.loads((OUT/f'{kind}_history.json').read_text(encoding='utf-8'))
        ck = torch.load(OUT/f'{kind}_clean_best.pt', map_location='cpu', weights_only=False)
        last = torch.load(OUT/f'{kind}_last.pt', map_location='cpu', weights_only=False)
        selected = min(hist, key=lambda h: h['validation_clean_MSE_K2'])
        assert ck['epoch'] == selected['epoch'] == training[kind]['best_epoch']
        assert ck['validation_clean_MSE_K2'] == selected['validation_clean_MSE_K2']
        assert len(hist) == training[kind]['epochs'] == last['epoch']
        assert ck['training_indices_sha256'] == data['metadata']['indices_sha256']['train']
        assert 'optimizer' in last and 'scheduler' in last
        model, _ = load_model(kind)
        actual = validation(model, data, bench['batch_size'])
        assert abs(actual-ck['validation_clean_MSE_K2']) < 1e-12
        predictor = load_predictor(OUT/f'{kind}_clean_best.pt')
        ss = data['sensors']['validation'][:2].cpu().numpy()
        with torch.inference_mode():
            reference = model(tensor(ss, torch.float64)).cpu().numpy().reshape(-1, H, W)
        assert np.allclose(predictor(ss), reference, rtol=0, atol=1e-10)
        result[kind] = {'checkpoint_epoch_matches_clean_history_argmin': True,
                        'reloaded_validation_MSE_K2': actual, 'standalone_predictor_verified': True,
                        'best_checkpoint_sha256': sha(OUT/f'{kind}_clean_best.pt'),
                        'last_checkpoint_sha256': sha(OUT/f'{kind}_last.pt')}
        del model, predictor, last, ck
    dump(OUT/'checkpoint_verification.json', result)
    return result


def main(mode):
    if mode == 'benchmark':
        if OUT.exists():
            raise FileExistsError('新目录已存在，拒绝覆盖预基准')
        OUT.mkdir()
        status('preparing_benchmark')
        data = prepare()
        dump(OUT/'metadata.json', data['metadata'])
        np.savez(OUT/'subset_indices.npz', **data['ids'])
        bench = benchmark(data)
        sanity(data, bench)
        status('benchmarked', batch_size=bench['batch_size'], channels=bench['channels'], script_sha256=sha(__file__))
        return
    assert OUT.exists() and json.loads((OUT/'status.json').read_text())['stage'] == 'benchmarked'
    bench = json.loads((OUT/'benchmark.json').read_text())
    assert bench['script_sha256'] == sha(__file__), '预基准后脚本变化，拒绝静默运行'
    assert not any(OUT.glob('*_history.json')), '已有训练轨迹，拒绝覆盖'
    status('preparing_training')
    data = prepare(); metadata = data['metadata']
    metadata.update(batch_size=bench['batch_size'], spatial_channels=bench['channels'])
    dump(OUT/'metadata.json', metadata)
    shutil.copy2(__file__, OUT/'spatial_decoder_clean_test.py')
    np.savez(OUT/'shared_preprocessing.npz', basis=data['basis'], residual_mean=data['mean'], residual_scale=data['scale'], train_indices=data['ids']['train'])
    checks = sanity(data, bench)
    training = {}
    for kind in NAMES:
        training[kind] = train(kind, data, bench)
        dump(OUT/'training_summary.json', training)
    # checkpoint固定和验证结束后才允许任何测试指标计算。
    verification = verify(data, training, bench)
    status('testing', checkpoints_frozen=True)
    rows, aggregate, paired, timings, plot, old_pair = test(data, bench)
    summary = {'experiment': 'clean均值分解ResMLP+POD16 vs ResMLP+learned空间decoder配对GPU实验', 'completed': True,
        'metadata': metadata, 'benchmark': bench, 'sanity': checks, 'training': training, 'checkpoint_verification': verification,
        'main_aggregate': aggregate, 'per_seed_metrics': rows, 'paired_comparisons': paired, 'timings': timings,
        'test_noise_exact_equal_previous': old_pair,
        'limitations': ['仅seed42一次训练，三噪声种子不等于三次训练，不宣称显著性。',
            '旧测试集已经暴露，本轮不是盲测；结构只按预基准资源规则在测试前固定，本轮未用测试调参或选checkpoint。',
            '8192训练子集pilot、原随机split内取样，不代表跨工况/跨电池泛化或全73340训练规模效果。',
            '空间decoder参数更多，不是严格等参数消融；比较的是同encoder、输入、预处理、损失、优化预算与选模规则下的两种decoder。',
            '前轮mixed-selected协议不同；本报告只比较本轮clean-selected重训模型，不用旧指标证明结构收益。',
            '合成高斯及固定偏置只用于鲁棒性退化测量，不代表真实传感器噪声；clean-only训练不保证噪声鲁棒。',
            'POD复用float64随机化SVD基底（96草图、2次幂迭代），不是完整SVD；已核验索引、哈希及正交。',
            '共享固定残差尺度是一维全场标量，不是逐模态白化；训练网络float32，验证/推断物理重建float64。',
            '卷积零填充与MLP展开隐含固定网格，无显式坐标；双线性解码不是FNO。',
            'GPU批量吞吐不等于单样本在线延迟；不与旧CPU时延比较，计时不含主机传输和IO。',
            'CUDA双线性插值反向可能非确定，固定seed仍不承诺位级重复。'],
        'elapsed_seconds_before_report': time.perf_counter()-START}
    report(summary, plot)
    summary['elapsed_seconds_total'] = time.perf_counter()-START
    dump(OUT/'summary.json', summary)
    manifest = {p.name: {'size': p.stat().st_size, 'sha256': sha(p)} for p in OUT.iterdir() if p.is_file() and p.name not in ['artifact_manifest.json', 'run.log', 'status.json']}
    dump(OUT/'artifact_manifest.json', manifest)
    assert (OUT/'result_summary.html').stat().st_size > 10000
    status('completed', completed=True, checkpoints_verified=True, summary_sha256=sha(OUT/'summary.json'),
           report_path=str(OUT/'result_summary.html'), elapsed_seconds_total=summary['elapsed_seconds_total'])
    log('全部训练、配对测试、自足checkpoint核验及HTML报告完成')
    print(json.dumps({'completed': True, 'main_aggregate': aggregate, 'training': {k: {f: v[f] for f in ['epochs','best_epoch','best_validation_clean_RMSE_K','parameters','train_seconds_synchronized']} for k, v in training.items()}}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['benchmark', 'run'], required=True)
    args = parser.parse_args()
    try:
        main(args.mode)
    except Exception:
        if OUT.exists():
            (OUT/'failure_traceback.txt').write_text(traceback.format_exc(), encoding='utf-8')
            status('failed', completed=False, traceback=traceback.format_exc())
        traceback.print_exc()
        raise
