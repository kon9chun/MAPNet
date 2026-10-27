"""四模型共享预算比较；源文件只读，所有新产物位于独立结果目录。
运行：python outputs/four_model_benchmark.py --benchmark-only
      python outputs/four_model_benchmark.py --run
仅使用训练统计及固定训练基底；测试集曾暴露，不是全新盲测。
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
from pathlib import Path
import argparse
import base64
import csv
import hashlib
import html
import io
import json
import math
import platform
import shutil
import subprocess
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
from threadpoolctl import threadpool_limits

import paths

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'four_model_comparison_20261021'
OLD = ROOT / 'pod_check_20261021'
CACHE = ROOT / 'pod_noise_20261021'
SOURCE = paths.DATA
NAMES = ('pod16_resmlp_aug', 'deeponet', 'fno2d', 'direct_mlp')
D, H, W = 9600, 40, 240
START = time.perf_counter()


def dump(path, obj):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(tmp, path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def log(text):
    line = f'[{time.perf_counter()-START:.2f}s] {text}'
    print(line, flush=True)
    with (OUT/'run.log').open('a', encoding='utf-8') as f:
        f.write(line+'\n')


def save_torch(path, obj):
    path = Path(path)
    tmp = path.with_name(path.name+'.tmp')
    torch.save(obj, tmp)
    os.replace(tmp, path)


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open('w', encoding='utf-8-sig', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)


def parameter_count(net):
    # complex64 的一个复参数按两个实标量计数。
    return sum(p.numel() * (2 if p.is_complex() else 1) for p in net.parameters())


def coordinates():
    yy, xx = torch.meshgrid(torch.linspace(0, 1, H), torch.linspace(0, 1, W), indexing='ij')
    return torch.stack((yy, xx), -1).reshape(D, 2)


def interpolation_matrix(positions):
    rows, cols = np.unique(positions[:, 0]), np.unique(positions[:, 1])
    assert len(rows) == len(cols) == 3
    assert np.array_equal(positions, np.array([(r, c) for r in rows for c in cols]))
    def linear(knots, n):
        mat = np.zeros((n, 3), np.float32)
        for i in range(n):
            j = int(np.clip(np.searchsorted(knots, i, side='right')-1, 0, 1))
            t = (i-knots[j])/(knots[j+1]-knots[j])
            mat[i, j], mat[i, j+1] = 1-t, t
        return mat
    a, b = linear(rows, H), linear(cols, W)
    matrix = np.einsum('hi,wj->hwij', a, b).reshape(D, 9)
    assert np.allclose(matrix.sum(1), 1)
    assert np.allclose(matrix[positions[:, 0]*W+positions[:, 1]], np.eye(9))
    return torch.from_numpy(matrix.T.copy())


class ResidualBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 128))
    def forward(self, x):
        return F.silu(x+self.net(x))


class PODNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(9, 128), nn.SiLU(), *[ResidualBlock() for _ in range(3)], nn.Linear(128, 16))
    def forward(self, x):
        return self.net(x)


class DirectMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(9, 128), nn.SiLU(), nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, D))
    def forward(self, x):
        return self.net(x)


class DeepONet(nn.Module):
    def __init__(self):
        super().__init__()
        self.branch = nn.Sequential(nn.Linear(9, 128), nn.SiLU(), nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 64))
        self.trunk = nn.Sequential(nn.Linear(2, 128), nn.SiLU(), nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 64))
        self.bias = nn.Parameter(torch.zeros(()))
        self.register_buffer('coords', coordinates())
    def forward(self, x):
        return self.branch(x) @ self.trunk(self.coords).T / 8.0 + self.bias


class SpectralConv2d(nn.Module):
    def __init__(self, width=12, modes=(6, 12)):
        super().__init__()
        self.m1, self.m2 = modes
        shape = (width, width, *modes)
        self.positive = nn.Parameter(torch.randn(*shape, dtype=torch.cfloat)/(width**2))
        self.negative = nn.Parameter(torch.randn(*shape, dtype=torch.cfloat)/(width**2))
    def forward(self, x):
        ft = torch.fft.rfft2(x, norm='ortho')
        out = torch.zeros(x.shape[0], x.shape[1], H, W//2+1, dtype=ft.dtype, device=x.device)
        out[:, :, :self.m1, :self.m2] = torch.einsum('bixy,ioxy->boxy', ft[:, :, :self.m1, :self.m2], self.positive)
        out[:, :, -self.m1:, :self.m2] = torch.einsum('bixy,ioxy->boxy', ft[:, :, -self.m1:, :self.m2], self.negative)
        return torch.fft.irfft2(out, s=(H, W), norm='ortho')


class FNO2d(nn.Module):
    def __init__(self, positions):
        super().__init__()
        self.register_buffer('interp', interpolation_matrix(positions))
        self.register_buffer('coords', coordinates().T.reshape(1, 2, H, W))
        self.lift = nn.Conv2d(3, 12, 1)
        self.spectral = nn.ModuleList([SpectralConv2d() for _ in range(4)])
        self.local = nn.ModuleList([nn.Conv2d(12, 12, 1) for _ in range(4)])
        self.project = nn.Sequential(nn.Conv2d(12, 32, 1), nn.GELU(), nn.Conv2d(32, 1, 1))
    def forward(self, x):
        # 输入是九个标准化测点的双线性插值，不读取任何真值场。
        field = (x @ self.interp).reshape(-1, 1, H, W)
        z = self.lift(torch.cat((field, self.coords.expand(len(x), -1, -1, -1)), 1))
        for spectral, local in zip(self.spectral, self.local):
            z = F.gelu(spectral(z)+local(z))
        return self.project(z).reshape(-1, D)


def model_factory(name, positions):
    return {'pod16_resmlp_aug': PODNet, 'deeponet': DeepONet, 'direct_mlp': DirectMLP,
            'fno2d': lambda: FNO2d(positions)}[name]()


def load_data():
    z = np.load(OLD/'pod_basis.npz', allow_pickle=False)
    splits = np.load(OLD/'split_indices.npz', allow_pickle=False)
    proj = np.load(CACHE/'projection_cache.npz', allow_pickle=False)
    mean, basis = z['mean'], z['basis'][:, :16]
    assert np.max(np.abs(basis.T@basis-np.eye(16))) < 1e-10
    y = np.load(SOURCE/'ydata.npy', mmap_mode='r', allow_pickle=False)
    sp = np.load(SOURCE/'sp.npy', mmap_mode='r', allow_pickle=False)
    assert y.shape == (93340, H, W) and sp.shape == (93340, 9)
    assert y.dtype == sp.dtype == np.float32
    ids = np.concatenate([splits[k] for k in ('train', 'validation', 'test_reserved')])
    assert np.array_equal(np.sort(ids), np.arange(len(y)))
    assert np.all(z['sp_std'] > 0)
    prior = json.loads((CACHE/'projection_manifest.json').read_text(encoding='utf-8'))
    sig = prior['signature']
    assert digest(OLD/'pod_basis.npz') == sig['basis_sha256']
    assert digest(OLD/'split_indices.npz') == sig['split_sha256']
    for key, expected in sig['sources'].items():
        st = Path(key).stat()
        assert st.st_size == expected['size'] and st.st_mtime_ns == expected['mtime_ns']
    scale = float(np.sqrt(proj['energy'][splits['train']].mean()/D))
    return dict(y=y, sp=sp, mean=mean, basis=basis, sm=z['sp_mean'], ss=z['sp_std'], positions=z['positions'],
                split={k: splits[k] for k in splits.files}, coef=proj['coef'], residual=proj['residual'][:, 1], scale=scale,
                signature=sig)


def benchmark(data):
    log('开始实测四模型前向、反向及优化步；不使用测试数据。')
    rows, gradients = [], {}
    for threads in (1, 4):
        torch.set_num_threads(threads)
        for name in NAMES:
            torch.manual_seed(99)
            net = model_factory(name, data['positions'])
            opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
            x = torch.randn(32, 9)
            target = torch.randn(32, 16 if name == NAMES[0] else D)
            measurements = []
            for step in range(5):
                t = time.perf_counter()
                opt.zero_grad(set_to_none=True)
                loss = (net(x)-target).square().mean()
                loss.backward()
                if name == 'fno2d' and step == 0:
                    norms = {k: float(p.grad.abs().norm()) for k, p in net.named_parameters() if 'spectral' in k}
                    assert all(math.isfinite(v) and v > 0 for v in norms.values())
                    gradients[str(threads)] = norms
                assert math.isfinite(float(loss.detach()))
                opt.step()
                if step >= 2:
                    measurements.append(time.perf_counter()-t)
            row = dict(model=name, threads=threads, batch_size=32, mean_step_seconds=float(np.mean(measurements)),
                       real_scalar_parameters=parameter_count(net))
            rows.append(row)
            log(f'基准 {row}')
    chosen = min([r for r in rows if r['model']=='fno2d'], key=lambda r: r['mean_step_seconds'])['threads']
    selected = [r for r in rows if r['threads']==chosen]
    steps = {r['model']: r['mean_step_seconds'] for r in selected}
    # 计算预算选择只依赖耗时，不依赖模型质量。保留全格、100轮与统一样本。
    def estimate(n, v):
        return 100*(sum(steps.values())*math.ceil(n/32)+sum(steps.values())*.4*math.ceil(v/32))
    full = estimate(73340, 10000)
    if full <= 10800:
        n, v, test, mode = 73340, 10000, 10000, 'full_training'
    else:
        n, v, test, mode = 8192, 1000, 2000, 'shared_subset_pilot'
    if estimate(n, v) > 14400:
        n, v, test, mode = 4096, 1000, 1000, 'resource_limited_shared_subset_pilot'
    result = {'measurements': rows, 'spectral_gradient_norms': gradients, 'selected_threads': chosen,
              'batch_size': 32, 'max_epochs': 100, 'patience': 15, 'train_n': n, 'validation_n': v, 'test_n': test,
              'mode': mode, 'cost_model': '100*(sum measured train step * ceil(n/32) + 0.4*sum step*ceil(v/32)); 不含I/O等开销，仅用于预算选择',
              'full_100epoch_compute_estimate_seconds': full, 'selected_compute_estimate_seconds': estimate(n, v),
              'training_seeds': [42], 'selection_uses_test_quality': False}
    dump(OUT/'compute_benchmark.json', result)
    log(f'统一预算选定：{mode}; train={n}, val={v}, test={test}, batch=32, max100/patience15, seed42, threads={chosen}')
    return result


def make_protocol(data, bench):
    return {
        'experiment': 'paper7 四模型共享预算真实训练', 'budget': bench,
        'architecture': {
            NAMES[0]: '9->128 SiLU; 3x residual(128->128 SiLU->128); ->16; 固定训练POD基底解码',
            'direct_mlp': '9->128 SiLU->128 SiLU->128 SiLU->9600；无POD、无CNN',
            'deeponet': 'branch9->128 SiLU->128 SiLU->64; trunk2->128 SiLU->128 SiLU->64; dot/sqrt64+bias',
            'fno2d': '九标准化测点按真实坐标双线性插值+xy；lift3->12；4x(FourierConv modes6,12 + Conv1x1), GELU；12->32 GELU->1；40x240全格'},
        'optimizer': {'name': 'AdamW', 'lr': .001, 'weight_decay': .0001, 'lr_schedule': None},
        'loss': '统一归一化场MSE；POD sum(coef/(scale*sqrtD)误差平方)+不可约残差/(D*scale²)，精确等价全场MSE',
        'normalization': {'input': '九点共享原始训练73340的逐测点mean/std',
                          'output': '所有模型同样加既有训练mean9600；统一除以训练中心化场标量RMS，不逐像素/逐模态白化',
                          'output_scale_K': data['scale'], 'input_mean_K': data['sm'].tolist(), 'input_std_K': data['ss'].tolist(),
                          'statistics_fit_ids': '原始训练73340，包含未参与pilot梯度更新样本；属于训练专用额外先验/预处理，四模型共享；POD基底仅POD使用'},
        'augmentation': {'sigmas_K': [0, .05, .1, .2, .5], 'probabilities': [.2]*5,
                         'method': '每轮每样本随机等概率情形，各传感器独立高斯；四模型同seed每轮相同order/情形/噪声'},
        'validation': {'seed': 24001, 'method': '每样本固定分配一个情形，五组等样本数(若整除)，固定噪声，仅评估一次混合，不是每样本五情形；均值场MSE选模',
                       'sigmas_K': [0, .05, .1, .2, .5]},
        'sampling': {'train_seed': 20261031, 'validation_seed': 20261032, 'test_seed': 20261033,
                     'method': '在既有对应划分内无放回随机选取，排序后保存全局索引；所有模型完全相同'},
        'test': {'gaussian_sigmas_K': [0, .05, .1, .2, .5, 1.0], 'noise_seeds': [31001, 31002, 31003],
                 'deterministic_biases_K': {'common_all_channels': .2, 'channel0_only': .5},
                 'evaluation': '统一测试子集所有9600点；正噪声3次，clean和确定性偏置各一次；训练seed仅42，噪声重复不是训练重复',
                 'previously_exposed': True, 'new_blind_test': False, 'used_in_current_selection': False,
                 'limitation': '原测试集此前被查看；本轮无测试选模，不能称全新盲测'},
        'training_seeds': bench['training_seeds'], 'source_signature': data['signature'],
        'positions_row_col': data['positions'].tolist(), 'basis_refitted': False,
        'limitations': ['共享子集pilot若启用，不是最终论文验证', '各实现是朴素基线，不是原论文官方最优实现或调参结果',
                       '只一训练种子，无训练随机性误差条或显著性结论', '随机划分可能存在时间/工况相关泄漏，不等于跨工况泛化',
                       '合成测点噪声与偏置，不代表真实传感器标定性能', 'POD16九点逆问题欠定，依赖训练分布先验',
                       '参数量不等；FNO保留全格无粗网格或坐标子采样；边界无额外padding']}


def prepare_subsets(data, protocol, protocol_hash):
    budget = protocol['budget']
    indices_path = OUT/'benchmark_indices.npz'
    counts = {'train': budget['train_n'], 'validation': budget['validation_n'], 'test_reserved': budget['test_n']}
    seeds = (20261031, 20261032, 20261033)
    ids = {key: np.sort(np.random.default_rng(seed).choice(data['split'][key], counts[key], replace=False))
           for key, seed in zip(counts, seeds)}
    if indices_path.exists():
        previous = np.load(indices_path)
        assert all(np.array_equal(previous[k], v) for k, v in ids.items())
    else:
        np.savez(indices_path, **ids)
    n = counts['train']+counts['validation']
    cache_path = OUT/'train_validation_fields_float32.npy'
    manifest = {'protocol_hash': protocol_hash, 'shape': [n, D], 'dtype': 'float32',
                'layout': 'train followed by validation; (raw-train_mean)/train_field_RMS'}
    manifest_path = OUT/'field_cache_manifest.json'
    good = cache_path.exists() and manifest_path.exists() and json.loads(manifest_path.read_text(encoding='utf-8'))==manifest
    if not good:
        arr = np.lib.format.open_memmap(cache_path, mode='w+', dtype=np.float32, shape=(n, D))
        allids = np.concatenate([ids['train'], ids['validation']])
        for a in range(0, n, 128):
            sl = slice(a, min(a+128, n))
            arr[sl] = ((np.asarray(data['y'][allids[sl]], np.float64).reshape(-1, D)-data['mean'])/data['scale']).astype(np.float32)
        arr.flush()
        del arr
        dump(manifest_path, manifest)
    arr = np.load(cache_path, mmap_mode='r')
    targets = torch.from_numpy(np.array(arr, copy=True))
    x = {k: torch.tensor((np.asarray(data['sp'][v], np.float64)-data['sm'])/data['ss'], dtype=torch.float32) for k, v in ids.items()}
    coef = {k: torch.tensor(data['coef'][v]/(data['scale']*math.sqrt(D)), dtype=torch.float32) for k, v in ids.items()}
    comp = {k: torch.tensor(data['residual'][v]/(D*data['scale']**2), dtype=torch.float32) for k, v in ids.items()}
    rng = np.random.default_rng(24001)
    categories = np.arange(counts['validation']) % 5
    rng.shuffle(categories)
    sigmas = np.array([0, .05, .1, .2, .5])[categories]
    eps = rng.normal(size=(len(categories), 9))*sigmas[:, None]
    valx = x['validation']+torch.tensor(eps/data['ss'], dtype=torch.float32)
    np.savez(OUT/'validation_fixed_noise.npz', categories=categories, noise_K=eps, sigmas_K=sigmas)
    # 检查缓存投影、传感器和全场的精确对应。
    for key in ('train', 'validation'):
        sample = ids[key][:8]
        raw = np.asarray(data['y'][sample], np.float64).reshape(-1, D)
        measured = raw[:, data['positions'][:, 0]*W+data['positions'][:, 1]]
        assert np.max(np.abs(measured-data['sp'][sample])) < 1e-3
        actual = (raw-data['mean']) @ data['basis']
        assert np.max(np.abs(actual-data['coef'][sample])) < 1e-6
    torch.manual_seed(334)
    trial = torch.randn(8, 16)
    explicit = ((trial @ torch.tensor(data['basis'].T*math.sqrt(D), dtype=torch.float32))-targets[:8]).square().mean(1)
    equivalent = (trial-coef['train'][:8]).square().sum(1)+comp['train'][:8]
    equivalence_delta = float(torch.max(torch.abs(explicit-equivalent)))
    assert torch.allclose(explicit, equivalent, rtol=3e-6, atol=3e-6)
    dump(OUT/'sanity_checks.json', {'POD_normalized_field_loss_equivalence_max_abs_delta': equivalence_delta,
                                  'sampled_projection_recomputed': True, 'sensor_field_match': True,
                                  'interpolation_exact_at_sensors': True})
    return dict(ids=ids, x=x, coef=coef, comp=comp, y=targets, valx=valx, categories=categories)


def objective(name, pred, target, coef, comp):
    if name == NAMES[0]:
        return ((pred-coef).square().sum(1)+comp).mean()
    return (pred-target).square().mean()


def augmentation(seed, epoch, n, ss):
    generator = torch.Generator().manual_seed(seed*100000+epoch)
    order = torch.randperm(n, generator=generator)
    levels = torch.tensor([0, .05, .1, .2, .5])[torch.randint(0, 5, (n, 1), generator=generator)]
    eps = torch.randn(n, 9, generator=generator)*levels/torch.tensor(ss, dtype=torch.float32)
    return order, eps


def fit(name, seed, data, cache, protocol, phash):
    torch.manual_seed(seed)
    net = model_factory(name, data['positions'])
    opt = torch.optim.AdamW(net.parameters(), lr=.001, weight_decay=.0001)
    tag = f'{name}_seed{seed}'
    lastpath, bestpath = OUT/(tag+'_last.pt'), OUT/(tag+'_best.pt')
    history, best, best_epoch, stale, first = [], math.inf, 0, 0, 1
    elapsed_before = 0.
    if lastpath.exists():
        state = torch.load(lastpath, map_location='cpu', weights_only=False)
        assert state['protocol_hash']==phash and state['script_sha256']==protocol['script_sha256']
        net.load_state_dict(state['state_dict'])
        opt.load_state_dict(state['optimizer'])
        history, best, best_epoch, stale = state['history'], state['best_score'], state['best_epoch'], state['stale']
        first, elapsed_before = state['epoch']+1, state['elapsed_seconds']
        log(f'{tag} 从已完成第{first-1}轮恢复，协议/脚本hash吻合。')
    budget, n, v = protocol['budget'], len(cache['ids']['train']), len(cache['ids']['validation'])
    bs = budget['batch_size']
    start = time.perf_counter()
    log(f'真实训练 {tag}: n={n}, val={v}, 参数实标量={parameter_count(net)}')
    if stale < budget['patience']:
        for epoch in range(first, budget['max_epochs']+1):
            t = time.perf_counter()
            net.train()
            order, eps = augmentation(seed, epoch, n, data['ss'])
            total = 0.
            for a in range(0, n, bs):
                idx = order[a:a+bs]
                bx = cache['x']['train'][idx]+eps[idx]
                opt.zero_grad(set_to_none=True)
                pred = net(bx)
                loss = objective(name, pred, cache['y'][idx], cache['coef']['train'][idx], cache['comp']['train'][idx])
                if not torch.isfinite(loss):
                    raise RuntimeError(f'{tag} 第{epoch}轮损失非有限')
                loss.backward()
                opt.step()
                total += float(loss.detach())*len(idx)
            train_seconds = time.perf_counter()-t
            net.eval()
            valtotal = 0.
            scenario_sums = np.zeros(5)
            scenario_counts = np.zeros(5, dtype=int)
            with torch.inference_mode():
                for a in range(0, v, bs):
                    sl = slice(a, min(a+bs, v))
                    pred = net(cache['valx'][sl])
                    if name == NAMES[0]:
                        per = (pred-cache['coef']['validation'][sl]).square().sum(1)+cache['comp']['validation'][sl]
                    else:
                        per = (pred-cache['y'][n+a:n+min(a+bs, v)]).square().mean(1)
                    valtotal += float(per.sum())
                    for k in range(5):
                        mask = cache['categories'][sl]==k
                        scenario_sums[k] += float(per[torch.from_numpy(mask)].sum())
                        scenario_counts[k] += int(mask.sum())
            score = valtotal/v*data['scale']**2
            assert math.isfinite(score)
            improved = score < best
            if improved:
                best, best_epoch, stale = score, epoch, 0
            else:
                stale += 1
            row = {'epoch': epoch, 'train_field_RMSE_K': math.sqrt(total/n)*data['scale'],
                   'validation_mixed_field_MSE_K2': score, 'validation_mixed_field_RMSE_K': math.sqrt(score),
                   'validation_clean_RMSE_K': math.sqrt(scenario_sums[0]/scenario_counts[0])*data['scale'],
                   'validation_sigma005_RMSE_K': math.sqrt(scenario_sums[1]/scenario_counts[1])*data['scale'],
                   'validation_sigma01_RMSE_K': math.sqrt(scenario_sums[2]/scenario_counts[2])*data['scale'],
                   'validation_sigma02_RMSE_K': math.sqrt(scenario_sums[3]/scenario_counts[3])*data['scale'],
                   'validation_sigma05_RMSE_K': math.sqrt(scenario_sums[4]/scenario_counts[4])*data['scale'],
                   'train_seconds': train_seconds, 'epoch_compute_seconds': time.perf_counter()-t,
                   'best_epoch_so_far': best_epoch, 'stale': stale}
            history.append(row)
            state = {'protocol_hash': phash, 'script_sha256': protocol['script_sha256'], 'model': name, 'training_seed': seed,
                     'state_dict': net.state_dict(), 'optimizer': opt.state_dict(), 'epoch': epoch, 'best_score': best,
                     'best_epoch': best_epoch, 'stale': stale, 'history': history,
                     'elapsed_seconds': elapsed_before+time.perf_counter()-start,
                     'normalization': protocol['normalization'], 'basis_file': str(OLD/'pod_basis.npz')}
            save_torch(lastpath, state)
            if improved:
                save_torch(bestpath, {k: value for k, value in state.items() if k not in ('optimizer', 'history')})
            dump(OUT/(tag+'_history.json'), {'protocol_hash': phash, 'history': history})
            write_csv(OUT/(tag+'_history.csv'), history)
            if epoch == 1 or epoch % 5 == 0 or stale >= budget['patience']:
                log(f'{tag} epoch={epoch} val={math.sqrt(score):.7g}K best@{best_epoch}={math.sqrt(best):.7g}K 本轮{row["epoch_compute_seconds"]:.2f}s stale={stale}')
            dump(OUT/'progress.json', {'status': 'training', 'active_model': name, 'training_seed': seed, 'epoch': epoch,
                                      'best_epoch': best_epoch, 'validation_RMSE_K': math.sqrt(best), 'protocol_hash': phash})
            if stale >= budget['patience']:
                break
    best_state = torch.load(bestpath, map_location='cpu', weights_only=False)
    net.load_state_dict(best_state['state_dict'])
    net.eval()
    summary = {'model': name, 'training_seed': seed, 'parameters_real_scalars': parameter_count(net),
               'actual_epochs': len(history), 'best_epoch': best_epoch, 'max_epochs': budget['max_epochs'], 'patience': budget['patience'],
               'stop_reason': 'patience15' if stale>=budget['patience'] else 'max100',
               'validation_mixed_field_RMSE_K': math.sqrt(best), 'elapsed_seconds': elapsed_before+time.perf_counter()-start,
               'best_checkpoint': str(bestpath), 'checkpoint_sha256': digest(bestpath), 'history': history}
    dump(OUT/(tag+'_training_summary.json'), summary)
    log(f'{tag}训练完成，{len(history)}轮，最佳{best_epoch}，val={math.sqrt(best):.7g}K。')
    return net, summary


def physical_prediction(name, net, x, data):
    out = net(x)
    if name == NAMES[0]:
        # 在CPU torch中完成固定基底解码，计入端到端推理时间。
        basis = torch.tensor(data['basis'].T, dtype=torch.float32)
        out = out @ basis * math.sqrt(D)
    return out*data['scale']+torch.tensor(data['mean'], dtype=torch.float32)


def inference_timing(name, net, data):
    sample = np.asarray(data['sp'][data['split']['train'][0]:data['split']['train'][0]+1], np.float64).copy()
    def one():
        x = torch.tensor((sample-data['sm'])/data['ss'], dtype=torch.float32)
        return physical_prediction(name, net, x, data)
    with torch.inference_mode():
        for _ in range(10):
            one()
        times = []
        for _ in range(50):
            t = time.perf_counter()
            one()
            times.append(time.perf_counter()-t)
    return {'batch_size': 1, 'device': 'cpu', 'threads': torch.get_num_threads(), 'warmup': 10, 'repetitions': 50,
            'median_ms': float(np.median(times)*1000), 'mean_ms': float(np.mean(times)*1000),
            'p95_ms': float(np.percentile(times, 95)*1000),
            'includes': 'numpy测点输入归一化、tensor转换、网络、加训练均值和恢复K；POD固定基底解码；FNO双线性映射及坐标拼接；不含磁盘I/O',
            'note': '固定不变的mean/basis转tensor也计时，是当前脚本实现而非极致优化延迟'}


def metric_arrays(pred, truth):
    error = pred-truth
    ae = np.abs(error)
    peak = np.abs(pred.max(1)-truth.max(1))
    maximum = ae.max(1)
    mse = np.mean(error*error, axis=1)
    mae = np.mean(ae, axis=1)
    bias = np.mean(error, axis=1)
    return np.column_stack((mse, mae, peak, maximum, bias))


def summarize_arrays(array):
    mse, mae, peak, maximum, bias = array.T
    return {'n_samples': len(array), 'field_MSE_K2': float(mse.mean()), 'field_RMSE_K': float(np.sqrt(mse.mean())),
            'field_MAE_K': float(mae.mean()), 'field_signed_bias_K': float(bias.mean()),
            'sample_field_RMSE_P95_K': float(np.percentile(np.sqrt(mse), 95)),
            'sample_field_RMSE_max_K': float(np.sqrt(mse.max())),
            'peak_temperature_MAE_K': float(peak.mean()), 'peak_temperature_abs_error_P95_K': float(np.percentile(peak, 95)),
            'peak_temperature_max_abs_error_K': float(peak.max()),
            'mean_sample_max_point_abs_error_K': float(maximum.mean()),
            'sample_max_point_abs_error_P95_K': float(np.percentile(maximum, 95)),
            'global_max_point_abs_error_K': float(maximum.max())}


def test_models(models, data, cache, result):
    locked = [{'model': name, 'training_seed': seed, 'checkpoint_sha256': digest(OUT/f'{name}_seed{seed}_best.pt')}
              for name, seed, _ in models]
    dump(OUT/'all_models_locked_before_test.json', {'protocol_hash': result['protocol_hash'], 'models': locked,
                                                  'previously_exposed_test': True, 'current_run_test_used_for_selection': False})
    ids = cache['ids']['test_reserved']
    raw_sensors = np.asarray(data['sp'][ids], np.float64)
    scenarios = [('clean', 0, None)]+[(f'gaussian_{s:g}K', s, None) for s in (.05, .1, .2, .5, 1.)]+[
        ('common_bias_0.2K', None, 'common'), ('channel0_bias_0.5K', None, 'channel0')]
    rows, sample_rows, rep = [], [], {'indices': ids[:2].tolist()}
    rep['truth'] = np.asarray(data['y'][ids[:2]], np.float32)
    for name, seed, net in models:
        teststart = time.perf_counter()
        result['inference_timing'][f'{name}_seed{seed}'] = inference_timing(name, net, data)
        for condition, sigma, bias in scenarios:
            runs, arrays = [], []
            for noise_seed in ([31001, 31002, 31003] if sigma else [None]):
                sensors = raw_sensors.copy()
                if sigma:
                    sensors += np.random.default_rng(noise_seed).normal(0, sigma, sensors.shape)
                elif bias == 'common':
                    sensors += .2
                elif bias == 'channel0':
                    sensors[:, 0] += .5
                x = torch.tensor((sensors-data['sm'])/data['ss'], dtype=torch.float32)
                collected = []
                with torch.inference_mode():
                    for a in range(0, len(ids), result['protocol']['budget']['batch_size']):
                        sl = slice(a, min(a+result['protocol']['budget']['batch_size'], len(ids)))
                        pred = physical_prediction(name, net, x[sl], data).numpy().astype(np.float64)
                        truth = np.asarray(data['y'][ids[sl]], np.float64).reshape(-1, D)
                        collected.append(metric_arrays(pred, truth))
                        if a == 0 and (noise_seed is None or noise_seed==31001) and condition in ('clean', 'gaussian_0.2K', 'gaussian_1K', 'common_bias_0.2K', 'channel0_bias_0.5K'):
                            rep[f'{name}_seed{seed}__{condition}'] = pred[:2].reshape(-1, H, W).astype(np.float32)
                array = np.concatenate(collected)
                stats = summarize_arrays(array)
                stats['noise_seed'] = noise_seed
                runs.append(stats)
                arrays.append(array)
                rows.append({'model': name, 'training_seed': seed, 'condition': condition, 'noise_seed': noise_seed, **stats})
                for idx, values in zip(ids, array):
                    sample_rows.append({'model': name, 'training_seed': seed, 'condition': condition, 'noise_seed': noise_seed,
                                        'global_index': int(idx), 'field_MSE_K2': values[0], 'field_MAE_K': values[1],
                                        'peak_abs_error_K': values[2], 'max_point_abs_error_K': values[3], 'signed_bias_K': values[4]})
            pooled = summarize_arrays(np.concatenate(arrays))
            pooled['n_unique_samples'] = len(ids)
            pooled['n_noise_repeats'] = len(runs)
            pooled['noise_repeat_RMSE_std_K_ddof0'] = float(np.std([v['field_RMSE_K'] for v in runs]))
            pooled['seed_runs'] = runs
            result['test'][f'{name}_seed{seed}'][condition] = pooled
            dump(OUT/'metrics.json', result)
            log(f'测试 {name} {condition}: RMSE={pooled["field_RMSE_K"]:.7g}K MAE={pooled["field_MAE_K"]:.7g}K peakMAE={pooled["peak_temperature_MAE_K"]:.7g}K')
        result['test_elapsed_seconds'][f'{name}_seed{seed}'] = time.perf_counter()-teststart
    write_csv(OUT/'test_metrics_per_noise_repeat.csv', rows)
    write_csv(OUT/'test_metrics_per_sample.csv', sample_rows)
    pooled_rows = []
    for tag, conditions in result['test'].items():
        for cond, metrics in conditions.items():
            pooled_rows.append({'model_training_seed': tag, 'condition': cond, **{k: v for k, v in metrics.items() if k!='seed_runs'}})
    write_csv(OUT/'test_metrics_pooled.csv', pooled_rows)
    np.savez_compressed(OUT/'representative_fields.npz', **rep)
    result['representative_selection'] = {'global_indices': ids[:2].tolist(), 'rule': '固定测试子集按全局索引排序的首两个；不按模型误差或图形效果挑选',
                                        'noise_seed': 31001, 'shared_across_models': True}


def figures_and_html(result):
    image_files = []
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, info in zip(axes.flat, result['training']):
        hist = info['history']
        ax.plot([x['epoch'] for x in hist], [x['train_field_RMSE_K'] for x in hist], label='Train, resampled noise')
        ax.plot([x['epoch'] for x in hist], [x['validation_mixed_field_RMSE_K'] for x in hist], label='Fixed validation mixture')
        ax.axvline(info['best_epoch'], ls='--', color='gray')
        ax.set(title=info['model'], xlabel='Epoch', ylabel='Full-grid field RMSE (K)', yscale='log')
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.tight_layout()
    path = OUT/'training_curves.png'
    fig.savefig(path, dpi=150)
    plt.close(fig)
    image_files.append(path)
    fig, ax = plt.subplots(figsize=(9, 5))
    for tag, values in result['test'].items():
        conditions = ['clean']+[f'gaussian_{s:g}K' for s in (.05, .1, .2, .5, 1.)]
        ax.plot([0, .05, .1, .2, .5, 1.], [values[k]['field_RMSE_K'] for k in conditions], marker='o', label=tag)
    ax.set(xlabel='Sensor iid Gaussian sigma (K)', ylabel='Pooled full-grid RMSE (K)', yscale='log')
    ax.axvspan(.5, 1., color='gray', alpha=.1)
    ax.legend()
    ax.grid(alpha=.2)
    fig.tight_layout()
    path = OUT/'noise_robustness.png'
    fig.savefig(path, dpi=150)
    plt.close(fig)
    image_files.append(path)
    rep = np.load(OUT/'representative_fields.npz')
    for i, idx in enumerate(rep['indices']):
        for condition in ('clean', 'gaussian_0.2K', 'gaussian_1K', 'common_bias_0.2K', 'channel0_bias_0.5K'):
            fields = [rep['truth'][i]]+[rep[f'{name}_seed42__{condition}'][i] for name in NAMES]
            vmin, vmax = min(float(f.min()) for f in fields), max(float(f.max()) for f in fields)
            lim = max(max(float(np.max(np.abs(f-fields[0]))) for f in fields[1:]), 1e-6)
            fig, axes = plt.subplots(2, 5, figsize=(20, 5.8))
            for j, (field, label) in enumerate(zip(fields, ('Truth', *NAMES))):
                im = axes[0, j].imshow(field, origin='lower', aspect='auto', cmap='inferno', vmin=vmin, vmax=vmax)
                axes[0, j].set_title(label, fontsize=9)
                fig.colorbar(im, ax=axes[0, j], fraction=.046)
                im = axes[1, j].imshow(field-fields[0], origin='lower', aspect='auto', cmap='coolwarm', vmin=-lim, vmax=lim)
                axes[1, j].set_title('Prediction - truth (K)', fontsize=9)
                fig.colorbar(im, ax=axes[1, j], fraction=.046)
            fig.suptitle(f'Fixed sample #{int(idx)} | {condition} | shared scales, all 9600 points')
            fig.tight_layout()
            path = OUT/f'fields_{int(idx)}_{condition}.png'
            fig.savefig(path, dpi=140)
            plt.close(fig)
            image_files.append(path)
    def table(headers, rows):
        def cell(v):
            return f'{v:.7g}' if isinstance(v, float) else html.escape(str(v))
        return '<div class="scroll"><table><tr>'+''.join('<th>'+cell(h)+'</th>' for h in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+cell(c)+'</td>' for c in r)+'</tr>' for r in rows)+'</table></div>'
    body = '<h1>Paper7 四模型共享预算实际比较</h1><p>2026-10-21 · CPU · 40×240全格 · 训练seed仅42</p>'
    b = result['protocol']['budget']
    body += '<section class="warn"><h2>实验定位与结论边界</h2><p>本次是 '+html.escape(b['mode'])+'，不是最终论文验证；所有模型共同使用训练 '+str(b['train_n'])+' / 验证 '+str(b['validation_n'])+' / 测试 '+str(b['test_n'])+'，统一batch32、max100、patience15。实际轮数由同一固定验证混合MSE早停决定。原测试集此前已查看，本轮只在全部模型锁定后统一测试，不能称全新盲测。没有依据测试指标调参、重选架构或选择样本。</p><p>三次测试噪声重复不是三次训练。只有一个训练seed，不能作训练稳定性、统计显著性或官方最优基线的结论。随机划分不证明跨工况、跨时间或真实传感器泛化。参数量不等，计算量不等；公平性是共享数据、增强、优化器、上限及早停协议，不是等FLOP或等参数量。</p></section>'
    body += '<section><h2>协议与归一化</h2><p>所有模型输入用原训练73340的逐测点mean/std；输出同样加固定训练均值场，只使用一个训练场RMS标量 '+f'{result["protocol"]["normalization"]["output_scale_K"]:.8g}'+' K。POD基底亦来自原训练73340，是POD独有的预计算先验；pilot不意味着基底只用8192拟合。三种非POD基线没有POD解码。</p><p>POD按正交能量公式训练，加入常数截断能量后与全场MSE精确等价；其他模型直接全场MSE。DeepONet branch/trunk均训练，全网格trunk反向；FNO四层标准谱卷积，width12、modes(6,12)、真实40×240，无粗网格、无真值插值、无坐标抽样loss。FNO九标准化点按真实非等距位置双线性插值，并拼接xy。所有谱权重梯度已检查非零且有限。</p><p>训练每样本20% clean、四个sigma .05/.1/.2/.5 K各20%随机抽样，每轮重扰动；模型间同seed相同样本次序和扰动。验证每个样本固定一种情形、五组等数，而非每样本五情形，五组均值MSE等权。测试1 K和两种确定偏置不用于选模。</p><p>协议SHA256：'+result['protocol_hash']+'；脚本SHA256：'+result['protocol']['script_sha256']+'。</p></section>'
    body += '<section><h2>训练与推理资源</h2>'+table(['模型', '实标量参数', '实际轮次', '最佳轮', '验证RMSE/K', '训练总秒', 'batch1 median/ms'], [[s['model'], s['parameters_real_scalars'], s['actual_epochs'], s['best_epoch'], s['validation_mixed_field_RMSE_K'], s['elapsed_seconds'], result['inference_timing'][f'{s["model"]}_seed{s["training_seed"]}']['median_ms']] for s in result['training']])+'<p>complex参数按两个实标量；参数量不含固定mean/POD基底、插值矩阵或坐标。训练时间含每轮验证/检查点/日志；CPU推理10次warmup、50次计时，包含输入归一化、tensor转换、物理K恢复、POD解码、FNO插值/坐标，不含磁盘I/O；实现未针对极限延迟优化。</p></section>'
    for cond in next(iter(result['test'].values())):
        body += '<section><h2>测试：'+html.escape(cond)+'</h2>'+table(['模型/训练seed', '场RMSE/K', '场MAE/K', '峰温MAE/K', '峰温最大绝对误差/K', '全局最大点绝对误差/K', '噪声重复RMSE SD/K'], [[tag, m[cond]['field_RMSE_K'], m[cond]['field_MAE_K'], m[cond]['peak_temperature_MAE_K'], m[cond]['peak_temperature_max_abs_error_K'], m[cond]['global_max_point_abs_error_K'], m[cond]['noise_repeat_RMSE_std_K_ddof0']] for tag, m in result['test'].items()])+'</section>'
    body += '<section><h2>聚合方法与复现</h2><p>正sigma三次噪声以全部样本×重复合并MSE后开根号；MAE/峰温MAE按合并样本均值；最大误差取所有重复最大值。P95以合并样本计算；clean与偏置只算一次。峰温误差是 |max预测温度−max真值温度|，不是热点坐标误差。所有物理指标均使用共同测试子集完整9600点，不是全10000测试集极值。每样本指标、每噪声重复、pooled结果均有CSV。</p><p>命令：指定Python outputs/four_model_benchmark.py --run；同协议自动恢复最近完整epoch，每轮保存last、最佳时保存best；--benchmark-only仅做计算基准。若完整结果已存在，--run校验协议后直接退出，不覆盖重训。索引、固定验证噪声、规范化缓存、协议、检查点、历史和数据签名均保存。源npy以只读mmap打开。</p></section>'
    for path in image_files:
        body += '<section><h2>'+html.escape(path.stem)+'</h2><img src="data:image/png;base64,'+base64.b64encode(path.read_bytes()).decode()+'"></section>'
    body += '<section><h2>完整机器指标</h2><details><summary>展开完整JSON</summary><pre>'+html.escape(json.dumps(result, ensure_ascii=False, indent=2))+'</pre></details></section>'
    style = 'body{background:#f3f6fa;color:#223047;font:15px/1.7 system-ui,"Microsoft YaHei",sans-serif;margin:0}main{max-width:1500px;margin:auto;padding:32px}section{background:white;border:1px solid #dce3ed;border-radius:12px;padding:24px;margin:20px 0}.warn{border-left:6px solid #b67b20}table{border-collapse:collapse;width:100%;font-size:12px}td,th{padding:10px;border-bottom:1px solid #dde4ee;text-align:right}th{background:#edf2f8}.scroll{overflow:auto}img{width:100%;height:auto}pre{white-space:pre-wrap;word-break:break-all;font-size:11px}p{overflow-wrap:anywhere}'
    (OUT/'result_summary.html').write_text('<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>四模型实际比较</title><style>'+style+'</style></head><body><main>'+body+'</main></body></html>', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description='四模型真实训练与共享预算比较')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--benchmark-only', action='store_true')
    group.add_argument('--run', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    torch.set_num_interop_threads(1)
    data = load_data()
    benchpath = OUT/'compute_benchmark.json'
    if args.benchmark_only or not benchpath.exists():
        bench = benchmark(data)
    else:
        bench = json.loads(benchpath.read_text(encoding='utf-8'))
    if args.benchmark_only:
        return 0
    torch.set_num_threads(bench['selected_threads'])
    torch.use_deterministic_algorithms(True)
    protocol = make_protocol(data, bench)
    protocol['script_sha256'] = digest(Path(__file__))
    phash = hashlib.sha256(json.dumps(protocol, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    pp = OUT/'protocol.json'
    if pp.exists():
        assert json.loads(pp.read_text(encoding='utf-8')) == {'protocol_hash': phash, 'protocol': protocol}, '既有协议不同，请使用新目录，不覆盖原实验'
    else:
        dump(pp, {'protocol_hash': phash, 'protocol': protocol})
    finalpath = OUT/'metrics.json'
    if finalpath.exists():
        prev = json.loads(finalpath.read_text(encoding='utf-8'))
        if prev.get('status')=='complete':
            assert prev['protocol_hash']==phash
            log('完整实验已存在且协议吻合；不覆盖重训。')
            return 0
    cache = prepare_subsets(data, protocol, phash)
    gpu_info = None
    if shutil.which('nvidia-smi'):
        gpu_info = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv'], capture_output=True, text=True).stdout
    result = {'status': 'training', 'protocol_hash': phash, 'protocol': protocol,
              'environment': {'python': sys.version, 'executable': sys.executable, 'platform': platform.platform(),
                              'numpy': np.__version__, 'torch': str(torch.__version__), 'cuda_available': torch.cuda.is_available(),
                              'device': 'cpu', 'threads': torch.get_num_threads(), 'logical_cpu_count': os.cpu_count(), 'nvidia_smi': gpu_info},
              'training': [], 'inference_timing': {}, 'test': {}, 'test_elapsed_seconds': {}}
    dump(OUT/'metrics.json', result)
    try:
        models = []
        for seed in bench['training_seeds']:
            for name in NAMES:
                net, summary = fit(name, seed, data, cache, protocol, phash)
                models.append((name, seed, net))
                result['training'].append(summary)
                result['test'][f'{name}_seed{seed}'] = {}
                dump(OUT/'metrics.json', result)
        result['status'] = 'all_models_locked_test_evaluation'
        dump(OUT/'metrics.json', result)
        log('四模型均按验证集锁定；现在统一进行既往已暴露测试集评估。')
        test_models(models, data, cache, result)
        result['status'] = 'all_computation_complete_generating_visualizations'
        result['total_training_seconds'] = sum(v['elapsed_seconds'] for v in result['training'])
        result['elapsed_seconds_current_process'] = time.perf_counter()-START
        dump(OUT/'metrics.json', result)
        figures_and_html(result)
        result['status'] = 'complete'
        result['elapsed_seconds_current_process'] = time.perf_counter()-START
        dump(OUT/'metrics.json', result)
        # HTML完整状态与最终metrics保持一致。
        summarypath = OUT/'result_summary.html'
        text = summarypath.read_text(encoding='utf-8').replace('all_computation_complete_generating_visualizations', 'complete')
        summarypath.write_text(text, encoding='utf-8')
        files = [p for p in OUT.iterdir() if p.is_file() and p.suffix in ('.json', '.csv', '.png', '.npz', '.pt', '.html')]
        dump(OUT/'artifact_manifest.json', {'protocol_hash': phash, 'files': {p.name: {'size_bytes': p.stat().st_size, 'sha256': digest(p)} for p in files}})
        dump(OUT/'progress.json', {'status': 'complete', 'protocol_hash': phash, 'elapsed_seconds_current_process': time.perf_counter()-START})
        log(f'COMPLETE {OUT}; 总训练{result["total_training_seconds"]:.2f}s；本进程{time.perf_counter()-START:.2f}s')
        return 0
    except Exception:
        result['status'] = 'failed'
        result['error'] = traceback.format_exc()
        dump(OUT/'metrics.json', result)
        log(result['error'])
        raise


if __name__ == '__main__':
    with threadpool_limits(limits=4):
        sys.exit(main())
