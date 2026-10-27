"""全量数据（73,340 训练样本）五配置训练与统一测试。

配置
  pod16_noise      POD16 直接 sigma=0.05 K 噪声训练
  pod16_staged     先干净预训练（300轮），再从干净最佳权重用 lr=1e-4、sigma=0.05 微调（300轮）
  deeponet_noise   DeepONet 直接 sigma=0.05
  cnn_small_noise  CNN（105,489 参数，与 POD16 同量级）直接 sigma=0.05
  fno2d_noise      FNO 直接 sigma=0.05

数据沿用 pod_check 的 73,340 / 10,000 / 10,000 划分；验证取旧 1,000 子集、测试取旧 2,000 子集，
因此与既有 8,192 子集实验的唯一差别是训练样本数，便于直接对照。

选模：干净验证与 sigma=0.05 含噪验证各保留一套 best；主表用含噪 best，另报干净 best。

运行
  python -B outputs/full_data_five_test.py --run                 顺序跑全部配置
  python -B outputs/full_data_five_test.py --resume              从断点继续
  python -B outputs/full_data_five_test.py --only pod16_noise --run
  python -B outputs/full_data_five_test.py --test-only           只做统一测试与汇总
"""
import argparse
import copy
import gc
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

import paths
import spatial_decoder_clean_test as m
import low_noise_param_matched_test as pm

ROOT = Path(__file__).resolve().parent
OUT = paths.RUNS / 'full_data_five'
SOURCE = paths.DATA
SPLIT = paths.DATA / 'split_indices.npz'
OLDSUB = paths.DATA / 'subset_indices.npz'

H, W, D = 40, 240, 9600
BATCH = 256
RANK = 16
SIGMA = 0.05
TRAIN_SEED = 42
MAX_EPOCHS = 300
PATIENCE = 40
VAL_NOISE_SEED = 77001
TEST_SEEDS = (31001, 31002, 31003)
TEST_SIGMAS = (0.02, 0.05, 0.10, 0.20, 0.50)

ORDER = ('pod16_noise', 'deeponet_noise', 'pod16_staged', 'cnn_small_noise', 'fno2d_noise')
KIND = {'pod16_noise': 'pod16_clean', 'pod16_staged': 'pod16_clean',
        'deeponet_noise': 'deeponet', 'cnn_small_noise': 'cnn_small',
        'fno2d_noise': 'fno2d'}
LABEL = {'pod16_noise': 'POD16 直接sigma0.05',
         'pod16_staged': 'POD16 干净预训练+sigma0.05微调',
         'deeponet_noise': 'DeepONet sigma0.05',
         'cnn_small_noise': 'CNN(105K) sigma0.05',
         'fno2d_noise': 'FNO sigma0.05'}

LOGFILE = None


def log(text):
    line = f'[{time.strftime("%H:%M:%S")}] {text}'
    print(line, flush=True)
    if LOGFILE is not None:
        with open(LOGFILE, 'a', encoding='utf-8') as fh:
            fh.write(line + '\n')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def require(cond, msg):
    if not cond:
        raise RuntimeError(msg)


def write_json(path, obj):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding='utf-8')
    os.replace(tmp, path)


def save_pt(path, obj):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    torch.save(obj, tmp)
    os.replace(tmp, path)


def cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def canonical_hash(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def config_dir(name):
    return OUT / name


def stage_dir(name, index):
    return config_dir(name) / f'stage{index}'


def real_params(net):
    return sum(p.numel() * (2 if p.is_complex() else 1) for p in net.parameters())


# ----------------------------------------------------------------------------- 数据
def build_pod(train_ids, y, s):
    """全量训练残差场上的中心化随机化 POD（sketch=96，两次幂迭代，float64）。"""
    chunk = 4096
    total = np.zeros(D, np.float64)
    for i in range(0, len(train_ids), chunk):
        ids = train_ids[i:i + chunk]
        a = np.asarray(y[ids], dtype=np.float64).reshape(-1, D)
        b = np.asarray(s[ids], dtype=np.float64).mean(1)
        total += (a - b[:, None]).sum(0)
    mean = total / len(train_ids)
    log(f'POD: 全量均值完成，n={len(train_ids)}')

    sketch = np.random.default_rng(20261022).normal(size=(D, 96))
    for _ in range(2):
        z = np.zeros((D, 96), np.float64)
        for i in range(0, len(train_ids), chunk):
            ids = train_ids[i:i + chunk]
            a = np.asarray(y[ids], dtype=np.float64).reshape(-1, D)
            b = np.asarray(s[ids], dtype=np.float64).mean(1)
            x = a - b[:, None] - mean
            z += x.T @ (x @ sketch)
        sketch = np.linalg.qr(z, mode='reduced')[0]
    log('POD: 幂迭代完成')

    c = np.zeros((96, 96), np.float64)
    for i in range(0, len(train_ids), chunk):
        ids = train_ids[i:i + chunk]
        a = np.asarray(y[ids], dtype=np.float64).reshape(-1, D)
        b = np.asarray(s[ids], dtype=np.float64).mean(1)
        x = a - b[:, None] - mean
        z = x @ sketch
        c += z.T @ z
    eig, vec = np.linalg.eigh(c)
    order = np.argsort(eig)[::-1]
    basis = sketch @ vec[:, order[:RANK]]
    basis /= np.linalg.norm(basis, axis=0, keepdims=True)
    for j in range(basis.shape[1]):
        if basis[np.abs(basis[:, j]).argmax(), j] < 0:
            basis[:, j] *= -1
    log(f'POD: 基底完成 {basis.shape}，正交误差 {np.abs(basis.T @ basis - np.eye(RANK)).max():.2e}')
    return mean, basis


def prepare(resume):
    require(torch.cuda.is_available(), '必须使用 CUDA')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(4)

    split = np.load(SPLIT, allow_pickle=False)
    old = np.load(OLDSUB, allow_pickle=False)
    train_ids = split['train']
    val_ids = old['validation']
    test_ids = old['test_reserved']
    require(len(train_ids) == 73340 and len(val_ids) == 1000 and len(test_ids) == 2000,
            '数据划分规模不符')
    require(not (set(train_ids) & set(val_ids)) and not (set(train_ids) & set(test_ids))
            and not (set(val_ids) & set(test_ids)), '划分存在交集')

    y = np.load(SOURCE / 'ydata.npy', mmap_mode='r', allow_pickle=False)
    s = np.load(SOURCE / 'sp.npy', mmap_mode='r', allow_pickle=False)
    require(y.shape == (93340, H, W) and s.shape == (93340, 9), '源数组形状不符')

    cache = OUT / 'preprocessing.npz'
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        mean, basis = z['mean'].copy(), z['basis'].copy()
        norm = {k: float(z[k]) for k in ('residual_sensor_scale', 'b_mean', 'b_std')}
        scale = float(z['scale'])
        require(np.array_equal(z['train_ids'], train_ids), '缓存索引不一致')
        log(f'预处理缓存命中：{cache.name}')
    else:
        mean, basis = build_pod(train_ids, y, s)
        st = np.asarray(s[train_ids], dtype=np.float64)
        bt = st.mean(1)
        scatter = float((st - bt[:, None]).__pow__(2).mean())
        norm = {'residual_sensor_scale': math.sqrt(scatter),
                'b_mean': float(bt.mean()),
                'b_std': float(bt.std())}
        sq = 0.0
        for i in range(0, len(train_ids), 4096):
            ids = train_ids[i:i + 4096]
            a = np.asarray(y[ids], dtype=np.float64).reshape(-1, D)
            b = np.asarray(s[ids], dtype=np.float64).mean(1)
            sq += float(((a - b[:, None] - mean) ** 2).sum())
        scale = math.sqrt(sq / (len(train_ids) * D))
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, mean=mean, basis=basis, scale=scale,
                 train_ids=train_ids, split_sha=np.array(hashlib.sha256(
                     np.concatenate([train_ids, val_ids, test_ids]).tobytes()).hexdigest()),
                 **norm)
        log(f'预处理完成：residual scale={scale:.6f} K，sensor scale={norm["residual_sensor_scale"]:.6f} K')

    mu = torch.tensor(mean, dtype=torch.float64, device='cuda')
    scale_t = torch.tensor(scale, dtype=torch.float32, device='cuda')

    def to_centered(ids):
        out = torch.empty((len(ids), D), dtype=torch.float32, device='cuda')
        for i in range(0, len(ids), 4096):
            part = ids[i:i + 4096]
            a = np.asarray(y[part], dtype=np.float64).reshape(-1, D)
            out[i:i + len(part)] = torch.tensor(a - mean, dtype=torch.float32, device='cuda')
        return out

    log('装载训练集到 GPU')
    centered_train = to_centered(train_ids)
    sensors_train = torch.tensor(np.asarray(s[train_ids], dtype=np.float64),
                                 dtype=torch.float64, device='cuda')
    centered_val = to_centered(val_ids)
    sensors_val = torch.tensor(np.asarray(s[val_ids], dtype=np.float64),
                               dtype=torch.float64, device='cuda')
    centered_test = to_centered(test_ids)
    sensors_test = torch.tensor(np.asarray(s[test_ids], dtype=np.float64),
                                dtype=torch.float64, device='cuda')
    log(f'显存占用约 {torch.cuda.memory_allocated() / 2**30:.2f} GB')

    data = {'mean': mean, 'basis': basis, 'scale': scale, 'norm': norm,
            'mu': mu, 'scale_t': scale_t,
            'centered': {'train': centered_train, 'validation': centered_val, 'test': centered_test},
            'sensors': {'train': sensors_train, 'validation': sensors_val, 'test': sensors_test},
            'ids': {'train': train_ids, 'validation': val_ids, 'test': test_ids},
            'split_sha': hashlib.sha256(np.concatenate([train_ids, val_ids, test_ids]).tobytes()).hexdigest()}
    return data


def val_noise(data):
    g = torch.Generator(device='cuda').manual_seed(VAL_NOISE_SEED)
    return torch.randn(data['sensors']['validation'].shape, generator=g, device='cuda',
                       dtype=torch.float64) * SIGMA


def val_mse(net, data, sensors, noise=None, batch=500):
    """物理场 MSE（K^2），float64 计算。"""
    centered = data['centered']['validation']
    total, count = 0.0, 0
    with torch.inference_mode():
        for i in range(0, len(centered), batch):
            s = sensors[i:i + batch] if noise is None else sensors[i:i + batch] + noise[i:i + batch]
            pred = net(s)
            truth = centered[i:i + batch].double() + data['mu']
            total += float((pred - truth).square().sum())
            count += pred.numel()
    return total / count


# ----------------------------------------------------------------------------- 训练
def new_net(kind, data):
    torch.manual_seed(TRAIN_SEED)
    random.seed(TRAIN_SEED)
    net = pm.Net(kind, data['mean'], data['scale'], data['norm'], data['basis']).cuda()
    return net


def checkpoint_base(name, net, data, protocol_hash, stage_index, sigma, lr):
    return {'format_version': 4, 'config_name': name, 'kind': net.kind,
            'structure_config': pm.CONFIG[net.kind], 'state_dict': cpu_state(net),
            'residual_mean': data['mean'], 'residual_scale': data['scale'],
            'normalization': data['norm'], 'basis': data['basis'],
            'batch_size': BATCH, 'inference_batch_size': 500,
            'split_sha256': data['split_sha'], 'script_sha256': m.sha(__file__),
            'protocol_hash': protocol_hash, 'train_seed': TRAIN_SEED,
            'train_samples': int(len(data['ids']['train'])), 'rank': RANK,
            'stage_index': stage_index, 'train_sigma_K': sigma, 'learning_rate': lr,
            'train_noise': 'all nine channels independent Gaussian, no clean fraction'
                           if sigma > 0 else 'clean sensors',
            'selection': 'noise best = min sigma0.05 validation physical MSE; '
                         'clean best = min clean validation physical MSE'}


def train_stage(name, data, phash, stage_index, sigma, lr, init_state=None,
                max_epochs=MAX_EPOCHS, patience=PATIENCE, resume=False):
    sdir = stage_dir(name, stage_index)
    sdir.mkdir(parents=True, exist_ok=True)
    hist_path = sdir / 'history.json'
    last_path = sdir / 'last.pt'
    noise_best_path = sdir / 'noise_best.pt'
    clean_best_path = sdir / 'clean_best.pt'
    done_path = sdir / 'done.json'

    if done_path.exists() and resume:
        log(f'{name} stage{stage_index} 已完成，跳过')
        return read_json(done_path)

    net = new_net(KIND[name], data)
    if init_state is not None:
        net.load_state_dict(init_state, strict=True)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=10, min_lr=1e-6)
    history = read_json(hist_path) if (resume and hist_path.exists()) else []
    best_noise, best_clean = float('inf'), float('inf')
    if resume and last_path.exists():
        last = torch.load(last_path, map_location='cuda', weights_only=False)
        require(last['protocol_hash'] == phash and last['stage_index'] == stage_index,
                'last 与当前协议不符')
        net.load_state_dict(last['state_dict'])
        opt.load_state_dict(last['optimizer'])
        sched.load_state_dict(last['scheduler'])
        best_noise, best_clean = last['best_noise'], last['best_clean']
        log(f'{name} stage{stage_index} 恢复自第 {last["epoch"]} 轮')
    elif not resume and hist_path.exists():
        hist_path.unlink()

    n = len(data['ids']['train'])
    steps = (n + BATCH - 1) // BATCH
    start = history[-1]['epoch'] if history else 0
    wait = 0
    vnoise = val_noise(data)
    train_seconds, start_time = 0.0, time.time()

    for epoch in range(start + 1, max_epochs + 1):
        t0 = time.time()
        net.train()
        g = torch.Generator(device='cuda').manual_seed(TRAIN_SEED * 100000 + epoch)
        perm = torch.randperm(n, generator=g, device='cuda')
        run, seen = torch.zeros((), device='cuda'), 0
        for i in range(0, n, BATCH):
            ix = perm[i:i + BATCH]
            s_clean = data['sensors']['train'][ix]
            centered = data['centered']['train'][ix]
            if sigma > 0:
                noise = torch.randn(s_clean.shape, generator=g, device='cuda',
                                    dtype=torch.float64) * sigma
                s = s_clean + noise
            else:
                s = s_clean
            b = s.mean(1)
            x, _ = net.features(s)
            target = (centered - b.float()[:, None]) / data['scale_t']
            pred = net.normalized_residual(x)
            loss = (pred - target).square().mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            run = run + loss.detach() * len(ix)
            seen += len(ix)
        run = float(run)
        require(math.isfinite(run), '本轮 loss 非有限')
        train_seconds += time.time() - t0

        net.eval()
        clean_score = val_mse(net, data, data['sensors']['validation'])
        noise_score = val_mse(net, data, data['sensors']['validation'], vnoise)
        monitor = noise_score if sigma > 0 else clean_score
        sched.step(monitor)
        lr_now = float(opt.param_groups[0]['lr'])

        improved = False
        if noise_score < best_noise:
            best_noise = noise_score
            improved = True
            ck = checkpoint_base(name, net, data, phash, stage_index, sigma, lr)
            ck.update(epoch=epoch, validation_noise_MSE_K2=noise_score,
                      validation_clean_MSE_K2=clean_score, selection_metric='noise')
            save_pt(noise_best_path, ck)
        if clean_score < best_clean:
            best_clean = clean_score
            ck = checkpoint_base(name, net, data, phash, stage_index, sigma, lr)
            ck.update(epoch=epoch, validation_noise_MSE_K2=noise_score,
                      validation_clean_MSE_K2=clean_score, selection_metric='clean')
            save_pt(clean_best_path, ck)

        history.append({'epoch': epoch, 'train_normalized_MSE': run / max(seen, 1),
                        'train_field_RMSE_K': math.sqrt(run / max(seen, 1)) * data['scale'],
                        'validation_clean_RMSE_K': math.sqrt(clean_score),
                        'validation_noise_MSE_K2': noise_score,
                        'validation_noise_RMSE_K': math.sqrt(noise_score),
                        'lr': lr_now, 'epoch_seconds': time.time() - t0})
        write_json(hist_path, history)

        state = checkpoint_base(name, net, data, phash, stage_index, sigma, lr)
        state.update(epoch=epoch, optimizer=opt.state_dict(), scheduler=sched.state_dict(),
                     best_noise=best_noise, best_clean=best_clean,
                     history_length=len(history))
        save_pt(last_path, state)

        wait = 0 if improved else wait + 1
        if epoch % 10 == 0 or epoch == 1:
            log(f'{name} s{stage_index} ep{epoch}/{max_epochs} '
                f'train {math.sqrt(run / max(seen, 1)) * data["scale"]:.5f}K '
                f'valClean {math.sqrt(clean_score):.5f}K valNoise {math.sqrt(noise_score):.5f}K '
                f'lr {lr_now:.2e} {time.time() - t0:.1f}s/轮')
        if wait >= patience:
            log(f'{name} s{stage_index} 第 {epoch} 轮早停（连续 {patience} 轮无改善）')
            break

    info = {'config': name, 'stage_index': stage_index, 'kind': KIND[name],
            'sigma_K': sigma, 'initial_lr': lr, 'epochs': len(history),
            'best_noise_validation_MSE_K2': best_noise,
            'best_noise_validation_RMSE_K': math.sqrt(best_noise),
            'best_clean_validation_MSE_K2': best_clean,
            'best_clean_validation_RMSE_K': math.sqrt(best_clean),
            'best_noise_epoch': min(history, key=lambda r: r['validation_noise_MSE_K2'])['epoch'],
            'best_clean_epoch': min(history, key=lambda r: r['validation_clean_RMSE_K'])['epoch'],
            'parameters': real_params(net),
            'steps_per_epoch': steps, 'train_samples': n,
            'train_seconds_synchronized': train_seconds,
            'wall_seconds': time.time() - start_time}
    write_json(done_path, info)
    del net, opt, sched
    gc.collect()
    torch.cuda.empty_cache()
    return info


def run_config(name, data, phash, resume):
    cdir = config_dir(name)
    cdir.mkdir(parents=True, exist_ok=True)
    final = cdir / 'done.json'
    if final.exists() and resume:
        log(f'{name} 已完成，跳过')
        return read_json(final)

    if name == 'pod16_staged':
        s0 = train_stage(name, data, phash, 0, 0.0, 1e-3, resume=resume)
        pretrained = stage_dir(name, 0) / 'clean_best.pt'
        if not pretrained.exists():
            pretrained = stage_dir(name, 0) / 'noise_best.pt'
        require(pretrained.exists(), f'缺少预训练权重 {pretrained}')
        ck = torch.load(pretrained, map_location='cuda', weights_only=False)
        require(ck['kind'] == 'pod16_clean', '预训练权重类型不符')
        log(f'{name} 阶段0 完成：干净验证 RMSE {s0["best_clean_validation_RMSE_K"]:.6f} K '
            f'（第 {s0["best_clean_epoch"]} 轮）')
        s1 = train_stage(name, data, phash, 1, SIGMA, 1e-4,
                         init_state=ck['state_dict'], resume=resume)
        info = {'config': name, 'stages': [s0, s1],
                'pretrain_seconds': s0['train_seconds_synchronized'],
                'finetune_seconds': s1['train_seconds_synchronized']}
    else:
        s0 = train_stage(name, data, phash, 0, SIGMA, 1e-3, resume=resume)
        info = {'config': name, 'stages': [s0],
                'pretrain_seconds': 0.0, 'finetune_seconds': s0['train_seconds_synchronized']}
    write_json(final, info)
    return info


# ----------------------------------------------------------------------------- 测试
def metrics(pred, truth):
    err = pred - truth
    peak_pred = pred.max(1).values
    peak_true = truth.max(1).values
    return {'MSE_K2': float(err.square().mean()),
            'RMSE_K': float(err.square().mean().sqrt()),
            'MAE_K': float(err.abs().mean()),
            'peak_MAE_K': float((peak_pred - peak_true).abs().mean()),
            'maxpoint_K': float(err.abs().max()),
            'signed_bias_K': float(err.mean())}


def load_config_net(name, which):
    path = stage_dir(name, 1 if name == 'pod16_staged' else 0) / f'{which}_best.pt'
    require(path.exists(), f'缺少权重 {path}')
    ck = torch.load(path, map_location='cuda', weights_only=False)
    net = pm.Net(ck['kind'], ck['residual_mean'], ck['residual_scale'],
                 ck['normalization'], ck['basis']).cuda()
    net.load_state_dict(ck['state_dict'], strict=True)
    net.eval()
    return net, ck


def run_test(data):
    centered = data['centered']['test']
    sensors = data['sensors']['test']
    truth = centered.double() + data['mu']
    truth_np = truth.cpu().numpy()
    cases = [('clean', None, 0.0)]
    for sg in TEST_SIGMAS:
        for sd in TEST_SEEDS:
            cases.append((f'noise_{sg:g}', sd, sg))
    noises = {}
    for tag, sd, sg in cases:
        if sd is None:
            continue
        g = torch.Generator(device='cuda').manual_seed(sd)
        noises[(tag, sd)] = torch.randn(sensors.shape, generator=g, device='cuda',
                                        dtype=torch.float64) * sg

    rows = []
    for name in ORDER:
        for which in ('noise', 'clean'):
            net, ck = load_config_net(name, which)
            for tag, sd, sg in cases:
                nz = noises.get((tag, sd))
                preds = []
                with torch.inference_mode():
                    for i in range(0, len(sensors), 500):
                        s = sensors[i:i + 500] if nz is None else sensors[i:i + 500] + nz[i:i + 500]
                        preds.append(net(s).cpu().numpy())
                pred = np.concatenate(preds)
                require(np.isfinite(pred).all(), f'{name} 预测非有限')
                row = {'config': name, 'label': LABEL[name], 'selection': which,
                       'parameters': real_params(net), 'scenario': tag,
                       'sigma_K': sg, 'noise_seed': sd, 'n_test': len(sensors),
                       **metrics(torch.tensor(pred), torch.tensor(truth_np))}
                rows.append(row)
                log(f'测试 {name}/{which} {tag} seed={sd} '
                    f'RMSE {row["RMSE_K"]:.6f} K peakMAE {row["peak_MAE_K"]:.6f} K')
            del net, ck
            gc.collect()
            torch.cuda.empty_cache()

    write_json(OUT / 'test_metrics_per_case.json', rows)
    import csv
    with open(OUT / 'test_metrics.csv', 'w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def summarize(rows):
    per = {}
    for r in rows:
        key = (r['config'], r['selection'], r['scenario'])
        per.setdefault(key, []).append(r)
    table = {}
    for (name, which, scenario), group in per.items():
        table.setdefault(name, {}).setdefault(which, {})[scenario] = {
            'RMSE_K': float(np.mean([g['RMSE_K'] for g in group])),
            'RMSE_std_seed_K': float(np.std([g['RMSE_K'] for g in group], ddof=1))
            if len(group) > 1 else 0.0,
            'MAE_K': float(np.mean([g['MAE_K'] for g in group])),
            'peak_MAE_K': float(np.mean([g['peak_MAE_K'] for g in group])),
            'maxpoint_K': float(np.mean([g['maxpoint_K'] for g in group])),
            'signed_bias_K': float(np.mean([g['signed_bias_K'] for g in group])),
            'noise_repetitions': len(group)}
    ranking = []
    for scenario in ['clean'] + [f'noise_{s:g}' for s in TEST_SIGMAS]:
        ranked = sorted(((table[n]['noise'][scenario]['RMSE_K'], n) for n in ORDER))
        for rank, (value, n) in enumerate(ranked, 1):
            ranking.append({'scenario': scenario, 'rank': rank, 'config': n,
                            'label': LABEL[n], 'RMSE_K': value,
                            'relative_to_best_pct': 100 * (value / ranked[0][0] - 1)})
    return table, ranking


def main():
    global LOGFILE
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--only', default=None)
    parser.add_argument('--test-only', action='store_true')
    args = parser.parse_args()
    require(args.run or args.resume or args.test_only, '需要 --run/--resume/--test-only')

    OUT.mkdir(parents=True, exist_ok=True)
    LOGFILE = OUT / 'run.log'
    log(f'启动：{sys.argv[1:]}')

    data = prepare(resume=args.resume)
    protocol = {'train_samples': int(len(data['ids']['train'])),
                'validation_samples': int(len(data['ids']['validation'])),
                'test_samples': int(len(data['ids']['test'])),
                'train_sigma_K': SIGMA, 'rank': RANK, 'batch_size': BATCH,
                'max_epochs': MAX_EPOCHS, 'patience': PATIENCE,
                'train_seed': TRAIN_SEED, 'val_noise_seed': VAL_NOISE_SEED,
                'test_seeds': list(TEST_SEEDS), 'test_sigmas_K': list(TEST_SIGMAS),
                'split_sha256': data['split_sha'], 'script_sha256': m.sha(__file__),
                'configs': {n: {'kind': KIND[n], 'stages': (
                    [{'sigma_K': 0.0, 'lr': 1e-3}, {'sigma_K': SIGMA, 'lr': 1e-4}]
                    if n == 'pod16_staged' else [{'sigma_K': SIGMA, 'lr': 1e-3}])}
                    for n in ORDER}}
    core = {k: v for k, v in protocol.items() if k != 'script_sha256'}
    phash = canonical_hash(core)
    ppath = OUT / 'protocol.json'
    if ppath.exists():
        old = read_json(ppath)
        old_core = {k: v for k, v in old.items()
                    if k not in ('script_sha256', 'protocol_hash', 'code_patches')}
        require(old_core == core, '协议字段变化，拒绝继续')
        patches = old.get('code_patches', [])
        if old.get('script_sha256') != protocol['script_sha256']:
            patches.append({'time': time.strftime('%F %T'),
                            'from_script_sha256': old.get('script_sha256'),
                            'to_script_sha256': protocol['script_sha256'],
                            'reason': '修复sigma=0时未保存clean_best导致两阶段读取失败；'
                                      '协议数据/结构/优化字段均未改动'})
            log('记录代码补丁：' + patches[-1]['reason'])
        old.update(protocol_fields=core, script_sha256=protocol['script_sha256'],
                   protocol_hash=phash, code_patches=patches)
        write_json(ppath, old)
    else:
        protocol['protocol_hash'] = phash
        write_json(ppath, protocol)
    write_json(OUT / 'status.json', {'stage': 'training', 'started': time.strftime('%F %T')})

    if not args.test_only:
        targets = [args.only] if args.only else list(ORDER)
        for name in targets:
            require(name in ORDER, f'未知配置 {name}')
            t0 = time.time()
            run_config(name, data, phash, resume=args.resume)
            log(f'{name} 完成，用时 {time.time() - t0:.0f} 秒')
            write_json(OUT / 'status.json', {'stage': 'training', 'last_completed': name,
                                             'updated': time.strftime('%F %T')})

    write_json(OUT / 'status.json', {'stage': 'testing', 'updated': time.strftime('%F %T')})
    rows = run_test(data)
    table, ranking = summarize(rows)
    info = {n: read_json(config_dir(n) / 'done.json') for n in ORDER
            if (config_dir(n) / 'done.json').exists()}
    summary = {'completed': len(info) == len(ORDER), 'protocol_hash': phash,
               'protocol': protocol, 'training': info, 'table': table,
               'ranking': ranking, 'rows': len(rows),
               'elapsed_local': time.strftime('%F %T')}
    write_json(OUT / 'summary.json', summary)
    write_json(OUT / 'status.json', {'stage': 'complete', 'updated': time.strftime('%F %T')})
    log('全部完成')


if __name__ == '__main__':
    main()
