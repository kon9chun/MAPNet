# -*- coding: utf-8 -*-
"""Full-field error of the six trained models at sigma = 0.02, 0.04, 0.06, 0.08 K.

Uses the same three noise seeds as every other table in the paper, and the same
generator convention (one standard-normal draw per seed, scaled by sigma), so the
0.02/0.04/0.06 columns reproduce the earlier fine sweep exactly.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import full_data_five_test as base          # noqa: E402
import paths                                # noqa: E402

OUT = paths.RUNS / "noise_grid"
OUT.mkdir(parents=True, exist_ok=True)
SIGMAS = (0.02, 0.04, 0.06, 0.08)
SEEDS = base.TEST_SEEDS


def main() -> None:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    old = np.load(base.OLDSUB, allow_pickle=False)
    test_ids = old["test_reserved"]
    y = np.load(base.SOURCE / "ydata.npy", mmap_mode="r", allow_pickle=False)
    s = np.load(base.SOURCE / "sp.npy", mmap_mode="r", allow_pickle=False)

    truth = np.empty((len(test_ids), base.D), dtype=np.float32)
    for i in range(0, len(test_ids), 2048):
        part = test_ids[i:i + 2048]
        truth[i:i + len(part)] = np.asarray(y[part], dtype=np.float32).reshape(-1, base.D)
    truth_t = torch.tensor(truth, dtype=torch.float64, device="cuda")

    sensors = torch.tensor(np.asarray(s[test_ids], dtype=np.float64),
                           dtype=torch.float64, device="cuda")
    bases = {}
    for sd in SEEDS:
        g = torch.Generator(device="cuda").manual_seed(sd)
        bases[sd] = torch.randn(sensors.shape, generator=g, device="cuda", dtype=torch.float64)

    rows, per_seed = [], []
    jobs = [(n, base.OUT) for n in base.ORDER]
    jobs.append(("mlp_small_noise", paths.RUNS / "full_data_mlp"))
    for name, out_dir in jobs:
        base.OUT = out_dir
        if name == "mlp_small_noise":
            base.KIND[name] = "mlp_small"
        net, ck = base.load_config_net(name, "noise")
        for sigma in SIGMAS:
            vals = []
            for sd in SEEDS:
                preds = []
                with torch.inference_mode():
                    for i in range(0, len(sensors), 500):
                        s_in = sensors[i:i + 500] + bases[sd][i:i + 500] * sigma
                        preds.append(net(s_in).double().cpu())
                pred = torch.cat(preds)
                err = pred - truth_t.cpu()
                vals.append({'rmse': float(err.square().mean().sqrt()),
                             'mae': float(err.abs().mean()),
                             'peak': float((pred.max(1).values - truth_t.cpu().max(1).values).abs().mean())})
            rmse = np.array([v['rmse'] for v in vals])
            for sd, v in zip(SEEDS, vals):
                per_seed.append({'config': name, 'sigma_K': sigma,
                                 'noise_seed': sd, 'RMSE_K': v['rmse'],
                                 'MAE_K': v['mae'], 'peak_MAE_K': v['peak']})
            rows.append({'config': name, 'label': base.LABEL.get(name, name),
                         'sigma_K': sigma, 'n_test': int(len(test_ids)),
                         'n_seed': len(SEEDS),
                         'RMSE_K': float(rmse.mean()), 'RMSE_std_seed_K': float(rmse.std(ddof=1)),
                         'MAE_K': float(np.mean([v['mae'] for v in vals])),
                         'peak_MAE_K': float(np.mean([v['peak'] for v in vals]))})
            print(f"{name:16s} sigma {sigma:.2f}  RMSE {rmse.mean():.5f} ± {rmse.std(ddof=1):.5f} K")
        del net, ck
        torch.cuda.empty_cache()

    with open(OUT / "metrics.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    with open(OUT / "metrics_per_seed.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(per_seed[0]))
        w.writeheader()
        w.writerows(per_seed)

    order = ["pod16_noise", "cnn_small_noise", "mlp_small_noise", "deeponet_noise", "fno2d_noise"]
    print("\n sigma " + "".join(f"{n[:9]:>12}" for n in order))
    for sigma in SIGMAS:
        line = f"{sigma:6.2f} "
        for n in order:
            v = next(r for r in rows if r['config'] == n and r['sigma_K'] == sigma)
            line += f"{v['RMSE_K']:12.5f}"
        print(line)


if __name__ == "__main__":
    main()
