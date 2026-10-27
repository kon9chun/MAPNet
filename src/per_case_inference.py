# -*- coding: utf-8 -*-
"""Per-sample test errors at sigma = 0.05 K for the five models.

Loads the frozen checkpoints, adds one fixed noise realisation to the nine
readings of the 2,000 test cases, and stores the per-sample MSE/RMSE so that
paired statistics (ECDF, bootstrap CI) can be computed without retraining.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import full_data_five_test as base          # noqa: E402  (constants + loaders)
import paths                                # noqa: E402

OUT = paths.RUNS / "per_case"
OUT.mkdir(parents=True, exist_ok=True)
SIGMA = 0.05
NOISE_SEED = 31001                          # first of the three reported seeds


def main() -> None:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    split = np.load(base.SPLIT, allow_pickle=False)
    old = np.load(base.OLDSUB, allow_pickle=False)
    test_ids = old["test_reserved"]
    y = np.load(base.SOURCE / "ydata.npy", mmap_mode="r", allow_pickle=False)
    s = np.load(base.SOURCE / "sp.npy", mmap_mode="r", allow_pickle=False)

    z = np.load(base.OUT / "preprocessing.npz", allow_pickle=False)
    mean = z["mean"].copy()
    scale = float(z["scale"])

    truth = np.empty((len(test_ids), base.D), dtype=np.float32)
    for i in range(0, len(test_ids), 2048):
        part = test_ids[i:i + 2048]
        truth[i:i + len(part)] = np.asarray(y[part], dtype=np.float32).reshape(-1, base.D)
    sensors = torch.tensor(np.asarray(s[test_ids], dtype=np.float64),
                           dtype=torch.float64, device="cuda")
    truth_t = torch.tensor(truth - mean, dtype=torch.float64, device="cuda") + \
        torch.tensor(mean, dtype=torch.float64, device="cuda")

    g = torch.Generator(device="cuda").manual_seed(NOISE_SEED)
    noise = torch.randn(sensors.shape, generator=g, device="cuda",
                        dtype=torch.float64) * SIGMA
    s_noisy = sensors + noise

    errs = {}
    jobs = [(n, base.OUT) for n in base.ORDER]
    jobs.append(("mlp_small_noise", paths.RUNS / "full_data_mlp"))
    for name, out_dir in jobs:
        base.OUT = out_dir
        if name == "mlp_small_noise":
            base.KIND[name] = "mlp_small"
        net, ck = base.load_config_net(name, "noise")
        chunks = []
        with torch.inference_mode():
            for i in range(0, len(sensors), 500):
                chunks.append(net(s_noisy[i:i + 500]).double().cpu())
        pred = torch.cat(chunks)
        per_case_mse = (pred - truth_t.cpu()).square().mean(1).numpy()
        errs[name] = per_case_mse
        print(f"{name:16s} global RMSE {np.sqrt(per_case_mse.mean()):.6f} K "
              f"| median per-case {np.sqrt(np.median(per_case_mse)):.6f} K")
        del net, ck
        torch.cuda.empty_cache()

    order = list(errs)
    np.savez(OUT / "per_case_mse.npz",
             names=np.array(order),
             mse=np.stack([errs[n] for n in order]),
             test_ids=test_ids, sigma_K=SIGMA, noise_seed=NOISE_SEED)

    a, c = errs["pod16_noise"], errs["cnn_small_noise"]
    rng = np.random.default_rng(20261026)
    n = len(a)
    dif = np.empty(4000)
    rel = np.empty(4000)
    for k in range(4000):
        idx = rng.integers(0, n, n)
        ra, rc = np.sqrt(a[idx].mean()), np.sqrt(c[idx].mean())
        dif[k] = ra - rc
        rel[k] = ra / rc - 1.0
    summary = {
        "sigma_K": SIGMA, "noise_seed": NOISE_SEED, "n_test": int(n),
        "global_RMSE_K": {k: float(np.sqrt(v.mean())) for k, v in errs.items()},
        "MAPNet_minus_CNN_K": {
            "point": float(np.sqrt(a.mean()) - np.sqrt(c.mean())),
            "ci95": [float(np.percentile(dif, 2.5)), float(np.percentile(dif, 97.5))],
            "p_MAPNet_better": float((dif < 0).mean()),
        },
        "MAPNet_vs_CNN_relative": {
            "point": float(np.sqrt(a.mean()) / np.sqrt(c.mean()) - 1.0),
            "ci95": [float(np.percentile(rel, 2.5)), float(np.percentile(rel, 97.5))],
        },
        "share_of_cases_MAPNet_better": float((a < c).mean()),
        "median_per_case_RMSE_K": {k: float(np.sqrt(np.median(v))) for k, v in errs.items()},
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
