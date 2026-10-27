# -*- coding: utf-8 -*-
"""Pick one test case and dump fields for the reconstruction figure."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import full_data_five_test as base          # noqa: E402
import paths                                # noqa: E402

OUT = paths.RUNS / "field_case"
OUT.mkdir(parents=True, exist_ok=True)
SIGMA, NOISE_SEED, PICK_SEED = 0.05, 31001, 20261026


def main() -> None:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    old = np.load(base.OLDSUB, allow_pickle=False)
    test_ids = old["test_reserved"]
    y = np.load(base.SOURCE / "ydata.npy", mmap_mode="r", allow_pickle=False)
    s = np.load(base.SOURCE / "sp.npy", mmap_mode="r", allow_pickle=False)
    z = np.load(base.OUT / "preprocessing.npz", allow_pickle=False)
    mean = z["mean"].copy()

    truth = np.empty((len(test_ids), base.D), dtype=np.float32)
    for i in range(0, len(test_ids), 2048):
        part = test_ids[i:i + 2048]
        truth[i:i + len(part)] = np.asarray(y[part], dtype=np.float32).reshape(-1, base.D)

    sensors = torch.tensor(np.asarray(s[test_ids], dtype=np.float64),
                           dtype=torch.float64, device="cuda")
    g = torch.Generator(device="cuda").manual_seed(NOISE_SEED)
    noise = torch.randn(sensors.shape, generator=g, device="cuda",
                        dtype=torch.float64) * SIGMA

    truth_t = torch.tensor(truth, dtype=torch.float64, device="cuda")
    preds, per_case = {}, {}
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
                chunks.append(net(sensors[i:i + 500] + noise[i:i + 500]).double().cpu())
        pred = torch.cat(chunks).numpy()
        preds[name] = pred
        per_case[name] = np.sqrt(((pred - truth) ** 2).mean(1))
        print(f"{name:16s} mean per-case RMSE {per_case[name].mean():.6f} K")
        del net, ck
        torch.cuda.empty_cache()

    span = truth.max(1) - truth.min(1)
    q = float(sys.argv[1]) if len(sys.argv) > 1 else 50.0
    target = float(np.percentile(span, q))
    k = int(np.argmin(np.abs(span - target)))
    rnd = int(np.random.default_rng(PICK_SEED).integers(0, len(test_ids)))
    print(f"\nspan p{q:g} = {target:.3f} K; random draw would be {rnd} "
          f"(span {span[rnd]:.3f} K)")
    print(f"picked index {k} (test id {int(test_ids[k])}), span {span[k]:.3f} K")
    for n in preds:
        print(f"  {n:16s} RMSE {per_case[n][k]:.5f} K")

    np.savez_compressed(OUT / "case.npz", truth=truth[k],
                        **{f"pred_{n}": preds[n][k] for n in preds},
                        index=k, test_id=int(test_ids[k]),
                        sigma_K=SIGMA, noise_seed=NOISE_SEED,
                        per_case_rmse=np.array([per_case[n][k] for n in preds]),
                        names=np.array(list(preds)))


if __name__ == "__main__":
    main()
