"""Minimal inference demo for MAPNet.

Loads the released checkpoint, reconstructs the 40 x 240 temperature field from
the nine embedded readings of 200 held-out cases, and reports the same error
metrics used in the paper. No training, no access to the 3.6 GB field array:
the demo fields are shipped in ``data/demo_fields.npy``.

    python src/demo_inference.py                 # sigma = 0.05 K, seed 31001
    python src/demo_inference.py --sigma 0       # clean readings
    python src/demo_inference.py --plot          # also save a three-panel figure

Runs on CPU in a few seconds; uses the GPU when one is available.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

import paths
from spatial_decoder_clean_test import FieldModel

H, W = 40, 240


def load_net(checkpoint):
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    net = FieldModel(
        ck["kind"], ck.get("channels"), ck["basis"], ck["residual_mean"],
        ck["residual_scale"], ck["normalization"],
    )
    net.load_state_dict(ck["state_dict"], strict=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return net.to(device).eval(), ck, device


def predict(net, sensors, device, batch=500):
    out = []
    with torch.inference_mode():
        for i in range(0, len(sensors), batch):
            s = torch.as_tensor(sensors[i:i + batch], dtype=torch.float64, device=device)
            out.append(net(s).cpu().numpy())
    return np.concatenate(out).reshape(-1, H, W)


def metrics(pred, truth):
    err = pred - truth
    return {
        "MSE_K2": float((err ** 2).mean()),
        "RMSE_K": float(np.sqrt((err ** 2).mean())),
        "MAE_K": float(np.abs(err).mean()),
        "peak_MAE_K": float(np.abs(pred.max(1).max(1) - truth.max(1).max(1)).mean()),
        "maxpoint_K": float(np.abs(err).max()),
        "signed_bias_K": float(err.mean()),
    }


def plot(truth, pred, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    err = pred - truth
    lim = float(np.percentile(np.abs(err), 99))
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 1.55), constrained_layout=True)
    for ax, data, title in zip(axes, (truth, pred, err), ("truth", "MAPNet", "error")):
        if title == "error":
            im = ax.imshow(data, cmap="RdBu_r", vmin=-lim, vmax=lim, origin="lower", aspect="auto")
            fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02).set_label("K", fontsize=7)
        else:
            vmin, vmax = truth.min(), truth.max()
            im = ax.imshow(data, cmap="inferno", vmin=vmin, vmax=vmax, origin="lower", aspect="auto")
            fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02).set_label("K", fontsize=7)
        ax.set_title(title, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.savefig(path, dpi=300)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default=str(paths.ROOT / "checkpoints" / "mapnet_noise_best.pt"))
    ap.add_argument("--sigma", type=float, default=0.05, help="sensing-noise standard deviation in K")
    ap.add_argument("--seed", type=int, default=31001, help="noise seed used in the paper")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    fields = np.load(paths.DATA / "demo_fields.npy", mmap_mode="r")
    sensors = np.load(paths.DATA / "demo_sensors.npy").astype(np.float64)
    meta = json.loads((paths.DATA / "demo_meta.json").read_text(encoding="utf-8"))
    assert len(sensors) == len(fields) == meta["n_samples"]

    if args.sigma > 0:
        g = torch.Generator().manual_seed(args.seed)
        sensors = sensors + args.sigma * torch.randn(
            sensors.shape, generator=g, dtype=torch.float64).numpy()

    net, ck, device = load_net(args.checkpoint)
    params = sum(p.numel() for p in net.parameters())
    pred = predict(net, sensors, device)
    truth = np.asarray(fields, dtype=np.float64)

    m = metrics(pred, truth)
    print(f"checkpoint : {args.checkpoint}")
    print(f"device     : {device}")
    print(f"parameters : {params:,}")
    print(f"cases      : {len(sensors)}  (sigma = {args.sigma:g} K, seed = {args.seed})")
    print(f"RMSE       : {m['RMSE_K']:.6f} K")
    print(f"MAE        : {m['MAE_K']:.6f} K")
    print(f"peak MAE   : {m['peak_MAE_K']:.6f} K")
    print(f"max point  : {m['maxpoint_K']:.6f} K")
    print(f"bias       : {m['signed_bias_K']:+.6f} K")

    out_dir = paths.RUNS / "demo"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps({
        "checkpoint": args.checkpoint, "sigma_K": args.sigma, "seed": args.seed,
        "n_cases": int(len(sensors)), "parameters": int(params),
        "device": device, **m}, indent=1), encoding="utf-8")
    if args.plot:
        idx = int(np.argmax(np.ptp(truth.reshape(len(truth), -1), axis=1)))
        plot(truth[idx], pred[idx], out_dir / "demo_field.png")
        print(f"figure     : {out_dir / 'demo_field.png'}  (case {idx}, widest spread)")
    print(f"metrics    : {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
