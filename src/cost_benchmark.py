"""Uniform cost benchmark for the five models of the MAPNet paper.

Measures, on the same device and with the same protocol:
  * parameters actually used by the network
  * per-sample inference latency at batch 2000 (as in Table 5) and at batch 1
  * peak GPU memory during one inference pass at batch 2000
  * multiply-accumulate operations per sample
Per-epoch training time is read from the training logs written by
full_data_five_test.py / full_data_mlp_test.py, because it cannot be
re-measured without retraining.

Run:  python -B outputs/cost_benchmark.py
"""
from __future__ import annotations

import csv
import json
import platform
import statistics
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import full_data_five_test as base  # noqa: E402
import paths                        # noqa: E402

OUT = paths.RUNS / "cost"
OUT.mkdir(parents=True, exist_ok=True)

MLP_DIR = paths.RUNS / "full_data_mlp"
ORDER = ["pod16_noise", "cnn_small_noise", "mlp_small_noise",
         "deeponet_noise", "fno2d_noise"]
LABEL = {"pod16_noise": "MAPNet", "cnn_small_noise": "CNN",
         "mlp_small_noise": "MLP", "deeponet_noise": "DeepONet",
         "fno2d_noise": "FNO"}

BATCH = 2000
WARMUP = 30
REPEATS = 200
DTYPE = torch.float64          # the dtype used for all reported accuracies


def training_stats(name: str) -> dict:
    """Epochs and wall clock of the run that produced the reported weights."""
    out_dir = MLP_DIR if name == "mlp_small_noise" else base.OUT
    done = sorted(out_dir.joinpath(name).glob("stage*/done.json"))[-1]
    d = json.loads(done.read_text(encoding="utf-8"))
    return {"epochs": d["epochs"], "steps_per_epoch": d["steps_per_epoch"],
            "train_seconds": d["train_seconds_synchronized"],
            "seconds_per_epoch": d["train_seconds_synchronized"] / d["epochs"],
            "parameters_logged": d["parameters"]}


MAIN_OUT = base.OUT


def load(name: str):
    if name == "mlp_small_noise":
        base.OUT = MLP_DIR
        base.KIND[name] = "mlp_small"
    else:
        base.OUT = MAIN_OUT
    net, _ = base.load_config_net(name, "noise")
    return net.eval()


def latency(net, batch: int, dtype=DTYPE) -> dict:
    x = torch.randn(batch, 9, dtype=dtype, device="cuda")
    with torch.inference_mode():
        for _ in range(WARMUP):
            net(x)
        torch.cuda.synchronize()
        samples = []
        for _ in range(REPEATS if batch > 1 else REPEATS * 5):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            net(x)
            end.record()
            torch.cuda.synchronize()
            samples.append(start.elapsed_time(end) * 1e3)      # microseconds
    per_sample = [t / batch for t in samples]
    return {"batch": batch,
            "median_us_per_sample": statistics.median(per_sample),
            "mean_us_per_sample": statistics.fmean(per_sample),
            "median_us_per_call": statistics.median(samples)}


def peak_memory(net, batch: int = BATCH) -> float:
    x = torch.randn(batch, 9, dtype=DTYPE, device="cuda")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        net(x)
    torch.cuda.synchronize()
    mib = torch.cuda.max_memory_allocated() / 2 ** 20
    torch.cuda.reset_peak_memory_stats()
    del x
    torch.cuda.empty_cache()
    return mib


def macs(net) -> float:
    from torch.utils.flop_counter import FlopCounterMode
    x = torch.randn(1, 9, dtype=DTYPE, device="cuda")
    with torch.inference_mode():
        net(x)
    with FlopCounterMode(display=False) as counter:
        with torch.inference_mode():
            net(x)
    return counter.get_total_flops() / 2


def main() -> None:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False

    device = torch.cuda.get_device_properties(0)
    env = {
        "gpu": torch.cuda.get_device_name(0),
        "compute_capability": f"{device.major}.{device.minor}",
        "sm_count": device.multi_processor_count,
        "vram_GiB": round(device.total_memory / 2 ** 30, 1),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "os": platform.platform(),
        "cpu": platform.processor(),
        "dtype": str(DTYPE),
        "tf32": False,
    }

    rows = []
    for name in ORDER:
        net = load(name)
        st = training_stats(name)
        row = {"config": name, "label": LABEL[name],
               "parameters": sum(p.numel() for p in net.parameters()),
               "macs_per_sample": macs(net),
               "inference_peak_MiB": peak_memory(net),
               **st}
        row["latency_b2000"] = latency(net, BATCH)
        row["latency_b1"] = latency(net, 1)
        rows.append(row)
        print(f"{LABEL[name]:9s} params {row['parameters']:7d}  "
              f"MACs {row['macs_per_sample']:9.2e}  "
              f"ep {st['epochs']:3d}  s/epoch {st['seconds_per_epoch']:5.2f}  "
              f"lat(2000) {row['latency_b2000']['median_us_per_sample']:6.2f} us  "
              f"lat(1) {row['latency_b1']['median_us_per_sample']:7.2f} us  "
              f"peak {row['inference_peak_MiB']:7.1f} MiB")
        del net
        torch.cuda.empty_cache()

    (OUT / "cost.json").write_text(
        json.dumps({"environment": env, "rows": rows},
                   ensure_ascii=False, indent=2), encoding="utf-8")

    with open(OUT / "cost.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "parameters", "macs_per_sample", "epochs",
                    "seconds_per_epoch", "train_seconds",
                    "latency_b2000_us", "latency_b1_us",
                    "inference_peak_MiB"])
        for r in rows:
            w.writerow([r["label"], r["parameters"],
                        f"{r['macs_per_sample']:.4e}", r["epochs"],
                        f"{r['seconds_per_epoch']:.3f}",
                        f"{r['train_seconds']:.1f}",
                        f"{r['latency_b2000']['median_us_per_sample']:.3f}",
                        f"{r['latency_b1']['median_us_per_sample']:.3f}",
                        f"{r['inference_peak_MiB']:.1f}"])

    print("\nrelative to MAPNet")
    ref = rows[0]
    for r in rows[1:]:
        print(f"  {r['label']:9s} s/epoch x{r['seconds_per_epoch']/ref['seconds_per_epoch']:4.1f}"
              f"   total x{r['train_seconds']/ref['train_seconds']:5.1f}"
              f"   latency x{r['latency_b2000']['median_us_per_sample']/ref['latency_b2000']['median_us_per_sample']:5.1f}"
              f"   MACs x{r['macs_per_sample']/ref['macs_per_sample']:6.1f}"
              f"   mem x{r['inference_peak_MiB']/ref['inference_peak_MiB']:4.1f}")
    print("\nwritten", OUT / "cost.json")


if __name__ == "__main__":
    main()
