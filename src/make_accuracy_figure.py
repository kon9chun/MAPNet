# -*- coding: utf-8 -*-
"""Figure for Section 4.1: (a) ECDF of per-case RMSE with zoom, (b) paired difference."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import paths

ROOT = Path(__file__).resolve().parent
DATA = paths.RUNS / "per_case" / "per_case_mse.npz"
if not DATA.exists():                      # fall back to the archived per-case errors
    DATA = paths.RESULTS / "per_case_mse.npz"
OUT = paths.FIG
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Nimbus Roman", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "pdf.fonttype": 42,
    "axes.linewidth": 0.6,
})

z = np.load(DATA, allow_pickle=False)
names = [str(n) for n in z["names"]]
rmse = np.sqrt(z["mse"])
idx = {n: i for i, n in enumerate(names)}

labels = {"pod16_noise": "MAPNet", "cnn_small_noise": "CNN",
          "mlp_small_noise": "MLP", "deeponet_noise": "DeepONet",
          "fno2d_noise": "FNO"}
order = ["pod16_noise", "cnn_small_noise", "mlp_small_noise",
         "deeponet_noise", "fno2d_noise"]
style = {
    "pod16_noise":     dict(color="#1f4e79", lw=1.6, ls="-"),
    "cnn_small_noise": dict(color="#c0504d", lw=1.1, ls="--"),
    "mlp_small_noise": dict(color="#8064a2", lw=1.0, ls=(0, (6, 1, 1, 1))),
    "deeponet_noise":  dict(color="#7f7f7f", lw=1.0, ls=(0, (3, 1, 1, 1))),
    "fno2d_noise":     dict(color="#a6a6a6", lw=1.0, ls=":"),
}

W, H = 137.0, 52.0
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(W / 25.4, H / 25.4), dpi=600,
                               gridspec_kw=dict(width_ratios=[1.0, 1.0], wspace=0.26))
fig.subplots_adjust(left=0.072, right=0.995, bottom=0.175, top=0.93)

# ---------------------------------------------------------------- panel (a): ECDF
for n in order:
    v = np.sort(rmse[idx[n]])
    y = np.arange(1, len(v) + 1) / len(v) * 100
    st = style[n]
    ax1.plot(v, y, color=st["color"], lw=st["lw"], ls=st["ls"], label=labels[n])

ax1.set_xscale("log")
ax1.set_xlim(0.004, 0.09)
ax1.set_ylim(0, 100)
ax1.set_xticks([0.005, 0.01, 0.02, 0.05])
ax1.set_xticklabels(["0.005", "0.01", "0.02", "0.05"])
ax1.set_xlabel("per-case RMSE (K)", labelpad=2)
ax1.set_ylabel("cumulative share of cases (%)", labelpad=2)
ax1.grid(True, which="major", color="0.88", lw=0.4)
ax1.set_axisbelow(True)
ax1.legend(loc="upper left", frameon=False, fontsize=6.2, handlelength=2.4,
           borderaxespad=0.3, labelspacing=0.15,
           bbox_to_anchor=(0.0, 0.90))
ax1.text(0.03, 0.99, "(a)", transform=ax1.transAxes, fontsize=8, va="top")

# zoom on the bulk of the distribution
axin = ax1.inset_axes([0.52, 0.10, 0.44, 0.40])
axin.set_facecolor("white")
for n in order:
    v = np.sort(rmse[idx[n]])
    y = np.arange(1, len(v) + 1) / len(v) * 100
    st = style[n]
    axin.plot(v, y, color=st["color"], lw=st["lw"] * 0.8, ls=st["ls"])
axin.set_xlim(0.009, 0.020)
axin.set_ylim(35, 80)
axin.set_xticks([0.010, 0.015, 0.020])
axin.set_xticklabels(["0.010", "0.015", "0.020"], fontsize=6.0)
axin.set_yticks([40, 60, 80])
axin.tick_params(labelsize=6.0, length=2, width=0.5, pad=1.2)
axin.grid(True, color="0.9", lw=0.3)
axin.set_axisbelow(True)

# ---------------------------------------------------------------- panel (b): paired
a = rmse[idx["pod16_noise"]]
c = rmse[idx["cnn_small_noise"]]
d = (a - c) * 1e3

ax2.axhline(0.0, color="0.35", lw=0.7, zorder=2)
ax2.scatter(c, d, s=3.6, color="#1f4e79", alpha=0.28, linewidths=0, zorder=3)

q = np.quantile(c, np.linspace(0, 1, 9))
xc, yc = [], []
for lo, hi in zip(q[:-1], q[1:]):
    m = (c >= lo) & (c <= hi)
    if m.sum() > 10:
        xc.append(np.median(c[m]))
        yc.append(np.mean(d[m]))
ax2.plot(xc, yc, color="#c0504d", lw=1.3, marker="o", ms=2.8, zorder=5)

ax2.set_xscale("log")
ax2.set_xlim(0.004, 0.09)
ax2.set_xticks([0.005, 0.01, 0.02, 0.05])
ax2.set_xticklabels(["0.005", "0.01", "0.02", "0.05"])
ax2.set_xlabel("CNN per-case RMSE (K)", labelpad=2)
ax2.set_ylabel("MAPNet $-$ CNN (mK)", labelpad=2)
ax2.grid(True, which="major", color="0.88", lw=0.4)
ax2.set_axisbelow(True)
ax2.text(0.03, 0.99, "(b)", transform=ax2.transAxes, fontsize=8, va="top")

fig.savefig(OUT / "accuracy.pdf")
fig.savefig(OUT / "accuracy.png", dpi=600)
print("saved")
