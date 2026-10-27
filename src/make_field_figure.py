# -*- coding: utf-8 -*-
"""Reconstruction figure: truth, five model predictions and their error maps."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

import paths

ROOT = Path(__file__).resolve().parent
DATA = paths.RUNS / "field_case" / "case.npz"
if not DATA.exists():                      # fall back to the archived case
    DATA = paths.RESULTS / "field_case.npz"
OUT = paths.FIG
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Nimbus Roman", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 7.5,
    "pdf.fonttype": 42,
    "axes.linewidth": 0.4,
})

z = np.load(DATA, allow_pickle=False)
truth = z["truth"].reshape(40, 240)
names = [str(n) for n in z["names"]]
label = {"pod16_noise": "MAPNet", "cnn_small_noise": "CNN", "mlp_small_noise": "MLP",
         "deeponet_noise": "DeepONet", "fno2d_noise": "FNO"}
order = ["pod16_noise", "cnn_small_noise", "mlp_small_noise", "deeponet_noise", "fno2d_noise"]
rmse = {n: float(np.sqrt(((z[f"pred_{n}"] - z["truth"]) ** 2).mean())) for n in names}
shape = {n: float(np.std(z[f"pred_{n}"] - z["truth"])) for n in names}

W, H = 137.0, 86.0
fig = plt.figure(figsize=(W / 25.4, H / 25.4), dpi=600)

x0, x1 = 0.135, 0.505           # field column
y1, y0 = 0.945, 0.205           # top of first row, bottom of last row
rows = 6
hgap = 0.006
hh = (y1 - y0 - (rows - 1) * hgap) / rows
dx_err = 0.025

vmin, vmax = truth.min(), truth.max()
# shape error: remove the case mean (the anchor offset, common to all models)
errs = {n: (z[f"pred_{n}"] - z["truth"]) for n in names}
bias = {n: float(errs[n].mean()) for n in names}
emax = max(np.percentile(np.abs(errs[n] - bias[n]), 99) for n in names)
emax = float(np.ceil(emax * 1000) / 1000)

ext = (0.0, 590.0, 0.0, 97.67)
axes_f, axes_e = [], []
for i in range(rows):
    y = y1 - (i + 1) * hh - i * hgap
    af = fig.add_axes([x0, y, x1 - x0, hh])
    ae = fig.add_axes([x1 + dx_err, y, x1 - x0, hh])
    axes_f.append(af)
    axes_e.append(ae)

for ax in axes_f + axes_e:
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.3)

# truth row
axes_f[0].imshow(truth, origin="lower", extent=ext, cmap="inferno",
                 vmin=vmin, vmax=vmax, aspect="auto")
fig.text(x0 - 0.008, y1 - hh / 2, "truth", ha="right", va="center", fontsize=7.5)
axes_e[0].axis("off")
axes_e[0].text(0.5, 0.55, "error with the case mean removed", ha="center", va="center",
               fontsize=6.2, color="0.45", transform=axes_e[0].transAxes)
axes_e[0].text(0.5, 0.30, f"common offset $\\approx${np.mean(list(bias.values()))*1e3:+.0f} mK",
               ha="center", va="center", fontsize=6.0, color="0.55",
               transform=axes_e[0].transAxes)

for i, n in enumerate(order, start=1):
    pred = z[f"pred_{n}"].reshape(40, 240)
    err = pred - truth - bias[n]
    axes_f[i].imshow(pred, origin="lower", extent=ext, cmap="inferno",
                     vmin=vmin, vmax=vmax, aspect="auto")
    im = axes_e[i].imshow(err, origin="lower", extent=ext, cmap="RdBu_r",
                          vmin=-emax, vmax=emax, aspect="auto")
    fig.text(x0 - 0.008, y1 - (i + 1) * hh - i * hgap + hh / 2, label[n],
             ha="right", va="center", fontsize=7.5)
    fig.text(x1 + dx_err + (x1 - x0) + 0.008,
             y1 - (i + 1) * hh - i * hgap + hh / 2,
             f"{rmse[n]*1e3:.1f} / {shape[n]*1e3:.1f}", ha="left", va="center", fontsize=6.2)

fig.text(x1 + dx_err + (x1 - x0) + 0.008, y1 - hh / 2, "RMSE / shape\n(mK)",
         ha="left", va="center", fontsize=6.0)

# axis labels on the bottom row
bottom = axes_f[-1]
bottom.set_xticks([0, 295, 590])
bottom.set_xticklabels(["0", "295", "590"], fontsize=6)
bottom.tick_params(length=1.6, width=0.4, pad=1.2)
bottom.set_xlabel("x (mm)", fontsize=6.6, labelpad=1.5)


# colour bars
cax1 = fig.add_axes([x0, 0.085, x1 - x0, 0.017])
cax2 = fig.add_axes([x1 + dx_err, 0.085, x1 - x0, 0.017])
cb1 = fig.colorbar(ScalarMappable(norm=Normalize(vmin, vmax), cmap="inferno"),
                   cax=cax1, orientation="horizontal")
cb1.set_label("$T$ (K)", fontsize=6.6, labelpad=1.5)
cb1.ax.tick_params(labelsize=6, length=1.6, width=0.4, pad=1)
cb1.outline.set_linewidth(0.3)
cb2 = fig.colorbar(ScalarMappable(norm=Normalize(-emax, emax), cmap="RdBu_r"),
                   cax=cax2, orientation="horizontal")
cb2.set_label("prediction $-$ truth, case mean removed (K; clipped)", fontsize=6.6, labelpad=1.5)
cb2.ax.tick_params(labelsize=6, length=1.6, width=0.4, pad=1)
cb2.outline.set_linewidth(0.3)

fig.savefig(OUT / "fields.pdf")
fig.savefig(OUT / "fields.png", dpi=600)
print("saved", OUT / "fields.pdf")
