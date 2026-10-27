# -*- coding: utf-8 -*-
"""Draw the MAPNet architecture / concept diagram (vector PDF for the paper).

Flow: nine readings -> anchor split -> residual MLP -> 16 POD coefficients
-> frozen POD decoder, where the anchor is added back -> 40x240 field.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle

import paths

DATA = paths.DATA
OUT = paths.FIG
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Nimbus Roman", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "pdf.fonttype": 42,
})

BLUE = "#1f4e79"
GREY = "#6f6f6f"
LBLUE = "#e9f0f8"
LGREY = "#efefef"

W, H = 150.0, 54.0
fig = plt.figure(figsize=(W / 25.4, H / 25.4), dpi=600)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.set_aspect("equal")
ax.axis("off")

YC = 30.0          # pipeline centre line


def box(x0, y0, w, h, fc="white", ec="black", lw=0.8, r=1.2, z=2):
    ax.add_patch(FancyBboxPatch((x0, y0), w, h,
                                boxstyle=f"round,pad=0,rounding_size={r}",
                                linewidth=lw, edgecolor=ec, facecolor=fc, zorder=z))


def arrow(p0, p1, color="black", lw=0.8, z=3, ms=7):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=ms,
                                 linewidth=lw, color=color, zorder=z,
                                 shrinkA=0, shrinkB=0))


def label(x, y, s, **kw):
    kw.setdefault("ha", "center")
    kw.setdefault("va", "center")
    kw.setdefault("fontsize", 7.6)
    ax.text(x, y, s, zorder=7, **kw)


# ---------------------------------------------------------------- 1. sensors
box(3.0, 20.0, 18.0, 20.0)
for i in range(3):
    for j in range(3):
        ax.add_patch(Circle((7.0 + i * 5.0, 25.5 + j * 5.0), 0.95,
                            facecolor=BLUE, edgecolor="black", lw=0.4, zorder=5))
label(12.0, 45.6, "nine readings", fontsize=7.2)
label(12.0, 42.4, r"$\mathbf{T}\in\mathbb{R}^{9}$", fontsize=8.2)
label(12.0, 16.8, "sensor noise $\\sigma$", fontsize=6.6, color=GREY)

arrow((21.2, YC), (24.6, YC))

# ---------------------------------------------------------------- 2. anchoring
box(24.8, 21.0, 21.0, 18.0)
label(35.3, 34.2, r"$b=\frac{1}{9}\sum_{i=1}^{9}T_i$", fontsize=9.2)
label(35.3, 27.6, r"$\mathbf{T}-b\mathbf{1}$", fontsize=8.4)

arrow((46.0, YC), (49.4, YC))

# ---------------------------------------------------------------- 3. residual MLP
box(49.6, 18.0, 28.0, 24.0, fc=LBLUE, ec=BLUE)
# fully connected layers, centred in the box
xs = [53.7, 59.9, 66.9, 73.5]
counts = [4, 7, 7, 4]
spreads = [10.5, 13.6, 13.6, 10.5]
yc_net = 29.6
pos = []
for x, n, sp in zip(xs, counts, spreads):
    ys = [yc_net + (j - (n - 1) / 2.0) * sp / (n - 1) for j in range(n)]
    pos.append([(x, y) for y in ys])
for a, b in zip(pos[:-1], pos[1:]):
    for (x0, y0) in a:
        for (x1, y1) in b:
            ax.plot([x0, x1], [y0, y1], color=BLUE, lw=0.22, alpha=0.55, zorder=4)
for layer in pos:
    for (x, y) in layer:
        ax.add_patch(Circle((x, y), 0.48, facecolor="white", edgecolor=BLUE,
                            lw=0.5, zorder=6))
label(63.6, 39.4, "residual MLP", fontsize=8, color=BLUE, fontweight="bold")
label(63.6, 14.6, "trainable, 102,544 parameters", fontsize=6.6, color=BLUE)
label(63.6, 46.4, "learned from data", fontsize=6.6, color=BLUE)
ax.plot([49.6, 77.6], [44.6, 44.6], color=BLUE, lw=0.5, zorder=4)

arrow((77.8, YC), (81.2, YC))

# ---------------------------------------------------------------- 4. coefficients
for k in range(16):
    ax.add_patch(Rectangle((81.6 + k * 0.98, YC - 1.6), 0.9, 3.2,
                           facecolor=LBLUE, edgecolor=BLUE, lw=0.4, zorder=5))
label(89.5, YC + 4.4, r"$\mathbf{a}\in\mathbb{R}^{16}$", fontsize=7.6, color=BLUE)

arrow((97.8, YC), (101.2, YC))

# ---------------------------------------------------------------- 5. POD decoder
box(101.4, 21.0, 23.0, 18.0, fc=LGREY)
label(112.9, 35.4, "POD decoder", fontsize=8, fontweight="bold")
label(112.9, 30.2, r"$b+\mu_R+s_R\,U\mathbf{a}$", fontsize=8.6)
label(112.9, 24.6, "16 modes, frozen", fontsize=6.6, color=GREY)
label(112.9, 46.4, "computed once, then frozen", fontsize=6.6, color=GREY)
ax.plot([101.4, 124.4], [44.6, 44.6], color=GREY, lw=0.5, zorder=4)

# anchor bypass: from the bottom of the anchor box into the decoder
ax.plot([35.3, 35.3], [20.8, 11.5], color="black", lw=0.8, zorder=3)
ax.plot([35.3, 112.9], [11.5, 11.5], color="black", lw=0.8, zorder=3)
arrow((112.9, 11.5), (112.9, 20.6))
label(72.0, 8.9, r"the anchor $b$ by-passes the network and is added back unchanged",
      fontsize=6.6)

# ---------------------------------------------------------------- 6. field
arrow((124.6, YC), (125.8, YC))
y = np.load(DATA / "demo_fields.npy", mmap_mode="r")[0].astype(float).reshape(40, 240)
FW, FH = 22.0, 22.0 * 40.0 / 240.0          # true 40x240 aspect ratio
ax.imshow(y, extent=(126.0, 126.0 + FW, YC - FH / 2, YC + FH / 2), origin="lower",
          cmap="inferno", aspect="auto", zorder=4)
ax.add_patch(Rectangle((126.0, YC - FH / 2), FW, FH, fill=False,
                       edgecolor="black", lw=0.6, zorder=5))
label(125.6, YC + 5.4, "temperature field", fontsize=7.4, ha="left")
label(125.6, YC - 5.2, r"$\hat{T}(\xi)$,  $40\times240$", fontsize=6.8, color=GREY, ha="left")

# enlarge every label so that, after scaling to \linewidth in the manuscript,
# the smallest annotation still prints at about 7 pt
for t in ax.texts:
    t.set_fontsize(t.get_fontsize() * 1.12)

fig.savefig(OUT / "architecture.pdf")
fig.savefig(OUT / "architecture.png", dpi=600)
print("saved")
