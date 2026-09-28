#!/usr/bin/env python
"""Trained priors report the blind widths the theory predicts, in the two-dimensional example.

The blind spread, on the fiber x_R = 0, of a Gaussian mixture fit by EM (squares) and of a HINT flow
at three seeds (circles), trained on each of the regularizer's two archives (red) and on the truths
(blue). Dashes are the same quantity for the exact dequantized training law. On the single-best
archive that law is the dequantization itself, widened through the slope of rho's regression line:
the smoothing floor.

Reads ``results/toy_estimation.npz`` (produced by ``scripts/toy_estimation.py``) and refuses it if
it was generated from other parameters. Writes ``figures/fig_toy_money.pdf``, sized to be included
at natural size. Seconds, CPU.

Run:  python scripts/fig_toy_money.py
"""
from __future__ import annotations

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import PathCollection  # noqa: E402

from resolvability.download import REPO, ensure  # noqa: E402
from resolvability.style import PALETTE, apply_paper_style  # noqa: E402
from resolvability.toy.estimation import CACHE, check_cache  # noqa: E402

apply_paper_style()
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm"})
EM, MAD, GREEN = PALETTE["em"], PALETTE["mad"], PALETTE["truth"]
OUT = os.path.join(REPO, "figures/fig_toy_money.pdf")
GROUPS = (("post", MAD, "posterior-sample\narchive"),
          ("sb", MAD, "single-best\narchive"),
          ("oracle", EM, "truths\n(oracle)"))
FIBER = "c1"


def check_label_clearance(fig, ax, labels, min_pt: float) -> None:
    """Every label must clear every scatter marker by at least ``min_pt`` points.

    The labels are anchored to the marks, so a regenerated cache moves them; this catches a label
    that has moved onto a marker.
    """
    fig.canvas.draw()
    px_per_pt = fig.dpi / 72.0
    renderer = fig.canvas.get_renderer()
    marks = []
    for coll in ax.collections:
        if not isinstance(coll, PathCollection) or len(coll.get_offsets()) == 0:
            continue
        n = len(coll.get_offsets())
        sizes = np.broadcast_to(coll.get_sizes(), (n,))
        lws = np.broadcast_to(coll.get_linewidths(), (n,))
        for (x, y), s, lw in zip(coll.get_offsets(), sizes, lws):
            cx, cy = ax.transData.transform((x, y))
            marks.append((cx, cy, (np.sqrt(s) / 2 + lw / 2) * px_per_pt))
    for lab in labels:
        bb = lab.get_window_extent(renderer)
        for cx, cy, r in marks:
            dx = max(bb.x0 - cx, 0, cx - bb.x1)
            dy = max(bb.y0 - cy, 0, cy - bb.y1)
            gap_pt = (np.hypot(dx, dy) - r) / px_per_pt
            if gap_pt < min_pt:
                raise RuntimeError(f"label '{lab.get_text()}' is {gap_pt:.2f} pt from a marker")


def main():
    z = np.load(ensure(CACHE))
    check_cache(z, FIBER)
    seeds = [int(s) for s in z["seeds"]]
    fig, ax = plt.subplots(figsize=(3.48, 1.65))
    truth_sd = float(z[f"anchor_oracle_sd_{FIBER}"])
    ax.axhline(truth_sd, color=GREEN, lw=0.7, ls=":", zorder=1)
    truth_label = ax.annotate("truth", (-0.45, truth_sd), xytext=(0, 3.0),
                              textcoords="offset points", fontsize=7, color=GREEN)
    for i, (arch, col, _) in enumerate(GROUPS):
        ax.hlines(float(z[f"anchor_{arch}_sd_{FIBER}"]), i - 0.22, i + 0.30, color="k",
                  lw=0.8, ls="--", zorder=2)
        ax.scatter([i - 0.11], [float(z[f"gmm_{arch}_sd_{FIBER}"])], marker="s", s=24,
                   facecolor="white", edgecolor=col, linewidth=0.9, zorder=4)
        for j, s in enumerate(seeds):
            ax.scatter([i + 0.08 + 0.055 * j], [float(z[f"hint_{arch}_sd_{FIBER}_s{s}"])],
                       marker="o", s=17, facecolor="white", edgecolor=col, linewidth=0.8,
                       zorder=4)
    x_sq, y_gmm = -0.11, float(z[f"gmm_post_sd_{FIBER}"])
    x_c0 = 0.08
    y_low = min(float(z[f"hint_post_sd_{FIBER}_s{s}"]) for s in seeds)
    y_post = float(z[f"anchor_post_sd_{FIBER}"])
    r_sq, r_c = np.sqrt(24) / 2, np.sqrt(17) / 2             # marker radii, points
    labels = [
        ax.annotate("theory", (0.30, y_post), xytext=(3.0, 0), textcoords="offset points",
                    fontsize=7, color="k", ha="left", va="center"),
        ax.annotate("GMM", (x_sq, y_gmm), xytext=(0, r_sq + 2.5), textcoords="offset points",
                    fontsize=7, color="0.3", ha="center", va="bottom"),
        ax.annotate(f"flow ({len(seeds)} seeds)", (x_c0, y_low), xytext=(-r_c, -(r_c + 2.5)),
                    textcoords="offset points", fontsize=7, color="0.3", ha="left", va="top"),
    ]
    labels.append(ax.text(0.55, 0.42, "curated", fontsize=7, color=MAD, ha="center",
                          va="center"))
    labels.append(truth_label)
    ax.set_yscale("log")
    ax.set_yticks([0.3, 1.0, 3.0]); ax.set_yticklabels(["0.3", "1", "3"])
    ylim = (0.22, 7.2)
    plotted = [float(z[f"{kind}_{a}_sd_{FIBER}"]) for a, _, _ in GROUPS
               for kind in ("anchor", "gmm")]
    plotted += [float(z[f"hint_{a}_sd_{FIBER}_s{s}"]) for a, _, _ in GROUPS for s in seeds]
    if not all(ylim[0] < v < ylim[1] for v in plotted):
        raise RuntimeError("a plotted value falls outside the y-limits")
    ax.set_ylim(*ylim); ax.set_xlim(-0.5, 2.55)
    ax.set_xticks(range(len(GROUPS)))
    ax.set_xticklabels([g[2] for g in GROUPS], fontsize=7)
    ax.set_xlabel("what the prior was trained on", fontsize=8, labelpad=2)
    ax.set_ylabel("blind spread of\nthe trained prior", fontsize=8, labelpad=1)
    ax.tick_params(labelsize=7)
    check_label_clearance(fig, ax, labels, min_pt=2.0)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", pad_inches=0.02, dpi=300,
                metadata={"CreationDate": None})
    plt.close(fig)
    print(f"saved {os.path.relpath(OUT, REPO)}")


if __name__ == "__main__":
    main()
