#!/usr/bin/env python
"""The certified blind band, on the groundwater operator.

(a) coverage against nominal, for the band each prior reports (filled) and the certified band
    built from held-out references (open). The band each prior reports tracks nominal for the oracle and
    collapses far below it for the curated prior; the certified band sits on the diagonal for both.
    Validity does not depend on the prior, which is the proposition's claim.
(b) the certified half-width divided by the one each prior reports, against the number of
    references. Both bands scale with the prior's own conditional spread, so the ratio is exactly
    the quantity a practitioner can report: how much wider the truth is on a blind direction than
    the prior claims. It is near one for the calibrated prior and far above it for the curated one,
    and it settles after a handful of references.

Reads ``results/darcy_certify.npz`` (produced by ``scripts/darcy_certify.py``); writes
``figures/fig_darcy_certify.pdf``. Seconds, CPU.

Run:  python scripts/fig_darcy_certify.py
"""
from __future__ import annotations

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from resolvability.download import REPO, ensure  # noqa: E402
from resolvability.style import PALETTE, apply_paper_style  # noqa: E402
apply_paper_style(); plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm"})
EM, MAD = PALETTE["em"], PALETTE["mad"]

CACHE = "results/darcy_certify.npz"
OUT = os.path.join(REPO, "figures/fig_darcy_certify.pdf")


def main():
    z = np.load(ensure(CACHE), allow_pickle=True)
    lev, ks, seeds = z["levels"], z["ks"], z["seeds"]
    keys = list(z["keys"]); vals = z["vals"]

    fig, ax = plt.subplots(1, 2, figsize=(3.30, 1.41))

    # ---- (a) coverage against nominal ----------------------------------------------------------
    ax[0].plot([0, 1], [0, 1], color="0.55", lw=0.6, ls=":", zorder=0)
    # The two certified curves and the oracle's reported curve all land on the diagonal. A small
    # opposite dodge along the nominal axis, plus distinct marker sizes, keeps every series
    # readable without moving any plotted value off its level.
    for tag, col, dodge, ms in (("oracle", EM, -0.006, 3.6), ("curated", MAD, +0.006, 2.7)):
        ax[0].plot(lev + dodge, z[f"{tag}_cert"].mean(0), "s--", ms=ms, lw=0.8, color=col,
                   markerfacecolor="none", markeredgewidth=0.6, label=f"{tag}, certified")
    for tag, col in (("oracle", EM), ("curated", MAD)):
        ax[0].plot(lev, z[f"{tag}_raw"].mean(0), "o-", ms=2.2, lw=0.8, color=col,
                   label=f"{tag}, reported")
    ax[0].set_xlabel("nominal credible level", fontsize=8, labelpad=1)
    ax[0].set_ylabel("blind\ncoverage", fontsize=8, labelpad=2)
    ax[0].set_xlim(0.45, 1.0); ax[0].set_ylim(-0.03, 1.03)
    ax[0].set_xticks([0.5, 0.7, 0.9]); ax[0].set_yticks([0.0, 0.5, 1.0])
    ax[0].tick_params(labelsize=7, length=2.5, pad=2.0)
    ax[0].set_title("(a)  blind coverage", fontsize=8.0, loc="left", pad=3.0)

    # ---- (b) certified width, relative to what each prior reports -----------------------------
    for tag, col, mk in (("oracle", EM, "o"), ("curated", MAD, "s")):
        w = np.array([[dict(zip(keys, vals[:, 1]))[f"{tag}_s{s}_k{k}"] for k in ks] for s in seeds])
        for row in w:                                              # one line per training seed
            ax[1].plot(ks, row, mk + "-", ms=2.2, lw=0.75, color=col, alpha=0.85,
                       markerfacecolor=("none" if tag == "curated" else col), markeredgewidth=0.55)
    ax[1].axhline(1.0, color="0.55", lw=0.6, ls=":", zorder=0)
    ax[1].set_xscale("log"); ax[1].set_yscale("log")
    ax[1].set_xticks(list(ks)); ax[1].set_xticklabels([str(int(k)) for k in ks])
    ax[1].set_xlabel("reference truths", fontsize=8, labelpad=1)
    ax[1].set_ylabel("certified /\nreported", fontsize=8, labelpad=2)
    ax[1].set_yticks([1, 10]); ax[1].set_yticklabels(["1", "10"])
    ax[1].tick_params(labelsize=7, which="both", length=2.5, pad=2.0)
    ax[1].tick_params(which="minor", length=1.5)
    ax[1].set_title("(b)  certified width", fontsize=8.0, loc="left", pad=3.0)
    ax[1].set_ylim(0.72, 34)                       # room for the labels above each family
    for tag, col in (("curated", MAD), ("oracle", EM)):
        w = np.array([[dict(zip(keys, vals[:, 1]))[f"{tag}_s{s}_k{k}"] for k in ks] for s in seeds])
        ax[1].text(ks[0], w[:, 0].max() * 1.4, tag, fontsize=7, color=col, ha="left")

    fig.subplots_adjust(left=0.152, right=0.975, bottom=0.3302, top=0.8647, wspace=0.52)
    # Panel (a)'s key, as one row under both panels, each band word followed by its two series.
    h = dict(zip(*ax[0].get_legend_handles_labels()[::-1]))
    r = fig.canvas.get_renderer()
    x, y = 0.012, 0.07
    for band in ("certified", "reported"):
        t = fig.text(x, y, band + ":", fontsize=7, ha="left", va="center")
        x = t.get_window_extent(r).x1 / fig.bbox.width + 0.008
        leg = fig.legend([h[f"oracle, {band}"], h[f"curated, {band}"]], ["oracle", "curated"],
                         fontsize=7, frameon=False, ncol=2, loc="center left",
                         bbox_to_anchor=(x, y), handlelength=1.3, handletextpad=0.35,
                         columnspacing=0.45, borderpad=0.0, borderaxespad=0.0)
        fig.canvas.draw()
        x = leg.get_window_extent(r).x1 / fig.bbox.width + 0.038
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", pad_inches=0.02, dpi=300)
    print("saved", os.path.relpath(OUT, REPO))


if __name__ == "__main__":
    main()
