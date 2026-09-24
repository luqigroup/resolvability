#!/usr/bin/env python
"""The deep end of one survey, and what each prior claims about the operator's own blind end.

Left: the deep end of the section at true physical proportions, full lateral extent, on the same
shared grayscale the teaser (``fig_hero.py``) uses -- truth beside the two posterior means, so a
reader can see that both priors put reflectors where the record barely reaches.

Right: what each prior claims, ordered by the operator's illumination rather than by depth. Every
pixel below the water mute is ranked by ``diag(A^T A)_j = ||A e_j||^2``, the energy the receivers
return from a unit reflector there, and the two reported spreads are read in equal-count bins of
that rank. The ranking uses the operator alone, no truth and no prior.

Two earlier versions of the right-hand panel were wrong, and are worth recording so they are not
tried again. Mean traces at three lateral positions: the wrong QUANTITY, since both posterior means
carry the reflectors and the difference between the priors is in the spread, not the amplitude. A
reach-against-depth profile: the wrong COORDINATE, since at a fixed depth the reach varies across
the section by decades -- the left edge is dark below about 1.3 km while the right edge is not --
so a laterally reduced profile summarizes a quantity that has no lateral summary. Ranking by
illumination is the fix: it puts every pixel where the operator says it belongs.

Reads ``results/seismic_dps_recon.npz`` and ``results/seismic_illum_diag.npz`` (the Hutchinson
probe of ``diag(A^T A)``, produced by ``scripts/seismic_m0_stability.py --phase hutch``); writes
``figures/fig_seismic_zoom.pdf``. Seconds, CPU.

Run:  python scripts/fig_seismic_zoom.py
"""
from __future__ import annotations

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402

from resolvability.download import REPO, ensure  # noqa: E402
from resolvability.seismic import DX, DZ, MUTE_END, N  # noqa: E402
from resolvability.style import PALETTE, apply_paper_style  # noqa: E402

NBIN = 24                                        # equal-count bins in illumination rank
ZTOP_KM, ZBOT_KM = 2.5, N * DZ / 1000.0          # the deep end of the section
CACHE = "results/seismic_dps_recon.npz"
DIAG = "results/seismic_illum_diag.npz"
OUT = os.path.join(REPO, "figures/fig_seismic_zoom.pdf")
EM, MAD = PALETTE["em"], PALETTE["mad"]
TRUTHC = "0.10"


def box(a, w=5):
    """A w x w average. The probe's per-pixel error is independent; the illumination is not."""
    from numpy.lib.stride_tricks import sliding_window_view
    return sliding_window_view(np.pad(a, w // 2, mode="edge"), (w, w)).mean((-1, -2))


def img(v):
    return np.asarray(v, float).reshape(N, N)


def main():
    apply_paper_style(); plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm"})
    z = np.load(ensure(CACHE), allow_pickle=True)
    truth, om, cm = img(z["truth"]), img(z["oracle"]), img(z["curated"])
    osd, csd = img(z["oracle_std"]), img(z["curated_std"])
    # the operator's own sensitivity to a reflector at each pixel, ||A e_j||^2, Hutchinson-probed
    diag = img(np.load(ensure(DIAG))["diag"])

    j0 = int(round(ZTOP_KM * 1000.0 / DZ))
    ext = [0.0, N * DX / 1000.0, ZBOT_KM, ZTOP_KM]        # km, depth increasing downward
    # the identical window fig_hero.py uses: 2.3 x the median, over the three images, of
    # each image's standard deviation below the muted water column
    win = 2.3 * float(np.median([np.std(v.T[MUTE_END:]) for v in (truth, om, cm)]))
    scale = float(truth[:, MUTE_END:].std())

    fig = plt.figure(figsize=(6.02, 2.28))
    gs = GridSpec(3, 2, figure=fig, width_ratios=[1.0, 0.46], hspace=0.55, wspace=0.16)

    for k, (v, name, col) in enumerate([(truth, "(a) truth", TRUTHC),
                                        (om, "(b) oracle mean", EM),
                                        (cm, "(c) curated mean", MAD)]):
        ax = fig.add_subplot(gs[k, 0])
        ax.imshow(v[:, j0:].T, cmap="gray", vmin=-win, vmax=win, extent=ext,
                  aspect="equal", interpolation="bicubic")
        ax.set_title(name, fontsize=8.0, color=col, pad=1.8)
        ax.set_yticks([2.6, 3.0]); ax.set_xticks([0, 2, 4])
        ax.tick_params(labelsize=6.8, length=2.0, pad=1.2)
        if k == 1: ax.set_ylabel("depth (km)", fontsize=7.6, labelpad=1.5)
        ax.set_xticklabels([]) if k < 2 else ax.set_xlabel("distance (km)", fontsize=7.6, labelpad=1.5)

    # (d) what each prior claims, against the operator's illumination. Ranking is monotone, so the
    # probe's sign noise costs no pixels; the small spatial average tames it before ranking, since
    # the true illumination is smooth across neighbours and the estimator's error is not.
    live = np.zeros((N, N), bool); live[:, MUTE_END:] = True
    rank = 100.0 * box(diag, 5)[live].argsort().argsort() / (live.sum() - 1)
    edges = np.linspace(0, 100, NBIN + 1)
    ctr = 0.5 * (edges[:-1] + edges[1:])
    axd = fig.add_subplot(gs[:, 1])
    for sd, c in ((osd, EM), (csd, MAD)):
        v = sd[live] / scale
        lo, md, hi = (np.array([np.percentile(v[(rank >= a) & (rank < b)], q)
                                for a, b in zip(edges[:-1], edges[1:])]) for q in (25, 50, 75))
        axd.fill_between(ctr, lo, hi, color=c, alpha=0.18, lw=0)
        axd.plot(ctr, md, lw=1.3, color=c)
    axd.set_xlim(0, 100); axd.set_ylim(0.0, 0.50)
    axd.set_xticks([0, 50, 100]); axd.set_yticks([0.0, 0.2, 0.4])
    axd.tick_params(labelsize=6.8, length=2.0, pad=1.2)
    axd.set_xlabel("illumination rank (%)", fontsize=7.6, labelpad=1.5)
    axd.set_ylabel("reported spread", fontsize=7.6, labelpad=2.5)
    axd.set_title("(d)  what each prior claims", fontsize=8.0, loc="left", color="0.15", pad=1.8)
    axd.text(2, 0.485, "blind", fontsize=6.2, color="0.35", ha="left", va="top")
    axd.text(98, 0.485, "resolved", fontsize=6.2, color="0.35", ha="right", va="top")
    axd.text(66, 0.435, "curated", fontsize=6.4, color=MAD, ha="center")
    axd.text(66, 0.128, "oracle", fontsize=6.4, color=EM, ha="center")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", pad_inches=0.02, dpi=300)
    print("saved", os.path.relpath(OUT, REPO))


if __name__ == "__main__":
    main()
