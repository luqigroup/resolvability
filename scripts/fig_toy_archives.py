#!/usr/bin/env python
"""The four archives of the two-dimensional example: {posterior-sample, single-best} x
{reconstructed under the truth p*, under the regularizer rho}.

Every cell is drawn through the same pipeline -- a measurement y ~ p*(y), then one posterior sample
or the global posterior mode -- rather than from an assumed law. The posterior-sample archive under
the truth returns the truth, since T[p*] = p*; under rho its blind conditional is rho's; the
single-best archive has zero blind width under either. Solid contours are each posterior-sample
archive's exact law; the truth (green dashed) is overlaid on every cell and rho (grey dotted) on the
rho column. Blue: under the truth; red: under rho.

No inputs. Writes ``figures/fig_toy_archives.pdf``, sized to be included at natural size. Seconds,
CPU.

Run:  python scripts/fig_toy_archives.py
"""
from __future__ import annotations

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from resolvability.download import REPO  # noqa: E402
from resolvability.style import PALETTE, apply_paper_style  # noqa: E402
from resolvability.toy import (RHO, S2, TRUTH, curated_prior, log_density,  # noqa: E402
                               posterior_archive, single_best_archive)

apply_paper_style()
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm"})
EM, MAD = PALETTE["em"], PALETTE["mad"]
GHOST_TRUTH, GHOST_RHO, TRUTH_LABEL = "#8cc690", "#9d9b96", "#4f8f53"
OUT = os.path.join(REPO, "figures/fig_toy_archives.pdf")

N_POST, N_SB, SEED = 1100, 700, 3
XR_LIM, XB_LIM = 3.4, 8.6


def main():
    rng = np.random.default_rng(SEED)
    T = curated_prior(RHO, TRUTH, S2)
    archives = {
        "post_true": posterior_archive(TRUTH, TRUTH, S2, N_POST, rng),
        "post_rho": posterior_archive(RHO, TRUTH, S2, N_POST, rng),
        "sb_true": single_best_archive(TRUTH, TRUTH, S2, N_SB, rng),
        "sb_rho": single_best_archive(RHO, TRUTH, S2, N_SB, rng),
    }

    xr = np.linspace(-XR_LIM, XR_LIM, 161)
    xb = np.linspace(-XB_LIM, XB_LIM, 161)
    XR, XB = np.meshgrid(xr, xb, indexing="ij")
    P = np.stack([XR, XB], -1)
    Z_truth = np.exp(log_density(TRUTH, P))
    Z_rho = np.exp(log_density(RHO, P))
    Z_curated = np.exp(log_density(T, P))

    fig, axes = plt.subplots(2, 2, figsize=(3.53, 3.35), sharex=True, sharey=True,
                             gridspec_kw=dict(wspace=0.07, hspace=0.10))
    cells = (
        (axes[0, 0], "post_true", Z_truth, EM, False, "recovers\nthe truth"),
        (axes[0, 1], "post_rho", Z_curated, MAD, True, r"$\rho$'s spread," + "\nfrozen"),
        (axes[1, 0], "sb_true", None, EM, False, "zero blind\nwidth"),
        (axes[1, 1], "sb_rho", None, MAD, True, "zero blind\nwidth"),
    )
    halo = [pe.withStroke(linewidth=2.0, foreground="white")]
    for ax, key, Z, col, show_rho, note in cells:
        X = archives[key]
        top_row = key.startswith("post")
        ax.scatter(X[:, 0], X[:, 1], s=1.1 if top_row else 1.6, color=col,
                   alpha=0.15 if top_row else 0.25, edgecolors="none", rasterized=True,
                   zorder=1)
        if Z is not None:
            ax.contour(XR, XB, Z, levels=4, colors=col, linewidths=1.05)
        # The truth and rho are drawn above the archive, so a match reads as colour under dashes.
        g = ax.contour(XR, XB, Z_truth, levels=4, colors=GHOST_TRUTH, linewidths=0.65,
                       linestyles=[(0, (3.2, 1.8))])
        g.set_zorder(3)
        if show_rho:
            r = ax.contour(XR, XB, Z_rho, levels=4, colors=GHOST_RHO, linewidths=0.65,
                           linestyles=[(0, (1, 1.6))])
            r.set_zorder(3)
        # Below the contours and above the dots: the box may mask dots, never a density.
        ax.text(0.04, 0.03, note, transform=ax.transAxes, fontsize=7, color="0.35",
                ha="left", va="bottom", zorder=1.5,
                bbox=dict(boxstyle="square,pad=0.3", fc="white", ec="none"))
        for sp in ax.spines.values():
            sp.set_zorder(6)
        ax.set_xticks([-2, 0, 2]); ax.set_yticks([-8, -4, 0, 4, 8])
        ax.set_xlim(-XR_LIM, XR_LIM); ax.set_ylim(-XB_LIM, XB_LIM)
        ax.tick_params(labelsize=7)
    axes[0, 0].set_title(r"under the truth $p_\star$", fontsize=8, pad=3)
    axes[0, 1].set_title(r"under the regularizer $\rho$", fontsize=8, pad=3)
    axes[0, 0].set_ylabel("posterior-sample archive\n" + r"$x_B$ (blind)", fontsize=8,
                          labelpad=1)
    axes[1, 0].set_ylabel("single-best archive\n" + r"$x_B$ (blind)", fontsize=8, labelpad=1)
    for j in (0, 1):
        axes[1, j].set_xlabel(r"$x_R$ (resolved)", fontsize=8, labelpad=1.5)
    axes[0, 1].text(-1.95, 7.45, "truth", fontsize=7, color=TRUTH_LABEL, path_effects=halo)
    axes[1, 1].text(0.1, 2.35, r"$\rho$", fontsize=7.4, color="0.35", ha="center")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", pad_inches=0.02, dpi=300,
                metadata={"CreationDate": None})
    plt.close(fig)
    print(f"saved {os.path.relpath(OUT, REPO)}")


if __name__ == "__main__":
    main()
