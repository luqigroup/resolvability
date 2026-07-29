#!/usr/bin/env python
"""Each prior against itself: the blind-minus-resolved coverage gap, and where the spread goes.

The two-prior comparison invites an easy dismissal -- train on reconstructions instead of truths and
of course you get a worse prior. This figure answers it with quantities that hold each source to its
own behavior rather than to the other's.

(a,b) For one prior on one operator, the gap is its blind coverage minus its OWN resolved coverage,
      per training seed. A prior's generic approximation error moves both terms together and drops
      out; what survives is the part that is specific to the directions the data cannot reach. The
      truth-trained control sits on zero at every level. The archive-trained prior falls away from
      it, and on the groundwater operator the fall deepens with the level, which is what a collapsed
      interval must do: it never widens, so asking for more coverage buys none. Panel (a) also
      carries the legacy ARCHIVE itself, scored on the same two subspaces -- the prior reproduces
      the archive's own deficit rather than merely being worse than the control.
(c)   The same three sources by spread, var(source)/var(truth) with the first moment removed, on the
      resolved subspace and on the blind one. Everything is order one where the data can adjudicate;
      the archive-trained prior is in fact nearer the truth's spread there than the control is. The
      separation appears only on the blind subspace. So the failure is not a worse prior, it is a
      good prior with a hole where the operator is blind.

The estimand a reader should take away is a difference in differences: the curated gap minus the
oracle gap. The within-prior difference cancels each prior's own approximation error; the
between-prior difference cancels whatever makes the blind subspace intrinsically harder than the
resolved one.

Amplitude calibration is measurement-only throughout (`kap=sqrt(E_sig/E_tag)` from
`results/seismic_data_kappa.npz`, including the archive's own `E_archive`); no ground truth enters it.

Run:  python scripts/fig_gap.py
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
ARCH = "0.45"

SEIS_LEVELS = np.array([0.5, 0.7, 0.8, 0.9, 0.95])
DARCY_SEEDS = (0, 1, 2)
N_RAND = 24                    # random subspaces averaged for the control in panel (c)
OUT = os.path.join(REPO, "figures/fig_gap.pdf")
CACHE = os.path.join(REPO, "results/gap_statistic.npz")


def _rand_ratio(Pu, Tu, du, db, rng):
    """Spread ratio on a random `db`-dimensional subspace of the probed span."""
    M = np.linalg.qr(rng.standard_normal((du, db)))[0]
    return float((Pu @ M).var(0).sum() / (Tu @ M).var(0).sum())


def central_cov(P, Pt, a):
    """Share of truth coordinates inside the source's own central-`a` band, averaged over truths."""
    q0, q1 = np.quantile(P, (1 - a) / 2, 0), np.quantile(P, (1 + a) / 2, 0)
    return float(np.mean([((Pt[i] >= q0) & (Pt[i] <= q1)).mean() for i in range(len(Pt))]))


def seismic():
    """Per-training-seed gap curves and spread ratios, from the coords `fig:reliability` plots."""
    from resolvability.seismic.blind import build_blind_subspace
    from resolvability.seismic.priors import archive_row, load_eval

    z = np.load(ensure("results/seismic_prior_seed_coords.npz"))
    nseed, lo, ne = int(z["SEEDS"]), int(z["HELDOUT_A"]), int(z["NEVAL"])
    truth = load_eval("broadband_dm", lo, lo + ne)
    sub = build_blind_subspace()
    bases = {"resolved": sub.Q_resolved, "blind": sub.Q_blind}
    arch = archive_row(bases, lo, ne, levels=SEIS_LEVELS)       # one shared definition

    cov, spread = {}, {}
    for key, kk in (("resolved", "res"), ("blind", "bl")):
        Pt = truth @ bases[key].T
        vt = Pt.var(0).sum()
        for arm in ("oracle", "curated"):
            P = [np.asarray(z[f"{arm}_{kk}_s{s}"], float) for s in range(nseed)]
            cov[arm, key] = np.array([[central_cov(p, Pt, a) for a in SEIS_LEVELS] for p in P])
            spread[arm, key] = np.array([p.var(0).sum() / vt for p in P])
        cov["archive", key] = arch[key][0][None, :]
        spread["archive", key] = np.array([arch[key][1]])

    # ---- the control a referee asks for: is the SPLIT doing the work, or the directions? --------
    # Draw subspaces of the blind subspace's own dimension at random from the span of the probed
    # directions, i.e. the same directions, no longer sorted by illumination. If the separation were
    # an artifact of cutting the space in two rather than of where the operator is blind, a random
    # cut of the same size would show it too. Everything here is a rotation of coordinates already
    # cached, so no prior is resampled.
    rng = np.random.default_rng(0)
    Qu = np.concatenate([sub.Q_resolved, sub.Q_blind], 0)       # (48, N*N), orthonormal
    du, db = Qu.shape[0], sub.Q_blind.shape[0]
    Tu = truth @ Qu.T
    for arm in ("oracle", "curated"):
        Pu = [np.concatenate([np.asarray(z[f"{arm}_res_s{s}"], float),
                              np.asarray(z[f"{arm}_bl_s{s}"], float)], 1) for s in range(nseed)]
        spread[arm, "random"] = np.array([
            np.mean([_rand_ratio(p, Tu, du, db, rng) for _ in range(N_RAND)]) for p in Pu])
    return cov, spread


def groundwater():
    cov = {}
    for arm in ("oracle", "curated"):
        Z = [np.load(ensure(f"results/darcy_pcn_{arm}_s{s}.npz")) for s in DARCY_SEEDS]
        cov[arm, "resolved"] = np.array([z["resolved"] for z in Z])
        cov[arm, "blind"] = np.array([z["blind"] for z in Z])
        cov["levels"] = Z[0]["levels"]
    return cov


def gap_panel(ax, levels, cov, sources, title, ylab=None):
    ax.axhline(0.0, color="0.55", lw=0.7, ls=":", zorder=0)
    for name, col in sources:
        g = cov[name, "blind"] - cov[name, "resolved"]
        for row in g:                                            # one line per training seed
            ax.plot(levels, row, "-", lw=0.95, color=col, alpha=0.9,
                    marker=("o" if name == "oracle" else "s" if name == "curated" else "^"),
                    ms=2.7, markerfacecolor=("none" if name == "curated" else col),
                    markeredgewidth=0.7)
    ax.set_xlabel("nominal credible level", fontsize=8)
    if ylab:
        ax.set_ylabel(ylab, fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_title(title, fontsize=8.0, loc="left", pad=3.0)


def main():
    sc, ss = seismic()
    gc = groundwater()

    fig, ax = plt.subplots(1, 3, figsize=(6.02, 2.05),
                           gridspec_kw={"width_ratios": [1.0, 1.0, 1.3]})

    gap_panel(ax[0], SEIS_LEVELS, sc,
              (("oracle", EM), ("curated", MAD), ("archive", ARCH)),
              "(a)  seismic Born", "blind $-$ resolved coverage")
    ax[0].set_ylim(-0.22, 0.10)
    ax[0].text(0.505, 0.082, "oracle", fontsize=6.8, color=EM, ha="left")
    ax[0].text(0.505, -0.108, "curated", fontsize=6.8, color=MAD, ha="left")
    ax[0].text(0.505, -0.205, "legacy archive", fontsize=6.8, color=ARCH, ha="left")

    gap_panel(ax[1], gc["levels"], gc, (("oracle", EM), ("curated", MAD)),
              "(b)  groundwater flow")
    ax[1].set_ylim(-0.70, 0.16)
    ax[1].text(0.505, 0.075, "oracle", fontsize=6.8, color=EM, ha="left")
    ax[1].text(0.505, -0.245, "curated", fontsize=6.8, color=MAD, ha="left")

    # ---- (c) where the spread goes, on the same two subspaces --------------------------------
    xs = [0, 1, 2]
    for name, col, mk in (("oracle", EM, "o"), ("curated", MAD, "s"), ("archive", ARCH, "^")):
        r, b = ss[name, "resolved"], ss[name, "blind"]
        # the archive has no random-cut entry (the control is about the two priors separating)
        m = ss.get((name, "random"))
        for i in range(len(r)):
            y = [r[i], m[i], b[i]] if m is not None else [r[i], np.nan, b[i]]
            ax[2].plot(xs, y, "-", lw=0.95, color=col, alpha=0.9, marker=mk, ms=3.2,
                       markerfacecolor=("none" if name == "curated" else col), markeredgewidth=0.7)
            if m is None:                     # keep the archive's two real points connected
                ax[2].plot([xs[0], xs[2]], [r[i], b[i]], "--", lw=0.85, color=col,
                           alpha=0.55, dashes=(3, 2))   # archive: no random-cut entry to join
    ax[2].axhline(1.0, color="0.55", lw=0.7, ls=":", zorder=0)
    ax[2].set_yscale("log")
    ax[2].set_xticks(xs); ax[2].set_xticklabels(["resolved", "random", "blind"], fontsize=6.8)
    ax[2].set_xlim(-0.22, 2.86); ax[2].set_ylim(0.024, 2.1)
    ax[2].set_ylabel("spread, relative to truth", fontsize=8)
    ax[2].tick_params(labelsize=7, which="both")
    ax[2].set_title("(c)  seismic: where the spread goes", fontsize=8.0, loc="left", pad=3.0)
    for name, col, mul in (("oracle", EM, 1.0), ("curated", MAD, 1.10), ("archive", ARCH, 0.86)):
        ax[2].text(2.11, float(ss[name, "blind"].mean()) * mul, name, fontsize=6.8, color=col,
                   ha="left", va="center")

    fig.tight_layout(w_pad=1.7)
    fig.savefig(OUT, bbox_inches="tight", pad_inches=0.02, dpi=300)

    np.savez(CACHE,
             seis_levels=SEIS_LEVELS, darcy_levels=gc["levels"],
             **{f"seis_{a}_{k}": sc[a, k] for a in ("oracle", "curated", "archive")
                for k in ("blind", "resolved")},
             **{f"seis_spread_{a}_{k}": v for (a, k), v in ss.items()},
             **{f"gw_{a}_{k}": gc[a, k] for a in ("oracle", "curated")
                for k in ("blind", "resolved")})
    print("saved", os.path.relpath(OUT, REPO), "and", os.path.relpath(CACHE, REPO))
    for tag, c, lv in (("seismic", sc, SEIS_LEVELS), ("groundwater", gc, gc["levels"])):
        i = int(np.argmin(abs(np.asarray(lv) - 0.9)))
        for a in ("oracle", "curated", "archive"):
            if (a, "blind") not in c:
                continue
            g = (c[a, "blind"] - c[a, "resolved"])[:, i]
            print("  %-12s %-8s gap@central-90%%  %+.4f  (seed sd %.4f)"
                  % (tag, a, g.mean(), g.std()))


if __name__ == "__main__":
    main()
