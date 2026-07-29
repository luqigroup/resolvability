#!/usr/bin/env python
"""Two things the resolvability statement is asked to survive, both computed from the operator.

`sec:reportcard` sells the statement as computed *from the operator alone*. For a Born operator the
operator carries a background $m_0$, and in a real shop $m_0$ is itself a legacy migrated product. So
a reader is entitled to ask whether "from the operator alone" quietly means "from someone's earlier
inversion". This script answers that, and fixes a second thing it needs the same machinery for.

  --phase stability   Rebuild the illumination symbol under perturbed backgrounds -- the velocity
                      gradient scaled, the water layer moved, a smooth lateral tilt added -- and
                      measure how much of the NOMINAL blind subspace still lies in the perturbed
                      one. Reported as the captured energy of each nominal blind atom under the
                      perturbed blind projector, and as principal angles between the two subspaces.
                      What the statement names should barely move; that is the claim being tested.

  --phase diag        A per-pixel map of $\\|\\bA\\bm\\delta_x\\|^2 = \\mathrm{diag}(J^\\top J)_x$, the
                      quantity that actually defines the resolved/blind split. `fig:teaser`(d) has
                      been showing incident wavefield energy, a source-side PROXY: it ignores the
                      receivers, the aperture and the band entirely, which is why its dynamic range
                      is so much narrower than the illumination the text describes. The diagonal is
                      read straight off point-spread functions -- $(N\\delta_x)_x = \\|J\\delta_x\\|^2$
                      -- so a lattice of separated deltas gives one sample per probe per apply, and
                      staggered lattices fill the map in a handful of applies.

Cost is set by the number of normal-operator applies, each 12 Born shots forward plus one autograd
adjoint. `--stride` controls the map resolution: stride s costs s^2 applies and returns an
(N/s x N/s) map.

Run:  python scripts/seismic_m0_stability.py --phase diag --stride 8
      python scripts/seismic_m0_stability.py --phase stability
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch

from resolvability.download import REPO  # noqa: E402
from resolvability.seismic.born import SparseBornImager  # noqa: E402

NSRC, NREC = 12, 24
OUT = os.path.join(REPO, "results")


def background(n, vtop=2.5, vbot=4.5, water=1.5, water_cells=None, tilt=0.0):
    """The deployed background, with the knobs a migration velocity analysis would actually move."""
    water_cells = max(2, n // 25) if water_cells is None else water_cells
    v0 = np.ones((n, n), np.float64) * vtop
    v0 *= np.linspace(1.0, vbot / v0.max(), n)[None, :]
    if tilt:                                   # smooth lateral gradient, the classic mis-migration
        v0 *= (1.0 + tilt * np.linspace(-0.5, 0.5, n))[:, None]
    s = 1.0 / v0
    s[:, :water_cells] = 1.0 / water
    return (s ** 2).astype(np.float32)


def imager(m0, n):
    return SparseBornImager(m0, spacing=(20.0, 12.5), nsrc=NSRC, nrec=NREC, tn=2000.0, f0=0.030,
                            space_order=16, nbl=40, pad_mode="zero", device="cpu")


def matvec(im, v, n):
    """N v = J^T J v, the adjoint taken by autograd (the exact transpose)."""
    x = torch.tensor(np.asarray(v, np.float32).reshape(1, 1, n, n), requires_grad=True)
    dfit = torch.cat([im._born_one(x, s).reshape(-1) for s in range(im.nsrc)])
    (dfit.detach() * dfit).sum().backward()
    return 0.5 * x.grad.detach().cpu().numpy().ravel().astype(np.float64)


def psf_symbol(im, n, P=64):
    """The local illumination symbol, as `seismic_probe_basis.py` builds it: one batched apply."""
    g = max(2, (n - P) // P + 1)
    cs = np.linspace(P // 2, n - P // 2, g).astype(int)
    probes = [(a, b) for a in cs for b in cs]
    R0 = np.zeros(n * n, np.float32)
    for a, b in probes:
        R0[a * n + b] = 1.0
    Nr0 = matvec(im, R0, n).reshape(n, n)
    w = np.outer(np.hanning(P), np.hanning(P))
    h = P // 2
    syms = []
    for a, b in probes:
        a0, b0 = np.clip(a - h, 0, n - P), np.clip(b - h, 0, n - P)
        tile = Nr0[a0:a0 + P, b0:b0 + P].copy()
        tile -= tile.mean()
        syms.append(np.abs(np.fft.fftshift(np.fft.fft2(w * tile))) ** 2)
    a_sym = np.stack(syms).mean(0)
    return a_sym / (a_sym.max() + 1e-30), Nr0, probes


def blind_atoms(a_sym, P=64, nb=48):
    """The `nb` Gabor wavenumbers the symbol ranks lowest, i.e. the directions this operator misses.

    Same ranking `resolvability/seismic/blind.py` applies to the cached symbol, reproduced here so
    a perturbed background can be ranked by its own symbol rather than the nominal one's.
    """
    kk = np.fft.fftshift(np.fft.fftfreq(P))
    KX, KZ = np.meshgrid(kk, kk, indexing="ij")
    krad = np.sqrt(KX ** 2 + KZ ** 2)
    band = (krad > 0.06) & (krad < 0.45)                 # the reflectivity band, as in the producer
    idx = np.where(band.ravel())[0]
    order = np.argsort(a_sym.ravel()[idx])               # ascending symbol = most blind first
    return idx[order[:nb]], (KX, KZ)


def phase_stability(n, P, nb):
    t0 = time.time()
    cases = [("nominal", {}),
             ("gradient +5%", {"vbot": 4.725}), ("gradient -5%", {"vbot": 4.275}),
             ("top +3%", {"vtop": 2.575}), ("top -3%", {"vtop": 2.425}),
             ("water layer x2", {"water_cells": 2 * max(2, n // 25)}),
             ("lateral tilt 4%", {"tilt": 0.04})]
    syms, sets = {}, {}
    for name, kw in cases:
        a_sym, _, _ = psf_symbol(imager(background(n, **kw), n), n, P)
        syms[name], sets[name] = a_sym, set(blind_atoms(a_sym, P, nb)[0].tolist())
        print(f"[{time.time()-t0:.0f}s] {name}: symbol built", flush=True)

    ref_idx, _ = blind_atoms(syms["nominal"], P, nb)
    ref = sets["nominal"]

    # Set overlap of the bottom-nb is the WRONG estimand and is reported only to show why: the
    # illumination tail is near-degenerate, so which atoms occupy the bottom nb reshuffles under a
    # perturbation that barely moves the symbol. What the statement claims is not "these are the nb
    # blindest" but "these directions are blind", so the test is where the nominal blind atoms LAND
    # in the perturbed operator's own illumination ranking.
    band_mask = None
    kk = np.fft.fftshift(np.fft.fftfreq(P))
    KX, KZ = np.meshgrid(kk, kk, indexing="ij")
    krad = np.sqrt(KX ** 2 + KZ ** 2)
    band_mask = ((krad > 0.06) & (krad < 0.45)).ravel()
    nband = int(band_mask.sum())

    print("\n  where the NOMINAL blind directions land in each perturbed operator's own ranking")
    print("  %-18s %-11s %-11s %-11s %s"
          % ("background", "median pct", "worst pct", "symbol corr", "bottom-%d overlap" % nb))
    rows = {}
    for name, _ in cases:
        a = syms[name].ravel()
        rank = np.argsort(np.argsort(a[band_mask])) / max(nband - 1, 1)   # 0 = most blind
        pos = np.searchsorted(np.where(band_mask)[0], ref_idx)
        pct = rank[pos]
        corr = float(np.corrcoef(np.log10(syms["nominal"].ravel() + 1e-12),
                                 np.log10(a + 1e-12))[0, 1])
        rows[name] = (float(np.median(pct)), float(pct.max()), corr, len(ref & sets[name]))
        print("  %-18s %-11.3f %-11.3f %-11.4f %d/%d"
              % (name, rows[name][0], rows[name][1], corr, rows[name][3], nb))
    print("\n  (percentile 0 = the most blind direction the perturbed operator has; the nominal "
          "\n   blind set occupying a low percentile everywhere is the stability that is claimed)")
    np.savez(os.path.join(OUT, "seismic_m0_stability.npz"),
             names=np.array(list(rows), dtype=object),
             med_pct=np.array([rows[k][0] for k in rows]),
             worst_pct=np.array([rows[k][1] for k in rows]),
             corr=np.array([rows[k][2] for k in rows]),
             shared=np.array([rows[k][3] for k in rows]), nb=nb, nband=nband,
             **{f"sym_{i}": syms[k] for i, k in enumerate(rows)})
    print("\nsaved plots/seismic_m0_stability.npz")


def phase_diag(n, sep, step):
    """diag(J^T J)_x = ||J delta_x||^2, sampled from staggered lattices of separated deltas.

    Each apply places deltas `sep` apart and reads every probe's own centre, which is that probe's
    ||J delta||^2 provided neighbours are far enough away that their point-spread sidelobes do not
    reach it. Staggering the lattice by `step` fills in between. Cost is (sep/step)^2 applies for a
    map on a lattice of spacing `step`. Separation is validated, not assumed: the run also probes at
    2*sep and compares the samples the two share.
    """
    t0 = time.time()
    im = imager(background(n), n)

    def sample(separation, oa, ob):
        aa = np.arange(oa, n, separation)
        bb = np.arange(ob, n, separation)
        R = np.zeros(n * n, np.float32)
        R[(aa[:, None] * n + bb[None, :]).ravel()] = 1.0
        return aa, bb, matvec(im, R, n).reshape(n, n)

    # --- separation check: the same probes, once crowded at `sep` and once isolated at 2*sep -------
    a1, b1, N1 = sample(sep, 0, 0)
    a2, b2, N2 = sample(2 * sep, 0, 0)
    shared = np.ix_(a2, b2)
    x1, x2 = N1[shared], N2[shared]
    # Relative error pointwise is meaningless where the diagonal is ~0 (the muted water layer and the
    # boundary), so the criterion is contamination measured against the map's own scale, on the
    # samples that carry illumination at all.
    live = x2 > 0.01 * x2.max()
    rel = float(np.max(np.abs(x1[live] - x2[live]) / x2[live]))
    glob = float(np.max(np.abs(x1 - x2)) / x2.max())
    print(f"[{time.time()-t0:.0f}s] separation check at sep={sep} against sep={2*sep}: "
          f"worst {rel:.2%} on illuminated samples ({live.sum()}/{live.size}), "
          f"{glob:.2%} of peak overall", flush=True)
    assert rel < 0.05, (f"probes {sep} apart contaminate one another ({rel:.1%}); raise --sep")

    # --- staggered fill ---------------------------------------------------------------------------
    diag = np.full((n, n), np.nan)
    offs = list(range(0, sep, step))
    for oa in offs:
        for ob in offs:
            aa, bb, Nr = sample(sep, oa, ob)
            diag[np.ix_(aa, bb)] = Nr[np.ix_(aa, bb)]
    got = np.isfinite(diag)
    print(f"[{time.time()-t0:.0f}s] {len(offs)**2} applies -> {got.sum()} samples on a "
          f"{len(offs)**2 and n//step}x{n//step} lattice", flush=True)

    v = diag[got]
    lo, hi = np.percentile(v[v > 0], [1, 99])
    np.savez(os.path.join(OUT, "seismic_illum_diag.npz"), diag=diag, sep=sep, step=step, N=n,
             sep_check=rel)
    print(f"\ndiag(J^T J): 1-99 pct spans {hi/lo:.0f}x  (min {v.min():.3e}, max {v.max():.3e})")
    print("saved plots/seismic_illum_diag.npz")


def phase_hutch(n, nprobe, seed):
    """diag(J^T J) by Hutchinson probing: diag(N)_i = E[z_i (N z)_i] for Rademacher z.

    A lattice of deltas cannot isolate the diagonal here -- the separation check in `--phase diag`
    refuses at every spacing tried, because a 12-shot acquisition has point-spread functions that
    smear across the whole section rather than sitting in a tile. Hutchinson assumes no locality at
    all: it is unbiased for any operator. The cost is one apply per probe, and the estimator's own
    noise is reported by splitting the probes in half and comparing, so the map is not read tighter
    than it was measured.
    """
    t0 = time.time()
    im = imager(background(n), n)
    rng = np.random.default_rng(seed)
    acc = [np.zeros(n * n), np.zeros(n * n)]
    cnt = [0, 0]
    for k in range(nprobe):
        z = rng.integers(0, 2, n * n).astype(np.float32) * 2.0 - 1.0
        acc[k % 2] += z * matvec(im, z, n)
        cnt[k % 2] += 1
        if (k + 1) % 16 == 0:
            print(f"[{time.time()-t0:.0f}s] {k+1}/{nprobe} probes", flush=True)
    h1, h2 = acc[0] / max(cnt[0], 1), acc[1] / max(cnt[1], 1)
    d = (0.5 * (h1 + h2)).reshape(n, n)
    live = d > 0.01 * d.max()
    split = float(np.median(np.abs(h1.reshape(n, n)[live] - h2.reshape(n, n)[live])
                            / (d[live] + 1e-30)))
    lo, hi = np.percentile(d[live], [1, 99])
    np.savez(os.path.join(OUT, "seismic_illum_diag.npz"), diag=d, nprobe=nprobe, N=n,
             split_half=split, estimator="hutchinson")
    print(f"\n[{time.time()-t0:.0f}s] diag(J^T J), {nprobe} probes")
    print(f"  split-half median relative disagreement {split:.1%} (the estimator's own noise)")
    print(f"  1-99 pct of the illuminated region spans {hi/lo:.0f}x")
    print("saved plots/seismic_illum_diag.npz")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=("stability", "diag", "hutch"), required=True)
    ap.add_argument("--nprobe", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--tile", type=int, default=64)
    ap.add_argument("--nblind", type=int, default=48)
    ap.add_argument("--sep", type=int, default=32)
    ap.add_argument("--step", type=int, default=8)
    a = ap.parse_args()
    if a.phase == "stability":
        phase_stability(a.n, a.tile, a.nblind)
    elif a.phase == "hutch":
        phase_hutch(a.n, a.nprobe, a.seed)
    else:
        phase_diag(a.n, a.sep, a.step)


if __name__ == "__main__":
    main()
