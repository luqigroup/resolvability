#!/usr/bin/env python
"""Fit priors to the two-dimensional example's archives, and read the blind spread they report.

Three training sets, each drawn through the pipeline of :mod:`resolvability.toy`:

  post     the regularizer's posterior-sample archive, whose law is the curated prior T[rho].
  sb       the regularizer's single-best archive, which lies on rho's regression line.
  oracle   the truths themselves.

Every set is dequantized in the same absolute units, ``N(0, h^2 I)`` with ``h = H_FLOOR``, so each
fit targets the dequantized law, whose blind conditionals are exact Gaussian mixtures; on the
single-best archive that law is the smoothing floor in closed form. Two model classes are fit to
each set: a two-component Gaussian mixture by EM, and a HINT flow at three seeds (Adam, full batch,
early stopping on a held-out split). The blind spread of every fit, and of the exact dequantized
law, is read on the fibers ``x_R = c`` of ``FIBERS``.

The three flow seeds share one dequantization and one split, so they measure the fit's variability,
not the data's. The same torch seed is used for all three sets, so the three flows at a seed start
from the same weights and permutations.

Writes ``results/toy_estimation.npz`` (everything ``fig_toy_money.py`` reads, together with the
parameters it was generated from) and ``data/checkpoints/toy_flows/{post,sb,oracle}_s{seed}.pth``.
CPU only, under a minute. The flows' early-stopping epoch depends on the number of CPU threads,
which moves their fitted spread by up to a percent or so; the mixture fits do not depend on it.

Run:  python scripts/toy_estimation.py [--seeds 0 1 2]
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch
from sklearn.mixture import GaussianMixture

from resolvability.download import REPO
from resolvability.toy import (FIBERS, GMM2, H_FLOOR, RHO, S2, TRUTH, curated_prior, dequantized,
                               fiber_conditional, log_density, posterior_archive, sample_gmm2,
                               sb_pushforward, single_best_archive)
from resolvability.toy.estimation import (CACHE, CKPT, FLOW, N_TRAIN, N_VAL, XB_GRID, fiber_slice,
                                          flow_logpdf, make_flow, slice_sd)

DATA_SEED = 0
OUT = os.path.join(REPO, CACHE)


def make_archives(n: int, rng) -> dict[str, np.ndarray]:
    """The three training sets: y ~ p*(y), then a posterior sample (post) or the posterior mode
    (sb) under rho; the oracle set is truths."""
    post = posterior_archive(RHO, TRUTH, S2, n, rng)
    sb = single_best_archive(RHO, TRUTH, S2, n, rng)
    oracle = sample_gmm2(TRUTH, n, rng)
    return {"post": post, "sb": sb, "oracle": oracle}


def fit_gmm(X: np.ndarray, rng) -> GMM2:
    """Two-component, full-covariance EM. Each dequantized training law is itself a
    two-component Gaussian mixture, so this model class contains it."""
    gm = GaussianMixture(n_components=2, covariance_type="full", n_init=4,
                         reg_covar=1e-10, random_state=int(rng.integers(2**31)))
    gm.fit(X)
    return GMM2(gm.weights_ / gm.weights_.sum(), gm.means_, gm.covariances_)


def train_hint(X_tr, X_va, seed: int, arch: str, epochs=400, patience=40, lr=2e-3):
    torch.manual_seed(seed)
    flow = make_flow()
    opt = torch.optim.Adam(flow.parameters(), lr=lr)
    tr = torch.as_tensor(X_tr, dtype=torch.float32)
    va = torch.as_tensor(X_va, dtype=torch.float32)

    hist_tr, hist_va, best, best_state, since = [], [], np.inf, None, 0
    for _ in range(epochs):
        flow.train()
        opt.zero_grad()
        loss = -flow.log_prob(tr).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(flow.parameters(), 5.0)
        opt.step()
        flow.eval()
        with torch.no_grad():
            v = float(-flow.log_prob(va).mean())
        hist_tr.append(loss.item()); hist_va.append(v)
        if v < best - 1e-4:
            best, since = v, 0
            best_state = {k: t.clone() for k, t in flow.state_dict().items()}
        else:
            since += 1
            if since > patience:
                break
    flow.load_state_dict(best_state)
    path = os.path.join(REPO, CKPT.format(arch=arch, seed=seed))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"state": best_state, "arch": arch, "seed": seed, "deq_abs": H_FLOOR, **FLOW,
                "lr": lr, "patience": patience, "n_train": len(X_tr), "val_nll": best,
                "epochs_run": len(hist_tr)}, path)
    return flow, np.array(hist_tr), np.array(hist_va)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = ap.parse_args()
    t0 = time.time()
    rng = np.random.default_rng(DATA_SEED)
    raw = make_archives(N_TRAIN + N_VAL, rng)
    print(f"[{time.time() - t0:.0f}s] archives drawn", flush=True)

    c1, c2 = FIBERS
    out = {"xb_grid": XB_GRID, "fibers": np.array([c1, c2]), "deq_abs": H_FLOOR,
           "s_rho": np.sqrt(RHO.Sig[1, 1] - RHO.Sig[0, 1] ** 2 / RHO.Sig[0, 0])}
    # The fits estimate the dequantized laws, so those are the reference; the raw fiber
    # conditionals are kept alongside (suffix _raw).
    laws = {"post": dequantized(curated_prior(RHO, TRUTH, S2), H_FLOOR),
            "sb": dequantized(sb_pushforward(RHO, TRUTH, S2), H_FLOOR),
            "oracle": dequantized(TRUTH, H_FLOOR)}
    for c, tag in ((c1, "c1"), (c2, "c2")):
        out[f"ref_frozen_raw_{tag}"] = fiber_conditional(RHO, c).pdf(XB_GRID)
        out[f"ref_truth_raw_{tag}"] = fiber_conditional(TRUTH, c).pdf(XB_GRID)
        out[f"s_star_raw_{tag}"] = np.sqrt(fiber_conditional(TRUTH, c).moments()[1])
        for arch, law in laws.items():
            fc = fiber_conditional(law, c)
            out[f"anchor_{arch}_{tag}"] = fc.pdf(XB_GRID)
            out[f"anchor_{arch}_sd_{tag}"] = np.sqrt(fc.moments()[1])

    for arch, X in raw.items():
        deq = X + H_FLOOR * rng.standard_normal(X.shape)
        X_tr, X_va = deq[:N_TRAIN], deq[N_TRAIN:]           # iid rows: a head/tail split is random
        gm = fit_gmm(X_tr, rng)
        for c, tag in ((c1, "c1"), (c2, "c2")):
            p = fiber_slice(lambda pts: log_density(gm, pts), c)
            out[f"gmm_{arch}_{tag}"] = p
            out[f"gmm_{arch}_sd_{tag}"] = slice_sd(p)
        for seed in args.seeds:
            flow, h_tr, h_va = train_hint(X_tr, X_va, seed, arch)
            out[f"loss_tr_{arch}_s{seed}"] = h_tr
            out[f"loss_va_{arch}_s{seed}"] = h_va
            for c, tag in ((c1, "c1"), (c2, "c2")):
                p = fiber_slice(lambda pts: flow_logpdf(flow, pts), c)
                out[f"hint_{arch}_{tag}_s{seed}"] = p
                out[f"hint_{arch}_sd_{tag}_s{seed}"] = slice_sd(p)
            print(f"[{time.time() - t0:.0f}s] {arch} flow s{seed}: {len(h_tr)} epochs, "
                  f"val NLL {h_va.min():.4f}, blind sd {out[f'hint_{arch}_sd_c1_s{seed}']:.3f}",
                  flush=True)
        print(f"[{time.time() - t0:.0f}s] {arch} GMM: blind sd {out[f'gmm_{arch}_sd_c1']:.3f}, "
              f"exact {out[f'anchor_{arch}_sd_c1']:.4f}", flush=True)

    out.update(dict(gen_s2=S2, gen_rho_mu=RHO.mu, gen_rho_Sig=RHO.Sig, gen_truth_w=TRUTH.w,
                    gen_truth_mu=TRUTH.mu, gen_truth_Sig=TRUTH.Sig, gen_h=H_FLOOR,
                    gen_n_train=N_TRAIN, gen_n_val=N_VAL, gen_data_seed=DATA_SEED,
                    seeds=np.array(args.seeds)))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez(OUT, **out)
    print(f"saved {os.path.relpath(OUT, REPO)} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
