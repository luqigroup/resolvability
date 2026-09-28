"""The two-dimensional example's estimation layer.

The fit cache and the flow checkpoints are produced by ``scripts/toy_estimation.py`` and are not
downloaded; the tests that read them skip when they are absent.
"""
from __future__ import annotations

import os

import numpy as np
import pytest
from numpy.polynomial.hermite_e import hermegauss
from scipy.integrate import trapezoid

from resolvability.download import REPO
from resolvability.toy import (FIBERS, H_FLOOR, RHO, S2, TRUTH, data_marginal, dequantized,
                               fiber_conditional, map_ridge, sample_gmm2, sb_pushforward)
from resolvability.toy.estimation import (CACHE, CKPT, check_cache, fiber_slice, flow_logpdf,
                                          load_flow)

HAVE_CACHE = os.path.exists(os.path.join(REPO, CACHE))
HAVE_CKPT = os.path.exists(os.path.join(REPO, CKPT.format(arch="post", seed=0)))


def test_sampler_moments():
    rng = np.random.default_rng(1)
    X = sample_gmm2(TRUTH, 200_000, rng)
    w, mu, Sig = TRUTH.w, TRUTH.mu, TRUTH.Sig
    mean = (w[:, None] * mu).sum(0)
    ex2 = (w[:, None, None] * (Sig + np.einsum("ji,jk->jik", mu, mu))).sum(0)
    assert np.allclose(X.mean(0), mean, atol=0.05)
    assert np.allclose(np.cov(X.T), ex2 - np.outer(mean, mean), atol=0.15)


def test_dequantized_single_best_law():
    """The dequantized single-best law's blind spread, in closed form and by quadrature over the
    data law of ``E_y[N((c, x_B) - (b0 + K y); h^2 I)]``; the ridge's slope widens it above h."""
    h, c = H_FLOOR, FIBERS[0]
    law = dequantized(sb_pushforward(RHO, TRUTH, S2), h)
    sd_closed = np.sqrt(fiber_conditional(law, c).moments()[1])
    b0, K = map_ridge(RHO, S2)
    dm = data_marginal(TRUTH, S2)
    nodes, wts = hermegauss(200)
    wts = wts / np.sqrt(2 * np.pi)
    xb = np.linspace(-6, 6, 12001)
    dens = np.zeros_like(xb)
    for wj, aj, vj in zip(dm.w, dm.mean, dm.var):
        for yn, wy in zip(aj + np.sqrt(vj) * nodes, wts):
            px = b0 + K * yn
            dens += wj * wy * np.exp(-((c - px[0]) ** 2 + (xb - px[1]) ** 2) / (2 * h**2))
    dens /= trapezoid(dens, xb)
    m = trapezoid(dens * xb, xb)
    sd_route = np.sqrt(trapezoid(dens * (xb - m) ** 2, xb))
    assert abs(sd_closed - sd_route) / sd_route < 1e-6
    assert sd_closed > h


@pytest.mark.skipif(not HAVE_CKPT, reason="run scripts/toy_estimation.py first")
def test_flow_density_integrates_to_one():
    flow = load_flow("post", 0)
    g = np.linspace(-16, 16, 481)
    XX, YY = np.meshgrid(g, g, indexing="ij")
    lp = flow_logpdf(flow, np.stack([XX.ravel(), YY.ravel()], axis=-1)).reshape(XX.shape)
    total = trapezoid(trapezoid(np.exp(lp), g, axis=1), g)
    assert abs(total - 1.0) < 5e-3


@pytest.mark.skipif(not HAVE_CACHE, reason="run scripts/toy_estimation.py first")
def test_cache_matches_the_parameters():
    check_cache(np.load(os.path.join(REPO, CACHE)))


@pytest.mark.skipif(not HAVE_CACHE, reason="run scripts/toy_estimation.py first")
def test_fits_land_on_the_dequantized_widths():
    """The mixture fit's model class contains every dequantized law, so it is held to a tight band;
    the flow's to a looser one."""
    z = np.load(os.path.join(REPO, CACHE))
    seeds = [int(s) for s in z["seeds"]]
    for arch, tol, flow_tol in (("post", 0.03, 0.15), ("sb", 0.03, 0.15), ("oracle", 0.12, 0.15)):
        anchor = float(z[f"anchor_{arch}_sd_c1"])
        assert abs(float(z[f"gmm_{arch}_sd_c1"]) - anchor) / anchor < tol
        for s in seeds:
            assert abs(float(z[f"hint_{arch}_sd_c1_s{s}"]) - anchor) / anchor < flow_tol


@pytest.mark.skipif(not (HAVE_CACHE and HAVE_CKPT), reason="run scripts/toy_estimation.py first")
def test_checkpoint_reproduces_the_cached_slice():
    z = np.load(os.path.join(REPO, CACHE))
    flow = load_flow("post", 0)
    p = fiber_slice(lambda pts: flow_logpdf(flow, pts), FIBERS[0])
    assert np.max(np.abs(p - z["hint_post_c1_s0"])) < 1e-6
