"""The two-dimensional example's closed forms, each checked against an independent route."""
from __future__ import annotations

import numpy as np
from numpy.polynomial.hermite_e import hermegauss
from scipy.integrate import trapezoid
from scipy.special import ndtri

from resolvability.toy import (FIBERS, LAPLACE_RHO, LG_TRUTH, RHO, S2, TRUTH, TWIN_A, Gauss2,
                               as_gmm, blind_marginal, curated_prior, data_marginal,
                               fiber_conditional, gaussian_freeze_residual,
                               laplace_freeze_residual, lg_coverage, log_density, log_reweight,
                               map_gmm_posterior, map_ridge, posterior_archive, posterior_gmm,
                               sample_measurements, sbc_pit_moments, sbc_rank_bins,
                               single_best_archive)


def _grid(pad=6.0, n=161):
    xr = np.linspace(-pad * 1.2, pad * 1.2, n)
    xb = np.linspace(-pad * 1.1, pad * 1.1, n)
    XR, XB = np.meshgrid(xr, xb, indexing="ij")
    return np.stack([XR, XB], axis=-1)


def _law_moments(p):
    g = as_gmm(p)
    mean = (g.w[:, None] * g.mu).sum(0)
    ex2 = (g.w[:, None, None] * (g.Sig + np.einsum("ji,jk->jik", g.mu, g.mu))).sum(0)
    return mean, ex2 - np.outer(mean, mean)


def test_em_step_identity():
    """The curated prior two ways: the data law pushed through the posterior, and rho times the
    EM reweight."""
    X = _grid()
    lhs = log_density(curated_prior(RHO, TWIN_A, S2), X)
    rhs = log_density(RHO, X) + log_reweight(RHO, TWIN_A, S2, X[..., 0])
    live = lhs > lhs.max() + np.log(1e-12)
    assert np.max(np.abs(lhs[live] - rhs[live])) < 1e-8


def test_data_marginal_adds_the_noise_variance():
    """Both routes of the identity above share ``data_marginal``, so it is checked on its own, by
    quadrature of the resolved marginal against the likelihood."""
    dm = data_marginal(TWIN_A, S2)
    assert np.allclose(dm.var, TWIN_A.Sig[:, 0, 0] + S2)
    w0, m0, v0 = TWIN_A.w, TWIN_A.mu[:, 0], TWIN_A.Sig[:, 0, 0]
    xr = np.linspace(-9, 9, 20001)
    for y in (-1.5, 0.0, 0.7, 2.2):
        resolved = sum(wj * np.exp(-0.5 * ((xr - mj) ** 2 / vj)) / np.sqrt(2 * np.pi * vj)
                       for wj, mj, vj in zip(w0, m0, v0))
        lik = np.exp(-0.5 * (y - xr) ** 2 / S2) / np.sqrt(2 * np.pi * S2)
        assert abs(trapezoid(resolved * lik, xr) - dm.pdf(y)) < 1e-10


def test_blind_conditional_is_frozen():
    """The curated prior's blind conditional is rho's on every fiber, and in the linear-Gaussian
    case the blind-block precision is unchanged."""
    assert gaussian_freeze_residual(RHO, TWIN_A, S2, FIBERS + (-2.0, 0.7)) < 1e-8
    Sig = curated_prior(RHO, LG_TRUTH, S2).Sig[0]
    prec22 = Sig[0, 0] / np.linalg.det(Sig)
    prec22_rho = RHO.Sig[0, 0] / np.linalg.det(RHO.Sig)
    assert abs(prec22 - prec22_rho) < 1e-10


def test_em_step_does_not_decrease_the_data_likelihood():
    def ll(prior):
        dm_p = data_marginal(prior, S2)
        dm_t = data_marginal(LG_TRUTH, S2)
        a, u2 = dm_p.mean[0], dm_p.var[0]
        b, t2 = dm_t.mean[0], dm_t.var[0]
        return -0.5 * (np.log(2 * np.pi * u2) + (t2 + (b - a) ** 2) / u2)
    rho0 = Gauss2(np.array([0.5, 0.0]), np.array([[0.6, 0.2], [0.2, 1.0]]))
    q = curated_prior(rho0, LG_TRUTH, S2)
    assert ll(Gauss2(q.mu[0], q.Sig[0])) > ll(rho0)
    q_star = curated_prior(Gauss2(LG_TRUTH.mu, LG_TRUTH.Sig), LG_TRUTH, S2)
    assert abs(ll(Gauss2(q_star.mu[0], q_star.Sig[0]))
               - ll(Gauss2(LG_TRUTH.mu, LG_TRUTH.Sig))) < 1e-9


def test_map_lies_on_the_regression_line():
    b0, K = map_ridge(RHO, S2)
    for y in np.linspace(-4, 4, 41):
        x = b0 + K * y
        assert abs(x[1] - fiber_conditional(RHO, x[0]).moments()[0]) < 1e-12


def test_freeze_under_a_laplace_regularizer():
    """The freeze does not need a Gaussian regularizer; the same code with a leaky operator
    ``A = [1, 0.5]`` must fail, so the check has power."""
    assert laplace_freeze_residual(LAPLACE_RHO, TWIN_A, S2, FIBERS, leak=0.0, n_nodes=201) < 1e-8
    assert laplace_freeze_residual(LAPLACE_RHO, TWIN_A, S2, FIBERS, leak=0.5, n_nodes=201) > 1e-2


def test_coverage_law_against_quadrature():
    for r, delta in ((0.5, 0.0), (2.0, 0.0), (0.5, 1.5)):
        for lv in (0.6, 0.9):
            z = ndtri((1 + lv) / 2)
            xb = np.linspace(-z * r, z * r, 100001)
            truth = np.exp(-0.5 * (xb + delta) ** 2) / np.sqrt(2 * np.pi)
            assert abs(trapezoid(truth, xb) - lg_coverage(np.array([lv]), r, delta=delta)[0]) < 1e-8
    levels = np.linspace(0.5, 0.99, 50)
    assert np.allclose(lg_coverage(levels, 1.0), levels, atol=1e-12)
    for r in (0.5, 2.0):
        assert (lg_coverage(levels, r, delta=1.5) < lg_coverage(levels, r) - 1e-6).all()


def test_coverage_of_a_floor_width_band_vanishes_with_the_bandwidth():
    s_star = 2.5
    covs = [lg_coverage(np.array([0.9]), h / s_star)[0] for h in (0.5, 0.25, 0.1, 0.02)]
    assert all(a > b for a, b in zip(covs, covs[1:]))
    assert covs[-1] < 0.02


def test_sbc_pit_is_uniform():
    mom = sbc_pit_moments(TWIN_A, S2, k_max=6, gh_order=80)
    assert np.allclose(mom, 1.0 / np.arange(2, 8), atol=1e-6)


def test_sbc_rank_histogram_is_flat_for_the_curated_prior():
    bins = sbc_rank_bins(curated_prior(RHO, TWIN_A, S2), S2, n_bins=10)
    assert np.max(np.abs(bins - 1.0)) < 1e-3


def test_twin_truths_share_the_data_law_and_the_curated_prior():
    assert np.array_equal(TWIN_A.mu[:, 0], TRUTH.mu[:, 0])
    assert np.array_equal(TWIN_A.Sig[:, 0, 0], TRUTH.Sig[:, 0, 0])
    for c in (-1.0, 0.0, 1.2):
        sa = np.sqrt(fiber_conditional(TWIN_A, c).moments()[1])
        sb = np.sqrt(fiber_conditional(TRUTH, c).moments()[1])
        assert abs(sb / sa - 10.0) < 1e-12
    dm_a, dm_b = data_marginal(TWIN_A, S2), data_marginal(TRUTH, S2)
    assert np.array_equal(dm_a.mean, dm_b.mean) and np.array_equal(dm_a.var, dm_b.var)
    qa, qb = curated_prior(RHO, TWIN_A, S2), curated_prior(RHO, TRUTH, S2)
    assert np.array_equal(qa.mu, qb.mu) and np.array_equal(qa.Sig, qb.Sig)


def test_blind_marginal_moves_only_through_the_correlation():
    """With a correlated rho the curated prior's blind marginal moves while its blind conditional
    stays frozen; with an uncorrelated rho the marginal does not move either."""
    xb = np.linspace(-4, 4, 801)
    T = curated_prior(RHO, TWIN_A, S2)
    moved = np.max(np.abs(blind_marginal(T).pdf(xb) - blind_marginal(RHO).pdf(xb)))
    assert moved > 1e-2
    rho0 = Gauss2(RHO.mu, np.array([[RHO.Sig[0, 0], 0.0], [0.0, RHO.Sig[1, 1]]]))
    T0 = curated_prior(rho0, TWIN_A, S2)
    frozen = np.max(np.abs(blind_marginal(T0).pdf(xb) - blind_marginal(rho0).pdf(xb)))
    assert frozen < 1e-12


def test_map_is_the_global_mode():
    """The multi-start search agrees with a dense-grid argmax, including near ties."""
    for y in (-1.8, -0.3, -0.02, 0.02, 0.4, 2.1):
        post = posterior_gmm(TRUTH, S2, y)
        x_hat = map_gmm_posterior(TRUTH, S2, y)
        g1 = np.linspace(y - 1.5, y + 1.5, 601)
        g2 = np.linspace(-12, 12, 1201)
        G = np.stack(np.meshgrid(g1, g2, indexing="ij"), -1)
        lp = log_density(post, G)
        i, j = np.unravel_index(np.argmax(lp), lp.shape)
        assert log_density(post, x_hat[None, :])[0] >= lp[i, j] - 1e-9
        assert np.hypot(x_hat[0] - g1[i], x_hat[1] - g2[j]) < 0.05


def test_truth_is_a_fixed_point():
    """Posteriors under the true prior, averaged over the true data law, return the truth."""
    nodes, wts = hermegauss(300)
    wts = wts / np.sqrt(2 * np.pi)
    dm = data_marginal(TRUTH, S2)
    X = np.array([[-1.2, 2.0], [0.3, -1.0], [1.5, -3.5], [0.0, 0.0]])
    mix = np.zeros(len(X))
    for wj, aj, vj in zip(dm.w, dm.mean, dm.var):
        for yn, wy in zip(aj + np.sqrt(vj) * nodes, wts):
            mix += wj * wy * np.exp(log_density(posterior_gmm(TRUTH, S2, float(yn)), X))
    assert np.allclose(mix, np.exp(log_density(TRUTH, X)), rtol=1e-9, atol=0)


def test_archive_samplers_draw_the_laws_they_claim():
    rng = np.random.default_rng(11)
    ys = sample_measurements(TRUTH, S2, 200_000, rng)
    m, v = data_marginal(TRUTH, S2).moments()
    assert abs(ys.mean() - m) < 0.02 and abs(ys.var() / v - 1) < 0.02
    X = posterior_archive(RHO, TRUTH, S2, 20_000, rng)
    mu0, C0 = _law_moments(curated_prior(RHO, TRUTH, S2))
    assert np.allclose(X.mean(0), mu0, atol=0.04) and np.allclose(np.cov(X.T), C0, atol=0.06)
    X = posterior_archive(TRUTH, TRUTH, S2, 20_000, rng)
    mu0, C0 = _law_moments(TRUTH)
    assert np.allclose(X.mean(0), mu0, atol=0.12) and np.allclose(np.cov(X.T) / C0, 1, atol=0.06)
    X = single_best_archive(RHO, TRUTH, S2, 2_000, rng)
    slope = RHO.Sig[0, 1] / RHO.Sig[0, 0]
    assert np.max(np.abs(X[:, 1] - (RHO.mu[1] + slope * (X[:, 0] - RHO.mu[0])))) < 1e-12
