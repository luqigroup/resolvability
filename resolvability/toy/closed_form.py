"""Closed forms for the two-dimensional example, and the archives drawn through the pipeline.

The unknown ``x = (x_R, x_B)`` is read through ``A = [1, 0]`` as ``y = x_R + eps``,
``eps ~ N(0, s2)``, so ``x_B`` spans the blind subspace. The regularizer is a correlated Gaussian
and the truth a two-component Gaussian mixture, so every object below is exact:

``posterior_kalman``        the regularizer's posterior, in Kalman form
``data_marginal``           the data law ``p(y)`` of a mixture (each variance gains ``s2``)
``curated_prior``           the curated prior ``T[rho]``, an exact two-component mixture
``log_reweight``            ``log E_y[p(y|x) / rho(y)]``, the EM reweight, in closed form
``fiber_conditional``       the blind conditional ``p(x_B | x_R)``
``map_ridge``               the affine MAP map; a single-best archive lies on its line
``lg_coverage``             the linear-Gaussian coverage law on a blind fiber
``sbc_pit_moments``         moments of the blind PIT, which is uniform for every prior
``blind_dilation``          twin truths that share the data law and differ on the blind fibers
``laplace_freeze_residual`` the freeze under a non-Gaussian (Laplace) regularizer, by quadrature

Variances are carried as variances throughout (``var``). Density ratios are never formed pointwise:
they are integrated in closed form or assembled in log-space. The posterior path never inverts a
covariance. Relative residuals are read only where the reference density exceeds ``1e-12`` of its
maximum.

``curated_prior`` (the pushforward of the data law through the posterior) and ``log_reweight`` (the
ratio integral) are two independent derivations of the same law; the tests compare them, which is
only a check while they stay independent.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from numpy.polynomial.hermite_e import hermegauss
from numpy.polynomial.legendre import leggauss
from scipy.integrate import trapezoid
from scipy.optimize import minimize
from scipy.special import erfcx, logsumexp, ndtr, ndtri

LOG2PI = float(np.log(2.0 * np.pi))


def _log_norm(x, mean, var):
    """``log N(x; mean, var)`` with ``var`` a variance (scalar or array)."""
    return -0.5 * (LOG2PI + np.log(var) + (np.asarray(x) - mean) ** 2 / var)


def _check_spd(S):
    S = np.asarray(S, dtype=float)
    if S.shape != (2, 2) or not np.allclose(S, S.T):
        raise ValueError("covariance must be symmetric 2x2")
    if np.linalg.det(S) <= 0 or S[0, 0] <= 0:
        raise ValueError("covariance must be SPD")
    return S


@dataclass(frozen=True)
class Gauss2:
    """A two-dimensional Gaussian ``N(mu, Sig)``."""
    mu: np.ndarray
    Sig: np.ndarray

    def __post_init__(self):
        object.__setattr__(self, "mu", np.asarray(self.mu, dtype=float))
        object.__setattr__(self, "Sig", _check_spd(self.Sig))


@dataclass(frozen=True)
class GMM2:
    """A two-dimensional Gaussian mixture: weights ``(J,)``, means ``(J, 2)``, covariances
    ``(J, 2, 2)``."""
    w: np.ndarray
    mu: np.ndarray
    Sig: np.ndarray

    def __post_init__(self):
        w = np.asarray(self.w, dtype=float)
        if abs(w.sum() - 1.0) > 1e-12 or (w <= 0).any():
            raise ValueError("weights must be positive and sum to one")
        mu = np.asarray(self.mu, dtype=float)
        Sg = np.asarray(self.Sig, dtype=float)
        if mu.shape != (len(w), 2) or Sg.shape != (len(w), 2, 2):
            raise ValueError("mu must be (J,2) and Sig (J,2,2) with J = len(w)")
        object.__setattr__(self, "w", w)
        object.__setattr__(self, "mu", mu)
        object.__setattr__(self, "Sig", np.stack([_check_spd(S) for S in Sg]))


def as_gmm(p: Gauss2 | GMM2) -> GMM2:
    if isinstance(p, Gauss2):
        return GMM2(np.array([1.0]), p.mu[None, :], p.Sig[None, :, :])
    return p


@dataclass(frozen=True)
class Mix1:
    """A one-dimensional Gaussian mixture: weights, means, variances."""
    w: np.ndarray
    mean: np.ndarray
    var: np.ndarray

    def logpdf(self, x):
        x = np.asarray(x, dtype=float)[..., None]
        return logsumexp(np.log(self.w) + _log_norm(x, self.mean, self.var), axis=-1)

    def pdf(self, x):
        return np.exp(self.logpdf(x))

    def cdf(self, x):
        x = np.asarray(x, dtype=float)[..., None]
        return (self.w * ndtr((x - self.mean) / np.sqrt(self.var))).sum(axis=-1)

    def moments(self):
        """``(mean, variance)``, by the law of total variance."""
        m = float((self.w * self.mean).sum())
        v = float((self.w * (self.var + self.mean**2)).sum() - m**2)
        return m, v


# ------------------------------------------------------------------ posterior, data law, curation

def posterior_kalman(rho: Gauss2, s2: float):
    """``pi_rho(x | y) = N(b0 + K y, Sig_post)``; the gain is a scalar divide."""
    S = rho.Sig
    K = S[:, 0] / (S[0, 0] + s2)
    Sig_post = S - np.outer(K, S[0, :])
    Sig_post = 0.5 * (Sig_post + Sig_post.T)
    b0 = rho.mu - K * rho.mu[0]
    return K, Sig_post, b0


def data_marginal(p: Gauss2 | GMM2, s2: float) -> Mix1:
    """``p(y)``: each component pushed through ``A``, with the noise variance added."""
    g = as_gmm(p)
    return Mix1(g.w, g.mu[:, 0].copy(), g.Sig[:, 0, 0] + s2)


def curated_prior(rho: Gauss2, p_star: Gauss2 | GMM2, s2: float) -> GMM2:
    """``T[rho] = E_{y ~ p_star(y)}[pi_rho(. | y)]``, one component per data-law component."""
    K, Sig_post, b0 = posterior_kalman(rho, s2)
    dm = data_marginal(p_star, s2)
    means = b0[None, :] + K[None, :] * dm.mean[:, None]
    covs = Sig_post[None, :, :] + dm.var[:, None, None] * np.outer(K, K)[None, :, :]
    return GMM2(dm.w.copy(), means, covs)


def log_reweight(rho: Gauss2, p_star: Gauss2 | GMM2, s2: float, x_R):
    """``log g(x_R) = log E_{y ~ p_star(y)}[N(y; x_R, s2) / rho(y)]``, in closed form.

    With ``rho(y) = N(m_R, v)``, ``v = S11 + s2``, completing the square gives
    ``N(y; x_R, s2) / N(y; m_R, v) = exp(c(x_R)) exp(-(y - mu_t)^2 / (2 vt))`` with
    ``1/vt = 1/s2 - 1/v > 0`` (``vt = s2 v / S11``), so the integral against each data component
    ``N(a_j, tau2_j)`` is ``sqrt(2 pi vt) N(a_j; mu_t, vt + tau2_j)``.
    """
    x_R = np.asarray(x_R, dtype=float)
    S11 = rho.Sig[0, 0]
    m_R = rho.mu[0]
    v = S11 + s2
    vt = s2 * v / S11
    mu_t = vt * (x_R / s2 - m_R / v)
    log_c = 0.5 * (np.log(v / s2) + mu_t**2 / vt - x_R**2 / s2 + m_R**2 / v)
    dm = data_marginal(p_star, s2)
    lg = logsumexp(
        np.log(dm.w) + _log_norm(dm.mean, mu_t[..., None], vt + dm.var), axis=-1
    )
    return log_c + 0.5 * (LOG2PI + np.log(vt)) + lg


def log_density(p: Gauss2 | GMM2, x):
    """Log density of a ``Gauss2``/``GMM2`` at points ``x`` of shape ``(..., 2)``."""
    g = as_gmm(p)
    x = np.asarray(x, dtype=float)
    d = x[..., None, :] - g.mu
    Sinv = np.linalg.inv(g.Sig)
    quad = np.einsum("...ji,jik,...jk->...j", d, Sinv, d)
    logdet = np.log(np.linalg.det(g.Sig))
    comp = -0.5 * (2 * LOG2PI + logdet + quad)
    return logsumexp(np.log(g.w) + comp, axis=-1)


# ------------------------------------------------------------------ fibers, marginals, the ridge

def fiber_conditional(p: Gauss2 | GMM2, x_R: float) -> Mix1:
    """``p(x_B | x_R)``: the component conditionals, reweighted by each component's ``p(x_R)``."""
    g = as_gmm(p)
    logw = np.log(g.w) + _log_norm(x_R, g.mu[:, 0], g.Sig[:, 0, 0])
    w = np.exp(logw - logsumexp(logw))
    slope = g.Sig[:, 0, 1] / g.Sig[:, 0, 0]
    mean = g.mu[:, 1] + slope * (x_R - g.mu[:, 0])
    var = g.Sig[:, 1, 1] - g.Sig[:, 0, 1] ** 2 / g.Sig[:, 0, 0]
    return Mix1(w, mean, var)


def blind_marginal(p: Gauss2 | GMM2) -> Mix1:
    g = as_gmm(p)
    return Mix1(g.w.copy(), g.mu[:, 1].copy(), g.Sig[:, 1, 1].copy())


def map_ridge(rho: Gauss2, s2: float):
    """``MAP_rho(y) = b0 + K y``, the posterior mean (a Gaussian's mode).

    The pushforward of any data law through this map lies on rho's regression line
    ``x_B = m_B + (S12 / S11)(x_R - m_R)``: given ``x_R`` the blind value is the conditional mean,
    a point mass.
    """
    K, _, b0 = posterior_kalman(rho, s2)
    return b0, K


# ------------------------------------------------------------------ the coverage law

def lg_coverage(levels, ratio: float, delta: float = 0.0):
    """Achieved coverage of the central-``level`` reported band on a blind fiber.

    ``C = Phi(delta + z r) - Phi(delta - z r)`` with ``z = Phi^{-1}((1 + level) / 2)`` and
    ``r = s_rho / s_star``; ``delta = 0`` gives ``2 Phi(z r) - 1``.
    """
    z = ndtri((1.0 + np.asarray(levels, dtype=float)) / 2.0)
    return ndtr(delta + z * ratio) - ndtr(delta - z * ratio)


# ------------------------------------------------------------------ self-consistency (SBC)

def posterior_gmm(prior: Gauss2 | GMM2, s2: float, y: float) -> GMM2:
    """The posterior under a mixture prior: a Kalman update per component, reweighted."""
    g = as_gmm(prior)
    logw = np.log(g.w) + _log_norm(y, g.mu[:, 0], g.Sig[:, 0, 0] + s2)
    w = np.exp(logw - logsumexp(logw))
    mus, covs = [], []
    for j in range(len(g.w)):
        K, Sig_post, b0 = posterior_kalman(Gauss2(g.mu[j], g.Sig[j]), s2)
        mus.append(b0 + K * y)
        covs.append(Sig_post)
    return GMM2(w, np.stack(mus), np.stack(covs))


def sbc_pit_moments(prior: Gauss2 | GMM2, s2: float, k_max: int = 6, gh_order: int = 80):
    """``E[U^k]`` for the blind PIT ``U = F_{x_B | y}(x_B*)``, ``x* ~ prior``, ``y ~ N(x*_R, s2)``.

    ``U`` is uniform for every prior, so the exact values are ``1 / (k + 1)``; this evaluates them
    by nested Gauss--Hermite quadrature.
    """
    nodes, wts = hermegauss(gh_order)
    wts = wts / np.sqrt(2.0 * np.pi)
    dm = data_marginal(prior, s2)
    moments = np.zeros(k_max)
    for wj, aj, t2j in zip(dm.w, dm.mean, dm.var):
        for yn, wy in zip(aj + np.sqrt(t2j) * nodes, wts):
            post = posterior_gmm(prior, s2, yn)
            bm = blind_marginal(post)
            for wi, mi, vi in zip(bm.w, bm.mean, bm.var):
                xb = mi + np.sqrt(vi) * nodes
                U = bm.cdf(xb)
                for k in range(1, k_max + 1):
                    moments[k - 1] += wj * wy * wi * (wts * U**k).sum()
    return moments


def sbc_rank_bins(prior: Gauss2 | GMM2, s2: float, n_bins: int = 10, gh_order: int = 80):
    """The SBC rank histogram as bin densities (exactly 1 per bin when uniform)."""
    nodes, wts = hermegauss(gh_order)
    wts = wts / np.sqrt(2.0 * np.pi)
    dm = data_marginal(prior, s2)
    masses = np.zeros(n_bins)
    for wj, aj, t2j in zip(dm.w, dm.mean, dm.var):
        for yn, wy in zip(aj + np.sqrt(t2j) * nodes, wts):
            post = posterior_gmm(prior, s2, yn)
            bm = blind_marginal(post)
            m0, v0 = bm.moments()
            # The bin indicator is discontinuous, where Gauss--Hermite converges poorly, so x_B is
            # integrated on a dense trapezoid grid instead.
            xb = np.linspace(m0 - 9 * np.sqrt(v0), m0 + 9 * np.sqrt(v0), 40001)
            dens = bm.pdf(xb)
            dens = dens / trapezoid(dens, xb)
            U = bm.cdf(xb)
            idx = np.clip((U * n_bins).astype(int), 0, n_bins - 1)
            wcell = dens * np.gradient(xb)
            np.add.at(masses, idx, wj * wy * wcell)
    return masses * n_bins


# ------------------------------------------------------------------ twin truths

def blind_dilation(p: GMM2, c: float, center: float = 0.0) -> GMM2:
    """``(x_R, x_B) -> (x_R, center + c (x_B - center))``.

    The resolved parameters are unchanged and every blind conditional is dilated by exactly ``c``,
    so the two truths share the data law.
    """
    mu = p.mu.copy()
    mu[:, 1] = center + c * (mu[:, 1] - center)
    Sig = p.Sig.copy()
    Sig[:, 0, 1] *= c
    Sig[:, 1, 0] *= c
    Sig[:, 1, 1] *= c**2
    return GMM2(p.w.copy(), mu, Sig)


# ------------------------------------------------------------------ the freeze, two regularizers

def gaussian_freeze_residual(rho: Gauss2, p_star: Gauss2 | GMM2, s2: float, fibers,
                             span: float = 7.0, n: int = 1401) -> float:
    """Largest relative residual of ``T[rho](x_B | x_R)`` against ``rho(x_B | x_R)``, over
    ``fibers``."""
    T = curated_prior(rho, p_star, s2)
    xb = np.linspace(-span, span, n)
    worst = 0.0
    for c in fibers:
        a = fiber_conditional(T, c).pdf(xb)
        b = fiber_conditional(rho, c).pdf(xb)
        live = b > 1e-12 * b.max()
        worst = max(worst, float(np.max(np.abs(a[live] - b[live]) / b[live])))
    return worst


@dataclass(frozen=True)
class LaplaceRho:
    """A correlated Laplace regularizer: ``x_R ~ Laplace(m_R, b_R)`` and
    ``x_B | x_R ~ Laplace(m_B + gamma (x_R - m_R), b_B)``."""
    m: np.ndarray
    b_R: float
    gamma: float
    b_B: float

    def log_pdf_R(self, x_R):
        return -np.abs(np.asarray(x_R) - self.m[0]) / self.b_R - np.log(2 * self.b_R)

    def cond_mean(self, x_R):
        return self.m[1] + self.gamma * (np.asarray(x_R) - self.m[0])

    def log_pdf_cond(self, x_B, x_R):
        return -np.abs(np.asarray(x_B) - self.cond_mean(x_R)) / self.b_B - np.log(2 * self.b_B)


def _log_erfcx(u):
    """``log erfcx(u)`` for both signs: for ``u < -1``, ``erfcx(u) = 2 exp(u^2) - erfcx(-u)``."""
    u = np.asarray(u, dtype=float)
    out = np.empty_like(u)
    pos = u >= -1.0
    out[pos] = np.log(erfcx(u[pos]))
    un = u[~pos]
    corr = erfcx(-un) * np.exp(-np.minimum(un**2, 700.0))
    out[~pos] = un**2 + np.log(2.0 - corr)
    return out


def _log_laplace_gauss_conv(d, beta: float, s2: float):
    """``log (Laplace(0, beta) * N(0, s2))(d)``, via
    ``exp(-d^2 / 2 s2) / (4 beta) [erfcx(u-) + erfcx(u+)]``,
    ``u+- = sqrt(s2 / 2) / beta +- d / sqrt(2 s2)``."""
    d = np.asarray(d, dtype=float)
    a = np.sqrt(s2 / 2.0) / beta
    b = d / np.sqrt(2.0 * s2)
    base = -d**2 / (2.0 * s2) - np.log(4.0 * beta)
    return base + np.logaddexp(_log_erfcx(a - b), _log_erfcx(a + b))


def laplace_freeze_residual(lap: LaplaceRho, p_star: Gauss2 | GMM2, s2: float, fibers,
                            leak: float = 0.0, span: float = 9.0, n_nodes: int = 501) -> float:
    """Largest relative residual of ``T[rho_L](x_B | x_R)`` against ``rho_L(x_B | x_R)``.

    The likelihood is ``y = x_R + leak * x_B + eps``. The curated density is assembled from the
    integrand ``p_star(y) N(y | x_R + leak x_B, s2) rho_L(x) / rho_L(y)`` in log-space, with the
    ``y`` quadrature run separately at every ``x_B``, so at ``leak = 0`` the reweight's independence
    of ``x_B`` comes out of the numerics rather than being factored out by hand. At ``leak > 0``
    the operator sees ``x_B`` and the residual is large.
    """
    tt, wtt = leggauss(n_nodes)
    m_R = lap.m[0]
    # rho_L(y): x_R' is integrated piecewise on either side of the Laplace kink; for leak != 0 the
    # inner x_B' integral is the exact Laplace-Gaussian convolution.
    pieces = []
    for lo, hi in ((-span, m_R), (m_R, span)):
        half, mid = 0.5 * (hi - lo), 0.5 * (hi + lo)
        pieces.append((mid + half * tt, half * wtt))
    xr_nodes = np.concatenate([p[0] for p in pieces])
    xr_w = np.concatenate([p[1] for p in pieces])

    def log_rho_y(y):
        y = np.asarray(y, dtype=float)
        mu_c = xr_nodes + leak * lap.cond_mean(xr_nodes)
        if leak == 0.0:
            lk = _log_norm(y[..., None], xr_nodes, s2)
        else:
            lk = _log_laplace_gauss_conv(y[..., None] - mu_c, abs(leak) * lap.b_B, s2)
        return logsumexp(lk + lap.log_pdf_R(xr_nodes) + np.log(xr_w), axis=-1)

    dm = data_marginal(p_star, s2)
    s = np.sqrt(s2)
    worst = 0.0
    for c in fibers:
        cond = lap.cond_mean(c)
        xb = cond + np.linspace(-8 * lap.b_B, 8 * lap.b_B, 401)
        centers = c + leak * xb
        half = 10.0 * s
        y = centers[:, None] + half * tt[None, :]
        logw_y = np.log(half * wtt)[None, :]
        base = dm.logpdf(y) + _log_norm(y, centers[:, None], s2) - log_rho_y(y) + logw_y
        log_T = lap.log_pdf_R(c) + lap.log_pdf_cond(xb, c) + logsumexp(base, axis=1)
        T = np.exp(log_T - log_T.max())
        T /= trapezoid(T, xb)
        ref = np.exp(lap.log_pdf_cond(xb, c))
        ref /= trapezoid(ref, xb)             # same discrete measure for both, so the
        mask = ref > 1e-12 * ref.max()        # comparison is of shape only, kink included
        worst = max(worst, float(np.max(np.abs(T[mask] - ref[mask]) / ref[mask])))
    return worst


# ------------------------------------------------------------------ archives, through the pipeline

def sample_gmm2(g: Gauss2 | GMM2, n: int, rng) -> np.ndarray:
    """``n`` independent samples of a ``Gauss2``/``GMM2``."""
    g = as_gmm(g)
    comp = rng.choice(len(g.w), size=n, p=g.w)
    L = np.linalg.cholesky(g.Sig)
    return g.mu[comp] + np.einsum("nij,nj->ni", L[comp], rng.standard_normal((n, 2)))


def sample_measurements(p_star: Gauss2 | GMM2, s2: float, n: int, rng) -> np.ndarray:
    """``n`` measurements ``y ~ p_star(y)``."""
    dm = data_marginal(p_star, s2)
    comp = rng.choice(len(dm.w), size=n, p=dm.w)
    return dm.mean[comp] + np.sqrt(dm.var[comp]) * rng.standard_normal(n)


def map_gmm_posterior(prior: Gauss2 | GMM2, s2: float, y: float) -> np.ndarray:
    """The global mode of ``pi(x | y)`` under a mixture prior.

    Nelder--Mead from every posterior component mean and from their weighted mean; the best local
    optimum wins. This finds the global mode when the prior's components share one covariance, as
    every prior here does: the posterior components then share one too, every critical point lies
    on the segment between the means (Ray and Lindsay, 2005), and an ascent from each mean reaches
    its side's mode. Unequal covariances can put modes off the segment.
    """
    post = posterior_gmm(prior, s2, y)
    starts = list(post.mu) + [(post.w[:, None] * post.mu).sum(0)]
    best, best_val = None, np.inf
    for x0 in starts:
        r = minimize(lambda x: -log_density(post, x[None, :])[0], x0, method="Nelder-Mead",
                     options=dict(xatol=1e-10, fatol=1e-12, maxiter=4000))
        if not r.success:
            warnings.warn(f"Nelder-Mead did not converge at y={y}")
        if r.fun < best_val:
            best, best_val = r.x, r.fun
    return best


def posterior_archive(prior: Gauss2 | GMM2, p_star: Gauss2 | GMM2, s2: float, n: int,
                      rng) -> np.ndarray:
    """A posterior-sample archive: ``y ~ p_star(y)``, then one sample of ``pi_prior(x | y)``
    for each."""
    ys = sample_measurements(p_star, s2, n, rng)
    return np.stack([sample_gmm2(posterior_gmm(prior, s2, float(y)), 1, rng)[0] for y in ys])


def single_best_archive(prior: Gauss2 | GMM2, p_star: Gauss2 | GMM2, s2: float, n: int,
                        rng) -> np.ndarray:
    """A single-best archive: ``y ~ p_star(y)``, then the global posterior mode for each.

    Under a single Gaussian prior the mode is the affine map of ``map_ridge``.
    """
    ys = sample_measurements(p_star, s2, n, rng)
    g = as_gmm(prior)
    if len(g.w) == 1:
        b0, K = map_ridge(Gauss2(g.mu[0], g.Sig[0]), s2)
        return b0[None, :] + K[None, :] * ys[:, None]
    return np.stack([map_gmm_posterior(prior, s2, float(y)) for y in ys])


def dequantized(g: Gauss2 | GMM2, h: float) -> GMM2:
    """The law of ``X + N(0, h^2 I)`` for ``X ~ g``: every component gains ``h^2 I``.

    On the single-best archive, which lies on a line, the blind conditional of this law is the
    dequantization widened by the line's slope, about ``h sqrt(1 + b^2)`` rather than ``h``.
    """
    g = as_gmm(g)
    return GMM2(g.w.copy(), g.mu.copy(), g.Sig + (h**2) * np.eye(2)[None, :, :])


def sb_pushforward(rho: Gauss2, p_star: Gauss2 | GMM2, s2: float) -> GMM2:
    """The single-best archive's law under a Gaussian regularizer.

    The data law pushed through the affine MAP map; each component has the rank-one covariance
    ``tau2_j K K^T``, lifted by ``1e-12 I`` so it is a valid ``GMM2``. Meant to be read through
    :func:`dequantized`, which restores full rank.
    """
    b0, K = map_ridge(rho, s2)
    dm = data_marginal(p_star, s2)
    mus = b0[None, :] + K[None, :] * dm.mean[:, None]
    Sigs = dm.var[:, None, None] * np.outer(K, K)[None, :, :] + 1e-12 * np.eye(2)[None, :, :]
    return GMM2(dm.w.copy(), mus, Sigs)


# ------------------------------------------------------------------ the example's parameters

S2 = 0.0625                                                # noise variance, s = 0.25
RHO = Gauss2(np.array([0.0, 0.0]), np.array([[1.0, 0.6], [0.6, 1.0]]))
TWIN_A = GMM2(
    np.array([0.5, 0.5]),
    np.array([[-1.2, 0.2], [1.2, -0.2]]),
    np.array([[[0.35, 0.10], [0.10, 0.09]], [[0.35, 0.10], [0.10, 0.09]]]),
)
DILATION = 10.0
TRUTH = blind_dilation(TWIN_A, DILATION)                   # the truth of the figures
FIBERS = (0.0, 1.2)                                        # the blind fibers x_R = c read
H_FLOOR = 0.25                                             # the dequantization bandwidth h
LAPLACE_RHO = LaplaceRho(np.array([0.0, 0.0]), 0.7, 0.6, 0.55)
LG_TRUTH = Gauss2(np.array([0.0, 0.0]), np.array([[1.0, 0.6], [0.6, 1.5]]))


def estimation_anchors(h: float, fiber: float) -> dict[str, float]:
    """Exact blind standard deviations, on the fiber ``x_R = fiber``, of the three dequantized
    training laws: the posterior-sample archive, the single-best archive, and the truth."""
    laws = {"post": curated_prior(RHO, TRUTH, S2),
            "sb": sb_pushforward(RHO, TRUTH, S2),
            "oracle": TRUTH}
    return {k: float(np.sqrt(fiber_conditional(dequantized(v, h), fiber).moments()[1]))
            for k, v in laws.items()}
