"""The two-dimensional example: the paper's mechanism where every prediction has a closed form.

``x = (x_R, x_B)`` is read through ``A = [1, 0]``, so ``x_B`` spans the blind subspace. A
two-component truth is reconstructed under a single correlated Gaussian regularizer, or under the
truth itself, into posterior-sample and single-best archives, all drawn through one pipeline.

``closed_form``  the exact laws, the archive samplers, and the example's parameters (numpy/scipy).
``estimation``   reading a fitted density's blind spread, and the fit cache's checks (torch).

The closed forms are re-exported here; ``estimation`` is imported on its own.
"""
from __future__ import annotations

from resolvability.toy.closed_form import (DILATION, FIBERS, GMM2, H_FLOOR, LAPLACE_RHO, LG_TRUTH,
                                           RHO, S2, TRUTH, TWIN_A, Gauss2, LaplaceRho, Mix1,
                                           as_gmm, blind_dilation, blind_marginal, curated_prior,
                                           data_marginal, dequantized, estimation_anchors,
                                           fiber_conditional, gaussian_freeze_residual,
                                           laplace_freeze_residual, lg_coverage, log_density,
                                           log_reweight, map_gmm_posterior, map_ridge,
                                           posterior_archive, posterior_gmm, posterior_kalman,
                                           sample_gmm2, sample_measurements, sb_pushforward,
                                           sbc_pit_moments, sbc_rank_bins, single_best_archive)

__all__ = ["DILATION", "FIBERS", "GMM2", "H_FLOOR", "LAPLACE_RHO", "LG_TRUTH", "RHO", "S2", "TRUTH",
           "TWIN_A", "Gauss2", "LaplaceRho", "Mix1", "as_gmm", "blind_dilation", "blind_marginal",
           "curated_prior", "data_marginal", "dequantized", "estimation_anchors",
           "fiber_conditional", "gaussian_freeze_residual", "laplace_freeze_residual",
           "lg_coverage", "log_density", "log_reweight", "map_gmm_posterior", "map_ridge",
           "posterior_archive", "posterior_gmm", "posterior_kalman", "sample_gmm2",
           "sample_measurements", "sb_pushforward", "sbc_pit_moments", "sbc_rank_bins",
           "single_best_archive"]
