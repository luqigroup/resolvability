"""Reading a fitted prior's blind spread in the two-dimensional example, and checking the fit cache.

Priors are fit to three training sets -- the regularizer's posterior-sample archive (``post``), its
single-best archive (``sb``), and the truths themselves (``oracle``) -- each dequantized by
``N(0, h^2 I)``. A fit's blind spread is the standard deviation of its density restricted to a fiber
``x_R = c`` and normalized over ``x_B``, which for the exact dequantized law is known in closed form
(:func:`resolvability.toy.closed_form.estimation_anchors`).

``scripts/toy_estimation.py`` writes the fits to ``results/toy_estimation.npz`` and the flows to
``data/checkpoints/toy_flows/``.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.integrate import trapezoid

from resolvability.download import ensure
from resolvability.groundwater.hint_flow import HINTFlow
from resolvability.toy.closed_form import (FIBERS, H_FLOOR, RHO, S2, TRUTH,
                                           estimation_anchors)

CACHE = "results/toy_estimation.npz"
CKPT = "data/checkpoints/toy_flows/{arch}_s{seed}.pth"
ARCHIVES = ("post", "sb", "oracle")
N_TRAIN, N_VAL = 4000, 1000
XB_GRID = np.linspace(-14.0, 14.0, 1121)      # wide enough that the widest law's tail is < 1e-3

# The HINT flow of the groundwater example, at the size of this one: eight coupling layers of tree
# depth one, three-layer networks of width 64.
FLOW = dict(n_hidden=64, n_flow_layers=8, depth=1, n_mlp_layers=3)


def make_flow(**kw) -> HINTFlow:
    return HINTFlow(2, n_cond=0, **{**FLOW, **kw})


def load_flow(arch: str, seed: int) -> HINTFlow:
    """A trained flow from ``data/checkpoints/toy_flows/``, in eval mode."""
    ck = torch.load(ensure(CKPT.format(arch=arch, seed=seed)), map_location="cpu",
                    weights_only=False)
    flow = make_flow(n_hidden=ck["n_hidden"], n_flow_layers=ck["n_flow_layers"],
                     depth=ck["depth"], n_mlp_layers=ck["n_mlp_layers"])
    flow.load_state_dict(ck["state"])
    return flow.eval()


def flow_logpdf(flow: HINTFlow, pts: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return flow.log_prob(torch.as_tensor(pts, dtype=torch.float32)).numpy()


def fiber_slice(logpdf_fn, c: float) -> np.ndarray:
    """A density restricted to the fiber ``x_R = c``, normalized over ``XB_GRID``."""
    pts = np.stack([np.full_like(XB_GRID, c), XB_GRID], axis=-1)
    lp = logpdf_fn(pts)
    p = np.exp(lp - lp.max())
    return p / trapezoid(p, XB_GRID)


def slice_sd(p: np.ndarray) -> float:
    m = trapezoid(p * XB_GRID, XB_GRID)
    return float(np.sqrt(trapezoid(p * (XB_GRID - m) ** 2, XB_GRID)))


def check_cache(z, fiber: str = "c1") -> None:
    """Refuse a cache made from other parameters than the ones in :mod:`resolvability.toy`.

    The cache stores the inputs it was generated from; they, and its exact anchors, must match the
    current parameters.
    """
    stale = "results/toy_estimation.npz is stale: rerun scripts/toy_estimation.py"
    for key, val in (("gen_s2", S2), ("gen_rho_mu", RHO.mu), ("gen_rho_Sig", RHO.Sig),
                     ("gen_truth_w", TRUTH.w), ("gen_truth_mu", TRUTH.mu),
                     ("gen_truth_Sig", TRUTH.Sig), ("gen_h", H_FLOOR)):
        if key not in z.files or not np.allclose(z[key], val, rtol=0, atol=1e-12):
            raise RuntimeError(f"{stale} ({key})")
    c = FIBERS[0] if fiber == "c1" else FIBERS[1]
    for arch, sd in estimation_anchors(H_FLOOR, c).items():
        if abs(float(z[f"anchor_{arch}_sd_{fiber}"]) - sd) > 1e-9:
            raise RuntimeError(f"{stale} (anchor {arch})")
