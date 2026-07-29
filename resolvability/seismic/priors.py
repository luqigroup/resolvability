"""The two deployed seismic priors: architecture, checkpoint lookup, evaluation images, sampling.

Both priors are the same unconditional UNet DDPM over 256 x 256 reflectivity, trained on
per-pixel standardized images. They differ only in what they were shown:

``oracle``   the broadband reflectivity itself (``broadband_dm``) -- the calibrated reference.
``curated``  the legacy least-squares migration archive (``lsrtm5``) -- the deployed prior.

Three training seeds per arm are shipped, as ``data/checkpoints/seismic_prior_<tag>_seed<i>.pth``.
Each checkpoint carries its own per-pixel normalization statistics, so samples are returned in
physical reflectivity units.
"""

from __future__ import annotations

import numpy as np
import torch
from diffusers import DDIMScheduler, DDPMScheduler, UNet2DModel

from . import MUTE_END, N, TAPER
from ..download import ensure
from ..utils import Normalizer
from .pseudosynth import water_mute

BASE = 64                    # UNet width of the deployed priors
N_SEEDS = 3                  # training seeds shipped per arm

CKPT = "data/checkpoints/seismic_prior_{tag}_seed{seed}.pth"
EVAL_WINDOW = "data/seismic/eval_window.npz"
EVAL_H5 = "data/seismic/dataset_eval.h5"


def make_unet(base: int = BASE) -> UNet2DModel:
    """The prior's score network: a four-scale UNet with attention at the coarsest two scales."""
    return UNet2DModel(
        sample_size=N, in_channels=1, out_channels=1, layers_per_block=2,
        block_out_channels=(base, base * 2, base * 4, base * 4), norm_num_groups=min(32, base),
        down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D"))


def find_all_ckpts(tag: str, n_seeds: int = N_SEEDS) -> list[tuple[str, int]]:
    """Every training seed of the ``oracle`` or ``curated`` prior, in seed order.

    Returns ``(path, base)`` pairs ordered by the seed index in the filename, so the arm's seed
    ``i`` is the same checkpoint on every machine and after any clone or unpack.
    """
    if tag not in ("oracle", "curated"):
        raise ValueError(f"tag must be 'oracle' or 'curated'; got {tag!r}")
    return [(ensure(CKPT.format(tag=tag, seed=s)), BASE) for s in range(n_seeds)]


def find_ckpt(tag: str) -> tuple[str, int]:
    """Seed 0 of the ``oracle`` or ``curated`` prior, as ``(path, base)``."""
    return find_all_ckpts(tag, n_seeds=1)[0]


def _load_eval_h5(key: str, a: int, b: int) -> np.ndarray:
    """Read rows ``[a, b)`` of ``key`` from the full evaluation archive."""
    import h5py

    try:
        path = ensure(EVAL_H5)
    except RuntimeError as exc:
        raise FileNotFoundError(
            f"rows [{a}, {b}) of {key!r} fall outside the shipped evaluation window, so they must "
            f"come from {EVAL_H5}, which is not on disk: {exc}") from exc
    with h5py.File(path, "r") as f:
        if key not in f:
            raise KeyError(f"{key!r} is not a dataset in {EVAL_H5}")
        return f[key][a:b].astype(np.float32)


def load_eval(key: str, a: int, b: int) -> np.ndarray:
    """Water-muted evaluation images, rows ``[a, b)`` of ``key``, flattened.

    Reads the slim window ``data/seismic/eval_window.npz`` when the requested rows lie inside it
    (they do for everything the figures draw) and falls back to the full archive
    ``data/seismic/dataset_eval.h5`` otherwise. Returns a (b - a, N*N) float32 array in physical
    reflectivity units, with the water column muted exactly as the imager mutes it.
    """
    z = np.load(ensure(EVAL_WINDOW))
    row0 = int(z["row0"])
    if key in z.files and a >= row0 and b <= row0 + z[key].shape[0]:
        X = z[key][a - row0:b - row0].astype(np.float32)
    else:
        X = _load_eval_h5(key, a, b)
    if len(X) != b - a:
        raise IndexError(f"asked for rows [{a}, {b}) of {key!r} but got {len(X)}")
    return np.stack([water_mute(x, MUTE_END, TAPER) for x in X]).reshape(b - a, N * N)


def archive_row(bases, lo, ne, levels=(0.9,)):
    """The legacy archive itself, scored on the same subspaces as the priors it trains.

    One definition, shared by the calibration script and ``scripts/fig_gap.py``: the archive is the
    deficit the curated prior is said to have inherited, so the two must never drift apart. The
    archive carries a physical amplitude rather than a generative model's, so it is put on the same
    data-fixed scale the priors use, ``kap = sqrt(E_sig / E_archive)``, which is measurement-only and
    never references a truth.

    ``bases`` is ``{name: Q}`` with ``Q`` of shape ``(r, N*N)``. Returns
    ``{name: (coverage_at_levels, spread_ratio)}``.
    """
    truth = load_eval("broadband_dm", lo, lo + ne)
    arch = load_eval("lsrtm5", lo, lo + ne)
    kz = np.load(ensure("results/seismic_data_kappa.npz"))
    kap = float(np.sqrt(kz["E_sig_rows"][lo:lo + ne].mean() / kz["E_archive"].mean()))

    out = {}
    for name, Q in bases.items():
        Pt, P = truth @ Q.T, (arch @ Q.T) * kap
        cov = []
        for a in levels:
            q0, q1 = np.quantile(P, (1 - a) / 2, 0), np.quantile(P, (1 + a) / 2, 0)
            cov.append(float(np.mean([((Pt[i] >= q0) & (Pt[i] <= q1)).mean()
                                      for i in range(len(Pt))])))
        out[name] = (np.array(cov), float(P.var(0).sum() / Pt.var(0).sum()))
    return out


@torch.no_grad()
def sample_prior(ckpt_path: str, base: int, n: int, chunk: int, n_steps: int,
                 device, sampler: str = "ddpm") -> np.ndarray:
    """Draw ``n`` unconditional samples from a trained prior, in physical units.

    ``sampler="ddim"`` is deterministic (eta = 0), so no per-step noise injection smears into the
    blind subspace the samples are read on; ``"ddpm"`` is the ancestral, stochastic sampler.
    Returns an (n, N*N) float32 array. Needs a GPU to be practical.
    """
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    unet = make_unet(base).to(device)
    unet.load_state_dict(ck["model"])
    unet.eval()
    norm = Normalizer.from_stats(ck["norm_mean"].to(device), ck["norm_std"].to(device))
    Sched = DDIMScheduler if sampler == "ddim" else DDPMScheduler
    # clip_sample=False: the images are per-pixel standardized, not in [-1, 1], so the scheduler's
    # default clamp of the predicted x0 would flatten every reflector.
    sched = Sched(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2", clip_sample=False)
    sched.set_timesteps(n_steps, device=device)
    out = []
    for i in range(0, n, chunk):
        b = min(chunk, n - i)
        x = torch.randn(b, 1, N, N, device=device)
        for t in sched.timesteps:
            x = sched.step(unet(x, t.expand(b)).sample, t, x).prev_sample
        out.append(norm.unnormalize(x.squeeze(1)).cpu().numpy())
    return np.concatenate(out, 0).reshape(n, N * N)
