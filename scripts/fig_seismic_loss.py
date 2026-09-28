"""Train/validation denoising-loss curves for the two seismic diffusion priors, side by side (1x2).

Companion to fig_seismic_training.py (the sample galleries). Reads only the loss history stored in
each prior's seed-0 checkpoint -- no GPU, no image download.
Run:  python scripts/fig_seismic_loss.py
"""
from __future__ import annotations

import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MultipleLocator  # noqa: E402

from resolvability.download import REPO, ensure                 # noqa: E402
from resolvability.seismic.priors import find_ckpt              # noqa: E402
from resolvability.style import PALETTE, apply_paper_style      # noqa: E402
apply_paper_style(); plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm"})
EM, MAD = PALETTE["em"], PALETTE["mad"]
OUT = os.path.join(REPO, "figures/fig_seismic_loss.pdf")


def main():
    fig, axes = plt.subplots(1, 2, figsize=(6.986, 2.597))
    # Short titles: a half-width panel has no room for the training targets.
    panels = [("curated", "(a)  curated prior", MAD),
              ("oracle", "(b)  oracle prior", EM)]
    for ax, (tag, title, col) in zip(axes, panels):
        path, _ = find_ckpt(tag)
        h = torch.load(ensure(path), map_location="cpu", weights_only=False)["hist"]
        st, tr, vl = np.asarray(h["step"]), np.asarray(h["train"]), np.asarray(h["val"])
        n = min(len(st), len(tr), len(vl))
        ax.plot(st[:n], tr[:n], color="0.6", lw=1.0, label="train")
        ax.plot(st[:n], vl[:n], color=col, lw=1.6, label="val")
        ax.set_yscale("log")
        ax.xaxis.set_major_locator(MultipleLocator(1000))
        ax.set_xlabel("training step", fontsize=8)
        ax.set_ylabel("denoising MSE", fontsize=8)
        ax.set_title(title, fontsize=8, color=col, loc="left", pad=3.0)
        ax.tick_params(labelsize=7, length=2.5)
        ax.tick_params(which="minor", labelsize=7)
        ax.legend(fontsize=7, frameon=False, loc="upper right")
    fig.tight_layout(w_pad=1.8)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", pad_inches=0.02, dpi=300)
    print("saved", os.path.relpath(OUT, REPO))


if __name__ == "__main__":
    main()
