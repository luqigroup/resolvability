# resolvability

Code, data, and machine-checked proofs for

> **Priors learned from legacy reconstructions inherit undetectable overconfidence.**
> Ali Siahkoohi and Sina Alemohammad. Preprint, 2026.

## Overview

Training a generative prior for an ill-posed inverse problem needs truths, which seismic and
medical imaging do not have. The recourse is an archive of legacy reconstructions. Where the
measurements are uninformative the posterior reverts to the prior, so the confidence reported
there is the archive's — and nothing in practice reveals it: truths differing only on those
directions induce identical data laws, and self-consistency diagnostics pass whatever the prior
assumes.

This repository reproduces the paper's examples and its theory:

| example | operator | prior | what it shows |
|---|---|---|---|
| two-dimensional | `A = [1, 0]` | Gaussian mixture, HINT flow | the predicted blind widths, in closed form |
| seismic | linearized Born, Parihaka | DDPM | under-coverage where the wavefield never reaches |
| groundwater | Darcy flow, 33 sensors | HINT flow | the same, with a single-best archive |

The theoretical results are machine-checked in Lean 4 (see [Formal verification](#formal-verification)).

## Installation

```bash
git clone https://github.com/luqigroup/resolvability
cd resolvability
pip install -e .
```

Python 3.10+ with PyTorch. A GPU speeds up prior training and sampling but is not needed to
reproduce any figure from the released caches.

Regenerating the seismic dataset additionally needs [Devito](https://www.devitoproject.org)
(`pip install -e ".[seismic]"`); nothing else in the repository requires an external PDE solver.

## Data and checkpoints

The datasets and trained checkpoints each example needs are hosted publicly and **downloaded
automatically on first use** — each script calls `resolvability.download.ensure` on what it
needs, which is a no-op once the file is on disk. There is nothing to fetch by hand, and no
figures to download: the figures are produced by the scripts below. The two-dimensional example
needs no data at all; its scripts regenerate everything they read.

To pre-fetch a tier instead of letting it stream in:

```python
from resolvability.download import ensure_tier
ensure_tier("checkpoints")     # the trained priors, three seeds per prior
ensure_tier("datasets")        # the full seismic training and evaluation archives
```

The full path→URL table is `resolvability/download.py`.

## Reproducing the paper's figures

The main figures, in paper order. Each renders in seconds on a CPU, from a cache that downloads on
first use or, for the two-dimensional example, from what `toy_estimation.py` writes in under a
minute.

```bash
python scripts/fig_hero.py               # Figure 1, the seismic teaser
python scripts/fig_toy_archives.py       # Figure 2, the four archives of the two-dimensional example
python scripts/toy_estimation.py         # the fits Figure 3 reads
python scripts/fig_toy_money.py          # Figure 3, fitted blind spread against the prediction
python scripts/fig_reliability_grid.py   # Figure 4, coverage on both operators
python scripts/fig_darcy_certify.py      # Figure 5, the certified blind band
python scripts/fig_gap.py                # Figure 6, the blind-minus-resolved gap
python scripts/fig_darcy_posterior.py    # Figure 7, the groundwater posterior
```

Output goes to `figures/`. The appendix figures and the scripted table are listed in
[FIGURES.md](FIGURES.md).

## Regenerating from scratch

Each example is a chain of producer scripts, with the figure script at the end. The
two-dimensional example has one stage (`scripts/toy_estimation.py`): it builds the archives, fits
the Gaussian mixture and the flows, and writes the cache and checkpoints. The seismic pipeline
(`scripts/seismic_*.py`) covers survey simulation and the migration archive, the probe basis and
illumination spectrum, prior training and sampling, the measurement-only amplitude calibration,
and diffusion posterior sampling on the Born operator. The groundwater pipeline
(`scripts/darcy_*.py`) covers the dataset and its MAP archive, flow training, and pCN posterior
sampling. Each stage's defaults are the configuration that produced the shipped caches.

### Devito threading

Devito's default kernel is serial C. To have the wave solves use the machine's cores, set the
environment before running anything that touches `resolvability.seismic.born`:

```bash
export DEVITO_LANGUAGE=openmp   # threaded kernels; the default, "C", is serial
export DEVITO_ARCH=gcc          # any OpenMP-capable compiler
export OMP_NUM_THREADS=8        # one thread per physical core
export OMP_PLACES=cores         # pin them
export OMP_PROC_BIND=close
```

On macOS, Apple `clang` rejects `-fopenmp`: install a real GCC (`brew install gcc`) and point Devito
at it with `export DEVITO_ARCH=custom CC=gcc-16 CXX=g++-16` (adjust the version suffix).

To confirm the threads are actually used, run one Born forward. `born.py` quiets Devito on import,
so turn its log level back up:

```python
import torch
from devito import configuration
from resolvability.seismic.born import parihaka_imager

configuration["log-level"] = "PERF"
imager = parihaka_imager()
imager.born_one(torch.zeros(1, 1, imager.nlat, imager.ndep), 0)
```

Each solve then reports its timing and the thread count it ran with:

```
Operator `Born` ran in 0.35 s
Performance[mode=advanced] arguments: {'nthreads': 8, 'nthreads_nonaffine': 8, 'pthreads': 0}
```

`nthreads: 1` while `OMP_NUM_THREADS` is higher means the kernel was built serial — `DEVITO_LANGUAGE`
or the compiler is wrong.

## Formal verification

`formal/` is a Lean 4 development that machine-checks the paper's theoretical results against
`mathlib`. Every theorem is kernel-verified with **no `sorry`**: `#print axioms` on each result
lists only Lean's three standard axioms (`propext`, `Classical.choice`, `Quot.sound`). Nothing
domain-specific is axiomatized — where an analytic step is not in `mathlib` it is either built
here or appears as an explicit hypothesis, never as an axiom.

```bash
cd formal
lake exe cache get     # prebuilt mathlib oleans for the pinned toolchain
lake build
```

`formal/README.md` maps each paper result to the file that proves it, and states precisely what
is and is not covered.

## Tests

```bash
pytest tests/ -v
```

Fast, CPU-only, no downloads: the discrete adjoints against finite differences, the blind/resolved
bases orthonormal and mutually orthogonal, the flow's forward and inverse mutually inverting
with the change-of-variables log-density, and the two-dimensional example's closed forms against
independent quadrature. The tests that read the two-dimensional fits skip until
`scripts/toy_estimation.py` has run.

## License

MIT — see [LICENSE](LICENSE).

## Contact

Ali Siahkoohi — <alisk@ucf.edu>
