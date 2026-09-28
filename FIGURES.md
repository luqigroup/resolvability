# Reproducing the figures

Every figure but one renders in seconds on a CPU (`fig_seismic_training` generates fresh
unconditional samples and is fastest on a GPU). The seismic and groundwater caches download on
first use; the two-dimensional example needs no data, and its one cache is written by
`toy_estimation.py` in under a minute. A fresh clone needs nothing but `pip install -e .`:

```bash
python scripts/fig_hero.py
python scripts/fig_reliability_grid.py
python scripts/fig_darcy_posterior.py
python scripts/toy_estimation.py && python scripts/fig_toy_money.py
```

Output lands in `figures/`.

## Figure → script

The main figures, in paper order:

| paper figure | script | reads |
|---|---|---|
| 1, the seismic teaser | `fig_hero.py` | seismic DPS reconstructions and illumination |
| 2, the four archives of the two-dimensional example | `fig_toy_archives.py` | nothing (the archives are drawn in the script) |
| 3, fitted blind spread against the exact dequantized width | `fig_toy_money.py` | the two-dimensional fits (`toy_estimation.py`) |
| 4, reliability grid | `fig_reliability_grid.py` | seismic per-seed coordinates; groundwater pCN coverage, 3 seeds per prior |
| 5, the certified blind band, and its width | `fig_darcy_certify.py` | certified-band cache (coverage and width, per prior and seed) |
| 6, blind-minus-resolved gap, and where the spread goes | `fig_gap.py` | seismic per-seed coordinates and the legacy archive; groundwater pCN coverage, 3 seeds per prior |
| 7, groundwater single-observation posterior | `fig_darcy_posterior.py` | groundwater single-observation pCN record |

The appendix figures:

| paper figure | script | reads |
|---|---|---|
| 8, seismic prior samples vs training data | `fig_seismic_training.py` | seismic checkpoints (fresh unconditional samples) and the training archive |
| 9, seismic prior train/val loss | `fig_seismic_loss.py` | loss histories stored in the seismic checkpoints |
| 10, groundwater flow samples vs training fields | `fig_darcy_training.py` | groundwater seed-0 checkpoints and the dataset |
| 11, groundwater flow train/val NLL | `fig_darcy_loss.py` | loss histories stored in the groundwater checkpoints |
| 12, what the cut is applied to, and that the reading survives it | `fig_seismic_cutoff.py` | illumination spectrum (measured by forward apply); cutoff sweep |
| 13, what an added measurement corrects, and what it leaves frozen | `fig_darcy_augment.py` | groundwater boreholes, joint archive, flow checkpoints |

The tables:

| paper table | script | reads |
|---|---|---|
| 1, reference counts of the audit | none | analytic: exact χ² power at each `(r, b)` |
| 2, coverage table | `table_coverage.py` | the reliability grid's own caches, through the same loaders |

## Regenerating the caches

The stages below produce what the figures read. Each is independent of the others unless listed
as a dependency. Runtimes are on one modern GPU where marked, otherwise one CPU core.

### Two-dimensional example

| # | stage | script | time | GPU |
|---|---|---|---|---|
| 1 | the archives, the Gaussian-mixture fits, and the flows at seeds 0–2 | `toy_estimation.py` | ~40 s | no |

Nothing is downloaded: the archives are drawn from the closed-form laws in `resolvability/toy`,
and the cache and checkpoints land in `results/` and `data/checkpoints/toy_flows/`.

### Seismic (linearized Born)

| # | stage | script | time | GPU |
|---|---|---|---|---|
| 1 | survey simulation and the least-squares migration archive | `seismic_make_dataset.py` | days | no (Devito) |
| 2 | probe basis and illumination spectrum | `seismic_probe_basis.py` | ~2 h | no (Devito) |
| 3 | train one prior; run for 2 priors × 3 seeds | `seismic_prior_train.py` | ~4 h each | yes |
| 4 | unconditional prior samples | `seismic_prior_sample.py` | ~40 min | yes |
| 5 | measurement-only amplitude calibration κ | `seismic_data_kappa.py` | ~5 h | no (Devito) |
| 6 | incident-wavefield illumination | `seismic_illumination.py` | ~10 min | no (Devito) |
| 7 | per-seed blind/resolved coverage coordinates | `seismic_calibrate.py` | ~6 h | yes |
| 8 | diffusion posterior sampling on the Born operator | `seismic_dps_posterior.py` | ~4 h | yes |

### Groundwater (Darcy flow)

| # | stage | script | time | GPU |
|---|---|---|---|---|
| 1 | truths, observations, and the MAP archive | `darcy_make_dataset.py` | ~18 min | no |
| 2 | train both flows; run for seeds 0–2 | `darcy_flow_train.py --seed {0,1,2}` | ~1.5 min each | no |
| 3 | coverage over held-out observations | `darcy_pcn.py --mode coverage --prior {oracle,curated} --seed {0,1,2}` | ~75 min each | no |
| 4 | single-observation posterior fields | `darcy_pcn.py --mode single` | ~45 s per prior | no |
| 5 | boreholes and the null split | `darcy_build_borehole.py` | <1 min | no |
| 6 | the jointly curated archive | `darcy_joint_archive.py` | ~1 h | no |
| 7 | the certified blind band | `darcy_certify.py` | minutes | no |

Nothing here needs a GPU or an external PDE solver: the Darcy forward map is a conservative
finite-difference assembly with an exact adjoint, checked against finite differences in
`tests/test_groundwater.py`.

Each mode's defaults are the configuration that produced the shipped caches, so the commands
above reproduce them without extra flags.
