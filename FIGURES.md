# Reproducing the figures

Every figure but one renders from a cache in seconds on a CPU (`fig_seismic_training` draws
fresh unconditional samples and is fastest on a GPU). The inputs download on first use, so a
fresh clone needs nothing but `pip install -e .`:

```bash
python scripts/fig_hero.py
python scripts/fig_reliability_grid.py
python scripts/fig_darcy_posterior.py
```

Output lands in `figures/`.

## Figure → script

The main figures, in paper order:

| paper figure | script | reads |
|---|---|---|
| the seismic teaser | `fig_hero.py` | seismic DPS reconstructions, prior samples, illumination, κ, evaluation window |
| reliability grid | `fig_reliability_grid.py` | seismic per-seed coordinates; groundwater pCN coverage, 3 seeds per prior |
| blind-minus-resolved gap, and where the spread goes | `fig_gap.py` | seismic per-seed coordinates and the legacy archive; groundwater pCN coverage, 3 seeds per prior |
| groundwater single-observation posterior | `fig_darcy_posterior.py` | groundwater single-observation pCN record |
| the certified blind band, and its width | `fig_darcy_certify.py` | certified-band cache (coverage and width, per prior and seed) |

The appendix figures:

| paper figure | script | reads |
|---|---|---|
| the deep end of one survey, on the operator's own illumination | `fig_seismic_zoom.py` | seismic DPS reconstructions; per-pixel `diag(AᵀA)` (`seismic_m0_stability.py --phase hutch`) |
| what the cut is applied to, and that the reading survives it | `fig_seismic_cutoff.py` | illumination spectrum (measured by forward apply); cutoff sweep |
| what an added measurement corrects, and what it leaves frozen | `fig_darcy_augment.py` | groundwater boreholes, joint archive, flow checkpoints |
| seismic prior samples vs training data | `fig_seismic_training.py` | seismic checkpoints (fresh unconditional samples) and the training archive |
| seismic prior train/val loss | `fig_seismic_loss.py` | loss histories stored in the seismic checkpoints |
| groundwater flow samples vs training fields | `fig_darcy_training.py` | groundwater seed-0 checkpoints and the dataset |
| groundwater flow train/val NLL | `fig_darcy_loss.py` | loss histories stored in the groundwater checkpoints |

The scripted tables:

| paper table | script | reads |
|---|---|---|
| coverage table | `table_coverage.py` | the reliability grid's own caches, through the same loaders |
| background-stability table (`tab:m0`) | `seismic_m0_stability.py --phase stability` | the background model and acquisition (Devito) |

The audit-count table is analytic (exact χ² power at each `(r, b)`) and has no script.

## Regenerating the caches

The stages below produce what the figures read. Each is independent of the others unless listed
as a dependency. Runtimes are on one modern GPU where marked, otherwise one CPU core.

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
| 9 | background-model stability of the blind set, and `tab:m0` | `seismic_m0_stability.py --phase stability` | ~2 min | no (Devito) |
| 10 | per-pixel illumination `diag(AᵀA)` for the zoom figure | `seismic_m0_stability.py --phase hutch` | hours | no (Devito) |

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
