# validation

Statistical validation and uncertainty quantification for the U-Net seabed
detector: conformal prediction bands for the bottom depth, MC-dropout and
ensemble uncertainty, calibration metrics, and dataset-shift analysis between
survey campaigns.

Scripts expect `code/` on `sys.path`; most insert it themselves. Run them from
the repository root (the SLURM scripts `cd` there). Default paths resolve
through `code/project_paths.py` (`REPO_ROOT`, overridable by environment
variable).

## Library modules

| Module | Contents |
| --- | --- |
| `conformal.py` | Split-conformal quantile, constant-width, profile-based, locally-weighted and pixel-level bands; coverage evaluation; transition-width sigma; plotting helpers. |
| `conformal_labelless.py` | Label-free conformal calibration from repeated ferry crossings: crossing detection, arc-length projection, pairwise scores, binning. |
| `uncertainty.py` | MC-dropout prediction, prediction entropy, bottom uncertainty band. |
| `ensemble.py` | Loading a deep ensemble and per-file ensemble prediction/evaluation. |
| `metrics.py` | Threshold sensitivity sweep and reliability (calibration) curve. |
| `dataset_shift/` | Sv distribution comparison (KS, JSD), per-survey noise floor, cross-survey generalisation matrix, within-survey baseline. |

`__init__.py` re-exports the main functions of these modules.

## Entry-point scripts

| Script | Output |
| --- | --- |
| `generate_conformal_plot.py` | Three-method conformal comparison on the survey split (calibration jan_2025, test feb/jun_2025). `Figures/conformal/`. Also provides `load_model` and the default `CHECKPOINT` (`exp_20260513_100101_arch_sweep_2`) imported by the other conformal scripts. |
| `generate_conformal_plot_v2.py` | Recalibrated comparison on the pooled exchangeable jan/feb/jun_2025 split with micro-averaged coverage. `Figures/conformal/v2_recalibrated/`. |
| `generate_conformal_scores.py` | Five conformity-score formulations (absolute/signed, constant/locally-weighted, CQR). `Figures/conformal/scores/`. |
| `generate_conformal_adaptive.py` | Sigma-threshold sweep and Mondrian conformal by backscatter and slope. `Figures/conformal/adaptive/`. |
| `generate_sigma_comparison.py` | Locally-weighted bands with global vs local transition-width sigma. `Figures/conformal/sigma_compare/`. |
| `conformal_exchangeability.py` | Exchangeability checks across surveys (two-sample score tests, conformal test martingale, stratified coverage); caches scores in `code/validation/cache/exchangeability/`. `Figures/conformal/exchangeability/`. |
| `generate_labelless_conformal_plot.py` | Label-free conformal band from repeated crossings. `Figures/conformal/labelless/`. |
| `generate_conformal_hake.py` | In-distribution calibrated bands and reliability applied to the deep-water NOAA Hake files. `Figures/conformal_hake_deep_water/`. |
| `generate_conformal_hake_recalibration.py` | In-situ split-conformal recalibration on the Hake files. |
| `evaluate_ensemble_critical.py` | Critical evaluation of a heterogeneous ensemble (gain over best member, diversity, size saturation, disagreement vs error). |
| `dataset_shift/distribution_comparison.py`, `noise_floor.py`, `cross_survey.py`, `within_survey_baseline.py` | Dataset-shift analyses; each has its own CLI. |

## slurm/

One SLURM wrapper per entry point (`run_conformal_base.sh`, `run_conformal_v2.sh`,
`run_conformal_scores.sh`, `run_conformal_adaptive.sh`, `run_sigma_comparison.sh`,
`run_conformal_exchangeability.sh`, `run_labelless_conformal.sh`,
`run_pred_crosspass_fullrange.sh`, `run_conformal_hake_recalibration.sh`), two
wrappers for ensemble scripts in `code/data_visualization/`
(`run_ensemble_uncertainty_bands.sh`, `run_ensemble_uncertainty_assessment.sh`),
and `submit_all_conformal.sh`, which submits the full conformal batch as
independent jobs. Extra arguments are forwarded to the Python script.

## notebooks/

`06_validation_uncertainty.ipynb` is an exploratory notebook (MC-dropout,
entropy, threshold sensitivity, calibration, uncertainty band) on one file;
`run_uncertainty_nb.sh` executes it with `nbconvert` on the cluster.
