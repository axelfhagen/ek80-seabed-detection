# deep_learning.training

Training pipeline, configuration and post-training analysis. Scripts are run
from the repository root, e.g. `python code/deep_learning/training/train.py --help`.

## Library modules

- `config.py` - `PipelineConfig`, the shared configuration for paths,
  patch extraction, model and optimiser settings.
- `cli.py` - argument parser for `train.py` and `evaluate_checkpoint.py` and
  construction of `PipelineConfig` from it.
- `datasets.py` - named patch-extraction presets selected with `--dataset`.
- `model_setup.py` - model construction shared by training and evaluation.
- `checkpointing.py` - resume from a checkpoint (unfreeze state, optimiser,
  history).
- `evaluation.py` - validation-set evaluation (`run_evaluation`).
- `diagnostics.py` - post-training plots: training curves, prediction grids,
  error maps, calibration, ROC/PR, entropy maps, per-file summaries.
- `cam_utils.py` - Grad-CAM helpers used by `diagnostics.py`.
- `results_log.py` - append-only `results.csv` log, one row per run.
- `survey_groups.py` - assignment of zarr files to survey periods.

## Entry points

- `train.py` - end-to-end training (raw/zarr input, patch cache, training,
  evaluation, diagnostics, results log).
- `evaluate_checkpoint.py` - evaluation of a saved checkpoint without training.
- `evaluate_ensemble.py` - evaluation of a deep ensemble of checkpoints.
- `evaluate_new_data.py` - evaluation of selected models on out-of-distribution
  zarr files; figures go to `Figures/new_data_eval/`.
- `analyze_results.py` - ranking and grouping of runs in `results.csv`, with
  optional next-sweep suggestions.
- `show_runs.py` - tabular view of `results.csv`.
- `extract_missing_patches.py` - patch-cache extraction for the calibration and
  test splits.
- `convert_cache_to_fast.py` - builds the pre-resized `.npy` fast cache from
  existing `.npz` files.
- `clean_patch_cache.py` - lists (and with `--delete` removes) orphaned
  patch-cache files.
- `generate_input_channel_examples.py` - input-channel and label example
  figures (`Figures/input_channels/`, `Figures/label_examples/`).
- `generate_presentation_plots.py` - prediction grids and an ensemble
  uncertainty band for presentations.
- `smoketest_fast_cache.py`, `smoketest_geo_loss.py`, `smoketest_regression.py`
  - quick CPU checks of the fast cache, the geometry-aware loss and the
  regression path.

## Other directories

- `slurm/` - cluster job scripts and the experiment-grid tooling; see its README.
- `notebooks/` - exploratory notebooks (training v4, data dimensions,
  reliability diagram, clustering, outlier investigation). Paths are derived
  from the notebook location relative to the repository root.
