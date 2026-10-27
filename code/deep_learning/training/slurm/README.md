# training/slurm

Slurm job scripts for training, evaluation and data preparation on the cluster,
plus the experiment-grid tooling for hyperparameter sweeps.

Paths: each script defines `REPO_ROOT` (default `/cluster/work/axelfh/Master`)
and, where needed, `DATA_ROOT` (default `/cluster/work/axelfh`). Set them in the
environment before `sbatch` to run elsewhere. The `#SBATCH --output/--error`
lines still hold absolute log paths because Slurm does not expand variables in
directives; edit them or pass `--output/--error` to `sbatch`.

## Experiment grids

- `experiment_grid.py` - named grids of `train.py` argument dicts
  (`--list`, `--index`, `--count`). Grids are plain lists or cartesian products
  built with `_expand_grid()`.
- `launch_sweep.sh` - array-job template. Task `$SLURM_ARRAY_TASK_ID` reads its
  config from `experiment_grid.py` and runs `train.py`.
- `submit_sweep.sh <grid>` - previews a grid, asks for confirmation and submits
  it as an array job with a concurrency limit.
- `resume_sweep.sh`, `resume_lower_lr.sh`, `eval_checkpoints.sh` - resume or
  eval-only backfill of individual sweep tasks.
- `prune_sweep.py` - successive halving: ranks a finished sweep in
  `../results.csv` and emits a new grid with the top-K configs.

```bash
cd code/deep_learning/training/slurm
python experiment_grid.py --grid loss_sweep --list
./submit_sweep.sh loss_sweep
python ../analyze_results.py --suggest     # after the jobs finish
```

`train.py` appends one row per completed run to `../results.csv`
(`../results_log.py`); view it with `../show_runs.py` or `../analyze_results.py`.

## Script groups

- Data preparation: `convert_raw_to_zarr.sh`, `convert_single_raw.sh`,
  `submit_convert_all.sh`, `submit_convert_june.sh`, `generate_patches.sh`,
  `run_generate_datasets.sh`, `run_extract_missing.sh`,
  `extract_line_mask_patches.sh`, `convert_cache_to_fast.sh`,
  `prewarm_aug_ablation.sh` with `prewarm_aug_ablation_cache.py`.
- Single training runs: `run_train_8h.sh` (auto-resuming 8 h chunks),
  `run_train_regression_8h.sh`, `run_finetune*.sh`, `run_full_finetune.sh`,
  `run_cosine_gentle.sh`, `run_earlystop*.sh`, `run_resnet50.sh`,
  `run_loss_sweep.sh`, `run_dataset_*.sh` (one per dataset preset),
  `run_line_*.sh` (line-mask target), `run_jun2025*.sh`,
  `run_ablation_target_h1024_w512.sh`, `retrain_arch_6_*.sh`, `resume_*.sh`,
  `repro_24393977_dec2023.sh`, `run_overfit_demo.sh`.
- Augmentation ablation: `launch_aug_ablation.sh`, `resume_aug_ablation.sh`,
  `aug_ablation_file_list.txt`.
- Survey-level experiments: `run_per_survey.sh`, `submit_per_survey.sh`,
  `run_all_surveys.sh`, `submit_new_surveys.sh`, `launch_cross_survey.sh`,
  `submit_cross_survey.sh`, `run_cross_survey_chunk_arch_sweep_2.sh`.
- Ensembles: `ensemble_train.sh`, `ensemble_eval.sh`, `submit_ensemble.sh`,
  `ensemble_critical_eval.sh`, `ensemble_seeds/` (five ResNet-34 seeds,
  `run_ensemble_eval.sh`, `submit_all.sh`).
- Evaluation and figures: `eval_common_0*.sh`, `eval_test_generalization.sh`,
  `run_thesis_figure_inference.sh`, `run_gen_prediction_grids_oct2024.sh`,
  `gen_prediction_grids_regen_conformal.sh`, `gen_prediction_patches_regen.sh`,
  `diagnostics_conformal_gradcam.sh`, `run_regen_diagnostics.sh` and
  `submit_regen_diagnostics.sh` (manifests `diag_regen_manifest.txt`,
  `regen_bottomline_regression.txt`), `run_mc_dropout_calibrated_oct2024.sh`,
  `run_bottom_extraction_compare.sh`, `run_rolling_window_fidelity.sh`,
  `run_spatial_grid_comparison.sh`, `run_survey_eval_plots.sh`,
  `label_overview_all.sh`, `run_visualize_labels.sh`, `run_line_mask_eval.sh`,
  `run_test_raw_inference.sh`, `run_test_raw_echograms.sh`,
  `run_infer_test_raw_2.sh`.
- Checks and benchmarks: `run_smoketest_pipeline.sh`,
  `run_smoketest_line_mask.sh`, `run_smoketest_fast_cache.sh`,
  `run_test_diagnostics.sh`, `run_dl_bench.sh`.
