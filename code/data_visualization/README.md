# data_visualization

Figure and analysis scripts for the thesis. Each script is a standalone entry point,
run from the repository root:

```bash
python code/data_visualization/<script>.py [--help]
```

Scripts add `code/` to `sys.path` themselves and write their output into
`Figures/<topic>/` relative to the repository root. `Figures/` is not part of this
repository. Input data (zarr stores, patch caches, checkpoints) is read from
`Datasets/` and `DATA_ROOT` (see `code/project_paths.py`).

## Naming conventions

| Prefix | Purpose |
|---|---|
| `plot_*.py` | One figure, or a small set of figures, per topic. |
| `visualize_*.py` | Batch panels, typically one image per input file. |
| `analyze_*.py` | An analysis plus its summary figure(s), often with a CSV or JSON of the numbers. |
| `gen_*.py` | Multi-panel grids and galleries (prediction grids, Grad-CAM reports, comparisons). |
| `compare_*.py`, `compute_*.py`, `eval_*.py`, `crosspass*.py` | Targeted comparisons, summary statistics and evaluations. |
| `stitch_*.py` | Data composition (e.g. stitching consecutive files) plus a figure. |
| `regenerate_*.py` | Rebuilds a derived artefact from existing outputs. |
| `smoketest_*.py` | Quick experiment on a few files to decide whether an option is worth pursuing. |

Versioned families (`foo.py`, `foo_v2.py`, ...) are kept where an older version still
produces a figure or is imported by a newer one.

## Shared style: `thesis_style.py`

Common matplotlib settings for every thesis figure. Fonts and line widths are oversized
because LaTeX downscales the figures to half or full page width.

- `apply_thesis_style()`: sets the rcParams; call once near the top of a script.
- `FIGSIZE_FULL`, `FIGSIZE_STD`, `FIGSIZE_HALF`: figure sizes in inches.
- `LABEL_SIZE`, `TICK_SIZE`, `LEGEND_SIZE`, `TITLE_SIZE`, `SUPTITLE_SIZE`, `LINEWIDTH`.
- `REFERENCE_BOTTOM_COLOR` (cyan, labelled bottom) and `UNET_PREDICTION_COLOR` (red).
- `style_colorbar(cbar, label=None)`: applies the font sizes to a colorbar.
- `keep_spines(ax)`, `savefig(fig, path, **kwargs)`.

## plot_, visualize_ and other figure scripts

Background and method illustrations:
`plot_convolution_schematic.py`, `plot_freeze_strategy_schematic.py`,
`plot_model_fit_regimes.py`, `plot_pulse_types.py`, `plot_example_echogram.py`,
`plot_example_echograms_intro.py`, `plot_survey_echogram_panel.py`,
`plot_survey_echogram_panel_v2.py`, `plot_freq_panel_38_200.py`, `plot_hake_channels.py`,
`plot_augmentation_examples.py`, `plot_synthetic_insertions_gaussian.py`.

Data, routes and GPS:
`plot_basto_route_v6.py`, `plot_data_split_timeline.py`, `plot_data_explorer.py`,
`plot_gps_overview.py`, `plot_gps_overview_test_raw_2.py`, `plot_raw_file_overview.py`,
`plot_ping_gps_quality.py` (imports helpers from `plot_raw_file_overview.py`),
`plot_crossings_arclength.py`, `plot_crossings_stack.py` (uses `stitch_panorama.py`),
`stitch_panorama.py`, `plot_june_echograms.py`, `plot_transfernow_echograms.py`.

Bottom labels (classical detectors):
`plot_bottom_label_comparison.py`, `plot_bottom_label_config_sweep.py`,
`plot_bottom_label_config_sweep_v3.py` (imports from the v1 script),
`plot_multifreq_bottom_ensemble.py`, `plot_temporal_smoothing.py`,
`plot_single_ping_detection.py`, `smoketest_gradient_ensemble.py`,
`visualize_labels.py`, `visualize_label_panels.py`, `visualize_outlier_bottom_detection.py`,
`visualize_split_echograms.py`.

Patch quality filtering (reads `Datasets/patch_quality_metrics_extended.csv` from
`code/data_cleaning/run_extended_metrics.py`):
`plot_quality_metric_distributions.py`, `plot_quality_filter_groups.py`,
`plot_quality_filter_groups_clean.py`, `plot_quality_filter_groups_extra.py`,
`regenerate_quality_filter_readme.py`.

Training and results:
`plot_training_curves.py`, `plot_overfitting_curves.py`,
`plot_convergence_loss_comparison.py`, `plot_per_file_performance.py`,
`plot_signed_bottom_error.py`, `plot_survey_evaluations.py`, `plot_raw_inference.py`,
`plot_failure_modes_combined.py`, `plot_discussion_figures.py`.

Uncertainty:
`plot_ensemble_uncertainty_bands.py`, `plot_probability_uncertainty_bands.py`,
`plot_requested_ensemble_maps.py`, `plot_ensemble_report_batch.py` (imports from
`plot_requested_ensemble_maps.py`).

## analyze_, gen_, compare_, compute_, eval_ scripts

Analyses behind the Results and Discussion chapters. Each script's module docstring
states its inputs and output directory.

Model evaluation:
`analyze_arch_sweep_eval.py`, `analyze_unet_variant_stats.py`,
`analyze_unet_vs_detector.py`, `analyze_detector_stats.py`, `analyze_threshold_sweep.py`,
`analyze_cross_survey_generalization.py`, `analyze_cross_survey_generalization_v2.py`,
`analyze_aug_ablation.py`, `analyze_aug_ablation_v2.py` (augmentation ablation), `gen_results_eval_figures.py`,
`gen_spatial_grid_comparison.py`, `eval_hake_test_raw.py` (NOAA Hake test files; imported
by other scripts).

Uncertainty and conformal prediction:
`analyze_ensemble_uncertainty.py`, `analyze_mc_dropout_uncertainty.py`, `analyze_mc_dropout_uncertainty_zarr.py`,
`analyze_rolling_window_uncertainty.py`, `analyze_conformal_exchangeability.py`,
`analyze_labelless_adaptive.py`, `analyze_prediction_probabilities.py`,
`gen_patch_conformal_failures.py`, `gen_patch_wide_band_diagnostics.py`,
`gen_conformal_wide_band_diagnostics.py`, `gen_calibration_stacked.py`.

Inference behaviour (patch edges, window width and stride, bottom-line extraction):
`analyze_edge_degradation.py`, `analyze_edge_mirror_padding.py`,
`analyze_rolling_window_fidelity.py`, `gen_patch_width_comparison.py`,
`gen_stride_comparison.py`, `compare_bottom_extraction.py`,
`analyze_bottom_line_spikes.py`, `analyze_bottom_smoothing_eval.py`.

Interpretability:
`analyze_channel_importance.py`, `analyze_channel_importance_v2.py`, `gen_gradcam_unet_v3.py` (library imported by v4
and v5_report), `gen_gradcam_unet_v4.py`, `gen_gradcam_unet_v5_report.py`,
`analyze_synthetic_insertions.py`, `analyze_spatial_correlation.py`.

Repeated ferry crossings:
`crosspass_data.py` (shared loader), `analyze_crosspass_consistency.py`,
`analyze_failure_mining.py`, `analyze_label_consistency_route.py`.

Labels and data:
`analyze_label_noise_ceiling.py`, `gen_label_error_gallery.py`, `gen_empty_label_files.py`,
`gen_ensemble_overlay_grids.py`, `compute_class_balance_per_survey.py`,
`analyze_dock_distribution.py`, `analyze_geographic_outliers.py`, `analyze_time_of_day.py`,
`analyze_timestamp_coverage.py`, `analyze_crossings_per_day.py`,
`analyze_joint_sv_distribution.py`, `analyze_freq_divergence.py`.

Bastø dock and shallow-water failure cases:
`analyze_basto_failure.py`, `analyze_shallow_C2_failure.py`,
`analyze_shallow_C2_bottomline_strategies.py`, `compare_basto_calibration.py`.

Prediction galleries:
`gen_prediction_grids_thesis.py`, `gen_prediction_grids_regen.py`,
`gen_prediction_patches_regen.py`, `gen_patch_overlay_only.py`, `gen_patch_overlay_worst.py`,
`gen_patch_predictions.py`, `gen_patch_prob_eval.py`, `gen_hard_cases_gallery.py`,
`gen_openwater_zero_iou.py`, `gen_echogram_grids.py`, `gen_rgb_image_predictions.py`.

## archive/

Superseded earlier versions of scripts in this directory (`plot_basto_route` v1-v5,
`gen_gradcam_unet` v1-v2, `analyze_channel_importance_v1`). Kept for reference only;
nothing imports them.

## slurm/

SLURM batch scripts for the figure jobs that need a GPU or more memory than a login
node. Each script sets `REPO_ROOT` (default `/cluster/work/axelfh/Master`, overridable
through the environment), changes into it and runs one script from this directory.
The `#SBATCH --output/--error` paths are fixed to the cluster layout.
