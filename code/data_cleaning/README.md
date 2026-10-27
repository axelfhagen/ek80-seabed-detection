# data_cleaning

Quality metrics, scoring and re-labelling for the cached training patches
(`patch`: normalised echogram channels, `mask`: binary below-bottom label).
Used by the data-cleaning notebooks (`raw_sensor_integration/notebooks/20_*`, `21_*`)
and by the quality-filter figure scripts in `code/data_visualization/`.

| Module | Contents |
|---|---|
| `__init__.py` | Re-exports the main metric, scoring and I/O functions. |
| `metrics.py` | `get_bottom_line`, `compute_patch_metrics` (signal and basic mask properties), `compute_mask_metrics` (label-failure metrics), `compute_all_metrics`, `compute_metrics_for_npz`. |
| `metrics_extended.py` | Additional metrics for surface noise, mask holes, bottom-line jumps and outliers, cross-frequency disagreement and low-information patches: `compute_extended_metrics`, `compute_full_metrics_for_npz`. |
| `scoring.py` | `rank_normalise`, patch / mask / combined quality scores and `classify_tiers` (reject / review / accept); metric column lists `PATCH_METRIC_COLS`, `MASK_METRIC_COLS`. |
| `io.py` | `load_patch_by_uid`, `load_quality_df`, `save_exclusion_list`, `load_exclusion_list`. |
| `file_analysis.py` | Per-source-file aggregation of patch scores: `aggregate_file_quality`, `identify_bad_files`, `plot_file_quality_bar`. |
| `relabel.py` | Patch-level mask re-labelling: `relabel_patch` (re-runs bottom detectors on de-normalised patches and keeps the best candidate), `smooth_erratic_bottom`, `fill_bottom_gaps`. |
| `visualization.py` | `plot_patch_with_mask`, `gallery`, `plot_metric_distributions`, `plot_score_distribution`. |
| `run_extended_metrics.py` | Entry point: recomputes all metrics for every patch listed in `Datasets/patch_quality_metrics.csv` and writes `Datasets/patch_quality_metrics_extended.csv`. |

```bash
python code/data_cleaning/run_extended_metrics.py --help
```

Tests: `tests/data_cleaning/`.
