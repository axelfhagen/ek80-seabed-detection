# thesis_metrics_regen

Scripts that recompute the numbers and figures cited in the thesis with the
Bastø Standard U-Net (`chunk_arch_sweep_2`). Letters in the docstrings (A-G)
refer to the groups of thesis claims each script covers. Machine-readable
summaries are written to `thesis_metrics_regen/outputs/`; regenerated figures
go to `Latex/Figures/regen_2026/` unless stated otherwise. Run from the
repository root with `code/` on `PYTHONPATH` (the SLURM scripts set this).

| Script | Recomputes |
| --- | --- |
| `recompute_headline.py` | (A, B) Oct-2024 headline metrics from the saved `eval_metrics.json` and per-ping errors. No inference. Writes `headline_summary.json`. |
| `diag_bottom_error.py` | (B) Per-ping signed bottom-depth errors on the held-out Oct-2024 files and the bottom-error distribution figure. |
| `eval_aug_p95.py` | (C) Per-file P95 depth error of one file under the augmentation-ablation checkpoints; appends to `headline_summary.json`. |
| `diag_calibration.py` | (D) Probability-bin fractions, ECE, transition width; reliability and probability-distribution figures. |
| `recompute_conformal.py` | (E) Supervised split-conformal q_hat and coverage from the cached scores in `code/validation/cache/exchangeability/`. |
| `diag_mc_dropout.py` | (F) MC-dropout rate sweep, boundary std and its correlation with prediction entropy. |
| `eval_hake.py` | (G) NOAA Hake transfer metrics (IoU, depth deviation, open-water fraction), reusing `code/data_visualization/eval_hake_test_raw.py`. |

## slurm/

| Script | Purpose |
| --- | --- |
| `run_regen_gpu.sh` | Runs the full chain: exchangeability cache, then the scripts above, then `code/validation/generate_conformal_hake.py`. |
| `run_final_figure_recompute.sh` | Array job (tasks 0-2) regenerating the final thesis figures (bottom-error distribution, prediction grids, channel importance, Grad-CAM, patch galleries, Hake recalibration) with `chunk_arch_sweep_2`. |
| `run_prediction_grids_arch2.sh`, `run_prediction_grids_qhat_arch2.sh`, `run_prediction_conformal_singles_arch2.sh` | Prediction-grid figures, optionally with a fixed locally-weighted conformal q_hat. |
| `run_patch_prob_arch2.sh`, `run_patch_conformal_arch2.sh` | Hard-patch probability gallery and patch conformal-failure gallery. |
