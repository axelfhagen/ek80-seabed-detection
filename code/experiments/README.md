# experiments

Self-contained experiments that sit outside the main training and validation
pipelines. Run from the repository root.

## hake_finetune/

Fine-tuning of the Bastø Standard U-Net (`chunk_arch_sweep_2`) on the NOAA Hake
EK80 files, evaluated against the zero-shot model on both Hake and Bastø data,
including conformal coverage after fine-tuning.

| File | Purpose |
| --- | --- |
| `run_hake_finetune.py` | Main experiment. `--mode full\|train\|eval\|conformal`; `--init-mode checkpoint\|scratch` selects fine-tuning from `chunk_arch_sweep_2` or Hake-only training from scratch. Checkpoints go to `code/checkpoints/hake_finetune_chunk_arch_sweep_2_seed0/`, figures and CSV/JSON results to `Figures/hake_finetune_chunk_arch_sweep_2/`, cached arrays to `hake_finetune/cache/`. |
| `summarize_results.py` | Builds summary figures (including `constant_conformal_focus.png`) and a results note from the CSV/JSON outputs only; does not load checkpoints. |
| `compare_hake_training_modes.py` | Compares zero-shot, fine-tuned and Hake-only scratch results; writes `training_mode_comparison.png` to `Figures/hake_only_resnet34_seed0/`. |
| `run_hake_finetune.slurm` | SLURM wrapper for `run_hake_finetune.py`; extra arguments are forwarded. |
| `rerun_conformal_cpu.sh` | Reruns only the conformal stage on CPU, reusing existing outputs. |
| `summarize_results.sh` | Runs `summarize_results.py` in the cluster environment. |
