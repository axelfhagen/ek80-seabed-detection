# HPC Experiment System

Automated hyperparameter search for seabed segmentation using Slurm array jobs.

## How it works

The system has three layers:

1. **`experiment_grid.py`** defines named experiment grids — each grid is a list
   of hyperparameter configurations. A configuration is a dict of CLI args for
   `train.py`. Grids can be simple lists or generated with `_expand_grid()` for
   cartesian products.

2. **`launch_sweep.sh`** is a generic Slurm array job script. When submitted as
   `sbatch --array=1-N launch_sweep.sh <grid_name>`, each array task reads its
   config from `experiment_grid.py` using `$SLURM_ARRAY_TASK_ID` as the index,
   converts the dict to CLI args, and runs `train.py`.

3. **`submit_sweep.sh`** is a convenience wrapper that counts the grid size,
   shows a preview, asks for confirmation, and submits with a concurrency limit
   (default 10 simultaneous jobs via Slurm's `%10` array throttle).

After jobs complete, results are automatically logged to `../results.csv` by
`train.py`. Use `analyze_results.py` or the `/analyze-experiments` Claude
command to rank runs, group by factors, and get suggestions for the next sweep.

## Quick start

```bash
cd /cluster/work/axelfh/Master/code/deep_learning/training/slurm

# 1. See available grids
python experiment_grid.py --grid loss_sweep --list

# 2. Submit
./submit_sweep.sh loss_sweep

# 3. Monitor
squeue -u axelfh

# 4. Analyze (after jobs finish)
python ../analyze_results.py --suggest
```

## Available grids

| Grid | Size | Tests |
|------|------|-------|
| `baseline` | 1 | Single reference run with defaults |
| `loss_sweep` | 6 | dice_bce (3 weights) + focal + focal_dice + tversky |
| `tversky_sweep` | 12 | alpha/beta combinations for Tversky loss |
| `lr_sweep` | 12 | encoder_lr × decoder_lr combinations |
| `encoder_sweep` | 5 | resnet34, resnet50, efficientnet-b0/b3, mobilenet_v2 |
| `scheduler_sweep` | 4 | plateau vs cosine annealing variants |
| `freeze_sweep` | 4 | encoder, stages, none, unfreeze-after |
| `dataset_sweep` | 7 | patch size, normalization, overlap, mask mode |

## Adding a new grid

Edit `experiment_grid.py`:

```python
GRIDS["my_sweep"] = _expand_grid(
    fixed={**_COMMON, "encoder": "resnet34"},
    variable={
        "encoder-lr": [1e-5, 1e-4, 1e-3],
        "batch-size": [4, 8],
    },
)
```

Preview with `python experiment_grid.py --grid my_sweep --list`, then submit.

## Successive halving

For large sweeps, run a short round first:

```bash
# Temporarily lower epochs in the grid, submit all configs
./submit_sweep.sh my_big_sweep

# After jobs finish, keep top 4 and generate a longer round
python prune_sweep.py --grid my_big_sweep --top 4 --next-epochs 250
```

This outputs a new grid (`my_big_sweep_round2`) with just the winners.
Per-job early stopping (`--early-stopping-patience 25`) also naturally kills
bad configurations before they waste the full epoch budget.

## File overview

```
experiment_grid.py   — Grid definitions + CLI (--list, --index, --count)
launch_sweep.sh      — Slurm array job template (called by submit_sweep.sh)
submit_sweep.sh      — Convenience wrapper: preview → confirm → sbatch
prune_sweep.py       — Successive halving: rank results, pick top-K
../analyze_results.py — Results analysis, group-by, flags, suggestions
../results.csv       — Append-only log of all training runs
../show_runs.py      — Pretty-print results.csv with sorting/filtering
../CLAUDE.md         — Protocol for Claude-assisted experiment iteration
```

## Claude-in-the-loop workflow

1. Run `/analyze-experiments` — Claude reads results.csv, ranks runs, and suggests next configs
2. Review suggestions, adjust if needed
3. Claude formats the `sbatch` command
4. Submit, wait for jobs to complete
5. Repeat from step 1
