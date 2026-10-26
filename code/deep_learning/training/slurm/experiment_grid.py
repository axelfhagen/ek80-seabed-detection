"""Experiment grid definitions for Slurm array jobs.

Each grid is a list of dicts mapping CLI arg names (without --) to values.
The launcher reads this file, selects a grid by name, and maps
SLURM_ARRAY_TASK_ID to the corresponding dict.

Usage:
    python experiment_grid.py --grid loss_sweep --list
    python experiment_grid.py --grid loss_sweep --index 3
    python experiment_grid.py --grid loss_sweep --count
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import product


def _expand_grid(fixed: dict, variable: dict[str, list]) -> list[dict]:
    """Cartesian product of variable params merged with fixed."""
    keys = list(variable.keys())
    combos = list(product(*variable.values()))
    return [{**fixed, **dict(zip(keys, combo))} for combo in combos]


# ── Shared defaults ──────────────────────────────────────────────────────

_COMMON = {
    "epochs": 250,
    "batch-size": 8,
    "freeze": "encoder",
    "early-stopping-patience": 25,
    "scheduler-type": "plateau",
    "patch-width": 512,
    "stride": 512,
}


# ── Grid definitions ─────────────────────────────────────────────────────

GRIDS: dict[str, list[dict]] = {}

# Phase 0: Single baseline run
GRIDS["baseline"] = [
    {**_COMMON, "encoder": "resnet34", "loss-type": "dice_bce", "bce-weight": 0.5},
]

# Random-split ablation: identical to the baseline recipe, but the train/val
# partition is a random file-level shuffle of the SAME train+val survey pool
# (split=random_pool) instead of the survey-grouped split. File counts match
# the survey baseline; only the partition strategy changes. Three seeds expose
# the run-to-run variance of a random split against the single survey baseline.
# Compare each run's own validation metrics AND its held-out test-survey
# numbers (re-eval with --eval-split test) to gauge over-optimism.
GRIDS["baseline_random_split"] = [
    {**_COMMON, "encoder": "resnet34", "loss-type": "dice_bce", "bce-weight": 0.5,
     "split": "random_pool", "seed": s}
    for s in (42, 1, 7)
]

# Asymmetric input ablation: baseline config, model input resized to a
# tall 1024(H) x 512(W) instead of the 512x512 default — keeps the 512-ping
# width, doubles the depth/range height. target_size is (height, width).
# Batch size halved to absorb the doubled activation footprint. Standalone
# 6h launcher: run_ablation_target_h1024_w512.sh.
GRIDS["target_size_asym"] = [
    {**_COMMON, "encoder": "resnet34", "loss-type": "dice_bce", "bce-weight": 0.5,
     "target-size": "1024 512", "batch-size": 4},
]

# Phase 0.5: Loss ablation — baseline config, only loss changes
# dice_bce baseline is already covered by the baseline grid
GRIDS["loss_ablation"] = [
    {**_COMMON, "encoder": "resnet34", "loss-type": "focal"},
    {**_COMMON, "encoder": "resnet34", "loss-type": "focal_dice"},
    {**_COMMON, "encoder": "resnet34", "loss-type": "tversky"},
]

# Phase 1: Loss function sweep
_loss_raw = _expand_grid(
    fixed={**_COMMON, "encoder": "resnet34"},
    variable={
        "loss-type": ["dice_bce", "focal", "focal_dice", "tversky"],
        "bce-weight": [0.3, 0.5, 0.7],
    },
)
GRIDS["loss_sweep"] = [
    c for c in _loss_raw
    if c["loss-type"] == "dice_bce" or c["bce-weight"] == 0.5
]

# geo_seg: depth-regression + curvature (anti-spike) loss test run
GRIDS["geo_seg_test"] = _expand_grid(
    fixed={**_COMMON, "encoder": "resnet34", "loss-type": "geo_seg",
           "bce-weight": 0.5, "geo-depth-scale": 0.1},
    variable={
        "geo-reg-weight": [0.05, 0.1],
        "geo-smooth-weight": [0.05, 0.1],
    },
)

# Phase 2: Tversky alpha/beta sweep
GRIDS["tversky_sweep"] = _expand_grid(
    fixed={**_COMMON, "encoder": "resnet34", "loss-type": "tversky"},
    variable={
        "tversky-alpha": [0.3, 0.5, 0.7, 0.8],
        "tversky-beta":  [0.2, 0.3, 0.5],
    },
)

# Phase 3: Learning rate sweep (update loss-type after phase 1)
GRIDS["lr_sweep"] = _expand_grid(
    fixed={**_COMMON, "encoder": "resnet34", "loss-type": "dice_bce"},
    variable={
        "encoder-lr": [1e-5, 5e-5, 1e-4, 5e-4],
        "decoder-lr": [5e-4, 1e-3, 5e-3],
    },
)

# Phase 4: Encoder backbone comparison
GRIDS["encoder_sweep"] = _expand_grid(
    fixed={**_COMMON, "loss-type": "dice_bce"},
    variable={
        "encoder": ["resnet34", "resnet50", "efficientnet-b0",
                     "efficientnet-b3", "mobilenet_v2"],
    },
)

# Phase 5: Scheduler comparison
GRIDS["scheduler_sweep"] = [
    {**_COMMON, "encoder": "resnet34", "loss-type": "dice_bce",
     "scheduler-type": "plateau"},
    {**_COMMON, "encoder": "resnet34", "loss-type": "dice_bce",
     "scheduler-type": "cosine", "cosine-t0": 10, "cosine-t-mult": 2},
    {**_COMMON, "encoder": "resnet34", "loss-type": "dice_bce",
     "scheduler-type": "cosine", "cosine-t0": 25, "cosine-t-mult": 2},
    {**_COMMON, "encoder": "resnet34", "loss-type": "dice_bce",
     "scheduler-type": "cosine", "cosine-t0": 50, "cosine-t-mult": 1},
]

# Phase 6: Freeze strategy comparison
GRIDS["freeze_sweep"] = [
    {**_COMMON, "encoder": "resnet34", "loss-type": "dice_bce",
     "freeze": "encoder"},
    {**_COMMON, "encoder": "resnet34", "loss-type": "dice_bce",
     "freeze": "encoder", "unfreeze-after": 25, "encoder-lr": 1e-5},
    {**_COMMON, "encoder": "resnet34", "loss-type": "dice_bce",
     "freeze": "stages"},
    {**_COMMON, "encoder": "resnet34", "loss-type": "dice_bce",
     "freeze": "none", "encoder-lr": 5e-5, "decoder-lr": 5e-4},
]

# Phase 7: Dataset variant sweep (uses --cache-dir)
GRIDS["dataset_sweep"] = [
    {**_COMMON, "encoder": "resnet34", "loss-type": "dice_bce",
     "cache-dir": f"Datasets/patch_datasets/{name}"}
    for name in [
        "00_baseline",
        "01_wide_patches",
        "05_symmetric_norm",
        "07_percentile_norm",
        "08_dense_overlap",
        "11_line_mask",
        "12_single_freq_38k",
    ]
]
# single_freq needs in_channels=1
for c in GRIDS["dataset_sweep"]:
    if "single_freq" in c.get("cache-dir", ""):
        c["in-channels"] = 1


# Phase 8: Architecture comparison (from-scratch vs transfer learning)
GRIDS["arch_sweep"] = [
    # smp U-Net pretrained (baseline reference)
    {**_COMMON, "architecture": "smp_unet", "encoder": "resnet34",
     "encoder-weights": "imagenet", "freeze": "encoder"},
    # smp U-Net from scratch
    {**_COMMON, "architecture": "smp_unet", "encoder": "resnet34",
     "encoder-weights": "none", "freeze": "none",
     "encoder-lr": 1e-3, "decoder-lr": 1e-3},
    # smp U-Net from scratch (larger encoder)
    {**_COMMON, "architecture": "smp_unet", "encoder": "resnet50",
     "encoder-weights": "none", "freeze": "none",
     "encoder-lr": 1e-3, "decoder-lr": 1e-3},
    # Vanilla U-Net sizes
    {**_COMMON, "architecture": "vanilla_unet_small",
     "freeze": "none", "encoder-lr": 1e-3, "decoder-lr": 1e-3},
    {**_COMMON, "architecture": "vanilla_unet",
     "freeze": "none", "encoder-lr": 1e-3, "decoder-lr": 1e-3},
    {**_COMMON, "architecture": "vanilla_unet_large",
     "freeze": "none", "encoder-lr": 1e-3, "decoder-lr": 1e-3},
    # smp FPN from scratch
    {**_COMMON, "architecture": "smp_fpn", "encoder": "resnet34",
     "encoder-weights": "none", "freeze": "none",
     "encoder-lr": 1e-3, "decoder-lr": 1e-3},
    # smp DeepLabV3+ from scratch
    {**_COMMON, "architecture": "smp_deeplabv3plus", "encoder": "resnet34",
     "encoder-weights": "none", "freeze": "none",
     "encoder-lr": 1e-3, "decoder-lr": 1e-3},
    # Lightweight baseline
    {**_COMMON, "architecture": "lightweight_cnn",
     "freeze": "none", "encoder-lr": 1e-3, "decoder-lr": 1e-3},
]


# Phase 8b: Attention architectures on the existing 512/512 patches.
# All three share the smp_unet baseline recipe (imagenet encoder, frozen,
# default LRs) except SegFormer, whose transformer encoder is fine-tuned
# end-to-end with low LRs. Compare against the smp_unet baseline already in
# results.csv (eval_iou ~0.984, bottom_err ~0.6 m on 1728 files).
GRIDS["attention_sweep"] = [
    # scSE U-Net: spatial+channel attention gates in the decoder.
    {**_COMMON, "architecture": "smp_unet_scse", "encoder": "resnet34",
     "encoder-weights": "imagenet", "freeze": "encoder"},
    # MAnet: multi-scale attention decoder.
    {**_COMMON, "architecture": "smp_manet", "encoder": "resnet34",
     "encoder-weights": "imagenet", "freeze": "encoder", "batch-size": 4},
    # SegFormer: MiT-B1 transformer encoder + all-MLP decoder, fine-tuned.
    {**_COMMON, "architecture": "smp_segformer", "encoder": "mit_b1",
     "encoder-weights": "imagenet", "freeze": "none",
     "encoder-lr": 6e-5, "decoder-lr": 6e-4, "batch-size": 4},
]


# Phase 8c: Ultra-lightweight from-scratch models (far fewer params than the
# resnet34 baseline, ~24.4M). No pretraining at all — every config trains from
# scratch with freeze=none and equal encoder/decoder LRs. Param counts (3-ch):
#   vanilla_unet bf16 d4  ~1.94M   (matches vanilla_unet_small)
#   vanilla_unet bf8  d4   ~487k
#   vanilla_unet bf16 d3   ~483k
#   vanilla_unet bf8  d3   ~121k
#   lightweight_cnn        ~67k    (depthwise-separable)
# Compare against the smp_unet baseline (eval_iou ~0.984, bottom_err ~0.6 m).
_LW_COMMON = {**_COMMON, "loss-type": "dice_bce", "freeze": "none",
              "encoder-lr": 1e-3, "decoder-lr": 1e-3}
GRIDS["lightweight_sweep"] = [
    {**_LW_COMMON, "architecture": "vanilla_unet", "base-filters": 16, "model-depth": 4},
    {**_LW_COMMON, "architecture": "vanilla_unet", "base-filters": 8,  "model-depth": 4},
    {**_LW_COMMON, "architecture": "vanilla_unet", "base-filters": 16, "model-depth": 3},
    {**_LW_COMMON, "architecture": "vanilla_unet", "base-filters": 8,  "model-depth": 3},
    {**_LW_COMMON, "architecture": "lightweight_cnn"},
]


# Fine-tuning: bce_weight at the best freeze strategy (stages).
# Best known config is dice_bce + resnet34 + stages (eval_iou=0.9978); sweep
# bce_weight to test FP/FN balance around the default of 0.5.
GRIDS["bce_weight_sweep"] = [
    {**_COMMON, "encoder": "resnet34", "loss-type": "dice_bce",
     "freeze": "stages", "bce-weight": bw}
    for bw in [0.3, 0.5, 0.7]
]


# Re-run of six selected sweep winners on the FULL train+val survey pool
# (split=pool, 20% held out for early stopping) instead of the survey split's
# fixed val set. Patience tightened to 20. Frozen-encoder configs (scSE, the
# three bce_weight=stages runs) get staged unfreezing at epoch 20 with a low
# encoder LR so the pretrained encoder fine-tunes after the head settles; the
# two from-scratch configs (smp_unet/none, vanilla_unet_small) have nothing to
# unfreeze. Scheduler stays plateau. Sources:
#   1 attention_sweep[1], 2 arch_sweep[2], 3 arch_sweep[4],
#   4-6 bce_weight_sweep[1..3].
_POOL_RERUN = {"split": "pool", "val-fraction": 0.2,
               "early-stopping-patience": 20}
_POOL_UNFREEZE = {"unfreeze-after": 20, "encoder-lr": 1e-5}
GRIDS["pool_rerun"] = [
    # scSE attention U-Net (imagenet, frozen encoder → unfreeze at 20).
    {**_COMMON, **_POOL_RERUN, **_POOL_UNFREEZE,
     "architecture": "smp_unet_scse", "encoder": "resnet34",
     "encoder-weights": "imagenet", "freeze": "encoder",
     "loss-type": "dice_bce"},
    # smp U-Net from scratch.
    {**_COMMON, **_POOL_RERUN,
     "architecture": "smp_unet", "encoder": "resnet34",
     "encoder-weights": "none", "freeze": "none",
     "encoder-lr": 1e-3, "decoder-lr": 1e-3, "loss-type": "dice_bce"},
    # vanilla U-Net (small) from scratch.
    {**_COMMON, **_POOL_RERUN,
     "architecture": "vanilla_unet_small", "freeze": "none",
     "encoder-lr": 1e-3, "decoder-lr": 1e-3, "loss-type": "dice_bce"},
    # bce_weight at freeze=stages → unfreeze at 20, for bw in 0.3/0.5/0.7.
    *[{**_COMMON, **_POOL_RERUN, **_POOL_UNFREEZE,
       "encoder": "resnet34", "loss-type": "dice_bce",
       "freeze": "stages", "bce-weight": bw}
      for bw in (0.3, 0.5, 0.7)],
]


# Final full-pool re-run of the six selected sweep winners, trained on the
# ENTIRE train+val survey pool (split=pool) with the last 10% held out for
# early stopping (val-fraction=0.1, patience 25 from _COMMON). Unlike pool_rerun
# this keeps each architecture's ORIGINAL sweep recipe verbatim (no added staged
# unfreezing); only the data split changes. Submitted as standalone 12h jobs.
# Sources (1-based): arch_sweep[2], bce_weight_sweep[1..3], attention_sweep[1],
# arch_sweep[4] — order matches the checkpoint dirs the user listed.
_FINAL_POOL = {"split": "pool", "val-fraction": 0.1}
GRIDS["final_pool"] = [
    # 1. smp U-Net from scratch (arch_sweep[2]).
    {**_COMMON, **_FINAL_POOL, "architecture": "smp_unet", "encoder": "resnet34",
     "encoder-weights": "none", "freeze": "none",
     "encoder-lr": 1e-3, "decoder-lr": 1e-3, "loss-type": "dice_bce"},
    # 2-4. smp U-Net, dice_bce + freeze=stages, bce_weight 0.3/0.5/0.7
    # (bce_weight_sweep[1..3]).
    *[{**_COMMON, **_FINAL_POOL, "encoder": "resnet34", "loss-type": "dice_bce",
       "freeze": "stages", "bce-weight": bw} for bw in (0.3, 0.5, 0.7)],
    # 5. scSE attention U-Net, imagenet encoder, frozen (attention_sweep[1]).
    {**_COMMON, **_FINAL_POOL, "architecture": "smp_unet_scse", "encoder": "resnet34",
     "encoder-weights": "imagenet", "freeze": "encoder", "loss-type": "dice_bce"},
    # 6. vanilla U-Net small from scratch (arch_sweep[4]).
    {**_COMMON, **_FINAL_POOL, "architecture": "vanilla_unet_small", "freeze": "none",
     "encoder-lr": 1e-3, "decoder-lr": 1e-3, "loss-type": "dice_bce"},
]


# Quick sanity check: arch_sweep with 1 epoch (pipeline validation)
_TEST_COMMON = {**_COMMON, "epochs": 1}
del _TEST_COMMON["early-stopping-patience"]
GRIDS["test_arch"] = [
    {**cfg, "epochs": 1}
    for cfg in GRIDS["arch_sweep"]
]
for c in GRIDS["test_arch"]:
    c.pop("early-stopping-patience", None)


# Augmentation ablation: quick study on the 89-file jan_2024 subset.
# The launch script (launch_aug_ablation.sh) injects --zarr-paths for those
# 89 files; here we only vary the augmentation probabilities. All configs
# share identical patches (256/64). Early stopping (patience 20) ends each
# run once bottom error stops improving. Order is significant — the analysis
# script labels configs by array index.
_ABL_COMMON = {
    "encoder": "resnet34",
    "loss-type": "dice_bce",
    "bce-weight": 0.5,
    "freeze": "encoder",
    "scheduler-type": "plateau",
    "split": "random",
    "val-fraction": 0.2,
    "seed": 42,
    "patch-width": 256,
    "stride": 64,
    "batch-size": 8,
    "epochs": 120,
    "early-stopping-patience": 20,
    "monitor-metric": "loss",  # val loss: smooth, sensitive trigger; IoU/bottom_mae saturate/jitter
}

# Basic augmentations (flip / gauss-noise / brightness) on by default.
_BASIC_ON = {"p-hflip": 0.5, "p-gauss-noise": 0.3, "p-brightness": 0.3}
_BASIC_OFF = {"p-hflip": 0.0, "p-gauss-noise": 0.0, "p-brightness": 0.0}
# Advanced augmentations all off by default; each config turns on one.
_ADV_OFF = {"p-channel-dropout": 0.0, "p-ping-dropout-zero": 0.0,
            "p-ping-dropout-shuffle": 0.0, "p-ping-dropout-noise": 0.0}

GRIDS["aug_ablation"] = [
    # 1. baseline: basic augs only, no advanced
    {**_ABL_COMMON, **_BASIC_ON, **_ADV_OFF},
    # 2. + channel dropout
    {**_ABL_COMMON, **_BASIC_ON, **_ADV_OFF, "p-channel-dropout": 0.072},
    # 3. + ping dropout (zero)
    {**_ABL_COMMON, **_BASIC_ON, **_ADV_OFF, "p-ping-dropout-zero": 0.10},
    # 4. + ping dropout (shuffle)
    {**_ABL_COMMON, **_BASIC_ON, **_ADV_OFF, "p-ping-dropout-shuffle": 0.05},
    # 5. + ping dropout (noise)
    {**_ABL_COMMON, **_BASIC_ON, **_ADV_OFF, "p-ping-dropout-noise": 0.05},
    # 6. all augmentations (basic + every advanced)
    {**_ABL_COMMON, **_BASIC_ON, "p-channel-dropout": 0.072,
     "p-ping-dropout-zero": 0.10, "p-ping-dropout-shuffle": 0.05,
     "p-ping-dropout-noise": 0.05},
    # 7. no augmentation at all (basic off, advanced off)
    {**_ABL_COMMON, **_BASIC_OFF, **_ADV_OFF},
]


# ── CLI ──────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Experiment grid helper")
    p.add_argument("--grid", required=True, choices=list(GRIDS.keys()),
                   help="Grid name")
    p.add_argument("--list", action="store_true", help="Print all configs")
    p.add_argument("--index", type=int, default=None,
                   help="Print config at 1-based index as JSON")
    p.add_argument("--count", action="store_true",
                   help="Print number of configs")
    args = p.parse_args()

    grid = GRIDS[args.grid]

    if args.count:
        print(len(grid))
    elif args.list:
        for i, cfg in enumerate(grid, 1):
            print(f"[{i}/{len(grid)}] {cfg}")
    elif args.index is not None:
        if args.index < 1 or args.index > len(grid):
            print(f"ERROR: index {args.index} out of range [1, {len(grid)}]",
                  file=sys.stderr)
            sys.exit(1)
        print(json.dumps(grid[args.index - 1]))
    else:
        p.print_help()


if __name__ == "__main__":
    main()
