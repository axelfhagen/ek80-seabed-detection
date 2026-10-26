"""CLI argument parsing and PipelineConfig construction for training/eval.

Kept separate from train.py so the ~200 lines of argparse boilerplate don't
crowd the training logic. Both train.py and evaluate_checkpoint.py share these.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from deep_learning.training.config import PipelineConfig
from deep_learning.training.results_log import DEFAULT_LOG_PATH


def parse_args():
    p = argparse.ArgumentParser(
        description="Train U-Net for seabed segmentation")

    # Dataset preset (single authoritative knob for patch extraction)
    p.add_argument("--dataset", type=str, default=None,
                   help="Named dataset preset (e.g. '00_baseline', 'baseline', "
                        "or index '0'). Sets ALL patch-extraction parameters at "
                        "once so comparison runs share identical patches. "
                        "Authoritative: conflicting individual extraction flags "
                        "are ignored. See --list-datasets.")
    p.add_argument("--list-datasets", action="store_true",
                   help="Print the available dataset presets and exit.")

    # Data sources
    p.add_argument("--zarr-dir", type=Path, default=None,
                   help="Directory to glob for *.zarr stores")
    p.add_argument("--zarr-paths", nargs="+", type=Path, default=[],
                   help="Paths to individual zarr stores (appended to --zarr-dir matches)")
    p.add_argument("--raw-dirs", nargs="+", type=Path, default=[],
                   help="Directories containing .raw files to convert")
    p.add_argument("--zarr-cache-dir", type=Path, default=None,
                   help="Where to write converted zarr stores")

    # Frequencies
    p.add_argument("--freq1", type=float, default=38_000,
                   help="First frequency in Hz (default: 38000)")
    p.add_argument("--freq2", type=float, default=200_000,
                   help="Second frequency in Hz (default: 200000)")

    # Patches
    p.add_argument("--patch-width", type=int, default=256)
    p.add_argument("--stride", type=int, default=64)
    p.add_argument("--max-range-m", type=float, default=None)
    p.add_argument("--target-size", type=int, nargs=2, default=[512, 512],
                   help="Model input size (H W)")
    p.add_argument("--crop-mode", type=str, default="full",
                   choices=["full", "bottom"],
                   help="'full' keeps entire depth; 'bottom' crops around "
                        "detected bottom line")
    p.add_argument("--crop-padding", type=int, default=64,
                   help="Samples of context above/below bottom extent "
                        "(only used with --crop-mode bottom)")
    p.add_argument("--mask-mode", type=str, default="below",
                   choices=["below", "line"],
                   help="'below' fills the region below the bottom; "
                        "'line' marks only the bottom boundary")
    p.add_argument("--line-thickness", type=int, default=3,
                   help="Boundary line height in pixels (only used with "
                        "--mask-mode line)")

    # Model
    p.add_argument("--architecture", type=str, default="smp_unet",
                   help="Architecture registry key (e.g. smp_unet, vanilla_unet, "
                        "smp_fpn, lightweight_cnn)")
    p.add_argument("--encoder", type=str, default="resnet34",
                   help="Encoder backbone for smp architectures")
    p.add_argument("--encoder-weights", type=str, default="imagenet",
                   help="Pretrained weights for smp encoders ('imagenet' or 'none')")
    p.add_argument("--base-filters", type=int, default=32,
                   help="Base filter count for vanilla_unet")
    p.add_argument("--model-depth", type=int, default=4,
                   help="Encoder depth for vanilla_unet")
    p.add_argument("--in-channels", type=int, default=3,
                   help="Number of input channels (1 for single-freq, 3 for multi-freq)")
    p.add_argument("--freeze", type=str, default="encoder",
                   choices=["encoder", "stages", "none"])
    p.add_argument("--task", type=str, default="segmentation",
                   choices=["segmentation", "regression"],
                   help="'segmentation' (per-pixel bottom mask) or 'regression' "
                        "(soft-argmax bottom-line regression). Use with "
                        "--architecture unet_regression --loss-type regression "
                        "--monitor-metric bottom_mae.")

    # Training
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=0,
                   help="DataLoader worker processes (default: 0)")
    p.add_argument("--fast-cache", action="store_true",
                   help="Use the fast patch cache: patches pre-resized to "
                        "--target-size and stored as uncompressed, mmap-able "
                        ".npy pairs (built once from the existing .npz, no zarr "
                        "re-read). Much faster dataloading; the .npz cache and "
                        "non-fast jobs are left untouched.")
    p.add_argument("--unfreeze-after", type=int, default=None,
                   help="Unfreeze encoder after this many epochs")
    p.add_argument("--early-stopping-patience", type=int, default=None,
                   help="Stop if the monitored metric does not improve for "
                        "this many epochs")
    p.add_argument("--monitor-metric", type=str, default="loss",
                   choices=["iou", "dice", "loss", "bottom_mae"],
                   help="Metric used for checkpointing + early stopping. "
                        "'iou'/'dice' maximised; 'loss'/'bottom_mae' minimised. "
                        "'bottom_mae' is the soft per-ping bottom-row error, "
                        "which stays sensitive after IoU saturates")
    p.add_argument("--encoder-lr", type=float, default=1e-4)
    p.add_argument("--decoder-lr", type=float, default=1e-3)
    p.add_argument("--bce-weight", type=float, default=0.5,
                   help="Weight for BCE term in dice_bce loss (Dice = 1 - bce_weight)")
    p.add_argument("--loss-type", type=str, default="dice_bce",
                   choices=["dice_bce", "focal", "focal_dice", "tversky", "geo_seg",
                            "regression"],
                   help="Loss function type")
    p.add_argument("--focal-alpha", type=float, default=0.25,
                   help="Focal loss alpha (class balance weight)")
    p.add_argument("--focal-gamma", type=float, default=2.0,
                   help="Focal loss gamma (focusing parameter)")
    p.add_argument("--tversky-alpha", type=float, default=0.7,
                   help="Tversky loss alpha (false-negative weight)")
    p.add_argument("--tversky-beta", type=float, default=0.3,
                   help="Tversky loss beta (false-positive weight)")
    p.add_argument("--geo-reg-weight", type=float, default=0.1,
                   help="geo_seg: weight of the bottom-depth regression term")
    p.add_argument("--geo-smooth-weight", type=float, default=0.1,
                   help="geo_seg: weight of the curvature (anti-spike) term")
    p.add_argument("--geo-depth-scale", type=float, default=1.0,
                   help="geo_seg: metres per depth row (1.0 = work in row units)")
    p.add_argument("--geo-huber-delta", type=float, default=5.0,
                   help="geo_seg: Huber transition point for the regression term")
    p.add_argument("--split", type=str, default="survey",
                   choices=["random", "random_pool", "pool", "survey"],
                   help="'survey' (canonical train/val/calibration/test "
                        "by survey period), 'random_pool' (random file-level "
                        "split over the SAME train+val survey pool, matching "
                        "the survey split's file counts), 'pool' (random "
                        "file-level split over the train+val survey pool using "
                        "--val-fraction for the held-out validation), or "
                        "'random' (file-level %% over --zarr-dir/--zarr-paths)")
    p.add_argument("--val-fraction", type=float, default=0.2,
                   help="Validation fraction for --split random and --split pool")
    p.add_argument("--eval-split", type=str, default="auto",
                   choices=["auto", "val", "calibration", "test"],
                   help="Which data to evaluate on. 'auto' follows --split; "
                        "'val'/'calibration'/'test' force a canonical survey "
                        "group (use 'test' for held-out generalization).")
    p.add_argument("--seed", type=int, default=42)

    # Scheduler
    p.add_argument("--scheduler-type", type=str, default="plateau",
                   choices=["plateau", "cosine"],
                   help="LR scheduler: 'plateau' (ReduceLROnPlateau) or "
                        "'cosine' (CosineAnnealingWarmRestarts)")
    p.add_argument("--cosine-t0", type=int, default=10,
                   help="Cosine: epochs until first restart")
    p.add_argument("--cosine-t-mult", type=int, default=2,
                   help="Cosine: restart period multiplier")
    p.add_argument("--cosine-eta-min", type=float, default=1e-6,
                   help="Cosine: minimum LR")
    p.add_argument("--lr-patience", type=int, default=5,
                   help="Plateau scheduler: epochs without val-loss "
                        "improvement before reducing LR")
    p.add_argument("--lr-factor", type=float, default=0.5,
                   help="Plateau scheduler: LR multiplier on reduction")
    p.add_argument("--grad-clip-norm", type=float, default=1.0,
                   help="Max global gradient norm for clipping (safety net "
                        "against loss spikes / resume divergence). 0 disables.")

    # Augmentation
    p.add_argument("--p-hflip", type=float, default=0.5,
                   help="Probability of horizontal flip (reverses survey "
                        "direction — physically valid). 0.0 disables.")
    p.add_argument("--p-gauss-noise", type=float, default=0.3,
                   help="Probability of additive Gaussian noise. 0.0 disables.")
    p.add_argument("--p-brightness", type=float, default=0.3,
                   help="Probability of brightness/contrast shift. 0.0 disables.")
    p.add_argument("--p-channel-dropout", type=float, default=0.0,
                   help="Probability of coherent frequency dropout per patch "
                        "(zeros one of freq1/freq2 plus the diff channel). "
                        "0.0 disables.")
    p.add_argument("--p-ping-dropout-zero", type=float, default=0.10,
                   help="Probability of zeroing 1-5 vertical stripes "
                        "(1-20 px wide) — simulates dropped pings. "
                        "0.0 disables.")
    p.add_argument("--p-ping-dropout-shuffle", type=float, default=0.05,
                   help="Probability of shuffling pixels within 1-5 vertical "
                        "stripes (1-20 px wide). 0.0 disables.")
    p.add_argument("--p-ping-dropout-noise", type=float, default=0.05,
                   help="Probability of replacing 1-5 vertical stripes "
                        "(1-20 px wide) with low-amplitude Gaussian noise. "
                        "0.0 disables.")
    p.add_argument("--n-no-bottom-patches", type=int, default=0,
                   help="Number of synthetic no-bottom patches to inject "
                        "(all-zero mask, water-column noise). 0 disables.")

    # Output
    p.add_argument("--checkpoint-dir", type=Path,
                   default=Path("code/checkpoints"))
    p.add_argument("--cache-dir", type=Path, default=Path("code/patch_cache"),
                   help="Directory for cached patch .npz files")
    p.add_argument("--resume", type=Path, default=None,
                   help="Path to a checkpoint .pth to resume from. Restores "
                        "model + optimizer state and continues the epoch "
                        "counter. If it points at a directory, the single "
                        "*_best.pth inside is used.")
    p.add_argument("--reset-lr-on-resume", action="store_true",
                   help="On --resume, re-apply --encoder-lr/--decoder-lr to the "
                        "restored optimizer instead of keeping the checkpoint's "
                        "saved LRs. Use to continue training at a different LR.")
    p.add_argument("--eval-only", action="store_true",
                   help="Skip training entirely: load the checkpoint given by "
                        "--resume and run evaluation + diagnostics only. "
                        "Requires --resume.")
    p.add_argument("--no-eval", action="store_true",
                   help="Skip evaluation after training")
    p.add_argument("--no-diagnostics", action="store_true",
                   help="Skip diagnostic plot generation after evaluation")
    p.add_argument("--results-log", type=str, default=str(DEFAULT_LOG_PATH),
                   help="Path to shared CSV results log "
                        "(empty string disables)")

    args = p.parse_args()

    if args.list_datasets:
        from deep_learning.training.datasets import format_dataset_list
        print(format_dataset_list())
        raise SystemExit(0)

    # Record which flags the user set explicitly (value differs from default),
    # so build_config can warn when a --dataset preset overrides them.
    args._explicit = {
        a.dest for a in p._actions
        if a.dest != 'help' and getattr(args, a.dest, None) != p.get_default(a.dest)
    }
    return args


# Patch-extraction fields a --dataset preset is authoritative over.
_EXTRACTION_FLAGS = {
    'patch_width', 'stride', 'max_range_m',
    'crop_mode', 'crop_padding', 'mask_mode', 'line_thickness',
}


def build_config(args) -> PipelineConfig:
    spec = None
    if args.dataset is not None:
        from deep_learning.training.datasets import resolve_dataset
        spec = resolve_dataset(args.dataset)
        overridden = _EXTRACTION_FLAGS & getattr(args, '_explicit', set())
        if overridden:
            print(f"WARN: --dataset {spec.name} is authoritative for patch "
                  f"extraction; ignoring explicitly-set "
                  f"{', '.join('--' + f.replace('_', '-') for f in sorted(overridden))}")

    zarr_paths = sorted(args.zarr_dir.glob("*.zarr")) if args.zarr_dir else []
    zarr_paths += [Path(p) for p in args.zarr_paths]
    if (not zarr_paths and not args.raw_dirs
            and args.split not in ("survey", "random_pool", "pool")):
        raise SystemExit("ERROR: No zarr files found. Provide --zarr-dir, "
                         "--zarr-paths, or use --split survey/random_pool/pool.")

    explicit = getattr(args, '_explicit', set())
    if spec is not None:
        # Preset is authoritative over every patch-extraction parameter,
        # including ones with no CLI flag (vmin/vmax, norm, single_freq, ...).
        extraction = dict(
            patch_width=spec.patch_width,
            stride=spec.stride,
            max_range_m=spec.max_range_m,
            max_range_samples=spec.max_range_samples,
            vmin=spec.vmin,
            vmax=spec.vmax,
            norm_method=spec.norm_method,
            clip_percentile=spec.clip_percentile,
            single_freq=spec.single_freq,
            mask_mode=spec.mask_mode,
            line_thickness=spec.line_thickness,
            crop_mode=spec.crop_mode,
            crop_padding=spec.crop_padding,
        )
        # Single-freq presets need 1-channel input; honour an explicit override.
        in_channels = (args.in_channels if 'in_channels' in explicit
                       else spec.recommended_in_channels)
        # Each preset gets its own cache subdir unless --cache-dir was given,
        # so comparison runs reuse one clean cache per dataset.
        cache_dir = (args.cache_dir if 'cache_dir' in explicit
                     else args.cache_dir / spec.name)
    else:
        extraction = dict(
            patch_width=args.patch_width,
            stride=args.stride,
            max_range_m=args.max_range_m,
            mask_mode=args.mask_mode,
            line_thickness=args.line_thickness,
            crop_mode=args.crop_mode,
            crop_padding=args.crop_padding,
        )
        in_channels = args.in_channels
        cache_dir = args.cache_dir

    return PipelineConfig(
        zarr_paths=zarr_paths,
        raw_dirs=[Path(p) for p in args.raw_dirs],
        zarr_cache_dir=args.zarr_cache_dir,
        split_mode=args.split,
        freq1_hz=args.freq1,
        freq2_hz=args.freq2,
        target_size=tuple(args.target_size),
        **extraction,
        architecture=args.architecture,
        encoder_name=args.encoder,
        encoder_weights=args.encoder_weights,
        base_filters=args.base_filters,
        model_depth=args.model_depth,
        in_channels=in_channels,
        freeze_strategy=args.freeze,
        task=args.task,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        num_workers=args.num_workers,
        fast_cache=args.fast_cache,
        unfreeze_after=args.unfreeze_after,
        early_stopping_patience=args.early_stopping_patience,
        monitor_metric=args.monitor_metric,
        encoder_lr=args.encoder_lr,
        decoder_lr=args.decoder_lr,
        bce_weight=args.bce_weight,
        loss_type=args.loss_type,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
        tversky_alpha=args.tversky_alpha,
        tversky_beta=args.tversky_beta,
        geo_reg_weight=args.geo_reg_weight,
        geo_smooth_weight=args.geo_smooth_weight,
        geo_depth_scale=args.geo_depth_scale,
        geo_huber_delta=args.geo_huber_delta,
        val_fraction=args.val_fraction,
        eval_split=args.eval_split,
        seed=args.seed,
        scheduler_type=args.scheduler_type,
        lr_patience=args.lr_patience,
        lr_factor=args.lr_factor,
        grad_clip_norm=(args.grad_clip_norm
                        if args.grad_clip_norm and args.grad_clip_norm > 0
                        else None),
        cosine_t0=args.cosine_t0,
        cosine_t_mult=args.cosine_t_mult,
        cosine_eta_min=args.cosine_eta_min,
        checkpoint_dir=args.checkpoint_dir,
        cache_dir=cache_dir,
        resume=args.resume,
        reset_lr_on_resume=args.reset_lr_on_resume,
        eval_only=args.eval_only,
        p_hflip=args.p_hflip,
        p_gauss_noise=args.p_gauss_noise,
        p_brightness=args.p_brightness,
        p_channel_dropout=args.p_channel_dropout,
        p_ping_dropout_zero=args.p_ping_dropout_zero,
        p_ping_dropout_shuffle=args.p_ping_dropout_shuffle,
        p_ping_dropout_noise=args.p_ping_dropout_noise,
        n_no_bottom_patches=args.n_no_bottom_patches,
    )
