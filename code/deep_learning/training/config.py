"""Centralized pipeline configuration for end-to-end U-Net training.

All paths, hyperparameters, and model settings live here so that
notebooks and CLI scripts share the same defaults.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class PipelineConfig:
    """Configuration for the raw → zarr → patches → train → evaluate pipeline."""

    # ── Data paths ────────────────────────────────────────────────────
    zarr_paths: list[Path] = field(default_factory=list)
    raw_dirs: list[Path] = field(default_factory=list)
    zarr_cache_dir: Optional[Path] = None

    # ── Frequencies ───────────────────────────────────────────────────
    freq1_hz: float = 38_000
    freq2_hz: float = 200_000

    # ── Patch extraction ──────────────────────────────────────────────
    patch_width: int = 256
    stride: int = 64
    max_range_m: Optional[float] = None
    max_range_samples: Optional[int] = None
    vmin: float = -80.0
    vmax: float = -34.0
    require_bottom: bool = True
    norm_method: str = 'minmax'
    clip_percentile: Optional[tuple[float, float]] = None
    single_freq: bool = False
    mask_mode: str = 'below'
    line_thickness: int = 3
    crop_mode: str = 'full'
    crop_padding: int = 64

    # ── Train / val split ─────────────────────────────────────────────
    # 'survey' (canonical split), 'random_pool' (random file-level split over
    # the same train+val survey pool, matching the survey val count), 'pool'
    # (random file-level split over the train+val pool using val_fraction), or
    # 'random' (file-level over zarr_paths)
    split_mode: str = "survey"
    val_fraction: float = 0.2      # used when split_mode='random' or 'pool'
    seed: int = 42
    # Which group to evaluate on: 'auto' follows split_mode; 'val'/'calibration'/
    # 'test' force a canonical survey group (e.g. held-out test generalization).
    eval_split: str = "auto"

    # ── Model ─────────────────────────────────────────────────────────
    architecture: str = "smp_unet"     # registry key (see deep_learning.architectures)
    encoder_name: str = "resnet34"     # backbone for smp models
    encoder_weights: str = "imagenet"  # "imagenet" or "none" (from-scratch)
    base_filters: int = 32            # initial filter count for vanilla_unet
    model_depth: int = 4              # encoder depth for vanilla_unet
    freeze_strategy: str = "encoder"   # 'encoder', 'stages', or 'none'
    freeze_stages: int = 3
    in_channels: int = 3
    target_size: tuple[int, int] = (512, 512)
    # 'segmentation' (per-pixel bottom mask) or 'regression' (soft-argmax
    # bottom-line regression — see architectures.RegressionUNet / RegressionLoss).
    task: str = "segmentation"

    # ── Training ──────────────────────────────────────────────────────
    batch_size: int = 4
    num_epochs: int = 30
    num_workers: int = 0
    # Fast patch cache: store patches pre-resized to ``target_size`` as
    # uncompressed, mmap-able .npy pairs instead of full-height compressed
    # .npz. Eliminates per-sample deflate + downsample (the dominant dataloader
    # cost) and lets each __getitem__ read only its one patch via mmap. Off by
    # default so existing .npz caches and in-flight jobs are untouched; the npy
    # path self-describes per file, and a missing .npy is built once from the
    # existing .npz (no zarr re-read). See echo_dataset.ensure_cached.
    fast_cache: bool = False
    persistent_workers: bool = True   # only applied when num_workers > 0
    prefetch_factor: int = 4          # only applied when num_workers > 0
    encoder_lr: float = 1e-4
    decoder_lr: float = 1e-3
    weight_decay: float = 1e-4
    bce_weight: float = 0.5
    loss_type: str = "dice_bce"       # 'dice_bce', 'focal', 'focal_dice', 'tversky', or 'geo_seg'
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    tversky_alpha: float = 0.7
    tversky_beta: float = 0.3
    geo_reg_weight: float = 0.1       # geo_seg: depth-regression term weight
    geo_smooth_weight: float = 0.1    # geo_seg: curvature (anti-spike) term weight
    geo_depth_scale: float = 1.0      # geo_seg: metres per depth row (1.0 = row units)
    geo_huber_delta: float = 5.0      # geo_seg: Huber transition point
    lr_patience: int = 5
    lr_factor: float = 0.5
    grad_clip_norm: Optional[float] = 1.0   # max global grad-norm; None disables.
    # Safety net against loss spikes — notably the resume-divergence failure
    # where a converged checkpoint's stale (tiny) Adam second moments turn a
    # single larger gradient into a huge step. Near convergence grad norms are
    # << 1, so clipping at 1.0 is a no-op in healthy training and only bites
    # during a genuine explosion.
    unfreeze_after: Optional[int] = None   # epoch at which to unfreeze encoder
    early_stopping_patience: Optional[int] = None  # None disables early stopping
    monitor_metric: str = "loss"  # 'iou'/'dice' (maximise) or 'loss'/'bottom_mae' (minimise)

    # ── Scheduler ─────────────────────────────────────────────────────
    scheduler_type: str = "plateau"   # 'plateau' or 'cosine'
    cosine_t0: int = 10
    cosine_t_mult: int = 2
    cosine_eta_min: float = 1e-6

    # ── Data cleaning ─────────────────────────────────────────────────
    exclusion_list_path: Optional[Path] = None   # JSON with excluded UIDs
    excluded_files: list[str] = field(default_factory=list)
    min_quality_score: Optional[float] = None    # threshold on combined_score
    quality_csv_path: Optional[Path] = None      # combined_quality.csv

    # ── Augmentation ──────────────────────────────────────────────────
    p_hflip: float = 0.5               # horizontal flip (reverses survey direction)
    p_gauss_noise: float = 0.3         # additive Gaussian noise
    p_brightness: float = 0.3          # brightness/contrast shift
    p_channel_dropout: float = 0.072   # per-channel drop prob; ~80% keep all 3; 0.0 = off
    p_ping_dropout_zero: float = 0.10  # zeros 1-5 vertical stripes (1-20 px wide)
    p_ping_dropout_shuffle: float = 0.05  # same shape; shuffles pixels in stripe
    p_ping_dropout_noise: float = 0.05    # same shape; fills stripe with noise
    n_no_bottom_patches: int = 0     # synthetic all-water patches (mask=0) to inject

    # ── echopype raw→zarr settings ────────────────────────────────────
    waveform_mode: str = "BB"
    encode_mode: str = "complex"

    # ── Output ────────────────────────────────────────────────────────
    checkpoint_dir: Path = Path("checkpoints")
    cache_dir: Optional[Path] = None
    resume: Optional[Path] = None
    reset_lr_on_resume: bool = False
    eval_only: bool = False

    # ── Convenience ───────────────────────────────────────────────────

    @property
    def device(self) -> str:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @property
    def use_amp(self) -> bool:
        return self.device == "cuda"
