"""Named dataset presets — the single source of truth for patch extraction.

A "dataset" is fully defined by its patch-extraction parameters. Selecting a
preset by name (``--dataset NAME``) applies all of them at once, so that
model-comparison runs share byte-identical patches without having to juggle a
dozen individual CLI flags.

``target_size`` and ``in_channels`` are training-time knobs, not part of the
patch identity — the ``recommended_*`` fields are advisory only. In particular
``target_size`` is deliberately *not* forced by a preset so every run can keep
the same 512x512 model input for comparison.

Both ``train.py`` (via ``cli.resolve_dataset``) and ``generate_all_datasets.py``
import from here, so the params can never drift apart.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    description: str
    patch_width: int = 256
    stride: int = 64
    max_range_m: Optional[float] = None
    max_range_samples: Optional[int] = None
    vmin: float = -80.0
    vmax: float = -34.0
    norm_method: str = 'minmax'
    clip_percentile: Optional[tuple[float, float]] = None
    single_freq: bool = False
    mask_mode: str = 'below'
    line_thickness: int = 3
    crop_mode: str = 'full'
    crop_padding: int = 64
    # Training-time recommendations (advisory; not forced by --dataset)
    recommended_target_size: tuple[int, int] = (512, 512)
    recommended_in_channels: int = 3


DATASET_SPECS: list[DatasetSpec] = [
    # ── 0. Baseline (matches existing training runs) ──────────────────
    DatasetSpec(
        name="00_baseline",
        description=(
            "Current default: 256-wide patches, stride 64, minmax [0,1] "
            "normalisation, vmin=-80/vmax=-34, 3-channel (38k, 200k, diff). "
            "Resized to 512x512 at training time."
        ),
    ),

    # ── 1. Wide patches ───────────────────────────────────────────────
    DatasetSpec(
        name="01_wide_patches",
        description=(
            "512-wide patches (2x baseline). Each patch covers more pings, "
            "giving the model more spatial context of bottom morphology. "
            "Fewer patches per file but each contains more information."
        ),
        patch_width=512,
        stride=128,
    ),

    # ── 2. Narrow patches, large target ───────────────────────────────
    DatasetSpec(
        name="02_narrow_patches",
        description=(
            "128-wide patches with recommended 1024x1024 target. Each ping "
            "column gets 8x the horizontal pixels compared to baseline — "
            "high detail per ping, less along-track context."
        ),
        patch_width=128,
        stride=64,
        recommended_target_size=(1024, 1024),
    ),

    # ── 3. Aspect-ratio-preserving resize ─────────────────────────────
    DatasetSpec(
        name="03_preserve_aspect",
        description=(
            "Same extraction as baseline but with recommended target size "
            "1024x256 to preserve the natural aspect ratio of the echogram "
            "(~10:1 height:width). No horizontal stretching."
        ),
        recommended_target_size=(1024, 256),
    ),

    # ── 4. Height-only resample ───────────────────────────────────────
    DatasetSpec(
        name="04_height_only_resample",
        description=(
            "Same extraction as baseline but recommended target 512x256 — "
            "no horizontal distortion, moderate vertical compression "
            "(~5.4x → ~2.7x compared to raw). Tests whether preserving "
            "horizontal resolution helps."
        ),
        recommended_target_size=(512, 256),
    ),

    # ── 5. Symmetric normalisation ────────────────────────────────────
    DatasetSpec(
        name="05_symmetric_norm",
        description=(
            "All channels normalised to [-1, 1] instead of [0, 1]. "
            "Pretrained ImageNet encoders were trained on mean-subtracted "
            "data, so centered input may transfer better."
        ),
        norm_method='symmetric',
    ),

    # ── 6. Wider Sv range ─────────────────────────────────────────────
    DatasetSpec(
        name="06_wide_sv_range",
        description=(
            "Normalisation bounds expanded to vmin=-90, vmax=-25 (from "
            "-80/-34). Captures more dynamic range — keeps signal from "
            "deep water (below -80 dB) and strong scatterers (above -34 dB) "
            "that baseline clips away."
        ),
        vmin=-90.0,
        vmax=-25.0,
    ),

    # ── 7. Percentile normalisation ───────────────────────────────────
    DatasetSpec(
        name="07_percentile_norm",
        description=(
            "Per-echogram adaptive normalisation using 2nd/98th percentile "
            "instead of fixed vmin/vmax. Each file's bounds adjust to its "
            "own SNR. More robust to varying conditions but introduces "
            "inconsistency across patches from different files."
        ),
        clip_percentile=(2, 98),
    ),

    # ── 8. Dense overlap ──────────────────────────────────────────────
    DatasetSpec(
        name="08_dense_overlap",
        description=(
            "Stride reduced to 32 (from 64). Roughly 2x more patches per "
            "file. More training data from the same zarr stores, at the "
            "cost of higher patch correlation."
        ),
        stride=32,
    ),

    # ── 9. No overlap ────────────────────────────────────────────────
    DatasetSpec(
        name="09_no_overlap",
        description=(
            "Stride equals patch width (256). Each ping appears in exactly "
            "one patch — no data leakage between adjacent patches. Smaller "
            "dataset but cleaner independence between samples."
        ),
        stride=256,
    ),

    # ── 10. Shallow crop ──────────────────────────────────────────────
    DatasetSpec(
        name="10_shallow_crop",
        description=(
            "Depth cropped to 1500 samples (~55% of full range). Removes "
            "the deep water column where there is rarely any bottom signal. "
            "Forces the model to focus on the bottom-detection region."
        ),
        max_range_samples=1500,
    ),

    # ── 11. Line mask ────────────────────────────────────────────────
    DatasetSpec(
        name="11_line_mask",
        description=(
            "Mask mode changed to 'line' (thin boundary at the bottom) "
            "instead of 'below' (fill everything below the bottom). "
            "Frames the task as boundary detection rather than region "
            "segmentation — may produce sharper depth estimates."
        ),
        mask_mode='line',
        line_thickness=3,
    ),

    # ── 12. Single frequency ─────────────────────────────────────────
    DatasetSpec(
        name="12_single_freq_38k",
        description=(
            "Only the 38 kHz channel (1-channel input). Tests whether "
            "the 200 kHz and frequency-difference channels actually help "
            "or just add noise. Requires in_channels=1 at training time."
        ),
        single_freq=True,
        recommended_in_channels=1,
    ),

    # ── 13. Bottom-crop, stride 128 ─────────────────────────────────
    DatasetSpec(
        name="13_bottom_crop_s128",
        description=(
            "Dynamically crops each patch around the detected bottom line. "
            "Finds the shallowest and deepest bottom pixels, adds 64 samples "
            "of padding, and rounds height to the nearest 256. Stride 128 "
            "gives moderate overlap."
        ),
        patch_width=256,
        stride=128,
        crop_mode='bottom',
        crop_padding=64,
    ),

    # ── 14. Bottom-crop, stride 256 (no overlap) ────────────────────
    DatasetSpec(
        name="14_bottom_crop_s256",
        description=(
            "Bottom-crop with no overlap (stride = patch width). Each ping "
            "appears in exactly one patch. Cleanest independence between "
            "samples, combined with focused depth range around the bottom."
        ),
        patch_width=256,
        stride=256,
        crop_mode='bottom',
        crop_padding=64,
    ),

    # ── 15. Bottom-crop, stride 64 (dense) ──────────────────────────
    DatasetSpec(
        name="15_bottom_crop_s64",
        description=(
            "Bottom-crop with dense overlap (stride 64). Maximises training "
            "data while focusing on the bottom region. Higher patch "
            "correlation but every bottom pixel is seen in ~4 patches."
        ),
        patch_width=256,
        stride=64,
        crop_mode='bottom',
        crop_padding=64,
    ),
]

DATASETS: dict[str, DatasetSpec] = {s.name: s for s in DATASET_SPECS}


def resolve_dataset(name_or_index) -> DatasetSpec:
    """Look up a dataset preset by full name, suffix, or list index.

    Accepts ``"00_baseline"``, the prefix-stripped ``"baseline"``, or the
    integer index ``0`` / ``"0"``. Raises KeyError with the list of available
    names if nothing matches.
    """
    key = str(name_or_index).strip()

    if key in DATASETS:
        return DATASETS[key]

    if key.isdigit():
        idx = int(key)
        if 0 <= idx < len(DATASET_SPECS):
            return DATASET_SPECS[idx]

    for spec in DATASET_SPECS:
        stripped = spec.name.split('_', 1)[1] if '_' in spec.name else spec.name
        if key == stripped:
            return spec

    available = ", ".join(s.name for s in DATASET_SPECS)
    raise KeyError(
        f"Unknown dataset preset {key!r}. Available presets: {available}")


def format_dataset_list() -> str:
    """Return a human-readable table of all presets for ``--list-datasets``."""
    lines = [f"{'idx':>3}  {'name':<24} description"]
    lines.append("-" * 78)
    for i, spec in enumerate(DATASET_SPECS):
        first = spec.description.split('. ')[0]
        lines.append(f"{i:>3}  {spec.name:<24} {first}")
    return "\n".join(lines)
