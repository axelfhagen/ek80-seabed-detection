"""Single-panel echogram patches with the LARGEST seabed prediction error.

Companion to gen_patch_overlay_only.py. Instead of picking clean mid-patch
interfaces (where prediction ~ label), this scans the patch cache, runs the
model on candidate patches, scores each by the bottom-line error between the
reference label and the U-Net prediction, and renders the worst offenders.

Error metric (per patch, in range samples): for every ping (column) that has a
reference seabed, take the absolute row distance to the predicted seabed; if the
model predicts no seabed in that column, count it as a full-height miss. The
patch score is the mean of these per-column errors, so both gross misalignment
and missed-seabed columns push the score up.

Output: Figures/patch_overlay_worst/
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from data_visualization.thesis_style import (
    apply_thesis_style,
    FIGSIZE_HALF,
    savefig,
)
from deep_learning.unet_model import create_unet
from deep_learning.evaluate import extract_bottom_line
from deep_learning.training.survey_groups import SURVEYS, SPLITS

CACHE_DIR = PROJECT_ROOT / "Datasets" / "patch_cache"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "code/checkpoints/chunk_arch_sweep_2/smp_unet_resnet34_best.pth"
OUTPUT_DIR = PROJECT_ROOT / "Figures" / "patch_overlay_worst"


def load_model(checkpoint_path, device):
    model = create_unet(encoder_name="resnet34", in_channels=3)
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    sd = state.get("model_state_dict", state)
    cleaned = {k.removeprefix("model."): v for k, v in sd.items()}
    model.load_state_dict(cleaned, strict=False)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def predict_batch(model, patches, device):
    """patches: (B, 3, 512, 512) float32 -> (B, 512, 512) prob maps in [0, 1]."""
    inp = torch.from_numpy(patches).float().to(device)
    logits = model(inp)
    return torch.sigmoid(logits)[:, 0].cpu().numpy().astype(np.float32)


def patch_error(mask, pred):
    """Mean per-column bottom-line error in range samples (missed columns = full height)."""
    h = mask.shape[0]
    ref = extract_bottom_line(mask, threshold=0.5)
    prd = extract_bottom_line(pred, threshold=0.5)
    rv = ref >= 0
    if not rv.any():
        return 0.0, 0.0
    cols = np.where(rv)[0]
    pv = prd[cols] >= 0
    err = np.empty(cols.shape[0], dtype=np.float64)
    err[pv] = np.abs(ref[cols][pv] - prd[cols][pv])
    err[~pv] = h  # model predicts no seabed where the label has one
    miss_frac = float((~pv).mean())
    return float(err.mean()), miss_frac


def list_fast_patch_files(cache_dir):
    return sorted(cache_dir.glob("*.512x512.patches.npy"))


def val_prefixes():
    """Date prefixes (e.g. 'D20241001') for the canonical validation survey(s)."""
    return tuple(pre for survey in SPLITS["val"] for pre in SURVEYS[survey]["date_prefixes"])


def filter_validation(patch_files):
    prefixes = val_prefixes()
    return [pf for pf in patch_files if pf.name.startswith(prefixes)]


def rank_all_patches(patch_files, model, device, batch_size):
    """Score every visible-interface patch and return records sorted worst-first.

    Each record is (score, survey, idx, file_path, miss_frac). No subsampling and
    no per-file cap, so the returned list is a true global ranking — element i is
    the (i+1)-th worst patch across all scanned files.
    """
    records = []
    n_scored = 0
    for fi, pf in enumerate(patch_files):
        mf = pf.with_name(pf.name.replace(".patches.npy", ".masks.npy"))
        if not mf.exists():
            continue
        patches = np.load(pf, mmap_mode="r")
        masks = np.load(mf, mmap_mode="r")
        n = patches.shape[0]
        if n == 0:
            continue

        # Keep patches with a clearly visible seabed interface so the error is
        # about getting the bottom wrong, not about empty/full patches.
        frac = masks[:].reshape(n, -1).mean(axis=1)
        visible = np.where((frac > 0.05) & (frac < 0.85))[0]
        if visible.size == 0:
            continue

        survey = pf.name.split(".")[0]
        for start in range(0, visible.size, batch_size):
            idxs = visible[start : start + batch_size]
            batch = np.asarray(patches[idxs], dtype=np.float32)
            preds = predict_batch(model, batch, device)
            for k, idx in enumerate(idxs):
                score, miss = patch_error(np.asarray(masks[idx]), preds[k])
                records.append((score, survey, int(idx), pf, miss))
                n_scored += 1

        if (fi + 1) % 25 == 0:
            print(f"  scanned {fi + 1}/{len(patch_files)} files, {n_scored} patches")

    records.sort(key=lambda r: r[0], reverse=True)
    print(f"  ranked {len(records)} patches total")
    return records


def make_figure(survey, idx, patch, mask, pred, score, out_dir, rank=None):
    h, w = pred.shape

    fig, ax = plt.subplots(figsize=FIGSIZE_HALF)
    ax.imshow(patch[0], cmap="viridis", aspect="equal", vmin=0, vmax=1, origin="upper")

    ref_line = extract_bottom_line(mask, threshold=0.5)
    pred_line = extract_bottom_line(pred, threshold=0.5)
    rv = ref_line >= 0
    pv = pred_line >= 0
    if rv.any():
        ax.plot(
            np.where(rv)[0],
            np.clip(ref_line[rv], 0, h - 1),
            color="cyan",
            linewidth=2.0,
            alpha=0.9,
            label="Reference label",
        )
    if pv.any():
        ax.plot(
            np.where(pv)[0],
            np.clip(pred_line[pv], 0, h - 1),
            color="red",
            linewidth=2.0,
            alpha=0.9,
            label="U-Net prediction",
        )

    ax.set_title("Difference in prediction and label")
    ax.set_xlabel("Ping")
    ax.set_ylabel("Range sample")
    ax.set_xlim(0, w - 1)
    ax.set_ylim(h - 1, 0)

    handles = [
        Line2D([0], [0], color="cyan", lw=2.5, label="Reference label"),
        Line2D([0], [0], color="red", lw=2.5, label="U-Net prediction"),
    ]
    ax.legend(handles=handles, loc="lower left", framealpha=0.9)

    fig.tight_layout()
    prefix = f"rank{rank:04d}_" if rank is not None else "worst_"
    out_path = out_dir / f"{prefix}{survey}_p{idx}.png"
    savefig(fig, out_path)
    plt.close(fig)
    print(f"  saved {out_path}  (rank {rank}, score {score:.1f})")
    return out_path


def write_readme(out_dir, checkpoint, n_total, scores, rank_start, rank_end, n_ranked, split_label):
    readme = out_dir / "_README.md"
    lo = min(scores) if scores else 0.0
    hi = max(scores) if scores else 0.0
    readme.write_text(
        "# Patch seabed overlays — error-ranked cases\n\n"
        "patch  echogram  reference-label  prediction  overlay  failure\n\n"
        "Single-panel 38 kHz echogram patches with the reference label (cyan) and "
        "the U-Net predicted seabed line (red) overlaid. Every patch with a visible "
        f"seabed in the {split_label} split is scored and globally ranked by "
        "bottom-line error; this folder shows ranks "
        f"{rank_start}-{rank_end} of {n_ranked} (rank 1 = worst). The very worst "
        "ranks are dominated by broken/erratic classical labels, so a mid-tail "
        "slice shows genuinely hard but real cases.\n\n"
        "Score = mean per-column absolute distance between predicted and "
        "reference seabed (range samples); columns where the model predicts no "
        "seabed under a labelled one are counted as a full-height miss.\n\n"
        f"Score range shown: {lo:.0f}-{hi:.0f} samples.\n\n"
        f"Model: {Path(checkpoint).relative_to(PROJECT_ROOT)}\n\n"
        "Source: code/data_visualization/gen_patch_overlay_worst.py\n\n"
        f"Patches shown: {n_total}\n\n"
        f"Created: {datetime.now():%Y-%m-%d %H:%M}\n"
    )
    print(f"  wrote {readme}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--cache-dir", default=str(CACHE_DIR))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--rank-start", type=int, default=100, help="first global error rank to render (1 = worst)"
    )
    parser.add_argument(
        "--rank-end", type=int, default=120, help="last global error rank to render (inclusive)"
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--val-only",
        action="store_true",
        default=True,
        help="rank only the canonical validation survey files",
    )
    parser.add_argument(
        "--all-files",
        dest="val_only",
        action="store_false",
        help="rank every cached file instead of validation only",
    )
    args = parser.parse_args()

    apply_thesis_style()

    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Device: {device}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {args.checkpoint}")
    model = load_model(args.checkpoint, device)

    patch_files = list_fast_patch_files(Path(args.cache_dir))
    if not patch_files:
        raise SystemExit(f"No fast patch cache (*.512x512.patches.npy) in {args.cache_dir}")
    print(f"Found {len(patch_files)} cached patch files")

    if args.val_only:
        patch_files = filter_validation(patch_files)
        split_label = "validation (oct_2024)"
        print(f"Restricted to {len(patch_files)} validation patch files")
    else:
        split_label = "all-files"

    ranked = rank_all_patches(patch_files, model, device, args.batch_size)
    if not ranked:
        raise SystemExit("No scorable patches found.")

    lo = max(1, args.rank_start)
    hi = min(len(ranked), args.rank_end)
    chosen = ranked[lo - 1 : hi]
    print(f"Rendering ranks {lo}-{hi} of {len(ranked)}")

    scores = []
    for offset, (score, survey, idx, pf, miss) in enumerate(chosen):
        rank = lo + offset
        patches = np.load(pf, mmap_mode="r")
        masks = np.load(pf.with_name(pf.name.replace(".patches.npy", ".masks.npy")), mmap_mode="r")
        patch = np.asarray(patches[idx], dtype=np.float32)
        mask = np.asarray(masks[idx], dtype=np.uint8)
        pred = predict_batch(model, patch[np.newaxis], device)[0]
        make_figure(survey, idx, patch, mask, pred, score, out_dir, rank=rank)
        scores.append(score)

    write_readme(out_dir, args.checkpoint, len(chosen), scores, lo, hi, len(ranked), split_label)
    print(f"\nDone. {len(chosen)} patches -> {out_dir}")


if __name__ == "__main__":
    main()
