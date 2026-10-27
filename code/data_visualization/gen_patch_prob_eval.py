"""Hake-style evaluation of the U-Net on held-out test 512x512 patches.

Mirrors the two-panel layout of eval_hake_test_raw.py, on the exact three-
channel patches read straight from the patch cache (no sliding-window
reconstruction). Patches are restricted to the canonical held-out survey split
(default: test = feb_2025 + jun_2025) and evaluated with a checkpoint that did
NOT train on those surveys, so the metrics are genuine generalization numbers.

For each patch: left panel is either the 38 kHz input Sv or an RGB composite
of the three input channels, with the classical-detector reference label and
the U-Net prediction overlaid. Right panel is the U-Net bottom probability
with the same lines. IoU, Dice, and per-ping bottom-line error (in samples)
are shown in the legend.

Output: Figures/patch_prob_eval/v2/

Usage:
    python code/data_visualization/gen_patch_prob_eval.py --n 8 --split test
    python code/data_visualization/gen_patch_prob_eval.py --n 12 --split test \
        --display rgb --selection hardest \
        --output-dir Figures/patch_prob_eval/v4_rgb_hard
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from data_visualization.thesis_style import apply_thesis_style, style_colorbar, savefig
from deep_learning.unet_model import create_unet
from deep_learning.evaluate import extract_bottom_line, bottom_depth_error
from deep_learning.training.survey_groups import SURVEYS, SPLITS

CACHE_DIR = PROJECT_ROOT / "Datasets" / "patch_cache"
# Survey-split checkpoint that holds out feb_2025 + jun_2025 (does NOT train on
# the test surveys), so test-patch metrics reflect true generalization.
DEFAULT_CHECKPOINT = PROJECT_ROOT / "code/checkpoints/chunk_arch_sweep_2/smp_unet_resnet34_best.pth"
OUTPUT_DIR = PROJECT_ROOT / "Figures" / "patch_prob_eval" / "v2"


def split_prefixes(split):
    """Date-prefix tuple for a canonical survey split ('test', 'val', ...)."""
    return tuple(p for survey in SPLITS[split] for p in SURVEYS[survey]["date_prefixes"])


def load_model(checkpoint_path, device):
    model = create_unet(encoder_name="resnet34", in_channels=3)
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    sd = state.get("model_state_dict", state)
    cleaned = {k.removeprefix("model."): v for k, v in sd.items()}
    model.load_state_dict(cleaned, strict=True)
    model.to(device).eval()
    return model


@torch.no_grad()
def predict_patches(model, patches, device, batch_size=8):
    """patches: (N, 3, 512, 512) float32 -> predictions (N, 512, 512)."""
    preds = []
    for start in range(0, len(patches), batch_size):
        batch_np = np.array(patches[start : start + batch_size], dtype=np.float32, copy=True)
        inp = torch.from_numpy(batch_np).to(device)
        logits = model(inp)
        batch_pred = torch.sigmoid(logits)[:, 0].cpu().numpy().astype(np.float32)
        preds.append(batch_pred)
    return np.concatenate(preds, axis=0) if preds else np.empty((0,))


@torch.no_grad()
def predict_patch(model, patch, device):
    """patch: (3, 512, 512) float32 -> prediction (512, 512) float32 in [0, 1]."""
    return predict_patches(model, patch[np.newaxis], device, batch_size=1)[0]


def sample_patches(patch_files, n_patches, prefer_bottom, rng):
    """Pick n_patches (survey, patch_idx, patch, mask), spread across files."""
    selected = []
    files = list(patch_files)
    rng.shuffle(files)
    fi = 0
    while len(selected) < n_patches and files:
        pf = files[fi % len(files)]
        mf = pf.with_name(pf.name.replace(".patches.npy", ".masks.npy"))
        if not mf.exists():
            files.remove(pf)
            continue
        patches = np.load(pf, mmap_mode="r")
        masks = np.load(mf, mmap_mode="r")
        n = patches.shape[0]
        if n == 0:
            files.remove(pf)
            continue

        idxs = np.arange(n)
        if prefer_bottom:
            frac = masks[:].reshape(n, -1).mean(axis=1)
            visible = (frac > 0.10) & (frac < 0.75)
            if not visible.any():
                files.remove(pf)
                continue
            cand = idxs[visible]
            idxs = cand[np.argsort(np.abs(frac[cand] - 0.4))]
        else:
            rng.shuffle(idxs)

        picked = idxs[0]
        survey = pf.name.split(".")[0]
        selected.append(
            (
                survey,
                int(picked),
                np.asarray(patches[picked], dtype=np.float32),
                np.asarray(masks[picked], dtype=np.uint8),
            )
        )
        fi += 1
        if fi >= len(files) and prefer_bottom:
            break
    return selected[:n_patches]


def channels_to_rgb_composite(patch_chw):
    """Map normalized [38 kHz, 200 kHz, diff] channels to display RGB."""

    def to01(channel):
        return np.clip((channel + 1.0) * 0.5, 0.0, 1.0)

    return np.stack([to01(patch_chw[0]), to01(patch_chw[1]), to01(patch_chw[2])], axis=-1)


def candidate_indices(
    masks, max_per_file, prefer_bottom, rng, min_mask_frac=0.05, max_mask_frac=0.95
):
    """Candidate patch indices for hard-case mining."""
    n = masks.shape[0]
    if n == 0:
        return np.array([], dtype=int)
    idxs = np.arange(n)
    if prefer_bottom:
        frac = masks[:].reshape(n, -1).mean(axis=1)
        visible = (frac > min_mask_frac) & (frac < max_mask_frac)
        idxs = idxs[visible]
    if idxs.size <= max_per_file:
        return idxs
    return rng.choice(idxs, size=max_per_file, replace=False)


def sort_key_for_hard_case(metric):
    iou = metric["iou"]
    mae = metric["mae_px"]
    iou_sort = iou if np.isfinite(iou) else -1.0
    mae_sort = mae if np.isfinite(mae) else 1e9
    return iou_sort, -mae_sort


def mine_hard_patches(
    patch_files,
    n_patches,
    prefer_bottom,
    rng,
    model,
    device,
    candidates_per_file=8,
    max_candidates=128,
    batch_size=8,
):
    """Score candidate patches and return the worst model/label disagreements."""
    scored = []
    files = list(patch_files)
    rng.shuffle(files)
    if not files:
        mine_hard_patches.last_scored = 0
        return []
    per_file = max(1, min(candidates_per_file, int(np.ceil(max_candidates / len(files)))))
    seen = 0
    for pf in files:
        if seen >= max_candidates:
            break
        mf = pf.with_name(pf.name.replace(".patches.npy", ".masks.npy"))
        if not mf.exists():
            continue
        patches = np.load(pf, mmap_mode="r")
        masks = np.load(mf, mmap_mode="r")
        idxs = candidate_indices(masks, per_file, prefer_bottom, rng)
        if idxs.size == 0:
            continue
        remaining = max_candidates - seen
        idxs = idxs[:remaining]
        survey = pf.name.split(".")[0]
        patch_batch = np.asarray(patches[idxs], dtype=np.float32)
        mask_batch = np.asarray(masks[idxs], dtype=np.uint8)
        pred_batch = predict_patches(model, patch_batch, device, batch_size=batch_size)
        for picked, patch, mask, pred in zip(idxs, patch_batch, mask_batch, pred_batch):
            metric = compute_metrics(mask, pred)
            scored.append(
                (sort_key_for_hard_case(metric), survey, int(picked), patch, mask, pred, metric)
            )
            seen += 1
            if seen % 25 == 0:
                print(f"Scored {seen} candidate patches")

    scored.sort(key=lambda x: x[0])
    selected = []
    used_surveys = set()
    selected_keys = set()
    for _, survey, idx, patch, mask, pred, metric in scored:
        if survey in used_surveys:
            continue
        selected.append((survey, idx, patch, mask, pred, metric))
        selected_keys.add((survey, idx))
        used_surveys.add(survey)
        if len(selected) >= n_patches:
            mine_hard_patches.last_scored = len(scored)
            return selected
    for _, survey, idx, patch, mask, pred, metric in scored:
        if (survey, idx) in selected_keys:
            continue
        selected.append((survey, idx, patch, mask, pred, metric))
        selected_keys.add((survey, idx))
        if len(selected) >= n_patches:
            break
    mine_hard_patches.last_scored = len(scored)
    return selected


def compute_metrics(mask, prediction):
    h = mask.shape[0]
    pred_mask = (prediction >= 0.5).astype(np.uint8)
    inter = np.logical_and(mask, pred_mask).sum()
    union = np.logical_or(mask, pred_mask).sum()
    iou = inter / union if union else float("nan")
    denom = mask.sum() + pred_mask.sum()
    dice = 2 * inter / denom if denom else float("nan")
    gt_line = extract_bottom_line(mask, 0.5)
    pred_line = extract_bottom_line(prediction, 0.5)
    # No metres for patches: report bottom-line error in samples (pixels).
    errs = bottom_depth_error(pred_line, gt_line, np.arange(h))
    return dict(
        iou=iou,
        dice=dice,
        mae_px=float(np.mean(errs)) if errs.size else float("nan"),
        median_px=float(np.median(errs)) if errs.size else float("nan"),
        gt_line=gt_line,
        pred_line=pred_line,
    )


def _overlay(ax, gt_line, pred_line, h):
    for line, color, lw in ((gt_line, "cyan", 2.6), (pred_line, "red", 1.8)):
        v = line >= 0
        if v.any():
            ax.plot(
                np.where(v)[0], np.clip(line[v], 0, h - 1), color=color, linewidth=lw, alpha=0.9
            )


def plot_single(survey, idx, patch, mask, pred, m, save_path, display_mode="38khz"):
    h, w = pred.shape
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    if display_mode == "rgb":
        axes[0].imshow(channels_to_rgb_composite(patch), aspect="auto", origin="upper")
        axes[0].set_title("RGB input composite with reference label and prediction")
    else:
        axes[0].imshow(patch[0], aspect="auto", cmap="viridis", vmin=0, vmax=1, origin="upper")
        axes[0].set_title("38 kHz input Sv with reference label and prediction")
    _overlay(axes[0], m["gt_line"], m["pred_line"], h)
    im = axes[1].imshow(pred, aspect="auto", cmap="RdYlBu_r", vmin=0, vmax=1, origin="upper")
    _overlay(axes[1], m["gt_line"], m["pred_line"], h)
    axes[1].set_title("U-Net bottom probability")
    style_colorbar(fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04), label="P(bottom)")
    for ax in axes:
        ax.set_xlabel("Ping")
        ax.set_ylabel("Range (sample)")
    leg = fig.legend(
        handles=[
            Line2D([0], [0], color="cyan", lw=2.6, label="Classical detector (reference label)"),
            Line2D([0], [0], color="red", lw=1.8, label="U-Net prediction"),
        ],
        loc="lower center",
        ncol=2,
        framealpha=0.9,
        bbox_to_anchor=(0.5, -0.13),
        title=f"IoU {m['iou']:.3f}    Dice {m['dice']:.3f}    bottom MAE {m['mae_px']:.1f} samples",
    )
    leg.get_title().set_fontsize(14)
    fig.suptitle(f"Held-out test patch: {survey} p{idx}", y=1.02)
    savefig(fig, save_path)
    plt.close(fig)


def plot_summary_grid(items, save_path, display_mode="38khz"):
    n = len(items)
    ncols = 2
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(10 * ncols, 5.5 * nrows))
    axes_flat = np.atleast_1d(axes).ravel()
    for ax, (survey, idx, patch, mask, pred, m) in zip(axes_flat, items):
        h = patch.shape[1]
        if display_mode == "rgb":
            ax.imshow(channels_to_rgb_composite(patch), aspect="auto", origin="upper")
        else:
            ax.imshow(patch[0], aspect="auto", cmap="viridis", vmin=0, vmax=1, origin="upper")
        _overlay(ax, m["gt_line"], m["pred_line"], h)
        ax.set_title(
            f"{survey} p{idx}\nIoU {m['iou']:.2f} | MAE {m['mae_px']:.1f} samp", fontsize=13
        )
        ax.set_xlabel("Ping")
        ax.set_ylabel("Range (sample)")
    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)
    fig.legend(
        handles=[
            Line2D([0], [0], color="cyan", lw=2.6, label="Classical detector (reference label)"),
            Line2D([0], [0], color="red", lw=1.8, label="U-Net prediction"),
        ],
        loc="lower center",
        ncol=2,
        framealpha=0.9,
        bbox_to_anchor=(0.5, -0.01),
    )
    fig.suptitle("Test patches: classical-detector label vs U-Net prediction", y=1.005)
    fig.tight_layout(rect=[0, 0.02, 1, 0.99])
    savefig(fig, save_path)
    plt.close(fig)


def write_readme(
    out_dir, checkpoint, split, items, display_mode, selection, candidates_scored=None
):
    lines = "\n".join(
        f"  {survey} p{idx}: IoU {m['iou']:.2f}, bottom MAE {m['mae_px']:.1f} samples"
        for survey, idx, _, _, _, m in items
    )
    if display_mode == "rgb":
        left_panel = (
            "RGB composite of the normalized three input channels "
            "(red=38 kHz, green=200 kHz, blue=38-200 kHz diff)"
        )
    else:
        left_panel = "38 kHz input Sv"
    candidate_line = (
        f"Candidate patches scored: {candidates_scored}\n" if candidates_scored is not None else ""
    )
    (out_dir / "_README.md").write_text(f"""# Patch probability evaluation ({out_dir.name})

patches  test-set  probability  inference  classical-label

Two-panel U-Net evaluation on the exact 512x512 three-channel patches read
straight from the patch cache, in the same layout as the Hake test-raw figures.
Left: {left_panel} with the classical-detector reference label (cyan) and the
U-Net prediction (red). Right: predicted P(bottom). IoU/Dice and per-ping
bottom-line error (in samples, since patches carry no metric range axis) are in
each legend.

Revised from the initial patch_prob_eval set: reference label recoloured cyan
and renamed "Classical detector (reference label)", patches restricted to the
held-out '{split}' survey split with a checkpoint not trained on those surveys
(genuine generalization), and metrics moved out of the title into the legend.

Model: {Path(checkpoint).relative_to(PROJECT_ROOT)}
Split: {split} ({", ".join(SPLITS[split])})
Display: {display_mode}
Selection: {selection}
{candidate_line}Source: code/data_visualization/gen_patch_prob_eval.py
Created: {datetime.now():%Y-%m-%d %H:%M}

Patches shown: {len(items)}
{lines}
""")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    ap.add_argument("--cache-dir", default=str(CACHE_DIR))
    ap.add_argument("--output-dir", default=str(OUTPUT_DIR))
    ap.add_argument("--device", default=None)
    ap.add_argument("--n", type=int, default=8, help="number of patches to render")
    ap.add_argument(
        "--split",
        default="test",
        choices=list(SPLITS),
        help="canonical survey split to sample patches from",
    )
    ap.add_argument(
        "--all-patches",
        action="store_true",
        help="sample randomly instead of preferring clear-interface patches",
    )
    ap.add_argument(
        "--display",
        default="38khz",
        choices=["38khz", "rgb"],
        help="left-panel display: single 38 kHz channel or RGB composite",
    )
    ap.add_argument(
        "--selection",
        default="sample",
        choices=["sample", "hardest"],
        help="sample patches or mine low-IoU/high-error hard cases",
    )
    ap.add_argument(
        "--candidates-per-file",
        type=int,
        default=8,
        help="max candidate patches per file for --selection hardest",
    )
    ap.add_argument(
        "--max-candidates",
        type=int,
        default=128,
        help="max total candidates scored for --selection hardest",
    )
    ap.add_argument(
        "--batch-size", type=int, default=8, help="inference batch size for --selection hardest"
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    apply_thesis_style()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model = load_model(args.checkpoint, device)

    prefixes = split_prefixes(args.split)
    patch_files = sorted(
        p for p in Path(args.cache_dir).glob("*.512x512.patches.npy") if p.name.startswith(prefixes)
    )
    if not patch_files:
        raise SystemExit(
            f"No '{args.split}'-split patches (prefixes {prefixes}) in {args.cache_dir}"
        )
    print(f"Found {len(patch_files)} cached '{args.split}'-split patch files")

    rng = np.random.default_rng(args.seed)
    if args.selection == "hardest":
        items = mine_hard_patches(
            patch_files,
            args.n,
            prefer_bottom=not args.all_patches,
            rng=rng,
            model=model,
            device=device,
            candidates_per_file=args.candidates_per_file,
            max_candidates=args.max_candidates,
            batch_size=args.batch_size,
        )
        print(f"Selected {len(items)} hard patches")
    else:
        picked = sample_patches(patch_files, args.n, prefer_bottom=not args.all_patches, rng=rng)
        print(f"Selected {len(picked)} patches")
        items = []
        for survey, idx, patch, mask in picked:
            pred = predict_patch(model, patch, device)
            m = compute_metrics(mask, pred)
            items.append((survey, idx, patch, mask, pred, m))

    for i, (survey, idx, patch, mask, pred, m) in enumerate(items, 1):
        print(
            f"[{i}/{len(items)}] {survey} p{idx}  IoU {m['iou']:.3f}  "
            f"Dice {m['dice']:.3f}  MAE {m['mae_px']:.1f} samp"
        )
        plot_single(
            survey,
            idx,
            patch,
            mask,
            pred,
            m,
            out_dir / f"{survey}_p{idx}_eval.png",
            display_mode=args.display,
        )

    if items:
        plot_summary_grid(items, out_dir / "summary_grid.png", display_mode=args.display)
        candidates_scored = None
        if args.selection == "hardest":
            candidates_scored = getattr(mine_hard_patches, "last_scored", None)
        write_readme(
            out_dir,
            args.checkpoint,
            args.split,
            items,
            display_mode=args.display,
            selection=args.selection,
            candidates_scored=candidates_scored,
        )
    print(f"\nDone. {len(items)} patches -> {out_dir}")


if __name__ == "__main__":
    main()
