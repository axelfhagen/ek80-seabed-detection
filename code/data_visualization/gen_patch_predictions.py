"""Run the trained U-Net on the actual 512x512 patches fed during training.

Loads patches straight from the fast patch cache (already the exact 3-channel
512x512 tensors the network sees), runs each through the model individually,
and plots input / ground truth / prediction per patch. No sliding-window
full-echogram reconstruction.

Output: Figures/patch_predictions/
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

from data_visualization.thesis_style import apply_thesis_style, style_colorbar
from deep_learning.unet_model import create_unet
from deep_learning.evaluate import extract_bottom_line

CACHE_DIR = PROJECT_ROOT / "Datasets" / "patch_cache"
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "code/checkpoints/exp_20260511_191547_all_surveys_jun2025/smp_unet_resnet34_best.pth"
)
OUTPUT_DIR = PROJECT_ROOT / "Figures" / "patch_predictions"

ROWS_PER_FIG = 4  # patches stacked per output figure


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
def predict_patch(model, patch, device):
    """patch: (3, 512, 512) float32 -> prediction (512, 512) float32 in [0, 1]."""
    inp = torch.from_numpy(patch[np.newaxis]).float().to(device)
    logits = model(inp)
    return torch.sigmoid(logits)[0, 0].cpu().numpy().astype(np.float32)


def list_fast_patch_files(cache_dir):
    return sorted(cache_dir.glob("*.512x512.patches.npy"))


def sample_patches(patch_files, n_patches, prefer_bottom, rng):
    """Pick n_patches (file_stem, patch_idx, patch, mask), spread across files."""
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
            # Prefer patches where the seabed interface sits mid-patch (a clear
            # transition to find), not patches that are almost entirely seabed.
            frac = masks[:].reshape(n, -1).mean(axis=1)
            visible = (frac > 0.10) & (frac < 0.75)
            if not visible.any():
                files.remove(pf)  # no clear interface in this file, skip it
                continue
            cand = idxs[visible]
            idxs = cand[np.argsort(np.abs(frac[cand] - 0.4))]
        else:
            rng.shuffle(idxs)

        # take one patch per file pass to spread variety across surveys
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


def make_figure(items, model, device, fig_idx, out_dir):
    n = len(items)
    fig, axes = plt.subplots(n, 4, figsize=(20, 4.6 * n), squeeze=False)

    for r, (survey, idx, patch, mask) in enumerate(items):
        pred = predict_patch(model, patch, device)
        h, w = pred.shape

        ax_in, ax_gt, ax_prob, ax_ov = axes[r]

        # Channel 0 = normalised 38 kHz Sv (the echogram the model reads)
        im0 = ax_in.imshow(patch[0], cmap="viridis", aspect="auto", vmin=0, vmax=1, origin="upper")
        ax_in.set_title(f"{survey}  patch {idx}: input (38 kHz)", fontsize=13)
        cb0 = fig.colorbar(im0, ax=ax_in, fraction=0.046, pad=0.04)
        style_colorbar(cb0, label="norm Sv", label_size=11, tick_size=9)

        ax_gt.imshow(mask, cmap="gray_r", aspect="auto", vmin=0, vmax=1, origin="upper")
        ax_gt.set_title(f"ground truth ({mask.mean() * 100:.0f}% seabed)", fontsize=13)

        im2 = ax_prob.imshow(pred, cmap="RdYlBu_r", aspect="auto", vmin=0, vmax=1, origin="upper")
        ax_prob.set_title("predicted P(bottom)", fontsize=13)
        cb2 = fig.colorbar(im2, ax=ax_prob, fraction=0.046, pad=0.04)
        style_colorbar(cb2, label="P", label_size=11, tick_size=9)

        ax_ov.imshow(patch[0], cmap="viridis", aspect="auto", vmin=0, vmax=1, origin="upper")
        gt_line = extract_bottom_line(mask, threshold=0.5)
        pr_line = extract_bottom_line(pred, threshold=0.5)
        gv = gt_line >= 0
        pv = pr_line >= 0
        if gv.any():
            ax_ov.plot(
                np.where(gv)[0],
                np.clip(gt_line[gv], 0, h - 1),
                color="red",
                linewidth=1.8,
                alpha=0.9,
            )
        if pv.any():
            ax_ov.plot(
                np.where(pv)[0],
                np.clip(pr_line[pv], 0, h - 1),
                color="cyan",
                linewidth=1.6,
                alpha=0.9,
            )
        ax_ov.set_title("bottom: GT vs predicted", fontsize=13)

        for ax in (ax_in, ax_gt, ax_prob, ax_ov):
            ax.set_xlabel("Ping", fontsize=11)
            ax.set_ylabel("Range (sample)", fontsize=11)
            ax.tick_params(labelsize=9)

    legend = [
        Line2D([0], [0], color="red", lw=2.5, label="Ground-truth bottom"),
        Line2D([0], [0], color="cyan", lw=2.5, label="Predicted bottom (p=0.5)"),
    ]
    fig.legend(handles=legend, loc="upper right", fontsize=12, framealpha=0.9)
    fig.suptitle(
        f"U-Net predictions on actual 512x512 input patches (part {fig_idx})", fontsize=18, y=1.0
    )
    fig.text(
        0.01,
        0.005,
        "Source: code/data_visualization/gen_patch_predictions.py | "
        "patches read straight from the training patch cache",
        fontsize=8,
        color="grey",
    )
    fig.tight_layout(rect=[0, 0.01, 1, 0.99])

    out_path = out_dir / f"patch_predictions_part{fig_idx:02d}.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")
    return out_path


def write_readme(out_dir, checkpoint, n_total):
    readme = out_dir / "_README.md"
    readme.write_text(
        "# U-Net predictions on actual input patches\n\n"
        "patches  input  seabed  inference  ground-truth\n\n"
        "What the trained seabed U-Net predicts on the exact 512x512 three-channel "
        "patches it is fed during training (read straight from the patch cache), "
        "rather than the sliding-window full-echogram reconstruction. Each row shows "
        "the 38 kHz input, the ground-truth mask, the predicted probability, and the "
        "ground-truth vs predicted bottom lines.\n\n"
        f"Model: {Path(checkpoint).relative_to(PROJECT_ROOT)}\n\n"
        "Source: code/data_visualization/gen_patch_predictions.py\n\n"
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
    parser.add_argument("--n", type=int, default=12, help="total number of patches to render")
    parser.add_argument(
        "--all-patches",
        action="store_true",
        help="do not prefer seabed-heavy patches; sample randomly",
    )
    parser.add_argument("--seed", type=int, default=0)
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

    rng = np.random.default_rng(args.seed)
    items = sample_patches(patch_files, args.n, prefer_bottom=not args.all_patches, rng=rng)
    print(f"Selected {len(items)} patches")

    for fig_idx, start in enumerate(range(0, len(items), ROWS_PER_FIG), 1):
        make_figure(items[start : start + ROWS_PER_FIG], model, device, fig_idx, out_dir)

    write_readme(out_dir, args.checkpoint, len(items))
    print(f"\nDone. {len(items)} patches -> {out_dir}")


if __name__ == "__main__":
    main()
