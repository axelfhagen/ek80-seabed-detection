"""Single-panel echogram patches with the reference label and U-Net bottom overlaid.

Reads 512x512 patches straight from the training patch cache, runs the model on
each, and plots one figure per patch: the 38 kHz echogram with the reference
label line (cyan) and the U-Net predicted bottom line (red). No probability panel.

Output: Figures/patch_overlay/
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

CACHE_DIR = PROJECT_ROOT / "Datasets" / "patch_cache"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "code/checkpoints/chunk_arch_sweep_2/smp_unet_resnet34_best.pth"
OUTPUT_DIR = PROJECT_ROOT / "Figures" / "patch_overlay"


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


def sample_patches(patch_files, n_patches, rng):
    """Pick n_patches (survey, idx, patch, mask) with a clear seabed interface."""
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

        # Prefer patches where the seabed interface sits mid-patch.
        frac = masks[:].reshape(n, -1).mean(axis=1)
        visible = (frac > 0.10) & (frac < 0.75)
        if not visible.any():
            files.remove(pf)
            continue
        cand = np.arange(n)[visible]
        order = cand[np.argsort(np.abs(frac[cand] - 0.4))]

        picked = int(order[0])
        survey = pf.name.split(".")[0]
        selected.append(
            (
                survey,
                picked,
                np.asarray(patches[picked], dtype=np.float32),
                np.asarray(masks[picked], dtype=np.uint8),
            )
        )
        fi += 1
        if fi >= len(files):
            break
    return selected[:n_patches]


def make_figure(survey, idx, patch, mask, model, device, out_dir):
    pred = predict_patch(model, patch, device)
    h, w = pred.shape

    fig, ax = plt.subplots(figsize=FIGSIZE_HALF)
    # Channel 0 = normalised 38 kHz Sv (the echogram the model reads).
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

    ax.set_title("Seabed: reference label and U-Net prediction")
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
    out_path = out_dir / f"overlay_{survey}_p{idx}.png"
    savefig(fig, out_path)
    plt.close(fig)
    print(f"  saved {out_path}")
    return out_path


def write_readme(out_dir, checkpoint, n_total):
    readme = out_dir / "_README.md"
    readme.write_text(
        "# Patch seabed overlays\n\n"
        "patch  echogram  reference-label  prediction  overlay\n\n"
        "Single-panel 38 kHz echogram patches with the reference label (cyan) and "
        "the U-Net predicted seabed line (red) overlaid, without the probability "
        "panel. Patches are read straight from the training patch cache.\n\n"
        f"Model: {Path(checkpoint).relative_to(PROJECT_ROOT)}\n\n"
        "Source: code/data_visualization/gen_patch_overlay_only.py\n\n"
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
    parser.add_argument("--n", type=int, default=12, help="number of patches to render")
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
    items = sample_patches(patch_files, args.n, rng=rng)
    print(f"Selected {len(items)} patches")

    for survey, idx, patch, mask in items:
        make_figure(survey, idx, patch, mask, model, device, out_dir)

    write_readme(out_dir, args.checkpoint, len(items))
    print(f"\nDone. {len(items)} patches -> {out_dir}")


if __name__ == "__main__":
    main()
