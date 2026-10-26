"""Two-panel Gaussian-noise insertion figure.

Creates a focused figure for the Gaussian-noise synthetic insertion case:

  1. Gaussian-noised RGB input composite.
  2. U-Net bottom-probability map, with no bottom-line overlay.

Outputs are written to both:
  Figures/synthetic_insertions/
  Latex/Figures/synthetic_insertions/
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "code"))
sys.path.insert(0, str(PROJECT_ROOT / "code" / "data_visualization"))

from data_visualization.plot_augmentation_examples import (  # noqa: E402
    DATASET_DIR,
    FREQ1,
    FREQ2,
    PATCH_PINGS,
    SEED,
    TARGET,
    VMAX,
    VMIN,
    _to_rgb,
    extract_patch,
)
from data_visualization.thesis_style import (  # noqa: E402
    apply_thesis_style,
    savefig,
    style_colorbar,
)
from deep_learning.architectures import create_model  # noqa: E402
from deep_learning.evaluate import load_file_data  # noqa: E402

DEFAULT_ZARR = DATASET_DIR / "D20240201-T035002.zarr"
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "code/checkpoints/chunk_arch_sweep_2/smp_unet_resnet34_best.pth"
)
OUTPUT_DIR = PROJECT_ROOT / "Figures" / "synthetic_insertions"
LATEX_OUTPUT_DIR = PROJECT_ROOT / "Latex" / "Figures" / "synthetic_insertions"


def load_model(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    model = create_model(
        cfg.get("architecture", "smp_unet"),
        encoder_name=cfg.get("encoder_name", cfg.get("encoder", "resnet34")),
        encoder_weights=None,
        in_channels=cfg.get("in_channels", 3),
        classes=cfg.get("classes", 1),
        decoder_attention_type=cfg.get("decoder_attention_type", None),
    )
    state = ckpt.get("model_state_dict", ckpt)
    state = {k.removeprefix("model."): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    return model


def add_gaussian_noise(img: np.ndarray, sigma: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noisy = img + rng.normal(0.0, sigma, size=img.shape).astype(np.float32)
    return np.clip(noisy, 0.0, 1.0).astype(np.float32)


@torch.no_grad()
def predict_probability(
    model: torch.nn.Module,
    img_hwc: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    patch = np.transpose(img_hwc, (2, 0, 1))[None].astype(np.float32)
    logits = model(torch.from_numpy(patch).to(device))
    return torch.sigmoid(logits)[0, 0].detach().cpu().numpy().astype(np.float32)


def make_figure(
    noisy_img: np.ndarray,
    probability: np.ndarray,
    out_path: Path,
    title: str,
) -> None:
    apply_thesis_style()
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.0), constrained_layout=True)

    axes[0].imshow(_to_rgb(noisy_img), aspect="auto")
    axes[0].set_title("Gaussian-noised RGB composite")
    axes[0].set_xlabel("Ping")
    axes[0].set_ylabel("Range sample")

    im = axes[1].imshow(
        probability,
        aspect="auto",
        cmap="magma",
        vmin=0.0,
        vmax=1.0,
    )
    axes[1].set_title("Bottom-probability map")
    axes[1].set_xlabel("Ping")
    axes[1].set_ylabel("Range sample")
    cbar = fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    style_colorbar(cbar, label="P(bottom)")

    fig.suptitle(title)
    savefig(fig, out_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zarr", type=Path, default=DEFAULT_ZARR)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--sigma", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--latex-out-dir", type=Path, default=LATEX_OUTPUT_DIR)
    args = parser.parse_args()

    device = torch.device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.latex_out_dir.mkdir(parents=True, exist_ok=True)

    sv_channels, gt_mask, _, _ = load_file_data(
        args.zarr,
        FREQ1,
        FREQ2,
        vmin=VMIN,
        vmax=VMAX,
    )
    img, _ = extract_patch(sv_channels, gt_mask)
    noisy = add_gaussian_noise(img, sigma=args.sigma, seed=args.seed)

    model = load_model(args.checkpoint, device)
    probability = predict_probability(model, noisy, device)

    out_name = "gaussian_noise_rgb_probability.png"
    out_path = args.out_dir / out_name
    title = (
        f"Gaussian noise insertion "
        f"(sigma={args.sigma:g}, {args.zarr.stem}, {TARGET}x{TARGET}, "
        f"{PATCH_PINGS}-ping source crop)"
    )
    make_figure(noisy, probability, out_path, title)

    latex_path = args.latex_out_dir / out_name
    shutil.copy2(out_path, latex_path)
    print(f"Wrote {out_path}")
    print(f"Wrote {latex_path}")


if __name__ == "__main__":
    main()
