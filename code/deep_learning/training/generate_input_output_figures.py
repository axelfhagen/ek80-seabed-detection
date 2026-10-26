"""Generate figures showing the actual model input channels and output prediction."""

import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from deep_learning.unet_model import create_unet, predict_full_echogram
from deep_learning.evaluate import load_file_data

ZARR_PATH = PROJECT_ROOT / "Datasets/echopype_zarr/D20240131-T092044.zarr"
CHECKPOINT_DIR = PROJECT_ROOT / "code/checkpoints/exp_20260420_080950_finetune_stages"
OUTPUT_DIR = PROJECT_ROOT / "Figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FREQ1 = 38_000
FREQ2 = 200_000
VMIN, VMAX = -80.0, -34.0
TARGET_SIZE = (512, 512)
PATCH_WIDTH = 512
STRIDE = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model():
    model = create_unet(encoder_name="resnet34", in_channels=3)
    ckpt_path = CHECKPOINT_DIR / "unet_resnet34_best.pth"
    state = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.to(DEVICE).eval()
    return model


def main():
    print("Loading data...")
    sv_channels, gt_mask, _, range_arr = load_file_data(
        ZARR_PATH, FREQ1, FREQ2, vmin=VMIN, vmax=VMAX,
    )

    print("Loading model and running inference...")
    model = load_model()
    pred_prob = predict_full_echogram(
        model, sv_channels, TARGET_SIZE, DEVICE,
        patch_width=PATCH_WIDTH, stride=STRIDE,
    )
    pred_mask = (pred_prob >= 0.5).astype(np.uint8)

    ch_names = ["38 kHz (Sv)", "200 kHz (Sv)", "Frequency difference"]

    # --- Figure 1: Three grayscale channels ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for i, (ax, name) in enumerate(zip(axes, ch_names)):
        im = ax.imshow(sv_channels[i], cmap="gray", aspect="auto",
                       vmin=0, vmax=1)
        ax.set_title(name, fontsize=13)
        ax.set_xlabel("Ping")
        if i == 0:
            ax.set_ylabel("Range sample")
        else:
            ax.set_yticklabels([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Model input channels (normalised to [0, 1])", fontsize=15, y=1.02)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "input_channels.png", dpi=200, bbox_inches="tight")
    print(f"Saved {OUTPUT_DIR / 'input_channels.png'}")
    plt.close(fig)

    # --- Figure 2: RGB composite + GT mask + prediction ---
    rgb = np.stack([
        sv_channels[0],
        sv_channels[1],
        np.clip(sv_channels[2], 0, 1),
    ], axis=-1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].imshow(rgb, aspect="auto")
    axes[0].set_title("RGB composite (R=38 kHz, G=200 kHz, B=diff)", fontsize=11)
    axes[0].set_xlabel("Ping")
    axes[0].set_ylabel("Range sample")

    axes[1].imshow(gt_mask, cmap="gray", aspect="auto", vmin=0, vmax=1)
    axes[1].set_title("Reference mask", fontsize=13)
    axes[1].set_xlabel("Ping")
    axes[1].set_yticklabels([])

    axes[2].imshow(pred_mask, cmap="gray", aspect="auto", vmin=0, vmax=1)
    axes[2].set_title("Model prediction", fontsize=13)
    axes[2].set_xlabel("Ping")
    axes[2].set_yticklabels([])

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "input_rgb_vs_masks.png", dpi=200, bbox_inches="tight")
    print(f"Saved {OUTPUT_DIR / 'input_rgb_vs_masks.png'}")
    plt.close(fig)

    print("Done.")


if __name__ == "__main__":
    main()
