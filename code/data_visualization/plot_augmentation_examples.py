"""Visualise each training augmentation applied to a single example patch.

Loads one zarr file, extracts a representative 256-ping patch centred on the
seabed, resizes it to the 512x512 network input size, then applies each
augmentation from the training pipeline in isolation (probability forced to 1)
and renders the results as a grid of RGB composites.

Output: Figures/augmentations/
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from deep_learning.evaluate import load_file_data
from deep_learning.echo_dataset import PingDropout, RandomChannelDropout
from data_visualization.thesis_style import apply_thesis_style, savefig

apply_thesis_style()

DATASET_DIR = PROJECT_ROOT / "Datasets/echopype_zarr"
OUTPUT_DIR = PROJECT_ROOT / "Figures" / "augmentations"

FREQ1 = 38_000
FREQ2 = 200_000
VMIN, VMAX = -80.0, -34.0
TARGET = 512
PATCH_PINGS = 256
SEED = 7

# (label, zarr file) — one patch per survey month for variety.
EXAMPLES = [
    ("jan2025", "D20250119-T131025.zarr"),
    ("dec2023", "D20231220-T221825.zarr"),
    ("jan2024", "D20240131-T095953.zarr"),
    ("feb2024", "D20240201-T035002.zarr"),
    ("apr2024", "D20240408-T102749.zarr"),
    ("jul2024", "D20240701-T042645.zarr"),
    ("sep2024", "D20240914-T162824.zarr"),
    ("oct2024", "D20241001-T031443.zarr"),
    ("jan2025b", "D20250114-T075912.zarr"),
    ("feb2025", "D20250201-T003154.zarr"),
    ("jun2025", "D20250624-T134159.zarr"),
]


def _to_rgb(img_hwc):
    """Build a displayable RGB composite from a (H, W, 3) channel image."""
    r = np.clip(img_hwc[..., 0], 0, 1)
    g = np.clip(img_hwc[..., 1], 0, 1)
    b = np.clip(img_hwc[..., 2], 0, 1)
    return np.stack([r, g, b], axis=-1)


def extract_patch(sv_channels, gt_mask):
    """Crop a 256-ping window centred on the strongest seabed, then resize."""
    import cv2

    _, n_samples, n_pings = sv_channels.shape
    width = min(PATCH_PINGS, n_pings)

    # Pick the ping window with the most seabed pixels.
    col_bottom = gt_mask.sum(axis=0).astype(np.float64)
    kernel = np.ones(width)
    window_score = np.convolve(col_bottom, kernel, mode="valid")
    c0 = int(np.argmax(window_score)) if window_score.size else 0
    c1 = c0 + width

    # Crop range to span the water column down to just below the seabed line.
    # The mask is filled below the seabed (mask_mode='below'), so the seabed
    # line is the top edge of the mask: the first masked row in each column.
    win_mask = gt_mask[:, c0:c1].astype(bool)
    has_bottom = win_mask.any(axis=0)
    if has_bottom.any():
        top_edge = np.argmax(win_mask, axis=0)[has_bottom]
        r1 = min(n_samples, int(top_edge.max()) + 60)
    else:
        r1 = n_samples
    r0 = 0

    img = sv_channels[:, r0:r1, c0:c1]
    msk = gt_mask[r0:r1, c0:c1]

    img_hwc = np.transpose(img, (1, 2, 0))
    img_r = cv2.resize(img_hwc, (TARGET, TARGET), interpolation=cv2.INTER_LINEAR)
    msk_r = cv2.resize(msk.astype(np.uint8), (TARGET, TARGET),
                       interpolation=cv2.INTER_NEAREST)
    return img_r.astype(np.float32), msk_r.astype(np.uint8)


def build_augmentations():
    """Return (label, callable) pairs, each forcing its transform to p=1."""
    import albumentations as A

    def alb(transform):
        comp = A.Compose([transform])

        def fn(image):
            return comp(image=image, mask=np.zeros(image.shape[:2], np.uint8))["image"]

        return fn

    cd = RandomChannelDropout(p=0.6)
    pz = PingDropout(p=1.0, mode="zero")
    ps = PingDropout(p=1.0, mode="shuffle")
    pn = PingDropout(p=1.0, mode="noise")

    return [
        ("Original", None),
        ("Horizontal flip", alb(A.HorizontalFlip(p=1.0))),
        ("Gaussian noise", alb(A.GaussNoise(std_range=(0.04, 0.05), p=1.0))),
        ("Brightness / contrast",
         alb(A.RandomBrightnessContrast(brightness_limit=0.2,
                                        contrast_limit=0.2, p=1.0))),
        ("Channel dropout", lambda image: cd(image=image)),
        ("Ping dropout (zero)", lambda image: pz(image=image)),
        ("Ping dropout (shuffle)", lambda image: ps(image=image)),
        ("Ping dropout (noise)", lambda image: pn(image=image)),
    ]


def render_file(label, zarr_path, augs):
    print(f"[{label}] loading {zarr_path.name}")
    sv_channels, gt_mask, _, _ = load_file_data(
        zarr_path, FREQ1, FREQ2, vmin=VMIN, vmax=VMAX,
    )
    img, _ = extract_patch(sv_channels, gt_mask)

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    for ax, (name, fn) in zip(axes.ravel(), augs):
        np.random.seed(SEED)  # deterministic per-panel result
        out = img if fn is None else fn(img.copy())
        ax.imshow(_to_rgb(out), aspect="auto")
        ax.set_title(name)
        ax.set_xticks([])
        ax.set_yticks([])

    axes[1, 0].set_xlabel("Ping")
    axes[1, 0].set_ylabel("Range sample")

    fig.suptitle(
        f"Training augmentations applied to one input patch "
        f"({label} / {zarr_path.stem}, RGB: R=38 kHz, G=200 kHz, B=diff)"
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = OUTPUT_DIR / f"augmentation_examples_{label}_{zarr_path.stem}.png"
    savefig(fig, out_path)
    plt.close(fig)
    print(f"  saved {out_path}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    augs = build_augmentations()

    for label, zarr_name in EXAMPLES:
        zarr_path = DATASET_DIR / zarr_name
        if not zarr_path.exists():
            print(f"SKIP {zarr_name}: not found")
            continue
        render_file(label, zarr_path, augs)

    OUTPUT_DIR.joinpath("_README.md").write_text(
        "# Augmentation Examples\n\n"
        "augmentation   examples   unet   seabed   input_patch\n\n"
        "Each training augmentation applied in isolation (probability forced "
        "to 1) to a 512x512 input patch, rendered as RGB composites "
        "(R=38 kHz, G=200 kHz, B=frequency difference). One figure per survey "
        "month, each showing horizontal flip, Gaussian noise, "
        "brightness/contrast, channel dropout, and the three ping-dropout "
        "modes (zero, shuffle, noise).\n\n"
        "Source: code/data_visualization/plot_augmentation_examples.py\n"
    )
    print("Done.")


if __name__ == "__main__":
    main()
