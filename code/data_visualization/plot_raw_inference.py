"""Plot echograms from raw files and run U-Net seabed inference.

Loads raw EK80 files (two frequencies), plots the echograms,
runs the trained U-Net model, and overlays the predicted bottom line.

Usage:
    python code/data_visualization/plot_raw_inference.py [--checkpoint PATH] [--max-files N]
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from raw_sensor_integration.data_loader import load_raw
from raw_sensor_integration.preprocessing import normalize_echogram
from deep_learning.unet_model import create_unet, predict_full_echogram
from deep_learning.evaluate import extract_bottom_line


RAW_DIRS = [
    "Datasets/extra_raw_files/basto1",
    "Datasets/extra_raw_files/diffraw",
    "Datasets/extra_raw_files/TransferNow-RawFilesEK80_4",
]

FREQ1 = 38_000
FREQ2 = 200_000
TARGET_SIZE = (512, 512)
PATCH_WIDTH = 512
STRIDE = 256
OUT_DIR = "code/plots_raw_inference"

DEFAULT_CHECKPOINT = (
    "code/checkpoints/exp_20260420_082458_cosine_gentle/unet_resnet34_best.pth"
)


def find_raw_files(raw_dirs, project_root, max_files=None):
    files = []
    for d in raw_dirs:
        full = os.path.join(project_root, d)
        found = sorted(glob.glob(os.path.join(full, "*.raw")))
        found = [f for f in found if not os.path.basename(f).startswith("._")]
        files.extend(found)
    if max_files:
        files = files[:max_files]
    return files


def has_both_freqs(raw_path):
    from raw_sensor_integration.metadata import get_channel_settings
    try:
        settings = get_channel_settings(raw_path)
        freqs = [s.get("frequency", 0) for s in settings]
        has_38 = any(abs(f - FREQ1) < 5000 for f in freqs)
        has_200 = any(abs(f - FREQ2) < 50000 for f in freqs)
        return has_38 and has_200
    except Exception:
        return False


def load_model(checkpoint_path, device):
    model = create_unet(encoder_name="resnet34", in_channels=3)
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    sd = state["model_state_dict"] if "model_state_dict" in state else state
    cleaned = {}
    for k, v in sd.items():
        cleaned[k.removeprefix("model.")] = v
    model.load_state_dict(cleaned, strict=False)
    model.to(device)
    model.eval()
    return model


def prepare_raw_channels(raw_path, max_pings=2000):
    """Load a raw file and build 3-channel normalized input.

    Raw files are uncalibrated (power in ~[-100, 0] dB). We use per-file
    percentile bounds to normalize to [0, 1], matching the distribution
    the model expects from calibrated data.
    """
    data1 = load_raw(raw_path, frequency=FREQ1, max_pings=max_pings)
    data2 = load_raw(raw_path, frequency=FREQ2, max_pings=max_pings)

    min_r = min(data1.n_samples, data2.n_samples)
    sv1 = data1.sv_db[:min_r]
    sv2 = data2.sv_db[:min_r]
    range_arr = data1.range_array[:min_r]

    finite1 = sv1[np.isfinite(sv1)]
    finite2 = sv2[np.isfinite(sv2)]
    vmin = float(np.percentile(np.concatenate([finite1, finite2]), 2))
    vmax = float(np.percentile(np.concatenate([finite1, finite2]), 98))

    norm1 = normalize_echogram(sv1, method='minmax',
                               vmin=vmin, vmax=vmax,
                               fill_nan=0.0).astype(np.float32)
    norm2 = normalize_echogram(sv2, method='minmax',
                               vmin=vmin, vmax=vmax,
                               fill_nan=0.0).astype(np.float32)
    diff = (norm1 - norm2).astype(np.float32)
    sv_channels = np.stack([norm1, norm2, diff], axis=0)

    return sv_channels, sv1, sv2, range_arr, data1.timestamps


def plot_echogram_with_inference(sv_channels, sv1_raw, sv2_raw, range_arr,
                                 prediction, fname, out_dir):
    """Plot echograms for both frequencies and overlay prediction."""
    pred_bottom = extract_bottom_line(prediction, threshold=0.5)

    fig, axes = plt.subplots(2, 2, figsize=(20, 12))

    n_range, n_pings = sv1_raw.shape
    extent = [0, n_pings, range_arr[-1], range_arr[0]]

    # 38 kHz echogram
    ax = axes[0, 0]
    vmin1, vmax1 = np.nanpercentile(sv1_raw, [2, 98])
    im = ax.imshow(sv1_raw, aspect='auto', cmap='viridis',
                   vmin=vmin1, vmax=vmax1, origin='upper', extent=extent)
    plt.colorbar(im, ax=ax, label='Power (dB)')
    ax.set_title(f'38 kHz — {fname}')
    ax.set_ylabel('Range (m)')
    ax.set_xlabel('Ping')

    # 200 kHz echogram
    ax = axes[0, 1]
    vmin2, vmax2 = np.nanpercentile(sv2_raw, [2, 98])
    im = ax.imshow(sv2_raw, aspect='auto', cmap='viridis',
                   vmin=vmin2, vmax=vmax2, origin='upper', extent=extent)
    plt.colorbar(im, ax=ax, label='Power (dB)')
    ax.set_title(f'200 kHz — {fname}')
    ax.set_ylabel('Range (m)')
    ax.set_xlabel('Ping')

    # 200 kHz with prediction overlay
    ax = axes[1, 0]
    im = ax.imshow(sv_channels[1], aspect='auto', cmap='viridis',
                   vmin=0, vmax=1, origin='upper', extent=extent)
    plt.colorbar(im, ax=ax, label='Normalized')

    valid_pred = pred_bottom >= 0
    if valid_pred.any():
        pings_p = np.where(valid_pred)[0]
        depths_p = range_arr[np.clip(pred_bottom[valid_pred], 0, len(range_arr) - 1)]
        ax.plot(pings_p, depths_p, color='red', linewidth=1.0, alpha=0.9)

    ax.set_title('200 kHz + predicted bottom (red)')
    ax.set_ylabel('Range (m)')
    ax.set_xlabel('Ping')

    # Prediction probability map
    ax = axes[1, 1]
    im = ax.imshow(prediction, aspect='auto', cmap='RdYlBu_r',
                   vmin=0, vmax=1, origin='upper', extent=extent)
    plt.colorbar(im, ax=ax, label='P(seabed)')
    ax.set_title('Seabed probability map')
    ax.set_ylabel('Range (m)')
    ax.set_xlabel('Ping')

    fig.suptitle(fname, fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_path = os.path.join(out_dir, f"{fname}_inference.png")
    fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {save_path}")
    return save_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--max-files", type=int, default=8)
    parser.add_argument("--max-pings", type=int, default=2000)
    parser.add_argument("--output-dir", default=OUT_DIR)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    project_root = str(Path(__file__).resolve().parents[2])

    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Device: {device}")

    os.makedirs(os.path.join(project_root, args.output_dir), exist_ok=True)

    ckpt_path = os.path.join(project_root, args.checkpoint)
    print(f"Loading model from {args.checkpoint} ...")
    model = load_model(ckpt_path, device)

    raw_files = find_raw_files(RAW_DIRS, project_root, max_files=None)
    dual_freq_files = [f for f in raw_files if has_both_freqs(f)]
    print(f"Found {len(raw_files)} raw files, {len(dual_freq_files)} with both 38+200 kHz")

    if args.max_files:
        dual_freq_files = dual_freq_files[:args.max_files]

    for i, raw_path in enumerate(dual_freq_files, 1):
        fname = Path(raw_path).stem
        print(f"\n[{i}/{len(dual_freq_files)}] {fname}")
        print(f"  File: {raw_path} ({os.path.getsize(raw_path) / 1e6:.1f} MB)")

        try:
            sv_channels, sv1, sv2, range_arr, timestamps = \
                prepare_raw_channels(raw_path, max_pings=args.max_pings)
            print(f"  Shape: {sv_channels.shape} ({sv_channels.shape[2]} pings, "
                  f"{sv_channels.shape[1]} range samples)")
            print(f"  Range: 0 - {range_arr[-1]:.1f} m")

            prediction = predict_full_echogram(
                model, sv_channels, TARGET_SIZE, device,
                patch_width=PATCH_WIDTH, stride=STRIDE)

            pred_bottom = extract_bottom_line(prediction, threshold=0.5)
            n_detected = np.sum(pred_bottom >= 0)
            print(f"  Bottom detected in {n_detected}/{len(pred_bottom)} pings")

            out_dir = os.path.join(project_root, args.output_dir)
            plot_echogram_with_inference(
                sv_channels, sv1, sv2, range_arr,
                prediction, fname, out_dir)

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    print("\nDone!")


if __name__ == "__main__":
    main()
