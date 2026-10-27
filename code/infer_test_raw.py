"""Run U-Net inference on raw files and produce 3x3 grid plots.

For each batch of 9 raw files, generates two 3x3 grids:
  1. Hard predictions (binary mask at threshold 0.5)
  2. Soft predictions (probability map)

Each panel shows the 200 kHz echogram with the predicted bottom line overlaid.
Files without both 38+200 kHz are skipped.

Also generates individual echogram PNGs for each file if they don't exist yet.

Usage:
    python code/infer_test_raw.py --raw-dir $DATA_ROOT/test_raw_2
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from raw_sensor_integration.data_loader import load_raw, load_zarr, raw_to_zarr
from raw_sensor_integration.preprocessing import normalize_echogram
from raw_sensor_integration.metadata import get_channel_settings
from raw_sensor_integration.ping_parser import extract_pings, pings_to_power_image
from deep_learning.unet_model import create_unet, predict_full_echogram
from deep_learning.evaluate import extract_bottom_line

FREQ1 = 38_000
FREQ2 = 200_000
TARGET_SIZE = (512, 512)
PATCH_WIDTH = 512
STRIDE = 256

# Training normalisation window (calibrated Sv, dB re 1 m⁻¹). Channels built
# from calibrated data are clipped to this fixed window — matching the
# train/val pipeline in echo_dataset.prepare_patches_from_zarr.
TRAIN_VMIN, TRAIN_VMAX = -80.0, -34.0

DEFAULT_CHECKPOINT = (
    "code/checkpoints/exp_20260511_191547_all_surveys_jun2025/smp_unet_resnet34_best.pth"
)
DEFAULT_OUTPUT_DIR = "code/plots_test_raw"
DEFAULT_ZARR_CACHE = "Datasets/echopype_zarr/from_raw"


def has_both_freqs(raw_path):
    try:
        settings = get_channel_settings(raw_path)
        freqs = [s.get("frequency", 0) for s in settings.values()]
        has_38 = any(abs(f - FREQ1) < 5000 for f in freqs)
        has_200 = any(abs(f - FREQ2) < 50000 for f in freqs)
        return has_38 and has_200
    except Exception:
        return False


def load_model(checkpoint_path, device):
    model = create_unet(encoder_name="resnet34", in_channels=3)
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    sd = state.get("model_state_dict", state)
    cleaned = {k.removeprefix("model."): v for k, v in sd.items()}
    model.load_state_dict(cleaned, strict=False)
    model.to(device)
    model.eval()
    return model


def prepare_raw_channels(raw_path, max_pings=2000):
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

    norm1 = normalize_echogram(sv1, method="minmax", vmin=vmin, vmax=vmax, fill_nan=0.0).astype(
        np.float32
    )
    norm2 = normalize_echogram(sv2, method="minmax", vmin=vmin, vmax=vmax, fill_nan=0.0).astype(
        np.float32
    )
    diff = (norm1 - norm2).astype(np.float32)
    sv_channels = np.stack([norm1, norm2, diff], axis=0)
    return sv_channels, sv1, sv2, range_arr


def ensure_calibrated_zarr(raw_path, zarr_dir, waveform_mode="BB", encode_mode="complex"):
    """Convert a .raw file to a calibrated Sv zarr (cached). Returns the Path.

    The conversion (echopype ``compute_Sv``: TVG, absorption, gain, beam
    correction) is the expensive step, so the result is cached under
    ``zarr_dir`` and reused on subsequent runs.
    """
    raw_path = Path(raw_path)
    zarr_dir = Path(zarr_dir)
    zarr_dir.mkdir(parents=True, exist_ok=True)
    zarr_path = zarr_dir / f"{raw_path.stem}.zarr"
    if zarr_path.exists():
        print(f"  Using cached calibrated zarr: {zarr_path}")
        return zarr_path
    print(f"  Calibrating {raw_path.name} -> {zarr_path.name} via echopype ...")
    raw_to_zarr(
        raw_path, output_path=zarr_path, waveform_mode=waveform_mode, encode_mode=encode_mode
    )
    return zarr_path


def prepare_calibrated_channels(
    raw_path,
    zarr_dir=DEFAULT_ZARR_CACHE,
    max_range_m=None,
    vmin=TRAIN_VMIN,
    vmax=TRAIN_VMAX,
    waveform_mode="BB",
    encode_mode="complex",
):
    """Build model channels from calibrated Sv with the training fixed clip.

    Unlike ``prepare_raw_channels`` (uncalibrated power, per-file pooled
    percentile), this converts the raw file to calibrated Sv via echopype —
    so TVG and absorption are applied — and normalises each channel with the
    fixed ``[vmin, vmax]`` window used in training. This matches the train/val
    pipeline (``echo_dataset.prepare_patches_from_zarr``) instead of skewing
    the input distribution at inference time.
    """
    zarr_path = ensure_calibrated_zarr(raw_path, zarr_dir, waveform_mode, encode_mode)
    d1 = load_zarr(str(zarr_path), frequency=FREQ1, max_range_m=max_range_m)
    d2 = load_zarr(str(zarr_path), frequency=FREQ2, max_range_m=max_range_m)

    min_r = min(d1.sv_db.shape[0], d2.sv_db.shape[0])
    sv1 = d1.sv_db[:min_r]
    sv2 = d2.sv_db[:min_r]
    range_arr = d1.range_array[:min_r]

    norm1 = normalize_echogram(sv1, method="minmax", vmin=vmin, vmax=vmax, fill_nan=0.0).astype(
        np.float32
    )
    norm2 = normalize_echogram(sv2, method="minmax", vmin=vmin, vmax=vmax, fill_nan=0.0).astype(
        np.float32
    )
    diff = (norm1 - norm2).astype(np.float32)
    sv_channels = np.stack([norm1, norm2, diff], axis=0)
    return sv_channels, sv1, sv2, range_arr


def generate_echogram(raw_path, out_dir, max_pings=1000):
    """Generate a multi-channel echogram PNG for a single raw file."""
    fname = Path(raw_path).stem
    out_path = os.path.join(out_dir, f"{fname}_echogram.png")
    if os.path.exists(out_path):
        return out_path

    pings = extract_pings(raw_path, max_pings=max_pings)
    if not pings:
        print(f"  No pings in {fname} — skipping echogram")
        return None

    ping_channels = sorted(set(p.channel_id for p in pings))
    n_ch = len(ping_channels)
    fig, axes = plt.subplots(1, n_ch, figsize=(7 * n_ch, 8))
    if n_ch == 1:
        axes = [axes]

    for ax, ch in zip(axes, ping_channels):
        power_db = pings_to_power_image(pings, ch)
        if power_db.size == 0:
            ax.set_title(f"{ch}\n(no data)")
            continue
        n_samples, n_pings_ch = power_db.shape
        finite = power_db[np.isfinite(power_db)]
        if finite.size == 0:
            ax.set_title(f"{ch}\n(all NaN)")
            continue
        vmin, vmax = np.percentile(finite, [2, 98])
        im = ax.imshow(
            power_db, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax, origin="upper"
        )
        plt.colorbar(im, ax=ax, label="Power (dB)")
        freq_label = ch.split()[-1] if " " in ch else ch
        ax.set_title(f"{freq_label}\n{n_pings_ch} pings, {n_samples} samples")
        ax.set_xlabel("Ping index")
        ax.set_ylabel("Sample index")

    fig.suptitle(fname, fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"  Echogram: {out_path}")
    return out_path


def plot_3x3_grid(results, mode, title, save_path):
    """Plot a 3x3 grid of inference results.

    mode: 'hard' for binary mask overlay, 'soft' for probability map.
    """
    n = len(results)
    nrows = 3
    ncols = 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(24, 20))
    axes_flat = axes.ravel()

    for i, res in enumerate(results):
        if i >= 9:
            break
        ax = axes_flat[i]
        echogram = res["sv_channels"][1]  # 200 kHz
        n_range, n_pings = echogram.shape
        range_arr = res["range_arr"]
        extent = [0, n_pings, range_arr[-1], range_arr[0]]

        if mode == "hard":
            ax.imshow(
                echogram,
                aspect="auto",
                cmap="viridis",
                vmin=0,
                vmax=1,
                origin="upper",
                extent=extent,
            )
            pred_binary = (res["prediction"] >= 0.5).astype(np.uint8)
            pred_bottom = extract_bottom_line(pred_binary, threshold=0.5)
            valid = pred_bottom >= 0
            if valid.any():
                pings_p = np.where(valid)[0]
                depths_p = range_arr[np.clip(pred_bottom[valid], 0, len(range_arr) - 1)]
                ax.plot(pings_p, depths_p, color="red", linewidth=1.0, alpha=0.9)
            ax.set_title(res["name"], fontsize=9)
        else:
            ax.imshow(
                res["prediction"],
                aspect="auto",
                cmap="RdYlBu_r",
                vmin=0,
                vmax=1,
                origin="upper",
                extent=extent,
            )
            pred_bottom = extract_bottom_line(res["prediction"], threshold=0.5)
            valid = pred_bottom >= 0
            if valid.any():
                pings_p = np.where(valid)[0]
                depths_p = range_arr[np.clip(pred_bottom[valid], 0, len(range_arr) - 1)]
                ax.plot(pings_p, depths_p, color="cyan", linewidth=0.8, alpha=0.8)
            ax.set_title(res["name"], fontsize=9)

        ax.set_ylabel("Range (m)", fontsize=7)
        ax.set_xlabel("Ping", fontsize=7)
        ax.tick_params(labelsize=7)

    for j in range(n, 9):
        axes_flat[j].set_visible(False)

    if mode == "hard":
        legend_elements = [
            Line2D([0], [0], color="red", lw=1.5, label="Predicted bottom"),
        ]
    else:
        legend_elements = [
            Line2D([0], [0], color="cyan", lw=1.5, label="Predicted bottom (p=0.5)"),
        ]
    fig.legend(handles=legend_elements, loc="upper right", fontsize=10, framealpha=0.9)

    mode_label = "Hard Predictions" if mode == "hard" else "Probability Maps"
    fig.suptitle(f"{title} — {mode_label}", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-pings", type=int, default=2000)
    parser.add_argument("--device", default=None)
    parser.add_argument("--skip-echograms", action="store_true")
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Convert raw files to calibrated Sv via echopype "
        "(TVG + absorption) and use the training fixed "
        "[-80,-34] clip, instead of uncalibrated power "
        "with a pooled percentile clip.",
    )
    parser.add_argument(
        "--zarr-cache",
        default=DEFAULT_ZARR_CACHE,
        help="Where calibrated zarr stores are cached (only used with --calibrate).",
    )
    args = parser.parse_args()

    project_root = str(Path(__file__).resolve().parents[1])

    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Device: {device}")

    out_dir = os.path.join(project_root, args.output_dir)
    echogram_dir = os.path.join(out_dir, "echograms")
    grid_dir = os.path.join(out_dir, "grids")
    os.makedirs(echogram_dir, exist_ok=True)
    os.makedirs(grid_dir, exist_ok=True)

    raw_files = sorted(Path(args.raw_dir).rglob("*.raw"))
    raw_files = [f for f in raw_files if not f.name.startswith("._")]
    print(f"Found {len(raw_files)} raw files in {args.raw_dir}")

    # Step 1: generate echograms for files that don't have one yet
    if not args.skip_echograms:
        print("\n=== Generating echograms ===")
        for i, rf in enumerate(raw_files, 1):
            print(f"[{i}/{len(raw_files)}] {rf.name}")
            try:
                generate_echogram(str(rf), echogram_dir)
            except Exception as e:
                print(f"  ERROR generating echogram: {e}")

    # Step 2: filter to dual-freq files and run inference
    print("\n=== Checking frequencies ===")
    dual_freq = []
    for rf in raw_files:
        if has_both_freqs(str(rf)):
            dual_freq.append(rf)
        else:
            print(f"  Skipping {rf.name} — missing 38 or 200 kHz")
    print(f"{len(dual_freq)} files with both 38+200 kHz")

    if not dual_freq:
        print("No dual-frequency files found. Done.")
        return

    ckpt_path = os.path.join(project_root, args.checkpoint)
    print(f"\nLoading model from {args.checkpoint} ...")
    model = load_model(ckpt_path, device)

    # Step 3: run inference on all dual-freq files
    print("\n=== Running inference ===")
    all_results = []
    for i, rf in enumerate(dual_freq, 1):
        fname = rf.stem
        print(f"\n[{i}/{len(dual_freq)}] {fname} ({rf.stat().st_size / 1e6:.1f} MB)")
        try:
            if args.calibrate:
                zarr_cache = os.path.join(project_root, args.zarr_cache)
                sv_channels, sv1, sv2, range_arr = prepare_calibrated_channels(
                    str(rf), zarr_dir=zarr_cache
                )
            else:
                sv_channels, sv1, sv2, range_arr = prepare_raw_channels(
                    str(rf), max_pings=args.max_pings
                )
            print(f"  Shape: {sv_channels.shape}")

            prediction = predict_full_echogram(
                model, sv_channels, TARGET_SIZE, device, patch_width=PATCH_WIDTH, stride=STRIDE
            )

            pred_bottom = extract_bottom_line(prediction, threshold=0.5)
            n_detected = np.sum(pred_bottom >= 0)
            print(f"  Bottom detected in {n_detected}/{len(pred_bottom)} pings")

            all_results.append(
                {
                    "name": fname,
                    "sv_channels": sv_channels,
                    "prediction": prediction,
                    "range_arr": range_arr,
                }
            )
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback

            traceback.print_exc()

    # Step 4: generate 3x3 grid plots
    print(f"\n=== Generating 3x3 grids ({len(all_results)} files) ===")
    batch_size = 9
    for batch_idx in range(0, len(all_results), batch_size):
        batch = all_results[batch_idx : batch_idx + batch_size]
        batch_num = batch_idx // batch_size + 1
        total_batches = (len(all_results) + batch_size - 1) // batch_size

        title = f"Test Raw 2 — Batch {batch_num}/{total_batches}"

        hard_path = os.path.join(grid_dir, f"grid_hard_batch{batch_num:02d}.png")
        plot_3x3_grid(batch, "hard", title, hard_path)

        soft_path = os.path.join(grid_dir, f"grid_soft_batch{batch_num:02d}.png")
        plot_3x3_grid(batch, "soft", title, soft_path)

    print(f"\nDone! {len(all_results)} files processed, output in {args.output_dir}")


if __name__ == "__main__":
    main()
