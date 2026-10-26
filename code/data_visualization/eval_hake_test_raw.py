"""Evaluation plots for the NOAA Hake test-raw files.

Output: Figures/hake_test_eval/

These EK80 files store CW power/angle samples, which the project's lightweight
binary parser mis-decodes (it assumes complex float32). They are therefore
decoded with echopype into calibrated Sv (raw_to_zarr, CW/power) before use.

For a handful of files this builds the algorithmic bottom *label* (the ensemble
pipeline used to generate training labels) on the 38 kHz channel, runs the
U-Net *prediction*, overlays both, and reports below-region IoU/Dice and
per-ping bottom-depth error in metres. There is no human ground truth, so the
metrics measure label/prediction agreement.

Important caveat: the seabed in these transects sits at ~375-670 m, but the
200 kHz channel only records to ~300 m. The model's 200 kHz and frequency-
difference channels are therefore blank at the seabed, so the prediction relies
on 38 kHz alone, an out-of-distribution regime for the three-channel model.

Usage:
    python code/data_visualization/eval_hake_test_raw.py --n-files 6
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "code"))
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from raw_sensor_integration.data_loader import raw_to_zarr, load_zarr
from raw_sensor_integration.preprocessing import load_sv_from_zarr, normalize_echogram
from raw_sensor_integration.bottom_labelling import (
    detect_bottom_ensemble_pipeline, bottom_to_sample_index, create_bottom_mask)
from deep_learning.unet_model import create_unet, predict_full_echogram
from deep_learning.evaluate import extract_bottom_line, bottom_depth_error
from deep_learning.bottom_extraction import extract_bottom_dp
from data_visualization.thesis_style import apply_thesis_style, style_colorbar, savefig

FREQ1 = 38_000
FREQ2 = 200_000
VMIN, VMAX = -80.0, -34.0          # calibrated-Sv normalisation (training defaults)
BT_THRESHOLD = -31.0               # calibrated bottom-detection threshold
TARGET_SIZE = (512, 512)
PATCH_WIDTH = 512
STRIDE = 256
CROP_PAD_SAMPLES = 160             # rows of context kept above/below the seabed

RAW_DIR = Path("/cluster/work/axelfh/test_raw_hake")
ZARR_DIR = Path("/cluster/work/axelfh/test_raw_hake_zarr")
OUT_DIR = PROJECT_ROOT / "Figures" / "hake_test_eval"
# Most general checkpoint: 1728-file random-pool split, patch_width 512, with
# hflip/noise/brightness augmentation — the best bet for out-of-distribution data.
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT / "code/checkpoints/exp_20260525_220040_baseline_random_split_2/"
    "smp_unet_resnet34_best.pth")


def ensure_zarr(raw_path: Path) -> Path:
    zp = ZARR_DIR / (raw_path.stem + ".zarr")
    if not zp.exists():
        ZARR_DIR.mkdir(parents=True, exist_ok=True)
        raw_to_zarr(raw_path, output_path=zp,
                    waveform_mode="CW", encode_mode="power")
    return zp


def load_model(checkpoint_path, device):
    model = create_unet(encoder_name="resnet34", in_channels=3)
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    sd = state.get("model_state_dict", state)
    model.load_state_dict({k.removeprefix("model."): v for k, v in sd.items()},
                          strict=False)
    model.to(device).eval()
    return model


def resample_200k_to_38k_grid(sv2, range2, range1):
    """Interpolate each 200 kHz ping column onto the 38 kHz range grid.

    Depths beyond the 200 kHz recording range are left NaN (filled later)."""
    n_pings = sv2.shape[1]
    out = np.full((len(range1), n_pings), np.nan, dtype=np.float32)
    in_range = range1 <= range2[-1]
    for p in range(n_pings):
        out[in_range, p] = np.interp(range1[in_range], range2, sv2[:, p])
    return out


def bottom_crop_bounds(label_idx, n_range):
    rows = label_idx[label_idx >= 0]
    if rows.size == 0:
        return 0, min(n_range, 512)
    top = int(rows.min()) - CROP_PAD_SAMPLES
    bot = int(rows.max()) + CROP_PAD_SAMPLES
    top = max(0, top)
    height = int(np.ceil((bot - top) / 256) * 256)
    bot = min(n_range, top + height)
    top = max(0, bot - height)
    return top, bot


def load_group(stems):
    """Load and concatenate consecutive (time-contiguous) files along the ping
    axis so the stitched echogram is wide enough for native 512-ping windows."""
    sv1s, sv2s, heaves = [], [], []
    range1 = range2 = pulse = freq = draft = None
    for stem in stems:
        zp = ensure_zarr(RAW_DIR / (stem + ".raw"))
        s1, r1, _, p1, _, _ = load_sv_from_zarr(zp, FREQ1)
        s2, r2, _, _, _, _ = load_sv_from_zarr(zp, FREQ2)
        d1 = load_zarr(zp, frequency=FREQ1)
        sv1s.append(s1); sv2s.append(s2)
        heaves.append(d1.heave if d1.heave is not None else np.zeros(s1.shape[1]))
        range1, range2 = r1, r2
        pulse, freq, draft = p1, d1.frequency_hz, d1.transducer_draft or 0.0
    n1 = min(s.shape[0] for s in sv1s)
    n2 = min(s.shape[0] for s in sv2s)
    sv1 = np.concatenate([s[:n1] for s in sv1s], axis=1)
    sv2 = np.concatenate([s[:n2] for s in sv2s], axis=1)
    heave = np.concatenate(heaves)
    return sv1, sv2, range1[:n1], range2[:n2], pulse, freq, draft, heave


def prepare_group(stems):
    sv1, sv2, range1, range2, pulse, freq, draft, heave = load_group(stems)

    # Algorithmic label on the full 38 kHz range (the only channel that reaches
    # the seabed in these deep transects).
    bottom_range, _, _ = detect_bottom_ensemble_pipeline(
        sv1, range1, heave=heave, pulse_duration=pulse,
        frequency_hz=freq, transducer_draft=draft,
        threshold_log_sv=BT_THRESHOLD, detectors=("simple", "edge"))
    label_idx = bottom_to_sample_index(bottom_range, range1)

    top, bot = bottom_crop_bounds(label_idx, len(range1))
    range_crop = range1[top:bot]
    sv1_crop = sv1[top:bot]
    sv2_on1 = resample_200k_to_38k_grid(sv2, range2, range1)[top:bot]

    norm1 = normalize_echogram(sv1_crop, method="minmax", vmin=VMIN, vmax=VMAX,
                               fill_nan=0.0).astype(np.float32)
    norm2 = normalize_echogram(sv2_on1, method="minmax", vmin=VMIN, vmax=VMAX,
                               fill_nan=0.0).astype(np.float32)
    diff = (norm1 - norm2).astype(np.float32)
    sv_channels = np.stack([norm1, norm2, diff], axis=0)

    label_idx_crop = np.where(label_idx >= 0, label_idx - top, -1)
    label_mask = create_bottom_mask(len(range_crop), sv1_crop.shape[1],
                                     label_idx_crop, mode="below", line_thickness=3)
    return dict(sv1_crop=sv1_crop, sv_channels=sv_channels, range_crop=range_crop,
                label_mask=label_mask, label_idx_crop=label_idx_crop)


def compute_metrics(label_mask, prediction, range_crop):
    pred_mask = (prediction >= 0.5).astype(np.uint8)
    inter = np.logical_and(label_mask, pred_mask).sum()
    union = np.logical_or(label_mask, pred_mask).sum()
    iou = inter / union if union else float("nan")
    denom = label_mask.sum() + pred_mask.sum()
    dice = 2 * inter / denom if denom else float("nan")
    label_line = extract_bottom_line(label_mask, 0.5)
    # Robust Viterbi line for the prediction: suppresses the naive first-crossing
    # spikes into the water column that otherwise dominate the depth error.
    pred_line = extract_bottom_dp(prediction, smoothness=0.5)
    errs = bottom_depth_error(pred_line, label_line, range_crop)
    return dict(iou=iou, dice=dice,
                mae_m=float(np.mean(errs)) if errs.size else float("nan"),
                median_m=float(np.median(errs)) if errs.size else float("nan"),
                detect_frac=float((pred_mask.any(axis=0)).mean()),
                label_line=label_line, pred_line=pred_line)


def _overlay(ax, range_crop, label_line, pred_line):
    for line, color, lw in ((label_line, "cyan", 2.6), (pred_line, "red", 1.8)):
        v = line >= 0
        if v.any():
            ax.plot(np.where(v)[0],
                    range_crop[np.clip(line[v], 0, len(range_crop) - 1)],
                    color=color, linewidth=lw, alpha=0.9)


def plot_single(name, res, prediction, m, save_path):
    sv1, range_crop = res["sv1_crop"], res["range_crop"]
    n_range, n_pings = sv1.shape
    extent = [0, n_pings, range_crop[-1], range_crop[0]]
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    axes[0].imshow(sv1, aspect="auto", cmap="viridis", vmin=VMIN, vmax=VMAX,
                   origin="upper", extent=extent)
    _overlay(axes[0], range_crop, m["label_line"], m["pred_line"])
    axes[0].set_title("38 kHz calibrated Sv with bottom label and prediction")
    im = axes[1].imshow(prediction, aspect="auto", cmap="RdYlBu_r", vmin=0, vmax=1,
                        origin="upper", extent=extent)
    _overlay(axes[1], range_crop, m["label_line"], m["pred_line"])
    axes[1].set_title("U-Net bottom probability")
    style_colorbar(fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04),
                   label="P(bottom)")
    for ax in axes:
        ax.set_xlabel("Ping")
        ax.set_ylabel("Range (m)")
    leg = fig.legend(
        handles=[Line2D([0], [0], color="cyan", lw=2.6, label="Algorithmic label"),
                 Line2D([0], [0], color="red", lw=1.8, label="U-Net prediction")],
        title=(f"IoU {m['iou']:.2f}   Dice {m['dice']:.2f}   depth MAE "
               f"{m['mae_m']:.1f} m   pred coverage {m['detect_frac']*100:.0f}%"),
        loc="lower center", ncol=2, framealpha=0.9, bbox_to_anchor=(0.5, -0.06))
    leg.get_title().set_fontsize(14)
    fig.suptitle(f"{name}: bottom label vs U-Net prediction", y=1.02)
    savefig(fig, save_path)
    plt.close(fig)


def plot_summary_grid(items, save_path):
    n = len(items)
    ncols = 2
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(11 * ncols, 6 * nrows))
    axes_flat = np.atleast_1d(axes).ravel()
    for ax, (name, res, m) in zip(axes_flat, items):
        sv1, range_crop = res["sv1_crop"], res["range_crop"]
        extent = [0, sv1.shape[1], range_crop[-1], range_crop[0]]
        ax.imshow(sv1, aspect="auto", cmap="viridis", vmin=VMIN, vmax=VMAX,
                  origin="upper", extent=extent)
        _overlay(ax, range_crop, m["label_line"], m["pred_line"])
        ax.set_title(f"{name}\nIoU {m['iou']:.2f} | depth MAE {m['mae_m']:.1f} m | "
                     f"cover {m['detect_frac']*100:.0f}%", fontsize=14)
        ax.set_xlabel("Ping")
        ax.set_ylabel("Range (m)")
    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)
    fig.legend(handles=[Line2D([0], [0], color="cyan", lw=2.6, label="Algorithmic label"),
                        Line2D([0], [0], color="red", lw=1.8, label="U-Net prediction")],
               loc="lower center", ncol=2, framealpha=0.9, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Hake test-raw: bottom label vs U-Net prediction", y=1.005)
    fig.tight_layout(rect=[0, 0.02, 1, 0.99])
    savefig(fig, save_path)
    plt.close(fig)


def write_readme(items, n_groups):
    import datetime
    lines = "\n".join(f"  {name}: IoU {m['iou']:.2f}, depth MAE {m['mae_m']:.1f} m, "
                      f"coverage {m['detect_frac']*100:.0f}%"
                      for name, _, m in items)
    (OUT_DIR / "_README.md").write_text(f"""# Hake test-raw evaluation

evaluation  bottom-detection  U-Net  EK80  out-of-distribution

Algorithmic 38 kHz bottom label overlaid against the U-Net prediction on NOAA
Hake EK80 test-raw files, with below-region IoU/Dice and per-ping depth error.
Consecutive time-contiguous files are stitched along the ping axis so each
echogram is wide enough for native 512-ping inference windows (each raw file is
only ~214 pings, which would otherwise be stretched 2.4x). The seabed (375-670 m)
lies beyond the 200 kHz range (300 m), so the three-channel model runs
out-of-distribution and relies on 38 kHz alone. Checkpoint: the 1728-file
random-pool augmented model (baseline_random_split_2).

Source: code/data_visualization/eval_hake_test_raw.py
Created: {datetime.datetime.now():%Y-%m-%d %H:%M}

Evaluated {len(items)} stitched echograms:
{lines}
""")


def consecutive_groups(group_size):
    """Build groups of `group_size` consecutive (time-contiguous) raw stems from
    two seabed-bearing blocks: a shallow shelf transect (~375 m) and a deeper
    slope transect (~515-670 m), so the eval spans a range of seabed depths."""
    stems = [p.stem for p in sorted(RAW_DIR.glob("*.raw"))
             if not p.name.startswith("._")]
    shelf = stems[0:9]    # 15:22-16:05, seabed ~375 m
    slope = stems[56:65]  # from T204958, seabed ~515-670 m
    groups = []
    for block in (shelf, slope):
        groups += [block[i:i + group_size]
                   for i in range(0, len(block), group_size)
                   if len(block[i:i + group_size]) == group_size]
    return groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group-size", type=int, default=3,
                    help="consecutive files to stitch per echogram (3 ~= 642 pings)")
    ap.add_argument("--n-groups", type=int, default=6)
    ap.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    ap.add_argument("--device", default=None)
    ap.add_argument("--groups", nargs="*", action="append", default=None,
                    help="explicit group(s) of stems; repeat the flag per group")
    args = ap.parse_args()

    apply_thesis_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Checkpoint: {Path(args.checkpoint).parent.name}")
    model = load_model(args.checkpoint, device)

    groups = args.groups or consecutive_groups(args.group_size)[:args.n_groups]

    items = []
    for i, stems in enumerate(groups, 1):
        name = f"{stems[0]}+{len(stems)-1}more" if len(stems) > 1 else stems[0]
        print(f"[{i}/{len(groups)}] {name}  ({len(stems)} files)")
        try:
            res = prepare_group(stems)
            print(f"  stitched echogram: {res['sv_channels'].shape[2]} pings")
            pred = predict_full_echogram(model, res["sv_channels"], TARGET_SIZE,
                                         device, patch_width=PATCH_WIDTH, stride=STRIDE)
            m = compute_metrics(res["label_mask"], pred, res["range_crop"])
            print(f"  IoU {m['iou']:.3f}  Dice {m['dice']:.3f}  depth MAE "
                  f"{m['mae_m']:.1f} m  coverage {m['detect_frac']*100:.0f}%")
            plot_single(name, res, pred, m, OUT_DIR / f"{name}_eval.png")
            items.append((name, res, m))
        except Exception as e:
            import traceback
            print(f"  ERROR: {e}")
            traceback.print_exc()

    if items:
        plot_summary_grid(items, OUT_DIR / "summary_grid.png")
        write_readme(items, len(groups))
    print(f"\nDone. {len(items)} stitched echograms plotted into {OUT_DIR}")


if __name__ == "__main__":
    main()
