"""Diagnose why the U-Net fails on the basto-electric / 'ved kai' raw recordings.

Produces, per raw file:
  - input channels (38 kHz, 200 kHz, freq-diff) as the model sees them
  - probability map under the two normalisation regimes
    (per-file percentile, as in infer_test_raw, vs the training fixed [-80,-34] clip)
  - Grad-CAM on a central 512-wide patch
  - a dB-distribution panel comparing raw power to the training Sv window

Output: Figures/basto_failure_analysis/
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "code"))
sys.path.insert(0, str(PROJECT_ROOT / "code" / "data_visualization"))

from raw_sensor_integration.data_loader import load_raw, load_zarr
from raw_sensor_integration.preprocessing import normalize_echogram
from deep_learning.unet_model import predict_full_echogram
from deep_learning.evaluate import extract_bottom_line

import gen_gradcam_unet_v3 as gv

FREQ1 = 38_000
FREQ2 = 200_000
TARGET_SIZE = (512, 512)
TRAIN_VMIN, TRAIN_VMAX = -80.0, -34.0

RAW_DIR = PROJECT_ROOT / "Datasets" / "extra_raw_files" / "basto1"
ZARR_DIR = PROJECT_ROOT / "Datasets" / "echopype_zarr"
OUTPUT_DIR = PROJECT_ROOT / "Figures" / "basto_failure_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = gv.DEVICE


def build_channels(sv1, sv2, mode):
    """Return (3,H,W) channels. mode='percentile' (infer_test_raw) or 'fixed' (training)."""
    if mode == "percentile":
        finite = np.concatenate([sv1[np.isfinite(sv1)], sv2[np.isfinite(sv2)]])
        vmin = float(np.percentile(finite, 2))
        vmax = float(np.percentile(finite, 98))
    else:
        vmin, vmax = TRAIN_VMIN, TRAIN_VMAX
    n1 = normalize_echogram(sv1, method="minmax", vmin=vmin, vmax=vmax, fill_nan=0.0).astype(
        np.float32
    )
    n2 = normalize_echogram(sv2, method="minmax", vmin=vmin, vmax=vmax, fill_nan=0.0).astype(
        np.float32
    )
    diff = (n1 - n2).astype(np.float32)
    return np.stack([n1, n2, diff], axis=0), (vmin, vmax)


def run_inference(model, channels):
    pred = predict_full_echogram(model, channels, TARGET_SIZE, DEVICE, patch_width=512, stride=256)
    bottom = extract_bottom_line(pred, threshold=0.5)
    n_det = int(np.sum(bottom >= 0))
    return pred, bottom, n_det


def central_patch(channels):
    """Extract a 512x512 patch from the centre, resized to model input."""
    _, h, w = channels.shape
    c = w // 2
    start = max(0, c - 256)
    patch = channels[:, :, start : start + 512]
    if patch.shape[2] < 512:
        patch = np.pad(patch, ((0, 0), (0, 0), (0, 512 - patch.shape[2])))
    return gv.resize_to_input(patch)


def load_train_sv_sample(max_files=3):
    """Pool 38 kHz Sv values from a few training zarr files for the histogram."""
    vals = []
    zfiles = sorted(ZARR_DIR.glob("*.zarr"))[:max_files]
    for z in zfiles:
        try:
            d = load_zarr(str(z), frequency=FREQ1, ping_slice=slice(0, 300))
            v = d.sv_db[np.isfinite(d.sv_db)]
            vals.append(v[::7])
        except Exception as e:
            print(f"  zarr {z.name} skipped: {e}")
    return np.concatenate(vals) if vals else np.array([])


def fig_channels(stem, channels, vrange):
    titles = ["38 kHz (normalised)", "200 kHz (normalised)", "38 - 200 kHz diff"]
    cmaps = ["viridis", "viridis", "coolwarm"]
    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    for ax, ch, t, cm in zip(axes, channels, titles, cmaps):
        vmin, vmax = (0, 1) if cm == "viridis" else (-1, 1)
        im = ax.imshow(ch, aspect="auto", cmap=cm, vmin=vmin, vmax=vmax)
        ax.set_title(t)
        ax.set_xlabel("Ping")
        ax.set_ylabel("Range sample")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(
        f"Model input channels: {stem}  (percentile clip {vrange[0]:.0f}..{vrange[1]:.0f} dB)",
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{stem}_01_input_channels.png", dpi=140, facecolor="white")
    plt.close(fig)


def fig_probmaps(stem, sv2, range_arr, results):
    extent = [0, sv2.shape[1], range_arr[-1], range_arr[0]]
    fig, axes = plt.subplots(1, 3, figsize=(22, 7))

    axes[0].imshow(
        sv2,
        aspect="auto",
        cmap="viridis",
        extent=extent,
        vmin=np.nanpercentile(sv2, 2),
        vmax=np.nanpercentile(sv2, 98),
    )
    axes[0].set_title("200 kHz echogram (dB)")
    axes[0].set_ylabel("Range (m)")

    for ax, (label, pred, bottom, n_det) in zip(axes[1:], results):
        im = ax.imshow(pred, aspect="auto", cmap="RdYlBu_r", vmin=0, vmax=1, extent=extent)
        valid = bottom >= 0
        if valid.any():
            pp = np.where(valid)[0]
            dd = range_arr[np.clip(bottom[valid], 0, len(range_arr) - 1)]
            ax.plot(pp, dd, color="black", lw=1.0)
        ax.set_title(f"p(seabed): {label}\nbottom in {n_det}/{len(bottom)} pings")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="p(bottom)")
    for ax in axes:
        ax.set_xlabel("Ping")
    fig.suptitle(f"Seabed probability map: {stem}", fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{stem}_02_probability_maps.png", dpi=140, facecolor="white")
    plt.close(fig)


def fig_gradcam(stem, model, channels_fixed):
    patch = central_patch(channels_fixed)
    _, probs = gv.predict_logits(model, patch)
    pred_mask = (probs > 0.5).astype(np.uint8)
    layer = gv.primary_layer(model)
    # Grad-CAM needs a non-empty target; fall back to top-activation region.
    target = pred_mask if pred_mask.sum() > 20 else (probs > probs.max() * 0.6).astype(np.uint8)
    cam = gv.gradcam_single(model, [layer], patch, target)
    echo = gv.freq_to_display(patch[0] * 2 - 1) if patch[0].max() <= 1 else patch[0]
    echo = np.clip(patch[0], 0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    axes[0].imshow(echo, cmap="viridis", aspect="auto")
    axes[0].set_title("38 kHz patch (fixed clip)")
    im = axes[1].imshow(probs, cmap="magma", vmin=0, vmax=1, aspect="auto")
    axes[1].set_title(f"p(seabed)  max={probs.max():.2f}")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    axes[2].imshow(echo, cmap="gray", aspect="auto")
    cs = axes[2].contourf(
        cam, levels=np.linspace(0.2, 0.9, 6), cmap="jet", alpha=0.55, extend="max"
    )
    axes[2].set_title("Grad-CAM attention")
    fig.colorbar(cs, ax=axes[2], fraction=0.046, pad=0.04, label="importance")
    for ax in axes:
        ax.set_xlabel("Ping (patch)")
        ax.set_ylabel("Range (patch)")
    fig.suptitle(f"Grad-CAM on central patch: {stem}", fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{stem}_03_gradcam.png", dpi=140, facecolor="white")
    plt.close(fig)


def fig_distribution(stem, sv1, sv2, train_sv):
    fig, ax = plt.subplots(figsize=(11, 6))
    bins = np.linspace(-100, 0, 101)
    ax.hist(
        sv1[np.isfinite(sv1)].ravel(),
        bins=bins,
        density=True,
        alpha=0.5,
        color="C0",
        label="raw 38 kHz power",
    )
    ax.hist(
        sv2[np.isfinite(sv2)].ravel(),
        bins=bins,
        density=True,
        alpha=0.5,
        color="C1",
        label="raw 200 kHz power",
    )
    if train_sv.size:
        ax.hist(
            train_sv,
            bins=bins,
            density=True,
            alpha=0.5,
            color="C2",
            label="training Sv 38 kHz (calibrated)",
        )
    ax.axvspan(
        TRAIN_VMIN,
        TRAIN_VMAX,
        color="green",
        alpha=0.12,
        label=f"training clip [{TRAIN_VMIN:.0f},{TRAIN_VMAX:.0f}]",
    )
    ax.axvline(TRAIN_VMIN, color="green", ls="--", lw=1)
    ax.axvline(TRAIN_VMAX, color="green", ls="--", lw=1)
    ax.set_xlabel("dB")
    ax.set_ylabel("density")
    ax.set_title(f"dB distribution vs training window: {stem}")
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{stem}_04_db_distribution.png", dpi=140, facecolor="white")
    plt.close(fig)


def main():
    print(f"Device: {DEVICE}")
    raw_files = sorted(RAW_DIR.glob("*.raw"))
    raw_files = [f for f in raw_files if not f.name.startswith("._")]
    print(f"Found {len(raw_files)} basto raw files")

    model = gv.load_model()
    train_sv = load_train_sv_sample()
    print(f"Training Sv sample: {train_sv.size} values")

    summary = []
    for rf in raw_files:
        stem = rf.stem
        print(f"\n=== {stem} ===")
        try:
            d1 = load_raw(str(rf), frequency=FREQ1, max_pings=2000)
            d2 = load_raw(str(rf), frequency=FREQ2, max_pings=2000)
        except Exception as e:
            print(f"  load failed: {e}")
            continue
        mr = min(d1.n_samples, d2.n_samples)
        mp = min(d1.sv_db.shape[1], d2.sv_db.shape[1])
        sv1, sv2 = d1.sv_db[:mr, :mp], d2.sv_db[:mr, :mp]
        range_arr = d1.range_array[:mr]
        res_m = float(range_arr[1] - range_arr[0]) if len(range_arr) > 1 else np.nan
        f1 = sv1[np.isfinite(sv1)]
        in_win = float(np.mean((f1 >= TRAIN_VMIN) & (f1 <= TRAIN_VMAX)))
        print(f"  shape {sv1.shape}, max range {range_arr[-1]:.1f} m, res {res_m:.3f} m/sample")
        print(
            f"  38kHz dB p2/p50/p98: "
            f"{np.percentile(f1, [2, 50, 98]).round(1)}, "
            f"frac in [-80,-34]={in_win:.2f}"
        )

        ch_pct, vr_pct = build_channels(sv1, sv2, "percentile")
        ch_fix, vr_fix = build_channels(sv1, sv2, "fixed")

        pred_p, bot_p, nd_p = run_inference(model, ch_pct)
        pred_f, bot_f, nd_f = run_inference(model, ch_fix)
        print(
            f"  bottom detected: percentile={nd_p}/{len(bot_p)}, "
            f"fixed-clip={nd_f}/{len(bot_f)}, "
            f"pmax pct={pred_p.max():.2f} fix={pred_f.max():.2f}"
        )

        fig_channels(stem, ch_pct, vr_pct)
        fig_probmaps(
            stem,
            sv2,
            range_arr,
            [
                ("percentile clip", pred_p, bot_p, nd_p),
                ("fixed [-80,-34] clip", pred_f, bot_f, nd_f),
            ],
        )
        fig_gradcam(stem, model, ch_fix)
        fig_distribution(stem, sv1, sv2, train_sv)

        summary.append(
            {
                "file": stem,
                "max_range_m": range_arr[-1],
                "res_m": res_m,
                "frac_in_train_window": in_win,
                "p2_38": np.percentile(f1, 2),
                "p98_38": np.percentile(f1, 98),
                "n_det_percentile": nd_p,
                "n_det_fixed": nd_f,
                "pmax_percentile": float(pred_p.max()),
                "pmax_fixed": float(pred_f.max()),
            }
        )

    import pandas as pd

    df = pd.DataFrame(summary)
    df.to_csv(OUTPUT_DIR / "summary.csv", index=False)
    print("\n=== SUMMARY ===")
    print(df.to_string(index=False))
    print(f"\nFigures + summary.csv in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
