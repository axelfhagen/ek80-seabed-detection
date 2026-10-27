"""Generate presentation plots: 2x2 prediction grids + ensemble uncertainty band."""

import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from deep_learning.unet_model import create_unet, predict_full_echogram
from deep_learning.evaluate import load_file_data, extract_bottom_line

# ── Configuration ────────────────────────────────────────────────────────────

TOP5_CHECKPOINTS = [
    PROJECT_ROOT / "code/checkpoints/exp_20260420_080950_finetune_stages",
    PROJECT_ROOT / "code/checkpoints/exp_20260420_091726_full_finetune",
    PROJECT_ROOT / "code/checkpoints/exp_20260423_165114_loss_04_focal_aggressive",
    PROJECT_ROOT / "code/checkpoints/exp_20260420_082458_cosine_gentle",
    PROJECT_ROOT / "code/checkpoints/exp_20260423_230543_ds_pctile",
]

TOP5_NAMES = [
    "Stages freeze",
    "Full fine-tune",
    "Focal loss",
    "Cosine schedule",
    "Percentile norm",
]

ZARR_DIR = PROJECT_ROOT / "Datasets/echopype_zarr"
OUTPUT_DIR = PROJECT_ROOT / "Figures/presentation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FREQ1 = 38_000
FREQ2 = 200_000
TARGET_SIZE = (512, 512)
PATCH_WIDTH = 512
STRIDE = 128
VMIN, VMAX = -80.0, -34.0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SEED = 42


def load_model(checkpoint_dir):
    model = create_unet(encoder_name="resnet34", in_channels=3)
    ckpt_path = checkpoint_dir / "unet_resnet34_best.pth"
    state = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.to(DEVICE).eval()
    return model


def get_zarr_files(n=20, seed=SEED):
    all_zarrs = sorted(ZARR_DIR.glob("*.zarr"))
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(all_zarrs), size=min(n, len(all_zarrs)), replace=False)
    return [all_zarrs[i] for i in sorted(chosen)]


# ── Part 1: 2x2 prediction grids (best model on 20 files) ───────────────────

def plot_prediction_grid(model, zarr_paths, out_path, fig_idx):
    """2x2 grid: each cell shows echogram with prediction overlay + GT bottom."""
    fig, axes = plt.subplots(2, 2, figsize=(18, 10))
    axes = axes.ravel()

    for i, zp in enumerate(zarr_paths):
        ax = axes[i]
        fname = zp.name.replace(".zarr", "")

        try:
            sv_ch, gt_mask, gt_bottom_idx, range_arr = load_file_data(
                zp, FREQ1, FREQ2, vmin=VMIN, vmax=VMAX)
        except Exception as e:
            ax.text(0.5, 0.5, f"Load error:\n{e}", transform=ax.transAxes,
                    ha='center', va='center', fontsize=8)
            ax.set_title(fname, fontsize=9)
            continue

        prediction = predict_full_echogram(
            model, sv_ch, TARGET_SIZE, DEVICE,
            patch_width=PATCH_WIDTH, stride=STRIDE)

        echogram = sv_ch[0]
        n_range, n_pings = echogram.shape
        extent = [0, n_pings, range_arr[-1], range_arr[0]]

        ax.imshow(echogram, aspect='auto', cmap='viridis', vmin=0, vmax=1,
                  origin='upper', extent=extent)

        ax.contour(prediction, levels=[0.5], colors='#ff4444', linewidths=1.2,
                   extent=extent, origin='upper')

        gt_line = extract_bottom_line(gt_mask, threshold=0.5)
        valid = gt_line >= 0
        pings = np.arange(n_pings)
        if valid.any():
            gt_depths = range_arr[np.clip(gt_line[valid], 0, len(range_arr)-1)]
            ax.plot(pings[valid], gt_depths, color='#00ccff', linewidth=0.8,
                    alpha=0.7, label='Reference bottom' if i == 0 else None)

        pred_line = extract_bottom_line(prediction, threshold=0.5)
        pred_valid = pred_line >= 0
        if pred_valid.any():
            pred_depths = range_arr[np.clip(pred_line[pred_valid], 0, len(range_arr)-1)]
            ax.plot(pings[pred_valid], pred_depths, color='#ff4444', linewidth=0.8,
                    alpha=0.7, linestyle='--', label='Pred bottom' if i == 0 else None)

        ax.set_title(fname, fontsize=10, fontweight='bold')
        ax.set_xlabel('Ping', fontsize=8)
        ax.set_ylabel('Range (m)', fontsize=8)
        ax.tick_params(labelsize=7)

    axes[0].legend(loc='lower right', fontsize=7, framealpha=0.8)
    fig.suptitle(f'Seabed Prediction — Best Model (batch {fig_idx+1}/5)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved {out_path}")


# ── Part 2: Ensemble uncertainty ─────────────────────────────────────────────

def ensemble_predict(models, sv_channels, target_size, device,
                     patch_width=512, stride=128):
    """Run all models and stack predictions."""
    preds = []
    for model in models:
        pred = predict_full_echogram(
            model, sv_channels, target_size, device,
            patch_width=patch_width, stride=stride)
        preds.append(pred)
    return np.stack(preds, axis=0)


def plot_ensemble_uncertainty(models, model_names, zarr_path, out_path):
    """Detailed uncertainty plot from model ensemble."""
    sv_ch, gt_mask, gt_bottom_idx, range_arr = load_file_data(
        zarr_path, FREQ1, FREQ2, vmin=VMIN, vmax=VMAX)

    all_preds = ensemble_predict(
        models, sv_ch, TARGET_SIZE, DEVICE,
        patch_width=PATCH_WIDTH, stride=STRIDE)

    mean_pred = all_preds.mean(axis=0)
    std_pred = all_preds.std(axis=0)

    n_range, n_pings = mean_pred.shape
    extent = [0, n_pings, range_arr[-1], range_arr[0]]
    echogram = sv_ch[0]
    pings = np.arange(n_pings)

    fig, axes = plt.subplots(3, 2, figsize=(20, 14))

    # (0,0) Echogram
    ax = axes[0, 0]
    ax.imshow(echogram, aspect='auto', cmap='viridis', vmin=0, vmax=1,
              origin='upper', extent=extent)
    ax.set_title('Echogram (38 kHz)', fontsize=11)
    ax.set_ylabel('Range (m)')

    # (0,1) Mean prediction
    ax = axes[0, 1]
    im = ax.imshow(mean_pred, aspect='auto', cmap='RdYlBu_r', vmin=0, vmax=1,
                   origin='upper', extent=extent)
    ax.set_title('Ensemble mean probability', fontsize=11)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # (1,0) Prediction std (uncertainty)
    ax = axes[1, 0]
    im = ax.imshow(std_pred, aspect='auto', cmap='inferno', vmin=0, vmax=std_pred.max(),
                   origin='upper', extent=extent)
    ax.set_title('Ensemble disagreement (std)', fontsize=11)
    ax.set_ylabel('Range (m)')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # (1,1) Echogram with uncertainty band overlay
    ax = axes[1, 1]
    ax.imshow(echogram, aspect='auto', cmap='viridis', vmin=0, vmax=1,
              origin='upper', extent=extent)

    gt_valid = gt_bottom_idx >= 0
    if gt_valid.any():
        gt_depths = range_arr[np.clip(gt_bottom_idx[gt_valid], 0, len(range_arr)-1)]
        ax.plot(pings[gt_valid], gt_depths, color='#00ccff', linewidth=1.0,
                alpha=0.8, label='Reference bottom')

    upper_depths = np.full(n_pings, np.nan)
    lower_depths = np.full(n_pings, np.nan)
    median_depths = np.full(n_pings, np.nan)

    for ping in range(n_pings):
        bottom_indices = []
        for k in range(len(models)):
            col = all_preds[k, :, ping]
            hits = np.where(col >= 0.5)[0]
            if len(hits) > 0:
                bottom_indices.append(hits[0])
        if bottom_indices:
            arr = np.array(bottom_indices)
            upper_depths[ping] = range_arr[np.clip(int(np.min(arr)), 0, len(range_arr)-1)]
            lower_depths[ping] = range_arr[np.clip(int(np.max(arr)), 0, len(range_arr)-1)]
            median_depths[ping] = range_arr[np.clip(int(np.median(arr)), 0, len(range_arr)-1)]

    valid_band = ~np.isnan(upper_depths) & ~np.isnan(lower_depths)
    ax.fill_between(pings[valid_band], upper_depths[valid_band], lower_depths[valid_band],
                    alpha=0.35, color='#ff4444', label='Ensemble spread')
    valid_med = ~np.isnan(median_depths)
    ax.plot(pings[valid_med], median_depths[valid_med], color='#ff4444',
            linewidth=1.0, label='Ensemble median')

    ax.set_title('Uncertainty band overlay', fontsize=11)
    ax.legend(loc='lower right', fontsize=8, framealpha=0.8)

    # (2,0) Per-model bottom lines
    ax = axes[2, 0]
    ax.imshow(echogram, aspect='auto', cmap='viridis', vmin=0, vmax=1,
              origin='upper', extent=extent)

    colors = ['#ff4444', '#ff8800', '#ffcc00', '#44cc44', '#4488ff']
    for k in range(len(models)):
        bl = extract_bottom_line(all_preds[k], threshold=0.5)
        bl_valid = bl >= 0
        if bl_valid.any():
            depths = range_arr[np.clip(bl[bl_valid], 0, len(range_arr)-1)]
            ax.plot(pings[bl_valid], depths, color=colors[k % len(colors)],
                    linewidth=0.8, alpha=0.8, label=model_names[k])

    if gt_valid.any():
        ax.plot(pings[gt_valid], gt_depths, color='#00ccff', linewidth=1.0,
                alpha=0.8, linestyle='--', label='Reference')

    ax.set_title('Individual model predictions', fontsize=11)
    ax.set_ylabel('Range (m)')
    ax.set_xlabel('Ping')
    ax.legend(loc='lower right', fontsize=7, framealpha=0.8, ncol=2)

    # (2,1) Band width histogram
    ax = axes[2, 1]
    band_width = np.abs(lower_depths - upper_depths)
    valid_bw = ~np.isnan(band_width)
    if valid_bw.any():
        bw = band_width[valid_bw]
        ax.hist(bw, bins=50, color='#ff6666', edgecolor='#cc0000', alpha=0.8)
        ax.axvline(np.median(bw), color='k', linestyle='--', linewidth=1.5,
                   label=f'Median: {np.median(bw):.2f} m')
        ax.axvline(np.mean(bw), color='#333333', linestyle=':', linewidth=1.5,
                   label=f'Mean: {np.mean(bw):.2f} m')
        ax.set_xlabel('Band width (m)')
        ax.set_ylabel('Count')
        ax.set_title('Bottom uncertainty width distribution', fontsize=11)
        ax.legend(fontsize=8)

    fname = zarr_path.name.replace('.zarr', '')
    fig.suptitle(f'Ensemble Uncertainty — {fname} ({len(models)} models)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved {out_path}")


# ── Part 3: Extra models for bagging ────────────────────────────────────────

def get_top_n_checkpoints(n=10):
    """Get top N checkpoint dirs from results.csv."""
    import csv
    results_csv = PROJECT_ROOT / "code/deep_learning/training/results.csv"
    rows = list(csv.DictReader(open(results_csv)))
    rows.sort(key=lambda r: float(r['eval_mean_iou']), reverse=True)

    seen = set()
    checkpoints = []
    names = []
    for r in rows[:n*2]:
        d = r['checkpoint_dir']
        if not d.startswith('/'):
            d = str(PROJECT_ROOT / d)
        p = Path(d)
        if p.name in seen:
            continue
        if (p / "unet_resnet34_best.pth").exists() and r['in_channels'] == '3':
            seen.add(p.name)
            checkpoints.append(p)
            iou = float(r['eval_mean_iou'])
            label = p.name.split('_', 3)[3] if p.name.count('_') >= 3 else p.name
            names.append(f"{label} (IoU={iou:.3f})")
            if len(checkpoints) >= n:
                break
    return checkpoints, names


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print(f"Device: {DEVICE}")
    print("Loading best model for prediction grids...")
    best_model = load_model(TOP5_CHECKPOINTS[0])

    zarr_files = get_zarr_files(n=20)
    print(f"Selected {len(zarr_files)} zarr files")

    for fig_idx in range(5):
        batch = zarr_files[fig_idx*4 : (fig_idx+1)*4]
        if len(batch) < 4:
            break
        out_path = OUTPUT_DIR / f"predictions_grid_{fig_idx+1}.png"
        print(f"\nPlotting grid {fig_idx+1}/5 ...")
        plot_prediction_grid(best_model, batch, out_path, fig_idx)

    print("\n\nLoading top 10 models for ensemble uncertainty...")
    ckpt_dirs, ckpt_names = get_top_n_checkpoints(n=10)
    print(f"Found {len(ckpt_dirs)} models")

    ensemble_models = []
    for d in ckpt_dirs:
        m = load_model(d)
        ensemble_models.append(m)
        print(f"  Loaded {d.name}")

    first_zarr = zarr_files[0]
    print(f"\nGenerating ensemble uncertainty plot for {first_zarr.name} ...")
    plot_ensemble_uncertainty(
        ensemble_models, ckpt_names, first_zarr,
        OUTPUT_DIR / "ensemble_uncertainty.png")

    print("\nDone! All plots saved to", OUTPUT_DIR)


if __name__ == "__main__":
    main()
