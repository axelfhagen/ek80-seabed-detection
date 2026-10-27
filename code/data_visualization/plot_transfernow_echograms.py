"""
Investigate and plot 200 kHz echograms from TransferNow raw files with bottom labels.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os
from raw_sensor_integration.data_loader import load_raw, list_channels
from raw_sensor_integration.bottom_labelling import detect_bottom_tvg, detect_bottom_dp
from project_paths import REPO_ROOT, DATA_ROOT

TRANSFER_DIRS = {
    "TransferNow-3": str(REPO_ROOT / "Datasets/extra_raw_files/TransferNow-RawFilesEK80_3"),
    "TransferNow-4": str(REPO_ROOT / "Datasets/extra_raw_files/TransferNow-RawFilesEK80_4"),
    "TransferNow-5": str(DATA_ROOT / "extra_raw/TransferNow-RawFilesEK80_5"),
    "TransferNow-6": str(DATA_ROOT / "extra_raw/TransferNow-RawFilesEK80_6"),
}

OUT_DIR = str(REPO_ROOT / "code" / "plots_transfernow")
os.makedirs(OUT_DIR, exist_ok=True)

TARGET_FREQ = 200000  # 200 kHz

for dir_label, dir_path in TRANSFER_DIRS.items():
    raw_files = sorted(Path(dir_path).glob("*.raw"))
    print(f"\n{'=' * 70}")
    print(f"{dir_label}: {dir_path}")
    print(f"  {len(raw_files)} raw files")
    if not raw_files:
        continue

    # Check channels on first file
    first_file = raw_files[0]
    print(f"\n  First file: {first_file.name}")
    print(f"  File size: {first_file.stat().st_size / 1e6:.1f} MB")

    try:
        channels = list_channels(str(first_file))
        print("  Available channels:")
        for ch in channels:
            print(
                f"    {ch['channel_id']}: {ch['frequency_hz'] / 1000:.0f} kHz, "
                f"{ch.get('sample_count', '?')} samples, "
                f"max_range={ch.get('max_range', '?')} m"
            )
    except Exception as e:
        print(f"  Could not list channels: {e}")
        continue

    # Check if 200 kHz is available
    has_200 = any(
        ch["frequency_hz"] and abs(ch["frequency_hz"] - TARGET_FREQ) < 10000 for ch in channels
    )
    if not has_200:
        print(
            f"  WARNING: No 200 kHz channel found. Available: "
            f"{[ch['frequency_hz'] / 1000 for ch in channels if ch['frequency_hz']]} kHz"
        )
        # Use nearest frequency
        available_freqs = [ch["frequency_hz"] for ch in channels if ch["frequency_hz"]]
        if not available_freqs:
            print("  No frequencies available, skipping.")
            continue
        nearest = min(available_freqs, key=lambda f: abs(f - TARGET_FREQ))
        print(f"  Using nearest frequency: {nearest / 1000:.0f} kHz")
        use_freq = nearest
    else:
        use_freq = TARGET_FREQ

    # Process every file (or sample if too many)
    files_to_plot = raw_files

    for raw_file in files_to_plot:
        fname = raw_file.stem
        print(f"\n  Processing {raw_file.name} ({raw_file.stat().st_size / 1e6:.1f} MB)...")

        try:
            data = load_raw(str(raw_file), frequency=use_freq)
        except Exception as e:
            print(f"    FAILED to load: {e}")
            continue

        sv_db = data.sv_db
        range_arr = data.range_array
        n_samples, n_pings = sv_db.shape

        # Metrics
        finite_mask = np.isfinite(sv_db)
        pct_finite = 100.0 * finite_mask.sum() / sv_db.size
        if finite_mask.any():
            sv_min = float(np.nanmin(sv_db))
            sv_max = float(np.nanmax(sv_db))
            sv_mean = float(np.nanmean(sv_db))
            sv_std = float(np.nanstd(sv_db))
        else:
            sv_min = sv_max = sv_mean = sv_std = float("nan")

        print(f"    Channel: {data.channel_id}")
        print(f"    Frequency: {data.frequency_hz / 1000:.0f} kHz")
        print(f"    Shape: ({n_samples}, {n_pings}) — {n_samples} samples x {n_pings} pings")
        print(f"    Range: 0 – {data.max_range:.1f} m")
        print(f"    Sample interval: {data.sample_interval * 1e6:.1f} µs")
        print(
            f"    Sv range: [{sv_min:.1f}, {sv_max:.1f}] dB, mean={sv_mean:.1f}, std={sv_std:.1f}"
        )
        print(f"    Finite data: {pct_finite:.1f}%")
        print(f"    Calibrated: {data.calibrated}")
        if data.heave is not None:
            print(
                f"    Heave: mean={np.nanmean(data.heave):.3f} m, "
                f"range=[{np.nanmin(data.heave):.3f}, {np.nanmax(data.heave):.3f}]"
            )
        if data.gps is not None:
            print(
                f"    GPS: {len(data.gps)} fixes, "
                f"lat=[{data.gps['lat'].min():.4f}, {data.gps['lat'].max():.4f}], "
                f"lon=[{data.gps['lon'].min():.4f}, {data.gps['lon'].max():.4f}]"
            )

        # Bottom detection
        try:
            tvg_bottom = detect_bottom_tvg(sv_db, range_arr, minimum_range=3.0)
            dp_bottom = detect_bottom_dp(sv_db, range_arr, smoothness=1.0, minimum_range=3.0)

            valid_tvg = np.isfinite(tvg_bottom)
            valid_dp = np.isfinite(dp_bottom)
            print(
                f"    TVG bottom: {valid_tvg.sum()}/{n_pings} valid, "
                f"mean={np.nanmean(tvg_bottom):.1f} m, "
                f"range=[{np.nanmin(tvg_bottom[valid_tvg]) if valid_tvg.any() else float('nan'):.1f}, "
                f"{np.nanmax(tvg_bottom[valid_tvg]) if valid_tvg.any() else float('nan'):.1f}]"
            )
            print(
                f"    DP bottom:  {valid_dp.sum()}/{n_pings} valid, "
                f"mean={np.nanmean(dp_bottom):.1f} m, "
                f"range=[{np.nanmin(dp_bottom[valid_dp]) if valid_dp.any() else float('nan'):.1f}, "
                f"{np.nanmax(dp_bottom[valid_dp]) if valid_dp.any() else float('nan'):.1f}]"
            )
        except Exception as e:
            print(f"    Bottom detection failed: {e}")
            tvg_bottom = dp_bottom = None

        # Plot
        fig, ax = plt.subplots(figsize=(max(12, n_pings * 0.015), 7))

        vmin, vmax = data.default_vmin_vmax
        extent = [0, n_pings, data.max_range, 0]
        im = ax.imshow(
            sv_db,
            aspect="auto",
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            extent=extent,
            origin="upper",
        )
        plt.colorbar(im, ax=ax, label="Power (dB)")

        if dp_bottom is not None:
            ping_idx = np.arange(n_pings) + 0.5
            valid = np.isfinite(dp_bottom)
            ax.plot(
                ping_idx[valid], dp_bottom[valid], "r-", linewidth=1.0, alpha=0.9, label="DP bottom"
            )
        if tvg_bottom is not None:
            valid = np.isfinite(tvg_bottom)
            ax.plot(
                ping_idx[valid],
                tvg_bottom[valid],
                "c--",
                linewidth=0.7,
                alpha=0.7,
                label="TVG bottom",
            )

        ax.set_xlabel("Ping index")
        ax.set_ylabel("Range (m)")
        ax.set_title(
            f"{dir_label} / {fname}\n"
            f"{data.frequency_hz / 1000:.0f} kHz | "
            f"{n_samples}×{n_pings} | "
            f"range: 0–{data.max_range:.0f} m"
        )
        ax.legend(loc="lower right", fontsize=8)

        plt.tight_layout()
        out_path = os.path.join(OUT_DIR, f"{dir_label}_{fname}.png")
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"    Saved: {out_path}")

print(f"\n{'=' * 70}")
print(f"All plots saved to {OUT_DIR}")
