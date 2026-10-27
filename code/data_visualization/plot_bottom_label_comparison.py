"""Compare CRIMAC ensemble vs echopype basic bottom-detection labels.

For each sampled echopype zarr (Sv-only), runs both pipelines and plots
the 38 kHz echogram with both bottom lines overlaid. Postprocessing matches
what was applied when building the existing labeled zarrs:

- CRIMAC ensemble: median of detect_bottom_simple + detect_bottom_edge,
  minimum_range=10 m, offset=0.5, threshold='auto', no extra smoothing
  (matches code/raw_sensor_integration/zarr_dataset.py::_run_bottom_detection).

- echopype basic: detect_seafloor(method='basic', threshold=(-40,-20),
  offset_m=0.3, bin_skip_from_surface=100), 38 kHz channel
  (matches notebooks/14_echopype_pipeline.ipynb DETECTION_PARAMS).

Both bottoms are converted to range from transducer and overlaid on the
Sv echogram. Output PNGs go to Figures/bottom_label_comparison/.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np

_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root / "code") not in sys.path:
    sys.path.insert(0, str(_repo_root / "code"))

ZARR_DIR = _repo_root / "Datasets" / "echopype_zarr"
OUT_DIR = _repo_root / "Figures" / "bottom_label_comparison"

# Detection params (mirroring existing labeled-zarr generators)
CRIM_PARAMS = dict(minimum_range=10.0, threshold_log_sv="auto", offset=0.5)
EP_PARAMS = dict(threshold=(-40.0, -20.0), offset_m=0.3, bin_skip_from_surface=100)
FREQ_HZ = 38_000
PULSE_DURATION = 0.001024  # standard 1 ms EK80 pulse


def run_crimac_ensemble(
    sv_2d: np.ndarray, ranges: np.ndarray, heave: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (simple, edge, ensemble_median) bottom ranges in meters."""
    from raw_sensor_integration.bottom_labelling import (
        detect_bottom_simple,
        detect_bottom_edge,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        simple = detect_bottom_simple(
            sv_2d,
            ranges,
            heave,
            PULSE_DURATION,
            minimum_range=CRIM_PARAMS["minimum_range"],
            threshold_log_sv=CRIM_PARAMS["threshold_log_sv"],
            offset=CRIM_PARAMS["offset"],
        )
        edge = detect_bottom_edge(
            sv_2d,
            ranges,
            heave,
            PULSE_DURATION,
            frequency_hz=FREQ_HZ,
            minimum_range=CRIM_PARAMS["minimum_range"],
            threshold_log_sv=CRIM_PARAMS["threshold_log_sv"],
            offset=CRIM_PARAMS["offset"],
        )
    ensemble = np.nanmedian(np.stack([simple, edge]), axis=0)
    return simple, edge, ensemble


def run_echopype_basic(ds, channel_id: str) -> np.ndarray:
    """Run echopype detect_seafloor(basic). Returns bottom depth in meters
    (range from transducer; matches the labeled-zarr convention with
    DEPTH_OFFSET=0)."""
    import echopype as ep
    from echopype.mask import detect_seafloor

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ds_d = ep.consolidate.add_depth(ds)
        params = {"var_name": "Sv", "channel": channel_id, **EP_PARAMS}
        bd = detect_seafloor(ds_d, method="basic", params=params)
    return bd.values.astype(np.float64)


def load_38k_echogram(zarr_path: Path):
    """Return (sv_2d, ranges, heave, channel_id, ds, ch_idx).

    sv_2d : (n_range, n_pings) float — Sv in dB, NaN-range rows trimmed.
    ranges: (n_range,)        float — meters from transducer.
    heave : (n_pings,)        float — meters.
    """
    import xarray as xr

    ds = xr.open_zarr(zarr_path, consolidated=False)
    freqs = ds.frequency_nominal.values
    ch_idx = int(np.argmin(np.abs(freqs - FREQ_HZ)))
    channel_id = str(ds.channel.values[ch_idx])

    sv = ds.Sv.isel(channel=ch_idx).values.T  # (range, ping)
    er = ds.echo_range.isel(channel=ch_idx).values  # (ping, range)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ranges = np.nanmean(er, axis=0)
    valid_r = np.isfinite(ranges)
    ranges = ranges[valid_r]
    sv = sv[valid_r]

    if "heave" in ds:
        heave = ds.heave.values.astype(np.float64)
    else:
        heave = np.zeros(sv.shape[1])

    return sv, ranges, heave, channel_id, ds, ch_idx


def plot_one(zarr_path: Path, out_path: Path) -> dict:
    """Build the comparison figure for one file. Returns summary stats."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sv, ranges, heave, channel_id, ds, ch_idx = load_38k_echogram(zarr_path)
    n_range, n_pings = sv.shape

    # Run both pipelines
    crim_simple, crim_edge, crim_ens = run_crimac_ensemble(sv, ranges, heave)
    ep_bottom = run_echopype_basic(ds, channel_id)
    if len(ep_bottom) != n_pings:
        # Align on min length defensively
        m = min(len(ep_bottom), n_pings)
        ep_bottom = ep_bottom[:m]

    pings = np.arange(n_pings)

    # ── Stats ────────────────────────────────────────────────────────────
    both = np.isfinite(crim_ens) & np.isfinite(ep_bottom)
    diff = crim_ens - ep_bottom
    median_diff = float(np.nanmedian(diff[both])) if both.any() else float("nan")
    mad = float(np.median(np.abs(diff[both] - median_diff))) if both.any() else float("nan")
    within_1m = float(np.mean(np.abs(diff[both]) <= 1.0)) * 100 if both.any() else float("nan")
    crim_valid_pct = float(np.isfinite(crim_ens).mean()) * 100
    ep_valid_pct = float(np.isfinite(ep_bottom).mean()) * 100

    # ── Figure ───────────────────────────────────────────────────────────
    fig, (ax_e, ax_d) = plt.subplots(
        2,
        1,
        figsize=(14, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.04},
        constrained_layout=True,
    )

    extent = [0, n_pings, float(ranges[-1]), float(ranges[0])]
    ax_e.imshow(
        sv,
        aspect="auto",
        origin="upper",
        extent=extent,
        vmin=-80,
        vmax=-34,
        cmap="viridis",
        interpolation="nearest",
    )
    ax_e.plot(
        pings,
        crim_ens,
        color="#ff3b30",
        lw=1.0,
        label=f"CRIMAC ensemble ({crim_valid_pct:.0f}% valid)",
    )
    ax_e.plot(
        pings,
        ep_bottom,
        color="#ffd60a",
        lw=1.0,
        label=f"echopype basic ({ep_valid_pct:.0f}% valid)",
    )
    ax_e.set_ylabel("Range (m)")
    ax_e.set_xlim(0, n_pings)
    ax_e.set_ylim(float(ranges[-1]), float(ranges[0]))
    ax_e.legend(loc="upper right", framealpha=0.85, fontsize=9)

    title = (
        f"{zarr_path.stem}  —  38 kHz  —  median(crim − ep) = "
        f"{median_diff:+.2f} m  (MAD {mad:.2f} m,  |Δ|≤1 m: {within_1m:.0f}%)"
    )
    ax_e.set_title(title, fontsize=10)

    ax_d.axhline(0.0, color="0.5", lw=0.6)
    ax_d.plot(pings, diff, color="0.15", lw=0.6)
    ax_d.set_ylabel("Δ bottom (m)\ncrim − ep")
    ax_d.set_xlabel("Ping")
    # Clip Y axis to interesting range — but show outliers as clipped
    finite = diff[np.isfinite(diff)]
    if finite.size:
        lo, hi = np.nanpercentile(finite, [2, 98])
        pad = max(2.0, 0.3 * (hi - lo))
        ax_d.set_ylim(lo - pad, hi + pad)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)

    ds.close()
    return {
        "file": zarr_path.name,
        "n_pings": n_pings,
        "crim_valid_pct": crim_valid_pct,
        "ep_valid_pct": ep_valid_pct,
        "median_diff_m": median_diff,
        "mad_m": mad,
        "within_1m_pct": within_1m,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=30, help="number of files to plot")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--zarr-dir", type=Path, default=ZARR_DIR)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = p.parse_args()

    zarr_paths = sorted(args.zarr_dir.glob("*.zarr"))
    if not zarr_paths:
        print(f"No zarrs in {args.zarr_dir}", file=sys.stderr)
        sys.exit(1)

    rng = np.random.default_rng(args.seed)
    pick = rng.choice(len(zarr_paths), size=min(args.n, len(zarr_paths)), replace=False)
    pick = sorted(pick.tolist())
    selected = [zarr_paths[i] for i in pick]
    print(f"Plotting {len(selected)} files into {args.out_dir}")

    rows = []
    for i, zp in enumerate(selected):
        out_path = args.out_dir / f"{zp.stem}.png"
        try:
            stats = plot_one(zp, out_path)
            rows.append(stats)
            print(
                f"[{i + 1:>2}/{len(selected)}] {zp.stem}: "
                f"crim {stats['crim_valid_pct']:.0f}%, "
                f"ep {stats['ep_valid_pct']:.0f}%, "
                f"Δmedian {stats['median_diff_m']:+.2f} m, "
                f"|Δ|≤1m {stats['within_1m_pct']:.0f}%"
            )
        except Exception as exc:
            print(f"[{i + 1:>2}/{len(selected)}] {zp.stem}: FAILED — {exc}")

    if rows:
        import csv

        summary_path = args.out_dir / "summary.csv"
        with open(summary_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
