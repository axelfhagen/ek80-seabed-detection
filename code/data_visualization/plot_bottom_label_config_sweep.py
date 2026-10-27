"""Sweep bottom-labelling post-processing configs on 10 random echograms.

Output: Figures/bottom_label_config_sweep/

For each of 5 post-processing configurations (varying ensemble outlier
tolerance, sliding-median window, deviation threshold, optional second
pass, optional Savitzky-Golay smoothing) we run the CRIMAC ensemble
pipeline on the same 10 random 38 kHz zarr echograms and overlay the
resulting bottom trace as a red line. A combined overlay PNG plots all
five configs together on each echogram for direct comparison.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy.signal as sig

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from raw_sensor_integration.bottom_labelling import (  # noqa: E402
    detect_bottom_ensemble_pipeline,
    post_process_bottom,
)

ZARR_DIR = PROJECT_ROOT / "Datasets" / "echopype_zarr"
OUTPUT_DIR = PROJECT_ROOT / "Figures" / "bottom_label_config_sweep"

FREQ_HZ = 38_000
PULSE_DURATION = 0.001024


@dataclass
class Config:
    name: str
    description: str
    detectors: tuple = ("simple", "edge")
    ensemble_max_deviation: float = 5.0
    pp_window: int = 31
    pp_max_deviation_m: float = 10.0
    second_pass: Optional[dict] = None
    savgol: Optional[dict] = None
    minimum_range: float = 10.0
    offset: float = 0.5


CONFIGS = [
    Config(
        name="C1 baseline_loose",
        description=(
            "Current default. Two detectors, sliding median window 31 pings, "
            "loose 10 m deviation tolerance. Spikes mostly pass through."
        ),
        detectors=("simple", "edge"),
        ensemble_max_deviation=5.0,
        pp_window=31,
        pp_max_deviation_m=10.0,
    ),
    Config(
        name="C2 strict_local",
        description=(
            "Same detectors, but tight 2 m deviation threshold in the median "
            "post-process. Kills single-ping outliers; keeps fine detail."
        ),
        detectors=("simple", "edge"),
        ensemble_max_deviation=3.0,
        pp_window=31,
        pp_max_deviation_m=2.0,
    ),
    Config(
        name="C3 wider_window",
        description=(
            "Window grown to 61 pings, 3 m deviation. Catches short-burst "
            "spikes by averaging over more neighbours, may round edges."
        ),
        detectors=("simple", "edge"),
        ensemble_max_deviation=4.0,
        pp_window=61,
        pp_max_deviation_m=3.0,
    ),
    Config(
        name="C4 two_pass",
        description=(
            "Three detectors. First a wide+loose pass (window 51, dev 3 m) "
            "to kill bursts, then narrow+strict (window 21, dev 1.5 m) for "
            "remaining singletons."
        ),
        detectors=("simple", "edge", "gradient"),
        ensemble_max_deviation=5.0,
        pp_window=51,
        pp_max_deviation_m=3.0,
        second_pass=dict(window=21, max_deviation_m=1.5),
    ),
    Config(
        name="C5 median_plus_savgol",
        description=(
            "Median spike rejection (window 31, dev 3 m) followed by "
            "Savitzky-Golay smoothing (window 21, order 3). Smoothest "
            "trace; rounds sharp depth transitions."
        ),
        detectors=("simple", "edge", "gradient"),
        ensemble_max_deviation=4.0,
        pp_window=31,
        pp_max_deviation_m=3.0,
        savgol=dict(window=21, order=3),
    ),
]


def load_38k_echogram(zarr_path: Path):
    """Return (sv, ranges, heave) for the 38 kHz channel of one zarr."""
    import xarray as xr

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        ds = xr.open_zarr(zarr_path, consolidated=False)
        freqs = ds.frequency_nominal.values
        ch_idx = int(np.argmin(np.abs(freqs - FREQ_HZ)))

        sv = ds.Sv.isel(channel=ch_idx).values.T  # (range, ping)
        er = ds.echo_range.isel(channel=ch_idx).values  # (ping, range)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            ranges = np.nanmean(er, axis=0)
        valid_r = np.isfinite(ranges)
        ranges = ranges[valid_r]
        sv = sv[valid_r]

        heave = ds.heave.values.astype(np.float64) if "heave" in ds else np.zeros(sv.shape[1])
        ds.close()
    return sv, ranges, heave


def run_config(sv: np.ndarray, ranges: np.ndarray, heave: np.ndarray, cfg: Config) -> np.ndarray:
    """Run ensemble + custom post-processing for one config."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        bottom, _, _ = detect_bottom_ensemble_pipeline(
            sv,
            ranges,
            heave,
            pulse_duration=PULSE_DURATION,
            frequency_hz=FREQ_HZ,
            minimum_range=cfg.minimum_range,
            offset=cfg.offset,
            detectors=cfg.detectors,
            max_deviation=cfg.ensemble_max_deviation,
            post_process=True,
            pp_max_deviation_m=cfg.pp_max_deviation_m,
            pp_window=cfg.pp_window,
        )
        if cfg.second_pass:
            bottom, _ = post_process_bottom(
                bottom,
                max_deviation_m=cfg.second_pass["max_deviation_m"],
                window=cfg.second_pass["window"],
                fill_gaps=True,
            )
        if cfg.savgol:
            valid = ~np.isnan(bottom)
            if valid.sum() >= cfg.savgol["window"]:
                x = np.arange(len(bottom))
                filled = np.interp(x, x[valid], bottom[valid])
                bottom = sig.savgol_filter(
                    filled,
                    cfg.savgol["window"],
                    cfg.savgol["order"],
                )
    return bottom


def _trim_to_data(sv: np.ndarray, ranges: np.ndarray):
    """Drop trailing all-NaN range rows so the y-axis hugs real data."""
    valid_row = ~np.all(np.isnan(sv), axis=1)
    if not valid_row.any():
        return sv, ranges
    last = int(np.where(valid_row)[0].max())
    last = min(len(ranges) - 1, last + 5)  # keep a small margin
    return sv[: last + 1], ranges[: last + 1]


def style_axes(ax, sv, ranges, n_pings, title: str):
    sv_t, r_t = _trim_to_data(sv, ranges)
    extent = [0, n_pings, float(r_t[-1]), float(r_t[0])]
    ax.imshow(
        sv_t,
        aspect="auto",
        origin="upper",
        extent=extent,
        vmin=-80,
        vmax=-34,
        cmap="viridis",
        interpolation="nearest",
    )
    ax.set_ylim(float(r_t[-1]), float(r_t[0]))
    ax.set_xlim(0, n_pings)
    ax.set_title(title, fontsize=17)
    ax.tick_params(labelsize=14)


def plot_config_grid(cfg: Config, idx: int, echograms: list[dict], out_path: Path) -> None:
    """2x5 grid of echograms with red bottom line for one config."""
    n = len(echograms)
    n_cols, n_rows = 2, (n + 1) // 2
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(11.0 * n_cols, 3.8 * n_rows),
        constrained_layout=True,
    )
    axes = np.atleast_2d(axes)

    for k, e in enumerate(echograms):
        r, c = divmod(k, n_cols)
        ax = axes[r, c]
        style_axes(ax, e["sv"], e["ranges"], e["n_pings"], title=e["name"])
        bottom = e["traces"][cfg.name]
        ax.plot(np.arange(e["n_pings"]), bottom, color="#ff3b30", lw=1.8)
        if r == n_rows - 1:
            ax.set_xlabel("Ping", fontsize=15)
        if c == 0:
            ax.set_ylabel("Range (m)", fontsize=15)

    for k in range(n, n_rows * n_cols):
        r, c = divmod(k, n_cols)
        axes[r, c].axis("off")

    # Wrap the description so long single-line titles don't shrink the panels
    import textwrap

    desc_wrap = textwrap.fill(cfg.description, width=110)
    fig.suptitle(f"{cfg.name}\n{desc_wrap}", fontsize=18, y=1.04, linespacing=1.3)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_overlay_comparison(echograms: list[dict], out_path: Path) -> None:
    """10 echograms, each panel overlays all 5 config bottom lines."""
    n = len(echograms)
    n_cols, n_rows = 2, (n + 1) // 2
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(11.0 * n_cols, 4.2 * n_rows),
        constrained_layout=True,
    )
    axes = np.atleast_2d(axes)

    colors = ["#ff3b30", "#ffd60a", "#30d158", "#0a84ff", "#bf5af2"]
    config_names = [c.name for c in CONFIGS]

    for k, e in enumerate(echograms):
        r, c = divmod(k, n_cols)
        ax = axes[r, c]
        style_axes(ax, e["sv"], e["ranges"], e["n_pings"], title=e["name"])
        x = np.arange(e["n_pings"])
        for name, col in zip(config_names, colors):
            ax.plot(x, e["traces"][name], color=col, lw=1.2, label=name, alpha=0.85)
        if k == 0:
            ax.legend(loc="upper right", fontsize=12, framealpha=0.85, ncol=1)
        if r == n_rows - 1:
            ax.set_xlabel("Ping", fontsize=15)
        if c == 0:
            ax.set_ylabel("Range (m)", fontsize=15)

    for k in range(n, n_rows * n_cols):
        r, c = divmod(k, n_cols)
        axes[r, c].axis("off")

    fig.suptitle("Bottom-labelling config sweep, all 5 configs overlaid", fontsize=22, y=1.02)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--zarr-dir", type=Path, default=ZARR_DIR)
    p.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    zarr_paths = sorted(args.zarr_dir.glob("*.zarr"))
    if not zarr_paths:
        print(f"No zarrs in {args.zarr_dir}", file=sys.stderr)
        sys.exit(1)

    rng = np.random.default_rng(args.seed)
    pick = rng.choice(len(zarr_paths), size=min(args.n, len(zarr_paths)), replace=False)
    selected = [zarr_paths[i] for i in sorted(pick.tolist())]
    print(f"Selected {len(selected)} echograms (seed={args.seed}):")
    for zp in selected:
        print(f"  {zp.stem}")

    echograms = []
    for zi, zp in enumerate(selected):
        try:
            sv, ranges, heave = load_38k_echogram(zp)
        except Exception as exc:
            print(f"[{zi + 1}/{len(selected)}] {zp.stem}: load FAILED — {exc}")
            continue
        n_pings = sv.shape[1]
        traces = {}
        for cfg in CONFIGS:
            try:
                traces[cfg.name] = run_config(sv, ranges, heave, cfg)
            except Exception as exc:
                print(f"  {cfg.name}: FAILED — {exc}")
                traces[cfg.name] = np.full(n_pings, np.nan)
        echograms.append(
            {
                "name": zp.stem,
                "sv": sv,
                "ranges": ranges,
                "n_pings": n_pings,
                "traces": traces,
            }
        )
        print(f"[{zi + 1}/{len(selected)}] {zp.stem}: {n_pings} pings")

    if not echograms:
        print("No echograms loaded; aborting.", file=sys.stderr)
        sys.exit(1)

    for ci, cfg in enumerate(CONFIGS, start=1):
        out_path = args.out_dir / f"config_{ci}_{cfg.name.split()[0]}.png"
        plot_config_grid(cfg, ci, echograms, out_path)
        print(f"Wrote {out_path}")

    overlay_path = args.out_dir / "overlay_all_configs.png"
    plot_overlay_comparison(echograms, overlay_path)
    print(f"Wrote {overlay_path}")


if __name__ == "__main__":
    main()
