"""Edge-preserving bottom-label de-jitter sweep (v2).

Output: Figures/bottom_label_config_sweep/v2/

Follow-up to plot_bottom_label_config_sweep.py. The v1 sweep showed that a
wide sliding median (C3) removes jitter but over-cuts the tops of real rocks
and the edges of steep slopes: a feature that spans only ~10 pings gets
dragged toward the surrounding flat seabed by a 61-ping median.

v2 keeps sharp *real* structure while removing jitter by discriminating on
temporal extent and local variability instead of magnitude alone:

  - run-length protection: only outlier runs short enough to be noise
    (<= max_run pings) are replaced; sustained deviations (rocks, slopes
    that "build up" over many pings) are kept.
  - adaptive MAD threshold (Hampel): the rejection threshold scales with
    *local* spread, so flat stretches are cleaned tightly while already
    rough/steep regions tolerate large sharp changes.

Same 10 random 38 kHz echograms as v1 (seed 7). Red line = resulting label.
"""
from __future__ import annotations

import argparse
import sys
import textwrap
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import scipy.ndimage as ndi

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "code"))
sys.path.insert(0, str(PROJECT_ROOT / "code" / "data_visualization"))

from raw_sensor_integration.bottom_labelling import (  # noqa: E402
    detect_bottom_ensemble_pipeline,
)
from plot_bottom_label_config_sweep import (  # noqa: E402
    load_38k_echogram, style_axes, FREQ_HZ, PULSE_DURATION,
)

OUTPUT_DIR = PROJECT_ROOT / "Figures" / "bottom_label_config_sweep" / "v2"


# ── Edge-preserving filters ────────────────────────────────────────────────

def _fill_gaps(trace: np.ndarray) -> np.ndarray:
    """Linear-interpolate NaN gaps so downstream filters see a continuous line."""
    out = trace.copy()
    valid = ~np.isnan(out)
    if valid.sum() >= 2:
        x = np.arange(len(out))
        out = np.interp(x, x[valid], out[valid])
    return out


def _local_median(trace: np.ndarray, window: int) -> np.ndarray:
    """Sliding-median reference (gaps assumed already filled)."""
    return ndi.median_filter(trace, size=window, mode="nearest")


def runlength_filter(trace: np.ndarray, window: int, dev_m: float,
                     max_run: int) -> np.ndarray:
    """Replace only outlier *runs* shorter than max_run; keep sustained ones.

    A point is a candidate outlier when it deviates from the local median by
    more than dev_m. Candidates are grouped into consecutive runs; runs up to
    max_run pings long are treated as jitter and snapped to the local median,
    while longer runs are kept untouched (real features that build up over
    many pings).
    """
    ref = _local_median(trace, window)
    dev = np.abs(trace - ref)
    is_out = (dev > dev_m) & ~np.isnan(trace) & ~np.isnan(ref)

    out = trace.copy()
    n = len(trace)
    i = 0
    while i < n:
        if is_out[i]:
            j = i
            while j < n and is_out[j]:
                j += 1
            if (j - i) <= max_run:
                out[i:j] = ref[i:j]
            i = j
        else:
            i += 1
    return out


def hampel_filter(trace: np.ndarray, window: int, n_sigma: float,
                  mad_floor_m: float) -> np.ndarray:
    """Adaptive MAD outlier rejection.

    Replaces a point with the local median when it deviates by more than
    max(n_sigma * 1.4826 * MAD, mad_floor_m). The MAD term makes the
    threshold scale with local variability: tight on flat stretches, loose
    on rough/steep regions, so sharp real structure survives.
    """
    med = ndi.median_filter(trace, size=window, mode="nearest")
    mad = 1.4826 * ndi.median_filter(np.abs(trace - med), size=window,
                                     mode="nearest")
    thresh = np.maximum(n_sigma * mad, mad_floor_m)
    out = trace.copy()
    mask = np.abs(trace - med) > thresh
    out[mask] = med[mask]
    return out


# ── Config definitions ─────────────────────────────────────────────────────

@dataclass
class Config:
    name: str
    description: str
    method: str  # 'median' | 'runlength' | 'hampel' | 'hampel_runlength'
    window: int = 31
    dev_m: float = 3.0
    max_run: int = 4
    n_sigma: float = 3.0
    mad_floor_m: float = 1.0
    # second-stage params (for combined methods)
    rl_window: int = 31
    rl_dev_m: float = 3.0
    rl_max_run: int = 4


CONFIGS = [
    Config(
        name="V1 ref_wide_median",
        description=(
            "Reference: the v1 'wider_window' config (61-ping sliding median, "
            "3 m deviation). Removes jitter but over-cuts rock tops and steep "
            "slope edges."
        ),
        method="median", window=61, dev_m=3.0,
    ),
    Config(
        name="V2 runlength",
        description=(
            "Snap only short outlier runs (<=4 pings) to a 31-ping local "
            "median; keep sustained deviations. Isolated spikes removed, "
            "multi-ping rocks and slopes preserved."
        ),
        method="runlength", window=31, dev_m=3.0, max_run=4,
    ),
    Config(
        name="V3 runlength_strict",
        description=(
            "Tighter run-length: 21-ping median, 2 m deviation, only runs "
            "<=3 pings removed. Cleans flat stretches harder while still "
            "keeping anything that builds up over >3 pings."
        ),
        method="runlength", window=21, dev_m=2.0, max_run=3,
    ),
    Config(
        name="V4 hampel_adaptive",
        description=(
            "Hampel filter (21-ping window, 3 sigma, 1 m floor). Rejection "
            "threshold scales with local spread: flat regions cleaned "
            "tightly, rough/steep regions tolerate large sharp changes."
        ),
        method="hampel", window=21, n_sigma=3.0, mad_floor_m=1.0,
    ),
    Config(
        name="V5 hampel_plus_runlength",
        description=(
            "Hampel (15-ping, 3 sigma, 0.8 m floor) for adaptive spike "
            "removal, then run-length cleanup (21-ping median, 2 m, runs "
            "<=3). Combines local-variability awareness with extent-based "
            "protection of real features."
        ),
        method="hampel_runlength", window=15, n_sigma=3.0, mad_floor_m=0.8,
        rl_window=21, rl_dev_m=2.0, rl_max_run=3,
    ),
]


def run_config(sv, ranges, heave, cfg: Config) -> np.ndarray:
    """Raw ensemble (3 detectors) + the config's edge-preserving de-jitter."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        raw, _, _ = detect_bottom_ensemble_pipeline(
            sv, ranges, heave,
            pulse_duration=PULSE_DURATION, frequency_hz=FREQ_HZ,
            minimum_range=10.0, offset=0.5,
            detectors=("simple", "edge", "gradient"),
            max_deviation=4.0,
            post_process=False,
        )
    trace = _fill_gaps(raw)

    if cfg.method == "median":
        ref = _local_median(trace, cfg.window)
        dev = np.abs(trace - ref)
        out = trace.copy()
        mask = (dev > cfg.dev_m) & ~np.isnan(ref)
        out[mask] = ref[mask]
        return out
    if cfg.method == "runlength":
        return runlength_filter(trace, cfg.window, cfg.dev_m, cfg.max_run)
    if cfg.method == "hampel":
        return hampel_filter(trace, cfg.window, cfg.n_sigma, cfg.mad_floor_m)
    if cfg.method == "hampel_runlength":
        t = hampel_filter(trace, cfg.window, cfg.n_sigma, cfg.mad_floor_m)
        return runlength_filter(t, cfg.rl_window, cfg.rl_dev_m, cfg.rl_max_run)
    raise ValueError(cfg.method)


# ── Plotting ───────────────────────────────────────────────────────────────

def plot_config_grid(cfg: Config, echograms, out_path: Path) -> None:
    n = len(echograms)
    n_cols, n_rows = 2, (n + 1) // 2
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(11.0 * n_cols, 3.8 * n_rows),
        constrained_layout=True,
    )
    axes = np.atleast_2d(axes)
    for k, e in enumerate(echograms):
        r, c = divmod(k, n_cols)
        ax = axes[r, c]
        style_axes(ax, e["sv"], e["ranges"], e["n_pings"], title=e["name"])
        ax.plot(np.arange(e["n_pings"]), e["traces"][cfg.name],
                color="#ff3b30", lw=1.8)
        if r == n_rows - 1:
            ax.set_xlabel("Ping", fontsize=15)
        if c == 0:
            ax.set_ylabel("Range (m)", fontsize=15)
    for k in range(n, n_rows * n_cols):
        r, c = divmod(k, n_cols)
        axes[r, c].axis("off")
    desc = textwrap.fill(cfg.description, width=110)
    fig.suptitle(f"{cfg.name}\n{desc}", fontsize=18, y=1.04, linespacing=1.3)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_overlay(echograms, out_path: Path) -> None:
    n = len(echograms)
    n_cols, n_rows = 2, (n + 1) // 2
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(11.0 * n_cols, 4.2 * n_rows),
        constrained_layout=True,
    )
    axes = np.atleast_2d(axes)
    colors = ["#ff3b30", "#ffd60a", "#30d158", "#0a84ff", "#bf5af2"]
    names = [c.name for c in CONFIGS]
    for k, e in enumerate(echograms):
        r, c = divmod(k, n_cols)
        ax = axes[r, c]
        style_axes(ax, e["sv"], e["ranges"], e["n_pings"], title=e["name"])
        x = np.arange(e["n_pings"])
        for name, col in zip(names, colors):
            ax.plot(x, e["traces"][name], color=col, lw=1.2, label=name,
                    alpha=0.85)
        if k == 0:
            ax.legend(loc="upper right", fontsize=12, framealpha=0.85)
        if r == n_rows - 1:
            ax.set_xlabel("Ping", fontsize=15)
        if c == 0:
            ax.set_ylabel("Range (m)", fontsize=15)
    for k in range(n, n_rows * n_cols):
        r, c = divmod(k, n_cols)
        axes[r, c].axis("off")
    fig.suptitle("Edge-preserving de-jitter sweep (v2), all 5 configs overlaid",
                 fontsize=22, y=1.02)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _raw_trace(e) -> np.ndarray:
    """The pre-smoothing ensemble trace, stored alongside the configs."""
    return e["traces"]["_raw"]


# Hand-picked zoom windows over regions with genuinely rough/bumpy bottom
# (read off the v1/v2 config grids). (file_stem, center_ping, half_width).
ZOOM_SPOTS = [
    ("D20240716-T051147", 300, 260),
    ("D20241006-T192708", 650, 320),
    ("D20241004-T192702", 560, 320),
    ("D20241009-T043209", 4400, 320),
]


def plot_peak_zooms(echograms, out_path: Path) -> None:
    """Zoom into rough-bottom regions; overlay raw + V1/V2/V4/V5 with a
    y-axis tightened around the bottom so peak preservation is visible."""
    by_name = {e["name"]: e for e in echograms}
    spots = [(by_name[n], cp, hw) for (n, cp, hw) in ZOOM_SPOTS if n in by_name]
    if not spots:
        return

    n_rows = (len(spots) + 1) // 2
    fig, axes = plt.subplots(n_rows, 2, figsize=(11.0 * 2, 4.2 * n_rows),
                             constrained_layout=True, squeeze=False)
    for k, (e, cp, hw) in enumerate(spots):
        r, c = divmod(k, 2)
        ax = axes[r][c]
        lo, hi = max(0, cp - hw), min(e["n_pings"], cp + hw)
        ranges = e["ranges"]
        sv = e["sv"][:, lo:hi]
        x = np.arange(lo, hi)

        # tighten y-axis around the bottom trace in this window
        win = _raw_trace(e)[lo:hi]
        win = win[np.isfinite(win)]
        if win.size:
            ytop = max(0.0, float(np.nanmin(win)) - 12.0)
            ybot = float(np.nanmax(win)) + 18.0
        else:
            ytop, ybot = float(ranges[0]), float(ranges[-1])
        row_lo = int(np.searchsorted(ranges, ytop))
        row_hi = int(np.searchsorted(ranges, ybot))
        row_lo = max(0, min(row_lo, len(ranges) - 2))
        row_hi = max(row_lo + 1, min(row_hi, len(ranges) - 1))

        extent = [lo, hi, float(ranges[row_hi]), float(ranges[row_lo])]
        ax.imshow(sv[row_lo:row_hi], aspect="auto", origin="upper",
                  extent=extent, vmin=-80, vmax=-34, cmap="viridis",
                  interpolation="nearest")
        ax.set_ylim(ybot, ytop)
        ax.set_xlim(lo, hi)

        ax.plot(x, _raw_trace(e)[lo:hi], color="0.85", lw=1.2, ls="--",
                label="raw ensemble")
        ax.plot(x, e["traces"]["V1 ref_wide_median"][lo:hi],
                color="#ff3b30", lw=2.4, label="V1 wide median")
        ax.plot(x, e["traces"]["V2 runlength"][lo:hi],
                color="#30d158", lw=2.0, label="V2 run-length")
        ax.plot(x, e["traces"]["V4 hampel_adaptive"][lo:hi],
                color="#0a84ff", lw=2.0, label="V4 Hampel")
        ax.plot(x, e["traces"]["V5 hampel_plus_runlength"][lo:hi],
                color="#bf5af2", lw=1.8, label="V5 combined")
        ax.set_title(f"{e['name']}  pings {lo}-{hi}", fontsize=16)
        ax.tick_params(labelsize=13)
        ax.set_xlabel("Ping", fontsize=14)
        if c == 0:
            ax.set_ylabel("Range (m)", fontsize=14)
        if k == 0:
            ax.legend(loc="upper right", fontsize=11, framealpha=0.9)
    for k in range(len(spots), n_rows * 2):
        r, c = divmod(k, 2)
        axes[r][c].axis("off")
    fig.suptitle("Rough-bottom zoom: does each config keep real rocks/slopes?",
                 fontsize=20, y=1.03)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _synthetic_bottom():
    """Ground-truth bottom (m) with a sharp rock, steep slope, and flats,
    plus a noisy observation (jitter + isolated spikes)."""
    n = 460
    gt = np.full(n, 80.0)
    # flat 0-150 at 80 m
    # sharp rock 195-205: rises to 64 m over ~5 pings and back (real feature)
    rock_x = np.arange(190, 211)
    gt[rock_x] = 80.0 - 16.0 * np.exp(-((rock_x - 200) ** 2) / (2 * 3.0 ** 2))
    # steep slope 250-310: 80 -> 112 m
    gt[250:310] = np.linspace(80.0, 112.0, 60)
    gt[310:] = 112.0
    # second smaller rock on the deep flat 360-372
    rock2 = np.arange(356, 377)
    gt[rock2] = np.minimum(
        gt[rock2],
        112.0 - 9.0 * np.exp(-((rock2 - 366) ** 2) / (2 * 3.0 ** 2)),
    )

    rng = np.random.default_rng(0)
    obs = gt + rng.normal(0, 0.4, n)  # mild measurement noise everywhere
    # jittery flat patch 100-150 (±2 m wobble we want removed)
    obs[100:150] += rng.normal(0, 2.0, 50)
    # isolated spikes (1-2 pings) we want removed
    for p, val in [(60, 50.0), (130, 132.0), (175, 55.0),
                   (270, 60.0), (330, 138.0), (420, 70.0)]:
        obs[p] = val
    obs[230] = 48.0
    obs[231] = 50.0  # 2-ping spike
    return gt, obs


def plot_synthetic_stress_test(out_path: Path) -> None:
    """Run each config's filter on a synthetic trace with known ground truth."""
    gt, obs = _synthetic_bottom()
    x = np.arange(len(gt))

    outputs = {}
    for cfg in CONFIGS:
        trace = obs.copy()
        if cfg.method == "median":
            ref = _local_median(trace, cfg.window)
            dev = np.abs(trace - ref)
            out = trace.copy()
            out[dev > cfg.dev_m] = ref[dev > cfg.dev_m]
        elif cfg.method == "runlength":
            out = runlength_filter(trace, cfg.window, cfg.dev_m, cfg.max_run)
        elif cfg.method == "hampel":
            out = hampel_filter(trace, cfg.window, cfg.n_sigma, cfg.mad_floor_m)
        else:
            t = hampel_filter(trace, cfg.window, cfg.n_sigma, cfg.mad_floor_m)
            out = runlength_filter(t, cfg.rl_window, cfg.rl_dev_m, cfg.rl_max_run)
        outputs[cfg.name] = out

    fig, axes = plt.subplots(len(CONFIGS), 1, figsize=(15, 3.0 * len(CONFIGS)),
                             sharex=True, constrained_layout=True)
    for ax, cfg in zip(axes, CONFIGS):
        ax.plot(x, obs, color="0.7", lw=1.0, label="noisy observation")
        ax.plot(x, gt, color="k", lw=1.6, ls="--", label="ground truth")
        ax.plot(x, outputs[cfg.name], color="#ff3b30", lw=2.0,
                label=f"{cfg.name} output")
        rmse = float(np.sqrt(np.mean((outputs[cfg.name] - gt) ** 2)))
        ax.set_title(f"{cfg.name}   (RMSE vs truth = {rmse:.2f} m)",
                     fontsize=15)
        ax.set_ylim(135, 45)  # depth increasing downward
        ax.set_ylabel("Depth (m)", fontsize=13)
        ax.tick_params(labelsize=12)
        ax.legend(loc="lower right", fontsize=10, ncol=3, framealpha=0.9)
    axes[-1].set_xlabel("Ping", fontsize=14)
    fig.suptitle("Synthetic stress test: keep the rock + steep slope, "
                 "remove jitter + isolated spikes", fontsize=18, y=1.02)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--zarr-dir", type=Path,
                   default=PROJECT_ROOT / "Datasets" / "echopype_zarr")
    p.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    zarr_paths = sorted(args.zarr_dir.glob("*.zarr"))
    rng = np.random.default_rng(args.seed)
    pick = rng.choice(len(zarr_paths), size=min(args.n, len(zarr_paths)),
                      replace=False)
    selected = [zarr_paths[i] for i in sorted(pick.tolist())]
    print(f"Selected {len(selected)} echograms (seed={args.seed})")

    echograms = []
    for zi, zp in enumerate(selected):
        try:
            sv, ranges, heave = load_38k_echogram(zp)
        except Exception as exc:
            print(f"[{zi+1}/{len(selected)}] {zp.stem}: load FAILED — {exc}")
            continue
        n_pings = sv.shape[1]
        traces = {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            raw, _, _ = detect_bottom_ensemble_pipeline(
                sv, ranges, heave,
                pulse_duration=PULSE_DURATION, frequency_hz=FREQ_HZ,
                minimum_range=10.0, offset=0.5,
                detectors=("simple", "edge", "gradient"),
                max_deviation=4.0, post_process=False,
            )
        traces["_raw"] = _fill_gaps(raw)
        for cfg in CONFIGS:
            try:
                traces[cfg.name] = run_config(sv, ranges, heave, cfg)
            except Exception as exc:
                print(f"  {cfg.name}: FAILED — {exc}")
                traces[cfg.name] = np.full(n_pings, np.nan)
        # Decimate the echogram for display only (detection ran at full res).
        sv_disp = sv[::2].astype(np.float32)
        ranges_disp = ranges[::2]
        echograms.append({"name": zp.stem, "sv": sv_disp, "ranges": ranges_disp,
                          "n_pings": n_pings, "traces": traces})
        del sv
        print(f"[{zi+1}/{len(selected)}] {zp.stem}: {n_pings} pings")

    if not echograms:
        print("No echograms loaded; aborting.", file=sys.stderr)
        sys.exit(1)

    for ci, cfg in enumerate(CONFIGS, start=1):
        out_path = args.out_dir / f"config_{ci}_{cfg.name.split()[0]}.png"
        plot_config_grid(cfg, echograms, out_path)
        print(f"Wrote {out_path}")

    overlay_path = args.out_dir / "overlay_all_configs.png"
    plot_overlay(echograms, overlay_path)
    print(f"Wrote {overlay_path}")

    zoom_path = args.out_dir / "peak_preservation_zoom.png"
    plot_peak_zooms(echograms, zoom_path)
    print(f"Wrote {zoom_path}")

    synth_path = args.out_dir / "synthetic_stress_test.png"
    plot_synthetic_stress_test(synth_path)
    print(f"Wrote {synth_path}")


if __name__ == "__main__":
    main()
