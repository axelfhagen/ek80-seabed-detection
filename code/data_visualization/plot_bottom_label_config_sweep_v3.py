"""Edge-preserving *smooth* bottom-label sweep (v3).

Output: Figures/bottom_label_config_sweep/v3/

v1's wide median was smooth but shaved the tops off real rocks/steep slopes;
v2 kept detail but looked jittery. v3 targets the actual ask: C3-level
smoothness without cutting real features.

Recipe: robust spike removal (small median + Hampel) to kill isolated
outliers and rapid ping-to-ping oscillation, then a 1-D bilateral filter.
The bilateral averages along gentle slopes and flats (removing jitter, like
a wide median) but stops averaging across a large value jump, so a genuine
rock or cliff stays sharp. Smoothness is set by the spatial sigma; how
sharp an edge survives is set by the range sigma (smaller = sharper).

Same 10 random 38 kHz echograms as v1/v2 (seed 7). Red line = label.
Also writes a synthetic stress test with known ground truth.
"""

from __future__ import annotations

import argparse
import sys
import textwrap
import warnings
from dataclasses import dataclass
from pathlib import Path

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
    load_38k_echogram,
    style_axes,
    FREQ_HZ,
    PULSE_DURATION,
)

OUTPUT_DIR = PROJECT_ROOT / "Figures" / "bottom_label_config_sweep" / "v3"


# ── Filters ────────────────────────────────────────────────────────────────


def _fill_gaps(trace: np.ndarray) -> np.ndarray:
    out = trace.copy()
    valid = ~np.isnan(out)
    if valid.sum() >= 2:
        x = np.arange(len(out))
        out = np.interp(x, x[valid], out[valid])
    return out


def hampel(trace: np.ndarray, window: int, n_sigma: float, floor_m: float) -> np.ndarray:
    """Adaptive MAD outlier rejection (gaps assumed filled)."""
    med = ndi.median_filter(trace, size=window, mode="nearest")
    mad = 1.4826 * ndi.median_filter(np.abs(trace - med), size=window, mode="nearest")
    thresh = np.maximum(n_sigma * mad, floor_m)
    out = trace.copy()
    mask = np.abs(trace - med) > thresh
    out[mask] = med[mask]
    return out


def bilateral_1d(trace: np.ndarray, spatial_sigma: float, range_sigma: float) -> np.ndarray:
    """Edge-preserving smoothing. Averages neighbours weighted by spatial
    distance AND value similarity, so flats/slopes smooth but sharp jumps
    (real rocks, cliffs) are preserved."""
    n = len(trace)
    radius = max(1, int(round(3 * spatial_sigma)))
    offs = np.arange(-radius, radius + 1)
    spatial_w = np.exp(-(offs**2) / (2 * spatial_sigma**2))
    out = np.empty_like(trace)
    inv2r2 = 1.0 / (2 * range_sigma**2)
    for i in range(n):
        lo, hi = max(0, i - radius), min(n, i + radius + 1)
        seg = trace[lo:hi]
        sw = spatial_w[(lo - i + radius) : (hi - i + radius)]
        rw = np.exp(-((seg - trace[i]) ** 2) * inv2r2)
        w = sw * rw
        out[i] = float(np.dot(w, seg) / np.sum(w))
    return out


# ── Configs ────────────────────────────────────────────────────────────────


@dataclass
class Config:
    name: str
    description: str
    method: str  # 'median_ref' | 'bilateral'
    # wide-median reference
    med_window: int = 61
    med_dev: float = 3.0
    # pre-smoothing spike removal
    pre_median: int = 5
    hampel_window: int = 15
    hampel_sigma: float = 3.0
    hampel_floor: float = 1.0
    # bilateral
    spatial_sigma: float = 12.0
    range_sigma: float = 3.5


CONFIGS = [
    Config(
        name="S1 ref_wide_median",
        description=(
            "Reference: v1 wide median (61-ping, 3 m). Smooth, but shaves the "
            "tops off real rocks and rounds steep slope edges."
        ),
        method="median_ref",
        med_window=61,
        med_dev=3.0,
    ),
    Config(
        name="S2 bilateral_sharp",
        description=(
            "Robust spike removal then bilateral smoothing (spatial 10 pings, "
            "range 2.5 m). Smooth flats, but only fairly sharp jumps are kept "
            "as edges, so most real rocks/slopes survive."
        ),
        method="bilateral",
        pre_median=5,
        hampel_window=15,
        hampel_sigma=3.0,
        hampel_floor=1.0,
        spatial_sigma=10.0,
        range_sigma=2.5,
    ),
    Config(
        name="S3 bilateral_balanced",
        description=(
            "Bilateral spatial 14 pings, range 3.5 m. More smoothing of "
            "flats/jitter while still preserving genuine slope and rock edges."
        ),
        method="bilateral",
        pre_median=5,
        hampel_window=15,
        hampel_sigma=3.0,
        hampel_floor=1.0,
        spatial_sigma=14.0,
        range_sigma=3.5,
    ),
    Config(
        name="S4 bilateral_smooth",
        description=(
            "Bilateral spatial 20 pings, range 4.5 m. Smoothest: close to C3 "
            "on flats, only strong sharp features stay sharp."
        ),
        method="bilateral",
        pre_median=7,
        hampel_window=21,
        hampel_sigma=3.0,
        hampel_floor=1.0,
        spatial_sigma=20.0,
        range_sigma=4.5,
    ),
]


def _apply(cfg: Config, trace: np.ndarray) -> np.ndarray:
    if cfg.method == "median_ref":
        ref = ndi.median_filter(trace, size=cfg.med_window, mode="nearest")
        dev = np.abs(trace - ref)
        out = trace.copy()
        out[dev > cfg.med_dev] = ref[dev > cfg.med_dev]
        return out
    # bilateral path: small median -> hampel -> bilateral
    t = ndi.median_filter(trace, size=cfg.pre_median, mode="nearest")
    t = hampel(t, cfg.hampel_window, cfg.hampel_sigma, cfg.hampel_floor)
    return bilateral_1d(t, cfg.spatial_sigma, cfg.range_sigma)


def run_config(sv, ranges, heave, cfg: Config) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        raw, _, _ = detect_bottom_ensemble_pipeline(
            sv,
            ranges,
            heave,
            pulse_duration=PULSE_DURATION,
            frequency_hz=FREQ_HZ,
            minimum_range=10.0,
            offset=0.5,
            detectors=("simple", "edge", "gradient"),
            max_deviation=4.0,
            post_process=False,
        )
    return _apply(cfg, _fill_gaps(raw))


# ── Plotting ───────────────────────────────────────────────────────────────


def plot_config_grid(cfg: Config, echograms, out_path: Path) -> None:
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
        ax.plot(np.arange(e["n_pings"]), e["traces"][cfg.name], color="#ff3b30", lw=1.8)
        if r == n_rows - 1:
            ax.set_xlabel("Ping", fontsize=15)
        if c == 0:
            ax.set_ylabel("Range (m)", fontsize=15)
    for k in range(n, n_rows * n_cols):
        r, c = divmod(k, n_cols)
        axes[r, c].axis("off")
    desc = textwrap.fill(cfg.description, width=110)
    fig.suptitle(f"{cfg.name}\n{desc}", fontsize=18, y=1.04, linespacing=1.3)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_overlay(echograms, out_path: Path) -> None:
    n = len(echograms)
    n_cols, n_rows = 2, (n + 1) // 2
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(11.0 * n_cols, 4.2 * n_rows),
        constrained_layout=True,
    )
    axes = np.atleast_2d(axes)
    colors = ["#ff3b30", "#ffd60a", "#30d158", "#0a84ff"]
    names = [c.name for c in CONFIGS]
    for k, e in enumerate(echograms):
        r, c = divmod(k, n_cols)
        ax = axes[r, c]
        style_axes(ax, e["sv"], e["ranges"], e["n_pings"], title=e["name"])
        x = np.arange(e["n_pings"])
        for name, col in zip(names, colors):
            ax.plot(x, e["traces"][name], color=col, lw=1.2, label=name, alpha=0.85)
        if k == 0:
            ax.legend(loc="upper right", fontsize=12, framealpha=0.85)
        if r == n_rows - 1:
            ax.set_xlabel("Ping", fontsize=15)
        if c == 0:
            ax.set_ylabel("Range (m)", fontsize=15)
    for k in range(n, n_rows * n_cols):
        r, c = divmod(k, n_cols)
        axes[r, c].axis("off")
    fig.suptitle("Edge-preserving smooth sweep (v3), all configs overlaid", fontsize=22, y=1.02)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def _synthetic_bottom():
    n = 460
    gt = np.full(n, 80.0)
    rock_x = np.arange(190, 211)
    gt[rock_x] = 80.0 - 16.0 * np.exp(-((rock_x - 200) ** 2) / (2 * 3.0**2))
    gt[250:310] = np.linspace(80.0, 112.0, 60)
    gt[310:] = 112.0
    rock2 = np.arange(356, 377)
    gt[rock2] = np.minimum(gt[rock2], 112.0 - 9.0 * np.exp(-((rock2 - 366) ** 2) / (2 * 3.0**2)))
    rng = np.random.default_rng(0)
    obs = gt + rng.normal(0, 0.4, n)
    obs[100:150] += rng.normal(0, 2.0, 50)
    for p, val in [(60, 50.0), (130, 132.0), (175, 55.0), (270, 60.0), (330, 138.0), (420, 70.0)]:
        obs[p] = val
    obs[230] = 48.0
    obs[231] = 50.0
    return gt, obs


def plot_synthetic(out_path: Path) -> None:
    gt, obs = _synthetic_bottom()
    x = np.arange(len(gt))
    fig, axes = plt.subplots(
        len(CONFIGS), 1, figsize=(15, 3.0 * len(CONFIGS)), sharex=True, constrained_layout=True
    )
    for ax, cfg in zip(axes, CONFIGS):
        out = _apply(cfg, obs.copy())
        rmse = float(np.sqrt(np.mean((out - gt) ** 2)))
        ax.plot(x, obs, color="0.7", lw=1.0, label="noisy observation")
        ax.plot(x, gt, color="k", lw=1.6, ls="--", label="ground truth")
        ax.plot(x, out, color="#ff3b30", lw=2.0, label=f"{cfg.name} output")
        ax.set_title(f"{cfg.name}   (RMSE vs truth = {rmse:.2f} m)", fontsize=15)
        ax.set_ylim(135, 45)
        ax.set_ylabel("Depth (m)", fontsize=13)
        ax.tick_params(labelsize=12)
        ax.legend(loc="lower right", fontsize=10, ncol=3, framealpha=0.9)
    axes[-1].set_xlabel("Ping", fontsize=14)
    fig.suptitle(
        "Synthetic stress test: smooth flats, keep rock + steep slope, remove jitter + spikes",
        fontsize=18,
        y=1.02,
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--zarr-dir", type=Path, default=PROJECT_ROOT / "Datasets" / "echopype_zarr")
    p.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    zarr_paths = sorted(args.zarr_dir.glob("*.zarr"))
    rng = np.random.default_rng(args.seed)
    pick = rng.choice(len(zarr_paths), size=min(args.n, len(zarr_paths)), replace=False)
    selected = [zarr_paths[i] for i in sorted(pick.tolist())]
    print(f"Selected {len(selected)} echograms (seed={args.seed})")

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
        # Display-only decimation to a fixed pixel budget (keeps memory and
        # figure size small; traces are in data coords so stay aligned).
        step_c = max(1, sv.shape[1] // 1400)
        step_r = max(1, sv.shape[0] // 500)
        sv_disp = sv[::step_r, ::step_c].astype(np.float32)
        echograms.append(
            {
                "name": zp.stem,
                "sv": sv_disp,
                "ranges": ranges[::step_r],
                "n_pings": n_pings,
                "traces": traces,
            }
        )
        del sv
        print(f"[{zi + 1}/{len(selected)}] {zp.stem}: {n_pings} pings")

    if not echograms:
        print("No echograms loaded; aborting.", file=sys.stderr)
        sys.exit(1)

    for ci, cfg in enumerate(CONFIGS, start=1):
        out_path = args.out_dir / f"config_{ci}_{cfg.name.split()[0]}.png"
        plot_config_grid(cfg, echograms, out_path)
        print(f"Wrote {out_path}")
    plot_overlay(echograms, args.out_dir / "overlay_all_configs.png")
    print("Wrote overlay")
    plot_synthetic(args.out_dir / "synthetic_stress_test.png")
    print("Wrote synthetic")


if __name__ == "__main__":
    main()
