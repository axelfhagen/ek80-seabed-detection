#!/usr/bin/env python
"""Zoomed diagnostics for unusually wide Hake conformal bands.

Reuses the deep-water Hake conformal pipeline and cached calibration thresholds
from ``code/validation/generate_conformal_hake.py``. The selection criterion is
the rolling mean width of the locally weighted conformal band, which is the
adaptive band whose width scales with the local probability-transition width.

Output: Figures/conformal_hake_deep_water/wide_band_diagnostics/
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from data_visualization.thesis_style import (  # noqa: E402
    REFERENCE_BOTTOM_COLOR,
    UNET_PREDICTION_COLOR,
    apply_thesis_style,
    savefig,
)
from validation.conformal import conformal_quantile  # noqa: E402
from validation.generate_conformal_hake import (  # noqa: E402
    ALPHA,
    METHODS,
    OUT_DIR as HAKE_OUT_DIR,
    VMAX,
    VMIN,
    _CKPT,
    bands_for_group,
    consecutive_groups,
    group_coverage,
    load_model,
    load_or_build_calib,
    process_hake_group,
)


DEFAULT_OUTPUT_DIR = HAKE_OUT_DIR / "wide_band_diagnostics"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "code/checkpoints/chunk_arch_sweep_2/smp_unet_resnet34_best.pth"
LOCAL_BAND_COLOR = METHODS["local"][1]
PROFILE_BAND_COLOR = METHODS["profile"][1]
CONSTANT_BAND_COLOR = METHODS["constant"][1]


@dataclass
class Example:
    rank: int
    name: str
    center: int
    start: int
    stop: int
    score_m: float
    mean_width_m: float
    max_width_m: float
    coverage: float
    ref_median_m: float
    pred_median_m: float
    out_path: Path


def default_output_dir() -> Path:
    for key in ("ARTIFACT_DIR", "RUNNER_ARTIFACT_DIR", "GUESTFISH_OUTPUT"):
        value = os.environ.get(key)
        if value and value.startswith("/") and "\x1b" not in value and "\\e" not in value:
            return Path(value)
    return DEFAULT_OUTPUT_DIR


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(arr)
    filled = np.where(finite, arr, 0.0)
    kernel = np.ones(window, dtype=np.float64)
    sums = np.convolve(filled, kernel, mode="same")
    counts = np.convolve(finite.astype(np.float64), kernel, mode="same")
    out = np.full(arr.shape, np.nan, dtype=np.float64)
    valid = counts >= max(8, window // 4)
    out[valid] = sums[valid] / counts[valid]
    return out


def percentile(values: list[np.ndarray], q: float) -> float:
    finite = np.concatenate([v[np.isfinite(v)] for v in values if np.isfinite(v).any()])
    if finite.size == 0:
        return float("nan")
    return float(np.nanpercentile(finite, q))


def score_candidates(
    items, band_sets, window: int, min_separation: int, threshold_m: float
) -> list[tuple[float, str, int]]:
    candidates: list[tuple[float, str, int]] = []
    for name, _g in items:
        width = band_sets[name]["local"][2]
        score = rolling_mean(width, window)
        order = np.argsort(np.where(np.isfinite(score), score, -np.inf))[::-1]
        accepted: list[int] = []
        for idx in order:
            s = float(score[idx])
            if not np.isfinite(s):
                continue
            if s < threshold_m:
                break
            if any(abs(int(idx) - old) < min_separation for old in accepted):
                continue
            accepted.append(int(idx))
            candidates.append((s, name, int(idx)))
    candidates.sort(reverse=True, key=lambda t: t[0])
    return candidates


def select_examples(
    items,
    band_sets,
    n_examples: int,
    window: int,
    min_separation: int,
    max_per_group: int,
    threshold_m: float,
) -> list[tuple[float, str, int]]:
    candidates = score_candidates(items, band_sets, window, min_separation, threshold_m)
    selected: list[tuple[float, str, int]] = []
    counts = {name: 0 for name, _ in items}
    used = {name: [] for name, _ in items}
    for score, name, center in candidates:
        if counts[name] >= max_per_group:
            continue
        if any(abs(center - old) < min_separation for old in used[name]):
            continue
        selected.append((score, name, center))
        counts[name] += 1
        used[name].append(center)
        if len(selected) >= n_examples:
            return selected

    # If the percentile threshold is too strict, fill from the same ranking.
    if len(selected) < n_examples:
        looser = score_candidates(items, band_sets, window, min_separation, 0.0)
        for score, name, center in looser:
            if len(selected) >= n_examples:
                break
            if (score, name, center) in selected:
                continue
            if counts[name] >= max_per_group:
                continue
            if any(abs(center - old) < min_separation for old in used[name]):
                continue
            selected.append((score, name, center))
            counts[name] += 1
            used[name].append(center)

    return selected


def finite_depth_window(*arrays: np.ndarray) -> np.ndarray:
    vals = []
    for arr in arrays:
        a = np.asarray(arr, dtype=np.float64)
        vals.append(a[np.isfinite(a) & (a > 0)])
    vals = [v for v in vals if v.size]
    if not vals:
        return np.array([], dtype=np.float64)
    return np.concatenate(vals)


def plot_example(
    rank: int,
    name: str,
    g: dict,
    bands: dict,
    center: int,
    score_m: float,
    window: int,
    plot_width: int,
    out_path: Path,
) -> Example:
    sv1 = g["sv1"]
    ra = g["range_crop"]
    n_pings = sv1.shape[1]
    start = max(0, center - plot_width // 2)
    stop = min(n_pings, start + plot_width)
    start = max(0, stop - plot_width)
    x = np.arange(start, stop)
    extent = [start, stop - 1, ra[-1], ra[0]]

    local_u, local_l, local_w = bands["local"]
    profile_u, profile_l, _profile_w = bands["profile"]
    const_u, const_l = bands["constant"]
    pred = g["pred_d"]
    ref = g["ref_d"]

    depth_vals = finite_depth_window(
        local_u[start:stop],
        local_l[start:stop],
        profile_u[start:stop],
        profile_l[start:stop],
        pred[start:stop],
        ref[start:stop],
    )
    if depth_vals.size:
        pad = max(20.0, 0.12 * float(np.nanmax(depth_vals) - np.nanmin(depth_vals)))
        y0 = max(float(ra[0]), float(np.nanmin(depth_vals) - pad))
        y1 = min(float(ra[-1]), float(np.nanmax(depth_vals) + pad))
    else:
        y0, y1 = float(ra[0]), float(ra[-1])

    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.imshow(
        sv1[:, start:stop],
        aspect="auto",
        cmap="viridis",
        vmin=VMIN,
        vmax=VMAX,
        origin="upper",
        extent=extent,
    )

    for upper, lower, color, alpha, zorder in (
        (profile_u, profile_l, PROFILE_BAND_COLOR, 0.14, 2),
        (const_u, const_l, CONSTANT_BAND_COLOR, 0.16, 3),
        (local_u, local_l, LOCAL_BAND_COLOR, 0.32, 4),
    ):
        v = np.isfinite(upper[x]) & np.isfinite(lower[x])
        if v.any():
            ax.fill_between(
                x[v], upper[x][v], lower[x][v], color=color, alpha=alpha, linewidth=0, zorder=zorder
            )

    for upper, lower, color, linestyle, linewidth in (
        (local_u, local_l, LOCAL_BAND_COLOR, "--", 1.4),
        (profile_u, profile_l, PROFILE_BAND_COLOR, ":", 1.0),
        (const_u, const_l, CONSTANT_BAND_COLOR, ":", 0.9),
    ):
        v = np.isfinite(upper[x]) & np.isfinite(lower[x])
        if v.any():
            ax.plot(
                x[v],
                upper[x][v],
                color=color,
                linestyle=linestyle,
                linewidth=linewidth,
                alpha=0.9,
                zorder=6,
            )
            ax.plot(
                x[v],
                lower[x][v],
                color=color,
                linestyle=linestyle,
                linewidth=linewidth,
                alpha=0.9,
                zorder=6,
            )

    vpred = np.isfinite(pred[x]) & (pred[x] > 0)
    vref = np.isfinite(ref[x]) & (ref[x] > 0)
    if vpred.any():
        ax.plot(
            x[vpred],
            pred[x][vpred],
            color=UNET_PREDICTION_COLOR,
            linewidth=2.0,
            zorder=8,
            label="U-Net prediction",
        )
    if vref.any():
        ax.plot(
            x[vref],
            ref[x][vref],
            color=REFERENCE_BOTTOM_COLOR,
            linewidth=2.2,
            linestyle="--",
            zorder=9,
            label="Reference label",
        )

    ax.axvline(center, color="white", linewidth=1.0, alpha=0.65, zorder=10)
    ax.set_xlim(start, stop - 1)
    ax.set_ylim(y1, y0)
    ax.set_xlabel("Ping")
    ax.set_ylabel("Depth (m)")
    ax.set_title(
        f"Wide local conformal band #{rank}: {name} | {window}-ping mean width {score_m:.1f} m"
    )
    handles = [
        Line2D([0], [0], color=UNET_PREDICTION_COLOR, lw=2.0, label="U-Net prediction"),
        Line2D(
            [0], [0], color=REFERENCE_BOTTOM_COLOR, lw=2.2, linestyle="--", label="Reference label"
        ),
        Patch(facecolor=LOCAL_BAND_COLOR, alpha=0.32, label="Local conformal band"),
        Patch(facecolor=PROFILE_BAND_COLOR, alpha=0.14, label="Profile band"),
        Patch(facecolor=CONSTANT_BAND_COLOR, alpha=0.16, label="Constant band"),
    ]
    ax.legend(handles=handles, loc="lower right", ncol=2, framealpha=0.92, fontsize=12)
    savefig(fig, out_path)
    plt.close(fig)

    width_slice = local_w[start:stop]
    valid_cov = (
        np.isfinite(local_u[start:stop])
        & np.isfinite(local_l[start:stop])
        & np.isfinite(ref[start:stop])
        & (ref[start:stop] > 0)
    )
    if valid_cov.any():
        inside = (ref[start:stop][valid_cov] >= local_u[start:stop][valid_cov]) & (
            ref[start:stop][valid_cov] <= local_l[start:stop][valid_cov]
        )
        coverage = float(inside.mean())
    else:
        coverage = float("nan")

    return Example(
        rank=rank,
        name=name,
        center=center,
        start=start,
        stop=stop,
        score_m=score_m,
        mean_width_m=float(np.nanmean(width_slice)),
        max_width_m=float(np.nanmax(width_slice)),
        coverage=coverage,
        ref_median_m=float(np.nanmedian(ref[start:stop])),
        pred_median_m=float(np.nanmedian(pred[start:stop])),
        out_path=out_path,
    )


def plot_gallery(examples: list[Example], out_path: Path) -> None:
    if not examples:
        return
    import matplotlib.image as mpimg

    ncols = 2
    nrows = int(np.ceil(len(examples) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(10 * ncols, 5.6 * nrows))
    axes_flat = np.atleast_1d(axes).ravel()
    for ax, ex in zip(axes_flat, examples):
        ax.imshow(mpimg.imread(ex.out_path))
        ax.set_axis_off()
        ax.set_title(
            f"#{ex.rank} {ex.name}, p{ex.center} | mean {ex.score_m:.1f} m",
            fontsize=12,
        )
    for ax in axes_flat[len(examples) :]:
        ax.set_visible(False)
    fig.suptitle("Zoomed diagnostics: unusually wide local conformal bands", y=1.005)
    fig.tight_layout()
    savefig(fig, out_path)
    plt.close(fig)


def write_report(
    out_dir: Path,
    examples: list[Example],
    paths_used: list[Path],
    threshold_m: float,
    percentile_q: float,
    window: int,
    plot_width: int,
    q_const: float,
    q_local: float,
    q_profile: float,
    n_groups: int,
    source: Path,
) -> None:
    rows = "\n".join(
        f"| {ex.rank} | `{ex.name}` | {ex.center} | {ex.start}-{ex.stop - 1} | "
        f"{ex.score_m:.2f} | {ex.mean_width_m:.2f} | {ex.max_width_m:.2f} | "
        f"{ex.coverage:.2f} | `{ex.out_path.name}` |"
        for ex in examples
    )
    paths = "\n".join(f"- `{p}`" for p in paths_used)
    image_list = "\n".join(f"- `{ex.out_path}`" for ex in examples)
    report = f"""# Wide Conformal Band Diagnostics

Generated: {datetime.now():%Y-%m-%d %H:%M}

## Selection Criterion

I used the locally weighted conformal band from the deep-water Hake conformal
pipeline because its width is explicitly adaptive:
`prediction +/- q_local * max(transition_width, epsilon)`.

For every processed Hake group, I computed the local band width per ping,
smoothed it with a centered {window}-ping rolling mean, and defined "large band"
as a smoothed width at or above the global {percentile_q:.0f}th percentile
({threshold_m:.2f} m) across the processed groups. Examples are the highest
scoring non-overlapping windows, with at most two examples per stitched group.
If the percentile filter did not produce the requested count, the next highest
non-overlapping windows were used.

Each per-example plot shows a {plot_width}-ping zoom around the selected center
ping. The filled purple band is the selection band. The green profile band and
orange constant-width band are included as context.

## Data And Code Paths Used

{paths}

Source script: `{source}`

Calibrated thresholds at alpha={ALPHA}:
- constant q = {q_const:.3f} m
- locally weighted q = {q_local:.3f}
- profile q = {q_profile:.4f}

Processed Hake groups: {n_groups}

## Generated Images

- Gallery/contact sheet: `{out_dir / "wide_band_gallery.png"}`
{image_list}

## Selected Examples

| Rank | Group | Center ping | Plot pings | Rolling width m | Plot mean m | Plot max m | Local coverage | File |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | --- |
{rows}

## What Seems To Drive The Wide Bands

The wide local bands are driven by broad or ambiguous probability transitions
around the predicted bottom. In these Hake transects the seabed is deep
relative to the training regime and the 200 kHz channel does not reach the
bottom, so the three-channel model is effectively relying on 38 kHz alone near
the seabed. In the selected windows, the reference bottom often lies near areas
with weak or diffuse backscatter, sloped structure, or disagreement between the
red model line and cyan reference line. Those conditions inflate the transition
width sigma, and the locally weighted conformal rule expands the band.

The profile band is frequently much broader than the local band here because
the Hake calibration reached q_profile near the maximum threshold, which makes
it conservative under this out-of-distribution shift.
"""
    out_dir.joinpath("REPORT.md").write_text(report)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, default=default_output_dir())
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    ap.add_argument("--device", default=None)
    ap.add_argument("--group-size", type=int, default=3)
    ap.add_argument("--n-groups", type=int, default=6)
    ap.add_argument("--n-calib", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-examples", type=int, default=10)
    ap.add_argument("--window", type=int, default=96)
    ap.add_argument("--plot-width", type=int, default=256)
    ap.add_argument("--min-separation", type=int, default=192)
    ap.add_argument("--max-per-group", type=int, default=2)
    ap.add_argument("--large-percentile", type=float, default=95.0)
    ap.add_argument("--recompute-calib", action="store_true")
    args = ap.parse_args()

    apply_thesis_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _CKPT["path"] = str(args.checkpoint)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")
    print(f"Output: {args.output_dir}")
    model = load_model(args.checkpoint, device)

    depth_s, local_s, profile_s, _rel = load_or_build_calib(
        model, device, args.n_calib, args.seed, args.recompute_calib
    )
    q_const = conformal_quantile(depth_s, ALPHA)
    q_local = conformal_quantile(local_s, ALPHA)
    q_profile = conformal_quantile(profile_s, ALPHA)
    print(f"q_constant={q_const:.3f} q_local={q_local:.3f} q_profile={q_profile:.4f}")

    groups = consecutive_groups(args.group_size)[: args.n_groups]
    items = []
    band_sets = {}
    paths_used = [
        Path("code/validation/generate_conformal_hake.py"),
        Path("Figures/conformal_hake_deep_water/cache"),
        Path("Figures/conformal_hake_deep_water/_README.md"),
        args.checkpoint,
    ]
    for i, stems in enumerate(groups, 1):
        name = f"{stems[0]}+{len(stems) - 1}more" if len(stems) > 1 else stems[0]
        print(f"[hake {i}/{len(groups)}] {name}", flush=True)
        g = process_hake_group(stems, model, device)
        bands = bands_for_group(g, q_const, q_profile, q_local)
        u, l = bands["local"]
        band_sets[name] = {
            "constant": bands["constant"],
            "profile": (*bands["profile"], np.abs(bands["profile"][1] - bands["profile"][0])),
            "local": (*bands["local"], np.abs(l - u)),
        }
        print(f"  local coverage {group_coverage(g, 'local', bands) * 100:.0f}%")
        items.append((name, g))

    if not items:
        raise SystemExit("No Hake groups were processed.")

    smoothed_widths = [rolling_mean(band_sets[name]["local"][2], args.window) for name, _ in items]
    threshold_m = percentile(smoothed_widths, args.large_percentile)
    selected = select_examples(
        items,
        band_sets,
        args.n_examples,
        args.window,
        args.min_separation,
        args.max_per_group,
        threshold_m,
    )
    if not selected:
        raise SystemExit("No finite local conformal-band widths found.")

    by_name = {name: g for name, g in items}
    examples: list[Example] = []
    for rank, (score, name, center) in enumerate(selected, 1):
        out_path = args.output_dir / f"wide_band_{rank:02d}_{name}_p{center}.png"
        ex = plot_example(
            rank,
            name,
            by_name[name],
            band_sets[name],
            center,
            score,
            args.window,
            args.plot_width,
            out_path,
        )
        examples.append(ex)
        print(f"  saved {out_path}")

    plot_gallery(examples, args.output_dir / "wide_band_gallery.png")
    write_report(
        args.output_dir,
        examples,
        paths_used,
        threshold_m,
        args.large_percentile,
        args.window,
        args.plot_width,
        q_const,
        q_local,
        q_profile,
        len(items),
        Path(__file__),
    )
    print(f"Done. Report -> {args.output_dir / 'REPORT.md'}")


if __name__ == "__main__":
    main()
