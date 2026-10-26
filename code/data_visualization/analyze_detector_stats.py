"""Paired statistical comparison of the classic (algorithmic) bottom detectors.

Runs the three CRIMAC-ported per-ping detectors (simple threshold, edge,
gradient) at 38 kHz on a sample of zarr files, then compares them pairwise on
per-file mean detected depth and per-file detection rate (paired t-test +
Wilcoxon). Quantifies how much the choice of detector shifts the training
labels. Per-file outputs are cached to JSON for instant re-plotting.

Output: Figures/statistical_tests/
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

warnings.filterwarnings("ignore")  # nan-slice noise from detectors

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from data_visualization.thesis_style import apply_thesis_style, savefig
from raw_sensor_integration.preprocessing import load_sv_from_zarr
from raw_sensor_integration.bottom_labelling import (
    detect_bottom_simple, detect_bottom_edge, detect_bottom_gradient,
)

ZARR_DIR = PROJECT_ROOT / "Datasets" / "echopype_zarr"
OUTPUT_DIR = PROJECT_ROOT / "Figures" / "statistical_tests"
FREQ = 38000
MAX_RANGE_M = 500
DETECTORS = ["simple", "edge", "gradient"]


def eval_files(n: int) -> list[str]:
    ref = PROJECT_ROOT / "code" / "checkpoints" / "exp_20260417_130130_cosine" / "eval_metrics.json"
    if ref.exists():
        files = [r["file"] for r in json.load(open(ref))["per_file"]]
    else:
        files = sorted(p.name for p in ZARR_DIR.glob("*.zarr"))
    return files[:n]


def run_detectors(zarr_path: Path):
    sv, rng, _t, pd, hv, _f = load_sv_from_zarr(str(zarr_path), FREQ, max_range_m=MAX_RANGE_M)
    pd = pd if pd and pd > 0 else 1e-3
    return {
        "simple": detect_bottom_simple(sv, rng, hv, pd),
        "edge": detect_bottom_edge(sv, rng, hv, pd, frequency_hz=FREQ),
        "gradient": detect_bottom_gradient(sv, rng),
    }


def gather(files, cache: Path, recompute: bool) -> dict:
    if cache.exists() and not recompute:
        data = json.load(open(cache))
        if set(data.get("files", [])) >= set(files):
            print(f"[cache] {cache}")
            return data
    rows = []
    for i, fn in enumerate(files, 1):
        p = ZARR_DIR / fn
        if not p.exists():
            print(f"[miss] {fn}"); continue
        print(f"[{i}/{len(files)}] {fn}")
        det = run_detectors(p)
        row = {"file": fn}
        for name, b in det.items():
            valid = ~np.isnan(b)
            row[f"{name}_mean_depth"] = float(np.nanmean(b)) if valid.any() else np.nan
            row[f"{name}_rate"] = float(valid.mean())
        rows.append(row)
    data = {"files": [r["file"] for r in rows], "rows": rows}
    json.dump(data, open(cache, "w"), indent=2)
    print(f"[saved] {cache}")
    return data


def paired_stats(a, b):
    m = ~(np.isnan(a) | np.isnan(b))
    a, b = a[m], b[m]
    diff = a - b
    _, p_t = stats.ttest_rel(a, b)
    try:
        _, p_w = stats.wilcoxon(a, b)
    except ValueError:
        p_w = np.nan
    sd = diff.std(ddof=1)
    return {"n": int(m.sum()), "mean_diff": float(diff.mean()),
            "ci": float(1.96 * sd / np.sqrt(m.sum())) if m.sum() else np.nan,
            "dz": float(diff.mean() / sd) if sd > 0 else 0.0,
            "p_t": float(p_t), "p_w": float(p_w), "a": a, "b": b}


def plot(data: dict):
    rows = data["rows"]

    def col(metric):
        return {d: np.array([r[f"{d}_{metric}"] for r in rows]) for d in DETECTORS}

    depth, rate = col("mean_depth"), col("rate")
    pairs = list(combinations(DETECTORS, 2))

    fig = plt.figure(figsize=(22, 13))
    fig.suptitle("Classic detector comparison (38 kHz, per-file)")

    # Row 1: Bland-Altman for each detector pair (mean depth)
    for k, (x, y) in enumerate(pairs):
        ax = fig.add_subplot(2, 3, k + 1)
        s = paired_stats(depth[x], depth[y])
        mean_xy = (s["a"] + s["b"]) / 2
        d = s["a"] - s["b"]
        ax.scatter(mean_xy, d, s=55, color="#1f5f80", alpha=0.75)
        md, sd = d.mean(), d.std(ddof=1)
        ax.axhline(md, color="#c0392b", lw=2, label=f"bias {md:+.1f} m")
        ax.axhline(md + 1.96 * sd, color="gray", ls="--", lw=1.2)
        ax.axhline(md - 1.96 * sd, color="gray", ls="--", lw=1.2)
        ax.axhline(0, color="black", lw=0.8, alpha=0.4)
        sig = "*" if s["p_t"] < 0.05 else "n.s."
        ax.set_title(f"{x} vs {y}: p={s['p_t']:.3f} ({sig})")
        ax.set_xlabel("mean depth of pair (m)")
        ax.set_ylabel(f"{x} minus {y} (m)")
        ax.legend(); ax.grid(alpha=0.3)

    # Row 2a: per-file depth across detectors
    ax = fig.add_subplot(2, 3, 4)
    xs = np.arange(len(DETECTORS))
    for i in range(len(rows)):
        ax.plot(xs, [depth[d][i] for d in DETECTORS], "-o", ms=5, lw=1.0, alpha=0.5)
    ax.plot(xs, [np.nanmean(depth[d]) for d in DETECTORS], "-s",
            color="black", lw=3, ms=11, zorder=5)
    ax.set_xticks(xs); ax.set_xticklabels(DETECTORS)
    ax.set_ylabel("mean bottom depth (m)")
    ax.set_title("per-file depth across detectors")
    ax.invert_yaxis(); ax.grid(alpha=0.3)

    # Row 2b: detection rate distribution
    ax = fig.add_subplot(2, 3, 5)
    rdata = [rate[d] for d in DETECTORS]
    bp = ax.boxplot(rdata, tick_labels=DETECTORS, showmeans=True, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#cfe3f0")
    rng = np.random.default_rng(0)
    for i, vals in enumerate(rdata):
        ax.scatter(rng.normal(i + 1, 0.04, len(vals)), vals, s=24,
                   color="#1f5f80", alpha=0.6, zorder=3)
    ax.set_ylabel("valid-ping fraction")
    ax.set_title("detection rate per file")
    ax.grid(axis="y", alpha=0.3)

    # Row 2c: pairwise stats text
    ax = fig.add_subplot(2, 3, 6)
    ax.axis("off")
    lines = ["paired tests: mean depth", "",
             f"{'pair':<16}{'D(m)':>8}{'dz':>7}{'p_t':>8}{'p_w':>8}"]
    summary = []
    for x, y in pairs:
        s = paired_stats(depth[x], depth[y])
        lines.append(f"{x+'-'+y:<16}{s['mean_diff']:>+8.1f}{s['dz']:>7.2f}"
                     f"{s['p_t']:>8.3f}{s['p_w']:>8.3f}")
        summary.append([x, y, s["n"], s["mean_diff"], s["dz"], s["p_t"], s["p_w"]])
    lines += ["", "paired tests: detection rate", "",
              f"{'pair':<16}{'Drate':>8}{'dz':>7}{'p_t':>8}{'p_w':>8}"]
    for x, y in pairs:
        s = paired_stats(rate[x], rate[y])
        lines.append(f"{x+'-'+y:<16}{s['mean_diff']:>+8.3f}{s['dz']:>7.2f}"
                     f"{s['p_t']:>8.3f}{s['p_w']:>8.3f}")
    ax.text(0.0, 1.0, "\n".join(lines), family="monospace", fontsize=13,
            va="top", transform=ax.transAxes)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = OUTPUT_DIR / "detector_comparison.pdf"
    savefig(fig, out)
    plt.close(fig)

    csv = OUTPUT_DIR / "detector_depth_stats.csv"
    with open(csv, "w") as f:
        f.write("detector_a,detector_b,n,mean_diff_m,cohen_dz,p_ttest,p_wilcoxon\n")
        for r in summary:
            f.write(",".join(str(x) for x in r) + "\n")
    return out, csv


def main():
    global OUTPUT_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-files", type=int, default=12)
    ap.add_argument("--recompute", action="store_true")
    ap.add_argument("--output-dir", default=str(OUTPUT_DIR))
    args = ap.parse_args()
    apply_thesis_style()

    OUTPUT_DIR = Path(args.output_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    files = eval_files(args.n_files)
    data = gather(files, OUTPUT_DIR / "detector_per_file.json", args.recompute)
    out, csv = plot(data)
    print(f"\nFigure -> {out}\nStats   -> {csv}")


if __name__ == "__main__":
    main()
