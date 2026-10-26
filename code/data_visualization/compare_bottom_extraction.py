"""Compare bottom-line extraction methods on the U-Net probability map.

Runs full-echogram inference over a model's zarr evaluation set, then derives a
per-ping bottom line several ways: the current baseline (first pixel >= 0.5) and
the robust methods in ``deep_learning.bottom_extraction`` - contiguous run,
temporal gating with hysteresis, DP/Viterbi (step-edge cost), a temporal Hampel
de-spiker, and an Otsu adaptive threshold. Files are ranked once by the baseline
bottom-depth error so the same worst-9 / best-9 / random-9 files are shown across
every method.

Two parameter sweeps are also run per checkpoint: detection threshold (for the
baseline first-edge rule) and DP smoothness. The whole thing loops over one or
more checkpoints and writes a cross-checkpoint summary.

Outputs (Figures/bottom_extraction_methods/):
  <checkpoint>/<method>_<set>.png   nine 3x3 grids per method
  <checkpoint>/threshold_sweep.png  error / jumps vs detection threshold
  <checkpoint>/dp_smoothness_sweep.png
  <checkpoint>/metrics.json         aggregate error + jumpiness + coverage
  summary_across_checkpoints.json   every checkpoint, every method
  summary_across_checkpoints.png    grouped bars (mean error, big jumps)
  _README.md                        folder card

This sweep is GPU work: run it on Slurm
(code/deep_learning/training/slurm/run_bottom_extraction_compare.sh), not the
login node.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from deep_learning.unet_model import create_unet, predict_full_echogram
from deep_learning.evaluate import (
    load_file_data, extract_bottom_line, bottom_depth_error,
)
from deep_learning.bottom_extraction import (
    extract_bottom_contiguous, extract_bottom_gated, extract_bottom_dp,
    extract_bottom_hampel, extract_bottom_otsu, bottom_jumpiness,
)
from data_visualization.thesis_style import apply_thesis_style

ZARR_DIR = PROJECT_ROOT / "Datasets" / "echopype_zarr"
DEFAULT_CHECKPOINT = (PROJECT_ROOT / "code" / "checkpoints"
                      / "exp_20260420_082458_cosine_gentle"
                      / "unet_resnet34_best.pth")
OUTPUT_DIR = PROJECT_ROOT / "Figures" / "bottom_extraction_methods"

FREQ1 = 38_000
FREQ2 = 200_000
TARGET_SIZE = (512, 512)
PATCH_WIDTH = 512
STRIDE = 256
THRESHOLD = 0.5

METHODS = ("contiguous", "gated", "dp", "hampel", "otsu")
ALL_LINES = ("baseline",) + METHODS
SETS = ("random", "worst", "best")

THRESHOLD_GRID = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
SMOOTHNESS_GRID = (0.1, 0.25, 0.5, 1.0, 2.0)


def encoder_from_checkpoint(path: Path) -> str:
    """U-Net checkpoints are named unet_<encoder>_best.pth."""
    m = re.match(r"unet_(.+?)_(?:best|final|last)$", path.stem)
    return m.group(1) if m else "resnet34"


def load_model(checkpoint_path, device):
    encoder = encoder_from_checkpoint(Path(checkpoint_path))
    model = create_unet(encoder_name=encoder, in_channels=3)
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    sd = state.get("model_state_dict", state)
    cleaned = {k.removeprefix("model."): v for k, v in sd.items()}
    model.load_state_dict(cleaned, strict=False)
    model.to(device)
    model.eval()
    return model


def method_lines(prob, args):
    """All bottom lines for one probability map."""
    return {
        "baseline": extract_bottom_line(prob, threshold=THRESHOLD),
        "contiguous": extract_bottom_contiguous(prob, threshold=THRESHOLD,
                                                min_run=args.min_run),
        "gated": extract_bottom_gated(prob, threshold=THRESHOLD,
                                      high_threshold=args.high_threshold,
                                      max_jump=args.max_jump,
                                      min_run=max(1, args.min_run // 2)),
        "dp": extract_bottom_dp(prob, smoothness=args.smoothness),
        "hampel": extract_bottom_hampel(prob, threshold=THRESHOLD,
                                        window=args.hampel_window,
                                        n_sigma=args.hampel_sigma),
        "otsu": extract_bottom_otsu(prob, min_run=args.min_run),
    }


def cache_path(cache_dir, name):
    return cache_dir / f"{name}.npz"


def predict_and_cache(model, zp, device, cache_dir):
    """Return (prob, gt_mask, echo, range_arr), using an NPZ cache if present."""
    cp = cache_path(cache_dir, zp.stem)
    if cp.exists():
        d = np.load(cp)
        return d["prob"], d["gt_mask"], d["echo"], d["range_arr"]

    sv_channels, gt_mask, _gt_idx, range_arr = load_file_data(
        zp, FREQ1, FREQ2, vmin=-80.0, vmax=-34.0)
    prob = predict_full_echogram(model, sv_channels, TARGET_SIZE, device,
                                 patch_width=PATCH_WIDTH, stride=STRIDE)
    echo = sv_channels[1].astype(np.float32)  # 200 kHz normalised
    np.savez_compressed(cp, prob=prob.astype(np.float32),
                        gt_mask=gt_mask.astype(np.uint8),
                        echo=echo, range_arr=range_arr)
    return prob, gt_mask, echo, range_arr


def aggregate_metrics(per_file, jump_threshold_m):
    """Aggregate per-method errors / jumpiness / coverage across all files."""
    out = {}
    for method in ALL_LINES:
        errs = np.concatenate([r["errors"][method] for r in per_file
                               if len(r["errors"][method])]) \
            if any(len(r["errors"][method]) for r in per_file) else np.array([])
        steps = [r["jump"][method][0] for r in per_file
                 if not np.isnan(r["jump"][method][0])]
        big = sum(r["jump"][method][1] for r in per_file)
        cov = [r["coverage"][method] for r in per_file]
        out[method] = {
            "bottom_error_mean_m": float(np.mean(errs)) if len(errs) else float("nan"),
            "bottom_error_median_m": float(np.median(errs)) if len(errs) else float("nan"),
            "bottom_error_p95_m": float(np.percentile(errs, 95)) if len(errs) else float("nan"),
            "mean_abs_step_m": float(np.mean(steps)) if steps else float("nan"),
            "n_big_jumps": int(big),
            "coverage": float(np.mean(cov)) if cov else float("nan"),
            "jump_threshold_m": jump_threshold_m,
        }
    return out


def print_table(metrics, title=""):
    cols = ["bottom_error_mean_m", "bottom_error_median_m", "bottom_error_p95_m",
            "mean_abs_step_m", "n_big_jumps", "coverage"]
    hdr = ["method"] + cols
    if title:
        print(f"\n[{title}]")
    print("\n| " + " | ".join(hdr) + " |")
    print("|" + "|".join(["---"] * len(hdr)) + "|")
    for method in ALL_LINES:
        m = metrics[method]
        row = [method] + [
            f"{m['bottom_error_mean_m']:.3f}", f"{m['bottom_error_median_m']:.3f}",
            f"{m['bottom_error_p95_m']:.3f}", f"{m['mean_abs_step_m']:.3f}",
            str(m["n_big_jumps"]), f"{m['coverage']:.1%}",
        ]
        print("| " + " | ".join(row) + " |")


def plot_grid(records, names, method, set_name, out_dir):
    """3x3 grid for one method on a selection of files."""
    fig, axes = plt.subplots(3, 3, figsize=(24, 20))
    axes = axes.ravel()
    for i, ax in enumerate(axes):
        if i >= len(names):
            ax.set_visible(False)
            continue
        rec = records[names[i]]
        echo, range_arr = rec["echo"], rec["range_arr"]
        n_range, n_pings = echo.shape
        extent = [0, n_pings, range_arr[-1], range_arr[0]]
        ax.imshow(echo, aspect="auto", cmap="viridis", vmin=0, vmax=1,
                  origin="upper", extent=extent)

        pings = np.arange(n_pings)
        for line_key, color, alpha, lw in [
            ("gt", "red", 0.9, 1.4),
            ("baseline", "0.85", 0.5, 1.0),
            (method, "cyan", 0.95, 1.2),
        ]:
            idx = rec["lines"][line_key]
            valid = idx >= 0
            if valid.any():
                depths = range_arr[np.clip(idx[valid], 0, n_range - 1)]
                ax.plot(pings[valid], depths, color=color, linewidth=lw, alpha=alpha)

        err = rec["errors"][method]
        err_str = f"{np.mean(err):.2f} m" if len(err) else "n/a"
        ax.set_title(f"{rec['name']}: err {err_str}", fontsize=9)
        ax.set_ylabel("Range (m)", fontsize=7)
        ax.set_xlabel("Ping", fontsize=7)
        ax.tick_params(labelsize=7)

    handles = [
        Line2D([0], [0], color="red", lw=2, label="Ground truth"),
        Line2D([0], [0], color="0.85", lw=2, label="Baseline (first p>=0.5)"),
        Line2D([0], [0], color="cyan", lw=2, label=f"{method}"),
    ]
    fig.legend(handles=handles, loc="upper right", fontsize=12, framealpha=0.9)
    fig.suptitle(f"Bottom extraction: {method} ({set_name} 9)", fontsize=20)
    out = out_dir / f"{method}_{set_name}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out}")


def plot_param_sweep(sweep, x_label, title, out_path):
    """Twin-axis plot: mean / p95 error and big-jump count vs a parameter."""
    xs = sweep["values"]
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(xs, sweep["mean_m"], "o-", color="C0", label="mean error")
    ax1.plot(xs, sweep["p95_m"], "s--", color="C0", alpha=0.6, label="p95 error")
    ax1.set_xlabel(x_label)
    ax1.set_ylabel("Bottom-depth error (m)", color="C0")
    ax1.tick_params(axis="y", labelcolor="C0")

    ax2 = ax1.twinx()
    ax2.plot(xs, sweep["n_big_jumps"], "^-", color="C3", label="big jumps")
    ax2.set_ylabel("Big jumps (count)", color="C3")
    ax2.tick_params(axis="y", labelcolor="C3")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize=9)
    ax1.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out_path}")


def run_param_sweeps(per_file_probs, jump_threshold_m):
    """Threshold (baseline first-edge) and DP-smoothness sweeps over the set.

    ``per_file_probs`` is a list of (prob, gt_idx, range_arr) tuples.
    """
    def aggregate(lines_per_file):
        errs, steps, big = [], [], 0
        for line, gt_idx, range_arr in lines_per_file:
            e = bottom_depth_error(line, gt_idx, range_arr)
            if len(e):
                errs.append(e)
            ms, nb, _ = bottom_jumpiness(line, range_arr, jump_threshold_m)
            if not np.isnan(ms):
                steps.append(ms)
            big += nb
        errs = np.concatenate(errs) if errs else np.array([])
        return {
            "mean": float(np.mean(errs)) if len(errs) else float("nan"),
            "median": float(np.median(errs)) if len(errs) else float("nan"),
            "p95": float(np.percentile(errs, 95)) if len(errs) else float("nan"),
            "mean_abs_step_m": float(np.mean(steps)) if steps else float("nan"),
            "n_big_jumps": int(big),
        }

    thr_rows = []
    for thr in THRESHOLD_GRID:
        lines = [(extract_bottom_line(prob, threshold=thr), gt, ra)
                 for prob, gt, ra in per_file_probs]
        thr_rows.append(aggregate(lines))
    smooth_rows = []
    for s in SMOOTHNESS_GRID:
        lines = [(extract_bottom_dp(prob, smoothness=s), gt, ra)
                 for prob, gt, ra in per_file_probs]
        smooth_rows.append(aggregate(lines))

    def pack(values, rows):
        return {
            "values": list(values),
            "mean_m": [r["mean"] for r in rows],
            "median_m": [r["median"] for r in rows],
            "p95_m": [r["p95"] for r in rows],
            "mean_abs_step_m": [r["mean_abs_step_m"] for r in rows],
            "n_big_jumps": [r["n_big_jumps"] for r in rows],
        }

    return (pack(THRESHOLD_GRID, thr_rows),
            pack(SMOOTHNESS_GRID, smooth_rows))


def plot_summary_across_checkpoints(summary, out_path):
    """Grouped bars: mean bottom error and big jumps per method per checkpoint."""
    ckpts = list(summary.keys())
    methods = ALL_LINES
    n_m = len(methods)
    x = np.arange(n_m)
    width = 0.8 / max(1, len(ckpts))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7))
    for j, ck in enumerate(ckpts):
        m = summary[ck]["metrics"]
        means = [m[meth]["bottom_error_mean_m"] for meth in methods]
        jumps = [m[meth]["n_big_jumps"] for meth in methods]
        off = (j - (len(ckpts) - 1) / 2) * width
        ax1.bar(x + off, means, width, label=ck)
        ax2.bar(x + off, jumps, width, label=ck)

    for ax, ylabel, title in [
        (ax1, "Mean bottom-depth error (m)", "Mean error by method"),
        (ax2, "Big jumps (count)", "Inter-ping big jumps by method"),
    ]:
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=30, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
    ax2.legend(fontsize=8, loc="upper left")
    fig.suptitle("Bottom extraction methods across checkpoints", fontsize=16)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out_path}")


def write_readme(out_dir, summary):
    lines = ["# Bottom extraction methods", "",
             "Comparison of per-ping bottom-line extraction methods on the U-Net",
             "probability map: baseline (first pixel p>=0.5) vs contiguous run,",
             "temporal gating, DP/Viterbi (step-edge cost), Hampel de-spiking, and",
             "Otsu adaptive thresholding. Includes detection-threshold and DP",
             "smoothness sweeps, run across several checkpoints.", "",
             "- Keywords: bottom detection, seabed, U-Net, post-processing, jump suppression",
             "- Source: code/data_visualization/compare_bottom_extraction.py", ""]
    lines.append("## Checkpoints")
    for ck, d in summary.items():
        best = min(METHODS, key=lambda m: d["metrics"][m]["bottom_error_mean_m"])
        be = d["metrics"][best]["bottom_error_mean_m"]
        base = d["metrics"]["baseline"]["bottom_error_mean_m"]
        lines.append(f"- {ck} ({d['encoder']}, {d['n_files']} files): best '{best}' "
                     f"{be:.2f} m vs baseline {base:.2f} m")
    lines += ["", f"Generated: {datetime.now():%Y-%m-%d %H:%M}", ""]
    (out_dir / "_README.md").write_text("\n".join(lines))


def append_changelog(summary, primary_ckpt):
    cl = PROJECT_ROOT / "Latex" / "changelog.md"
    if not cl.exists():
        return
    d = summary[primary_ckpt]
    metrics = d["metrics"]
    best = min(METHODS, key=lambda m: metrics[m]["bottom_error_mean_m"])
    base, b = metrics["baseline"], metrics[best]
    entry = (
        f"\n## {datetime.now():%Y-%m-%d} - Bottom-line extraction sweep\n\n"
        f"Fixed the DP/Viterbi extractor (it had collapsed to the deepest row "
        f"because the 'below' masks make the bottom a filled half-plane; it now "
        f"uses a step-edge cost) and added Hampel de-spiking and Otsu adaptive "
        f"thresholding. Compared {len(ALL_LINES)} methods plus threshold and "
        f"smoothness sweeps across {len(summary)} checkpoints. On '{primary_ckpt}', "
        f"best method '{best}' cut mean bottom-depth error from "
        f"{base['bottom_error_mean_m']:.2f} m to {b['bottom_error_mean_m']:.2f} m "
        f"and big jumps from {base['n_big_jumps']} to {b['n_big_jumps']}.\n"
    )
    with cl.open("a") as f:
        f.write(entry)


def run_checkpoint(checkpoint, args, device, out_root):
    """Full method comparison + sweeps for one checkpoint. Returns metrics dict."""
    checkpoint = Path(checkpoint)
    encoder = encoder_from_checkpoint(checkpoint)
    out_dir = out_root / checkpoint.parent.name
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_root / "_cache" / checkpoint.parent.name
    cache_dir.mkdir(parents=True, exist_ok=True)

    eval_metrics = checkpoint.parent / "eval_metrics.json"
    names = [r["file"] for r in json.loads(eval_metrics.read_text())["per_file"]]
    paths = [ZARR_DIR / n for n in names if (ZARR_DIR / n).exists()]
    if args.max_files:
        paths = paths[:args.max_files]
    print(f"\n=== {checkpoint.parent.name} ({encoder}) | {len(paths)} eval files ===")

    model = None
    if not args.replot_only:
        model = load_model(checkpoint, device)

    per_file = []
    sweep_probs = []
    for i, zp in enumerate(paths):
        cp = cache_path(cache_dir, zp.stem)
        if model is None and not cp.exists():
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                prob, gt_mask, echo, range_arr = predict_and_cache(
                    model, zp, device, cache_dir)
        except Exception as e:  # noqa: BLE001
            print(f"  {zp.name}: FAILED - {e}")
            continue

        gt_idx = extract_bottom_line(gt_mask, threshold=0.5)
        lines = method_lines(prob, args)
        lines["gt"] = gt_idx
        errors = {m: bottom_depth_error(lines[m], gt_idx, range_arr)
                  for m in ALL_LINES}
        jump = {m: bottom_jumpiness(lines[m], range_arr, args.jump_threshold_m)
                for m in ALL_LINES}
        coverage = {m: float(np.mean(lines[m] >= 0)) for m in ALL_LINES}
        base_err = float(np.mean(errors["baseline"])) \
            if len(errors["baseline"]) else float("nan")

        per_file.append({
            "name": zp.stem, "base_err": base_err, "errors": errors,
            "jump": jump, "coverage": coverage,
            "echo": echo, "range_arr": range_arr, "lines": lines,
        })
        sweep_probs.append((np.asarray(prob), gt_idx, range_arr))
        if (i + 1) % 10 == 0:
            print(f"  processed {i + 1}/{len(paths)}")

    if not per_file:
        print("  No files processed (no cache and no model).")
        return None

    metrics = aggregate_metrics(per_file, args.jump_threshold_m)
    print_table(metrics, title=checkpoint.parent.name)

    thr_sweep, smooth_sweep = run_param_sweeps(sweep_probs, args.jump_threshold_m)
    plot_param_sweep(thr_sweep, "Detection threshold",
                     f"Threshold sweep (baseline first-edge): {checkpoint.parent.name}",
                     out_dir / "threshold_sweep.png")
    plot_param_sweep(smooth_sweep, "DP smoothness",
                     f"DP smoothness sweep: {checkpoint.parent.name}",
                     out_dir / "dp_smoothness_sweep.png")

    (out_dir / "metrics.json").write_text(json.dumps({
        "checkpoint": checkpoint.parent.name, "encoder": encoder,
        "n_files": len(per_file), "metrics": metrics,
        "threshold_sweep": thr_sweep, "dp_smoothness_sweep": smooth_sweep,
    }, indent=2))
    print(f"Saved -> {out_dir / 'metrics.json'}")

    ranked = sorted([r for r in per_file if not np.isnan(r["base_err"])],
                    key=lambda r: r["base_err"])
    n_sel = min(9, len(ranked))
    rng = random.Random(args.seed)
    sel = {
        "best": [r["name"] for r in ranked[:n_sel]],
        "worst": [r["name"] for r in ranked[-n_sel:]][::-1],
        "random": rng.sample([r["name"] for r in per_file], min(9, len(per_file))),
    }
    records = {r["name"]: r for r in per_file}
    for method in METHODS:
        for set_name in SETS:
            plot_grid(records, sel[set_name], method, set_name, out_dir)

    return {"encoder": encoder, "n_files": len(per_file), "metrics": metrics,
            "threshold_sweep": thr_sweep, "dp_smoothness_sweep": smooth_sweep}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", default=[str(DEFAULT_CHECKPOINT)],
                    help="one or more checkpoint .pth paths")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-files", type=int, default=None,
                    help="cap eval files (debug)")
    ap.add_argument("--min-run", type=int, default=5)
    ap.add_argument("--max-jump", type=int, default=15)
    ap.add_argument("--high-threshold", type=float, default=0.7)
    ap.add_argument("--smoothness", type=float, default=0.5)
    ap.add_argument("--hampel-window", type=int, default=15)
    ap.add_argument("--hampel-sigma", type=float, default=3.0)
    ap.add_argument("--jump-threshold-m", type=float, default=5.0)
    ap.add_argument("--output-dir", default=str(OUTPUT_DIR))
    ap.add_argument("--replot-only", action="store_true",
                    help="reuse cached probability maps, no model load")
    ap.add_argument("--no-changelog", action="store_true")
    args = ap.parse_args()

    apply_thesis_style()
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Checkpoints: {len(args.checkpoints)}")

    summary = {}
    for ckpt in args.checkpoints:
        result = run_checkpoint(ckpt, args, device, out_root)
        if result is not None:
            summary[Path(ckpt).parent.name] = result

    if not summary:
        print("No checkpoints produced results.")
        return

    (out_root / "summary_across_checkpoints.json").write_text(
        json.dumps(summary, indent=2))
    print(f"Saved -> {out_root / 'summary_across_checkpoints.json'}")
    if len(summary) >= 1:
        plot_summary_across_checkpoints(
            summary, out_root / "summary_across_checkpoints.png")

    write_readme(out_root, summary)
    if not args.no_changelog:
        append_changelog(summary, next(iter(summary)))
    print("\nDone.")


if __name__ == "__main__":
    main()
