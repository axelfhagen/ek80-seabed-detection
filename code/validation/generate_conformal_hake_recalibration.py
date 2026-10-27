#!/usr/bin/env python
"""In-situ split-conformal recalibration on NOAA Hake EK80 files.

This is the companion to ``generate_conformal_hake.py``.  The older script
applies Bastoe-calibrated bands to deep-water Hake and shows that coverage does
not transfer.  This script asks the constructive follow-up question: if a small
held-out Hake calibration split is available, does split conformal recover the
target coverage on the remaining Hake files?

The split unit is the raw file.  Within each file, the algorithmic 38 kHz
bottom label is treated as the reference, matching ``eval_hake_test_raw.py``.
Per-file nonconformity scores are cached, so the GPU/IO pass is paid once and
all split-size sweeps are cheap.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import gc
import sys
import warnings
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "code"))
warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from data_visualization.eval_hake_test_raw import (
    RAW_DIR,
    DEFAULT_CHECKPOINT,
    STRIDE,
    TARGET_SIZE,
    PATCH_WIDTH,
    load_model,
    prepare_group,
)
from deep_learning.unet_model import predict_full_echogram
from validation.conformal import (
    calibrate_depth,
    calibrate_locally_weighted,
    calibrate_profile,
    conformal_quantile,
    compute_transition_width,
    extract_bottom_depths,
    predict_profile_band,
)

ALPHA = 0.10
EPSILON = 0.5
OUT_DIR = PROJECT_ROOT / "Figures" / "conformal_hake_recalibration"
CACHE_DIR = OUT_DIR / "cache"
BASTO_CACHE_DIR = PROJECT_ROOT / "Figures" / "conformal_hake_deep_water" / "cache"

METHODS = {
    "constant": "Constant-width",
    "local": "Locally-weighted",
    "profile": "Profile-based",
}


def raw_stems(limit: int | None = None) -> list[str]:
    stems = [p.stem for p in sorted(RAW_DIR.glob("*.raw")) if not p.name.startswith("._")]
    return stems[:limit] if limit else stems


def ref_depths_from_label_idx(label_idx_crop: np.ndarray, range_crop: np.ndarray) -> np.ndarray:
    idx = np.asarray(label_idx_crop)
    out = np.full(idx.shape, np.nan, dtype=np.float64)
    valid = idx >= 0
    out[valid] = range_crop[np.clip(idx[valid], 0, len(range_crop) - 1)]
    return out


def score_one_file(stem: str, model, device: str, recompute: bool = False) -> Path | None:
    cache = CACHE_DIR / f"{stem}_scores.npz"
    if cache.exists() and not recompute:
        return cache

    print(f"  scoring {stem}", flush=True)
    try:
        res = prepare_group([stem])
        prob = predict_full_echogram(
            model,
            res["sv_channels"],
            TARGET_SIZE,
            device,
            patch_width=PATCH_WIDTH,
            stride=STRIDE,
        )
        range_crop = res["range_crop"]
        pred_d = extract_bottom_depths(prob, range_crop, threshold=0.5)
        ref_d = ref_depths_from_label_idx(res["label_idx_crop"], range_crop)
        sigma = compute_transition_width(prob, range_crop)
        depth_scores = calibrate_depth(pred_d, ref_d)
        local_scores = calibrate_locally_weighted(pred_d, ref_d, sigma, epsilon=EPSILON)
        profile_scores = calibrate_profile([prob], [res["label_mask"]])
        pu, pl, pw = predict_profile_band(prob, range_crop, 0.5)
        valid_profile = np.isfinite(pu) & np.isfinite(pl) & np.isfinite(ref_d) & (ref_d > 0)
        profile_widths = pw[valid_profile]
        valid_local = (
            np.isfinite(pred_d)
            & np.isfinite(ref_d)
            & np.isfinite(sigma)
            & (pred_d > 0)
            & (ref_d > 0)
        )
        local_sigmas = np.maximum(sigma[valid_local], EPSILON)
    except Exception as exc:
        print(f"    skip {stem}: {exc}", flush=True)
        return None

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache,
        stem=stem,
        n_pings=np.array([prob.shape[1]], dtype=np.int64),
        depth_scores=depth_scores.astype(np.float32),
        local_scores=local_scores.astype(np.float32),
        local_sigmas=local_sigmas.astype(np.float32),
        profile_scores=profile_scores.astype(np.float32),
        profile_widths=profile_widths.astype(np.float32),
        pred_depths=pred_d.astype(np.float32),
        ref_depths=ref_d.astype(np.float32),
    )
    del res, prob, pred_d, ref_d, sigma
    gc.collect()
    return cache


def build_score_cache(stems: list[str], model, device: str, recompute: bool) -> list[Path]:
    paths = []
    for i, stem in enumerate(stems, 1):
        if i % 10 == 1 or recompute:
            print(f"[score {i}/{len(stems)}]", flush=True)
        path = score_one_file(stem, model, device, recompute=recompute)
        if path is not None:
            paths.append(path)
    return paths


def load_scores(paths: list[Path]) -> dict[str, dict[str, np.ndarray]]:
    out = {}
    for path in paths:
        with np.load(path, allow_pickle=False) as d:
            stem = str(d["stem"])
            out[stem] = {
                "constant": d["depth_scores"].astype(np.float64),
                "local": d["local_scores"].astype(np.float64),
                "local_sigmas": d["local_sigmas"].astype(np.float64),
                "profile": d["profile_scores"].astype(np.float64),
                "profile_widths": d["profile_widths"].astype(np.float64),
                "n_pings": d["n_pings"].astype(np.int64),
            }
    return out


def concat_scores(scores_by_file, files, key):
    arrays = [scores_by_file[f][key] for f in files if scores_by_file[f][key].size]
    if not arrays:
        return np.array([], dtype=np.float64)
    return np.concatenate(arrays)


def eval_method(scores_by_file, files, method: str, q_hat: float) -> dict[str, float]:
    per_file_cov = []
    per_file_n = []
    covered_total = 0
    n_total = 0
    widths = []
    for f in files:
        s = scores_by_file[f][method]
        if s.size == 0:
            continue
        covered = s <= q_hat
        per_file_cov.append(float(covered.mean()))
        per_file_n.append(int(s.size))
        covered_total += int(covered.sum())
        n_total += int(s.size)
        if method == "constant":
            widths.append(np.full(s.size, 2.0 * q_hat, dtype=np.float64))
        elif method == "local":
            widths.append(2.0 * q_hat * scores_by_file[f]["local_sigmas"])
        elif method == "profile":
            # The cached profile score is sufficient for coverage at any q_hat,
            # but the exact profile width depends on re-thresholding the full
            # probability map. Avoid reporting a stale q=0.5 width here.
            pass
    width_arr = np.concatenate(widths) if widths else np.array([], dtype=np.float64)
    return {
        "coverage": covered_total / n_total if n_total else float("nan"),
        "n": n_total,
        "n_files": len(per_file_cov),
        "file_mean": float(np.mean(per_file_cov)) if per_file_cov else float("nan"),
        "file_sd": float(np.std(per_file_cov, ddof=1)) if len(per_file_cov) > 1 else float("nan"),
        "median_width_m": float(np.nanmedian(width_arr)) if width_arr.size else float("nan"),
        "mean_width_m": float(np.nanmean(width_arr)) if width_arr.size else float("nan"),
    }


def bootstrap_file_ci(scores_by_file, files, method: str, q_hat: float, rng, n_boot: int):
    vals = []
    files = list(files)
    for _ in range(n_boot):
        sample = list(rng.choice(files, size=len(files), replace=True))
        vals.append(eval_method(scores_by_file, sample, method, q_hat)["coverage"])
    return tuple(float(x) for x in np.nanpercentile(vals, [2.5, 97.5]))


def split_files(files: list[str], n_calib: int, seed: int) -> tuple[list[str], list[str]]:
    rng = np.random.default_rng(seed)
    perm = list(rng.permutation(files))
    n = min(n_calib, max(1, len(perm) - 1))
    calib = sorted(perm[:n])
    test = sorted(perm[n:])
    return calib, test


def find_basto_cache(checkpoint: str, explicit: str | None) -> Path | None:
    if explicit:
        p = Path(explicit)
        return p if p.exists() else None
    ckpt_name = Path(checkpoint).parent.name
    matches = sorted(BASTO_CACHE_DIR.glob(f"calib_{ckpt_name}_n*_s0.npz"))
    if matches:
        return matches[-1]
    matches = sorted(BASTO_CACHE_DIR.glob("calib_*_n*_s0.npz"))
    return matches[-1] if matches else None


def load_basto_thresholds(cache_path: Path | None, alpha: float) -> dict[str, float]:
    if cache_path is None:
        return {}
    with np.load(cache_path, allow_pickle=False) as d:
        out = {}
        if "depth_scores" in d:
            out["constant"] = conformal_quantile(d["depth_scores"], alpha)
        if "local_scores" in d:
            out["local"] = conformal_quantile(d["local_scores"], alpha)
        if "profile_scores" in d:
            out["profile"] = conformal_quantile(d["profile_scores"], alpha)
        return out


def run_primary_split(scores_by_file, files, n_calib, seed, alpha, n_boot, basto_q):
    calib, test = split_files(files, n_calib, seed)
    rng = np.random.default_rng(seed + 10_000)
    rows = []
    for method in METHODS:
        calib_scores = concat_scores(scores_by_file, calib, method)
        if calib_scores.size == 0:
            continue
        q_hat = conformal_quantile(calib_scores, alpha)
        calib_eval = eval_method(scores_by_file, calib, method, q_hat)
        test_eval = eval_method(scores_by_file, test, method, q_hat)
        ci_lo, ci_hi = bootstrap_file_ci(scores_by_file, test, method, q_hat, rng, n_boot=n_boot)
        rows.append(
            {
                "source": "hake_recalibrated",
                "method": method,
                "q_hat": q_hat,
                "calib_files": len(calib),
                "test_files": len(test),
                "calib_coverage": calib_eval["coverage"],
                "test_coverage": test_eval["coverage"],
                "test_ci_lo": ci_lo,
                "test_ci_hi": ci_hi,
                "test_n": test_eval["n"],
                "test_file_mean": test_eval["file_mean"],
                "test_file_sd": test_eval["file_sd"],
                "test_median_width_m": test_eval["median_width_m"],
                "test_mean_width_m": test_eval["mean_width_m"],
            }
        )
        if method in basto_q:
            z = eval_method(scores_by_file, test, method, basto_q[method])
            zlo, zhi = bootstrap_file_ci(
                scores_by_file, test, method, basto_q[method], rng, n_boot=n_boot
            )
            rows.append(
                {
                    "source": "basto_zero_shot",
                    "method": method,
                    "q_hat": basto_q[method],
                    "calib_files": 0,
                    "test_files": len(test),
                    "calib_coverage": float("nan"),
                    "test_coverage": z["coverage"],
                    "test_ci_lo": zlo,
                    "test_ci_hi": zhi,
                    "test_n": z["n"],
                    "test_file_mean": z["file_mean"],
                    "test_file_sd": z["file_sd"],
                    "test_median_width_m": z["median_width_m"],
                    "test_mean_width_m": z["mean_width_m"],
                }
            )
    return calib, test, rows


def run_size_sweep(scores_by_file, files, sizes, repeats, seed, alpha):
    rows = []
    for n_calib in sizes:
        for r in range(repeats):
            split_seed = seed + r
            calib, test = split_files(files, n_calib, split_seed)
            for method in METHODS:
                c = concat_scores(scores_by_file, calib, method)
                if c.size == 0:
                    continue
                q_hat = conformal_quantile(c, alpha)
                e = eval_method(scores_by_file, test, method, q_hat)
                rows.append(
                    {
                        "n_calib": len(calib),
                        "repeat": r,
                        "seed": split_seed,
                        "method": method,
                        "q_hat": q_hat,
                        "coverage": e["coverage"],
                        "test_files": len(test),
                        "test_n": e["n"],
                        "file_mean": e["file_mean"],
                        "file_sd": e["file_sd"],
                        "median_width_m": e["median_width_m"],
                    }
                )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_primary(rows, save_path: Path, alpha: float):
    fig, ax = plt.subplots(figsize=(8, 4.8))
    labels = []
    vals = []
    err_lo = []
    err_hi = []
    colors = []
    for method in METHODS:
        for source, color in (("basto_zero_shot", "#d95f02"), ("hake_recalibrated", "#1b9e77")):
            row = next((r for r in rows if r["method"] == method and r["source"] == source), None)
            if row is None:
                continue
            labels.append(f"{METHODS[method]}\n{source.replace('_', ' ')}")
            vals.append(row["test_coverage"])
            err_lo.append(row["test_coverage"] - row["test_ci_lo"])
            err_hi.append(row["test_ci_hi"] - row["test_coverage"])
            colors.append(color)
    x = np.arange(len(vals))
    ax.bar(x, vals, color=colors, edgecolor="black", linewidth=0.6)
    ax.errorbar(x, vals, yerr=[err_lo, err_hi], fmt="none", color="black", capsize=3)
    ax.axhline(1.0 - alpha, color="black", linestyle="--", linewidth=1.4)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Held-out Hake coverage")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_sweep(rows, save_path: Path, alpha: float):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = {"constant": "#1b9e77", "local": "#7570b3", "profile": "#66a61e"}
    sizes = sorted({r["n_calib"] for r in rows})
    for method in METHODS:
        means, lo, hi = [], [], []
        for n in sizes:
            vals = np.array(
                [r["coverage"] for r in rows if r["method"] == method and r["n_calib"] == n]
            )
            means.append(float(np.nanmean(vals)))
            lo.append(float(np.nanpercentile(vals, 10)))
            hi.append(float(np.nanpercentile(vals, 90)))
        means = np.array(means)
        lo = np.array(lo)
        hi = np.array(hi)
        ax.plot(sizes, means, marker="o", color=colors[method], label=METHODS[method])
        ax.fill_between(sizes, lo, hi, color=colors[method], alpha=0.15)
    ax.axhline(1.0 - alpha, color="black", linestyle="--", linewidth=1.4)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Hake calibration files")
    ax.set_ylabel("Held-out Hake coverage")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_readme(
    rows,
    sweep_rows,
    files,
    calib,
    test,
    checkpoint,
    basto_cache,
    alpha,
    save_path: Path,
):
    def fmt(x):
        return "nan" if not np.isfinite(x) else f"{x:.3f}"

    primary = [r for r in rows if r["source"] == "hake_recalibrated" and r["method"] == "constant"][
        0
    ]
    zero = next(
        (r for r in rows if r["source"] == "basto_zero_shot" and r["method"] == "constant"),
        None,
    )
    lines = []
    for r in rows:
        lines.append(
            f"  {r['source']} / {METHODS[r['method']]}: q={r['q_hat']:.3f}, "
            f"test coverage {fmt(r['test_coverage'])} "
            f"[{fmt(r['test_ci_lo'])}, {fmt(r['test_ci_hi'])}], "
            f"file sd {fmt(r['test_file_sd'])}, median width {fmt(r['test_median_width_m'])} m"
        )
    sweep_summary = []
    for n in sorted({r["n_calib"] for r in sweep_rows}):
        vals = [
            r["coverage"] for r in sweep_rows if r["method"] == "constant" and r["n_calib"] == n
        ]
        sweep_summary.append(
            f"  n_calib={n}: constant-width mean coverage {np.nanmean(vals):.3f} "
            f"(10-90% {np.nanpercentile(vals, 10):.3f}-{np.nanpercentile(vals, 90):.3f})"
        )

    save_path.write_text(
        f"""# Hake in-situ conformal recalibration

conformal  hake  recalibration  split-conformal  coverage

Split-conformal recalibration on NOAA Hake EK80 files using the algorithmic
38 kHz bottom label as reference. The split unit is the raw file; pings within
a file are aggregated only after assigning the entire file to calibration or
test. Target coverage is {1 - alpha:.0%}. Checkpoint:
{Path(checkpoint).parent.name}.

Source: code/validation/generate_conformal_hake_recalibration.py
Created: {dt.datetime.now():%Y-%m-%d %H:%M}

Data: {len(files)} scored Hake files from {RAW_DIR}; primary split uses
{len(calib)} calibration files and {len(test)} held-out test files.
Bastoe zero-shot cache: {basto_cache if basto_cache else "not available"}.

Headline constant-width result:
  Hake recalibrated q={primary["q_hat"]:.3f} m, held-out coverage
  {primary["test_coverage"]:.3f} over {primary["test_n"]} pings
  ({primary["test_files"]} files), file-bootstrap 95% CI
  [{primary["test_ci_lo"]:.3f}, {primary["test_ci_hi"]:.3f}].
"""
        + (
            f"  Bastoe-fitted q={zero['q_hat']:.3f} m gives held-out Hake coverage "
            f"{zero['test_coverage']:.3f} on the same test files.\n"
            if zero
            else ""
        )
        + """
Primary split all methods:
"""
        + "\n".join(lines)
        + """

Calibration-size sensitivity, constant-width:
"""
        + "\n".join(sweep_summary)
        + """

Outputs:
  primary_results.csv
  calibration_size_sweep.csv
  primary_coverage.png
  calibration_size_sweep.png
  cache/*_scores.npz
"""
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-calib", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--alpha", type=float, default=ALPHA)
    ap.add_argument("--max-files", type=int, default=None)
    ap.add_argument("--sweep-sizes", type=int, nargs="*", default=[10, 20, 30])
    ap.add_argument("--sweep-repeats", type=int, default=30)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    ap.add_argument("--device", default=None)
    ap.add_argument("--recompute-cache", action="store_true")
    ap.add_argument("--from-cache", action="store_true")
    ap.add_argument("--basto-calib-cache", default=None)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    stems = raw_stems(args.max_files)
    print(f"Hake raw files requested: {len(stems)}")
    print(f"Device: {device}")
    print(f"Checkpoint: {Path(args.checkpoint).parent.name}")

    if args.from_cache:
        paths = [
            CACHE_DIR / f"{s}_scores.npz" for s in stems if (CACHE_DIR / f"{s}_scores.npz").exists()
        ]
    else:
        model = load_model(args.checkpoint, device)
        paths = build_score_cache(stems, model, device, recompute=args.recompute_cache)

    scores_by_file = load_scores(paths)
    all_scored_files = sorted(scores_by_file)
    files = [f for f in all_scored_files if scores_by_file[f]["constant"].size > 0]
    empty_files = [f for f in all_scored_files if scores_by_file[f]["constant"].size == 0]
    if len(files) <= args.n_calib:
        raise SystemExit(f"Need more than {args.n_calib} scored files; found {len(files)}")
    print(f"Scored files available: {len(all_scored_files)}")
    print(f"Files with valid nonconformity scores: {len(files)}")
    if empty_files:
        print(f"Files omitted because they have zero valid score pings: {len(empty_files)}")

    basto_cache = find_basto_cache(args.checkpoint, args.basto_calib_cache)
    basto_q = load_basto_thresholds(basto_cache, args.alpha)
    if basto_cache:
        print(f"Bastoe zero-shot cache: {basto_cache}")
    else:
        print("Bastoe zero-shot cache unavailable; only Hake-recalibrated rows will be written.")

    calib, test, primary_rows = run_primary_split(
        scores_by_file,
        files,
        args.n_calib,
        args.seed,
        args.alpha,
        args.bootstrap,
        basto_q,
    )
    sweep_rows = run_size_sweep(
        scores_by_file,
        files,
        args.sweep_sizes,
        args.sweep_repeats,
        args.seed,
        args.alpha,
    )

    write_csv(OUT_DIR / "primary_results.csv", primary_rows)
    write_csv(OUT_DIR / "calibration_size_sweep.csv", sweep_rows)
    plot_primary(primary_rows, OUT_DIR / "primary_coverage.png", args.alpha)
    plot_sweep(sweep_rows, OUT_DIR / "calibration_size_sweep.png", args.alpha)
    write_readme(
        primary_rows,
        sweep_rows,
        files,
        calib,
        test,
        args.checkpoint,
        basto_cache,
        args.alpha,
        OUT_DIR / "_README.md",
    )

    print("\nPrimary split:")
    for row in primary_rows:
        print(
            f"  {row['source']:17s} {row['method']:8s} "
            f"q={row['q_hat']:.3f} coverage={row['test_coverage']:.3f} "
            f"CI=[{row['test_ci_lo']:.3f},{row['test_ci_hi']:.3f}] "
            f"n={row['test_n']}"
        )
    print(f"Done -> {OUT_DIR}")


if __name__ == "__main__":
    main()
