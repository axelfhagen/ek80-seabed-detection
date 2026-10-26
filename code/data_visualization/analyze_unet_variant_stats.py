"""Paired statistical comparison of U-Net training variants.

Pairs per-file metrics (from eval_metrics.json of checkpoints evaluated on the
same held-out set) and runs paired t-tests + Wilcoxon between variants, with
Holm correction across pairs. Tests whether config choices differ significantly,
not whether the U-Net matches its own training labels.

Output: Figures/statistical_tests/
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from data_visualization.thesis_style import apply_thesis_style, savefig

CKPT = PROJECT_ROOT / "code" / "checkpoints"
OUTPUT_DIR = PROJECT_ROOT / "Figures" / "statistical_tests"

# Comparison groups: variant -> (label, checkpoint dir). All variants in a group
# must have been evaluated on the same held-out file set.
GROUPS = {
    "freeze_strategy": [
        ("frozen encoder", "exp_20260417_130130_cosine"),
        ("staged unfreeze", "exp_20260420_080950_finetune_stages"),
        ("full finetune", "exp_20260420_091726_full_finetune"),
    ],
    "encoder": [
        ("ResNet-34", "exp_20260417_130130_cosine"),
        ("ResNet-50", "exp_20260420_093231_resnet50"),
    ],
    "loss": [
        ("dice-heavy", "exp_20260423_165114_loss_01_dice_heavy"),
        ("bce-heavy", "exp_20260423_165114_loss_02_bce_heavy"),
        ("focal", "exp_20260423_165114_loss_03_focal"),
        ("focal-aggr", "exp_20260423_165114_loss_04_focal_aggressive"),
        ("focal-dice", "exp_20260423_165114_loss_05_focal_dice"),
    ],
}

METRICS = {
    "bottom_error": ("bottom_error_mean_m", "bottom depth error (m)", True),
    "iou": ("iou", "IoU", False),
    "dice": ("dice", "Dice", False),
}


def load_per_file(exp_dir: str, json_key: str) -> dict[str, float]:
    data = json.load(open(CKPT / exp_dir / "eval_metrics.json"))
    return {r["file"]: float(r[json_key]) for r in data["per_file"]}


def paired_arrays(a, b):
    keys = sorted(set(a) & set(b))
    return keys, np.array([a[k] for k in keys]), np.array([b[k] for k in keys])


def cohen_dz(diff):
    sd = diff.std(ddof=1)
    return float(diff.mean() / sd) if sd > 0 else 0.0


def holm(pvals):
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * pvals[idx])
        adj[idx] = min(running, 1.0)
    return adj.tolist()


def analyze_group(variants, json_key):
    series = {lab: load_per_file(d, json_key) for lab, d in variants}
    labels = [lab for lab, _ in variants]
    raw_p, results = [], []
    for x, y in combinations(labels, 2):
        keys, ax, by = paired_arrays(series[x], series[y])
        diff = ax - by
        _, p_t = stats.ttest_rel(ax, by)
        try:
            _, p_w = stats.wilcoxon(ax, by)
        except ValueError:
            p_w = np.nan
        results.append({"x": x, "y": y, "n": len(keys),
                        "mean_diff": diff.mean(),
                        "ci": 1.96 * diff.std(ddof=1) / np.sqrt(len(keys)),
                        "dz": cohen_dz(diff), "p_t": p_t, "p_w": p_w})
        raw_p.append(p_t)
    for r, padj in zip(results, holm(raw_p)):
        r["p_t_holm"] = padj
    return series, labels, results


def plot_group(name, series, labels, results, metric_label, lower_better):
    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    fig.suptitle(f"U-Net variant comparison: {name.replace('_', ' ')}")

    # Panel 1: per-variant distribution
    ax = axes[0]
    data = [list(series[l].values()) for l in labels]
    bp = ax.boxplot(data, tick_labels=labels, showmeans=True, widths=0.55,
                    patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#cfe3f0"); patch.set_alpha(0.8)
    rng = np.random.default_rng(0)
    for i, vals in enumerate(data):
        ax.scatter(rng.normal(i + 1, 0.05, len(vals)), vals, s=20,
                   color="#1f5f80", alpha=0.6, zorder=3)
    ax.set_ylabel(metric_label)
    ax.set_title("per-file distribution")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.3)

    # Panel 2: pairwise matrix (lower: Holm p, upper: |dz|)
    ax = axes[1]
    n = len(labels)
    M = np.full((n, n), np.nan)
    annot = np.empty((n, n), dtype=object)
    idx = {l: i for i, l in enumerate(labels)}
    for r in results:
        i, j = idx[r["x"]], idx[r["y"]]
        M[j, i] = r["p_t_holm"]; annot[j, i] = f"p={r['p_t_holm']:.3f}"
        M[i, j] = abs(r["dz"]); annot[i, j] = f"dz={r['dz']:.2f}"
    lower = np.where(np.tril(np.ones_like(M), -1).astype(bool),
                     np.clip(1 - M, 0, 1), np.nan)
    upper = np.where(np.triu(np.ones_like(M), 1).astype(bool),
                     np.clip(M, 0, 2) / 2, np.nan)
    ax.imshow(lower, cmap="Reds", vmin=0, vmax=1)
    ax.imshow(upper, cmap="Blues", vmin=0, vmax=1)
    for i in range(n):
        for j in range(n):
            if annot[i, j] is not None:
                sig = (i > j and M[i, j] < 0.05)
                ax.text(j, i, annot[i, j], ha="center", va="center",
                        fontsize=13, fontweight="bold" if sig else "normal")
    ax.set_xticks(range(n)); ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_yticks(range(n)); ax.set_yticklabels(labels)
    ax.set_title("Holm p (lower), effect size (upper)")

    # Panel 3: paired slope plot for largest-effect pair
    ax = axes[2]
    head = max(results, key=lambda r: abs(r["dz"]))
    x, y = head["x"], head["y"]
    _, av, bv = paired_arrays(series[x], series[y])
    for a_val, b_val in zip(av, bv):
        worse = (a_val > b_val) if lower_better else (a_val < b_val)
        ax.plot([0, 1], [a_val, b_val], "-o", ms=5, lw=1.0,
                color="#c0392b" if worse else "#27ae60", alpha=0.55)
    ax.plot([0, 1], [av.mean(), bv.mean()], "-s", color="black", lw=3,
            ms=11, zorder=5)
    ax.set_xticks([0, 1]); ax.set_xticklabels([x, y])
    ax.set_ylabel(metric_label)
    sig = "***" if head["p_t_holm"] < 0.001 else "**" if head["p_t_holm"] < 0.01 \
        else "*" if head["p_t_holm"] < 0.05 else "n.s."
    ax.set_title(f"largest effect: {x} vs {y}")
    ax.text(0.5, 0.98,
            f"n={head['n']}   dz={head['dz']:.2f}\n"
            f"mean diff = {head['mean_diff']:+.3f} +/- {head['ci']:.3f} m\n"
            f"t-test p={head['p_t_holm']:.4f} ({sig})   "
            f"Wilcoxon p={head['p_w']:.4f}",
            transform=ax.transAxes, va="top", ha="center", fontsize=13,
            bbox=dict(boxstyle="round", fc="white", alpha=0.85))
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = OUTPUT_DIR / f"unet_variants_{name}.pdf"
    savefig(fig, out)
    plt.close(fig)
    return out


def main():
    global OUTPUT_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", default="bottom_error", choices=list(METRICS))
    ap.add_argument("--output-dir", default=str(OUTPUT_DIR))
    args = ap.parse_args()
    apply_thesis_style()

    OUTPUT_DIR = Path(args.output_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    json_key, metric_label, lower_better = METRICS[args.metric]
    print(f"\nMetric: {metric_label}  ({'lower' if lower_better else 'higher'} is better)\n")
    summary = []
    for name, variants in GROUPS.items():
        try:
            series, labels, results = analyze_group(variants, json_key)
        except FileNotFoundError as e:
            print(f"[skip] {name}: {e}"); continue
        out = plot_group(name, series, labels, results, metric_label, lower_better)
        print(f"=== {name} ===")
        for r in results:
            star = "*" if r["p_t_holm"] < 0.05 else " "
            print(f"  {r['x']:>16s} vs {r['y']:<16s}  Δ={r['mean_diff']:+7.3f}  "
                  f"dz={r['dz']:+5.2f}  p_holm={r['p_t_holm']:.4f}{star}  "
                  f"p_wilcox={r['p_w']:.4f}")
            summary.append([name, r["x"], r["y"], r["n"], r["mean_diff"],
                            r["dz"], r["p_t"], r["p_t_holm"], r["p_w"]])
        print(f"  -> {out}\n")

    csv = OUTPUT_DIR / f"unet_variants_{args.metric}_stats.csv"
    with open(csv, "w") as f:
        f.write("group,variant_a,variant_b,n,mean_diff,cohen_dz,p_ttest,p_ttest_holm,p_wilcoxon\n")
        for row in summary:
            f.write(",".join(str(x) for x in row) + "\n")
    print(f"Stats table -> {csv}")


if __name__ == "__main__":
    main()
