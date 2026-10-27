#!/usr/bin/env python3
"""Post-process the Hake fine-tune experiment outputs.

This script deliberately works from CSV/JSON artifacts only. It does not load or
modify model checkpoints, so it is safe to rerun after the Slurm job finishes.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
FIG_DIR = ROOT / "Figures" / "hake_finetune_chunk_arch_sweep_2"
CKPT_DIR = ROOT / "code" / "checkpoints" / "hake_finetune_chunk_arch_sweep_2_seed0"
BASE_CKPT = ROOT / "code" / "checkpoints" / "chunk_arch_sweep_2" / "smp_unet_resnet34_best.pth"
MANIFEST = CKPT_DIR / "hake_split_manifest.json"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def f(row: dict[str, str], key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def finite_mean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else float("nan")


def finite_std(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(arr.std()) if arr.size else float("nan")


def summarize_hake() -> dict[str, dict[str, float]]:
    rows = read_rows(FIG_DIR / "hake_test_metrics.csv")
    summary: dict[str, dict[str, float]] = {}
    for model in ("original", "finetuned"):
        selected = [row for row in rows if row["model"] == model]
        summary[model] = {
            "n_groups": len(selected),
            "mean_iou": finite_mean([f(r, "iou") for r in selected]),
            "mean_dice": finite_mean([f(r, "dice") for r in selected]),
            "mean_precision": finite_mean([f(r, "precision") for r in selected]),
            "mean_recall": finite_mean([f(r, "recall") for r in selected]),
            "mean_depth_mae_m": finite_mean([f(r, "depth_mae_m") for r in selected]),
            "median_depth_mae_m": float(np.nanmedian([f(r, "depth_mae_m") for r in selected])),
            "mean_detect_frac": finite_mean([f(r, "detect_frac") for r in selected]),
        }
    (FIG_DIR / "hake_test_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def summarize_basto() -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for model in ("original", "finetuned"):
        rows = read_rows(FIG_DIR / f"basto_test_metrics_{model}.csv")
        depth = [f(r, "bottom_error_mean_m") for r in rows]
        valid_depth = [v for v in depth if np.isfinite(v)]
        summary[model] = {
            "n_files": len(rows),
            "n_files_depth_valid": len(valid_depth),
            "mean_iou": finite_mean([f(r, "iou") for r in rows]),
            "mean_dice": finite_mean([f(r, "dice") for r in rows]),
            "mean_precision": finite_mean([f(r, "precision") for r in rows]),
            "mean_recall": finite_mean([f(r, "recall") for r in rows]),
            "mean_bottom_error_m": finite_mean(depth),
            "std_iou": finite_std([f(r, "iou") for r in rows]),
        }
    (FIG_DIR / "basto_test_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def delta(after: float, before: float) -> float:
    return float(after - before)


def plot_metric_panels(hake: dict, basto: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3))
    specs = [
        (
            axes[0],
            "Hake test",
            hake,
            ["mean_iou", "mean_dice", "mean_depth_mae_m"],
            ["IoU", "Dice", "Depth MAE (m)"],
        ),
        (
            axes[1],
            "Basto test",
            basto,
            ["mean_iou", "mean_dice", "mean_bottom_error_m"],
            ["IoU", "Dice", "Depth MAE (m)"],
        ),
    ]
    colors = {"original": "#4c78a8", "finetuned": "#f58518"}
    for ax, title, summary, keys, labels in specs:
        x = np.arange(len(keys), dtype=float)
        width = 0.34
        for offset, model in ((-width / 2, "original"), (width / 2, "finetuned")):
            values = [summary[model][key] for key in keys]
            bars = ax.bar(x + offset, values, width, label=model, color=colors[model])
            for bar, value in zip(bars, values):
                label = f"{value:.3f}" if value < 10 else f"{value:.1f}"
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    label,
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
        ax.set_title(title)
        ax.set_xticks(x, labels)
        ax.grid(True, axis="y", alpha=0.25)
        ax.margins(y=0.18)
    axes[0].legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "metric_summary_panels.png", dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_split_overview() -> None:
    if not MANIFEST.exists():
        return
    manifest = json.loads(MANIFEST.read_text())
    counts = manifest["counts"]
    labels = ["train", "val", "calibration", "test"]
    values = [counts[label] for label in labels]
    colors = ["#54a24b", "#4c78a8", "#f58518", "#e45756"]

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    bars = ax.bar(labels, values, color=colors)
    for bar, value in zip(bars, values):
        pct = value / sum(values) * 100
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value}\n{pct:.0f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_ylabel("Hake files")
    ax.set_title("Hake split used for isolated fine-tuning experiment")
    ax.grid(True, axis="y", alpha=0.25)
    ax.margins(y=0.18)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "hake_split_overview.png", dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_conformal_hake_methods() -> None:
    path = FIG_DIR / "conformal_results.csv"
    if not path.exists():
        return
    rows = [
        r
        for r in read_rows(path)
        if r["model"] == "finetuned"
        and r["test_domain"] == "hake"
        and r["calibration_domain"] in ("basto", "hake", "balanced_pooled")
    ]
    order = {
        ("basto", "constant"): 0,
        ("basto", "local"): 1,
        ("basto", "profile"): 2,
        ("hake", "constant"): 3,
        ("hake", "local"): 4,
        ("hake", "profile"): 5,
        ("balanced_pooled", "constant"): 6,
        ("balanced_pooled", "local"): 7,
        ("balanced_pooled", "profile"): 8,
    }
    rows.sort(key=lambda r: order.get((r["calibration_domain"], r["method"]), 99))

    def short_domain(name: str) -> str:
        return "balanced" if name == "balanced_pooled" else name

    labels = [f"{short_domain(r['calibration_domain'])}\n{r['method']}" for r in rows]
    values = [f(r, "coverage") for r in rows]
    widths = [f(r, "mean_width_m") for r in rows]
    colors = [
        "#4c78a8"
        if r["method"] == "constant"
        else "#f58518"
        if r["method"] == "local"
        else "#54a24b"
        for r in rows
    ]

    fig, ax = plt.subplots(figsize=(8.8, 4.2))
    bars = ax.bar(np.arange(len(rows)), values, color=colors)
    ax.axhline(0.9, color="black", linestyle="--", linewidth=1.0)
    for bar, value, width in zip(bars, values, widths):
        width_text = "" if not np.isfinite(width) else f"\n{width:.1f}m"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.3f}{width_text}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Hake empirical coverage")
    ax.set_xticks(np.arange(len(rows)), labels, fontsize=8)
    ax.set_title("Fine-tuned model: Hake conformal coverage by calibration source")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "finetuned_hake_conformal_methods.png", dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_hake_group_deltas() -> None:
    rows = read_rows(FIG_DIR / "hake_test_metrics.csv")
    groups = sorted({row["group"] for row in rows})
    original = {row["group"]: row for row in rows if row["model"] == "original"}
    tuned = {row["group"]: row for row in rows if row["model"] == "finetuned"}
    iou_delta = [delta(f(tuned[g], "iou"), f(original[g], "iou")) for g in groups]
    mae_delta = [delta(f(tuned[g], "depth_mae_m"), f(original[g], "depth_mae_m")) for g in groups]

    labels = [g.replace("+2more", "+2") for g in groups]
    y = np.arange(len(groups), dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.7), sharey=True)
    axes[0].barh(y, iou_delta, color="#54a24b")
    axes[0].axvline(0, color="black", linewidth=0.8)
    axes[0].set_title("IoU change")
    axes[0].set_xlabel("fine-tuned - original")
    axes[0].set_yticks(y, labels)
    axes[0].grid(True, axis="x", alpha=0.25)

    axes[1].barh(y, mae_delta, color=["#54a24b" if v < 0 else "#e45756" for v in mae_delta])
    axes[1].axvline(0, color="black", linewidth=0.8)
    axes[1].set_title("Depth MAE change")
    axes[1].set_xlabel("meters; negative is better")
    axes[1].grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "hake_group_delta_bars.png", dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_effect_summary(hake: dict, basto: dict) -> None:
    items = [
        ("Hake IoU", (hake["finetuned"]["mean_iou"] / hake["original"]["mean_iou"] - 1) * 100),
        ("Hake Dice", (hake["finetuned"]["mean_dice"] / hake["original"]["mean_dice"] - 1) * 100),
        (
            "Hake depth MAE",
            (hake["finetuned"]["mean_depth_mae_m"] / hake["original"]["mean_depth_mae_m"] - 1)
            * 100,
        ),
        ("Basto IoU", (basto["finetuned"]["mean_iou"] / basto["original"]["mean_iou"] - 1) * 100),
        (
            "Basto Dice",
            (basto["finetuned"]["mean_dice"] / basto["original"]["mean_dice"] - 1) * 100,
        ),
        (
            "Basto depth MAE",
            (
                basto["finetuned"]["mean_bottom_error_m"] / basto["original"]["mean_bottom_error_m"]
                - 1
            )
            * 100,
        ),
    ]
    labels = [label for label, _ in items]
    values = [value for _, value in items]
    colors = [
        "#54a24b"
        if (("MAE" not in label and value > 0) or ("MAE" in label and value < 0))
        else "#e45756"
        for label, value in items
    ]

    fig, ax = plt.subplots(figsize=(8.2, 4.1))
    y = np.arange(len(items), dtype=float)
    bars = ax.barh(y, values, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    for bar, value in zip(bars, values):
        inside = abs(value) >= 8
        if inside and value >= 0:
            x, ha, color = value - 1.2, "right", "white"
        elif inside:
            x, ha, color = value + 1.2, "left", "white"
        else:
            x, ha, color = (
                value + (1.0 if value >= 0 else -1.0),
                ("left" if value >= 0 else "right"),
                "black",
            )
        ax.text(
            x,
            bar.get_y() + bar.get_height() / 2,
            f"{value:+.1f}%",
            ha=ha,
            va="center",
            fontsize=8,
            color=color,
        )
    ax.set_yticks(y, labels)
    ax.set_xlabel("Relative change after fine-tuning")
    ax.set_title("Fine-tuning effect summary")
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "effect_summary_percent.png", dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_conformal_focus() -> list[dict[str, str]]:
    path = FIG_DIR / "conformal_results.csv"
    if not path.exists():
        return []
    rows = read_rows(path)
    constant = [r for r in rows if r["method"] == "constant"]
    order = {
        ("original", "basto", "hake"): 0,
        ("original", "hake", "hake"): 1,
        ("original", "pooled", "hake"): 2,
        ("original", "balanced_pooled", "hake"): 3,
        ("finetuned", "basto", "hake"): 4,
        ("finetuned", "hake", "hake"): 5,
        ("finetuned", "pooled", "hake"): 6,
        ("finetuned", "balanced_pooled", "hake"): 7,
        ("original", "basto", "basto"): 8,
        ("original", "hake", "basto"): 9,
        ("original", "pooled", "basto"): 10,
        ("original", "balanced_pooled", "basto"): 11,
        ("finetuned", "basto", "basto"): 12,
        ("finetuned", "hake", "basto"): 13,
        ("finetuned", "pooled", "basto"): 14,
        ("finetuned", "balanced_pooled", "basto"): 15,
    }
    constant.sort(
        key=lambda r: order.get((r["model"], r["calibration_domain"], r["test_domain"]), 99)
    )
    labels = [f"{r['model']} {r['calibration_domain']}->{r['test_domain']}" for r in constant]
    colors = ["#4c78a8" if r["model"] == "original" else "#f58518" for r in constant]
    fig, ax = plt.subplots(figsize=(9.2, max(4.8, 0.34 * len(constant))))
    y = np.arange(len(constant), dtype=float)
    bars = ax.barh(y, [f(r, "coverage") for r in constant], color=colors)
    ax.axvline(0.9, color="black", linestyle="--", linewidth=1.0, label="target 0.90")
    for bar, row in zip(bars, constant):
        value = f(row, "coverage")
        label_x = min(value + 0.012, 1.015)
        ax.text(
            label_x,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            ha="left",
            va="center",
            fontsize=8,
        )
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Empirical coverage")
    ax.set_yticks(y, labels, fontsize=8)
    ax.invert_yaxis()
    ax.grid(True, axis="x", alpha=0.25)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "constant_conformal_focus.png", dpi=250, bbox_inches="tight")
    plt.close(fig)
    return rows


def write_interpretation(hake: dict, basto: dict, conformal_rows: list[dict[str, str]]) -> None:
    hake_iou_gain = delta(hake["finetuned"]["mean_iou"], hake["original"]["mean_iou"])
    hake_mae_gain = delta(
        hake["finetuned"]["mean_depth_mae_m"], hake["original"]["mean_depth_mae_m"]
    )
    basto_iou_gain = delta(basto["finetuned"]["mean_iou"], basto["original"]["mean_iou"])
    basto_mae_gain = delta(
        basto["finetuned"]["mean_bottom_error_m"], basto["original"]["mean_bottom_error_m"]
    )

    lines = [
        "# Hake fine-tune experiment",
        "",
        f"Base checkpoint: `{BASE_CKPT.relative_to(ROOT)}`",
        f"Fine-tuned checkpoint: `{(CKPT_DIR / 'smp_unet_resnet34_best.pth').relative_to(ROOT)}`",
        "",
        "## Test metrics",
        f"- Hake improved from IoU {hake['original']['mean_iou']:.3f} to {hake['finetuned']['mean_iou']:.3f} "
        f"and depth MAE {hake['original']['mean_depth_mae_m']:.2f} m to {hake['finetuned']['mean_depth_mae_m']:.2f} m "
        f"(delta IoU {hake_iou_gain:+.3f}, delta MAE {hake_mae_gain:+.2f} m).",
        f"- Basto changed from IoU {basto['original']['mean_iou']:.3f} to {basto['finetuned']['mean_iou']:.3f} "
        f"and depth MAE {basto['original']['mean_bottom_error_m']:.3f} m to {basto['finetuned']['mean_bottom_error_m']:.3f} m "
        f"(delta IoU {basto_iou_gain:+.4f}, delta MAE {basto_mae_gain:+.3f} m).",
        "",
        "## Visual outputs",
        "- `metric_summary_panels.png`: Hake and Basto metric comparison.",
        "- `hake_split_overview.png`: Hake split sizes used in the experiment.",
        "- `effect_summary_percent.png`: relative metric changes after fine-tuning.",
        "- `hake_group_delta_bars.png`: per-group Hake changes.",
        "- `finetuned_hake_conformal_methods.png`: Hake coverage by calibration source/method.",
        "- `constant_conformal_focus.png`: constant-width conformal coverage, when conformal scoring has finished.",
        "- `hake_overlays/*_before_after.png`: qualitative before/after overlays.",
        "",
        "## Interpretation",
        "- The fine-tune is useful evidence for domain adaptation: it substantially improves Hake on the held-out Hake test groups.",
        "- It should not replace the general Basto checkpoint: Basto performance regresses slightly after Hake adaptation.",
        "- Add this as an isolated Hake/domain-adaptation subsection or appendix, with the caveat that Hake references are derived labels and the Hake test split is small.",
    ]
    if conformal_rows:
        constant = [r for r in conformal_rows if r["method"] == "constant"]
        lines += ["", "## Constant conformal coverage"]
        for row in constant:
            lines.append(
                f"- {row['model']} {row['calibration_domain']}->{row['test_domain']}: "
                f"coverage {f(row, 'coverage'):.3f}, q={f(row, 'q_hat'):.3f}, n={int(f(row, 'n'))}."
            )
    (FIG_DIR / "_README.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    hake = summarize_hake()
    basto = summarize_basto()
    plot_split_overview()
    plot_metric_panels(hake, basto)
    plot_hake_group_deltas()
    plot_effect_summary(hake, basto)
    plot_conformal_hake_methods()
    conformal_rows = plot_conformal_focus()
    write_interpretation(hake, basto, conformal_rows)
    print(f"Wrote post-processed summaries and figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
