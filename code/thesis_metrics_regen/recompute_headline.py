"""Recompute the Oct-2024 headline metrics (A) and bottom-error distribution (B).

Single canonical source: the chunk_arch_sweep_2 eval_metrics.json (378 held-out Oct-2024
files) and the saved per-ping signed_bottom_errors.npy. No inference; reads existing artifacts.

Writes outputs/headline_summary.json and prints a human-readable block. The augmentation P95
check (C) is appended by eval_aug_p95.py when it runs.
"""

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EVAL_JSON = ROOT / "code/checkpoints/chunk_arch_sweep_2/eval_metrics.json"
SIGNED_ERR = ROOT / "Figures/bottom_error_distribution/signed_bottom_errors.npy"
OUT = Path(__file__).resolve().parent / "outputs/headline_summary.json"


def main():
    d = json.loads(EVAL_JSON.read_text())
    summ = d.get("summary", d)
    pf = d["per_file"]
    iou = np.array([f["iou"] for f in pf], dtype=np.float64)
    be_mean = np.array([f.get("bottom_error_mean_m", np.nan) for f in pf], dtype=np.float64)
    be_med = np.array([f.get("bottom_error_median_m", np.nan) for f in pf], dtype=np.float64)
    files = [f["file"] for f in pf]

    zero = iou < 1e-9
    keep = ~zero
    empty_files = [files[i] for i in np.where(zero)[0]]

    short = 1.0 - iou
    short_sorted = np.sort(short)[::-1]

    res = {
        "source": str(EVAL_JSON.relative_to(ROOT)),
        "checkpoint": "chunk_arch_sweep_2",
        "n_files": int(len(pf)),
        "mean_iou": float(iou.mean()),
        "mean_dice": float(summ["mean_dice"]),
        "mean_bottom_error_m": float(summ["mean_bottom_error_m"]),
        "std_iou": float(summ["std_iou"]),
        "median_per_file_iou": float(np.median(iou)),
        "n_files_iou_above_0p99": int((iou > 0.99).sum()),
        "n_empty_label_files": int(zero.sum()),
        "empty_label_files": empty_files,
        "mean_iou_excl_empty": float(iou[keep].mean()),
        "median_per_file_median_dev_m_excl_empty": float(np.nanmedian(be_med[keep])),
        "frac_files_median_dev_below_1m_excl_empty": float((be_med[keep] < 1.0).mean()),
        "frac_files_mean_dev_below_1m_excl_empty": float((be_mean[keep] < 1.0).mean()),
        "total_iou_shortfall": float(short.sum()),
        "top5_empty_share_of_shortfall": float(short_sorted[:5].sum() / short.sum()),
        "top4_empty_share_of_shortfall": float(short_sorted[:4].sum() / short.sum()),
    }

    # B. bottom-error distribution from the saved per-ping signed errors.
    a = np.load(SIGNED_ERR)
    absel = np.abs(a)
    res["bottom_error_distribution"] = {
        "source": str(SIGNED_ERR.relative_to(ROOT)),
        "n_pings": int(a.size),
        "median_abs_err_m": float(np.median(absel)),
        "p95_abs_err_m": float(np.percentile(absel, 95)),
        "mean_abs_err_m": float(absel.mean()),
        "signed_median_m": float(np.median(a)),
        "signed_mean_m": float(a.mean()),
    }

    OUT.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
