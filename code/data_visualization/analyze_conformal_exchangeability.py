"""Conformal exchangeability diagnostics for the Discussion chapter.

The depth-band conformal guarantee assumes the calibration and test points are
exchangeable, and the formulation treats each ping as one point. Two features of
the route data strain that: consecutive pings are correlated (so the effective
sample size is closer to the file count than the ping count), and the file-level
split assigns whole days, sea-states, and tidal phases to one side of the split.
This script turns that argument into measured diagnostics, using the cached
per-survey nonconformity scores (no model inference needed).

Three panels:

  (a) Realized per-survey coverage at the 90% and 95% levels, with q_hat
      calibrated on the calibration survey (jan_2025). Bars coloured by the
      survey's role in the canonical split. Test surveys reveal whether the
      marginal guarantee survives the day-to-day shift.
  (b) The file, not the ping, as the unit: realized coverage per file across
      the pooled test surveys at fixed q_hat (the spread exposes the inflated
      ping-level sample size), annotated with the coefficient of variation of
      q_hat under leave-one-file-out recalibration on the calibration survey.
  (c) Covariate descriptors per survey (median absolute depth error with the
      interquartile range, and the central depth range), with the calibration
      survey marked, showing whether the test surveys sit inside the
      calibration spread.

Output: Figures/conformal_exchangeability/
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from data_visualization.thesis_style import apply_thesis_style, savefig, LEGEND_SIZE
from validation.conformal import conformal_quantile, evaluate_coverage
from deep_learning.training.survey_groups import SURVEYS, SPLITS

CACHE_DIR = PROJECT_ROOT / "code" / "validation" / "cache" / "exchangeability"
OUTPUT_DIR = PROJECT_ROOT / "Figures" / "conformal_exchangeability"

# Chronological survey order, used on every axis.
SURVEY_ORDER = [
    "dec_2023",
    "jan_2024",
    "apr_2024",
    "jul_2024",
    "sep_2024",
    "oct_2024",
    "jan_2025",
    "feb_2025",
    "jun_2025",
]

# Map each survey to its split role and a colour.
SPLIT_OF = {s: split for split, surveys in SPLITS.items() for s in surveys}
ROLE_COLOR = {
    "train": "#9ecae1",
    "val": "#fdae6b",
    "calibration": "#74c476",
    "test": "#d62728",
}
ROLE_LABEL = {
    "train": "Train",
    "val": "Validation",
    "calibration": "Calibration",
    "test": "Test",
}

CALIB_SURVEY = "jan_2025"
ALPHAS = {0.10: "90%", 0.05: "95%"}


def load_survey(name):
    """Return (pred_depth, gt_depth, abs_err, file_id) for a survey, excluding
    label-failure files whose per-file median abs error exceeds 2 m (a wrong
    reference bottom on almost every ping, the same files set aside in the
    segmentation analysis)."""
    d = np.load(CACHE_DIR / f"scores_{name}.npz")
    pred, gt, err, fid = d["pred_depth"], d["gt_depth"], d["abs_err"], d["file_id"]
    keep = np.ones(err.size, dtype=bool)
    for f in np.unique(fid):
        sel = fid == f
        mm = err[sel]
        mm = mm[np.isfinite(mm)]
        if mm.size and np.median(mm) > 2.0:
            keep[sel] = False
    return pred[keep], gt[keep], err[keep], fid[keep]


def main():
    apply_thesis_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data = {s: load_survey(s) for s in SURVEY_ORDER}
    calib_pred, calib_gt, calib_err, calib_fid = data[CALIB_SURVEY]

    # q_hat from the calibration survey at each level.
    calib_scores = np.abs(calib_pred - calib_gt)
    calib_scores = calib_scores[np.isfinite(calib_scores)]
    q_hat = {a: conformal_quantile(calib_scores, a) for a in ALPHAS}

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    labels = [SURVEYS[s]["label"] for s in SURVEY_ORDER]
    x = np.arange(len(SURVEY_ORDER))

    # --- Panel (a): realized per-survey coverage ----------------------------
    ax = axes[0]
    width = 0.38
    for k, (a, lvl) in enumerate(ALPHAS.items()):
        cov = []
        for s in SURVEY_ORDER:
            pred, gt, _, _ = data[s]
            c, _, _ = evaluate_coverage(pred, gt, q_hat[a])
            cov.append(c)
        offset = (k - 0.5) * width
        ax.bar(
            x + offset,
            cov,
            width,
            color="#4c78a8" if k == 0 else "#f58518",
            edgecolor="black",
            linewidth=0.6,
            label=f"{lvl} target",
        )
        ax.axhline(1 - a, color="#4c78a8" if k == 0 else "#f58518", ls="--", lw=1.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Realized coverage")
    ax.set_ylim(0.80, 1.0)
    ax.set_title("Per-survey coverage (q calibrated on Jan 2025)")
    ax.legend(fontsize=LEGEND_SIZE - 2, loc="lower left")
    # Mark the calibration survey on the axis.
    ci = SURVEY_ORDER.index(CALIB_SURVEY)
    ax.get_xticklabels()[ci].set_fontweight("bold")

    # --- Panel (b): per-file coverage on the test surveys -------------------
    ax = axes[1]
    test_surveys = SPLITS["test"]
    a_main = 0.10
    file_cov, file_lbl, file_col = [], [], []
    for s in test_surveys:
        pred, gt, _, fid = data[s]
        for f in np.unique(fid):
            m = fid == f
            c, _, n = evaluate_coverage(pred[m], gt[m], q_hat[a_main])
            if n > 0 and np.isfinite(c):
                file_cov.append(c)
                file_lbl.append(s)
                file_col.append(ROLE_COLOR["test"])
    file_cov = np.array(file_cov)
    jitter = (np.random.RandomState(0).rand(len(file_cov)) - 0.5) * 0.25
    ax.scatter(jitter, file_cov, s=90, c=file_col, edgecolor="black", linewidth=0.6, zorder=3)
    ax.axhline(1 - a_main, color="#4c78a8", ls="--", lw=2, label="90% target")
    ax.axhline(file_cov.mean(), color="black", ls=":", lw=2, label=f"mean = {file_cov.mean():.3f}")

    # Leave-one-file-out q_hat coefficient of variation on calibration survey.
    loo_q = []
    for f in np.unique(calib_fid):
        keep = calib_fid != f
        sc = np.abs(calib_pred[keep] - calib_gt[keep])
        sc = sc[np.isfinite(sc)]
        loo_q.append(conformal_quantile(sc, a_main))
    loo_q = np.array(loo_q)
    cv = loo_q.std() / loo_q.mean()
    ax.set_xlim(-0.5, 0.5)
    ax.set_xticks([])
    ax.set_ylabel("Per-file realized coverage")
    ax.set_ylim(min(0.5, file_cov.min() - 0.05), 1.02)
    ax.set_title("Test files as the unit (90% band)")
    ax.text(
        0.02,
        0.04,
        f"leave-one-file-out q CV = {100 * cv:.1f}%\n"
        f"per-file coverage sd = {file_cov.std():.3f}\n"
        f"n files = {len(file_cov)}",
        transform=ax.transAxes,
        fontsize=LEGEND_SIZE - 2,
        va="bottom",
        ha="left",
        bbox=dict(boxstyle="round", fc="white", ec="0.5"),
    )
    ax.legend(fontsize=LEGEND_SIZE - 2, loc="lower right")

    # --- Panel (c): covariate descriptors per survey ------------------------
    ax = axes[2]
    med, q1, q3, dlo, dhi = [], [], [], [], []
    for s in SURVEY_ORDER:
        pred, gt, err, _ = data[s]
        e = err[np.isfinite(err)]
        med.append(np.median(e))
        q1.append(np.percentile(e, 25))
        q3.append(np.percentile(e, 75))
        gtv = gt[np.isfinite(gt) & (gt > 0)]
        dlo.append(np.percentile(gtv, 10))
        dhi.append(np.percentile(gtv, 90))
    med = np.array(med)
    yerr = np.vstack([med - np.array(q1), np.array(q3) - med])
    colors = [ROLE_COLOR[SPLIT_OF[s]] for s in SURVEY_ORDER]
    ax.bar(
        x,
        med,
        color=colors,
        edgecolor="black",
        linewidth=0.6,
        yerr=yerr,
        capsize=4,
        error_kw={"elinewidth": 1.5},
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Median abs. depth error (m)")
    ax.set_title("Per-survey error and depth spread")
    ax.get_xticklabels()[ci].set_fontweight("bold")
    # Twin axis for the central depth range.
    ax2 = ax.twinx()
    ax2.plot(x, dlo, "o-", color="#54278f", lw=1.8, ms=6, label="10th pct depth")
    ax2.plot(x, dhi, "s--", color="#54278f", lw=1.8, ms=6, label="90th pct depth")
    ax2.set_ylabel("Bottom depth (m)", color="#54278f")
    ax2.tick_params(axis="y", labelcolor="#54278f")
    ax2.spines["top"].set_visible(False)
    ax2.legend(fontsize=LEGEND_SIZE - 4, loc="upper left")
    # Split-role legend.
    handles = [
        plt.Rectangle((0, 0), 1, 1, fc=ROLE_COLOR[r], ec="black")
        for r in ["train", "val", "calibration", "test"]
    ]
    ax.legend(
        handles,
        [ROLE_LABEL[r] for r in ["train", "val", "calibration", "test"]],
        fontsize=LEGEND_SIZE - 4,
        loc="upper right",
        ncol=2,
    )

    fig.tight_layout()
    savefig(fig, OUTPUT_DIR / "conformal_exchangeability.pdf")
    plt.close(fig)

    # Console summary for the verification step.
    print(f"q_hat(90%) = {q_hat[0.10]:.2f} m, q_hat(95%) = {q_hat[0.05]:.2f} m")
    for s in SURVEY_ORDER:
        pred, gt, _, _ = data[s]
        c90, _, _ = evaluate_coverage(pred, gt, q_hat[0.10])
        c95, _, _ = evaluate_coverage(pred, gt, q_hat[0.05])
        print(f"  {SURVEYS[s]['label']:<10s} [{SPLIT_OF[s]:<11s}] cov90={c90:.3f} cov95={c95:.3f}")
    print(f"leave-one-file-out q CV = {100 * cv:.1f}%, per-file coverage sd = {file_cov.std():.3f}")
    print(f"Wrote {OUTPUT_DIR / 'conformal_exchangeability.pdf'}")


if __name__ == "__main__":
    main()
