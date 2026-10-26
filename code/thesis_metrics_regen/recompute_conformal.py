"""Recompute the supervised split-conformal numbers (E) from ONE canonical source.

Source: the cached per-ping nonconformity scores in
code/validation/cache/exchangeability/scores_<survey>.npz, which meta.json confirms were
produced with the chunk_arch_sweep_2 checkpoint. Keys per file:
    abs_err   : |pred_depth - gt_depth|        (constant-width nonconformity score, metres)
    norm_err  : abs_err / max(sigma, eps)      (locally-weighted nonconformity score, unitless)
    pred_depth, gt_depth : per-ping depths (m)
    file_id   : per-survey 0-based file index

Calibration survey = jan_2025; test = feb_2025 + jun_2025 (the thesis split). Computes q_hat,
pooled/per-survey/per-file coverage, KS calib-vs-test, the within-calibration no-shift
baseline, leave-one-file-out CV of q_hat, and per-survey median |error|.

Reuses the finite-sample quantile rule from validation/conformal.py.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
from validation.conformal import conformal_quantile  # noqa: E402

CACHE = ROOT / "code/validation/cache/exchangeability"
OUT = Path(__file__).resolve().parent / "outputs/conformal_summary.json"
OUT_CSV = Path(__file__).resolve().parent / "outputs/conformal_perfile_coverage.csv"

ALPHA = 0.10
CALIB = "jan_2025"
TEST = ["feb_2025", "jun_2025"]
ALL_SURVEYS = ["dec_2023", "jan_2024", "apr_2024", "jul_2024", "sep_2024",
               "oct_2024", "jan_2025", "feb_2025", "jun_2025"]


def load(survey):
    z = np.load(CACHE / f"scores_{survey}.npz", allow_pickle=True)
    return {k: z[k] for k in z.files}


def coverage(scores, q):
    return float((np.asarray(scores) <= q).mean())


def main():
    cal = load(CALIB)
    n_calib_files = int(np.unique(cal["file_id"]).size)

    # q_hat fitted on the calibration survey.
    q_const = conformal_quantile(cal["abs_err"], ALPHA)
    q_lw = conformal_quantile(cal["norm_err"], ALPHA)

    tests = {s: load(s) for s in TEST}
    test_abs = np.concatenate([tests[s]["abs_err"] for s in TEST])
    test_norm = np.concatenate([tests[s]["norm_err"] for s in TEST])
    n_test_files = sum(int(np.unique(tests[s]["file_id"]).size) for s in TEST)

    res = {
        "source": "code/validation/cache/exchangeability/scores_*.npz",
        "checkpoint": "chunk_arch_sweep_2",
        "alpha": ALPHA,
        "nominal_coverage": 1 - ALPHA,
        "calib_survey": CALIB,
        "n_calib_files": n_calib_files,
        "n_calib_pings": int(cal["abs_err"].size),
        "test_surveys": TEST,
        "n_test_files": n_test_files,
        "n_test_pings": int(test_abs.size),
        "q_hat_constant_m": q_const,
        "q_hat_locally_weighted": q_lw,
        "calib_coverage_constant": coverage(cal["abs_err"], q_const),
        "calib_coverage_lw": coverage(cal["norm_err"], q_lw),
        "pooled_test_coverage_constant": coverage(test_abs, q_const),
        "pooled_test_coverage_lw": coverage(test_norm, q_lw),
    }

    # Per-survey test coverage.
    res["per_survey_test_coverage_constant"] = {
        s: coverage(tests[s]["abs_err"], q_const) for s in TEST
    }
    res["per_survey_test_coverage_lw"] = {
        s: coverage(tests[s]["norm_err"], q_lw) for s in TEST
    }

    # Per-file coverage (constant band) across all scored test files.
    per_file = []
    for s in TEST:
        fid = tests[s]["file_id"]
        ae = tests[s]["abs_err"]
        for f in np.unique(fid):
            m = fid == f
            per_file.append((s, int(f), int(m.sum()), coverage(ae[m], q_const)))
    cov_vals = np.array([c for *_, c in per_file])
    res["per_file_coverage_constant"] = {
        "n_files": len(per_file),
        "mean": float(cov_vals.mean()),
        "std": float(cov_vals.std()),
        "min": float(cov_vals.min()),
        "max": float(cov_vals.max()),
    }

    # KS distance: calibration vs pooled test (both score types).
    res["ks_calib_vs_test_constant"] = float(stats.ks_2samp(cal["abs_err"], test_abs).statistic)
    res["ks_calib_vs_test_lw"] = float(stats.ks_2samp(cal["norm_err"], test_norm).statistic)

    # No-shift baseline: random-half split of the calibration scores, averaged over seeds.
    def half_split_ks(scores, n_seeds=50):
        scores = np.asarray(scores)
        ks = []
        for seed in range(n_seeds):
            rng = np.random.default_rng(seed)
            idx = rng.permutation(scores.size)
            half = scores.size // 2
            ks.append(stats.ks_2samp(scores[idx[:half]], scores[idx[half:]]).statistic)
        return float(np.mean(ks))

    res["ks_no_shift_baseline_constant"] = half_split_ks(cal["abs_err"])
    res["ks_no_shift_baseline_lw"] = half_split_ks(cal["norm_err"])

    # Leave-one-file-out CV of q_hat over the calibration files.
    fids = cal["file_id"]
    ae = cal["abs_err"]
    loo_q = []
    for f in np.unique(fids):
        loo_q.append(conformal_quantile(ae[fids != f], ALPHA))
    loo_q = np.array(loo_q)
    res["loo_qhat_constant"] = {
        "mean": float(loo_q.mean()),
        "std": float(loo_q.std()),
        "cv": float(loo_q.std() / loo_q.mean()),
    }

    # Per-survey median |error| across all available surveys.
    res["per_survey_median_abs_err_m"] = {}
    for s in ALL_SURVEYS:
        p = CACHE / f"scores_{s}.npz"
        if not p.exists():
            continue
        z = np.load(p, allow_pickle=True)
        res["per_survey_median_abs_err_m"][s] = float(np.median(z["abs_err"]))

    OUT.write_text(json.dumps(res, indent=2))
    with OUT_CSV.open("w") as fh:
        fh.write("survey,file_id,n_pings,coverage_constant\n")
        for s, f, n, c in per_file:
            fh.write(f"{s},{f},{n},{c:.6f}\n")

    print(json.dumps(res, indent=2))
    print(f"\nwrote {OUT}\nwrote {OUT_CSV}")


if __name__ == "__main__":
    main()
