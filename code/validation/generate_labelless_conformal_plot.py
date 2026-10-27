#!/usr/bin/env python
"""Label-less conformal prediction calibrated from repeated ferry crossings.

End-to-end pipeline:
  1. Load best checkpoint and the echopype zarr files.
  2. For each file, run U-Net inference, extract per-ping bottom depth.
  3. Detect ferry crossings via the midpoint-line state machine, project
     each ping onto the route arc-length, and emit a long-format DataFrame.
  4. Split crossings into calibration and validation halves.
  5. Compute residual-from-bin-median calibration scores, derive q_hat.
  6. Validate by checking pairwise coverage on held-out crossings.
  7. Sweep alpha for the empirical-vs-nominal coverage curve.
  8. Write artefacts:
        ping_table.parquet            — long-format predictions
        bin_summary.csv               — per-bin medians and counts
        calibration.json              — q_hat and metadata
        coverage_curve.csv            — alpha sweep
        coverage_curve.png            — calibration plot
        score_distribution.png        — histogram of calibration scores
        example_crossing_band.png     — one held-out crossing with band
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_project_root = str(Path(__file__).resolve().parents[1])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from project_paths import REPO_ROOT  # noqa: E402

import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from deep_learning.unet_model import create_unet, predict_full_echogram
from raw_sensor_integration.preprocessing import load_sv_from_zarr, normalize_echogram
from validation.conformal import extract_bottom_depths, compute_transition_width
from validation.conformal_labelless import (
    LabellessCalibration,
    build_calibration_scores,
    calibrate_labelless,
    pairwise_coverage,
    pairwise_coverage_curve,
    plot_labelless_band,
    plot_pairwise_coverage_curve,
    split_crossings,
)


# ── Defaults ──────────────────────────────────────────────────────────
DEFAULT_CHECKPOINT = str(
    REPO_ROOT
    / "code"
    / "checkpoints"
    / "exp_20260511_191547_all_surveys_jun2025"
    / "smp_unet_resnet34_best.pth"
)
DEFAULT_ZARR_DIR = str(REPO_ROOT / "Datasets" / "echopype_zarr")
DEFAULT_OUT_DIR = str(REPO_ROOT / "Figures" / "conformal" / "labelless")

FREQ1, FREQ2 = 38_000, 200_000
VMIN, VMAX = -80.0, -34.0
TARGET_SIZE = (512, 512)
PATCH_WIDTH, STRIDE = 256, 128


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(checkpoint_path: Path, device: torch.device):
    model = create_unet(encoder_name="resnet34", in_channels=3)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and any(k.startswith(("model.", "module.")) for k in ckpt.keys()):
        state = ckpt
    else:
        state = ckpt
    # Some checkpoints were saved from a wrapping module ("model." or "module.")
    state = {
        (
            k[len("model.") :]
            if k.startswith("model.")
            else k[len("module.") :]
            if k.startswith("module.")
            else k
        ): v
        for k, v in state.items()
    }
    model.load_state_dict(state)
    model.to(device).eval()
    return model


def predict_depths_for_zarr(
    zarr_path: Path, model, device: torch.device
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return per-ping (pred_depth_m, range_array, prob_map, sigma_m)."""
    sv1, range1, _, _, _, _ = load_sv_from_zarr(zarr_path, FREQ1)
    sv2, range2, _, _, _, _ = load_sv_from_zarr(zarr_path, FREQ2)
    min_r = min(sv1.shape[0], sv2.shape[0])
    sv1, sv2 = sv1[:min_r], sv2[:min_r]
    range_arr = range1[:min_r]

    norm1 = normalize_echogram(sv1, method="minmax", vmin=VMIN, vmax=VMAX, fill_nan=0.0).astype(
        np.float32
    )
    norm2 = normalize_echogram(sv2, method="minmax", vmin=VMIN, vmax=VMAX, fill_nan=0.0).astype(
        np.float32
    )
    diff = (norm1 - norm2).astype(np.float32)
    sv_channels = np.stack([norm1, norm2, diff], axis=0)

    with torch.no_grad():
        prob = predict_full_echogram(
            model, sv_channels, TARGET_SIZE, device, patch_width=PATCH_WIDTH, stride=STRIDE
        )
    pred_depths = extract_bottom_depths(prob, range_arr, threshold=0.5)
    # Per-ping difficulty: probability-transition width (metres). Sharp
    # water->bottom transition = small sigma (confident), gradual = large.
    sigma = compute_transition_width(prob, range_arr)
    return pred_depths, range_arr, prob, sigma


def _load_file_records(
    zp: Path, model, device, out_dir: Path, save_prob_artifact: bool = False
) -> dict | None:
    """Inference on one zarr file. Returns per-ping arrays or None on failure."""
    import xarray as xr

    try:
        pred_depths, range_arr, prob, sigma = predict_depths_for_zarr(zp, model, device)
    except Exception as e:
        print(f"  [skip] {zp.name}: {e}")
        return None

    ds = xr.open_zarr(str(zp), consolidated=False)
    try:
        ping_time = pd.to_datetime(ds.ping_time.values, utc=True).values
        if "latitude" in ds and "longitude" in ds:
            lat = np.asarray(ds.latitude.values, dtype=np.float64)
            lon = np.asarray(ds.longitude.values, dtype=np.float64)
        else:
            lat = np.full(len(ping_time), np.nan)
            lon = np.full(len(ping_time), np.nan)
    finally:
        ds.close()

    n = min(len(pred_depths), len(ping_time))
    pred_depths, sigma = pred_depths[:n], sigma[:n]
    ping_time, lat, lon = ping_time[:n], lat[:n], lon[:n]

    if save_prob_artifact:
        np.savez_compressed(
            out_dir / f"qa_probmap_{zp.stem}.npz",
            prob=prob.astype(np.float16),
            range_array=range_arr,
            pred_depths=pred_depths,
        )

    return {
        "file_id": zp.stem,
        "ping_time": ping_time,
        "lat": lat,
        "lon": lon,
        "pred_depth": pred_depths,
        "sigma": sigma,
        "n_pings": int(n),
    }


def _stitch_groups(records: list[dict], gap_tolerance_s: float = 600.0) -> list[list[dict]]:
    """Sort by ping_time and group files with small time gaps into runs."""
    records = [r for r in records if r is not None and r["n_pings"] > 0]
    records.sort(key=lambda r: r["ping_time"][0])
    if not records:
        return []
    groups: list[list[dict]] = [[records[0]]]
    for r in records[1:]:
        prev_end = groups[-1][-1]["ping_time"][-1]
        gap = (r["ping_time"][0] - prev_end) / np.timedelta64(1, "s")
        if gap <= gap_tolerance_s:
            groups[-1].append(r)
        else:
            groups.append([r])
    return groups


def collect_ping_table(
    zarr_paths: list[Path],
    model,
    device,
    out_dir: Path,
    save_example_probs: int = 0,
    gap_tolerance_s: float = 600.0,
) -> pd.DataFrame:
    """Run inference per file, stitch consecutive files, then detect crossings
    on the stitched trace so crossings spanning file boundaries are kept.
    """
    from validation.conformal_labelless import (
        detect_crossings,
        filter_by_duration,
        project_to_arclength,
        LON_HORTEN,
        LON_MOSS,
        DOCK_HALFWIDTH_DEG,
    )

    diagnostics: list[dict] = []
    records: list[dict] = []
    saved_probs = 0
    for zp in zarr_paths:
        rec = _load_file_records(
            zp, model, device, out_dir, save_prob_artifact=(save_example_probs > saved_probs)
        )
        if rec is None:
            continue
        records.append(rec)
        if save_example_probs > saved_probs:
            saved_probs += 1

    groups = _stitch_groups(records, gap_tolerance_s=gap_tolerance_s)
    print(f"\n  Stitched into {len(groups)} consecutive group(s):")
    for gi, g in enumerate(groups):
        print(f"    group {gi}: {len(g)} files, {g[0]['ping_time'][0]} -> {g[-1]['ping_time'][-1]}")

    rows: list[dict] = []
    for gi, g in enumerate(groups):
        # Concatenate per-ping arrays for this run.
        lat = np.concatenate([r["lat"] for r in g])
        lon = np.concatenate([r["lon"] for r in g])
        ping_time = np.concatenate([r["ping_time"] for r in g])
        pred_depth = np.concatenate([r["pred_depth"] for r in g])
        sigma = np.concatenate([r["sigma"] for r in g])
        # Per-ping file_id for traceability.
        file_id_per_ping = np.concatenate(
            [np.repeat(np.asarray(r["file_id"]), r["n_pings"]) for r in g]
        )

        crossings = detect_crossings(lon, ping_time)
        crossings = filter_by_duration(crossings)
        n_dropped = 0

        for c in crossings:
            s_idx, e_idx = c["start"], c["end"]
            cid = f"group{gi:02d}_{s_idx:07d}_{c['direction']}"
            idx = np.arange(s_idx, e_idx + 1)
            lat_c, lon_c = lat[idx], lon[idx]
            pt_c = ping_time[idx]
            pd_c = pred_depth[idx]
            sig_c = sigma[idx]
            fid_c = file_id_per_ping[idx]

            valid = (
                np.isfinite(lat_c)
                & np.isfinite(lon_c)
                & np.isfinite(pd_c)
                & (pd_c > 0)
                & (lon_c > LON_HORTEN + DOCK_HALFWIDTH_DEG)
                & (lon_c < LON_MOSS - DOCK_HALFWIDTH_DEG)
            )
            if not valid.any():
                n_dropped += 1
                continue
            s_arc = project_to_arclength(lat_c[valid], lon_c[valid])
            for k, i_local in enumerate(np.where(valid)[0]):
                sig_val = float(sig_c[i_local])
                rows.append(
                    {
                        "file_id": str(fid_c[i_local]),
                        "group_id": gi,
                        "ping_idx": int(idx[i_local]),
                        "ping_time": pt_c[i_local],
                        "lat": float(lat_c[i_local]),
                        "lon": float(lon_c[i_local]),
                        "pred_depth_m": float(pd_c[i_local]),
                        "sigma_m": sig_val if np.isfinite(sig_val) else np.nan,
                        "crossing_id": cid,
                        "direction": c["direction"],
                        "arc_length_m": float(s_arc[k]),
                    }
                )

        diagnostics.append(
            {
                "group_id": gi,
                "n_files": len(g),
                "n_pings": int(len(lat)),
                "n_crossings_detected": len(crossings),
                "n_crossings_dropped_no_valid_pings": n_dropped,
                "t_start": str(g[0]["ping_time"][0]),
                "t_end": str(g[-1]["ping_time"][-1]),
            }
        )
        print(
            f"    group {gi}: detected {len(crossings)} crossings "
            f"(dropped {n_dropped} with no valid pings)"
        )

    pd.DataFrame(diagnostics).to_csv(out_dir / "group_diagnostics.csv", index=False)
    return pd.DataFrame(rows)


def _robust_reject(df: pd.DataFrame, bin_size_m: float, k: float) -> pd.DataFrame:
    """Drop pings whose depth is > k robust-MADs from their bin median.

    Uses the cross-crossing redundancy to suppress gross model failures
    (e.g. a single crossing reading 90 m where every other crossing reads
    30 m) that would otherwise dominate the q_hat tail.
    """
    from validation.conformal_labelless import bin_arclength

    work = df.copy()
    work["_bin"] = bin_arclength(work["arc_length_m"].to_numpy(), bin_size_m)
    med = work.groupby("_bin")["pred_depth_m"].transform("median")
    mad = work.groupby("_bin")["pred_depth_m"].transform(
        lambda x: np.median(np.abs(x - np.median(x)))
    )
    robust_sigma = 1.4826 * mad
    # Where MAD is ~0 (all crossings agree) keep everything.
    dev = np.abs(work["pred_depth_m"] - med)
    keep = (robust_sigma <= 0) | (dev <= k * robust_sigma)
    n_drop = int((~keep).sum())
    print(
        f"  Robust-reject (k={k}): dropped {n_drop:,} outlier pings ({len(work) - n_drop:,} remain)"
    )
    return work[keep].drop(columns="_bin").reset_index(drop=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    p.add_argument("--zarr-dir", default=DEFAULT_ZARR_DIR)
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument(
        "--max-files", type=int, default=None, help="cap the number of zarr files processed"
    )
    p.add_argument("--bin-size", type=float, default=20.0, help="arc-length bin size in metres")
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--min-crossings-per-bin", type=int, default=2)
    p.add_argument(
        "--per-direction", action="store_true", help="calibrate H2M and M2H bins separately"
    )
    p.add_argument(
        "--score-mode",
        choices=("pairwise", "median"),
        default="pairwise",
        help="conformity score: pairwise |A-B| (self-consistency) "
        "or |pred-bin_median| (single-prediction truth band)",
    )
    p.add_argument(
        "--adaptive",
        action="store_true",
        help="per-ping adaptive band: normalize the score by a "
        "per-ping sigma, giving pred ± q_hat·sigma (narrow "
        "where reliable, wide where not)",
    )
    p.add_argument(
        "--sigma-source",
        choices=("transition", "dispersion"),
        default="dispersion",
        help='source of the adaptive sigma: "transition" = the '
        "probability-transition width (weak signal when the "
        'model fails confidently); "dispersion" = per-bin '
        "cross-crossing depth spread estimated on calibration "
        "crossings (direct label-free difficulty signal). "
        "Only used with --adaptive.",
    )
    p.add_argument("--calib-frac", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--qa-prob-maps", type=int, default=0, help="save N example probability maps as .npz for QA"
    )
    p.add_argument(
        "--max-pred-depth",
        type=float,
        default=None,
        help="drop pings whose predicted depth exceeds this many "
        "metres before calibration (filters obvious model "
        "failures; e.g. 100 for the Oslofjord)",
    )
    p.add_argument(
        "--robust-reject",
        type=float,
        default=None,
        help="drop pings whose depth deviates more than this many "
        "robust-MADs from their bin median before calibration "
        "(e.g. 5). Uses the crossing redundancy to suppress "
        "gross model failures that dominate the q_hat tail.",
    )
    args = p.parse_args()

    # Resolve the effective score mode and the sigma column for adaptive bands.
    score_mode = args.score_mode + ("_norm" if args.adaptive else "")
    sigma_col = "sigma_m" if args.adaptive else None

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    zarr_paths = sorted(Path(args.zarr_dir).glob("*.zarr"))
    if args.max_files is not None:
        zarr_paths = zarr_paths[: args.max_files]
    if not zarr_paths:
        raise SystemExit(f"No zarr files found in {args.zarr_dir}")

    print("── Loading model ──")
    device = _device()
    print(f"  device: {device}")
    model = load_model(Path(args.checkpoint), device)
    print(f"  checkpoint: {args.checkpoint}")

    print(f"\n── Inference + crossing extraction ({len(zarr_paths)} files) ──")
    df = collect_ping_table(
        zarr_paths, model, device, out_dir, save_example_probs=args.qa_prob_maps
    )
    if df.empty:
        raise SystemExit("No pings in any crossing; cannot calibrate.")

    print(f"\n  Total pings in crossings: {len(df):,}")
    print(f"  Distinct crossings: {df['crossing_id'].nunique()}")
    print(f"  H2M / M2H: {(df['direction'] == 'H2M').sum()} / {(df['direction'] == 'M2H').sum()}")

    if args.max_pred_depth is not None:
        before = len(df)
        df = df[df["pred_depth_m"] <= args.max_pred_depth].copy()
        print(
            f"  Dropped {before - len(df):,} pings with pred_depth > "
            f"{args.max_pred_depth} m  ({len(df):,} remain)"
        )

    # Fill NaN sigma (transition width undefined where the probability never
    # crosses the upper threshold) with the global median so adaptive scoring
    # doesn't lose those pings.
    if "sigma_m" in df.columns:
        med_sigma = float(np.nanmedian(df["sigma_m"].to_numpy()))
        df["sigma_m"] = df["sigma_m"].fillna(med_sigma if np.isfinite(med_sigma) else 1.0)

    if args.robust_reject is not None:
        df = _robust_reject(df, args.bin_size, args.robust_reject)

    df.to_parquet(out_dir / "ping_table.parquet", index=False)

    # ── Split crossings ───────────────────────────────────────────────
    df_calib, df_test = split_crossings(df, calib_frac=args.calib_frac, seed=args.seed)
    print(f"\n  Calibration crossings: {df_calib['crossing_id'].nunique()} ({len(df_calib)} pings)")
    print(f"  Validation crossings:  {df_test['crossing_id'].nunique()} ({len(df_test)} pings)")

    # ── Adaptive sigma source ─────────────────────────────────────────
    if args.adaptive and args.sigma_source == "dispersion":
        from validation.conformal_labelless import compute_bin_dispersion, assign_bin_sigma

        disp = compute_bin_dispersion(
            df_calib,
            bin_size_m=args.bin_size,
            per_direction=args.per_direction,
            min_crossings_per_bin=args.min_crossings_per_bin,
        )
        # Estimate on calibration crossings only, then map onto both splits.
        df_calib = assign_bin_sigma(
            df_calib, disp, bin_size_m=args.bin_size, per_direction=args.per_direction
        )
        df_test = assign_bin_sigma(
            df_test, disp, bin_size_m=args.bin_size, per_direction=args.per_direction
        )
        print(
            f"  Adaptive sigma = per-bin cross-crossing dispersion "
            f"({len(disp)} bins; median {disp['sigma_disp_m'].median():.2f} m)"
        )
    elif args.adaptive:
        print("  Adaptive sigma = probability-transition width (per ping)")

    # ── Calibration ───────────────────────────────────────────────────
    scores, bin_info = build_calibration_scores(
        df_calib,
        bin_size_m=args.bin_size,
        min_crossings_per_bin=args.min_crossings_per_bin,
        per_direction=args.per_direction,
        score_mode=score_mode,
        sigma_col=sigma_col,
    )
    if len(scores) == 0:
        raise SystemExit(
            "No calibration scores produced. Increase --max-files or lower --min-crossings-per-bin."
        )

    q_hat = calibrate_labelless(scores, alpha=args.alpha)
    coverage = pairwise_coverage(
        df_test,
        q_hat,
        bin_size_m=args.bin_size,
        per_direction=args.per_direction,
        score_mode=score_mode,
        sigma_col=sigma_col,
    )
    print(f"\n── Calibration result ({score_mode}) ──")
    if args.adaptive:
        print(
            f"  α = {args.alpha:.2f}  →  q_hat = {q_hat:.3f} (×σ multiplier); "
            f"mean realised band ±{coverage['mean_band_width_m'] / 2:.2f} m"
        )
    else:
        print(
            f"  α = {args.alpha:.2f}  →  q_hat = {q_hat:.3f} m "
            f"(band ±{q_hat:.2f} m, total {2 * q_hat:.2f} m)"
        )
    print(
        f"  Pairwise coverage on validation: "
        f"{coverage['coverage']:.1%} "
        f"({coverage['n_pairs']} pairs, "
        f"median |Δ| = {coverage['median_abs_diff_m']:.2f} m)"
    )

    bin_info.to_csv(out_dir / "bin_summary.csv", index=False)

    cal = LabellessCalibration(
        alpha=args.alpha,
        q_hat_m=float(q_hat),
        n_calib_scores=int(len(scores)),
        n_calib_pings=int(len(df_calib)),
        n_test_pings=int(len(df_test)),
        bin_size_m=float(args.bin_size),
        min_crossings_per_bin=int(args.min_crossings_per_bin),
        per_direction=bool(args.per_direction),
        score_mode=score_mode,
        coverage=coverage,
    )
    cal.to_json(out_dir / "calibration.json")

    # ── Coverage curve sweep ──────────────────────────────────────────
    print("\n── Sweeping α for coverage curve ──")
    curve = pairwise_coverage_curve(
        df_calib,
        df_test,
        alphas=np.arange(0.02, 0.51, 0.02),
        bin_size_m=args.bin_size,
        per_direction=args.per_direction,
        min_crossings_per_bin=args.min_crossings_per_bin,
        score_mode=score_mode,
        sigma_col=sigma_col,
    )
    curve.to_csv(out_dir / "coverage_curve.csv", index=False)
    plot_pairwise_coverage_curve(
        curve,
        save_path=out_dir / "coverage_curve.png",
        title=(
            f"Label-less conformal ({score_mode}, bin={args.bin_size:.0f} m, α={args.alpha:.2f})"
        ),
    )
    plt.close("all")

    # ── Score distribution ────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
    # Clip the long failure tail for a readable histogram.
    hi = float(np.quantile(scores, 0.995))
    ax.hist(np.clip(scores, None, hi), bins=60, color="#1f77b4", alpha=0.85, edgecolor="white")
    unit = "" if args.adaptive else " m"
    ax.axvline(
        q_hat,
        color="#d62728",
        lw=2,
        ls="--",
        label=rf"$\hat q$ = {q_hat:.2f}{unit}  (α={args.alpha:.2f})",
    )
    if args.adaptive:
        base = (
            r"$|\mathrm{pred}_A - \mathrm{pred}_B| / \sigma_A$"
            if args.score_mode == "pairwise"
            else r"$|\mathrm{pred} - \mathrm{med}| / \sigma$"
        )
        score_xlabel = base + "  (dimensionless)"
    else:
        score_xlabel = (
            r"$|\mathrm{pred}_A - \mathrm{pred}_B|$  (m)"
            if args.score_mode == "pairwise"
            else r"$|\mathrm{pred} - \mathrm{bin\ median}|$  (m)"
        )
    ax.set_xlabel(score_xlabel)
    ax.set_ylabel("Count")
    ax.set_title(f"Label-less nonconformity scores ({score_mode})")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "score_distribution.png", bbox_inches="tight")
    plt.close(fig)

    # ── Example crossing band ─────────────────────────────────────────
    if not df_test.empty:
        # Pick the validation crossing with the most pings.
        counts = df_test.groupby("crossing_id").size().sort_values(ascending=False)
        example_cid = counts.index[0]
        ex = df_test[df_test["crossing_id"] == example_cid]
        # Overlay other validation crossings of the same direction.
        same_dir = ex["direction"].iloc[0]
        neighbours = df_test[
            (df_test["direction"] == same_dir) & (df_test["crossing_id"] != example_cid)
        ]
        ymax = args.max_pred_depth if args.max_pred_depth is not None else None
        plot_labelless_band(
            ex,
            q_hat,
            df_neighbour=neighbours,
            bin_size_m=args.bin_size,
            save_path=out_dir / "example_crossing_band.png",
            title=f"Label-less conformal band  •  {example_cid}",
            sigma_col=sigma_col,
            ymax=ymax,
        )
        plt.close("all")

        # Also dump the per-ping band as CSV.
        ex_sorted = ex.sort_values("arc_length_m").copy()
        if args.adaptive:
            half = q_hat * np.maximum(ex_sorted["sigma_m"].to_numpy(), 0.5)
        else:
            half = q_hat
        ex_sorted["lower_m"] = ex_sorted["pred_depth_m"] - half
        ex_sorted["upper_m"] = ex_sorted["pred_depth_m"] + half
        ex_sorted.to_csv(out_dir / "example_crossing_band.csv", index=False)

    print(f"\n── Done. Outputs in {out_dir} ──")
    for name in sorted(out_dir.glob("*")):
        print(f"  {name.relative_to(out_dir)}")


if __name__ == "__main__":
    main()
