"""Label-less conformal prediction via repeated vessel crossings.

The Bastø ferry repeatedly traverses the same physical route between Horten
and Moss, so the seabed beneath any spot is observed many times across days.
We exploit this redundancy to calibrate a conformal prediction interval
*without* ground-truth labels: at each spatial bin along the route, the
spread of predicted bottom depths across crossings stands in for the
unknown error distribution.

Conformity scores (``score_mode``), with deliberately different targets:

* ``pairwise`` (default) — score = |pred_A − pred_B| over all ordered pairs of
  crossings in a bin. The resulting band ``pred ± q_hat`` is interpreted as:
  with probability ≥ 1−α, a *second* prediction at the same spot falls inside
  it. This is the self-consistency interval the master's thesis question asks
  about, and it is directly testable on held-out pairs (no labels needed).

* ``median`` — score = |pred − bin_median|. The bin median stands in for an
  approximate truth and the band targets *truth* coverage at 1−α. Strictly
  narrower than the pairwise band (by a factor ~√2 under i.i.d. errors), but
  pair coverage on held-out crossings will be lower than 1−α.

* ``pairwise_norm`` / ``median_norm`` — adaptive variants that divide the score
  by a per-ping sigma, giving a band ``pred ± q_hat·sigma`` that is narrow
  where the spot is reliable and wide where it is not (constant marginal
  coverage, better conditional coverage). Two sigma sources exist:
    - probability-transition width (``compute_transition_width``): cheap, but
      a *weak* signal here — the U-Net's transition is sharp even where it is
      wrong (it fails confidently), so on Bastø data this barely beats the
      constant band (Spearman ρ ≈ 0.16 vs cross-crossing disagreement).
    - per-bin cross-crossing dispersion (``compute_bin_dispersion``): the
      direct label-free difficulty signal — bins where repeated passes
      disagree get wide bands. This is the recommended adaptive sigma; it
      noticeably tightens the band on the calm channel while widening it on
      the noisy harbour approaches at the same coverage.

What this catches
-----------------
Random per-crossing error: tidal noise, ferry motion, ping-time-of-day
effects, model variance across spatial neighbourhoods.

What it cannot catch
--------------------
A systematic bias shared by every crossing (e.g. the U-Net always picking
the upper bottom return rather than the seabed proper) shows up as zero
inter-crossing spread but real label error. Compare against a labelled
subset before reading too much into the resulting interval width.

References
----------
The "exchangeable predictions across passes" calibration idea is similar
in spirit to cross-conformal prediction with leave-one-replicate-out
scores. The 1-D arc-length projection is specific to the linear ferry
route geometry.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from validation.conformal import conformal_quantile


# ---------------------------------------------------------------------------
# Bastø route geometry (consistent with plot_crossings_stack.py)
# ---------------------------------------------------------------------------

LON_HORTEN = 10.50
LON_MOSS = 10.66
DOCK_HALFWIDTH_DEG = 0.010  # ~580 m east-west: counts as "at the dock"

# Approximate terminal coordinates used for the route projection.
HORTEN_LAT_LON = (59.412, 10.493)
MOSS_LAT_LON = (59.426, 10.653)

# Plausible one-way pass duration. Bastø Electric crosses in ~22 min;
# anything outside this window is GPS dropout / extended dock dwell.
MIN_PASS_MIN = 15.0
MAX_PASS_MIN = 35.0

EARTH_R_M = 6_371_000.0


# ---------------------------------------------------------------------------
# Crossing detection
# ---------------------------------------------------------------------------


def detect_crossings(lon: np.ndarray, times: np.ndarray) -> list[dict]:
    """One-way pass detection by midpoint-line hysteresis.

    Identical state-machine to plot_crossings_stack.detect_crossings;
    duplicated here to avoid a sys.path dance into data_visualization.
    """
    horten_max = LON_HORTEN + DOCK_HALFWIDTH_DEG
    moss_min = LON_MOSS - DOCK_HALFWIDTH_DEG

    crossings: list[dict] = []
    state = "between"
    last_horten = None
    last_moss = None
    for i, x in enumerate(lon):
        if np.isnan(x):
            continue
        if x <= horten_max:
            new_state = "horten"
            last_horten = i
        elif x >= moss_min:
            new_state = "moss"
            last_moss = i
        else:
            new_state = "between"

        if new_state == "moss" and state == "horten" and last_horten is not None:
            crossings.append(
                {
                    "direction": "H2M",
                    "start": last_horten,
                    "end": i,
                    "t_start": times[last_horten],
                    "t_end": times[i],
                }
            )
            last_moss = i
        elif new_state == "horten" and state == "moss" and last_moss is not None:
            crossings.append(
                {
                    "direction": "M2H",
                    "start": last_moss,
                    "end": i,
                    "t_start": times[last_moss],
                    "t_end": times[i],
                }
            )
            last_horten = i

        if new_state in ("horten", "moss"):
            state = new_state
    return crossings


def filter_by_duration(
    crossings: list[dict], min_min: float = MIN_PASS_MIN, max_min: float = MAX_PASS_MIN
) -> list[dict]:
    """Drop crossings whose elapsed time is implausible."""
    kept: list[dict] = []
    for c in crossings:
        dt = pd.Timestamp(c["t_end"]) - pd.Timestamp(c["t_start"])
        minutes = dt.total_seconds() / 60.0
        if min_min <= minutes <= max_min:
            kept.append(c)
    return kept


# ---------------------------------------------------------------------------
# Arc-length projection along the Horten-Moss route
# ---------------------------------------------------------------------------


def project_to_arclength(
    lat: np.ndarray,
    lon: np.ndarray,
    origin: tuple[float, float] = HORTEN_LAT_LON,
    end: tuple[float, float] = MOSS_LAT_LON,
) -> np.ndarray:
    """Project geographic points onto the 1-D route corridor.

    Equirectangular projection around the route midpoint, then a dot
    product onto the unit vector from origin to end. Resolution of the
    equirectangular approximation is well under 1 m over this 10 km route.

    Returns
    -------
    s : (n,) float64 arc-length in metres, positive = toward Moss.
    """
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)

    lat0, lon0 = origin
    lat1, lon1 = end
    mean_lat = 0.5 * (lat0 + lat1)
    cos_lat = np.cos(np.deg2rad(mean_lat))

    deg_lon_m = cos_lat * (np.pi / 180.0) * EARTH_R_M
    deg_lat_m = (np.pi / 180.0) * EARTH_R_M

    x = (lon - lon0) * deg_lon_m
    y = (lat - lat0) * deg_lat_m

    dx = (lon1 - lon0) * deg_lon_m
    dy = (lat1 - lat0) * deg_lat_m
    L = float(np.hypot(dx, dy))
    ux, uy = dx / L, dy / L

    return x * ux + y * uy


def route_length_m(
    origin: tuple[float, float] = HORTEN_LAT_LON, end: tuple[float, float] = MOSS_LAT_LON
) -> float:
    lat0, lon0 = origin
    lat1, lon1 = end
    mean_lat = 0.5 * (lat0 + lat1)
    cos_lat = np.cos(np.deg2rad(mean_lat))
    dx = (lon1 - lon0) * cos_lat * (np.pi / 180.0) * EARTH_R_M
    dy = (lat1 - lat0) * (np.pi / 180.0) * EARTH_R_M
    return float(np.hypot(dx, dy))


def bin_arclength(s: np.ndarray, bin_size_m: float = 20.0) -> np.ndarray:
    """Integer bin index for each arc-length value."""
    s = np.asarray(s, dtype=np.float64)
    return np.floor(s / float(bin_size_m)).astype(np.int64)


# ---------------------------------------------------------------------------
# Per-file ping extraction
# ---------------------------------------------------------------------------


@dataclass
class CrossingPings:
    """Long-format record produced for one zarr file."""

    df: pd.DataFrame
    n_pings_total: int
    n_pings_in_crossing: int
    n_crossings: int


def _load_latlon(zarr_path: Path | str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read per-ping (lat, lon, ping_time) from an echopype zarr.

    Returns NaN-padded arrays so length always matches the Sv ping axis.
    """
    import xarray as xr

    ds = xr.open_zarr(str(zarr_path), consolidated=False)
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
    return lat, lon, ping_time


def extract_crossing_pings(
    zarr_path: Path | str,
    pred_depths: np.ndarray,
    *,
    file_id: str | None = None,
    min_duration_min: float = MIN_PASS_MIN,
    max_duration_min: float = MAX_PASS_MIN,
) -> CrossingPings:
    """Assemble the long-format DataFrame for one zarr file.

    The caller has already run inference and extracted per-ping
    ``pred_depths`` (NaN where no bottom). This routine matches those to
    GPS, segments by crossing, applies duration filtering, and drops
    in-dock + invalid pings.

    Returns
    -------
    CrossingPings:
        ``df``: long-format DataFrame with columns
            file_id, ping_idx, ping_time, lat, lon, pred_depth_m,
            crossing_id, direction, arc_length_m
        plus diagnostic counts.
    """
    zarr_path = Path(zarr_path)
    fid = file_id or zarr_path.stem

    lat, lon, ping_time = _load_latlon(zarr_path)
    n_total = len(lat)

    if len(pred_depths) != n_total:
        # In some echopype dumps the Sv ping axis is shorter than the GPS
        # one if a trailing partial ping was dropped. Truncate to the
        # shorter of the two.
        n = min(n_total, len(pred_depths))
        lat, lon, ping_time = lat[:n], lon[:n], ping_time[:n]
        pred_depths = np.asarray(pred_depths)[:n]
        n_total = n

    crossings = detect_crossings(lon, ping_time)
    crossings = filter_by_duration(crossings, min_duration_min, max_duration_min)

    rows: list[dict] = []
    in_cross = 0
    for c in crossings:
        s_idx, e_idx = c["start"], c["end"]
        cid = f"{fid}_{s_idx:06d}_{c['direction']}"
        idx_range = np.arange(s_idx, e_idx + 1)
        lat_c, lon_c = lat[idx_range], lon[idx_range]
        pt_c = ping_time[idx_range]
        pd_c = np.asarray(pred_depths)[idx_range]

        # Drop NaN GPS, NaN depths, and points still inside dock buffers.
        valid = (
            np.isfinite(lat_c)
            & np.isfinite(lon_c)
            & np.isfinite(pd_c)
            & (pd_c > 0)
            & (lon_c > LON_HORTEN + DOCK_HALFWIDTH_DEG)
            & (lon_c < LON_MOSS - DOCK_HALFWIDTH_DEG)
        )
        if not valid.any():
            continue

        s = project_to_arclength(lat_c[valid], lon_c[valid])
        in_cross += int(valid.sum())

        for k, i_local in enumerate(np.where(valid)[0]):
            rows.append(
                {
                    "file_id": fid,
                    "ping_idx": int(idx_range[i_local]),
                    "ping_time": pt_c[i_local],
                    "lat": float(lat_c[i_local]),
                    "lon": float(lon_c[i_local]),
                    "pred_depth_m": float(pd_c[i_local]),
                    "crossing_id": cid,
                    "direction": c["direction"],
                    "arc_length_m": float(s[k]),
                }
            )

    df = pd.DataFrame(rows)
    return CrossingPings(
        df=df,
        n_pings_total=int(n_total),
        n_pings_in_crossing=in_cross,
        n_crossings=len(crossings),
    )


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

EPSILON_SIGMA_M = 0.5  # floor on per-ping sigma to avoid divide-by-zero


def _collapse_per_crossing(
    work: pd.DataFrame, group_cols: list[str], sigma_col: str | None
) -> pd.DataFrame:
    """One row per (bin, crossing): median depth (+ median sigma) in the bin."""
    agg = {"pred_depth_m": "median"}
    if sigma_col is not None:
        agg[sigma_col] = "median"
    return work.groupby(group_cols + ["crossing_id"]).agg(agg).reset_index()


def build_calibration_scores(
    df: pd.DataFrame,
    *,
    bin_size_m: float = 20.0,
    min_crossings_per_bin: int = 2,
    per_direction: bool = False,
    score_mode: str = "pairwise",
    sigma_col: str | None = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Per-bin nonconformity scores from repeated crossings.

    Parameters
    ----------
    df : long-format DataFrame from ``extract_crossing_pings``.
    bin_size_m : arc-length bin width.
    min_crossings_per_bin : minimum distinct crossings that must populate
        a bin for it to contribute scores (>=2).
    per_direction : if True, calibrate H2M and M2H bins separately and
        concatenate the scores. Useful if directional bias is suspected.
    score_mode : {'pairwise', 'median', 'pairwise_norm', 'median_norm'}
        * 'pairwise':      |pred_A - pred_B| over unordered crossing pairs.
                           Constant self-consistency band ``pred ± q_hat``.
        * 'median':        |pred - bin_median|. Constant truth-style band.
        * 'pairwise_norm': |pred_A - pred_B| / max(sigma_A, eps) over *ordered*
                           pairs. Adaptive self-consistency band
                           ``pred ± q_hat * sigma_ping`` — see module docs.
        * 'median_norm':   |pred - bin_median| / max(sigma, eps). Adaptive
                           truth-style band ``pred ± q_hat * sigma_ping``.
    sigma_col : name of the per-ping uncertainty column (e.g. 'sigma_m',
        the probability-transition width). Required for the '*_norm' modes.

    Returns
    -------
    scores : 1-D float64 array of nonconformity scores (dimensionless for the
             normalized modes, metres otherwise).
    bin_info : DataFrame with one row per used bin
        (bin, direction, median_depth_m, n_pings, n_crossings).
    """
    valid_modes = ("pairwise", "median", "pairwise_norm", "median_norm")
    if score_mode not in valid_modes:
        raise ValueError(f"score_mode must be one of {valid_modes}, got {score_mode!r}")
    normalized = score_mode.endswith("_norm")
    if normalized and sigma_col is None:
        raise ValueError(f"score_mode={score_mode!r} requires a sigma_col")
    if df.empty:
        return np.array([], dtype=np.float64), pd.DataFrame()

    work = df.copy()
    work["bin"] = bin_arclength(work["arc_length_m"].to_numpy(), bin_size_m)
    group_cols = ["bin", "direction"] if per_direction else ["bin"]

    per_crossing = _collapse_per_crossing(work, group_cols, sigma_col)
    n_pings_per_bin = work.groupby(group_cols)["pred_depth_m"].size().rename("n_pings")

    scores: list[np.ndarray] = []
    bin_records: list[dict] = []
    for key, sub in per_crossing.groupby(group_cols):
        if len(sub) < min_crossings_per_bin:
            continue
        vals = sub["pred_depth_m"].to_numpy()
        med = float(np.median(vals))
        if normalized:
            sig = np.maximum(sub[sigma_col].to_numpy(), EPSILON_SIGMA_M)

        if score_mode == "pairwise":
            i, j = np.triu_indices(len(vals), k=1)
            s = np.abs(vals[i] - vals[j])
        elif score_mode == "median":
            s = np.abs(vals - med)
        elif score_mode == "pairwise_norm":
            # Ordered pairs: interval owned by crossing i, "other" is j.
            # Denominator is the owner's sigma so the band at a new ping is
            # pred ± q_hat * sigma_ping (consistent with validation).
            i, j = np.where(~np.eye(len(vals), dtype=bool))
            s = np.abs(vals[i] - vals[j]) / sig[i]
        else:  # 'median_norm'
            s = np.abs(vals - med) / sig

        scores.append(s)
        n_pings = int(n_pings_per_bin.loc[key]) if key in n_pings_per_bin.index else int(len(sub))
        rec = {"median_depth_m": med, "n_crossings": int(len(sub)), "n_pings": n_pings}
        if per_direction:
            bin_val, dir_val = key
            rec["bin"] = int(bin_val)
            rec["direction"] = dir_val
        else:
            bin_val = key[0] if isinstance(key, tuple) else key
            rec["bin"] = int(bin_val)
            rec["direction"] = "BOTH"
        bin_records.append(rec)

    if not scores:
        return np.array([], dtype=np.float64), pd.DataFrame(bin_records)

    return np.concatenate(scores), pd.DataFrame(bin_records)


def calibrate_labelless(scores: np.ndarray, alpha: float = 0.1) -> float:
    """Wrapper around ``conformal_quantile`` for label-less scores."""
    return conformal_quantile(scores, alpha=alpha)


def compute_bin_dispersion(
    df: pd.DataFrame,
    *,
    bin_size_m: float = 20.0,
    per_direction: bool = False,
    min_crossings_per_bin: int = 2,
) -> pd.DataFrame:
    """Per-bin robust dispersion of predicted depth across crossings.

    This is the most direct label-free difficulty signal: a bin where
    repeated crossings disagree a lot is genuinely hard, regardless of how
    confident the model's probability profile looks there. Returns a frame
    of (bin[, direction], sigma_disp_m) where sigma is 1.4826 * MAD of the
    per-crossing bin-median depths (robust standard deviation).

    Estimate this on the *calibration* crossings only, then map it onto test
    pings via ``assign_bin_sigma`` to keep the held-out estimate honest.
    """
    if df.empty:
        return pd.DataFrame()
    work = df.copy()
    work["bin"] = bin_arclength(work["arc_length_m"].to_numpy(), bin_size_m)
    group_cols = ["bin", "direction"] if per_direction else ["bin"]
    per_crossing = work.groupby(group_cols + ["crossing_id"])["pred_depth_m"].median().reset_index()

    recs = []
    for key, sub in per_crossing.groupby(group_cols):
        if len(sub) < min_crossings_per_bin:
            continue
        vals = sub["pred_depth_m"].to_numpy()
        mad = float(np.median(np.abs(vals - np.median(vals))))
        rec = {"sigma_disp_m": max(1.4826 * mad, EPSILON_SIGMA_M)}
        if per_direction:
            rec["bin"], rec["direction"] = int(key[0]), key[1]
        else:
            rec["bin"] = int(key[0] if isinstance(key, tuple) else key)
        recs.append(rec)
    return pd.DataFrame(recs)


def assign_bin_sigma(
    df: pd.DataFrame,
    dispersion: pd.DataFrame,
    *,
    bin_size_m: float = 20.0,
    per_direction: bool = False,
    out_col: str = "sigma_m",
) -> pd.DataFrame:
    """Map per-bin dispersion onto each ping's ``out_col``.

    Pings in bins absent from ``dispersion`` (e.g. test-only bins) fall back
    to the global median dispersion so they still receive a finite band.
    """
    work = df.copy()
    work["bin"] = bin_arclength(work["arc_length_m"].to_numpy(), bin_size_m)
    keys = ["bin", "direction"] if per_direction else ["bin"]
    if dispersion.empty:
        work[out_col] = 1.0
        return work.drop(columns="bin")
    fallback = float(dispersion["sigma_disp_m"].median())
    merged = work.merge(dispersion, on=keys, how="left")
    merged[out_col] = merged["sigma_disp_m"].fillna(fallback)
    return merged.drop(columns=["bin", "sigma_disp_m"])


# ---------------------------------------------------------------------------
# Validation: pairwise coverage on held-out crossings
# ---------------------------------------------------------------------------


def split_crossings(
    df: pd.DataFrame, calib_frac: float = 0.5, seed: int = 0
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split rows by crossing_id (not by ping) for calibration vs test."""
    if df.empty:
        return df.copy(), df.copy()
    rng = np.random.default_rng(seed)
    # Stratify by direction so both H2M and M2H appear in both splits.
    parts_calib: list[pd.DataFrame] = []
    parts_test: list[pd.DataFrame] = []
    for direction, sub in df.groupby("direction"):
        crossings = np.asarray(sub["crossing_id"].unique(), dtype=object)
        rng.shuffle(crossings)
        n_calib = int(round(len(crossings) * calib_frac))
        calib_ids = set(crossings[:n_calib])
        in_calib = sub["crossing_id"].isin(calib_ids)
        parts_calib.append(sub[in_calib])
        parts_test.append(sub[~in_calib])
    df_calib = pd.concat(parts_calib, ignore_index=True)
    df_test = pd.concat(parts_test, ignore_index=True)
    return df_calib, df_test


def pairwise_coverage(
    df_test: pd.DataFrame,
    q_hat: float,
    bin_size_m: float = 20.0,
    per_direction: bool = False,
    score_mode: str = "pairwise",
    sigma_col: str | None = None,
) -> dict:
    """Held-out pairwise coverage: fraction of crossing pairs at the same bin
    whose second prediction falls inside the first's interval.

    For constant modes the criterion is ``|pred_A - pred_B| <= q_hat``.
    For the adaptive '*_norm' modes the interval owned by A has half-width
    ``q_hat * max(sigma_A, eps)``, so the criterion is
    ``|pred_A - pred_B| <= q_hat * max(sigma_A, eps)`` over *ordered* pairs.

    A 90%-nominal q_hat should give ~90% empirical pair coverage when the
    crossings are exchangeable.
    """
    normalized = score_mode.endswith("_norm")
    empty = {
        "coverage": float("nan"),
        "n_pairs": 0,
        "median_abs_diff_m": float("nan"),
        "mean_band_width_m": float("nan"),
    }
    if df_test.empty:
        return empty
    if normalized and sigma_col is None:
        raise ValueError(f"score_mode={score_mode!r} requires a sigma_col")

    work = df_test.copy()
    work["bin"] = bin_arclength(work["arc_length_m"].to_numpy(), bin_size_m)
    group_cols = ["bin", "direction"] if per_direction else ["bin"]
    per_crossing = _collapse_per_crossing(work, group_cols, sigma_col)

    diffs: list[np.ndarray] = []
    widths: list[np.ndarray] = []
    covered: list[np.ndarray] = []
    for _, sub in per_crossing.groupby(group_cols):
        vals = sub["pred_depth_m"].to_numpy()
        if len(vals) < 2:
            continue
        if normalized:
            sig = np.maximum(sub[sigma_col].to_numpy(), EPSILON_SIGMA_M)
            i, j = np.where(~np.eye(len(vals), dtype=bool))  # ordered pairs
            d = np.abs(vals[i] - vals[j])
            half = q_hat * sig[i]
        else:
            i, j = np.triu_indices(len(vals), k=1)  # unordered pairs
            d = np.abs(vals[i] - vals[j])
            half = np.full(len(d), q_hat)
        diffs.append(d)
        widths.append(2.0 * half)
        covered.append(d <= half)

    if not diffs:
        return empty

    all_d = np.concatenate(diffs)
    all_w = np.concatenate(widths)
    all_c = np.concatenate(covered)
    return {
        "coverage": float(all_c.mean()),
        "n_pairs": int(len(all_d)),
        "median_abs_diff_m": float(np.median(all_d)),
        "mean_abs_diff_m": float(np.mean(all_d)),
        "mean_band_width_m": float(np.mean(all_w)),
        "median_band_width_m": float(np.median(all_w)),
    }


def predict_adaptive_band(pred_depths, sigma, q_hat, epsilon: float = EPSILON_SIGMA_M):
    """Per-ping adaptive band: pred ± q_hat * max(sigma, eps).

    Returns (upper, lower, band_width) — upper is the shallower edge.
    """
    pred = np.asarray(pred_depths, dtype=np.float64)
    sig = np.maximum(np.asarray(sigma, dtype=np.float64), epsilon)
    half = q_hat * sig
    return pred - half, pred + half, 2.0 * half


def pairwise_coverage_curve(
    df_calib: pd.DataFrame,
    df_test: pd.DataFrame,
    alphas: Iterable[float] | None = None,
    *,
    bin_size_m: float = 20.0,
    per_direction: bool = False,
    min_crossings_per_bin: int = 2,
    score_mode: str = "pairwise",
    sigma_col: str | None = None,
) -> pd.DataFrame:
    """Sweep alpha; return (alpha, q_hat, empirical_coverage, n_pairs).

    For the adaptive modes ``q_hat`` is dimensionless (a multiplier on the
    per-ping sigma) and the reported ``mean_band_width_m`` is the realised
    metric width.
    """
    if alphas is None:
        alphas = np.arange(0.02, 0.51, 0.02)
    alphas = np.asarray(list(alphas), dtype=np.float64)

    scores, _ = build_calibration_scores(
        df_calib,
        bin_size_m=bin_size_m,
        min_crossings_per_bin=min_crossings_per_bin,
        per_direction=per_direction,
        score_mode=score_mode,
        sigma_col=sigma_col,
    )
    if len(scores) == 0:
        return pd.DataFrame(
            {
                "alpha": alphas,
                "q_hat": np.nan,
                "empirical_coverage": np.nan,
                "mean_band_width_m": np.nan,
                "n_pairs": 0,
            }
        )

    records = []
    for a in alphas:
        q = conformal_quantile(scores, alpha=float(a))
        cov = pairwise_coverage(
            df_test,
            q,
            bin_size_m=bin_size_m,
            per_direction=per_direction,
            score_mode=score_mode,
            sigma_col=sigma_col,
        )
        records.append(
            {
                "alpha": float(a),
                "q_hat": float(q),
                "empirical_coverage": cov["coverage"],
                "mean_band_width_m": cov.get("mean_band_width_m", np.nan),
                "n_pairs": cov["n_pairs"],
            }
        )
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def plot_pairwise_coverage_curve(
    curve: pd.DataFrame, save_path: Path | str | None = None, title: str | None = None
):
    """Empirical vs nominal coverage curve, label-less."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    nominal = 1.0 - curve["alpha"].to_numpy()
    empirical = curve["empirical_coverage"].to_numpy()
    ax.plot([0, 1], [0, 1], color="gray", linestyle=":", lw=1, label="ideal (y = x)")
    ax.plot(nominal, empirical, marker="o", lw=1.5, color="#1f77b4", label="label-less (pairwise)")
    ax.set_xlabel("Nominal coverage (1 - α)")
    ax.set_ylabel("Empirical pairwise coverage")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    ax.set_title(title or "Label-less conformal calibration")
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(str(save_path), bbox_inches="tight")
    return fig


def plot_labelless_band(
    df_crossing: pd.DataFrame,
    q_hat: float,
    df_neighbour: pd.DataFrame | None = None,
    bin_size_m: float = 20.0,
    save_path: Path | str | None = None,
    title: str | None = None,
    sigma_col: str | None = None,
    ymax: float | None = None,
):
    """Depth profile of one crossing with the conformal band shaded.

    If ``sigma_col`` is given the band is adaptive (``pred ± q_hat * sigma``,
    per-ping); otherwise it is the constant ``pred ± q_hat`` band. If
    ``df_neighbour`` is given, overlay other crossings sharing bins as a
    visual coverage check. ``ymax`` clips the depth axis (the white overlays
    can otherwise spike to model-failure depths and squash the channel).
    """
    import matplotlib.pyplot as plt

    if df_crossing.empty:
        raise ValueError("df_crossing is empty")

    sub = df_crossing.sort_values("arc_length_m").reset_index(drop=True)
    s = sub["arc_length_m"].to_numpy() / 1000.0  # plot in km
    d = sub["pred_depth_m"].to_numpy()

    if sigma_col is not None:
        half = q_hat * np.maximum(sub[sigma_col].to_numpy(), EPSILON_SIGMA_M)
        band_label = f"adaptive band (q={q_hat:.2f}×σ, mean ±{np.nanmean(half):.2f} m)"
    else:
        half = np.full(len(d), q_hat)
        band_label = f"constant band (±{q_hat:.2f} m)"

    fig, ax = plt.subplots(figsize=(11, 4.5), dpi=150)
    ax.fill_between(s, d - half, d + half, color="#ff7f0e", alpha=0.30, label=band_label)
    ax.plot(s, d, color="#d62728", lw=1.4, label="this crossing")

    if df_neighbour is not None and not df_neighbour.empty:
        for cid, g in df_neighbour.groupby("crossing_id"):
            g = g.sort_values("arc_length_m")
            ax.plot(
                g["arc_length_m"].to_numpy() / 1000.0,
                g["pred_depth_m"].to_numpy(),
                color="white",
                alpha=0.85,
                lw=0.9,
            )
        ax.plot([], [], color="white", lw=0.9, label="other crossings")

    top = ymax if ymax is not None else float(np.nanmax(d + half)) * 1.05
    ax.set_ylim(top, 0)  # inverted depth axis, clipped
    ax.set_xlabel("Arc-length along route (km, Horten → Moss)")
    ax.set_ylabel("Predicted bottom depth (m)")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right")
    if title is None:
        cid0 = sub["crossing_id"].iloc[0]
        title = f"Label-less conformal band — {cid0}"
    ax.set_title(title)
    ax.set_facecolor("#222222")
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(str(save_path), bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Convenience: dump calibration metadata
# ---------------------------------------------------------------------------


@dataclass
class LabellessCalibration:
    alpha: float
    q_hat_m: float
    n_calib_scores: int
    n_calib_pings: int
    n_test_pings: int
    bin_size_m: float
    min_crossings_per_bin: int
    per_direction: bool
    score_mode: str
    coverage: dict

    def to_json(self, path: Path | str) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2, default=str))
