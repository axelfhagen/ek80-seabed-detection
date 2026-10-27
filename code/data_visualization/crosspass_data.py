"""Shared loader for repeated-crossing statistics.

The Basto ferry traverses the same Horten-Moss route many times, so the seabed
beneath any spot is observed across many crossings. This module turns the
per-ping bathymetry CSV into a tidy per-ping table tagged with a crossing id,
direction, and arc-length along the route, which the cross-pass-consistency and
failure-mining figure scripts both build on. The parsed table is cached to a
dependency-free .npz so repeat runs are instant.

Used by analyze_crosspass_consistency.py and analyze_failure_mining.py.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from validation.conformal_labelless import project_to_arclength, route_length_m

BATHY_CSV = PROJECT_ROOT / "Figures" / "bathymetry" / "bathymetry_data.csv"
CACHE = PROJECT_ROOT / "Figures" / "crosspass_consistency" / "_cache_pings.npz"

TZ = "Europe/Oslo"
LON_MIN, LON_MAX = 10.470, 10.680
LAT_MIN, LAT_MAX = 59.395, 59.445
LON_MIDPOINT = 10.58
HYST_DEG = 0.005  # ~280 m east-west hysteresis buffer
GAP_SECONDS = 300.0  # break a crossing after this recording gap
MIN_PINGS = 200  # ignore tiny crossing fragments
MIN_SPAN_KM = 3.0  # a real crossing covers a good stretch of the route

DIR_H2M, DIR_M2H = 0, 1  # integer direction codes (toward Moss / toward Horten)


def _iter_crossings(df: pd.DataFrame):
    """Yield (start, end, direction) index spans for each one-way crossing.

    A new crossing starts whenever the ferry moves clearly across the
    longitude midpoint (with hysteresis) or after a long recording gap.
    Direction is inferred from the net east-west travel of the span.
    """
    lon = df["lon"].to_numpy()
    ts = df["ts_local"].to_numpy()
    lo, hi = LON_MIDPOINT - HYST_DEG, LON_MIDPOINT + HYST_DEG

    starts = [0]
    side = 0
    dt = np.diff(ts).astype("timedelta64[s]").astype(float)
    for i in range(len(lon)):
        if i > 0 and dt[i - 1] > GAP_SECONDS:
            starts.append(i)
            side = 0
        if lon[i] < lo:
            new = -1
        elif lon[i] > hi:
            new = 1
        else:
            continue
        if side != 0 and new != side:
            starts.append(i)
        side = new
    starts.append(len(lon))

    for a, b in zip(starts[:-1], starts[1:]):
        if b - a < MIN_PINGS:
            continue
        direction = DIR_H2M if lon[b - 1] >= lon[a] else DIR_M2H
        yield a, b, direction


def _build_table() -> pd.DataFrame:
    """Stream the CSV, segment into crossings, project to arc-length."""
    print(f"Streaming {BATHY_CSV} ...")
    frames: list[pd.DataFrame] = []
    n_rows = 0
    for chunk in pd.read_csv(
        BATHY_CSV,
        usecols=["timestamp", "lat", "lon", "depth_m"],
        dtype={"lat": "float64", "lon": "float64", "depth_m": "float64"},
        parse_dates=["timestamp"],
        chunksize=1_000_000,
    ):
        chunk = chunk.dropna(subset=["timestamp", "lat", "lon", "depth_m"])
        chunk = chunk[
            (chunk["lon"] > LON_MIN)
            & (chunk["lon"] < LON_MAX)
            & (chunk["lat"] > LAT_MIN)
            & (chunk["lat"] < LAT_MAX)
            & (chunk["depth_m"] > 0)
            & (chunk["depth_m"] < 200)
        ]
        if chunk.empty:
            continue
        ts_local = chunk["timestamp"].dt.tz_localize("UTC").dt.tz_convert(TZ)
        chunk = chunk.assign(ts_local=ts_local, date_local=ts_local.dt.normalize())
        frames.append(chunk[["ts_local", "date_local", "lon", "lat", "depth_m"]])
        n_rows += len(chunk)
        print(f"  {n_rows:>11,} pings kept")

    df = pd.concat(frames, ignore_index=True).sort_values("ts_local").reset_index(drop=True)

    route_km = route_length_m() / 1000.0
    cid = np.full(len(df), -1, dtype=np.int64)
    direction = np.zeros(len(df), dtype=np.int8)
    next_id = 0
    for _, g in df.groupby("date_local", sort=True):
        gi = g.index.to_numpy()
        for a, b, d in _iter_crossings(g.reset_index(drop=True)):
            lat = g["lat"].to_numpy()[a:b]
            lon = g["lon"].to_numpy()[a:b]
            s_km = project_to_arclength(lat, lon) / 1000.0
            if s_km.max() - s_km.min() < MIN_SPAN_KM:
                continue
            cid[gi[a:b]] = next_id
            direction[gi[a:b]] = d
            next_id += 1

    df["crossing_id"] = cid
    df["direction"] = direction
    df = df[df["crossing_id"] >= 0].reset_index(drop=True)

    s_km = project_to_arclength(df["lat"].to_numpy(), df["lon"].to_numpy()) / 1000.0
    df["s_km"] = s_km
    df = df[(df["s_km"] >= 0) & (df["s_km"] <= route_km)].reset_index(drop=True)
    print(f"{df['crossing_id'].nunique()} crossings, {len(df):,} pings retained.")
    return df


def load_ping_table(rebuild: bool = False) -> pd.DataFrame:
    """Per-ping table with crossing id, direction, and arc-length (km).

    Columns: lat, lon, depth_m, crossing_id, direction (0=H2M, 1=M2H), s_km.
    Cached to a numeric .npz; pass rebuild=True to force a re-parse.
    """
    if CACHE.exists() and not rebuild:
        z = np.load(CACHE)
        return pd.DataFrame(
            {
                "lat": z["lat"],
                "lon": z["lon"],
                "depth_m": z["depth_m"],
                "crossing_id": z["crossing_id"],
                "direction": z["direction"],
                "s_km": z["s_km"],
            }
        )

    df = _build_table()
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        CACHE,
        lat=df["lat"].to_numpy(np.float32),
        lon=df["lon"].to_numpy(np.float32),
        depth_m=df["depth_m"].to_numpy(np.float32),
        crossing_id=df["crossing_id"].to_numpy(np.int32),
        direction=df["direction"].to_numpy(np.int8),
        s_km=df["s_km"].to_numpy(np.float32),
    )
    print(f"cached -> {CACHE}")
    return df


def robust_std(x: np.ndarray) -> float:
    """1.4826 * MAD: a robust standard-deviation estimate."""
    x = np.asarray(x, dtype=np.float64)
    if x.size < 2:
        return np.nan
    return 1.4826 * float(np.median(np.abs(x - np.median(x))))


def per_crossing_bin_medians(df: pd.DataFrame, key_cols: list[str]) -> pd.DataFrame:
    """Collapse to one median depth per (cell, crossing) to de-bias autocorrelation.

    Within a single crossing many consecutive pings hit the same cell, so we
    first take the median depth per (cell, crossing) before computing any
    across-crossing statistic. Returns the collapsed long frame.
    """
    return df.groupby(key_cols + ["crossing_id"])["depth_m"].median().reset_index()


if __name__ == "__main__":
    t = load_ping_table(rebuild=True)
    print(t.head())
    print(t.describe())
