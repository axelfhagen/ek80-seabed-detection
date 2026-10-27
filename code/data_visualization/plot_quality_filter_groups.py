"""Visualise rule-based outlier filter groups on the training patch set.

Output: Figures/quality_filter_groups/

For each of ~16 rule-based filter groups (e.g. SURFACE_NOISE, JAGGED_BOTTOM,
WEAK_BOTTOM, BOTTOM_JUMPS, ...), this script:
  - selects matching patches from Datasets/patch_quality_metrics_extended.csv,
  - draws a 4x4 grid of (ch0 echogram + red mask-bottom overlay),
  - and writes a CSV listing every matched UID with the driving metric values.

A `_README.md` and a global `filter_summary.csv` are written into the same folder.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from data_cleaning.metrics import get_bottom_line  # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

CSV_IN = PROJECT_ROOT / "Datasets" / "patch_quality_metrics_extended.csv"
PATCH_DIR = PROJECT_ROOT / "Datasets" / "patch_cache"
OUTPUT_DIR = PROJECT_ROOT / "Figures" / "quality_filter_groups"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Filter-group definitions
# ---------------------------------------------------------------------------
# Each entry: name -> dict with
#   description : human-readable rule, used in the figure title
#   query       : function(df) -> boolean Series, True for matched patches
#   sort_by     : column name to sort matches by (descending = worst first); None = random
#   show_cols   : extra metric columns to display under each panel
#
# We deliberately span many orthogonal failure modes so the user can verify
# which rules actually flag real problems.


# Convenience for queries
def q(expr):
    """Return a closure that evaluates ``df.eval(expr)`` for safety."""

    def _f(df):
        return df.eval(expr)

    return _f


FILTER_GROUPS = {
    # ---- echogram noise / artefact groups ----
    "A1_HIGH_SATURATION": {
        "description": "saturation_frac > 0.05  (many pixels clipped at intensity 1.0)",
        "query": q("saturation_frac > 0.05"),
        "sort_by": "saturation_frac",
        "show_cols": ["saturation_frac", "high_intensity_frac", "top_band_intensity"],
    },
    "A2_SURFACE_GLARE": {
        "description": "top_band_intensity > 0.65 and ping_streak_score < 0.2  (broad surface glare, not interference)",
        "query": q("top_band_intensity > 0.65 and ping_streak_score < 0.2"),
        "sort_by": "top_band_intensity",
        "show_cols": ["top_band_intensity", "ping_streak_score", "water_col_high_frac"],
    },
    "A3_PING_INTERFERENCE": {
        "description": "ping_streak_score > 0.5  (per-ping column means vary wildly: vertical streak artefacts)",
        "query": q("ping_streak_score > 0.5"),
        "sort_by": "ping_streak_score",
        "show_cols": ["ping_streak_score", "row_streak_score", "top_band_intensity"],
    },
    "A4_HORIZONTAL_BANDS": {
        "description": "row_streak_score > 3.5  (row means vary strongly: persistent reflections or calibration bands)",
        "query": q("row_streak_score > 3.5"),
        "sort_by": "row_streak_score",
        "show_cols": ["row_streak_score", "ping_streak_score", "water_col_high_frac"],
    },
    "A5_LOW_INFO": {
        "description": "intensity_entropy < 0.5 or dynamic_range < 0.3  (mostly-zero or low-contrast patch)",
        "query": q("intensity_entropy < 0.5 or dynamic_range < 0.3"),
        "sort_by": "intensity_entropy",
        "show_cols": ["intensity_entropy", "dynamic_range", "zero_frac"],
    },
    "A6_WATER_COL_BRIGHT": {
        "description": "water_col_high_frac > 0.7  (water column has many >0.5 pixels: schools or noise above bottom)",
        "query": q("water_col_high_frac > 0.7"),
        "sort_by": "water_col_high_frac",
        "show_cols": ["water_col_high_frac", "water_col_p99", "top_band_intensity"],
    },
    # ---- bottom-line label problems ----
    "B1_JAGGED_BOTTOM": {
        "description": "bottom_quad_residual > 80  (mask bottom departs strongly from a smooth quadratic)",
        "query": q("bottom_quad_residual > 80"),
        "sort_by": "bottom_quad_residual",
        "show_cols": ["bottom_quad_residual", "bottom_gradient_p95", "bottom_outlier_frac"],
    },
    "B2_BOTTOM_JUMPS": {
        "description": "bottom_gradient_p95 > 25  (large between-ping jumps in the bottom line)",
        "query": q("bottom_gradient_p95 > 25"),
        "sort_by": "bottom_gradient_p95",
        "show_cols": ["bottom_gradient_p95", "bottom_jump_count", "bottom_quad_residual"],
    },
    "B3_BOTTOM_OUTLIERS": {
        "description": "bottom_outlier_frac > 0.15  (>15% of pings are >4*MAD outliers in the bottom line)",
        "query": q("bottom_outlier_frac > 0.15"),
        "sort_by": "bottom_outlier_frac",
        "show_cols": ["bottom_outlier_frac", "bottom_mad", "bottom_quad_residual"],
    },
    "B4_VERY_SHALLOW": {
        "description": "bottom_rel_depth < 0.02  (median bottom in top 2% of image: tiny water column)",
        "query": q("bottom_rel_depth < 0.02"),
        "sort_by": "bottom_rel_depth",  # ascending = worst first via reverse later
        "show_cols": ["bottom_rel_depth", "bottom_too_shallow", "bottom_intensity_p10"],
    },
    "B5_DEEP_BOTTOM": {
        "description": "bottom_rel_depth > 0.5  (bottom in lower half: unusual for this dataset)",
        "query": q("bottom_rel_depth > 0.5"),
        "sort_by": "bottom_rel_depth",
        "show_cols": ["bottom_rel_depth", "bottom_pingwise_mean", "bottom_intensity_p10"],
    },
    # ---- mask / labelling failures ----
    "C1_MASK_LEAKAGE": {
        "description": "mask_holes_above > 1e-4  (mask pixels exist above the per-column bottom-line: leak / multi-component)",
        "query": q("mask_holes_above > 1e-4"),
        "sort_by": "mask_holes_above",
        "show_cols": ["mask_holes_above", "largest_component_frac", "mask_connectivity"],
    },
    "C2_WEAK_BOTTOM": {
        "description": "bottom_intensity_p10 < 0.005 and bottom_pingwise_mean < 0.03  (label sits on dim pixels along the bottom)",
        "query": q("bottom_intensity_p10 < 0.005 and bottom_pingwise_mean < 0.03"),
        "sort_by": "bottom_pingwise_mean",
        "show_cols": ["bottom_pingwise_mean", "bottom_intensity_p10", "boundary_contrast"],
    },
    "C3_MISALIGNED_LABEL": {
        "description": "bottom_argmax_misalign > 50  (mask bottom > 50 px from local ch0 argmax)",
        "query": q("bottom_argmax_misalign > 50"),
        "sort_by": "bottom_argmax_misalign",
        "show_cols": ["bottom_argmax_misalign", "cross_freq_bottom_diff", "mask_edge_intensity"],
    },
    "C4_CROSS_FREQ_DISAGREE": {
        "description": "cross_freq_bottom_diff > 12  (ch0 vs ch1 disagree about bottom location)",
        "query": q("cross_freq_bottom_diff > 12"),
        "sort_by": "cross_freq_bottom_diff",
        "show_cols": ["cross_freq_bottom_diff", "bottom_argmax_misalign", "freq_coherence"],
    },
    "C5_HIGH_CURVATURE": {
        "description": "bottom_curvature > 5  (high mean |second-derivative| of the bottom line)",
        "query": q("bottom_curvature > 5"),
        "sort_by": "bottom_curvature",
        "show_cols": ["bottom_curvature", "bottom_quad_residual", "bottom_gradient_p95"],
    },
    # ---- combined positive/negative reference groups ----
    "D1_LIKELY_BAD_OR": {
        "description": "Union: BOTTOM_JUMPS or BOTTOM_OUTLIERS or MASK_LEAKAGE or WEAK_BOTTOM or MISALIGNED_LABEL",
        "query": lambda df: (
            (df["bottom_gradient_p95"] > 25)
            | (df["bottom_outlier_frac"] > 0.15)
            | (df["mask_holes_above"].fillna(0) > 1e-4)
            | ((df["bottom_intensity_p10"] < 0.005) & (df["bottom_pingwise_mean"] < 0.03))
            | (df["bottom_argmax_misalign"] > 50)
        ),
        "sort_by": None,  # random sample to cover diversity
        "show_cols": ["bottom_gradient_p95", "bottom_outlier_frac", "bottom_argmax_misalign"],
    },
    "D2_LIKELY_GOOD": {
        "description": (
            "Clean candidates: bottom_quad_residual <= 5, bottom_gradient_p95 <= 5, bottom_outlier_frac <= 0.02, "
            "bottom_argmax_misalign <= 15, mask_holes_above <= 1e-5, dynamic_range >= 0.5, top_band_intensity <= 0.5"
        ),
        "query": lambda df: (
            (df["bottom_quad_residual"] <= 5)
            & (df["bottom_gradient_p95"] <= 5)
            & (df["bottom_outlier_frac"] <= 0.02)
            & (df["bottom_argmax_misalign"] <= 15)
            & (df["mask_holes_above"].fillna(0) <= 1e-5)
            & (df["dynamic_range"] >= 0.5)
            & (df["top_band_intensity"] <= 0.5)
        ),
        "sort_by": None,
        "show_cols": ["bottom_quad_residual", "bottom_argmax_misalign", "dynamic_range"],
    },
}


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def _load_patch(uid, patch_dir):
    stem, idx_str = uid.split("::")
    idx = int(idx_str)
    data = np.load(patch_dir / f"{stem}.npz")
    return data["patches"][idx], data["masks"][idx]


def _zoom_around_bottom(ch0, mask, margin=200):
    bottom = get_bottom_line(mask)
    valid = bottom >= 0
    H, W = ch0.shape
    if valid.any():
        med = int(np.median(bottom[valid]))
        lo = max(0, med - margin)
        hi = min(H, med + margin)
    else:
        lo, hi = 0, H
    ch0_crop = ch0[lo:hi, :]
    bottom_shift = np.where(valid, bottom - lo, -1)
    return ch0_crop, bottom_shift, (lo, hi)


def _draw_panel(ax, uid, df_row, show_cols, zoom=True, margin=200):
    """Draw a single 4x4 cell."""
    try:
        patch, mask = _load_patch(uid, PATCH_DIR)
    except Exception as e:
        ax.set_axis_off()
        ax.text(
            0.5,
            0.5,
            f"LOAD ERR\n{e}",
            ha="center",
            va="center",
            fontsize=8,
            transform=ax.transAxes,
            color="red",
        )
        return

    ch0 = patch[0]
    if zoom:
        ch0_show, bottom, _ = _zoom_around_bottom(ch0, mask, margin=margin)
    else:
        ch0_show = ch0
        bottom = get_bottom_line(mask)

    ax.imshow(ch0_show, aspect="auto", cmap="viridis", vmin=0, vmax=1, interpolation="nearest")
    xs = np.arange(ch0_show.shape[1])
    valid = bottom >= 0
    if valid.any():
        ax.plot(xs[valid], bottom[valid], "r-", lw=1.0, alpha=0.95)

    # Title: uid (small font)
    ax.set_title(uid, fontsize=7, fontfamily="monospace", pad=2)
    ax.set_xticks([])
    ax.set_yticks([])

    # Subtitle: a couple of driving metrics
    if df_row is not None and show_cols:
        parts = []
        for c in show_cols:
            if c in df_row.index and pd.notna(df_row[c]):
                v = df_row[c]
                if isinstance(v, float):
                    parts.append(f"{c}={v:.3g}")
                else:
                    parts.append(f"{c}={v}")
        if parts:
            ax.set_xlabel(
                " | ".join(parts), fontsize=6, fontfamily="monospace", color="0.3", labelpad=2
            )


def render_group_panel(name, group, df, ncols=4, nrows=4, zoom_bottom=True, margin=200, seed=0):
    """Render one 4x4 (or fewer if small group) gallery for a filter group."""
    mask_bool = group["query"](df)
    sub = df[mask_bool]
    n_total = len(sub)

    if n_total == 0:
        print(f"  [skip] {name}: 0 matched patches", flush=True)
        return None, 0

    # Always render two views: zoomed-to-bottom AND full-echogram.
    sort_by = group.get("sort_by")
    if sort_by and sort_by in sub.columns:
        # Pick worst (descending), but for "shallow" we want ascending = closest to top
        ascending = sort_by == "bottom_rel_depth" and "SHALLOW" in name
        sub = sub.sort_values(sort_by, ascending=ascending)
        picks = sub.head(ncols * nrows)
    else:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(sub), size=min(ncols * nrows, len(sub)), replace=False)
        picks = sub.iloc[np.sort(idx)]

    uids = picks["uid"].tolist()
    n_show = len(uids)

    for view_name, zoom in [("zoom", True), ("full", False)]:
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
        axes = np.atleast_2d(axes).flat

        for i in range(nrows * ncols):
            ax = axes[i]
            if i < n_show:
                uid = uids[i]
                df_row = picks.iloc[i]
                _draw_panel(ax, uid, df_row, group["show_cols"], zoom=zoom, margin=margin)
            else:
                ax.set_visible(False)

        suptitle = (
            f"{name}  ({n_total} matched / {len(df)} total, "
            f"showing {n_show} {'worst' if sort_by else 'random'})"
            f"\n{group['description']}"
        )
        fig.suptitle(suptitle, fontsize=11, fontweight="bold", y=1.0)
        plt.tight_layout()
        out = OUTPUT_DIR / f"{name}_{view_name}.png"
        fig.savefig(out, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(
            f"  -> {out.relative_to(PROJECT_ROOT)}  ({n_show} panels, {n_total} matched)",
            flush=True,
        )

    return picks, n_total


# ---------------------------------------------------------------------------
# Per-group CSV listings
# ---------------------------------------------------------------------------


def write_group_csv(name, group, df, summary_rows):
    mask_bool = group["query"](df)
    sub = df[mask_bool].copy()
    keep_cols = ["uid", "source_zarr", "patch_idx"] + group["show_cols"]
    keep_cols = [c for c in keep_cols if c in sub.columns]
    sub = sub[keep_cols]
    sort_by = group.get("sort_by")
    if sort_by and sort_by in sub.columns:
        ascending = sort_by == "bottom_rel_depth" and "SHALLOW" in name
        sub = sub.sort_values(sort_by, ascending=ascending)
    out = OUTPUT_DIR / f"{name}_matches.csv"
    sub.to_csv(out, index=False)
    summary_rows.append(
        {
            "group": name,
            "n_matched": int(mask_bool.sum()),
            "frac_matched": float(mask_bool.mean()),
            "description": group["description"],
            "csv": out.name,
        }
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    df = pd.read_csv(CSV_IN)
    print(f"Loaded {len(df)} patches with {len(df.columns)} columns from {CSV_IN.name}")

    summary_rows = []
    for name, group in FILTER_GROUPS.items():
        print(f"[{name}]", flush=True)
        render_group_panel(name, group, df)
        write_group_csv(name, group, df, summary_rows)

    # Global summary CSV
    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values("n_matched", ascending=False)
    summary_csv = OUTPUT_DIR / "filter_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"\nWrote summary -> {summary_csv.relative_to(PROJECT_ROOT)}")

    # README
    readme = OUTPUT_DIR / "_README.md"
    lines = []
    lines.append("# Rule-based quality filter groups\n")
    lines.append("outliers  filtering  echogram  label-quality  diagnostic\n")
    lines.append(
        "One 4x4 grid per filter rule, drawn from `Datasets/patch_quality_metrics_extended.csv`. "
    )
    lines.append(
        "Each group has two views: `*_zoom.png` (zoomed +/-200 px around the median bottom) "
    )
    lines.append("and `*_full.png` (full echogram). Per-group CSV lists every matched UID.\n\n")
    lines.append(f"Source script: `code/data_visualization/{Path(__file__).name}`\n\n")
    lines.append("## Summary\n\n")
    lines.append("| group | n_matched | frac | description |\n")
    lines.append("|---|---:|---:|---|\n")
    for _, r in summary_df.iterrows():
        lines.append(
            f"| {r['group']} | {r['n_matched']} | {r['frac_matched']:.1%} | {r['description']} |\n"
        )
    readme.write_text("".join(lines))
    print(f"Wrote README   -> {readme.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
