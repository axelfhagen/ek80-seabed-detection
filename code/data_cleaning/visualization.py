"""Plotting helpers for data quality analysis."""

import numpy as np
import matplotlib.pyplot as plt
from .metrics import get_bottom_line
from .io import load_patch_by_uid


def plot_patch_with_mask(patch, mask, uid, ax, zoom_bottom=True, margin=150, subtitle=None):
    """Show ch0 echogram with bottom line overlay on a given axes.

    When zoom_bottom is True, the view is cropped to +-margin samples around
    the median bottom depth so the boundary region is visible.
    """
    ch0 = patch[0]
    bottom = get_bottom_line(mask)
    valid = bottom >= 0
    H, W = ch0.shape

    if zoom_bottom and valid.any():
        med = int(np.median(bottom[valid]))
        row_lo = max(0, med - margin)
        row_hi = min(H, med + margin)
        ch0_crop = ch0[row_lo:row_hi, :]
        bottom_shifted = np.where(valid, bottom - row_lo, -1)
    else:
        ch0_crop = ch0
        bottom_shifted = bottom

    ax.imshow(ch0_crop, aspect="auto", cmap="viridis", vmin=0, vmax=1, interpolation="nearest")

    xs = np.arange(W)
    valid_shifted = bottom_shifted >= 0
    if valid_shifted.any():
        ax.plot(xs[valid_shifted], bottom_shifted[valid_shifted], "r-", lw=1, alpha=0.9)

    ax.set_title(uid, fontsize=7, fontfamily="monospace", pad=2)
    ax.set_xticks([])
    ax.set_yticks([])
    if subtitle:
        ax.set_xlabel(subtitle, fontsize=6, fontfamily="monospace", color="0.4", labelpad=2)


def gallery(
    uids,
    title,
    df=None,
    patch_dir=None,
    ncols=5,
    nrows=4,
    zoom_bottom=True,
    margin=150,
    score_col=None,
):
    """Plot a grid of patches identified by their uids.

    Parameters
    ----------
    uids : list of str
    title : str
    df : DataFrame (optional) — used for subtitle lookup and score display
    patch_dir : path-like — directory containing .npz files
    ncols, nrows : grid dimensions
    score_col : str (optional) — if set, display this score in the title
    """
    n = min(len(uids), ncols * nrows)
    actual_nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(actual_nrows, ncols, figsize=(4 * ncols, 3 * actual_nrows))
    if actual_nrows == 1:
        axes = np.atleast_2d(axes)
    axes = axes.flat

    for i in range(n):
        uid = uids[i]
        patch, mask = load_patch_by_uid(uid, patch_dir)

        subtitle = None
        patch_title = uid
        if df is not None:
            row = df.loc[df["uid"] == uid]
            if not row.empty:
                subtitle = row.iloc[0].get("source_zarr", None)
                if score_col and score_col in row.columns:
                    score = row.iloc[0][score_col]
                    patch_title = f"{uid}\n{score_col}={score:.3f}"

        plot_patch_with_mask(
            patch,
            mask,
            patch_title,
            axes[i],
            zoom_bottom=zoom_bottom,
            margin=margin,
            subtitle=subtitle,
        )

    for j in range(n, actual_nrows * ncols):
        axes[j].set_visible(False)

    fig.suptitle(title, fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.show()


def plot_metric_distributions(df, metric_cols):
    """Plot histograms for a list of metrics with percentile markers."""
    ncols = 4
    nrows = int(np.ceil(len(metric_cols) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 3 * nrows))
    axes = axes.flat

    for i, col in enumerate(metric_cols):
        ax = axes[i]
        vals = df[col].dropna()
        ax.hist(vals, bins=50, color="steelblue", edgecolor="none", alpha=0.8)
        ax.set_title(col, fontsize=10)
        ax.set_ylabel("count")
        if len(vals) > 0:
            p5, p50, p95 = np.percentile(vals, [5, 50, 95])
            ax.axvline(p5, color="orange", ls="--", lw=1, label=f"p5={p5:.3f}")
            ax.axvline(p50, color="red", ls="-", lw=1, label=f"p50={p50:.3f}")
            ax.axvline(p95, color="orange", ls="--", lw=1, label=f"p95={p95:.3f}")
            ax.legend(fontsize=7, loc="upper right")

    for j in range(len(metric_cols), nrows * ncols):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


def plot_score_distribution(df, score_col, reject_threshold=0.20, accept_threshold=0.60):
    """Plot score histogram with tier boundaries."""
    fig, ax = plt.subplots(figsize=(10, 4))
    scores = df[score_col].dropna()
    ax.hist(scores, bins=60, color="steelblue", edgecolor="none", alpha=0.8)
    ax.axvline(reject_threshold, color="red", ls="--", lw=2, label=f"reject < {reject_threshold}")
    ax.axvline(
        accept_threshold, color="green", ls="--", lw=2, label=f"accept >= {accept_threshold}"
    )

    n_reject = (scores < reject_threshold).sum()
    n_review = ((scores >= reject_threshold) & (scores < accept_threshold)).sum()
    n_accept = (scores >= accept_threshold).sum()
    ax.set_xlabel(score_col)
    ax.set_ylabel("Count")
    ax.set_title(f"reject={n_reject}, review={n_review}, accept={n_accept}")
    ax.legend()
    plt.tight_layout()
    plt.show()
