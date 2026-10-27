"""File-level quality aggregation and analysis."""

import matplotlib.pyplot as plt


def aggregate_file_quality(df, score_col="combined_score"):
    """Group patches by source file and compute aggregate quality metrics.

    Parameters
    ----------
    df : DataFrame with uid, source_zarr, and metric/score columns
    score_col : str — the score column to aggregate

    Returns
    -------
    file_df : DataFrame indexed by source_zarr, sorted worst-first
    """
    agg = (
        df.groupby("source_zarr")
        .agg(
            n_patches=("uid", "count"),
            mean_score=(score_col, "mean"),
            min_score=(score_col, "min"),
            max_score=(score_col, "max"),
            mean_zero_frac=("zero_frac", "mean"),
            mean_boundary_contrast=("boundary_contrast", "mean"),
            mean_bottom_smoothness=("bottom_smoothness", "mean"),
            mean_bottom_snr=("bottom_snr", "mean"),
        )
        .sort_values("mean_score", ascending=True)
    )
    return agg


def identify_bad_files(file_df, threshold=0.30):
    """Return source_zarr names where mean quality is below threshold.

    Parameters
    ----------
    file_df : DataFrame from aggregate_file_quality()
    threshold : float

    Returns
    -------
    bad_files : list of str (source_zarr names)
    """
    return file_df[file_df["mean_score"] < threshold].index.tolist()


def plot_file_quality_bar(file_df, score_col="mean_score"):
    """Horizontal bar chart of per-file quality scores.

    Parameters
    ----------
    file_df : DataFrame from aggregate_file_quality()
    score_col : str — column to plot
    """
    fig, ax = plt.subplots(figsize=(14, max(4, len(file_df) * 0.35)))
    scores = file_df[score_col]
    colors = plt.cm.RdYlGn(scores / scores.max() if scores.max() > 0 else scores)
    ax.barh(range(len(file_df)), scores, color=colors)
    ax.set_yticks(range(len(file_df)))
    labels = [f"{name} (n={int(row['n_patches'])})" for name, row in file_df.iterrows()]
    ax.set_yticklabels(labels, fontsize=7, fontfamily="monospace")
    ax.set_xlabel(score_col)
    ax.set_title("Per-file quality (sorted worst-first)")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.show()
