"""Shared matplotlib styling for thesis figures.

Big-font regime: figures end up downscaled to roughly half- or full-page width
in the LaTeX report, so fonts and line widths are oversized to stay readable
after that shrink. Call apply_thesis_style() once near the top of every figure
script, size figures with the FIGSIZE_* constants, and use style_colorbar()
for any colorbar (rcParams font sizes do not propagate to colorbars).

    from data_visualization.thesis_style import (
        apply_thesis_style, FIGSIZE_STD, style_colorbar, savefig,
    )

    apply_thesis_style()
    fig, ax = plt.subplots(figsize=FIGSIZE_STD)
    ...
    savefig(fig, OUTPUT_DIR / 'name.pdf')
"""

import matplotlib.pyplot as plt

# Figure sizes in inches. Kept generous on purpose: LaTeX downscales them, and
# a larger canvas keeps the oversized fonts in proportion.
FIGSIZE_FULL = (12, 7)  # geo/map plots and full-width panels
FIGSIZE_STD = (10, 6)  # standard single-axes plot
FIGSIZE_HALF = (7, 5)  # half-width or paired plots

LINEWIDTH = 2.0

LABEL_SIZE = 18
TICK_SIZE = 16
LEGEND_SIZE = 16
TITLE_SIZE = 20
SUPTITLE_SIZE = 22

# Shared overlay colors for bottom-line figures.
REFERENCE_BOTTOM_COLOR = "#00ccff"  # cyan reference / labelled bottom
UNET_PREDICTION_COLOR = "#d62728"  # red U-Net prediction

_RC = {
    # Computer Modern, to match the thesis body font (book class, default CM).
    # cmr10 has no Unicode minus glyph, so render numbers/math through mathtext's
    # cm fontset and fall back to an ASCII hyphen for plain text minus signs;
    # this avoids missing-glyph warnings on negative tick labels.
    "font.family": "serif",
    "font.serif": ["cmr10", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True,
    "axes.unicode_minus": False,
    "axes.labelsize": LABEL_SIZE,
    "axes.titlesize": TITLE_SIZE,
    "figure.titlesize": SUPTITLE_SIZE,
    "xtick.labelsize": TICK_SIZE,
    "ytick.labelsize": TICK_SIZE,
    "legend.fontsize": LEGEND_SIZE,
    "lines.linewidth": LINEWIDTH,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
}


def apply_thesis_style():
    """Apply the thesis rcParams. Call once before creating figures."""
    plt.rcParams.update(_RC)


def style_colorbar(cbar, label=None, label_size=LABEL_SIZE, tick_size=TICK_SIZE):
    """Size a colorbar's label and ticks. rcParams do not reach colorbars."""
    if label is not None:
        cbar.set_label(label, fontsize=label_size)
    cbar.ax.tick_params(labelsize=tick_size)


def keep_spines(ax):
    """Re-enable the top and right spines that apply_thesis_style() removes.

    Use on map or boxed axes (e.g. geographic plots) that should keep a frame.
    """
    for side in ("top", "right"):
        ax.spines[side].set_visible(True)


def savefig(fig, path, **kwargs):
    """Save with thesis defaults (dpi=300, tight bbox). Format follows the
    path suffix: .pdf for line/scatter/bar plots, .png for map/image plots."""
    kwargs.setdefault("dpi", 300)
    kwargs.setdefault("bbox_inches", "tight")
    fig.savefig(path, **kwargs)
