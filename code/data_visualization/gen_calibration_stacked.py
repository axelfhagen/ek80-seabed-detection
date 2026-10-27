"""Combined calibration figure: reliability diagram stacked on the probability histogram.

The two panels share a single x-axis so the predicted-probability scale lines up exactly
(0 below 0, 1 below 1) and the histogram reads as the base of the reliability diagram.

The numeric content here is reconstructed from the vector geometry of the original
reliability_diagram.pdf and prob_distribution.pdf (marker centres and bar heights extracted
and mapped back to data coordinates via the tick-label positions). The canonical source that
recomputes these arrays from the model is code/thesis_metrics_regen/diag_calibration.py, which
also emits this combined figure when run where the checkpoint and zarr data are available.
This standalone script regenerates the figure without the model.
"""

from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = ROOT / "Latex/Figures/regen_2026"
ECE = 0.0007

# Reliability curve (20 bins), reconstructed model points.
REL_X = np.array(
    [
        0.0012,
        0.0734,
        0.1265,
        0.1759,
        0.2263,
        0.2737,
        0.3243,
        0.3759,
        0.4256,
        0.4753,
        0.5257,
        0.5768,
        0.6249,
        0.676,
        0.7276,
        0.7756,
        0.8274,
        0.8792,
        0.9287,
        1.0008,
    ]
)
REL_Y = np.array(
    [
        0.0048,
        0.1631,
        0.1733,
        0.1905,
        0.2596,
        0.3841,
        0.4977,
        0.5762,
        0.678,
        0.7041,
        0.7404,
        0.8089,
        0.8024,
        0.8118,
        0.8265,
        0.8515,
        0.8522,
        0.8531,
        0.8694,
        1.0037,
    ]
)

# Probability histogram (200 bins), bar centres and log10 pixel counts.
HIST_X = np.linspace(0.0025, 0.9975, 200)
HIST_LOGC = np.array(
    [
        7.553382,
        5.395705,
        5.06167,
        4.819276,
        4.659795,
        4.551985,
        4.4385,
        4.361009,
        4.296721,
        4.234492,
        4.186567,
        4.151492,
        4.11992,
        4.108368,
        4.150174,
        4.078586,
        4.015573,
        3.986361,
        3.962356,
        3.97037,
        3.983388,
        3.994868,
        4.013954,
        4.025837,
        4.047042,
        4.04125,
        4.055419,
        4.059903,
        4.036965,
        4.049696,
        4.069222,
        4.061193,
        4.075662,
        4.073283,
        4.087831,
        4.091412,
        4.081333,
        4.078107,
        4.075903,
        4.076706,
        4.089392,
        4.081769,
        4.081254,
        4.074575,
        4.08422,
        4.09149,
        4.112413,
        4.114587,
        4.109076,
        4.128761,
        4.120718,
        4.11642,
        4.083668,
        4.064216,
        4.029966,
        4.02831,
        4.001594,
        3.981445,
        3.943263,
        3.912542,
        3.885439,
        3.869065,
        3.843285,
        3.818139,
        3.804698,
        3.787016,
        3.759858,
        3.768014,
        3.755848,
        3.745313,
        3.757356,
        3.734605,
        3.727227,
        3.73275,
        3.738204,
        3.722631,
        3.736979,
        3.735134,
        3.73531,
        3.76605,
        3.754755,
        3.737941,
        3.753153,
        3.761683,
        3.815732,
        3.788576,
        3.813973,
        3.753238,
        3.714769,
        3.696684,
        3.67001,
        3.665177,
        3.661125,
        3.663727,
        3.63984,
        3.640497,
        3.616071,
        3.631204,
        3.628399,
        3.625916,
        3.633657,
        3.633435,
        3.62205,
        3.628287,
        3.626256,
        3.639949,
        3.631092,
        3.632767,
        3.604454,
        3.607653,
        3.621936,
        3.637752,
        3.666106,
        3.683105,
        3.710131,
        3.723807,
        3.741513,
        3.730887,
        3.719175,
        3.697165,
        3.721724,
        3.712919,
        3.709291,
        3.688235,
        3.688529,
        3.697549,
        3.648522,
        3.664866,
        3.642572,
        3.631651,
        3.641372,
        3.671337,
        3.643877,
        3.68251,
        3.658192,
        3.666416,
        3.653225,
        3.659975,
        3.657877,
        3.68251,
        3.692723,
        3.69214,
        3.702986,
        3.71661,
        3.707513,
        3.715322,
        3.729641,
        3.81434,
        3.812429,
        3.844039,
        3.852923,
        3.85023,
        3.857862,
        3.826993,
        3.810361,
        3.832304,
        3.821182,
        3.805222,
        3.83894,
        3.845066,
        3.854665,
        3.882377,
        3.844039,
        3.845339,
        3.893584,
        3.916271,
        3.911898,
        3.948358,
        3.956896,
        3.997763,
        3.990538,
        4.016909,
        4.034629,
        4.066234,
        4.076666,
        4.124471,
        4.164136,
        4.223734,
        4.249782,
        4.293203,
        4.324001,
        4.380392,
        4.38085,
        4.412874,
        4.431557,
        4.458188,
        4.49072,
        4.526502,
        4.568093,
        4.60423,
        4.627777,
        4.649318,
        4.690642,
        4.749991,
        4.822025,
        4.924038,
        5.009591,
        5.148906,
        5.446607,
        8.319654,
    ]
)


def style_serif():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["cmr10", "CMU Serif", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "axes.formatter.use_mathtext": True,
            "axes.unicode_minus": False,
            "font.size": 12,
        }
    )


def make_figure(rel_x, rel_y, hist_x, hist_counts, ece, out_path):
    fig = plt.figure(figsize=(6.2, 6.6))
    gs = GridSpec(2, 1, height_ratios=[3.0, 1.35], hspace=0.07, figure=fig)
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1], sharex=ax0)

    ax0.plot([0, 1], [0, 1], "k--", label="perfect")
    ax0.plot(rel_x, rel_y, "o-", color="crimson", label="model")
    ax0.set_ylabel("Observed frequency")
    ax0.set_title(f"Reliability diagram (ECE = {ece:.4f})")
    ax0.legend(loc="upper left")
    ax0.set_ylim(-0.03, 1.05)
    plt.setp(ax0.get_xticklabels(), visible=False)

    ax1.bar(hist_x, hist_counts, width=1 / 200, color="steelblue", edgecolor="none")
    ax1.set_yscale("log")
    ax1.set_ylabel("Pixel count (log)")
    ax1.set_xlabel("Predicted bottom probability")
    ax1.set_ylim(10**3.4, 10**8.7)
    ax1.set_xlim(-0.03, 1.03)

    fig.align_ylabels([ax0, ax1])
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main():
    style_serif()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "calibration_stacked.pdf"
    make_figure(REL_X, REL_Y, HIST_X, 10.0**HIST_LOGC, ECE, out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
