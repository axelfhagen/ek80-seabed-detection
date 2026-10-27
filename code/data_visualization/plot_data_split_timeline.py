"""Chronological data-split timeline for the Basto Electric dataset.

Draws each recording period as a coloured box left to right in calendar order,
coloured by its role in the canonical split (train, validation, conformal
calibration, test). Box widths scale with the square root of the recording
period's raw data volume in gigabytes, so larger periods read as wider boxes
while the smallest periods stay legible. The NOAA Pacific hake survey is drawn
as an isolated box set apart from the Basto timeline to mark it as an
out-of-distribution test set that never enters training, calibration, or model
selection; it is drawn at a fixed width and is not on the volume scale.

Raw sizes (GB) are from Table tab:surveys in Latex/Chapters/04Data.tex. The
Sep-Oct 2024 campaign (65 GB combined) is split at the month boundary into a
September training block and an October validation block, with the 65 GB
apportioned by file count (292 vs 378 files -> ~28 / ~37 GB).

Source of the split: Latex/Chapters/04Data.tex (canonical split) and
Latex/Chapters/06Method.tex (experiment matrix).
"""

import sys
from math import sqrt
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_visualization.thesis_style import apply_thesis_style, savefig  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "Figures" / "data_split"

# Okabe-Ito colourblind-safe palette, one colour per split role.
ROLE_COLOR = {
    "train": "#0072B2",  # blue
    "val": "#E69F00",  # orange
    "calib": "#009E73",  # green
    "test": "#D55E00",  # vermillion
    "hake": "#999999",  # grey, isolated
}
ROLE_LABEL = {
    "train": "Training",
    "val": "Validation",
    "calib": "Conformal calibration",
    "test": "Test",
    "hake": "External OOD test",
}

# Basto Electric recording periods in calendar order: (label, role, raw GB,
# GB display string). Widths scale with sqrt(GB). The Sep-Oct 2024 campaign
# (65 GB) is split at the month boundary and the volume apportioned by file
# count (292 September vs 378 October files).
PERIODS = [
    ("Dec\n2023", "train", 2, "2 GB"),
    ("Jan\n2024", "train", 9, "9 GB"),
    ("Apr\n2024", "train", 6, "6 GB"),
    ("Jul\n2024", "train", 35, "35 GB"),
    ("Sep\n2024", "train", 28, "~28 GB"),
    ("Oct\n2024", "val", 37, "~37 GB"),
    ("Jan\n2025", "calib", 38, "38 GB"),
    ("Feb\n2025", "test", 12, "12 GB"),
    ("Jun\n2025", "test", 14, "14 GB"),
]

BOX_H = 1.18
BOX_LABEL_FONTSIZE = 16
BOX_VOLUME_FONTSIZE = 13
GAP = 0.16  # gap between adjacent timeline boxes
ISOLATION_GAP = 1.1  # extra gap before the isolated hake box
HAKE_W = 1.3  # fixed width for the isolated hake box (not on GB scale)
MIN_BOX_W = 0.62  # width of the smallest period, keeps its label legible

# Width scale: width = WIDTH_SCALE * sqrt(GB), fixed so the smallest period
# reaches MIN_BOX_W.
WIDTH_SCALE = MIN_BOX_W / sqrt(min(gb for _, _, gb, _ in PERIODS))


def _box_width(gb):
    return WIDTH_SCALE * sqrt(gb)


def _text_color(hex_color):
    """Black or white text depending on background luminance."""
    r, g, b = (int(hex_color[i : i + 2], 16) / 255 for i in (1, 3, 5))
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    return "white" if lum < 0.55 else "black"


def _box(ax, x, y, w, h, color, label, gb_label=None, dashed=False):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=2.0,
        edgecolor="black" if not dashed else "0.3",
        facecolor=color,
        linestyle="--" if dashed else "-",
        mutation_aspect=1.0,
    )
    ax.add_patch(patch)
    txt_color = _text_color(color)
    cx = x + w / 2
    if gb_label is None:
        ax.text(
            cx,
            y + h / 2,
            label,
            ha="center",
            va="center",
            fontsize=BOX_LABEL_FONTSIZE,
            fontweight="bold",
            color=txt_color,
        )
    else:
        ax.text(
            cx,
            y + h * 0.66,
            label,
            ha="center",
            va="center",
            fontsize=BOX_LABEL_FONTSIZE,
            fontweight="bold",
            color=txt_color,
        )
        ax.text(
            cx,
            y + h * 0.16,
            gb_label,
            ha="center",
            va="center",
            fontsize=BOX_VOLUME_FONTSIZE,
            color=txt_color,
        )


def main():
    apply_thesis_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(18.5, 4.6))

    y0 = 0.0

    # Timeline boxes, widths proportional to sqrt(GB).
    x = 0.0
    for label, role, gb, gb_label in PERIODS:
        w = _box_width(gb)
        _box(ax, x, y0, w, BOX_H, ROLE_COLOR[role], label, gb_label=gb_label)
        x += w + GAP

    timeline_right = x - GAP

    # Isolated hake box, set apart after a divider. Drawn at a fixed width
    # because the external survey is not on the Basto volume scale.
    hake_x = timeline_right + ISOLATION_GAP
    _box(ax, hake_x, y0, HAKE_W, BOX_H, ROLE_COLOR["hake"], "NOAA\nhake\nSH2306", dashed=True)

    # Dashed vertical divider between the Basto route and the external survey.
    div_x = timeline_right + ISOLATION_GAP / 2
    ax.plot(
        [div_x, div_x], [y0 - 0.55, y0 + BOX_H + 0.55], linestyle=":", color="0.4", linewidth=1.8
    )

    # Group labels above each block.
    ax.text(
        timeline_right / 2,
        y0 + BOX_H + 0.42,
        "Basto Electric route (Oslofjord)",
        ha="center",
        va="bottom",
        fontsize=24,
        style="italic",
        color="0.2",
    )
    ax.text(
        hake_x + HAKE_W / 2,
        y0 + BOX_H + 0.42,
        "Pacific (deep water)",
        ha="center",
        va="bottom",
        fontsize=24,
        style="italic",
        color="0.2",
    )

    # Time arrow beneath the timeline.
    arrow = FancyArrowPatch(
        (0, y0 - 0.45),
        (timeline_right, y0 - 0.45),
        arrowstyle="-|>",
        mutation_scale=22,
        linewidth=2.0,
        color="0.35",
    )
    ax.add_patch(arrow)
    ax.text(
        timeline_right / 2,
        y0 - 0.72,
        "chronological order",
        ha="center",
        va="top",
        fontsize=21,
        color="0.35",
    )

    # Legend for the roles.
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=ROLE_COLOR[r], edgecolor="black")
        for r in ("train", "val", "calib", "test", "hake")
    ]
    labels = [ROLE_LABEL[r] for r in ("train", "val", "calib", "test", "hake")]
    ax.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.42),
        ncol=5,
        frameon=False,
        fontsize=20,
        handlelength=1.3,
        columnspacing=1.4,
        handletextpad=0.5,
    )

    ax.set_xlim(-0.4, hake_x + HAKE_W + 0.4)
    ax.set_ylim(y0 - 1.15, y0 + BOX_H + 0.95)
    ax.set_aspect("equal")
    ax.axis("off")

    out = OUTPUT_DIR / "data_split_timeline.pdf"
    savefig(fig, out)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
