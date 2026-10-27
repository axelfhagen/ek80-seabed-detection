"""Schematic of the 2D convolution operation for the theory chapter.

Shows a single kernel sliding over an input feature map to produce an output
feature map, illustrating local connectivity and weight sharing (the same
kernel is applied at every spatial position, which gives translation
equivariance).

Output: Figures/convolution_schematic/
"""

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from data_visualization.thesis_style import apply_thesis_style, savefig  # noqa: E402

OUTPUT_DIR = PROJECT_ROOT / "Figures" / "convolution_schematic"

# Colour scheme: window A (primary illustration) and window B (slid position).
COL_A = "#1f77b4"
COL_B = "#d95f02"
GRID_EDGE = "#444444"
KERNEL_FILL = "#cfcfcf"


def draw_grid(ax, x0, y_top, n_rows, n_cols, cell, edge=GRID_EDGE, lw=1.4):
    """Draw an n_rows x n_cols grid with row 0 at the top. Returns a function
    mapping (row, col) -> the cell's centre coordinate."""
    for r in range(n_rows):
        for c in range(n_cols):
            x = x0 + c * cell
            y = y_top - (r + 1) * cell
            ax.add_patch(Rectangle((x, y), cell, cell, facecolor="white", edgecolor=edge, lw=lw))

    def centre(r, c):
        return (x0 + (c + 0.5) * cell, y_top - (r + 0.5) * cell)

    return centre


def highlight_window(ax, x0, y_top, r0, c0, size, cell, colour, lw=3.0, ls="-"):
    """Outline a size x size window whose top-left cell is (r0, c0)."""
    x = x0 + c0 * cell
    y = y_top - (r0 + size) * cell
    ax.add_patch(
        Rectangle(
            (x, y),
            size * cell,
            size * cell,
            facecolor=colour,
            alpha=0.18,
            edgecolor=colour,
            lw=lw,
            ls=ls,
            zorder=3,
        )
    )


def highlight_cell(ax, x0, y_top, r, c, cell, colour, lw=3.0, ls="-"):
    x = x0 + c * cell
    y = y_top - (r + 1) * cell
    ax.add_patch(
        Rectangle(
            (x, y),
            cell,
            cell,
            facecolor=colour,
            alpha=0.30,
            edgecolor=colour,
            lw=lw,
            ls=ls,
            zorder=3,
        )
    )


def arrow(ax, start, end, colour, lw=2.4, ls="-"):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=22,
            lw=lw,
            ls=ls,
            color=colour,
            zorder=4,
            shrinkA=2,
            shrinkB=2,
        )
    )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    apply_thesis_style()

    fig, ax = plt.subplots(figsize=(13, 6.5))
    ax.set_aspect("equal")
    ax.axis("off")

    # --- input feature map: 6x6 ---
    in_cell = 0.75
    in_n = 6
    in_x0 = 0.0
    in_top = 6.0
    draw_grid(ax, in_x0, in_top, in_n, in_n, in_cell)
    # Window A (top-left) and window B (slid to bottom-right).
    highlight_window(ax, in_x0, in_top, 0, 0, 3, in_cell, COL_A)
    highlight_window(ax, in_x0, in_top, 3, 3, 3, in_cell, COL_B, ls=(0, (4, 3)))

    in_w = in_n * in_cell
    ax.text(
        in_x0 + in_w / 2, in_top + 0.35, "Input feature map", ha="center", va="bottom", fontsize=18
    )
    ax.text(
        in_x0 + in_w / 2,
        in_top - in_w - 0.30,
        r"$6 \times 6$",
        ha="center",
        va="top",
        fontsize=16,
        color="#333333",
    )

    # --- kernel: 3x3, drawn centred between input and output ---
    k_cell = 0.55
    k_n = 3
    k_x0 = in_x0 + in_w + 1.6
    k_top = 4.9
    draw_grid(ax, k_x0, k_top, k_n, k_n, k_cell, edge=COL_A, lw=2.0)
    k_w = k_n * k_cell
    for r in range(k_n):
        for c in range(k_n):
            cx, cy = k_x0 + (c + 0.5) * k_cell, k_top - (r + 0.5) * k_cell
            ax.add_patch(
                Rectangle(
                    (cx - k_cell / 2, cy - k_cell / 2),
                    k_cell,
                    k_cell,
                    facecolor=KERNEL_FILL,
                    edgecolor=COL_A,
                    lw=2.0,
                    zorder=2,
                )
            )
            ax.text(cx, cy, r"$w$", ha="center", va="center", fontsize=13, color="#333333")
    ax.text(
        k_x0 + k_w / 2, k_top + 0.30, r"$3 \times 3$ kernel", ha="center", va="bottom", fontsize=18
    )
    ax.text(
        k_x0 + k_w / 2,
        k_top - k_w - 0.25,
        "shared weights",
        ha="center",
        va="top",
        fontsize=15,
        color="#333333",
        style="italic",
    )

    # --- output feature map: 4x4 (6 - 3 + 1) ---
    out_cell = 0.75
    out_n = 4
    out_x0 = k_x0 + k_w + 1.9
    out_w = out_n * out_cell
    # vertically centre the output grid on the input grid
    in_mid = in_top - in_w / 2
    out_top = in_mid + out_w / 2
    out_centre = draw_grid(ax, out_x0, out_top, out_n, out_n, out_cell)
    highlight_cell(ax, out_x0, out_top, 0, 0, out_cell, COL_A)
    highlight_cell(ax, out_x0, out_top, 3, 3, out_cell, COL_B, ls=(0, (4, 3)))
    ax.text(
        out_x0 + out_w / 2,
        out_top + 0.35,
        "Output feature map",
        ha="center",
        va="bottom",
        fontsize=18,
    )
    ax.text(
        out_x0 + out_w / 2,
        out_top - out_w - 0.30,
        r"$4 \times 4$",
        ha="center",
        va="top",
        fontsize=16,
        color="#333333",
    )

    # --- arrows: window A -> kernel -> output cell A ---
    winA_right = (in_x0 + 3 * in_cell, in_top - 1.5 * in_cell)
    arrow(ax, winA_right, (k_x0 - 0.15, k_top - k_w / 2), COL_A)
    arrow(
        ax,
        (k_x0 + k_w + 0.15, k_top - k_w / 2),
        (out_centre(0, 0)[0] - out_cell / 2 - 0.05, out_centre(0, 0)[1]),
        COL_A,
    )

    # window B -> output cell B (dashed, conveys the slide)
    winB_right = (in_x0 + in_w, in_top - 4.5 * in_cell)
    arrow(
        ax,
        winB_right,
        (out_centre(3, 3)[0] - out_cell / 2 - 0.05, out_centre(3, 3)[1]),
        COL_B,
        lw=2.0,
        ls=(0, (4, 3)),
    )

    # explanatory note on weight sharing / equivariance
    ax.text(
        (in_x0 + out_x0 + out_w) / 2,
        0.55,
        "The same kernel slides across every position (weight sharing),\n"
        "so a feature is detected wherever it appears (translation equivariance).",
        ha="center",
        va="top",
        fontsize=15,
        color="#222222",
    )

    ax.set_title("2D convolution: local connectivity and weight sharing", fontsize=20, pad=14)

    ax.set_xlim(-0.5, out_x0 + out_w + 0.5)
    ax.set_ylim(-0.2, in_top + 1.1)

    savefig(fig, OUTPUT_DIR / "convolution_schematic.pdf")
    plt.close(fig)
    print(f"wrote {OUTPUT_DIR / 'convolution_schematic.pdf'}")


if __name__ == "__main__":
    main()
