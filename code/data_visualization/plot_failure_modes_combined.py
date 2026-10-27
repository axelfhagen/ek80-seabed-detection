"""Combine the noise and second-echo failure-mode echograms into one figure.

Places the two example echograms side by side, marks that black pixels are the
detected bottom, and adds a qualitative intensity colorbar (white = very low,
blue, green, red = high) matching the echogram colour scale.
"""

from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

from data_visualization.thesis_style import apply_thesis_style, style_colorbar, savefig

apply_thesis_style()

ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = ROOT / "Latex" / "Figures"
OUT_DIR = FIG_DIR / "failure_modes_combined"
OUT_DIR.mkdir(parents=True, exist_ok=True)

img_noise = mpimg.imread(FIG_DIR / "failure_mode_noise.png")
img_second = mpimg.imread(FIG_DIR / "failure_mode_secondecho.png")

# Echogram intensity scale: high -> low is red, green, blue, white.
intensity_cmap = LinearSegmentedColormap.from_list(
    "echo_intensity",
    [(0.0, "white"), (0.34, "blue"), (0.67, "green"), (1.0, "red")],
)

fig, axes = plt.subplots(1, 2, figsize=(13, 6))

axes[0].imshow(img_noise, aspect="auto")
axes[0].set_title("Surface noise and aeration", fontsize=20)

axes[1].imshow(img_second, aspect="auto")
axes[1].set_title("Second bottom echo", fontsize=20)

for ax, label in zip(axes, ("(a)", "(b)")):
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(True)
    ax.text(
        0.02,
        0.97,
        label,
        transform=ax.transAxes,
        fontsize=20,
        fontweight="bold",
        va="top",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8),
    )

# Legend: black pixels are the detected bottom.
legend_handle = Line2D(
    [0],
    [0],
    marker="s",
    color="none",
    markerfacecolor="black",
    markersize=16,
    label="Detected bottom",
)
fig.legend(
    handles=[legend_handle],
    loc="lower center",
    ncol=1,
    fontsize=16,
    frameon=False,
    bbox_to_anchor=(0.5, -0.04),
)

# Qualitative intensity colorbar on the right.
sm = plt.cm.ScalarMappable(cmap=intensity_cmap)
sm.set_array([])
cbar = fig.colorbar(sm, ax=axes, fraction=0.046, pad=0.03)
cbar.set_ticks([0.0, 0.33, 0.67, 1.0])
cbar.set_ticklabels(["Very low", "Low", "Medium", "High"])
style_colorbar(cbar, label="Backscatter intensity")

savefig(fig, OUT_DIR / "failure_modes_combined.png")
print(f"Saved to {OUT_DIR / 'failure_modes_combined.png'}")
