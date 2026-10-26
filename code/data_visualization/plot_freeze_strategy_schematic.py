"""Encoder freezing strategies for the transfer-learning section of the Method.

Schematic of the ResNet-34 U-Net under the three transfer-learning regimes
compared in the Results: a fully frozen encoder, staged unfreezing (only the
input stem frozen, every residual stage and the decoder trained), and full
fine-tuning. freeze_encoder_stages(m, 3) freezes the first three encoder
children -- conv1, bn1, relu -- which are the input stem, so 99.96% of the
weights stay trainable. Frozen blocks are greyed and hatched; trainable blocks
are coloured. Trainable parameter fractions are computed from the actual model.

Output: Figures/freeze_strategy_schematic/
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from data_visualization.thesis_style import apply_thesis_style, savefig

OUTPUT_DIR = PROJECT_ROOT / "Figures" / "freeze_strategy_schematic"

ENC_COLOR = "#4c78a8"
DEC_COLOR = "#54a24b"
FROZEN_FC = "#d9d9d9"
NEUTRAL_FC = "#f0f0f0"

# Pipeline blocks, top to bottom. role: enc | dec | neutral.
BLOCKS = [
    ("Input (3 ch)", "neutral"),
    ("Stage 1 (stem)\nconv1 + bn", "enc"),
    ("Stage 2\nlayer1", "enc"),
    ("Stage 3\nlayer2", "enc"),
    ("Stage 4\nlayer3", "enc"),
    ("Stage 5\nlayer4", "enc"),
    ("Decoder\n(5 blocks + skips)", "dec"),
    ("Seg. head", "dec"),
    ("Bottom mask", "neutral"),
]

# Which encoder stages (1-indexed) are trained under each strategy; decoder and
# head are always trained.
STRATEGIES = [
    ("Encoder frozen", set()),
    ("Staged unfreezing", {2, 3, 4, 5}),
    ("Full fine-tuning", {1, 2, 3, 4, 5}),
]


def trainable_fractions():
    """Return {strategy_name: trainable_pct} from the real ResNet-34 U-Net."""
    try:
        from deep_learning.unet_model import (
            create_unet, freeze_encoder, freeze_encoder_stages,
            get_trainable_param_count)
    except Exception as exc:  # pragma: no cover
        print(f"[warn] model import failed ({exc}); skipping param counts")
        return {}

    out = {}
    for name, builder in {
        "Encoder frozen": lambda m: freeze_encoder(m),
        "Staged unfreezing": lambda m: freeze_encoder_stages(m, 3),
        "Full fine-tuning": lambda m: m,
    }.items():
        m = create_unet("resnet34", encoder_weights=None, in_channels=3)
        builder(m)
        out[name] = get_trainable_param_count(m)["trainable_pct"]
    return out


def main():
    apply_thesis_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pct = trainable_fractions()

    n = len(BLOCKS)
    bw, bh = 2.4, 0.82          # block width/height
    gap = 0.34                  # vertical gap between blocks
    col_dx = 5.2                # horizontal spacing of strategy columns
    y0 = 0.0

    fig, ax = plt.subplots(figsize=(15, 9))
    ys = [y0 + (n - 1 - i) * (bh + gap) for i in range(n)]

    for c, (strat, train_stages) in enumerate(STRATEGIES):
        cx = c * col_dx
        for i, (label, role) in enumerate(BLOCKS):
            y = ys[i]
            trainable = True
            if role == "enc":
                stage = int(label.split()[1])
                trainable = stage in train_stages
            elif role == "neutral":
                trainable = None  # not a parameterised block

            if trainable is None:
                fc, hatch, ec = NEUTRAL_FC, None, "0.6"
            elif trainable:
                fc = ENC_COLOR if role == "enc" else DEC_COLOR
                hatch, ec = None, "0.2"
            else:
                fc, hatch, ec = FROZEN_FC, "////", "0.55"

            box = FancyBboxPatch(
                (cx - bw / 2, y - bh / 2), bw, bh,
                boxstyle="round,pad=0.02,rounding_size=0.12",
                linewidth=1.4, edgecolor=ec, facecolor=fc, hatch=hatch,
                mutation_aspect=0.5, zorder=2)
            ax.add_patch(box)
            tc = "white" if (trainable and role in ("enc", "dec")) else "0.15"
            ax.text(cx, y, label, ha="center", va="center",
                    fontsize=12.5, color=tc, zorder=3)

            # Arrow to the next block down.
            if i < n - 1:
                ax.add_patch(FancyArrowPatch(
                    (cx, y - bh / 2), (cx, ys[i + 1] + bh / 2),
                    arrowstyle="-|>", mutation_scale=14,
                    color="0.45", lw=1.4, zorder=1))

        title = strat
        if strat in pct:
            p = pct[strat]
            ptxt = f"{p:.2f}%" if 99 < p < 100 else f"{p:.0f}%"
            title += f"\n({ptxt} of weights trained)"
        ax.text(cx, ys[0] + bh / 2 + 0.55, title, ha="center", va="bottom",
                fontsize=15, fontweight="bold")

    # Legend.
    handles = [
        plt.Rectangle((0, 0), 1, 1, fc=ENC_COLOR, ec="0.2"),
        plt.Rectangle((0, 0), 1, 1, fc=DEC_COLOR, ec="0.2"),
        plt.Rectangle((0, 0), 1, 1, fc=FROZEN_FC, ec="0.55", hatch="////"),
    ]
    ax.legend(handles,
              ["Trainable encoder", "Trainable decoder / head", "Frozen"],
              loc="lower center", bbox_to_anchor=(0.5, -0.04),
              ncol=3, fontsize=14, frameon=False)

    ax.set_xlim(-bw, (len(STRATEGIES) - 1) * col_dx + bw)
    ax.set_ylim(y0 - bh, ys[0] + bh / 2 + 1.4)
    ax.axis("off")
    ax.set_aspect("equal")
    fig.suptitle("Encoder freezing strategies for transfer learning", y=0.95)

    savefig(fig, OUTPUT_DIR / "freeze_strategy_schematic.pdf")
    plt.close(fig)
    print(f"trainable %: {pct}")
    print(f"Wrote {OUTPUT_DIR / 'freeze_strategy_schematic.pdf'}")


if __name__ == "__main__":
    main()
