"""Per-column accuracy across the patch width (live-deployment edge effect).

Training and offline evaluation feed the network patches in which the ping of
interest sits in the interior, with context on both sides. Live deployment puts
the most recent ping at the right edge, with no future context. If per-column
accuracy degrades toward the patch edges in offline inference, the live cost of
that asymmetry can be bounded from the offline data. This measures the effect:
single-patch inference (no overlap averaging) on a sample of held-out patches,
accumulating the bottom-line error and column IoU at each column across the
patch width.

Patches are the pre-resized 512-wide fast-cache patches from the validation and
test surveys. Runs on GPU; results cached to NPZ. Run the compute step on Slurm.

Output: Figures/edge_degradation/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from data_visualization.thesis_style import apply_thesis_style, savefig, LEGEND_SIZE

CACHE_DIR = PROJECT_ROOT / "Datasets" / "patch_cache"
CKPT = (
    PROJECT_ROOT
    / "code"
    / "checkpoints"
    / "exp_20260420_080950_finetune_stages"
    / "unet_resnet34_best.pth"
)
OUTPUT_DIR = PROJECT_ROOT / "Figures" / "edge_degradation"
CACHE_NPZ = OUTPUT_DIR / "per_column.npz"

W = 512  # fast-cache patch width
THRESHOLD = 0.5


def _eval_survey_prefixes():
    from deep_learning.training.survey_groups import SURVEYS, SPLITS

    surveys = SPLITS["val"] + SPLITS["test"]
    return tuple(p for s in surveys for p in SURVEYS[s]["date_prefixes"])


def _bottom_idx(binary_col):
    hits = np.where(binary_col)[0]
    return hits[0] if len(hits) else -1


def compute(max_patches, batch):
    import torch
    from deep_learning.unet_model import create_unet

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = create_unet("resnet34", encoder_weights=None, in_channels=3)
    sd = torch.load(CKPT, map_location=device)
    if isinstance(sd, dict):
        sd = sd.get("model_state_dict", sd)
    model.load_state_dict(sd)
    model.to(device).eval()

    prefixes = _eval_survey_prefixes()
    files = sorted(
        p for p in CACHE_DIR.glob("*.512x512.patches.npy") if p.name.startswith(prefixes)
    )
    print(f"{len(files)} val/test fast-cache files")

    # Accumulators over columns.
    abs_err_sum = np.zeros(W)  # |pred_row - gt_row| per column
    err_count = np.zeros(W)  # columns with both bottoms valid
    tp = np.zeros(W)
    fp = np.zeros(W)
    fn = np.zeros(W)
    n_used = 0
    rng = np.random.RandomState(42)
    rng.shuffle(files)

    buf_imgs = []
    buf_msks = []

    def flush():
        nonlocal n_used
        if not buf_imgs:
            return
        x = torch.from_numpy(np.stack(buf_imgs)).float().to(device)
        with torch.no_grad():
            logits = model(x)
            prob = torch.sigmoid(logits).cpu().numpy()[:, 0]  # (B, H, W)
        for pb, mk in zip(prob, buf_msks):
            predb = pb >= THRESHOLD
            gtb = mk.astype(bool)
            tp_c = (predb & gtb).sum(axis=0)
            fp_c = (predb & ~gtb).sum(axis=0)
            fn_c = (~predb & gtb).sum(axis=0)
            tp[:] += tp_c
            fp[:] += fp_c
            fn[:] += fn_c
            for c in range(W):
                gi = _bottom_idx(gtb[:, c])
                pi = _bottom_idx(predb[:, c])
                if gi >= 0 and pi >= 0:
                    abs_err_sum[c] += abs(pi - gi)
                    err_count[c] += 1
            n_used += 1
        buf_imgs.clear()
        buf_msks.clear()

    for f in files:
        if n_used >= max_patches:
            break
        patches = np.load(f, mmap_mode="r")  # (N, 3, 512, 512)
        masks = np.load(
            str(f).replace(".patches.npy", ".masks.npy"), mmap_mode="r"
        )  # (N, 512, 512)
        n = patches.shape[0]
        take = min(n, max(1, max_patches // max(1, len(files) // 1)))
        sel = rng.choice(n, min(take, n), replace=False)
        for i in sel:
            if n_used + len(buf_imgs) >= max_patches:
                break
            buf_imgs.append(np.asarray(patches[i]))
            buf_msks.append(np.asarray(masks[i]))
            if len(buf_imgs) >= batch:
                flush()
    flush()

    mean_abs_err = np.divide(abs_err_sum, err_count, out=np.full(W, np.nan), where=err_count > 0)
    col_iou = np.divide(tp, tp + fp + fn, out=np.full(W, np.nan), where=(tp + fp + fn) > 0)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        CACHE_NPZ, mean_abs_err=mean_abs_err, col_iou=col_iou, err_count=err_count, n_patches=n_used
    )
    print(f"{n_used} patches; cached -> {CACHE_NPZ}")
    return np.load(CACHE_NPZ)


def plot(d):
    apply_thesis_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    mean_abs_err = d["mean_abs_err"]
    col_iou = d["col_iou"]
    n = int(d["n_patches"])
    ping = np.arange(W)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)

    ax1.plot(ping, col_iou, color="#1f77b4", lw=2)
    interior = col_iou[int(0.25 * W) : int(0.75 * W)]
    ax1.axhline(
        np.nanmean(interior),
        color="0.4",
        ls="--",
        lw=1.5,
        label=f"interior mean = {np.nanmean(interior):.3f}",
    )
    ax1.axvline(W - 1, color="#d62728", ls=":", lw=2)
    ax1.set_ylabel("Per-column IoU")
    ax1.set_title(f"Accuracy across the patch width (n = {n} patches)")
    ax1.legend(fontsize=LEGEND_SIZE - 2, loc="lower center")

    ax2.plot(ping, mean_abs_err, color="#7f3b08", lw=2)
    ax2.axhline(
        np.nanmean(mean_abs_err[int(0.25 * W) : int(0.75 * W)]),
        color="0.4",
        ls="--",
        lw=1.5,
        label="interior mean",
    )
    ax2.axvline(W - 1, color="#d62728", ls=":", lw=2, label="most recent ping (live edge)")
    ax2.set_ylabel("Mean bottom error (rows)")
    ax2.set_xlabel(f"Ping index within patch (0 = oldest, {W - 1} = newest)")
    ax2.set_title("Bottom-line error across the patch width")
    ax2.legend(fontsize=LEGEND_SIZE - 2, loc="upper center")

    fig.tight_layout()
    savefig(fig, OUTPUT_DIR / "edge_degradation.pdf")
    plt.close(fig)
    # Report the edge effect.
    edge = np.nanmean(col_iou[-int(0.05 * W) :])
    interior_m = np.nanmean(col_iou[int(0.25 * W) : int(0.75 * W)])
    print(f"interior IoU={interior_m:.4f}, right-edge IoU={edge:.4f}, drop={interior_m - edge:.4f}")
    print(f"Wrote {OUTPUT_DIR / 'edge_degradation.pdf'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--compute", action="store_true")
    ap.add_argument("--max-patches", type=int, default=600)
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()
    if args.compute or not CACHE_NPZ.exists():
        d = compute(args.max_patches, args.batch)
    else:
        d = np.load(CACHE_NPZ)
        print(f"Loaded cached {CACHE_NPZ}")
    plot(d)


if __name__ == "__main__":
    main()
