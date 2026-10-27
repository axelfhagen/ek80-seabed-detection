"""Per-channel importance analysis for the U-Net seabed segmenter.

Loads the all-surveys-jun2025 checkpoint and quantifies how much each of
the three input channels (38 kHz, 200 kHz, freq-diff) contributes to the
prediction, using six complementary methods:

  1. Channel ablation: zero, mean, and shuffled-baseline replacement
  2. Pairwise channel ablation (drop two at a time) for synergy analysis
  3. Mean |input-gradient| per channel, aggregated across many patches
  4. Integrated Gradients per channel (50-step Riemann sum from zero baseline)
  5. Grad-CAM masked by per-channel input-gradient maps
  6. Per-channel input distributions for bottom vs water pixels

Output: Figures/channel_importance/
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path
import sys
import gc
import json
import argparse
import numpy as np
import cv2
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from deep_learning.unet_model import create_unet
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import SemanticSegmentationTarget


CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "code/checkpoints"
    / "exp_20260511_191547_all_surveys_jun2025"
    / "smp_unet_resnet34_best.pth"
)
CACHE_DIR = PROJECT_ROOT / "Datasets/patch_cache"
OUTPUT_DIR = PROJECT_ROOT / "Figures" / "channel_importance"

TEST_CACHE_FILES = [
    "D20250624-T181715_053ec569bebc.npz",
    "D20250625-T015027_905c23f15ed7.npz",
    "D20250625-T070947_5c198acd81b6.npz",
    "D20250625-T115550_a5e6c643dbf5.npz",
    "D20250625-T155644_3fea320b24bc.npz",
    "D20250625-T202259_67549b305f7e.npz",
    "D20250626-T042617_9658927628a0.npz",
    "D20250626-T084615_f0923604b66b.npz",
    "D20250626-T125158_1a735b9e662e.npz",
    "D20250626-T163945_6b70d0692d95.npz",
    "D20250626-T211527_1a43046a88e8.npz",
]

CHANNEL_NAMES = ["38 kHz", "200 kHz", "38 kHz minus 200 kHz"]
CHANNEL_SHORT = ["38 kHz", "200 kHz", "diff"]
CHANNEL_COLORS = ["#1f77b4", "#d62728", "#2ca02c"]

TARGET_SIZE = (512, 512)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

plt.rcParams.update(
    {
        "axes.titlesize": 20,
        "axes.labelsize": 18,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 16,
        "figure.titlesize": 22,
    }
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_model():
    model = create_unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1)
    sd = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    state = sd["model_state_dict"]
    cleaned = {
        k.replace("model.", "", 1) if k.startswith("model.") else k: v for k, v in state.items()
    }
    model.load_state_dict(cleaned, strict=True)
    return model.to(DEVICE).eval()


def resize_patch(patch_chw, mask=None, size=TARGET_SIZE):
    h, w = size
    hwc = np.transpose(patch_chw, (1, 2, 0))
    hwc_r = cv2.resize(hwc, (w, h), interpolation=cv2.INTER_LINEAR)
    out_chw = np.transpose(hwc_r, (2, 0, 1)).astype(np.float32)
    if mask is not None:
        m_r = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        return out_chw, m_r
    return out_chw


def predict_probs(model, input_chw):
    inp = torch.from_numpy(input_chw[np.newaxis]).float().to(DEVICE)
    with torch.no_grad():
        probs = torch.sigmoid(model(inp)).squeeze().cpu().numpy()
    return probs


def predict_probs_batch(model, input_nchw):
    inp = torch.from_numpy(input_nchw).float().to(DEVICE)
    with torch.no_grad():
        probs = torch.sigmoid(model(inp)).cpu().numpy()
    return probs.squeeze(1)


def iou_dice(pred_binary, gt_binary):
    pred = pred_binary.astype(bool)
    gt = gt_binary.astype(bool)
    inter = (pred & gt).sum()
    union = (pred | gt).sum()
    iou = inter / max(union, 1)
    dice = 2 * inter / max(pred.sum() + gt.sum(), 1)
    return float(iou), float(dice)


def bottom_row_per_col(binary_mask):
    """First row (smallest depth index) where the mask is positive, per column.
    Returns array of length n_cols with -1 where no bottom is detected."""
    h, w = binary_mask.shape
    out = np.full(w, -1, dtype=np.int64)
    for c in range(w):
        rows = np.where(binary_mask[:, c] > 0)[0]
        if rows.size > 0:
            out[c] = int(rows[0])
    return out


def bottom_depth_error(pred_binary, gt_binary):
    """Mean absolute bottom-row error in pixels over columns where both have
    a detected bottom. Returns nan if no overlapping columns."""
    pr = bottom_row_per_col(pred_binary)
    gt = bottom_row_per_col(gt_binary)
    mask = (pr >= 0) & (gt >= 0)
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs(pr[mask] - gt[mask])))


def channels_to_rgb_composite(patch_chw):
    def to01(c):
        return np.clip((c + 1.0) * 0.5, 0.0, 1.0)

    return np.stack([to01(patch_chw[0]), to01(patch_chw[1]), to01(patch_chw[2])], axis=-1)


def add_cb(im, ax, label=None):
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if label:
        cb.set_label(label, fontsize=14)
    cb.ax.tick_params(labelsize=13)
    return cb


# ---------------------------------------------------------------------------
# Sample loading
# ---------------------------------------------------------------------------


def load_test_patches(max_per_file=20, min_cov=0.05, max_cov=0.9, seed=0):
    """Stream and resize patches from the test cache, keeping a balanced set.

    NPZ caches are compressed, so any indexed access decompresses the whole
    array. We do one decompression per file, then pick indices from the
    in-memory arrays.

    Returns:
        patches : list of (3, H, W) float32 numpy arrays (resized to TARGET_SIZE)
        masks   : list of (H, W) uint8 numpy arrays
        meta    : list of dicts with 'file' and 'index'
    """
    rng = np.random.RandomState(seed)
    patches, masks, meta = [], [], []
    for fname in TEST_CACHE_FILES:
        path = CACHE_DIR / fname
        if not path.exists():
            continue
        print(f"  Loading {fname} ...", flush=True)
        with np.load(path) as d:
            all_p = d["patches"]  # forces decompression once
            all_m = d["masks"]
            n = all_p.shape[0]
            idxs = list(range(n))
            rng.shuffle(idxs)
            n_kept = 0
            for idx in idxs:
                if n_kept >= max_per_file:
                    break
                p = all_p[idx]
                m = all_m[idx]
                cov = m.mean()
                if cov < min_cov or cov > max_cov:
                    continue
                rp, rm = resize_patch(p, m)
                patches.append(rp)
                masks.append(rm.astype(np.uint8))
                meta.append({"file": fname.split("_")[0], "index": int(idx)})
                n_kept += 1
        # Release the large arrays before moving on
        del all_p, all_m
        gc.collect()
    return patches, masks, meta


# ---------------------------------------------------------------------------
# Analysis 1: channel ablation (single channel zero / mean / shuffled)
# ---------------------------------------------------------------------------


def ablate_channel(patch_chw, channel, mode, replacement=None):
    """Return a copy of patch_chw with `channel` replaced.

    mode = 'zero', 'mean' (per-image mean), or 'replace' (use replacement
    array of shape (H, W))."""
    out = patch_chw.copy()
    if mode == "zero":
        out[channel] = 0.0
    elif mode == "mean":
        out[channel] = float(out[channel].mean())
    elif mode == "replace":
        out[channel] = replacement
    else:
        raise ValueError(mode)
    return out


def run_ablation(model, patches, masks, n_shuffle_trials=3, batch_size=16, threshold=0.5, seed=0):
    """Compute IoU/Dice/bottom-error for baseline and per-channel ablations.

    Returns a dict with results per mode and per channel.
    """
    rng = np.random.RandomState(seed)

    def batched_probs(arr_list):
        out = []
        for i in range(0, len(arr_list), batch_size):
            chunk = np.stack(arr_list[i : i + batch_size])
            out.append(predict_probs_batch(model, chunk))
        return np.concatenate(out, axis=0)

    print(f"  Ablation: running baseline on {len(patches)} patches")
    base_probs = batched_probs(patches)
    base_metrics = []
    for prob, m in zip(base_probs, masks):
        pred = (prob > threshold).astype(np.uint8)
        iou, dice = iou_dice(pred, m)
        err = bottom_depth_error(pred, m)
        base_metrics.append((iou, dice, err))
    base_metrics = np.array(base_metrics)

    results = {
        "baseline": {
            "iou": base_metrics[:, 0],
            "dice": base_metrics[:, 1],
            "err": base_metrics[:, 2],
        },
    }

    modes = ["zero", "mean"]
    for mode in modes:
        results[mode] = {}
        for ch in range(3):
            print(f"  Ablation [{mode}] channel {ch} ({CHANNEL_SHORT[ch]})")
            ablated = [ablate_channel(p, ch, mode) for p in patches]
            probs = batched_probs(ablated)
            mets = []
            for prob, m in zip(probs, masks):
                pred = (prob > threshold).astype(np.uint8)
                iou, dice = iou_dice(pred, m)
                err = bottom_depth_error(pred, m)
                mets.append((iou, dice, err))
            mets = np.array(mets)
            results[mode][ch] = {
                "iou": mets[:, 0],
                "dice": mets[:, 1],
                "err": mets[:, 2],
            }

    # Shuffle baseline: replace channel c with the same channel from a
    # different randomly chosen patch.  Repeat trials to average out
    # the choice of replacement.
    results["shuffle"] = {}
    n = len(patches)
    for ch in range(3):
        print(f"  Ablation [shuffle x{n_shuffle_trials}] channel {ch} ({CHANNEL_SHORT[ch]})")
        iou_all, dice_all, err_all = [], [], []
        for trial in range(n_shuffle_trials):
            perm = rng.permutation(n)
            # Make sure no patch maps to itself
            while np.any(perm == np.arange(n)) and n > 1:
                perm = rng.permutation(n)
            ablated = []
            for i, p in enumerate(patches):
                repl = patches[perm[i]][ch]
                ablated.append(ablate_channel(p, ch, "replace", replacement=repl))
            probs = batched_probs(ablated)
            mets = []
            for prob, m in zip(probs, masks):
                pred = (prob > threshold).astype(np.uint8)
                iou, dice = iou_dice(pred, m)
                err = bottom_depth_error(pred, m)
                mets.append((iou, dice, err))
            mets = np.array(mets)
            iou_all.append(mets[:, 0])
            dice_all.append(mets[:, 1])
            err_all.append(mets[:, 2])
        results["shuffle"][ch] = {
            "iou": np.mean(iou_all, axis=0),
            "dice": np.mean(dice_all, axis=0),
            "err": np.nanmean(err_all, axis=0),
        }

    return results


# ---------------------------------------------------------------------------
# Analysis 2: pairwise channel ablation (drop two of three)
# ---------------------------------------------------------------------------


def run_pairwise_ablation(model, patches, masks, batch_size=16, threshold=0.5):
    """For each single channel kept (others zeroed), measure IoU/Dice."""

    def batched_probs(arr_list):
        out = []
        for i in range(0, len(arr_list), batch_size):
            chunk = np.stack(arr_list[i : i + batch_size])
            out.append(predict_probs_batch(model, chunk))
        return np.concatenate(out, axis=0)

    results = {}
    for keep in range(3):
        print(f"  Keep-only channel {keep} ({CHANNEL_SHORT[keep]})")
        ablated = []
        for p in patches:
            mod = p.copy()
            for c in range(3):
                if c != keep:
                    mod[c] = 0.0
            ablated.append(mod)
        probs = batched_probs(ablated)
        ious, dices = [], []
        for prob, m in zip(probs, masks):
            pred = (prob > threshold).astype(np.uint8)
            iou, dice = iou_dice(pred, m)
            ious.append(iou)
            dices.append(dice)
        results[keep] = {"iou": np.array(ious), "dice": np.array(dices)}

    print("  Keep none (all zeros)")
    ablated = [np.zeros_like(p) for p in patches]
    probs = batched_probs(ablated)
    ious, dices = [], []
    for prob, m in zip(probs, masks):
        pred = (prob > threshold).astype(np.uint8)
        iou, dice = iou_dice(pred, m)
        ious.append(iou)
        dices.append(dice)
    results["none"] = {"iou": np.array(ious), "dice": np.array(dices)}

    return results


# ---------------------------------------------------------------------------
# Analysis 3 & 4: input gradient and integrated gradients per channel
# ---------------------------------------------------------------------------


def input_gradient(model, patch_chw, target_mask=None):
    """Return d(score)/d(input) where score = sum over predicted-bottom pixels
    of sigmoid(logit). Shape (3, H, W) numpy array."""
    inp = torch.from_numpy(patch_chw[np.newaxis]).float().to(DEVICE)
    inp.requires_grad_(True)
    model.zero_grad(set_to_none=True)
    logits = model(inp)
    probs = torch.sigmoid(logits)
    if target_mask is None:
        with torch.no_grad():
            tmask = (probs.squeeze() > 0.5).float()
    else:
        tmask = torch.from_numpy(target_mask.astype(np.float32)).to(DEVICE)
    score = (probs.squeeze() * tmask).sum()
    score.backward()
    grad = inp.grad.detach().squeeze(0).cpu().numpy()
    inp.grad = None
    return grad


def integrated_gradients(model, patch_chw, steps=32, target_mask=None):
    """Riemann-sum approximation of integrated gradients with zero baseline.

    Returns (3, H, W) numpy array. The sum over all pixels and channels
    approximately equals the score difference predict(patch) - predict(0).
    """
    inp = torch.from_numpy(patch_chw[np.newaxis]).float().to(DEVICE)
    if target_mask is None:
        with torch.no_grad():
            tmask = (torch.sigmoid(model(inp)).squeeze() > 0.5).float()
    else:
        tmask = torch.from_numpy(target_mask.astype(np.float32)).to(DEVICE)

    accum = torch.zeros_like(inp)
    for k in range(1, steps + 1):
        alpha = k / steps
        interp = (alpha * inp).clone().detach().requires_grad_(True)
        model.zero_grad(set_to_none=True)
        logits = model(interp)
        probs = torch.sigmoid(logits)
        score = (probs.squeeze() * tmask).sum()
        score.backward()
        accum = accum + interp.grad.detach()
    accum = accum / steps
    ig = (inp.detach() - 0) * accum
    return ig.squeeze(0).cpu().numpy()


def aggregate_attribution(model, patches, masks, method="grad", steps=32, max_patches=None):
    """Run grad or IG over all patches and accumulate:
       - per-channel mean absolute attribution per patch (n_patches, 3)
       - per-channel mean spatial attribution map (3, H, W)
       - per-channel bottom-aligned profile

    Returns dict.
    """
    if max_patches is not None:
        patches = patches[:max_patches]
        masks = masks[:max_patches]

    n = len(patches)
    per_patch = np.zeros((n, 3), dtype=np.float64)
    accum_map = np.zeros((3,) + TARGET_SIZE, dtype=np.float64)
    aligned_profile = []  # rows centered on bottom edge, summed across cols

    window = 200
    cnt_aligned = 0

    for i, (p, m) in enumerate(zip(patches, masks)):
        if method == "grad":
            attr = input_gradient(model, p)
        elif method == "ig":
            attr = integrated_gradients(model, p, steps=steps)
        else:
            raise ValueError(method)
        abs_attr = np.abs(attr)

        per_patch[i] = abs_attr.reshape(3, -1).mean(axis=1)
        accum_map += abs_attr

        # Bottom alignment for profile: use the GT mask top edge
        cols = np.where(m.any(axis=0))[0]
        if cols.size == 0:
            continue
        top_rows = np.argmax(m > 0, axis=0)[cols]
        br = int(np.median(top_rows))
        H = m.shape[0]
        lo = br - window
        hi = br + window
        pad_lo = max(0, -lo)
        pad_hi = max(0, hi - H)
        lo = max(0, lo)
        hi = min(H, hi)
        prof = abs_attr.mean(axis=2)  # (3, H) mean over columns
        seg = prof[:, lo:hi]
        if pad_lo > 0:
            seg = np.concatenate([np.full((3, pad_lo), np.nan), seg], axis=1)
        if pad_hi > 0:
            seg = np.concatenate([seg, np.full((3, pad_hi), np.nan)], axis=1)
        aligned_profile.append(seg)
        cnt_aligned += 1

        if (i + 1) % 25 == 0:
            print(f"    {method} {i + 1}/{n}")

    accum_map /= n
    aligned_profile = np.array(aligned_profile)  # (n, 3, 2*window)
    return {
        "per_patch": per_patch,
        "mean_map": accum_map,
        "aligned": aligned_profile,
        "rel_y": np.arange(-window, window),
    }


# ---------------------------------------------------------------------------
# Analysis 5: GradCAM masked by per-channel gradient
# ---------------------------------------------------------------------------


def gradcam_target(model, target_layer, input_chw, target_mask):
    inp = torch.from_numpy(input_chw[np.newaxis]).float().to(DEVICE)
    targets = [SemanticSegmentationTarget(category=0, mask=target_mask.astype(np.float32))]
    with GradCAM(model=model, target_layers=[target_layer]) as cam:
        return cam(input_tensor=inp, targets=targets)[0]


# ---------------------------------------------------------------------------
# Analysis 6: per-channel input distributions for bottom vs water
# ---------------------------------------------------------------------------


def collect_channel_distributions(patches, masks, max_samples_per_class=200_000, seed=0):
    """Sample input values from bottom vs water pixels for each channel."""
    rng = np.random.RandomState(seed)
    bottom_vals = [[], [], []]
    water_vals = [[], [], []]
    per_class_quota = max_samples_per_class // max(1, len(patches))

    for p, m in zip(patches, masks):
        bot_mask = m > 0
        wat_mask = ~bot_mask
        for c in range(3):
            bv = p[c][bot_mask]
            wv = p[c][wat_mask]
            if bv.size > per_class_quota:
                bv = bv[rng.choice(bv.size, per_class_quota, replace=False)]
            if wv.size > per_class_quota:
                wv = wv[rng.choice(wv.size, per_class_quota, replace=False)]
            bottom_vals[c].append(bv)
            water_vals[c].append(wv)

    return ([np.concatenate(v) for v in bottom_vals], [np.concatenate(v) for v in water_vals])


# ---------------------------------------------------------------------------
# Figure generation
# ---------------------------------------------------------------------------


def fig_ablation_bars(ablation, out_path):
    """Mean and 95% CI of IoU/Dice/bottom-error under each ablation mode."""
    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    metrics = ["iou", "dice", "err"]
    titles = [
        "IoU (higher better)",
        "Dice (higher better)",
        "Bottom-row error in pixels (lower better)",
    ]
    modes = ["zero", "mean", "shuffle"]
    mode_labels = ["Zero out", "Replace with mean", "Replace from other patch"]

    for ax, metric, title in zip(axes, metrics, titles):
        base = ablation["baseline"][metric]
        if metric == "err":
            base_mean = np.nanmean(base)
        else:
            base_mean = np.mean(base)

        x = np.arange(3)
        width = 0.25
        for j, (mode, lbl) in enumerate(zip(modes, mode_labels)):
            means, lows, highs = [], [], []
            for ch in range(3):
                arr = ablation[mode][ch][metric]
                arr = arr[~np.isnan(arr)] if metric == "err" else arr
                m = np.mean(arr) if len(arr) else float("nan")
                lo = np.percentile(arr, 2.5) if len(arr) else float("nan")
                hi = np.percentile(arr, 97.5) if len(arr) else float("nan")
                means.append(m)
                lows.append(m - lo)
                highs.append(hi - m)
            ax.bar(
                x + (j - 1) * width,
                means,
                width=width,
                label=lbl,
                yerr=[lows, highs],
                capsize=4,
                color=["#8da0cb", "#fc8d62", "#66c2a5"][j],
                alpha=0.9,
            )
        ax.axhline(
            base_mean,
            color="black",
            linestyle="--",
            linewidth=2,
            label=f"Baseline ({base_mean:.3f})",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(CHANNEL_NAMES)
        ax.set_ylabel(title)
        ax.set_title(f"Channel ablation: {metric.upper()}")
        ax.legend(loc="lower right" if metric != "err" else "upper left", fontsize=12)
        ax.grid(axis="y", alpha=0.3)
        if metric == "err":
            ax.set_yscale("log")

    fig.suptitle(
        "Per-channel ablation: how IoU, Dice, and bottom error "
        "change when a single channel is removed",
        y=1.03,
    )
    fig.text(
        0.01,
        0.005,
        "Source: code/data_visualization/analyze_channel_importance.py",
        fontsize=10,
        color="gray",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_drop_distribution(ablation, out_path):
    """Per-patch IoU drop distributions under shuffle ablation."""
    fig, axes = plt.subplots(1, 3, figsize=(22, 6))
    base = ablation["baseline"]["iou"]
    for ch, ax in enumerate(axes):
        drops = base - ablation["shuffle"][ch]["iou"]
        ax.hist(drops, bins=30, color=CHANNEL_COLORS[ch], alpha=0.75, edgecolor="black")
        ax.axvline(
            np.median(drops),
            color="black",
            linestyle="--",
            linewidth=2,
            label=f"Median drop {np.median(drops):+.3f}",
        )
        ax.axvline(0, color="gray", linestyle=":", linewidth=1.5)
        ax.set_xlabel("IoU drop (baseline minus ablated)")
        ax.set_ylabel("Patch count")
        ax.set_title(f"Drop in IoU when {CHANNEL_NAMES[ch]} is shuffled")
        ax.legend(loc="upper right", fontsize=13)
        ax.grid(alpha=0.3)
    fig.suptitle("Distribution of per-patch IoU drop under channel shuffling", y=1.02)
    fig.text(
        0.01,
        0.005,
        "Source: code/data_visualization/analyze_channel_importance.py",
        fontsize=10,
        color="gray",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_pairwise_bars(pairwise, ablation, out_path):
    """Keep-only-one-channel and keep-none performance."""
    fig, ax = plt.subplots(1, 1, figsize=(13, 7))
    base_iou = float(np.mean(ablation["baseline"]["iou"]))
    base_dice = float(np.mean(ablation["baseline"]["dice"]))

    labels = ["Baseline\n(all 3)"] + [f"Only\n{CHANNEL_SHORT[c]}" for c in range(3)] + ["All zeros"]
    iou_means = (
        [base_iou]
        + [float(np.mean(pairwise[c]["iou"])) for c in range(3)]
        + [float(np.mean(pairwise["none"]["iou"]))]
    )
    dice_means = (
        [base_dice]
        + [float(np.mean(pairwise[c]["dice"])) for c in range(3)]
        + [float(np.mean(pairwise["none"]["dice"]))]
    )

    x = np.arange(len(labels))
    width = 0.36
    ax.bar(x - width / 2, iou_means, width, label="IoU", color="#1f77b4", alpha=0.85)
    ax.bar(x + width / 2, dice_means, width, label="Dice", color="#ff7f0e", alpha=0.85)
    for i, (iv, dv) in enumerate(zip(iou_means, dice_means)):
        ax.text(i - width / 2, iv + 0.01, f"{iv:.3f}", ha="center", fontsize=12)
        ax.text(i + width / 2, dv + 0.01, f"{dv:.3f}", ha="center", fontsize=12)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Metric")
    ax.set_title("Performance when only a single channel is retained")
    ax.legend(loc="upper right")
    ax.set_ylim(0, max(iou_means + dice_means) * 1.15)
    ax.grid(axis="y", alpha=0.3)
    fig.text(
        0.01,
        0.005,
        "Source: code/data_visualization/analyze_channel_importance.py",
        fontsize=10,
        color="gray",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_attribution_bars(grad_agg, ig_agg, out_path):
    """Per-channel mean |gradient| and mean |IG| with patch-level CIs."""
    fig, axes = plt.subplots(1, 2, figsize=(20, 7))

    for ax, agg, title in zip(
        axes,
        [grad_agg, ig_agg],
        [
            "Mean absolute input gradient per channel",
            "Mean absolute integrated gradient per channel",
        ],
    ):
        means = agg["per_patch"].mean(axis=0)
        lo = np.percentile(agg["per_patch"], 2.5, axis=0)
        hi = np.percentile(agg["per_patch"], 97.5, axis=0)
        share = means / means.sum() * 100
        x = np.arange(3)
        ax.bar(
            x,
            means,
            color=CHANNEL_COLORS,
            alpha=0.85,
            yerr=[means - lo, hi - means],
            capsize=5,
            edgecolor="black",
        )
        for i, (m, s) in enumerate(zip(means, share)):
            ax.text(i, m + 0.02 * means.max(), f"{m:.3e}\n({s:.1f}%)", ha="center", fontsize=13)
        ax.set_xticks(x)
        ax.set_xticklabels(CHANNEL_NAMES)
        ax.set_ylabel("Mean |attribution|")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Per-channel attribution magnitudes across test patches", y=1.02)
    fig.text(
        0.01,
        0.005,
        "Source: code/data_visualization/analyze_channel_importance.py",
        fontsize=10,
        color="gray",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_mean_attribution_maps(agg, title_prefix, out_path):
    """Per-channel mean spatial attribution map across the dataset."""
    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    vmax = float(agg["mean_map"].max())
    for c, ax in enumerate(axes):
        im = ax.imshow(agg["mean_map"][c], cmap="hot", aspect="auto", vmin=0, vmax=vmax)
        share = agg["mean_map"][c].sum() / agg["mean_map"].sum() * 100
        ax.set_title(f"{CHANNEL_NAMES[c]} ({share:.1f}% of total)")
        ax.set_xticks([])
        ax.set_yticks([])
        add_cb(im, ax, "mean |attribution|")
    fig.suptitle(f"{title_prefix}: mean per-channel spatial attribution", y=1.03)
    fig.text(
        0.01,
        0.005,
        "Source: code/data_visualization/analyze_channel_importance.py",
        fontsize=10,
        color="gray",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_bottom_aligned_profiles(grad_agg, ig_agg, out_path):
    """Bottom-aligned vertical attribution profiles per channel."""
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    for ax, agg, title in zip(
        axes,
        [grad_agg, ig_agg],
        [
            "Input gradient profile aligned to bottom edge",
            "Integrated gradient profile aligned to bottom edge",
        ],
    ):
        rel_y = agg["rel_y"]
        for c in range(3):
            mean = np.nanmean(agg["aligned"][:, c, :], axis=0)
            std = np.nanstd(agg["aligned"][:, c, :], axis=0)
            ax.plot(mean, rel_y, color=CHANNEL_COLORS[c], linewidth=2.5, label=CHANNEL_NAMES[c])
            ax.fill_betweenx(rel_y, mean - std, mean + std, color=CHANNEL_COLORS[c], alpha=0.15)
        ax.axhline(0, color="red", linestyle="--", linewidth=2, label="Bottom edge")
        ax.invert_yaxis()
        ax.set_xlabel("Mean |attribution| over columns")
        ax.set_ylabel("Rows relative to bottom edge")
        ax.set_title(title)
        ax.legend(loc="lower right", fontsize=13)
        ax.grid(alpha=0.3)
    fig.suptitle("Where each channel matters relative to the seabed line", y=1.02)
    fig.text(
        0.01,
        0.005,
        "Source: code/data_visualization/analyze_channel_importance.py",
        fontsize=10,
        color="gray",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_per_patch_grad_examples(model, patches, masks, meta, out_path, n_examples=3):
    """For a few patches, show input channels alongside per-channel input
    gradient and integrated gradient maps."""
    chosen = []
    for i, m in enumerate(masks):
        cov = m.mean()
        if 0.1 < cov < 0.8:
            chosen.append(i)
        if len(chosen) >= n_examples:
            break
    chosen = chosen[:n_examples]
    fig, axes = plt.subplots(len(chosen), 7, figsize=(28, 5 * len(chosen)))
    if len(chosen) == 1:
        axes = axes[np.newaxis]
    for r, idx in enumerate(chosen):
        p = patches[idx]
        m = masks[idx]
        composite = channels_to_rgb_composite(p)
        probs = predict_probs(model, p)

        axes[r, 0].imshow(composite, aspect="auto")
        axes[r, 0].set_title(f"Input: {meta[idx]['file']}#{meta[idx]['index']}")
        axes[r, 0].set_xticks([])
        axes[r, 0].set_yticks([])

        im = axes[r, 1].imshow(probs, cmap="magma", aspect="auto", vmin=0, vmax=1)
        axes[r, 1].set_title("Predicted p(bottom)")
        axes[r, 1].set_xticks([])
        axes[r, 1].set_yticks([])
        add_cb(im, axes[r, 1])

        g = input_gradient(model, p)
        ig = integrated_gradients(model, p, steps=24)

        g_abs = np.abs(g)
        ig_abs = np.abs(ig)
        g_max = max(g_abs.max(), 1e-12)
        for c in range(3):
            im = axes[r, 2 + c].imshow(g_abs[c], cmap="hot", aspect="auto", vmin=0, vmax=g_max)
            pct = g_abs[c].sum() / g_abs.sum() * 100
            axes[r, 2 + c].set_title(f"|grad| {CHANNEL_SHORT[c]} ({pct:.1f}%)")
            axes[r, 2 + c].set_xticks([])
            axes[r, 2 + c].set_yticks([])
            add_cb(im, axes[r, 2 + c])

        # Show first IG channel composite for compactness
        ig_pcts = ig_abs.reshape(3, -1).sum(axis=1)
        ig_share = ig_pcts / max(ig_pcts.sum(), 1e-12) * 100
        ig_comp = ig_abs.sum(axis=0)
        im = axes[r, 5].imshow(ig_comp, cmap="hot", aspect="auto", vmin=0, vmax=ig_comp.max())
        axes[r, 5].set_title("|IG| summed channels")
        axes[r, 5].set_xticks([])
        axes[r, 5].set_yticks([])
        add_cb(im, axes[r, 5])

        # Bar chart of IG share per channel
        axes[r, 6].bar(np.arange(3), ig_share, color=CHANNEL_COLORS, alpha=0.85, edgecolor="black")
        axes[r, 6].set_xticks(np.arange(3))
        axes[r, 6].set_xticklabels(CHANNEL_SHORT, fontsize=14)
        axes[r, 6].set_ylabel("IG share (%)")
        axes[r, 6].set_ylim(0, 100)
        axes[r, 6].set_title("IG attribution per channel")
        axes[r, 6].grid(axis="y", alpha=0.3)
        for i, s in enumerate(ig_share):
            axes[r, 6].text(i, s + 2, f"{s:.0f}%", ha="center", fontsize=14)

    fig.suptitle(
        "Per-patch input attribution: gradient and integrated gradient per channel", y=1.005
    )
    fig.text(
        0.01,
        0.005,
        "Source: code/data_visualization/analyze_channel_importance.py",
        fontsize=10,
        color="gray",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_distribution_per_channel(bot_vals, wat_vals, out_path):
    """Histograms of input values for bottom vs water pixels."""
    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    for c, ax in enumerate(axes):
        bins = np.linspace(-1, 1, 60)
        ax.hist(wat_vals[c], bins=bins, alpha=0.55, density=True, color="steelblue", label="Water")
        ax.hist(bot_vals[c], bins=bins, alpha=0.55, density=True, color="firebrick", label="Bottom")
        # Bhattacharyya overlap as a quick separability indicator
        hb, _ = np.histogram(bot_vals[c], bins=bins, density=True)
        hw, _ = np.histogram(wat_vals[c], bins=bins, density=True)
        hb /= hb.sum() + 1e-12
        hw /= hw.sum() + 1e-12
        bhatt = np.sum(np.sqrt(hb * hw))
        sep = -np.log(bhatt + 1e-12)
        ax.set_title(f"{CHANNEL_NAMES[c]}\nseparability (-log Bhatt) = {sep:.2f}")
        ax.set_xlabel("Normalised value (channel scale)")
        ax.set_ylabel("Density")
        ax.legend(loc="upper left")
        ax.grid(alpha=0.3)
    fig.suptitle("Per-channel input distributions: bottom vs water pixels", y=1.03)
    fig.text(
        0.01,
        0.005,
        "Source: code/data_visualization/analyze_channel_importance.py",
        fontsize=10,
        color="gray",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_gradcam_per_channel(model, patches, masks, meta, out_path, n_examples=2):
    """Show Grad-CAM and how it overlaps with each channel's |gradient| map."""
    layer = model.decoder.blocks[-1]
    chosen = [i for i, m in enumerate(masks) if 0.15 < m.mean() < 0.7][:n_examples]
    fig, axes = plt.subplots(len(chosen), 5, figsize=(24, 5 * len(chosen)))
    if len(chosen) == 1:
        axes = axes[np.newaxis]

    for r, idx in enumerate(chosen):
        p = patches[idx]
        composite = channels_to_rgb_composite(p)
        probs = predict_probs(model, p)
        pred = (probs > 0.5).astype(np.uint8)

        axes[r, 0].imshow(composite, aspect="auto")
        axes[r, 0].set_title(f"Input: {meta[idx]['file']}#{meta[idx]['index']}")
        axes[r, 0].set_xticks([])
        axes[r, 0].set_yticks([])

        cam = gradcam_target(model, layer, p, pred)
        im = axes[r, 1].imshow(cam, cmap="jet", aspect="auto", vmin=0, vmax=1)
        axes[r, 1].set_title("Grad-CAM (decoder block -1)")
        axes[r, 1].set_xticks([])
        axes[r, 1].set_yticks([])
        add_cb(im, axes[r, 1])

        g = np.abs(input_gradient(model, p))
        for c in range(3):
            # Spatial overlap of |grad_c| with GradCAM, expressed as cosine sim
            v1 = g[c].ravel()
            v2 = cam.ravel()
            cos = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12))
            blended = 0.55 * plt.get_cmap("jet")(cam)[..., :3] + 0.45 * composite
            blended = np.clip(blended, 0, 1)
            im = axes[r, 2 + c].imshow(
                g[c] / max(g[c].max(), 1e-12), cmap="hot", aspect="auto", vmin=0, vmax=1
            )
            axes[r, 2 + c].set_title(f"|grad| {CHANNEL_SHORT[c]} (cos with CAM = {cos:.2f})")
            axes[r, 2 + c].set_xticks([])
            axes[r, 2 + c].set_yticks([])

    fig.suptitle("Grad-CAM at the decoder vs per-channel input gradient magnitude", y=1.005)
    fig.text(
        0.01,
        0.005,
        "Source: code/data_visualization/analyze_channel_importance.py",
        fontsize=10,
        color="gray",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_summary_dashboard(ablation, pairwise, grad_agg, ig_agg, out_path):
    """One-page summary: normalised importance from each method."""
    fig, axes = plt.subplots(1, 2, figsize=(22, 7))

    base_iou = float(np.mean(ablation["baseline"]["iou"]))

    # Drop in IoU from shuffle ablation
    shuffle_drop = np.array(
        [base_iou - float(np.mean(ablation["shuffle"][c]["iou"])) for c in range(3)]
    )
    zero_drop = np.array([base_iou - float(np.mean(ablation["zero"][c]["iou"])) for c in range(3)])
    only_iou = np.array([float(np.mean(pairwise[c]["iou"])) for c in range(3)])
    grad_score = grad_agg["per_patch"].mean(axis=0)
    ig_score = ig_agg["per_patch"].mean(axis=0)

    def norm(x):
        x = np.array(x)
        s = x.sum()
        return x / s if s > 0 else x

    methods = {
        "IoU drop (zero)": norm(zero_drop),
        "IoU drop (shuffle)": norm(shuffle_drop),
        "Only-channel IoU": norm(only_iou),
        "|grad|": norm(grad_score),
        "|IG|": norm(ig_score),
    }
    method_names = list(methods.keys())
    n_methods = len(methods)
    x = np.arange(3)
    width = 0.16

    ax = axes[0]
    for j, (name, vals) in enumerate(methods.items()):
        ax.bar(x + (j - (n_methods - 1) / 2) * width, vals, width, label=name, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(CHANNEL_NAMES)
    ax.set_ylabel("Normalised importance share")
    ax.set_title("Channel importance share by method (each method sums to 1)")
    ax.legend(loc="upper right", fontsize=12)
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    rows = np.array([methods[k] for k in method_names])
    im = ax.imshow(rows, cmap="YlGnBu", aspect="auto", vmin=0, vmax=rows.max())
    ax.set_yticks(np.arange(n_methods))
    ax.set_yticklabels(method_names)
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(CHANNEL_NAMES)
    for i in range(n_methods):
        for j in range(3):
            ax.text(j, i, f"{rows[i, j]:.2f}", ha="center", va="center", fontsize=14, color="black")
    ax.set_title("Importance share heatmap")
    add_cb(im, ax, "normalised share")

    fig.suptitle("Cross-method summary of per-channel importance", y=1.03)
    fig.text(
        0.01,
        0.005,
        "Source: code/data_visualization/analyze_channel_importance.py",
        fontsize=10,
        color="gray",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=None, help="Override the output directory"
    )
    parser.add_argument(
        "--max-per-file", type=int, default=12, help="Patches sampled per test file"
    )
    parser.add_argument(
        "--max-attribution-patches",
        type=int,
        default=60,
        help="Cap for gradient/IG aggregation (slowest step)",
    )
    parser.add_argument("--ig-steps", type=int, default=24)
    parser.add_argument("--shuffle-trials", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    global OUTPUT_DIR
    if args.output_dir is not None:
        OUTPUT_DIR = args.output_dir
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Device: {DEVICE}")
    print(f"Checkpoint: {CHECKPOINT_PATH.name}")
    print(f"Output: {OUTPUT_DIR}")

    model = load_model()

    print(f"\n[1/9] Loading test patches (max {args.max_per_file} per file)...")
    patches, masks, meta = load_test_patches(max_per_file=args.max_per_file, seed=args.seed)
    print(f"  Total kept patches: {len(patches)}")

    print("\n[2/9] Channel ablation (zero, mean, shuffle)...")
    ablation = run_ablation(
        model,
        patches,
        masks,
        n_shuffle_trials=args.shuffle_trials,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    _cleanup()

    print("\n[3/9] Pairwise ablation (keep only one channel)...")
    pairwise = run_pairwise_ablation(model, patches, masks, batch_size=args.batch_size)
    _cleanup()

    print(f"\n[4/9] Input gradient aggregation (capped at {args.max_attribution_patches})...")
    grad_agg = aggregate_attribution(
        model, patches, masks, method="grad", max_patches=args.max_attribution_patches
    )
    _cleanup()

    print(
        f"\n[5/9] Integrated gradient aggregation "
        f"({args.ig_steps} steps, capped at "
        f"{args.max_attribution_patches})..."
    )
    ig_agg = aggregate_attribution(
        model,
        patches,
        masks,
        method="ig",
        steps=args.ig_steps,
        max_patches=args.max_attribution_patches,
    )
    _cleanup()

    print("\n[6/9] Channel distribution stats...")
    bot_vals, wat_vals = collect_channel_distributions(patches, masks)
    _cleanup()

    print("\n[7/9] Generating figures...")

    fig_ablation_bars(ablation, OUTPUT_DIR / "01_ablation_iou_dice_err.png")
    fig_drop_distribution(ablation, OUTPUT_DIR / "02_ablation_iou_drop_distribution.png")
    fig_pairwise_bars(pairwise, ablation, OUTPUT_DIR / "03_pairwise_keep_only_one_channel.png")
    fig_attribution_bars(grad_agg, ig_agg, OUTPUT_DIR / "04_attribution_bars.png")
    fig_mean_attribution_maps(grad_agg, "Input gradient", OUTPUT_DIR / "05_mean_gradient_maps.png")
    fig_mean_attribution_maps(ig_agg, "Integrated gradient", OUTPUT_DIR / "06_mean_ig_maps.png")
    fig_bottom_aligned_profiles(grad_agg, ig_agg, OUTPUT_DIR / "07_bottom_aligned_profiles.png")
    fig_per_patch_grad_examples(
        model, patches, masks, meta, OUTPUT_DIR / "08_per_patch_grad_examples.png"
    )
    fig_distribution_per_channel(bot_vals, wat_vals, OUTPUT_DIR / "09_channel_distributions.png")
    fig_gradcam_per_channel(
        model, patches, masks, meta, OUTPUT_DIR / "10_gradcam_vs_per_channel_grad.png"
    )
    fig_summary_dashboard(
        ablation, pairwise, grad_agg, ig_agg, OUTPUT_DIR / "11_summary_dashboard.png"
    )

    print("\n[8/9] Writing numeric summary CSVs and JSON...")
    base_iou = float(np.mean(ablation["baseline"]["iou"]))
    base_dice = float(np.mean(ablation["baseline"]["dice"]))

    summary = {
        "n_patches": len(patches),
        "baseline_iou_mean": base_iou,
        "baseline_dice_mean": base_dice,
        "baseline_bottom_err_mean_px": float(np.nanmean(ablation["baseline"]["err"])),
        "channels": CHANNEL_NAMES,
        "ablation": {
            mode: {
                CHANNEL_SHORT[c]: {
                    "iou_mean": float(np.mean(ablation[mode][c]["iou"])),
                    "dice_mean": float(np.mean(ablation[mode][c]["dice"])),
                    "err_mean_px": float(np.nanmean(ablation[mode][c]["err"])),
                    "iou_drop": base_iou - float(np.mean(ablation[mode][c]["iou"])),
                }
                for c in range(3)
            }
            for mode in ["zero", "mean", "shuffle"]
        },
        "pairwise_keep_only": {
            CHANNEL_SHORT[c]: {
                "iou_mean": float(np.mean(pairwise[c]["iou"])),
                "dice_mean": float(np.mean(pairwise[c]["dice"])),
            }
            for c in range(3)
        },
        "pairwise_keep_none": {
            "iou_mean": float(np.mean(pairwise["none"]["iou"])),
            "dice_mean": float(np.mean(pairwise["none"]["dice"])),
        },
        "attribution": {
            "mean_abs_grad": {
                CHANNEL_SHORT[c]: float(grad_agg["per_patch"].mean(axis=0)[c]) for c in range(3)
            },
            "mean_abs_ig": {
                CHANNEL_SHORT[c]: float(ig_agg["per_patch"].mean(axis=0)[c]) for c in range(3)
            },
        },
    }
    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n[9/9] Done.")
    print(f"All figures written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
