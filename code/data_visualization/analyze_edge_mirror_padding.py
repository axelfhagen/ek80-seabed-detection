"""Quick test: does reflection-padding the patch edges reduce edge-column error?

The edge-degradation analysis shows per-column bottom-line error rising toward
the left/right patch boundaries. Hypothesis: that rise is a missing-context /
zero-padding artefact of the convolutions at the image border. If we instead
mirror the image around the edge pixel (reflection padding) before inference and
crop the centre back out, the border columns get genuine (mirrored) context, so
the edge error should shrink.

This is a fast, local, directional check on the handful of multi-frequency
patches stored under notebooks/tensors/ using the generic resnet34 checkpoint.
Not the full val/test sweep -- just enough to see whether the effect moves.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import cv2
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from deep_learning.unet_model import create_unet

CKPT = PROJECT_ROOT / "code" / "checkpoints" / "unet_resnet34_best.pth"
TENSOR_DIR = PROJECT_ROOT / "code" / "raw_sensor_integration" / "notebooks" / "tensors"
OUT = PROJECT_ROOT / "Figures" / "edge_degradation"

H = W = 512  # model input size (matches training target_size)
F1_IDX, F2_IDX = 0, 4  # 38.5 kHz, 210 kHz in the 5-freq stack
PADS = [0, 32, 64, 128]
THRESHOLD = 0.5


def load_patches():
    """Concatenate the unique multi-frequency patches available locally."""
    imgs, msks = [], []
    seen = set()
    for name in [
        "bottom_patches_train.npz",
        "bottom_patches_val.npz",
        "bottom_patches_multifreq.npz",
        "bottom_patches.npz",
    ]:
        p = TENSOR_DIR / name
        if not p.exists():
            continue
        d = np.load(p)
        for img5, lab in zip(d["images"], d["labels"]):
            key = (round(float(img5.mean()), 6), round(float(lab.mean()), 6), img5.shape)
            if key in seen:
                continue
            seen.add(key)
            imgs.append(img5)
            msks.append(lab)
    return imgs, msks


def to_input(img5):
    """5-freq native patch -> (3, H, W) resized model input (freq1, freq2, diff)."""
    f1 = img5[F1_IDX].astype(np.float32)
    f2 = img5[F2_IDX].astype(np.float32)
    ch = np.stack([f1, f2, f1 - f2], axis=0)  # (3, n_range, n_ping)
    chw = np.transpose(ch, (1, 2, 0))  # (n_range, n_ping, 3)
    res = cv2.resize(chw, (W, H), interpolation=cv2.INTER_LINEAR)
    return np.transpose(res, (2, 0, 1))  # (3, H, W)


def to_mask(lab):
    return cv2.resize(lab.astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST) >= 0.5


def bottom_rows(binary_hw):
    """First seabed row per column; -1 if column has none."""
    rows = np.full(binary_hw.shape[1], -1, dtype=np.float64)
    for c in range(binary_hw.shape[1]):
        hit = np.where(binary_hw[:, c])[0]
        if len(hit):
            rows[c] = hit[0]
    return rows


def bottom_rows_transition(prob_hw):
    """Bottom line = dominant downward water->seabed transition per column.

    Robust to scattered false-positive seabed pixels high in the water column
    (which break a naive 'first pixel above threshold' decode): the seabed
    interface is the row with the largest positive low->high jump in a lightly
    smoothed column profile. -1 if the column never crosses threshold.
    """
    H, Wc = prob_hw.shape
    k = np.ones(5) / 5.0
    rows = np.full(Wc, -1.0)
    for c in range(Wc):
        col = np.convolve(prob_hw[:, c], k, mode="same")
        if col.max() < THRESHOLD:
            continue
        rows[c] = int(np.argmax(np.diff(col))) + 1
    return rows


@torch.no_grad()
def infer(model, inp_chw, pad, device):
    """Reflect-pad along the ping (width) axis by `pad`, run, crop centre back."""
    if pad > 0:
        inp_chw = np.pad(inp_chw, ((0, 0), (0, 0), (pad, pad)), mode="reflect")
    x = torch.from_numpy(inp_chw[None]).float().to(device)
    prob = torch.sigmoid(model(x))[0, 0].cpu().numpy()
    if pad > 0:
        prob = prob[:, pad : pad + W]
    return prob


def main():
    device = (
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    model = create_unet("resnet34", encoder_weights=None, in_channels=3)
    sd = torch.load(CKPT, map_location=device)
    if isinstance(sd, dict):
        sd = sd.get("model_state_dict", sd)
    model.load_state_dict(sd)
    model.to(device).eval()

    imgs, msks = load_patches()
    print(f"device={device}  n_patches={len(imgs)}")

    # Per-column |pred_row - gt_row| accumulators, one curve per pad setting.
    err_sum = {p: np.zeros(W) for p in PADS}
    err_cnt = {p: np.zeros(W) for p in PADS}

    for img5, lab in zip(imgs, msks):
        inp = to_input(img5)
        gt_rows = bottom_rows(to_mask(lab))
        for p in PADS:
            prob = infer(model, inp, p, device)
            pr_rows = bottom_rows_transition(prob)
            valid = (gt_rows >= 0) & (pr_rows >= 0)
            err_sum[p][valid] += np.abs(pr_rows[valid] - gt_rows[valid])
            err_cnt[p][valid] += 1

    mean_err = {
        p: np.divide(err_sum[p], err_cnt[p], out=np.full(W, np.nan), where=err_cnt[p] > 0)
        for p in PADS
    }

    edge = slice(0, int(0.05 * W))  # leftmost 5%
    redge = slice(int(0.95 * W), W)  # rightmost 5%
    interior = slice(int(0.25 * W), int(0.75 * W))
    print(f"\n{'pad':>5} {'interior':>10} {'left5%':>10} {'right5%':>10} {'edge-mean':>10}")
    for p in PADS:
        m = mean_err[p]
        i = np.nanmean(m[interior])
        le = np.nanmean(m[edge])
        re = np.nanmean(m[redge])
        em = np.nanmean(np.concatenate([m[edge], m[redge]]))
        tag = "  <- baseline" if p == 0 else ""
        print(f"{p:>5} {i:>10.3f} {le:>10.3f} {re:>10.3f} {em:>10.3f}{tag}")

    # Figure
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 5))
    for p in PADS:
        lbl = "baseline (no pad)" if p == 0 else f"reflect-pad {p}px"
        ax.plot(np.arange(W), mean_err[p], lw=2, label=lbl)
    ax.set_xlabel("Ping column within patch")
    ax.set_ylabel("Mean bottom error (rows)")
    ax.set_title(f"Reflection-padding vs edge error (n={len(imgs)} patches)")
    ax.legend()
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "mirror_pad_test.pdf")
    fig.savefig(OUT / "mirror_pad_test.png", dpi=130)
    print(f"\nWrote {OUT / 'mirror_pad_test.pdf'}")


if __name__ == "__main__":
    main()
