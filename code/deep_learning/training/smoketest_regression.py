"""Tiny CPU smoketest for the bottom-line regression path.

Exercises RegressionUNet + RegressionLoss + compute_metrics(task='regression')
+ predict_full_echogram(task='regression') on synthetic data, checking shapes,
finiteness, a backward pass, and that the loss/MAE drop when the predicted line
is moved toward the truth. No zarr / cache / GPU needed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

_root = str(Path(__file__).resolve().parents[2])
if _root not in sys.path:
    sys.path.insert(0, _root)

from deep_learning.architectures import create_model, list_architectures
from deep_learning.unet_model import (
    build_loss,
    compute_metrics,
    soft_argmax_depth,
    target_depth_from_mask,
    predict_full_echogram,
)


def _below_mask(depth_rows, H, W):
    """Synthetic 'below' mask (H, W): 1 at/below the given per-column bottom row."""
    rows = np.arange(H)[:, None]
    return (rows >= depth_rows[None, :]).astype(np.float32)


def main():
    torch.manual_seed(0)
    np.random.seed(0)
    H = W = 64
    B = 2

    assert "unet_regression" in list_architectures(), list_architectures()
    model = create_model(
        "unet_regression", encoder_name="resnet34", encoder_weights=None, in_channels=3
    )
    print("architecture_id:", model.architecture_id)

    x = torch.randn(B, 3, H, W)
    true_depth = np.array([H // 2 + int(5 * np.sin(j / 4)) for j in range(W)], dtype=np.float32)
    masks = torch.from_numpy(np.stack([_below_mask(true_depth, H, W)] * B))

    # Target depth read back from the 'below' mask must equal the true line.
    d_true = target_depth_from_mask(masks, mask_mode="below")
    assert torch.allclose(d_true[0], torch.from_numpy(true_depth)), (d_true[0][:5], true_depth[:5])
    print("target_depth_from_mask OK")

    logits = model(x)
    assert logits.shape == (B, 1, H, W), logits.shape
    d_pred = soft_argmax_depth(logits)
    assert d_pred.shape == (B, W), d_pred.shape
    assert torch.isfinite(d_pred).all()
    print("forward + soft_argmax OK, shapes", tuple(logits.shape), tuple(d_pred.shape))

    loss_fn = build_loss(
        "regression",
        geo_depth_scale=1.0,
        geo_huber_delta=5.0,
        geo_smooth_weight=0.1,
        mask_mode="below",
    )
    loss = loss_fn(logits, masks)
    assert torch.isfinite(loss), loss
    loss.backward()
    n_grad = sum(p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None)
    assert n_grad > 0, "no gradients flowed"
    print(f"loss={loss.item():.4f}  backward OK  (grad mass={n_grad:.2f})")

    m = compute_metrics(logits.detach(), masks, mask_mode="below", task="regression")
    assert set(m) == {"iou", "dice", "precision", "recall", "bottom_mae"}, m
    assert all(np.isfinite(v) for v in m.values()), m
    print("compute_metrics OK:", {k: round(v, 3) for k, v in m.items()})

    # Sanity: loss/MAE for logits whose argmax sits ON the true line should beat
    # logits whose argmax sits far above it. Build peaked heatmaps directly.
    def peaked_logits(depth):
        rows = torch.arange(H).view(1, H, 1).float()
        d = torch.from_numpy(depth).view(1, 1, W).float()
        return (-((rows - d) ** 2) / 4.0).unsqueeze(1)  # (1,1,H,W), peak at depth

    good = peaked_logits(true_depth)
    bad = peaked_logits(np.full(W, 5.0, dtype=np.float32))
    mae_good = compute_metrics(good, masks[:1], mask_mode="below", task="regression")["bottom_mae"]
    mae_bad = compute_metrics(bad, masks[:1], mask_mode="below", task="regression")["bottom_mae"]
    assert mae_good < mae_bad, (mae_good, mae_bad)
    print(f"monotonicity OK: MAE on-line={mae_good:.2f} < off-line={mae_bad:.2f}")

    # Full-echogram regression inference -> 'below' probability map in [0,1].
    sv = np.random.randn(3, 128, 100).astype(np.float32)
    pred = predict_full_echogram(
        model, sv, target_size=(H, W), device="cpu", patch_width=64, stride=32, task="regression"
    )
    assert pred.shape == (128, 100), pred.shape
    assert np.isfinite(pred).all()
    assert pred.min() >= 0.0 and pred.max() <= 1.0, (pred.min(), pred.max())
    print("predict_full_echogram OK, shape", pred.shape)

    print("\nALL REGRESSION SMOKETESTS PASSED")


if __name__ == "__main__":
    main()
