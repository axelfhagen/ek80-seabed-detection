"""Smoketest for the geo_seg loss: real 512-wide patches, real model,
forward + backward in fp32 and AMP. Checks the loss runs and produces
finite gradients before committing to a Slurm run. Does not extract patches.
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from deep_learning.unet_model import build_loss, GeoSegLoss
from deep_learning.training.model_setup import build_model
from deep_learning.training.config import PipelineConfig

CACHE = (
    Path(__file__).resolve().parents[2].parent
    / "Datasets/patch_cache/D20231220-T212138_7dddd39f3f4b.npz"
)


def load_batch(n=4, target_size=512):
    d = np.load(CACHE)
    patches, masks = d["patches"], d["masks"]
    assert patches.shape[-1] == 512, f"expected 512-wide, got {patches.shape}"
    idx = np.arange(min(n, patches.shape[0]))
    x = torch.from_numpy(patches[idx]).float()  # (B,3,H,512)
    y = torch.from_numpy(masks[idx].astype(np.float32))  # (B,H,512)
    # resize to square target like the real PatchDataset does
    x = torch.nn.functional.interpolate(
        x, size=(target_size, target_size), mode="bilinear", align_corners=False
    )
    y = torch.nn.functional.interpolate(
        y.unsqueeze(1), size=(target_size, target_size), mode="nearest"
    ).squeeze(1)
    return x, y


def check(name, logits, targets, loss_fn, amp=False, device="cpu"):
    logits = logits.clone().detach().requires_grad_(True)
    if amp:
        with torch.autocast(device_type=device, dtype=torch.float16):
            loss = loss_fn(logits, targets)
    else:
        loss = loss_fn(logits, targets)
    loss.backward()
    g = logits.grad
    ok = torch.isfinite(loss).item() and torch.isfinite(g).all().item()
    print(
        f"  [{name}] loss={loss.item():.5f} grad_finite={torch.isfinite(g).all().item()} "
        f"grad_norm={g.norm().item():.4f} {'OK' if ok else 'FAIL'}"
    )
    assert ok, f"{name} produced non-finite loss/grad"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}, torch={torch.__version__}")

    x, y = load_batch()
    print(f"batch: x={tuple(x.shape)}, y={tuple(y.shape)}, y_frac_bottom={y.mean().item():.3f}")

    # --- 1. loss only, on synthetic logits matched to label geometry ---
    print("loss-only checks (logits = scaled labels):")
    base_logits = (y * 8.0 - 4.0).unsqueeze(1)  # confident-ish toward truth
    for mode in ("below", "line"):
        lf = GeoSegLoss(mask_mode=mode, reg_weight=0.1, smooth_weight=0.1)
        check(f"geo_seg/{mode}", base_logits, y, lf)
    # extreme inputs: all-zero and all-one logits
    lf = build_loss("geo_seg", mask_mode="below")
    check("all-zero-logits", torch.zeros_like(base_logits), y, lf)
    check("all-one-logits", torch.ones_like(base_logits) * 10, y, lf)

    # --- 2. real model forward + backward, fp32 and AMP ---
    print("real-model checks (smp_unet/resnet34):")
    model = build_model(PipelineConfig(encoder_weights="none")).to(device)
    x_d, y_d = x.to(device), y.to(device)
    lf = build_loss(
        "geo_seg", mask_mode="below", geo_reg_weight=0.1, geo_smooth_weight=0.1, geo_depth_scale=0.1
    )

    model.zero_grad()
    logits = model(x_d)
    loss = lf(logits, y_d)
    loss.backward()
    gnorm = (
        torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])
        .norm()
        .item()
    )
    finite = torch.isfinite(loss).item() and np.isfinite(gnorm)
    print(
        f"  [fp32] loss={loss.item():.5f} param_grad_norm={gnorm:.4f} {'OK' if finite else 'FAIL'}"
    )
    assert finite

    if device == "cuda":
        scaler = torch.amp.GradScaler()
        model.zero_grad()
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(x_d)
            loss = lf(logits, y_d)
        scaler.scale(loss).backward()
        print(f"  [amp ] loss={loss.item():.5f} {'OK' if torch.isfinite(loss).item() else 'FAIL'}")
        assert torch.isfinite(loss).item()
    else:
        print("  [amp ] skipped (no CUDA)")

    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
