"""Run synthetic inputs through the seabed U-Net and plot the outputs.

This script probes the model with a few deliberately degenerate 3-channel
inputs:
  - all zeros
  - all ones
  - Gaussian noise clipped to [0, 1]

For each case it writes a diagnostic figure with:
  - the three input channels
  - the full probability map
  - the extracted bottom line
  - a conformal band around that line
  - a Grad-CAM heatmap and overlay

The conformal band uses the calibrated constant-width q_hat from the existing
``Figures/conformal/sigma_compare/summary.txt`` file, which was generated with
the same checkpoint.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm

sys.path.insert(0, str(CODE_ROOT))
sys.path.insert(0, str(CODE_ROOT / "validation"))
sys.path.insert(0, str(CODE_ROOT / "data_visualization"))
sys.path.insert(0, str(CODE_ROOT / "deep_learning" / "training"))

from cam_utils import (
    channels_to_rgb_composite,
    gradcam_target_layer,
    overlay_cam,
)
from conformal import extract_bottom_depths, predict_depth_band
from thesis_style import apply_thesis_style


CHECKPOINT = PROJECT_ROOT / "operational_use" / "model_path" / "smp_unet_resnet34_best.pth"
CONFORMAL_SUMMARY = PROJECT_ROOT / "Figures" / "conformal" / "sigma_compare" / "summary.txt"
OUTPUT_DIR = PROJECT_ROOT / "Figures" / "synthetic_insertions"
TARGET_SIZE = (512, 512)
PATCH_WIDTH = 512
STRIDE = 256
ALPHA_LABEL = 0.10
RNG_SEED = 7


class Conv2dReLU(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        base = tvm.resnet34(weights=None)
        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x0 = self.relu(self.bn1(self.conv1(x)))
        x1 = self.layer1(self.maxpool(x0))
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return [x0, x1, x2, x3, x4]


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.skip_channels = skip_channels
        self.conv1 = Conv2dReLU(in_channels + skip_channels, out_channels)
        self.conv2 = Conv2dReLU(out_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None = None):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                DecoderBlock(512, 256, 256),
                DecoderBlock(256, 128, 128),
                DecoderBlock(128, 64, 64),
                DecoderBlock(64, 64, 32),
                DecoderBlock(32, 0, 16),
            ]
        )

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        x = features[-1]
        skips = features[-2::-1]
        for i, block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = block(x, skip)
        return x


class SimpleSmpLikeUnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(16, 1, kernel_size=3, padding=1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        x = self.decoder(features)
        x = self.segmentation_head(x)
        return x


def _load_model(checkpoint_path: Path, device: str) -> SimpleSmpLikeUnet:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = SimpleSmpLikeUnet()
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = (
        ckpt["model_state_dict"]
        if (isinstance(ckpt, dict) and "model_state_dict" in ckpt)
        else ckpt
    )
    cleaned = {k.removeprefix("model."): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint mismatch. Missing={missing[:5]} unexpected={unexpected[:5]}"
        )
    model.to(device)
    model.eval()
    return model


def _predict_full_echogram(
    model: nn.Module,
    sv_channels: np.ndarray,
    target_size: tuple[int, int],
    device: str,
    patch_width: int = 512,
    stride: int = 128,
) -> np.ndarray:
    """Sliding-window inference, rewritten to avoid the missing smp dependency."""
    n_ch, n_range, n_pings = sv_channels.shape
    h, w = target_size

    sum_pred = np.zeros((n_range, n_pings), dtype=np.float64)
    count = np.zeros((n_range, n_pings), dtype=np.float64)

    if n_pings <= patch_width:
        windows = [(0, n_pings)]
    else:
        starts = list(range(0, n_pings - patch_width + 1, stride))
        if starts[-1] != n_pings - patch_width:
            starts.append(n_pings - patch_width)
        windows = [(s, s + patch_width) for s in starts]

    for start, end in windows:
        patch = sv_channels[:, :, start:end]
        patch_t = torch.from_numpy(patch[None]).float().to(device)
        with torch.no_grad():
            logits = model(patch_t)
            probs = torch.sigmoid(logits)[0, 0]
            probs = F.interpolate(
                probs[None, None],
                size=(n_range, end - start),
                mode="bilinear",
                align_corners=False,
            )[0, 0]
        probs_np = probs.cpu().numpy()
        sum_pred[:, start:end] += probs_np
        count[:, start:end] += 1.0

    prediction = np.divide(sum_pred, count, out=np.zeros_like(sum_pred), where=count > 0).astype(
        np.float32
    )
    return prediction


def _device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():  # pragma: no cover - Apple Silicon only
        return "mps"
    return "cpu"


def _load_q_hat(summary_path: Path) -> float:
    text = summary_path.read_text(encoding="utf-8")
    match = re.search(r"^q_constant:\s*([0-9.]+)\s*$", text, flags=re.MULTILINE)
    if not match:
        raise ValueError(f"Could not find q_constant in {summary_path}")
    return float(match.group(1))


def _synthetic_inputs(height: int, width: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(RNG_SEED)
    return {
        "all_zeros": np.zeros((3, height, width), dtype=np.float32),
        "all_ones": np.ones((3, height, width), dtype=np.float32),
        "gaussian_noise": np.clip(
            rng.normal(loc=0.5, scale=0.20, size=(3, height, width)),
            0.0,
            1.0,
        ).astype(np.float32),
    }


def _fallback_bottom(prob: np.ndarray, range_arr: np.ndarray) -> np.ndarray:
    """Use a deterministic full-length line when thresholded detection is empty."""
    idx = np.argmax(prob, axis=0)
    return range_arr[idx].astype(np.float64)


def _target_mask(prob: np.ndarray) -> np.ndarray:
    """Construct a non-empty Grad-CAM target mask from a probability map."""
    thr = float(np.quantile(prob, 0.95))
    mask = (prob >= thr).astype(np.uint8)
    if mask.sum() == 0 or mask.sum() == mask.size:
        mask = np.zeros_like(prob, dtype=np.uint8)
        mask[np.unravel_index(int(np.argmax(prob)), prob.shape)] = 1
    return mask


def _gradcam_single(
    model: nn.Module,
    target_layer: nn.Module,
    input_chw: np.ndarray,
    target_mask: np.ndarray,
    device: str,
) -> np.ndarray:
    """Minimal Grad-CAM implementation without external dependencies."""
    activations: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []

    def _forward_hook(_module, _inputs, output):
        activations.append(output)

    def _backward_hook(_module, _grad_input, grad_output):
        gradients.append(grad_output[0])

    h_fwd = target_layer.register_forward_hook(_forward_hook)
    h_bwd = target_layer.register_full_backward_hook(_backward_hook)

    try:
        inp = torch.from_numpy(input_chw[None]).float().to(device)
        mask = torch.from_numpy(target_mask.astype(np.float32)).to(device)

        model.zero_grad(set_to_none=True)
        logits = model(inp)[0, 0]
        target = (logits * mask).sum()
        target.backward()

        if not activations or not gradients:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients")

        acts = activations[0][0]  # (C, H, W)
        grads = gradients[0][0]  # (C, H, W)
        weights = grads.mean(dim=(1, 2), keepdim=True)
        cam = torch.relu((weights * acts).sum(dim=0, keepdim=True))
        cam = cam[None]  # (1, 1, H, W)
        cam = F.interpolate(cam, size=target_mask.shape, mode="bilinear", align_corners=False)[0, 0]
        cam = cam - cam.min()
        denom = cam.max().clamp_min(1e-8)
        cam = (cam / denom).detach().cpu().numpy().astype(np.float32)
        return cam
    finally:
        h_fwd.remove()
        h_bwd.remove()


def _plot_case(
    name: str,
    input_chw: np.ndarray,
    prob: np.ndarray,
    range_arr: np.ndarray,
    q_hat: float,
    model,
    device: str,
    out_dir: Path,
) -> None:
    _, n_pings = prob.shape
    extent = [0, n_pings, float(range_arr[-1]), float(range_arr[0])]
    ping_axis = np.arange(n_pings)

    pred_depths = extract_bottom_depths(prob, range_arr, threshold=0.5)
    valid_pred = np.isfinite(pred_depths)
    if valid_pred.any():
        line_depths = pred_depths
        line_label = f"thresholded bottom line ({valid_pred.sum()}/{n_pings} pings)"
    else:
        line_depths = _fallback_bottom(prob, range_arr)
        valid_pred = np.isfinite(line_depths)
        line_label = "fallback argmax line (no pings crossed 0.5)"

    upper, lower = predict_depth_band(line_depths, q_hat)
    if not valid_pred.any():
        upper = np.asarray(upper, dtype=np.float64)
        lower = np.asarray(lower, dtype=np.float64)

    layer = gradcam_target_layer(model)
    if layer is None:
        raise RuntimeError("Model does not expose a decoder block for Grad-CAM")

    target = _target_mask(prob)
    cam = _gradcam_single(model, layer, input_chw, target, device)
    composite = channels_to_rgb_composite(input_chw)
    cam_overlay = overlay_cam(cam, composite, alpha=0.55, cmap="jet")

    apply_thesis_style()
    fig, axes = plt.subplots(2, 3, figsize=(22, 12), dpi=250)

    channel_titles = ["38 kHz input", "200 kHz input", "38 - 200 kHz diff"]
    cmaps = ["viridis", "viridis", "coolwarm"]
    for i in range(3):
        vmin, vmax = (0.0, 1.0) if i < 2 else (-1.0, 1.0)
        im = axes[0, i].imshow(
            input_chw[i],
            aspect="auto",
            cmap=cmaps[i],
            vmin=vmin,
            vmax=vmax,
            origin="upper",
            extent=extent,
        )
        axes[0, i].set_title(channel_titles[i], fontsize=11)
        axes[0, i].set_xlabel("Ping")
        axes[0, i].set_ylabel("Range (m)")
        fig.colorbar(im, ax=axes[0, i], fraction=0.046, pad=0.03)

    im = axes[1, 0].imshow(
        prob, aspect="auto", cmap="RdYlBu_r", vmin=0.0, vmax=1.0, origin="upper", extent=extent
    )
    axes[1, 0].fill_between(
        ping_axis[valid_pred],
        upper[valid_pred],
        lower[valid_pred],
        color="#ff7f0e",
        alpha=0.28,
        label=f"conformal band (±{q_hat:.2f} m)",
    )
    axes[1, 0].plot(
        ping_axis[valid_pred], line_depths[valid_pred], color="#d62728", lw=1.25, label=line_label
    )
    axes[1, 0].set_title(
        f"Probability map  max={prob.max():.3f}  mean={prob.mean():.3f}",
        fontsize=11,
    )
    axes[1, 0].set_xlabel("Ping")
    axes[1, 0].set_ylabel("Range (m)")
    axes[1, 0].legend(loc="lower right", fontsize=8, framealpha=0.92)
    fig.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.03, label="p(bottom)")

    im = axes[1, 1].imshow(cam_overlay, aspect="auto", origin="upper", extent=extent)
    axes[1, 1].set_title("Grad-CAM overlay", fontsize=11)
    axes[1, 1].set_xlabel("Ping")
    axes[1, 1].set_ylabel("Range (m)")

    im = axes[1, 2].imshow(
        cam, aspect="auto", cmap="jet", vmin=0, vmax=1, origin="upper", extent=extent
    )
    axes[1, 2].set_title("Grad-CAM heatmap", fontsize=11)
    axes[1, 2].set_xlabel("Ping")
    axes[1, 2].set_ylabel("Range (m)")
    fig.colorbar(im, ax=axes[1, 2], fraction=0.046, pad=0.03, label="importance")

    fig.suptitle(
        f"Synthetic insertion test: {name}  |  q_hat={q_hat:.2f} m  "
        f"|  target detections={int(valid_pred.sum())}/{n_pings}",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path = out_dir / f"{name}.png"
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {out_path}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = _device()
    q_hat = _load_q_hat(CONFORMAL_SUMMARY)

    print(f"Device: {device}")
    print(f"Checkpoint: {CHECKPOINT}")
    print(f"Conformal band q_hat: {q_hat:.3f} m")

    model = _load_model(CHECKPOINT, device)

    cases = _synthetic_inputs(*TARGET_SIZE)
    range_arr = np.linspace(0.0, 150.0, TARGET_SIZE[0], dtype=np.float32)

    for name, input_chw in cases.items():
        prob = _predict_full_echogram(
            model,
            input_chw,
            TARGET_SIZE,
            device,
            patch_width=PATCH_WIDTH,
            stride=STRIDE,
        )
        _plot_case(name, input_chw, prob, range_arr, q_hat, model, device, OUTPUT_DIR)

    summary_path = OUTPUT_DIR / "summary.txt"
    summary_path.write_text(
        "\n".join(
            [
                f"checkpoint: {CHECKPOINT}",
                f"q_hat_m: {q_hat:.6f}",
                f"cases: {', '.join(cases.keys())}",
                f"output_dir: {OUTPUT_DIR}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
