"""U-Net with transfer learning for seabed segmentation.

Wraps segmentation-models-pytorch (smp) to provide:
  - Pretrained U-Net creation with configurable encoder
  - Three encoder freezing strategies for transfer learning
  - Combined Dice + BCE loss for binary segmentation
  - Training/validation step functions and segmentation metrics
  - Sliding-window inference for full echograms
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp


# ---------------------------------------------------------------------------
# Model creation
# ---------------------------------------------------------------------------

def create_unet(encoder_name='resnet34',
                encoder_weights='imagenet',
                in_channels=3,
                classes=1):
    """
    Create a U-Net with a pretrained encoder.

    Parameters
    ----------
    encoder_name    : backbone architecture ('resnet34', 'resnet50',
                      'efficientnet-b0', etc.)
    encoder_weights : pretrained weights ('imagenet' or None)
    in_channels     : number of input channels (3 for freq1/freq2/diff)
    classes         : number of output classes (1 for binary segmentation)

    Returns
    -------
    smp.Unet model
    """
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
        activation=None,  # raw logits — apply sigmoid in loss / inference
    )
    return model


def add_mc_dropout(model, p=0.1):
    """Insert Dropout2d into each decoder block for MC-Dropout inference.

    Each decoder block has two Conv2dReLU sub-modules (conv1, conv2), each
    a Sequential of [Conv2d, BatchNorm2d, ReLU].  This function appends a
    Dropout2d layer after the ReLU in every Conv2dReLU, so that calling
    ``model.train()`` at inference time produces stochastic outputs.

    Dropout has no learnable parameters, so existing checkpoints load
    without modification.

    Parameters
    ----------
    model : smp.Unet
    p     : dropout probability (0.1–0.3 is typical for MC-Dropout)

    Returns
    -------
    The model (for chaining).
    """
    for block in model.decoder.blocks:
        for conv_relu in [block.conv1, block.conv2]:
            conv_relu.add_module(str(len(conv_relu)), nn.Dropout2d(p=p))
    return model


# ---------------------------------------------------------------------------
# Freezing strategies
# ---------------------------------------------------------------------------

def freeze_encoder(model):
    """
    Freeze the entire encoder (all layers).

    Only the decoder and segmentation head are trainable.  This is the
    safest option when training data is limited — the ImageNet feature
    extractor is preserved exactly, and only the task-specific decoder
    learns from the echogram data.

    Returns the model for chaining.
    """
    for param in model.encoder.parameters():
        param.requires_grad = False
    return model


def unfreeze_encoder(model):
    """Unfreeze the entire encoder.  Returns the model for chaining."""
    for param in model.encoder.parameters():
        param.requires_grad = True
    return model


def freeze_encoder_stages(model, freeze_stages=3):
    """
    Freeze the first ``freeze_stages`` *named children* of the encoder.

    Note that the unit here is an encoder child module, not a ResNet stage.
    A torchvision/smp ResNet encoder lists its children in this order:

        conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4

    So ``freeze_stages=3`` (the default) freezes ``conv1``, ``bn1``, ``relu``
    — i.e. only the input stem, about 9,500 of the ~24M parameters. Every
    residual block (layer1–layer4) and the decoder stay trainable, which
    leaves 99.96% of the network trainable. At the default this is therefore
    near-full fine-tuning, not partial freezing of the deeper stages.

    Parameters
    ----------
    model          : smp.Unet model
    freeze_stages  : number of leading encoder children to freeze

    Returns
    -------
    The model (for chaining).
    """
    # Freeze the first `freeze_stages` named children of the encoder.
    # For ResNet these are: conv1, bn1, relu, maxpool, layer1, ...
    encoder = model.encoder

    # First, unfreeze everything
    for param in encoder.parameters():
        param.requires_grad = True

    # Then freeze the requested stages
    stages_frozen = 0
    for name, child in encoder.named_children():
        if stages_frozen >= freeze_stages:
            break
        for param in child.parameters():
            param.requires_grad = False
        stages_frozen += 1

    return model


def get_trainable_param_count(model):
    """
    Count trainable and frozen parameters.

    Returns
    -------
    dict with keys 'trainable', 'frozen', 'total', 'trainable_pct'
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    frozen = total - trainable
    return {
        'trainable': trainable,
        'frozen': frozen,
        'total': total,
        'trainable_pct': 100.0 * trainable / total if total > 0 else 0.0,
    }


def get_param_groups(model, encoder_lr, decoder_lr):
    """
    Build optimizer parameter groups with differential learning rates.

    The encoder (pretrained) uses a lower learning rate to preserve
    ImageNet features, while the decoder (randomly initialised) uses
    a higher rate to learn the segmentation task faster.

    Parameters
    ----------
    model       : smp.Unet
    encoder_lr  : learning rate for encoder parameters
    decoder_lr  : learning rate for decoder + segmentation head

    Returns
    -------
    list of dicts suitable for torch.optim optimizers
    """
    encoder_params = list(model.encoder.parameters())
    encoder_ids = {id(p) for p in encoder_params}
    decoder_params = [p for p in model.parameters()
                      if id(p) not in encoder_ids]

    return [
        {'params': [p for p in encoder_params if p.requires_grad],
         'lr': encoder_lr},
        {'params': [p for p in decoder_params if p.requires_grad],
         'lr': decoder_lr},
    ]


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------

class DiceBCELoss(nn.Module):
    """
    Combined Dice loss and binary cross-entropy.

    BCE provides per-pixel gradient signal for stable training.  Dice loss
    directly optimises the overlap metric and handles the class imbalance
    inherent in seabed segmentation (water column >> seabed area in most
    patches).

    Parameters
    ----------
    bce_weight : weight for the BCE term (Dice weight = 1 − bce_weight)
    smooth     : Laplace smoothing for Dice to avoid division by zero
    """

    def __init__(self, bce_weight=0.5, smooth=1.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        Parameters
        ----------
        logits  : (B, 1, H, W) raw model output (before sigmoid)
        targets : (B, H, W) float ground truth (0 or 1)
        """
        squeezed = logits.squeeze(1)  # (B, H, W)

        # BCE (use logits directly — safe under autocast)
        bce = F.binary_cross_entropy_with_logits(squeezed, targets, reduction='mean')

        probs = torch.sigmoid(squeezed)

        # Dice
        intersection = (probs * targets).sum(dim=(1, 2))
        union = probs.sum(dim=(1, 2)) + targets.sum(dim=(1, 2))
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice.mean()

        return self.bce_weight * bce + (1.0 - self.bce_weight) * dice_loss


class FocalLoss(nn.Module):
    """Binary focal loss — down-weights well-classified pixels."""

    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        squeezed = logits.squeeze(1)
        bce = F.binary_cross_entropy_with_logits(squeezed, targets, reduction='none')
        probs = torch.sigmoid(squeezed)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = self.alpha * (1 - p_t) ** self.gamma
        return (focal_weight * bce).mean()


class FocalDiceLoss(nn.Module):
    """Weighted combination of focal loss and Dice loss."""

    def __init__(self, focal_weight=0.5, alpha=0.25, gamma=2.0, smooth=1.0):
        super().__init__()
        self.focal_weight = focal_weight
        self.focal = FocalLoss(alpha=alpha, gamma=gamma)
        self.smooth = smooth

    def forward(self, logits, targets):
        f_loss = self.focal(logits, targets)

        probs = torch.sigmoid(logits.squeeze(1))
        intersection = (probs * targets).sum(dim=(1, 2))
        union = probs.sum(dim=(1, 2)) + targets.sum(dim=(1, 2))
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice.mean()

        return self.focal_weight * f_loss + (1.0 - self.focal_weight) * dice_loss


class TverskyLoss(nn.Module):
    """Tversky loss — generalises Dice with independent FN/FP weights.

    alpha controls false-negative penalty, beta controls false-positive.
    alpha=beta=0.5 is equivalent to Dice loss.
    """

    def __init__(self, alpha=0.7, beta=0.3, smooth=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits.squeeze(1))
        tp = (probs * targets).sum(dim=(1, 2))
        fn = ((1 - probs) * targets).sum(dim=(1, 2))
        fp = (probs * (1 - targets)).sum(dim=(1, 2))
        tversky = (tp + self.smooth) / (tp + self.alpha * fn + self.beta * fp + self.smooth)
        return 1.0 - tversky.mean()


class GeoSegLoss(nn.Module):
    """Segmentation loss with bottom-geometry regularisation.

    Combines a pixel-wise segmentation backbone (Dice + BCE) with two terms
    that act on the per-ping bottom depth derived differentiably from the
    probability map:

    * a regression term that penalises the bottom-depth error directly, and
    * a curvature term that penalises sharp per-ping spikes while leaving
      genuine (smoothly sloping) bathymetry unpenalised.

    The per-ping bottom depth is read off the probabilities without an
    argmax, so the whole loss is differentiable. For 'below' masks (water
    above the seabed, filled below) the bottom row index of column j equals
    the number of water pixels, i.e. ``sum_i (1 - p_ij)``. For 'line' masks
    the depth is the probability-weighted mean row (soft argmax).

    Parameters
    ----------
    bce_weight   : weight of the BCE term inside the Dice+BCE backbone
    reg_weight   : weight of the depth-regression term
    smooth_weight: weight of the curvature (anti-spike) term
    depth_scale  : metres per depth row; converts row units to metres so the
                   regression/curvature terms are in physical units. Leave at
                   1.0 to work in row units and fold the scale into the weights.
    huber_delta  : transition point (in scaled units) of the Huber regression
    mask_mode    : 'below' (default) or 'line' — selects the depth proxy
    smooth       : Laplace smoothing for the Dice backbone
    """

    def __init__(self, bce_weight=0.5, reg_weight=0.1, smooth_weight=0.1,
                 depth_scale=1.0, huber_delta=5.0, mask_mode='below', smooth=1.0):
        super().__init__()
        self.base = DiceBCELoss(bce_weight=bce_weight, smooth=smooth)
        self.reg_weight = reg_weight
        self.smooth_weight = smooth_weight
        self.depth_scale = depth_scale
        self.huber_delta = huber_delta
        self.mask_mode = mask_mode

    def _soft_depth(self, x):
        """Per-ping bottom depth (B, W) from a (B, H, W) probability/label map."""
        if self.mask_mode == 'line':
            H = x.shape[1]
            rows = torch.arange(H, device=x.device, dtype=x.dtype).view(1, H, 1)
            weight = x.sum(dim=1).clamp_min(1e-6)
            return (x * rows).sum(dim=1) / weight
        # 'below': bottom row = number of water pixels above it
        return (1.0 - x).sum(dim=1)

    def forward(self, logits, targets):
        base = self.base(logits, targets)

        probs = torch.sigmoid(logits.squeeze(1))      # (B, H, W)
        d_pred = self._soft_depth(probs)               # (B, W)
        d_true = self._soft_depth(targets)             # (B, W)

        diff = (d_pred - d_true) * self.depth_scale
        reg = F.huber_loss(diff, torch.zeros_like(diff), delta=self.huber_delta)

        if d_pred.shape[1] >= 3:
            curv = d_pred[:, 2:] - 2.0 * d_pred[:, 1:-1] + d_pred[:, :-2]
            smooth = (curv * self.depth_scale).abs().mean()
        else:
            smooth = d_pred.new_zeros(())

        return base + self.reg_weight * reg + self.smooth_weight * smooth


def soft_argmax_depth(logits):
    """Per-ping bottom row ``(B, W)`` from ``(B, 1, H, W)`` logits via soft-argmax.

    Softmax over the depth (height) axis turns each column into a distribution
    over rows; the expectation is the (continuous, differentiable) bottom row.
    This is the regression decoding shared by the loss, metrics, and inference.
    """
    x = logits.squeeze(1)                                       # (B, H, W)
    H = x.shape[1]
    p = torch.softmax(x, dim=1)                                 # over rows, per col
    rows = torch.arange(H, device=x.device, dtype=x.dtype).view(1, H, 1)
    return (p * rows).sum(dim=1)                                # (B, W)


def target_depth_from_mask(targets, mask_mode='below'):
    """Ground-truth per-ping bottom row ``(B, W)`` from a ``(B, H, W)`` mask.

    For 'below' masks the bottom row equals the number of water pixels above it
    (``sum_i (1 - m_ij)``); for 'line' masks it is the probability-weighted mean
    row (soft argmax of the thin line).
    """
    if mask_mode == 'line':
        H = targets.shape[1]
        rows = torch.arange(H, device=targets.device, dtype=targets.dtype).view(1, H, 1)
        w = targets.sum(dim=1).clamp_min(1e-6)
        return (targets * rows).sum(dim=1) / w
    return (1.0 - targets).sum(dim=1)


class RegressionLoss(nn.Module):
    """Direct bottom-line regression via differentiable soft-argmax.

    The model output is read, per ping (column), as a distribution over depth
    rows (softmax over the height axis); its expectation is the predicted bottom
    row. A Huber term pulls that prediction to the true bottom row and a
    curvature term keeps the line smooth (genuine sloping bathymetry is
    unpenalised, sharp per-ping spikes are). Unlike GeoSegLoss there is no
    per-pixel Dice/BCE backbone — the seabed is recovered purely as a regressed
    line, not a segmented region.

    Parameters
    ----------
    depth_scale  : metres per depth row (1.0 = work in row units)
    huber_delta  : Huber transition point (in scaled units)
    smooth_weight: weight of the curvature (anti-spike) term
    mask_mode    : 'below' or 'line' — how the target depth is read off the mask
    """

    def __init__(self, depth_scale=1.0, huber_delta=5.0, smooth_weight=0.1,
                 mask_mode='below'):
        super().__init__()
        self.depth_scale = depth_scale
        self.huber_delta = huber_delta
        self.smooth_weight = smooth_weight
        self.mask_mode = mask_mode

    def forward(self, logits, targets):
        d_pred = soft_argmax_depth(logits)                      # (B, W)
        d_true = target_depth_from_mask(targets, self.mask_mode)

        diff = (d_pred - d_true) * self.depth_scale
        reg = F.huber_loss(diff, torch.zeros_like(diff), delta=self.huber_delta)

        if d_pred.shape[1] >= 3:
            curv = d_pred[:, 2:] - 2.0 * d_pred[:, 1:-1] + d_pred[:, :-2]
            smooth = (curv * self.depth_scale).abs().mean()
        else:
            smooth = d_pred.new_zeros(())

        return reg + self.smooth_weight * smooth


def build_loss(loss_type, bce_weight=0.5, focal_alpha=0.25, focal_gamma=2.0,
               tversky_alpha=0.7, tversky_beta=0.3,
               geo_reg_weight=0.1, geo_smooth_weight=0.1,
               geo_depth_scale=1.0, geo_huber_delta=5.0, mask_mode='below'):
    """Factory for loss functions based on loss_type string."""
    if loss_type == "dice_bce":
        return DiceBCELoss(bce_weight=bce_weight)
    elif loss_type == "focal":
        return FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
    elif loss_type == "focal_dice":
        return FocalDiceLoss(focal_weight=0.5, alpha=focal_alpha, gamma=focal_gamma)
    elif loss_type == "tversky":
        return TverskyLoss(alpha=tversky_alpha, beta=tversky_beta)
    elif loss_type == "geo_seg":
        return GeoSegLoss(bce_weight=bce_weight, reg_weight=geo_reg_weight,
                          smooth_weight=geo_smooth_weight, depth_scale=geo_depth_scale,
                          huber_delta=geo_huber_delta, mask_mode=mask_mode)
    elif loss_type == "regression":
        return RegressionLoss(depth_scale=geo_depth_scale, huber_delta=geo_huber_delta,
                              smooth_weight=geo_smooth_weight, mask_mode=mask_mode)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_metrics(logits, targets, threshold=0.5, mask_mode='below',
                    task='segmentation'):
    """
    Compute binary segmentation metrics.

    Parameters
    ----------
    logits    : (B, 1, H, W) raw model output
    targets   : (B, H, W) ground truth
    threshold : probability threshold for positive class
    mask_mode : 'below' or 'line' — selects the soft bottom-depth proxy used
                for ``bottom_mae`` (mirrors GeoSegLoss._soft_depth)
    task      : 'segmentation' (default) or 'regression'. In regression mode the
                bottom row is the soft-argmax of the column-softmax output and
                IoU/Dice are computed on 'below' masks reconstructed from the
                predicted and true bottom lines (so they stay comparable to the
                segmentation runs while measuring line agreement).

    Returns
    -------
    dict with 'iou', 'dice', 'precision', 'recall', 'bottom_mae'

    ``bottom_mae`` is the mean absolute per-ping bottom-row error (in rows),
    read off the *probabilities* without thresholding. Unlike IoU/Dice it keeps
    moving as confidence sharpens (e.g. bottom pixels 0.6 → 0.9), so it stays
    sensitive after the thresholded metrics have saturated.
    """
    if task == 'regression':
        d_pred = soft_argmax_depth(logits)                       # (B, W)
        d_true = target_depth_from_mask(targets, mask_mode)      # (B, W)
        bottom_mae = (d_pred - d_true).abs().mean()

        H = logits.shape[2]
        rows = torch.arange(H, device=logits.device,
                            dtype=d_pred.dtype).view(1, H, 1)
        preds = (rows >= d_pred.unsqueeze(1)).float()            # below-mask
        gt = (rows >= d_true.unsqueeze(1)).float()
        intersection = (preds * gt).sum()
        pred_sum = preds.sum()
        target_sum = gt.sum()
        union = pred_sum + target_sum - intersection
        eps = 1e-7
        return {
            'iou': ((intersection + eps) / (union + eps)).item(),
            'dice': ((2.0 * intersection + eps) / (pred_sum + target_sum + eps)).item(),
            'precision': ((intersection + eps) / (pred_sum + eps)).item(),
            'recall': ((intersection + eps) / (target_sum + eps)).item(),
            'bottom_mae': bottom_mae.item(),
        }

    probs = torch.sigmoid(logits.squeeze(1))
    preds = (probs > threshold).float()

    intersection = (preds * targets).sum()
    pred_sum = preds.sum()
    target_sum = targets.sum()
    union = pred_sum + target_sum - intersection

    eps = 1e-7
    iou = (intersection + eps) / (union + eps)
    dice = (2.0 * intersection + eps) / (pred_sum + target_sum + eps)
    precision = (intersection + eps) / (pred_sum + eps)
    recall = (intersection + eps) / (target_sum + eps)

    # Soft per-ping bottom row, then mean absolute error against the target.
    if mask_mode == 'line':
        H = probs.shape[1]
        rows = torch.arange(H, device=probs.device, dtype=probs.dtype).view(1, H, 1)
        d_pred = (probs * rows).sum(dim=1) / probs.sum(dim=1).clamp_min(1e-6)
        d_true = (targets * rows).sum(dim=1) / targets.sum(dim=1).clamp_min(1e-6)
    else:  # 'below': bottom row = number of water pixels above it
        d_pred = (1.0 - probs).sum(dim=1)
        d_true = (1.0 - targets).sum(dim=1)
    bottom_mae = (d_pred - d_true).abs().mean()

    return {
        'iou': iou.item(),
        'dice': dice.item(),
        'precision': precision.item(),
        'recall': recall.item(),
        'bottom_mae': bottom_mae.item(),
    }


# ---------------------------------------------------------------------------
# Training and validation steps
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, loss_fn, device, scaler=None,
                    mask_mode='below', grad_clip_norm=None,
                    task='segmentation'):
    """
    Train for one epoch.

    Parameters
    ----------
    model         : U-Net model
    loader        : DataLoader yielding (images, masks)
    optimizer     : torch optimizer
    loss_fn       : loss function (e.g. DiceBCELoss)
    device        : 'cuda', 'mps', or 'cpu'
    scaler        : torch.amp.GradScaler for mixed precision (None = disabled)
    mask_mode     : passed to compute_metrics for the bottom-row error proxy
    grad_clip_norm: max global gradient L2-norm; None disables clipping. Caps
                    runaway steps so a single spike can't blow the run to NaN.

    Returns
    -------
    dict with 'loss', 'iou', 'dice', 'bottom_mae', 'n_nonfinite'

    A batch whose loss is non-finite is skipped (no backward/step) so it cannot
    poison the weights; ``n_nonfinite`` counts these. If every batch is skipped
    the loss is reported as NaN rather than 0 so a dead model never registers as
    an improvement.
    """
    model.train()
    running_loss = 0.0
    running_iou = 0.0
    running_dice = 0.0
    running_bottom_mae = 0.0
    n_batches = 0
    n_nonfinite = 0

    use_amp = scaler is not None
    amp_dtype = torch.float16 if device == 'cuda' else torch.bfloat16

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.amp.autocast(device_type=str(device), dtype=amp_dtype):
                logits = model(images)
                loss = loss_fn(logits, masks)
            if not torch.isfinite(loss):
                n_nonfinite += 1
                continue
            scaler.scale(loss).backward()
            # unscale_ records any inf/nan grads in the scaler's state, so a
            # subsequent step is still skipped even though clipping rewrites
            # the (inf) grads — this is the documented AMP+clip ordering.
            if grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = loss_fn(logits, masks)
            if not torch.isfinite(loss):
                n_nonfinite += 1
                continue
            loss.backward()
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

        metrics = compute_metrics(logits.detach(), masks, mask_mode=mask_mode,
                                  task=task)
        running_loss += loss.item()
        running_iou += metrics['iou']
        running_dice += metrics['dice']
        running_bottom_mae += metrics['bottom_mae']
        n_batches += 1

    return {
        'loss': running_loss / n_batches if n_batches else float('nan'),
        'iou': running_iou / n_batches if n_batches else 0.0,
        'dice': running_dice / n_batches if n_batches else 0.0,
        'bottom_mae': running_bottom_mae / n_batches if n_batches else float('nan'),
        'n_nonfinite': n_nonfinite,
    }


@torch.no_grad()
def validate(model, loader, loss_fn, device, mask_mode='below',
             task='segmentation'):
    """
    Run validation.  Same signature/return as ``train_one_epoch`` but
    without gradient computation.
    """
    model.eval()
    running_loss = 0.0
    running_iou = 0.0
    running_dice = 0.0
    running_bottom_mae = 0.0
    n_batches = 0

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        logits = model(images)
        loss = loss_fn(logits, masks)

        metrics = compute_metrics(logits, masks, mask_mode=mask_mode, task=task)
        running_loss += loss.item()
        running_iou += metrics['iou']
        running_dice += metrics['dice']
        running_bottom_mae += metrics['bottom_mae']
        n_batches += 1

    return {
        'loss': running_loss / max(n_batches, 1),
        'iou': running_iou / max(n_batches, 1),
        'dice': running_dice / max(n_batches, 1),
        'bottom_mae': running_bottom_mae / max(n_batches, 1),
    }


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    """Stop training when a monitored score stops improving."""

    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None

    def __call__(self, score):
        if self.best_score is None or score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


# ---------------------------------------------------------------------------
# Full-echogram inference (sliding window with overlap averaging)
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_full_echogram(model, sv_channels, target_size, device,
                          patch_width=512, stride=128, task='segmentation'):
    """
    Run inference on a complete echogram using a sliding window.

    Overlapping patches are averaged for seamless prediction.

    Parameters
    ----------
    model        : trained U-Net
    sv_channels  : (3, n_range, n_pings) float32 numpy array
                   (normalised freq1, freq2, difference)
    target_size  : (height, width) that the model expects
    device       : 'cuda', 'mps', or 'cpu'
    patch_width  : number of pings per patch (must match training)
    stride       : step between patches

    Returns
    -------
    prediction : (n_range, n_pings) float32 probability map [0, 1]
    """
    import cv2

    model.eval()
    n_ch, n_range, n_pings = sv_channels.shape
    h, w = target_size

    # Accumulate predictions with overlap averaging
    sum_pred = np.zeros((n_range, n_pings), dtype=np.float64)
    count = np.zeros((n_range, n_pings), dtype=np.float64)

    # Window start positions covering the full ping axis. Echograms narrower
    # than patch_width get a single full-width window (otherwise the strided
    # range is empty and the whole file predicts zeros). The final window is
    # pinned to the right edge so trailing pings are always covered.
    if n_pings <= patch_width:
        windows = [(0, n_pings)]
    else:
        starts = list(range(0, n_pings - patch_width + 1, stride))
        if starts[-1] != n_pings - patch_width:
            starts.append(n_pings - patch_width)
        windows = [(s, s + patch_width) for s in starts]

    for start, end in windows:
        pw = end - start
        patch = sv_channels[:, :, start:end]  # (C, n_range, pw)

        # Resize to model input size
        patch_hwc = np.transpose(patch, (1, 2, 0))  # (n_range, pw, C)
        patch_resized = cv2.resize(patch_hwc, (w, h),
                                   interpolation=cv2.INTER_LINEAR)
        if patch_resized.ndim == 2:          # cv2 drops a singleton channel
            patch_resized = patch_resized[..., None]
        patch_chw = np.transpose(patch_resized, (2, 0, 1))  # (C, h, w)

        # Model forward pass
        inp = torch.from_numpy(patch_chw[np.newaxis]).float().to(device)
        logits = model(inp)  # (1, 1, h, w)
        if task == 'regression':
            # Decode the column-softmax to a per-ping bottom row, then paint a
            # 'below' mask so the downstream thresholding / extract_bottom_line
            # path (shared with segmentation) recovers the same line.
            d = soft_argmax_depth(logits)[0]                  # (w,) rows in [0, h)
            rr = torch.arange(h, device=device, dtype=d.dtype).view(h, 1)
            probs = (rr >= d.view(1, w)).float().cpu().numpy()  # (h, w)
        else:
            probs = torch.sigmoid(logits)[0, 0].cpu().numpy()  # (h, w)

        # Resize prediction back to this window's ping width
        probs_orig = cv2.resize(probs, (pw, n_range),
                                interpolation=cv2.INTER_LINEAR)

        sum_pred[:, start:end] += probs_orig
        count[:, start:end] += 1.0

    # Average overlapping predictions
    prediction = np.divide(sum_pred, count,
                           out=np.zeros_like(sum_pred),
                           where=count > 0).astype(np.float32)
    return prediction
