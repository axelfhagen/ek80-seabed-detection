"""Checkpoint resume logic for the training loop.

Factored out of train.py because the resume path is fiddly: it must mirror the
encoder unfreeze state before restoring the optimizer (so the param groups line
up) and re-load the prior training history so the saved curve stays continuous.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from deep_learning.training.config import PipelineConfig


def resolve_checkpoint_path(resume, prefer_last: bool = False) -> Path:
    """Resolve a --resume value (file or directory) to a concrete .pth path.

    With ``prefer_last`` set, a ``*_last.pth`` (rewritten every epoch) is chosen
    over ``*_best.pth`` when present, so a resumed run continues from where it
    actually stopped rather than rewinding to the best-val epoch. Evaluation
    callers leave it unset to keep loading the best checkpoint.
    """
    resume_path = Path(resume)
    if resume_path.is_dir():
        if prefer_last:
            last = sorted(resume_path.glob("*_last.pth"))
            if last:
                return last[0]
        candidates = sorted(resume_path.glob("*_best.pth"))
        if not candidates:
            raise SystemExit(f"ERROR: no *_best.pth found in {resume_path}")
        resume_path = candidates[0]
    if not resume_path.exists():
        raise SystemExit(f"ERROR: resume checkpoint not found: {resume_path}")
    return resume_path


def load_resume_state(cfg: PipelineConfig, model, optimizer, scaler, device, history):
    """Restore model (+ optimizer + AMP scaler + history) from cfg.resume.

    Mutates ``model``, ``scaler`` and ``history`` in place. Returns
    ``(optimizer, best_val_iou, best_epoch, start_epoch, scheduler_state)`` —
    the optimizer is rebuilt (and returned) when the checkpoint was past the
    unfreeze epoch, so the caller must use the returned object. ``scheduler_state``
    is the saved scheduler state_dict (or ``None`` for older checkpoints); the
    caller restores it onto the freshly built scheduler so the LR schedule
    continues across the resume.

    The GradScaler's loss-scale calibration is restored when present so the
    resumed run does not start from the default (high) scale on a converged
    model.
    """
    # Training resumes from _last.pth (continue where it stopped); eval-only
    # loads _best.pth (report the best-val model).
    resume_path = resolve_checkpoint_path(cfg.resume, prefer_last=not cfg.eval_only)

    ckpt = torch.load(resume_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    # Eval-only never touches the optimizer, and its frozen-encoder param
    # groups won't match a checkpoint trained with the encoder unfrozen, so
    # skip optimizer restoration entirely.
    if not cfg.eval_only:
        # If the run had already unfrozen the encoder before it stopped, mirror
        # that state so the optimizer param groups line up before loading.
        if cfg.unfreeze_after is not None and ckpt["epoch"] >= cfg.unfreeze_after:
            model.unfreeze_encoder()
            param_groups = [
                {
                    "params": [p for p in model.get_encoder_params() if p.requires_grad],
                    "lr": cfg.encoder_lr,
                },
                {
                    "params": [p for p in model.get_decoder_params() if p.requires_grad],
                    "lr": cfg.decoder_lr,
                },
            ]
            optimizer = torch.optim.AdamW(param_groups, weight_decay=cfg.weight_decay)

        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        # Older checkpoints predate AMP-scaler checkpointing — restore only when
        # both the live scaler and a saved state exist.
        if scaler is not None and ckpt.get("scaler_state_dict") is not None:
            scaler.load_state_dict(ckpt["scaler_state_dict"])

        # load_state_dict above also restored the saved per-group LRs, clobbering
        # any CLI override. When asked, re-apply the configured LRs so the resume
        # can continue at a different rate. Param groups are built encoder-first,
        # decoder-second (see train.py optimizer construction).
        if cfg.reset_lr_on_resume:
            group_lrs = [cfg.encoder_lr, cfg.decoder_lr]
            for group, lr in zip(optimizer.param_groups, group_lrs):
                group["lr"] = lr
            print(f"  reset LR on resume: encoder_lr={cfg.encoder_lr}, decoder_lr={cfg.decoder_lr}")
    resume_val_iou = ckpt.get("val_iou", 0.0)
    best_val_iou = resume_val_iou
    best_epoch = ckpt["epoch"]
    start_epoch = ckpt["epoch"] + 1

    # Reload prior training curve if present, so the saved history stays
    # continuous across the resume boundary. Truncate to ``start_epoch`` so the
    # curve never duplicates epochs — relevant when falling back to _best.pth,
    # whose epoch sits behind the trained tail. (Resuming from _last.pth the
    # history already ends exactly at start_epoch, so this is a no-op.)
    prior_history_path = cfg.checkpoint_dir / "training_history.json"
    if prior_history_path.exists():
        with open(prior_history_path) as f:
            prior = json.load(f)
        for k in history:
            if k in prior:
                history[k] = list(prior[k])[:start_epoch]

    # Enforce the loop invariant: the epoch loop appends each metric at index
    # len(history), so the epoch counter MUST equal the curve length. If they
    # diverge, the saved curve desyncs from the epoch number, producing the
    # impossible best_epoch > n_epochs_completed seen in legacy chunk dirs whose
    # _last.pth epoch ran ahead of a truncated/short training_history.json.
    # Truncation above caps the curve at start_epoch; if a corrupt or legacy dir
    # left it shorter, realign the counter down to the curve length. The model
    # and optimizer weights stay valid — only the epoch label shifts — so
    # bookkeeping is consistent from here on.
    n_hist = len(history.get("val_loss", []))
    if n_hist != start_epoch:
        print(
            f"  WARN: training_history.json has {n_hist} epochs but the "
            f"checkpoint is at epoch {start_epoch} — realigning the epoch "
            f"counter to the curve length to keep bookkeeping consistent."
        )
        start_epoch = n_hist
        best_epoch = min(best_epoch, start_epoch - 1)

    print(f"Resumed from {resume_path}")
    print(f"  checkpoint epoch {best_epoch + 1} (val_iou={resume_val_iou:.4f})")
    print(f"  continuing from epoch {start_epoch + 1}/{cfg.num_epochs}")

    scheduler_state = ckpt.get("scheduler_state_dict")
    return optimizer, best_val_iou, best_epoch, start_epoch, scheduler_state
