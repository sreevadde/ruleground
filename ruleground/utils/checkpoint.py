"""
Model Checkpointing

Save/load model checkpoints with metadata for reproducibility.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    step: int,
    metrics: Dict[str, float],
    path: str | Path,
    scheduler: Optional[Any] = None,
    config: Optional[Dict] = None,
) -> None:
    """Save a training checkpoint.

    Args:
        model:     Model (handles DDP unwrapping).
        optimizer: Optimizer state.
        epoch:     Current epoch.
        step:      Global step count.
        metrics:   Current metrics dict.
        path:      Save path.
        scheduler: Optional LR scheduler.
        config:    Optional config dict for reproducibility.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Unwrap DDP if needed
    state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()

    checkpoint = {
        "model_state_dict": state_dict,
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "step": step,
        "metrics": metrics,
    }

    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    if config is not None:
        checkpoint["config"] = config

    torch.save(checkpoint, path)
    logger.info(f"Checkpoint saved: {path} (epoch={epoch}, step={step})")


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    strict: bool = True,
) -> Dict[str, Any]:
    """Load a training checkpoint.

    Args:
        path:      Checkpoint file path.
        model:     Model to load weights into.
        optimizer: Optional optimizer to restore state.
        scheduler: Optional scheduler to restore state.
        strict:    Whether to enforce strict state dict matching.

    Returns:
        Dict with 'epoch', 'step', 'metrics', and optional 'config'.
    """
    path = Path(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    # Handle DDP-wrapped models
    target = model.module if hasattr(model, "module") else model
    target.load_state_dict(checkpoint["model_state_dict"], strict=strict)

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    logger.info(
        f"Checkpoint loaded: {path} "
        f"(epoch={checkpoint.get('epoch', '?')}, step={checkpoint.get('step', '?')})"
    )

    return {
        "epoch": checkpoint.get("epoch", 0),
        "step": checkpoint.get("step", 0),
        "metrics": checkpoint.get("metrics", {}),
        "config": checkpoint.get("config"),
    }


class BestModelTracker:
    """Track the best model by a monitored metric."""

    def __init__(
        self,
        metric_name: str = "q1_accuracy",
        mode: str = "max",
        save_dir: str | Path = "checkpoints",
    ):
        self.metric_name = metric_name
        self.mode = mode
        self.save_dir = Path(save_dir)
        self.best_value = float("-inf") if mode == "max" else float("inf")
        self.best_epoch = -1

    def is_better(self, value: float) -> bool:
        if self.mode == "max":
            return value > self.best_value
        return value < self.best_value

    def update(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        step: int,
        metrics: Dict[str, float],
        config: Optional[Dict] = None,
    ) -> bool:
        """Check if current metrics are best, save if so.

        Returns:
            True if this was a new best.
        """
        value = metrics.get(self.metric_name, 0.0)
        if self.is_better(value):
            self.best_value = value
            self.best_epoch = epoch
            save_checkpoint(
                model, optimizer, epoch, step, metrics,
                self.save_dir / "best.pt",
                config=config,
            )
            return True
        return False
