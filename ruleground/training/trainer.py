"""
Supervised Trainer (Stage 1)

Standard supervised pre-training with multi-task loss.
L = L_task + gamma * L_pred + delta * L_cons

Paper Section 5.3.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ruleground.training.losses import RuleGroundLoss
from ruleground.evaluation.metrics import compute_all_metrics
from ruleground.utils.checkpoint import save_checkpoint, BestModelTracker
from ruleground.utils.logging import MetricLogger
from ruleground.utils.distributed import is_main_process, reduce_mean

logger = logging.getLogger(__name__)


class SupervisedTrainer:
    """Stage 1: Supervised pre-training for RuleGround.

    Args:
        model:          RuleGround model.
        train_loader:   Training DataLoader.
        val_loader:     Validation DataLoader.
        optimizer:      Optimizer.
        scheduler:      LR scheduler (optional).
        loss_fn:        Loss function (defaults to RuleGroundLoss).
        max_epochs:     Maximum training epochs.
        output_dir:     Directory for checkpoints and logs.
        log_every:      Log metrics every N steps.
        eval_every:     Evaluate every N epochs.
        grad_clip:      Max gradient norm (0 = disabled).
        use_amp:        Use automatic mixed precision.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        loss_fn: Optional[RuleGroundLoss] = None,
        max_epochs: int = 10,
        output_dir: str = "experiments/supervised",
        log_every: int = 50,
        eval_every: int = 1,
        grad_clip: float = 1.0,
        use_amp: bool = False,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.max_epochs = max_epochs
        self.output_dir = Path(output_dir)
        self.log_every = log_every
        self.eval_every = eval_every
        self.grad_clip = grad_clip
        self.use_amp = use_amp

        # Defaults
        self.optimizer = optimizer or torch.optim.AdamW(
            model.parameters(), lr=1e-4, weight_decay=0.01
        )
        self.scheduler = scheduler
        self.loss_fn = loss_fn or RuleGroundLoss()

        # AMP scaler
        self.scaler = torch.amp.GradScaler("cuda") if use_amp else None

        # Tracking
        self.global_step = 0
        self.best_tracker = BestModelTracker(
            metric_name="q1_accuracy",
            mode="max",
            save_dir=self.output_dir / "checkpoints",
        )
        self.metric_logger = MetricLogger()

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Run one training epoch."""
        self.model.train()
        self.metric_logger.reset()

        for batch_idx, batch in enumerate(self.train_loader):
            # Move to device
            device = next(self.model.parameters()).device
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            # Forward
            with torch.amp.autocast("cuda", enabled=self.use_amp):
                outputs = self.model(
                    video=batch["video"],
                    mask=batch.get("mask"),
                    sport_ids=batch.get("sport_ids"),
                )
                loss, breakdown = self.loss_fn(outputs, batch)

            # Backward
            self.optimizer.zero_grad()
            if self.scaler:
                self.scaler.scale(loss).backward()
                if self.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.grad_clip > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()

            if self.scheduler:
                self.scheduler.step()

            # Log
            self.metric_logger.update(breakdown)
            self.global_step += 1

            if self.global_step % self.log_every == 0 and is_main_process():
                avg = self.metric_logger.average()
                lr = self.optimizer.param_groups[0]["lr"]
                logger.info(
                    f"Epoch {epoch} | Step {self.global_step} | LR {lr:.2e} | {self.metric_logger}"
                )

        return self.metric_logger.average()

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Run evaluation on validation set."""
        if self.val_loader is None:
            return {}

        self.model.eval()
        all_outputs = {"q1_logits": [], "q2_logits": [], "predicate_probs": [], "q5_preds": []}
        all_targets = {"q1_labels": [], "q2_labels": [], "predicate_labels": []}

        device = next(self.model.parameters()).device

        for batch in self.val_loader:
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            outputs = self.model(
                video=batch["video"],
                mask=batch.get("mask"),
                sport_ids=batch.get("sport_ids"),
            )

            for k in all_outputs:
                if k in outputs:
                    all_outputs[k].append(outputs[k].cpu())
            for k in all_targets:
                if k in batch:
                    all_targets[k].append(batch[k].cpu())

        # Concatenate
        cat_outputs = {k: torch.cat(v) for k, v in all_outputs.items() if v}
        cat_targets = {k: torch.cat(v) for k, v in all_targets.items() if v}

        metrics = compute_all_metrics(cat_outputs, cat_targets)
        return metrics

    def train(self) -> Dict[str, float]:
        """Full training loop."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        best_metrics = {}
        for epoch in range(1, self.max_epochs + 1):
            train_metrics = self.train_epoch(epoch)

            if is_main_process():
                logger.info(f"Epoch {epoch} train: {train_metrics}")

            # Evaluate
            if epoch % self.eval_every == 0 and self.val_loader is not None:
                val_metrics = self.evaluate()
                if is_main_process():
                    logger.info(f"Epoch {epoch} val: {val_metrics}")

                    # Track best model
                    is_best = self.best_tracker.update(
                        self.model, self.optimizer, epoch, self.global_step, val_metrics
                    )
                    if is_best:
                        best_metrics = val_metrics
                        logger.info(f"New best: {self.best_tracker.metric_name}="
                                    f"{self.best_tracker.best_value:.4f}")

            # Save periodic checkpoint
            if is_main_process():
                save_checkpoint(
                    self.model, self.optimizer, epoch, self.global_step,
                    train_metrics,
                    self.output_dir / "checkpoints" / f"epoch_{epoch}.pt",
                )

        return best_metrics
