"""
Training Callbacks

W&B logging, checkpointing, early stopping, and LR scheduling.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class Callback:
    """Base callback class."""

    def on_train_start(self, trainer: Any) -> None:
        pass

    def on_train_end(self, trainer: Any) -> None:
        pass

    def on_epoch_start(self, trainer: Any, epoch: int) -> None:
        pass

    def on_epoch_end(self, trainer: Any, epoch: int, metrics: Dict[str, float]) -> None:
        pass

    def on_step(self, trainer: Any, step: int, metrics: Dict[str, float]) -> None:
        pass


class WandbCallback(Callback):
    """Weights & Biases logging callback.

    Args:
        project:  W&B project name.
        name:     Run name.
        config:   Config dict to log.
        log_every: Log every N steps.
    """

    def __init__(
        self,
        project: str = "ruleground",
        name: Optional[str] = None,
        config: Optional[Dict] = None,
        log_every: int = 50,
    ):
        self.project = project
        self.name = name
        self.config = config
        self.log_every = log_every
        self._run = None

    def on_train_start(self, trainer: Any) -> None:
        try:
            import wandb
            self._run = wandb.init(
                project=self.project,
                name=self.name,
                config=self.config or {},
                reinit=True,
            )
            logger.info(f"W&B initialized: {self._run.url}")
        except ImportError:
            logger.warning("wandb not installed, skipping W&B logging")
        except Exception as e:
            logger.warning(f"W&B init failed: {e}")

    def on_step(self, trainer: Any, step: int, metrics: Dict[str, float]) -> None:
        if self._run and step % self.log_every == 0:
            import wandb
            wandb.log(metrics, step=step)

    def on_epoch_end(self, trainer: Any, epoch: int, metrics: Dict[str, float]) -> None:
        if self._run:
            import wandb
            wandb.log({f"epoch/{k}": v for k, v in metrics.items()}, step=epoch)

    def on_train_end(self, trainer: Any) -> None:
        if self._run:
            import wandb
            wandb.finish()


class EarlyStopping(Callback):
    """Stop training if validation metric doesn't improve.

    Args:
        metric_name: Metric to monitor.
        mode:        'max' or 'min'.
        patience:    Epochs to wait before stopping.
        min_delta:   Minimum change to qualify as improvement.
    """

    def __init__(
        self,
        metric_name: str = "q1_accuracy",
        mode: str = "max",
        patience: int = 5,
        min_delta: float = 1e-4,
    ):
        self.metric_name = metric_name
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("-inf") if mode == "max" else float("inf")
        self.counter = 0
        self.should_stop = False

    def on_epoch_end(self, trainer: Any, epoch: int, metrics: Dict[str, float]) -> None:
        value = metrics.get(self.metric_name)
        if value is None:
            return

        if self.mode == "max":
            improved = value > self.best + self.min_delta
        else:
            improved = value < self.best - self.min_delta

        if improved:
            self.best = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                logger.info(
                    f"Early stopping: {self.metric_name} not improved for "
                    f"{self.patience} epochs (best={self.best:.4f})"
                )


class LRScheduleCallback(Callback):
    """Log learning rate at each step."""

    def on_step(self, trainer: Any, step: int, metrics: Dict[str, float]) -> None:
        if hasattr(trainer, "optimizer"):
            lr = trainer.optimizer.param_groups[0]["lr"]
            metrics["lr"] = lr


class CallbackRunner:
    """Runs a list of callbacks."""

    def __init__(self, callbacks: Optional[List[Callback]] = None):
        self.callbacks = callbacks or []

    def on_train_start(self, trainer: Any) -> None:
        for cb in self.callbacks:
            cb.on_train_start(trainer)

    def on_train_end(self, trainer: Any) -> None:
        for cb in self.callbacks:
            cb.on_train_end(trainer)

    def on_epoch_start(self, trainer: Any, epoch: int) -> None:
        for cb in self.callbacks:
            cb.on_epoch_start(trainer, epoch)

    def on_epoch_end(self, trainer: Any, epoch: int, metrics: Dict[str, float]) -> None:
        for cb in self.callbacks:
            cb.on_epoch_end(trainer, epoch, metrics)

    def on_step(self, trainer: Any, step: int, metrics: Dict[str, float]) -> None:
        for cb in self.callbacks:
            cb.on_step(trainer, step, metrics)

    @property
    def should_stop(self) -> bool:
        return any(
            getattr(cb, "should_stop", False)
            for cb in self.callbacks
        )
