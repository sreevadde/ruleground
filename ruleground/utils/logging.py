"""
Logging Utilities

Configures structured logging for training, evaluation, and extraction.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    name: str = "ruleground",
) -> logging.Logger:
    """Configure logging for RuleGround.

    Args:
        level:    Logging level string.
        log_file: Optional path to write logs to disk.
        name:     Logger name.

    Returns:
        Configured logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    # Prevent duplicate handlers on repeated calls
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)

    # File handler
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


class MetricLogger:
    """Accumulates and reports metrics during training."""

    def __init__(self):
        self._meters: dict[str, list[float]] = {}

    def update(self, metrics: dict[str, float]) -> None:
        for k, v in metrics.items():
            if k not in self._meters:
                self._meters[k] = []
            self._meters[k].append(v)

    def average(self) -> dict[str, float]:
        return {k: sum(v) / len(v) for k, v in self._meters.items() if v}

    def reset(self) -> None:
        self._meters.clear()

    def __repr__(self) -> str:
        avg = self.average()
        parts = [f"{k}={v:.4f}" for k, v in avg.items()]
        return " | ".join(parts)
