"""
Training CLI

Orchestrates the 3-stage training pipeline:
    Stage 1: Supervised pre-training
    Stage 2: GRPO post-training
    Stage 3: RSA risk alignment
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import torch

from ruleground.utils.config import load_config, DEFAULT_CONFIG
from ruleground.utils.logging import setup_logging
from ruleground.utils.checkpoint import load_checkpoint

logger = logging.getLogger(__name__)


def run_training(
    config_path: str,
    output_dir: str = "experiments",
    num_gpus: int = 1,
    resume_path: Optional[str] = None,
    overrides: Optional[List[str]] = None,
) -> None:
    """Run the full training pipeline.

    Args:
        config_path: Path to YAML config.
        output_dir:  Output directory for checkpoints and logs.
        num_gpus:    Number of GPUs.
        resume_path: Optional checkpoint to resume from.
        overrides:   Config overrides.
    """
    setup_logging(log_file=f"{output_dir}/train.log")

    # Load config
    config = load_config(config_path, overrides)
    logger.info(f"Config loaded from {config_path}")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Build model
    from ruleground.models.ruleground import RuleGround
    model = RuleGround.from_config(config).to(device)
    logger.info(f"Model built: {sum(p.numel() for p in model.parameters())} parameters")

    # Build data loaders
    from ruleground.data.dataset import build_dataloader
    from ruleground.data.transforms import get_train_transform, get_eval_transform

    train_transform = get_train_transform(
        tuple(config.data.frame_size), config.data.num_frames
    )
    val_transform = get_eval_transform(
        tuple(config.data.frame_size), config.data.num_frames
    )

    train_loader = build_dataloader(
        root=config.data.root,
        split="train",
        num_frames=config.data.num_frames,
        batch_size=config.training.batch_size,
        num_workers=config.data.num_workers,
        transform=train_transform,
    )
    val_loader = build_dataloader(
        root=config.data.root,
        split="val",
        num_frames=config.data.num_frames,
        batch_size=config.training.batch_size,
        num_workers=config.data.num_workers,
        transform=val_transform,
    )

    # Resume if specified
    start_epoch = 0
    if resume_path:
        info = load_checkpoint(resume_path, model)
        start_epoch = info["epoch"]
        logger.info(f"Resumed from epoch {start_epoch}")

    # ---- Stage 1: Supervised Pre-training ----
    logger.info("=" * 60)
    logger.info("Stage 1: Supervised Pre-training")
    logger.info("=" * 60)

    from ruleground.training.trainer import SupervisedTrainer
    from ruleground.training.losses import RuleGroundLoss

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.lr,
        weight_decay=config.training.weight_decay,
    )
    loss_fn = RuleGroundLoss(
        gamma=config.training.gamma,
        delta=config.training.delta,
    )

    supervised_trainer = SupervisedTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        max_epochs=config.training.pretrain_epochs,
        output_dir=f"{output_dir}/stage1_supervised",
    )
    supervised_metrics = supervised_trainer.train()
    logger.info(f"Stage 1 complete. Best: {supervised_metrics}")

    # ---- Stage 2: GRPO Post-training ----
    if config.training.use_grpo:
        logger.info("=" * 60)
        logger.info("Stage 2: GRPO Post-training")
        logger.info("=" * 60)

        from ruleground.training.grpo import GRPOTrainer
        from ruleground.training.rewards import RewardFunction

        grpo_trainer = GRPOTrainer(
            model=model,
            reward_fn=RewardFunction(),
            group_size=config.grpo.group_size,
            clip_ratio=config.grpo.clip_ratio,
            kl_coef=config.grpo.kl_coef,
            pred_dropout=config.grpo.predicate_dropout,
            lr=config.training.lr * 0.1,  # Lower LR for fine-tuning
        )

        for epoch in range(1, config.training.grpo_epochs + 1):
            epoch_loss = 0.0
            for batch in train_loader:
                batch = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                loss, metrics = grpo_trainer.step(batch)
                epoch_loss += loss.item()
            logger.info(f"GRPO Epoch {epoch} | Loss: {epoch_loss / len(train_loader):.4f}")

    # ---- Stage 3: RSA Risk Alignment ----
    if config.training.use_rsa:
        logger.info("=" * 60)
        logger.info("Stage 3: RSA Risk Alignment")
        logger.info("=" * 60)

        from ruleground.training.rsa import RSAConstraint, RSATrainer

        rsa_constraint = RSAConstraint(
            alpha=config.rsa.alpha,
            lambda_risk=config.rsa.lambda_risk,
            fp_penalty=config.rsa.fp_penalty,
        )

        # Re-create GRPO if not already
        if not config.training.use_grpo:
            from ruleground.training.grpo import GRPOTrainer
            grpo_trainer = GRPOTrainer(model=model)

        rsa_trainer = RSATrainer(grpo_trainer, rsa_constraint)

        for epoch in range(1, config.training.rsa_epochs + 1):
            epoch_loss = 0.0
            for batch in train_loader:
                batch = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                loss, metrics = rsa_trainer.step(batch)
                epoch_loss += loss.item()
            logger.info(f"RSA Epoch {epoch} | Loss: {epoch_loss / len(train_loader):.4f}")

    logger.info("Training complete!")


def main():
    """Entry point for rg-train command."""
    import sys
    if len(sys.argv) < 3:
        print("Usage: rg-train -c <config.yaml> [-o output_dir] [-g gpus]")
        sys.exit(1)
    run_training(config_path=sys.argv[2])
