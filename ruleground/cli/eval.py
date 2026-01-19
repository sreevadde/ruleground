"""
Evaluation CLI

Run evaluation on a trained model checkpoint.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import torch

from ruleground.utils.config import load_config
from ruleground.utils.logging import setup_logging

logger = logging.getLogger(__name__)


def run_evaluation(
    checkpoint_path: str,
    config_path: str,
    split: str = "test",
    output_path: Optional[str] = None,
    per_sport: bool = True,
) -> None:
    """Run model evaluation.

    Args:
        checkpoint_path: Path to model checkpoint.
        config_path:     Path to config YAML.
        split:           Dataset split to evaluate.
        output_path:     Where to save results JSON.
        per_sport:       Whether to compute per-sport breakdowns.
    """
    setup_logging()

    config = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build and load model
    from ruleground.models.ruleground import RuleGround
    from ruleground.utils.checkpoint import load_checkpoint

    model = RuleGround.from_config(config).to(device)
    load_checkpoint(checkpoint_path, model)
    model.eval()

    # Build data loader
    from ruleground.data.dataset import build_dataloader
    from ruleground.data.transforms import get_eval_transform

    transform = get_eval_transform(
        tuple(config.data.frame_size), config.data.num_frames
    )
    dataloader = build_dataloader(
        root=config.data.root,
        split=split,
        num_frames=config.data.num_frames,
        batch_size=config.training.batch_size,
        num_workers=config.data.num_workers,
        transform=transform,
    )

    # Run evaluation
    from ruleground.evaluation.evaluator import Evaluator
    from ruleground.evaluation.error_analysis import ErrorAnalyzer

    evaluator = Evaluator(model, device)
    results = evaluator.run(dataloader, per_sport=per_sport)

    # Print results
    logger.info("=" * 60)
    logger.info(f"Evaluation Results ({split})")
    logger.info("=" * 60)
    for k, v in results["overall"].items():
        logger.info(f"  {k}: {v:.4f}")

    if per_sport and "per_sport" in results:
        for sport, metrics in results["per_sport"].items():
            logger.info(f"\n  {sport.upper()}:")
            for k, v in metrics.items():
                logger.info(f"    {k}: {v:.4f}")

    # Error analysis
    preds = results.get("predictions", {})
    if preds.get("predicate_probs") is not None and preds.get("q1_preds") is not None:
        analyzer = ErrorAnalyzer()
        # Note: q1_labels would come from targets; simplified for CLI
        logger.info("\nError analysis available via Python API")

    # Save results
    if output_path:
        evaluator.save_results(results, output_path)
    else:
        default_path = Path(checkpoint_path).parent / f"eval_{split}.json"
        evaluator.save_results(results, default_path)


def main():
    """Entry point for rg-eval command."""
    import sys
    if len(sys.argv) < 3:
        print("Usage: rg-eval -ckpt <checkpoint> -c <config.yaml> [-s split]")
        sys.exit(1)
    # Simple arg parsing
    args = {}
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] in ("-ckpt", "--checkpoint"):
            args["checkpoint_path"] = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] in ("-c", "--config"):
            args["config_path"] = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] in ("-s", "--split"):
            args["split"] = sys.argv[i + 1]
            i += 2
        else:
            i += 1
    run_evaluation(**args)
