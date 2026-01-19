"""
Full Evaluation Pipeline

Runs model inference on a dataset split and computes all metrics.
Supports per-sport breakdown and error attribution.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ruleground.evaluation.metrics import compute_all_metrics
from ruleground.predicates.ontology import ID_TO_SPORT, SPORT_TO_ID

logger = logging.getLogger(__name__)


class Evaluator:
    """Full evaluation pipeline for RuleGround.

    Runs inference, computes Q1/Q2/Q5 metrics, FP analysis,
    predicate accuracy, and per-sport breakdowns.

    Args:
        model:    Trained RuleGround model.
        device:   Device for inference.
    """

    def __init__(
        self,
        model: nn.Module,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.device = device or next(model.parameters()).device
        self.model.eval()

    @torch.no_grad()
    def run(
        self,
        dataloader: DataLoader,
        per_sport: bool = True,
    ) -> Dict[str, Any]:
        """Run full evaluation.

        Args:
            dataloader: Evaluation DataLoader.
            per_sport:  Whether to compute per-sport breakdowns.

        Returns:
            Dict with 'overall' metrics and optionally 'per_sport' breakdowns.
        """
        all_outputs = {
            "q1_logits": [], "q2_logits": [], "q5_preds": [],
            "predicate_probs": [], "predicate_logits": [],
        }
        all_targets = {
            "q1_labels": [], "q2_labels": [], "q5_spans": [],
            "predicate_labels": [], "predicate_mask": [],
        }
        all_sport_ids = []
        all_video_ids = []

        for batch in dataloader:
            batch_device = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            outputs = self.model(
                video=batch_device["video"],
                mask=batch_device.get("mask"),
                sport_ids=batch_device.get("sport_ids"),
            )

            # Collect outputs
            for k in all_outputs:
                if k in outputs:
                    all_outputs[k].append(outputs[k].cpu())

            for k in all_targets:
                if k in batch:
                    if isinstance(batch[k], torch.Tensor):
                        all_targets[k].append(batch[k].cpu())

            if "sport_ids" in batch:
                all_sport_ids.append(batch["sport_ids"])
            if "video_ids" in batch:
                all_video_ids.extend(batch["video_ids"])

        # Concatenate
        cat_out = {k: torch.cat(v) for k, v in all_outputs.items() if v}
        cat_tgt = {k: torch.cat(v) for k, v in all_targets.items() if v}
        sport_ids = torch.cat(all_sport_ids) if all_sport_ids else None

        # Overall metrics
        overall = compute_all_metrics(cat_out, cat_tgt)

        result: Dict[str, Any] = {"overall": overall}

        # Per-sport breakdown
        if per_sport and sport_ids is not None:
            sport_metrics = {}
            for sport_id_val, sport_enum in ID_TO_SPORT.items():
                mask = sport_ids == sport_id_val
                if mask.sum() == 0:
                    continue
                sport_out = {k: v[mask] for k, v in cat_out.items()}
                sport_tgt = {k: v[mask] for k, v in cat_tgt.items()}
                sport_metrics[sport_enum.name.lower()] = compute_all_metrics(
                    sport_out, sport_tgt
                )
            result["per_sport"] = sport_metrics

        # Store predictions for error analysis
        result["predictions"] = {
            "q1_preds": cat_out.get("q1_logits", torch.tensor([])).argmax(dim=-1)
            if "q1_logits" in cat_out else None,
            "q2_preds": cat_out.get("q2_logits", torch.tensor([])).argmax(dim=-1)
            if "q2_logits" in cat_out else None,
            "predicate_probs": cat_out.get("predicate_probs"),
            "video_ids": all_video_ids,
        }

        return result

    def save_results(self, results: Dict[str, Any], path: str | Path) -> None:
        """Save evaluation results to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Convert tensors to lists for JSON serialization
        serializable = {}
        for k, v in results.items():
            if k == "predictions":
                continue  # Skip raw tensor predictions
            if isinstance(v, dict):
                serializable[k] = {
                    sk: sv if not isinstance(sv, torch.Tensor) else sv.tolist()
                    for sk, sv in v.items()
                }
            else:
                serializable[k] = v

        with open(path, "w") as f:
            json.dump(serializable, f, indent=2)
        logger.info(f"Results saved to {path}")
