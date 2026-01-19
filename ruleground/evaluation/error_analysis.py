"""
Error Analysis and Attribution

Categorizes model errors into perception, grounding, and reasoning
failures per the Perceptual Gap taxonomy (Paper Section 3).

Error taxonomy:
    - Perception error:  Low relevant predicate activation when evidence exists
    - Grounding error:   Predicates active but wrong classification of rule status
    - Reasoning error:   Correct predicates but wrong task output
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import Tensor

from ruleground.predicates.ontology import (
    ALL_PREDICATE_NAMES,
    get_predicate_names_for_sport,
    SPORT_FROM_STR,
    ID_TO_SPORT,
)

logger = logging.getLogger(__name__)


class ErrorType:
    PERCEPTION = "perception"
    GROUNDING = "grounding"
    REASONING = "reasoning"
    CORRECT = "correct"


class ErrorAnalyzer:
    """Analyze and attribute model errors.

    For each incorrect prediction, determines whether the error is
    due to perception (failure to detect evidence), grounding
    (misclassifying rule-relevant status), or reasoning (correct
    predicates but wrong task conclusion).

    Args:
        predicate_threshold: Threshold for considering a predicate "active".
        evidence_threshold:  Minimum predicate evidence to rule out perception error.
    """

    def __init__(
        self,
        predicate_threshold: float = 0.5,
        evidence_threshold: float = 0.3,
    ):
        self.predicate_threshold = predicate_threshold
        self.evidence_threshold = evidence_threshold

    def classify_error(
        self,
        pred_probs: Tensor,
        gt_predicates: Optional[Tensor],
        q1_pred: int,
        q1_label: int,
        sport_id: int,
    ) -> str:
        """Classify a single prediction error.

        Args:
            pred_probs:    [K] predicted predicate probabilities.
            gt_predicates: [K] ground truth predicate labels (or None).
            q1_pred:       Predicted Q1 class.
            q1_label:      Ground truth Q1 label.
            sport_id:      Sport integer ID.

        Returns:
            Error type string.
        """
        if q1_pred == (1 if q1_label > 0 else 0):
            return ErrorType.CORRECT

        # Get sport-relevant predicates
        sport_enum = ID_TO_SPORT.get(sport_id)
        if sport_enum is None:
            return ErrorType.REASONING

        sport_names = set(get_predicate_names_for_sport(sport_enum))
        sport_indices = [
            i for i, n in enumerate(ALL_PREDICATE_NAMES) if n in sport_names
        ]

        relevant_probs = pred_probs[sport_indices]
        max_activation = relevant_probs.max().item()
        mean_activation = relevant_probs.mean().item()

        # Check if predicates have ground truth
        if gt_predicates is not None:
            gt_relevant = gt_predicates[sport_indices]
            gt_active = (gt_relevant > 0.5).any().item()
            pred_active = (relevant_probs > self.predicate_threshold).any().item()

            if gt_active and not pred_active:
                # Evidence exists but predicates didn't detect it
                return ErrorType.PERCEPTION

            if gt_active and pred_active:
                # Predicates detected something but classification failed
                # Check if predicate values match GT
                pred_binary = (relevant_probs > self.predicate_threshold).float()
                gt_binary = (gt_relevant > 0.5).float()
                agreement = (pred_binary == gt_binary).float().mean().item()

                if agreement < 0.7:
                    return ErrorType.GROUNDING
                else:
                    return ErrorType.REASONING

        # Without GT predicates, use heuristics
        if max_activation < self.evidence_threshold:
            return ErrorType.PERCEPTION
        elif mean_activation < self.predicate_threshold:
            return ErrorType.GROUNDING
        else:
            return ErrorType.REASONING

    def analyze_batch(
        self,
        predicate_probs: Tensor,
        q1_preds: Tensor,
        q1_labels: Tensor,
        sport_ids: Tensor,
        gt_predicates: Optional[Tensor] = None,
    ) -> Dict[str, Any]:
        """Analyze a batch of predictions.

        Args:
            predicate_probs: [B, K] predicted predicate probabilities.
            q1_preds:        [B] predicted Q1 classes.
            q1_labels:       [B] ground truth Q1 labels.
            sport_ids:       [B] sport integer IDs.
            gt_predicates:   [B, K] ground truth predicate labels (optional).

        Returns:
            Dict with error counts, rates, and per-sample attributions.
        """
        B = q1_preds.shape[0]
        errors = []
        counts = Counter()

        for i in range(B):
            gt_pred_i = gt_predicates[i] if gt_predicates is not None else None
            error_type = self.classify_error(
                predicate_probs[i],
                gt_pred_i,
                q1_preds[i].item(),
                q1_labels[i].item(),
                sport_ids[i].item(),
            )
            errors.append(error_type)
            counts[error_type] += 1

        total_errors = sum(
            v for k, v in counts.items() if k != ErrorType.CORRECT
        )

        result = {
            "error_counts": dict(counts),
            "total_errors": total_errors,
            "total_samples": B,
            "accuracy": counts[ErrorType.CORRECT] / B if B > 0 else 0.0,
            "per_sample": errors,
        }

        # Error rate breakdown (among incorrect predictions)
        if total_errors > 0:
            result["error_rates"] = {
                ErrorType.PERCEPTION: counts[ErrorType.PERCEPTION] / total_errors,
                ErrorType.GROUNDING: counts[ErrorType.GROUNDING] / total_errors,
                ErrorType.REASONING: counts[ErrorType.REASONING] / total_errors,
            }
        else:
            result["error_rates"] = {
                ErrorType.PERCEPTION: 0.0,
                ErrorType.GROUNDING: 0.0,
                ErrorType.REASONING: 0.0,
            }

        return result

    def format_report(self, analysis: Dict[str, Any]) -> str:
        """Format error analysis as a readable report."""
        lines = [
            "=" * 60,
            "RuleGround Error Analysis Report",
            "=" * 60,
            f"Total samples:  {analysis['total_samples']}",
            f"Correct:        {analysis['error_counts'].get(ErrorType.CORRECT, 0)}",
            f"Total errors:   {analysis['total_errors']}",
            f"Accuracy:       {analysis['accuracy']:.2%}",
            "",
            "Error Breakdown (among errors):",
            f"  Perception:  {analysis['error_rates'][ErrorType.PERCEPTION]:.1%}"
            f"  (failed to detect visual evidence)",
            f"  Grounding:   {analysis['error_rates'][ErrorType.GROUNDING]:.1%}"
            f"  (detected evidence but misclassified rule status)",
            f"  Reasoning:   {analysis['error_rates'][ErrorType.REASONING]:.1%}"
            f"  (correct predicates but wrong task conclusion)",
            "=" * 60,
        ]

        # Paper comparison
        lines.extend([
            "",
            "Paper Reference (GPT-4o on SportR):",
            "  Perception errors: ~38%",
            "  Grounding errors:  ~62%  <-- RuleGround targets this",
        ])

        return "\n".join(lines)
