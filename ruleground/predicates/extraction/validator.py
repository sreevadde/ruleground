"""
Cross-Model Predicate Validation

Validates extraction quality by comparing outputs from two LLMs.
Paper reports 88.3% cross-model agreement, Cohen's kappa = 0.76.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from ruleground.predicates.extraction.extractor import (
    PredicateExtraction,
    PredicateExtractor,
)
from ruleground.predicates.extraction.prompts import build_validation_prompt

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of cross-model validation for one sample."""

    video_id: str
    agreement: float = 0.0  # fraction of predicates that agree
    primary: Optional[PredicateExtraction] = None
    secondary: Optional[PredicateExtraction] = None
    merged_predicates: Dict[str, bool] = field(default_factory=dict)
    merged_confidences: Dict[str, float] = field(default_factory=dict)


class CrossModelValidator:
    """Validate extractions across two LLM backends.

    Uses a primary extractor and a secondary extractor, then merges
    results based on agreement. Predicates where both models agree
    get full confidence; disagreements get downweighted.

    Args:
        primary:   Primary PredicateExtractor (e.g., Claude).
        secondary: Secondary PredicateExtractor (e.g., GPT-4o).
        agreement_threshold: Minimum agreement to keep a predicate.
        agree_boost:   Confidence multiplier when models agree.
        disagree_penalty: Confidence multiplier when models disagree.
    """

    def __init__(
        self,
        primary: PredicateExtractor,
        secondary: PredicateExtractor,
        agreement_threshold: float = 0.5,
        agree_boost: float = 1.0,
        disagree_penalty: float = 0.5,
    ):
        self.primary = primary
        self.secondary = secondary
        self.agreement_threshold = agreement_threshold
        self.agree_boost = agree_boost
        self.disagree_penalty = disagree_penalty

    def validate_single(
        self,
        rationale: str,
        sport: str,
        video_id: str = "",
    ) -> ValidationResult:
        """Extract from both models and compare."""
        ext_a = self.primary.extract(rationale, sport, video_id)
        ext_b = self.secondary.extract(rationale, sport, video_id)

        # Compute agreement
        all_preds = set(ext_a.predicates.keys()) | set(ext_b.predicates.keys())
        if not all_preds:
            return ValidationResult(video_id=video_id, primary=ext_a, secondary=ext_b)

        agree_count = 0
        merged_preds = {}
        merged_confs = {}

        for name in all_preds:
            a_val = ext_a.predicates.get(name)
            b_val = ext_b.predicates.get(name)
            a_conf = ext_a.confidences.get(name, 0.0)
            b_conf = ext_b.confidences.get(name, 0.0)

            if a_val is not None and b_val is not None:
                if a_val == b_val:
                    # Both agree on value
                    agree_count += 1
                    merged_preds[name] = a_val
                    merged_confs[name] = min(
                        max(a_conf, b_conf) * self.agree_boost, 1.0
                    )
                else:
                    # Disagree: take higher confidence, penalize
                    if a_conf >= b_conf:
                        merged_preds[name] = a_val
                        merged_confs[name] = a_conf * self.disagree_penalty
                    else:
                        merged_preds[name] = b_val
                        merged_confs[name] = b_conf * self.disagree_penalty
            elif a_val is not None:
                merged_preds[name] = a_val
                merged_confs[name] = a_conf * self.disagree_penalty
            else:
                merged_preds[name] = b_val
                merged_confs[name] = b_conf * self.disagree_penalty

        agreement = agree_count / len(all_preds) if all_preds else 0.0

        return ValidationResult(
            video_id=video_id,
            agreement=agreement,
            primary=ext_a,
            secondary=ext_b,
            merged_predicates=merged_preds,
            merged_confidences=merged_confs,
        )

    def validate_batch(
        self,
        items: List[Dict[str, str]],
    ) -> Tuple[List[ValidationResult], Dict[str, float]]:
        """Validate a batch and compute aggregate statistics.

        Returns:
            Tuple of (results, stats_dict).
        """
        results = []
        for item in items:
            result = self.validate_single(
                rationale=item["rationale"],
                sport=item["sport"],
                video_id=item.get("video_id", ""),
            )
            results.append(result)

        # Aggregate stats
        agreements = [r.agreement for r in results if r.primary and r.secondary]
        stats = {
            "mean_agreement": float(np.mean(agreements)) if agreements else 0.0,
            "std_agreement": float(np.std(agreements)) if agreements else 0.0,
            "num_samples": len(results),
            "num_validated": len(agreements),
        }

        # Cohen's kappa approximation
        if agreements:
            stats["cohens_kappa"] = self._compute_kappa(results)

        return results, stats

    def _compute_kappa(self, results: List[ValidationResult]) -> float:
        """Compute Cohen's kappa across all predicate decisions."""
        agree = 0
        total = 0
        p_a_pos = 0
        p_b_pos = 0

        for r in results:
            if not r.primary or not r.secondary:
                continue
            all_names = set(r.primary.predicates.keys()) & set(
                r.secondary.predicates.keys()
            )
            for name in all_names:
                total += 1
                a = r.primary.predicates[name]
                b = r.secondary.predicates[name]
                if a == b:
                    agree += 1
                if a:
                    p_a_pos += 1
                if b:
                    p_b_pos += 1

        if total == 0:
            return 0.0

        po = agree / total
        pa_pos = p_a_pos / total
        pb_pos = p_b_pos / total
        pe = pa_pos * pb_pos + (1 - pa_pos) * (1 - pb_pos)

        if pe >= 1.0:
            return 1.0
        return (po - pe) / (1 - pe)
