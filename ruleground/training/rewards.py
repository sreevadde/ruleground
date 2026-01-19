"""
Reward Functions for GRPO

Multi-component reward: R = R_correct + alpha*R_faithful + beta*R_consistent

Paper Section 5.5: reward combines correctness, faithfulness (predicates
support the conclusion), and consistency (rules align with task).
"""

from __future__ import annotations

from typing import Dict

import torch
from torch import Tensor


class RewardFunction:
    """Multi-component reward for GRPO training.

    Components:
        R_correct:    +1 if task prediction matches label, 0 otherwise.
        R_faithful:   Measures whether active predicates support the decision.
        R_consistent: Measures alignment between rule scores and task output.

    Args:
        alpha: Weight for faithfulness reward.
        beta:  Weight for consistency reward.
    """

    def __init__(self, alpha: float = 0.3, beta: float = 0.2):
        self.alpha = alpha
        self.beta = beta

    def compute_correctness(
        self,
        q1_logits: Tensor,
        q2_logits: Tensor,
        q1_labels: Tensor,
        q2_labels: Tensor,
    ) -> Tensor:
        """Binary correctness reward.

        Returns:
            [B] tensor of 0/1 rewards.
        """
        q1_correct = (q1_logits.argmax(dim=-1) == (q1_labels > 0).long()).float()
        q2_correct = (q2_logits.argmax(dim=-1) == q2_labels).float()
        # Weight Q1 more heavily (primary task)
        return 0.6 * q1_correct + 0.4 * q2_correct

    def compute_faithfulness(
        self,
        predicate_probs: Tensor,
        q1_logits: Tensor,
    ) -> Tensor:
        """Faithfulness reward: predicates should support the conclusion.

        For infraction predictions, at least some predicates should be active.
        For non-infraction predictions, predicate evidence should be low.

        Returns:
            [B] tensor of faithfulness scores in [0, 1].
        """
        pred_is_infraction = (q1_logits.argmax(dim=-1) > 0).float()  # [B]
        max_pred = predicate_probs.max(dim=-1).values  # [B]
        mean_pred = predicate_probs.mean(dim=-1)  # [B]

        # Infraction => want high predicate evidence
        # Non-infraction => want low predicate evidence
        faith_infraction = max_pred * pred_is_infraction
        faith_clean = (1 - mean_pred) * (1 - pred_is_infraction)

        return faith_infraction + faith_clean

    def compute_consistency(
        self,
        rule_scores: Dict[str, Tensor],
        q1_labels: Tensor,
    ) -> Tensor:
        """Consistency reward: rules should align with ground truth.

        Returns:
            [B] tensor of consistency scores in [0, 1].
        """
        if not rule_scores:
            return torch.zeros_like(q1_labels, dtype=torch.float)

        stacked = torch.stack(list(rule_scores.values()), dim=-1)  # [B, R]
        max_rule = stacked.max(dim=-1).values  # [B]

        is_infraction = (q1_labels > 0).float()

        # Reward: rule fires when there's an infraction, stays low otherwise
        aligned = max_rule * is_infraction + (1 - max_rule) * (1 - is_infraction)
        return aligned

    def __call__(
        self,
        outputs: Dict[str, Tensor],
        targets: Dict[str, Tensor],
    ) -> Tensor:
        """Compute total reward.

        Args:
            outputs: Model output dict.
            targets: Target dict.

        Returns:
            [B] tensor of total rewards.
        """
        r_correct = self.compute_correctness(
            outputs["q1_logits"],
            outputs["q2_logits"],
            targets["q1_labels"],
            targets["q2_labels"],
        )
        r_faithful = self.compute_faithfulness(
            outputs["predicate_probs"],
            outputs["q1_logits"],
        )
        r_consistent = self.compute_consistency(
            outputs.get("rule_scores", {}),
            targets["q1_labels"],
        )

        return r_correct + self.alpha * r_faithful + self.beta * r_consistent
