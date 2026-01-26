"""
Loss Functions for RuleGround

Three-component supervised loss (Paper Section 5.3):
    L = L_task + gamma * L_pred + delta * L_cons

- L_task: Cross-entropy on Q1/Q2 task labels
- L_pred: Weighted BCE on extracted predicate labels (weak supervision)
- L_cons: Rule consistency loss (composed rule scores match task labels)
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class WeightedBCELoss(nn.Module):
    """Binary cross-entropy with confidence weighting for weak supervision.

    Handles per-predicate confidence weights from the extraction pipeline
    and sport-conditional masking (unextracted predicates get mask=0).
    """

    def forward(
        self,
        logits: Tensor,
        targets: Tensor,
        weights: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            logits:  [B, K] predicate logits.
            targets: [B, K] binary targets (0/1).
            weights: [B, K] confidence weights from extraction.
            mask:    [B, K] binary mask (1 = has label, 0 = unlabeled).

        Returns:
            Scalar loss.
        """
        # Mask out -inf logits from sport-conditional masking before BCE
        valid = torch.isfinite(logits)
        safe_logits = logits.clone()
        safe_logits[~valid] = 0.0

        loss = F.binary_cross_entropy_with_logits(safe_logits, targets, reduction="none")
        loss = loss * valid.float()  # Zero out loss for masked predicates

        if weights is not None:
            loss = loss * weights
        if mask is not None:
            loss = loss * mask
            valid = valid & (mask > 0)

        denom = valid.float().sum().clamp(min=1.0)
        return loss.sum() / denom


class RuleConsistencyLoss(nn.Module):
    """Ensures composed rule scores align with task predictions.

    For infraction samples (q1_label > 0), at least one rule should fire.
    For non-infraction samples, no rule should fire strongly.
    """

    def forward(
        self,
        rule_scores: Dict[str, Tensor],
        q1_labels: Tensor,
    ) -> Tensor:
        """
        Args:
            rule_scores: Dict of rule_name -> [B] scores in [0, 1].
            q1_labels:   [B] ground truth Q1 labels.

        Returns:
            Scalar consistency loss.
        """
        if not rule_scores:
            return torch.tensor(0.0, requires_grad=True)

        # Max rule activation across all rules: [B]
        stacked = torch.stack(list(rule_scores.values()), dim=-1)
        max_rule = stacked.max(dim=-1).values

        # Target: should be high (1) for infractions, low (0) for clean plays
        infraction = (q1_labels > 0).float()

        return F.binary_cross_entropy(
            max_rule.clamp(1e-7, 1 - 1e-7), infraction
        )


class Q5TemporalLoss(nn.Module):
    """Loss for Q5 temporal grounding predictions.

    Smooth L1 loss on predicted (start, end) spans vs ground truth.
    Only applied to samples that have temporal annotations.
    """

    def forward(
        self,
        pred_spans: Tensor,
        gt_spans: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            pred_spans: [B, 2] predicted (start, end) in [0, 1].
            gt_spans:   [B, 2] ground truth (start, end) in [0, 1].
            mask:       [B] binary mask (1 = has Q5 annotation).

        Returns:
            Scalar loss.
        """
        loss = F.smooth_l1_loss(pred_spans, gt_spans, reduction="none")  # [B, 2]
        loss = loss.mean(dim=-1)  # [B]

        if mask is not None:
            loss = loss * mask
            denom = mask.sum().clamp(min=1.0)
            return loss.sum() / denom

        return loss.mean()


class RuleGroundLoss(nn.Module):
    """Combined multi-task loss for RuleGround supervised training.

    L = L_q1 + L_q2 + eta * L_q5 + gamma * L_pred + delta * L_cons

    Args:
        gamma: Weight for predicate supervision loss.
        delta: Weight for rule consistency loss.
        eta:   Weight for Q5 temporal grounding loss.
        q2_weight: Weight for Q2 classification loss (downweight if class imbalanced).
    """

    def __init__(
        self,
        gamma: float = 0.5,
        delta: float = 0.1,
        eta: float = 0.5,
        q2_weight: float = 1.0,
    ):
        super().__init__()
        self.gamma = gamma
        self.delta = delta
        self.eta = eta
        self.q2_weight = q2_weight

        self.q1_loss = nn.CrossEntropyLoss()
        self.q2_loss = nn.CrossEntropyLoss()
        self.pred_loss = WeightedBCELoss()
        self.cons_loss = RuleConsistencyLoss()
        self.q5_loss = Q5TemporalLoss()

    def forward(
        self,
        outputs: Dict[str, Tensor],
        targets: Dict[str, Tensor],
    ) -> Tuple[Tensor, Dict[str, float]]:
        """
        Args:
            outputs: Model output dict.
            targets: Target dict with q1_labels, q2_labels, predicate_labels, etc.

        Returns:
            (total_loss, loss_breakdown_dict)
        """
        breakdown = {}

        # Q1: Infraction identification
        l_q1 = self.q1_loss(outputs["q1_logits"], targets["q1_labels"])
        breakdown["loss_q1"] = l_q1.item()

        # Q2: Foul classification
        l_q2 = self.q2_loss(outputs["q2_logits"], targets["q2_labels"])
        breakdown["loss_q2"] = l_q2.item()

        total = l_q1 + self.q2_weight * l_q2

        # Q5: Temporal grounding (if annotations available)
        if "q5_spans" in targets and "q5_preds" in outputs:
            q5_mask = targets.get("q5_mask")
            l_q5 = self.q5_loss(outputs["q5_preds"], targets["q5_spans"], q5_mask)
            breakdown["loss_q5"] = l_q5.item()
            total = total + self.eta * l_q5

        # Predicate supervision (weak labels from extraction)
        if "predicate_labels" in targets and "predicate_logits" in outputs:
            l_pred = self.pred_loss(
                outputs["predicate_logits"],
                targets["predicate_labels"],
                targets.get("predicate_weights"),
                targets.get("predicate_mask"),
            )
            breakdown["loss_pred"] = l_pred.item()
            total = total + self.gamma * l_pred

        # Rule consistency
        if "rule_scores" in outputs:
            l_cons = self.cons_loss(outputs["rule_scores"], targets["q1_labels"])
            breakdown["loss_cons"] = l_cons.item()
            total = total + self.delta * l_cons

        breakdown["loss_total"] = total.item()
        return total, breakdown
