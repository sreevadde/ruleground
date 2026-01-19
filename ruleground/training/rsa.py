"""
Risk-aware Stepwise Alignment (RSA) -- Stage 3

Paper Section 5.6: CVaR penalty on false-positive infractions.
Targets the worst alpha fraction of FP risk to suppress overconfident
false calls while maintaining recall.

L_RSA = L_GRPO + lambda * CVaR_alpha[1{false_positive} * c_penalty]

Result: 34% false-positive reduction at matched recall.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)


class RSAConstraint:
    """CVaR penalty on false-positive infraction calls.

    Uses Conditional Value at Risk (CVaR) to penalize the worst-case
    false positives, focusing on the tail of the FP risk distribution.

    Args:
        alpha:       CVaR quantile (0.1 = focus on worst 10%).
        lambda_risk: Weight of the CVaR penalty in total loss.
        fp_penalty:  Base penalty multiplier for false positives.
    """

    def __init__(
        self,
        alpha: float = 0.1,
        lambda_risk: float = 0.3,
        fp_penalty: float = 2.0,
    ):
        self.alpha = alpha
        self.lambda_risk = lambda_risk
        self.fp_penalty = fp_penalty

    def compute_cvar(self, losses: Tensor) -> Tensor:
        """Compute Conditional Value at Risk.

        CVaR_alpha = E[X | X >= VaR_alpha]
        Focus on the worst alpha fraction of the loss distribution.

        Args:
            losses: [N] per-sample losses.

        Returns:
            Scalar CVaR estimate.
        """
        if losses.numel() == 0:
            return torch.tensor(0.0, device=losses.device, requires_grad=True)

        sorted_losses, _ = losses.sort(descending=True)
        k = max(1, int(self.alpha * len(sorted_losses)))
        return sorted_losses[:k].mean()

    def compute_fp_risk(
        self,
        outputs: Dict[str, Tensor],
        labels: Tensor,
    ) -> Tensor:
        """Compute per-sample false positive risk.

        Risk = P(infraction) * (1 - predicate_evidence) * is_negative * penalty

        High-confidence infraction predictions on non-infraction samples
        with low predicate evidence are most heavily penalized.

        Args:
            outputs: Model output dict.
            labels:  [B] ground truth Q1 labels.

        Returns:
            [B] per-sample FP risk scores.
        """
        # Probability of predicting infraction
        task_probs = F.softmax(outputs["q1_logits"], dim=-1)
        infraction_prob = 1 - task_probs[:, 0]  # P(infraction)

        # Predicate evidence: max activation across predicates
        pred_evidence = outputs["predicate_probs"].max(dim=-1).values

        # Identify actual negatives (no infraction)
        is_negative = (labels == 0).float()

        # FP risk: confident + unsupported + wrong
        fp_risk = infraction_prob * (1 - pred_evidence) * is_negative * self.fp_penalty

        return fp_risk

    def __call__(
        self,
        outputs: Dict[str, Tensor],
        labels: Tensor,
    ) -> Tuple[Tensor, Dict[str, float]]:
        """Compute RSA penalty.

        Args:
            outputs: Model output dict.
            labels:  [B] ground truth Q1 labels.

        Returns:
            (penalty_loss, metrics_dict)
        """
        fp_risk = self.compute_fp_risk(outputs, labels)
        cvar = self.compute_cvar(fp_risk)
        penalty = self.lambda_risk * cvar

        metrics = {
            "rsa_cvar": cvar.item(),
            "rsa_penalty": penalty.item(),
            "fp_risk_mean": fp_risk.mean().item(),
            "fp_risk_max": fp_risk.max().item() if fp_risk.numel() > 0 else 0.0,
        }

        return penalty, metrics


class RSATrainer:
    """Stage 3: RSA-augmented GRPO training.

    Wraps GRPO with an additional CVaR penalty for false positive
    risk reduction.

    Args:
        grpo_trainer:  Initialized GRPOTrainer.
        rsa_constraint: RSAConstraint for FP penalty.
    """

    def __init__(self, grpo_trainer, rsa_constraint: Optional[RSAConstraint] = None):
        self.grpo = grpo_trainer
        self.rsa = rsa_constraint or RSAConstraint()

    def step(self, batch: Dict[str, Tensor]) -> Tuple[Tensor, Dict[str, float]]:
        """One RSA-augmented training step.

        Args:
            batch: Input batch dict.

        Returns:
            (total_loss, combined_metrics)
        """
        # GRPO step (computes loss but we need to add RSA before optimizer step)
        self.grpo.model.train()

        # Forward through model
        outputs = self.grpo.model(
            video=batch["video"],
            mask=batch.get("mask"),
            sport_ids=batch.get("sport_ids"),
        )

        # Compute RSA penalty
        rsa_penalty, rsa_metrics = self.rsa(outputs, batch["q1_labels"])

        # Standard GRPO loss
        grpo_loss, grpo_metrics = self.grpo.step(batch)

        # Combined
        total_loss = grpo_loss + rsa_penalty

        metrics = {**grpo_metrics, **rsa_metrics, "rsa_total_loss": total_loss.item()}

        return total_loss, metrics
