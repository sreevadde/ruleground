"""
Group Relative Policy Optimization (GRPO) -- Stage 2

Paper Section 5.5: GRPO post-training with group-relative advantage
estimation (no learned critic needed).

Key fix from review: predicate dropout for stochasticity.
In the original GRPO paper, stochasticity comes from token sampling.
For our discriminative model, we introduce predicate dropout to generate
diverse group members from the same input.

Â_GRPO(x, y_i) = (R(x, y_i) - mu_group) / sigma_group
"""

from __future__ import annotations

import copy
import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ruleground.training.rewards import RewardFunction

logger = logging.getLogger(__name__)


class PredicateDropout(nn.Module):
    """Stochasticity source for GRPO group sampling.

    Applies independent dropout masks to predicate probabilities,
    generating diverse outputs from the same input. This is the
    key architectural fix for adapting GRPO to discriminative models.

    Args:
        p: Dropout probability for each predicate.
    """

    def __init__(self, p: float = 0.1):
        super().__init__()
        self.p = p

    def forward(self, predicate_probs: Tensor) -> Tensor:
        """Apply random dropout to predicate probabilities.

        Args:
            predicate_probs: [B, K] predicate probabilities.

        Returns:
            Perturbed [B, K] probabilities.
        """
        if not self.training or self.p == 0:
            return predicate_probs
        mask = torch.bernoulli(
            torch.full_like(predicate_probs, 1 - self.p)
        )
        return predicate_probs * mask


class GRPOTrainer:
    """GRPO: Group-relative advantage estimation without learned critic.

    For each input, generates G group members via predicate dropout,
    computes rewards, normalizes advantages within the group, and
    updates the policy with clipped ratio.

    Args:
        model:         Active policy model.
        ref_model:     Frozen reference model (for KL constraint).
        reward_fn:     Reward function.
        group_size:    Number of group samples per input (G).
        clip_ratio:    PPO-style clip ratio (epsilon).
        kl_coef:       KL divergence penalty coefficient.
        pred_dropout:  Predicate dropout rate for stochasticity.
        lr:            Learning rate for GRPO fine-tuning.
    """

    def __init__(
        self,
        model: nn.Module,
        ref_model: Optional[nn.Module] = None,
        reward_fn: Optional[RewardFunction] = None,
        group_size: int = 8,
        clip_ratio: float = 0.2,
        kl_coef: float = 0.1,
        pred_dropout: float = 0.1,
        lr: float = 1e-5,
    ):
        self.model = model
        self.group_size = group_size
        self.clip_ratio = clip_ratio
        self.kl_coef = kl_coef

        # Reference model (frozen copy)
        if ref_model is None:
            self.ref_model = copy.deepcopy(model)
        else:
            self.ref_model = ref_model
        for p in self.ref_model.parameters():
            p.requires_grad = False
        self.ref_model.eval()

        self.reward_fn = reward_fn or RewardFunction()
        self.pred_dropout = PredicateDropout(p=pred_dropout)

        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    def _forward_with_dropout(
        self,
        model: nn.Module,
        batch: Dict[str, Tensor],
    ) -> Dict[str, Tensor]:
        """Forward pass with predicate dropout for stochasticity."""
        outputs = model(
            video=batch["video"],
            mask=batch.get("mask"),
            sport_ids=batch.get("sport_ids"),
        )
        # Apply predicate dropout for diversity
        outputs["predicate_probs"] = self.pred_dropout(outputs["predicate_probs"])
        return outputs

    def compute_advantages(self, rewards: Tensor) -> Tensor:
        """Group-relative advantage normalization.

        Â(x, y_i) = (R(x, y_i) - mu_group) / sigma_group

        Args:
            rewards: [B, G] rewards for each group member.

        Returns:
            [B, G] normalized advantages.
        """
        mu = rewards.mean(dim=-1, keepdim=True)
        sigma = rewards.std(dim=-1, keepdim=True).clamp(min=1e-8)
        return (rewards - mu) / sigma

    def step(self, batch: Dict[str, Tensor]) -> Tuple[Tensor, Dict[str, float]]:
        """One GRPO optimization step.

        Args:
            batch: Input batch dict.

        Returns:
            (loss, metrics_dict)
        """
        self.model.train()
        B = batch["video"].shape[0]
        G = self.group_size
        device = batch["video"].device

        # Sample G outputs per input with different dropout masks
        group_q1_logprobs = []
        group_ref_logprobs = []
        group_rewards = torch.zeros(B, G, device=device)

        for g in range(G):
            # Policy forward with dropout
            outputs = self._forward_with_dropout(self.model, batch)
            log_probs = F.log_softmax(outputs["q1_logits"], dim=-1)

            # Reference forward (no dropout)
            with torch.no_grad():
                ref_outputs = self.ref_model(
                    video=batch["video"],
                    mask=batch.get("mask"),
                    sport_ids=batch.get("sport_ids"),
                )
                ref_log_probs = F.log_softmax(ref_outputs["q1_logits"], dim=-1)

            # Get action (argmax of current policy)
            action = outputs["q1_logits"].argmax(dim=-1)  # [B]
            action_lp = log_probs.gather(-1, action.unsqueeze(-1)).squeeze(-1)
            ref_action_lp = ref_log_probs.gather(-1, action.unsqueeze(-1)).squeeze(-1)

            group_q1_logprobs.append(action_lp)
            group_ref_logprobs.append(ref_action_lp)

            # Compute reward
            group_rewards[:, g] = self.reward_fn(outputs, batch)

        # Stack across group
        policy_lp = torch.stack(group_q1_logprobs, dim=1)  # [B, G]
        ref_lp = torch.stack(group_ref_logprobs, dim=1)  # [B, G]

        # Advantages
        advantages = self.compute_advantages(group_rewards)  # [B, G]

        # Clipped policy gradient
        ratio = (policy_lp - ref_lp.detach()).exp()
        clipped_ratio = ratio.clamp(1 - self.clip_ratio, 1 + self.clip_ratio)
        policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()

        # KL penalty
        kl_div = (ref_lp.detach() - policy_lp).mean()
        kl_loss = self.kl_coef * kl_div

        total_loss = policy_loss + kl_loss

        # Optimize
        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        metrics = {
            "grpo_loss": total_loss.item(),
            "policy_loss": policy_loss.item(),
            "kl_loss": kl_loss.item(),
            "mean_reward": group_rewards.mean().item(),
            "mean_advantage": advantages.mean().item(),
        }

        return total_loss, metrics
