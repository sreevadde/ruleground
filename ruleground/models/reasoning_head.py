"""
Reasoning Head (Multi-Task)

Lightweight cross-attention transformer that consumes predicate states and
pooled visual features to produce task outputs (Paper Section 5.4).

Separate output branches for Q1 (infraction ID), Q2 (foul classification),
and Q5 (temporal grounding).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, Optional


class ReasoningHead(nn.Module):
    """Multi-task reasoning head with cross-attention.

    Architecture (matches paper Section 5.4):
        1. Predicate embeddings cross-attend to pooled visual features
        2. Attended features are projected to separate task heads
        3. Q1: binary infraction, Q2: multi-class foul, Q5: temporal span

    Args:
        num_predicates: Number of predicate inputs (K).
        visual_dim:     Dimension of pooled visual features (D).
        hidden_dim:     Internal hidden dimension.
        num_q1_classes: Classes for infraction identification (default 2).
        num_q2_classes: Classes for foul classification.
        num_heads:      Cross-attention heads.
        dropout:        Dropout rate.
    """

    def __init__(
        self,
        num_predicates: int,
        visual_dim: int,
        hidden_dim: int = 256,
        num_q1_classes: int = 2,
        num_q2_classes: int = 17,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.num_predicates = num_predicates
        self.visual_dim = visual_dim
        self.hidden_dim = hidden_dim

        # Project predicate probs to hidden dim (predicate tokens)
        self.pred_proj = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
        )

        # Learnable predicate position embeddings
        self.pred_pos_embed = nn.Parameter(
            torch.randn(1, num_predicates, hidden_dim)
        )

        # Project visual features to hidden dim
        self.vis_proj = nn.Linear(visual_dim, hidden_dim)

        # Cross-attention: predicate tokens attend to visual features
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(hidden_dim)

        # FFN after cross-attention
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim)

        # Pooling over predicate tokens -> single representation
        self.pool = nn.Sequential(
            nn.Linear(hidden_dim, 1),
        )

        # Task-specific output heads
        self.q1_head = nn.Linear(hidden_dim, num_q1_classes)
        self.q2_head = nn.Linear(hidden_dim, num_q2_classes)
        self.q5_head = nn.Linear(hidden_dim, 2)  # temporal start/end offsets

    def forward(
        self,
        predicate_probs: Tensor,
        pooled_visual: Tensor,
        rule_scores: Optional[Dict[str, Tensor]] = None,
    ) -> Dict[str, Tensor]:
        """
        Args:
            predicate_probs: [B, K] predicate probabilities from RGM.
            pooled_visual:   [B, D] pooled visual features from temporal pool.
            rule_scores:     Optional dict of rule_name -> [B] scores.

        Returns:
            Dict with 'q1_logits' [B, C1], 'q2_logits' [B, C2],
            'q5_preds' [B, 2], 'attended_features' [B, K, H].
        """
        B = predicate_probs.shape[0]

        # Build predicate tokens: [B, K, 1] -> [B, K, H]
        pred_tokens = self.pred_proj(predicate_probs.unsqueeze(-1))
        pred_tokens = pred_tokens + self.pred_pos_embed

        # Build visual key/value: [B, D] -> [B, 1, H]
        vis_kv = self.vis_proj(pooled_visual).unsqueeze(1)

        # If rule scores available, append as additional tokens
        if rule_scores:
            rule_vals = torch.stack(list(rule_scores.values()), dim=-1)  # [B, R]
            rule_tokens = self.pred_proj(rule_vals.unsqueeze(-1))  # [B, R, H]
            vis_kv = torch.cat([vis_kv, rule_tokens], dim=1)

        # Cross-attention: predicates (query) attend to visual + rule (key/value)
        attended, _ = self.cross_attn(
            query=pred_tokens,
            key=vis_kv,
            value=vis_kv,
        )
        attended = self.attn_norm(pred_tokens + attended)

        # FFN
        ffn_out = self.ffn(attended)
        features = self.ffn_norm(attended + ffn_out)  # [B, K, H]

        # Pool predicate tokens into single representation
        weights = self.pool(features).softmax(dim=1)  # [B, K, 1]
        pooled = (features * weights).sum(dim=1)  # [B, H]

        return {
            "q1_logits": self.q1_head(pooled),
            "q2_logits": self.q2_head(pooled),
            "q5_preds": self.q5_head(pooled).sigmoid(),
            "attended_features": features,
        }
