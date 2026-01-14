"""
Rule Grounding Module (RGM)

The core innovation of RuleGround: an explicit predicate bottleneck between
perception and reasoning. Predicts structured predicate states from video
embeddings using temporal attention pooling + per-predicate classifiers.

State/spatial predicates use clip-level pooled features.
Instant predicates use frame-level SnapFormer heatmaps.

Paper Section 5.2.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict, List, Optional

from ruleground.predicates.ontology import (
    ALL_PREDICATE_NAMES,
    NUM_PREDICATES,
    PREDICATE_NAME_TO_IDX,
    INSTANT_PREDICATES,
    get_state_predicate_indices,
    get_instant_predicate_indices,
)
from ruleground.models.temporal_pool import TemporalAttentionPool
from ruleground.models.predicate_head import PredicateHead
from ruleground.models.snapformer_head import SnapFormerHead


class RuleGroundingModule(nn.Module):
    """Rule Grounding Module (RGM).

    Composes:
        1. TemporalAttentionPool: aggregates frame embeddings -> clip vector
        2. PredicateHead: state/spatial predicates from clip vector
        3. SnapFormerHead: instant predicates from frame-level heatmaps

    The two predicate heads are merged into a unified predicate vector
    following the canonical ordering from the ontology.

    Args:
        embed_dim:    Video encoder feature dimension.
        num_heads:    Attention heads for temporal pooling.
        max_seq_len:  Maximum temporal sequence length.
        hidden_dim:   MLP hidden dimension for classifiers.
        dropout:      Dropout rate.
        use_rope:     Whether to use RoPE in temporal attention.
        use_flash_attn: Whether to use Flash Attention.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        max_seq_len: int = 64,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        use_rope: bool = True,
        use_flash_attn: bool = True,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_predicates = NUM_PREDICATES

        # Indices for state/spatial vs instant predicates
        self.state_indices = get_state_predicate_indices()
        self.instant_indices = get_instant_predicate_indices()

        state_names = [ALL_PREDICATE_NAMES[i] for i in self.state_indices]
        instant_names = [ALL_PREDICATE_NAMES[i] for i in self.instant_indices]

        # Shared temporal pooling for state predicates
        self.temporal_pool = TemporalAttentionPool(
            embed_dim=embed_dim,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            dropout=dropout,
            use_rope=use_rope,
            use_flash_attn=use_flash_attn,
        )

        # State/spatial predicates: operate on pooled clip features
        self.state_head = PredicateHead(
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            predicate_names=state_names,
        )

        # Instant predicates: operate on frame-level features (SnapFormer)
        self.instant_head = SnapFormerHead(
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            instant_predicate_names=instant_names,
        )

        # Mapping from local head indices to global ontology indices
        self._state_name_to_global = {
            name: PREDICATE_NAME_TO_IDX[name] for name in state_names
        }
        self._instant_name_to_global = {
            name: PREDICATE_NAME_TO_IDX[name] for name in instant_names
        }

    def forward(
        self,
        frame_embeddings: Tensor,
        mask: Optional[Tensor] = None,
        sport_ids: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        Args:
            frame_embeddings: [B, T, D] from video encoder.
            mask:             [B, T] attention mask.
            sport_ids:        [B] sport IDs for conditional masking.

        Returns:
            Dict with:
                'logits':            [B, K] all predicate logits (canonical order)
                'probs':             [B, K] all predicate probs
                'predicate_dict':    {name: [B]} predicate probabilities
                'pooled':            [B, D] pooled clip features
                'frame_activations': [B, T, K_instant] instant pred frame probs
        """
        B = frame_embeddings.shape[0]
        device = frame_embeddings.device

        # --- Temporal pooling ---
        pooled = self.temporal_pool(frame_embeddings, mask)  # [B, D]

        # --- State/spatial predicates (from pooled features) ---
        state_out = self.state_head(pooled, sport_ids)

        # --- Instant predicates (from frame-level features) ---
        instant_out = self.instant_head(frame_embeddings, mask, sport_ids)

        # --- Merge into canonical ordering ---
        logits = torch.full(
            (B, self.num_predicates), float("-inf"), device=device
        )
        probs = torch.zeros(B, self.num_predicates, device=device)

        # Fill state/spatial
        for i, name in enumerate(self.state_head.predicate_names):
            global_idx = self._state_name_to_global[name]
            logits[:, global_idx] = state_out["logits"][:, i]
            probs[:, global_idx] = state_out["probs"][:, i]

        # Fill instant
        for i, name in enumerate(self.instant_head.predicate_names):
            global_idx = self._instant_name_to_global[name]
            logits[:, global_idx] = instant_out["logits"][:, i]
            probs[:, global_idx] = instant_out["probs"][:, i]

        # Build unified predicate dict
        pred_dict: Dict[str, Tensor] = {}
        pred_dict.update(state_out["predicate_dict"])
        pred_dict.update(instant_out["predicate_dict"])

        return {
            "logits": logits,
            "probs": probs,
            "predicate_dict": pred_dict,
            "pooled": pooled,
            "frame_activations": instant_out["frame_activations"],
        }
