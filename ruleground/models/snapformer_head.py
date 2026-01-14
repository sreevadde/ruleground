"""
SnapFormer Head for Instant Predicates

Frame-level predicate detection via 1D heatmap regression (Paper Section 5.2).
Produces per-frame activations for instant-type predicates (e.g., contact_occurred,
ball_released), enabling both predicate prediction and temporal grounding for Q5.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, List, Optional, Tuple

from ruleground.predicates.ontology import (
    ALL_PREDICATE_NAMES,
    INSTANT_PREDICATES,
    get_instant_predicate_indices,
    get_sport_mask,
    Sport,
    ID_TO_SPORT,
)


class SnapFormerHead(nn.Module):
    """Frame-level predicate detection for instant events.

    For each instant predicate, produces a per-frame activation heatmap [B, T]
    that peaks at the frame where the event occurs. The clip-level predicate
    score is obtained via max-pooling over the temporal axis.

    Also provides frame-level activations for Q5 temporal grounding.

    Args:
        embed_dim:  Frame embedding dimension.
        hidden_dim: Internal projection dimension.
        dropout:    Dropout rate.
        instant_predicate_names: Names of instant predicates.
            Defaults to all INSTANT_PREDICATES from ontology.
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        instant_predicate_names: Optional[List[str]] = None,
    ):
        super().__init__()

        if instant_predicate_names is None:
            self.predicate_names = sorted(INSTANT_PREDICATES)
        else:
            self.predicate_names = list(instant_predicate_names)

        self.num_predicates = len(self.predicate_names)
        self.name_to_idx = {n: i for i, n in enumerate(self.predicate_names)}

        # Shared temporal convolution for local context
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(embed_dim, hidden_dim, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        # Per-predicate heatmap heads: 1D conv producing per-frame logits
        self.heatmap_heads = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Conv1d(hidden_dim, hidden_dim // 2, kernel_size=1),
                    nn.SiLU(),
                    nn.Conv1d(hidden_dim // 2, 1, kernel_size=1),
                )
                for name in self.predicate_names
            }
        )

        # Precompute sport masks for instant predicates
        for sport in (Sport.BASKETBALL, Sport.FOOTBALL, Sport.SOCCER):
            full_mask = get_sport_mask(sport)
            local_mask = []
            for name in self.predicate_names:
                global_idx = ALL_PREDICATE_NAMES.index(name)
                local_mask.append(full_mask[global_idx])
            self.register_buffer(
                f"_mask_{sport.name.lower()}",
                torch.tensor(local_mask, dtype=torch.bool),
            )

    def _get_sport_mask(self, sport_id: int) -> Tensor:
        sport = ID_TO_SPORT[sport_id]
        return getattr(self, f"_mask_{sport.name.lower()}")

    def forward(
        self,
        frame_embeddings: Tensor,
        mask: Optional[Tensor] = None,
        sport_ids: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        Args:
            frame_embeddings: [B, T, D] per-frame features from encoder.
            mask:             [B, T] attention mask (True = valid frame).
            sport_ids:        [B] integer sport IDs for masking.

        Returns:
            Dict with:
                'logits':            [B, K_instant] clip-level logits (max over time)
                'probs':             [B, K_instant] clip-level probabilities
                'frame_logits':      [B, T, K_instant] per-frame logits
                'frame_activations': [B, T, K_instant] per-frame probabilities
                'predicate_dict':    {name: Tensor[B]} clip-level probs
        """
        B, T, D = frame_embeddings.shape
        device = frame_embeddings.device

        # Shared temporal convolution: [B, T, D] -> [B, D, T] -> [B, H, T]
        x = frame_embeddings.transpose(1, 2)
        x = self.temporal_conv(x)  # [B, hidden_dim, T]

        # Per-predicate frame-level heatmaps
        frame_logits = torch.zeros(
            B, T, self.num_predicates, device=device
        )
        for i, name in enumerate(self.predicate_names):
            h = self.heatmap_heads[name](x)  # [B, 1, T]
            frame_logits[:, :, i] = h.squeeze(1)  # [B, T]

        # Mask out invalid frames
        if mask is not None:
            frame_logits = frame_logits.masked_fill(
                ~mask.unsqueeze(-1), float("-inf")
            )

        # Clip-level: max-pool over time
        clip_logits = frame_logits.max(dim=1).values  # [B, K_instant]

        # Sport-conditional masking
        if sport_ids is not None:
            sport_mask = torch.ones(
                B, self.num_predicates, dtype=torch.bool, device=device
            )
            for sport_id in sport_ids.unique():
                sm = self._get_sport_mask(sport_id.item())
                batch_idx = sport_ids == sport_id
                sport_mask[batch_idx] = sm.unsqueeze(0)
            clip_logits = clip_logits.masked_fill(~sport_mask, float("-inf"))
            frame_logits = frame_logits.masked_fill(
                ~sport_mask.unsqueeze(1), float("-inf")
            )

        clip_probs = torch.sigmoid(clip_logits)
        frame_probs = torch.sigmoid(frame_logits)

        pred_dict = {
            name: clip_probs[:, i]
            for i, name in enumerate(self.predicate_names)
        }

        return {
            "logits": clip_logits,
            "probs": clip_probs,
            "frame_logits": frame_logits,
            "frame_activations": frame_probs,
            "predicate_dict": pred_dict,
        }
