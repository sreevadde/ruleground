"""
Predicate Classifiers

Per-predicate 2-layer MLPs with sport-conditional masking (Paper Section 5.2, Eq. 1).
State/spatial predicates operate on clip-level pooled features.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict, List, Optional

from ruleground.predicates.ontology import (
    ALL_PREDICATE_NAMES,
    NUM_PREDICATES,
    SPORT_TO_ID,
    get_sport_mask,
    get_state_predicate_indices,
    Sport,
    ID_TO_SPORT,
)


class PredicateClassifier(nn.Module):
    """Single predicate classifier: 2-layer MLP with SiLU activation."""

    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        """x: [B, D] -> [B, 1]"""
        return self.mlp(x)


class PredicateHead(nn.Module):
    """Collection of per-predicate classifiers with sport-conditional masking.

    Each predicate has an independent 2-layer MLP. During forward, predicates
    not relevant to the input sport are masked to -inf in logits (0.0 in probs).

    Args:
        embed_dim:  Input feature dimension.
        hidden_dim: MLP hidden dimension.
        dropout:    Dropout rate.
        predicate_names: Ordered list of predicate names to classify.
            Defaults to ALL_PREDICATE_NAMES from ontology.
        predicate_indices: If provided, only create classifiers for these
            indices into ALL_PREDICATE_NAMES (for state-only heads).
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        predicate_names: Optional[List[str]] = None,
        predicate_indices: Optional[List[int]] = None,
    ):
        super().__init__()

        if predicate_names is not None:
            self.predicate_names = predicate_names
        elif predicate_indices is not None:
            self.predicate_names = [ALL_PREDICATE_NAMES[i] for i in predicate_indices]
        else:
            self.predicate_names = list(ALL_PREDICATE_NAMES)

        self.num_predicates = len(self.predicate_names)
        self.name_to_local_idx = {n: i for i, n in enumerate(self.predicate_names)}

        # Per-predicate classifiers
        self.classifiers = nn.ModuleDict(
            {
                name: PredicateClassifier(embed_dim, hidden_dim, dropout)
                for name in self.predicate_names
            }
        )

        # Precompute sport masks (registered as buffers so they move to device)
        for sport in (Sport.BASKETBALL, Sport.FOOTBALL, Sport.SOCCER):
            full_mask = get_sport_mask(sport)
            # Map full-ontology mask to our predicate subset
            local_mask = []
            for name in self.predicate_names:
                global_idx = ALL_PREDICATE_NAMES.index(name)
                local_mask.append(full_mask[global_idx])
            self.register_buffer(
                f"_mask_{sport.name.lower()}",
                torch.tensor(local_mask, dtype=torch.bool),
            )

    def _get_sport_mask(self, sport_id: int) -> Tensor:
        """Retrieve precomputed sport mask buffer."""
        sport = ID_TO_SPORT[sport_id]
        return getattr(self, f"_mask_{sport.name.lower()}")

    def forward(
        self,
        pooled: Tensor,
        sport_ids: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        Args:
            pooled:    [B, D] clip-level features from temporal pooling.
            sport_ids: [B] integer sport IDs. If None, no masking applied.

        Returns:
            Dict with 'logits' [B, K], 'probs' [B, K], 'predicate_dict'.
        """
        B = pooled.shape[0]
        device = pooled.device

        # Compute all predicate logits
        logits = torch.zeros(B, self.num_predicates, device=device)
        for i, name in enumerate(self.predicate_names):
            logits[:, i] = self.classifiers[name](pooled).squeeze(-1)

        # Apply sport-conditional masking
        if sport_ids is not None:
            mask = torch.ones(B, self.num_predicates, dtype=torch.bool, device=device)
            for sport_id in sport_ids.unique():
                sport_mask = self._get_sport_mask(sport_id.item())
                batch_mask = sport_ids == sport_id
                mask[batch_mask] = sport_mask.unsqueeze(0)
            # Mask irrelevant predicates to -inf
            logits = logits.masked_fill(~mask, float("-inf"))

        probs = torch.sigmoid(logits)
        # Masked predicates get sigmoid(-inf) ~ 0.0, which is correct

        pred_dict = {
            name: probs[:, i] for i, name in enumerate(self.predicate_names)
        }

        return {
            "logits": logits,
            "probs": probs,
            "predicate_dict": pred_dict,
        }
