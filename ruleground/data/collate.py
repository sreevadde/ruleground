"""
Batch Collation

Custom collate functions that handle variable-length videos,
optional predicate labels, and sport-conditional masking.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
from torch import Tensor

from ruleground.predicates.ontology import (
    ALL_PREDICATE_NAMES,
    NUM_PREDICATES,
    SPORT_TO_ID,
    SPORT_FROM_STR,
)


def sportr_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Tensor]:
    """Collate a batch of SportR samples.

    Handles:
        - Stacking video tensors (pads shorter clips)
        - Converting sport strings to integer IDs
        - Building predicate label and mask tensors
        - Building attention masks for variable-length clips

    Args:
        batch: List of dicts from SportRDataset.__getitem__.

    Returns:
        Batched dict with all tensors.
    """
    B = len(batch)

    # Stack videos (assume uniform after transforms)
    videos = torch.stack([item["video"] for item in batch])  # [B, T, C, H, W]

    # Q1 labels (binary: 0 = no infraction, 1+ = infraction)
    q1_labels = torch.tensor([item["q1_label"] for item in batch], dtype=torch.long)

    # Q2 labels (foul class index)
    q2_labels = torch.tensor([item["q2_label"] for item in batch], dtype=torch.long)

    # Sport IDs
    sport_ids = torch.tensor(
        [
            SPORT_TO_ID[SPORT_FROM_STR[item["sport"].lower()]]
            if isinstance(item["sport"], str)
            else item["sport"]
            for item in batch
        ],
        dtype=torch.long,
    )

    result = {
        "video": videos,
        "q1_labels": q1_labels,
        "q2_labels": q2_labels,
        "sport_ids": sport_ids,
    }

    # Video IDs (for evaluation tracking)
    if "video_id" in batch[0]:
        result["video_ids"] = [item["video_id"] for item in batch]

    # Attention mask (all ones if clips are uniform length)
    T = videos.shape[1]
    result["mask"] = torch.ones(B, T, dtype=torch.bool)

    # Predicate labels and weights (if available from extraction)
    if "predicates" in batch[0] and batch[0]["predicates"] is not None:
        pred_labels = torch.zeros(B, NUM_PREDICATES)
        pred_weights = torch.zeros(B, NUM_PREDICATES)
        pred_mask = torch.zeros(B, NUM_PREDICATES)

        for i, item in enumerate(batch):
            preds = item.get("predicates", {})
            if not preds:
                continue
            labels = preds.get("labels", {})
            weights = preds.get("weights", {})

            for name, val in labels.items():
                if name in ALL_PREDICATE_NAMES:
                    idx = ALL_PREDICATE_NAMES.index(name)
                    pred_labels[i, idx] = float(val)
                    pred_weights[i, idx] = weights.get(name, 1.0)
                    pred_mask[i, idx] = 1.0 if weights.get(name, 0.0) > 0 else 0.0

        result["predicate_labels"] = pred_labels
        result["predicate_weights"] = pred_weights
        result["predicate_mask"] = pred_mask

    # Q5 temporal spans (if available)
    if "q5_span" in batch[0] and batch[0]["q5_span"] is not None:
        q5_spans = torch.tensor(
            [item["q5_span"] for item in batch], dtype=torch.float
        )
        result["q5_spans"] = q5_spans

    return result
