"""
Evaluation Metrics for SportR Benchmark

Q1: Infraction identification (binary accuracy/F1)
Q2: Foul classification (multi-class accuracy/F1-macro)
Q5: Temporal grounding (IoU)
FP: False positive rate reduction
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
from torch import Tensor

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)


def compute_q1_metrics(
    preds: Tensor,
    labels: Tensor,
) -> Dict[str, float]:
    """Q1: Infraction identification (binary).

    Args:
        preds:  [N] predicted class indices or logits (>0 = infraction).
        labels: [N] ground truth labels (>0 = infraction).

    Returns:
        Dict with q1_accuracy, q1_f1, q1_precision, q1_recall.
    """
    if preds.dim() > 1:
        # logits -> class index
        preds = preds.argmax(dim=-1)

    p = (preds > 0).cpu().numpy()
    t = (labels > 0).cpu().numpy()

    return {
        "q1_accuracy": float(accuracy_score(t, p)),
        "q1_f1": float(f1_score(t, p, average="binary", zero_division=0)),
        "q1_precision": float(precision_score(t, p, average="binary", zero_division=0)),
        "q1_recall": float(recall_score(t, p, average="binary", zero_division=0)),
    }


def compute_q2_metrics(
    preds: Tensor,
    labels: Tensor,
    num_classes: int = 17,
) -> Dict[str, float]:
    """Q2: Foul classification (multi-class).

    Args:
        preds:       [N] predicted class indices or [N, C] logits.
        labels:      [N] ground truth class indices.
        num_classes: Number of foul classes.

    Returns:
        Dict with q2_accuracy, q2_f1_macro, q2_f1_weighted.
    """
    if preds.dim() > 1:
        preds = preds.argmax(dim=-1)

    p = preds.cpu().numpy()
    t = labels.cpu().numpy()

    return {
        "q2_accuracy": float(accuracy_score(t, p)),
        "q2_f1_macro": float(f1_score(t, p, average="macro", zero_division=0)),
        "q2_f1_weighted": float(f1_score(t, p, average="weighted", zero_division=0)),
    }


def compute_temporal_iou(
    pred_spans: Tensor,
    gt_spans: Tensor,
) -> Dict[str, float]:
    """Q5: Temporal grounding via span IoU.

    Args:
        pred_spans: [N, 2] predicted (start, end) normalized to [0, 1].
        gt_spans:   [N, 2] ground truth (start, end) normalized to [0, 1].

    Returns:
        Dict with q5_iou_mean, q5_iou_median.
    """
    # Ensure proper ordering
    pred_start = pred_spans[:, 0].clamp(0, 1)
    pred_end = pred_spans[:, 1].clamp(0, 1)
    gt_start = gt_spans[:, 0]
    gt_end = gt_spans[:, 1]

    # Intersection
    inter_start = torch.max(pred_start, gt_start)
    inter_end = torch.min(pred_end, gt_end)
    intersection = (inter_end - inter_start).clamp(min=0)

    # Union
    pred_len = (pred_end - pred_start).clamp(min=0)
    gt_len = (gt_end - gt_start).clamp(min=0)
    union = pred_len + gt_len - intersection

    iou = intersection / (union + 1e-8)

    return {
        "q5_iou_mean": float(iou.mean().item()),
        "q5_iou_median": float(iou.median().item()),
    }


def compute_false_positive_metrics(
    preds: Tensor,
    labels: Tensor,
) -> Dict[str, float]:
    """Compute false positive analysis metrics.

    Args:
        preds:  [N] predicted class indices (0 = no infraction).
        labels: [N] ground truth labels (0 = no infraction).

    Returns:
        Dict with fp_rate, fp_count, tn_count, fp_reduction potential.
    """
    if preds.dim() > 1:
        preds = preds.argmax(dim=-1)

    pred_pos = (preds > 0)
    true_neg = (labels == 0)

    fp = (pred_pos & true_neg).float().sum().item()
    tn = (~pred_pos & true_neg).float().sum().item()
    fn = (~pred_pos & ~true_neg).float().sum().item()
    tp = (pred_pos & ~true_neg).float().sum().item()

    total_neg = fp + tn
    fpr = fp / total_neg if total_neg > 0 else 0.0

    return {
        "fp_rate": fpr,
        "fp_count": int(fp),
        "tn_count": int(tn),
        "tp_count": int(tp),
        "fn_count": int(fn),
    }


def compute_predicate_accuracy(
    pred_probs: Tensor,
    gt_labels: Tensor,
    mask: Optional[Tensor] = None,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute predicate prediction accuracy.

    Args:
        pred_probs: [N, K] predicted predicate probabilities.
        gt_labels:  [N, K] ground truth predicate labels (0/1).
        mask:       [N, K] mask for valid predicate labels.
        threshold:  Classification threshold.

    Returns:
        Dict with pred_accuracy, pred_f1, pred_precision, pred_recall.
    """
    preds_binary = (pred_probs > threshold).float()
    gt_binary = (gt_labels > threshold).float()  # Binarize GT for weak supervision

    if mask is not None:
        # Only evaluate on labeled predicates
        valid = mask.bool()
        p = preds_binary[valid].cpu().numpy()
        t = gt_binary[valid].cpu().numpy()
    else:
        p = preds_binary.cpu().numpy().ravel()
        t = gt_binary.cpu().numpy().ravel()

    if len(t) == 0:
        return {"pred_accuracy": 0.0, "pred_f1": 0.0}

    return {
        "pred_accuracy": float(accuracy_score(t, p)),
        "pred_f1": float(f1_score(t, p, average="binary", zero_division=0)),
        "pred_precision": float(precision_score(t, p, average="binary", zero_division=0)),
        "pred_recall": float(recall_score(t, p, average="binary", zero_division=0)),
    }


def compute_all_metrics(
    outputs: Dict[str, Tensor],
    targets: Dict[str, Tensor],
) -> Dict[str, float]:
    """Compute all metrics from model outputs and targets.

    Args:
        outputs: Model output dict with q1_logits, q2_logits, q5_preds, predicate_probs.
        targets: Target dict with q1_labels, q2_labels, q5_spans, predicate_labels.

    Returns:
        Combined metrics dict.
    """
    metrics = {}

    if "q1_logits" in outputs and "q1_labels" in targets:
        metrics.update(compute_q1_metrics(outputs["q1_logits"], targets["q1_labels"]))

    if "q2_logits" in outputs and "q2_labels" in targets:
        metrics.update(compute_q2_metrics(outputs["q2_logits"], targets["q2_labels"]))

    if "q5_preds" in outputs and "q5_spans" in targets:
        metrics.update(compute_temporal_iou(outputs["q5_preds"], targets["q5_spans"]))

    if "q1_logits" in outputs and "q1_labels" in targets:
        metrics.update(compute_false_positive_metrics(outputs["q1_logits"], targets["q1_labels"]))

    if "predicate_probs" in outputs and "predicate_labels" in targets:
        metrics.update(
            compute_predicate_accuracy(
                outputs["predicate_probs"],
                targets["predicate_labels"],
                targets.get("predicate_mask"),
            )
        )

    return metrics
