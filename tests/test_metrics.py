"""Tests for evaluation metrics."""

import pytest
import torch

from ruleground.evaluation.metrics import (
    compute_q1_metrics,
    compute_q2_metrics,
    compute_temporal_iou,
    compute_false_positive_metrics,
    compute_predicate_accuracy,
    compute_all_metrics,
)


class TestQ1Metrics:
    def test_perfect_accuracy(self):
        preds = torch.tensor([0, 1, 1, 0])
        labels = torch.tensor([0, 1, 1, 0])
        metrics = compute_q1_metrics(preds, labels)
        assert metrics["q1_accuracy"] == 1.0
        assert metrics["q1_f1"] == 1.0

    def test_from_logits(self):
        preds = torch.tensor([[5.0, -5.0], [-5.0, 5.0], [-5.0, 5.0]])
        labels = torch.tensor([0, 1, 1])
        metrics = compute_q1_metrics(preds, labels)
        assert metrics["q1_accuracy"] == 1.0

    def test_all_wrong(self):
        preds = torch.tensor([1, 0, 0, 1])
        labels = torch.tensor([0, 1, 1, 0])
        metrics = compute_q1_metrics(preds, labels)
        assert metrics["q1_accuracy"] == 0.0


class TestQ2Metrics:
    def test_perfect(self):
        preds = torch.tensor([0, 1, 2, 3])
        labels = torch.tensor([0, 1, 2, 3])
        metrics = compute_q2_metrics(preds, labels)
        assert metrics["q2_accuracy"] == 1.0

    def test_from_logits(self):
        # 3-class, batch of 2
        logits = torch.tensor([[5.0, 0.0, 0.0], [0.0, 0.0, 5.0]])
        labels = torch.tensor([0, 2])
        metrics = compute_q2_metrics(logits, labels)
        assert metrics["q2_accuracy"] == 1.0


class TestTemporalIoU:
    def test_perfect_overlap(self):
        pred = torch.tensor([[0.2, 0.8], [0.1, 0.5]])
        gt = torch.tensor([[0.2, 0.8], [0.1, 0.5]])
        metrics = compute_temporal_iou(pred, gt)
        assert abs(metrics["q5_iou_mean"] - 1.0) < 1e-5

    def test_no_overlap(self):
        pred = torch.tensor([[0.0, 0.3]])
        gt = torch.tensor([[0.5, 1.0]])
        metrics = compute_temporal_iou(pred, gt)
        assert metrics["q5_iou_mean"] < 1e-5

    def test_partial_overlap(self):
        pred = torch.tensor([[0.0, 0.5]])
        gt = torch.tensor([[0.25, 0.75]])
        metrics = compute_temporal_iou(pred, gt)
        # Intersection: 0.25-0.5 = 0.25, Union: 0.75
        assert 0.3 < metrics["q5_iou_mean"] < 0.4


class TestFalsePositiveMetrics:
    def test_no_false_positives(self):
        preds = torch.tensor([0, 1, 0, 1])
        labels = torch.tensor([0, 1, 0, 1])
        metrics = compute_false_positive_metrics(preds, labels)
        assert metrics["fp_rate"] == 0.0
        assert metrics["fp_count"] == 0

    def test_all_false_positives(self):
        preds = torch.tensor([1, 1])
        labels = torch.tensor([0, 0])
        metrics = compute_false_positive_metrics(preds, labels)
        assert metrics["fp_rate"] == 1.0
        assert metrics["fp_count"] == 2


class TestPredicateAccuracy:
    def test_perfect(self):
        probs = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        labels = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        metrics = compute_predicate_accuracy(probs, labels)
        assert metrics["pred_accuracy"] == 1.0

    def test_with_mask(self):
        probs = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        labels = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        mask = torch.tensor([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]])  # only some labeled
        metrics = compute_predicate_accuracy(probs, labels, mask=mask)
        assert metrics["pred_accuracy"] == 1.0


class TestComputeAllMetrics:
    def test_combines_all(self):
        outputs = {
            "q1_logits": torch.tensor([[5.0, -5.0], [-5.0, 5.0]]),
            "q2_logits": torch.tensor([[5.0, 0.0], [0.0, 5.0]]),
            "predicate_probs": torch.tensor([[0.9, 0.1], [0.1, 0.9]]),
        }
        targets = {
            "q1_labels": torch.tensor([0, 1]),
            "q2_labels": torch.tensor([0, 1]),
            "predicate_labels": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        }
        metrics = compute_all_metrics(outputs, targets)
        assert "q1_accuracy" in metrics
        assert "q2_accuracy" in metrics
        assert "pred_accuracy" in metrics
        assert "fp_rate" in metrics
