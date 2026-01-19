"""Tests for loss functions."""

import pytest
import torch

from ruleground.training.losses import (
    WeightedBCELoss,
    RuleConsistencyLoss,
    Q5TemporalLoss,
    RuleGroundLoss,
)


class TestWeightedBCELoss:
    def test_basic_output(self):
        loss_fn = WeightedBCELoss()
        logits = torch.randn(4, 20)
        targets = torch.randint(0, 2, (4, 20)).float()
        loss = loss_fn(logits, targets)
        assert loss.dim() == 0
        assert loss.item() > 0

    def test_with_weights(self):
        loss_fn = WeightedBCELoss()
        logits = torch.randn(4, 20)
        targets = torch.randint(0, 2, (4, 20)).float()
        weights = torch.rand(4, 20)
        loss = loss_fn(logits, targets, weights=weights)
        assert loss.dim() == 0

    def test_with_mask(self):
        loss_fn = WeightedBCELoss()
        logits = torch.randn(4, 20)
        targets = torch.randint(0, 2, (4, 20)).float()
        mask = torch.zeros(4, 20)
        mask[:, :5] = 1.0  # only first 5 predicates labeled
        loss = loss_fn(logits, targets, mask=mask)
        assert loss.dim() == 0

    def test_gradient_flows(self):
        loss_fn = WeightedBCELoss()
        logits = torch.randn(4, 20, requires_grad=True)
        targets = torch.randint(0, 2, (4, 20)).float()
        loss = loss_fn(logits, targets)
        loss.backward()
        assert logits.grad is not None


class TestRuleConsistencyLoss:
    def test_with_rules(self):
        loss_fn = RuleConsistencyLoss()
        rule_scores = {
            "blocking_foul": torch.rand(4),
            "charging_foul": torch.rand(4),
        }
        q1_labels = torch.tensor([0, 1, 1, 0])
        loss = loss_fn(rule_scores, q1_labels)
        assert loss.dim() == 0

    def test_empty_rules(self):
        loss_fn = RuleConsistencyLoss()
        loss = loss_fn({}, torch.tensor([0, 1]))
        assert loss.dim() == 0

    def test_gradient_flows(self):
        loss_fn = RuleConsistencyLoss()
        score = torch.rand(4, requires_grad=True)
        rule_scores = {"test": score.sigmoid()}
        q1_labels = torch.tensor([0, 1, 1, 0])
        loss = loss_fn(rule_scores, q1_labels)
        loss.backward()
        assert score.grad is not None


class TestQ5TemporalLoss:
    def test_basic(self):
        loss_fn = Q5TemporalLoss()
        pred = torch.rand(4, 2)
        gt = torch.rand(4, 2)
        loss = loss_fn(pred, gt)
        assert loss.dim() == 0

    def test_with_mask(self):
        loss_fn = Q5TemporalLoss()
        pred = torch.rand(4, 2)
        gt = torch.rand(4, 2)
        mask = torch.tensor([1.0, 1.0, 0.0, 0.0])
        loss = loss_fn(pred, gt, mask=mask)
        assert loss.dim() == 0

    def test_perfect_prediction(self):
        loss_fn = Q5TemporalLoss()
        gt = torch.tensor([[0.2, 0.8], [0.1, 0.5]])
        loss = loss_fn(gt, gt)
        assert loss.item() < 1e-6


class TestRuleGroundLoss:
    def test_full_loss(self):
        loss_fn = RuleGroundLoss(gamma=0.5, delta=0.1, eta=0.5)
        outputs = {
            "q1_logits": torch.randn(4, 2),
            "q2_logits": torch.randn(4, 17),
            "predicate_logits": torch.randn(4, 20),
            "q5_preds": torch.rand(4, 2),
            "rule_scores": {
                "blocking_foul": torch.rand(4),
            },
        }
        targets = {
            "q1_labels": torch.randint(0, 2, (4,)),
            "q2_labels": torch.randint(0, 17, (4,)),
            "predicate_labels": torch.randint(0, 2, (4, 20)).float(),
            "predicate_weights": torch.rand(4, 20),
            "predicate_mask": torch.ones(4, 20),
            "q5_spans": torch.rand(4, 2),
        }
        loss, breakdown = loss_fn(outputs, targets)
        assert loss.dim() == 0
        assert "loss_q1" in breakdown
        assert "loss_q2" in breakdown
        assert "loss_pred" in breakdown
        assert "loss_cons" in breakdown
        assert "loss_total" in breakdown

    def test_minimal_loss(self):
        """Test with only Q1/Q2 (no predicates, no rules)."""
        loss_fn = RuleGroundLoss()
        outputs = {
            "q1_logits": torch.randn(4, 2),
            "q2_logits": torch.randn(4, 17),
        }
        targets = {
            "q1_labels": torch.randint(0, 2, (4,)),
            "q2_labels": torch.randint(0, 17, (4,)),
        }
        loss, breakdown = loss_fn(outputs, targets)
        assert loss.dim() == 0
        assert "loss_q1" in breakdown
        assert "loss_q2" in breakdown

    def test_gradient_flows(self):
        loss_fn = RuleGroundLoss()
        logits = torch.randn(4, 2, requires_grad=True)
        outputs = {"q1_logits": logits, "q2_logits": torch.randn(4, 17)}
        targets = {"q1_labels": torch.randint(0, 2, (4,)), "q2_labels": torch.randint(0, 17, (4,))}
        loss, _ = loss_fn(outputs, targets)
        loss.backward()
        assert logits.grad is not None
