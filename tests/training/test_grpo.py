"""Tests for GRPO and RSA training."""

import pytest
import torch
import torch.nn as nn

from ruleground.training.grpo import PredicateDropout, GRPOTrainer
from ruleground.training.rsa import RSAConstraint
from ruleground.training.rewards import RewardFunction


class TestPredicateDropout:
    def test_no_dropout_eval(self):
        pd = PredicateDropout(p=0.5)
        pd.eval()
        x = torch.rand(2, 20)
        out = pd(x)
        assert torch.equal(x, out)

    def test_dropout_train(self):
        pd = PredicateDropout(p=0.5)
        pd.train()
        x = torch.ones(2, 20)
        out = pd(x)
        # Some values should be zeroed
        assert (out == 0).any()

    def test_shape_preserved(self):
        pd = PredicateDropout(p=0.3)
        pd.train()
        x = torch.rand(4, 20)
        assert pd(x).shape == (4, 20)

    def test_zero_dropout(self):
        pd = PredicateDropout(p=0.0)
        pd.train()
        x = torch.rand(2, 20)
        out = pd(x)
        assert torch.equal(x, out)


class TestRewardFunction:
    def test_correctness_reward(self):
        rf = RewardFunction()
        q1_logits = torch.tensor([[0.0, 1.0], [1.0, 0.0]])  # pred: [1, 0]
        q2_logits = torch.tensor([[1.0, 0.0], [0.0, 1.0]])  # pred: [0, 1]
        q1_labels = torch.tensor([1, 0])  # both correct for Q1
        q2_labels = torch.tensor([0, 1])  # both correct for Q2
        reward = rf.compute_correctness(q1_logits, q2_logits, q1_labels, q2_labels)
        assert reward.shape == (2,)
        assert (reward > 0).all()

    def test_faithfulness_reward(self):
        rf = RewardFunction()
        pred_probs = torch.tensor([[0.9, 0.8, 0.1], [0.1, 0.1, 0.1]])
        q1_logits = torch.tensor([[0.0, 1.0], [1.0, 0.0]])  # [infraction, no_infraction]
        reward = rf.compute_faithfulness(pred_probs, q1_logits)
        assert reward.shape == (2,)
        # First sample: infraction with high evidence -> high faithfulness
        # Second sample: no infraction with low evidence -> high faithfulness
        assert reward[0] > 0

    def test_consistency_reward(self):
        rf = RewardFunction()
        # Sample 0: rule fires (0.9) and IS infraction -> consistent
        # Sample 1: rule fires (0.9) but NOT infraction -> inconsistent
        rule_scores = {"rule1": torch.tensor([0.9, 0.9])}
        q1_labels = torch.tensor([1, 0])
        reward = rf.compute_consistency(rule_scores, q1_labels)
        assert reward.shape == (2,)
        assert reward[0] > reward[1]  # first is more consistent

    def test_total_reward(self):
        rf = RewardFunction()
        outputs = {
            "q1_logits": torch.randn(4, 2),
            "q2_logits": torch.randn(4, 17),
            "predicate_probs": torch.rand(4, 20),
            "rule_scores": {"r1": torch.rand(4)},
        }
        targets = {
            "q1_labels": torch.randint(0, 2, (4,)),
            "q2_labels": torch.randint(0, 17, (4,)),
        }
        reward = rf(outputs, targets)
        assert reward.shape == (4,)


class TestRSAConstraint:
    def test_cvar_computation(self):
        rsa = RSAConstraint(alpha=0.2)
        losses = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        cvar = rsa.compute_cvar(losses)
        # Top 20% = 1 sample = 5.0
        assert cvar.item() == 5.0

    def test_fp_risk(self):
        rsa = RSAConstraint()
        outputs = {
            "q1_logits": torch.tensor([[0.0, 5.0], [0.0, 5.0], [5.0, 0.0]]),
            "predicate_probs": torch.tensor([[0.1, 0.1], [0.9, 0.9], [0.1, 0.1]]),
        }
        labels = torch.tensor([0, 0, 1])  # first two are negatives
        fp_risk = rsa.compute_fp_risk(outputs, labels)
        assert fp_risk.shape == (3,)
        # First: high infraction prob + low evidence + negative -> high risk
        # Second: high infraction prob + high evidence + negative -> lower risk
        # Third: positive, so risk = 0
        assert fp_risk[0] > fp_risk[1]
        assert fp_risk[2] == 0.0

    def test_call_returns_penalty(self):
        rsa = RSAConstraint()
        outputs = {
            "q1_logits": torch.randn(4, 2),
            "predicate_probs": torch.rand(4, 20),
        }
        labels = torch.randint(0, 2, (4,))
        penalty, metrics = rsa(outputs, labels)
        assert penalty.dim() == 0
        assert "rsa_cvar" in metrics
        assert "rsa_penalty" in metrics
