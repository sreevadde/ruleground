"""Tests for the differentiable logic layer."""

import torch
import pytest

from ruleground.models.logic import DifferentiableLogic, RuleComposer
from ruleground.predicates.rules import RULE_LIBRARY


class TestDifferentiableLogic:
    """Verify t-norm operations at boundaries and differentiability."""

    def test_and_boundaries(self):
        a = torch.tensor([0.0, 1.0, 1.0, 0.0])
        b = torch.tensor([0.0, 0.0, 1.0, 1.0])
        result = DifferentiableLogic.AND(a, b)
        expected = torch.tensor([0.0, 0.0, 1.0, 0.0])
        assert torch.allclose(result, expected)

    def test_or_boundaries(self):
        a = torch.tensor([0.0, 1.0, 1.0, 0.0])
        b = torch.tensor([0.0, 0.0, 1.0, 1.0])
        result = DifferentiableLogic.OR(a, b)
        expected = torch.tensor([0.0, 1.0, 1.0, 1.0])
        assert torch.allclose(result, expected)

    def test_not_boundaries(self):
        a = torch.tensor([0.0, 1.0, 0.5])
        result = DifferentiableLogic.NOT(a)
        expected = torch.tensor([1.0, 0.0, 0.5])
        assert torch.allclose(result, expected)

    def test_implies_boundaries(self):
        # True -> True = True
        assert DifferentiableLogic.IMPLIES(torch.tensor(1.0), torch.tensor(1.0)).item() == 1.0
        # True -> False = False
        assert DifferentiableLogic.IMPLIES(torch.tensor(1.0), torch.tensor(0.0)).item() == 0.0
        # False -> anything = True
        assert DifferentiableLogic.IMPLIES(torch.tensor(0.0), torch.tensor(0.0)).item() == 1.0
        assert DifferentiableLogic.IMPLIES(torch.tensor(0.0), torch.tensor(1.0)).item() == 1.0

    def test_fuzzy_intermediate(self):
        a = torch.tensor(0.7)
        b = torch.tensor(0.6)
        and_result = DifferentiableLogic.AND(a, b)
        or_result = DifferentiableLogic.OR(a, b)
        assert and_result.item() == pytest.approx(0.42, abs=1e-6)
        assert or_result.item() == pytest.approx(0.88, abs=1e-6)

    def test_multi_and(self):
        tensors = [torch.tensor(0.9), torch.tensor(0.8), torch.tensor(0.7)]
        result = DifferentiableLogic.multi_AND(*tensors)
        expected = 0.9 * 0.8 * 0.7
        assert result.item() == pytest.approx(expected, abs=1e-6)

    def test_gradient_flows(self):
        a = torch.tensor(0.5, requires_grad=True)
        b = torch.tensor(0.5, requires_grad=True)
        result = DifferentiableLogic.AND(a, b)
        result.backward()
        assert a.grad is not None
        assert b.grad is not None

    def test_gradient_through_not(self):
        a = torch.tensor(0.3, requires_grad=True)
        result = DifferentiableLogic.NOT(a)
        result.backward()
        assert a.grad is not None
        assert a.grad.item() == pytest.approx(-1.0)


class TestRuleComposer:
    """Test rule parsing, compilation, and evaluation."""

    def test_simple_and_rule(self):
        composer = RuleComposer()
        preds = {
            "contact_occurred": torch.tensor([1.0]),
            "defender_set": torch.tensor([0.0]),
            "restricted_area": torch.tensor([0.0]),
            "shooting_motion": torch.tensor([0.0]),
            "verticality_maintained": torch.tensor([0.0]),
        }
        scores = composer(preds)
        assert "blocking_foul" in scores
        # contact_occurred(1.0) AND NOT defender_set(0.0) = 1.0 * 1.0 = 1.0
        assert scores["blocking_foul"].item() == pytest.approx(1.0)

    def test_charging_foul_rule(self):
        composer = RuleComposer()
        preds = {
            "contact_occurred": torch.tensor([1.0]),
            "defender_set": torch.tensor([1.0]),
            "restricted_area": torch.tensor([0.0]),
            "shooting_motion": torch.tensor([0.0]),
            "verticality_maintained": torch.tensor([0.0]),
        }
        scores = composer(preds)
        # contact(1) AND defender_set(1) AND NOT restricted(0) = 1*1*1 = 1.0
        assert scores["charging_foul"].item() == pytest.approx(1.0)

    def test_dpi_rule(self):
        composer = RuleComposer()
        preds = {
            "contact_before_arrival": torch.tensor([0.9]),
            "incidental_contact": torch.tensor([0.1]),
            "ball_catchable": torch.tensor([0.8]),
            "ball_in_air": torch.tensor([0.9]),
            "offensive_push_off": torch.tensor([0.0]),
            "within_five_yards": torch.tensor([0.0]),
            "contact_occurred": torch.tensor([0.9]),
        }
        scores = composer(preds)
        # before_arrival(0.9) AND NOT incidental(0.1) AND catchable(0.8)
        # = 0.9 * 0.9 * 0.8 = 0.648
        assert "dpi" in scores
        assert scores["dpi"].item() == pytest.approx(0.648, abs=1e-3)

    def test_missing_predicates_skip_gracefully(self):
        composer = RuleComposer()
        # Only provide basketball predicates -- football/soccer rules should be skipped
        preds = {
            "contact_occurred": torch.tensor([1.0]),
            "defender_set": torch.tensor([0.0]),
            "restricted_area": torch.tensor([0.0]),
            "shooting_motion": torch.tensor([0.0]),
            "verticality_maintained": torch.tensor([0.0]),
            "pivot_foot_lifted": torch.tensor([0.0]),
            "ball_released": torch.tensor([0.0]),
        }
        scores = composer(preds)
        assert "blocking_foul" in scores
        assert "dpi" not in scores  # missing football predicates

    def test_for_sport_filters_rules(self):
        bb_composer = RuleComposer().for_sport("basketball")
        rule_names = bb_composer.rule_names
        assert "blocking_foul" in rule_names
        assert "dpi" not in rule_names
        assert "handball" not in rule_names

    def test_batched_evaluation(self):
        composer = RuleComposer()
        B = 4
        preds = {
            "contact_occurred": torch.rand(B),
            "defender_set": torch.rand(B),
            "restricted_area": torch.rand(B),
            "shooting_motion": torch.rand(B),
            "verticality_maintained": torch.rand(B),
            "pivot_foot_lifted": torch.rand(B),
            "ball_released": torch.rand(B),
        }
        scores = composer(preds)
        for name, score in scores.items():
            assert score.shape == (B,), f"{name} has wrong shape"

    def test_nested_formula_parsing(self):
        """Verify the parser handles the 3-way AND in charging foul."""
        from ruleground.predicates.rules import Rule, RULE_LIBRARY

        charging = [r for r in RULE_LIBRARY if r.name == "charging_foul"][0]
        assert "AND" in charging.formula
        # Should have two AND operators
        tokens = charging.formula.split()
        assert tokens.count("AND") == 2

    def test_gradient_through_rules(self):
        composer = RuleComposer()
        preds = {
            "contact_occurred": torch.tensor([0.8], requires_grad=True),
            "defender_set": torch.tensor([0.3], requires_grad=True),
            "restricted_area": torch.tensor([0.2], requires_grad=True),
            "shooting_motion": torch.tensor([0.5], requires_grad=True),
            "verticality_maintained": torch.tensor([0.4], requires_grad=True),
        }
        scores = composer(preds)
        loss = sum(s.sum() for s in scores.values())
        loss.backward()
        assert preds["contact_occurred"].grad is not None
