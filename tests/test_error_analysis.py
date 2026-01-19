"""Tests for error analysis."""

import pytest
import torch

from ruleground.evaluation.error_analysis import ErrorAnalyzer, ErrorType
from ruleground.predicates.ontology import NUM_PREDICATES


class TestErrorAnalyzer:
    @pytest.fixture
    def analyzer(self):
        return ErrorAnalyzer(predicate_threshold=0.5, evidence_threshold=0.3)

    def test_correct_prediction(self, analyzer):
        pred_probs = torch.rand(NUM_PREDICATES)
        result = analyzer.classify_error(pred_probs, None, q1_pred=1, q1_label=1, sport_id=0)
        assert result == ErrorType.CORRECT

    def test_perception_error_low_activation(self, analyzer):
        """Low predicate activation = perception failure."""
        pred_probs = torch.full((NUM_PREDICATES,), 0.1)  # very low
        result = analyzer.classify_error(pred_probs, None, q1_pred=0, q1_label=1, sport_id=0)
        assert result == ErrorType.PERCEPTION

    def test_reasoning_error_high_activation(self, analyzer):
        """High predicate activation but wrong conclusion = reasoning failure."""
        pred_probs = torch.full((NUM_PREDICATES,), 0.8)  # high activation
        result = analyzer.classify_error(pred_probs, None, q1_pred=0, q1_label=1, sport_id=0)
        assert result == ErrorType.REASONING

    def test_batch_analysis(self, analyzer):
        B = 10
        pred_probs = torch.rand(B, NUM_PREDICATES)
        q1_preds = torch.randint(0, 2, (B,))
        q1_labels = torch.randint(0, 2, (B,))
        sport_ids = torch.zeros(B, dtype=torch.long)

        result = analyzer.analyze_batch(pred_probs, q1_preds, q1_labels, sport_ids)

        assert "error_counts" in result
        assert "total_errors" in result
        assert "total_samples" in result
        assert result["total_samples"] == B
        assert len(result["per_sample"]) == B

    def test_format_report(self, analyzer):
        analysis = {
            "total_samples": 100,
            "total_errors": 30,
            "accuracy": 0.7,
            "error_counts": {
                ErrorType.CORRECT: 70,
                ErrorType.PERCEPTION: 10,
                ErrorType.GROUNDING: 15,
                ErrorType.REASONING: 5,
            },
            "error_rates": {
                ErrorType.PERCEPTION: 0.333,
                ErrorType.GROUNDING: 0.5,
                ErrorType.REASONING: 0.167,
            },
        }
        report = analyzer.format_report(analysis)
        assert "Perception" in report
        assert "Grounding" in report
        assert "Reasoning" in report
        assert "70.00%" in report
