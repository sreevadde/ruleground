"""Tests for RGM and its sub-components (temporal pool, predicate head, snapformer).

These tests use synthetic tensors and do not require GPU or pretrained weights.
ActionFormer is required as a dependency.
"""

import torch
import pytest

from ruleground.predicates.ontology import (
    NUM_PREDICATES,
    ALL_PREDICATE_NAMES,
    INSTANT_PREDICATES,
    get_state_predicate_indices,
    get_instant_predicate_indices,
    SPORT_TO_ID,
    Sport,
)


# ---------------------------------------------------------------------------
# Temporal Attention Pool
# ---------------------------------------------------------------------------

class TestTemporalAttentionPool:

    @pytest.fixture
    def pool(self):
        from ruleground.models.temporal_pool import TemporalAttentionPool
        return TemporalAttentionPool(embed_dim=64, num_heads=4, max_seq_len=32)

    def test_output_shape(self, pool):
        x = torch.randn(2, 16, 64)
        out = pool(x)
        assert out.shape == (2, 64)

    def test_with_mask(self, pool):
        x = torch.randn(2, 16, 64)
        mask = torch.ones(2, 16, dtype=torch.bool)
        mask[0, 12:] = False  # First sample has shorter sequence
        out = pool(x, mask=mask)
        assert out.shape == (2, 64)

    def test_gradient_flow(self, pool):
        x = torch.randn(2, 16, 64, requires_grad=True)
        out = pool(x)
        out.sum().backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_single_frame(self, pool):
        x = torch.randn(1, 1, 64)
        out = pool(x)
        assert out.shape == (1, 64)


# ---------------------------------------------------------------------------
# Predicate Head
# ---------------------------------------------------------------------------

class TestPredicateHead:

    @pytest.fixture
    def head(self):
        from ruleground.models.predicate_head import PredicateHead
        state_indices = get_state_predicate_indices()
        state_names = [ALL_PREDICATE_NAMES[i] for i in state_indices]
        return PredicateHead(embed_dim=64, hidden_dim=32, predicate_names=state_names)

    def test_output_shape(self, head):
        x = torch.randn(2, 64)
        out = head(x)
        assert out["logits"].shape == (2, head.num_predicates)
        assert out["probs"].shape == (2, head.num_predicates)

    def test_probs_in_range(self, head):
        x = torch.randn(4, 64)
        out = head(x)
        assert (out["probs"] >= 0).all()
        assert (out["probs"] <= 1).all()

    def test_sport_masking(self, head):
        x = torch.randn(2, 64)
        sport_ids = torch.tensor([SPORT_TO_ID[Sport.BASKETBALL], SPORT_TO_ID[Sport.SOCCER]])
        out = head(x, sport_ids=sport_ids)
        # Both samples should have valid outputs
        assert out["logits"].shape == (2, head.num_predicates)

    def test_predicate_dict_keys(self, head):
        x = torch.randn(2, 64)
        out = head(x)
        for name in head.predicate_names:
            assert name in out["predicate_dict"]
            assert out["predicate_dict"][name].shape == (2,)

    def test_per_predicate_independence(self, head):
        """Gradient of one predicate should not flow through another's classifier."""
        x = torch.randn(1, 64, requires_grad=True)
        out = head(x)
        # Pick two different predicate classifiers
        names = head.predicate_names
        if len(names) < 2:
            pytest.skip("Need at least 2 predicates")

        # Backprop through first predicate only
        out["logits"][0, 0].backward(retain_graph=True)

        # Get the classifier params for the second predicate
        second_name = names[1]
        second_params = list(head.classifiers[second_name].parameters())
        for p in second_params:
            if p.grad is not None:
                assert torch.all(p.grad == 0), "Cross-predicate gradient leak"


# ---------------------------------------------------------------------------
# SnapFormer Head
# ---------------------------------------------------------------------------

class TestSnapFormerHead:

    @pytest.fixture
    def snap_head(self):
        from ruleground.models.snapformer_head import SnapFormerHead
        return SnapFormerHead(embed_dim=64, hidden_dim=32)

    def test_output_shapes(self, snap_head):
        x = torch.randn(2, 16, 64)
        out = snap_head(x)
        K = snap_head.num_predicates
        assert out["logits"].shape == (2, K)
        assert out["probs"].shape == (2, K)
        assert out["frame_logits"].shape == (2, 16, K)
        assert out["frame_activations"].shape == (2, 16, K)

    def test_frame_activations_in_range(self, snap_head):
        x = torch.randn(4, 16, 64)
        out = snap_head(x)
        # After sigmoid, frame activations should be in [0, 1]
        fa = out["frame_activations"]
        assert (fa >= 0).all()
        assert (fa <= 1).all()

    def test_clip_logits_are_temporal_max(self, snap_head):
        x = torch.randn(2, 16, 64)
        out = snap_head(x)
        # Clip logits should be max over time dimension
        expected_max = out["frame_logits"].max(dim=1).values
        assert torch.allclose(out["logits"], expected_max, atol=1e-5)

    def test_only_instant_predicates(self, snap_head):
        for name in snap_head.predicate_names:
            assert name in INSTANT_PREDICATES, f"{name} is not instant"


# ---------------------------------------------------------------------------
# Full RGM
# ---------------------------------------------------------------------------

class TestRuleGroundingModule:

    @pytest.fixture
    def rgm(self):
        from ruleground.models.rgm import RuleGroundingModule
        return RuleGroundingModule(embed_dim=64, num_heads=4, hidden_dim=32)

    def test_output_shapes(self, rgm):
        x = torch.randn(2, 16, 64)
        out = rgm(x)
        assert out["logits"].shape == (2, NUM_PREDICATES)
        assert out["probs"].shape == (2, NUM_PREDICATES)
        assert out["pooled"].shape == (2, 64)
        # frame_activations shape: [B, T, K_instant]
        K_instant = len(get_instant_predicate_indices())
        assert out["frame_activations"].shape == (2, 16, K_instant)

    def test_predicate_dict_complete(self, rgm):
        x = torch.randn(2, 16, 64)
        out = rgm(x)
        # Should have entries for all 20 predicates
        assert len(out["predicate_dict"]) == NUM_PREDICATES

    def test_probs_in_range(self, rgm):
        x = torch.randn(4, 16, 64)
        out = rgm(x)
        valid_mask = out["probs"] > -1  # Exclude -inf masked entries
        valid_probs = out["probs"][valid_mask]
        assert (valid_probs >= 0).all()
        assert (valid_probs <= 1).all()

    def test_sport_masking_propagates(self, rgm):
        x = torch.randn(2, 16, 64)
        sport_ids = torch.tensor([0, 2])  # basketball, soccer
        out = rgm(x, sport_ids=sport_ids)
        assert out["logits"].shape == (2, NUM_PREDICATES)

    def test_gradient_flow_end_to_end(self, rgm):
        x = torch.randn(2, 16, 64, requires_grad=True)
        out = rgm(x)
        loss = out["probs"].sum() + out["pooled"].sum()
        loss.backward()
        assert x.grad is not None

    def test_with_mask(self, rgm):
        x = torch.randn(2, 16, 64)
        mask = torch.ones(2, 16, dtype=torch.bool)
        mask[0, 10:] = False
        out = rgm(x, mask=mask)
        assert out["logits"].shape == (2, NUM_PREDICATES)


# ---------------------------------------------------------------------------
# Reasoning Head
# ---------------------------------------------------------------------------

class TestReasoningHead:

    @pytest.fixture
    def head(self):
        from ruleground.models.reasoning_head import ReasoningHead
        return ReasoningHead(
            num_predicates=NUM_PREDICATES,
            visual_dim=64,
            hidden_dim=32,
            num_q1_classes=2,
            num_q2_classes=10,
            num_heads=2,
        )

    def test_output_shapes(self, head):
        pred_probs = torch.randn(2, NUM_PREDICATES).sigmoid()
        pooled = torch.randn(2, 64)
        out = head(pred_probs, pooled)
        assert out["q1_logits"].shape == (2, 2)
        assert out["q2_logits"].shape == (2, 10)
        assert out["q5_preds"].shape == (2, 2)

    def test_q5_in_range(self, head):
        pred_probs = torch.randn(4, NUM_PREDICATES).sigmoid()
        pooled = torch.randn(4, 64)
        out = head(pred_probs, pooled)
        # Q5 uses sigmoid so should be in [0, 1]
        assert (out["q5_preds"] >= 0).all()
        assert (out["q5_preds"] <= 1).all()

    def test_with_rule_scores(self, head):
        pred_probs = torch.randn(2, NUM_PREDICATES).sigmoid()
        pooled = torch.randn(2, 64)
        rules = {"blocking_foul": torch.rand(2), "dpi": torch.rand(2)}
        out = head(pred_probs, pooled, rule_scores=rules)
        assert out["q1_logits"].shape == (2, 2)

    def test_gradient_flow(self, head):
        raw = torch.randn(2, NUM_PREDICATES, requires_grad=True)
        pred_probs = raw.sigmoid()
        pooled = torch.randn(2, 64, requires_grad=True)
        out = head(pred_probs, pooled)
        loss = out["q1_logits"].sum() + out["q2_logits"].sum() + out["q5_preds"].sum()
        loss.backward()
        assert raw.grad is not None  # gradient flows through sigmoid to leaf
        assert pooled.grad is not None
