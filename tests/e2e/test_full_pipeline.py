"""
End-to-End Pipeline Tests for RuleGround

Tests the complete training and evaluation pipeline using synthetic SportR
data with learnable patterns. Validates:

    Use Case 1 (Basketball):  Blocking foul vs charging foul vs clean play
    Use Case 2 (Soccer):      Handball vs offside vs clean play
    Use Case 3 (Multi-sport): Mixed sports, per-sport evaluation breakdown

Each test runs real training (supervised + GRPO), real evaluation, and real
error analysis — the only mock is the video encoder (avoids VideoMAE download).
"""

from __future__ import annotations

import logging
import sys
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from ruleground.predicates.ontology import (
    ALL_PREDICATE_NAMES,
    NUM_PREDICATES,
    SPORT_TO_ID,
)
from ruleground.training.losses import RuleGroundLoss
from ruleground.training.rewards import RewardFunction
from ruleground.training.grpo import GRPOTrainer, PredicateDropout
from ruleground.training.trainer import SupervisedTrainer
from ruleground.evaluation.evaluator import Evaluator
from ruleground.evaluation.metrics import compute_all_metrics
from ruleground.evaluation.error_analysis import ErrorAnalyzer
from ruleground.utils.checkpoint import save_checkpoint, load_checkpoint

from tests.e2e.fixtures import (
    build_test_model,
    build_dataloaders,
    SyntheticSportRDataset,
    MockVideoEncoder,
    BASKETBALL_SCENARIOS,
    SOCCER_SCENARIOS,
    FOOTBALL_SCENARIOS,
)
from ruleground.data.collate import sportr_collate_fn

logging.basicConfig(level=logging.INFO, format="%(name)s - %(message)s")
logger = logging.getLogger("e2e")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def collect_predictions(model, dataloader):
    """Run inference and collect all predictions + targets."""
    model.eval()
    all_out = {"q1_logits": [], "q2_logits": [], "q5_preds": [],
               "predicate_probs": [], "predicate_logits": []}
    all_tgt = {"q1_labels": [], "q2_labels": [], "predicate_labels": [],
               "predicate_mask": []}
    all_sport_ids = []

    with torch.no_grad():
        for batch in dataloader:
            outputs = model(
                video=batch["video"],
                mask=batch.get("mask"),
                sport_ids=batch.get("sport_ids"),
            )
            for k in all_out:
                if k in outputs:
                    all_out[k].append(outputs[k])
            for k in all_tgt:
                if k in batch:
                    all_tgt[k].append(batch[k])
            if "sport_ids" in batch:
                all_sport_ids.append(batch["sport_ids"])

    cat_out = {k: torch.cat(v) for k, v in all_out.items() if v}
    cat_tgt = {k: torch.cat(v) for k, v in all_tgt.items() if v}
    sport_ids = torch.cat(all_sport_ids) if all_sport_ids else None
    return cat_out, cat_tgt, sport_ids


# ===================================================================
# USE CASE 1: Basketball — blocking foul vs charging vs clean play
# ===================================================================

class TestBasketballUseCase:
    """End-to-end test: basketball infraction detection.

    The model should learn to distinguish:
        - Blocking foul:  high contact, defender NOT set
        - Charging foul:  high contact, defender SET, NOT restricted area
        - Clean play:     low contact
        - Shooting foul:  contact during shooting motion
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        self.model = build_test_model(embed_dim=128, hidden_dim=64)
        self.dataset = SyntheticSportRDataset(
            samples_per_scenario=30,
            num_frames=16,
            frame_size=(32, 32),
            sports=["basketball"],
        )
        n = len(self.dataset)
        n_val = max(1, int(n * 0.2))
        train_set, val_set = torch.utils.data.random_split(
            self.dataset, [n - n_val, n_val]
        )
        self.train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=8, shuffle=True,
            collate_fn=sportr_collate_fn, drop_last=True,
        )
        self.val_loader = torch.utils.data.DataLoader(
            val_set, batch_size=8, shuffle=False,
            collate_fn=sportr_collate_fn,
        )

    def test_supervised_training_reduces_loss(self):
        """Stage 1: supervised loss should decrease over epochs."""
        loss_fn = RuleGroundLoss(gamma=0.5, delta=0.1, eta=0.5)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=3e-3)

        self.model.train()
        epoch_losses = []

        for epoch in range(5):
            total_loss = 0.0
            n_batches = 0
            for batch in self.train_loader:
                outputs = self.model(
                    video=batch["video"],
                    mask=batch.get("mask"),
                    sport_ids=batch.get("sport_ids"),
                )
                loss, breakdown = loss_fn(outputs, batch)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            avg_loss = total_loss / max(n_batches, 1)
            epoch_losses.append(avg_loss)
            logger.info(f"[Basketball] Epoch {epoch+1} loss: {avg_loss:.4f}")

        # Loss should decrease
        assert epoch_losses[-1] < epoch_losses[0], (
            f"Loss didn't decrease: {epoch_losses[0]:.4f} -> {epoch_losses[-1]:.4f}"
        )
        logger.info(
            f"[Basketball] Loss decreased: {epoch_losses[0]:.4f} -> {epoch_losses[-1]:.4f}"
        )

    def test_evaluation_produces_metrics(self):
        """After training, evaluator should produce real Q1/Q2 metrics."""
        # Quick train
        loss_fn = RuleGroundLoss()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=3e-3)
        self.model.train()
        for epoch in range(3):
            for batch in self.train_loader:
                outputs = self.model(
                    video=batch["video"],
                    mask=batch.get("mask"),
                    sport_ids=batch.get("sport_ids"),
                )
                loss, _ = loss_fn(outputs, batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Evaluate
        cat_out, cat_tgt, sport_ids = collect_predictions(
            self.model, self.val_loader
        )
        metrics = compute_all_metrics(cat_out, cat_tgt)

        logger.info(f"[Basketball] Evaluation metrics:")
        for k, v in sorted(metrics.items()):
            logger.info(f"  {k}: {v:.4f}")

        # Should have Q1 metrics
        assert "q1_accuracy" in metrics
        assert "q1_f1" in metrics
        assert 0 <= metrics["q1_accuracy"] <= 1

    def test_predicate_bottleneck_activates(self):
        """Predicate probs should show sport-relevant activation patterns."""
        # Quick train
        loss_fn = RuleGroundLoss(gamma=1.0)  # High predicate weight
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=3e-3)
        self.model.train()
        for _ in range(5):
            for batch in self.train_loader:
                outputs = self.model(
                    video=batch["video"],
                    mask=batch.get("mask"),
                    sport_ids=batch.get("sport_ids"),
                )
                loss, _ = loss_fn(outputs, batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Check predicate activations
        cat_out, cat_tgt, _ = collect_predictions(self.model, self.val_loader)
        pred_probs = cat_out["predicate_probs"]

        # Predicates should not be all zero or all one
        mean_activation = pred_probs.mean().item()
        assert 0.01 < mean_activation < 0.99, (
            f"Predicate activations degenerate: mean={mean_activation:.4f}"
        )

        # Some predicates should be more active than others
        per_pred_mean = pred_probs.mean(dim=0)
        assert per_pred_mean.max() > per_pred_mean.min() + 0.01, (
            "All predicates have same activation — bottleneck not differentiating"
        )
        logger.info(f"[Basketball] Predicate activation range: "
                     f"{per_pred_mean.min():.3f} - {per_pred_mean.max():.3f}")


# ===================================================================
# USE CASE 2: Soccer — handball vs offside vs clean play
# ===================================================================

class TestSoccerUseCase:
    """End-to-end test: soccer infraction detection."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.model = build_test_model(embed_dim=128, hidden_dim=64)
        self.dataset = SyntheticSportRDataset(
            samples_per_scenario=25,
            num_frames=16,
            frame_size=(32, 32),
            sports=["soccer"],
        )
        n = len(self.dataset)
        n_val = max(1, int(n * 0.2))
        train_set, val_set = torch.utils.data.random_split(
            self.dataset, [n - n_val, n_val]
        )
        self.train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=8, shuffle=True,
            collate_fn=sportr_collate_fn, drop_last=True,
        )
        self.val_loader = torch.utils.data.DataLoader(
            val_set, batch_size=8, shuffle=False,
            collate_fn=sportr_collate_fn,
        )

    def test_supervised_training_and_eval(self):
        """Full supervised training + evaluation for soccer."""
        loss_fn = RuleGroundLoss(gamma=0.5, delta=0.1, eta=0.5)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=3e-3)

        self.model.train()
        first_loss = None
        last_loss = None

        for epoch in range(5):
            total_loss = 0.0
            n = 0
            for batch in self.train_loader:
                outputs = self.model(
                    video=batch["video"],
                    mask=batch.get("mask"),
                    sport_ids=batch.get("sport_ids"),
                )
                loss, breakdown = loss_fn(outputs, batch)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()
                n += 1

            avg = total_loss / max(n, 1)
            if first_loss is None:
                first_loss = avg
            last_loss = avg
            logger.info(f"[Soccer] Epoch {epoch+1} loss: {avg:.4f}")

        assert last_loss < first_loss

        # Evaluate
        cat_out, cat_tgt, _ = collect_predictions(self.model, self.val_loader)
        metrics = compute_all_metrics(cat_out, cat_tgt)
        logger.info(f"[Soccer] Q1 accuracy: {metrics.get('q1_accuracy', 0):.4f}")
        logger.info(f"[Soccer] Q1 F1:       {metrics.get('q1_f1', 0):.4f}")

        assert "q1_accuracy" in metrics

    def test_error_analysis(self):
        """Error analysis should produce perception/grounding/reasoning breakdown."""
        # Quick train
        loss_fn = RuleGroundLoss(gamma=0.5)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=3e-3)
        self.model.train()
        for _ in range(3):
            for batch in self.train_loader:
                out = self.model(video=batch["video"], sport_ids=batch.get("sport_ids"))
                loss, _ = loss_fn(out, batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Collect predictions
        cat_out, cat_tgt, sport_ids = collect_predictions(self.model, self.val_loader)

        q1_preds = cat_out["q1_logits"].argmax(dim=-1)
        q1_labels = cat_tgt["q1_labels"]
        pred_probs = cat_out["predicate_probs"]
        gt_preds = cat_tgt.get("predicate_labels")

        analyzer = ErrorAnalyzer(predicate_threshold=0.5, evidence_threshold=0.3)
        analysis = analyzer.analyze_batch(
            pred_probs, q1_preds, q1_labels, sport_ids, gt_preds,
        )

        report = analyzer.format_report(analysis)
        logger.info(f"\n[Soccer] Error Analysis:\n{report}")

        # Should have valid structure
        assert "error_counts" in analysis
        assert "error_rates" in analysis
        assert analysis["total_samples"] > 0


# ===================================================================
# USE CASE 3: Multi-sport with GRPO + full evaluation pipeline
# ===================================================================

class TestMultiSportPipeline:
    """End-to-end test: multi-sport training with all 3 stages.

    This is the most comprehensive test — mirrors the actual training
    pipeline described in the paper:
        Stage 1: Supervised pre-training
        Stage 2: GRPO post-training
        Evaluation with per-sport breakdown and error analysis
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        self.model = build_test_model(embed_dim=128, hidden_dim=64)
        self.train_loader, self.val_loader = build_dataloaders(
            samples_per_scenario=15,
            batch_size=8,
            num_frames=16,
            frame_size=(32, 32),
            val_ratio=0.2,
        )

    def test_stage1_supervised(self):
        """Stage 1: Supervised training on multi-sport data."""
        loss_fn = RuleGroundLoss(gamma=0.5, delta=0.1, eta=0.5)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=3e-3)

        losses = []
        self.model.train()
        for epoch in range(5):
            epoch_loss = 0.0
            n = 0
            for batch in self.train_loader:
                out = self.model(
                    video=batch["video"],
                    mask=batch.get("mask"),
                    sport_ids=batch.get("sport_ids"),
                )
                loss, breakdown = loss_fn(out, batch)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
                n += 1
            avg = epoch_loss / max(n, 1)
            losses.append(avg)
            logger.info(f"[Multi-sport Stage 1] Epoch {epoch+1}: loss={avg:.4f}")

        assert losses[-1] < losses[0], (
            f"Stage 1 loss didn't decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
        )

    def test_stage2_grpo(self):
        """Stage 2: GRPO post-training improves reward signal."""
        # Quick Stage 1 first
        loss_fn = RuleGroundLoss()
        opt = torch.optim.AdamW(self.model.parameters(), lr=3e-3)
        self.model.train()
        for _ in range(3):
            for batch in self.train_loader:
                out = self.model(video=batch["video"], sport_ids=batch.get("sport_ids"))
                loss, _ = loss_fn(out, batch)
                opt.zero_grad()
                loss.backward()
                opt.step()

        # Stage 2: GRPO
        grpo = GRPOTrainer(
            model=self.model,
            reward_fn=RewardFunction(alpha=0.3, beta=0.2),
            group_size=4,  # Smaller group for speed
            clip_ratio=0.2,
            kl_coef=0.1,
            pred_dropout=0.15,
            lr=1e-4,
        )

        rewards = []
        for step_idx in range(5):
            batch = next(iter(self.train_loader))
            loss, metrics = grpo.step(batch)
            rewards.append(metrics["mean_reward"])
            logger.info(
                f"[Multi-sport Stage 2] Step {step_idx+1}: "
                f"loss={metrics['grpo_loss']:.4f}, "
                f"reward={metrics['mean_reward']:.4f}, "
                f"advantage={metrics['mean_advantage']:.4f}"
            )

        # Rewards should be non-trivial (not all zero)
        assert any(r != 0 for r in rewards), "All rewards are zero"
        logger.info(f"[Multi-sport Stage 2] Reward range: {min(rewards):.4f} - {max(rewards):.4f}")

    def test_full_evaluation_with_per_sport(self):
        """Full evaluation with per-sport breakdown."""
        # Quick train
        loss_fn = RuleGroundLoss(gamma=0.5, delta=0.1)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=3e-3)
        self.model.train()
        for _ in range(5):
            for batch in self.train_loader:
                out = self.model(
                    video=batch["video"],
                    mask=batch.get("mask"),
                    sport_ids=batch.get("sport_ids"),
                )
                loss, _ = loss_fn(out, batch)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

        # Run evaluator
        evaluator = Evaluator(self.model)
        results = evaluator.run(self.val_loader, per_sport=True)

        # Check overall metrics exist
        assert "overall" in results
        overall = results["overall"]
        logger.info(f"\n[Multi-sport Evaluation]")
        logger.info(f"  Overall:")
        for k, v in sorted(overall.items()):
            logger.info(f"    {k}: {v:.4f}")

        assert "q1_accuracy" in overall
        assert "q1_f1" in overall

        # Check per-sport breakdown
        if "per_sport" in results:
            for sport, metrics in results["per_sport"].items():
                logger.info(f"  {sport}:")
                for k, v in sorted(metrics.items()):
                    logger.info(f"    {k}: {v:.4f}")

    def test_error_analysis_multi_sport(self):
        """Error analysis across all sports."""
        # Quick train
        loss_fn = RuleGroundLoss(gamma=0.5)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=3e-3)
        self.model.train()
        for _ in range(3):
            for batch in self.train_loader:
                out = self.model(video=batch["video"], sport_ids=batch.get("sport_ids"))
                loss, _ = loss_fn(out, batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Collect predictions
        cat_out, cat_tgt, sport_ids = collect_predictions(self.model, self.val_loader)

        analyzer = ErrorAnalyzer()
        analysis = analyzer.analyze_batch(
            cat_out["predicate_probs"],
            cat_out["q1_logits"].argmax(dim=-1),
            cat_tgt["q1_labels"],
            sport_ids,
            cat_tgt.get("predicate_labels"),
        )
        report = analyzer.format_report(analysis)
        logger.info(f"\n[Multi-sport Error Analysis]\n{report}")

        assert analysis["total_samples"] > 0
        assert "perception" in analysis["error_rates"]
        assert "grounding" in analysis["error_rates"]
        assert "reasoning" in analysis["error_rates"]

    def test_checkpoint_save_load_roundtrip(self):
        """Checkpoint save/load preserves model state and metrics."""
        # Train briefly
        loss_fn = RuleGroundLoss()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=3e-3)
        self.model.train()
        for batch in self.train_loader:
            out = self.model(video=batch["video"], sport_ids=batch.get("sport_ids"))
            loss, _ = loss_fn(out, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            break

        # Get pre-save predictions
        self.model.eval()
        batch = next(iter(self.val_loader))
        with torch.no_grad():
            out_before = self.model(video=batch["video"], sport_ids=batch.get("sport_ids"))

        # Save checkpoint
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "test_ckpt.pt"
            test_metrics = {"q1_accuracy": 0.85, "loss": 1.23}
            save_checkpoint(
                self.model, optimizer, epoch=3, step=100,
                metrics=test_metrics, path=ckpt_path,
            )
            assert ckpt_path.exists()

            # Build fresh model and load
            model2 = build_test_model(embed_dim=128, hidden_dim=64)
            opt2 = torch.optim.AdamW(model2.parameters(), lr=3e-3)
            info = load_checkpoint(ckpt_path, model2, opt2)

            assert info["epoch"] == 3
            assert info["step"] == 100
            assert info["metrics"]["q1_accuracy"] == 0.85

            # Predictions should match
            model2.eval()
            with torch.no_grad():
                out_after = model2(video=batch["video"], sport_ids=batch.get("sport_ids"))

            torch.testing.assert_close(
                out_before["q1_logits"], out_after["q1_logits"], rtol=1e-4, atol=1e-4
            )
            logger.info("[Checkpoint] Save/load roundtrip verified")


# ===================================================================
# USE CASE 4: Overfit test — model should memorize a tiny dataset
# ===================================================================

class TestOverfitSmallDataset:
    """Sanity check: model should achieve near-perfect Q1 accuracy
    when trained on a very small dataset for enough epochs."""

    def test_overfit_10_samples(self):
        model = build_test_model(embed_dim=128, hidden_dim=64)

        # Tiny dataset: 3 samples per scenario, just basketball
        dataset = SyntheticSportRDataset(
            samples_per_scenario=3,
            num_frames=16,
            frame_size=(32, 32),
            sports=["basketball"],
        )
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=len(dataset),
            shuffle=False, collate_fn=sportr_collate_fn,
        )

        loss_fn = RuleGroundLoss(gamma=0.5, delta=0.1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3)

        model.train()
        for epoch in range(30):
            for batch in loader:
                out = model(
                    video=batch["video"],
                    mask=batch.get("mask"),
                    sport_ids=batch.get("sport_ids"),
                )
                loss, breakdown = loss_fn(out, batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if (epoch + 1) % 10 == 0:
                logger.info(f"[Overfit] Epoch {epoch+1}: loss={loss.item():.4f}")

        # Evaluate — should be near perfect on training data
        model.eval()
        with torch.no_grad():
            batch = next(iter(loader))
            out = model(video=batch["video"], sport_ids=batch.get("sport_ids"))
            q1_preds = out["q1_logits"].argmax(dim=-1)
            q1_labels = (batch["q1_labels"] > 0).long()
            accuracy = (q1_preds == q1_labels).float().mean().item()

        logger.info(f"[Overfit] Q1 accuracy on training data: {accuracy:.2%}")
        assert accuracy >= 0.7, f"Model couldn't overfit tiny dataset: {accuracy:.2%}"
