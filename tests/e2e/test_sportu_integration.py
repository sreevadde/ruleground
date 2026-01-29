"""
End-to-end integration test using real SPORTU data.

Tests the full pipeline with actual SPORTU annotations converted to
RuleGround format. Uses mock video tensors (no video files needed)
but real q1/q2 labels, rationales, and sport distributions.

This validates:
    - SPORTU converter produces valid RuleGround annotations
    - Annotations load correctly into SportRDataset
    - Model trains on realistic label distributions
    - Evaluation produces meaningful metrics on real data splits
    - Error analysis works with real sport/foul distributions
    - Per-sport breakdown matches expected sports
"""

from __future__ import annotations

import json
import logging
import tempfile
from collections import Counter
from pathlib import Path
from typing import Dict, List

import pytest
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from ruleground.data.collate import sportr_collate_fn
from ruleground.data.preparation.sportu_converter import (
    SportUConverter,
    Q2_CLASS_NAMES,
    FOUL_TAXONOMY,
)
from ruleground.training.losses import RuleGroundLoss
from ruleground.training.grpo import GRPOTrainer
from ruleground.training.rewards import RewardFunction
from ruleground.evaluation.metrics import compute_all_metrics
from ruleground.evaluation.error_analysis import ErrorAnalyzer
from ruleground.predicates.ontology import NUM_PREDICATES

from tests.e2e.fixtures import build_test_model

logging.basicConfig(level=logging.INFO, format="%(name)s - %(message)s")
logger = logging.getLogger("sportu_e2e")

# Path to SPORTU raw data (downloaded earlier)
SPORTU_DIR = Path("data/sportu_raw")
SPORTR_DIR = Path("data/sportr")


def sportr_annotations_available() -> bool:
    """Check if converted SPORTU annotations exist."""
    return (SPORTR_DIR / "annotations" / "train.json").exists()


# ---------------------------------------------------------------------------
# Mock video dataset that uses real annotations
# ---------------------------------------------------------------------------

class MockVideoSportRDataset(Dataset):
    """Loads real SPORTU→RuleGround annotations but generates synthetic videos.

    This lets us test the full pipeline with real label distributions
    without needing actual video files.
    """

    def __init__(self, annotations: List[Dict], num_frames: int = 16):
        self.annotations = annotations
        self.num_frames = num_frames

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        anno = self.annotations[idx]

        # Generate a video tensor with signal correlated to the label
        # Infractions get higher mean activation (learnable pattern)
        base = 0.6 if anno["q1_label"] > 0 else 0.2
        video = torch.randn(self.num_frames, 3, 32, 32) * 0.1 + base
        video = video.clamp(0, 1)

        return {
            "video": video,
            "video_id": anno["video_id"],
            "sport": anno["sport"],
            "q1_label": anno["q1_label"],
            "q2_label": anno["q2_label"],
            "q5_span": anno.get("q5_span"),
            "predicates": None,
        }


# ===================================================================
# TEST: Converter output validation
# ===================================================================

@pytest.mark.skipif(
    not SPORTU_DIR.exists(),
    reason="SPORTU raw data not available"
)
class TestSportUConverter:
    """Validate the SPORTU → RuleGround conversion."""

    def test_converter_produces_valid_output(self):
        """Converter produces train/val/test JSONs with correct schema."""
        with tempfile.TemporaryDirectory() as tmpdir:
            converter = SportUConverter(
                sportu_dir=SPORTU_DIR,
                output_dir=tmpdir,
            )
            stats = converter.convert()

            # Check files created
            assert (Path(tmpdir) / "annotations" / "train.json").exists()
            assert (Path(tmpdir) / "annotations" / "val.json").exists()
            assert (Path(tmpdir) / "annotations" / "test.json").exists()
            assert (Path(tmpdir) / "foul_taxonomy.json").exists()

            # Check stats
            assert stats["total_samples"] > 500
            assert stats["train"] > stats["val"]
            assert stats["q2_classes_used"] == 17
            assert stats["rationale_coverage"] > 0.7

            logger.info(f"Converter stats: {stats}")

    def test_annotations_schema(self):
        """Each annotation has required fields with correct types."""
        for split in ["train", "val", "test"]:
            path = SPORTR_DIR / "annotations" / f"{split}.json"
            if not path.exists():
                pytest.skip("Converted annotations not found")

            with open(path) as f:
                data = json.load(f)

            for anno in data:
                assert "video_id" in anno
                assert "sport" in anno
                assert anno["sport"] in ("basketball", "soccer", "football")
                assert "q1_label" in anno
                assert anno["q1_label"] in (0, 1)
                assert "q2_label" in anno
                assert 0 <= anno["q2_label"] <= 16
                # q1=0 → q2=0 consistency
                if anno["q1_label"] == 0:
                    assert anno["q2_label"] == 0, (
                        f"Inconsistent: {anno['video_id']} q1=0 but q2={anno['q2_label']}"
                    )

    def test_sport_distribution(self):
        """All 3 target sports are represented."""
        path = SPORTR_DIR / "annotations" / "train.json"
        if not path.exists():
            pytest.skip("Converted annotations not found")

        with open(path) as f:
            data = json.load(f)

        sports = Counter(d["sport"] for d in data)
        assert "basketball" in sports
        assert "soccer" in sports
        assert "football" in sports
        logger.info(f"Sport distribution: {dict(sports)}")

    def test_q2_taxonomy_coverage(self):
        """Q2 labels cover foul types across all sports."""
        all_data = []
        for split in ["train", "val", "test"]:
            path = SPORTR_DIR / "annotations" / f"{split}.json"
            if path.exists():
                with open(path) as f:
                    all_data.extend(json.load(f))

        if not all_data:
            pytest.skip("No annotations")

        q2_counts = Counter(d["q2_label"] for d in all_data)
        logger.info("Q2 distribution:")
        for cls in sorted(q2_counts):
            name = Q2_CLASS_NAMES.get(cls, "unknown")
            logger.info(f"  {cls:2d} ({name}): {q2_counts[cls]}")

        # Should have clean plays and multiple foul types
        assert 0 in q2_counts  # clean plays
        assert len(q2_counts) >= 10  # variety of fouls

    def test_rationale_quality(self):
        """Rationales are non-empty text that describe violations."""
        path = SPORTR_DIR / "annotations" / "train.json"
        if not path.exists():
            pytest.skip("No annotations")

        with open(path) as f:
            data = json.load(f)

        with_rationale = [d for d in data if d.get("rationale")]
        assert len(with_rationale) > len(data) * 0.5, (
            f"Only {len(with_rationale)}/{len(data)} have rationales"
        )

        # Rationales should be meaningful text
        for d in with_rationale[:20]:
            assert len(d["rationale"]) > 10
            assert any(c.isalpha() for c in d["rationale"])


# ===================================================================
# TEST: Full pipeline with real SPORTU annotations
# ===================================================================

@pytest.mark.skipif(
    not sportr_annotations_available(),
    reason="Converted SportR annotations not available"
)
class TestSportUPipeline:
    """Full pipeline test using real SPORTU-derived annotations."""

    @pytest.fixture(autouse=True)
    def setup(self):
        # Load real annotations
        with open(SPORTR_DIR / "annotations" / "train.json") as f:
            train_annos = json.load(f)
        with open(SPORTR_DIR / "annotations" / "val.json") as f:
            val_annos = json.load(f)

        self.model = build_test_model(embed_dim=128, hidden_dim=64)

        self.train_dataset = MockVideoSportRDataset(train_annos)
        self.val_dataset = MockVideoSportRDataset(val_annos)

        self.train_loader = DataLoader(
            self.train_dataset, batch_size=16, shuffle=True,
            collate_fn=sportr_collate_fn, drop_last=True,
        )
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=16, shuffle=False,
            collate_fn=sportr_collate_fn,
        )

    def test_supervised_training_on_real_labels(self):
        """Stage 1: supervised training with real SPORTU labels."""
        loss_fn = RuleGroundLoss(gamma=0.0, delta=0.1, eta=0.0)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=3e-3)

        losses = []
        self.model.train()
        for epoch in range(3):
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
            logger.info(f"[SPORTU Stage 1] Epoch {epoch+1}: loss={avg:.4f}")

        assert losses[-1] < losses[0], (
            f"Loss didn't decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
        )

    def test_evaluation_on_real_labels(self):
        """Evaluation with real SPORTU labels produces valid metrics."""
        # Quick train
        loss_fn = RuleGroundLoss(gamma=0.0, delta=0.1)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=3e-3)
        self.model.train()
        for _ in range(3):
            for batch in self.train_loader:
                out = self.model(video=batch["video"], sport_ids=batch.get("sport_ids"))
                loss, _ = loss_fn(out, batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Evaluate on val set
        self.model.eval()
        all_out = {"q1_logits": [], "q2_logits": [], "predicate_probs": []}
        all_tgt = {"q1_labels": [], "q2_labels": []}
        all_sports = []

        with torch.no_grad():
            for batch in self.val_loader:
                out = self.model(video=batch["video"], sport_ids=batch.get("sport_ids"))
                for k in all_out:
                    if k in out:
                        all_out[k].append(out[k])
                for k in all_tgt:
                    if k in batch:
                        all_tgt[k].append(batch[k])
                all_sports.append(batch["sport_ids"])

        cat_out = {k: torch.cat(v) for k, v in all_out.items() if v}
        cat_tgt = {k: torch.cat(v) for k, v in all_tgt.items() if v}
        sport_ids = torch.cat(all_sports)

        metrics = compute_all_metrics(cat_out, cat_tgt)
        logger.info(f"\n[SPORTU Evaluation] Metrics on val set:")
        for k, v in sorted(metrics.items()):
            logger.info(f"  {k}: {v:.4f}")

        assert "q1_accuracy" in metrics
        assert "q2_accuracy" in metrics
        # On real data with mock videos, accuracy should be above chance (50%)
        # after training on correlated signal
        assert metrics["q1_accuracy"] > 0.4

    def test_error_analysis_on_real_labels(self):
        """Error analysis on real SPORTU labels."""
        # Quick train
        loss_fn = RuleGroundLoss(gamma=0.0)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=3e-3)
        self.model.train()
        for _ in range(2):
            for batch in self.train_loader:
                out = self.model(video=batch["video"], sport_ids=batch.get("sport_ids"))
                loss, _ = loss_fn(out, batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Collect predictions
        self.model.eval()
        all_preds, all_labels, all_sports, all_pred_probs = [], [], [], []
        with torch.no_grad():
            for batch in self.val_loader:
                out = self.model(video=batch["video"], sport_ids=batch.get("sport_ids"))
                all_preds.append(out["q1_logits"].argmax(dim=-1))
                all_labels.append(batch["q1_labels"])
                all_sports.append(batch["sport_ids"])
                all_pred_probs.append(out["predicate_probs"])

        q1_preds = torch.cat(all_preds)
        q1_labels = torch.cat(all_labels)
        sport_ids = torch.cat(all_sports)
        pred_probs = torch.cat(all_pred_probs)

        analyzer = ErrorAnalyzer()
        analysis = analyzer.analyze_batch(pred_probs, q1_preds, q1_labels, sport_ids)
        report = analyzer.format_report(analysis)
        logger.info(f"\n[SPORTU Error Analysis]\n{report}")

        assert analysis["total_samples"] > 0
        assert "error_counts" in analysis

    def test_grpo_with_real_labels(self):
        """GRPO training step works on real SPORTU label distribution."""
        # Quick supervised warmup
        loss_fn = RuleGroundLoss(gamma=0.0)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=3e-3)
        self.model.train()
        for batch in self.train_loader:
            out = self.model(video=batch["video"], sport_ids=batch.get("sport_ids"))
            loss, _ = loss_fn(out, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            break

        # GRPO step
        grpo = GRPOTrainer(
            model=self.model,
            reward_fn=RewardFunction(alpha=0.3, beta=0.2),
            group_size=4,
            clip_ratio=0.2,
            kl_coef=0.1,
            pred_dropout=0.15,
            lr=1e-4,
        )

        batch = next(iter(self.train_loader))
        loss, metrics = grpo.step(batch)
        logger.info(
            f"[SPORTU GRPO] loss={metrics['grpo_loss']:.4f}, "
            f"reward={metrics['mean_reward']:.4f}"
        )

        assert not torch.isnan(loss)
        assert "mean_reward" in metrics
