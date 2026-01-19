"""Tests for data pipeline components."""

import pytest
import torch

from ruleground.data.transforms import (
    Resize,
    Normalize,
    RandomHorizontalFlip,
    UniformTemporalSubsample,
    ColorJitter,
    get_train_transform,
    get_eval_transform,
)
from ruleground.data.collate import sportr_collate_fn
from ruleground.predicates.ontology import NUM_PREDICATES


class TestTransforms:
    def test_resize(self):
        video = torch.rand(16, 3, 128, 128)
        t = Resize((224, 224))
        out = t(video)
        assert out.shape == (16, 3, 224, 224)

    def test_normalize(self):
        video = torch.rand(16, 3, 224, 224)
        t = Normalize()
        out = t(video)
        assert out.shape == (16, 3, 224, 224)
        # After normalization, mean should be near 0
        assert out.mean().abs() < 2.0

    def test_horizontal_flip(self):
        torch.manual_seed(0)
        video = torch.arange(4).float().view(1, 1, 1, 4)
        t = RandomHorizontalFlip(p=1.0)  # always flip
        out = t(video)
        assert torch.equal(out, video.flip(-1))

    def test_temporal_subsample(self):
        video = torch.rand(32, 3, 64, 64)
        t = UniformTemporalSubsample(16)
        out = t(video)
        assert out.shape[0] == 16

    def test_temporal_subsample_short(self):
        """If video is already shorter, don't change it."""
        video = torch.rand(8, 3, 64, 64)
        t = UniformTemporalSubsample(16)
        out = t(video)
        assert out.shape[0] == 16

    def test_color_jitter(self):
        video = torch.rand(4, 3, 32, 32)
        t = ColorJitter(brightness=0.2, contrast=0.2)
        out = t(video)
        assert out.shape == video.shape
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_train_transform_pipeline(self):
        video = torch.rand(32, 3, 128, 128)
        t = get_train_transform((224, 224), 16)
        out = t(video)
        assert out.shape == (16, 3, 224, 224)

    def test_eval_transform_pipeline(self):
        video = torch.rand(32, 3, 128, 128)
        t = get_eval_transform((224, 224), 16)
        out = t(video)
        assert out.shape == (16, 3, 224, 224)


class TestCollate:
    def _make_sample(self, vid_id, sport="basketball", q1=0, q2=0, with_preds=False):
        sample = {
            "video": torch.rand(16, 3, 224, 224),
            "video_id": vid_id,
            "q1_label": q1,
            "q2_label": q2,
            "sport": sport,
            "q5_span": None,
            "predicates": None,
        }
        if with_preds:
            sample["predicates"] = {
                "labels": {"contact_occurred": 1.0, "defender_set": 0.0},
                "weights": {"contact_occurred": 0.95, "defender_set": 0.8},
            }
        return sample

    def test_basic_collation(self):
        batch = [self._make_sample(f"v{i}") for i in range(4)]
        collated = sportr_collate_fn(batch)

        assert collated["video"].shape == (4, 16, 3, 224, 224)
        assert collated["q1_labels"].shape == (4,)
        assert collated["q2_labels"].shape == (4,)
        assert collated["sport_ids"].shape == (4,)
        assert collated["mask"].shape == (4, 16)

    def test_sport_id_mapping(self):
        batch = [
            self._make_sample("v0", sport="basketball"),
            self._make_sample("v1", sport="football"),
            self._make_sample("v2", sport="soccer"),
        ]
        collated = sportr_collate_fn(batch)
        assert collated["sport_ids"].tolist() == [0, 1, 2]

    def test_predicate_labels(self):
        batch = [self._make_sample(f"v{i}", with_preds=True) for i in range(2)]
        collated = sportr_collate_fn(batch)

        assert "predicate_labels" in collated
        assert collated["predicate_labels"].shape == (2, NUM_PREDICATES)
        assert "predicate_weights" in collated
        assert "predicate_mask" in collated

    def test_q5_spans(self):
        batch = [self._make_sample(f"v{i}") for i in range(2)]
        batch[0]["q5_span"] = [0.2, 0.8]
        batch[1]["q5_span"] = [0.1, 0.5]
        collated = sportr_collate_fn(batch)
        assert collated["q5_spans"].shape == (2, 2)

    def test_video_ids_preserved(self):
        batch = [self._make_sample(f"v{i}") for i in range(3)]
        collated = sportr_collate_fn(batch)
        assert collated["video_ids"] == ["v0", "v1", "v2"]
