"""
End-to-end test fixtures: synthetic SportR data with learnable patterns.

Creates realistic synthetic datasets for each sport where the video signal
correlates with the ground truth, so the model can actually learn.

Pattern encoding (baked into video tensors):
    - Channel 0 (R): contact intensity (high → contact_occurred)
    - Channel 1 (G): positional signal (sport-specific meaning)
    - Channel 2 (B): motion signal (high → action happening)
    - Temporal pattern: Q5 spans have higher activation in the marked frames

This means the mock encoder + RGM can learn to extract predicates from
meaningful signal, and we can verify the full pipeline learns.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

from ruleground.predicates.ontology import (
    ALL_PREDICATE_NAMES,
    NUM_PREDICATES,
    PREDICATE_NAME_TO_IDX,
    get_predicate_names_for_sport,
    get_sport_mask,
    SPORT_TO_ID,
)
from ruleground.data.collate import sportr_collate_fn


# ---------------------------------------------------------------------------
# Mock encoder: lightweight trainable replacement for VideoMAE-v2
# ---------------------------------------------------------------------------

class MockVideoEncoder(nn.Module):
    """Trainable mock encoder that maps video tensors to frame embeddings.

    Uses a simple Conv3D + projection instead of VideoMAE-v2. This is
    lightweight (no HuggingFace download) but still produces meaningful
    gradients so the full pipeline can train.
    """

    def __init__(self, embed_dim: int = 128):
        super().__init__()
        self.embed_dim = embed_dim
        # Simple feature extractor: [B, C, T, H, W] -> [B, T, D]
        self.conv = nn.Conv3d(3, 32, kernel_size=(1, 4, 4), stride=(1, 4, 4), padding=0)
        self.pool = nn.AdaptiveAvgPool3d((None, 1, 1))  # Keep T, collapse spatial
        self.proj = nn.Linear(32, embed_dim)

    def forward(self, video: Tensor) -> Tensor:
        # Input: [B, T, C, H, W]
        B, T = video.shape[0], video.shape[1]
        x = video.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]
        x = self.conv(x)                   # [B, 32, T, H', W']
        x = self.pool(x)                   # [B, 32, T, 1, 1]
        x = x.squeeze(-1).squeeze(-1)      # [B, 32, T]
        x = x.permute(0, 2, 1)             # [B, T, 32]
        x = self.proj(x)                   # [B, T, D]
        return x


# ---------------------------------------------------------------------------
# Scenario definitions: sport-specific use cases with predicate patterns
# ---------------------------------------------------------------------------

# Basketball use cases
BASKETBALL_SCENARIOS = [
    {
        "name": "blocking_foul",
        "q1_label": 1,
        "q2_label": 1,
        "q5_span": [0.2, 0.6],
        "predicates": {
            "ball_in_play": 1.0,
            "contact_occurred": 1.0,
            "defender_set": 0.0,       # NOT set → blocking
            "restricted_area": 0.3,
            "shooting_motion": 0.0,
        },
        "video_signal": {"contact": 0.8, "position": 0.2, "motion": 0.7},
    },
    {
        "name": "charging_foul",
        "q1_label": 1,
        "q2_label": 2,
        "q5_span": [0.3, 0.7],
        "predicates": {
            "ball_in_play": 1.0,
            "contact_occurred": 1.0,
            "defender_set": 1.0,       # SET → charging
            "restricted_area": 0.0,    # NOT in restricted area
            "shooting_motion": 0.0,
        },
        "video_signal": {"contact": 0.9, "position": 0.8, "motion": 0.6},
    },
    {
        "name": "clean_play_basketball",
        "q1_label": 0,
        "q2_label": 0,
        "q5_span": None,
        "predicates": {
            "ball_in_play": 1.0,
            "contact_occurred": 0.0,
            "defender_set": 0.5,
            "restricted_area": 0.2,
            "shooting_motion": 0.0,
        },
        "video_signal": {"contact": 0.1, "position": 0.5, "motion": 0.3},
    },
    {
        "name": "shooting_foul",
        "q1_label": 1,
        "q2_label": 3,
        "q5_span": [0.4, 0.8],
        "predicates": {
            "ball_in_play": 1.0,
            "contact_occurred": 1.0,
            "shooting_motion": 1.0,
            "defender_set": 0.3,
            "verticality_maintained": 0.0,
        },
        "video_signal": {"contact": 0.7, "position": 0.4, "motion": 0.9},
    },
]

# Football use cases
FOOTBALL_SCENARIOS = [
    {
        "name": "dpi",
        "q1_label": 1,
        "q2_label": 4,
        "q5_span": [0.3, 0.6],
        "predicates": {
            "ball_in_play": 1.0,
            "contact_before_arrival": 1.0,
            "incidental_contact": 0.0,
            "ball_catchable": 1.0,
            "ball_in_air": 1.0,
        },
        "video_signal": {"contact": 0.85, "position": 0.3, "motion": 0.8},
    },
    {
        "name": "clean_play_football",
        "q1_label": 0,
        "q2_label": 0,
        "q5_span": None,
        "predicates": {
            "ball_in_play": 1.0,
            "contact_before_arrival": 0.0,
            "incidental_contact": 0.8,
            "ball_catchable": 0.6,
            "ball_in_air": 0.4,
        },
        "video_signal": {"contact": 0.15, "position": 0.6, "motion": 0.2},
    },
    {
        "name": "opi",
        "q1_label": 1,
        "q2_label": 5,
        "q5_span": [0.2, 0.5],
        "predicates": {
            "ball_in_play": 1.0,
            "contact_before_arrival": 1.0,
            "offensive_push_off": 1.0,
            "ball_in_air": 1.0,
            "incidental_contact": 0.0,
        },
        "video_signal": {"contact": 0.75, "position": 0.7, "motion": 0.85},
    },
]

# Soccer use cases
SOCCER_SCENARIOS = [
    {
        "name": "handball",
        "q1_label": 1,
        "q2_label": 6,
        "q5_span": [0.4, 0.7],
        "predicates": {
            "ball_in_play": 1.0,
            "ball_contact_arm": 1.0,
            "arm_natural_position": 0.0,  # NOT natural → handball
            "contact_occurred": 1.0,
        },
        "video_signal": {"contact": 0.7, "position": 0.2, "motion": 0.6},
    },
    {
        "name": "offside",
        "q1_label": 1,
        "q2_label": 7,
        "q5_span": [0.1, 0.3],
        "predicates": {
            "ball_in_play": 1.0,
            "offside_position": 1.0,
            "involved_in_play": 1.0,
            "played_by_opponent": 0.0,
        },
        "video_signal": {"contact": 0.2, "position": 0.9, "motion": 0.5},
    },
    {
        "name": "clean_play_soccer",
        "q1_label": 0,
        "q2_label": 0,
        "q5_span": None,
        "predicates": {
            "ball_in_play": 1.0,
            "contact_occurred": 0.1,
            "ball_contact_arm": 0.0,
            "offside_position": 0.0,
        },
        "video_signal": {"contact": 0.05, "position": 0.5, "motion": 0.4},
    },
    {
        "name": "dogso",
        "q1_label": 1,
        "q2_label": 8,
        "q5_span": [0.5, 0.9],
        "predicates": {
            "ball_in_play": 1.0,
            "denying_goal": 1.0,
            "contact_occurred": 1.0,
        },
        "video_signal": {"contact": 0.9, "position": 0.3, "motion": 0.95},
    },
]

SPORT_SCENARIOS = {
    "basketball": BASKETBALL_SCENARIOS,
    "football": FOOTBALL_SCENARIOS,
    "soccer": SOCCER_SCENARIOS,
}


# ---------------------------------------------------------------------------
# Synthetic video generation
# ---------------------------------------------------------------------------

def generate_video(
    signal: Dict[str, float],
    q5_span: Optional[List[float]],
    num_frames: int = 16,
    height: int = 32,
    width: int = 32,
    noise_scale: float = 0.1,
) -> Tensor:
    """Generate a synthetic video tensor with embedded signal.

    Args:
        signal:      Dict with 'contact', 'position', 'motion' intensities.
        q5_span:     Temporal span [start, end] in [0,1] or None.
        num_frames:  Number of frames.
        height:      Frame height.
        width:       Frame width.
        noise_scale: Random noise scale.

    Returns:
        [T, C, H, W] video tensor in [0, 1].
    """
    T, C, H, W = num_frames, 3, height, width
    video = torch.zeros(T, C, H, W)

    # Encode signal into channels
    video[:, 0, :, :] = signal.get("contact", 0.0)   # Red = contact
    video[:, 1, :, :] = signal.get("position", 0.0)   # Green = position
    video[:, 2, :, :] = signal.get("motion", 0.0)     # Blue = motion

    # Add temporal structure: amplify signal during Q5 span
    if q5_span is not None:
        start_f = int(q5_span[0] * T)
        end_f = int(q5_span[1] * T)
        # Boost activation during event
        video[start_f:end_f] *= 1.5
        # Add temporal gradient leading into event
        if start_f > 0:
            video[max(0, start_f - 2):start_f] *= 1.2

    # Add per-pixel noise for realism
    noise = torch.randn_like(video) * noise_scale
    video = (video + noise).clamp(0, 1)

    return video


def build_predicate_labels(
    scenario_predicates: Dict[str, float],
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Build full 20-predicate label and weight vectors from scenario predicates.

    Args:
        scenario_predicates: Sparse dict of {predicate_name: value}.

    Returns:
        (labels_dict, weights_dict) covering all 20 predicates.
    """
    labels = {}
    weights = {}
    for name in ALL_PREDICATE_NAMES:
        if name in scenario_predicates:
            labels[name] = scenario_predicates[name]
            weights[name] = 0.95  # High confidence for specified predicates
        else:
            labels[name] = 0.0
            weights[name] = 0.3   # Low confidence for unspecified
    return labels, weights


# ---------------------------------------------------------------------------
# Synthetic dataset class
# ---------------------------------------------------------------------------

class SyntheticSportRDataset(Dataset):
    """Synthetic SportR dataset with learnable patterns.

    Creates N samples per scenario, each with slightly different noise.
    The signal-to-noise ratio is designed so a neural network can learn
    to distinguish the scenarios within a few epochs.

    Args:
        samples_per_scenario: Number of samples to generate per scenario.
        num_frames:           Frames per video.
        frame_size:           (H, W) frame dimensions.
        noise_scale:          Random noise magnitude.
        sports:               List of sports to include (default: all).
    """

    def __init__(
        self,
        samples_per_scenario: int = 20,
        num_frames: int = 16,
        frame_size: Tuple[int, int] = (32, 32),
        noise_scale: float = 0.1,
        sports: Optional[List[str]] = None,
    ):
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.noise_scale = noise_scale

        sports = sports or ["basketball", "football", "soccer"]
        self.samples: List[Dict[str, Any]] = []

        for sport in sports:
            scenarios = SPORT_SCENARIOS[sport]
            for scenario in scenarios:
                for i in range(samples_per_scenario):
                    labels, weights = build_predicate_labels(scenario["predicates"])
                    self.samples.append({
                        "video_id": f"{sport}_{scenario['name']}_{i:03d}",
                        "sport": sport,
                        "q1_label": scenario["q1_label"],
                        "q2_label": scenario["q2_label"],
                        "q5_span": scenario["q5_span"],
                        "scenario_name": scenario["name"],
                        "video_signal": scenario["video_signal"],
                        "predicates": {"labels": labels, "weights": weights},
                    })

        # Shuffle for training
        random.shuffle(self.samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        video = generate_video(
            signal=sample["video_signal"],
            q5_span=sample["q5_span"],
            num_frames=self.num_frames,
            height=self.frame_size[0],
            width=self.frame_size[1],
            noise_scale=self.noise_scale,
        )
        return {
            "video": video,
            "video_id": sample["video_id"],
            "sport": sample["sport"],
            "q1_label": sample["q1_label"],
            "q2_label": sample["q2_label"],
            "q5_span": sample["q5_span"],
            "predicates": sample["predicates"],
        }


# ---------------------------------------------------------------------------
# Helper: build model with mock encoder
# ---------------------------------------------------------------------------

def build_test_model(embed_dim: int = 128, hidden_dim: int = 64) -> nn.Module:
    """Build a RuleGround model with mock encoder for testing.

    Constructs the real RGM, logic layer, and reasoning head, but
    replaces VideoMAE-v2 with a lightweight trainable mock.
    """
    from ruleground.models.rgm import RuleGroundingModule
    from ruleground.models.logic import RuleComposer
    from ruleground.models.reasoning_head import ReasoningHead

    class TestableRuleGround(nn.Module):
        """RuleGround with mock encoder for end-to-end testing."""

        def __init__(self, embed_dim: int, hidden_dim: int):
            super().__init__()
            self.encoder = MockVideoEncoder(embed_dim=embed_dim)
            self.rgm = RuleGroundingModule(
                embed_dim=embed_dim,
                num_heads=4,
                max_seq_len=64,
                hidden_dim=hidden_dim,
                dropout=0.1,
                use_rope=True,
                use_flash_attn=True,
            )
            self.rule_composer = RuleComposer()
            self.reasoning_head = ReasoningHead(
                num_predicates=NUM_PREDICATES,
                visual_dim=embed_dim,
                hidden_dim=hidden_dim,
                num_q1_classes=2,
                num_q2_classes=17,
                num_heads=4,
                dropout=0.1,
            )

        def forward(self, video, mask=None, sport_ids=None):
            frame_embeddings = self.encoder(video)
            rgm_out = self.rgm(frame_embeddings, mask, sport_ids)
            rule_scores = self.rule_composer(rgm_out["predicate_dict"])
            task_out = self.reasoning_head(
                predicate_probs=rgm_out["probs"],
                pooled_visual=rgm_out["pooled"],
                rule_scores=rule_scores if rule_scores else None,
            )
            return {
                "q1_logits": task_out["q1_logits"],
                "q2_logits": task_out["q2_logits"],
                "q5_preds": task_out["q5_preds"],
                "predicate_logits": rgm_out["logits"],
                "predicate_probs": rgm_out["probs"],
                "predicate_dict": rgm_out["predicate_dict"],
                "rule_scores": rule_scores,
                "frame_activations": rgm_out["frame_activations"],
                "pooled": rgm_out["pooled"],
            }

    return TestableRuleGround(embed_dim, hidden_dim)


def build_dataloaders(
    samples_per_scenario: int = 20,
    batch_size: int = 8,
    num_frames: int = 16,
    frame_size: Tuple[int, int] = (32, 32),
    val_ratio: float = 0.2,
) -> Tuple[DataLoader, DataLoader]:
    """Build train and val dataloaders from synthetic data.

    Returns:
        (train_loader, val_loader)
    """
    dataset = SyntheticSportRDataset(
        samples_per_scenario=samples_per_scenario,
        num_frames=num_frames,
        frame_size=frame_size,
    )
    n = len(dataset)
    n_val = max(1, int(n * val_ratio))
    n_train = n - n_val

    train_set, val_set = torch.utils.data.random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=sportr_collate_fn,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=sportr_collate_fn,
        drop_last=False,
    )
    return train_loader, val_loader
