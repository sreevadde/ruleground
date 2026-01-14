"""
RuleGround: Full Model Assembly

Composes encoder + RGM + differentiable logic + reasoning head into the
complete RuleGround architecture (Paper Section 5).

Pipeline:
    video -> VideoEncoder -> frame_embeddings -> RGM -> predicates
                                                     -> RuleComposer -> rule_scores
                                                     -> ReasoningHead -> Q1/Q2/Q5
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict, List, Optional

from ruleground.predicates.ontology import ALL_PREDICATE_NAMES, NUM_PREDICATES
from ruleground.models.encoder import VideoEncoder
from ruleground.models.rgm import RuleGroundingModule
from ruleground.models.logic import RuleComposer
from ruleground.models.reasoning_head import ReasoningHead


class RuleGround(nn.Module):
    """RuleGround: Rule-Grounded Representations for Sports Video.

    Components:
        1. Video Encoder (frozen VideoMAE-v2)
        2. Rule Grounding Module (RGM) -- predicate bottleneck
        3. Rule Composer (differentiable logic)
        4. Reasoning Head (multi-task: Q1/Q2/Q5)

    Args:
        encoder_name:   HuggingFace model ID for video encoder.
        num_q1_classes: Q1 infraction classes (default 2 = yes/no).
        num_q2_classes: Q2 foul type classes.
        hidden_dim:     Hidden dimension for RGM and reasoning head.
        num_heads:      Attention heads for temporal pooling.
        max_seq_len:    Maximum temporal sequence length.
        dropout:        Dropout rate.
        freeze_encoder: Whether to freeze the video encoder.
        use_rope:       Whether to use RoPE in temporal attention.
        use_flash_attn: Whether to use Flash Attention.
    """

    def __init__(
        self,
        encoder_name: str = "MCG-NJU/videomae-base",
        num_q1_classes: int = 2,
        num_q2_classes: int = 17,
        hidden_dim: int = 256,
        num_heads: int = 8,
        max_seq_len: int = 64,
        dropout: float = 0.1,
        freeze_encoder: bool = True,
        use_rope: bool = True,
        use_flash_attn: bool = True,
    ):
        super().__init__()

        # Video encoder
        self.encoder = VideoEncoder(encoder_name, freeze=freeze_encoder)
        embed_dim = self.encoder.embed_dim

        # Rule Grounding Module (predicate bottleneck)
        self.rgm = RuleGroundingModule(
            embed_dim=embed_dim,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            hidden_dim=hidden_dim,
            dropout=dropout,
            use_rope=use_rope,
            use_flash_attn=use_flash_attn,
        )

        # Rule Composer (differentiable logic)
        self.rule_composer = RuleComposer()

        # Reasoning Head (multi-task)
        self.reasoning_head = ReasoningHead(
            num_predicates=NUM_PREDICATES,
            visual_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_q1_classes=num_q1_classes,
            num_q2_classes=num_q2_classes,
            num_heads=min(num_heads, 4),
            dropout=dropout,
        )

    def forward(
        self,
        video: Tensor,
        mask: Optional[Tensor] = None,
        sport_ids: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        Args:
            video:     [B, T, C, H, W] video frames.
            mask:      [B, T] attention mask.
            sport_ids: [B] integer sport IDs.

        Returns:
            Dict with:
                'q1_logits':          [B, C1] infraction logits
                'q2_logits':          [B, C2] foul type logits
                'q5_preds':           [B, 2] temporal grounding predictions
                'predicate_logits':   [B, K] predicate logits
                'predicate_probs':    [B, K] predicate probabilities
                'predicate_dict':     {name: [B]} predicate probabilities
                'rule_scores':        {name: [B]} rule scores
                'frame_activations':  [B, T, K_instant] frame-level activations
                'pooled':             [B, D] pooled visual features
        """
        # 1. Encode video
        frame_embeddings = self.encoder(video)  # [B, T, D]

        # 2. Predicate prediction via RGM
        rgm_out = self.rgm(frame_embeddings, mask, sport_ids)

        # 3. Rule composition via differentiable logic
        rule_scores = self.rule_composer(rgm_out["predicate_dict"])

        # 4. Task prediction via reasoning head
        task_out = self.reasoning_head(
            predicate_probs=rgm_out["probs"],
            pooled_visual=rgm_out["pooled"],
            rule_scores=rule_scores if rule_scores else None,
        )

        return {
            # Task outputs
            "q1_logits": task_out["q1_logits"],
            "q2_logits": task_out["q2_logits"],
            "q5_preds": task_out["q5_preds"],
            # Predicate outputs
            "predicate_logits": rgm_out["logits"],
            "predicate_probs": rgm_out["probs"],
            "predicate_dict": rgm_out["predicate_dict"],
            # Rule outputs
            "rule_scores": rule_scores,
            # Intermediate features
            "frame_activations": rgm_out["frame_activations"],
            "pooled": rgm_out["pooled"],
        }

    @staticmethod
    def from_config(config) -> "RuleGround":
        """Build RuleGround from an OmegaConf config object."""
        return RuleGround(
            encoder_name=config.model.encoder.name,
            num_q1_classes=config.model.get("num_q1_classes", 2),
            num_q2_classes=config.model.get("num_q2_classes", 17),
            hidden_dim=config.model.rgm.hidden_dim,
            num_heads=config.model.rgm.num_heads,
            max_seq_len=config.data.get("num_frames", 16) * 4,
            dropout=config.model.rgm.dropout,
            freeze_encoder=config.model.encoder.freeze,
            use_rope=config.model.rgm.use_rope,
            use_flash_attn=config.model.rgm.use_flash_attn,
        )
