"""
Video Encoder

Wraps a frozen VideoMAE-v2 ViT-B encoder (Paper Section 5.1).
Produces per-frame embeddings from raw video input.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional


class VideoEncoder(nn.Module):
    """Frozen video encoder that produces frame-level embeddings.

    Wraps HuggingFace VideoMAE-v2 (or v1) models. The encoder is frozen
    by default -- RuleGround trains only the RGM and reasoning heads.

    Args:
        model_name: HuggingFace model ID. Default uses VideoMAE-v2 ViT-B.
        freeze:     Whether to freeze all encoder parameters.
    """

    def __init__(
        self,
        model_name: str = "MCG-NJU/videomae-base",
        freeze: bool = True,
    ):
        super().__init__()
        self.model_name = model_name
        self._freeze = freeze

        # Lazy import to avoid hard dependency at module level
        from transformers import VideoMAEModel

        self.backbone = VideoMAEModel.from_pretrained(model_name)
        self.embed_dim = self.backbone.config.hidden_size

        if freeze:
            self._freeze_params()

    def _freeze_params(self) -> None:
        """Freeze all encoder parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()

    def train(self, mode: bool = True):
        """Override train to keep encoder in eval mode when frozen."""
        super().train(mode)
        if self._freeze:
            self.backbone.eval()
        return self

    def forward(self, video: Tensor) -> Tensor:
        """
        Args:
            video: [B, T, C, H, W] video frames (T uniformly sampled frames).
                   Also accepts [B, C, T, H, W] (channels-first temporal).

        Returns:
            frame_embeddings: [B, T, D] per-frame embedding vectors.
        """
        # Handle both input formats
        if video.dim() == 5:
            if video.shape[1] == 3 or video.shape[1] == 1:
                # [B, C, T, H, W] -> [B, T, C, H, W]
                video = video.permute(0, 2, 1, 3, 4)

        B, T = video.shape[:2]

        # VideoMAE expects [B, C, T, H, W]
        pixel_values = video.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]

        with torch.set_grad_enabled(not self._freeze):
            outputs = self.backbone(pixel_values=pixel_values)

        # outputs.last_hidden_state: [B, num_patches, D]
        # We need to reshape back to [B, T, D]
        hidden = outputs.last_hidden_state  # [B, N, D]

        # VideoMAE produces N patches from T frames. We average patches per
        # frame to get frame-level embeddings [B, T, D].
        N = hidden.shape[1]
        patches_per_frame = N // T if T > 0 else N

        if patches_per_frame > 1 and N == patches_per_frame * T:
            # Reshape: [B, T * P, D] -> [B, T, P, D] -> mean over P
            hidden = hidden.view(B, T, patches_per_frame, -1).mean(dim=2)
        elif N != T:
            # Fallback: interpolate to T frames
            hidden = hidden.transpose(1, 2)  # [B, D, N]
            hidden = torch.nn.functional.interpolate(
                hidden, size=T, mode="linear", align_corners=False
            )
            hidden = hidden.transpose(1, 2)  # [B, T, D]

        return hidden  # [B, T, D]
