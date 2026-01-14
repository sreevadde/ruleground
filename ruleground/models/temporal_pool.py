"""
Temporal Attention Pooling

Aggregates frame-level features into a single clip-level vector using
learned cross-attention with ActionFormer v2 primitives (Paper Section 5.2).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional

from actionformer import RotaryPositionEmbedding, RMSNorm, rotate_half, HAS_FLASH_ATTN


class TemporalAttentionPool(nn.Module):
    """Temporal attention pooling using ActionFormer v2 primitives.

    A learnable query token cross-attends to frame embeddings via multi-head
    attention with RoPE and optional Flash Attention, producing a single
    clip-level representation.

    Args:
        embed_dim:   Dimensionality of frame embeddings.
        num_heads:   Number of attention heads.
        max_seq_len: Maximum temporal sequence length (for RoPE).
        dropout:     Dropout rate.
        use_rope:    Whether to apply Rotary Position Embeddings.
        use_flash_attn: Whether to use Flash Attention (PyTorch 2.x SDPA).
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        max_seq_len: int = 64,
        dropout: float = 0.1,
        use_rope: bool = True,
        use_flash_attn: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.use_rope = use_rope
        self.use_flash = use_flash_attn and HAS_FLASH_ATTN

        # RMSNorm from ActionFormer
        self.norm = RMSNorm(embed_dim)

        # Learnable query for pooling
        self.pool_query = nn.Parameter(torch.randn(1, 1, embed_dim))

        # Projections
        self.kv_proj = nn.Linear(embed_dim, 2 * embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # RoPE from ActionFormer
        if use_rope:
            self.rope = RotaryPositionEmbedding(self.head_dim, max_seq_len)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            x:    [B, T, D] frame embeddings
            mask: [B, T] boolean attention mask (True = attend, False = ignore)
        Returns:
            pooled: [B, D] clip-level representation
        """
        B, T, D = x.shape
        # ActionFormer RMSNorm expects (B, C, T) format; transpose around it
        x = self.norm(x.transpose(1, 2)).transpose(1, 2)

        # Query from learnable token, K/V from sequence
        q = self.pool_query.expand(B, -1, -1)  # [B, 1, D]
        kv = self.kv_proj(x)
        k, v = kv.chunk(2, dim=-1)

        # Reshape for multi-head attention
        q = q.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE to keys only (query is position-agnostic)
        if self.use_rope:
            cos, sin = self.rope(k)
            k = (k * cos) + (rotate_half(k) * sin)

        # Compute attention
        if self.use_flash:
            # Build causal-style mask for SDPA if needed
            attn_mask = None
            if mask is not None:
                # mask: [B, T] -> [B, 1, 1, T] for query_len=1
                attn_mask = mask.unsqueeze(1).unsqueeze(2)
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.dropout.p if self.training else 0.0,
            )
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            if mask is not None:
                attn = attn.masked_fill(
                    ~mask.unsqueeze(1).unsqueeze(2), float("-inf")
                )
            attn = attn.softmax(dim=-1)
            attn = self.dropout(attn)
            out = attn @ v

        out = out.transpose(1, 2).reshape(B, 1, D)
        out = self.out_proj(out)

        return out.squeeze(1)  # [B, D]
