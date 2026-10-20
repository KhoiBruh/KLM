"""
Mô-đun Gated Grouped-Query Attention (GQA), KV-Cache & Khối Transformer
=======================================================================
Bao gồm:
- `LayerKVCache`: Lưu trữ trạng thái K, V và ShortConv khi sinh văn bản tốc độ cao.
- `GatedGroupedQueryAttention`: Tích hợp ShortConv + QK-Norm + RoPE + Output Gate + FlashAttention.
- `TransformerBlock`: Khối Transformer hoàn chỉnh kết hợp Attention và SwiGLU MLP.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module

from .config_loader import ModelConfig
from .layers import (
    DepthwiseShortConv,
    RMSNorm,
    RotaryEmbedding,
    SwiGLUMLP,
    apply_rotary_pos_emb,
)


@dataclass
class LayerKVCache:
    """Bộ nhớ đệm KV và trạng thái ShortConv cho một tầng Transformer khi sinh văn bản."""

    k: torch.Tensor
    v: torch.Tensor
    conv_state: Optional[torch.Tensor] = None

    @property
    def seq_len(self) -> int:
        return int(self.k.shape[2])


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Lặp các đầu KV cho Grouped-Query Attention (GQA): `(B, n_kv_heads, T, D) -> (B, n_heads, T, D)`."""
    if n_rep == 1:
        return x
    bsz, n_kv_heads, seq_len, head_dim = x.shape
    return (
        x[:, :, None, :, :]
        .expand(bsz, n_kv_heads, n_rep, seq_len, head_dim)
        .reshape(bsz, n_kv_heads * n_rep, seq_len, head_dim)
    )


class GatedGroupedQueryAttention(Module):
    """
    Grouped-Query Attention (GQA) tích hợp:
    - Depthwise ShortConv trước QKV
    - QK-Norm (`RMSNorm` trên Q và K trước khi xoay RoPE)
    - Output Gate (nhân `attn_out = attn_out * torch.sigmoid(gate)` trước `o_proj`)
    - FlashAttention (`F.scaled_dot_product_attention`) & KV-Cache
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.d_model = cfg.d_model
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        self.n_rep = self.n_heads // self.n_kv_heads
        self.q_dim = self.n_heads * self.head_dim
        self.kv_dim = self.n_kv_heads * self.head_dim
        self.dropout = cfg.dropout

        self.use_short_conv = cfg.use_short_conv
        if self.use_short_conv:
            self.short_conv = DepthwiseShortConv(self.d_model, cfg.short_conv_kernel)

        self.q_proj = nn.Linear(self.d_model, self.q_dim, bias=False)
        self.k_proj = nn.Linear(self.d_model, self.kv_dim, bias=False)
        self.v_proj = nn.Linear(self.d_model, self.kv_dim, bias=False)

        self.use_qk_norm = cfg.use_qk_norm
        if self.use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)

        self.use_attn_output_gate = cfg.use_attn_output_gate
        self.attn_gate_rank = cfg.attn_gate_rank
        if self.use_attn_output_gate:
            if 0 < self.attn_gate_rank < self.d_model:
                self.gate_down = nn.Linear(self.d_model, self.attn_gate_rank, bias=False)
                self.gate_up = nn.Linear(self.attn_gate_rank, self.q_dim, bias=False)
                self.gate_proj: Optional[nn.Linear] = None
            else:
                self.gate_down = None
                self.gate_up = None
                self.gate_proj = nn.Linear(self.d_model, self.q_dim, bias=False)

        self.o_proj = nn.Linear(self.q_dim, self.d_model, bias=False)

    def _compute_gate(self, x: torch.Tensor) -> torch.Tensor:
        if self.gate_proj is not None:
            return self.gate_proj(x)
        return self.gate_up(self.gate_down(x))

    def forward(
        self,
        x: torch.Tensor,
        rope: RotaryEmbedding,
        kv_cache: Optional[LayerKVCache] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[LayerKVCache]]:
        bsz, seq_len, _ = x.shape

        conv_state = kv_cache.conv_state if kv_cache is not None else None
        if self.use_short_conv:
            x_conv, new_conv_state = self.short_conv(x, conv_state=conv_state, use_cache=use_cache)
        else:
            x_conv, new_conv_state = x, None

        q = self.q_proj(x_conv).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x_conv).view(bsz, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x_conv).view(bsz, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        offset = kv_cache.seq_len if kv_cache is not None else 0
        cos, sin = rope(q, seq_len, offset)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if kv_cache is not None:
            k = torch.cat([kv_cache.k, k], 2)
            v = torch.cat([kv_cache.v, v], 2)

        new_cache: Optional[LayerKVCache] = None
        if use_cache:
            new_cache = LayerKVCache(k.detach(), v.detach(), new_conv_state)

        k_exp = repeat_kv(k, self.n_rep)
        v_exp = repeat_kv(v, self.n_rep)

        is_causal = (kv_cache is None) and (seq_len > 1)
        attn_out = F.scaled_dot_product_attention(
            q, k_exp, v_exp, None,
            self.dropout if self.training else 0.0,
            is_causal
        )

        attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, seq_len, self.q_dim)

        if self.use_attn_output_gate:
            gate = self._compute_gate(x)
            attn_out = attn_out * torch.sigmoid(gate)

        return self.o_proj(attn_out), new_cache


class TransformerBlock(Module):
    """Khối Transformer kết hợp Gated GQA + ShortConv + QK-Norm và SwiGLU MLP."""

    def __init__(self, cfg: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.attn_norm = RMSNorm(cfg.d_model)
        self.attn = GatedGroupedQueryAttention(cfg)
        self.ffn_norm = RMSNorm(cfg.d_model)
        self.mlp = SwiGLUMLP(cfg.d_model, cfg.d_ff, cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        rope: RotaryEmbedding,
        kv_cache: Optional[LayerKVCache] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[LayerKVCache]]:
        attn_out, new_cache = self.attn(self.attn_norm(x), rope, kv_cache, use_cache)
        h = x + attn_out
        out = h + self.mlp(self.ffn_norm(h))
        return out, new_cache
