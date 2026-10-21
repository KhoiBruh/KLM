"""
Bidirectional Attention Module & Transformer Block for LLaDA Architecture
========================================================================
LLaDA (Large Language Diffusion with Masking) uses a Bidirectional Transformer:
- Full non-causal attention across the sequence (is_causal = False).
- QK-Norm (`RMSNorm` on Query and Key projections prior to RoPE) for training stability.
- 1D Rotary Position Embedding (RoPE).
- FlashAttention optimization via `F.scaled_dot_product_attention`.
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
	RMSNorm,
	RotaryEmbed,
	SwiGLUMLP,
	rotary_pos_emb,
)


@dataclass
class LayerKVCache:
	"""Legacy container kept for type compatibility."""
	k: torch.Tensor
	v: torch.Tensor
	conv_state: Optional[torch.Tensor] = None

	@property
	def seq_len(self) -> int:
		return int(self.k.shape[2])


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
	"""Repeat Key/Value heads for Grouped-Query Attention (GQA): `(B, n_kv_heads, T, D) -> (B, n_heads, T, D)`."""
	if n_rep == 1:
		return x
	bsz, n_kv_heads, seq_len, head_dim = x.shape
	return (
		x[:, :, None, :, :]
		.expand(bsz, n_kv_heads, n_rep, seq_len, head_dim)
		.reshape(bsz, n_kv_heads * n_rep, seq_len, head_dim)
	)


class GatedGQA(Module):
	"""
	Bidirectional Grouped-Query Attention (GQA) for LLaDA:
	- Full bidirectional attention between masked tokens and unmasked context.
	- QK-Norm (`RMSNorm` on Q and K) prevents gradient spikes under Muon/AdamW.
	- Optimized with PyTorch SDPA (FlashAttention / MemEfficient kernel).
	"""

	def __init__(self, cfg: ModelConfig):
		super().__init__()
		self.d_model = cfg.d_model
		self.n_heads = cfg.n_heads
		self.n_kv_heads = cfg.n_kv_heads
		self.head_dim = cfg.head_dim
		self.n_rep = self.n_heads // self.n_kv_heads
		self.q_dim = self.n_heads * self.head_dim
		self.kv_dim = self.n_kv_heads * self.head_dim
		self.dropout = cfg.dropout

		self.q_proj = nn.Linear(self.d_model, self.q_dim, bias=False)
		self.k_proj = nn.Linear(self.d_model, self.kv_dim, bias=False)
		self.v_proj = nn.Linear(self.d_model, self.kv_dim, bias=False)

		self.use_qk_norm = getattr(cfg, "use_qk_norm", True)
		if self.use_qk_norm:
			self.q_norm = RMSNorm(self.head_dim)
			self.k_norm = RMSNorm(self.head_dim)

		self.o_proj = nn.Linear(self.q_dim, self.d_model, bias=False)

	def forward(
		self,
		x: torch.Tensor,
		rope: RotaryEmbed,
		attention_mask: Optional[torch.Tensor] = None
	) -> Tuple[torch.Tensor, None]:
		bsz, seq_len, _ = x.shape

		q = self.q_proj(x).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
		k = self.k_proj(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
		v = self.v_proj(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

		if self.use_qk_norm:
			q = self.q_norm(q)
			k = self.k_norm(k)

		# Apply 2D RoPE over full bidirectional sequence
		cos, sin = rope(q, seq_len, 0)
		q, k = rotary_pos_emb(q, k, cos, sin)

		k_exp = repeat_kv(k, self.n_rep)
		v_exp = repeat_kv(v, self.n_rep)

		# LLaDA bidirectional non-causal attention
		attn_out = F.scaled_dot_product_attention(
			q, k_exp, v_exp,
			attn_mask=attention_mask,
			dropout_p=self.dropout if self.training else 0.0,
			is_causal=False,
		)

		attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, seq_len, self.q_dim)
		return self.o_proj(attn_out), None


class TransformerBlock(Module):
	"""Bidirectional Transformer Block combining GQA + QK-Norm and SwiGLU MLP."""

	def __init__(self, cfg: ModelConfig, layer_idx: int):
		super().__init__()
		self.layer_idx = layer_idx
		self.attn_norm = RMSNorm(cfg.d_model)
		self.attn = GatedGQA(cfg)
		self.ffn_norm = RMSNorm(cfg.d_model)
		self.mlp = SwiGLUMLP(cfg.d_model, cfg.d_ff, cfg.dropout)

	def forward(
		self,
		x: torch.Tensor,
		rope: RotaryEmbed,
		attention_mask: Optional[torch.Tensor] = None
	) -> Tuple[torch.Tensor, None]:
		attn_out, _ = self.attn(self.attn_norm(x), rope, attention_mask)
		h = x + attn_out
		out = h + self.mlp(self.ffn_norm(h))
		return out, None
