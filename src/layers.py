"""
Core Layers & Primitives
========================
Foundational neural network modules:
- `RMSNorm`: High-performance Root Mean Square Layer Normalization.
- `RotaryEmbed` & `rotary_pos_emb`: Rotary Position Embedding (RoPE) with dynamic context extension.
- `SwiGLUMLP`: SwiGLU feed-forward network.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
	"""Root Mean Square Layer Normalization (RMSNorm)."""

	def __init__(self, dim: int, eps: float = 1e-6):
		super().__init__()
		self.eps = eps
		self.weight = nn.Parameter(torch.ones(dim))

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		orig_dtype = x.dtype
		x_fp32 = x.float()
		variance = x_fp32.pow(2).mean(dim=-1, keepdim=True)
		x_normed = x_fp32 * torch.rsqrt(variance + self.eps)
		return (self.weight * x_normed).to(orig_dtype)


class RotaryEmbed(nn.Module):
	"""
	Rotary Positional Embedding (RoPE) supporting context extension (`scaling_factor`)
	with dynamic cache reallocation when sequences exceed `max_seq_len`.
	"""

	def __init__(
		self,
		dim: int,
		max_seq_len: int = 2048,
		theta: float = 10000.0,
		scaling_factor: float = 1.0,
	):
		super().__init__()
		self.dim = dim
		self.max_seq_len = max_seq_len
		self.theta = theta
		self.scaling_factor = max(1.0, float(scaling_factor))

		inv_freq = 1.0 / (
			(self.theta * self.scaling_factor)
			** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
		)
		self.register_buffer("inv_freq", inv_freq, persistent=False)
		self._build_cache(max_seq_len)

	def _build_cache(self, seq_len: int):
		self.max_seq_len = max(self.max_seq_len, seq_len)
		t = torch.arange(self.max_seq_len, device=self.inv_freq.device, dtype=torch.float32)
		freqs = torch.outer(t, self.inv_freq)
		emb = torch.cat((freqs, freqs), -1)
		self.register_buffer("cos_cached", emb.cos(), persistent=False)
		self.register_buffer("sin_cached", emb.sin(), persistent=False)

	def forward(
		self, x: torch.Tensor, seq_len: int, offset: int = 0
	) -> Tuple[torch.Tensor, torch.Tensor]:
		needed_len = offset + seq_len
		if needed_len > self.cos_cached.shape[0]:
			self._build_cache(needed_len * 2)
		cos = self.cos_cached[offset: offset + seq_len].to(dtype=x.dtype)
		sin = self.sin_cached[offset: offset + seq_len].to(dtype=x.dtype)
		return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
	"""Rotates half the hidden dimensions of a tensor for RoPE."""
	x1 = x[..., : x.shape[-1] // 2]
	x2 = x[..., x.shape[-1] // 2:]
	return torch.cat((-x2, x1), dim=-1)


def rotary_pos_emb(
	q: torch.Tensor,
	k: torch.Tensor,
	cos: torch.Tensor,
	sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
	"""Applies RoPE rotation to Q and K with shapes `(B, n_heads, T, head_dim)`."""
	cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, T, head_dim)
	sin = sin.unsqueeze(0).unsqueeze(0)  # (1, 1, T, head_dim)
	q_embed = (q * cos) + (rotate_half(q) * sin)
	k_embed = (k * cos) + (rotate_half(k) * sin)
	return q_embed, k_embed


class SwiGLUMLP(nn.Module):
	"""SwiGLU feed-forward network (SiLU Gated Linear Unit)."""

	def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
		super().__init__()
		self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
		self.up_proj = nn.Linear(d_model, d_ff, bias=False)
		self.down_proj = nn.Linear(d_ff, d_model, bias=False)
		self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))
