"""
Các khối tầng cơ sở (Core Layers) của kiến trúc Open-SLM Killer
===============================================================
Bao gồm:
- `RMSNorm`: Chuẩn hóa Root Mean Square tốc độ cao.
- `RotaryEmbedding` & `apply_rotary_pos_emb`: Mã hóa vị trí xoay (RoPE) hỗ trợ mở rộng ngữ cảnh.
- `DepthwiseShortConv`: Tích chập nhân quả chiều sâu (kernel=5) đặt trước QKV, hỗ trợ state cache O(1).
- `SwiGLUMLP`: Mạng tiến truyền SwiGLU.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Chuẩn hóa Root Mean Square Layer Normalization tốc độ cao."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x_fp32 = x.float()
        variance = x_fp32.pow(2).mean(dim=-1, keepdim=True)
        x_normed = x_fp32 * torch.rsqrt(variance + self.eps)
        return (self.weight * x_normed).to(orig_dtype)


class RotaryEmbedding(nn.Module):
    """
    Rotary Positional Embedding (RoPE) hỗ trợ mở rộng ngữ cảnh (`scaling_factor`)
    và tự động mở rộng bộ đệm `cos`/`sin` khi gặp chuỗi dài hơn `max_seq_len`.
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 2048,
        theta: float = 10000.0,
        scaling_factor: float = 1.0,
    ) -> None:
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

    def _build_cache(self, seq_len: int) -> None:
        self.max_seq_len = max(self.max_seq_len, seq_len)
        t = torch.arange(self.max_seq_len, device=self.inv_freq.device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(
        self, x: torch.Tensor, seq_len: int, offset: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        needed_len = offset + seq_len
        if needed_len > self.cos_cached.shape[0]:
            self._build_cache(needed_len * 2)
        cos = self.cos_cached[offset : offset + seq_len].to(dtype=x.dtype)
        sin = self.sin_cached[offset : offset + seq_len].to(dtype=x.dtype)
        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Xoay nửa chiều ẩn của vector phục vụ RoPE."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Áp dụng phép xoay RoPE cho Q và K có chiều `(B, n_heads, T, head_dim)`."""
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, T, head_dim)
    sin = sin.unsqueeze(0).unsqueeze(0)  # (1, 1, T, head_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class DepthwiseShortConv(nn.Module):
    """
    Lớp tích chập nhân quả chiều sâu (Causal Depthwise Conv1d) đặt ngay trước QKV
    giúp nắm bắt ngữ cảnh cục bộ của chuỗi BPE và hỗ trợ trạng thái đệm O(1) khi sinh văn bản.
    """

    def __init__(self, d_model: int, kernel_size: int = 5) -> None:
        super().__init__()
        self.d_model = d_model
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size,
            padding=kernel_size - 1,
            groups=d_model,
            bias=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        conv_state: Optional[torch.Tensor] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        bsz, seq_len, _ = x.shape
        x_t = x.transpose(1, 2)  # (B, D, T)

        if conv_state is not None and seq_len == 1:
            padded = torch.cat([conv_state, x_t], dim=-1)
            out = F.conv1d(
                padded, self.conv.weight, bias=None, stride=1, padding=0, groups=self.d_model
            )
            new_state = padded[:, :, 1:].detach() if use_cache else None
            return out.transpose(1, 2), new_state

        if conv_state is not None:
            padded = torch.cat([conv_state, x_t], dim=-1)
            out = F.conv1d(
                padded, self.conv.weight, bias=None, stride=1, padding=0, groups=self.d_model
            )
        else:
            # Cắt bỏ phần padding thừa ở cuối chuỗi để đảm bảo tính nhân quả (causal)
            out = self.conv(x_t)[..., :seq_len]

        new_state: Optional[torch.Tensor] = None
        if use_cache and self.kernel_size > 1:
            pad_len = self.kernel_size - 1
            if seq_len >= pad_len:
                new_state = x_t[:, :, -pad_len:].detach()
            else:
                zeros = torch.zeros(
                    bsz, self.d_model, pad_len - seq_len, device=x.device, dtype=x.dtype
                )
                new_state = torch.cat([zeros, x_t], dim=-1).detach()

        return out.transpose(1, 2), new_state


class SwiGLUMLP(nn.Module):
    """Mạng tiến truyền SwiGLU (SiLU Gated Linear Unit)."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))
