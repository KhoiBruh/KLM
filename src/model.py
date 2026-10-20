"""
Mô hình trung tâm KLM
====================================================
Tệp này chỉ tập trung định nghĩa mô hình cấp cao (`KLMModel` & `ModelOutput`),
trong khi các thành phần chi tiết được tách gọn sang:
- `src/layers.py`: `RMSNorm`, `RotaryEmbedding`, `DepthwiseShortConv`, `SwiGLUMLP`
- `src/attention.py`: `LayerKVCache`, `GatedGroupedQueryAttention`, `TransformerBlock`
- `src/checkpoint.py`: `count_parameters`, `estimate_config_parameters`,
  `save_safetensors_checkpoint`, `load_safetensors_checkpoint`
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .attention import (
	GatedGroupedQueryAttention,
	LayerKVCache,
	TransformerBlock
)
from .checkpoint import (
	_unwrap_model,
	count_parameters,
	estimate_config_parameters,
	load_safetensors_checkpoint,
	save_safetensors_checkpoint,
)
from .config_loader import FullConfig, ModelConfig
from .layers import (
	DepthwiseShortConv,
	RMSNorm,
	RotaryEmbedding,
	SwiGLUMLP,
)


@dataclass
class ModelOutput:
	"""Cấu trúc đầu ra thống nhất của mô hình."""

	logits: torch.Tensor
	loss: Optional[torch.Tensor] = None
	ntp_loss: Optional[torch.Tensor] = None
	mtp_loss: Optional[torch.Tensor] = None
	kv_caches: Optional[List[LayerKVCache]] = None


class KLMModel(nn.Module):
	"""
	Mô hình ngôn ngữ thế hệ mới Open-SLM Killer (<10M, 100M, mở rộng tới 1B-7B+ tham số).
	Hỗ trợ DenseFormer, Multi-Token Prediction (MTP), Token Superposition Training (TST),
	Gradient Checkpointing, Chunked Cross-Entropy và xuất mô hình sạch cho thi đấu/inference.
	"""

	def __init__(self, cfg: Union[FullConfig, ModelConfig]) -> None:
		super().__init__()
		if isinstance(cfg, FullConfig):
			self.full_cfg: Optional[FullConfig] = cfg
			self.cfg: ModelConfig = cfg.model
			self.gradient_checkpointing: bool = cfg.training.gradient_checkpointing
		else:
			self.full_cfg = None
			self.cfg = cfg
			self.gradient_checkpointing = False

		self.cfg.apply_preset_if_needed()

		self.vocab_size = self.cfg.vocab_size
		self.d_model = self.cfg.d_model
		self.n_layers = self.cfg.n_layers

		self.tok_embeddings = nn.Embedding(self.vocab_size, self.d_model)
		self.rope = RotaryEmbedding(
			self.cfg.head_dim,
			self.cfg.max_seq_len,
			self.cfg.rope_theta,
			self.cfg.rope_scaling_factor,
		)

		self.layers = nn.ModuleList(
			[TransformerBlock(self.cfg, i) for i in range(self.n_layers)]
		)

		# DenseFormer: Trọng số kết nối xuyên tầng alpha_i in R^{i+2}
		# Khởi tạo [0, ..., 0, 1] để tương đương hệt với Residual connection ở bước 0
		self.use_dense_connections = self.cfg.use_dense_connections
		if self.use_dense_connections:
			self.dense_alphas = nn.ParameterList()
			for i in range(self.n_layers):
				alpha = torch.zeros(i + 2, dtype=torch.float32)
				alpha[-1] = 1.0
				self.dense_alphas.append(nn.Parameter(alpha))
		else:
			self.dense_alphas = None

		self.norm = RMSNorm(self.d_model)
		self.lm_head = nn.Linear(self.d_model, self.vocab_size, False)

		# Multi-Token Prediction (MTP) siêu nhẹ: dự đoán thêm token t+2 lúc train
		self.mtp_num_heads = self.cfg.mtp_num_heads
		if self.mtp_num_heads > 0:
			self.mtp_proj: Optional[nn.Module] = (
				nn.Linear(self.d_model, self.d_model, False)
				if self.mtp_num_heads == 1
				else nn.ModuleList(
					[
						nn.Linear(self.d_model, self.d_model, False)
						for _ in range(self.mtp_num_heads)
					]
				)
			)
		else:
			self.mtp_proj = None

		self._init_weights()

		if self.cfg.tie_word_embeddings:
			self.lm_head.weight = self.tok_embeddings.weight

	def _init_weights(self) -> None:
		std = 0.02
		for module_name, module in self.named_modules():
			if isinstance(module, nn.Linear):
				init_std = std
				if module_name.endswith("o_proj") or module_name.endswith("down_proj"):
					init_std = std / math.sqrt(2.0 * max(1, self.n_layers))
				nn.init.normal_(module.weight, 0.0, init_std)
				if module.bias is not None:
					nn.init.zeros_(module.bias)
			elif isinstance(module, nn.Embedding):
				nn.init.normal_(module.weight, 0.0, std)
			elif isinstance(module, nn.Conv1d):
				nn.init.normal_(module.weight, 0.0, 0.02)
				with torch.no_grad():
					module.weight[:, 0, -1] += 1.0

	def tie_weights(self) -> None:
		"""Khôi phục liên kết trọng số (Weight Tying) sau khi nạp từ Safetensors."""
		if self.cfg.tie_word_embeddings:
			self.lm_head.weight = self.tok_embeddings.weight

	def export_for_inference(self) -> "KLMModel":
		"""Xóa sạch nhánh `mtp_proj` khi chấm điểm hoặc chạy thực tế, tiết kiệm 100% tham số MTP."""
		if self.mtp_proj is not None:
			del self.mtp_proj
			self.mtp_proj = None
		self.mtp_num_heads = 0
		self.gradient_checkpointing = False
		self.eval()
		return self

	def _apply_tst_embedding_pool(
		self,
		embeds: torch.Tensor,
		pool_factor: int,
	) -> torch.Tensor:
		"""Token Superposition Training (TST): Gộp `k` token liên tiếp bằng trung bình có trọng số."""
		if pool_factor <= 1:
			return embeds

		bsz, seq_len, dim = embeds.shape
		usable_len = (seq_len // pool_factor) * pool_factor
		if usable_len == 0:
			return embeds

		cropped = embeds[:, :usable_len, :].view(bsz, usable_len // pool_factor, pool_factor, dim)
		weights = torch.linspace(
			0.5, 1.5, pool_factor,
			device=embeds.device,
			dtype=embeds.dtype
		)
		weights = weights / weights.sum()
		return (cropped * weights.view(1, 1, pool_factor, 1)).sum(2)

	def _compute_ce_loss(
		self,
		hidden: torch.Tensor,
		targets: torch.Tensor,
		ignore_index: int = -100,
	) -> torch.Tensor:
		"""Tính hàm mất mát Cross-Entropy (hỗ trợ `chunked_loss_size > 0` cho mô hình nhiều tỷ tham số)."""
		chunk_size = self.cfg.chunked_loss_size
		flat_hidden = hidden.reshape(-1, self.d_model)
		flat_targets = targets.reshape(-1)

		if chunk_size <= 0 or flat_hidden.shape[0] <= chunk_size:
			logits = self.lm_head(flat_hidden).float()
			return F.cross_entropy(logits, flat_targets, ignore_index=ignore_index)

		total_loss = torch.zeros((), device=hidden.device, dtype=torch.float32)
		valid_tokens = (flat_targets != ignore_index).sum().clamp(min=1)
		for start in range(0, flat_hidden.shape[0], chunk_size):
			end = min(start + chunk_size, flat_hidden.shape[0])
			chunk_logits = self.lm_head(flat_hidden[start:end]).float()
			chunk_loss = F.cross_entropy(
				chunk_logits,
				flat_targets[start:end],
				ignore_index=ignore_index,
				reduction="sum",
			)
			total_loss = total_loss + chunk_loss
		return total_loss / valid_tokens

	def forward(
		self,
		input_ids: torch.Tensor,
		targets: Optional[torch.Tensor] = None,
		kv_caches: Optional[List[LayerKVCache]] = None,
		use_cache: bool = False,
		tst_pool_factor: int = 1,
		ignore_index: int = -100,
	) -> ModelOutput:
		x = self.tok_embeddings(input_ids)

		if tst_pool_factor > 1 and self.training:
			x = self._apply_tst_embedding_pool(x, tst_pool_factor)
			if targets is not None:
				usable_len = (targets.shape[1] // tst_pool_factor) * tst_pool_factor
				targets = targets[:, tst_pool_factor - 1: usable_len: tst_pool_factor]

		history: List[torch.Tensor] = [x] if self.use_dense_connections else []
		new_caches: Optional[List[LayerKVCache]] = [] if use_cache else None

		for i, layer in enumerate(self.layers):
			layer_cache = kv_caches[i] if (kv_caches is not None and i < len(kv_caches)) else None

			if self.training and self.gradient_checkpointing and not use_cache:
				def _custom_forward(
					hidden_in: torch.Tensor,
					block: TransformerBlock = layer
				) -> torch.Tensor:
					out_tensor, _ = block(hidden_in, self.rope)
					return out_tensor

				x = checkpoint(_custom_forward, x, use_reentrant=False)
				updated_cache = None
			else:
				x, updated_cache = layer(
					x, rope=self.rope, kv_cache=layer_cache, use_cache=use_cache
				)

			if new_caches is not None and updated_cache is not None:
				new_caches.append(updated_cache)

			if self.use_dense_connections and self.dense_alphas is not None:
				history.append(x)
				alpha = self.dense_alphas[i].to(dtype=x.dtype)
				stacked = torch.stack(history, 0)
				x = torch.einsum("k,kbtd->btd", alpha, stacked)
				history[-1] = x

		hidden_states = self.norm(x)

		if targets is None:
			logits = self.lm_head(hidden_states)
			return ModelOutput(logits, kv_caches=new_caches)

		if self.cfg.chunked_loss_size > 0:
			ntp_loss = self._compute_ce_loss(hidden_states, targets, ignore_index)
			logits = self.lm_head(hidden_states[:, -1:, :])
		else:
			logits = self.lm_head(hidden_states)
			ntp_loss = F.cross_entropy(
				logits.view(-1, self.vocab_size).float(),
				targets.reshape(-1),
				ignore_index=ignore_index,
			)

		total_loss = ntp_loss
		mtp_loss: Optional[torch.Tensor] = None

		if self.training and self.mtp_proj is not None and hidden_states.shape[1] > 1:
			mtp_heads = [self.mtp_proj] if isinstance(self.mtp_proj, nn.Linear) else list(self.mtp_proj)
			mtp_losses: List[torch.Tensor] = []
			for k_idx, proj_head in enumerate(mtp_heads, 1):
				if hidden_states.shape[1] <= k_idx:
					continue
				mtp_hidden = self.norm(proj_head(hidden_states[:, :-k_idx, :]))
				mtp_target = targets[:, k_idx:]
				mtp_losses.append(
					self._compute_ce_loss(mtp_hidden, mtp_target, ignore_index)
				)
			if mtp_losses:
				mtp_loss = torch.stack(mtp_losses).mean()
				total_loss = ntp_loss + 0.3 * mtp_loss

		return ModelOutput(
			logits, total_loss,
			ntp_loss.detach(),
			mtp_loss.detach() if mtp_loss is not None else None,
			new_caches
		)

	@torch.inference_mode()
	def generate(
		self,
		input_ids: torch.Tensor,
		max_new_tokens: int = 128,
		temperature: float = 0.7,
		top_k: int = 40,
		top_p: float = 0.9,
		eos_token_id: Optional[int] = None,
	) -> torch.Tensor:
		"""Sinh văn bản tự hồi quy tốc độ cao với KV-Cache và ShortConv State Cache."""
		self.eval()
		bsz = input_ids.shape[0]
		device = input_ids.device

		out = self(input_ids=input_ids, use_cache=True)
		kv_caches = out.kv_caches
		next_token_logits = out.logits[:, -1, :]

		generated = input_ids
		finished = torch.zeros(bsz, dtype=torch.bool, device=device)

		for _ in range(max_new_tokens):
			if temperature <= 1e-5:
				next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
			else:
				scaled_logits = next_token_logits / temperature
				if top_k > 0:
					v, _ = torch.topk(scaled_logits, min(top_k, scaled_logits.size(-1)))
					scaled_logits[scaled_logits < v[:, [-1]]] = -float("Inf")
				if 0.0 < top_p < 1.0:
					sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True)
					cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
					sorted_indices_to_remove = cumulative_probs > top_p
					sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
					sorted_indices_to_remove[..., 0] = 0
					indices_to_remove = sorted_indices_to_remove.scatter(
						1, sorted_indices, sorted_indices_to_remove
					)
					scaled_logits[indices_to_remove] = -float("Inf")
				probs = F.softmax(scaled_logits, dim=-1)
				next_token = torch.multinomial(probs, num_samples=1)

			generated = torch.cat([generated, next_token], dim=1)

			if eos_token_id is not None:
				finished = finished | (next_token.squeeze(-1) == eos_token_id)
				if bool(finished.all()):
					break

			step_out = self(input_ids=next_token, kv_caches=kv_caches, use_cache=True)
			kv_caches = step_out.kv_caches
			next_token_logits = step_out.logits[:, -1, :]

		return generated


__all__ = [
	"RMSNorm",
	"RotaryEmbedding",
	"DepthwiseShortConv",
	"SwiGLUMLP",
	"LayerKVCache",
	"GatedGroupedQueryAttention",
	"TransformerBlock",
	"ModelOutput",
	"KLMModel",
	"count_parameters",
	"estimate_config_parameters",
	"save_safetensors_checkpoint",
	"load_safetensors_checkpoint",
	"_unwrap_model",
]
