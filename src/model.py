"""
Mô hình trung tâm LLaDA (Large Language Diffusion with Masking)
==============================================================
Kiến trúc Transformer 2 chiều (Bidirectional Transformer) tối ưu cho học khuếch tán:
- Full Bidirectional Attention: Mọi token chú ý lẫn nhau (không dùng Causal Mask).
- RoPE 1D + RMSNorm + QK-Norm + SwiGLU MLP.
- Masked Generative Modeling: Dự đoán token gốc tại các vị trí bị che (mask).
- LLaDA Sampler: Giải mã khuếch tán bán tự hồi quy dựa trên độ tự tin (Confidence-based Re-masking).
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
	GatedGQA,
	LayerKVCache,
	TransformerBlock,
)
from .checkpoint import (
	_unwrap_model,
	count_params,
	est_config_params,
	load_checkpoint,
	save_checkpoint,
)
from .config_loader import FullConfig, ModelConfig
from .layers import (
	RMSNorm,
	RotaryEmbed,
	SwiGLUMLP,
)


@dataclass
class ModelOutput:
	"""Cấu trúc đầu ra thống nhất của mô hình LLaDA."""

	logits: torch.Tensor
	loss: Optional[torch.Tensor] = None
	ntp_loss: Optional[torch.Tensor] = None
	mtp_loss: Optional[torch.Tensor] = None
	kv_caches: Optional[List[LayerKVCache]] = None


class KLMModel(nn.Module):
	"""
	Mô hình ngôn ngữ khuếch tán LLaDA (Large Language Diffusion with Masking).
	Hỗ trợ Masked Cross-Entropy, QK-Norm, Gradient Checkpointing và LLaDA Sampler.
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
		self.mask_token_id = getattr(self.cfg, "mask_token_id", 4)
		self.sampling_steps = getattr(self.cfg, "sampling_steps", 64)

		self.tok_embeddings = nn.Embedding(self.vocab_size, self.d_model)
		self.rope = RotaryEmbed(
			self.cfg.head_dim,
			self.cfg.max_seq_len,
			self.cfg.rope_theta,
			self.cfg.rope_scaling_factor,
		)

		self.layers = nn.ModuleList(
			[TransformerBlock(self.cfg, i) for i in range(self.n_layers)]
		)

		self.norm = RMSNorm(self.d_model)
		self.lm_head = nn.Linear(self.d_model, self.vocab_size, False)

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

	def tie_weights(self) -> None:
		"""Khôi phục liên kết trọng số (Weight Tying) sau khi nạp từ Safetensors."""
		if self.cfg.tie_word_embeddings:
			self.lm_head.weight = self.tok_embeddings.weight

	def export_for_inference(self) -> "KLMModel":
		"""Chuẩn bị mô hình cho chế độ suy luận / thi đấu."""
		self.gradient_checkpointing = False
		self.eval()
		return self

	def forward(
		self,
		input_ids: torch.Tensor,
		targets: Optional[torch.Tensor] = None,
		mask_indices: Optional[torch.Tensor] = None,
		attention_mask: Optional[torch.Tensor] = None,
		ignore_index: int = -100
	) -> ModelOutput:
		"""
		Forward 2 chiều (Bidirectional) qua mạng LLaDA.
		- input_ids: Tensor (B, L) chứa chuỗi đã bị mask ngẫu nhiên x_t.
		- targets: Tensor (B, L) chứa chuỗi gốc x_0 cần phục hồi.
		- mask_indices: Tensor Boolean (B, L) đánh dấu các vị trí bị mask (True = bị mask).
		"""
		x = self.tok_embeddings(input_ids)

		for layer in self.layers:
			if self.training and self.gradient_checkpointing:
				def _custom_forward(hidden_in: torch.Tensor, block: TransformerBlock = layer) -> torch.Tensor:
					out_tensor, _ = block(hidden_in, self.rope, attention_mask=attention_mask)
					return out_tensor

				x = checkpoint(_custom_forward, x, use_reentrant=False)
			else:
				x, _ = layer(x, rope=self.rope, attention_mask=attention_mask)

		hidden_states = self.norm(x)
		logits = self.lm_head(hidden_states)

		if targets is None:
			return ModelOutput(logits=logits)

		# Xác định mặt nạ các vị trí bị mask nếu chưa được truyền tường minh
		if mask_indices is None:
			mask_indices = input_ids == self.mask_token_id

		# Tính Masked Cross-Entropy Loss (chỉ tính trên các token bị mask)
		num_masked = mask_indices.sum()
		if num_masked > 0:
			masked_targets = targets.clone()
			masked_targets[~mask_indices] = ignore_index

			loss = F.cross_entropy(
				logits.view(-1, self.vocab_size).float(),
				masked_targets.view(-1),
				ignore_index=ignore_index,
			)
		else:
			loss = F.cross_entropy(
				logits.view(-1, self.vocab_size).float(),
				targets.view(-1),
				ignore_index=ignore_index,
			)

		return ModelOutput(
			logits=logits,
			loss=loss,
			ntp_loss=loss.detach(),
			mtp_loss=None,
		)

	@torch.inference_mode()
	def generate(
		self,
		input_ids: Optional[torch.Tensor] = None,
		prompt_ids: Optional[torch.Tensor] = None,
		gen_length: int = 128,
		steps: Optional[int] = None,
		temperature: float = 0.0,
		remask_schedule: str = "cosine"
	) -> torch.Tensor:
		"""
		Thuật toán giải mã khuếch tán LLaDA (Confidence-based Masked Diffusion Sampler).
		- Hỗ trợ cả sinh nối tiếp từ Prompt (infilling / completion) và sinh tự do.
		- Tại mỗi bước diffusion, mô hình dự đoán đồng thời toàn bộ các vị trí mask,
		  sau đó unmask dần các token có độ tự tin cao nhất và re-mask các token ít tự tin.
		"""
		self.eval()
		total_steps = steps or self.sampling_steps
		device = next(self.parameters()).device

		# Chuẩn hóa đầu vào prompt
		if input_ids is not None:
			prompt = input_ids.to(device)
		elif prompt_ids is not None:
			prompt = prompt_ids.to(device)
		else:
			prompt = torch.empty((1, 0), dtype=torch.long, device=device)

		bsz, prompt_len = prompt.shape
		total_len = prompt_len + gen_length

		# Khởi tạo chuỗi x: giữ nguyên prompt, phần còn lại điền toàn bộ bằng MASK token
		x = torch.full((bsz, total_len), self.mask_token_id, dtype=torch.long, device=device)
		if prompt_len > 0:
			x[:, :prompt_len] = prompt

		# Đánh dấu vùng prompt cố định (không bao giờ re-mask prompt)
		is_prompt = torch.zeros((bsz, total_len), dtype=torch.bool, device=device)
		if prompt_len > 0:
			is_prompt[:, :prompt_len] = True

		# Vòng lặp khử nhiễu khuếch tán (Diffusion Denoising Steps)
		for s in range(total_steps):
			out = self(input_ids=x)
			logits = out.logits  # (B, L, V)

			# Tính xác suất và lấy mẫu token dự đoán
			if temperature <= 1e-5:
				pred_tokens = torch.argmax(logits, dim=-1)
				probs = F.softmax(logits, dim=-1)
				confidence = torch.max(probs, dim=-1).values
			else:
				scaled_logits = logits / max(1e-4, temperature)
				probs = F.softmax(scaled_logits, dim=-1)
				flat_probs = probs.view(-1, self.vocab_size)
				pred_tokens = torch.multinomial(flat_probs, num_samples=1).view(bsz, total_len)
				confidence = probs.gather(-1, pred_tokens.unsqueeze(-1)).squeeze(-1)

			# Cập nhật các vị trí đang là mask bằng token vừa dự đoán
			mask_pos = (x == self.mask_token_id)
			x = torch.where(mask_pos, pred_tokens, x)

			# Nếu là bước cuối cùng, dừng lại (tất cả token đã được điền)
			if s == total_steps - 1:
				break

			# Tính tỷ lệ mask cần giữ lại cho bước tiếp theo theo schedule
			progress = (s + 1) / total_steps
			if remask_schedule == "cosine":
				mask_ratio = math.cos(progress * (math.pi / 2.0))
			else:  # "linear"
				mask_ratio = 1.0 - progress

			num_to_mask = max(1, int(round(mask_ratio * gen_length)))

			# Độ tự tin ở vùng prompt được đặt vô cực để không bao giờ bị re-mask
			eff_confidence = confidence.clone()
			eff_confidence[is_prompt] = float("inf")

			# Re-mask lại num_to_mask token có độ tự tin thấp nhất trong vùng sinh
			# Để đảm bảo token có độ tự tin thấp nhất sẽ được mô hình xem xét lại ở bước sau
			for b in range(bsz):
				gen_confs = eff_confidence[b, prompt_len:]
				if len(gen_confs) > 0:
					actual_to_mask = min(num_to_mask, len(gen_confs))
					_, low_conf_indices = torch.topk(gen_confs, k=actual_to_mask, largest=False)
					x[b, prompt_len + low_conf_indices] = self.mask_token_id

		return x


__all__ = [
	"RMSNorm",
	"RotaryEmbed",
	"SwiGLUMLP",
	"LayerKVCache",
	"GatedGQA",
	"TransformerBlock",
	"ModelOutput",
	"KLMModel",
	"count_params",
	"est_config_params",
	"save_checkpoint",
	"load_checkpoint",
	"_unwrap_model",
]
