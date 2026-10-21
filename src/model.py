"""
Core LLaDA Model (Large Language Diffusion with Masking)
========================================================
Bidirectional Transformer architecture optimized for discrete masked diffusion:
- Full Bidirectional Attention: All tokens attend to each other without causal masking.
- RoPE 1D + RMSNorm + QK-Norm + SwiGLU MLP.
- Masked Generative Modeling: Predicts clean ground-truth tokens at masked positions.
- LLaDA Sampler: Semi-autoregressive diffusion decoding with confidence-based remasking.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, List, Optional, Union

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
	"""Unified model output structure for LLaDA."""

	logits: torch.Tensor
	loss: Optional[torch.Tensor] = None
	ntp_loss: Optional[torch.Tensor] = None
	mtp_loss: Optional[torch.Tensor] = None
	kv_caches: Optional[List[LayerKVCache]] = None


class KLMModel(nn.Module):
	"""
	LLaDA Discrete Diffusion Language Model.
	Supports Masked Cross-Entropy, QK-Norm, Gradient Checkpointing, and LLaDA Sampler.
	"""

	def __init__(self, cfg: Union[FullConfig, ModelConfig]):
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

	def _init_weights(self):
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

	def tie_weights(self):
		"""Restores weight tying after loading state from safetensors."""
		if self.cfg.tie_word_embeddings:
			self.lm_head.weight = self.tok_embeddings.weight

	def export_for_inference(self) -> "KLMModel":
		"""Prepares model for evaluation or production inference."""
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
		Bidirectional forward pass through LLaDA backbone.
		- input_ids: Tensor (B, L) containing noisy sequence x_t with mask tokens.
		- targets: Tensor (B, L) containing ground-truth sequence x_0 to recover.
		- mask_indices: Boolean Tensor (B, L) indicating masked token positions.
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

		# Auto-derive mask indices if not provided explicitly
		if mask_indices is None:
			mask_indices = input_ids == self.mask_token_id

		# Compute Masked Cross-Entropy Loss solely over masked positions
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
		remask_schedule: str = "cosine",
		step_callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
	) -> torch.Tensor:
		"""
		LLaDA Diffusion Sampler (Confidence-based Masked Diffusion Decoding).
		- Supports prompt continuation (infilling / completion) as well as unconditional generation.
		- Simultaneously predicts all positions, unmasking confident tokens and remasking low-confidence tokens.
		"""
		self.eval()
		total_steps = steps or self.sampling_steps
		device = next(self.parameters()).device

		# Standardize prompt input
		if input_ids is not None:
			prompt = input_ids.to(device)
		elif prompt_ids is not None:
			prompt = prompt_ids.to(device)
		else:
			prompt = torch.empty((1, 0), dtype=torch.long, device=device)

		bsz, prompt_len = prompt.shape
		total_len = prompt_len + gen_length

		# Initialize sequence x: preserve prompt tokens, fill remainder with MASK tokens
		x = torch.full((bsz, total_len), self.mask_token_id, dtype=torch.long, device=device)
		if prompt_len > 0:
			x[:, :prompt_len] = prompt

		# Mark fixed prompt positions (never remask prompt tokens)
		is_prompt = torch.zeros((bsz, total_len), dtype=torch.bool, device=device)
		if prompt_len > 0:
			is_prompt[:, :prompt_len] = True

		# Diffusion denoising iterations
		for s in range(total_steps):
			out = self(input_ids=x)
			logits = out.logits.clone()  # (B, L, V)

			# Disallow predicting MASK token directly
			logits[:, :, self.mask_token_id] = -float("inf")

			# Calculate token probabilities and confidence values
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

			# Update current mask positions with predicted tokens
			mask_pos = (x == self.mask_token_id)
			x = torch.where(mask_pos, pred_tokens, x)

			# Final step termination: sequence is completely unmasked
			if s == total_steps - 1:
				if step_callback is not None:
					step_callback(s + 1, total_steps, x.detach())
				break

			# Compute target mask ratio for next step
			progress = (s + 1) / total_steps
			unmask_target = max(1, math.ceil(progress * gen_length))
			if remask_schedule == "cosine":
				mask_ratio = math.cos(progress * (math.pi / 2.0))
			else:  # "linear"
				mask_ratio = 1.0 - progress

			num_to_mask = max(0, min(gen_length - unmask_target, int(round(mask_ratio * gen_length))))

			# Prompt tokens have infinite confidence to prevent remasking
			eff_confidence = confidence.clone()
			eff_confidence[is_prompt] = float("inf")

			# Remask lowest-confidence generated tokens
			if num_to_mask > 0:
				for b in range(bsz):
					gen_confs = eff_confidence[b, prompt_len:]
					if len(gen_confs) > 0:
						actual_to_mask = min(num_to_mask, len(gen_confs))
						_, low_conf_indices = torch.topk(gen_confs, k=actual_to_mask, largest=False)
						x[b, prompt_len + low_conf_indices] = self.mask_token_id

			if step_callback is not None:
				step_callback(s + 1, total_steps, x.detach())

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
