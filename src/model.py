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
	has_checkpoint,
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

		# Compute LLaDA ELBO Masked Cross-Entropy Loss (weighted by 1 / r_b per sequence)
		valid_mask = mask_indices & (targets != ignore_index)
		num_masked = valid_mask.sum()
		if num_masked > 0:
			token_losses = F.cross_entropy(
				logits.view(-1, self.vocab_size),
				targets.view(-1),
				ignore_index=ignore_index,
				reduction="none",
			).view_as(targets)
			masked_losses = token_losses * valid_mask
			per_seq_counts = valid_mask.sum(dim=-1).clamp(min=1).to(dtype=token_losses.dtype)
			per_seq_loss = masked_losses.sum(dim=-1) / per_seq_counts
			active_seqs = (valid_mask.sum(dim=-1) > 0).to(dtype=token_losses.dtype)
			loss = (per_seq_loss * active_seqs).sum() / active_seqs.sum().clamp(min=1.0)
		else:
			loss = F.cross_entropy(
				logits.view(-1, self.vocab_size),
				targets.view(-1),
				ignore_index=ignore_index,
			)

		return ModelOutput(
			logits=logits,
			loss=loss,
			ntp_loss=loss.detach(),
			mtp_loss=None,
		)

	@torch.no_grad()
	def generate(
		self,
		input_ids: Optional[torch.Tensor] = None,
		prompt_ids: Optional[torch.Tensor] = None,
		gen_length: int = 128,
		steps: Optional[int] = None,
		temperature: float = 0.0,
		remask_schedule: str = "cosine",
		block_length: int = 16,
		repetition_penalty: float = 1.25,
		step_callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
	) -> torch.Tensor:
		"""
		LLaDA Semi-Autoregressive Block Diffusion Sampler (Confidence-based Masked Decoding).
		- Divides `gen_length` into semi-autoregressive blocks (`block_length`) to prevent
		  long-range unigram mode collapse (e.g. all `\n` tokens).
		- Applies sliding-window repetition penalty and adjacent duplicate suppression.
		"""
		self.eval()
		total_steps = max(1, steps or self.sampling_steps)
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

		# Initialize full sequence x: preserve prompt tokens, fill remainder with MASK tokens
		x = torch.full((bsz, total_len), self.mask_token_id, dtype=torch.long, device=device)
		if prompt_len > 0:
			x[:, :prompt_len] = prompt

		eff_block_len = max(1, min(block_length, gen_length))
		num_blocks = math.ceil(gen_length / eff_block_len)
		steps_per_block = max(1, math.ceil(total_steps / num_blocks))
		global_step = 0
		effective_total_steps = num_blocks * steps_per_block

		for b_idx in range(num_blocks):
			b_start = prompt_len + b_idx * eff_block_len
			b_end = min(total_len, b_start + eff_block_len)
			cur_block_len = b_end - b_start

			for s in range(steps_per_block):
				global_step += 1
				# Run bidirectional forward pass up to the end of the current active block
				out = self(input_ids=x[:, :b_end])
				logits = out.logits[:, b_start:b_end, :].clone()  # (B, cur_block_len, V)

				# Disallow predicting control tokens (<|pad|>=0, <|bos|>=1, <|unk|>=3, <|mask|>=4)
				for banned_id in (0, 1, 3, self.mask_token_id):
					if 0 <= banned_id < self.vocab_size:
						logits[:, :, banned_id] = -float("inf")

				# Prevent immediate leading newline collapse at the start of generation
				if b_idx == 0 and cur_block_len > 0 and 208 < self.vocab_size:
					logits[:, 0, 208] -= 5.0

				# Apply sliding-window repetition penalty & adjacent duplicate suppression
				if repetition_penalty > 1.0:
					for b in range(bsz):
						win_start = max(0, b_start - 32)
						window_tokens = [
							int(t) for t in x[b, win_start:b_end].tolist()
							if int(t) != self.mask_token_id
						]
						if window_tokens:
							counts: dict[int, int] = {}
							for tid in window_tokens:
								counts[tid] = counts.get(tid, 0) + 1
							for tid, cnt in counts.items():
								pen = repetition_penalty ** min(cnt, 4)
								val = logits[b, :, tid]
								logits[b, :, tid] = torch.where(val > 0, val / pen, val * pen)

						for pos in range(cur_block_len):
							abs_pos = b_start + pos
							if abs_pos > 0:
								prev_tok = int(x[b, abs_pos - 1].item())
								if prev_tok != self.mask_token_id:
									logits[b, pos, prev_tok] -= 3.5

				# Calculate token probabilities and confidence values
				if temperature <= 1e-5:
					pred_tokens = torch.argmax(logits, dim=-1)
					probs = F.softmax(logits, dim=-1)
					confidence = torch.max(probs, dim=-1).values
				else:
					scaled_logits = logits / max(1e-4, temperature)
					probs = F.softmax(scaled_logits, dim=-1)
					flat_probs = probs.view(-1, self.vocab_size)
					pred_tokens = torch.multinomial(flat_probs, num_samples=1).view(bsz, cur_block_len)
					confidence = probs.gather(-1, pred_tokens.unsqueeze(-1)).squeeze(-1)

				# Update current block mask positions with predicted tokens
				block_slice = x[:, b_start:b_end]
				mask_pos = (block_slice == self.mask_token_id)
				x[:, b_start:b_end] = torch.where(mask_pos, pred_tokens, block_slice)

				if s == steps_per_block - 1:
					if step_callback is not None:
						step_callback(min(global_step, effective_total_steps), effective_total_steps, x.detach())
					break

				# Compute target mask ratio within current block for next step
				progress = (s + 1) / steps_per_block
				unmask_target = max(1, math.ceil(progress * cur_block_len))
				if remask_schedule == "cosine":
					mask_ratio = math.cos(progress * (math.pi / 2.0))
				else:
					mask_ratio = 1.0 - progress

				num_to_mask = max(
					0,
					min(cur_block_len - unmask_target, int(round(mask_ratio * cur_block_len)))
				)

				if num_to_mask > 0:
					for b in range(bsz):
						_, low_conf_indices = torch.topk(confidence[b], k=num_to_mask, largest=False)
						x[b, b_start + low_conf_indices] = self.mask_token_id

				if step_callback is not None:
					step_callback(min(global_step, effective_total_steps), effective_total_steps, x.detach())

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
