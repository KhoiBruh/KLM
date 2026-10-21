"""
LLaDA Training Pipeline (Large Language Diffusion with Masking)
===============================================================
- Automatically loads configuration from root `config.toml`.
- Trains under discrete Masked Generative Modeling (Diffusion):
  Sample clean sequences x_0 -> mask random tokens to create noisy x_t -> optimize Masked Cross-Entropy.
- Scales from edge devices (RTX 3050 4GB) to distributed cloud clusters (DDP/FSDP),
  utilizing Mixed Precision (AMP) and the hybrid Muon+AdamW optimizer.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import time
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn

from src.config_loader import FullConfig, load_config
from src.dataset import MemmapTokenDataset
from src.model import (
	KLMModel,
	count_params,
	load_checkpoint,
	save_checkpoint,
)
from src.optim import build_optimizer
from src.rl_env import GRPOTrainer
from tokenizer.build_tokenizer import Tokenizer


def set_global_seed(seed: int):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)


def resolve_device(cfg: FullConfig) -> Tuple[torch.device, torch.dtype, bool]:
	"""Detects available hardware (CUDA/CPU) and selects optimal AMP dtype."""
	req_device = cfg.project.device.lower()
	if req_device.startswith("cuda") and torch.cuda.is_available():
		local_rank = int(os.environ.get("LOCAL_RANK", "0"))
		device = torch.device(f"cuda:{local_rank}")
	else:
		device = torch.device("cpu")

	dtype_str = cfg.project.dtype.lower()
	if device.type == "cuda":
		if (
			dtype_str in ("bfloat16", "bf16", "fp8") and
			torch.cuda.is_bf16_supported()
		):
			return device, torch.bfloat16, True
		if dtype_str in ("float16", "fp16", "bfloat16", "bf16"):
			return device, torch.float16, True
	return device, torch.float32, False


def wrap_distributed_or_compile(
	model: KLMModel,
	cfg: FullConfig,
	device: torch.device,
) -> nn.Module:
	"""Wraps model with DDP / FSDP and optionally enables torch.compile."""
	wrapped: nn.Module = model
	world_size = int(os.environ.get("WORLD_SIZE", "1"))
	strategy = cfg.training.distributed_strategy.lower()

	if world_size > 1 and dist.is_available():
		if not dist.is_initialized():
			backend = "nccl" if device.type == "cuda" else "gloo"
			dist.init_process_group(backend=backend)
		if strategy == "fsdp" or (
			strategy == "auto" and
			count_params(model)["inference_params"] >= 500_000_000
		):
			from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

			wrapped = FSDP(wrapped, device_id=device if device.type == "cuda" else None)
		else:
			from torch.nn.parallel import DistributedDataParallel as DDP

			wrapped = DDP(wrapped, [device.index] if device.type == "cuda" else None)

	if cfg.project.compile and hasattr(torch, "compile") and device.type == "cuda":
		try:
			wrapped = torch.compile(wrapped, mode="max-autotune")  # type: ignore[assignment]
			print("[Compile] Activated torch.compile(mode='max-autotune').")
		except Exception as exc:
			print(f"[Compile] Skipping torch.compile (fallback to eager mode): {exc}")

	return wrapped


def diffusion_mask(
	clean_tokens: torch.Tensor,
	cfg: FullConfig,
	mask_token_id: int,
	is_sft: bool = False,
	assistant_token_id: int = 5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
	"""
	Applies masking noise according to LLaDA principles:
	Samples masking ratio r ~ [r_min, r_max], randomly replaces tokens with `mask_token_id`.
	"""
	bsz, seq_len = clean_tokens.shape
	device = clean_tokens.device
	min_ratio = getattr(cfg.training, "mask_ratio_min", 0.15)
	max_ratio = getattr(cfg.training, "mask_ratio_max", 1.0)
	schedule = getattr(cfg.training, "noise_schedule", "uniform")

	if schedule == "cosine":
		u = torch.rand(bsz, 1, device=device)
		r = torch.cos(u * (math.pi / 2.0)).clamp(min_ratio, max_ratio)
	else:
		r = torch.empty(bsz, 1, device=device).uniform_(min_ratio, max_ratio)

	mean_ratio = float(r.mean().item())
	rand_matrix = torch.rand(bsz, seq_len, device=device)
	mask = rand_matrix < r

	# Ensure at least one token is masked per row
	zero_mask = mask.sum(1) == 0
	if zero_mask.any():
		num_zeros = int(zero_mask.sum().item())
		rand_pos = torch.randint(0, seq_len, (num_zeros,), device=device)
		mask[zero_mask, rand_pos] = True

	if is_sft:
		# Preserve prompt tokens, only mask completion tokens after <|assistant|>
		for b in range(bsz):
			assist_pos = (clean_tokens[b] == assistant_token_id).nonzero(as_tuple=True)[0]
			if len(assist_pos) > 0:
				mask[b, : assist_pos[0] + 1] = False

	noisy_input = clean_tokens.clone()
	noisy_input[mask] = mask_token_id
	return noisy_input, clean_tokens, mask, mean_ratio


@torch.no_grad()
def evaluate_validation_loss(
	model: nn.Module,
	val_dataset: MemmapTokenDataset,
	cfg: FullConfig,
	device: torch.device,
	amp_dtype: torch.dtype,
	use_amp: bool,
	mask_token_id: int,
	eval_batches: int = 8,
) -> Dict[str, float]:
	model.eval()
	losses = []
	batch_size = max(1, min(cfg.training.micro_batch_size, 16))
	for _ in range(eval_batches):
		clean_tokens, _ = val_dataset.sample_batch(batch_size, device)
		noisy_input, targets, mask, _ = diffusion_mask(
			clean_tokens, cfg, mask_token_id
		)
		with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
			out = model(input_ids=noisy_input, targets=targets, mask_indices=mask)
			if out.loss is not None:
				losses.append(float(out.loss.item()))

	model.train()
	mean_loss = float(np.mean(losses)) if losses else 0.0
	ppl = float(math.exp(min(20.0, mean_loss)))
	return {"val_loss": mean_loss, "val_ppl": ppl}


def train(
	cfg: Optional[FullConfig] = None,
	stage_override: Optional[str] = None,
	max_steps_override: Optional[int] = None,
	scale_override: Optional[str] = None,
	batch_size_override: Optional[int] = None,
	grad_accum_override: Optional[int] = None,
):
	if cfg is None:
		cfg = load_config(scale_override=scale_override)
	elif scale_override:
		cfg.scale_preset(scale_override)

	if stage_override:
		cfg.training.stage = stage_override.lower()
	if max_steps_override is not None:
		if cfg.training.stage == "grpo":
			cfg.rl_grpo.steps = max_steps_override
		else:
			cfg.training.total_steps = max_steps_override

	if batch_size_override is not None:
		cfg.training.micro_batch_size = batch_size_override
	if grad_accum_override is not None:
		cfg.training.grad_accum_steps = grad_accum_override

	set_global_seed(cfg.project.seed)
	device, amp_dtype, use_amp = resolve_device(cfg)

	tokenizer = Tokenizer(cfg=cfg)
	base_model = KLMModel(cfg).to(device=device)
	param_stats = count_params(base_model)

	print("=" * 78)
	print(f" LAUNCHING LLADA DIFFUSION TRAINING: {cfg.project.name.upper()} | STAGE: {cfg.training.stage.upper()}")
	print("=" * 78)
	print(f" - Hardware device       : {device} (AMP={use_amp}, dtype={amp_dtype})")
	print(
		f" - Architecture scale     : d_model={cfg.model.d_model}, "
		f"n_layers={cfg.model.n_layers}, "
		f"n_heads={cfg.model.n_heads}/{cfg.model.n_kv_heads}, "
		f"d_ff={cfg.model.d_ff}, vocab={cfg.model.vocab_size}"
	)
	print(f" - Attention mode         : Bidirectional (is_causal={cfg.model.is_causal})")
	print(f" - Mask Token ID          : {tokenizer.mask_id} (<|mask|>)")
	print(
		f" - Total parameters       : {param_stats['training_params']:,} ({param_stats['training_params'] / 1e6:.3f}M)"
	)
	print(f" - Checkpoint storage     : Safetensors ({cfg.project.checkpoint_path})")
	print("=" * 78)

	stage = cfg.training.stage.lower()
	lr_override = cfg.rl_grpo.learning_rate if stage == "grpo" else None
	optimizer = build_optimizer(base_model, cfg=cfg, lr_override=lr_override)
	print(
		f"[Optimizer] HybridMuonAdamW -> Muon (2D): {optimizer.muon_param_count:,} params | "
		f"AdamW (Emb/Norm/1D): {optimizer.adamw_param_count:,} params"
	)

	# Restore from existing safetensors checkpoint if available
	ckpt_dir = cfg.project.checkpoint_path
	start_step = 0
	if (
		(ckpt_dir / "model.safetensors").exists() or
		(ckpt_dir / "model.safetensors.index.json").exists()
	):
		try:
			resume_info = load_checkpoint(base_model, ckpt_dir, device, optimizer)
			start_step = int(resume_info.get("step", 0))
			print(f"[Checkpoint] Restored weights from '{resume_info['loaded_from']}' (step={start_step}).")
		except Exception as exc:
			print(f"[Checkpoint] Initializing fresh weights (skipped checkpoint: {exc}).")

	model = wrap_distributed_or_compile(base_model, cfg, device)
	scaler = torch.amp.GradScaler(
		device.type,
		enabled=(
			use_amp and
			amp_dtype == torch.float16 and
			device.type == "cuda"
		)
	)

	if stage == "grpo":
		grpo_trainer = GRPOTrainer(model, tokenizer, cfg, device)
		total_rl_steps = cfg.rl_grpo.steps
		t0 = time.perf_counter()
		for step in range(start_step + 1, start_step + total_rl_steps + 1):
			metrics = grpo_trainer.grpo_step(optimizer, scaler, amp_dtype, use_amp)

			if (
				step == 1 or
				step % max(1, min(10, cfg.training.eval_interval)) == 0 or
				step == start_step + total_rl_steps
			):
				elapsed = time.perf_counter() - t0
				print(
					f"[GRPO Step {step:05d}/{start_step + total_rl_steps:05d}] "
					f"loss={metrics['loss']:.4f} | reward={metrics['mean_reward']:.3f} | time={elapsed:.1f}s"
				)

			if step % cfg.training.save_interval == 0 or step == start_step + total_rl_steps:
				saved_path = save_checkpoint(model, ckpt_dir, step, optimizer, cfg=cfg)
				print(f"[Safetensors] Saved GRPO checkpoint to: {saved_path}")

		return

	# Pre-training or SFT with Masked Generative Modeling (LLaDA)
	train_dataset = MemmapTokenDataset(cfg, "train")
	val_dataset = MemmapTokenDataset(cfg, "val")

	total_steps = cfg.training.total_steps
	grad_accum = max(1, cfg.training.grad_accum_steps)
	micro_bsz = cfg.training.micro_batch_size
	is_sft = stage == "sft"

	model.train()
	t_start = time.perf_counter()

	try:
		for step in range(start_step + 1, total_steps + 1):
			lr_mult = optimizer.update_lr(step - 1, total_steps)
			train_dataset.set_progress(step - 1, total_steps)

			optimizer.zero_grad(True)
			accum_loss = 0.0
			accum_ratio = 0.0

			for _ in range(grad_accum):
				clean_tokens, _ = train_dataset.sample_batch(micro_bsz, device)
				noisy_input, targets, mask, ratio = diffusion_mask(
					clean_tokens, cfg,
					tokenizer.mask_id, is_sft,
					tokenizer.assistant_id
				)
				accum_ratio += ratio / grad_accum

				with torch.amp.autocast(device.type, amp_dtype, use_amp):
					out = model(input_ids=noisy_input, targets=targets, mask_indices=mask)
					assert out.loss is not None
					loss = out.loss / grad_accum

				if scaler.is_enabled():
					scaler.scale(loss).backward()
				else:
					loss.backward()

				accum_loss += float(loss.detach().item())

			if scaler.is_enabled():
				scaler.unscale_(optimizer)
				grad_norm = float(
					nn.utils.clip_grad_norm_(model.parameters(), cfg.optimizer.grad_clip).item()
				)
				scaler.step(optimizer)
				scaler.update()
			else:
				grad_norm = float(
					nn.utils.clip_grad_norm_(model.parameters(), cfg.optimizer.grad_clip).item()
				)
				optimizer.step()

			if step > 0 or step == total_steps:
				dt = time.perf_counter() - t_start
				tokens_per_step = micro_bsz * grad_accum * cfg.model.max_seq_len
				tok_per_sec = (tokens_per_step * step) / max(1e-6, dt)
				print(
					f"[{stage.upper()} | Step {step:06d}/{total_steps:06d}] "
					f"loss={accum_loss:.4f} | mask_ratio={accum_ratio:.2f} | "
					f"lr_mult={lr_mult:.3f} | grad_norm={grad_norm:.3f} | "
					f"speed={tok_per_sec:,.0f} tok/s"
				)

			if step % cfg.training.eval_interval == 0 or step == total_steps:
				val_metrics = evaluate_validation_loss(
					model, val_dataset, cfg, device,
					amp_dtype, use_amp, tokenizer.mask_id, 4
				)
				print(
					f"  >>> [Eval @ Step {step:06d}] val_loss={val_metrics['val_loss']:.4f} | "
					f"val_ppl={val_metrics['val_ppl']:.2f}"
				)

			if step % cfg.training.save_interval == 0 or step == total_steps:
				saved_path = save_checkpoint(
					model, ckpt_dir, step, optimizer,
					{"stage": stage, "lr_mult": lr_mult}, cfg
				)
				print(f"  >>> [Safetensors] Saved checkpoint to: {saved_path}")

	except KeyboardInterrupt:
		print(f"\n[Interrupted] Detected Ctrl+C! Saving emergency checkpoint at step {step}...")
		saved_path = save_checkpoint(
			model, ckpt_dir, step, optimizer,
			{"stage": stage, "lr_mult": lr_mult}, cfg
		)
		print(f"[Safetensors] Safely saved checkpoint to: {saved_path}")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Trains LLaDA Diffusion model."
	)
	parser.add_argument("--config", type=str, default="config.toml", help="Path to config.toml")
	parser.add_argument(
		"--stage",
		type=str,
		default=None,
		choices=["pretrain", "sft", "grpo"],
		help="Overrides training stage (pretrain, sft, grpo)",
	)
	parser.add_argument("--steps", type=int, default=None, help="Overrides total training steps")
	parser.add_argument(
		"--scale",
		type=str,
		default=None,
		choices=["custom", "10m", "100m", "500m", "1b", "3b", "7b"],
		help="Quickly switches scale preset",
	)
	parser.add_argument("--batch-size", type=int, default=None, help="Overrides micro_batch_size")
	parser.add_argument("--grad-accum", type=int, default=None, help="Overrides grad_accum_steps")
	return parser.parse_args()


if __name__ == "__main__":
	args = parse_args()
	loaded_cfg = load_config(args.config, args.scale)
	train(
		loaded_cfg,
		args.stage, args.steps, args.scale,
		args.batch_size, args.grad_accum
	)
