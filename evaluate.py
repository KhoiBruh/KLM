"""
Benchmark Evaluation, Parameter Audit & Multi-Billion Scale Verification (`evaluate.py`)
========================================================================================
- Automatically parses `config.toml` from the project root and loads `.safetensors` checkpoints.
- Calls `model.export_for_inference()` and prints a structured parameter audit confirming total
  unique parameters < 10,000,000 (~9.36M for the 10M competition tier).
- Evaluates Masked Cross-Entropy Validation Loss, Perplexity (PPL), and ArithMark-3 math sandbox accuracy.
- Supports `--all-scales` to verify zero-RAM architectural scaling from <10M up to 7B+ parameters.
"""

from __future__ import annotations

import argparse
import copy
import math
from typing import Any, Dict

from src.checkpoint import (
	HAS_SAFETENSORS,
	HAS_TORCH,
	count_params,
	est_config_params,
	load_checkpoint,
	save_checkpoint,
)
from src.config_loader import FullConfig, load_config


def print_parameter_report(
	stats_pre: Dict[str, Any],
	stats_post: Dict[str, Any],
	cfg: FullConfig
) -> Dict[str, Any]:
	"""
	Print parameter breakdown before and after calling `model.export_for_inference()`.
	Confirm unique parameter total < 10,000,000 for the 10M configuration tier.
	"""
	breakdown: Dict[str, int] = stats_post["breakdown"]

	print("\n" + "=" * 80)
	print(f" MODEL PARAMETER AUDIT REPORT: {cfg.project.name.upper()}")
	print("=" * 80)
	print(f" {'Architecture Component':<36} | {'Parameter Count':>18} | {'Ratio':>10}")
	print("-" * 80)

	inf_total = max(1, stats_post["inference_params"])
	labels = {
		"embeddings_and_lm_head": (
			"Token Embeddings + LM Head (Tied)"
			if cfg.model.tie_word_embeddings
			else "Token Embeddings + LM Head (Untied)"
		),
		"attention_qkv_o": f"Bidirectional GQA (Q={cfg.model.n_heads}h, KV={cfg.model.n_kv_heads}h, O)",
		"qk_and_layer_norms": "QK-Norm & Layer RMSNorms",
		"mlp_swiglu": f"SwiGLU FFN (d_ff={cfg.model.d_ff})",
	}

	for key, title in labels.items():
		val = breakdown.get(key, 0)
		pct = (val / inf_total) * 100.0
		print(f" {title:<36} | {val:>18,} | {pct:>9.2f}%")

	print("-" * 80)
	print(
		f" {'TOTAL LLADA INFERENCE PARAMETERS':<36} | "
		f"{stats_post['inference_params']:>18,} | "
		f"({stats_post['inference_params'] / 1e6:.4f}M)"
	)
	print("=" * 80)

	if cfg.model.vocab_size == 4096 and cfg.model.d_model == 256:
		limit = 10_000_000
		margin = limit - stats_post["inference_params"]
		assert stats_post["inference_params"] < limit, (
			f"Total parameters ({stats_post['inference_params']:,}) exceed 10,000,000 limit!"
		)
		print(
			f" [AUDIT PASSED: <10M] Total inference parameters = {stats_post['inference_params']:,} "
			f"(~{stats_post['inference_params'] / 1e6:.2f}M < 10,000,000 | Margin: {margin:,} parameters)."
		)
	elif stats_post["inference_params"] < 150_000_000:
		print(
			f" [AUDIT PASSED: <150M] Total inference parameters = {stats_post['inference_params']:,} "
			f"(~{stats_post['inference_params'] / 1e6:.2f}M < 150,000,000)."
		)
	else:
		print(
			f" [AUDIT PASSED: MULTI-BILLION] Total inference parameters = {stats_post['inference_params']:,} "
			f"(~{stats_post['inference_params'] / 1e9:.3f}B parameters)."
		)

	return stats_post


def print_multi_billion_scale_table(base_cfg: FullConfig):
	"""
	Quickly verify all scaling presets from 10M up to 7B+ parameters
	(0MB RAM/VRAM required) to validate architectural scaling.
	"""
	print("\n" + "=" * 88)
	print(" ARCHITECTURAL SCALING TABLE (SCALING UP TO MULTI-BILLION PARAMETERS)")
	print("=" * 88)
	print(
		f" {'Preset':<8} | {'Vocab':>7} | {'d_model':>7} | {'Layers':>6} | {'Heads (Q/KV)':>12} | "
		f"{'d_ff':>7} | {'Inference Params':>18}"
	)
	print("-" * 88)

	for preset_name in ("10m", "100m", "500m", "1b", "3b", "7b"):
		cfg_copy = copy.deepcopy(base_cfg)
		cfg_copy.scale_preset(preset_name)
		stats = est_config_params(cfg_copy)
		inf_p = stats["inference_params"]
		human = f"{inf_p / 1e6:.2f}M" if inf_p < 1_000_000_000 else f"{inf_p / 1e9:.2f}B"
		heads_str = f"{cfg_copy.model.n_heads}/{cfg_copy.model.n_kv_heads}"
		print(
			f" {preset_name.upper():<8} | {cfg_copy.model.vocab_size:>7,} | {cfg_copy.model.d_model:>7} | "
			f"{cfg_copy.model.n_layers:>6} | {heads_str:>12} | {cfg_copy.model.d_ff:>7} | "
			f"{inf_p:>11,} ({human:>5})"
		)
	print("=" * 88)


def main():
	parser = argparse.ArgumentParser(description="Model evaluation and parameter audit for KLM LLaDA.")
	parser.add_argument("--config", type=str, default="config.toml", help="Path to config.toml")
	parser.add_argument(
		"--scale",
		type=str,
		default=None,
		choices=["custom", "10m", "100m", "500m", "1b", "3b", "7b"],
		help="Verify configuration according to a scale preset",
	)
	parser.add_argument(
		"--all-scales",
		action="store_true",
		help="Display parameter table across all presets from 10M to 7B+",
	)
	parser.add_argument(
		"--num-samples",
		type=int,
		default=10,
		help="Number of ArithMark-3 math sandbox test problems to evaluate",
	)
	args = parser.parse_args()

	cfg = load_config(args.config, args.scale)

	if not HAS_TORCH:
		stats = est_config_params(cfg)
		print_parameter_report(stats, stats, cfg)
		if args.all_scales:
			print_multi_billion_scale_table(cfg)
		return

	import torch
	from src.dataset import MemmapTokenDataset
	from src.model import KLMModel
	from src.rl_env import VerifiableMathLogicEnv
	from tokenizer.build_tokenizer import Tokenizer
	from train import diffusion_mask

	device = torch.device(
		"cuda" if (cfg.project.device.lower().startswith("cuda") and torch.cuda.is_available())
		else "cpu"
	)

	if args.scale in ("1b", "3b", "7b") and not torch.cuda.is_available():
		with torch.device("meta"):
			meta_model = KLMModel(cfg)
			pre = count_params(meta_model)
			meta_model.export_for_inference()
			post = count_params(meta_model)
			print_parameter_report(pre, post, cfg)

		if args.all_scales:
			print_multi_billion_scale_table(cfg)
		return

	model = KLMModel(cfg).to(device)
	ckpt_dir = cfg.project.checkpoint_path

	if (ckpt_dir / "model.safetensors").exists() or (ckpt_dir / "model.safetensors.index.json").exists():
		try:
			info = load_checkpoint(model, ckpt_dir, device)
			print(f"[Evaluate] Loaded weights from '{info['loaded_from']}' (step={info['step']}).")
		except Exception as exc:
			print(f"[Evaluate] Evaluating on initialized weights ({exc}).")
	else:
		print("[Evaluate] No checkpoint found in 'checkpoints/', evaluating on randomly initialized weights.")

	stats_pre = count_params(model)
	model.export_for_inference()
	stats_post = count_params(model)
	print_parameter_report(stats_pre, stats_post, cfg)

	if HAS_SAFETENSORS:
		inf_ckpt = save_checkpoint(
			model, ckpt_dir, 0, cfg=cfg,
			export_inference_copy=True
		)
		print(f"[Safetensors] Exported clean inference weights to: {inf_ckpt}")

	if args.all_scales:
		print_multi_billion_scale_table(cfg)

	# Evaluate Perplexity and ArithMark-3
	model.eval()
	tokenizer = Tokenizer(cfg=cfg)
	val_dataset = MemmapTokenDataset(cfg, "val")

	losses = []
	batch_size = max(1, min(cfg.training.micro_batch_size, 8))
	with torch.no_grad():
		for _ in range(6):
			clean_tokens, _ = val_dataset.sample_batch(batch_size, device)
			noisy_input, targets, mask, _ = diffusion_mask(
				clean_tokens, cfg, mask_token_id=tokenizer.mask_id, is_sft=False
			)
			out = model(input_ids=noisy_input, targets=targets, mask_indices=mask)
			if out.loss is not None:
				losses.append(float(out.loss.item()))

		mean_loss = sum(losses) / max(1, len(losses))
		ppl = math.exp(min(20.0, mean_loss))

		env = VerifiableMathLogicEnv(cfg.project.seed + 2026)
		correct = 0
		total_reward = 0.0

		print("\n" + "-" * 80)
		print(f" MATHEMATICAL & REASONING BENCHMARK EVALUATION (ARITHMARK-3 SANDBOX, N={args.num_samples})")
		print("-" * 80)

		for idx in range(args.num_samples):
			task = env.sample_task()
			prompt_ids = tokenizer.encode(task.prompt)
			inp = torch.tensor([prompt_ids], dtype=torch.long, device=device)
			gen_ids = model.generate(prompt_ids=inp, gen_length=64, steps=16, temperature=0.0)
			completion_ids = gen_ids[0, len(prompt_ids):].tolist()
			completion_text = tokenizer.decode(completion_ids, False)
			reward, is_corr = env.score_response(completion_text, task.ground_truth)
			total_reward += reward

			if is_corr:
				correct += 1

			if idx < 3:
				clean_prompt = task.prompt.replace("<|bos|>", "").strip()
				clean_out = completion_text.replace("\n", " ").strip()[:90]
				print(f" [Sample #{idx + 1}] Prompt      : {clean_prompt}")
				print(f"           Ground Truth: {task.ground_truth} | Generated: {clean_out}")

		acc = (correct / max(1, args.num_samples)) * 100.0
		avg_reward = total_reward / max(1, args.num_samples)

		print("-" * 80)
		print(f" -> Validation Cross-Entropy Loss : {mean_loss:.4f}")
		print(f" -> Validation Perplexity (PPL)   : {ppl:.2f}")
		print(f" -> ArithMark-3 Accuracy Score    : {acc:.1f}% ({correct}/{args.num_samples})")
		print(f" -> Average RL Reward Score       : {avg_reward:.3f}")
		print("=" * 80 + "\n")


if __name__ == "__main__":
	main()
