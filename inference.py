"""
LLaDA Diffusion Sampler & Inference Script (`inference.py`)
==========================================================
- Confidence-based Masked Diffusion Sampler (semi-autoregressive / non-autoregressive generation).
- Live Streaming Denoising Display:
  Watch the completion emerge from `<|mask|>` noise step-by-step,
  with real-time instantaneous `tok/s` and `steps/s` counters.
- Supports interactive CLI mode or single-prompt execution via `--prompt`.
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import List

import torch

from src.config_loader import load_config
from src.model import KLMModel, count_params, has_checkpoint, load_checkpoint
from tokenizer.build_tokenizer import Tokenizer


def decode_masked_tokens(
	tokenizer: Tokenizer,
	token_ids: List[int],
	mask_token_id: int,
	mask_char: str = "·",
) -> str:
	"""
	Decode a token sequence containing MASK tokens:
	- Normal tokens are decoded into smooth text.
	- Tokens equal to `mask_token_id` are displayed using `mask_char`.
	"""
	parts: List[str] = []
	buffer: List[int] = []

	def flush_buffer():
		if buffer:
			parts.append(tokenizer.decode(buffer))
			buffer.clear()

	for tid in token_ids:
		if tid == mask_token_id:
			flush_buffer()
			parts.append(mask_char)
		else:
			buffer.append(tid)

	flush_buffer()
	return "".join(parts)


@torch.inference_mode()
def generate_llada(
	model: KLMModel,
	tokenizer: Tokenizer,
	prompt_text: str,
	device: torch.device,
	gen_length: int = 64,
	steps: int = 32,
	temperature: float = 0.0,
	remask_schedule: str = "cosine",
	stream: bool = True,
	print_each_step: bool = False,
	stage: str = "sft",
) -> str:
	"""
	Generate text with the LLaDA Masked Diffusion Sampler accompanied by real-time speed monitoring.
	"""
	model.eval()
	if prompt_text.startswith("<|bos|>"):
		formatted_prompt = prompt_text
	elif stage == "pretrain":
		formatted_prompt = f"<|bos|>{prompt_text}"
	else:
		formatted_prompt = f"<|bos|><|user|> {prompt_text} <|assistant|>"
	prompt_ids = tokenizer.encode(formatted_prompt)
	prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)
	prompt_len = len(prompt_ids)

	t0 = time.perf_counter()
	last_status_len = 0

	def step_callback(step: int, total_steps: int, current_tensor: torch.Tensor):
		nonlocal last_status_len
		if not stream:
			return

		dt = max(1e-6, time.perf_counter() - t0)
		gen_tokens = current_tensor[0, prompt_len:].tolist()
		unmasked_count = sum(1 for t in gen_tokens if t != model.mask_token_id)
		tok_speed = unmasked_count / dt
		step_speed = step / dt

		preview_text = decode_masked_tokens(
			tokenizer, gen_tokens,
			model.mask_token_id, "·"
		)
		preview_clean = preview_text.replace("\n", " ").strip()
		if len(preview_clean) > 60:
			preview_clean = preview_clean[:57] + "..."

		status_line = (
			f"\r  [Denoise {step:02d}/{total_steps:02d} | Unmasked {unmasked_count:02d}/{gen_length} tok | "
			f"{tok_speed:5.1f} tok/s | {step_speed:4.1f} step/s] > {preview_clean}"
		)

		if print_each_step:
			print(status_line)
		else:
			pad = " " * max(0, last_status_len - len(status_line))
			sys.stdout.write(status_line + pad)
			sys.stdout.flush()
			last_status_len = len(status_line)

	generated_tensor = model.generate(
		prompt_ids=prompt_tensor,
		gen_length=gen_length,
		steps=steps,
		temperature=temperature,
		remask_schedule=remask_schedule,
		step_callback=step_callback if stream else None,
	)
	elapsed = max(1e-6, time.perf_counter() - t0)

	# Clear temporary status line to print final output cleanly
	if stream and not print_each_step and last_status_len > 0:
		sys.stdout.write("\r" + " " * (last_status_len + 10) + "\r")
		sys.stdout.flush()

	gen_ids = generated_tensor[0, prompt_len:].tolist()
	# Truncate at first <|eos|>, <|user|>, or <|pad|> token to eliminate trailing blocks
	stop_tokens = {tokenizer.eos_id, tokenizer.pad_id, tokenizer.user_id}
	cut_idx = len(gen_ids)
	for i, tid in enumerate(gen_ids):
		if tid in stop_tokens:
			cut_idx = i
			break

	trimmed_ids = gen_ids[:cut_idx]
	decoded_text = tokenizer.decode(trimmed_ids).strip()

	speed = gen_length / elapsed
	step_speed = steps / elapsed

	print(f"{decoded_text}")
	print(
		f"  [LLaDA Sampler: {steps} steps | {len(trimmed_ids)}/{gen_length} tokens in {elapsed:.2f}s -> "
		f"{speed:.1f} tok/s | {step_speed:.1f} step/s]"
	)
	return decoded_text


def main():
	parser = argparse.ArgumentParser(
		description="Inference and text generation with LLaDA Diffusion Sampler & Live Streaming."
	)
	parser.add_argument("--config", type=str, default="config.toml", help="Path to config.toml")
	parser.add_argument("--prompt", type=str, default=None, help="Input prompt")
	parser.add_argument("--gen-length", type=int, default=64, help="Number of generated tokens")
	parser.add_argument("--steps", type=int, default=32, help="Number of diffusion denoising steps")
	parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (0 = greedy confidence)")
	parser.add_argument("--schedule", type=str, default="cosine", choices=["cosine", "linear"], help="Unmasking schedule")
	parser.add_argument("--no-stream", action="store_true", help="Disable live denoising streaming")
	parser.add_argument("--show-steps", action="store_true", help="Print each step on a new line instead of in-place overwrite")
	args = parser.parse_args()

	cfg = load_config(args.config)
	device = torch.device(
		"cuda" if (cfg.project.device.lower().startswith("cuda") and torch.cuda.is_available())
		else "cpu"
	)

	ckpt_dir = cfg.project.checkpoint_path
	bundled_tok = ckpt_dir / "tokenizer.json"
	tokenizer = Tokenizer(
		tokenizer_path=bundled_tok if bundled_tok.exists() else None,
		cfg=cfg,
	)
	model = KLMModel(cfg).to(device)

	ckpt_stage = cfg.training.stage.lower()
	if has_checkpoint(ckpt_dir):
		try:
			info = load_checkpoint(model, ckpt_dir, device)
			ckpt_stage = str(info.get("stage", ckpt_stage)).lower()
			print(
				f"[Inference] Loaded Safetensors weights from '{info['loaded_from']}' "
				f"(step={info['step']}, stage={ckpt_stage})."
			)
		except Exception as exc:
			print(f"[Inference] Warning when loading checkpoint ({exc}), running with random weights.")
	else:
		print("[Inference] No checkpoint found in 'checkpoints/', running with random initialization.")

	model.export_for_inference()
	stats = count_params(model)
	print(
		f"[Inference] LLaDA model ready on '{device}' | "
		f"Parameters: {stats['inference_params']:,} ({stats['inference_params'] / 1e6:.2f}M) | "
		f"Attention: Bidirectional | Mask ID: {tokenizer.mask_id}"
	)

	stream_enabled = not args.no_stream

	if args.prompt is not None or not sys.stdin.isatty():
		prompt_str = args.prompt or "Calculate: (125 + 75) * 4"
		print(f"\nUser > {prompt_str}")
		print("Assistant > ", end="", flush=True)
		generate_llada(
			model, tokenizer, prompt_str, device,
			args.gen_length, args.steps, args.temperature, args.schedule,
			stream=stream_enabled, print_each_step=args.show_steps,
			stage=ckpt_stage,
		)
		return

	print("\n" + "=" * 74)
	print(" LLADA CONVERSATIONAL INTERACTIVE SHELL (Type 'exit' to quit, '/clear' to reset)")
	print("=" * 74)
	history_prompt = ""

	while True:
		try:
			user_in = input("\nUser > ").strip()
		except (EOFError, KeyboardInterrupt):
			print("\nSession exited.")
			break

		if not user_in:
			continue
		if user_in.lower() in ("/exit", "/quit", "exit", "quit"):
			print("Session exited.")
			break
		if user_in.lower() in ("/clear", "/reset", "clear", "reset"):
			history_prompt = ""
			print("Context cleared. Started new conversation.")
			continue

		if not history_prompt:
			turn_prompt = f"<|bos|><|user|> {user_in} <|assistant|>"
		else:
			turn_prompt = f"{history_prompt} <|user|> {user_in} <|assistant|>"

		print("Assistant > ", end="", flush=True)
		assistant_reply = generate_llada(
			model, tokenizer, turn_prompt, device,
			args.gen_length, args.steps, args.temperature, args.schedule,
			stream_enabled, args.show_steps,
			stage=ckpt_stage,
		)
		history_prompt = f"{turn_prompt} {assistant_reply}"


if __name__ == "__main__":
	main()
