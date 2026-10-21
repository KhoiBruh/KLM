"""
Kịch bản Suy luận & Trực quan hóa LLaDA Diffusion Sampler (`inference.py`)
=========================================================================
- Giải mã khuếch tán bán tự hồi quy (Confidence-based Masked Diffusion Sampler).
- Hiển thị trực quan theo thời gian thực (Live Streaming Denoising Display):
  Xem câu văn dần "hiện hình" từ nhiễu hạt `<|mask|>` qua từng bước, đồng thời
  đo tốc độ tức thời `tok/s` và `steps/s` nhảy số liên tục trên màn hình.
- Hỗ trợ CLI tương tác trực tiếp hoặc chạy đơn với tham số `--prompt`.
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import List

import torch

from src.config_loader import FullConfig, load_config
from src.model import KLMModel, count_params, load_checkpoint
from tokenizer.build_tokenizer import Tokenizer


def decode_masked_tokens(
	tokenizer: Tokenizer,
	token_ids: List[int],
	mask_token_id: int,
	mask_char: str = "·",
) -> str:
	"""
	Giải mã chuỗi token có chứa MASK:
	- Các token bình thường được giải mã thành văn bản mượt mà.
	- Các token đang là `mask_token_id` được biểu diễn bằng ký tự mờ `mask_char`.
	"""
	parts: List[str] = []
	buffer: List[int] = []

	def flush_buffer() -> None:
		if buffer:
			parts.append(tokenizer.decode(buffer, skip_special_tokens=True))
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
) -> str:
	"""
	Sinh văn bản với LLaDA Masked Diffusion Sampler kèm theo đồng hồ đo tok/s thời gian thực.
	"""
	model.eval()
	formatted_prompt = (
		prompt_text if prompt_text.startswith("<|bos|>")
		else f"<|bos|><|user|> {prompt_text} <|assistant|>"
	)
	prompt_ids = tokenizer.encode(formatted_prompt)
	prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)
	prompt_len = len(prompt_ids)

	t0 = time.perf_counter()
	last_status_len = 0

	def step_callback(step: int, total_steps: int, current_tensor: torch.Tensor) -> None:
		nonlocal last_status_len
		if not stream:
			return

		dt = max(1e-6, time.perf_counter() - t0)
		gen_tokens = current_tensor[0, prompt_len:].tolist()
		unmasked_count = sum(1 for t in gen_tokens if t != model.mask_token_id)
		tok_speed = unmasked_count / dt
		step_speed = step / dt

		preview_text = decode_masked_tokens(
			tokenizer, gen_tokens, model.mask_token_id, mask_char="·"
		)
		preview_clean = preview_text.replace("\n", " ").strip()
		if len(preview_clean) > 60:
			preview_clean = preview_clean[:57] + "..."

		status_line = (
			f"\r  [Khử nhiễu {step:02d}/{total_steps:02d} | Mở {unmasked_count:02d}/{gen_length} tok | "
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

	# Xóa dòng trạng thái tạm thời để in kết quả cuối cùng rõ ràng
	if stream and not print_each_step and last_status_len > 0:
		sys.stdout.write("\r" + " " * (last_status_len + 10) + "\r")
		sys.stdout.flush()

	gen_ids = generated_tensor[0, prompt_len:].tolist()
	decoded_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

	speed = gen_length / elapsed
	step_speed = steps / elapsed

	print(f"{decoded_text}")
	print(
		f"  [LLaDA Sampler: {steps} steps | {gen_length} tokens trong {elapsed:.2f}s -> "
		f"{speed:.1f} tok/s | {step_speed:.1f} step/s]"
	)
	return decoded_text


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Thử nghiệm sinh văn bản với LLaDA Diffusion Sampler & Live Streaming."
	)
	parser.add_argument("--config", type=str, default="config.toml", help="Đường dẫn tới config.toml")
	parser.add_argument("--prompt", type=str, default=None, help="Prompt câu hỏi")
	parser.add_argument("--gen-length", type=int, default=64, help="Độ dài văn bản sinh thêm (số token)")
	parser.add_argument("--steps", type=int, default=32, help="Số bước khuếch tán (diffusion steps)")
	parser.add_argument("--temperature", type=float, default=0.0, help="Nhiệt độ lấy mẫu (0 = greedy confidence)")
	parser.add_argument("--schedule", type=str, default="cosine", choices=["cosine", "linear"], help="Lịch trình unmasking")
	parser.add_argument("--no-stream", action="store_true", help="Tắt hiển thị khử nhiễu trực tiếp")
	parser.add_argument("--show-steps", action="store_true", help="In từng bước ra dòng mới thay vì ghi đè")
	args = parser.parse_args()

	cfg = load_config(args.config)
	device = torch.device(
		"cuda" if (cfg.project.device.lower().startswith("cuda") and torch.cuda.is_available())
		else "cpu"
	)

	tokenizer = Tokenizer(cfg=cfg)
	model = KLMModel(cfg).to(device)

	ckpt_dir = cfg.project.checkpoint_path
	if (ckpt_dir / "model.safetensors").exists() or (ckpt_dir / "model.safetensors.index.json").exists():
		try:
			info = load_checkpoint(model, ckpt_dir, device)
			print(f"[Inference] Đã nạp trọng số Safetensors từ '{info['loaded_from']}' (step={info['step']}).")
		except Exception as exc:
			print(f"[Inference] Cảnh báo khi nạp checkpoint ({exc}), chạy với trọng số ngẫu nhiên.")
	else:
		print("[Inference] Chưa có checkpoint trong 'checkpoints/', chạy thử với trọng số khởi tạo ngẫu nhiên.")

	model.export_for_inference()
	stats = count_params(model)
	print(
		f"[Inference] Mô hình LLaDA sẵn sàng trên '{device}' | "
		f"Tham số: {stats['inference_params']:,} ({stats['inference_params'] / 1e6:.2f}M) | "
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
		)
		return

	print("\n" + "=" * 74)
	print(" GIAO DIỆN THỬ NGHIỆM LLADA DIFFUSION SAMPLER (Gõ 'exit' để thoát)")
	print("=" * 74)
	while True:
		try:
			user_in = input("\nUser > ").strip()
		except (EOFError, KeyboardInterrupt):
			print("\nĐã thoát phiên làm việc.")
			break

		if not user_in:
			continue
		if user_in.lower() in ("/exit", "/quit", "exit", "quit"):
			print("Đã thoát phiên làm việc.")
			break

		print("Assistant > ", end="", flush=True)
		generate_llada(
			model, tokenizer, user_in, device,
			args.gen_length, args.steps, args.temperature, args.schedule,
			stream=stream_enabled, print_each_step=args.show_steps,
		)


if __name__ == "__main__":
	main()
