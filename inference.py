"""
Kịch bản Suy luận & Sinh văn bản LLaDA Diffusion (`inference.py`)
================================================================
- Giải mã khuếch tán bán tự hồi quy (Confidence-based Masked Diffusion Sampler).
- Khởi tạo vùng sinh bằng các token <|mask|>, unmask song song toàn bộ chuỗi qua N bước.
- Hỗ trợ CLI tương tác trực tiếp hoặc chạy đơn với tham số `--prompt`.
"""

from __future__ import annotations

import argparse
import sys
import time

import torch

from src.config_loader import FullConfig, load_config
from src.model import KLMModel, count_params, load_checkpoint
from tokenizer.build_tokenizer import Tokenizer


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
) -> str:
	"""
	Sinh văn bản với LLaDA Masked Diffusion Sampler:
	- Giữ nguyên Prompt, điền MASK vào phần sinh, unmask dần qua các bước khuếch tán.
	"""
	model.eval()
	formatted_prompt = (
		prompt_text if prompt_text.startswith("<|bos|>")
		else f"<|bos|><|user|> {prompt_text} <|assistant|>"
	)
	prompt_ids = tokenizer.encode(formatted_prompt)
	prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)

	t0 = time.perf_counter()
	generated_tensor = model.generate(
		prompt_ids=prompt_tensor,
		gen_length=gen_length,
		steps=steps,
		temperature=temperature,
		remask_schedule=remask_schedule,
	)
	elapsed = max(1e-6, time.perf_counter() - t0)

	# Lấy phần token vừa sinh ra (bỏ qua phần prompt)
	gen_ids = generated_tensor[0, len(prompt_ids):].tolist()
	decoded_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

	speed = gen_length / elapsed
	print(f"{decoded_text.strip()}")
	print(f"  [LLaDA Diffusion: {steps} steps | {gen_length} tokens trong {elapsed:.2f}s -> {speed:.1f} tok/s]")
	return decoded_text


def main() -> None:
	parser = argparse.ArgumentParser(description="Thử nghiệm sinh văn bản với LLaDA Diffusion.")
	parser.add_argument("--config", type=str, default="config.toml", help="Đường dẫn tới config.toml")
	parser.add_argument("--prompt", type=str, default=None, help="Prompt câu hỏi")
	parser.add_argument("--gen-length", type=int, default=64, help="Độ dài văn bản sinh thêm (số token)")
	parser.add_argument("--steps", type=int, default=32, help="Số bước khuếch tán (diffusion steps)")
	parser.add_argument("--temperature", type=float, default=0.0, help="Nhiệt độ lấy mẫu (0 = greedy confidence)")
	parser.add_argument("--schedule", type=str, default="cosine", choices=["cosine", "linear"], help="Lịch trình unmasking")
	args = parser.parse_args()

	cfg: FullConfig = load_config(args.config)
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

	if args.prompt is not None or not sys.stdin.isatty():
		prompt_str = args.prompt or "Calculate: (125 + 75) * 4"
		print(f"\nUser > {prompt_str}")
		print("Assistant > ", end="")
		generate_llada(
			model, tokenizer, prompt_str, device,
			args.gen_length, args.steps, args.temperature, args.schedule
		)
		return

	print("\n" + "=" * 72)
	print(" GIAO DIỆN THỬ NGHIỆM LLADA DIFFUSION (Gõ 'exit' hoặc 'quit' để thoát)")
	print("=" * 72)
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

		print("Assistant > ", end="")
		generate_llada(
			model, tokenizer, user_in, device,
			args.gen_length, args.steps, args.temperature, args.schedule
		)


if __name__ == "__main__":
	main()
