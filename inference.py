"""
Kịch bản Chạy Thử Nghiệm & Chat Tốc Độ Cao với KV-Cache (`inference.py`)
========================================================================
- Tự động đọc `config.toml` tại thư mục gốc, nạp bộ `tokenizer/` và trọng số `.safetensors` từ `checkpoints/`.
- Tự động gọi `model.export_for_inference()` để cắt bỏ nhánh MTP giúp tối ưu hóa bộ nhớ.
- Hỗ trợ KV-Cache + ShortConv State Cache hoàn chỉnh để sinh văn bản tốc độ cao trên cả RTX 3050 4GB lẫn CPU,
  kèm giao diện tương tác trực tiếp trên dòng lệnh (Interactive CLI) hoặc chạy 1 lần qua `--prompt`.
"""

from __future__ import annotations

import argparse
import sys
import time

import torch
import torch.nn.functional as F

from src.config_loader import FullConfig, load_config
from src.model import KLMModel, count_parameters, load_safetensors_checkpoint
from tokenizer.build_tokenizer import Tokenizer


@torch.inference_mode()
def generate_streaming(
	model: KLMModel,
	tokenizer: Tokenizer,
	prompt_text: str,
	device: torch.device,
	max_new_tokens: int = 128,
	temperature: float = 0.7,
	top_k: int = 40,
	top_p: float = 0.9,
	stream_to_stdout: bool = True,
) -> str:
	"""
	Sinh văn bản tốc độ cao bằng KV-Cache từng bước và in trực tiếp (streaming) ra màn hình.
	"""
	model.eval()
	formatted_prompt = (
		prompt_text if prompt_text.startswith("<|bos|>")
		else f"<|bos|><|user|> {prompt_text} <|assistant|>"
	)
	input_ids = tokenizer.encode(formatted_prompt)
	input_tensor = torch.tensor([input_ids], torch.long, device)

	t0 = time.perf_counter()
	# Bước 1: Prefill toàn bộ câu hỏi để tạo KV-Cache
	out = model(input_ids=input_tensor, use_cache=True)
	kv_caches = out.kv_caches
	next_logits = out.logits[:, -1, :]

	generated_ids: list[int] = []

	for _ in range(max_new_tokens):
		if temperature <= 1e-5:
			next_token = torch.argmax(next_logits, -1, True)
		else:
			scaled = next_logits / temperature
			if top_k > 0:
				v, _ = torch.topk(scaled, min(top_k, scaled.size(-1)))
				scaled[scaled < v[:, [-1]]] = -float("Inf")
			if 0.0 < top_p < 1.0:
				sorted_logits, sorted_indices = torch.sort(scaled, descending=True)
				cum_probs = torch.cumsum(F.softmax(sorted_logits, -1), -1)
				remove_mask = cum_probs > top_p
				remove_mask[..., 1:] = remove_mask[..., :-1].clone()
				remove_mask[..., 0] = 0
				indices_to_remove = remove_mask.scatter(1, sorted_indices, remove_mask)
				scaled[indices_to_remove] = -float("Inf")
			probs = F.softmax(scaled, -1)
			next_token = torch.multinomial(probs, 1)

		tok_id = int(next_token.item())
		generated_ids.append(tok_id)

		if stream_to_stdout:
			piece = tokenizer.decode([tok_id], False)
			sys.stdout.write(piece)
			sys.stdout.flush()

		if tok_id == tokenizer.eos_id:
			break

		# Bước 2: Giải mã tự hồi quy O(1) với KV-Cache
		step_out = model(next_token, kv_caches, True)
		kv_caches = step_out.kv_caches
		next_logits = step_out.logits[:, -1, :]

	elapsed = max(1e-6, time.perf_counter() - t0)
	num_gen = len(generated_ids)
	if stream_to_stdout:
		print(f"\n  [Tốc độ KV-Cache: {num_gen} tokens trong {elapsed:.2f}s -> {num_gen / elapsed:.1f} tok/s]")

	return tokenizer.decode(generated_ids, False)


def main() -> None:
	parser = argparse.ArgumentParser(description="Chạy thử nghiệm CLI / Sinh văn bản tốc độ cao với KV-Cache.")
	parser.add_argument("--config", type=str, default="config.toml", help="Đường dẫn tới config.toml")
	parser.add_argument("--prompt", type=str, default=None, help="Câu hỏi / Prompt cần sinh văn bản ngay")
	parser.add_argument("--max-new-tokens", type=int, default=96, help="Số token sinh tối đa")
	parser.add_argument("--temperature", type=float, default=0.6, help="Nhiệt độ lấy mẫu (0 = greedy)")
	parser.add_argument("--top-k", type=int, default=32, help="Top-K sampling")
	parser.add_argument("--top-p", type=float, default=0.9, help="Top-P nucleus sampling")
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
			info = load_safetensors_checkpoint(model, ckpt_dir, device)
			print(f"[Inference] Đã nạp trọng số Safetensors từ '{info['loaded_from']}' (step={info['step']}).")
		except Exception as exc:
			print(f"[Inference] Cảnh báo khi nạp checkpoint ({exc}), chạy với trọng số hiện tại.")
	else:
		print("[Inference] Chưa có checkpoint trong 'checkpoints/', chạy thử với trọng số khởi tạo.")

	# Cắt bỏ nhánh MTP trước khi chạy thực tế
	model.export_for_inference()
	stats = count_parameters(model)
	print(
		f"[Inference] Mô hình sẵn sàng trên '{device}' | "
		f"Tham số Inference: {stats['inference_params']:,} ({stats['inference_params'] / 1e6:.2f}M)"
	)

	# Nếu truyền --prompt hoặc môi trường không phải tương tác (non-interactive), chạy 1 câu rồi thoát
	if args.prompt is not None or not sys.stdin.isatty():
		prompt_str = args.prompt or "Calculate: (125 + 75) * 4"
		print(f"\nUser > {prompt_str}")
		sys.stdout.write("Assistant > ")
		generate_streaming(
			model, tokenizer, prompt_str, device,
			args.max_new_tokens, args.temperature,
			args.top_k, args.top_p,
		)
		return

	print("\n" + "=" * 72)
	print(" GIAO DIỆN CHAT TRỰC TIẾP KLM (Gõ 'exit' hoặc 'quit' để thoát)")
	print("=" * 72)
	while True:
		try:
			user_in = input("\nUser > ").strip()
		except (EOFError, KeyboardInterrupt):
			print("\nĐã thoát phiên làm việc.")
			break

		if not user_in:
			continue
		if user_in.lower() in ("/exit", "/quit"):
			print("Đã thoát phiên làm việc.")
			break

		sys.stdout.write("Assistant > ")
		generate_streaming(
			model, tokenizer, user_in, device,
			args.max_new_tokens, args.temperature,
			args.top_k, args.top_p,
		)


if __name__ == "__main__":
	main()
