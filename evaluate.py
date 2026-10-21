"""
Kịch bản Đánh giá Benchmark, Đếm Tham số Chính xác & Kiểm tra Quy mô Đa tỷ Tham số (`evaluate.py`)
==================================================================================================
- Tự động đọc `config.toml` tại thư mục gốc và nạp trọng số `.safetensors` từ `checkpoints/`.
- Gọi `model.export_for_inference()` để cắt bỏ hoàn toàn nhánh MTP (`mtp_proj`), sau đó in bảng
  thống kê chi tiết từng nhóm tham số và xác nhận tổng tham số < 10,000,000 (~9.98M cho cấu hình 10M).
- Đánh giá Perplexity (PPL) và chấm điểm trực tiếp bài test số học kiểu ArithMark-3 trên terminal.
- Hỗ trợ `--all-scales` để kiểm tra bảng mở rộng quy mô từ <10M lên tới 7B+ tham số.
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
	In bảng thống kê tham số trước và sau khi gọi `model.export_for_inference()`.
	Khẳng định tổng tham số độc nhất < 10,000,000 cho phân khúc 10M.
	"""
	breakdown: Dict[str, int] = stats_post["breakdown"]

	print("\n" + "=" * 80)
	print(f" BẢNG KIỂM TOÁN THAM SỐ MÔ HÌNH: {cfg.project.name.upper()}")
	print("=" * 80)
	print(f" {'Thành phần Kiến trúc':<36} | {'Số lượng Tham số':>18} | {'Tỷ trọng':>10}")
	print("-" * 80)

	inf_total = max(1, stats_post["inference_params"])
	labels = {
		"embeddings_and_lm_head": (
			"Token Embeddings + LM Head (Tied)"
			if cfg.model.tie_word_embeddings
			else "Token Embeddings + LM Head (Untied)"
		),
		"short_conv": f"Depthwise ShortConv (kernel={cfg.model.short_conv_kernel})",
		"attention_qkv_o": f"GQA Projections (Q={cfg.model.n_heads}h, KV={cfg.model.n_kv_heads}h, O)",
		"attention_output_gate": "Attention Output Gate (Sigmoid/SiLU)",
		"qk_and_layer_norms": "QK-Norm & Layer RMSNorms",
		"denseformer_alphas": "DenseFormer Cross-Layer Alphas",
		"mlp_swiglu": f"SwiGLU FFN (d_ff={cfg.model.d_ff})",
	}

	for key, title in labels.items():
		val = breakdown.get(key, 0)
		pct = (val / inf_total) * 100.0
		print(f" {title:<36} | {val:>18,} | {pct:>9.2f}%")

	print("-" * 80)
	print(
		f" {'Tổng tham số lúc Train (gồm MTP)':<36} | "
		f"{stats_pre['training_params']:>18,} | "
		f"({stats_pre['training_params'] / 1e6:.4f}M)"
	)
	print(
		f" {'Nhánh MTP đã cắt bỏ khi Inference':<36} | "
		f"{-stats_pre['mtp_params']:>18,} | (Đã xóa sạch)"
	)
	print(
		f" {'TỔNG THAM SỐ THI ĐẤU (INFERENCE)':<36} | "
		f"{stats_post['inference_params']:>18,} | "
		f"({stats_post['inference_params'] / 1e6:.4f}M)"
	)
	print("=" * 80)

	if cfg.model.vocab_size == 4096 and cfg.model.d_model == 256:
		limit = 10_000_000
		margin = limit - stats_post["inference_params"]
		assert stats_post["inference_params"] < limit, (
			f"Tổng tham số ({stats_post['inference_params']:,}) vượt ngưỡng 10,000,000!"
		)
		print(
			f" [XÁC NHẬN HỢP LỆ <10M] Tổng tham số Inference = {stats_post['inference_params']:,} "
			f"(~{stats_post['inference_params'] / 1e6:.2f}M < 10,000,000 | Dư {margin:,} tham số)."
		)
	elif stats_post["inference_params"] < 150_000_000:
		print(
			f" [XÁC NHẬN HỢP LỆ <150M] Tổng tham số Inference = {stats_post['inference_params']:,} "
			f"(~{stats_post['inference_params'] / 1e6:.2f}M < 150,000,000)."
		)
	else:
		print(
			f" [XÁC NHẬN QUY MÔ LỚN] Tổng tham số Inference = {stats_post['inference_params']:,} "
			f"(~{stats_post['inference_params'] / 1e9:.3f}B tham số)."
		)

	return stats_post


def print_multi_billion_scale_table(base_cfg: FullConfig) -> None:
	"""
	Kiểm tra nhanh toàn bộ các cấu hình mở rộng từ 10M lên tới 7B+ tham số
	(tốn 0MB RAM/VRAM) để chứng minh khả năng mở rộng kiến trúc lên hàng tỷ tham số.
	"""
	print("\n" + "=" * 88)
	print(" BẢNG MỞ RỘNG QUY MÔ KIẾN TRÚC (SCALING UP TO MULTI-BILLION PARAMETERS)")
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


def main() -> None:
	parser = argparse.ArgumentParser(description="Đánh giá mô hình & kiểm toán tham số Open-SLM Killer.")
	parser.add_argument("--config", type=str, default="config.toml", help="Đường dẫn tới config.toml")
	parser.add_argument(
		"--scale",
		type=str,
		default=None,
		choices=["custom", "10m", "100m", "500m", "1b", "3b", "7b"],
		help="Kiểm tra cấu hình theo preset quy mô",
	)
	parser.add_argument(
		"--all-scales",
		action="store_true",
		help="Hiển thị bảng tham số của mọi quy mô từ 10M lên tới 7B+ tham số",
	)
	parser.add_argument(
		"--num-samples",
		type=int,
		default=10,
		help="Số lượng bài toán ArithMark-3 dùng để chấm điểm",
	)
	args = parser.parse_args()

	cfg = load_config(args.config, args.scale)

	if not HAS_TORCH:
		stats_pre = est_config_params(cfg)
		cfg_inf = copy.deepcopy(cfg)
		cfg_inf.model.mtp_num_heads = 0
		stats_post = est_config_params(cfg_inf)
		print_parameter_report(stats_pre, stats_post, cfg)
		if args.all_scales:
			print_multi_billion_scale_table(cfg)

		return

	import torch
	from src.dataset import MemmapTokenDataset
	from src.model import KLMModel
	from src.rl_env import VerifiableMathLogicEnv
	from tokenizer.build_tokenizer import Tokenizer
	from train import apply_diffusion_mask

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
			print(f"[Evaluate] Đã nạp trọng số từ '{info['loaded_from']}' (step={info['step']}).")
		except Exception as exc:
			print(f"[Evaluate] Dùng trọng số khởi tạo để đánh giá cấu trúc ({exc}).")
	else:
		print("[Evaluate] Chưa có checkpoint trong 'checkpoints/', đánh giá trên trọng số khởi tạo.")

	stats_pre = count_params(model)
	model.export_for_inference()
	stats_post = count_params(model)
	print_parameter_report(stats_pre, stats_post, cfg)

	if HAS_SAFETENSORS:
		inf_ckpt = save_checkpoint(
			model, ckpt_dir, 0, cfg=cfg,
			export_inference_copy=True
		)
		print(f"[Safetensors] Đã xuất trọng số Inference sạch (không MTP) tại: {inf_ckpt}")

	if args.all_scales:
		print_multi_billion_scale_table(cfg)

	# Đánh giá Perplexity và ArithMark-3
	model.eval()
	tokenizer = Tokenizer(cfg=cfg)
	val_dataset = MemmapTokenDataset(cfg, "val")

	losses = []
	batch_size = max(1, min(cfg.training.micro_batch_size, 8))
	with torch.no_grad():
		for _ in range(6):
			clean_tokens, _ = val_dataset.sample_batch(batch_size, device)
			noisy_input, targets, mask, _ = apply_diffusion_mask(
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
		print(f" ĐÁNH GIÁ BENCHMARK TOÁN HỌC & SUY LUẬN (ARITHMARK-3 SANDBOX, N={args.num_samples})")
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
				print(f" [Mẫu #{idx + 1}] Đề bài : {clean_prompt}")
				print(f"          Đáp án chuẩn: {task.ground_truth} | Sinh ra: {clean_out}")

		acc = (correct / max(1, args.num_samples)) * 100.0
		avg_reward = total_reward / max(1, args.num_samples)

		print("-" * 80)
		print(f" -> Validation Cross-Entropy Loss : {mean_loss:.4f}")
		print(f" -> Validation Perplexity (PPL)   : {ppl:.2f}")
		print(f" -> Điểm ArithMark-3 Accuracy     : {acc:.1f}% ({correct}/{args.num_samples})")
		print(f" -> Điểm Thưởng Trung bình (RL)   : {avg_reward:.3f}")
		print("=" * 80 + "\n")


if __name__ == "__main__":
	main()
