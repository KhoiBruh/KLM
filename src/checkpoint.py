"""
Quản lý Lưu trữ Chuẩn Safetensors & Kiểm toán Tham số Mô hình
=============================================================
Bao gồm:
- `count_parameters`: Đếm chính xác tham số thực tế trên mô hình PyTorch.
- `estimate_config_parameters`: Tính giải tích chính xác 100% số lượng tham số từ `ModelConfig`
  (không cần khởi tạo tensor vào RAM, hỗ trợ kiểm tra ngay cả khi chưa cài đặt PyTorch).
- `save_safetensors_checkpoint` & `load_safetensors_checkpoint`: Lưu và nạp `.safetensors`
  đơn tệp hoặc phân mảnh (Sharded Safetensors) cho quy mô lên tới vài tỷ tham số, tự động
  xử lý Weight Tying (`lm_head.weight` <-> `tok_embeddings.weight`).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .config_loader import FullConfig, ModelConfig

try:
	import torch
	import torch.nn as nn

	HAS_TORCH = True
except ImportError:
	HAS_TORCH = False

try:
	from safetensors.torch import load_file as safetensors_load_file
	from safetensors.torch import save_file as safetensors_save_file

	HAS_SAFETENSORS = True
except ImportError:
	HAS_SAFETENSORS = False


def est_config_params(cfg: Union[FullConfig, ModelConfig]) -> Dict[str, Any]:
	"""
	Tính toán giải tích chính xác tuyệt đối số lượng tham số độc nhất của kiến trúc từ `ModelConfig`.
	Hữu ích để kiểm toán tức thì từ <10M lên tới 7B+ tham số mà không tốn RAM.
	"""
	m = cfg.model if isinstance(cfg, FullConfig) else cfg
	m.apply_preset_if_needed()

	d = m.d_model
	L = m.n_layers
	q_dim = m.n_heads * m.head_dim
	kv_dim = m.n_kv_heads * m.head_dim

	emb_params = m.vocab_size * d
	if not m.tie_word_embeddings:
		emb_params += m.vocab_size * d

	short_conv_params = L * (d * m.short_conv_kernel) if m.use_short_conv else 0
	attn_qkvo_params = L * (d * q_dim + 2 * d * kv_dim + q_dim * d)

	if m.use_attn_output_gate:
		if 0 < m.attn_gate_rank < d:
			gate_params = L * (d * m.attn_gate_rank + m.attn_gate_rank * q_dim)
		else:
			gate_params = L * (d * q_dim)
	else:
		gate_params = 0

	qk_norm_params = L * (2 * m.head_dim) if m.use_qk_norm else 0
	layer_norm_params = L * (2 * d) + d
	norm_params = qk_norm_params + layer_norm_params

	dense_params = sum(i + 2 for i in range(L)) if m.use_dense_connections else 0
	mlp_params = L * (3 * d * m.d_ff)
	mtp_params = m.mtp_num_heads * (d * d) if m.mtp_num_heads > 0 else 0

	inference_params = (
		emb_params
		+ short_conv_params
		+ attn_qkvo_params
		+ gate_params
		+ norm_params
		+ dense_params
		+ mlp_params
	)
	training_params = inference_params + mtp_params

	return {
		"training_params": training_params,
		"inference_params": inference_params,
		"mtp_params": mtp_params,
		"breakdown": {
			"embeddings_and_lm_head": emb_params,
			"short_conv": short_conv_params,
			"attention_qkv_o": attn_qkvo_params,
			"attention_output_gate": gate_params,
			"qk_and_layer_norms": norm_params,
			"denseformer_alphas": dense_params,
			"mlp_swiglu": mlp_params,
			"mtp_head": mtp_params,
			"other": 0,
		},
	}


def count_params(model: Any) -> Dict[str, Any]:
	"""
	Đếm chính xác số lượng tham số độc nhất trên đối tượng mô hình PyTorch
	(loại bỏ trùng lặp do Weight Tying và tách biệt rõ tham số MTP).
	"""
	if isinstance(model, (FullConfig, ModelConfig)) or not HAS_TORCH:
		return est_config_params(model)

	seen_data_ptrs: set[int] = set()
	total_unique = 0
	mtp_params = 0

	breakdown: Dict[str, int] = {
		"embeddings_and_lm_head": 0,
		"short_conv": 0,
		"attention_qkv_o": 0,
		"attention_output_gate": 0,
		"qk_and_layer_norms": 0,
		"denseformer_alphas": 0,
		"mlp_swiglu": 0,
		"mtp_head": 0,
		"other": 0,
	}

	for name, param in model.named_parameters():
		# Trên thiết bị meta, data_ptr() đều bằng 0 nên dùng tên tham số để lọc Weight Tying
		if param.is_meta:
			if name == "lm_head.weight" and getattr(model.cfg, "tie_word_embeddings", True):
				continue
		else:
			ptr = param.data_ptr()
			if ptr in seen_data_ptrs:
				continue
			seen_data_ptrs.add(ptr)

		num = param.numel()
		total_unique += num

		if "mtp_proj" in name:
			mtp_params += num
			breakdown["mtp_head"] += num
		elif "tok_embeddings" in name or "lm_head" in name:
			breakdown["embeddings_and_lm_head"] += num
		elif "short_conv" in name:
			breakdown["short_conv"] += num
		elif "gate_down" in name or "gate_up" in name or "attn.gate_proj" in name:
			breakdown["attention_output_gate"] += num
		elif any(k in name for k in ("q_proj", "k_proj", "v_proj", "o_proj")):
			breakdown["attention_qkv_o"] += num
		elif "norm" in name:
			breakdown["qk_and_layer_norms"] += num
		elif "dense_alphas" in name:
			breakdown["denseformer_alphas"] += num
		elif "mlp." in name:
			breakdown["mlp_swiglu"] += num
		else:
			breakdown["other"] += num

	inference_params = total_unique - mtp_params
	return {
		"training_params": total_unique,
		"inference_params": inference_params,
		"mtp_params": mtp_params,
		"breakdown": breakdown,
	}


def _unwrap_model(model: Any) -> Any:
	"""Lấy mô hình gốc nếu đang bọc trong `torch.compile` hoặc `DDP`/`FSDP`."""
	raw = model
	if hasattr(raw, "_orig_mod"):
		raw = raw._orig_mod
	if hasattr(raw, "module"):
		raw = raw.module
	return raw


def save_checkpoint(
	model: Any,
	checkpoint_dir: Union[str, Path],
	step: int,
	optimizer: Optional[Any] = None,
	extra_state: Optional[Dict[str, Any]] = None,
	cfg: Optional[FullConfig] = None,
	export_inference_copy: bool = False,
) -> Path:
	"""
	Lưu trọng số mô hình theo định dạng chuẩn `.safetensors` (an toàn, mmap 0-copy):
	- Tự động xử lý Weight Tying (`lm_head.weight` dùng chung bộ nhớ với `tok_embeddings.weight`).
	- Tự động chia nhỏ thành nhiều shard (`model-00001-of-0000N.safetensors` + `model.safetensors.index.json`)
	  nếu mô hình ở quy mô nhiều tỷ tham số vượt quá `max_shard_size_gb`.
	"""
	if not HAS_SAFETENSORS:
		raise ImportError(
			"Thư viện 'safetensors' chưa được cài đặt. Hãy chạy: pip install safetensors"
		)

	ckpt_root = Path(checkpoint_dir)
	ckpt_root.mkdir(parents=True, exist_ok=True)
	raw_model = _unwrap_model(model)

	state_dict = raw_model.state_dict()
	tensors_to_save: Dict[str, torch.Tensor] = {}
	seen_ptrs: Dict[int, str] = {}
	tied_keys: Dict[str, str] = {}

	for key, tensor in state_dict.items():
		if export_inference_copy and key.startswith("mtp_proj"):
			continue
		ptr = tensor.data_ptr()
		if ptr in seen_ptrs:
			tied_keys[key] = seen_ptrs[ptr]
			continue
		seen_ptrs[ptr] = key
		tensors_to_save[key] = tensor.detach().cpu().contiguous()

	metadata = {
		"format": "pt",
		"step": str(step),
		"tie_word_embeddings": str(raw_model.cfg.tie_word_embeddings),
		"tied_keys_json": json.dumps(tied_keys),
		"scale_preset": raw_model.cfg.scale_preset,
	}

	max_shard_bytes = int((cfg.project.max_shard_size_gb if cfg else 4.0) * (1024 ** 3))
	total_bytes = sum(t.numel() * t.element_size() for t in tensors_to_save.values())

	if total_bytes <= max_shard_bytes:
		latest_model_path = ckpt_root / "model.safetensors"
		safetensors_save_file(tensors_to_save, str(latest_model_path), metadata=metadata)
		step_model_path = ckpt_root / f"step_{step:07d}.safetensors"
		safetensors_save_file(tensors_to_save, str(step_model_path), metadata=metadata)
		saved_primary = latest_model_path
	else:
		shards: List[Dict[str, torch.Tensor]] = []
		current_shard: Dict[str, torch.Tensor] = {}
		current_size = 0

		for k, v in tensors_to_save.items():
			v_bytes = v.numel() * v.element_size()
			if current_shard and (current_size + v_bytes > max_shard_bytes):
				shards.append(current_shard)
				current_shard = {}
				current_size = 0
			current_shard[k] = v
			current_size += v_bytes
		if current_shard:
			shards.append(current_shard)

		num_shards = len(shards)
		weight_map: Dict[str, str] = {}
		for idx, shard_dict in enumerate(shards, start=1):
			shard_name = f"model-{idx:05d}-of-{num_shards:05d}.safetensors"
			safetensors_save_file(shard_dict, str(ckpt_root / shard_name), metadata=metadata)
			for k in shard_dict:
				weight_map[k] = shard_name

		index_path = ckpt_root / "model.safetensors.index.json"
		index_data = {
			"metadata": {"total_size": total_bytes, **metadata},
			"weight_map": weight_map,
		}
		index_path.write_text(json.dumps(index_data, indent=2), encoding="utf-8")
		saved_primary = index_path

	meta_info = {
		"step": step,
		"model_config": raw_model.cfg.__dict__,
		"tied_keys": tied_keys,
	}
	(ckpt_root / "checkpoint_meta.json").write_text(
		json.dumps(meta_info, indent=2, ensure_ascii=False), encoding="utf-8"
	)

	if optimizer is not None or extra_state is not None:
		trainer_state = {
			"step": step,
			"optimizer": optimizer.state_dict() if optimizer is not None else None,
			"extra_state": extra_state or {},
		}
		torch.save(trainer_state, ckpt_root / "trainer_state.pt")

	return saved_primary


def load_checkpoint(
	model: Any,
	checkpoint_dir_or_file: Union[str, Path],
	device: Union[str, Any] = "cpu",
	optimizer: Optional[Any] = None,
	strict: bool = False,
) -> Dict[str, Any]:
	"""
	Nạp trọng số mô hình từ tệp `.safetensors` (đơn tệp hoặc phân mảnh `model.safetensors.index.json`),
	tự động khôi phục liên kết Weight Tying và trạng thái huấn luyện.
	"""
	if not HAS_SAFETENSORS:
		raise ImportError(
			"Thư viện 'safetensors' chưa được cài đặt. Hãy chạy: pip install safetensors"
		)

	target = Path(checkpoint_dir_or_file)
	ckpt_dir = target.parent if target.is_file() else target

	if target.is_file() and target.suffix == ".safetensors":
		safetensor_file = target
		index_file = None
	elif (ckpt_dir / "model.safetensors.index.json").exists():
		index_file = ckpt_dir / "model.safetensors.index.json"
		safetensor_file = None
	elif (ckpt_dir / "model.safetensors").exists():
		safetensor_file = ckpt_dir / "model.safetensors"
		index_file = None
	else:
		candidates = sorted(ckpt_dir.glob("*.safetensors"), key=lambda p: p.stat().st_mtime)
		if not candidates:
			raise FileNotFoundError(f"Không tìm thấy tệp .safetensors nào trong '{ckpt_dir}'.")
		safetensor_file = candidates[-1]
		index_file = None

	state_dict: Dict[str, torch.Tensor] = {}
	if index_file is not None:
		index_data = json.loads(index_file.read_text(encoding="utf-8"))
		shard_files = sorted(set(index_data.get("weight_map", {}).values()))
		for shard_name in shard_files:
			shard_tensors = safetensors_load_file(str(ckpt_dir / shard_name), device=str(device))
			state_dict.update(shard_tensors)
	elif safetensor_file is not None:
		state_dict = safetensors_load_file(str(safetensor_file), device=str(device))

	raw_model = _unwrap_model(model)
	if (
		raw_model.cfg.tie_word_embeddings
		and "lm_head.weight" not in state_dict
		and "tok_embeddings.weight" in state_dict
	):
		state_dict["lm_head.weight"] = state_dict["tok_embeddings.weight"]

	missing, unexpected = raw_model.load_state_dict(state_dict, strict=strict)
	raw_model.tie_weights()

	info: Dict[str, Any] = {
		"step": 0,
		"missing_keys": missing,
		"unexpected_keys": unexpected,
		"loaded_from": str(index_file or safetensor_file),
	}

	meta_path = ckpt_dir / "checkpoint_meta.json"
	if meta_path.exists():
		try:
			meta = json.loads(meta_path.read_text(encoding="utf-8"))
			info["step"] = int(meta.get("step", 0))
		except Exception:
			pass

	trainer_state_path = ckpt_dir / "trainer_state.pt"
	if trainer_state_path.exists():
		try:
			trainer_state = torch.load(trainer_state_path, map_location=device, weights_only=False)
			info["step"] = int(trainer_state.get("step", info["step"]))
			info["extra_state"] = trainer_state.get("extra_state", {})
			if optimizer is not None and trainer_state.get("optimizer") is not None:
				optimizer.load_state_dict(trainer_state["optimizer"])
		except Exception as exc:
			print(f"[Cảnh báo] Không thể nạp trainer_state.pt: {exc}")

	return info
