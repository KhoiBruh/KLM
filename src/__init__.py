"""
KLM Core Library (`src`)
====================================
Kiến trúc được chia thành các mô-đun nhỏ gọn, dễ đọc và dễ bảo trì:
- `config_loader.py`: Tự động đọc `config.toml` và quản lý các preset quy mô (10M -> 7B+).
- `layers.py`: Các khối tầng cơ sở (`RMSNorm`, `RotaryEmbedding`, `DepthwiseShortConv`, `SwiGLUMLP`).
- `attention.py`: `GatedGQA` (Bidirectional Attention 2 chiều) và `TransformerBlock`.
- `checkpoint.py`: Lưu/nạp trọng số `.safetensors` (đơn tệp & phân mảnh đa tỷ tham số) và đếm tham số.
- `model.py`: Lớp mô hình chính `KLMModel` LLaDA (Masked Generative Modeling & Confidence Sampler).
- `optim.py`: Bộ tối ưu lai `HybridMuonAdamW` (Newton-Schulz 5 bước) & lịch trình WSD.
- `dataset.py`: Bộ nạp dữ liệu nhị phân `np.memmap` tốc độ cao.
- `rl_env.py`: Môi trường kiểm chứng toán học/logic (Verifiable Sandbox) & thuật toán GRPO.
"""

from .config_loader import (
	FullConfig,
	ModelConfig,
	OptimizerConfig,
	ProjectConfig,
	RLGRPOConfig,
	SCALE_PRESETS,
	TrainingConfig,
	load_config,
)
from .checkpoint import est_config_params

__all__ = [
	"ProjectConfig",
	"ModelConfig",
	"OptimizerConfig",
	"TrainingConfig",
	"RLGRPOConfig",
	"FullConfig",
	"SCALE_PRESETS",
	"load_config",
    "est_config_params",
]
