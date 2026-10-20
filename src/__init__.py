"""
KLM Core Library (`src`)
====================================
Kiến trúc được chia thành các mô-đun nhỏ gọn, dễ đọc và dễ bảo trì:
- `config_loader.py`: Tự động đọc `config.toml` và quản lý các preset quy mô (10M -> 7B+).
- `layers.py`: Các khối tầng cơ sở (`RMSNorm`, `RotaryEmbedding`, `DepthwiseShortConv`, `SwiGLUMLP`).
- `attention.py`: `GatedGroupedQueryAttention`, `LayerKVCache` và `TransformerBlock`.
- `checkpoint.py`: Lưu/nạp trọng số `.safetensors` (đơn tệp & phân mảnh đa tỷ tham số) và đếm tham số.
- `model.py`: Lớp mô hình chính `KLMModel` (kết hợp DenseFormer, MTP, TST).
- `optim.py`: Bộ tối ưu lai `HybridMuonAdamW` (Newton-Schulz 5 bước) & lịch trình WSD.
- `dataset.py`: Bộ nạp dữ liệu nhị phân `np.memmap` kèm Token Superposition Training (TST).
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
from .checkpoint import estimate_config_parameters

__all__ = [
    "ProjectConfig",
    "ModelConfig",
    "OptimizerConfig",
    "TrainingConfig",
    "RLGRPOConfig",
    "FullConfig",
    "SCALE_PRESETS",
    "load_config",
    "estimate_config_parameters",
]
