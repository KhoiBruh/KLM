"""
KLM Core Library (`src`)
====================================
Modular, highly-maintainable core architecture:
- `config_loader.py`: Master `config.toml` parser with multi-scale presets (10M -> 7B+).
- `layers.py`: Foundational primitives (`RMSNorm`, `RotaryEmbedding`, `SwiGLUMLP`).
- `attention.py`: `BidirectionalGQA` (Full 2-way attention) and `TransformerBlock`.
- `checkpoint.py`: Standard `.safetensors` I/O with auto-sharding and parameter auditing.
- `model.py`: Core `KLMModel` for LLaDA (Masked Generative Modeling & Confidence Sampler).
- `optim.py`: Hybrid `Muon + AdamW` optimizer with 5-step Newton-Schulz and WSD scheduler.
- `dataset.py`: High-throughput binary `np.memmap` dataset pipeline.
- `rl_env.py`: Verifiable mathematical reasoning sandbox and GRPO RL algorithm.
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
