"""
Bộ đọc tự động tệp cấu hình trung tâm "config.toml" ở thư mục gốc của dự án.
Hỗ trợ truy cập dạng thuộc tính lồng nhau (cfg.model.d_model) và hệ thống Preset
mở rộng quy mô từ <10M (9.98M), 100M, 500M lên tới 1B, 3B, 7B+ (vài tỷ tham số).
"""

from __future__ import annotations

import tomllib

import copy
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Thư mục gốc của dự án: pathlib.Path(__file__).resolve().parent.parent
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH: Path = PROJECT_ROOT / "config.toml"

# Bảng cấu hình mở rộng quy mô từ <10M lên tới hàng tỷ tham số (Multi-Billion Scale Presets)
SCALE_PRESETS: Dict[str, Dict[str, Any]] = {
	"10m": {
		"vocab_size": 4096,
		"d_model": 256,
		"n_layers": 12,
		"n_heads": 8,
		"n_kv_heads": 2,
		"head_dim": 32,
		"d_ff": 688,
		"max_seq_len": 1024,
		"attn_gate_rank": 98,
		"tie_word_embeddings": True,
		"chunked_loss_size": 0,
	},
	"100m": {
		# Đánh bại GPT-X3-150M ở phân khúc <150M tham số (~148.9M inference params)
		"vocab_size": 32768,
		"d_model": 640,
		"n_layers": 28,
		"n_heads": 10,
		"n_kv_heads": 2,
		"head_dim": 64,
		"d_ff": 1760,
		"max_seq_len": 2048,
		"attn_gate_rank": 128,
		"tie_word_embeddings": True,
		"chunked_loss_size": 0,
	},
	"500m": {
		# Phân khúc <500M tham số (~496.5M inference params)
		"vocab_size": 32768,
		"d_model": 1152,
		"n_layers": 28,
		"n_heads": 18,
		"n_kv_heads": 6,
		"head_dim": 64,
		"d_ff": 3328,
		"max_seq_len": 4096,
		"attn_gate_rank": 256,
		"tie_word_embeddings": True,
		"chunked_loss_size": 2048,
	},
	"1b": {
		# Quy mô ~1.02 tỷ tham số
		"vocab_size": 65536,
		"d_model": 1536,
		"n_layers": 32,
		"n_heads": 24,
		"n_kv_heads": 4,
		"head_dim": 64,
		"d_ff": 4096,
		"max_seq_len": 4096,
		"attn_gate_rank": 384,
		"tie_word_embeddings": False,
		"chunked_loss_size": 4096,
	},
	"3b": {
		# Quy mô ~2.93 tỷ tham số
		"vocab_size": 65536,
		"d_model": 2560,
		"n_layers": 36,
		"n_heads": 32,
		"n_kv_heads": 8,
		"head_dim": 80,
		"d_ff": 6912,
		"max_seq_len": 8192,
		"attn_gate_rank": 512,
		"tie_word_embeddings": False,
		"chunked_loss_size": 4096,
	},
	"7b": {
		# Quy mô ~6.96 tỷ tham số (<7B)
		"vocab_size": 128256,
		"d_model": 4096,
		"n_layers": 32,
		"n_heads": 32,
		"n_kv_heads": 8,
		"head_dim": 128,
		"d_ff": 11008,
		"max_seq_len": 8192,
		"attn_gate_rank": 512,
		"tie_word_embeddings": False,
		"chunked_loss_size": 4096,
	},
}


@dataclass
class ProjectConfig:
	name: str = "klm-10m"
	seed: int = 42
	device: str = "cuda"
	dtype: str = "bfloat16"
	compile: bool = True
	checkpoint_dir: str = "checkpoints"
	tokenizer_dir: str = "tokenizer"
	save_format: str = "safetensors"
	max_shard_size_gb: float = 4.0

	@property
	def checkpoint_path(self) -> Path:
		p = Path(self.checkpoint_dir)
		return p if p.is_absolute() else (PROJECT_ROOT / p)

	@property
	def tokenizer_path(self) -> Path:
		p = Path(self.tokenizer_dir)
		return p if p.is_absolute() else (PROJECT_ROOT / p)


@dataclass
class ModelConfig:
	scale_preset: str = "custom"
	vocab_size: int = 4096
	d_model: int = 256
	n_layers: int = 12
	n_heads: int = 8
	n_kv_heads: int = 2
	head_dim: int = 32
	d_ff: int = 688
	max_seq_len: int = 1024
	rope_theta: float = 10000.0
	rope_scaling_factor: float = 1.0
	dropout: float = 0.0
	tie_word_embeddings: bool = True

	# Cấu hình LLaDA Diffusion
	is_causal: bool = False
	mask_token_id: int = 4
	sampling_steps: int = 64

	# Các vũ khí kiến trúc đặc biệt
	use_qk_norm: bool = True
	use_attn_output_gate: bool = False
	attn_gate_rank: int = 0
	use_short_conv: bool = False
	short_conv_kernel: int = 5
	use_dense_connections: bool = False
	mtp_num_heads: int = 0
	chunked_loss_size: int = 0

	def apply_preset_if_needed(self) -> None:
		preset_key = self.scale_preset.strip().lower()
		if preset_key in SCALE_PRESETS and preset_key != "custom":
			preset_vals = SCALE_PRESETS[preset_key]
			for k, v in preset_vals.items():
				setattr(self, k, v)
		# Đảm bảo tính nhất quán của head_dim và n_kv_heads
		if self.head_dim <= 0:
			self.head_dim = self.d_model // self.n_heads
		if self.n_heads % self.n_kv_heads != 0:
			raise ValueError(
				f"n_heads ({self.n_heads}) phải chia hết cho n_kv_heads ({self.n_kv_heads})."
			)


@dataclass
class OptimizerConfig:
	muon_lr: float = 0.02
	muon_momentum: float = 0.95
	muon_ns_steps: int = 5
	adamw_lr: float = 0.003
	weight_decay: float = 0.01
	betas: List[float] = field(default_factory=lambda: [0.9, 0.95])
	grad_clip: float = 1.0


@dataclass
class TrainingConfig:
	stage: str = "pretrain"
	micro_batch_size: int = 32
	grad_accum_steps: int = 16
	total_steps: int = 100000
	warmup_ratio: float = 0.03
	stable_ratio: float = 0.82
	decay_ratio: float = 0.15

	# Cấu hình Masking Diffusion
	mask_ratio_min: float = 0.15
	mask_ratio_max: float = 1.0
	noise_schedule: str = "uniform"

	use_tst: bool = False
	tst_ratio: float = 0.0
	tst_pool_factor: int = 1
	gradient_checkpointing: bool = True
	distributed_strategy: str = "auto"
	low_cpu_mem_usage: bool = False
	eval_interval: int = 1000
	save_interval: int = 5000


@dataclass
class RLGRPOConfig:
	group_size: int = 8
	kl_coef: float = 0.04
	max_gen_len: int = 256
	learning_rate: float = 1e-5
	steps: int = 2000


@dataclass
class DataSourceConfig:
	"""
	Một nguồn dữ liệu trong hỗn hợp huấn luyện (khai báo bằng `[[data.sources]]` trong config.toml).

	type:
	  - "text"    : tệp văn bản thuần (.txt/.md/...), hỗ trợ glob `data/raw/**/*.txt`
	  - "jsonl"   : JSON Lines (.jsonl / .jsonl.gz), lấy `text_field` hoặc ghép bằng `template`
	  - "parquet" : tệp Parquet (cần `pyarrow`)
	  - "hf"      : Hugging Face Hub dataset, đọc streaming (cần `datasets`)
	  - "bin"     : shard token đã mã hoá sẵn (uint16/uint32 memmap) — bỏ qua bước token hoá
	"""

	name: str = "source"
	type: str = "text"
	path: Union[str, List[str]] = ""
	weight: float = 1.0
	enabled: bool = True
	stages: List[str] = field(default_factory=lambda: ["pretrain"])

	# Trích xuất văn bản
	text_field: str = "text"
	template: str = ""  # Ví dụ: "<|user|> {question} <|assistant|> {answer}"
	doc_separator: str = ""  # "" = mỗi tệp là 1 tài liệu; "\n\n" = tách theo đoạn
	min_chars: int = 0  # Lọc tài liệu quá ngắn
	max_docs: int = 0  # 0 = không giới hạn
	max_tokens: int = 0  # 0 = không giới hạn (giới hạn số token sau khi mã hoá)

	# Hugging Face Hub
	hf_name: str = ""
	hf_config: str = ""
	hf_split: str = "train"
	hf_revision: str = ""

	# Nguồn "bin" có sẵn
	bin_dtype: str = "auto"  # "auto" | "uint16" | "uint32"


@dataclass
class DataPhaseConfig:
	"""
	Lịch trình trộn dữ liệu theo tiến độ huấn luyện (curriculum / annealing).
	Áp dụng cho các bước có `progress < until` (progress = step / total_steps).
	`weights` ghi đè trọng số của từng nguồn theo tên; nguồn không nêu tên giữ trọng số gốc.
	"""

	until: float = 1.0
	weights: Dict[str, float] = field(default_factory=dict)


@dataclass
class DataConfig:
	cache_dir: str = "data/cache"  # Nơi lưu shard token đã mã hoá
	shard_size_tokens: int = 100_000_000  # Số token tối đa mỗi shard (~200MB với uint16)
	val_ratio: float = 0.005  # Tỷ lệ tài liệu dành cho tập validation
	append_eos: bool = True  # Chèn <|eos|> giữa các tài liệu khi đóng gói
	prepend_bos: bool = False
	tokenize_batch_size: int = 1024  # Số tài liệu mỗi lô khi token hoá (Rust đa luồng)
	synthetic_fallback: bool = True  # Dùng dữ liệu toán giả lập nếu chưa khai báo nguồn nào
	sources: List[DataSourceConfig] = field(default_factory=list)
	phases: List[DataPhaseConfig] = field(default_factory=list)

	@property
	def cache_path(self) -> Path:
		p = Path(self.cache_dir)
		return p if p.is_absolute() else (PROJECT_ROOT / p)

	def active_sources(self, stage: str) -> List[DataSourceConfig]:
		stage = stage.lower()
		return [
			s for s in self.sources
			if s.enabled and s.weight > 0 and (not s.stages or stage in [x.lower() for x in s.stages])
		]

	def weights_at(self, progress: float, sources: List[DataSourceConfig]) -> List[float]:
		"""Trả về trọng số trộn (chưa chuẩn hoá) của các nguồn tại tiến độ `progress` ∈ [0, 1]."""
		override: Dict[str, float] = {}
		for phase in sorted(self.phases, key=lambda p: p.until):
			if progress < phase.until:
				override = phase.weights
				break
		return [float(override.get(s.name, s.weight)) for s in sources]


@dataclass
class FullConfig:
	project: ProjectConfig = field(default_factory=ProjectConfig)
	model: ModelConfig = field(default_factory=ModelConfig)
	optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
	training: TrainingConfig = field(default_factory=TrainingConfig)
	rl_grpo: RLGRPOConfig = field(default_factory=RLGRPOConfig)
	data: DataConfig = field(default_factory=DataConfig)
	config_path: Path = field(default=DEFAULT_CONFIG_PATH, repr=False)

	def to_dict(self) -> Dict[str, Any]:
		return {
			"project": asdict(self.project),
			"model": asdict(self.model),
			"optimizer": asdict(self.optimizer),
			"training": asdict(self.training),
			"rl_grpo": asdict(self.rl_grpo),
			"data": asdict(self.data),
		}

	def scale_preset(self, preset_name: str):
		"""Chuyển đổi nhanh cấu hình sang quy mô lớn hơn (10m, 100m, 500m, 1b, 3b, 7b)."""
		preset_key = preset_name.strip().lower()
		if preset_key not in SCALE_PRESETS:
			valid = ", ".join(SCALE_PRESETS.keys())
			raise ValueError(f"Preset '{preset_name}' không hợp lệ. Hỗ trợ: {valid}")
		self.model.scale_preset = preset_key
		self.model.apply_preset_if_needed()
		self.project.name = f"slm-killer-{preset_key}"


def _filter_dataclass_kwargs(cls: Any, data: Dict[str, Any]) -> Dict[str, Any]:
	valid_fields = set(cls.__dataclass_fields__.keys())
	return {k: v for k, v in data.items() if k in valid_fields}


def _deep_merge_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
	"""
	Gộp đệ quy 2 từ điển cấu hình TOML:
	- Nếu cả hai cùng là dict (ví dụ `[model]`, `[training]`, `[data]`): gộp từng khóa con.
	- Nếu là danh sách hoặc giá trị đơn (ví dụ `[[data.sources]]`): giá trị mới ghi đè lên giá trị cũ.
	"""
	merged = copy.deepcopy(base)
	for k, v in override.items():
		if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
			merged[k] = _deep_merge_dict(merged[k], v)
		else:
			merged[k] = copy.deepcopy(v)
	return merged


def _load_toml_includes(path: Path, visited: Optional[set[Path]] = None) -> Dict[str, Any]:
	"""
	Đọc tệp TOML và tự động nạp gộp toàn bộ các tệp con được khai báo trong `includes = [...]`.
	Các khóa khai báo trực tiếp trong tệp cha sẽ được ưu tiên ghi đè lên tệp được include.
	"""
	resolved = path.resolve()
	if visited is None:
		visited = set()
	if resolved in visited:
		raise ValueError(f"Phát hiện vòng lặp nạp cấu hình (circular include) tại: {resolved}")
	visited.add(resolved)

	# utf-8-sig: chấp nhận cả tệp có BOM (Notepad / PowerShell trên Windows hay tự thêm)
	current_raw: Dict[str, Any] = tomllib.loads(resolved.read_text(encoding="utf-8-sig"))
	include_list = current_raw.pop("includes", None)

	accumulated: Dict[str, Any] = {}
	if isinstance(include_list, list):
		for inc_item in include_list:
			inc_path = Path(str(inc_item))
			if not inc_path.is_absolute():
				# Ưu tiên tìm tương đối theo thư mục chứa tệp cấu hình hiện tại, sau đó tới PROJECT_ROOT
				candidate_local = resolved.parent / inc_path
				inc_resolved = candidate_local if candidate_local.exists() else (PROJECT_ROOT / inc_path)
			else:
				inc_resolved = inc_path

			if not inc_resolved.exists():
				raise FileNotFoundError(
					f"Không tìm thấy tệp cấu hình thành phần '{inc_item}' (được khai báo trong '{resolved}')."
				)
			sub_dict = _load_toml_includes(inc_resolved, visited=visited)
			accumulated = _deep_merge_dict(accumulated, sub_dict)

	return _deep_merge_dict(accumulated, current_raw)


def load_config(
	path: Union[str, Path] = "config.toml",
	scale_override: Optional[str] = None,
) -> FullConfig:
	"""
	Tự động tìm và đọc tệp cấu hình `config.toml` tại thư mục gốc của dự án (kèm các tệp
	thành phần trong `configs/` thông qua `includes`).
	Đảm bảo dù người dùng gọi script từ bất kỳ thư mục làm việc nào vẫn nạp đúng file.
	"""
	candidate = Path(path)
	if not candidate.is_absolute():
		# Ưu tiên tìm tại thư mục gốc dự án (Path(__file__).resolve().parent.parent / "config.toml")
		root_candidate = PROJECT_ROOT / candidate
		if root_candidate.exists():
			resolved_path = root_candidate
		elif candidate.resolve().exists():
			resolved_path = candidate.resolve()
		else:
			resolved_path = DEFAULT_CONFIG_PATH
	else:
		resolved_path = candidate

	if not resolved_path.exists():
		raise FileNotFoundError(
			f"Không tìm thấy tệp cấu hình tại '{resolved_path}'. "
			f"Hãy đảm bảo tệp 'config.toml' nằm ở thư mục gốc dự án ({PROJECT_ROOT})."
		)

	raw_copy = _load_toml_includes(resolved_path)
	project_cfg = ProjectConfig(**_filter_dataclass_kwargs(ProjectConfig, raw_copy.get("project", {})))
	model_cfg = ModelConfig(**_filter_dataclass_kwargs(ModelConfig, raw_copy.get("model", {})))
	optim_cfg = OptimizerConfig(**_filter_dataclass_kwargs(OptimizerConfig, raw_copy.get("optimizer", {})))
	train_cfg = TrainingConfig(**_filter_dataclass_kwargs(TrainingConfig, raw_copy.get("training", {})))
	rl_cfg = RLGRPOConfig(**_filter_dataclass_kwargs(RLGRPOConfig, raw_copy.get("rl_grpo", {})))

	raw_data: Dict[str, Any] = raw_copy.get("data", {})
	raw_sources = raw_data.pop("sources", [])
	raw_phases = raw_data.pop("phases", [])
	data_cfg = DataConfig(**_filter_dataclass_kwargs(DataConfig, raw_data))
	data_cfg.sources = [
		DataSourceConfig(**_filter_dataclass_kwargs(DataSourceConfig, s)) for s in raw_sources
	]
	data_cfg.phases = [
		DataPhaseConfig(**_filter_dataclass_kwargs(DataPhaseConfig, p)) for p in raw_phases
	]
	names = [s.name for s in data_cfg.sources]
	if len(names) != len(set(names)):
		raise ValueError(f"Tên nguồn dữ liệu trong [[data.sources]] bị trùng: {names}")

	if scale_override:
		model_cfg.scale_preset = scale_override
	model_cfg.apply_preset_if_needed()

	return FullConfig(
		project=project_cfg,
		model=model_cfg,
		optimizer=optim_cfg,
		training=train_cfg,
		rl_grpo=rl_cfg,
		data=data_cfg,
		config_path=resolved_path,
	)
