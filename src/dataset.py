"""
Bộ nạp dữ liệu tốc độ cao (Memory-Mapped Binary Dataset & Token Superposition Training)
======================================================================================
- Sử dụng `np.memmap` định dạng `uint16` (hoặc `uint32` khi vocab > 65535 cho mô hình nhiều tỷ tham số)
  giúp nạp dữ liệu với tốc độ tối đa và tốn ~0MB RAM phụ trội.
- Tự động sinh dữ liệu giả lập (Synthetic Arithmetic, Logic & Reasoning Corpus) nếu chưa có
  tệp `.bin` thực tế để người dùng chạy thử `python train.py` ngay lập tức.
- Hỗ trợ cơ chế Token Superposition Training (TST) ở giai đoạn đầu Pre-train và chế độ SFT có mask.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from .config_loader import FullConfig, load_config
from .data_sources import prepare_source
from tokenizer.build_tokenizer import Tokenizer


def _select_memmap_dtype(vocab_size: int) -> np.dtype:
	"""Chọn uint16 cho vocab <= 65535 (10M/100M) và uint32 cho vocab lớn (1B-7B+)."""
	return np.dtype(np.uint16) if vocab_size <= 65535 else np.dtype(np.uint32)


def generate_synthetic_corpus_bin(
	bin_path: Path,
	cfg: FullConfig,
	num_tokens: int = 262_144,
	seed: int = 42,
) -> Path:
	"""
	Tạo tệp nhị phân `.bin` chứa các bài toán số học nhiều bước, quy luật dãy số và suy luận logic
	được mã hóa bằng bộ Tokenizer (tách từng chữ số 0-9) nếu người dùng chưa có dữ liệu thực tế.
	"""
	bin_path.parent.mkdir(parents=True, exist_ok=True)
	rng = random.Random(seed)
	tok = Tokenizer(cfg=cfg, auto_build=True)
	dtype = _select_memmap_dtype(cfg.model.vocab_size)

	buffer: list[int] = []
	ops = ["+", "-", "*"]

	while len(buffer) < num_tokens:
		mode = rng.randint(0, 3)
		if mode == 0:
			# Toán số học 2 bước kiểu ArithMark-3
			a = rng.randint(10, 500)
			b = rng.randint(5, 200)
			c = rng.randint(2, 25)
			op1 = rng.choice(["+", "-"])
			step1 = (a + b) if op1 == "+" else (a - b)
			ans = step1 * c
			text = (
				f"<|bos|><|user|> Calculate: ({a} {op1} {b}) * {c} <|assistant|>"
				f"<|think|> Step 1: {a} {op1} {b} = {step1}. Step 2: {step1} * {c} = {ans}. <|/think|>"
				f"<|answer|> {ans} <|eos|>\n"
			)
		elif mode == 1:
			# Tìm quy luật dãy số
			start = rng.randint(1, 30)
			diff = rng.randint(2, 15)
			seq = [start + i * diff for i in range(5)]
			nxt = start + 5 * diff
			seq_str = ", ".join(str(x) for x in seq)
			text = (
				f"<|bos|><|user|> Find the next number in the sequence: {seq_str}, ? <|assistant|>"
				f"<|think|> Common difference is {seq[1]} - {seq[0]} = {diff}. "
				f"Next term: {seq[-1]} + {diff} = {nxt}. <|/think|>"
				f"<|answer|> {nxt} <|eos|>\n"
			)
		elif mode == 2:
			# Giải phương trình bậc nhất
			x_val = rng.randint(2, 40)
			m = rng.randint(2, 12)
			b = rng.randint(1, 50)
			rhs = m * x_val + b
			text = (
				f"<|bos|><|user|> Solve for x: {m} * x + {b} = {rhs} <|assistant|>"
				f"<|think|> {m} * x = {rhs} - {b} = {rhs - b}. "
				f"x = {rhs - b} / {m} = {x_val}. <|/think|>"
				f"<|answer|> {x_val} <|eos|>\n"
			)
		else:
			# Suy luận toán lời văn ngắn
			apples = rng.randint(12, 90)
			bought = rng.randint(5, 45)
			shared = rng.randint(2, 10)
			total = apples + bought - shared
			text = (
				f"<|bos|><|user|> Lan has {apples} books, buys {bought} more, and gives {shared} away. "
				f"How many books are left? <|assistant|>"
				f"<|think|> {apples} + {bought} = {apples + bought}. "
				f"{apples + bought} - {shared} = {total}. <|/think|>"
				f"<|answer|> {total} <|eos|>\n"
			)

		ids = tok.encode(text)
		buffer.extend(ids)

	arr = np.array(buffer[:num_tokens], dtype=dtype)
	mmap = np.memmap(str(bin_path), dtype=dtype, mode="w+", shape=(len(arr),))
	mmap[:] = arr[:]
	mmap.flush()
	del mmap
	return bin_path


def get_tst_pool_factor(step: int, cfg: FullConfig) -> int:
	"""
	Xác định hệ số gộp Token Superposition Training (TST) theo bước huấn luyện hiện tại:
	- Trong `tst_ratio` (30%) số bước đầu của giai đoạn `pretrain` khi `use_tst=True`,
	  trả về `tst_pool_factor` (ví dụ 4).
	- Sau ngưỡng đó hoặc ở giai đoạn `sft`/`grpo`, trả về 1 (chiều dài chuẩn).
	"""
	if (
		cfg.training.stage.lower() == "pretrain"
		and cfg.training.use_tst
		and cfg.training.tst_pool_factor > 1
	):
		tst_end_step = int(cfg.training.total_steps * cfg.training.tst_ratio)
		if step < tst_end_step:
			return int(cfg.training.tst_pool_factor)
	return 1


class MemmapTokenDataset(Dataset):
	"""
	Bộ nạp dữ liệu nhị phân sử dụng `np.memmap` tốc độ cao, tốn 0MB RAM thừa.
	- Nếu `config.toml` khai báo `[[data.sources]]` cho stage hiện tại → tự động trả về
	  `MixtureTokenDataset` (trộn nhiều nguồn theo trọng số).
	- Ngược lại → dùng tệp `.bin` đơn lẻ (tự sinh dữ liệu giả lập nếu chưa có).
	"""

	def __new__(cls, cfg: Optional[FullConfig] = None, split: str = "train", *args: Any, **kwargs: Any):
		cfg = cfg or load_config()
		if cls is MemmapTokenDataset and kwargs.get("bin_path") is None and cfg.data.active_sources(cfg.training.stage):
			return MixtureTokenDataset(cfg, split)
		return super().__new__(cls)

	def set_progress(self, step: int, total_steps: int) -> None:
		"""Không dùng cho tệp đơn lẻ (giữ tương thích với MixtureTokenDataset)."""

	def __init__(
		self,
		cfg: Optional[FullConfig] = None,
		split: str = "train",
		bin_path: Optional[Union[str, Path]] = None,
	) -> None:
		super().__init__()
		self.cfg = cfg or load_config()
		self.split = split
		self.seq_len = self.cfg.model.max_seq_len
		self.vocab_size = self.cfg.model.vocab_size
		self.dtype = _select_memmap_dtype(self.vocab_size)

		if bin_path is not None:
			self.bin_path = Path(bin_path)
		else:
			# Lưu tệp .bin mặc định trong thư mục `checkpoints/` để giữ đúng 3 thư mục gốc
			ckpt_dir = self.cfg.project.checkpoint_path
			ckpt_dir.mkdir(parents=True, exist_ok=True)
			self.bin_path = ckpt_dir / f"{split}_tokens.bin"

		min_required_tokens = max(65_536, (self.seq_len + 1) * 64)
		if not self.bin_path.exists() or self.bin_path.stat().st_size < min_required_tokens * self.dtype.itemsize:
			num_syn = 262_144 if split == "train" else 32_768
			seed = self.cfg.project.seed + (0 if split == "train" else 999)
			print(
				f"[Dataset] Chưa tìm thấy dữ liệu '{self.bin_path.name}'. "
				f"Đang khởi tạo dữ liệu giả lập ({num_syn:,} tokens, {self.dtype.name})..."
			)
			generate_synthetic_corpus_bin(
				self.bin_path, cfg=self.cfg, num_tokens=num_syn, seed=seed
			)

		file_bytes = self.bin_path.stat().st_size
		self.total_tokens = file_bytes // self.dtype.itemsize
		self.num_samples = max(1, (self.total_tokens - 1) // self.seq_len)

	def _open_memmap(self) -> np.memmap:
		return np.memmap(
			str(self.bin_path),
			dtype=self.dtype,
			mode="r",
			shape=(self.total_tokens,),
		)

	def __len__(self) -> int:
		return self.num_samples

	def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
		mmap = self._open_memmap()
		start = (idx * self.seq_len) % max(1, self.total_tokens - self.seq_len - 1)
		chunk = np.asarray(mmap[start: start + self.seq_len + 1], dtype=np.int64)
		# Đảm bảo mọi token ID nằm trong phạm vi vocab_size hiện tại
		if chunk.max() >= self.vocab_size:
			chunk = chunk % self.vocab_size
		x = torch.from_numpy(chunk[:-1])
		y = torch.from_numpy(chunk[1:])
		return x, y

	def sample_batch(
		self,
		batch_size: int,
		device: Union[str, torch.device] = "cpu",
		seq_len_override: Optional[int] = None,
		sft_masking: bool = False,
		assistant_token_id: int = 5,
	) -> Tuple[torch.Tensor, torch.Tensor]:
		"""
		Lấy ngẫu nhiên một micro-batch trực tiếp từ `np.memmap` với tốc độ cực cao.
		Khi `sft_masking=True` (giai đoạn SFT), tự động đặt `ignore_index = -100` cho phần câu hỏi
		trước thẻ `<|assistant|>` để mô hình chỉ tập trung tối ưu hóa câu trả lời và suy luận.
		"""
		mmap = self._open_memmap()
		T = seq_len_override if seq_len_override is not None else self.seq_len
		max_start = max(1, self.total_tokens - T - 1)
		starts = np.random.randint(0, max_start, size=(batch_size,))

		x_np = np.empty((batch_size, T), dtype=np.int64)
		y_np = np.empty((batch_size, T), dtype=np.int64)

		for i, s in enumerate(starts):
			segment = mmap[s: s + T + 1].astype(np.int64)
			if segment.max() >= self.vocab_size:
				segment = segment % self.vocab_size
			x_np[i] = segment[:-1]
			y_np[i] = segment[1:]

			if sft_masking:
				# Tìm vị trí của token <|assistant|> và mask các vị trí câu hỏi nếu tìm thấy
				assist_positions = np.where(x_np[i] == assistant_token_id)[0]
				if len(assist_positions) > 0:
					first_assist = int(assist_positions[0])
					y_np[i, :first_assist] = -100

		x = torch.from_numpy(x_np).to(device=device, non_blocking=True)
		y = torch.from_numpy(y_np).to(device=device, non_blocking=True)
		return x, y


def _apply_sft_mask(x_row: np.ndarray, y_row: np.ndarray, assistant_token_id: int) -> None:
	assist_positions = np.where(x_row == assistant_token_id)[0]
	if len(assist_positions) > 0:
		y_row[: int(assist_positions[0])] = -100


class MixtureTokenDataset:
	"""
	Trộn nhiều nguồn dữ liệu đã token hoá (`[[data.sources]]`) theo trọng số, kiểu
	Megatron/Llama data-blend:
	  1. Chọn nguồn theo xác suất ∝ trọng số (có thể đổi theo `[[data.phases]]`).
	  2. Chọn shard trong nguồn ∝ số token của shard.
	  3. Cắt ngẫu nhiên một đoạn `seq_len + 1` token (tài liệu được đóng gói, ngăn bởi <|eos|>).
	Mỗi rank (DDP/FSDP) dùng seed riêng → không trùng mẫu giữa các GPU.
	"""

	def __init__(self, cfg: FullConfig, split: str = "train") -> None:
		self.cfg = cfg
		self.split = split
		self.seq_len = cfg.model.max_seq_len
		self.vocab_size = cfg.model.vocab_size
		rank = int(os.environ.get("RANK", "0"))
		self.rng = np.random.default_rng(cfg.project.seed + rank * 1009 + (0 if split == "train" else 7))

		self.sources = cfg.data.active_sources(cfg.training.stage)
		self.pools: List[Dict[str, Any]] = []
		min_len = self.seq_len + 1

		for src in self.sources:
			meta = prepare_source(src, cfg)
			dtype = np.dtype(meta["dtype"])
			base_dir = cfg.data.cache_path / src.name
			shards = meta["splits"].get(split) or []
			if not shards or sum(s["num_tokens"] for s in shards) < min_len:
				# Nguồn quá nhỏ để tách val → dùng chung shard train cho val
				shards = meta["splits"].get("train") or []
			arrays, sizes = [], []
			for s in shards:
				if s["num_tokens"] < min_len:
					continue
				path = Path(s["file"]) if Path(s["file"]).is_absolute() else base_dir / s["file"]
				arrays.append(np.memmap(str(path), dtype=dtype, mode="r", shape=(s["num_tokens"],)))
				sizes.append(s["num_tokens"])
			if not arrays:
				print(f"[Data] Bỏ qua nguồn '{src.name}': không đủ {min_len} token cho split '{split}'.")
				continue
			sizes_np = np.asarray(sizes, dtype=np.float64)
			self.pools.append(
				{
					"source": src,
					"arrays": arrays,
					"shard_probs": sizes_np / sizes_np.sum(),
					"num_tokens": int(sizes_np.sum()),
				}
			)

		if not self.pools:
			raise RuntimeError(
				f"Không có nguồn dữ liệu hợp lệ cho stage '{cfg.training.stage}' / split '{split}'. "
				"Kiểm tra [[data.sources]] trong config.toml."
			)
		self.total_tokens = sum(p["num_tokens"] for p in self.pools)
		self.set_progress(0, max(1, cfg.training.total_steps))
		if split == "train":
			self.describe()

	def set_progress(self, step: int, total_steps: int) -> None:
		"""Cập nhật trọng số trộn theo tiến độ (dùng cho `[[data.phases]]`)."""
		progress = step / max(1, total_steps)
		w = np.asarray(
			self.cfg.data.weights_at(progress, [p["source"] for p in self.pools]), dtype=np.float64
		)
		w = np.clip(w, 0.0, None)
		if w.sum() <= 0:
			w = np.ones_like(w)
		self.source_probs = w / w.sum()

	def describe(self) -> None:
		print("[Data] Hỗn hợp dữ liệu huấn luyện:")
		for p, prob in zip(self.pools, self.source_probs):
			print(
				f"   - {p['source'].name:<20} type={p['source'].type:<8} "
				f"token={p['num_tokens']:>14,} | shard={len(p['arrays']):>4} | tỷ lệ lấy mẫu={prob * 100:5.1f}%"
			)
		print(f"   = Tổng: {self.total_tokens:,} token")

	def __len__(self) -> int:
		return max(1, self.total_tokens // self.seq_len)

	def sample_batch(
		self,
		batch_size: int,
		device: Union[str, torch.device] = "cpu",
		seq_len_override: Optional[int] = None,
		sft_masking: bool = False,
		assistant_token_id: int = 5,
	) -> Tuple[torch.Tensor, torch.Tensor]:
		T = seq_len_override or self.seq_len
		x_np = np.empty((batch_size, T), dtype=np.int64)
		y_np = np.empty((batch_size, T), dtype=np.int64)

		src_ids = self.rng.choice(len(self.pools), size=batch_size, p=self.source_probs)
		for i, si in enumerate(src_ids):
			pool = self.pools[si]
			arr = pool["arrays"][self.rng.choice(len(pool["arrays"]), p=pool["shard_probs"])]
			start = int(self.rng.integers(0, max(1, arr.shape[0] - T - 1)))
			segment = np.asarray(arr[start: start + T + 1], dtype=np.int64)
			if segment.max() >= self.vocab_size:
				segment %= self.vocab_size
			x_np[i] = segment[:-1]
			y_np[i] = segment[1:]
			if sft_masking:
				_apply_sft_mask(x_np[i], y_np[i], assistant_token_id)

		x = torch.from_numpy(x_np)
		y = torch.from_numpy(y_np)
		if torch.device(device).type == "cuda":
			x, y = x.pin_memory(), y.pin_memory()
		return x.to(device, non_blocking=True), y.to(device, non_blocking=True)
