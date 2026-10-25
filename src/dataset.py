"""
High-Throughput Binary Dataset Pipeline (Memory-Mapped Dataset for LLaDA Diffusion)
===================================================================================
- Leverages `np.memmap` with `uint16` (or `uint32` for large vocabs > 65535)
  for zero-overhead data loading and zero RAM bloat.
- Automatically generates synthetic arithmetic and reasoning tokens if no
  binary files exist yet, allowing instant testing.
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
	"""Selects uint16 for vocab <= 65535 (10M/100M) and uint32 for larger vocabs (1B-7B+)."""
	return np.dtype(np.uint16) if vocab_size <= 65535 else np.dtype(np.uint32)


def build_document_attention_mask(
	tokens: torch.Tensor,
	eos_token_id: int = 2,
	pad_token_id: int = 0,
) -> torch.Tensor:
	"""
	Constructs a (B, 1, T, T) boolean document attention mask for Bidirectional Attention:
	- Tokens only attend to other tokens within the same document (delimited by <|eos|>).
	- Completely prevents cross-document semantic contamination during continuous-stream pretraining.
	- Padding tokens (<|pad|>) cannot attend or be attended to.
	- The diagonal is strictly kept True to prevent NaNs in softmax when rows have no valid tokens.
	"""
	bsz, seq_len = tokens.shape
	device = tokens.device

	is_eos = (tokens == eos_token_id)
	is_pad = (tokens == pad_token_id)

	# Cumulative sum increments document index after each <|eos|>
	doc_ids = torch.cumsum(is_eos, dim=-1) - is_eos.long()

	# Tokens belong to the same document if doc_ids match
	same_doc = (doc_ids.unsqueeze(-1) == doc_ids.unsqueeze(-2))

	# Exclude pad tokens from participating in document cross-attention
	not_pad = ~is_pad
	valid_mask = same_doc & not_pad.unsqueeze(-1) & not_pad.unsqueeze(-2)

	# Ensure diagonal is always True so that every query has at least one valid key
	diag = torch.eye(seq_len, dtype=torch.bool, device=device).unsqueeze(0)
	valid_mask = valid_mask | diag

	return valid_mask.unsqueeze(1)  # (B, 1, T, T)



def generate_synthetic_corpus_bin(
	bin_path: Path,
	cfg: FullConfig,
	num_tokens: int = 262_144,
	seed: int = 42,
) -> Path:
	"""
	Generates a binary `.bin` token file containing multi-step arithmetic problems,
	numerical sequences, and logical reasoning samples if no real data is found.
	"""
	bin_path.parent.mkdir(parents=True, exist_ok=True)
	rng = random.Random(seed)
	tok = Tokenizer(cfg=cfg)
	dtype = _select_memmap_dtype(cfg.model.vocab_size)

	buffer: list[int] = []
	ops = ["+", "-", "*"]

	while len(buffer) < num_tokens:
		mode = rng.randint(0, 3)
		if mode == 0:
			# Multi-step arithmetic explanation
			a = rng.randint(10, 500)
			b = rng.randint(5, 200)
			c = rng.randint(2, 25)
			op1 = rng.choice(["+", "-"])
			step1 = (a + b) if op1 == "+" else (a - b)
			ans = step1 * c
			text = (
				f"<|bos|>Arithmetic Explanation: Evaluating ({a} {op1} {b}) * {c}. "
				f"First calculate inside parentheses: {a} {op1} {b} = {step1}. "
				f"Then multiply {step1} by {c} to yield {ans}. "
				f"The result is {ans}.<|eos|>\n"
			)
		elif mode == 1:
			# Number sequences
			start = rng.randint(1, 30)
			diff = rng.randint(2, 15)
			seq = [start + i * diff for i in range(5)]
			nxt = start + 5 * diff
			seq_str = ", ".join(str(x) for x in seq)
			text = (
				f"<|bos|>Sequence Induction: Arithmetic sequence with initial term {start} and difference {diff}: "
				f"{seq_str}. The next consecutive term is {nxt}.<|eos|>\n"
			)
		elif mode == 2:
			# Linear equation solving
			x_val = rng.randint(2, 40)
			m = rng.randint(2, 12)
			b = rng.randint(1, 50)
			rhs = m * x_val + b
			text = (
				f"<|bos|>Algebraic Derivation: Given {m} * x + {b} = {rhs}. "
				f"Subtracting {b} gives {m} * x = {rhs - b}. "
				f"Dividing by {m} yields x = {x_val}.<|eos|>\n"
			)
		else:
			# Factual concepts and reasoning
			concepts = [
				("LLaDA Discrete Diffusion", "denoises masked discrete tokens iteratively across bidirectional transformer layers."),
				("Bidirectional Attention", "enables each position to access both preceding and following tokens simultaneously."),
				("Muon Optimizer", "applies fifth-order Newton-Schulz orthogonalization to optimize 2D parameter weights rapidly."),
				("QK-Norm Normalization", "stabilizes attention variance and prevents entropy explosion in deep diffusion transformers."),
				("Cơ chế chú ý hai chiều", "cho phép mô hình ngôn ngữ hiểu đồng thời toàn bộ ngữ cảnh hai phía của câu."),
			]
			term, desc = rng.choice(concepts)
			text = f"<|bos|>Knowledge: {term} {desc}<|eos|>\n"

		ids = tok.encode(text)
		buffer.extend(ids)

	arr = np.array(buffer[:num_tokens], dtype=dtype)
	mmap = np.memmap(str(bin_path), dtype=dtype, mode="w+", shape=(len(arr),))
	mmap[:] = arr[:]
	mmap.flush()
	del mmap
	return bin_path


class MemmapTokenDataset(Dataset):
	"""
	High-throughput binary dataset loader backed by `np.memmap`.
	- Returns `MixtureTokenDataset` if `[[data.sources]]` are defined.
	- Otherwise loads single `.bin` file (generating synthetic tokens if absent).
	"""

	def __new__(
		cls,
		cfg: Optional[FullConfig] = None,
		split: str = "train",
		*args: Any, **kwargs: Any
	):
		cfg = cfg or load_config()
		if (
			cls is MemmapTokenDataset and
			kwargs.get("bin_path") is None and
			cfg.data.active_sources(cfg.training.stage)
		):
			return MixtureTokenDataset(cfg, split)

		return super().__new__(cls)

	def set_progress(self, step: int, total_steps: int):
		"""Unused for single file dataset (kept for MixtureTokenDataset compatibility)."""

	def __init__(
		self,
		cfg: Optional[FullConfig] = None,
		split: str = "train",
		bin_path: Optional[Union[str, Path]] = None,
	):
		super().__init__()
		self.cfg = cfg or load_config()
		self.split = split
		self.seq_len = self.cfg.model.max_seq_len
		self.vocab_size = self.cfg.model.vocab_size
		self.dtype = _select_memmap_dtype(self.vocab_size)

		if bin_path is not None:
			self.bin_path = Path(bin_path)
		else:
			ckpt_dir = self.cfg.project.checkpoint_path
			ckpt_dir.mkdir(parents=True, exist_ok=True)
			self.bin_path = ckpt_dir / f"{split}_tokens.bin"

		min_required_tokens = max(65_536, (self.seq_len + 1) * 64)
		if (
			not self.bin_path.exists() or
			self.bin_path.stat().st_size < min_required_tokens * self.dtype.itemsize
		):
			num_syn = 262_144 if split == "train" else 32_768
			seed = self.cfg.project.seed + (0 if split == "train" else 999)
			print(
				f"[Dataset] File '{self.bin_path.name}' not found. "
				f"Generating synthetic corpus ({num_syn:,} tokens, {self.dtype.name})..."
			)
			generate_synthetic_corpus_bin(
				self.bin_path, cfg=self.cfg, num_tokens=num_syn, seed=seed
			)

		file_bytes = self.bin_path.stat().st_size
		self.total_tokens = file_bytes // self.dtype.itemsize
		self.num_samples = max(1, (self.total_tokens - 1) // self.seq_len)
		self._mmap: Optional[np.memmap] = None

	def _open_memmap(self) -> np.memmap:
		if self._mmap is None:
			self._mmap = np.memmap(
				str(self.bin_path),
				dtype=self.dtype,
				mode="r",
				shape=(self.total_tokens,),
			)
		return self._mmap

	def __len__(self) -> int:
		return self.num_samples

	def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
		mmap = self._open_memmap()
		start = (idx * self.seq_len) % max(1, self.total_tokens - self.seq_len)
		chunk = np.asarray(mmap[start: start + self.seq_len], dtype=np.int64)
		if chunk.max() >= self.vocab_size:
			chunk = chunk % self.vocab_size
		x = torch.from_numpy(chunk)
		return x, x

	def sample_batch(
		self,
		batch_size: int,
		device: Union[str, torch.device] = "cpu",
		seq_len_override: Optional[int] = None,
		sft_masking: bool = False,
		assistant_token_id: int = 6,
	) -> Tuple[torch.Tensor, torch.Tensor]:
		"""
		Randomly samples a micro-batch directly from `np.memmap`.
		"""
		mmap = self._open_memmap()
		T = seq_len_override if seq_len_override is not None else self.seq_len
		max_start = max(1, self.total_tokens - T)
		starts = np.random.randint(0, max_start, size=(batch_size,))

		x_np = np.empty((batch_size, T), dtype=np.int64)
		sft_targets_np = np.empty((batch_size, T), dtype=np.int64) if sft_masking else None

		for i, s in enumerate(starts):
			s_idx = int(s)
			# Align to document start (<|bos|>, id=1) if nearby within window T
			lookahead = mmap[s_idx: min(self.total_tokens, s_idx + T)]
			bos_hits = np.where(lookahead == 1)[0]
			if len(bos_hits) > 0 and (sft_masking or bos_hits[0] > 0):
				s_idx = min(max_start - 1, s_idx + int(bos_hits[0]))

			segment = mmap[s_idx: s_idx + T].astype(np.int64)
			if len(segment) < T:
				padded = np.zeros(T, dtype=np.int64)
				padded[:len(segment)] = segment
				segment = padded
			if segment.max() >= self.vocab_size:
				segment = segment % self.vocab_size
			x_np[i] = segment

			if sft_masking:
				sft_targets_np[i] = segment.copy()
				_apply_sft_mask(x_np[i], sft_targets_np[i], assistant_token_id)

		x = torch.from_numpy(x_np)
		y = torch.from_numpy(sft_targets_np) if sft_masking else None
		if torch.device(device).type == "cuda":
			x = x.pin_memory()
			if y is not None:
				y = y.pin_memory()
		return (
			x.to(device=device, non_blocking=True),
			y.to(device=device, non_blocking=True) if y is not None else None,
		)


def _apply_sft_mask(
	x_row: np.ndarray,
	y_row: np.ndarray,
	assistant_token_id: int = 6,
	user_token_id: int = 5,
	bos_token_id: int = 1,
	eos_token_id: int = 2,
	eos_keep_tokens: int = 16,
):
	"""
	Isolates a single SFT conversation starting at `<|bos|>`:
	1. Pads any subsequent unrelated document after `<|eos|>` with `<|eos|>` so
	   bidirectional attention never mixes unrelated conversations.
	2. Masks (`-100`) all `<|user|>` prompt spans, leaving only `<|assistant|>` responses
	   and 1 block (`eos_keep_tokens`) of `<|eos|>` tokens supervised.
	"""
	T = x_row.shape[0]
	y_row[:] = x_row

	assist_positions = np.where(x_row == assistant_token_id)[0]
	if len(assist_positions) == 0:
		return

	first_assist = int(assist_positions[0])

	# Find end of the conversation (<|eos|> after the first <|assistant|>)
	eos_after = np.where(x_row[first_assist + 1:] == eos_token_id)[0]
	if len(eos_after) > 0:
		eos_idx = first_assist + 1 + int(eos_after[0])
		if eos_idx + 1 < T:
			x_row[eos_idx + 1:] = eos_token_id
			y_row[eos_idx + 1:] = eos_token_id
		if eos_idx + eos_keep_tokens < T:
			y_row[eos_idx + eos_keep_tokens:] = -100
		active_end = eos_idx + 1
	else:
		active_end = T

	# Mask out the initial prompt up to and including the first <|assistant|> token
	y_row[: first_assist + 1] = -100

	# Mask out any subsequent multi-turn <|user|> prompt spans
	user_starts = np.where(
		(x_row[first_assist + 1: active_end] == user_token_id)
		| (x_row[first_assist + 1: active_end] == bos_token_id)
	)[0]
	if len(user_starts) > 0:
		for u_rel in user_starts:
			u_pos = first_assist + 1 + int(u_rel)
			next_assists = assist_positions[assist_positions > u_pos]
			if len(next_assists) > 0:
				y_row[u_pos: int(next_assists[0]) + 1] = -100
			else:
				y_row[u_pos: active_end] = -100


class MixtureTokenDataset:
	"""
	Blends multiple tokenized sources (`[[data.sources]]`) according to sampling weights:
	  1. Select source proportionally to active weight.
	  2. Select shard within source proportionally to token count.
	  3. Randomly slice `seq_len + 1` tokens.
	"""

	def __init__(self, cfg: FullConfig, split: str = "train"):
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
				shards = meta["splits"].get("train") or []
			arrays, sizes = [], []
			for s in shards:
				if s["num_tokens"] < min_len:
					continue
				path = Path(s["file"]) if Path(s["file"]).is_absolute() else base_dir / s["file"]
				arrays.append(np.memmap(str(path), dtype=dtype, mode="r", shape=(s["num_tokens"],)))
				sizes.append(s["num_tokens"])
			if not arrays:
				print(f"[Data] Skipping source '{src.name}': insufficient tokens for split '{split}'.")
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
				f"No valid data sources found for stage '{cfg.training.stage}' / split '{split}'. "
				"Check [[data.sources]] in config.toml."
			)
		self.total_tokens = sum(p["num_tokens"] for p in self.pools)
		self.set_progress(0, max(1, cfg.training.total_steps))
		if split == "train":
			self.describe()

	def set_progress(self, step: int, total_steps: int):
		"""Updates source blending probabilities according to progress."""
		progress = step / max(1, total_steps)
		w = np.asarray(
			self.cfg.data.weights_at(progress, [p["source"] for p in self.pools]),
			dtype=np.float64
		)
		w = np.clip(w, 0.0, None)
		if w.sum() <= 0:
			w = np.ones_like(w)
		self.source_probs = w / w.sum()

	def describe(self):
		print("[Data] Training dataset blend:")
		for p, prob in zip(self.pools, self.source_probs):
			print(
				f"   - {p['source'].name:<20} type={p['source'].type:<8} "
				f"tokens={p['num_tokens']:>14,} | shards={len(p['arrays']):>4} | sample_prob={prob * 100:5.1f}%"
			)
		print(f"   = Total: {self.total_tokens:,} tokens")

	def __len__(self) -> int:
		return max(1, self.total_tokens // self.seq_len)

	def sample_batch(
		self,
		batch_size: int,
		device: Union[str, torch.device] = "cpu",
		seq_len_override: Optional[int] = None,
		sft_masking: bool = False,
		assistant_token_id: int = 6,
	) -> Tuple[torch.Tensor, torch.Tensor]:
		T = seq_len_override or self.seq_len
		x_np = np.empty((batch_size, T), dtype=np.int64)
		sft_targets_np = np.empty((batch_size, T), dtype=np.int64) if sft_masking else None

		src_ids = self.rng.choice(len(self.pools), size=batch_size, p=self.source_probs)
		for i, si in enumerate(src_ids):
			pool = self.pools[si]
			arr = pool["arrays"][self.rng.choice(len(pool["arrays"]), p=pool["shard_probs"])]
			max_start = max(1, arr.shape[0] - T)
			start = int(self.rng.integers(0, max_start))
			# Align to document start (<|bos|>, id=1) if nearby within window T
			lookahead = arr[start: min(arr.shape[0], start + T)]
			bos_hits = np.where(lookahead == 1)[0]
			if len(bos_hits) > 0 and (sft_masking or bos_hits[0] > 0):
				start = min(max_start - 1, start + int(bos_hits[0]))

			segment = np.asarray(arr[start: start + T], dtype=np.int64)
			if len(segment) < T:
				padded = np.zeros(T, dtype=np.int64)
				padded[:len(segment)] = segment
				segment = padded
			if segment.max() >= self.vocab_size:
				segment %= self.vocab_size

			x_np[i] = segment
			if sft_masking:
				sft_targets_np[i] = segment.copy()
				_apply_sft_mask(x_np[i], sft_targets_np[i], assistant_token_id)

		x = torch.from_numpy(x_np)
		y = torch.from_numpy(sft_targets_np) if sft_masking else None
		if torch.device(device).type == "cuda":
			x = x.pin_memory()
			if y is not None:
				y = y.pin_memory()

		return (
			x.to(device, non_blocking=True),
			y.to(device, non_blocking=True) if y is not None else None,
		)
