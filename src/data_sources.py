"""
Data Ingestion & Binary Token Sharding Pipeline
===============================================
Streams raw corpora from declared `[[data.sources]]` (avoiding high memory consumption),
tokenizes batches using high-throughput Rust tokenizers, and writes shards into `np.memmap`:

    data/cache/<source_name>/
        meta.json               # configuration fingerprint & shard statistics
        train_00000.bin         # uint16 (vocab <= 65535) or uint32 (large vocab)
        train_00001.bin
        val_00000.bin

Cache automatically invalidates whenever source configuration or tokenizer changes.
"""

from __future__ import annotations

import glob
import gzip
import hashlib
import json
import time
import zlib
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import numpy as np

from .config_loader import PROJECT_ROOT, DataConfig, DataSourceConfig, FullConfig

META_VERSION = 1


def select_token_dtype(vocab_size: int) -> np.dtype:
	"""uint16 for vocab <= 65535, uint32 for large frontier vocabularies."""
	return np.dtype(np.uint16) if vocab_size <= 65535 else np.dtype(np.uint32)


# --------------------------------------------------------------------------- #
# Raw document reading
# --------------------------------------------------------------------------- #

def _expand_paths(path_spec: Any) -> List[Path]:
	specs = path_spec if isinstance(path_spec, list) else [path_spec]
	files: List[Path] = []
	for spec in specs:
		if not spec:
			continue
		p = Path(spec)
		pattern = str(p if p.is_absolute() else PROJECT_ROOT / p)
		matched = sorted(glob.glob(pattern, recursive=True))
		files.extend(Path(m) for m in matched if Path(m).is_file())
	return files


class _SafeDict(dict):
	def __missing__(self, key: str) -> str:
		return ""


def _render_messages(messages: List[Dict[str, Any]]) -> str:
	"""Formats OpenAI-style `messages=[{role, content}]` into chat special tokens."""
	parts: List[str] = []
	for m in messages:
		role = str(m.get("role", "user")).lower()
		content = str(m.get("content", ""))
		tag = "<|assistant|>" if role == "assistant" else "<|user|>"
		parts.append(f"{tag} {content}")
	return " ".join(parts)


def _record_to_text(record: Dict[str, Any], src: DataSourceConfig) -> str:
	if src.template:
		return src.template.format_map(_SafeDict(record))
	value = record.get(src.text_field)
	if value is None and isinstance(record.get("messages"), list):
		return _render_messages(record["messages"])
	if isinstance(value, list):
		return _render_messages(value) if value and isinstance(value[0], dict) \
			else "\n".join(map(str, value))

	return "" if value is None else str(value)


def _open_text(path: Path):
	if path.suffix == ".gz":
		return gzip.open(path, "rt", encoding="utf-8", errors="replace")
	return open(path, "r", encoding="utf-8", errors="replace")


def _iter_text(src: DataSourceConfig) -> Iterator[str]:
	for f in _expand_paths(src.path):
		with _open_text(f) as fh:
			content = fh.read()
		if src.doc_separator:
			yield from content.split(src.doc_separator)
		else:
			yield content


def _iter_jsonl(src: DataSourceConfig) -> Iterator[str]:
	for f in _expand_paths(src.path):
		with _open_text(f) as fh:
			for line in fh:
				line = line.strip()
				if not line:
					continue
				try:
					yield _record_to_text(json.loads(line), src)
				except json.JSONDecodeError:
					continue


def _iter_parquet(src: DataSourceConfig) -> Iterator[str]:
	try:
		import pyarrow.parquet as pq
	except ImportError as exc:
		raise ImportError(f"Source '{src.name}' with parquet type requires: pip install pyarrow") from exc
	for f in _expand_paths(src.path):
		pf = pq.ParquetFile(str(f))
		for batch in pf.iter_batches(batch_size=4096):
			for record in batch.to_pylist():
				yield _record_to_text(record, src)


def _iter_hf(src: DataSourceConfig) -> Iterator[str]:
	try:
		from datasets import load_dataset
	except ImportError as exc:
		raise ImportError(f"Source '{src.name}' with hf type requires: pip install datasets") from exc

	ds = load_dataset(
		src.hf_name,
		name=src.hf_config or None,
		split=src.hf_split,
		revision=src.hf_revision or None,
		streaming=True,
	)
	for record in ds:
		yield _record_to_text(record, src)


READERS = {
	"text": _iter_text,
	"jsonl": _iter_jsonl,
	"parquet": _iter_parquet,
	"hf": _iter_hf,
}


def iter_documents(src: DataSourceConfig) -> Iterator[str]:
	reader = READERS.get(src.type.lower())
	if reader is None:
		raise ValueError(
			f"Unsupported source type '{src.type}'. Choose from: {', '.join([*READERS, 'bin'])}"
		)

	count = 0
	for doc in reader(src):
		if len(doc) < max(1, src.min_chars):
			continue
		yield doc
		count += 1
		if src.max_docs and count >= src.max_docs:
			break


# --------------------------------------------------------------------------- #
# Shard Writing
# --------------------------------------------------------------------------- #

class _ShardWriter:
	def __init__(self, out_dir: Path, split: str, dtype: np.dtype, shard_size: int):
		self.out_dir, self.split, self.dtype, self.shard_size = out_dir, split, dtype, shard_size
		self.buffer: List[np.ndarray] = []
		self.buffered = 0
		self.shards: List[Dict[str, Any]] = []

	def add(self, ids: List[int]):
		if not ids:
			return

		self.buffer.append(np.asarray(ids, dtype=self.dtype))
		self.buffered += len(ids)
		if self.buffered >= self.shard_size:
			self.flush()

	def flush(self):
		if not self.buffered:
			return

		arr = np.concatenate(self.buffer)
		name = f"{self.split}_{len(self.shards):05d}.bin"
		arr.tofile(self.out_dir / name)
		self.shards.append({"file": name, "num_tokens": int(arr.size)})
		self.buffer, self.buffered = [], 0

	@property
	def total(self) -> int:
		return sum(s["num_tokens"] for s in self.shards) + self.buffered


def _file_hash(path: Path) -> str:
	return hashlib.sha1(path.read_bytes()).hexdigest()[:16] if path.exists() else "none"


def source_fingerprint(src: DataSourceConfig, data_cfg: DataConfig, cfg: FullConfig) -> str:
	payload = {
		"v": META_VERSION,
		"source": {k: v for k, v in asdict(src).items() if k not in ("weight", "enabled", "stages")},
		"val_ratio": data_cfg.val_ratio,
		"append_eos": data_cfg.append_eos,
		"prepend_bos": data_cfg.prepend_bos,
		"vocab_size": cfg.model.vocab_size,
		"tokenizer": _file_hash(cfg.project.tokenizer_path / "tokenizer.json"),
	}
	return hashlib.sha1(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:20]


def _is_val_doc(doc_idx: int, val_ratio: float) -> bool:
	if val_ratio <= 0:
		return False
	return (zlib.crc32(str(doc_idx).encode()) % 1_000_000) < int(val_ratio * 1_000_000)


def load_source_meta(src: DataSourceConfig, cfg: FullConfig) -> Optional[Dict[str, Any]]:
	meta_path = cfg.data.cache_path / src.name / "meta.json"
	if not meta_path.exists():
		return None
	meta = json.loads(meta_path.read_text(encoding="utf-8"))
	if meta.get("fingerprint") != source_fingerprint(src, cfg.data, cfg):
		return None
	return meta


def _register_bin_source(src: DataSourceConfig, cfg: FullConfig) -> Dict[str, Any]:
	"""Registers existing pre-tokenized binary files directly."""
	dtype = select_token_dtype(cfg.model.vocab_size) if src.bin_dtype == "auto" \
		else np.dtype(src.bin_dtype)

	files = _expand_paths(src.path)
	if not files:
		raise FileNotFoundError(
			f"Binary source '{src.name}' found no matching files at: {src.path}"
		)

	meta: Dict[str, Any] = {
		"dtype": dtype.name,
		"splits": {"train": [], "val": []},
		"external": True
	}

	for f in files:
		split = "val" if "val" in f.name.lower() else "train"
		meta["splits"][split].append(
			{
				"file": str(f),
				"num_tokens": f.stat().st_size // dtype.itemsize
			}
		)

	return meta


def prepare_source(src: DataSourceConfig, cfg: FullConfig, force: bool = False) -> Dict[str, Any]:
	"""Tokenizes data source into binary shards (skips if cache is valid). Returns metadata."""
	if src.type.lower() == "bin":
		return _register_bin_source(src, cfg)

	if not force:
		cached = load_source_meta(src, cfg)
		if cached is not None:
			return cached

	from tokenizer.build_tokenizer import Tokenizer

	out_dir = cfg.data.cache_path / src.name
	out_dir.mkdir(parents=True, exist_ok=True)
	for old in out_dir.glob("*.bin"):
		old.unlink()

	tok = Tokenizer(cfg=cfg, auto_build=True)
	dtype = select_token_dtype(cfg.model.vocab_size)
	writers = {
		split: _ShardWriter(out_dir, split, dtype, cfg.data.shard_size_tokens) for split in ("train", "val")
	}

	t0 = time.perf_counter()
	batch: List[str] = []
	doc_idx = 0
	num_docs = 0
	budget_hit = False

	def _flush_batch(docs: List[str], start_idx: int):
		nonlocal budget_hit
		for offset, ids in enumerate(tok.encode_batch(docs)):
			if cfg.data.prepend_bos:
				ids = [tok.bos_id, *ids]
			if cfg.data.append_eos:
				ids = [*ids, tok.eos_id]
			split = "val" if _is_val_doc(start_idx + offset, cfg.data.val_ratio) else "train"
			writers[split].add(ids)
			if src.max_tokens and writers["train"].total >= src.max_tokens:
				budget_hit = True
				return

	for doc in iter_documents(src):
		batch.append(doc)
		if len(batch) >= cfg.data.tokenize_batch_size:
			_flush_batch(batch, doc_idx)
			doc_idx += len(batch)
			num_docs += len(batch)
			batch = []
			print(
				f"\r[Data:{src.name}] {num_docs:,} documents | {writers['train'].total:,} train tokens",
				end="",
				flush=True,
			)
			if budget_hit:
				break
	if batch and not budget_hit:
		_flush_batch(batch, doc_idx)
		num_docs += len(batch)

	for w in writers.values():
		w.flush()

	meta = {
		"version": META_VERSION,
		"fingerprint": source_fingerprint(src, cfg.data, cfg),
		"source": src.name,
		"dtype": dtype.name,
		"num_docs": num_docs,
		"splits": {split: w.shards for split, w in writers.items()},
		"created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
	}
	(out_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
	n_train = sum(s["num_tokens"] for s in meta["splits"]["train"])
	n_val = sum(s["num_tokens"] for s in meta["splits"]["val"])
	print(
		f"\r[Data:{src.name}] Finished: {num_docs:,} documents -> {n_train:,} train tokens / {n_val:,} val tokens "
		f"({time.perf_counter() - t0:.1f}s)"
	)
	return meta


def prepare_all(
	cfg: FullConfig,
	stage: Optional[str] = None,
	force: bool = False
) -> Dict[str, Dict[str, Any]]:
	sources = cfg.data.active_sources(stage) if stage \
		else [s for s in cfg.data.sources if s.enabled]

	return {
		src.name: prepare_source(src, cfg, force) for src in sources
	}
