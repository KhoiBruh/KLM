"""
Training Data Preparation Script (`prepare_data.py`)
====================================================
Pre-tokenizes all enabled sources under `[[data.sources]]` in `config.toml` into `.bin` shards.
(`train.py` automatically triggers this step if the cache is missing, but running this separately
is useful when preparing large corpora before renting GPUs or running on dedicated CPU machines.)

Examples:
    python prepare_data.py                  # Prepare all enabled sources
    python prepare_data.py --stage sft      # Only sources tagged for SFT
    python prepare_data.py --only wiki_vi   # A specific source by name
    python prepare_data.py --force          # Re-tokenize from scratch
    python prepare_data.py --list           # Check cache status only
"""

from __future__ import annotations

import argparse

from src.config_loader import load_config
from src.data_sources import load_source_meta, prepare_source


def main():
	parser = argparse.ArgumentParser(description="Tokenize data sources defined in config.toml into .bin shards")
	parser.add_argument("--config", default="config.toml")
	parser.add_argument("--stage", default=None, choices=["pretrain", "sft", "grpo"])
	parser.add_argument("--only", nargs="*", default=None, help="Specific source names to process")
	parser.add_argument("--force", action="store_true", help="Bypass cache and force re-tokenization")
	parser.add_argument("--list", action="store_true", help="List cache status only")
	args = parser.parse_args()

	cfg = load_config(args.config)
	sources = cfg.data.active_sources(args.stage) if args.stage \
		else [s for s in cfg.data.sources if s.enabled]

	if args.only:
		sources = [s for s in sources if s.name in args.only]

	if not sources:
		print("No active sources found in [[data.sources]]. Check examples in config.toml.")
		return

	for src in sources:
		if args.list:
			meta = load_source_meta(src, cfg) if src.type != "bin" else {"splits": {}}
			if meta is None:
				print(f" - {src.name:<20} [MISSING / STALE]  type={src.type}")
			else:
				n = {k: sum(s['num_tokens'] for s in v) for k, v in meta["splits"].items()}
				print(f" - {src.name:<20} [OK]               type={src.type}  tokens={n}")
			continue
		prepare_source(src, cfg, force=args.force)


if __name__ == "__main__":
	main()
