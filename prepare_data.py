"""
Chuẩn bị dữ liệu huấn luyện (`prepare_data.py`)
===============================================
Token hoá trước toàn bộ nguồn trong `[[data.sources]]` của `config.toml` thành shard `.bin`.
(`train.py` cũng tự gọi bước này nếu cache chưa có, nhưng chạy riêng giúp chuẩn bị dữ liệu lớn
trước khi thuê GPU, hoặc chạy song song trên máy CPU khác.)

Ví dụ:
    python prepare_data.py                  # chuẩn bị mọi nguồn đang bật
    python prepare_data.py --stage sft      # chỉ các nguồn dùng cho SFT
    python prepare_data.py --only wiki_vi   # một nguồn cụ thể
    python prepare_data.py --force          # token hoá lại từ đầu
    python prepare_data.py --list           # xem trạng thái cache
"""

from __future__ import annotations

import argparse

from src.config_loader import load_config
from src.data_sources import load_source_meta, prepare_source


def main():
	parser = argparse.ArgumentParser(description="Token hoá nguồn dữ liệu trong config.toml thành shard .bin")
	parser.add_argument("--config", default="config.toml")
	parser.add_argument("--stage", default=None, choices=["pretrain", "sft", "grpo"])
	parser.add_argument("--only", nargs="*", default=None, help="Tên nguồn cần xử lý")
	parser.add_argument("--force", action="store_true", help="Bỏ qua cache, token hoá lại")
	parser.add_argument("--list", action="store_true", help="Chỉ liệt kê trạng thái cache")
	args = parser.parse_args()

	cfg = load_config(args.config)
	sources = cfg.data.active_sources(args.stage) if args.stage \
		else [s for s in cfg.data.sources if s.enabled]

	if args.only:
		sources = [s for s in sources if s.name in args.only]

	if not sources:
		print("Chưa có nguồn nào trong [[data.sources]] (hoặc bị lọc hết). Xem ví dụ trong config.toml.")
		return

	for src in sources:
		if args.list:
			meta = load_source_meta(src, cfg) if src.type != "bin" else {"splits": {}}
			if meta is None:
				print(f" - {src.name:<20} [CHƯA CÓ / CŨ]  type={src.type}")
			else:
				n = {k: sum(s['num_tokens'] for s in v) for k, v in meta["splits"].items()}
				print(f" - {src.name:<20} [OK]  type={src.type}  token={n}")
			continue
		prepare_source(src, cfg, force=args.force)


if __name__ == "__main__":
	main()
