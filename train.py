"""
Kịch bản Huấn luyện Hợp nhất Open-SLM Killer (`pretrain` WSD + TST -> `sft` -> `grpo` RL)
=========================================================================================
- Tự động nạp cấu hình từ `config.toml` tại thư mục gốc dự án (không cần truyền tham số phức tạp).
- Lưu và khôi phục trọng số chuẩn `.safetensors` (hỗ trợ cả đơn tệp và phân mảnh cho quy mô tỷ tham số).
- Tương thích từ GPU VRAM thấp (RTX 3050 4GB / CPU) tới cụm GPU lớn (A100 / H100 / RTX Pro 6000)
  với DDP / FSDP ZeRO-3, AMP (`bfloat16`/`float16`), Gradient Checkpointing và Muon+AdamW.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import time
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn

from src.config_loader import FullConfig, load_config
from src.dataset import MemmapTokenDataset, get_tst_pool_factor
from src.model import (
    KLMModel,
    count_parameters,
    load_safetensors_checkpoint,
    save_safetensors_checkpoint,
)
from src.optim import build_optimizer
from src.rl_env import GRPOTrainer
from tokenizer.build_tokenizer import Tokenizer


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(cfg: FullConfig) -> Tuple[torch.device, torch.dtype, bool]:
    """Tự động phát hiện GPU/CPU và kiểu dữ liệu AMP tối ưu."""
    req_device = cfg.project.device.lower()
    if req_device.startswith("cuda") and torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    dtype_str = cfg.project.dtype.lower()
    if device.type == "cuda":
        if dtype_str in ("bfloat16", "bf16", "fp8") and torch.cuda.is_bf16_supported():
            return device, torch.bfloat16, True
        if dtype_str in ("float16", "fp16", "bfloat16", "bf16"):
            return device, torch.float16, True
    return device, torch.float32, False


def wrap_distributed_or_compile(
    model: KLMModel,
    cfg: FullConfig,
    device: torch.device,
) -> nn.Module:
    """
    Hỗ trợ mở rộng lên quy mô nhiều tỷ tham số với FSDP (ZeRO-3) hoặc DDP khi chạy đa GPU (`torchrun`),
    và bật `torch.compile` an toàn nếu hệ thống hỗ trợ.
    """
    wrapped: nn.Module = model
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    strategy = cfg.training.distributed_strategy.lower()

    if world_size > 1 and dist.is_available():
        if not dist.is_initialized():
            backend = "nccl" if device.type == "cuda" else "gloo"
            dist.init_process_group(backend=backend)
        if strategy == "fsdp" or (strategy == "auto" and count_parameters(model)["inference_params"] >= 500_000_000):
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

            wrapped = FSDP(wrapped, device_id=device if device.type == "cuda" else None)
        else:
            from torch.nn.parallel import DistributedDataParallel as DDP

            wrapped = DDP(wrapped, [device.index] if device.type == "cuda" else None)

    if cfg.project.compile and hasattr(torch, "compile") and device.type == "cuda":
        try:
            wrapped = torch.compile(wrapped, mode="max-autotune")  # type: ignore[assignment]
            print("[Compile] Đã kích hoạt torch.compile(mode='max-autotune').")
        except Exception as exc:
            print(f"[Compile] Bỏ qua torch.compile (fallback về eager mode): {exc}")

    return wrapped


@torch.no_grad()
def evaluate_validation_loss(
    model: nn.Module,
    val_dataset: MemmapTokenDataset,
    cfg: FullConfig,
    device: torch.device,
    amp_dtype: torch.dtype,
    use_amp: bool,
    eval_batches: int = 8,
) -> Dict[str, float]:
    model.eval()
    losses = []
    batch_size = max(1, min(cfg.training.micro_batch_size, 16))
    for _ in range(eval_batches):
        x, y = val_dataset.sample_batch(batch_size=batch_size, device=device)
        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            out = model(input_ids=x, targets=y)
            if out.ntp_loss is not None:
                losses.append(float(out.ntp_loss.item()))
            elif out.loss is not None:
                losses.append(float(out.loss.item()))
    model.train()
    mean_loss = float(np.mean(losses)) if losses else 0.0
    ppl = float(math.exp(min(20.0, mean_loss)))
    return {"val_loss": mean_loss, "val_ppl": ppl}


def train(
    cfg: Optional[FullConfig] = None,
    stage_override: Optional[str] = None,
    max_steps_override: Optional[int] = None,
    scale_override: Optional[str] = None,
    batch_size_override: Optional[int] = None,
    grad_accum_override: Optional[int] = None,
) -> None:
    if cfg is None:
        cfg = load_config(scale_override=scale_override)
    elif scale_override:
        cfg.apply_scale_preset(scale_override)

    if stage_override:
        cfg.training.stage = stage_override.lower()
    if max_steps_override is not None:
        if cfg.training.stage == "grpo":
            cfg.rl_grpo.steps = max_steps_override
        else:
            cfg.training.total_steps = max_steps_override
    if batch_size_override is not None:
        cfg.training.micro_batch_size = batch_size_override
    if grad_accum_override is not None:
        cfg.training.grad_accum_steps = grad_accum_override

    set_global_seed(cfg.project.seed)
    device, amp_dtype, use_amp = resolve_device(cfg)

    tokenizer = Tokenizer(cfg=cfg, auto_build=True)
    base_model = KLMModel(cfg).to(device=device)
    param_stats = count_parameters(base_model)

    print("=" * 78)
    print(f" KHỞI ĐỘNG HUẤN LUYỆN: {cfg.project.name.upper()} | STAGE: {cfg.training.stage.upper()}")
    print("=" * 78)
    print(f" - Thiết bị chạy         : {device} (AMP={use_amp}, dtype={amp_dtype})")
    print(
        f" - Quy mô kiến trúc      : d_model={cfg.model.d_model}, "
        f"n_layers={cfg.model.n_layers}, "
        f"n_heads={cfg.model.n_heads}/{cfg.model.n_kv_heads}, "
        f"d_ff={cfg.model.d_ff}, vocab={cfg.model.vocab_size}"
    )
    print(f" - Tham số Huấn luyện    : {param_stats['training_params']:,} ({param_stats['training_params']/1e6:.3f}M)")
    print(f" - Tham số Thi đấu (Inf) : {param_stats['inference_params']:,} ({param_stats['inference_params']/1e6:.3f}M)")
    print(f" - Định dạng Checkpoint  : Safetensors ({cfg.project.checkpoint_path})")
    print("=" * 78)

    stage = cfg.training.stage.lower()
    lr_override = cfg.rl_grpo.learning_rate if stage == "grpo" else None
    optimizer = build_optimizer(base_model, cfg=cfg, lr_override=lr_override)
    print(
        f"[Optimizer] HybridMuonAdamW -> Muon (2D): {optimizer.muon_param_count:,} params | "
        f"AdamW (Emb/Norm/1D): {optimizer.adamw_param_count:,} params"
    )

    # Tự động khôi phục từ checkpoint .safetensors mới nhất nếu có
    ckpt_dir = cfg.project.checkpoint_path
    start_step = 0
    if (ckpt_dir / "model.safetensors").exists() or (ckpt_dir / "model.safetensors.index.json").exists():
        try:
            resume_info = load_safetensors_checkpoint(
                base_model, ckpt_dir, device, optimizer
            )
            start_step = int(resume_info.get("step", 0))
            print(f"[Checkpoint] Đã khôi phục trọng số từ '{resume_info['loaded_from']}' (step={start_step}).")
        except Exception as exc:
            print(f"[Checkpoint] Khởi tạo trọng số mới (bỏ qua checkpoint cũ do khác cấu hình: {exc}).")

    model = wrap_distributed_or_compile(base_model, cfg, device)
    scaler = torch.amp.GradScaler(
        device.type,
        enabled=(use_amp and amp_dtype == torch.float16 and device.type == "cuda"),
    )

    # Nhánh 1: Huấn luyện tăng cường GRPO với môi trường kiểm chứng toán học
    if stage == "grpo":
        grpo_trainer = GRPOTrainer(model, tokenizer, cfg, device)
        total_rl_steps = cfg.rl_grpo.steps
        t0 = time.perf_counter()
        for step in range(start_step + 1, start_step + total_rl_steps + 1):
            metrics = grpo_trainer.grpo_step(optimizer, scaler, amp_dtype, use_amp)
            if step == 1 or step % max(1, min(10, cfg.training.eval_interval)) == 0 or step == start_step + total_rl_steps:
                elapsed = time.perf_counter() - t0
                print(
                    f"[GRPO Step {step:05d}/{start_step + total_rl_steps:05d}] "
                    f"loss={metrics['loss']:.4f} | reward={metrics['mean_reward']:.3f} "
                    f"(max={metrics['max_reward']:.2f}) | acc={metrics['accuracy']*100:.1f}% | "
                    f"kl={metrics['kl']:.4f} | time={elapsed:.1f}s"
                )
            if step % cfg.training.save_interval == 0 or step == start_step + total_rl_steps:
                saved_path = save_safetensors_checkpoint(
                    model, ckpt_dir, step, optimizer, cfg=cfg
                )
                print(f"[Safetensors] Đã lưu checkpoint GRPO tại: {saved_path}")
        return

    # Nhánh 2: Pre-training (kèm WSD + TST + MTP) hoặc Supervised Fine-Tuning (SFT)
    train_dataset = MemmapTokenDataset(cfg, "train")
    val_dataset = MemmapTokenDataset(cfg, "val")

    total_steps = cfg.training.total_steps
    grad_accum = max(1, cfg.training.grad_accum_steps)
    micro_bsz = cfg.training.micro_batch_size
    is_sft = stage == "sft"

    model.train()
    t_start = time.perf_counter()

    try:
        for step in range(start_step + 1, total_steps + 1):
            lr_mult = optimizer.update_lr(step - 1, total_steps)
            tst_factor = get_tst_pool_factor(step - 1, cfg) if not is_sft else 1
            train_dataset.set_progress(step - 1, total_steps)

            optimizer.zero_grad(True)
            accum_loss = 0.0
            accum_ntp = 0.0
            accum_mtp = 0.0

            for _ in range(grad_accum):
                x, y = train_dataset.sample_batch(
                    batch_size=micro_bsz,
                    device=device,
                    sft_masking=is_sft,
                    assistant_token_id=tokenizer.assistant_id,
                )
                with torch.amp.autocast(device.type, amp_dtype, use_amp):
                    out = model(input_ids=x, targets=y, tst_pool_factor=tst_factor)
                    assert out.loss is not None
                    loss = out.loss / grad_accum

                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                accum_loss += float(loss.detach().item())
                if out.ntp_loss is not None:
                    accum_ntp += float(out.ntp_loss.item()) / grad_accum
                if out.mtp_loss is not None:
                    accum_mtp += float(out.mtp_loss.item()) / grad_accum

            if scaler.is_enabled():
                scaler.unscale_(optimizer)
                grad_norm = float(
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.optimizer.grad_clip).item()
                )
                scaler.step(optimizer)
                scaler.update()
            else:
                grad_norm = float(
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.optimizer.grad_clip).item()
                )
                optimizer.step()

            if step > 0 or step == total_steps:
                dt = time.perf_counter() - t_start
                tokens_per_step = micro_bsz * grad_accum * (cfg.model.max_seq_len // max(1, tst_factor))
                tok_per_sec = (tokens_per_step * step) / max(1e-6, dt)
                print(
                    f"[{stage.upper()} | Step {step:06d}/{total_steps:06d}] "
                    f"loss={accum_loss:.4f} (ntp={accum_ntp:.4f}, mtp={accum_mtp:.4f}) | "
                    f"lr_mult={lr_mult:.3f} | tst={tst_factor}x | grad_norm={grad_norm:.3f} | "
                    f"speed={tok_per_sec:,.0f} tok/s"
                )

            if step % cfg.training.eval_interval == 0 or step == total_steps:
                val_metrics = evaluate_validation_loss(
                    model, val_dataset, cfg, device, 
                    amp_dtype, use_amp, 4
                )
                print(
                    f"  >>> [Eval @ Step {step:06d}] val_loss={val_metrics['val_loss']:.4f} | "
                    f"val_ppl={val_metrics['val_ppl']:.2f}"
                )

            if step % cfg.training.save_interval == 0 or step == total_steps:
                saved_path = save_safetensors_checkpoint(
                    model, ckpt_dir, step, optimizer,
                    {"stage": stage, "lr_mult": lr_mult}, cfg
                )
                print(f"  >>> [Safetensors] Đã lưu checkpoint tại: {saved_path}")

    except KeyboardInterrupt:
        print(f"\n[Dừng khẩn cấp] Phát hiện Ctrl+C! Đang lưu checkpoint tại step {step}...")
        saved_path = save_safetensors_checkpoint(
            model, ckpt_dir, step, optimizer,
            {"stage": stage, "lr_mult": lr_mult}, cfg
        )
        print(f"[Safetensors] Đã lưu checkpoint an toàn tại: {saved_path}")
        print("Lần sau khi chạy lại 'python train.py', tiến trình sẽ tự động tiếp tục từ bước này!")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Huấn luyện KLM (Tự động đọc config.toml ở thư mục gốc)."
    )
    parser.add_argument("--config", type=str, default="config.toml", help="Đường dẫn tới config.toml")
    parser.add_argument(
        "--stage",
        type=str,
        default=None,
        choices=["pretrain", "sft", "grpo"],
        help="Ghi đè giai đoạn huấn luyện (pretrain, sft, grpo)",
    )
    parser.add_argument("--steps", type=int, default=None, help="Ghi đè tổng số bước chạy")
    parser.add_argument(
        "--scale",
        type=str,
        default=None,
        choices=["custom", "10m", "100m", "500m", "1b", "3b", "7b"],
        help="Chuyển nhanh sang cấu hình quy mô lớn (10m, 100m, 500m, 1b, 3b, 7b)",
    )
    parser.add_argument("--batch-size", type=int, default=None, help="Ghi đè micro_batch_size")
    parser.add_argument("--grad-accum", type=int, default=None, help="Ghi đè grad_accum_steps")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    loaded_cfg = load_config(args.config, args.scale)
    train(
        loaded_cfg,
        args.stage, args.steps, args.scale,
        args.batch_size, args.grad_accum
    )
