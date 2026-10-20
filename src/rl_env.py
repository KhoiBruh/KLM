"""
Môi trường Kiểm chứng Toán học / Logic (Verifiable Sandbox) & Thuật toán GRPO
=============================================================================
- Tự động sinh bài toán số học nhiều bước (ArithMark-3 style), tìm quy luật dãy số,
  và giải phương trình đại số có đáp án kiểm chứng chính xác 100%.
- Triển khai thuật toán GRPO (Group Relative Policy Optimization) không cần mạng Critic,
  tiết kiệm 50% VRAM, phù hợp chạy ngay trên RTX 3050 4GB lẫn cụm GPU đám mây.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config_loader import FullConfig, load_config
from .model import KLMModel, _unwrap_model
from tokenizer.build_tokenizer import Tokenizer


@dataclass
class VerifiableTask:
    """Một bài toán kiểm chứng được sinh tự động kèm đáp án chuẩn và lời giải mẫu."""

    task_type: str
    prompt: str
    ground_truth: int
    reference_solution: str


class VerifiableMathLogicEnv:
    """
    Môi trường kiểm soát tự động (Verifiable Arithmetic & Logic Sandbox):
    - Sinh đề bài ngẫu nhiên theo seed có đáp án nguyên kiểm chứng được 100%.
    - Chấm điểm thưởng (Reward) kết hợp:
      1. Điểm chính xác (`1.0` nếu đáp án trùng khớp `ground_truth`, `0.0` nếu sai).
      2. Điểm thưởng định dạng suy luận (`+0.2` khi tuân thủ cấu trúc `<|think|>...<|/think|><|answer|>`).
    """

    ANSWER_REGEX = re.compile(r"<\|answer\|>\s*([-+]?\d+)|Answer:\s*([-+]?\d+)|([-+]?\d+)\s*$")

    def __init__(self, seed: int = 42) -> None:
        self.rng = random.Random(seed)

    def sample_task(self, difficulty: int = 1) -> VerifiableTask:
        task_kind = self.rng.choice(["arith_multistep", "sequence_pattern", "linear_equation", "logic_word"])

        if task_kind == "arith_multistep":
            scale = max(1, difficulty)
            a = self.rng.randint(5 * scale, 50 * scale)
            b = self.rng.randint(3 * scale, 40 * scale)
            c = self.rng.randint(2, 12)
            op = self.rng.choice(["+", "-"])
            mid = (a + b) if op == "+" else (a - b)
            ans = mid * c
            prompt = f"<|bos|><|user|> Calculate: ({a} {op} {b}) * {c} <|assistant|>"
            ref = f"<|think|> Step 1: {a} {op} {b} = {mid}. Step 2: {mid} * {c} = {ans}. <|/think|><|answer|> {ans} <|eos|>"
            return VerifiableTask(task_type=task_kind, prompt=prompt, ground_truth=ans, reference_solution=ref)

        if task_kind == "sequence_pattern":
            start = self.rng.randint(1, 25)
            step = self.rng.randint(2, 12)
            terms = [start + i * step for i in range(5)]
            ans = start + 5 * step
            seq_str = ", ".join(str(x) for x in terms)
            prompt = f"<|bos|><|user|> Find the next number in the sequence: {seq_str}, ? <|assistant|>"
            ref = (
                f"<|think|> Difference is {terms[1]} - {terms[0]} = {step}. "
                f"Next number is {terms[-1]} + {step} = {ans}. <|/think|><|answer|> {ans} <|eos|>"
            )
            return VerifiableTask(task_type=task_kind, prompt=prompt, ground_truth=ans, reference_solution=ref)

        if task_kind == "linear_equation":
            x_val = self.rng.randint(2, 30)
            m = self.rng.randint(2, 10)
            b = self.rng.randint(1, 40)
            rhs = m * x_val + b
            prompt = f"<|bos|><|user|> Solve for x: {m} * x + {b} = {rhs} <|assistant|>"
            ref = (
                f"<|think|> {m} * x = {rhs} - {b} = {rhs - b}. "
                f"x = {rhs - b} / {m} = {x_val}. <|/think|><|answer|> {x_val} <|eos|>"
            )
            return VerifiableTask(task_type=task_kind, prompt=prompt, ground_truth=x_val, reference_solution=ref)

        # logic_word
        init_items = self.rng.randint(15, 80)
        added = self.rng.randint(5, 35)
        removed = self.rng.randint(3, 14)
        ans = init_items + added - removed
        prompt = (
            f"<|bos|><|user|> A warehouse has {init_items} units, receives {added} more, "
            f"and ships {removed}. How many units remain? <|assistant|>"
        )
        ref = (
            f"<|think|> {init_items} + {added} = {init_items + added}. "
            f"{init_items + added} - {removed} = {ans}. <|/think|><|answer|> {ans} <|eos|>"
        )
        return VerifiableTask(task_type=task_kind, prompt=prompt, ground_truth=ans, reference_solution=ref)

    @classmethod
    def extract_predicted_integer(cls, text: str) -> Optional[int]:
        """Trích xuất câu trả lời số nguyên từ chuỗi văn bản mô hình sinh ra."""
        if "<|answer|>" in text:
            after_ans = text.split("<|answer|>")[-1]
            nums = re.findall(r"[-+]?\d+", after_ans)
            if nums:
                try:
                    return int(nums[0])
                except ValueError:
                    pass

        all_nums = re.findall(r"[-+]?\d+", text)
        if all_nums:
            try:
                return int(all_nums[-1])
            except ValueError:
                return None
        return None

    def score_response(self, response_text: str, ground_truth: int) -> Tuple[float, bool]:
        """
        Chấm điểm câu trả lời của mô hình:
        - Chính xác tuyệt đối đáp án số học: +1.0 điểm
        - Có thẻ suy luận `<|think|>` ... `<|/think|>` và `<|answer|>`: +0.15 điểm định dạng
        - Sai số nhỏ (cùng chữ số cuối hoặc gần đúng): +0.05 điểm khuyến khích dẫn hướng ban đầu
        """
        pred = self.extract_predicted_integer(response_text)
        is_correct = pred is not None and (pred == ground_truth)

        reward = 1.0 if is_correct else 0.0

        # Điểm thưởng tuân thủ cấu trúc suy luận CoT
        has_think = ("<|think|>" in response_text) and ("<|/think|>" in response_text)
        has_answer_tag = "<|answer|>" in response_text
        if has_think and has_answer_tag:
            reward += 0.15
        elif has_think or has_answer_tag:
            reward += 0.05

        if not is_correct and pred is not None:
            if abs(pred - ground_truth) <= 2:
                reward += 0.1
            elif (pred % 10) == (ground_truth % 10):
                reward += 0.03

        return reward, is_correct


def compute_group_normalized_advantages(
    rewards: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """
    Chuẩn hóa lợi thế tương đối trong nhóm G câu trả lời (GRPO Advantage):
        A_i = (r_i - mean(r)) / (std(r) + eps)
    """
    mean_r = rewards.mean()
    std_r = rewards.std(unbiased=False)
    return (rewards - mean_r) / (std_r + eps)


class GRPOTrainer:
    """
    Bộ huấn luyện tăng cường GRPO (Group Relative Policy Optimization) không cần mạng Critic:
    1. Với mỗi đề bài, mô hình sinh ra G = `group_size` chuỗi suy luận + đáp án.
    2. Môi trường `VerifiableMathLogicEnv` chấm điểm thưởng r_i cho từng chuỗi.
    3. Chuẩn hóa lợi thế nhóm A_i = (r_i - mean(r)) / (std(r) + eps).
    4. Tối ưu hàm mục tiêu Policy Gradient kết hợp phạt phân kỳ KL so với trạng thái tham chiếu.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Tokenizer,
        cfg: Optional[FullConfig] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.cfg = cfg or load_config()
        self.model = model
        self.raw_model: KLMModel = _unwrap_model(model)
        self.tokenizer = tokenizer
        self.device = device or torch.device(
            "cuda" if (self.cfg.project.device == "cuda" and torch.cuda.is_available()) else "cpu"
        )
        self.env = VerifiableMathLogicEnv(seed=self.cfg.project.seed)

        self.group_size = self.cfg.rl_grpo.group_size
        self.kl_coef = self.cfg.rl_grpo.kl_coef
        self.max_gen_len = min(self.cfg.rl_grpo.max_gen_len, self.cfg.model.max_seq_len // 2)

    def grpo_step(
        self,
        optimizer: Any,
        scaler: Optional[torch.amp.GradScaler] = None,
        amp_dtype: torch.dtype = torch.float32,
        use_amp: bool = False,
    ) -> Dict[str, float]:
        """Thực hiện 1 bước cập nhật GRPO hoàn chỉnh trên một bài toán có kiểm chứng."""
        task = self.env.sample_task()
        prompt_ids = self.tokenizer.encode(task.prompt)
        prompt_len = len(prompt_ids)

        prompt_tensor = torch.tensor(
            [prompt_ids] * self.group_size, dtype=torch.long, device=self.device
        )

        # 1. Rollout: Sinh G chuỗi câu trả lời bằng KV-Cache
        self.raw_model.eval()
        with torch.no_grad():
            with torch.amp.autocast(device_type=self.device.type, dtype=amp_dtype, enabled=use_amp):
                sequences = self.raw_model.generate(
                    input_ids=prompt_tensor,
                    max_new_tokens=self.max_gen_len,
                    temperature=0.85,
                    top_k=32,
                    top_p=0.92,
                    eos_token_id=self.tokenizer.eos_id,
                )

                # Tính log-probs tham chiếu (Reference / Old Policy) mà không cần bản sao mô hình thứ 2
                ref_out = self.raw_model(input_ids=sequences[:, :-1])
                ref_log_probs = F.log_softmax(ref_out.logits.float(), dim=-1).detach()

        # 2. Chấm điểm từng câu trả lời trong nhóm G
        rewards_list: List[float] = []
        correct_count = 0
        for i in range(self.group_size):
            gen_ids = sequences[i, prompt_len:].tolist()
            gen_text = self.tokenizer.decode(gen_ids, skip_special_tokens=False)
            r_val, is_corr = self.env.score_response(gen_text, task.ground_truth)
            rewards_list.append(r_val)
            if is_corr:
                correct_count += 1

        rewards = torch.tensor(rewards_list, dtype=torch.float32, device=self.device)
        advantages = compute_group_normalized_advantages(rewards)  # (G,)

        # 3. Cập nhật Policy với lợi thế tương đối A_i và phạt KL
        self.model.train()
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=self.device.type, dtype=amp_dtype, enabled=use_amp):
            out = self.model(input_ids=sequences[:, :-1])
            logits = out.logits.float()
            log_probs = F.log_softmax(logits, dim=-1)

            target_ids = sequences[:, 1:]
            token_log_probs = torch.gather(log_probs, dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)
            old_token_log_probs = torch.gather(
                ref_log_probs, dim=-1, index=target_ids.unsqueeze(-1)
            ).squeeze(-1)

            # Mask chỉ tính trên phần token do mô hình sinh ra (sau prompt)
            gen_mask = torch.zeros_like(token_log_probs, dtype=torch.float32)
            gen_start = max(0, prompt_len - 1)
            gen_mask[:, gen_start:] = 1.0

            # Ước lượng KL không chệch (Schulman approximator): exp(ref - cur) - (ref - cur) - 1
            log_ratio = old_token_log_probs - token_log_probs
            kl_per_token = torch.exp(log_ratio.clamp(-10.0, 10.0)) - log_ratio - 1.0

            ratio = torch.exp((token_log_probs - old_token_log_probs).clamp(-10.0, 10.0))
            adv_expanded = advantages.unsqueeze(1)  # (G, 1)
            clipped_ratio = torch.clamp(ratio, 0.8, 1.2)
            surr1 = ratio * adv_expanded
            surr2 = clipped_ratio * adv_expanded
            policy_obj = torch.minimum(surr1, surr2)

            token_loss = -(policy_obj - self.kl_coef * kl_per_token) * gen_mask
            denom = gen_mask.sum().clamp(min=1.0)
            loss = token_loss.sum() / denom

        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.optimizer.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.optimizer.grad_clip)
            optimizer.step()

        return {
            "loss": float(loss.detach().item()),
            "mean_reward": float(rewards.mean().item()),
            "max_reward": float(rewards.max().item()),
            "accuracy": float(correct_count) / float(self.group_size),
            "kl": float(((kl_per_token * gen_mask).sum() / denom).detach().item()),
        }
