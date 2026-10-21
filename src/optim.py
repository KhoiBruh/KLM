"""
HybridMuonAdamW Optimizer & WSD (Warmup - Stable - Decay) Scheduler
===================================================================
- Employs the Muon algorithm (MomentUm Orthogonalized by Newton-Schulz) for 2D hidden
  weight matrices (Attention projections, SwiGLU FFN).
- Employs AdamW for Embeddings, LM Head, and RMSNorm scale weights.
- Incorporates matrix-size RMS scaling for smooth scaling from <10M up to 7B+ parameters.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import Optimizer

from .config_loader import FullConfig, OptimizerConfig, TrainingConfig


def zeropower_via_newtonschulz5(
	G: torch.Tensor,
	steps: int = 5,
	eps: float = 1e-7
) -> torch.Tensor:
	"""
	Orthogonalizes 2D gradient matrix `G` using 5th-order Newton-Schulz iterations on `bfloat16`.
	Approximates U @ V^T from SVD(G) without computing expensive singular value decompositions.
	"""
	if G.ndim < 2:
		raise ValueError(f"Newton-Schulz requires tensor with >= 2 dimensions, got {G.ndim} dims.")

	orig_shape = G.shape
	if G.ndim > 2:
		G = G.reshape(G.size(0), -1)

	a, b, c = (3.4445, -4.7750, 2.0315)
	work_dtype = torch.bfloat16 if (G.is_cuda or G.dtype == torch.bfloat16) else torch.float32
	X = G.to(work_dtype)

	transposed = X.size(0) > X.size(1)
	if transposed:
		X = X.T

	norm = X.norm() + eps
	X = X / norm

	for _ in range(steps):
		A = X @ X.T
		B = b * A + c * (A @ A)
		X = a * X + B @ X

	if transposed:
		X = X.T

	return X.reshape(orig_shape).to(G.dtype)


class Muon(Optimizer):
	"""
	Pure PyTorch implementation of the Muon optimizer with Nesterov momentum,
	5th-order Newton-Schulz orthogonalization, and decoupled weight decay.
	"""

	def __init__(
		self,
		params: Iterable[nn.Parameter],
		lr: float = 0.02,
		momentum: float = 0.95,
		nesterov: bool = True,
		ns_steps: int = 5,
		weight_decay: float = 0.01,
	):
		defaults = dict(
			lr=lr,
			momentum=momentum,
			nesterov=nesterov,
			ns_steps=ns_steps,
			weight_decay=weight_decay,
		)
		super().__init__(params, defaults)

	@torch.no_grad()
	def step(self, closure: Optional[Any] = None) -> Optional[torch.Tensor]:
		loss = None
		if closure is not None:
			with torch.enable_grad():
				loss = closure()

		for group in self.param_groups:
			lr: float = group["lr"]
			momentum: float = group["momentum"]
			nesterov: bool = group["nesterov"]
			ns_steps: int = group["ns_steps"]
			wd: float = group["weight_decay"]

			for p in group["params"]:
				if p.grad is None:
					continue
				grad = p.grad
				if grad.is_sparse:
					raise RuntimeError("Muon does not support sparse gradients.")

				state = self.state[p]
				if len(state) == 0:
					state["momentum_buffer"] = torch.zeros_like(p)

				buf: torch.Tensor = state["momentum_buffer"]
				buf.mul_(momentum).add_(grad)
				g_in = grad.add(buf, alpha=momentum) if nesterov else buf

				g_2d = g_in.reshape(g_in.size(0), -1) if g_in.ndim > 2 else g_in
				ortho = zeropower_via_newtonschulz5(g_2d, ns_steps)
				ortho = ortho.reshape_as(p)

				# Matrix aspect ratio scaling for training stability across scales
				rows, cols = g_2d.size(0), g_2d.size(1)
				scale = math.sqrt(max(1.0, rows / max(1, cols)))

				if wd > 0.0:
					p.mul_(1.0 - lr * wd)

				p.add_(ortho, alpha=-lr * scale)

		return loss


def get_wsd_lr_multiplier(
	step: int,
	total_steps: int,
	warmup_ratio: float = 0.03,
	stable_ratio: float = 0.82,
	min_lr_ratio: float = 0.05
) -> float:
	"""
	WSD (Warmup - Stable - Decay) Learning Rate Scheduler:
	- Phase 1 (Warmup): Linear ramp from 0 to 1.0.
	- Phase 2 (Stable): Plateau at maximum learning rate 1.0.
	- Phase 3 (Decay): Smooth cosine decay down to `min_lr_ratio`.
	"""
	if total_steps <= 0:
		return 1.0

	step = max(0, min(step, total_steps))
	warmup_steps = max(1, int(total_steps * warmup_ratio))
	stable_steps = int(total_steps * stable_ratio)
	decay_start = warmup_steps + stable_steps

	if step < warmup_steps:
		return float(step + 1) / float(warmup_steps)

	if step < decay_start:
		return 1.0

	decay_steps = max(1, total_steps - decay_start)
	progress = min(1.0, float(step - decay_start) / float(decay_steps))
	cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
	return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay


class HybridMuonAdamW:
	"""
	Hybrid optimizer orchestrating:
	1. `Muon`: 2D hidden weight matrices (Attention Q/K/V/O, SwiGLU MLP).
	2. `AdamW`: Embeddings, LM Head, and 1D RMSNorm scale parameters.
	"""

	def __init__(
		self,
		model: nn.Module,
		optim_cfg: OptimizerConfig,
		train_cfg: Optional[TrainingConfig] = None,
		lr_override: Optional[float] = None,
	):
		self.optim_cfg = optim_cfg
		self.train_cfg = train_cfg
		self.base_muon_lr = lr_override if lr_override is not None else optim_cfg.muon_lr
		self.base_adamw_lr = lr_override if lr_override is not None else optim_cfg.adamw_lr

		muon_params, adamw_decay_params, adamw_nodecay_params = self._split_parameters(model)

		self.muon_param_count = sum(p.numel() for p in muon_params)
		self.adamw_param_count = sum(
			p.numel() for p in (adamw_decay_params + adamw_nodecay_params)
		)

		self.muon: Optional[Muon] = (
			Muon(
				muon_params,
				self.base_muon_lr,
				optim_cfg.muon_momentum,
				ns_steps=optim_cfg.muon_ns_steps,
				weight_decay=optim_cfg.weight_decay,
			)
			if len(muon_params) > 0
			else None
		)

		adamw_groups: List[Dict[str, Any]] = []
		if adamw_decay_params:
			adamw_groups.append(
				{
					"params": adamw_decay_params,
					"weight_decay": optim_cfg.weight_decay,
					"lr": self.base_adamw_lr,
				}
			)
		if adamw_nodecay_params:
			adamw_groups.append(
				{
					"params": adamw_nodecay_params,
					"weight_decay": 0.0,
					"lr": self.base_adamw_lr,
				}
			)

		betas = (float(optim_cfg.betas[0]), float(optim_cfg.betas[1]))
		self.adamw: Optional[torch.optim.AdamW] = (
			torch.optim.AdamW(adamw_groups, self.base_adamw_lr, betas)
			if len(adamw_groups) > 0
			else None
		)

	@staticmethod
	def _split_parameters(
		model: nn.Module,
	) -> Tuple[List[nn.Parameter], List[nn.Parameter], List[nn.Parameter]]:
		"""
		Classifies model parameters into optimal optimization groups:
		- 2D hidden matrices -> Muon
		- Embedding / LM Head -> AdamW (with weight decay)
		- Norm scales & biases (1D) -> AdamW (without weight decay)
		"""
		muon_params: List[nn.Parameter] = []
		adamw_decay: List[nn.Parameter] = []
		adamw_nodecay: List[nn.Parameter] = []
		seen_ptrs: set[int] = set()

		for name, param in model.named_parameters():
			if not param.requires_grad:
				continue
			ptr = param.data_ptr()
			if ptr in seen_ptrs:
				continue
			seen_ptrs.add(ptr)

			is_embedding_or_head = ("tok_embeddings" in name) or ("lm_head" in name)
			if not is_embedding_or_head and param.ndim == 2:
				muon_params.append(param)
			elif is_embedding_or_head:
				adamw_decay.append(param)
			else:
				adamw_nodecay.append(param)

		return muon_params, adamw_decay, adamw_nodecay

	@property
	def param_groups(self) -> List[Dict[str, Any]]:
		groups: List[Dict[str, Any]] = []
		if self.muon is not None:
			groups.extend(self.muon.param_groups)
		if self.adamw is not None:
			groups.extend(self.adamw.param_groups)
		return groups

	def update_lr(self, step: int, total_steps: Optional[int] = None) -> float:
		"""Updates learning rates according to WSD schedule and returns current multiplier."""
		if self.train_cfg is not None:
			max_steps = total_steps if total_steps is not None else self.train_cfg.total_steps
			mult = get_wsd_lr_multiplier(
				step,
				max_steps,
				warmup_ratio=self.train_cfg.warmup_ratio,
				stable_ratio=self.train_cfg.stable_ratio,
				min_lr_ratio=0.05,
			)
		else:
			mult = 1.0

		if self.muon is not None:
			for g in self.muon.param_groups:
				g["lr"] = self.base_muon_lr * mult
		if self.adamw is not None:
			for g in self.adamw.param_groups:
				g["lr"] = self.base_adamw_lr * mult
		return mult

	def zero_grad(self, set_to_none: bool = True):
		if self.muon is not None:
			self.muon.zero_grad(set_to_none)
		if self.adamw is not None:
			self.adamw.zero_grad(set_to_none)

	def step(self):
		if self.muon is not None:
			self.muon.step()
		if self.adamw is not None:
			self.adamw.step()

	def state_dict(self) -> Dict[str, Any]:
		return {
			"muon": self.muon.state_dict() if self.muon is not None else None,
			"adamw": self.adamw.state_dict() if self.adamw is not None else None,
		}

	def load_state_dict(self, state: Dict[str, Any]):
		if self.muon is not None and state.get("muon") is not None:
			self.muon.load_state_dict(state["muon"])
		if self.adamw is not None and state.get("adamw") is not None:
			self.adamw.load_state_dict(state["adamw"])


def build_optimizer(
	model: nn.Module,
	cfg: FullConfig,
	lr_override: Optional[float] = None,
) -> HybridMuonAdamW:
	"""Utility helper initializing HybridMuonAdamW from configuration."""
	return HybridMuonAdamW(model, cfg.optimizer, cfg.training, lr_override)
