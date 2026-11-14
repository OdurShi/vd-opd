"""Optimizer and learning-rate schedule of Appendix B.4.

AdamW with learning rate ``3e-6``, ``(beta1, beta2) = (0.9, 0.95)``, epsilon ``1e-8``, zero
weight decay, gradient-norm clipping at 1.0, 3% warmup and cosine decay.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class OptimConfig:
    """Optimisation hyper-parameters (defaults are the paper's values)."""

    learning_rate: float = 3e-6
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    warmup_fraction: float = 0.03
    min_lr_ratio: float = 0.0


def build_optimizer(
    parameters: Iterable[torch.nn.Parameter], cfg: OptimConfig
) -> torch.optim.AdamW:
    """AdamW over all trainable parameters (full-parameter training, Appendix B.4)."""
    return torch.optim.AdamW(
        [p for p in parameters if p.requires_grad],
        lr=cfg.learning_rate,
        betas=(cfg.beta1, cfg.beta2),
        eps=cfg.eps,
        weight_decay=cfg.weight_decay,
    )


def warmup_steps(total_iterations: int, warmup_fraction: float) -> int:
    """Number of warmup iterations: ``ceil(fraction * J)``, at most ``J``."""
    if total_iterations <= 0:
        return 0
    return min(total_iterations, math.ceil(warmup_fraction * total_iterations))


def learning_rate_at(iteration: int, total_iterations: int, cfg: OptimConfig) -> float:
    """Learning rate of iteration ``k`` (0-based) out of ``J``: linear warmup, cosine decay.

    The schedule is a function of the iteration index, not of the number of optimizer steps
    actually taken, so a skipped batch (Algorithm 1, line 12) leaves both the optimizer
    state and the remaining schedule unchanged.
    """
    warmup = warmup_steps(total_iterations, cfg.warmup_fraction)
    if iteration < warmup:
        return cfg.learning_rate * (iteration + 1) / warmup
    decay_steps = max(1, total_iterations - warmup)
    progress = min(1.0, (iteration - warmup) / decay_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.learning_rate * (cfg.min_lr_ratio + (1.0 - cfg.min_lr_ratio) * cosine)


def set_learning_rate(optimizer: torch.optim.Optimizer, value: float) -> None:
    """Set the learning rate of every parameter group."""
    for group in optimizer.param_groups:
        group["lr"] = value
