"""Per-position distillation objectives that share the on-policy training loop.

Every objective maps a block of response positions to a per-position loss. The trainer sums
the per-position losses and divides by the *unweighted* number of valid response positions
``N_k`` (Equation 7), so the reduction is identical for all methods.

Implemented from the paper:

* the consensus family of Table 4, ``-(1/N_k) sum_t w_t s_t^T log q_t`` with target
  ``s_t`` in {geometric ``r_t = c_t / Z_t``, arithmetic ``p_bar_t``, single view} and weight
  ``w_t`` in {``Z_t``, 1, shuffled ``Z_pi(t)``, batch mean ``Z_bar_k``}; VD-OPD (Equation 7)
  is the member (geometric, ``Z_t``), evaluated directly on the unnormalised ``c_t``;
* GKD with forward KL, full-vocabulary reverse KL and generalised JSD (Appendix B.3.1);
* sampled-token OPD and EOPD with top-k or full-vocabulary forward KL (Appendix B.3.1);
* EOPD-2V (Equation 13).

ToDi, TSD-KD, Veto and CROP are *not* specified precisely enough in the paper to be
re-implemented from it; they are exposed as named hooks, see :func:`register_objective`.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

import torch
import torch.nn.functional as F
from torch import Tensor

from vdopd.consensus import (
    arithmetic_mean,
    entropy,
    geometric_consensus,
    kl_divergence,
    normalized_consensus,
)

TARGETS = ("geometric", "arithmetic", "view1", "view2")
WEIGHTS = ("mass", "unit", "shuffled", "batch_mean")


@dataclass
class PositionBlock:
    """A block of valid response positions ``(i, t)``.

    Attributes:
        student_logits: ``[n, d]`` student logits at ``pi_theta(. | x_i, y_{i,<t})``; the only
            tensor through which gradients flow.
        teacher_log_probs: one ``[n, d]`` tensor of full-vocabulary log-probabilities per
            teacher view, ``log p^(m)_{i,t}`` (Equation 2). Always treated as constants.
        tokens: ``[n]`` sampled response token ids ``y_{i,t}``.
        assigned_weight: ``[n]`` externally assigned weights for the weight-assignment
            controls of Table 4(b) (``None`` otherwise).
    """

    student_logits: Tensor
    teacher_log_probs: tuple[Tensor, ...]
    tokens: Tensor
    assigned_weight: Tensor | None = None


@dataclass
class BlockLoss:
    """Per-position loss and the supervision weight that was applied at each position."""

    loss: Tensor
    weight: Tensor


class Objective(Protocol):
    """Interface of a per-position objective."""

    num_views: int
    weight_assignment: str | None
    uses_consensus_mass: bool

    def __call__(self, block: PositionBlock) -> BlockLoss: ...


def _detached_views(block: PositionBlock, num_views: int) -> list[Tensor]:
    if len(block.teacher_log_probs) < num_views:
        raise ValueError(
            f"objective needs {num_views} teacher view(s), got {len(block.teacher_log_probs)}"
        )
    return [lp.detach().float() for lp in block.teacher_log_probs[:num_views]]


@dataclass
class ConsensusObjective:
    """Target/weight family of Table 4; ``(geometric, mass)`` is VD-OPD (Equation 7).

    Per-position loss ``-w_t s_t^T log q_t``. For VD-OPD the product ``w_t s_t`` equals the
    unnormalised consensus ``c_t``, so the loss is evaluated as ``-c_t^T log q_t`` without
    ever dividing by ``Z_t``; at ``Z_t = 0`` it is exactly zero (Section 4.3).
    """

    target: str = "geometric"
    weight: str = "mass"
    num_views: int = field(default=2, init=False)

    def __post_init__(self) -> None:
        if self.target not in TARGETS:
            raise ValueError(f"unknown target {self.target!r}; choose from {TARGETS}")
        if self.weight not in WEIGHTS:
            raise ValueError(f"unknown weight {self.weight!r}; choose from {WEIGHTS}")

    @property
    def weight_assignment(self) -> str | None:
        """Name of the batch-level weight control that needs all masses in advance."""
        return self.weight if self.weight in ("shuffled", "batch_mean") else None

    @property
    def uses_consensus_mass(self) -> bool:
        """Whether a batch with zero total consensus mass must skip the optimizer step."""
        return self.weight != "unit"

    def __call__(self, block: PositionBlock) -> BlockLoss:
        log_p1, log_p2 = _detached_views(block, 2)
        log_q = F.log_softmax(block.student_logits.float(), dim=-1)

        if self.target == "geometric" and self.weight == "mass":
            c, mass = geometric_consensus(log_p1, log_p2)
            return BlockLoss(loss=-(c * log_q).sum(-1), weight=mass)

        if self.target == "geometric":
            target, mass = normalized_consensus(log_p1, log_p2)
        else:
            mass = geometric_consensus(log_p1, log_p2)[1]
            if self.target == "arithmetic":
                target = arithmetic_mean(log_p1, log_p2)
            else:
                target = (log_p1 if self.target == "view1" else log_p2).exp()

        if self.weight == "unit":
            weight = torch.ones_like(mass)
        elif self.weight == "mass":
            weight = mass
        else:
            if block.assigned_weight is None:
                raise ValueError(f"weight={self.weight!r} needs PositionBlock.assigned_weight")
            weight = block.assigned_weight.detach().float().to(mass.device)
        return BlockLoss(loss=-weight * (target * log_q).sum(-1), weight=weight)


def assign_weights(mass: Tensor, mode: str, generator: torch.Generator | None = None) -> Tensor:
    """Weight-assignment controls of Table 4(b) over all valid positions of a rollout batch.

    ``"shuffled"`` returns ``Z_pi(t)`` for a seeded random permutation ``pi`` of the valid
    response positions, which preserves the weight multiset. ``"batch_mean"`` returns
    ``Z_bar_k = (1/N_k) sum_t Z_t`` at every position, which preserves the total weight.
    """
    mass = mass.detach().float()
    if mode == "shuffled":
        perm = torch.randperm(mass.numel(), generator=generator)
        return mass[perm.to(mass.device)]
    if mode == "batch_mean":
        return torch.full_like(mass, mass.mean().item()) if mass.numel() else mass
    raise ValueError(f"unknown weight assignment {mode!r}")


@dataclass
class ForwardKL:
    """GKD-FKL: ``KL(p || q)`` toward a single teacher view (Appendix B.3.1)."""

    view: int = 0
    num_views: int = field(default=1, init=False)
    weight_assignment: str | None = field(default=None, init=False)
    uses_consensus_mass: bool = field(default=False, init=False)

    def __call__(self, block: PositionBlock) -> BlockLoss:
        log_p = _detached_views(block, self.view + 1)[self.view]
        log_q = F.log_softmax(block.student_logits.float(), dim=-1)
        loss = kl_divergence(log_p, log_q)
        return BlockLoss(loss=loss, weight=torch.ones_like(loss))


@dataclass
class ReverseKL:
    """GKD-RKL (full): full-vocabulary ``KL(q || p)`` (Appendix B.3.1)."""

    num_views: int = field(default=1, init=False)
    weight_assignment: str | None = field(default=None, init=False)
    uses_consensus_mass: bool = field(default=False, init=False)

    def __call__(self, block: PositionBlock) -> BlockLoss:
        (log_p,) = _detached_views(block, 1)
        log_q = F.log_softmax(block.student_logits.float(), dim=-1)
        loss = kl_divergence(log_q, log_p)
        return BlockLoss(loss=loss, weight=torch.ones_like(loss))


@dataclass
class GeneralizedJSD:
    """GKD-JSD: ``beta KL(p || m) + (1 - beta) KL(q || m)`` with ``m = beta p + (1 - beta) q``.

    This is the generalised Jensen-Shannon divergence of GKD (Agarwal et al., 2024) with the
    interpolation parameter ``beta = 0.9`` named in Appendix B.3.1.
    """

    beta: float = 0.9
    num_views: int = field(default=1, init=False)
    weight_assignment: str | None = field(default=None, init=False)
    uses_consensus_mass: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if not 0.0 < self.beta < 1.0:
            raise ValueError("beta must lie strictly between 0 and 1")

    def __call__(self, block: PositionBlock) -> BlockLoss:
        (log_p,) = _detached_views(block, 1)
        log_q = F.log_softmax(block.student_logits.float(), dim=-1)
        log_m = torch.logaddexp(log_p + math.log(self.beta), log_q + math.log1p(-self.beta))
        loss = self.beta * kl_divergence(log_p, log_m)
        loss = loss + (1.0 - self.beta) * kl_divergence(log_q, log_m)
        return BlockLoss(loss=loss, weight=torch.ones_like(loss))


def sampled_token_opd_loss(log_p: Tensor, log_q: Tensor, tokens: Tensor) -> Tensor:
    """Score-function surrogate of the per-token reverse KL using only the sampled token.

    With ``y ~ q``, ``E_y[(log q(y) - log p(y)) grad log q(y)] = grad KL(q || p)``; the
    surrogate ``sg[log q(y) - log p(y)] * log q(y)`` therefore has the reverse-KL gradient in
    expectation while reading one teacher log-probability per position.
    """
    index = tokens.long().unsqueeze(-1)
    log_p_y = log_p.gather(-1, index).squeeze(-1)
    log_q_y = log_q.gather(-1, index).squeeze(-1)
    return (log_q_y - log_p_y).detach() * log_q_y


@dataclass
class SampledTokenOPD:
    """Sampled-token OPD (Appendix B.3.1), a different objective from full-vocabulary RKL."""

    view: int = 0
    num_views: int = field(default=1, init=False)
    weight_assignment: str | None = field(default=None, init=False)
    uses_consensus_mass: bool = field(default=False, init=False)

    def __call__(self, block: PositionBlock) -> BlockLoss:
        log_p = _detached_views(block, self.view + 1)[self.view]
        log_q = F.log_softmax(block.student_logits.float(), dim=-1)
        loss = sampled_token_opd_loss(log_p, log_q, block.tokens)
        return BlockLoss(loss=loss, weight=torch.ones_like(loss))


def truncated_forward_kl(
    log_p: Tensor, log_q: Tensor, top_k: int | None, renormalize: bool = False
) -> Tensor:
    """Forward KL restricted to the teacher's ``top_k`` tokens (``None`` = full vocabulary).

    Without renormalisation the truncated sum ``sum_{a in topk(p)} p(a) [log p(a) - log q(a)]``
    is returned; with ``renormalize=True`` both distributions are first renormalised on the
    teacher's top-k support.
    """
    if top_k is None or top_k >= log_p.shape[-1]:
        return kl_divergence(log_p, log_q)
    log_p_k, index = log_p.topk(top_k, dim=-1)
    log_q_k = log_q.gather(-1, index)
    if renormalize:
        log_p_k = log_p_k - torch.logsumexp(log_p_k, dim=-1, keepdim=True)
        log_q_k = log_q_k - torch.logsumexp(log_q_k, dim=-1, keepdim=True)
    return kl_divergence(log_p_k, log_q_k)


@dataclass
class EOPD:
    """Entropy-aware OPD: sampled-token OPD plus teacher-entropy-gated forward KL.

    ``l_t = l_t^OPD + lambda * 1[H(p_t) > tau] * FKL(p_t, q_t)`` with entropy threshold
    ``tau = 0.8`` nats and FKL weight ``lambda = 1.0`` (Appendix B.3.1 and B.5.3). ``top_k=16``
    gives "EOPD (top-16)", ``top_k=None`` gives "EOPD (full)".
    """

    entropy_threshold: float = 0.8
    fkl_weight: float = 1.0
    top_k: int | None = None
    renormalize_top_k: bool = False
    view: int = 0
    num_views: int = field(default=1, init=False)
    weight_assignment: str | None = field(default=None, init=False)
    uses_consensus_mass: bool = field(default=False, init=False)

    def __call__(self, block: PositionBlock) -> BlockLoss:
        log_p = _detached_views(block, self.view + 1)[self.view]
        log_q = F.log_softmax(block.student_logits.float(), dim=-1)
        gate = (entropy(log_p) > self.entropy_threshold).float()
        fkl = truncated_forward_kl(log_p, log_q, self.top_k, self.renormalize_top_k)
        loss = sampled_token_opd_loss(log_p, log_q, block.tokens) + self.fkl_weight * gate * fkl
        return BlockLoss(loss=loss, weight=gate)


@dataclass
class TwoViewAverage:
    """Average of a single-view objective over the two teacher views (Equation 13).

    ``L_2V = 0.5 [L(p^(1), q) + L(p^(2), q)]``. With ``base = EOPD(top_k=None)`` this is
    EOPD-2V: the entropy gate is evaluated separately for each view and both terms use the
    same student predictions on the original prompt.
    """

    base_factory: Callable[[int], Objective]
    num_views: int = field(default=2, init=False)
    weight_assignment: str | None = field(default=None, init=False)
    uses_consensus_mass: bool = field(default=False, init=False)

    def __call__(self, block: PositionBlock) -> BlockLoss:
        outputs = [self.base_factory(view)(block) for view in range(2)]
        return BlockLoss(
            loss=0.5 * (outputs[0].loss + outputs[1].loss),
            weight=0.5 * (outputs[0].weight + outputs[1].weight),
        )


@dataclass(frozen=True)
class ObjectiveConfig:
    """Configuration of the distillation objective (``objective:`` section of a YAML file)."""

    name: str = "vdopd"
    target: str | None = None
    weight: str | None = None
    eopd_entropy_threshold: float = 0.8
    eopd_fkl_weight: float = 1.0
    eopd_top_k: int | None = None
    eopd_renormalize_top_k: bool = False
    jsd_beta: float = 0.9


_CONSENSUS_PRESETS: dict[str, tuple[str, str]] = {
    "vdopd": ("geometric", "mass"),
    "arithmetic": ("arithmetic", "unit"),
    "weighted_arithmetic": ("arithmetic", "mass"),
    "geometric": ("geometric", "unit"),
    "shuffled_weight": ("geometric", "shuffled"),
    "batch_mean_weight": ("geometric", "batch_mean"),
}

# Baselines of Table 1 whose objectives the paper cites but does not state. They can be
# plugged in with `register_objective` once implemented from their original publications.
UNSPECIFIED_BASELINES: dict[str, str] = {
    "todi": "ToDi-OP (Jung et al., 2025): detached vocabulary-wise divergence weights",
    "tsd_kd": "TSD-KD (Kim & Baek, 2026): token ranking, selective distillation, entropy reg.",
    "veto": "Veto (Jang et al., 2026): geometric teacher-student bridge with decayed parameter",
    "crop": "CROP (Li et al., 2026): counterfactual token selection on validated triplets",
}

_REGISTRY: dict[str, Callable[[ObjectiveConfig], Objective]] = {}


def register_objective(name: str, factory: Callable[[ObjectiveConfig], Objective]) -> None:
    """Register an additional objective under ``name`` (hook for external baselines).

    ``factory`` receives the :class:`ObjectiveConfig` and must return an object implementing
    the :class:`Objective` protocol. The trainer needs nothing else: rollouts, teacher views,
    blockwise evaluation and the ``1/N_k`` reduction are shared.
    """
    _REGISTRY[name] = factory


def available_objectives() -> list[str]:
    """Names accepted by :func:`build_objective` (excluding unimplemented hooks)."""
    return sorted(set(_REGISTRY) | set(_CONSENSUS_PRESETS) | {"consensus"})


def build_objective(cfg: ObjectiveConfig) -> Objective:
    """Instantiate the objective named by ``cfg.name``."""
    if cfg.name in _REGISTRY:
        return _REGISTRY[cfg.name](cfg)
    if cfg.name in _CONSENSUS_PRESETS:
        target, weight = _CONSENSUS_PRESETS[cfg.name]
        return ConsensusObjective(target=cfg.target or target, weight=cfg.weight or weight)
    if cfg.name == "consensus":
        return ConsensusObjective(target=cfg.target or "geometric", weight=cfg.weight or "mass")
    if cfg.name in UNSPECIFIED_BASELINES:
        raise NotImplementedError(
            f"objective {cfg.name!r} is a hook only: {UNSPECIFIED_BASELINES[cfg.name]}. The "
            "VD-OPD paper does not state this objective; implement it from its original "
            "publication and register it with vdopd.objectives.register_objective()."
        )
    raise ValueError(f"unknown objective {cfg.name!r}; available: {available_objectives()}")


def _eopd(cfg: ObjectiveConfig, view: int = 0, top_k: int | None = None) -> EOPD:
    return EOPD(
        entropy_threshold=cfg.eopd_entropy_threshold,
        fkl_weight=cfg.eopd_fkl_weight,
        top_k=top_k,
        renormalize_top_k=cfg.eopd_renormalize_top_k,
        view=view,
    )


register_objective("gkd_fkl", lambda cfg: ForwardKL())
register_objective("gkd_rkl", lambda cfg: ReverseKL())
register_objective("gkd_jsd", lambda cfg: GeneralizedJSD(beta=cfg.jsd_beta))
register_objective("sampled_opd", lambda cfg: SampledTokenOPD())
register_objective("eopd", lambda cfg: _eopd(cfg, top_k=cfg.eopd_top_k))
register_objective("eopd_full", lambda cfg: _eopd(cfg, top_k=None))
register_objective("eopd_top16", lambda cfg: _eopd(cfg, top_k=cfg.eopd_top_k or 16))
register_objective(
    "eopd_2v", lambda cfg: TwoViewAverage(base_factory=lambda view: _eopd(cfg, view=view))
)
