"""Diagnostics analysed in the paper.

* Section 3 / Figure 2: first-view entropy versus cross-view JS divergence.
* Section 5.5 / Equation 9 / Figure 6: consensus mass versus consensus-fit utility.
* Appendix B.7 / Table 12: shared pre-update statistics and matched branch updates.
* Appendix B.8 / Equation 15: length-normalised candidate-answer logits.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import torch
from torch import Tensor

from vdopd.consensus import entropy, js_divergence, normalized_consensus
from vdopd.models import CausalLM
from vdopd.objectives import Objective
from vdopd.scoring import iter_log_prob_blocks, score_responses
from vdopd.trainer import RolloutBatch, Trainer


def entropy_quartile_summary(
    first_view_entropy: np.ndarray, js: np.ndarray
) -> list[dict[str, float]]:
    """Cross-view JS divergence within first-view entropy quartiles (Figure 2(b)).

    Pairs are sorted by entropy and partitioned into four equally sized groups. For each
    group the entropy range and the 5th/25th/50th/75th/95th percentiles of the JS divergence
    are returned (the box and whisker statistics of the figure).
    """
    first_view_entropy = np.asarray(first_view_entropy, dtype=np.float64)
    js = np.asarray(js, dtype=np.float64)
    order = np.argsort(first_view_entropy, kind="stable")
    summary = []
    for quartile, members in enumerate(np.array_split(order, 4), start=1):
        p5, p25, p50, p75, p95 = np.percentile(js[members], [5, 25, 50, 75, 95])
        summary.append(
            {
                "quartile": quartile,
                "count": int(len(members)),
                "entropy_min": float(first_view_entropy[members].min()),
                "entropy_max": float(first_view_entropy[members].max()),
                "js_p5": float(p5),
                "js_p25": float(p25),
                "js_median": float(p50),
                "js_p75": float(p75),
                "js_p95": float(p95),
            }
        )
    return summary


def consensus_fit_loss(log_q: Tensor, target: Tensor) -> Tensor:
    """``l^cons_t(theta) = -r_t^T log q_t(theta)`` with a fixed normalised target (Equation 9)."""
    return -(target * log_q.float()).sum(-1)


@torch.no_grad()
def position_statistics(
    student: CausalLM,
    teacher: CausalLM,
    batch: RolloutBatch,
    *,
    block_size: int = 32,
    microbatch_size: int = 8,
) -> dict[str, Tensor]:
    """Per-position statistics at fixed response prefixes.

    Returns CPU tensors of length ``N_k``: retained mass ``Z_t``, the consensus-fit loss
    ``l^cons_t`` of Equation 9, the full-vocabulary entropies ``H(p1)``, ``H(p2)``, ``H(q)``
    of Table 12(a), the cross-view JS divergence of Equation 1, and the index of the response
    each position belongs to.
    """
    keys = ("mass", "consensus_fit_loss", "entropy_p1", "entropy_p2", "entropy_q", "js_divergence")
    collected: dict[str, list[Tensor]] = {key: [] for key in keys}
    owners: list[int] = []
    rows_all = [i for i, response in enumerate(batch.responses) if response]
    for start in range(0, len(rows_all), microbatch_size):
        rows = rows_all[start : start + microbatch_size]
        responses = [batch.responses[i] for i in rows]
        original = [batch.pairs[i].original_ids for i in rows]
        wrapped = [batch.pairs[i].view2_ids for i in rows]
        for row, response in zip(rows, responses, strict=True):
            owners.extend([row] * len(response))
        blocks = zip(
            iter_log_prob_blocks(student, original, responses, block_size=block_size),
            iter_log_prob_blocks(teacher, original, responses, block_size=block_size),
            iter_log_prob_blocks(teacher, wrapped, responses, block_size=block_size),
            strict=True,
        )
        for log_q, log_p1, log_p2 in blocks:
            log_q, log_p1, log_p2 = log_q.cpu(), log_p1.cpu(), log_p2.cpu()
            target, mass = normalized_consensus(log_p1, log_p2)
            collected["mass"].append(mass)
            collected["consensus_fit_loss"].append(consensus_fit_loss(log_q, target))
            collected["entropy_p1"].append(entropy(log_p1))
            collected["entropy_p2"].append(entropy(log_p2))
            collected["entropy_q"].append(entropy(log_q))
            collected["js_divergence"].append(js_divergence(log_p1, log_p2))
    result = {
        key: torch.cat(values) if values else torch.empty(0) for key, values in collected.items()
    }
    result["response_index"] = torch.as_tensor(owners, dtype=torch.long)
    return result


@torch.no_grad()
def paired_view_statistics(
    teacher: CausalLM,
    prompts_a: Sequence[Sequence[int]],
    prompts_b: Sequence[Sequence[int]],
    responses: Sequence[Sequence[int]],
    *,
    block_size: int = 32,
    microbatch_size: int = 8,
) -> dict[str, Tensor]:
    """First-view entropy and cross-view JS divergence of paired teacher predictions.

    The preliminary study (Section 3, Figure 2) scores student-generated prefixes under two
    wrappers of the same task; every response position yields one pair ``(p1, p2)``.
    """
    entropies: list[Tensor] = []
    divergences: list[Tensor] = []
    for start in range(0, len(responses), microbatch_size):
        rows = slice(start, start + microbatch_size)
        blocks = zip(
            iter_log_prob_blocks(teacher, prompts_a[rows], responses[rows], block_size=block_size),
            iter_log_prob_blocks(teacher, prompts_b[rows], responses[rows], block_size=block_size),
            strict=True,
        )
        for log_p1, log_p2 in blocks:
            entropies.append(entropy(log_p1).cpu())
            divergences.append(js_divergence(log_p1, log_p2).cpu())
    empty = torch.empty(0)
    return {
        "entropy_p1": torch.cat(entropies) if entropies else empty,
        "js_divergence": torch.cat(divergences) if divergences else empty,
    }


def _moving_average_skip_empty(values: np.ndarray, window: int) -> np.ndarray:
    """Centered moving average over non-empty neighbours; empty (NaN) bins stay empty."""
    half = window // 2
    smoothed = np.full_like(values, np.nan)
    for index, value in enumerate(values):
        if np.isnan(value):
            continue
        neighbours = values[max(0, index - half) : index + half + 1]
        smoothed[index] = np.nanmean(neighbours)
    return smoothed


def binned_utility_summary(
    mass: np.ndarray,
    utility: np.ndarray,
    correct: np.ndarray | None = None,
    *,
    num_bins: int = 24,
    window: int = 3,
) -> dict[str, np.ndarray]:
    """Summaries of Figure 6 (Appendix B.7).

    ``[0, 1]`` is divided into ``num_bins`` equal-width bins of ``Z``. For each bin the
    median and the 10th/90th percentiles of the utility ``u_t`` and the token-weighted
    correctness rate are computed, and a ``window``-bin moving average is applied to these
    displayed summaries. Empty bins remain empty (NaN).
    """
    mass = np.asarray(mass, dtype=np.float64)
    utility = np.asarray(utility, dtype=np.float64)
    bins = np.clip(np.floor(mass * num_bins).astype(int), 0, num_bins - 1)
    edges = np.linspace(0.0, 1.0, num_bins + 1)
    raw = {key: np.full(num_bins, np.nan) for key in ("median", "p10", "p90", "correctness")}
    counts = np.zeros(num_bins, dtype=int)
    for index in range(num_bins):
        members = bins == index
        counts[index] = int(members.sum())
        if counts[index] == 0:
            continue
        raw["p10"][index], raw["median"][index], raw["p90"][index] = np.percentile(
            utility[members], [10, 50, 90]
        )
        if correct is not None:
            raw["correctness"][index] = float(np.asarray(correct, dtype=np.float64)[members].mean())
    summary: dict[str, np.ndarray] = {
        "bin_centers": 0.5 * (edges[:-1] + edges[1:]),
        "count": counts,
    }
    for key, values in raw.items():
        summary[key] = _moving_average_skip_empty(values, window)
    return summary


def matched_branch_updates(
    trainer: Trainer,
    batch: RolloutBatch,
    objectives: Mapping[str, Objective],
    evaluate: Callable[[], float] | None = None,
) -> dict[str, Any]:
    """Local-update diagnostic with matched branches (Appendix B.7, Table 12, Figure 6).

    Every branch starts from the same checkpoint and optimizer state, reuses the same
    sampled response token ids and teacher views in ``batch``, takes one update, and differs
    only in its objective. For each branch the token-level consensus-fit utility
    ``u_t = l^cons_t(theta_before) - l^cons_t(theta_after)`` (Equation 9) is measured at the
    fixed pre-update prefixes; ``evaluate`` (e.g. Avg@8 on the source-disjoint measurement
    questions) is called before and after each branch. The trainer is restored afterwards.
    """
    model_state = copy.deepcopy(trainer.student.model.state_dict())
    optimizer_state = copy.deepcopy(trainer.optimizer.state_dict())
    flops_state = trainer.flops.state_dict()

    def statistics() -> dict[str, Tensor]:
        return position_statistics(
            trainer.student,
            trainer.teacher,
            batch,
            block_size=trainer.cfg.position_block_size,
            microbatch_size=trainer.cfg.microbatch_size,
        )

    before = statistics()
    result: dict[str, Any] = {
        "pre_update": {
            key: float(before[key].double().mean())
            for key in ("entropy_p1", "entropy_p2", "entropy_q", "mass")
        },
        "mass": before["mass"],
        "response_index": before["response_index"],
        "before_score": evaluate() if evaluate is not None else None,
        "branches": {},
    }
    try:
        for name, objective in objectives.items():
            trainer.student.model.load_state_dict(model_state)
            trainer.optimizer.load_state_dict(copy.deepcopy(optimizer_state))
            stats = trainer.accumulate_gradients(batch, objective)
            stepped = trainer.apply_update(stats)
            after = statistics()
            result["branches"][name] = {
                "status": stats["status"],
                "stepped": stepped,
                "loss": stats.get("loss"),
                "utility": before["consensus_fit_loss"] - after["consensus_fit_loss"],
                "after_score": evaluate() if evaluate is not None else None,
            }
    finally:
        trainer.student.model.load_state_dict(model_state)
        trainer.optimizer.load_state_dict(optimizer_state)
        trainer.flops.load_state_dict(flops_state)
    return result


@torch.no_grad()
def candidate_logits(
    model: CausalLM, state_ids: Sequence[int], candidates: Sequence[Sequence[int]]
) -> np.ndarray:
    """Length-normalised log-likelihood of candidate answers at a fixed state (Equation 15).

    ``z_k = (1 / L_k) sum_l log pi(a_{k,l} | state, a_{k,<l})`` where ``state_ids`` is the
    serialised prompt view followed by the fixed response prefix ``y_{<t}`` and each
    candidate is the token sequence of a complete canonicalised final-answer string. A
    softmax over the returned vector gives the candidate distribution of Equation 16.
    """
    log_probs = score_responses(model, [list(state_ids)] * len(candidates), candidates)
    values, offset = [], 0
    for candidate in candidates:
        rows = log_probs[offset : offset + len(candidate)]
        tokens = torch.as_tensor(list(candidate), dtype=torch.long, device=rows.device)
        values.append(float(rows.gather(1, tokens.unsqueeze(1)).cpu().double().mean()))
        offset += len(candidate)
    return np.asarray(values, dtype=np.float64)
