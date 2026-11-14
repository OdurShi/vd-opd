"""Diagnostics analysed in the paper and in the supplementary experiments.

* Section 3 / Figure 2: first-view entropy versus cross-view JS divergence.
* Section 5.5 / Equation 9 / Figure 6: consensus mass versus consensus-fit utility.
* Appendix B.7 / Table 12: shared pre-update statistics and matched branch updates.
* Appendix B.8 / Equation 15: length-normalised candidate-answer logits.
* Supplementary E09: gradient agreement of the three possible (original, alternative) pairs.
* Supplementary E16: teacher diagnostics and continuation outcomes by quartile of the
  cross-view agreement ``Z`` at fixed student prefixes.
* Supplementary E17: matched local updates measured by the oracle KL of Equation 21, the
  probability of a shared wrong answer and continuation correctness, per ``Z`` stratum.
"""

from __future__ import annotations

import copy
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from vdopd.consensus import entropy, js_divergence, normalized_consensus
from vdopd.eval.answers import extract_answer, is_equivalent, normalize_answer
from vdopd.eval.bootstrap import DEFAULT_RESAMPLES, BootstrapCI, clustered_mean_bootstrap
from vdopd.eval.evaluator import EvalConfig
from vdopd.models import CausalLM
from vdopd.objectives import Objective
from vdopd.scoring import iter_log_prob_blocks, pack_prompt_and_response, response_hidden_states
from vdopd.seeding import derive_rng, derive_seed
from vdopd.theory import smoothed_reference
from vdopd.trainer import RolloutBatch, RolloutConfig, Trainer
from vdopd.views import TRAIN_WRAPPERS, Message, PromptCodec, View, select_views


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


def _blocks_from_hidden(model: CausalLM, hidden: Tensor, block_size: int) -> Iterator[Tensor]:
    """Full-vocabulary log-probability blocks from cached response hidden states."""
    for start in range(0, hidden.shape[0], block_size):
        yield F.log_softmax(model.project(hidden[start : start + block_size]).float(), dim=-1)


@torch.no_grad()
def position_statistics(
    student: CausalLM,
    teacher: CausalLM,
    batch: RolloutBatch,
    *,
    block_size: int = 32,
    microbatch_size: int = 8,
    teacher_hidden: Sequence[Sequence[Tensor]] | None = None,
) -> dict[str, Tensor]:
    """Per-position statistics at fixed response prefixes.

    Returns CPU tensors of length ``N_k``: retained mass ``Z_t``, the consensus-fit loss
    ``l^cons_t`` of Equation 9, the full-vocabulary entropies ``H(p1)``, ``H(p2)``, ``H(q)``
    of Table 12(a), the cross-view JS divergence of Equation 1, and the index of the response
    each position belongs to.

    ``teacher_hidden`` optionally supplies the teacher hidden states of the batch (the
    result of ``Trainer.teacher_hidden_states`` with the same microbatch size and at least
    two views). The frozen teacher is then not run again, which matters when the statistics
    are evaluated repeatedly while only the student changes.
    """
    keys = ("mass", "consensus_fit_loss", "entropy_p1", "entropy_p2", "entropy_q", "js_divergence")
    collected: dict[str, list[Tensor]] = {key: [] for key in keys}
    owners: list[int] = []
    rows_all = [i for i, response in enumerate(batch.responses) if response]
    starts = range(0, len(rows_all), microbatch_size)
    if teacher_hidden is not None and len(teacher_hidden) != len(starts):
        raise ValueError("teacher_hidden does not match the microbatches of the batch")
    for chunk, start in enumerate(starts):
        rows = rows_all[start : start + microbatch_size]
        responses = [batch.responses[i] for i in rows]
        student_prompts = [batch.pairs[i].student_prompt_ids for i in rows]
        for row, response in zip(rows, responses, strict=True):
            owners.extend([row] * len(response))
        if teacher_hidden is None:
            original = [batch.pairs[i].original_ids for i in rows]
            wrapped = [batch.pairs[i].view2_ids for i in rows]
            teacher_blocks = (
                iter_log_prob_blocks(teacher, original, responses, block_size=block_size),
                iter_log_prob_blocks(teacher, wrapped, responses, block_size=block_size),
            )
        else:
            positions = sum(len(response) for response in responses)
            if any(hidden.shape[0] != positions for hidden in teacher_hidden[chunk][:2]):
                raise ValueError("teacher_hidden does not match the microbatches of the batch")
            teacher_blocks = tuple(
                _blocks_from_hidden(teacher, hidden, block_size)
                for hidden in teacher_hidden[chunk][:2]
            )
        blocks = zip(
            iter_log_prob_blocks(student, student_prompts, responses, block_size=block_size),
            *teacher_blocks,
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
    evaluate: Callable[[], Any] | None = None,
) -> dict[str, Any]:
    """Local-update diagnostic with matched branches (Appendix B.7, Table 12, Figure 6).

    Every branch starts from the same checkpoint and optimizer state, reuses the same
    sampled response token ids and teacher views in ``batch``, takes one update, and differs
    only in its objective. For each branch the token-level consensus-fit utility
    ``u_t = l^cons_t(theta_before) - l^cons_t(theta_after)`` (Equation 9) is measured at the
    fixed pre-update prefixes; ``evaluate`` (e.g. Avg@8 on the source-disjoint measurement
    questions, or the state-level measurements of supplementary E17) is called before and
    after each branch and its return values are stored as ``before_score`` and
    ``after_score``. The trainer is restored afterwards.

    All branches share the rollouts, hence the same ``N_k`` update positions, and the same
    teacher distributions: the frozen teacher is run once for all branches and all
    statistics passes.
    """
    model_state = copy.deepcopy(trainer.student.model.state_dict())
    optimizer_state = copy.deepcopy(trainer.optimizer.state_dict())
    flops_state = trainer.flops.state_dict()
    # Under student-side augmentation a single-view objective scores the student's prompt,
    # which a shared two-view cache does not hold; the teacher is then run per branch.
    teacher_hidden = None
    if trainer.view_cfg.student_wrap_probability == 0 and batch.num_positions > 0:
        views = max([2, *(objective.num_views for objective in objectives.values())])
        teacher_hidden = trainer.teacher_hidden_states(batch, views)

    def statistics() -> dict[str, Tensor]:
        return position_statistics(
            trainer.student,
            trainer.teacher,
            batch,
            block_size=trainer.cfg.position_block_size,
            microbatch_size=trainer.cfg.microbatch_size,
            teacher_hidden=teacher_hidden,
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
            stats = trainer.accumulate_gradients(batch, objective, teacher_hidden)
            stepped = trainer.apply_update(stats)
            after = statistics()
            result["branches"][name] = {
                "status": stats["status"],
                "stepped": stepped,
                "num_positions": stats["num_positions"],
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
def sequence_log_likelihoods(
    model: CausalLM,
    prompts: Sequence[Sequence[int]],
    responses: Sequence[Sequence[int]],
    *,
    block_size: int = 32,
) -> np.ndarray:
    """Length-normalised log-likelihood ``(1/L) sum_l log pi(a_l | prompt, a_<l)`` of each
    response, in float64. One forward pass scores all responses; the per-token
    log-probabilities are read block by block and averaged per response without a Python
    loop over tokens."""
    packed = pack_prompt_and_response(prompts, responses, model.pad_token_id)
    lengths = torch.as_tensor([len(r) for r in responses], dtype=torch.long)
    if int(lengths.min()) == 0:
        raise ValueError("every response needs at least one token")
    hidden = response_hidden_states(model, packed)
    token_log_probs: list[Tensor] = []
    for start in range(0, hidden.shape[0], block_size):
        log_probs = F.log_softmax(model.project(hidden[start : start + block_size]).float(), -1)
        tokens = packed.tokens[start : start + block_size].to(log_probs.device)
        token_log_probs.append(log_probs.gather(1, tokens.unsqueeze(1)).squeeze(1).cpu().double())
    owner = torch.repeat_interleave(torch.arange(len(responses)), lengths)
    totals = torch.zeros(len(responses), dtype=torch.float64).index_add_(
        0, owner, torch.cat(token_log_probs)
    )
    return (totals / lengths).numpy()


def candidate_logits(
    model: CausalLM, state_ids: Sequence[int], candidates: Sequence[Sequence[int]]
) -> np.ndarray:
    """Length-normalised log-likelihood of candidate answers at a fixed state (Equation 15).

    ``z_k = (1 / L_k) sum_l log pi(a_{k,l} | state, a_{k,<l})`` where ``state_ids`` is the
    serialised prompt view followed by the fixed response prefix ``y_{<t}`` and each
    candidate is the token sequence of a complete canonicalised final-answer string. A
    softmax over the returned vector gives the candidate distribution of Equation 16.
    """
    return sequence_log_likelihoods(model, [list(state_ids)] * len(candidates), candidates)


def candidate_distributions(
    model: CausalLM,
    states: Sequence[Sequence[int]],
    candidates: Sequence[Sequence[Sequence[int]]],
    *,
    batch_size: int = 8,
    block_size: int = 32,
) -> list[np.ndarray]:
    """Candidate distributions of Equation 16 for many states (supplementary E17).

    ``states[i]`` is the token sequence of state ``i`` (prompt view followed by the fixed
    prefix) and ``candidates[i]`` its candidate answers as token sequences; the number of
    candidates may differ between states. ``batch_size`` states are scored per forward pass.
    """
    if len(states) != len(candidates):
        raise ValueError("states and candidates must have the same length")
    distributions: list[np.ndarray] = []
    for start in range(0, len(states), batch_size):
        chunk = range(start, min(start + batch_size, len(states)))
        prompts = [list(states[i]) for i in chunk for _ in candidates[i]]
        responses = [list(candidate) for i in chunk for candidate in candidates[i]]
        logits = sequence_log_likelihoods(model, prompts, responses, block_size=block_size)
        offset = 0
        for i in chunk:
            z = logits[offset : offset + len(candidates[i])]
            offset += len(candidates[i])
            z = z - z.max()
            distributions.append(np.exp(z) / np.exp(z).sum())
    return distributions


# ------------------------------------------------- supplementary E09: pair gradients
def pair_gradient_cosines(
    trainer: Trainer,
    batch: RolloutBatch,
    views: Sequence[View] | None = None,
    objective: Objective | None = None,
    *,
    storage_device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Gradient agreement of the possible (original, alternative) view pairs (E09).

    At the fixed states of ``batch`` (its sampled response token ids and the current student
    parameters), the gradient of the objective is accumulated once for every alternative
    view, each time pairing the original prompt of *every* response with that one view. The
    result holds the cosine similarity between each pair's gradient and the mean gradient
    over all pairs, the pairwise cosines and the gradient norms. With the three training
    wrappers these are the three possible pairs of which default training samples one per
    response.

    The gradients are compared through their Gram matrix, so one copy per pair is kept on
    ``storage_device``. Student parameters, optimizer state and the FLOPs meter are left
    unchanged; a response whose prompt cannot take a view falls back to the identity view as
    in Appendix A.1.
    """
    views = tuple(trainer.wrappers if views is None else views)
    objective = objective or trainer.objective
    if not views:
        raise ValueError("at least one alternative view is required")
    flops_state = trainer.flops.state_dict()
    rollout = trainer.rollout_cfg
    gradients: list[list[Tensor]] = []
    try:
        for view in views:
            pairs = [
                select_views(
                    trainer.prompts[index],
                    trainer.codec,
                    max_response_tokens=rollout.max_response_tokens,
                    teacher_context=trainer.teacher.max_context,
                    student_context=trainer.student.max_context,
                    rng=np.random.default_rng(0),  # a single candidate: nothing is sampled
                    wrappers=(view,),
                )
                for index in batch.prompt_indices
            ]
            branch = RolloutBatch(
                batch.iteration, batch.prompt_indices, pairs, batch.responses, batch.num_excluded
            )
            stats = trainer.accumulate_gradients(branch, objective)
            if stats["status"] != "ok":
                raise RuntimeError(f"view {view.name!r}: gradient status {stats['status']!r}")
            gradients.append(
                [
                    torch.zeros_like(p, dtype=torch.float32, device=storage_device)
                    if p.grad is None
                    else p.grad.detach().to(storage_device, torch.float32, copy=True)
                    for p in trainer.parameters
                ]
            )
    finally:
        trainer.optimizer.zero_grad(set_to_none=True)
        trainer.flops.load_state_dict(flops_state)

    count = len(views)
    gram = np.zeros((count, count), dtype=np.float64)
    for a in range(count):
        for b in range(a, count):
            gram[a, b] = gram[b, a] = sum(
                float(torch.dot(x.flatten().double().cpu(), y.flatten().double().cpu()))
                for x, y in zip(gradients[a], gradients[b], strict=True)
            )
    norms = np.sqrt(np.diag(gram))
    mean_norm = float(np.sqrt(gram.mean()))  # ||(1/V) sum_v g_v||
    with np.errstate(divide="ignore", invalid="ignore"):
        to_mean = gram.mean(axis=1) / (norms * mean_norm)  # <g_v, g_mean> / (|g_v| |g_mean|)
        pairwise = gram / np.outer(norms, norms)
    return {
        "views": [view.name for view in views],
        "cosine_to_mean": {view.name: float(c) for view, c in zip(views, to_mean, strict=True)},
        "pairwise_cosine": pairwise.tolist(),
        "gradient_norm": {view.name: float(n) for view, n in zip(views, norms, strict=True)},
        "mean_gradient_norm": mean_norm,
        "num_positions": batch.num_positions,
    }


# ------------------------------------- supplementary E16 / E17: fixed analysis states
@dataclass(frozen=True)
class AnalysisState:
    """A fixed student-generated prefix of a measurement question (Appendix B.8).

    The state is ``(x, y_<t)``: the question's prompt and the first ``t`` tokens of a student
    response; ``next_token`` is the token ``y_t`` the student sampled there. ``view2_ids`` is
    the alternative teacher view of the question.
    """

    question: int
    original_ids: tuple[int, ...]
    view2_ids: tuple[int, ...]
    wrapper: str
    prefix: tuple[int, ...]
    next_token: int


def build_analysis_states(
    student: CausalLM,
    codec: PromptCodec,
    prompts: Sequence[Sequence[Message]],
    *,
    rollout: RolloutConfig,
    seed: int,
    num_prefixes: int = 4,
    wrappers: Sequence[View] = TRAIN_WRAPPERS,
    teacher_context: int | None = None,
) -> list[AnalysisState]:
    """Sample fixed prefixes of student responses to the measurement questions (E16, E17).

    For every question the student generates ``num_prefixes`` independent responses with the
    rollout settings, and each response is cut at one uniformly sampled position ``t``,
    which gives the state ``(x, y_<t)``. Each question gets one alternative teacher view,
    sampled uniformly from the valid ``wrappers`` as in training. Questions whose original
    prompt does not fit the context limits are skipped. Everything is seeded through the
    ``states`` random stream.
    """
    view_rng = derive_rng(seed, "states", 0)
    cut_rng = derive_rng(seed, "states", 1)
    pairs = []
    for question, messages in enumerate(prompts):
        pair = select_views(
            messages,
            codec,
            max_response_tokens=rollout.max_response_tokens,
            teacher_context=teacher_context,
            student_context=student.max_context,
            rng=view_rng,
            wrappers=wrappers,
        )
        if pair is not None:
            pairs.extend([(question, pair)] * num_prefixes)

    states: list[AnalysisState] = []
    size = rollout.generation_batch_size
    for chunk, start in enumerate(range(0, len(pairs), size)):
        members = pairs[start : start + size]
        responses = student.generate(
            [pair.original_ids for _, pair in members],
            max_new_tokens=rollout.max_response_tokens,
            temperature=rollout.temperature,
            top_p=rollout.top_p,
            top_k=rollout.top_k,
            seed=derive_seed(seed, "states", 2, chunk),
        )
        for (question, pair), response in zip(members, responses, strict=True):
            if not response:
                continue
            cut = int(cut_rng.integers(len(response)))
            states.append(
                AnalysisState(
                    question=question,
                    original_ids=pair.original_ids,
                    view2_ids=pair.view2_ids,
                    wrapper=pair.wrapper.name,
                    prefix=tuple(response[:cut]),
                    next_token=int(response[cut]),
                )
            )
    return states


def hellinger_tail_split(log_p1: Tensor, log_p2: Tensor, top_k: int = 10) -> tuple[Tensor, Tensor]:
    """Squared Hellinger distance outside the union of both views' top-``k`` tokens.

    Returns ``(outside, total)`` with ``total = 0.5 ||sqrt(p1) - sqrt(p2)||^2 = 1 - Z`` and
    ``outside`` the part of that sum over tokens that are in neither view's ``top_k`` most
    likely tokens (supplementary E16).
    """
    log_p1, log_p2 = log_p1.float(), log_p2.float()
    terms = 0.5 * ((0.5 * log_p1).exp() - (0.5 * log_p2).exp()).square()
    k = min(top_k, terms.shape[-1])
    union = torch.cat([log_p1.topk(k, dim=-1).indices, log_p2.topk(k, dim=-1).indices], dim=-1)
    head = torch.zeros_like(terms, dtype=torch.bool).scatter_(-1, union, True)
    return terms.masked_fill(head, 0.0).sum(-1), terms.sum(-1)


@torch.no_grad()
def state_agreement(
    teacher: CausalLM,
    states: Sequence[AnalysisState],
    *,
    block_size: int = 32,
    microbatch_size: int = 8,
    top_k: int = 10,
) -> dict[str, np.ndarray]:
    """Cross-view teacher diagnostics at every analysis state (supplementary E16).

    For the next-token distributions ``p1``, ``p2`` of the two teacher views at each state:
    the retained mass ``Z`` (``mass``), whether both views have the same most likely token
    (``same_top1``), and the squared Hellinger distance in total and outside the union of
    both views' ``top_k`` tokens (``hellinger_total``, ``hellinger_outside``).
    """
    keys = ("mass", "same_top1", "hellinger_total", "hellinger_outside")
    collected: dict[str, list[Tensor]] = {key: [] for key in keys}
    for start in range(0, len(states), microbatch_size):
        members = states[start : start + microbatch_size]
        targets = [[state.next_token] for state in members]
        blocks = zip(
            iter_log_prob_blocks(
                teacher,
                [state.original_ids + state.prefix for state in members],
                targets,
                block_size=block_size,
            ),
            iter_log_prob_blocks(
                teacher,
                [state.view2_ids + state.prefix for state in members],
                targets,
                block_size=block_size,
            ),
            strict=True,
        )
        for log_p1, log_p2 in blocks:
            log_p1, log_p2 = log_p1.cpu(), log_p2.cpu()
            outside, total = hellinger_tail_split(log_p1, log_p2, top_k)
            collected["mass"].append(normalized_consensus(log_p1, log_p2)[1])
            collected["same_top1"].append(log_p1.argmax(-1) == log_p2.argmax(-1))
            collected["hellinger_total"].append(total)
            collected["hellinger_outside"].append(outside)
    return {
        key: (torch.cat(values).numpy() if values else np.empty(0))
        for key, values in collected.items()
    }


def sample_final_answers(
    model: CausalLM,
    codec: PromptCodec,
    prompts: Sequence[Sequence[int]],
    prefixes: Sequence[Sequence[int]],
    cfg: EvalConfig,
    *,
    num_samples: int,
    seed: int,
    stream: int = 0,
) -> list[list[str | None]]:
    """Final answers of ``num_samples`` sampled continuations of every state.

    State ``i`` is the encoded prompt ``prompts[i]`` followed by the fixed response prefix
    ``prefixes[i]``. The model continues it with the decoding settings of ``cfg``; the
    complete response (prefix plus continuation) is capped at ``cfg.max_response_tokens``
    and its final answer is extracted. Seeds depend only on ``seed``, ``stream`` and the
    batch, so calls with the same arguments use paired random streams: the two teacher
    views of a continuation pair (E16), or the student before and after an update (E17).
    """
    if len(prompts) != len(prefixes):
        raise ValueError("prompts and prefixes must have the same length")
    answers: list[list[str | None]] = [[None] * num_samples for _ in prompts]
    jobs = [(i, j) for i in range(len(prompts)) for j in range(num_samples)]
    for batch_index, start in enumerate(range(0, len(jobs), cfg.batch_size)):
        batch = jobs[start : start + cfg.batch_size]
        inputs = [list(prompts[i]) + list(prefixes[i]) for i, _ in batch]
        remaining = [max(0, cfg.max_response_tokens - len(prefixes[i])) for i, _ in batch]
        allowance = max(remaining)
        if model.max_context is not None:
            allowance = min(allowance, model.max_context - max(len(ids) for ids in inputs))
        continuations: list[list[int]] = [[] for _ in batch]
        if allowance > 0:
            continuations = model.generate(
                inputs,
                max_new_tokens=allowance,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                seed=derive_seed(seed, "evaluation", stream, batch_index),
            )
        for (i, j), continuation, budget in zip(batch, continuations, remaining, strict=True):
            text = codec.decode(list(prefixes[i]) + list(continuation[:budget]))
            answers[i][j] = extract_answer(text)
    return answers


OUTCOMES = ("both_correct", "one_correct", "same_wrong", "different_wrong")


def continuation_outcome(
    first: str | None, second: str | None, reference: str, rel_tol: float = 1e-9
) -> str:
    """Outcome category of a pair of continuations of the same state (supplementary E16).

    ``both_correct`` / ``one_correct`` compare each final answer with the reference;
    ``same_wrong`` requires two incorrect final answers that are equivalent to each other;
    every other incorrect pair, including pairs with a missing final answer, is
    ``different_wrong``.
    """
    correct = (is_equivalent(first, reference, rel_tol), is_equivalent(second, reference, rel_tol))
    if all(correct):
        return "both_correct"
    if any(correct):
        return "one_correct"
    if first is not None and second is not None and is_equivalent(first, second, rel_tol):
        return "same_wrong"
    return "different_wrong"


def pair_outcomes(
    answers_view1: Sequence[Sequence[str | None]],
    answers_view2: Sequence[Sequence[str | None]],
    references: Sequence[str],
    rel_tol: float = 1e-9,
) -> np.ndarray:
    """Outcome codes (indices into :data:`OUTCOMES`) of all continuation pairs, ``[n, pairs]``."""
    return np.array(
        [
            [
                OUTCOMES.index(continuation_outcome(a, b, reference, rel_tol))
                for a, b in zip(first, second, strict=True)
            ]
            for first, second, reference in zip(
                answers_view1, answers_view2, references, strict=True
            )
        ],
        dtype=np.int64,
    )


def shared_wrong_answers(
    answers_view1: Sequence[Sequence[str | None]],
    answers_view2: Sequence[Sequence[str | None]],
    references: Sequence[str],
    rel_tol: float = 1e-9,
) -> list[str | None]:
    """The wrong final answer shared by both teacher views at each state, if any (E17).

    A state is eligible when at least one of its continuation pairs ends in the same wrong
    final answer under both views; its shared wrong answer is the one that occurs in most
    such pairs (the first in case of a tie). Other states get ``None``.
    """
    shared: list[str | None] = []
    for first, second, reference in zip(answers_view1, answers_view2, references, strict=True):
        votes: Counter[str] = Counter()
        spelling: dict[str, str] = {}
        for a, b in zip(first, second, strict=True):
            if continuation_outcome(a, b, reference, rel_tol) == "same_wrong":
                key = normalize_answer(a)
                votes[key] += 1
                spelling.setdefault(key, a)
        shared.append(spelling[votes.most_common(1)[0][0]] if votes else None)
    return shared


def quartile_labels(values: np.ndarray | Sequence[float]) -> np.ndarray:
    """Quartile (1 = lowest, 4 = highest) of every value.

    Values are sorted and partitioned into four groups of equal size (up to rounding), the
    convention of the entropy quartiles of Figure 2(b).
    """
    array = np.asarray(values, dtype=np.float64)
    labels = np.zeros(len(array), dtype=np.int64)
    order = np.argsort(array, kind="stable")
    for quartile, members in enumerate(np.array_split(order, 4), start=1):
        labels[members] = quartile
    return labels


def agreement_summary(
    agreement: Mapping[str, np.ndarray], outcomes: np.ndarray | None = None
) -> list[dict[str, float]]:
    """Teacher diagnostics per quartile of the cross-view agreement ``Z`` (supplementary E16).

    Args:
        agreement: output of :func:`state_agreement`.
        outcomes: ``[states, pairs]`` outcome codes of the paired teacher continuations
            (:func:`pair_outcomes`), or ``None`` to omit the outcome columns.

    Returns:
        One record per quartile (Q1 = lowest ``Z``) with the number of states and
        continuation pairs, the mean ``Z``, the rate of states whose two views share the
        most likely next token, ``tail_share`` (the stratum's total squared Hellinger
        distance outside the union of both views' top tokens divided by its total squared
        Hellinger distance) and the fraction of continuation pairs in each outcome category
        (micro-average over the stratum's pairs).
    """
    labels = quartile_labels(agreement["mass"])
    rows = []
    for quartile in range(1, 5):
        members = labels == quartile
        total = float(agreement["hellinger_total"][members].sum())
        row: dict[str, float] = {
            "quartile": quartile,
            "states": int(members.sum()),
            "mean_mass": float(agreement["mass"][members].mean()) if members.any() else np.nan,
            "same_top1": float(agreement["same_top1"][members].mean()) if members.any() else np.nan,
            "tail_share": float(agreement["hellinger_outside"][members].sum()) / total
            if total > 0
            else np.nan,
        }
        if outcomes is not None:
            stratum = outcomes[members]
            row["pairs"] = int(stratum.size)
            for code, name in enumerate(OUTCOMES):
                row[name] = float((stratum == code).mean()) if stratum.size else np.nan
        rows.append(row)
    return rows


AGREEMENT_ROWS = (
    ("States", "states", 1.0, 0),
    ("Continuation pairs", "pairs", 1.0, 0),
    ("Mean Z", "mean_mass", 1.0, 2),
    ("Same top-1 next token (%)", "same_top1", 100.0, 0),
    ("Disagreement outside union of top-10 tokens (%)", "tail_share", 100.0, 0),
    ("Both continuations correct (%, approx.)", "both_correct", 100.0, 0),
    ("Exactly one correct (%, approx.)", "one_correct", 100.0, 0),
    ("Same wrong final answer (%, approx.)", "same_wrong", 100.0, 0),
    ("Different wrong answers (%, approx.)", "different_wrong", 100.0, 0),
)


def agreement_table(summary: Sequence[Mapping[str, float]]) -> list[dict[str, str]]:
    """Rows ``diagnostic, low_z_q1, high_z_q4`` of the E16 table from :func:`agreement_summary`."""
    low, high = summary[0], summary[-1]
    rows = []
    for label, key, scale, digits in AGREEMENT_ROWS:
        if key not in low:
            continue
        rows.append(
            {
                "diagnostic": label,
                "low_z_q1": f"{scale * low[key]:.{digits}f}",
                "high_z_q4": f"{scale * high[key]:.{digits}f}",
            }
        )
    return rows


# --------------------------------------- supplementary E17: matched local updates by Z
Z_STRATA = (("Low", 1), ("High", 4))  # bottom and top quartile of Z over the states


def oracle_kl(distribution: np.ndarray, correct_index: int = 0, epsilon: float = 0.01) -> float:
    """``D_KL(p* || q)`` of a candidate distribution ``q`` against the fixed oracle target.

    ``p*`` puts ``1 - epsilon`` on the correct candidate and ``epsilon / (K - 1)`` on each
    other one (Equation 17); for four candidates that is ``(0.99, 0.01/3, 0.01/3, 0.01/3)``.
    """
    q = np.asarray(distribution, dtype=np.float64)
    p_star = smoothed_reference(correct_index, len(q), epsilon)
    return float(np.sum(p_star * (np.log(p_star) - np.log(q))))


def oracle_kl_decrease(
    before: Sequence[np.ndarray],
    after: Sequence[np.ndarray],
    correct_index: int = 0,
    epsilon: float = 0.01,
) -> np.ndarray:
    """Per-state ``u = D_KL(p* || q_before) - D_KL(p* || q_after)`` (Equation 21; E17).

    Positive values mean that the update moved the student's candidate distribution towards
    the oracle target, i.e. the oracle KL decreased.
    """
    return np.array(
        [
            oracle_kl(b, correct_index, epsilon) - oracle_kl(a, correct_index, epsilon)
            for b, a in zip(before, after, strict=True)
        ]
    )


def stratum_means(
    values: np.ndarray | Sequence[float],
    questions: np.ndarray | Sequence[int],
    quartiles: np.ndarray | Sequence[int],
    *,
    eligible: np.ndarray | Sequence[bool] | None = None,
    strata: Sequence[tuple[str, int]] = Z_STRATA,
    num_resamples: int = DEFAULT_RESAMPLES,
    seed: int = 0,
) -> dict[str, BootstrapCI | None]:
    """Mean of a per-state quantity in each ``Z`` stratum with a 95% bootstrap interval.

    States are grouped by ``quartiles`` (:func:`quartile_labels` of ``Z``); ``eligible``
    optionally restricts the states (e.g. to those with a shared wrong answer). The
    interval resamples questions, each bringing along all of its states in the stratum
    (:func:`vdopd.eval.bootstrap.clustered_mean_bootstrap`). A stratum without states gives
    ``None``.
    """
    values = np.asarray(values, dtype=np.float64)
    questions = np.asarray(questions)
    quartiles = np.asarray(quartiles)
    keep = np.ones(len(values), dtype=bool) if eligible is None else np.asarray(eligible, bool)
    result: dict[str, BootstrapCI | None] = {}
    for label, quartile in strata:
        members = keep & (quartiles == quartile)
        result[label] = (
            clustered_mean_bootstrap(
                values[members], questions[members], num_resamples=num_resamples, seed=seed
            )
            if members.any()
            else None
        )
    return result
