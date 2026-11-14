"""Sampled evaluation over prompt views (Appendix B.2).

Mathematical evaluations use zero-shot prompting and eight independent completions per
question at temperature 1.0 with top-p 0.8 and an 8,192-token response cap. The student is
evaluated independently on every view, without teacher assistance or answer ensembling.

The same function evaluates any checkpoint - a distilled student, the initial student or the
frozen teacher - with one serialisation, one set of stop tokens and one answer checker, which
is the common protocol of the supplementary experiments E02-E05. Beyond the four held-out
prefix wrappers it can evaluate further held-out view families (structural changes, verified
paraphrases; E14 and E15), record the greedy-answer flips and parse failures of E12 and E15,
and keep the per-question results that the paired question bootstrap resamples.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from vdopd.data.problems import DEFAULT_PROMPT_TEMPLATE, Problem, build_messages
from vdopd.eval.answers import extract_answer, is_equivalent
from vdopd.eval.metrics import (
    avg_at_k,
    greedy_answer_flip_rate,
    parse_failure_rate,
    pass_at_k,
    sensitivity_gaps,
    view_robustness,
)
from vdopd.models import CausalLM
from vdopd.seeding import derive_seed
from vdopd.views import (
    EVAL_WRAPPERS,
    ORIGINAL_VIEW,
    PREFIX_FAMILY,
    PromptCodec,
    RewrittenView,
    View,
    ViewFamily,
    apply_wrapper,
    preserves_content_and_roles,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvalConfig:
    """Decoding protocol of Appendix B.2.1 (defaults are the paper's values)."""

    num_samples: int = 8
    temperature: float = 1.0
    top_p: float = 0.8
    max_response_tokens: int = 8192
    batch_size: int = 16
    seed: int = 11
    rel_tol: float = 1e-9
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE
    system_prompt: str | None = None
    exemplars: tuple[tuple[str, str], ...] = ()


@dataclass
class Completions:
    """Scored sampled completions of ``n`` questions under one view.

    ``correct`` and ``parsed`` are boolean ``[n, k]`` arrays: whether completion ``s`` of
    question ``i`` is correct, and whether a final answer could be extracted from it.
    """

    correct: np.ndarray
    parsed: np.ndarray


def _encode_prompts(
    codec: PromptCodec,
    problems: Sequence[Problem],
    view: View | Sequence[View],
    cfg: EvalConfig,
) -> list[list[int]]:
    """Encoded prompts of all questions under one view (or one view per question).

    A view whose content was written outside the repository (a paraphrase) must leave
    roles, earlier turns and the assistant generation prefix of the serialised prompt
    unchanged; otherwise the evaluation stops (Appendix A.1).
    """
    per_question = isinstance(view, (list, tuple))
    if per_question and len(view) != len(problems):
        raise ValueError("per-question views need one view per problem")
    prompts: list[list[int]] = []
    for index, problem in enumerate(problems):
        messages = build_messages(
            problem.question, cfg.prompt_template, cfg.system_prompt, cfg.exemplars
        )
        current = view[index] if per_question else view
        wrapped = apply_wrapper(messages, current)
        if isinstance(current, RewrittenView) and not preserves_content_and_roles(
            messages, wrapped, current, codec
        ):
            raise ValueError(
                f"view {current.name!r} of problem {problem.id!r} does not preserve the "
                "roles and context of the serialised prompt"
            )
        prompts.append(codec.encode(wrapped))
    return prompts


def _allowance(model: CausalLM, prompts: Sequence[Sequence[int]], cfg: EvalConfig) -> int:
    """Response cap of a batch: the configured cap, reduced to fit the context limit."""
    allowance = cfg.max_response_tokens
    if model.max_context is not None:
        allowance = min(allowance, model.max_context - max(len(p) for p in prompts))
    return allowance


def sample_completions(
    model: CausalLM,
    codec: PromptCodec,
    problems: Sequence[Problem],
    wrapper: View | Sequence[View],
    cfg: EvalConfig,
    view_index: int = 0,
) -> Completions:
    """Generate ``k`` completions per question under one view and score them.

    ``wrapper`` is one view for all questions or a sequence with one view per question
    (paraphrases). Random seeds depend only on the evaluation seed, the view and the batch,
    so different checkpoints are compared on paired random streams.
    """
    missing = [problem.id for problem in problems if problem.answer is None]
    if missing:  # checked before any completion is generated
        raise ValueError(f"problem {missing[0]} has no reference answer")
    prompts = _encode_prompts(codec, problems, wrapper, cfg)
    correct = np.zeros((len(problems), cfg.num_samples), dtype=bool)
    parsed = np.zeros_like(correct)
    jobs = [(q, s) for q in range(len(problems)) for s in range(cfg.num_samples)]
    for batch_index, start in enumerate(range(0, len(jobs), cfg.batch_size)):
        batch = jobs[start : start + cfg.batch_size]
        batch_prompts = [prompts[q] for q, _ in batch]
        allowance = _allowance(model, batch_prompts, cfg)
        if allowance <= 0:
            continue  # overlong input: no completion, zero credit
        responses = model.generate(
            batch_prompts,
            max_new_tokens=allowance,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            seed=derive_seed(cfg.seed, "evaluation", view_index, batch_index),
        )
        for (q, s), response in zip(batch, responses, strict=True):
            answer = extract_answer(codec.decode(response))
            parsed[q, s] = answer is not None
            correct[q, s] = is_equivalent(answer, problems[q].answer, cfg.rel_tol)
    return Completions(correct, parsed)


def sample_correctness(
    model: CausalLM,
    codec: PromptCodec,
    problems: Sequence[Problem],
    wrapper: View | Sequence[View],
    cfg: EvalConfig,
    view_index: int = 0,
) -> np.ndarray:
    """Boolean ``[n, k]`` correctness of the sampled completions under one view."""
    return sample_completions(model, codec, problems, wrapper, cfg, view_index).correct


def greedy_answers(
    model: CausalLM,
    codec: PromptCodec,
    problems: Sequence[Problem],
    wrapper: View | Sequence[View],
    cfg: EvalConfig,
) -> list[str | None]:
    """Final answer of one greedy completion per question under one view (E12, E15).

    ``None`` marks a completion without a parseable final answer (or an overlong input).
    """
    prompts = _encode_prompts(codec, problems, wrapper, cfg)
    answers: list[str | None] = [None] * len(problems)
    for start in range(0, len(prompts), cfg.batch_size):
        batch_prompts = prompts[start : start + cfg.batch_size]
        allowance = _allowance(model, batch_prompts, cfg)
        if allowance <= 0:
            continue
        responses = model.generate(
            batch_prompts, max_new_tokens=allowance, temperature=0.0, top_p=1.0, seed=None
        )
        for offset, response in enumerate(responses):
            answers[start + offset] = extract_answer(codec.decode(response))
    return answers


def same_final_answer(first: str | None, second: str | None, rel_tol: float = 1e-9) -> bool:
    """Whether two extracted final answers agree (two missing answers count as the same)."""
    if first is None or second is None:
        return first is None and second is None
    return first == second or is_equivalent(first, second, rel_tol)


def _family_metrics(canonical: Completions, views: Sequence[Completions]) -> dict[str, float]:
    """MeanView / WorstView / WorstTemplate, sensitivity gaps and parse failures."""
    metrics = view_robustness(np.stack([view.correct for view in views]))
    metrics.update(
        sensitivity_gaps(avg_at_k(canonical.correct), metrics["mean_view"], metrics["worst_view"])
    )
    metrics["parse_failures"] = parse_failure_rate(np.stack([view.parsed for view in views]))
    return metrics


def evaluate(
    model: CausalLM,
    codec: PromptCodec,
    problems: Sequence[Problem],
    cfg: EvalConfig | None = None,
    *,
    unseen_wrappers: Sequence[View] = EVAL_WRAPPERS,
    extra_wrappers: Sequence[View] = (),
    families: Sequence[ViewFamily] = (),
    greedy: bool = False,
    flip_mode: str = "pair",
    keep_questions: bool = True,
) -> dict[str, Any]:
    """Evaluate one benchmark under the canonical prompt and the held-out views.

    Returns Avg@k and Pass@k for the canonical prompt (Equation 10), per-view Avg@k, and
    MeanView / WorstView / WorstTemplate over ``unseen_wrappers`` (Equations 11 and 12).
    ``extra_wrappers`` (e.g. the training wrappers) are reported per view only. Pass an empty
    ``unseen_wrappers`` to evaluate the canonical prompt alone.

    Supplementary experiments E12-E15:

    * ``families`` are further held-out view families. ``result["families"]`` holds, for the
      prefix family (``unseen_wrappers``) and each of them, the three view metrics,
      ``canonical_minus_mean``, ``mean_minus_worst`` and ``parse_failures`` (the fraction of
      sampled completions under the family's views without a parseable final answer).
    * ``greedy=True`` additionally decodes one greedy completion per question and view and
      records ``greedy_answer_flips``: the fraction of (question, view) pairs - or, with
      ``flip_mode="question"``, of questions - whose greedy final answer differs from the
      one under the canonical prompt.
    * ``keep_questions=True`` stores the question ids and, per view, the number of correct
      completions of every question, which is what the paired question bootstrap resamples.
    """
    cfg = cfg or EvalConfig()
    result: dict[str, Any] = {"num_questions": len(problems), "num_samples": cfg.num_samples}
    canonical = sample_completions(model, codec, problems, ORIGINAL_VIEW, cfg, view_index=0)
    result["avg_at_k"] = avg_at_k(canonical.correct)
    result["pass_at_k"] = pass_at_k(canonical.correct)
    result["parse_failures"] = parse_failure_rate(canonical.parsed)
    result["views"] = {ORIGINAL_VIEW.name: result["avg_at_k"]}
    counts = {ORIGINAL_VIEW.name: canonical.correct.sum(axis=1)}

    # Views in evaluation order: held-out prefixes, per-view extras, then further families.
    # The position in this list is the view index of the random stream, so adding families
    # leaves the completions of the earlier views unchanged.
    groups: list[tuple[str | None, list[str], list[View | list[View]]]] = []
    if unseen_wrappers:
        groups.append((PREFIX_FAMILY, [w.name for w in unseen_wrappers], list(unseen_wrappers)))
    if extra_wrappers:
        groups.append((None, [w.name for w in extra_wrappers], list(extra_wrappers)))
    for family in families:
        if family.per_question is None:
            per_view: list[View | list[View]] = list(family.views)
        else:  # view m of the family = the m-th view of every question
            per_question = [family.views_for(problem.id) for problem in problems]
            per_view = [list(views) for views in zip(*per_question, strict=True)]
        groups.append((family.name, list(family.view_names), per_view))

    canonical_greedy = greedy_answers(model, codec, problems, ORIGINAL_VIEW, cfg) if greedy else []
    view_index = 0
    for family_name, names, views in groups:
        completions: list[Completions] = []
        greedy_views: list[list[str | None]] = []
        for name, view in zip(names, views, strict=True):
            view_index += 1
            if name in result["views"]:
                raise ValueError(f"duplicate view name {name!r}")
            scored = sample_completions(model, codec, problems, view, cfg, view_index=view_index)
            result["views"][name] = avg_at_k(scored.correct)
            counts[name] = scored.correct.sum(axis=1)
            completions.append(scored)
            if greedy and family_name is not None:
                greedy_views.append(greedy_answers(model, codec, problems, view, cfg))
        if family_name is None:
            continue
        metrics = _family_metrics(canonical, completions)
        metrics["views"] = list(names)
        if greedy:
            metrics["greedy_answer_flips"] = greedy_answer_flip_rate(
                canonical_greedy,
                greedy_views,
                lambda a, b: same_final_answer(a, b, cfg.rel_tol),
                flip_mode,
            )
        result.setdefault("families", {})[family_name] = metrics
        if family_name == PREFIX_FAMILY:
            result["unseen"] = {
                key: metrics[key] for key in ("mean_view", "worst_view", "worst_template")
            }
    if keep_questions:
        result["questions"] = {
            "ids": [problem.id for problem in problems],
            "correct_counts": {name: [int(c) for c in values] for name, values in counts.items()},
        }
    return result
