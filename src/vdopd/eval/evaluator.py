"""Sampled evaluation over prompt views (Appendix B.2).

Mathematical evaluations use zero-shot prompting and eight independent completions per
question at temperature 1.0 with top-p 0.8 and an 8,192-token response cap. The student is
evaluated independently on every view, without teacher assistance or answer ensembling.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from vdopd.data.problems import DEFAULT_PROMPT_TEMPLATE, Problem, build_messages
from vdopd.eval.answers import extract_answer, is_equivalent
from vdopd.eval.metrics import avg_at_k, pass_at_k, view_robustness
from vdopd.models import CausalLM
from vdopd.seeding import derive_seed
from vdopd.views import EVAL_WRAPPERS, ORIGINAL_VIEW, PromptCodec, Wrapper, apply_wrapper

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


def sample_correctness(
    model: CausalLM,
    codec: PromptCodec,
    problems: Sequence[Problem],
    wrapper: Wrapper,
    cfg: EvalConfig,
    view_index: int = 0,
) -> np.ndarray:
    """Generate ``k`` completions per question under one view and score them.

    Returns a boolean ``[n, k]`` array. Random seeds depend only on the evaluation seed, the
    view and the batch, so different checkpoints are compared on paired random streams.
    """
    prompts: list[list[int]] = []
    for problem in problems:
        messages = build_messages(
            problem.question, cfg.prompt_template, cfg.system_prompt, cfg.exemplars
        )
        prompts.append(codec.encode(apply_wrapper(messages, wrapper)))

    correct = np.zeros((len(problems), cfg.num_samples), dtype=bool)
    jobs = [(q, s) for q in range(len(problems)) for s in range(cfg.num_samples)]
    for batch_index, start in enumerate(range(0, len(jobs), cfg.batch_size)):
        batch = jobs[start : start + cfg.batch_size]
        batch_prompts = [prompts[q] for q, _ in batch]
        allowance = cfg.max_response_tokens
        if model.max_context is not None:
            allowance = min(allowance, model.max_context - max(len(p) for p in batch_prompts))
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
            reference = problems[q].answer
            if reference is None:
                raise ValueError(f"problem {problems[q].id} has no reference answer")
            text = codec.decode(response)
            correct[q, s] = is_equivalent(extract_answer(text), reference, cfg.rel_tol)
    return correct


def evaluate(
    model: CausalLM,
    codec: PromptCodec,
    problems: Sequence[Problem],
    cfg: EvalConfig | None = None,
    *,
    unseen_wrappers: Sequence[Wrapper] = EVAL_WRAPPERS,
    extra_wrappers: Sequence[Wrapper] = (),
) -> dict[str, Any]:
    """Evaluate one benchmark under the canonical prompt and the held-out wrappers.

    Returns Avg@k and Pass@k for the canonical prompt (Equation 10), per-view Avg@k, and
    MeanView / WorstView / WorstTemplate over ``unseen_wrappers`` (Equations 11 and 12).
    ``extra_wrappers`` (e.g. the training wrappers) are reported per view only. Pass an empty
    ``unseen_wrappers`` to evaluate the canonical prompt alone.
    """
    cfg = cfg or EvalConfig()
    result: dict[str, Any] = {"num_questions": len(problems), "num_samples": cfg.num_samples}
    canonical = sample_correctness(model, codec, problems, ORIGINAL_VIEW, cfg, view_index=0)
    result["avg_at_k"] = avg_at_k(canonical)
    result["pass_at_k"] = pass_at_k(canonical)
    result["views"] = {ORIGINAL_VIEW.name: result["avg_at_k"]}

    unseen: list[np.ndarray] = []
    for index, wrapper in enumerate([*unseen_wrappers, *extra_wrappers], start=1):
        correct = sample_correctness(model, codec, problems, wrapper, cfg, view_index=index)
        result["views"][wrapper.name] = avg_at_k(correct)
        if index <= len(unseen_wrappers):
            unseen.append(correct)
    if unseen:
        result["unseen"] = view_robustness(np.stack(unseen))
    return result
