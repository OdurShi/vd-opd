"""CPU quick start: the complete VD-OPD loop on a synthetic arithmetic task.

Tiny teacher and student transformers are built in code with random weights. Because a
randomly initialised teacher carries no knowledge, the demo first fits the teacher (and,
briefly, the initial student) to the synthetic task with ordinary supervised next-token
training. That pretraining only manufactures a toy teacher/student pair; it is *not* part of
the method. The distillation that follows is exactly Algorithm 1: on-policy rollouts,
two-view teacher scoring, the consensus loss, one update per rollout batch, and evaluation
under the canonical prompt and the held-out wrappers.

Numbers printed by this demo describe a toy problem and say nothing about the paper's
experiments.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from vdopd.chat import SimpleChatCodec
from vdopd.data.problems import Problem, build_messages
from vdopd.data.synthetic import (
    SYNTHETIC_TEMPLATE,
    make_arithmetic_problems,
    reference_response,
    split_problems,
)
from vdopd.eval.evaluator import EvalConfig, evaluate
from vdopd.models import CausalLM, build_tiny_causal_lm
from vdopd.objectives import ObjectiveConfig, build_objective
from vdopd.optim import OptimConfig
from vdopd.scoring import pack_prompt_and_response, response_hidden_states
from vdopd.trainer import RolloutConfig, TrainConfig, Trainer
from vdopd.views import EVAL_WRAPPERS, ORIGINAL_VIEW, TRAIN_WRAPPERS, Wrapper, apply_wrapper

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DemoConfig:
    """Settings of the synthetic quick start (chosen for speed, not taken from the paper)."""

    seed: int = 11
    objective: str = "vdopd"
    max_operand: int = 9
    teacher_hidden: int = 96
    teacher_layers: int = 2
    student_hidden: int = 48
    student_layers: int = 2
    teacher_pretrain_steps: int = 1200
    student_pretrain_steps: int = 150
    pretrain_batch_size: int = 32
    pretrain_learning_rate: float = 3e-3
    iterations: int = 300
    rollout_batch_size: int = 32
    max_response_tokens: int = 16
    learning_rate: float = 1e-3
    eval_questions: int = 40
    eval_samples: int = 4


def supervised_pretrain(
    model: CausalLM,
    codec: SimpleChatCodec,
    problems: Sequence[Problem],
    *,
    steps: int,
    batch_size: int,
    learning_rate: float,
    wrappers: Sequence[Wrapper],
    seed: int,
) -> float:
    """Fit a toy model to reference responses with next-token cross-entropy.

    Demo scaffolding only (see the module docstring). Each example uses a random view from
    ``wrappers`` so that the toy teacher has seen differently worded prompts. Returns the
    final training loss.
    """
    rng = np.random.default_rng(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0)
    loss_value = float("nan")
    for _ in range(steps):
        batch = [problems[i] for i in rng.integers(len(problems), size=batch_size)]
        prompts, responses = [], []
        for problem in batch:
            wrapper = wrappers[int(rng.integers(len(wrappers)))]
            messages = apply_wrapper(build_messages(problem.question, SYNTHETIC_TEMPLATE), wrapper)
            prompts.append(codec.encode(messages))
            responses.append(codec.encode_response(reference_response(problem)))
        packed = pack_prompt_and_response(prompts, responses, model.pad_token_id)
        logits = model.project(response_hidden_states(model, packed)).float()
        loss = F.cross_entropy(logits, packed.tokens.to(logits.device))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        loss_value = loss.item()
    return loss_value


def _summary(result: dict[str, Any]) -> dict[str, float]:
    return {
        "avg_at_k": result["avg_at_k"],
        "pass_at_k": result["pass_at_k"],
        "mean_view": result["unseen"]["mean_view"],
        "worst_view": result["unseen"]["worst_view"],
        "worst_template": result["unseen"]["worst_template"],
    }


@dataclass
class ToyPair:
    """A toy teacher/student pair with its codec, problems and evaluation settings."""

    teacher: CausalLM
    student: CausalLM
    codec: SimpleChatCodec
    train_problems: list[Problem]
    test_problems: list[Problem]
    eval_cfg: EvalConfig


def build_toy_pair(cfg: DemoConfig | None = None) -> ToyPair:
    """Build and pretrain the toy teacher and initial student (demo scaffolding)."""
    cfg = cfg or DemoConfig()
    codec = SimpleChatCodec()
    problems = make_arithmetic_problems(cfg.max_operand)
    train_problems, test_problems = split_problems(problems, test_fraction=0.2, seed=cfg.seed)

    def tiny(hidden: int, layers: int, seed: int, name: str) -> CausalLM:
        return build_tiny_causal_lm(
            len(codec.tokenizer),
            hidden_size=hidden,
            num_layers=layers,
            eos_token_id=codec.tokenizer.eos_token_id,
            pad_token_id=codec.tokenizer.pad_token_id,
            seed=seed,
            name=name,
        )

    teacher = tiny(cfg.teacher_hidden, cfg.teacher_layers, cfg.seed, "toy-teacher")
    student = tiny(cfg.student_hidden, cfg.student_layers, cfg.seed + 1, "toy-student")
    all_views = (ORIGINAL_VIEW, *TRAIN_WRAPPERS, *EVAL_WRAPPERS)
    logger.info("supervised pretraining of the toy teacher (%d steps)", cfg.teacher_pretrain_steps)
    supervised_pretrain(
        teacher,
        codec,
        problems,  # the toy teacher is fitted to every problem, like a knowledgeable teacher
        steps=cfg.teacher_pretrain_steps,
        batch_size=cfg.pretrain_batch_size,
        learning_rate=cfg.pretrain_learning_rate,
        wrappers=all_views,
        seed=cfg.seed,
    )
    logger.info("brief pretraining of the initial student (%d steps)", cfg.student_pretrain_steps)
    supervised_pretrain(
        student,
        codec,
        train_problems,
        steps=cfg.student_pretrain_steps,
        batch_size=cfg.pretrain_batch_size,
        learning_rate=cfg.pretrain_learning_rate,
        wrappers=all_views,
        seed=cfg.seed + 1,
    )
    eval_cfg = EvalConfig(
        num_samples=cfg.eval_samples,
        max_response_tokens=cfg.max_response_tokens,
        batch_size=128,
        seed=cfg.seed,
        prompt_template=SYNTHETIC_TEMPLATE,
    )
    return ToyPair(
        teacher, student, codec, train_problems, test_problems[: cfg.eval_questions], eval_cfg
    )


def build_toy_trainer(pair: ToyPair, cfg: DemoConfig, prompts: Sequence[Problem]) -> Trainer:
    """Trainer for the toy pair with the demo's rollout and optimisation settings."""
    return Trainer(
        student=pair.student,
        teacher=pair.teacher,
        codec=pair.codec,
        prompts=[build_messages(p.question, SYNTHETIC_TEMPLATE) for p in prompts],
        objective=build_objective(ObjectiveConfig(name=cfg.objective)),
        rollout=RolloutConfig(
            batch_size=cfg.rollout_batch_size,
            max_response_tokens=cfg.max_response_tokens,
            generation_batch_size=cfg.rollout_batch_size,
        ),
        optim=OptimConfig(learning_rate=cfg.learning_rate),
        train=TrainConfig(
            seed=cfg.seed, iterations=cfg.iterations, microbatch_size=16, log_every=25
        ),
    )


def run_demo(cfg: DemoConfig | None = None) -> dict[str, Any]:
    """Run pretraining of the toy pair, on-policy distillation and evaluation."""
    cfg = cfg or DemoConfig()
    started = time.time()
    logger.info("[1/3] building the toy teacher and initial student")
    pair = build_toy_pair(cfg)
    results: dict[str, Any] = {
        "teacher": _summary(evaluate(pair.teacher, pair.codec, pair.test_problems, pair.eval_cfg)),
        "initial_student": _summary(
            evaluate(pair.student, pair.codec, pair.test_problems, pair.eval_cfg)
        ),
    }

    logger.info(
        "[2/3] on-policy distillation with objective=%s (%d iterations)",
        cfg.objective,
        cfg.iterations,
    )
    trainer = build_toy_trainer(pair, cfg, pair.train_problems)
    history = trainer.train()

    logger.info("[3/3] evaluation of the distilled student")
    results["distilled_student"] = _summary(
        evaluate(pair.student, pair.codec, pair.test_problems, pair.eval_cfg)
    )
    masses = [h["mass_mean"] for h in history if "mass_mean" in h]
    results["training"] = {
        "objective": cfg.objective,
        "iterations": len(history),
        "first_loss": history[0].get("loss"),
        "last_loss": history[-1].get("loss"),
        "mean_mass": float(np.mean(masses)) if masses else None,
        "total_flops": trainer.flops.total,
        "teacher_scoring_flops": trainer.flops.teacher_scoring,
    }
    results["seconds"] = time.time() - started
    return results


def format_results(results: dict[str, Any], num_samples: int) -> str:
    """Render the demo metrics as a small text table (percent)."""
    header = (
        f"{'model':<20}{'Avg@' + str(num_samples):>8}{'Pass@' + str(num_samples):>9}"
        f"{'MeanView':>10}{'WorstView':>11}{'WorstTmpl':>11}"
    )
    lines = [header, "-" * len(header)]
    for key, label in (
        ("teacher", "toy teacher"),
        ("initial_student", "initial student"),
        ("distilled_student", "distilled student"),
    ):
        row = results[key]
        lines.append(
            f"{label:<20}{100 * row['avg_at_k']:>8.1f}{100 * row['pass_at_k']:>9.1f}"
            f"{100 * row['mean_view']:>10.1f}{100 * row['worst_view']:>11.1f}"
            f"{100 * row['worst_template']:>11.1f}"
        )
    return "\n".join(lines)
