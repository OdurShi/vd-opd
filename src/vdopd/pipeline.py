"""Glue between configuration files and the library: build models, data and runs."""

from __future__ import annotations

import dataclasses
import json
import logging
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import transformers

from vdopd import __version__
from vdopd.chat import HFChatCodec
from vdopd.config import ExperimentConfig, config_to_dict
from vdopd.data.benchmarks import BENCHMARKS, load_benchmark
from vdopd.data.math_train import prepare_diagnostic_split, prepare_math_splits, split_diagnostic
from vdopd.data.paths import set_data_dir
from vdopd.data.problems import Problem, build_messages, load_jsonl
from vdopd.data.synthetic import make_arithmetic_problems, split_problems
from vdopd.eval.evaluator import EvalConfig, evaluate
from vdopd.eval.metrics import macro_average
from vdopd.models import CausalLM, check_pair_compatibility, load_causal_lm, load_tokenizer
from vdopd.objectives import ObjectiveConfig, build_objective
from vdopd.optim import OptimConfig
from vdopd.trainer import RolloutConfig, TrainConfig, Trainer
from vdopd.views import EVAL_WRAPPERS, TRAIN_WRAPPERS, Message, PromptCodec

logger = logging.getLogger(__name__)

SYNTHETIC_SOURCE = "synthetic_arithmetic"


def _synthetic_split(cfg: ExperimentConfig) -> tuple[list[Problem], list[Problem]]:
    problems = make_arithmetic_problems(cfg.data.synthetic_max_operand)
    return split_problems(problems, cfg.data.synthetic_test_fraction, cfg.data.split_seed)


def use_configured_data_dir(cfg: ExperimentConfig) -> None:
    """Apply ``data.data_dir`` (if set) as the directory of the shipped dataset files."""
    if cfg.data.data_dir is not None:
        set_data_dir(cfg.data.data_dir)


def load_training_problems(cfg: ExperimentConfig) -> list[Problem]:
    """Optimisation problems named by ``data.source`` (Appendix B.1)."""
    use_configured_data_dir(cfg)
    data = cfg.data
    if data.source == "math_train":
        problems = prepare_math_splits(
            data.remove_overlap_with,
            num_diagnostic=data.num_diagnostic,
            num_development=data.num_development,
            seed=data.split_seed,
        ).optimization
    elif data.source == SYNTHETIC_SOURCE:
        problems = _synthetic_split(cfg)[0]
    else:
        problems = load_jsonl(data.source)
    return problems[: data.limit] if data.limit is not None else problems


def load_evaluation_problems(name: str, cfg: ExperimentConfig) -> list[Problem]:
    """Evaluation problems: a registered benchmark, the synthetic test split or a JSONL file."""
    use_configured_data_dir(cfg)
    limit = cfg.evaluation.limit
    if name in BENCHMARKS:
        return load_benchmark(name, limit=limit)
    if name == SYNTHETIC_SOURCE:
        problems = _synthetic_split(cfg)[1]
    else:
        problems = load_jsonl(name)
    return problems[:limit] if limit is not None else problems


def training_messages(cfg: ExperimentConfig, problems: list[Problem]) -> list[list[Message]]:
    """Original-view chat messages; only the question text is used (no answers)."""
    return [
        build_messages(
            p.question, cfg.data.prompt_template, cfg.data.system_prompt, cfg.data.exemplars
        )
        for p in problems
    ]


def build_models(
    cfg: ExperimentConfig, *, student_path: str | None = None, load_teacher: bool = True
) -> tuple[CausalLM | None, CausalLM, HFChatCodec]:
    """Load the frozen teacher, the student and the shared chat codec.

    With ``load_teacher=False`` (evaluation) the student is loaded without gradients.
    """
    model = cfg.model
    tokenizer = load_tokenizer(model.teacher, model.teacher_revision)
    codec = HFChatCodec(tokenizer, model.chat_template_kwargs)
    student = load_causal_lm(
        student_path or model.student,
        tokenizer=tokenizer,
        stop_tokens=model.stop_tokens,
        dtype=model.dtype,
        device=model.device,
        revision=None if student_path else model.student_revision,
        max_context=model.student_context,
        gradient_checkpointing=model.gradient_checkpointing and load_teacher,
        trainable=load_teacher,
    )
    teacher = None
    if load_teacher:
        teacher = load_causal_lm(
            model.teacher,
            tokenizer=tokenizer,
            stop_tokens=model.stop_tokens,
            dtype=model.dtype,
            device=model.teacher_device or model.device,
            revision=model.teacher_revision,
            max_context=model.teacher_context,
        )
        student_tokenizer = load_tokenizer(model.student, model.student_revision)
        check_pair_compatibility(teacher, student, tokenizer, student_tokenizer)
    return teacher, student, codec


def write_manifest(cfg: ExperimentConfig, output_dir: Path) -> None:
    """Record the resolved configuration and software versions (Appendix B.4)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "config": config_to_dict(cfg),
        "versions": {
            "vdopd": __version__,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "python": platform.python_version(),
        },
        "train_wrappers": [w.prefix for w in TRAIN_WRAPPERS],
        "eval_wrappers": [w.prefix for w in EVAL_WRAPPERS],
    }
    with open(output_dir / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)


def run_training(cfg: ExperimentConfig) -> Trainer:
    """Train a student as described by ``cfg`` and save it to ``train.output_dir``."""
    if cfg.train.output_dir is None:
        raise ValueError("train.output_dir must be set")
    output_dir = Path(cfg.train.output_dir)
    write_manifest(cfg, output_dir)
    teacher, student, codec = build_models(cfg)
    problems = load_training_problems(cfg)
    logger.info("optimisation prompts: %d", len(problems))
    trainer = Trainer(
        student=student,
        teacher=teacher,
        codec=codec,
        prompts=training_messages(cfg, problems),
        objective=build_objective(cfg.objective),
        rollout=cfg.rollout,
        optim=cfg.optim,
        train=cfg.train,
    )
    trainer.train()
    trainer.save_checkpoint(output_dir / "final")
    return trainer


def evaluate_student(
    cfg: ExperimentConfig, student: CausalLM, codec: HFChatCodec
) -> dict[str, Any]:
    """Evaluate a student on the configured benchmarks; adds the macro-average."""
    section = cfg.evaluation
    # Prompt wording is identical for training and evaluation (data section).
    decoding = dataclasses.replace(
        section.decoding,
        prompt_template=cfg.data.prompt_template,
        system_prompt=cfg.data.system_prompt,
        exemplars=cfg.data.exemplars,
    )
    results: dict[str, Any] = {"benchmarks": {}}
    for name in section.benchmarks:
        problems = load_evaluation_problems(name, cfg)
        results["benchmarks"][name] = evaluate(
            student,
            codec,
            problems,
            decoding,
            unseen_wrappers=EVAL_WRAPPERS if section.unseen_wrappers else (),
            extra_wrappers=TRAIN_WRAPPERS if section.seen_wrappers else (),
        )
        logger.info("%s: Avg@k=%.4f", name, results["benchmarks"][name]["avg_at_k"])
    per_benchmark = list(results["benchmarks"].values())
    results["macro"] = {
        "avg_at_k": macro_average([r["avg_at_k"] for r in per_benchmark]),
        "pass_at_k": macro_average([r["pass_at_k"] for r in per_benchmark]),
    }
    if section.unseen_wrappers:
        for key in ("mean_view", "worst_view", "worst_template"):
            results["macro"][key] = macro_average([r["unseen"][key] for r in per_benchmark])
    return results


def run_evaluation(
    cfg: ExperimentConfig, student_path: str | None = None, output: str | None = None
) -> dict[str, Any]:
    """Evaluate the student at ``student_path`` (default: the initial student)."""
    _, student, codec = build_models(cfg, student_path=student_path, load_teacher=False)
    results = evaluate_student(cfg, student, codec)
    results["student"] = student_path or cfg.model.student
    results["seed"] = cfg.evaluation.decoding.seed
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)
    return results


@dataclass
class DiagnosticSetting:
    """Models and problem sets of the mechanism diagnostics (Appendix B.7)."""

    teacher: CausalLM
    student: CausalLM
    codec: PromptCodec
    update_problems: list[Problem]
    measurement_problems: list[Problem]
    prompt_template: str
    rollout: RolloutConfig
    optim: OptimConfig
    train: TrainConfig
    decoding: EvalConfig


def diagnostic_setting_from_config(
    cfg: ExperimentConfig, student_path: str | None = None
) -> DiagnosticSetting:
    """Load a teacher/student pair and the reserved diagnostic problems of a config.

    For ``data.source = math_train`` the 500 reserved diagnostic problems are split by
    source-problem id into 250 update and 250 measurement problems with the split seed.
    """
    use_configured_data_dir(cfg)
    teacher, student, codec = build_models(cfg, student_path=student_path)
    if cfg.data.source == "math_train":
        splits = prepare_math_splits(
            cfg.data.remove_overlap_with,
            num_diagnostic=cfg.data.num_diagnostic,
            num_development=cfg.data.num_development,
            seed=cfg.data.split_seed,
        )
        update, measurement = prepare_diagnostic_split(splits.diagnostic, seed=cfg.data.split_seed)
    elif cfg.data.source == SYNTHETIC_SOURCE:
        update, measurement = _synthetic_split(cfg)
    else:
        problems = load_jsonl(cfg.data.source)
        update, measurement = split_diagnostic(
            problems, num_update=len(problems) // 2, seed=cfg.data.split_seed
        )
    return DiagnosticSetting(
        teacher=teacher,
        student=student,
        codec=codec,
        update_problems=update,
        measurement_problems=measurement,
        prompt_template=cfg.data.prompt_template,
        rollout=cfg.rollout,
        optim=cfg.optim,
        train=cfg.train,
        decoding=dataclasses.replace(
            cfg.evaluation.decoding,
            prompt_template=cfg.data.prompt_template,
            system_prompt=cfg.data.system_prompt,
            exemplars=cfg.data.exemplars,
        ),
    )


def toy_diagnostic_setting(seed: int = 11) -> DiagnosticSetting:
    """The synthetic toy pair of the quick start, for running the diagnostics on a laptop."""
    from vdopd.demo import DemoConfig, build_toy_pair, build_toy_trainer

    demo_cfg = DemoConfig(seed=seed)
    pair = build_toy_pair(demo_cfg)
    trainer = build_toy_trainer(pair, demo_cfg, pair.train_problems)
    return DiagnosticSetting(
        teacher=pair.teacher,
        student=pair.student,
        codec=pair.codec,
        update_problems=pair.train_problems,
        measurement_problems=pair.test_problems,
        prompt_template=pair.eval_cfg.prompt_template,
        rollout=trainer.rollout_cfg,
        optim=trainer.optim_cfg,
        train=trainer.cfg,
        decoding=pair.eval_cfg,
    )


def diagnostic_trainer(setting: DiagnosticSetting, objective_name: str = "vdopd") -> Trainer:
    """Trainer over the update problems of a diagnostic setting."""
    return Trainer(
        student=setting.student,
        teacher=setting.teacher,
        codec=setting.codec,
        prompts=[
            build_messages(
                p.question,
                setting.prompt_template,
                setting.decoding.system_prompt,
                setting.decoding.exemplars,
            )
            for p in setting.update_problems
        ],
        objective=build_objective(ObjectiveConfig(name=objective_name)),
        rollout=setting.rollout,
        optim=setting.optim,
        train=dataclasses.replace(setting.train, output_dir=None, checkpoint_every=0),
    )
