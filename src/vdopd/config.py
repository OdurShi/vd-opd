"""Experiment configuration: dataclasses loaded from YAML.

A YAML file may name a parent with ``_base_: relative/path.yaml``; the child is merged over
the parent. Command-line overrides use dotted keys, e.g. ``train.seed=22``. Unknown keys are
rejected so that a typo cannot silently fall back to a default.
"""

from __future__ import annotations

import dataclasses
import types
import typing
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from vdopd.data.benchmarks import SIX_MATH
from vdopd.data.problems import DEFAULT_PROMPT_TEMPLATE
from vdopd.eval.evaluator import EvalConfig
from vdopd.objectives import ObjectiveConfig
from vdopd.optim import OptimConfig
from vdopd.trainer import RolloutConfig, TrainConfig


@dataclass(frozen=True)
class ModelConfig:
    """Teacher/student pair and their shared serialisation (Section 5.1, Appendix B.4)."""

    teacher: str = "Qwen/Qwen3-8B"
    student: str = "Qwen/Qwen3-1.7B-Base"
    teacher_revision: str | None = None
    student_revision: str | None = None
    dtype: str = "bfloat16"
    device: str = "auto"
    teacher_device: str | None = None
    gradient_checkpointing: bool = True
    chat_template_kwargs: dict = field(default_factory=lambda: {"enable_thinking": False})
    stop_tokens: tuple[str, ...] = ("<|im_end|>",)
    teacher_context: int | None = None
    student_context: int | None = None


@dataclass(frozen=True)
class DataConfig:
    """Training prompts (Appendix B.1)."""

    source: str = "math_train"
    data_dir: str | None = None
    split_seed: int = 42
    num_diagnostic: int = 500
    num_development: int = 500
    remove_overlap_with: tuple[str, ...] = SIX_MATH
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE
    system_prompt: str | None = None
    exemplars: tuple[tuple[str, str], ...] = ()
    limit: int | None = None
    synthetic_max_operand: int = 19
    synthetic_test_fraction: float = 0.2


@dataclass(frozen=True)
class EvaluationConfig:
    """Benchmarks, views and decoding protocol (Appendix B.2)."""

    benchmarks: tuple[str, ...] = SIX_MATH
    limit: int | None = None
    unseen_wrappers: bool = True
    seen_wrappers: bool = False
    decoding: EvalConfig = field(default_factory=EvalConfig)


@dataclass(frozen=True)
class ExperimentConfig:
    """Complete description of one training/evaluation setting."""

    name: str = "vdopd"
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    rollout: RolloutConfig = field(default_factory=RolloutConfig)
    objective: ObjectiveConfig = field(default_factory=ObjectiveConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)


def _coerce(value: Any, hint: Any, where: str) -> Any:
    """Convert a YAML value to the annotated field type (nested dataclasses, tuples, floats)."""
    origin = typing.get_origin(hint)
    if origin in (typing.Union, types.UnionType):
        if value is None:
            return None
        options = [arg for arg in typing.get_args(hint) if arg is not type(None)]
        return _coerce(value, options[0], where)
    if dataclasses.is_dataclass(hint):
        if not isinstance(value, dict):
            raise TypeError(f"section {where!r} must be a mapping")
        return _build(hint, value, where)
    if origin is tuple:
        return tuple(tuple(v) if isinstance(v, list) else v for v in value)
    if hint is float and isinstance(value, (int, str)) and not isinstance(value, bool):
        return float(value)  # PyYAML reads "3e-6" as a string
    if hint is int and isinstance(value, (str, float)):
        if float(value) != int(float(value)):
            raise TypeError(f"{where!r} must be an integer, got {value!r}")
        return int(float(value))
    return value


def _build(cls: type, data: dict[str, Any], where: str) -> Any:
    hints = typing.get_type_hints(cls)
    unknown = sorted(set(data) - set(hints))
    if unknown:
        raise KeyError(f"unknown key(s) {unknown} in section {where!r}")
    return cls(**{key: _coerce(value, hints[key], f"{where}.{key}") for key, value in data.items()})


def _merge(base: dict[str, Any], child: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in child.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _read(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    parent = data.pop("_base_", None)
    if parent is not None:
        data = _merge(_read((path.parent / parent).resolve()), data)
    return data


def apply_overrides(data: dict[str, Any], overrides: Sequence[str]) -> dict[str, Any]:
    """Apply ``section.key=value`` overrides; values are parsed as YAML scalars."""
    for override in overrides:
        dotted, separator, raw = override.partition("=")
        if not separator:
            raise ValueError(f"override {override!r} is not of the form key=value")
        node = data
        *parents, leaf = dotted.split(".")
        for key in parents:
            node = node.setdefault(key, {})
        node[leaf] = yaml.safe_load(raw)
    return data


def config_from_dict(data: dict[str, Any]) -> ExperimentConfig:
    """Build an :class:`ExperimentConfig` from a nested dictionary."""
    return _build(ExperimentConfig, data, "config")


def load_config(path: str | Path, overrides: Sequence[str] = ()) -> ExperimentConfig:
    """Load a YAML experiment file, resolving ``_base_`` parents and overrides."""
    data = apply_overrides(_read(Path(path).resolve()), overrides)
    return config_from_dict(data)


def config_to_dict(config: ExperimentConfig) -> dict[str, Any]:
    """Plain nested dictionary of a configuration (for the run manifest)."""
    return dataclasses.asdict(config)
