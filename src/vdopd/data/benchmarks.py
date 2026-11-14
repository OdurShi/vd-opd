"""Evaluation benchmark loaders (Appendix B.2.1).

Each benchmark is read from ``data/benchmarks/<name>.jsonl`` when that file exists (see
``data/README.md``) and from the Hugging Face Hub otherwise. The Hub datasets are public
mirrors whose sizes match the counts stated in the paper (500 MATH500 questions, 40 AMC23
problems, 30 AIME problems per year, the English text-only open-ended mathematics subset of
OlympiadBench, the official GSM8K test split); their revisions are pinned.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass

from vdopd.data.paths import local_file, read_jsonl
from vdopd.data.problems import Problem
from vdopd.eval.answers import gsm8k_reference


@dataclass(frozen=True)
class BenchmarkSpec:
    """Where a benchmark lives on the Hugging Face Hub and how to read it."""

    path: str
    config: str | None
    split: str
    question_field: str
    answer_field: str
    id_field: str | None = None
    answer_format: str = "plain"  # "plain", "gsm8k" (text after ####) or "list" (first item)
    revision: str | None = None  # pinned Hub commit


BENCHMARKS: dict[str, BenchmarkSpec] = {
    "math500": BenchmarkSpec(
        "HuggingFaceH4/MATH-500",
        None,
        "test",
        "problem",
        "answer",
        "unique_id",
        revision="6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be",
    ),
    "amc23": BenchmarkSpec(
        "math-ai/amc23",
        None,
        "test",
        "question",
        "answer",
        "id",
        revision="80815d37005feb82cd7f8fbc6901d5d3eff43057",
    ),
    "aime24": BenchmarkSpec(
        "HuggingFaceH4/aime_2024",
        None,
        "train",
        "problem",
        "answer",
        "id",
        revision="2fe88a2f1091d5048c0f36abc874fb997b3dd99a",
    ),
    "aime25": BenchmarkSpec(
        "math-ai/aime25",
        None,
        "test",
        "problem",
        "answer",
        "id",
        revision="563bb8404243c5f09de6ec262f2db674fe5bce9b",
    ),
    "minerva": BenchmarkSpec(
        "math-ai/minervamath",
        None,
        "test",
        "question",
        "answer",
        revision="ee46ddc498933b1977577953250ca5c66be64f96",
    ),
    "olympiadbench": BenchmarkSpec(
        "Hothan/OlympiadBench",
        "OE_TO_maths_en_COMP",
        "train",
        "question",
        "final_answer",
        "id",
        answer_format="list",
        revision="91184b52131e7fc9455fef848035173aea8cc01a",
    ),
    "gsm8k": BenchmarkSpec(
        "openai/gsm8k",
        "main",
        "test",
        "question",
        "answer",
        answer_format="gsm8k",
        revision="740312add88f781978c0658806c59bc2815b9866",
    ),
}

# The six primary mathematics benchmarks whose unweighted mean is the "six-math" score.
SIX_MATH: tuple[str, ...] = ("math500", "amc23", "aime24", "aime25", "minerva", "olympiadbench")


def _reference(raw: object, answer_format: str) -> str:
    if answer_format == "gsm8k":
        return gsm8k_reference(str(raw))
    if answer_format == "list":
        items = ast.literal_eval(raw) if isinstance(raw, str) else list(raw)
        return str(items[0])
    return str(raw)


def records_to_problems(records: list[dict], spec: BenchmarkSpec, name: str) -> list[Problem]:
    """Convert raw dataset rows into :class:`Problem` records (pure, testable offline)."""
    problems = []
    for index, record in enumerate(records):
        identifier = record[spec.id_field] if spec.id_field else index
        problems.append(
            Problem(
                id=f"{name}/{identifier}",
                question=str(record[spec.question_field]),
                answer=_reference(record[spec.answer_field], spec.answer_format),
                subject=record.get("subject"),
                level=None if record.get("level") is None else str(record["level"]),
            )
        )
    return problems


def benchmark_file(name: str) -> str:
    """Path of a benchmark file relative to the data directory."""
    return f"benchmarks/{name}.jsonl"


def fetch_benchmark_records(name: str, revision: str | None = None) -> list[dict]:
    """Download the raw rows of a benchmark from the Hugging Face Hub (requires ``datasets``)."""
    from datasets import load_dataset

    spec = BENCHMARKS[name]
    dataset = load_dataset(
        spec.path, spec.config, split=spec.split, revision=revision or spec.revision
    )
    return [dict(row) for row in dataset]


def load_benchmark(
    name: str, limit: int | None = None, revision: str | None = None
) -> list[Problem]:
    """Load a benchmark by name: the shipped file if present, otherwise the Hub."""
    if name not in BENCHMARKS:
        raise KeyError(f"unknown benchmark {name!r}; available: {sorted(BENCHMARKS)}")
    path = local_file(benchmark_file(name))
    records = read_jsonl(path) if path is not None else fetch_benchmark_records(name, revision)
    problems = records_to_problems(records, BENCHMARKS[name], name)
    return problems[:limit] if limit is not None else problems
