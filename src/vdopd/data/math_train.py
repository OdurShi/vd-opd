"""MATH training split and its partition (Appendix B.1 and B.7).

After deduplication and removal of detected overlaps with evaluation questions, 500
problems are reserved for implementation and mechanism diagnostics and 500 for development;
the remaining problems form the optimisation set. Splits are stratified by subject and
difficulty with split seed 42. The diagnostic problems are further divided by
source-problem id into 250 update and 250 measurement problems with the same seed.

The training problems are read from ``data/math_train/train.jsonl`` and the partition from
the index files in ``data/math_train/splits/`` when they exist (see ``data/README.md``);
otherwise the problems are downloaded from the Hugging Face Hub and the partition is
recomputed, which gives the same result.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from vdopd.data.paths import (
    data_dir,
    local_file,
    read_jsonl,
    read_lines,
    write_lines,
)
from vdopd.data.problems import Problem
from vdopd.eval.answers import extract_answer

MATH_SUBJECTS: tuple[str, ...] = (
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
)
MATH_TRAIN_PATH = "EleutherAI/hendrycks_math"
MATH_TRAIN_REVISION = "21a5633873b6a120296cce3e2df9d5550074f4a3"  # pinned Hub commit
MATH_TRAIN_FILE = "math_train/train.jsonl"
SPLIT_DIR = "math_train/splits"
SPLIT_NAMES = ("optimization", "development", "diagnostic")


def question_key(question: str) -> str:
    """Normalised text used to detect duplicate and overlapping questions."""
    return re.sub(r"\s+", " ", question).strip().lower()


def deduplicate(problems: Iterable[Problem]) -> list[Problem]:
    """Keep the first occurrence of every distinct (normalised) question text."""
    seen: set[str] = set()
    unique: list[Problem] = []
    for problem in problems:
        key = question_key(problem.question)
        if key not in seen:
            seen.add(key)
            unique.append(problem)
    return unique


def remove_overlaps(
    problems: Iterable[Problem], evaluation_questions: Iterable[str]
) -> list[Problem]:
    """Drop training problems whose normalised text equals an evaluation question."""
    blocked = {question_key(q) for q in evaluation_questions}
    return [p for p in problems if question_key(p.question) not in blocked]


@dataclass(frozen=True)
class MathSplits:
    """Partition of the cleaned MATH training problems."""

    optimization: list[Problem]
    development: list[Problem]
    diagnostic: list[Problem]


def _allocate(sizes: Sequence[int], total: int) -> list[int]:
    """Proportional allocation with the largest-remainder method (deterministic ties)."""
    population = sum(sizes)
    if total > population:
        raise ValueError(f"cannot reserve {total} problems out of {population}")
    exact = [size * total / population for size in sizes]
    counts = [int(np.floor(value)) for value in exact]
    remainder_order = sorted(range(len(sizes)), key=lambda i: (-(exact[i] - counts[i]), i))
    for index in remainder_order[: total - sum(counts)]:
        counts[index] += 1
    return counts


def stratified_split(
    problems: Sequence[Problem],
    *,
    num_diagnostic: int = 500,
    num_development: int = 500,
    seed: int = 42,
) -> MathSplits:
    """Reserve diagnostic and development problems, stratified by subject and difficulty.

    Each (subject, level) stratum contributes to the two reserved sets in proportion to its
    size; within a stratum the problems are shuffled with a generator derived from ``seed``.
    Every problem lands in exactly one split, so all views of a source problem share a split.
    """
    strata: dict[tuple[str, str], list[Problem]] = defaultdict(list)
    for problem in problems:
        strata[(problem.subject or "", problem.level or "")].append(problem)
    keys = sorted(strata)
    sizes = [len(strata[key]) for key in keys]
    diagnostic_counts = _allocate(sizes, num_diagnostic)
    remaining = [size - count for size, count in zip(sizes, diagnostic_counts, strict=True)]
    development_counts = _allocate(remaining, num_development)

    rng = np.random.default_rng(seed)
    splits = MathSplits([], [], [])
    for key, n_diag, n_dev in zip(keys, diagnostic_counts, development_counts, strict=True):
        members = sorted(strata[key], key=lambda p: p.id)
        order = rng.permutation(len(members))
        shuffled = [members[i] for i in order]
        splits.diagnostic.extend(shuffled[:n_diag])
        splits.development.extend(shuffled[n_diag : n_diag + n_dev])
        splits.optimization.extend(shuffled[n_diag + n_dev :])
    return splits


def split_diagnostic(
    problems: Sequence[Problem], *, num_update: int = 250, seed: int = 42
) -> tuple[list[Problem], list[Problem]]:
    """Split the diagnostic problems by source id into update and measurement sets (B.7)."""
    ordered = sorted(problems, key=lambda p: p.id)
    order = np.random.default_rng(seed).permutation(len(ordered))
    update = [ordered[i] for i in order[:num_update]]
    measurement = [ordered[i] for i in order[num_update:]]
    return update, measurement


def fetch_math_train_records(revision: str | None = None) -> list[dict]:
    """Download the official MATH training split from the Hub (requires ``datasets``).

    Every row keeps its original fields (``problem``, ``level``, ``type``, ``solution``) and
    receives a stable ``id`` of the form ``math_train/<subject>/<row index>``.
    """
    from datasets import load_dataset

    records: list[dict] = []
    for subject in MATH_SUBJECTS:
        dataset = load_dataset(
            MATH_TRAIN_PATH, subject, split="train", revision=revision or MATH_TRAIN_REVISION
        )
        for index, row in enumerate(dataset):
            records.append({"id": f"math_train/{subject}/{index}", **dict(row)})
    return records


def records_to_training_problems(records: Sequence[dict]) -> list[Problem]:
    """Convert MATH rows to problems.

    Reference answers (the last boxed expression of each solution) are attached for
    development scoring only; the distillation update uses the question text alone.
    """
    return [
        Problem(
            id=record["id"],
            question=record["problem"],
            answer=extract_answer(record["solution"]),
            subject=record["type"],
            level=record["level"],
        )
        for record in records
    ]


def load_math_train(revision: str | None = None) -> list[Problem]:
    """Load the MATH training split: the shipped file if present, otherwise the Hub."""
    path = local_file(MATH_TRAIN_FILE)
    records = read_jsonl(path) if path is not None else fetch_math_train_records(revision)
    return records_to_training_problems(records)


def split_parameters(
    num_diagnostic: int, num_development: int, seed: int, overlap_benchmarks: Sequence[str]
) -> dict:
    """Parameters that identify a stored partition (written to ``split_info.json``)."""
    return {
        "seed": seed,
        "num_diagnostic": num_diagnostic,
        "num_development": num_development,
        "overlap_benchmarks": sorted(overlap_benchmarks),
    }


def write_split_files(
    cleaned: Sequence[Problem],
    splits: MathSplits,
    parameters: dict,
    directory: str | None = None,
) -> list[str]:
    """Write the partition as index files (one problem id per line).

    Files: ``deduplicated.txt`` (problems kept after deduplication and overlap removal),
    ``optimization.txt``, ``development.txt``, ``diagnostic.txt``, the 250/250 division of
    the diagnostic problems (``diagnostic_update.txt``, ``diagnostic_measurement.txt``) and
    ``split_info.json`` with the parameters and sizes. Returns the written file names.
    """
    target = data_dir() / (directory or SPLIT_DIR)
    update, measurement = split_diagnostic(
        splits.diagnostic, num_update=len(splits.diagnostic) // 2, seed=parameters["seed"]
    )
    contents = {
        "deduplicated": cleaned,
        "optimization": splits.optimization,
        "development": splits.development,
        "diagnostic": splits.diagnostic,
        "diagnostic_update": update,
        "diagnostic_measurement": measurement,
    }
    for name, problems in contents.items():
        write_lines(target / f"{name}.txt", [problem.id for problem in problems])
    info = dict(parameters, sizes={name: len(problems) for name, problems in contents.items()})
    (target / "split_info.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    return [f"{name}.txt" for name in contents] + ["split_info.json"]


def _stored_split(name: str, parameters: dict, by_id: dict[str, Problem]) -> list[Problem] | None:
    """Problems of a stored index file if it exists and was made with ``parameters``."""
    info_path = local_file(f"{SPLIT_DIR}/split_info.json")
    index_path = local_file(f"{SPLIT_DIR}/{name}.txt")
    if info_path is None or index_path is None:
        return None
    info = json.loads(info_path.read_text(encoding="utf-8"))
    if any(info.get(key) != value for key, value in parameters.items()):
        return None
    identifiers = read_lines(index_path)
    if any(identifier not in by_id for identifier in identifiers):
        return None
    return [by_id[identifier] for identifier in identifiers]


def prepare_math_splits(
    overlap_benchmarks: Sequence[str] = (),
    *,
    num_diagnostic: int = 500,
    num_development: int = 500,
    seed: int = 42,
    revision: str | None = None,
) -> MathSplits:
    """Load, clean and partition the MATH training split as in Appendix B.1.

    ``overlap_benchmarks`` names the evaluation benchmarks whose questions must not occur in
    the training problems. The stored index files are used when they were produced with the
    same parameters; otherwise the partition is recomputed from the problems.
    """
    problems = load_math_train(revision)
    parameters = split_parameters(num_diagnostic, num_development, seed, overlap_benchmarks)
    by_id = {problem.id: problem for problem in problems}
    stored = [_stored_split(name, parameters, by_id) for name in SPLIT_NAMES]
    if all(part is not None for part in stored):
        return MathSplits(*stored)

    from vdopd.data.benchmarks import load_benchmark

    evaluation_questions = [
        problem.question for name in overlap_benchmarks for problem in load_benchmark(name)
    ]
    cleaned = remove_overlaps(deduplicate(problems), evaluation_questions)
    return stratified_split(
        cleaned, num_diagnostic=num_diagnostic, num_development=num_development, seed=seed
    )


def prepare_diagnostic_split(
    diagnostic: Sequence[Problem], *, seed: int = 42
) -> tuple[list[Problem], list[Problem]]:
    """Update / measurement division of the diagnostic problems (Appendix B.7).

    Uses the stored index files when they cover exactly ``diagnostic``; otherwise the
    division is recomputed with :func:`split_diagnostic`.
    """
    by_id = {problem.id: problem for problem in diagnostic}
    stored = []
    for name in ("diagnostic_update", "diagnostic_measurement"):
        path = local_file(f"{SPLIT_DIR}/{name}.txt")
        stored.append(read_lines(path) if path is not None else None)
    info_path = local_file(f"{SPLIT_DIR}/split_info.json")
    if info_path is not None and all(part is not None for part in stored):
        same_seed = json.loads(info_path.read_text(encoding="utf-8")).get("seed") == seed
        if same_seed and sorted(stored[0] + stored[1]) == sorted(by_id):
            return [by_id[i] for i in stored[0]], [by_id[i] for i in stored[1]]
    return split_diagnostic(diagnostic, num_update=len(diagnostic) // 2, seed=seed)
