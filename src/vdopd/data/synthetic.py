r"""Synthetic integer-arithmetic task for the CPU quick start and the offline tests.

Questions such as ``What is 12+7?`` have a single integer answer; the expected response is
``\boxed{19}`` followed by EOS, so the quick start exercises the same answer-extraction and
scoring path as the real benchmarks. This task is *not* part of the paper's experiments.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from vdopd.data.problems import Problem

SYNTHETIC_TEMPLATE = "{question} Answer in \\boxed{{}}."


def make_arithmetic_problems(
    max_operand: int = 19, operations: Sequence[str] = ("+", "-")
) -> list[Problem]:
    """Enumerate ``a op b`` for all operands in ``[0, max_operand]``."""
    problems: list[Problem] = []
    for operation in operations:
        for a in range(max_operand + 1):
            for b in range(max_operand + 1):
                value = {"+": a + b, "-": a - b, "*": a * b}[operation]
                problems.append(
                    Problem(
                        id=f"synthetic/{a}{operation}{b}",
                        question=f"What is {a}{operation}{b}?",
                        answer=str(value),
                        subject=operation,
                    )
                )
    return problems


def split_problems(
    problems: Sequence[Problem], test_fraction: float = 0.2, seed: int = 0
) -> tuple[list[Problem], list[Problem]]:
    """Disjoint train/test split of the problem list."""
    order = np.random.default_rng(seed).permutation(len(problems))
    num_test = int(round(test_fraction * len(problems)))
    test = [problems[i] for i in order[:num_test]]
    train = [problems[i] for i in order[num_test:]]
    return train, test


def reference_response(problem: Problem) -> str:
    """Reference completion of a synthetic problem."""
    return f"\\boxed{{{problem.answer}}}"
