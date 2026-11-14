"""Evaluation metrics: Avg@k, Pass@k and the unseen-view robustness metrics.

Implements Equations 10-12 and 14 of the paper. Inputs are boolean correctness arrays; all
metrics are returned as fractions in ``[0, 1]`` (multiply by 100 for the paper's percent).

The sensitivity gaps, greedy answer flips and parse failures of the supplementary robustness
tables (E12-E15) are defined here as well.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

FLIP_MODES = ("pair", "question")


def _as_matrix(correct: Sequence[Sequence[bool]] | np.ndarray) -> np.ndarray:
    matrix = np.asarray(correct, dtype=np.float64)
    if matrix.ndim != 2 or matrix.size == 0:
        raise ValueError("expected a non-empty [questions, samples] correctness array")
    return matrix


def avg_at_k(correct: Sequence[Sequence[bool]] | np.ndarray) -> float:
    """``Avg@k = (1 / (k n)) sum_i c_i`` (Equation 10), ``c_i`` = correct completions."""
    return float(_as_matrix(correct).mean())


def pass_at_k(correct: Sequence[Sequence[bool]] | np.ndarray) -> float:
    """``Pass@k = (1 / n) sum_i 1[c_i > 0]`` (Equation 10)."""
    return float((_as_matrix(correct).sum(axis=1) > 0).mean())


def per_question_accuracy(correct: Sequence[Sequence[bool]] | np.ndarray) -> np.ndarray:
    """``a_i``: Avg@k of each question, shape ``[n]``."""
    return _as_matrix(correct).mean(axis=1)


def view_robustness(correct: np.ndarray | Sequence) -> dict[str, float]:
    """Unseen-view metrics of Equations 11 and 12.

    Args:
        correct: ``[M, n, k]`` correctness of ``k`` completions for ``n`` questions under
            each of ``M`` wrappers, evaluated independently (no ensembling).

    Returns:
        ``mean_view = (1 / (n M)) sum_{i,m} a_{i,m}``,
        ``worst_view = (1 / n) sum_i min_m a_{i,m}`` (per-question vulnerability) and
        ``worst_template = min_m (1 / n) sum_i a_{i,m}`` (globally weakest template), with
        ``a_{i,m}`` the Avg@k of question ``i`` under wrapper ``m``. By construction
        ``worst_view <= worst_template <= mean_view``.
    """
    array = np.asarray(correct, dtype=np.float64)
    if array.ndim != 3 or array.size == 0:
        raise ValueError("expected a non-empty [views, questions, samples] correctness array")
    accuracy = array.mean(axis=2)  # a_{i,m}, shape [M, n]
    return {
        "mean_view": float(accuracy.mean()),
        "worst_view": float(accuracy.min(axis=0).mean()),
        "worst_template": float(accuracy.mean(axis=1).min()),
    }


def sensitivity_gaps(canonical: float, mean_view: float, worst_view: float) -> dict[str, float]:
    """``Canonical - Mean`` and ``Mean - Worst`` (supplementary E12).

    The first is the drop from the canonical prompt to the mean over the held-out views, the
    second the spread between the mean view and the per-question worst view.
    """
    return {
        "canonical_minus_mean": float(canonical - mean_view),
        "mean_minus_worst": float(mean_view - worst_view),
    }


def greedy_answer_flip_rate(
    canonical: Sequence[str | None],
    views: Sequence[Sequence[str | None]],
    same_answer: Callable[[str | None, str | None], bool],
    mode: str = "pair",
) -> float:
    """Fraction of greedy final answers that change under held-out views (E12, E15).

    Args:
        canonical: the greedy final answer of each of ``n`` questions under the canonical
            prompt (``None`` = no parseable final answer).
        views: ``[M][n]`` greedy final answers under each of ``M`` held-out views.
        same_answer: predicate deciding whether two final answers agree.
        mode: ``"pair"`` counts (question, view) pairs whose answer differs from the
            canonical one; ``"question"`` counts questions whose answer differs under at
            least one view.
    """
    if mode not in FLIP_MODES:
        raise ValueError(f"unknown flip mode {mode!r}; choose from {FLIP_MODES}")
    if len(canonical) == 0 or len(views) == 0:
        raise ValueError("expected at least one question and one view")
    if any(len(answers) != len(canonical) for answers in views):
        raise ValueError("every view needs one answer per question")
    flipped = np.array(
        [
            [not same_answer(c, a) for c, a in zip(canonical, answers, strict=True)]
            for answers in views
        ]
    )
    return float(flipped.mean() if mode == "pair" else flipped.any(axis=0).mean())


def parse_failure_rate(parsed: np.ndarray | Sequence) -> float:
    """Fraction of completions without a parseable final answer (supplementary E15).

    ``parsed`` is a boolean array of any shape, true where a final answer was extracted.
    """
    array = np.asarray(parsed, dtype=np.float64)
    if array.size == 0:
        raise ValueError("no completions")
    return float(1.0 - array.mean())


def macro_average(values: Sequence[float]) -> float:
    """Unweighted mean over benchmarks (Appendix B.2.1)."""
    if len(values) == 0:
        raise ValueError("cannot average an empty list")
    return float(np.mean(values))


def seed_mean_and_sd(values: Sequence[float]) -> tuple[float, float]:
    """Mean and sample standard deviation over seeds (Equation 14, ``n - 1`` denominator)."""
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError("no seeds")
    sd = float(array.std(ddof=1)) if array.size > 1 else 0.0
    return float(array.mean()), sd
