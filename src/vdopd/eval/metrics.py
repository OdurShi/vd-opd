"""Evaluation metrics: Avg@k, Pass@k and the unseen-view robustness metrics.

Implements Equations 10-12 and 14 of the paper. Inputs are boolean correctness arrays; all
metrics are returned as fractions in ``[0, 1]`` (multiply by 100 for the paper's percent).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


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
