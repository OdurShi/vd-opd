"""Paired question bootstrap for differences between two methods.

The supplementary experiments report 95% confidence intervals for a gain of one method over
another (E01, E06, E07, E08, E10) and for per-stratum means of local-update effects (E17).
The convention is a *paired question bootstrap*: the question is the resampling unit, and a
resampled question brings along all of its sampled completions (eight under the paper's
protocol) and its paired training seeds for both methods. The interval therefore measures
question-sampling uncertainty conditional on the evaluated training runs; it does not
include the variation between training seeds, which is reported separately as mean and
sample standard deviation (:func:`seed_summary`, Equation 14).

All resampling is vectorised NumPy with a seeded ``numpy.random.Generator``; intervals are
percentile intervals.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from vdopd.eval.metrics import seed_mean_and_sd

DEFAULT_RESAMPLES = 10_000
_CHUNK_ELEMENTS = 1 << 21  # resampled indices drawn per block (bounds memory at ~16 MB)


@dataclass(frozen=True)
class BootstrapCI:
    """Point estimate with a percentile bootstrap interval."""

    estimate: float
    low: float
    high: float
    confidence: float = 0.95
    num_resamples: int = DEFAULT_RESAMPLES

    def contains(self, value: float) -> bool:
        return self.low <= value <= self.high


def question_scores(runs: Sequence[np.ndarray | Sequence] | np.ndarray) -> np.ndarray:
    """Per-question score of one method, averaged over its seeds and samples.

    ``runs`` holds one array per training seed, each of shape ``[n]`` (a per-question value
    such as the question's Avg@8, a Pass@8 indicator or a worst-view accuracy) or ``[n, k]``
    (the correctness of ``k`` sampled completions). All seeds must cover the same ``n``
    questions in the same order. The result has shape ``[n]``: question ``i`` keeps all of
    its samples and seeds together, so resampling entries of this vector resamples whole
    questions.
    """
    if isinstance(runs, np.ndarray) and runs.ndim == 1:
        runs = [runs]  # a single run of per-question values
    arrays = [np.asarray(run, dtype=np.float64) for run in runs]
    if not arrays or any(a.ndim == 0 for a in arrays):
        raise ValueError("expected one [n] or [n, k] array per seed")
    reduced = [a if a.ndim == 1 else a.reshape(a.shape[0], -1).mean(axis=1) for a in arrays]
    lengths = {len(a) for a in reduced}
    if len(lengths) != 1 or 0 in lengths:
        raise ValueError(f"runs must cover the same non-empty question set, got sizes {lengths}")
    return np.mean(reduced, axis=0)


def _resampled_means(
    values: np.ndarray, num_resamples: int, rng: np.random.Generator
) -> np.ndarray:
    """Means of ``num_resamples`` resamples (with replacement) of a 1-D array."""
    n = len(values)
    means = np.empty(num_resamples, dtype=np.float64)
    step = max(1, _CHUNK_ELEMENTS // n)
    for start in range(0, num_resamples, step):
        count = min(step, num_resamples - start)
        index = rng.integers(n, size=(count, n))
        means[start : start + count] = values[index].mean(axis=1)
    return means


def _interval(estimate: float, draws: np.ndarray, confidence: float) -> BootstrapCI:
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between 0 and 1")
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(draws, [tail, 1.0 - tail])
    return BootstrapCI(float(estimate), float(low), float(high), confidence, len(draws))


def _paired_difference(a, b) -> np.ndarray:
    scores_a, scores_b = question_scores(a), question_scores(b)
    if scores_a.shape != scores_b.shape:
        raise ValueError("both methods must be evaluated on the same questions")
    return scores_b - scores_a


def paired_question_bootstrap(
    a: Sequence[np.ndarray | Sequence] | np.ndarray,
    b: Sequence[np.ndarray | Sequence] | np.ndarray,
    *,
    num_resamples: int = DEFAULT_RESAMPLES,
    confidence: float = 0.95,
    seed: int = 0,
    rng: np.random.Generator | None = None,
) -> BootstrapCI:
    """Confidence interval for the mean difference ``b - a`` on one benchmark.

    ``a`` and ``b`` are the per-seed arrays of the two methods (see :func:`question_scores`)
    on the same questions. Questions are resampled with replacement; each resampled question
    contributes the paired difference of its scores, i.e. all samples and paired seeds of
    both methods stay together. The estimate is the difference of the two means (for
    correctness arrays: the Avg@k gain, as a fraction).
    """
    difference = _paired_difference(a, b)
    generator = rng if rng is not None else np.random.default_rng(seed)
    draws = _resampled_means(difference, num_resamples, generator)
    return _interval(difference.mean(), draws, confidence)


def stratified_paired_bootstrap(
    benchmarks: Mapping[str, tuple[Sequence, Sequence]] | Sequence[tuple[Sequence, Sequence]],
    *,
    num_resamples: int = DEFAULT_RESAMPLES,
    confidence: float = 0.95,
    seed: int = 0,
) -> BootstrapCI:
    """Confidence interval for a difference of macro averages over benchmarks.

    ``benchmarks`` maps each benchmark to its pair ``(a, b)`` of per-seed arrays. Questions
    are resampled *within* each benchmark (the benchmark sizes stay fixed), each benchmark's
    mean difference is recomputed, and the benchmark means are averaged without weights, as
    for the six-benchmark macro average (Appendix B.2.1).
    """
    pairs = list(benchmarks.values()) if isinstance(benchmarks, Mapping) else list(benchmarks)
    if not pairs:
        raise ValueError("no benchmarks")
    rng = np.random.default_rng(seed)
    differences = [_paired_difference(a, b) for a, b in pairs]
    draws = np.mean([_resampled_means(d, num_resamples, rng) for d in differences], axis=0)
    return _interval(np.mean([d.mean() for d in differences]), draws, confidence)


def clustered_mean_bootstrap(
    values: Sequence[float] | np.ndarray,
    clusters: Sequence | np.ndarray,
    *,
    num_resamples: int = DEFAULT_RESAMPLES,
    confidence: float = 0.95,
    seed: int = 0,
) -> BootstrapCI:
    """Confidence interval for a mean over items that are grouped by question.

    Used for the per-stratum means of supplementary E17, where the items are fixed prefixes
    (states) and several states stem from one question. Questions (``clusters``) are
    resampled with replacement and bring along all of their items; the mean is recomputed as
    the ratio of the resampled sum to the resampled item count.
    """
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("expected a non-empty 1-D array of values")
    _, inverse = np.unique(np.asarray(clusters), return_inverse=True)
    if inverse.shape != array.shape:
        raise ValueError("values and clusters must have the same length")
    sums = np.bincount(inverse, weights=array)
    counts = np.bincount(inverse).astype(np.float64)
    n = len(sums)
    rng = np.random.default_rng(seed)
    draws = np.empty(num_resamples, dtype=np.float64)
    step = max(1, _CHUNK_ELEMENTS // n)
    for start in range(0, num_resamples, step):
        size = min(step, num_resamples - start)
        index = rng.integers(n, size=(size, n))
        draws[start : start + size] = sums[index].sum(axis=1) / counts[index].sum(axis=1)
    return _interval(array.mean(), draws, confidence)


def seed_summary(values: Sequence[float]) -> tuple[float, float]:
    """Mean and sample standard deviation across training seeds (Equation 14)."""
    return seed_mean_and_sd(values)
