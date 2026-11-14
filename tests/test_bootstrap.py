"""Paired question bootstrap of the supplementary experiments (E01, E06-E08, E10, E17)."""

from __future__ import annotations

import numpy as np
import pytest

from vdopd.eval.bootstrap import (
    BootstrapCI,
    clustered_mean_bootstrap,
    paired_question_bootstrap,
    question_scores,
    seed_summary,
    stratified_paired_bootstrap,
)
from vdopd.eval.metrics import seed_mean_and_sd


def correctness(rng, seeds=3, questions=40, samples=8, rate=0.5) -> list[np.ndarray]:
    """Synthetic ``[n, k]`` correctness arrays, one per seed, with per-question difficulty."""
    difficulty = rng.random((questions, 1))
    return [rng.random((questions, samples)) < rate * 2 * difficulty for _ in range(seeds)]


def test_question_scores_average_samples_and_seeds_per_question():
    first = np.array([[1, 1, 0, 0], [0, 0, 0, 0]], dtype=bool)
    second = np.array([[1, 1, 1, 1], [0, 1, 0, 0]], dtype=bool)
    assert question_scores([first, second]).tolist() == [0.75, 0.125]
    assert question_scores([[0.5, 1.0], [0.0, 1.0]]).tolist() == [0.25, 1.0]  # [n] per seed
    assert question_scores(np.array([0.5, 1.0])).tolist() == [0.5, 1.0]  # one run
    with pytest.raises(ValueError):
        question_scores([np.zeros((3, 8)), np.zeros((4, 8))])  # different question sets
    with pytest.raises(ValueError):
        question_scores([])


def test_identical_methods_and_constant_shifts_have_degenerate_intervals():
    rng = np.random.default_rng(0)
    a = correctness(rng)
    same = paired_question_bootstrap(a, a, num_resamples=200)
    assert (same.estimate, same.low, same.high) == (0.0, 0.0, 0.0)
    scores = question_scores(a)
    shifted = paired_question_bootstrap([scores], [scores + 0.125], num_resamples=200)
    assert shifted.estimate == pytest.approx(0.125)
    assert shifted.low == pytest.approx(0.125) and shifted.high == pytest.approx(0.125)


def test_estimate_is_the_difference_of_the_two_means_and_lies_in_the_interval():
    rng = np.random.default_rng(1)
    a, b = correctness(rng, rate=0.4), correctness(rng, rate=0.5)
    ci = paired_question_bootstrap(a, b, num_resamples=2000, seed=3)
    assert isinstance(ci, BootstrapCI) and ci.confidence == 0.95 and ci.num_resamples == 2000
    assert ci.estimate == pytest.approx(np.mean(b) - np.mean(a))
    assert ci.low < ci.estimate < ci.high and ci.contains(ci.estimate)
    narrow = paired_question_bootstrap(a, b, num_resamples=2000, seed=3, confidence=0.5)
    assert ci.low < narrow.low < narrow.high < ci.high


def test_bootstrap_is_seeded():
    rng = np.random.default_rng(2)
    a, b = correctness(rng), correctness(rng)
    first = paired_question_bootstrap(a, b, num_resamples=500, seed=7)
    assert first == paired_question_bootstrap(a, b, num_resamples=500, seed=7)
    assert first != paired_question_bootstrap(a, b, num_resamples=500, seed=8)
    generator = np.random.default_rng(7)
    assert first == paired_question_bootstrap(a, b, num_resamples=500, rng=generator)


def test_the_question_is_the_resampling_unit():
    """Samples and seeds of a question stay together: only per-question means matter."""
    rng = np.random.default_rng(3)
    a, b = correctness(rng), correctness(rng)
    reference = paired_question_bootstrap(a, b, num_resamples=300, seed=5)
    # Rearranging the samples of each question, or the order of the paired seeds, or
    # repeating every sample changes nothing, because whole questions are resampled.
    shuffled = [rng.permuted(run, axis=1) for run in a]
    assert paired_question_bootstrap(shuffled, b, num_resamples=300, seed=5) == reference
    reordered = paired_question_bootstrap(a[::-1], b[::-1], num_resamples=300, seed=5)
    assert reordered.low == pytest.approx(reference.low, abs=1e-12)
    doubled = [np.concatenate([run, run], axis=1) for run in b]
    repeated = paired_question_bootstrap(a, doubled, num_resamples=300, seed=5)
    assert repeated.high == pytest.approx(reference.high, abs=1e-12)
    # Resampling the n * k samples independently would understate the uncertainty when the
    # samples of a question are correlated: here every question is all-correct or all-wrong.
    blocky = [np.repeat(rng.random((60, 1)) < 0.5, 8, axis=1)]
    zeros = [np.zeros((60, 8), dtype=bool)]
    ci = paired_question_bootstrap(zeros, blocky, num_resamples=4000, seed=0)
    half_width = (ci.high - ci.low) / 2
    question_level = 1.96 * question_scores(blocky).std(ddof=1) / np.sqrt(60)
    sample_level = 1.96 * np.std(blocky[0], ddof=1) / np.sqrt(60 * 8)
    assert half_width == pytest.approx(question_level, rel=0.1)
    assert half_width > 2 * sample_level


def test_interval_width_matches_the_standard_error_of_the_paired_differences():
    rng = np.random.default_rng(4)
    a, b = correctness(rng, questions=400), correctness(rng, questions=400)
    difference = question_scores(b) - question_scores(a)
    ci = paired_question_bootstrap(a, b, num_resamples=4000, seed=1)
    expected = 1.96 * difference.std(ddof=1) / np.sqrt(len(difference))
    assert (ci.high - ci.low) / 2 == pytest.approx(expected, rel=0.08)
    with pytest.raises(ValueError):
        paired_question_bootstrap(a, [run[:10] for run in b])
    with pytest.raises(ValueError):
        paired_question_bootstrap(a, b, confidence=1.0)


def test_stratified_bootstrap_resamples_within_benchmarks_and_macro_averages():
    rng = np.random.default_rng(5)
    large = (correctness(rng, questions=50), correctness(rng, questions=50))
    small = (correctness(rng, questions=6), correctness(rng, questions=6))
    ci = stratified_paired_bootstrap({"large": large, "small": small}, num_resamples=1000)
    gains = [np.mean(b) - np.mean(a) for a, b in (large, small)]
    assert ci.estimate == pytest.approx(np.mean(gains))  # unweighted over benchmarks
    assert ci.low < ci.estimate < ci.high
    assert ci == stratified_paired_bootstrap([large, small], num_resamples=1000)
    # A benchmark whose gain is the same on every question adds no width: the macro
    # interval is the average of the other benchmark's interval and that constant.
    base = question_scores(small[0])
    constant = ([base], [base + 0.25])
    alone = paired_question_bootstrap(*large, num_resamples=1000, seed=0)
    both = stratified_paired_bootstrap([large, constant], num_resamples=1000, seed=0)
    assert both.low == pytest.approx((alone.low + 0.25) / 2, abs=1e-12)
    assert both.high == pytest.approx((alone.high + 0.25) / 2, abs=1e-12)
    with pytest.raises(ValueError):
        stratified_paired_bootstrap({})


def test_clustered_mean_bootstrap_keeps_the_items_of_a_question_together():
    rng = np.random.default_rng(6)
    values = rng.normal(size=30)
    # One item per question: the ordinary bootstrap of a mean.
    single = clustered_mean_bootstrap(values, np.arange(30), num_resamples=500, seed=2)
    plain = paired_question_bootstrap([np.zeros(30)], [values], num_resamples=500, seed=2)
    assert single.estimate == pytest.approx(values.mean())
    assert single.low == pytest.approx(plain.low, abs=1e-12)
    assert single.high == pytest.approx(plain.high, abs=1e-12)
    # Four states per question with identical values: as uncertain as one state per question.
    repeated = clustered_mean_bootstrap(
        np.repeat(values, 4), np.repeat(np.arange(30), 4), num_resamples=500, seed=2
    )
    assert repeated.low == pytest.approx(single.low, abs=1e-12)
    # Unequal numbers of states per question: the mean is a ratio of resampled sums.
    clusters = np.array(["a", "a", "a", "b", "c", "c"])
    ci = clustered_mean_bootstrap([1.0, 1.0, 1.0, 0.0, 0.5, 0.5], clusters, num_resamples=2000)
    assert ci.estimate == pytest.approx(4.0 / 6.0) and 0.0 <= ci.low < ci.high <= 1.0
    with pytest.raises(ValueError):
        clustered_mean_bootstrap([], [])
    with pytest.raises(ValueError):
        clustered_mean_bootstrap([1.0, 2.0], [0])


def test_seed_summary_is_the_mean_and_sample_sd_over_seeds():
    values = [34.0, 35.0, 36.5]
    assert seed_summary(values) == seed_mean_and_sd(values)
    mean, sd = seed_summary(values)
    assert mean == pytest.approx(np.mean(values)) and sd == pytest.approx(np.std(values, ddof=1))
