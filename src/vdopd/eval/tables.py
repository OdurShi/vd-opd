"""Result tables in the layout of the supplementary experiments (E01-E15).

Evaluation files written by ``vdopd eval --output`` are turned into CSV tables whose file
names and column names equal those of the supplementary result tables, so that the output of
new runs can be placed next to (or compared with) the reported tables. Composite cells are
split into one column per number: ``<name>_mean`` / ``<name>_sd`` for mean and sample SD over
training seeds, ``<name>`` / ``<name>_ci_low`` / ``<name>_ci_high`` for a gain with its 95%
paired-question-bootstrap interval. Accuracies are in percent, gains in percentage points
and FLOPs in units of 1e18.

Expected layout (the one produced by ``scripts/run_experiment.sh``)::

    <results-dir>/<group>/<method>/seed<seed>.json        evaluation of one training seed
    <runs-dir>/<group>/<method>/seed<seed>/summary.json   optional: FLOPs and mean Z of the run

``<group>`` is the name of the config folder (``qwen3_1p7b``, ``qwen3_4b``, ``llama3p2_3b``)
and ``<method>`` the name of the config file, e.g. ``eopd_full``, ``eopd_2v``, ``vdopd``,
``vdopd_budget``, ``vdopd_k4``, ``ablation_geometric`` or ``eopd_2v_studentaug``; reference
models are stored as ``frozen_teacher`` and ``initial_student``
(``scripts/evaluate_reference.sh``).

Every number is computed from the files that are passed in. Cells whose inputs are missing
(a method without evaluation files, a file without per-question results, GPU-hours,
which this code does not record) are left empty.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from vdopd.eval.bootstrap import (
    DEFAULT_RESAMPLES,
    BootstrapCI,
    paired_question_bootstrap,
    stratified_paired_bootstrap,
)
from vdopd.eval.metrics import macro_average, seed_mean_and_sd
from vdopd.views import ORIGINAL_VIEW, PARAPHRASE_FAMILY, PREFIX_FAMILY, STRUCTURAL_FAMILY

Runs = Mapping[int, Mapping[str, Any]]  # training seed -> evaluation record

GROUP_LABELS = {
    "qwen3_1p7b": "Qwen3-1.7B",
    "qwen3_4b": "Qwen3-4B",
    "llama3p2_3b": "Llama-3.2-3B",
}
TEACHER_LABELS = {
    "qwen3_1p7b": "Qwen3-8B",
    "qwen3_4b": "Qwen3-8B",
    "llama3p2_3b": "Llama-3.1-8B",
}
BENCHMARK_LABELS = {
    "math500": "MATH500",
    "amc23": "AMC23",
    "aime24": "AIME24",
    "aime25": "AIME25",
    "minerva": "Minerva",
    "olympiadbench": "OlympiadBench",
}
# Method folders of the complete per-benchmark tables, in display order.
METHOD_LABELS = {
    "frozen_teacher": "Frozen teacher",
    "initial_student": "Initial student",
    "gkd_fkl": "GKD-FKL",
    "gkd_rkl": "GKD-RKL(full)",
    "gkd_jsd": "GKD-JSD",
    "sampled_opd": "Sampled OPD",
    "todi": "ToDi-OP",
    "tsd_kd": "TSD-KD",
    "veto": "Veto",
    "eopd_top16": "EOPD(top-16)",
    "eopd_full": "EOPD(full)",
    "vdopd": "VD-OPD",
}
ROBUSTNESS_METHODS = {
    "eopd_full": ("EOPD(full)", "eopd_full"),
    "eopd_2v": ("EOPD(2V)", "eopd_2v"),
    "eopd_full_studentaug": ("EOPD(full)+StudentAug", "eopd_full_studentaug"),
    "eopd_2v_studentaug": ("EOPD(2V)+StudentAug", "eopd_2v_studentaug"),
    "vdopd": ("VD-OPD", "vd_opd"),
}
FACTORIAL_CELLS = (
    ("Arithmetic", "1", "ablation_arithmetic"),
    ("Arithmetic", "Z", "ablation_weighted_arithmetic"),
    ("Geometric", "1", "ablation_geometric"),
    ("Geometric", "shuffled Z", "ablation_shuffled_weight"),
    ("Geometric", "batch-mean Z", "ablation_batch_mean_weight"),
    ("Geometric (VD-OPD)", "state-specific Z", "vdopd"),
)
COMPUTE_CONTROLS = (
    ("EOPD(full) vs. VD-Budget", "equal total FLOPs", "eopd_full", "vdopd_budget"),
    ("EOPD(2V) vs. VD-OPD", "equal two-view / update budget", "eopd_2v", "vdopd"),
    ("EOPD(full) vs. VD-OPD", "none (unmatched)", "eopd_full", "vdopd"),
)
VIEW_COUNTS = (
    ("K=2", "vdopd"),
    ("K=4, same updates", "vdopd_k4"),
    ("K=4, same FLOPs", "vdopd_k4_budget"),
)
VIEW_POOLS = (
    ("Simple templates", "eopd_2v", "vdopd"),
    ("Random valid paraphrases", "eopd_2v_paraphrase", "vdopd_paraphrase"),
    ("Lower-quality, task-preserving", "eopd_2v_lowquality", "vdopd_lowquality"),
)
HELD_OUT_FAMILIES = (("Structural", STRUCTURAL_FAMILY), ("Paraphrase", PARAPHRASE_FAMILY))


@dataclass
class ResultSet:
    """Evaluation records and optional training records of a set of runs.

    Attributes:
        evaluations: ``{group: {method: {seed: evaluation record}}}``.
        runs_dir: directory with the training outputs (FLOPs, mean Z), or ``None``.
        group: group used by the tables that concern one student (default: Qwen3-1.7B).
        num_resamples, bootstrap_seed: settings of the paired question bootstrap.
    """

    evaluations: dict[str, dict[str, dict[int, dict[str, Any]]]]
    runs_dir: Path | None = None
    group: str = "qwen3_1p7b"
    num_resamples: int = DEFAULT_RESAMPLES
    bootstrap_seed: int = 0

    def runs(self, method: str, group: str | None = None) -> Runs | None:
        return self.evaluations.get(group or self.group, {}).get(method)


def load_results(results_dir: str | Path) -> dict[str, dict[str, dict[int, dict[str, Any]]]]:
    """Read ``<results-dir>/<group>/<method>/seed<seed>.json`` into a nested dictionary."""
    evaluations: dict[str, dict[str, dict[int, dict[str, Any]]]] = {}
    for path in sorted(Path(results_dir).glob("*/*/seed*.json")):
        try:
            seed = int(path.stem[len("seed") :])
        except ValueError:
            continue
        with open(path, encoding="utf-8") as handle:
            record = json.load(handle)
        group, method = path.parent.parent.name, path.parent.name
        evaluations.setdefault(group, {}).setdefault(method, {})[seed] = record
    return evaluations


# ------------------------------------------------------------------ record access
def benchmarks_of(runs: Runs) -> list[str]:
    """Benchmarks of a method (those of its first seed, in file order)."""
    return list(next(iter(runs.values()))["benchmarks"])


def benchmark_metric(
    record: Mapping[str, Any], benchmark: str, metric: str, family: str | None = None
) -> float | None:
    """One metric of one benchmark as a fraction, or ``None`` if it was not recorded.

    ``family=None`` reads a canonical-prompt metric (``avg_at_k``, ``pass_at_k``); otherwise
    the metric of a held-out view family. Evaluation files written before view families
    existed store the prefix family under ``unseen``.
    """
    entry = record["benchmarks"].get(benchmark)
    if entry is None:
        return None
    if family is None:
        return entry.get(metric)
    families = entry.get("families", {})
    if family in families:
        return families[family].get(metric)
    if family == PREFIX_FAMILY:
        return entry.get("unseen", {}).get(metric)
    return None


def macro_metric(record: Mapping[str, Any], metric: str, family: str | None = None) -> float | None:
    """Unweighted mean of a metric over the benchmarks of one evaluation (one seed)."""
    values = [benchmark_metric(record, name, metric, family) for name in record["benchmarks"]]
    return None if not values or any(v is None for v in values) else macro_average(values)


def seed_statistics(values: Sequence[float | None]) -> tuple[float, float] | None:
    """Mean and sample SD over seeds in percent, or ``None`` if a seed lacks the value."""
    if not values or any(v is None for v in values):
        return None
    return seed_mean_and_sd([100.0 * v for v in values])


def benchmark_statistics(
    runs: Runs | None, benchmark: str, metric: str, family: str | None = None
) -> tuple[float, float] | None:
    if not runs:
        return None
    values = [benchmark_metric(record, benchmark, metric, family) for record in runs.values()]
    return seed_statistics(values)


def macro_statistics(
    runs: Runs | None, metric: str, family: str | None = None
) -> tuple[float, float] | None:
    """Per-seed macro average, then mean and sample SD across seeds (Appendix B.5.3)."""
    if not runs:
        return None
    return seed_statistics([macro_metric(record, metric, family) for record in runs.values()])


def question_accuracy(
    record: Mapping[str, Any], benchmark: str, view: str = ORIGINAL_VIEW.name
) -> tuple[list[str], np.ndarray] | None:
    """Question ids and per-question Avg@k of one view, if the evaluation stored them."""
    entry = record["benchmarks"].get(benchmark, {})
    questions = entry.get("questions")
    if not questions or view not in questions.get("correct_counts", {}):
        return None
    counts = np.asarray(questions["correct_counts"][view], dtype=np.float64)
    return list(questions["ids"]), counts / entry["num_samples"]


def paired_runs(
    baseline: Runs, method: Runs, benchmark: str
) -> tuple[list[np.ndarray], list[np.ndarray]] | None:
    """Per-question accuracies of two methods on their common seeds, seed by seed.

    Returns ``None`` when the seeds do not overlap or per-question results are missing.
    Raises if the two methods (or two seeds) were evaluated on different questions.
    """
    seeds = sorted(set(baseline) & set(method))
    first: list[np.ndarray] = []
    second: list[np.ndarray] = []
    reference_ids: list[str] | None = None
    for seed in seeds:
        a = question_accuracy(baseline[seed], benchmark)
        b = question_accuracy(method[seed], benchmark)
        if a is None or b is None:
            return None
        reference_ids = reference_ids or a[0]
        if a[0] != reference_ids or b[0] != reference_ids:
            raise ValueError(
                f"{benchmark}: the compared runs were evaluated on different questions"
            )
        first.append(a[1])
        second.append(b[1])
    return (first, second) if seeds else None


def benchmark_gain_ci(
    results: ResultSet, baseline: Runs, method: Runs, benchmark: str
) -> BootstrapCI | None:
    """95% CI of the Avg@k gain of ``method`` over ``baseline`` on one benchmark (fraction)."""
    pair = paired_runs(baseline, method, benchmark)
    if pair is None:
        return None
    return paired_question_bootstrap(
        *pair, num_resamples=results.num_resamples, seed=results.bootstrap_seed
    )


def macro_gain_ci(results: ResultSet, baseline: Runs, method: Runs) -> BootstrapCI | None:
    """95% CI of the macro-average Avg@k gain, resampling questions within benchmarks."""
    pairs = {}
    for benchmark in benchmarks_of(baseline):
        pair = paired_runs(baseline, method, benchmark)
        if pair is None:
            return None
        pairs[benchmark] = pair
    return stratified_paired_bootstrap(
        pairs, num_resamples=results.num_resamples, seed=results.bootstrap_seed
    )


def run_record(results: ResultSet, method: str, key: str, group: str | None = None) -> float | None:
    """Seed mean of a run-level quantity (``flops_total``, ``mean_mass``) of a method.

    Reads ``summary.json`` of every seed's training folder; if it is absent, FLOPs are taken
    from the last line of ``metrics.jsonl`` and the mean mass from its per-iteration totals.
    """
    if results.runs_dir is None:
        return None
    folder = results.runs_dir / (group or results.group) / method
    values: list[float] = []
    for seed_dir in sorted(folder.glob("seed*")):
        value = _read_run_value(seed_dir, key)
        if value is not None:
            values.append(value)
    return float(np.mean(values)) if values else None


def _read_run_value(seed_dir: Path, key: str) -> float | None:
    summary = seed_dir / "summary.json"
    if summary.is_file():
        with open(summary, encoding="utf-8") as handle:
            value = json.load(handle).get(key)
        return None if value is None else float(value)
    metrics = seed_dir / "metrics.jsonl"
    if not metrics.is_file():
        return None
    with open(metrics, encoding="utf-8") as handle:
        lines = [json.loads(line) for line in handle if line.strip()]
    if not lines:
        return None
    if key == "mean_mass":
        scored = [line for line in lines if "mass_total" in line]
        positions = sum(line["num_positions"] for line in scored)
        return sum(line["mass_total"] for line in scored) / positions if positions else None
    value = lines[-1].get(key)
    return None if value is None else float(value)


# ------------------------------------------------------------------ formatting
def number(value: float | None, digits: int = 1, signed: bool = False) -> str:
    """Fixed-point cell; empty for a missing value; ``+`` sign for gains."""
    if value is None:
        return ""
    rounded = round(float(value), digits) + 0.0  # "+ 0.0" turns -0.0 into 0.0
    return f"{rounded:+.{digits}f}" if signed else f"{rounded:.{digits}f}"


def _mean_sd_cells(statistics: tuple[float, float] | None, mean_digits: int = 1) -> tuple[str, str]:
    if statistics is None:
        return "", ""
    return number(statistics[0], mean_digits), number(statistics[1], 2)


def _gain_cells(
    baseline: tuple[float, float] | None,
    method: tuple[float, float] | None,
    interval: BootstrapCI | None,
) -> tuple[str, str, str]:
    """Gain of the seed means (pp) and the bounds of its bootstrap interval (pp)."""
    if baseline is None or method is None:
        return "", "", ""
    low = None if interval is None else 100.0 * interval.low
    high = None if interval is None else 100.0 * interval.high
    return number(method[0] - baseline[0], 1, True), number(low, 1, True), number(high, 1, True)


def _flops(results: ResultSet, method: str, group: str | None = None) -> str:
    value = run_record(results, method, "flops_total", group)
    return number(None if value is None else value / 1e18, 1)


# ------------------------------------------------------------------ table builders
def table_e01(results: ResultSet) -> list[dict[str, str]]:
    """Three-seed per-benchmark Avg@k of EOPD(full) and VD-OPD with the gain and its CI."""
    rows = []
    for group, label in GROUP_LABELS.items():
        baseline, method = results.runs("eopd_full", group), results.runs("vdopd", group)
        if not baseline or not method:
            continue
        for benchmark in benchmarks_of(baseline):
            a = benchmark_statistics(baseline, benchmark, "avg_at_k")
            b = benchmark_statistics(method, benchmark, "avg_at_k")
            gain = _gain_cells(a, b, benchmark_gain_ci(results, baseline, method, benchmark))
            rows.append(
                {
                    "student": label,
                    "benchmark": BENCHMARK_LABELS.get(benchmark, benchmark),
                    "eopd_full_avg8_mean": _mean_sd_cells(a)[0],
                    "eopd_full_avg8_sd": _mean_sd_cells(a)[1],
                    "vd_opd_avg8_mean": _mean_sd_cells(b)[0],
                    "vd_opd_avg8_sd": _mean_sd_cells(b)[1],
                    "vd_gain": gain[0],
                    "vd_gain_ci_low": gain[1],
                    "vd_gain_ci_high": gain[2],
                }
            )
    return rows


def _complete_table(group: str) -> Callable[[ResultSet], list[dict[str, str]]]:
    """Avg@k / Pass@k of every available method on the six benchmarks of one student."""

    def build(results: ResultSet) -> list[dict[str, str]]:
        rows = []
        for method, label in METHOD_LABELS.items():
            runs = results.runs(method, group)
            if not runs:
                continue
            row = {"method": label}
            for benchmark in BENCHMARK_LABELS:
                for metric, suffix in (("avg_at_k", "avg8"), ("pass_at_k", "pass8")):
                    statistics = benchmark_statistics(runs, benchmark, metric)
                    row[f"{benchmark}_{suffix}"] = _mean_sd_cells(statistics)[0]
            rows.append(row)
        return rows

    return build


def table_e05(results: ResultSet) -> list[dict[str, str]]:
    """Teacher and Qwen3-4B students on MATH500 under the common protocol."""
    group = "qwen3_4b"
    entries = (
        ("frozen_teacher", f"{TEACHER_LABELS[group]} teacher"),
        ("eopd_full", f"{GROUP_LABELS[group]}, EOPD(full)"),
        ("vdopd", f"{GROUP_LABELS[group]}, VD-OPD"),
    )
    rows = []
    for method, label in entries:
        runs = results.runs(method, group)
        if not runs:
            continue
        rows.append(
            {
                "model": label,
                "math500_avg8": _mean_sd_cells(benchmark_statistics(runs, "math500", "avg_at_k"))[
                    0
                ],
                "math500_pass8": _mean_sd_cells(benchmark_statistics(runs, "math500", "pass_at_k"))[
                    0
                ],
            }
        )
    return rows


def table_e06(results: ResultSet) -> list[dict[str, str]]:
    """Compute- and view-matched controls: macro Avg@k, FLOPs and the VD gain with its CI."""
    rows = []
    for comparison, budget, baseline_name, method_name in COMPUTE_CONTROLS:
        baseline, method = results.runs(baseline_name), results.runs(method_name)
        if not baseline or not method:
            continue
        a, b = macro_statistics(baseline, "avg_at_k"), macro_statistics(method, "avg_at_k")
        gain = _gain_cells(a, b, macro_gain_ci(results, baseline, method))
        rows.append(
            {
                "comparison": comparison,
                "budget_matched": budget,
                "baseline_flops_1e18": _flops(results, baseline_name),
                "vd_flops_1e18": _flops(results, method_name),
                "baseline_macro_avg8_mean": _mean_sd_cells(a, 2)[0],
                "baseline_macro_avg8_sd": _mean_sd_cells(a, 2)[1],
                "vd_macro_avg8_mean": _mean_sd_cells(b, 2)[0],
                "vd_macro_avg8_sd": _mean_sd_cells(b, 2)[1],
                "vd_gain": gain[0],
                "vd_gain_ci_low": gain[1],
                "vd_gain_ci_high": gain[2],
            }
        )
    return rows


def table_e07(results: ResultSet) -> list[dict[str, str]]:
    """Target x weight factorial: macro Avg@k of every cell and the gain of VD-OPD over it."""
    reference = results.runs("vdopd")
    rows = []
    for target, weight, method_name in FACTORIAL_CELLS:
        runs = results.runs(method_name)
        if not runs:
            continue
        statistics = macro_statistics(runs, "avg_at_k")
        gain = ("", "", "")
        if method_name != "vdopd" and reference:
            gain = _gain_cells(
                statistics,
                macro_statistics(reference, "avg_at_k"),
                macro_gain_ci(results, runs, reference),
            )
        rows.append(
            {
                "target": target,
                "weight": weight,
                "macro_avg8_mean": _mean_sd_cells(statistics)[0],
                "macro_avg8_sd": _mean_sd_cells(statistics)[1],
                "vd_gain": gain[0],
                "vd_gain_ci_low": gain[1],
                "vd_gain_ci_high": gain[2],
            }
        )
    return rows if len(rows) > 1 else []  # the VD-OPD row alone is not an ablation


def table_e08(results: ResultSet) -> list[dict[str, str]]:
    """Two versus four teacher views: accuracy, WorstView, FLOPs and the gain over K=2."""
    reference = results.runs("vdopd")
    rows = []
    for setting, method_name in VIEW_COUNTS:
        runs = results.runs(method_name)
        if not runs:
            continue
        statistics = macro_statistics(runs, "avg_at_k")
        worst = macro_statistics(runs, "worst_view", PREFIX_FAMILY)
        gain = ("", "", "")
        if method_name != "vdopd" and reference:
            gain = _gain_cells(
                macro_statistics(reference, "avg_at_k"),
                statistics,
                macro_gain_ci(results, reference, runs),
            )
        rows.append(
            {
                "setting": setting,
                "macro_avg8_mean": _mean_sd_cells(statistics)[0],
                "macro_avg8_sd": _mean_sd_cells(statistics)[1],
                "worst_view": _mean_sd_cells(worst)[0],
                "flops_1e18": _flops(results, method_name),
                "gpu_hours": "",  # not recorded by this code
                "gain_over_k2": gain[0],
                "gain_over_k2_ci_low": gain[1],
                "gain_over_k2_ci_high": gain[2],
            }
        )
    return rows if len(rows) > 1 else []  # the K = 2 row alone is not a comparison


def table_e10(results: ResultSet) -> list[dict[str, str]]:
    """Training view pools: mean Z, EOPD(2V) and VD-OPD accuracy, their gap and WorstView."""
    single_view = macro_statistics(results.runs("eopd_full"), "avg_at_k")
    rows = []
    for pool, baseline_name, method_name in VIEW_POOLS:
        baseline, method = results.runs(baseline_name), results.runs(method_name)
        if not baseline or not method:
            continue
        a, b = macro_statistics(baseline, "avg_at_k"), macro_statistics(method, "avg_at_k")
        gain = _gain_cells(a, b, macro_gain_ci(results, baseline, method))
        rows.append(
            {
                "view_pool": pool,
                "mean_z": number(run_record(results, method_name, "mean_mass"), 3),
                "eopd_full": _mean_sd_cells(single_view)[0],
                "eopd_2v_mean": _mean_sd_cells(a)[0],
                "eopd_2v_sd": _mean_sd_cells(a)[1],
                "vd_opd_mean": _mean_sd_cells(b)[0],
                "vd_opd_sd": _mean_sd_cells(b)[1],
                "vd_minus_2v": gain[0],
                "vd_minus_2v_ci_low": gain[1],
                "vd_minus_2v_ci_high": gain[2],
                "worst_view_eopd_2v": _mean_sd_cells(
                    macro_statistics(baseline, "worst_view", PREFIX_FAMILY)
                )[0],
                "worst_view_vd_opd": _mean_sd_cells(
                    macro_statistics(method, "worst_view", PREFIX_FAMILY)
                )[0],
            }
        )
    return rows


def _macro_cell(runs: Runs, metric: str, family: str | None = None, digits: int = 1) -> str:
    statistics = macro_statistics(runs, metric, family)
    return "" if statistics is None else number(statistics[0], digits)


def table_e12(results: ResultSet) -> list[dict[str, str]]:
    """Unseen-wrapper robustness (prefix family), macro averages over the benchmarks."""
    rows = []
    for method, (label, _) in ROBUSTNESS_METHODS.items():
        runs = results.runs(method)
        if not runs:
            continue
        canonical = macro_statistics(runs, "avg_at_k")
        mean_view = macro_statistics(runs, "mean_view", PREFIX_FAMILY)
        worst_view = macro_statistics(runs, "worst_view", PREFIX_FAMILY)
        gaps = canonical is not None and mean_view is not None and worst_view is not None
        rows.append(
            {
                "method": label,
                "canonical": _mean_sd_cells(canonical)[0],
                "mean_view": _mean_sd_cells(mean_view)[0],
                "worst_view": _mean_sd_cells(worst_view)[0],
                "worst_template": _macro_cell(runs, "worst_template", PREFIX_FAMILY),
                "canonical_minus_mean": number(canonical[0] - mean_view[0]) if gaps else "",
                "mean_minus_worst": number(mean_view[0] - worst_view[0]) if gaps else "",
                "greedy_answer_flips_pct": _macro_cell(
                    runs, "greedy_answer_flips", PREFIX_FAMILY, digits=0
                ),
            }
        )
    return rows


def table_e13(results: ResultSet) -> list[dict[str, str]]:
    """Unseen-wrapper MeanView / WorstView of every method on every benchmark."""
    available = {m: results.runs(m) for m in ROBUSTNESS_METHODS if results.runs(m)}
    if not available:
        return []
    rows = []
    for benchmark in benchmarks_of(next(iter(available.values()))):
        row = {"benchmark": BENCHMARK_LABELS.get(benchmark, benchmark)}
        for method, (_, prefix) in ROBUSTNESS_METHODS.items():
            runs = available.get(method)
            for metric in ("mean_view", "worst_view"):
                statistics = benchmark_statistics(runs, benchmark, metric, PREFIX_FAMILY)
                row[f"{prefix}_{metric}"] = _mean_sd_cells(statistics)[0]
        rows.append(row)
    return rows


def table_e14(results: ResultSet) -> list[dict[str, str]]:
    """Student-side augmentation: FLOPs, canonical accuracy and three held-out families."""
    families = (
        ("prefix", PREFIX_FAMILY),
        ("structural", STRUCTURAL_FAMILY),
        ("paraphrase", PARAPHRASE_FAMILY),
    )
    rows = []
    for method, (label, _) in ROBUSTNESS_METHODS.items():
        runs = results.runs(method)
        if not runs:
            continue
        row = {
            "method": label,
            "flops_1e18": _flops(results, method),
            "canonical": _macro_cell(runs, "avg_at_k"),
        }
        for column, family in families:
            row[f"{column}_mean_view"] = _macro_cell(runs, "mean_view", family)
            row[f"{column}_worst_view"] = _macro_cell(runs, "worst_view", family)
        rows.append(row)
    return rows


def table_e15(results: ResultSet) -> list[dict[str, str]]:
    """Held-out structural changes and verified paraphrases, detailed metrics."""
    rows = []
    for family_label, family in HELD_OUT_FAMILIES:
        for method, (label, _) in ROBUSTNESS_METHODS.items():
            runs = results.runs(method)
            if not runs or macro_statistics(runs, "mean_view", family) is None:
                continue
            rows.append(
                {
                    "family": family_label,
                    "method": label,
                    "mean_view": _macro_cell(runs, "mean_view", family),
                    "worst_view": _macro_cell(runs, "worst_view", family),
                    "worst_template": _macro_cell(runs, "worst_template", family),
                    "answer_flips_pct": _macro_cell(runs, "greedy_answer_flips", family, digits=0),
                    "parse_failures_pct": _macro_cell(runs, "parse_failures", family),
                }
            )
    return rows


# ------------------------------------------------------------------ registry and output
@dataclass(frozen=True)
class TableSpec:
    """File name, column names and builder of one supplementary table."""

    experiment: str
    path: str
    columns: tuple[str, ...]
    build: Callable[[ResultSet], list[dict[str, str]]]


_BENCHMARK_COLUMNS = tuple(
    f"{benchmark}_{suffix}" for benchmark in BENCHMARK_LABELS for suffix in ("avg8", "pass8")
)
_GAIN = ("vd_gain", "vd_gain_ci_low", "vd_gain_ci_high")

TABLES: dict[str, TableSpec] = {
    spec.experiment: spec
    for spec in (
        TableSpec(
            "E01",
            "01_main_results/E01_three_seed_per_benchmark.csv",
            (
                "student",
                "benchmark",
                "eopd_full_avg8_mean",
                "eopd_full_avg8_sd",
                "vd_opd_avg8_mean",
                "vd_opd_avg8_sd",
                *_GAIN,
            ),
            table_e01,
        ),
        TableSpec(
            "E02",
            "01_main_results/E02_complete_table6_qwen3_1.7b.csv",
            ("method", *_BENCHMARK_COLUMNS),
            _complete_table("qwen3_1p7b"),
        ),
        TableSpec(
            "E03",
            "01_main_results/E03_complete_table7_qwen3_4b.csv",
            ("method", *_BENCHMARK_COLUMNS),
            _complete_table("qwen3_4b"),
        ),
        TableSpec(
            "E04",
            "01_main_results/E04_complete_table8_llama3.2_3b.csv",
            ("method", *_BENCHMARK_COLUMNS),
            _complete_table("llama3p2_3b"),
        ),
        TableSpec(
            "E05",
            "01_main_results/E05_teacher_vs_student_math500.csv",
            ("model", "math500_avg8", "math500_pass8"),
            table_e05,
        ),
        TableSpec(
            "E06",
            "02_compute_controls/E06_compute_matched_controls.csv",
            (
                "comparison",
                "budget_matched",
                "baseline_flops_1e18",
                "vd_flops_1e18",
                "baseline_macro_avg8_mean",
                "baseline_macro_avg8_sd",
                "vd_macro_avg8_mean",
                "vd_macro_avg8_sd",
                *_GAIN,
            ),
            table_e06,
        ),
        TableSpec(
            "E07",
            "03_ablations/E07_target_weight_factorial.csv",
            ("target", "weight", "macro_avg8_mean", "macro_avg8_sd", *_GAIN),
            table_e07,
        ),
        TableSpec(
            "E08",
            "03_ablations/E08_number_of_views_K2_vs_K4.csv",
            (
                "setting",
                "macro_avg8_mean",
                "macro_avg8_sd",
                "worst_view",
                "flops_1e18",
                "gpu_hours",
                "gain_over_k2",
                "gain_over_k2_ci_low",
                "gain_over_k2_ci_high",
            ),
            table_e08,
        ),
        TableSpec(
            "E10",
            "03_ablations/E10_view_pool_sensitivity.csv",
            (
                "view_pool",
                "mean_z",
                "eopd_full",
                "eopd_2v_mean",
                "eopd_2v_sd",
                "vd_opd_mean",
                "vd_opd_sd",
                "vd_minus_2v",
                "vd_minus_2v_ci_low",
                "vd_minus_2v_ci_high",
                "worst_view_eopd_2v",
                "worst_view_vd_opd",
            ),
            table_e10,
        ),
        TableSpec(
            "E12",
            "05_robustness/E12_unseen_wrapper_macro.csv",
            (
                "method",
                "canonical",
                "mean_view",
                "worst_view",
                "worst_template",
                "canonical_minus_mean",
                "mean_minus_worst",
                "greedy_answer_flips_pct",
            ),
            table_e12,
        ),
        TableSpec(
            "E13",
            "05_robustness/E13_unseen_wrapper_per_benchmark.csv",
            (
                "benchmark",
                *(
                    f"{prefix}_{metric}"
                    for _, prefix in ROBUSTNESS_METHODS.values()
                    for metric in ("mean_view", "worst_view")
                ),
            ),
            table_e13,
        ),
        TableSpec(
            "E14",
            "05_robustness/E14_student_augmentation_baseline.csv",
            (
                "method",
                "flops_1e18",
                "canonical",
                "prefix_mean_view",
                "prefix_worst_view",
                "structural_mean_view",
                "structural_worst_view",
                "paraphrase_mean_view",
                "paraphrase_worst_view",
            ),
            table_e14,
        ),
        TableSpec(
            "E15",
            "05_robustness/E15_heldout_structural_and_paraphrase.csv",
            (
                "family",
                "method",
                "mean_view",
                "worst_view",
                "worst_template",
                "answer_flips_pct",
                "parse_failures_pct",
            ),
            table_e15,
        ),
    )
}


def build_table(experiment: str, results: ResultSet) -> list[dict[str, str]]:
    """Rows of one table; every row has exactly the columns of its :class:`TableSpec`."""
    spec = TABLES[experiment]
    rows = spec.build(results)
    for row in rows:
        if tuple(row) != spec.columns:
            raise RuntimeError(f"{experiment}: row columns {tuple(row)} differ from the schema")
    return rows


def write_csv(path: str | Path, columns: Sequence[str], rows: Sequence[Mapping[str, str]]) -> None:
    """Write rows as a UTF-8 CSV file with one header row.

    The dialect is the default of Python's ``csv`` module (CRLF line ends, minimal quoting),
    the same as that of the reported supplementary tables.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        writer.writerows(rows)


def write_tables(
    results: ResultSet, output_dir: str | Path, experiments: Sequence[str] | None = None
) -> dict[str, Path | None]:
    """Build the requested tables and write those that have at least one row.

    Returns ``{experiment: written path}``; the value is ``None`` for a table whose inputs
    are not present in ``results``.
    """
    written: dict[str, Path | None] = {}
    for experiment in experiments or list(TABLES):
        rows = build_table(experiment, results)
        if not rows:
            written[experiment] = None
            continue
        path = Path(output_dir) / TABLES[experiment].path
        write_csv(path, TABLES[experiment].columns, rows)
        written[experiment] = path
    return written
