"""Tables in the layout of the supplementary experiments, built from evaluation files.

The evaluation files used here are synthetic (random correctness drawn in this module); the
tests check the table layout and that every cell is computed from the files, not any value
reported for the paper.
"""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from vdopd.eval.bootstrap import paired_question_bootstrap, stratified_paired_bootstrap
from vdopd.eval.metrics import seed_mean_and_sd, view_robustness
from vdopd.eval.tables import (
    BENCHMARK_LABELS,
    TABLES,
    ResultSet,
    build_table,
    load_results,
    macro_metric,
    number,
    question_accuracy,
    write_tables,
)

REPOSITORY = Path(__file__).resolve().parents[1]
SEEDS = (11, 22, 33)
SIZES = {"math500": 14, "amc23": 8, "aime24": 6, "aime25": 6, "minerva": 9, "olympiadbench": 11}
FAMILIES = {
    "prefix": [f"unseen_{i}" for i in range(4)],
    "structural": [f"structural_{i}" for i in range(4)],
    "paraphrase": [f"paraphrase_{i}" for i in range(4)],
}
MAIN_METHODS = (
    "eopd_full", "eopd_2v", "vdopd", "vdopd_budget", "vdopd_k4", "vdopd_k4_budget",
    "eopd_full_studentaug", "eopd_2v_studentaug", "eopd_2v_paraphrase", "vdopd_paraphrase",
    "eopd_2v_lowquality", "vdopd_lowquality", "ablation_arithmetic",
    "ablation_weighted_arithmetic", "ablation_geometric", "ablation_shuffled_weight",
    "ablation_batch_mean_weight", "frozen_teacher", "initial_student", "gkd_fkl", "eopd_top16",
)  # fmt: skip
OTHER_METHODS = ("frozen_teacher", "eopd_full", "vdopd")
K = 8


def synthetic_evaluation(rng: np.random.Generator, seed: int, skill: float) -> dict:
    """An evaluation record in the format of ``vdopd eval --output`` with random results."""
    record: dict = {"benchmarks": {}, "seed": seed}
    for benchmark, size in SIZES.items():
        difficulty = rng.random(size)
        counts = {"original": rng.binomial(K, np.clip(skill * difficulty * 1.6, 0, 1))}
        families = {}
        for family, names in FAMILIES.items():
            for name in names:
                counts[name] = rng.binomial(K, np.clip(skill * difficulty * 1.3, 0, 1))
            stacked = np.stack([np.arange(K) < counts[name][:, None] for name in names])
            metrics = view_robustness(stacked)
            metrics["greedy_answer_flips"] = float(rng.random() * 0.4)
            metrics["parse_failures"] = float(rng.random() * 0.03)
            metrics["views"] = names
            families[family] = metrics
        entry = {
            "num_questions": size,
            "num_samples": K,
            "avg_at_k": float(counts["original"].mean() / K),
            "pass_at_k": float((counts["original"] > 0).mean()),
            "views": {name: float(c.mean() / K) for name, c in counts.items()},
            "unseen": {
                k: families["prefix"][k] for k in ("mean_view", "worst_view", "worst_template")
            },
            "families": families,
            "questions": {
                "ids": [f"{benchmark}/{i}" for i in range(size)],
                "correct_counts": {
                    name: [int(c) for c in values] for name, values in counts.items()
                },
            },
        }
        record["benchmarks"][benchmark] = entry
    return record


@pytest.fixture(scope="module")
def layout(tmp_path_factory) -> dict[str, Path]:
    """Synthetic results/ and runs/ folders in the layout of scripts/run_experiment.sh."""
    root = tmp_path_factory.mktemp("tables")
    rng = np.random.default_rng(0)
    groups = {"qwen3_1p7b": MAIN_METHODS, "qwen3_4b": OTHER_METHODS, "llama3p2_3b": OTHER_METHODS}
    for group, methods in groups.items():
        for index, method in enumerate(methods):
            skill = 0.35 + 0.02 * index
            for seed in SEEDS:
                path = root / "results" / group / method / f"seed{seed}.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(synthetic_evaluation(rng, seed, skill)))
                run = root / "runs" / group / method / f"seed{seed}"
                run.mkdir(parents=True)
                summary = {"flops_total": (4.0 + index + seed / 100) * 1e18, "mean_mass": 0.7}
                (run / "summary.json").write_text(json.dumps(summary))
    return {"results": root / "results", "runs": root / "runs", "root": root}


@pytest.fixture(scope="module")
def results(layout) -> ResultSet:
    return ResultSet(load_results(layout["results"]), runs_dir=layout["runs"], num_resamples=400)


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with open(path, encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    return list(reader.fieldnames), rows


def percent(runs: dict, benchmark: str, metric: str) -> tuple[float, float]:
    return seed_mean_and_sd([100 * runs[s]["benchmarks"][benchmark][metric] for s in SEEDS])


def test_every_table_is_written_with_its_schema(results, layout):
    written = write_tables(results, layout["root"] / "out")
    assert set(written) == set(TABLES) and all(path is not None for path in written.values())
    for experiment, path in written.items():
        header, rows = read_csv(path)
        assert tuple(header) == TABLES[experiment].columns
        assert path == layout["root"] / "out" / TABLES[experiment].path
        assert rows and all(set(row) == set(header) for row in rows)
    expected_rows = {"E01": 18, "E02": 6, "E03": 3, "E04": 3, "E05": 3, "E06": 3, "E07": 6,
                     "E08": 3, "E10": 3, "E12": 5, "E13": 6, "E14": 5, "E15": 10}  # fmt: skip
    assert {e: len(read_csv(p)[1]) for e, p in written.items()} == expected_rows
    again = write_tables(results, layout["root"] / "again")
    for experiment, path in written.items():  # deterministic, including the bootstrap
        assert path.read_text() == again[experiment].read_text()


@pytest.mark.parametrize("experiment", sorted(TABLES))
def test_columns_equal_those_of_the_reported_tables(experiment):
    """File names and column names follow the reported supplementary tables exactly."""
    path = REPOSITORY / "results" / "supplementary" / TABLES[experiment].path
    if not path.is_file():
        pytest.skip("the reported table is not present")
    header, _ = read_csv(path)
    assert tuple(header) == TABLES[experiment].columns


def test_e01_cells_are_seed_statistics_and_paired_bootstrap_intervals(results):
    rows = build_table("E01", results)
    assert [row["student"] for row in rows[::6]] == ["Qwen3-1.7B", "Qwen3-4B", "Llama-3.2-3B"]
    assert [row["benchmark"] for row in rows[:6]] == list(BENCHMARK_LABELS.values())
    row = rows[0]
    baseline, method = (results.runs(name) for name in ("eopd_full", "vdopd"))
    a, b = percent(baseline, "math500", "avg_at_k"), percent(method, "math500", "avg_at_k")
    assert row["eopd_full_avg8_mean"] == f"{a[0]:.1f}" and row["eopd_full_avg8_sd"] == f"{a[1]:.2f}"
    assert row["vd_opd_avg8_mean"] == f"{b[0]:.1f}" and row["vd_opd_avg8_sd"] == f"{b[1]:.2f}"
    assert row["vd_gain"] == f"{b[0] - a[0]:+.1f}"
    per_seed = [
        [question_accuracy(runs[s], "math500")[1] for s in SEEDS] for runs in (baseline, method)
    ]
    ci = paired_question_bootstrap(*per_seed, num_resamples=400, seed=0)
    assert row["vd_gain_ci_low"] == number(100 * ci.low, 1, True)
    assert row["vd_gain_ci_high"] == number(100 * ci.high, 1, True)
    assert float(row["vd_gain_ci_low"]) <= float(row["vd_gain"]) <= float(row["vd_gain_ci_high"])
    assert ci.estimate == pytest.approx((b[0] - a[0]) / 100)


def test_complete_tables_report_avg_and_pass_for_every_available_method(results):
    rows = build_table("E02", results)
    assert [row["method"] for row in rows] == [
        "Frozen teacher", "Initial student", "GKD-FKL", "EOPD(top-16)", "EOPD(full)", "VD-OPD",
    ]  # fmt: skip
    teacher = results.runs("frozen_teacher")
    assert rows[0]["aime24_avg8"] == f"{percent(teacher, 'aime24', 'avg_at_k')[0]:.1f}"
    assert rows[0]["aime24_pass8"] == f"{percent(teacher, 'aime24', 'pass_at_k')[0]:.1f}"
    assert all(float(r["math500_avg8"]) <= float(r["math500_pass8"]) for r in rows)
    assert [row["method"] for row in build_table("E04", results)] == [
        "Frozen teacher", "EOPD(full)", "VD-OPD",
    ]  # fmt: skip
    e05 = build_table("E05", results)
    assert [row["model"] for row in e05] == [
        "Qwen3-8B teacher", "Qwen3-4B, EOPD(full)", "Qwen3-4B, VD-OPD",
    ]  # fmt: skip
    students = results.runs("vdopd", "qwen3_4b")
    assert e05[2]["math500_pass8"] == f"{percent(students, 'math500', 'pass_at_k')[0]:.1f}"


def test_e06_macro_gain_flops_and_stratified_interval(results):
    rows = build_table("E06", results)
    assert [row["comparison"] for row in rows] == [
        "EOPD(full) vs. VD-Budget", "EOPD(2V) vs. VD-OPD", "EOPD(full) vs. VD-OPD",
    ]  # fmt: skip
    assert rows[0]["budget_matched"] == "equal total FLOPs"
    baseline, method = (results.runs(name) for name in ("eopd_2v", "vdopd"))
    macros = [
        seed_mean_and_sd([100 * macro_metric(runs[s], "avg_at_k") for s in SEEDS])
        for runs in (baseline, method)
    ]
    row = rows[1]
    assert row["baseline_macro_avg8_mean"] == f"{macros[0][0]:.2f}"
    assert row["vd_macro_avg8_sd"] == f"{macros[1][1]:.2f}"
    assert row["vd_gain"] == f"{macros[1][0] - macros[0][0]:+.1f}"
    pairs = {
        benchmark: tuple(
            [question_accuracy(runs[s], benchmark)[1] for s in SEEDS] for runs in (baseline, method)
        )
        for benchmark in SIZES
    }
    ci = stratified_paired_bootstrap(pairs, num_resamples=400, seed=0)
    assert row["vd_gain_ci_low"] == number(100 * ci.low, 1, True)
    assert row["vd_gain_ci_high"] == number(100 * ci.high, 1, True)
    # FLOPs are the seed mean of the recorded totals, in units of 1e18.
    index = MAIN_METHODS.index("eopd_2v")
    assert row["baseline_flops_1e18"] == f"{4.0 + index + 0.22:.1f}"
    without_runs = ResultSet(results.evaluations, num_resamples=50)
    assert build_table("E06", without_runs)[1]["baseline_flops_1e18"] == ""


def test_e07_e08_e10_reference_rows_and_gain_directions(results):
    e07 = build_table("E07", results)
    assert [(row["target"], row["weight"]) for row in e07] == [
        ("Arithmetic", "1"), ("Arithmetic", "Z"), ("Geometric", "1"),
        ("Geometric", "shuffled Z"), ("Geometric", "batch-mean Z"),
        ("Geometric (VD-OPD)", "state-specific Z"),
    ]  # fmt: skip
    assert (e07[-1]["vd_gain"], e07[-1]["vd_gain_ci_low"], e07[-1]["vd_gain_ci_high"]) == ("",) * 3
    vd = float(e07[-1]["macro_avg8_mean"])
    assert float(e07[0]["vd_gain"]) == pytest.approx(
        vd - float(e07[0]["macro_avg8_mean"]), abs=0.11
    )
    e08 = build_table("E08", results)
    assert [row["setting"] for row in e08] == ["K=2", "K=4, same updates", "K=4, same FLOPs"]
    assert e08[0]["gain_over_k2"] == "" and all(row["gpu_hours"] == "" for row in e08)
    assert float(e08[1]["gain_over_k2"]) == pytest.approx(
        float(e08[1]["macro_avg8_mean"]) - float(e08[0]["macro_avg8_mean"]), abs=0.11
    )
    assert float(e08[0]["worst_view"]) <= float(e08[0]["macro_avg8_mean"]) + 20
    e10 = build_table("E10", results)
    assert [row["view_pool"] for row in e10] == [
        "Simple templates", "Random valid paraphrases", "Lower-quality, task-preserving",
    ]  # fmt: skip
    assert all(row["mean_z"] == "0.700" for row in e10)
    assert len({row["eopd_full"] for row in e10}) == 1  # the single-view baseline is shared
    assert e10[0]["vd_opd_mean"] == e07[-1]["macro_avg8_mean"]


def test_robustness_tables(results):
    e12 = build_table("E12", results)
    assert [row["method"] for row in e12] == [
        "EOPD(full)", "EOPD(2V)", "EOPD(full)+StudentAug", "EOPD(2V)+StudentAug", "VD-OPD",
    ]  # fmt: skip
    runs = results.runs("vdopd")
    prefix = [
        np.mean([runs[s]["benchmarks"][b]["families"]["prefix"][key] for b in SIZES])
        for key in ("mean_view", "worst_view", "worst_template", "greedy_answer_flips")
        for s in SEEDS
    ]
    means = 100 * np.reshape(prefix, (4, 3)).mean(axis=1)
    row = e12[-1]
    assert [row[c] for c in ("mean_view", "worst_view", "worst_template")] == [
        f"{value:.1f}" for value in means[:3]
    ]
    assert row["greedy_answer_flips_pct"] == f"{means[3]:.0f}"
    canonical = np.mean([macro_metric(runs[s], "avg_at_k") for s in SEEDS]) * 100
    assert row["canonical"] == f"{canonical:.1f}"
    assert row["canonical_minus_mean"] == number(canonical - means[0])
    assert row["mean_minus_worst"] == number(means[0] - means[1])
    assert float(row["worst_view"]) <= float(row["worst_template"]) <= float(row["mean_view"])
    e13 = build_table("E13", results)
    assert [row["benchmark"] for row in e13] == list(BENCHMARK_LABELS.values())
    cell = np.mean(
        [runs[s]["benchmarks"]["minerva"]["families"]["prefix"]["worst_view"] for s in SEEDS]
    )
    assert e13[4]["vd_opd_worst_view"] == f"{100 * cell:.1f}"
    e14 = build_table("E14", results)
    assert (
        e14[-1]["prefix_mean_view"] == row["mean_view"] and e14[-1]["canonical"] == row["canonical"]
    )
    assert e14[0]["flops_1e18"] == f"{4.0 + 0.22:.1f}"
    e15 = build_table("E15", results)
    assert [r["family"] for r in e15] == ["Structural"] * 5 + ["Paraphrase"] * 5
    assert e15[4]["mean_view"] == e14[-1]["structural_mean_view"]
    assert e15[9]["worst_view"] == e14[-1]["paraphrase_worst_view"]
    parse = np.mean(
        [
            runs[s]["benchmarks"][b]["families"]["paraphrase"]["parse_failures"]
            for b in SIZES
            for s in SEEDS
        ]
    )
    assert e15[9]["parse_failures_pct"] == f"{100 * parse:.1f}"


def test_missing_inputs_leave_cells_empty_or_skip_the_table(results, layout):
    # Evaluation files without per-question results (older format): gains, no intervals.
    stripped = json.loads(json.dumps(results.evaluations, default=str))
    stripped = {
        group: {
            method: {int(seed): record for seed, record in runs.items()}
            for method, runs in methods.items()
        }
        for group, methods in stripped.items()
    }
    for methods in stripped.values():
        for runs in methods.values():
            for record in runs.values():
                for entry in record["benchmarks"].values():
                    del entry["questions"], entry["families"]
    old_format = ResultSet(stripped, num_resamples=50)
    row = build_table("E01", old_format)[0]
    assert row["vd_gain"] != "" and row["vd_gain_ci_low"] == "" and row["vd_gain_ci_high"] == ""
    e12 = build_table("E12", old_format)[0]  # the prefix family falls back to "unseen"
    assert e12["mean_view"] != "" and e12["greedy_answer_flips_pct"] == ""
    assert build_table("E15", old_format) == []
    # Only one student and two methods: tables about other runs are not written.
    partial = {
        "qwen3_1p7b": {m: results.evaluations["qwen3_1p7b"][m] for m in ("eopd_full", "vdopd")}
    }
    written = write_tables(ResultSet(partial, num_resamples=50), layout["root"] / "partial")
    assert written["E03"] is None and written["E10"] is None and written["E01"] is not None
    assert written["E07"] is None and written["E08"] is None  # reference rows only
    assert len(read_csv(written["E06"])[1]) == 1 and len(read_csv(written["E01"])[1]) == 6
    # Different question sets cannot be paired.
    broken = json.loads(json.dumps(partial, default=str))
    broken = {"qwen3_1p7b": {m: {int(s): r for s, r in runs.items()} for m, runs in
                             broken["qwen3_1p7b"].items()}}  # fmt: skip
    broken["qwen3_1p7b"]["vdopd"][11]["benchmarks"]["math500"]["questions"]["ids"][0] = "other"
    with pytest.raises(ValueError, match="different questions"):
        build_table("E01", ResultSet(broken, num_resamples=50))


def test_number_formatting():
    assert number(None) == "" and number(1.25, 1) in ("1.2", "1.3")
    assert number(-0.04, 1, signed=True) == "+0.0" and number(0.04, 1, signed=True) == "+0.0"
    assert number(-1.26, 1, signed=True) == "-1.3" and number(36.666, 2) == "36.67"


def test_tables_script(layout, capsys):
    path = REPOSITORY / "scripts" / "supplementary_tables.py"
    spec = importlib.util.spec_from_file_location("supplementary_tables", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    out = layout["root"] / "cli"
    module.main(
        ["--results-dir", str(layout["results"]), "--runs-dir", str(layout["runs"]),
         "--output-dir", str(out), "--tables", "E01", "E13", "--resamples", "100"]
    )  # fmt: skip
    printed = capsys.readouterr().out
    assert "E01:" in printed and "E13:" in printed and "E06" not in printed
    assert sorted(p.name for p in out.rglob("*.csv")) == [
        "E01_three_seed_per_benchmark.csv",
        "E13_unseen_wrapper_per_benchmark.csv",
    ]
    with pytest.raises(SystemExit):
        module.main(["--results-dir", str(layout["root"] / "none"), "--output-dir", str(out)])


def test_run_records_fall_back_to_the_training_log(tmp_path, results):
    from vdopd.eval.tables import run_record

    run = tmp_path / "qwen3_1p7b" / "vdopd" / "seed11"
    run.mkdir(parents=True)
    lines = [
        {"num_positions": 10, "mass_total": 7.0, "flops_total": 1e18},
        {"num_positions": 0, "flops_total": 2e18},  # an empty batch has no mass
        {"num_positions": 30, "mass_total": 15.0, "flops_total": 3e18},
    ]
    (run / "metrics.jsonl").write_text("".join(json.dumps(line) + "\n" for line in lines))
    logged = ResultSet(results.evaluations, runs_dir=tmp_path)
    assert run_record(logged, "vdopd", "flops_total") == 3e18
    assert run_record(logged, "vdopd", "mean_mass") == pytest.approx(22.0 / 40)
    assert run_record(logged, "eopd_full", "flops_total") is None
