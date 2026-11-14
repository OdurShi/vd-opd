"""Dataset files under ``data/``: checksums, local-first loading, Hub fallback and splits."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from vdopd.data import benchmarks, math_train, paths
from vdopd.data.benchmarks import BENCHMARKS, SIX_MATH, benchmark_file, load_benchmark
from vdopd.data.math_train import (
    MATH_TRAIN_FILE,
    SPLIT_DIR,
    MathSplits,
    deduplicate,
    load_math_train,
    prepare_diagnostic_split,
    prepare_math_splits,
    remove_overlaps,
    split_diagnostic,
    split_parameters,
    stratified_split,
    write_split_files,
)
from vdopd.data.problems import Problem

REPOSITORY = Path(__file__).resolve().parents[1]
DATA = REPOSITORY / "data"
MANIFEST = DATA / "manifest.json"
needs_data = pytest.mark.skipif(not MANIFEST.is_file(), reason="data/ is not present")


@pytest.fixture(autouse=True)
def default_data_dir(monkeypatch):
    """Every test starts from the repository's own data directory."""
    monkeypatch.delenv(paths.ENV_VAR, raising=False)
    paths.set_data_dir(None)
    yield
    paths.set_data_dir(None)


@pytest.fixture
def no_hub(monkeypatch):
    """Fail the test if a loader tries to reach the Hugging Face Hub."""

    def refuse(*args, **kwargs):
        raise AssertionError("unexpected download")

    monkeypatch.setattr(benchmarks, "fetch_benchmark_records", refuse)
    monkeypatch.setattr(math_train, "fetch_math_train_records", refuse)


def manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))["files"]


def test_data_directory_resolution(monkeypatch, tmp_path):
    assert paths.data_dir() == DATA
    monkeypatch.setenv(paths.ENV_VAR, str(tmp_path / "env"))
    assert paths.data_dir() == tmp_path / "env"
    paths.set_data_dir(tmp_path / "explicit")  # an explicit override wins
    assert paths.data_dir() == tmp_path / "explicit"
    assert paths.local_file("benchmarks/math500.jsonl") is None
    paths.write_jsonl(tmp_path / "explicit" / "x.jsonl", [{"a": 1}, {"a": "é"}])
    assert paths.read_jsonl(paths.local_file("x.jsonl")) == [{"a": 1}, {"a": "é"}]
    paths.write_lines(tmp_path / "explicit" / "ids.txt", ["a", "b"])
    assert paths.read_lines(tmp_path / "explicit" / "ids.txt") == ["a", "b"]


@needs_data
def test_shipped_files_match_the_manifest_checksums():
    entries = manifest()
    assert MATH_TRAIN_FILE in entries
    for relative, entry in entries.items():
        path = DATA / relative
        if entry.get("shipped", True):
            assert path.is_file(), f"{relative} should be distributed with the repository"
        if path.is_file():
            assert paths.sha256(path) == entry["sha256"], relative
    for path in sorted((DATA / "licenses").glob("*.txt")):
        assert path.stat().st_size > 500


@needs_data
def test_manifest_records_pinned_sources():
    entries = manifest()
    for name, spec in BENCHMARKS.items():
        entry = entries[benchmark_file(name)]
        assert (entry["source"], entry["split"], entry["revision"]) == (
            spec.path,
            spec.split,
            spec.revision,
        )
    train = entries[MATH_TRAIN_FILE]
    assert train["source"] == math_train.MATH_TRAIN_PATH
    assert train["revision"] == math_train.MATH_TRAIN_REVISION and train["num_examples"] == 7500
    expected = {"math500": 500, "amc23": 40, "aime24": 30, "aime25": 30, "minerva": 272}
    expected.update(olympiadbench=674, gsm8k=1319)
    assert {n: entries[benchmark_file(n)]["num_examples"] for n in BENCHMARKS} == expected


@needs_data
def test_shipped_datasets_load_without_network(no_hub):
    entries = manifest()
    for name in BENCHMARKS:
        if not (DATA / benchmark_file(name)).is_file():
            continue
        problems = load_benchmark(name)
        assert len(problems) == entries[benchmark_file(name)]["num_examples"]
        assert all(p.question and p.answer for p in problems)
        assert len(load_benchmark(name, limit=3)) == 3
    train = load_math_train()
    assert len(train) == 7500 and len({p.id for p in train}) == 7500
    assert train[0].id == "math_train/algebra/0" and train[0].level.startswith("Level")
    assert {p.subject for p in train} == {
        "Algebra",
        "Counting & Probability",
        "Geometry",
        "Intermediate Algebra",
        "Number Theory",
        "Prealgebra",
        "Precalculus",
    }


def test_loaders_fall_back_to_the_hub_when_a_file_is_absent(monkeypatch, tmp_path):
    monkeypatch.setenv(paths.ENV_VAR, str(tmp_path))
    calls = []

    def fake_benchmark(name, revision=None):
        calls.append(name)
        return [{"id": "7", "question": "1+1?", "answer": "2", "url": ""}]

    def fake_train(revision=None):
        calls.append("math_train")
        row = {"problem": "q", "level": "Level 1", "type": "Algebra", "solution": "\\boxed{3}"}
        return [{"id": "math_train/algebra/0", **row}]

    monkeypatch.setattr(benchmarks, "fetch_benchmark_records", fake_benchmark)
    monkeypatch.setattr(math_train, "fetch_math_train_records", fake_train)
    assert load_benchmark("amc23") == [Problem("amc23/7", "1+1?", "2")]
    assert load_math_train() == [Problem("math_train/algebra/0", "q", "3", "Algebra", "Level 1")]
    assert calls == ["amc23", "math_train"]
    # Once a file exists in the data directory it is used instead.
    paths.write_jsonl(
        tmp_path / benchmark_file("amc23"), [{"id": "9", "question": "x", "answer": "1"}]
    )
    assert load_benchmark("amc23") == [Problem("amc23/9", "x", "1")]
    assert calls == ["amc23", "math_train"]
    with pytest.raises(KeyError):
        load_benchmark("unknown")


@needs_data
def test_stored_partition_reproduces_appendix_b1(no_hub):
    """The index files equal a recomputation: deduplicate, remove overlaps, stratify (seed 42)."""
    problems = load_math_train()
    available = [name for name in SIX_MATH if (DATA / benchmark_file(name)).is_file()]
    questions = [p.question for name in available for p in load_benchmark(name)]
    cleaned = remove_overlaps(deduplicate(problems), questions)
    splits = stratified_split(cleaned, num_diagnostic=500, num_development=500, seed=42)
    update, measurement = split_diagnostic(splits.diagnostic, num_update=250, seed=42)

    def stored(name: str) -> list[str]:
        return paths.read_lines(DATA / SPLIT_DIR / f"{name}.txt")

    assert stored("deduplicated") == [p.id for p in cleaned] and len(cleaned) == 7499
    assert stored("optimization") == [p.id for p in splits.optimization]
    assert stored("development") == [p.id for p in splits.development]
    assert stored("diagnostic") == [p.id for p in splits.diagnostic]
    assert stored("diagnostic_update") == [p.id for p in update]
    assert stored("diagnostic_measurement") == [p.id for p in measurement]
    sizes = [len(stored(n)) for n in ("optimization", "development", "diagnostic")]
    assert sizes == [6499, 500, 500]
    everything = stored("optimization") + stored("development") + stored("diagnostic")
    assert sorted(everything) == sorted(stored("deduplicated"))  # a partition of the cleaned set
    info = json.loads((DATA / SPLIT_DIR / "split_info.json").read_text())
    assert info["seed"] == 42 and info["overlap_benchmarks"] == sorted(SIX_MATH)
    assert info["sizes"]["optimization"] == 6499


@needs_data
def test_prepare_math_splits_uses_the_index_files(no_hub):
    splits = prepare_math_splits(SIX_MATH)  # default parameters: stored partition, no download
    assert [len(splits.optimization), len(splits.development), len(splits.diagnostic)] == [
        6499,
        500,
        500,
    ]
    assert splits.optimization[0].id == paths.read_lines(DATA / SPLIT_DIR / "optimization.txt")[0]
    update, measurement = prepare_diagnostic_split(splits.diagnostic)
    assert [p.id for p in update] == paths.read_lines(DATA / SPLIT_DIR / "diagnostic_update.txt")
    assert len(measurement) == 250
    # Other parameters do not match the stored files, so the partition is recomputed.
    other = prepare_math_splits((), seed=7)
    assert [p.id for p in other.diagnostic] != [p.id for p in splits.diagnostic]
    assert len(other.optimization) == 6499


def test_split_files_round_trip(monkeypatch, tmp_path):
    paths.set_data_dir(tmp_path)
    rows = [
        {
            "id": f"math_train/algebra/{i}",
            "problem": f"q{i}",
            "level": f"Level {1 + i % 2}",
            "type": "Algebra",
            "solution": f"\\boxed{{{i}}}",
        }
        for i in range(40)
    ]
    paths.write_jsonl(tmp_path / MATH_TRAIN_FILE, rows)
    problems = load_math_train()
    splits = stratified_split(problems, num_diagnostic=8, num_development=6, seed=3)
    parameters = split_parameters(8, 6, 3, ())
    written = write_split_files(problems, splits, parameters)
    assert "split_info.json" in written and len(written) == 7
    again = prepare_math_splits((), num_diagnostic=8, num_development=6, seed=3)
    assert isinstance(again, MathSplits)
    assert [p.id for p in again.optimization] == [p.id for p in splits.optimization]
    assert [p.id for p in again.diagnostic] == [p.id for p in splits.diagnostic]
    update, measurement = prepare_diagnostic_split(again.diagnostic, seed=3)
    assert len(update) == len(measurement) == 4


def test_download_script_covers_every_dataset():
    spec = importlib.util.spec_from_file_location(
        "download_data", REPOSITORY / "scripts" / "download_data.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert set(module.LICENSES) == {"math_train", *BENCHMARKS}
    for name in module.LICENSES:
        entry = module.describe(name)
        assert entry["revision"] and entry["file"].endswith(".jsonl")
    not_shipped = {name for name, (_, shipped) in module.LICENSES.items() if not shipped}
    ignored = (REPOSITORY / ".gitignore").read_text()
    for name in not_shipped:
        assert f"/data/{benchmark_file(name)}" in ignored
