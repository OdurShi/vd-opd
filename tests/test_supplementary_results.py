"""The supplementary experiment tables: loading, rendering and cross-table consistency.

The tests never compare against literal results. They check the structure of the collection,
that tables reporting the same quantity agree, that the rendered document is current, and
that the checks notice an edited value.
"""

from __future__ import annotations

import csv
import importlib.util
import re
import shutil
from pathlib import Path

import pytest

from vdopd import supplementary
from vdopd.supplementary import (
    BENCHMARKS,
    Column,
    check_consistency,
    embed_tables,
    load,
    load_index,
    parse_distribution,
    parse_number,
    render_markdown,
    supplementary_dir,
)

REPOSITORY = Path(__file__).resolve().parents[1]
DOCUMENT = REPOSITORY / "docs" / "supplementary_results.md"


@pytest.fixture(scope="module")
def index():
    return load_index()


@pytest.fixture
def copy(tmp_path):
    """An editable copy of the data directory."""
    target = tmp_path / "supplementary"
    shutil.copytree(supplementary_dir(), target)
    return target


def _rewrite(path: Path, edit) -> None:
    """Apply ``edit(rows)`` to the rows of a CSV file and write it back."""
    with open(path, encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames, rows = list(reader.fieldnames), list(reader)
    edit(rows)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def test_index_lists_eighteen_experiments_in_order(index):
    assert list(index.experiments) == [f"E{n:02d}" for n in range(1, 19)]
    assert [group.dir for group in index.groups] == sorted(group.dir for group in index.groups)
    for experiment in index:
        assert experiment.rows, experiment.id
        assert experiment.title and experiment.setup and experiment.findings, experiment.id
        assert experiment.path.is_file()
        assert experiment.relative_path.startswith(experiment.group + "/")


def test_every_csv_field_is_displayed_exactly_once(index):
    for experiment in index:
        displayed = [name for column in experiment.columns for name in column.fields]
        assert sorted(displayed) == sorted(experiment.fieldnames), experiment.id


def test_cells_are_numbers_or_text_labels(index):
    """Composed columns hold numbers (or blanks); a row never mixes blank and filled parts."""
    for experiment in index:
        for row in experiment.rows:
            for column in experiment.columns:
                if column.kind == "col":
                    continue
                cells = [row[name].strip() for name in column.fields]
                assert all(cells) or not any(cells), (experiment.id, column.header)
                assert all(parse_number(cell) is not None for cell in cells if cell)


def test_per_method_tables_cover_the_six_benchmarks(index):
    for experiment_id in ("E02", "E03", "E04"):
        experiment = index[experiment_id]
        methods = [row["method"] for row in experiment.rows]
        assert len(methods) == len(set(methods))
        assert {"Frozen teacher", "Initial student", "EOPD(full)", "VD-OPD"} <= set(methods)
        for benchmark in BENCHMARKS:
            for metric in ("avg8", "pass8"):
                values = experiment.values(f"{benchmark.lower()}_{metric}")
                assert all(value is not None and 0.0 <= value <= 100.0 for value in values)
        for row in experiment.rows:
            for benchmark in BENCHMARKS:
                prefix = benchmark.lower()
                # Pass@8 counts a question as solved if any sample is correct, so it bounds Avg@8.
                assert float(row[f"{prefix}_pass8"]) >= float(row[f"{prefix}_avg8"])


def test_tables_agree_with_each_other(index):
    assert check_consistency(index) == []


def test_document_is_current(index):
    assert DOCUMENT.is_file(), "run scripts/render_supplementary.py"
    assert DOCUMENT.read_text(encoding="utf-8") == render_markdown(index)


def test_document_links_resolve():
    text = DOCUMENT.read_text(encoding="utf-8")
    for experiment in load_index():
        assert f'<a id="{experiment.id.lower()}"></a>' in text
        assert (DOCUMENT.parent / "../results/supplementary" / experiment.relative_path).is_file()


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, REPOSITORY / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_render_script_check_mode():
    assert _load_script("render_supplementary").main(["--check"]) == 0


def test_figures_are_drawn_from_the_tables(tmp_path):
    pytest.importorskip("matplotlib")
    module = _load_script("plot_supplementary")
    written = module.draw(tmp_path, dpi=50)
    assert [path.stem for path in written] == [name for name, _ in module.FIGURES.values()]
    assert all(path.stat().st_size > 2_000 for path in written)
    shipped = REPOSITORY / "assets" / "supplementary"
    assert all((shipped / path.name).is_file() for path in written)


def test_edited_score_is_reported(copy):
    def edit(rows):
        rows[0]["vd_opd_avg8_mean"] = f"{float(rows[0]['vd_opd_avg8_mean']) + 5:.1f}"

    _rewrite(copy / "01_main_results" / "E01_three_seed_per_benchmark.csv", edit)
    problems = check_consistency(load_index(copy))
    assert any("VD-OPD Avg@8" in problem for problem in problems)
    assert any("gain vs difference of means" in problem for problem in problems)


def test_edited_macro_average_is_reported(copy):
    def edit(rows):
        for row in rows:
            row["vd_macro_avg8_mean"] = f"{float(row['vd_macro_avg8_mean']) + 2:.2f}"

    _rewrite(copy / "02_compute_controls" / "E06_compute_matched_controls.csv", edit)
    problems = check_consistency(load_index(copy))
    assert any(p.startswith("E06") and "macro average of E02" in p for p in problems)


def test_interval_that_misses_its_estimate_is_reported(copy):
    def edit(rows):
        rows[0]["oracle_kl_decrease_ci_high"] = rows[0]["oracle_kl_decrease_ci_low"]

    _rewrite(copy / "06_agreement_analysis" / "E17_local_update_utility.csv", edit)
    problems = check_consistency(load_index(copy))
    assert any(problem.startswith("E17 line 2") and "outside" in problem for problem in problems)


def test_two_token_example_is_recomputed(copy):
    def edit(rows):
        rows[1]["z"] = "0.5"

    _rewrite(copy / "06_agreement_analysis" / "E18_two_token_example.csv", edit)
    problems = check_consistency(load_index(copy))
    assert any(problem.startswith("E18") and "sum_a sqrt(p1 p2)" in problem for problem in problems)


def test_unlisted_csv_is_rejected(copy):
    (copy / "03_ablations" / "E99_extra.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not listed"):
        load_index(copy)


def test_undisplayed_field_is_rejected(copy):
    path = copy / "03_ablations" / "E09_pair_gradient_cosine.csv"
    lines = path.read_text(encoding="utf-8").splitlines()
    lines = [lines[0] + ",extra"] + [line + ",1" for line in lines[1:]]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not displayed"):
        load_index(copy)


def test_environment_variable_selects_the_directory(copy, monkeypatch):
    monkeypatch.setenv(supplementary.ENV_VAR, str(copy))
    assert supplementary_dir() == copy
    assert load("E07").path.is_relative_to(copy)


@pytest.mark.parametrize(
    ("spec", "row", "expected"),
    [
        ({"header": "h", "col": "a"}, {"a": "K=2"}, "K=2"),
        ({"header": "h", "col": "a"}, {"a": ""}, "-"),
        ({"header": "h", "mean": "m", "sd": "s"}, {"m": "1.0", "s": "0.25"}, "1.0 ± 0.25"),
        (
            {"header": "h", "value": "v", "low": "l", "high": "u"},
            {"v": "-0.4", "l": "-1.2", "u": "+0.4"},
            "−0.4 [−1.2, +0.4]",
        ),
        ({"header": "h", "value": "v", "low": "l", "high": "u"}, {"v": "", "l": "", "u": ""}, "-"),
        ({"header": "h", "pair": ["a", "b"]}, {"a": "1.5", "b": "2.5"}, "1.5 / 2.5"),
    ],
)
def test_column_rendering(spec, row, expected):
    assert Column.from_spec(spec).render(row) == expected


def test_number_and_distribution_parsing():
    assert parse_number("+1.8") == 1.8
    assert parse_number("−0.3") == -0.3
    assert parse_number("") is None
    assert parse_number("K=2") is None
    assert parse_distribution("(0.25, 0.75)") == (0.25, 0.75)


def test_embedded_tables_are_refreshed(index):
    table = index["E09"].to_markdown()
    text = "before\n<!-- supplementary:E09 -->\nstale\n<!-- /supplementary:E09 -->\nafter\n"
    refreshed = embed_tables(text, index)
    assert refreshed == text.replace("stale", table)
    assert embed_tables(refreshed, index) == refreshed
    empty = "<!-- supplementary:E09 -->\n<!-- /supplementary:E09 -->\n"
    assert embed_tables(empty, index) == empty.replace("-->\n<!--", f"-->\n{table}\n<!--")


def test_readme_tables_are_current(index):
    text = (REPOSITORY / "README.md").read_text(encoding="utf-8")
    embedded = re.findall(r"<!-- supplementary:(E\d\d) -->", text)
    assert embedded, "README embeds no supplementary table"
    for experiment_id in embedded:
        assert index[experiment_id].to_markdown() in text
    assert embed_tables(text, index) == text


@pytest.mark.parametrize(
    "document",
    [
        "docs/supplementary_results.md",
        "results/supplementary/README.md",
        "assets/supplementary/README.md",
    ],
)
def test_relative_links_resolve(document):
    path = REPOSITORY / document
    for target in re.findall(r"\]\(([^)#\s]+)(?:#[^)]*)?\)", path.read_text(encoding="utf-8")):
        if not target.startswith(("http://", "https://")):
            assert (path.parent / target).exists(), f"{document}: broken link {target}"
