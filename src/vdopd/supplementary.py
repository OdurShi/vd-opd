"""Supplementary experiment tables (E01-E18) stored under ``results/supplementary``.

``results/supplementary/index.yaml`` lists the experiments. Every experiment has one CSV file
with the reported numbers; the index says how the CSV columns are displayed (``mean ± sd``,
``value [low, high]``, ``a / b``). This module

* loads the index and the CSV files (:func:`load_index`, :func:`load`);
* renders them as Markdown (:meth:`Experiment.to_markdown`, :func:`render_markdown`), which is
  how ``docs/supplementary_results.md`` is produced;
* checks that tables reporting the same quantity agree with each other
  (:func:`check_consistency`), e.g. that the macro averages equal the mean of the
  per-benchmark scores and that every confidence interval contains its point estimate.

The numbers are never stored in code: edit a CSV file and re-run
``python scripts/render_supplementary.py``.

The directory is resolved in this order: an explicit ``root`` argument, the
``VDOPD_SUPPLEMENTARY_DIR`` environment variable, ``results/supplementary`` at the root of a
source checkout.
"""

from __future__ import annotations

import csv
import math
import os
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml

ENV_VAR = "VDOPD_SUPPLEMENTARY_DIR"
INDEX_NAME = "index.yaml"
_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

MINUS = "−"
BENCHMARKS = ("MATH500", "AMC23", "AIME24", "AIME25", "Minerva", "OlympiadBench")

# Student of E01 -> experiment holding the complete per-method table of that student.
STUDENT_TABLES = {"Qwen3-1.7B": "E02", "Qwen3-4B": "E03", "Llama-3.2-3B": "E04"}
# Method name -> column prefix used by the per-benchmark robustness table E13.
ROBUSTNESS_PREFIXES = {
    "EOPD(full)": "eopd_full",
    "EOPD(2V)": "eopd_2v",
    "EOPD(full)+StudentAug": "eopd_full_studentaug",
    "EOPD(2V)+StudentAug": "eopd_2v_studentaug",
    "VD-OPD": "vd_opd",
}

# Two values printed with one decimal agree if they differ by at most half a unit in the
# last place; a difference of two rounded values can be off by a full unit plus that half.
SAME_VALUE_TOLERANCE = 0.05
DERIVED_TOLERANCE = 0.15
MACRO_TOLERANCE = 0.1


def supplementary_dir(root: str | Path | None = None) -> Path:
    """Directory that holds ``index.yaml`` and the CSV files."""
    if root is not None:
        return Path(root)
    if os.environ.get(ENV_VAR):
        return Path(os.environ[ENV_VAR])
    return _REPOSITORY_ROOT / "results" / "supplementary"


def parse_number(cell: str | None) -> float | None:
    """Numeric value of a CSV cell (``'+1.8'`` -> 1.8); ``None`` for blank or text cells."""
    if cell is None:
        return None
    text = cell.strip().replace(MINUS, "-")
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def parse_distribution(cell: str) -> tuple[float, ...]:
    """Parse a probability vector written as ``'(0.99, 0.01)'``."""
    return tuple(float(part) for part in cell.strip().strip("()").split(","))


def _typeset(cell: str | None) -> str:
    """Display form of one cell: ``-`` for blanks, a typographic minus for negative numbers."""
    if cell is None or not cell.strip():
        return "-"
    text = cell.strip()
    if re.fullmatch(r"-\d[\d.]*", text):
        return MINUS + text[1:]
    return text


@dataclass(frozen=True)
class Column:
    """One displayed column, built from one to three CSV fields.

    ``kind`` is ``col`` (one field as written), ``mean_sd`` (``mean ± sd``), ``interval``
    (``value [low, high]``) or ``pair`` (``a / b``).
    """

    header: str
    kind: str
    fields: tuple[str, ...]
    align: str = "right"

    @classmethod
    def from_spec(cls, spec: Mapping[str, object]) -> Column:
        header = str(spec["header"])
        align = str(spec.get("align", "right"))
        if align not in ("left", "right"):
            raise ValueError(f"column {header!r}: align must be 'left' or 'right'")
        if "col" in spec:
            return cls(header, "col", (str(spec["col"]),), align)
        if "mean" in spec:
            return cls(header, "mean_sd", (str(spec["mean"]), str(spec["sd"])), align)
        if "value" in spec:
            fields = (str(spec["value"]), str(spec["low"]), str(spec["high"]))
            return cls(header, "interval", fields, align)
        if "pair" in spec:
            first, second = spec["pair"]  # type: ignore[misc]
            return cls(header, "pair", (str(first), str(second)), align)
        raise ValueError(f"column {header!r}: expected one of col, mean/sd, value/low/high, pair")

    def render(self, row: Mapping[str, str]) -> str:
        cells = [_typeset(row.get(name)) for name in self.fields]
        if self.kind == "col":
            return cells[0]
        if all(cell == "-" for cell in cells):
            return "-"
        if self.kind == "mean_sd":
            return f"{cells[0]} ± {cells[1]}"
        if self.kind == "interval":
            return f"{cells[0]} [{cells[1]}, {cells[2]}]"
        return f"{cells[0]} / {cells[1]}"


@dataclass(frozen=True)
class Group:
    """A folder of related experiments."""

    dir: str
    title: str
    summary: str = ""


@dataclass(frozen=True)
class Experiment:
    """One supplementary experiment: its description and the rows of its CSV file."""

    id: str
    title: str
    group: str
    relative_path: str
    path: Path
    columns: tuple[Column, ...]
    fieldnames: tuple[str, ...]
    rows: tuple[dict[str, str], ...]
    extends: str | None = None
    question: str | None = None
    setup: tuple[str, ...] = ()
    findings: tuple[str, ...] = ()

    def number(self, row: Mapping[str, str], name: str) -> float | None:
        """Numeric value of ``name`` in ``row`` (``None`` if blank or not a number)."""
        if name not in self.fieldnames:
            raise KeyError(f"{self.id}: no column {name!r}")
        return parse_number(row.get(name))

    def values(self, name: str) -> list[float | None]:
        """The column ``name`` as numbers, in file order."""
        return [self.number(row, name) for row in self.rows]

    def find(self, **where: str) -> dict[str, str] | None:
        """The row whose text fields equal ``where``, or ``None`` if there is not exactly one."""
        matches = [
            row for row in self.rows if all(row.get(key) == value for key, value in where.items())
        ]
        return matches[0] if len(matches) == 1 else None

    def row(self, **where: str) -> dict[str, str]:
        """Like :meth:`find`, but raise ``KeyError`` when the row is missing or ambiguous."""
        found = self.find(**where)
        if found is None:
            raise KeyError(f"{self.id}: expected exactly one row with {where}")
        return found

    def to_markdown(self) -> str:
        """The experiment's table as GitHub-flavoured Markdown."""
        header = "| " + " | ".join(column.header for column in self.columns) + " |"
        rule = "|" + "|".join("---" if c.align == "left" else "---:" for c in self.columns) + "|"
        body = [
            "| " + " | ".join(column.render(row) for column in self.columns) + " |"
            for row in self.rows
        ]
        return "\n".join([header, rule, *body])


@dataclass(frozen=True)
class Index:
    """All supplementary experiments together with the shared conventions."""

    root: Path
    conventions: str
    groups: tuple[Group, ...]
    experiments: dict[str, Experiment]

    def __getitem__(self, experiment_id: str) -> Experiment:
        return self.experiments[experiment_id]

    def __iter__(self):
        return iter(self.experiments.values())

    def __len__(self) -> int:
        return len(self.experiments)

    def in_group(self, group: str) -> list[Experiment]:
        return [experiment for experiment in self if experiment.group == group]


def _read_csv(path: Path) -> tuple[tuple[str, ...], tuple[dict[str, str], ...]]:
    with open(path, encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = tuple(reader.fieldnames or ())
        rows = tuple({key: (value or "") for key, value in row.items()} for row in reader)
    return fieldnames, rows


def _as_tuple(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in value)  # type: ignore[union-attr]


def load_index(root: str | Path | None = None) -> Index:
    """Load ``index.yaml`` and every CSV file it lists.

    Raises ``ValueError`` if an experiment id is repeated, a listed file is missing, a display
    column refers to a field the CSV does not have, a CSV field is not displayed by any
    column, or a CSV file in the directory is not listed in the index.
    """
    directory = supplementary_dir(root)
    spec = yaml.safe_load((directory / INDEX_NAME).read_text(encoding="utf-8"))
    groups = tuple(
        Group(str(g["dir"]), str(g["title"]), str(g.get("summary", ""))) for g in spec["groups"]
    )
    group_dirs = {group.dir for group in groups}

    experiments: dict[str, Experiment] = {}
    for entry in spec["experiments"]:
        experiment_id = str(entry["id"])
        if experiment_id in experiments:
            raise ValueError(f"{INDEX_NAME}: duplicate experiment id {experiment_id}")
        relative = str(entry["file"])
        path = directory / relative
        if not path.is_file():
            raise ValueError(f"{experiment_id}: missing data file {relative}")
        group = Path(relative).parts[0]
        if group not in group_dirs:
            raise ValueError(f"{experiment_id}: folder {group!r} is not a listed group")
        fieldnames, rows = _read_csv(path)
        columns = tuple(Column.from_spec(column) for column in entry["columns"])
        displayed = {name for column in columns for name in column.fields}
        unknown = sorted(displayed - set(fieldnames))
        if unknown:
            raise ValueError(f"{experiment_id}: index refers to missing CSV fields {unknown}")
        hidden = sorted(set(fieldnames) - displayed)
        if hidden:
            raise ValueError(f"{experiment_id}: CSV fields not displayed by any column {hidden}")
        experiments[experiment_id] = Experiment(
            id=experiment_id,
            title=str(entry["title"]),
            group=group,
            relative_path=relative,
            path=path,
            columns=columns,
            fieldnames=fieldnames,
            rows=rows,
            extends=str(entry["extends"]) if entry.get("extends") else None,
            question=" ".join(str(entry["question"]).split()) if entry.get("question") else None,
            setup=tuple(" ".join(item.split()) for item in _as_tuple(entry.get("setup"))),
            findings=tuple(" ".join(item.split()) for item in _as_tuple(entry.get("findings"))),
        )

    listed = {experiment.relative_path for experiment in experiments.values()}
    on_disk = {path.relative_to(directory).as_posix() for path in directory.rglob("*.csv")}
    unlisted = sorted(on_disk - listed)
    if unlisted:
        raise ValueError(f"CSV files not listed in {INDEX_NAME}: {unlisted}")
    return Index(directory, " ".join(str(spec["conventions"]).split()), groups, experiments)


def load(experiment_id: str, root: str | Path | None = None) -> Experiment:
    """Load one experiment by id (``'E07'``)."""
    return load_index(root)[experiment_id]


# --------------------------------------------------------------------------- rendering


def render_markdown(
    index: Index | None = None, link_prefix: str = "../results/supplementary/"
) -> str:
    """The whole collection as one Markdown document (``docs/supplementary_results.md``).

    ``link_prefix`` is the path from the document to the data directory.
    """
    index = index or load_index()
    lines: list[str] = [
        "# Supplementary experiments (E01-E18)",
        "",
        "<!-- Generated by scripts/render_supplementary.py from results/supplementary/.",
        "     Edit the CSV files or index.yaml there, not this file. -->",
        "",
        "These tables are supplementary results reported by the authors of the paper. They",
        "extend the tables of the paper with seed variation, confidence intervals, matched",
        "controls, further ablations and robustness tests. The numbers are stored as CSV files",
        f"in [`results/supplementary/`]({link_prefix}) and loaded with `vdopd.supplementary`;",
        "the commands and configurations for each experiment are listed in the",
        "[README](../README.md#supplementary-experiments).",
        "",
        index.conventions,
        "",
        "## Index",
        "",
        "| ID | Experiment | Extends | Data |",
        "|---|---|---|---|",
    ]
    for experiment in index:
        name = Path(experiment.relative_path).name
        lines.append(
            f"| [{experiment.id}](#{experiment.id.lower()}) | {experiment.title} "
            f"| {experiment.extends or '-'} | [`{name}`]({link_prefix}{experiment.relative_path}) |"
        )
    for group in index.groups:
        lines += ["", f"## {group.title}", ""]
        if group.summary:
            lines += [group.summary, ""]
        for experiment in index.in_group(group.dir):
            lines += [f'<a id="{experiment.id.lower()}"></a>', ""]
            lines += [f"### {experiment.id}. {experiment.title}", ""]
            if experiment.extends:
                lines += [f"*Extends {experiment.extends} of the paper.*", ""]
            if experiment.question:
                lines += [f"**Question.** {experiment.question}", ""]
            if experiment.setup:
                lines += ["**Setup.**", ""]
                lines += [f"- {item}" for item in experiment.setup]
                lines.append("")
            lines += [experiment.to_markdown(), ""]
            if experiment.findings:
                lines += ["**Key findings.**", ""]
                lines += [f"- {item}" for item in experiment.findings]
                lines.append("")
            target = link_prefix + experiment.relative_path
            lines.append(f"Data: [`{experiment.relative_path}`]({target})")
    return "\n".join(lines).rstrip() + "\n"


EMBED = re.compile(
    r"(<!-- supplementary:(E\d\d) -->\n)(?:.*?\n)??(<!-- /supplementary:\2 -->)", re.DOTALL
)


def embed_tables(text: str, index: Index | None = None) -> str:
    """Refresh the tables a document embeds between ``<!-- supplementary:E07 -->`` markers.

    Everything between ``<!-- supplementary:E07 -->`` and ``<!-- /supplementary:E07 -->`` is
    replaced by the current table of that experiment, so a README can show a few tables
    without keeping a second copy of the numbers.
    """
    index = index or load_index()

    def table(match: re.Match[str]) -> str:
        return match.group(1) + index[match.group(2)].to_markdown() + "\n" + match.group(3)

    return EMBED.sub(table, text)


# --------------------------------------------------------------------------- consistency


def macro_average(values: Iterable[float | None]) -> float | None:
    """Mean of ``values``; ``None`` if any entry is missing."""
    items = list(values)
    if not items or any(item is None for item in items):
        return None
    return sum(items) / len(items)  # type: ignore[arg-type]


class _Checker:
    """Collects human-readable descriptions of disagreements between tables."""

    def __init__(self, index: Index) -> None:
        self.index = index
        self.problems: list[str] = []

    def row(self, experiment_id: str, **where: str) -> dict[str, str] | None:
        found = self.index[experiment_id].find(**where)
        if found is None:
            self.problems.append(f"{experiment_id}: expected exactly one row with {where}")
        return found

    def number(self, experiment_id: str, row: Mapping[str, str] | None, name: str) -> float | None:
        if row is None:
            return None
        value = self.index[experiment_id].number(row, name)
        if value is None:
            self.problems.append(f"{experiment_id}: column {name} is blank or not a number")
        return value

    def close(self, left: float | None, right: float | None, tolerance: float, label: str) -> None:
        if left is None or right is None:
            return
        if abs(left - right) > tolerance + 1e-9:
            self.problems.append(f"{label}: {left:g} vs {right:g} (tolerance {tolerance:g})")


def _check_intervals(check: _Checker) -> None:
    """Every interval contains its point estimate and every SD is non-negative."""
    for experiment in check.index:
        for number, row in enumerate(experiment.rows, start=2):
            for column in experiment.columns:
                values = [parse_number(row.get(name)) for name in column.fields]
                where = f"{experiment.id} line {number}, {column.header}"
                if column.kind == "interval" and all(v is not None for v in values):
                    value, low, high = values
                    if not low <= value <= high:  # type: ignore[operator]
                        check.problems.append(f"{where}: {value} is outside [{low}, {high}]")
                elif column.kind == "interval" and any(v is not None for v in values):
                    check.problems.append(f"{where}: interval is partly blank")
                elif column.kind == "mean_sd" and values[1] is not None and values[1] < 0:
                    check.problems.append(f"{where}: negative standard deviation")


def _check_main_results(check: _Checker) -> None:
    """E01 and E05 repeat entries of the complete per-method tables E02-E04."""
    e01 = check.index["E01"]
    for row in e01.rows:
        table = STUDENT_TABLES.get(row["student"])
        if table is None:
            check.problems.append(f"E01: unknown student {row['student']!r}")
            continue
        prefix = row["benchmark"].lower()
        label = f"E01 {row['student']} {row['benchmark']}"
        eopd = e01.number(row, "eopd_full_avg8_mean")
        vd = e01.number(row, "vd_opd_avg8_mean")
        check.close(
            eopd,
            check.number(table, check.row(table, method="EOPD(full)"), f"{prefix}_avg8"),
            SAME_VALUE_TOLERANCE,
            f"{label}: EOPD(full) Avg@8 vs {table}",
        )
        check.close(
            vd,
            check.number(table, check.row(table, method="VD-OPD"), f"{prefix}_avg8"),
            SAME_VALUE_TOLERANCE,
            f"{label}: VD-OPD Avg@8 vs {table}",
        )
        if eopd is not None and vd is not None:
            gain = e01.number(row, "vd_gain")
            check.close(gain, vd - eopd, DERIVED_TOLERANCE, f"{label}: gain vs difference of means")

    pairs = {
        "Qwen3-8B teacher": "Frozen teacher",
        "Qwen3-4B, EOPD(full)": "EOPD(full)",
        "Qwen3-4B, VD-OPD": "VD-OPD",
    }
    for model, method in pairs.items():
        left = check.row("E05", model=model)
        right = check.row("E03", method=method)
        for name in ("math500_avg8", "math500_pass8"):
            check.close(
                check.number("E05", left, name),
                check.number("E03", right, name),
                SAME_VALUE_TOLERANCE,
                f"E05 {model} {name} vs E03",
            )


def _macro_from_table(check: _Checker, table: str, method: str) -> float | None:
    row = check.row(table, method=method)
    return macro_average(
        check.number(table, row, f"{benchmark.lower()}_avg8") for benchmark in BENCHMARKS
    )


def _check_macro_averages(check: _Checker) -> None:
    """Macro Avg@8 of the Qwen3-1.7B student wherever a table repeats it."""
    index = check.index
    macro = {
        method: _macro_from_table(check, "E02", method)
        for method in ("EOPD(top-16)", "EOPD(full)", "ToDi-OP", "TSD-KD", "Veto", "VD-OPD")
    }
    full, vd = macro["EOPD(full)"], macro["VD-OPD"]

    def same(value: float | None, reference: float | None, label: str) -> None:
        check.close(value, reference, SAME_VALUE_TOLERANCE, f"{label} vs macro average of E02")

    e06 = index["E06"]
    for row in e06.rows:
        baseline = e06.number(row, "baseline_macro_avg8_mean")
        ours = e06.number(row, "vd_macro_avg8_mean")
        name = row["comparison"]
        if name.startswith("EOPD(full)"):
            same(baseline, full, f"E06 '{name}' baseline")
        if name.endswith("VD-OPD"):
            same(ours, vd, f"E06 '{name}' VD-OPD")
        if baseline is not None and ours is not None:
            check.close(
                e06.number(row, "vd_gain"),
                ours - baseline,
                DERIVED_TOLERANCE,
                f"E06 '{name}': gain vs difference of means",
            )

    e07 = index["E07"]
    reference = check.row("E07", weight="state-specific Z")
    reference_mean = check.number("E07", reference, "macro_avg8_mean")
    same(reference_mean, vd, "E07 VD-OPD row")
    for row in e07.rows:
        gain = parse_number(row.get("vd_gain"))
        mean = e07.number(row, "macro_avg8_mean")
        if gain is not None and mean is not None and reference_mean is not None:
            check.close(
                gain,
                reference_mean - mean,
                DERIVED_TOLERANCE,
                f"E07 {row['target']} / {row['weight']}: gain vs difference of means",
            )

    e08 = index["E08"]
    k2 = check.row("E08", setting="K=2")
    k2_mean = check.number("E08", k2, "macro_avg8_mean")
    same(k2_mean, vd, "E08 K=2")
    for row in e08.rows:
        gain = parse_number(row.get("gain_over_k2"))
        mean = e08.number(row, "macro_avg8_mean")
        if gain is not None and mean is not None and k2_mean is not None:
            check.close(
                gain,
                mean - k2_mean,
                DERIVED_TOLERANCE,
                f"E08 {row['setting']}: gain vs difference of means",
            )

    e10 = index["E10"]
    for row in e10.rows:
        same(e10.number(row, "eopd_full"), full, f"E10 {row['view_pool']} EOPD(full)")
        two, ours = e10.number(row, "eopd_2v_mean"), e10.number(row, "vd_opd_mean")
        if two is not None and ours is not None:
            check.close(
                e10.number(row, "vd_minus_2v"),
                ours - two,
                DERIVED_TOLERANCE,
                f"E10 {row['view_pool']}: VD - 2V vs difference of means",
            )
    simple = check.row("E10", view_pool="Simple templates")
    same(check.number("E10", simple, "vd_opd_mean"), vd, "E10 simple templates VD-OPD")

    for method, value in macro.items():
        row = check.row("E11", method=method)
        same(check.number("E11", row, "original_macro_avg8"), value, f"E11 {method} original")
    for experiment_id in ("E12", "E14"):
        for method in ("EOPD(full)", "VD-OPD"):
            row = check.row(experiment_id, method=method)
            same(
                check.number(experiment_id, row, "canonical"),
                macro[method],
                f"{experiment_id} {method} canonical",
            )

    # EOPD(2V) is not a row of E02; its macro average must agree across the tables.
    two_view = check.number(
        "E06", check.row("E06", comparison="EOPD(2V) vs. VD-OPD"), "baseline_macro_avg8_mean"
    )
    others = {
        "E10 simple templates": check.number("E10", simple, "eopd_2v_mean"),
        "E11 original": check.number(
            "E11", check.row("E11", method="EOPD(2V)"), "original_macro_avg8"
        ),
        "E12 canonical": check.number("E12", check.row("E12", method="EOPD(2V)"), "canonical"),
        "E14 canonical": check.number("E14", check.row("E14", method="EOPD(2V)"), "canonical"),
    }
    for label, value in others.items():
        check.close(value, two_view, SAME_VALUE_TOLERANCE, f"EOPD(2V) macro Avg@8, {label} vs E06")


def _check_robustness(check: _Checker) -> None:
    """E12-E15 and the robustness columns of E08 and E10."""
    index = check.index
    e12, e13 = index["E12"], index["E13"]
    for row in e12.rows:
        method = row["method"]
        canonical = e12.number(row, "canonical")
        mean, worst = e12.number(row, "mean_view"), e12.number(row, "worst_view")
        if canonical is not None and mean is not None:
            check.close(
                e12.number(row, "canonical_minus_mean"),
                canonical - mean,
                MACRO_TOLERANCE,
                f"E12 {method}: Canonical - Mean",
            )
        if mean is not None and worst is not None:
            check.close(
                e12.number(row, "mean_minus_worst"),
                mean - worst,
                MACRO_TOLERANCE,
                f"E12 {method}: Mean - Worst",
            )
        prefix = ROBUSTNESS_PREFIXES.get(method)
        if prefix is None:
            check.problems.append(f"E12: unknown method {method!r}")
            continue
        for metric, value in (("mean_view", mean), ("worst_view", worst)):
            check.close(
                value,
                macro_average(e13.values(f"{prefix}_{metric}")),
                MACRO_TOLERANCE,
                f"E12 {method} {metric} vs macro average of E13",
            )
        augmented = check.row("E14", method=method)
        for name, source in (
            ("canonical", "canonical"),
            ("prefix_mean_view", "mean_view"),
            ("prefix_worst_view", "worst_view"),
        ):
            check.close(
                check.number("E14", augmented, name),
                e12.number(row, source),
                SAME_VALUE_TOLERANCE,
                f"E14 {method} {name} vs E12 {source}",
            )
        for family in ("Structural", "Paraphrase"):
            detail = check.row("E15", family=family, method=method)
            for metric in ("mean_view", "worst_view"):
                check.close(
                    check.number("E14", augmented, f"{family.lower()}_{metric}"),
                    check.number("E15", detail, metric),
                    SAME_VALUE_TOLERANCE,
                    f"E14 {method} {family.lower()} {metric} vs E15",
                )

    ours = check.row("E12", method="VD-OPD")
    two = check.row("E12", method="EOPD(2V)")
    check.close(
        check.number("E08", check.row("E08", setting="K=2"), "worst_view"),
        check.number("E12", ours, "worst_view"),
        SAME_VALUE_TOLERANCE,
        "E08 K=2 WorstView vs E12 VD-OPD",
    )
    simple = check.row("E10", view_pool="Simple templates")
    check.close(
        check.number("E10", simple, "worst_view_vd_opd"),
        check.number("E12", ours, "worst_view"),
        SAME_VALUE_TOLERANCE,
        "E10 simple templates WorstView (VD-OPD) vs E12",
    )
    check.close(
        check.number("E10", simple, "worst_view_eopd_2v"),
        check.number("E12", two, "worst_view"),
        SAME_VALUE_TOLERANCE,
        "E10 simple templates WorstView (EOPD(2V)) vs E12",
    )
    matched = check.row("E06", comparison="EOPD(2V) vs. VD-OPD")
    flops = {
        "E06": check.number("E06", matched, "vd_flops_1e18"),
        "E08": check.number("E08", check.row("E08", setting="K=2"), "flops_1e18"),
    }
    reference = check.number("E14", check.row("E14", method="VD-OPD"), "flops_1e18")
    for label, value in flops.items():
        check.close(value, reference, SAME_VALUE_TOLERANCE, f"VD-OPD FLOPs, {label} vs E14")


def _check_agreement(check: _Checker) -> None:
    """E16 counts and percentages; E18 recomputed from its two teacher views."""
    e16 = check.index["E16"]
    states = check.row("E16", diagnostic="States")
    pairs = check.row("E16", diagnostic="Continuation pairs")
    for stratum in ("low_z_q1", "high_z_q4"):
        count = check.number("E16", states, stratum)
        if count is not None:
            check.close(
                check.number("E16", pairs, stratum),
                4 * count,
                0.0,
                f"E16 {stratum}: continuation pairs vs four per state",
            )
        outcomes = [
            e16.number(row, stratum)
            for row in e16.rows
            if "correct" in row["diagnostic"].lower() or "wrong" in row["diagnostic"].lower()
        ]
        total = macro_average(outcomes)
        if total is not None:
            check.close(
                total * len(outcomes), 100.0, 1.0, f"E16 {stratum}: outcome percentages sum to 100"
            )

    e18 = check.index["E18"]
    for row in e18.rows:
        try:
            p1, p2 = parse_distribution(row["p1"]), parse_distribution(row["p2"])
            reported = parse_distribution(row["r"])
        except ValueError:
            check.problems.append(f"E18 {row['case']}: cannot parse the distributions")
            continue
        consensus = [math.sqrt(a * b) for a, b in zip(p1, p2, strict=True)]
        mass = sum(consensus)
        check.close(e18.number(row, "z"), mass, 5e-4, f"E18 {row['case']}: Z vs sum_a sqrt(p1 p2)")
        for position, (value, expected) in enumerate(zip(reported, consensus, strict=True)):
            check.close(value, expected / mass, 5e-3, f"E18 {row['case']}: r[{position}] vs c / Z")


CHECKS = (
    _check_intervals,
    _check_main_results,
    _check_macro_averages,
    _check_robustness,
    _check_agreement,
)


def check_consistency(index: Index | None = None) -> list[str]:
    """Disagreements between the tables, one description per problem (empty if none).

    The checks cover quantities that appear in more than one table (the EOPD(full) and VD-OPD
    scores, the macro averages, the unseen-view metrics, training FLOPs), quantities that
    follow from others in the same table (gains, ``Canonical - Mean``, counts, percentages),
    the containment of every point estimate in its confidence interval, and the closed-form
    two-token example E18.
    """
    check = _Checker(index or load_index())
    for function in CHECKS:
        try:
            function(check)
        except KeyError as error:
            check.problems.append(f"{function.__name__}: missing table or column {error}")
    return check.problems


def summary_rows(index: Index | None = None) -> Sequence[tuple[str, str, str, str]]:
    """``(id, title, extends, relative path)`` of every experiment, for tables of contents."""
    index = index or load_index()
    return [(e.id, e.title, e.extends or "-", e.relative_path) for e in index]
