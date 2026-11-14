#!/usr/bin/env python
"""Aggregate evaluation files into the paper's result tables (Tables 1-4 and 6-8).

Expected layout, as written by ``scripts/run_experiment.sh``::

    <results-dir>/<method>/seed<seed>.json     # output of `vdopd eval --output ...`

Each cell is the mean over the available seeds; the six-benchmark score is the unweighted
benchmark mean within each seed followed by the mean across seeds (Appendix B.5.3). Use
``--metric`` to pick Avg@k (Tables 1, 4), Pass@k (Table 6) or an unseen-wrapper metric
(Table 3), and ``--baseline`` to print gains in percentage points (Figure 4).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vdopd.eval.metrics import macro_average, seed_mean_and_sd

UNSEEN_METRICS = ("mean_view", "worst_view", "worst_template")


def load_results(results_dir: Path) -> dict[str, list[dict]]:
    """Map each method directory to its per-seed evaluation records."""
    methods: dict[str, list[dict]] = {}
    for path in sorted(results_dir.glob("*/seed*.json")):
        with open(path, encoding="utf-8") as handle:
            methods.setdefault(path.parent.name, []).append(json.load(handle))
    return methods


def benchmark_value(record: dict, benchmark: str, metric: str) -> float:
    entry = record["benchmarks"][benchmark]
    return entry["unseen"][metric] if metric in UNSEEN_METRICS else entry[metric]


def summarise(records: list[dict], metric: str) -> dict[str, tuple[float, float]]:
    """Per-benchmark and macro ``(mean, sample SD)`` over seeds, in percent."""
    benchmarks = list(records[0]["benchmarks"])
    summary = {
        name: seed_mean_and_sd([100 * benchmark_value(r, name, metric) for r in records])
        for name in benchmarks
    }
    per_seed_macro = [
        100 * macro_average([benchmark_value(r, name, metric) for name in benchmarks])
        for r in records
    ]
    summary["mean"] = seed_mean_and_sd(per_seed_macro)
    return summary


def format_table(
    summaries: dict[str, tuple[int, dict[str, tuple[float, float]]]],
    baseline: str | None,
    show_sd: bool,
) -> str:
    """Markdown table with one row per method; cells are seed means in percent."""
    columns = list(next(iter(summaries.values()))[1])
    lines = ["| method | seeds | " + " | ".join(columns) + " |"]
    lines.append("|---|---|" + "---|" * len(columns))
    for method, (num_seeds, summary) in summaries.items():
        cells = []
        for column in columns:
            mean, sd = summary[column]
            if baseline is not None:
                cells.append(f"{mean - summaries[baseline][1][column][0]:+.1f}")
            else:
                cells.append(f"{mean:.1f} ± {sd:.1f}" if show_sd else f"{mean:.1f}")
        lines.append(f"| {method} | {num_seeds} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--results-dir", required=True, type=Path)
    parser.add_argument(
        "--metric", default="avg_at_k", choices=("avg_at_k", "pass_at_k", *UNSEEN_METRICS)
    )
    parser.add_argument("--baseline", default=None, help="print gains over this method (pp)")
    parser.add_argument("--sd", action="store_true", help="show the sample SD over seeds")
    args = parser.parse_args()

    methods = load_results(args.results_dir)
    if not methods:
        raise SystemExit(f"no evaluation files found under {args.results_dir}")
    if args.baseline is not None and args.baseline not in methods:
        raise SystemExit(f"baseline {args.baseline!r} not found; available: {sorted(methods)}")
    summaries = {
        method: (len(records), summarise(records, args.metric))
        for method, records in methods.items()
    }
    print(format_table(summaries, args.baseline, args.sd))


if __name__ == "__main__":
    main()
