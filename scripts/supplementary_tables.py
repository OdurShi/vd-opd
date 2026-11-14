#!/usr/bin/env python
"""Build the tables of the supplementary experiments from evaluation files.

Per-seed evaluation files written by ``vdopd eval --output`` (the layout produced by
``scripts/run_experiment.sh``) are aggregated into CSV tables that have the file names and
exactly the column names of the supplementary result tables E01-E08, E10 and E12-E15::

    <results-dir>/<group>/<method>/seed<seed>.json        e.g. results/qwen3_1p7b/vdopd/seed11.json
    <runs-dir>/<group>/<method>/seed<seed>/summary.json   optional, for the FLOPs and mean-Z columns

    python scripts/supplementary_tables.py --results-dir results --runs-dir runs \
        --output-dir out/tables
    python scripts/supplementary_tables.py --results-dir results --output-dir out/tables \
        --tables E01 E06 E07

Seed statistics are the mean and the sample standard deviation over the available seeds;
confidence intervals are 95% paired question bootstraps over the seeds two methods have in
common (stratified by benchmark for macro averages). A table is written only if the methods
it compares are present; cells whose inputs are missing stay empty. Numbers come from the
given files only.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from vdopd.eval.bootstrap import DEFAULT_RESAMPLES
from vdopd.eval.tables import TABLES, ResultSet, load_results, write_tables


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--results-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--runs-dir", type=Path, default=None, help="training outputs (FLOPs, Z)")
    parser.add_argument("--tables", nargs="+", default=None, choices=sorted(TABLES), metavar="ID")
    parser.add_argument(
        "--group", default="qwen3_1p7b", help="config folder of the single-student tables"
    )
    parser.add_argument("--resamples", type=int, default=DEFAULT_RESAMPLES)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    args = parser.parse_args(argv)

    evaluations = load_results(args.results_dir)
    if not evaluations:
        raise SystemExit(f"no evaluation files found under {args.results_dir}")
    results = ResultSet(
        evaluations,
        runs_dir=args.runs_dir,
        group=args.group,
        num_resamples=args.resamples,
        bootstrap_seed=args.bootstrap_seed,
    )
    for experiment, path in write_tables(results, args.output_dir, args.tables).items():
        print(f"{experiment}: {path if path is not None else 'skipped (no matching runs)'}")


if __name__ == "__main__":
    main()
