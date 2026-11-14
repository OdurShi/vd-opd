#!/usr/bin/env python
"""Accuracy versus teacher-scoring FLOPs (Figure 5, Equation 14).

Each point is one (method, nominal budget) pair run with three seeds. Provide, for every
run, the evaluation file written by `vdopd eval --output` and the `metrics.jsonl` of the
training run (its last line records the cumulative teacher-scoring FLOPs)::

    <root>/<method>/budget<b>/seed<seed>/eval.json
    <root>/<method>/budget<b>/seed<seed>/metrics.jsonl

Budgeted runs are produced with `--set train.max_teacher_flops=<value>`; the four nominal
budgets of the paper are 0.5x, 1.0x, 1.5x and 2.0x the default teacher-scoring cost of
EOPD (full). The script prints the mean cost, the mean Avg@k and its sample SD over seeds.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vdopd.eval.metrics import seed_mean_and_sd


def read_point(directory: Path, benchmark: str | None) -> tuple[float, float]:
    """Teacher-only FLOPs and Avg@k (percent) of one run."""
    with open(directory / "metrics.jsonl", encoding="utf-8") as handle:
        last = json.loads(handle.readlines()[-1])
    with open(directory / "eval.json", encoding="utf-8") as handle:
        evaluation = json.load(handle)
    score = (
        evaluation["macro"]["avg_at_k"]
        if benchmark is None
        else evaluation["benchmarks"][benchmark]["avg_at_k"]
    )
    return float(last["flops_teacher_scoring"]), 100.0 * score


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--benchmark", default=None, help="default: six-benchmark macro-average")
    parser.add_argument("--plot", default=None)
    args = parser.parse_args()

    curves: dict[str, list[tuple[float, float, float]]] = {}
    for budget_dir in sorted(args.root.glob("*/budget*")):
        points = [read_point(d, args.benchmark) for d in sorted(budget_dir.glob("seed*"))]
        if not points:
            continue
        cost = sum(c for c, _ in points) / len(points)  # mean recorded teacher-only FLOPs
        mean, sd = seed_mean_and_sd([score for _, score in points])
        curves.setdefault(budget_dir.parent.name, []).append((cost, mean, sd))
    if not curves:
        raise SystemExit(f"no runs found under {args.root}")

    for method, points in curves.items():
        for cost, mean, sd in sorted(points):
            print(f"{method:<16} teacher FLOPs={cost:.6e}  Avg@k={mean:.2f} +- {sd:.2f}")

    if args.plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figure, axis = plt.subplots(figsize=(4.5, 3.0))
        for method, points in curves.items():
            costs, means, sds = zip(*sorted(points), strict=True)
            axis.errorbar(costs, means, yerr=sds, marker="o", capsize=3, label=method)
        axis.set_xlabel("Teacher-scoring FLOPs")
        axis.set_ylabel("Avg@k (%)")
        axis.legend()
        figure.tight_layout()
        figure.savefig(args.plot, dpi=200)


if __name__ == "__main__":
    main()
