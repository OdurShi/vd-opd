#!/usr/bin/env python
"""Local-update diagnostic (Section 5.4-5.5, Appendix B.7, Table 12 and Figure 6).

Six matched branches (Table 4) start from the same checkpoint and optimizer state, reuse the
same sampled responses and teacher views, and take one update each. The script prints

* Table 12(a): token means of H(p1), H(p2), H(q) and the retained mass Z;
* Table 12(b): Avg@k on the source-disjoint measurement problems before/after each branch;
* Figure 6: per-bin median and 10-90% range of the consensus-fit utility
  ``u_t = l^cons_t(before) - l^cons_t(after)`` (Equation 9) for the VD-OPD branch, with the
  token-weighted correctness of the sampled responses.

    python scripts/local_update_diagnostic.py --toy --output results/local_update.json
    python scripts/local_update_diagnostic.py --config configs/qwen3_1p7b/vdopd.yaml

``--toy`` uses the synthetic toy pair of the quick start (not the paper's models).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from vdopd.config import load_config
from vdopd.diagnostics import binned_utility_summary, matched_branch_updates
from vdopd.eval.answers import extract_answer, is_equivalent
from vdopd.eval.evaluator import evaluate
from vdopd.objectives import ObjectiveConfig, build_objective
from vdopd.optim import set_learning_rate
from vdopd.pipeline import (
    diagnostic_setting_from_config,
    diagnostic_trainer,
    toy_diagnostic_setting,
)

BRANCHES = {
    "Arithmetic": "arithmetic",
    "Weighted arithmetic": "weighted_arithmetic",
    "Geometric": "geometric",
    "VD-OPD": "vdopd",
    "Shuffled weight": "shuffled_weight",
    "Batch-mean weight": "batch_mean_weight",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", help="YAML experiment file (real models)")
    source.add_argument("--toy", action="store_true", help="synthetic toy pair (CPU)")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--student", default=None, help="checkpoint to branch from")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--no-accuracy", action="store_true", help="skip the Avg@k evaluations")
    parser.add_argument("--output", default=None)
    parser.add_argument("--plot", default=None, help="write Figure 6 to this image file")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)

    if args.toy:
        setting = toy_diagnostic_setting(args.seed)
    else:
        setting = diagnostic_setting_from_config(
            load_config(args.config, args.overrides), args.student
        )
    trainer = diagnostic_trainer(setting)
    set_learning_rate(trainer.optimizer, setting.optim.learning_rate)  # no warmup in a branch
    batch = trainer.collect_rollouts(0)

    def accuracy() -> float:
        result = evaluate(
            setting.student,
            setting.codec,
            setting.measurement_problems,
            setting.decoding,
            unseen_wrappers=(),
        )
        return 100.0 * result["avg_at_k"]

    objectives = {
        label: build_objective(ObjectiveConfig(name=name)) for label, name in BRANCHES.items()
    }
    result = matched_branch_updates(
        trainer, batch, objectives, None if args.no_accuracy else accuracy
    )

    pre = result["pre_update"]
    print("Table 12(a) - shared pre-update diagnostics (token means)")
    print(
        f"  H(p1)={pre['entropy_p1']:.3f}  H(p2)={pre['entropy_p2']:.3f}  "
        f"H(q)={pre['entropy_q']:.3f}  Z={pre['mass']:.3f}  positions={len(result['mass'])}"
    )
    print("Table 12(b) - matched branch updates")
    rows = {}
    for label, branch in result["branches"].items():
        utility = branch["utility"].numpy()
        rows[label] = {
            "status": branch["status"],
            "mean_utility": float(utility.mean()),
            "before": result["before_score"],
            "after": branch["after_score"],
        }
        line = f"  {label:<20} status={branch['status']:<9} mean u_t={utility.mean():+.5f}"
        if branch["after_score"] is not None:
            before, after = result["before_score"], branch["after_score"]
            line += f"  Avg@k {before:.1f} -> {after:.1f} ({after - before:+.1f})"
        print(line)

    # Each position inherits the final-answer correctness of its sampled response.
    correct = np.array(
        [
            is_equivalent(
                extract_answer(setting.codec.decode(response)),
                setting.update_problems[index].answer or "",
            )
            for index, response in zip(batch.prompt_indices, batch.responses, strict=True)
        ]
    )
    mass = result["mass"].numpy()
    utility = result["branches"]["VD-OPD"]["utility"].numpy()
    token_correct = correct[result["response_index"].numpy()]
    summary = binned_utility_summary(mass, utility, token_correct)
    print("Figure 6 - consensus mass and utility (VD-OPD branch; 24 bins, 3-bin moving average)")
    print("  Z bin   count   p10       median    p90       correctness")
    for index, center in enumerate(summary["bin_centers"]):
        if summary["count"][index] == 0:
            continue
        print(
            f"  {center:.3f}  {summary['count'][index]:>6}  {summary['p10'][index]:+.5f}  "
            f"{summary['median'][index]:+.5f}  {summary['p90'][index]:+.5f}  "
            f"{summary['correctness'][index]:.3f}"
        )

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "pre_update": pre,
                    "branches": rows,
                    "figure6": {key: np.asarray(value).tolist() for key, value in summary.items()},
                },
                handle,
                indent=2,
            )
    if args.plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figure, axis = plt.subplots(figsize=(4.5, 3.2))
        axis.hexbin(mass, utility, gridsize=40, bins="log", cmap="Purples", mincnt=1)
        centers = summary["bin_centers"]
        axis.fill_between(centers, summary["p10"], summary["p90"], alpha=0.25, label="10-90%")
        axis.plot(centers, summary["median"], label="Median")
        axis.set_xlabel("$Z_t$")
        axis.set_ylabel("$u_t$ (nats)")
        twin = axis.twinx()
        twin.plot(centers, 100 * summary["correctness"], "--", color="tab:red", label="Correctness")
        twin.set_ylim(0, 100)
        twin.set_ylabel("Correctness (%)")
        axis.legend(loc="upper left", fontsize=7)
        figure.tight_layout()
        figure.savefig(args.plot, dpi=200)


if __name__ == "__main__":
    main()
