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

``--supplementary`` runs the matched local updates of supplementary experiment E17 instead:
five branches (EOPD(full), Geometric-1, Batch-mean-Z, Shuffled-Z, VD-OPD) that share the
checkpoint, the optimizer state, the student rollouts, the teacher distributions and the
update positions. Their effect is measured at fixed prefixes of the source-disjoint
measurement questions and reported for the bottom (Low) and top (High) quartile of the
cross-view agreement Z, with 95% question-bootstrap intervals:

* the decrease of the oracle KL against the fixed four-candidate answer target
  (0.99, 0.01/3, 0.01/3, 0.01/3) (Equation 21; positive = decreased);
* the change of the student's probability of a wrong answer shared by both teacher views
  (positive = reinforced), on the states that have one;
* the change of the accuracy of student continuations, in percentage points.

    python scripts/local_update_diagnostic.py --toy --supplementary \
        --csv out/E17_local_update_utility.csv
    python scripts/local_update_diagnostic.py --config configs/qwen3_1p7b/vdopd.yaml \
        --supplementary --checkpoint runs/qwen3_1p7b/vdopd/seed11/checkpoint-50 \
        --candidates candidates.jsonl

``--candidates`` is a JSON Lines file with one record per measurement problem,
``{"problem_id": ..., "candidates": [correct, wrong, wrong, wrong]}`` (final-answer strings,
the reference-correct one first). ``--toy`` uses the synthetic toy pair of the quick start
(not the paper's models) and derives its candidates from the integer answers.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from vdopd.config import load_config
from vdopd.data.problems import Problem, build_messages
from vdopd.diagnostics import (
    Z_STRATA,
    binned_utility_summary,
    build_analysis_states,
    candidate_distributions,
    matched_branch_updates,
    oracle_kl_decrease,
    quartile_labels,
    sample_final_answers,
    shared_wrong_answers,
    state_agreement,
    stratum_means,
)
from vdopd.eval.answers import extract_answer, is_equivalent
from vdopd.eval.bootstrap import DEFAULT_RESAMPLES, BootstrapCI
from vdopd.eval.evaluator import evaluate
from vdopd.eval.tables import number, write_csv
from vdopd.objectives import ObjectiveConfig, build_objective
from vdopd.optim import learning_rate_at, set_learning_rate
from vdopd.pipeline import (
    DiagnosticSetting,
    diagnostic_setting_from_config,
    diagnostic_trainer,
    toy_diagnostic_setting,
)
from vdopd.trainer import Trainer

BRANCHES = {
    "Arithmetic": "arithmetic",
    "Weighted arithmetic": "weighted_arithmetic",
    "Geometric": "geometric",
    "VD-OPD": "vdopd",
    "Shuffled weight": "shuffled_weight",
    "Batch-mean weight": "batch_mean_weight",
}
# Supplementary E17: the five matched updates, in the order of the table.
SUPPLEMENTARY_BRANCHES = {
    "EOPD(full)": "eopd_full",
    "Geometric-1": "geometric",
    "Batch-mean-Z": "batch_mean_weight",
    "Shuffled-Z": "shuffled_weight",
    "VD-OPD": "vdopd",
}
E17_COLUMNS = (
    "stratum",
    "update",
    "oracle_kl_decrease",
    "oracle_kl_decrease_ci_low",
    "oracle_kl_decrease_ci_high",
    "shared_wrong_prob_change",
    "shared_wrong_prob_change_ci_low",
    "shared_wrong_prob_change_ci_high",
    "correctness_change_pp",
    "correctness_change_pp_ci_low",
    "correctness_change_pp_ci_high",
)


def toy_candidates(problems: Sequence[Problem]) -> dict[str, list[str]]:
    """Candidate answers of the synthetic integer task: the answer and three near misses."""
    return {
        p.id: [str(int(p.answer) + offset) for offset in (0, 1, -1, 2)]
        for p in problems
        if p.answer is not None
    }


def load_candidates(path: str) -> dict[str, list[str]]:
    """Read ``{"problem_id", "candidates": [correct, wrong, ...]}`` records (JSON Lines)."""
    candidates: dict[str, list[str]] = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                candidates[str(record["problem_id"])] = [str(c) for c in record["candidates"]]
    return candidates


def state_measurement(
    setting: DiagnosticSetting, args: argparse.Namespace
) -> tuple[Callable[[], dict[str, np.ndarray]], dict[str, Any]]:
    """Fixed analysis states of the measurement questions and their student measurement.

    Returns ``measure`` (evaluated before and after every branch) and the fixed context:
    the question index and ``Z`` quartile of every state and the states that have a wrong
    answer shared by both teacher views.
    """
    problems = setting.measurement_problems[: args.num_questions]
    texts = toy_candidates(problems) if args.toy else load_candidates(args.candidates)
    problems = [p for p in problems if p.id in texts and p.answer is not None]
    if not problems:
        raise SystemExit("no measurement problem has candidate answers")
    codec, decoding = setting.codec, setting.decoding
    prompts = [
        build_messages(
            p.question, setting.prompt_template, decoding.system_prompt, decoding.exemplars
        )
        for p in problems
    ]
    states = build_analysis_states(
        setting.student,
        codec,
        prompts,
        rollout=setting.rollout,
        seed=args.seed,
        num_prefixes=args.prefixes_per_question,
        teacher_context=setting.teacher.max_context,
    )
    agreement = state_agreement(
        setting.teacher,
        states,
        block_size=setting.train.position_block_size,
        microbatch_size=setting.train.microbatch_size,
    )
    prefixes = [state.prefix for state in states]
    originals = [state.original_ids for state in states]
    references = [problems[state.question].answer for state in states]

    # Paired teacher continuations identify the wrong answers shared by both views.
    teacher_answers = [
        sample_final_answers(
            setting.teacher,
            codec,
            [getattr(state, view) for state in states],
            prefixes,
            decoding,
            num_samples=args.continuations,
            seed=args.seed,
        )
        for view in ("original_ids", "view2_ids")
    ]
    shared = shared_wrong_answers(*teacher_answers, references, decoding.rel_tol)
    eligible = np.array([answer is not None for answer in shared])

    def encode(answer: str) -> list[int]:
        return codec.encode_text(args.candidate_template.replace("{answer}", answer))

    state_ids = [list(state.original_ids) + list(state.prefix) for state in states]
    candidates = [[encode(text) for text in texts[problems[s.question].id]] for s in states]
    shared_candidates, shared_index = [], []
    for state, answer in zip(states, shared, strict=True):
        if answer is None:
            continue
        options = list(texts[problems[state.question].id])
        match = [k for k, option in enumerate(options) if is_equivalent(answer, option)]
        if not match:  # the shared wrong answer joins the candidates of this state
            options.append(answer)
            match = [len(options) - 1]
        shared_candidates.append([encode(text) for text in options])
        shared_index.append(match[0])
    eligible_ids = [ids for ids, keep in zip(state_ids, eligible, strict=True) if keep]

    def measure() -> dict[str, np.ndarray]:
        size = setting.train.microbatch_size
        distributions = candidate_distributions(
            setting.student, state_ids, candidates, batch_size=size
        )
        shared_distributions = candidate_distributions(
            setting.student, eligible_ids, shared_candidates, batch_size=size
        )
        answers = sample_final_answers(
            setting.student,
            codec,
            originals,
            prefixes,
            decoding,
            num_samples=decoding.num_samples,
            seed=args.seed,
            stream=1,
        )
        correct = [
            np.mean([is_equivalent(answer, reference, decoding.rel_tol) for answer in samples])
            for samples, reference in zip(answers, references, strict=True)
        ]
        shared_wrong = np.full(len(states), np.nan)
        shared_wrong[eligible] = [
            distribution[index]
            for distribution, index in zip(shared_distributions, shared_index, strict=True)
        ]
        return {
            "candidates": np.stack(distributions),
            "shared_wrong": shared_wrong,
            "correct": np.array(correct),
        }

    context = {
        "questions": np.array([state.question for state in states]),
        "quartiles": quartile_labels(agreement["mass"]),
        "mass": agreement["mass"],
        "eligible": eligible,
        "num_questions": len(problems),
    }
    return measure, context


def _cells(interval: BootstrapCI | None, digits: int) -> tuple[str, str, str]:
    """Signed estimate and interval bounds of one table cell group (empty if undefined)."""
    if interval is None:
        return "", "", ""
    values = (interval.estimate, interval.low, interval.high)
    return tuple(number(value, digits, signed=True) for value in values)


def supplementary_rows(
    result: dict[str, Any], context: dict[str, Any], num_resamples: int, seed: int
) -> list[dict[str, str]]:
    """Rows of the E17 table: one per Z stratum and update."""
    before = result["before_score"]
    rows: list[dict[str, str]] = []
    summaries = {}
    for label, branch in result["branches"].items():
        after = branch["after_score"]
        quantities = (
            (oracle_kl_decrease(before["candidates"], after["candidates"]), None, 4),
            (after["shared_wrong"] - before["shared_wrong"], context["eligible"], 3),
            (100.0 * (after["correct"] - before["correct"]), None, 2),
        )
        summaries[label] = [
            (
                stratum_means(
                    values,
                    context["questions"],
                    context["quartiles"],
                    eligible=eligible,
                    num_resamples=num_resamples,
                    seed=seed,
                ),
                digits,
            )
            for values, eligible, digits in quantities
        ]
    for stratum, _ in Z_STRATA:
        for label, summary in summaries.items():
            cells = [cell for means, digits in summary for cell in _cells(means[stratum], digits)]
            rows.append(dict(zip(E17_COLUMNS, [stratum, label, *cells], strict=True)))
    return rows


def print_supplementary(rows: Sequence[dict[str, str]], context: dict[str, Any]) -> None:
    quartiles, eligible = context["quartiles"], context["eligible"]
    counts = {
        label: (int((quartiles == q).sum()), int((eligible & (quartiles == q)).sum()))
        for label, q in Z_STRATA
    }
    print(
        f"measurement states: {len(quartiles)} from {context['num_questions']} questions; "
        + "; ".join(
            f"{label} Z: {states} states, {shared} with a shared wrong answer"
            for label, (states, shared) in counts.items()
        )
    )
    print(
        "stratum  update         oracle KL decrease           shared-wrong change        "
        "correctness change (pp)"
    )
    for row in rows:
        values = [row[column] for column in E17_COLUMNS[2:]]
        groups = [
            f"{values[i] or 'n/a'} [{values[i + 1] or 'n/a'}, {values[i + 2] or 'n/a'}]"
            for i in (0, 3, 6)
        ]
        print(f"{row['stratum']:<8} {row['update']:<14} " + "  ".join(f"{g:<27}" for g in groups))


def print_table12(result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Print Table 12 and return the per-branch summary written to ``--output``."""
    pre = result["pre_update"]
    print("Table 12(a) - shared pre-update diagnostics (token means)")
    print(
        f"  H(p1)={pre['entropy_p1']:.3f}  H(p2)={pre['entropy_p2']:.3f}  "
        f"H(q)={pre['entropy_q']:.3f}  Z={pre['mass']:.3f}  positions={len(result['mass'])}"
    )
    print("Table 12(b) - matched branch updates")
    rows = {}
    scalar_scores = isinstance(result["before_score"], float)
    for label, branch in result["branches"].items():
        utility = branch["utility"].numpy()
        rows[label] = {
            "status": branch["status"],
            "num_positions": branch["num_positions"],
            "mean_utility": float(utility.mean()),
            "before": result["before_score"] if scalar_scores else None,
            "after": branch["after_score"] if scalar_scores else None,
        }
        line = f"  {label:<20} status={branch['status']:<9} mean u_t={utility.mean():+.5f}"
        if scalar_scores:
            before, after = result["before_score"], branch["after_score"]
            line += f"  Avg@k {before:.1f} -> {after:.1f} ({after - before:+.1f})"
        print(line)
    return rows


def figure6(
    result: dict[str, Any], setting: DiagnosticSetting, batch, plot: str | None
) -> dict[str, np.ndarray]:
    """Print (and optionally plot) the binned utility of the VD-OPD branch."""
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
    if plot:
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
        figure.savefig(plot, dpi=200)
    return summary


def branch_trainer(setting: DiagnosticSetting, checkpoint: str | None) -> Trainer:
    """Trainer at the branching point with the learning rate of the branch update."""
    trainer = diagnostic_trainer(setting)
    if checkpoint is None:
        # A student without optimizer state: the configured rate, no warmup.
        set_learning_rate(trainer.optimizer, setting.optim.learning_rate)
    else:
        # A training checkpoint: its optimizer state and the scheduled rate of its iteration.
        trainer.load_checkpoint(checkpoint)
        rate = learning_rate_at(trainer.iteration, trainer.total_iterations, setting.optim)
        set_learning_rate(trainer.optimizer, rate)
    return trainer


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", help="YAML experiment file (real models)")
    source.add_argument("--toy", action="store_true", help="synthetic toy pair (CPU)")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--student", default=None, help="student weights to branch from")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="training checkpoint folder to branch from (student weights and optimizer state)",
    )
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--no-accuracy", action="store_true", help="skip the Avg@k evaluations")
    parser.add_argument("--output", default=None)
    parser.add_argument("--plot", default=None, help="write Figure 6 to this image file")
    extra = parser.add_argument_group("supplementary experiment E17")
    extra.add_argument("--supplementary", action="store_true", help="run the E17 analysis")
    extra.add_argument("--candidates", default=None, help="candidate answers (JSON Lines)")
    extra.add_argument("--candidate-template", default="{answer}", help="scored answer string")
    extra.add_argument("--num-questions", type=int, default=None, help="default: all")
    extra.add_argument("--prefixes-per-question", type=int, default=4)
    extra.add_argument("--continuations", type=int, default=4, help="teacher pairs per state")
    extra.add_argument("--resamples", type=int, default=DEFAULT_RESAMPLES)
    extra.add_argument("--csv", default=None, help="write the E17 table to this CSV file")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)
    if args.supplementary and not (args.toy or args.candidates):
        parser.error("--supplementary needs --candidates (or --toy)")

    if args.toy:
        setting = toy_diagnostic_setting(args.seed)
    else:
        setting = diagnostic_setting_from_config(
            load_config(args.config, args.overrides), args.student
        )
    trainer = branch_trainer(setting, args.checkpoint)
    batch = trainer.collect_rollouts(0)

    def accuracy() -> float:
        result = evaluate(
            setting.student,
            setting.codec,
            setting.measurement_problems,
            setting.decoding,
            unseen_wrappers=(),
            keep_questions=False,
        )
        return 100.0 * result["avg_at_k"]

    measure: Callable[[], Any] | None = None if args.no_accuracy else accuracy
    context: dict[str, Any] = {}
    branches = BRANCHES
    if args.supplementary:
        branches = SUPPLEMENTARY_BRANCHES
        measure, context = state_measurement(setting, args)
    objectives = {
        label: build_objective(ObjectiveConfig(name=name)) for label, name in branches.items()
    }
    result = matched_branch_updates(trainer, batch, objectives, measure)
    rows = print_table12(result)
    payload: dict[str, Any] = {"pre_update": result["pre_update"], "branches": rows}

    if args.supplementary:
        table = supplementary_rows(result, context, args.resamples, args.seed)
        print_supplementary(table, context)
        payload["e17"] = table
        if args.csv:
            write_csv(args.csv, E17_COLUMNS, table)
    else:
        summary = figure6(result, setting, batch, args.plot)
        payload["figure6"] = {key: np.asarray(value).tolist() for key, value in summary.items()}

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)


if __name__ == "__main__":
    main()
