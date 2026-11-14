#!/usr/bin/env python
"""Teacher diagnostics by cross-view agreement Z (supplementary experiment E16).

For every measurement question the student generates a few responses, each of which is cut
at a random position; the resulting fixed prefixes are the analysis states. At every state
the frozen teacher's next-token distributions under the original prompt and under a wrapped
prompt give the agreement ``Z``. States are split into quartiles of ``Z``; the script prints,
for the bottom (Q1) and top (Q4) quartile, the number of states and continuation pairs, the
mean ``Z``, how often both views share the most likely next token, the share of the squared
Hellinger distance that lies outside the union of both views' top-10 tokens, and the
outcomes of paired teacher continuations (both correct, exactly one correct, the same wrong
final answer, different wrong answers).

    python scripts/agreement_diagnostic.py --toy --csv out/E16_teacher_diagnostics_by_Z.csv
    python scripts/agreement_diagnostic.py --config configs/qwen3_1p7b/vdopd.yaml \
        --student runs/qwen3_1p7b/vdopd/seed11/checkpoint-50/student

``--toy`` uses the synthetic toy pair of the quick start (not the paper's models).
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence
from pathlib import Path

from vdopd.config import load_config
from vdopd.data.problems import build_messages
from vdopd.diagnostics import (
    agreement_summary,
    agreement_table,
    build_analysis_states,
    pair_outcomes,
    sample_final_answers,
    state_agreement,
)
from vdopd.eval.tables import write_csv
from vdopd.pipeline import diagnostic_setting_from_config, toy_diagnostic_setting

COLUMNS = ("diagnostic", "low_z_q1", "high_z_q4")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", help="YAML experiment file (real models)")
    source.add_argument("--toy", action="store_true", help="synthetic toy pair (CPU)")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--student", default=None, help="student checkpoint that generates")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--num-questions", type=int, default=None, help="default: all")
    parser.add_argument("--prefixes-per-question", type=int, default=4)
    parser.add_argument("--continuations", type=int, default=4, help="teacher pairs per state")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output", default=None, help="write all quartiles to this JSON file")
    parser.add_argument("--csv", default=None, help="write the Q1/Q4 table to this CSV file")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)

    if args.toy:
        setting = toy_diagnostic_setting(args.seed)
    else:
        setting = diagnostic_setting_from_config(
            load_config(args.config, args.overrides), args.student
        )
    problems = setting.measurement_problems[: args.num_questions]
    prompts = [
        build_messages(
            p.question,
            setting.prompt_template,
            setting.decoding.system_prompt,
            setting.decoding.exemplars,
        )
        for p in problems
    ]
    states = build_analysis_states(
        setting.student,
        setting.codec,
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
        top_k=args.top_k,
    )
    outcomes = None
    if args.continuations > 0:
        prefixes = [state.prefix for state in states]
        answers = [
            sample_final_answers(
                setting.teacher,
                setting.codec,
                [getattr(state, view) for state in states],
                prefixes,
                setting.decoding,
                num_samples=args.continuations,
                seed=args.seed,
            )
            for view in ("original_ids", "view2_ids")  # same seeds: paired continuations
        ]
        references = [problems[state.question].answer or "" for state in states]
        outcomes = pair_outcomes(*answers, references, setting.decoding.rel_tol)

    summary = agreement_summary(agreement, outcomes)
    rows = agreement_table(summary)
    print(f"analysis states: {len(states)} from {len(problems)} questions")
    print(f"{'Diagnostic':<52}{'Low Z (Q1)':>12}{'High Z (Q4)':>13}")
    for row in rows:
        print(f"{row['diagnostic']:<52}{row['low_z_q1']:>12}{row['high_z_q4']:>13}")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump({"num_states": len(states), "quartiles": summary}, handle, indent=2)
    if args.csv:
        write_csv(args.csv, COLUMNS, rows)


if __name__ == "__main__":
    main()
