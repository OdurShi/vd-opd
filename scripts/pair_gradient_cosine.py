#!/usr/bin/env python
"""Gradient agreement of the three possible view pairs (supplementary experiment E09).

Default training pairs the original prompt of every response with one alternative view
sampled uniformly from the three wrappers of Table 5. On one fixed rollout batch (the same
student parameters and the same sampled responses) this script accumulates the gradient of
the VD-OPD objective three times, each time pairing every response with the same wrapper,
and prints the cosine similarity between each pair's gradient and the mean gradient over the
three pairs.

    python scripts/pair_gradient_cosine.py --toy --csv out/E09_pair_gradient_cosine.csv
    python scripts/pair_gradient_cosine.py --config configs/qwen3_1p7b/vdopd.yaml \
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
from vdopd.diagnostics import pair_gradient_cosines
from vdopd.eval.tables import write_csv
from vdopd.pipeline import (
    diagnostic_setting_from_config,
    diagnostic_trainer,
    toy_diagnostic_setting,
)

COLUMNS = ("paired_view", "cosine_to_mean_gradient")
VIEW_LABELS = {
    "instruction_1": "Instruction 1",
    "instruction_2": "Instruction 2",
    "layout": "Layout wrapper",
}


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", help="YAML experiment file (real models)")
    source.add_argument("--toy", action="store_true", help="synthetic toy pair (CPU)")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--student", default=None, help="student checkpoint to evaluate at")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--batch", type=int, default=0, help="index of the rollout batch")
    parser.add_argument("--storage-device", default="cpu", help="where gradient copies are kept")
    parser.add_argument("--output", default=None, help="write all statistics to this JSON file")
    parser.add_argument("--csv", default=None, help="write the table to this CSV file")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)

    if args.toy:
        setting = toy_diagnostic_setting(args.seed)
    else:
        setting = diagnostic_setting_from_config(
            load_config(args.config, args.overrides), args.student
        )
    trainer = diagnostic_trainer(setting)
    batch = trainer.collect_rollouts(args.batch)
    result = pair_gradient_cosines(trainer, batch, storage_device=args.storage_device)

    rows = [
        {COLUMNS[0]: VIEW_LABELS.get(name, name), COLUMNS[1]: f"{cosine:.2f}"}
        for name, cosine in result["cosine_to_mean"].items()
    ]
    print(f"fixed states: {result['num_positions']} response positions of one rollout batch")
    print("Original prompt paired with   Cosine to three-pair mean gradient")
    for row in rows:
        print(f"  {row[COLUMNS[0]]:<28}{row[COLUMNS[1]]}")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
    if args.csv:
        write_csv(args.csv, COLUMNS, rows)


if __name__ == "__main__":
    main()
