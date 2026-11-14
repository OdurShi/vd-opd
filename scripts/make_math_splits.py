#!/usr/bin/env python
"""Derive the Appendix B.1 partition of the MATH training split as index files.

    python scripts/make_math_splits.py

Reads ``data/math_train/train.jsonl`` and the six evaluation benchmarks (shipped files, or
the Hub for those that are not shipped), removes duplicate problems and problems that
also occur in an evaluation benchmark, and writes the stratified partition with split seed 42
to ``data/math_train/splits/``:

    deduplicated.txt             problems kept after deduplication and overlap removal
    optimization.txt             optimisation set
    development.txt              500 development problems
    diagnostic.txt               500 diagnostic problems
    diagnostic_update.txt        250 diagnostic problems used for branch updates (App. B.7)
    diagnostic_measurement.txt   250 diagnostic problems used for measurement (App. B.7)
    split_info.json              parameters and sizes

The procedure is deterministic; the files are checked against a recomputation in the tests.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vdopd.data import paths
from vdopd.data.benchmarks import SIX_MATH, load_benchmark
from vdopd.data.math_train import (
    SPLIT_DIR,
    deduplicate,
    load_math_train,
    remove_overlaps,
    split_parameters,
    stratified_split,
    write_split_files,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-diagnostic", type=int, default=500)
    parser.add_argument("--num-development", type=int, default=500)
    args = parser.parse_args()
    if args.data_dir is not None:
        paths.set_data_dir(args.data_dir)

    problems = load_math_train()
    unique = deduplicate(problems)
    evaluation_questions = [p.question for name in SIX_MATH for p in load_benchmark(name)]
    cleaned = remove_overlaps(unique, evaluation_questions)
    splits = stratified_split(
        cleaned,
        num_diagnostic=args.num_diagnostic,
        num_development=args.num_development,
        seed=args.seed,
    )
    parameters = split_parameters(args.num_diagnostic, args.num_development, args.seed, SIX_MATH)
    written = write_split_files(cleaned, splits, parameters)
    print(
        f"problems {len(problems)} -> deduplicated {len(unique)} -> without overlaps "
        f"{len(cleaned)}; optimization {len(splits.optimization)}, development "
        f"{len(splits.development)}, diagnostic {len(splits.diagnostic)}"
    )

    directory = paths.data_dir()
    manifest_path = directory / "manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.is_file()
        else {"files": {}}
    )
    for name in written:
        relative = f"{SPLIT_DIR}/{name}"
        entry = {
            "derived_from": "math_train/train.jsonl",
            "sha256": paths.sha256(directory / relative),
        }
        if name.endswith(".txt"):
            entry["num_examples"] = len(paths.read_lines(directory / relative))
        manifest["files"][relative] = entry
    manifest["files"] = dict(sorted(manifest["files"].items()))
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
