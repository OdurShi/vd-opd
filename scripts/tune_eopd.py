#!/usr/bin/env python
"""Development-split grid over EOPD's coefficients (supplementary experiment E11).

For one EOPD variant (``eopd_top16``, ``eopd_full`` or ``eopd_2v``) every combination of the
entropy threshold tau and the forward-KL coefficient alpha in the ``tuning`` section of the
config (a 3 x 3 grid by default) is trained once and scored on the MATH development split
under the common decoding protocol. The configuration with the highest development Avg@k is
selected; ties go to the lower training cost (Appendix B.4).

    python scripts/tune_eopd.py --config configs/qwen3_1p7b/eopd_full.yaml --dry-run
    python scripts/tune_eopd.py --config configs/qwen3_1p7b/eopd_full.yaml \
        --output-dir runs/tuning/qwen3_1p7b/eopd_full
    python scripts/tune_eopd.py --config configs/qwen3_1p7b/eopd_2v.yaml \
        --set tuning.eopd_entropy_thresholds=[0.6,0.8,1.0] --set tuning.eopd_fkl_weights=[1.0,1.2]

Each trial is written to ``<output-dir>/<tau..._alpha...>/`` (training run and
``trial.json``); finished trials are reused when the script is started again. The summary is
``<output-dir>/grid.json``. The trials are part of model selection and are not counted in
the training FLOPs of the selected run.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from vdopd.config import load_config
from vdopd.tuning import eopd_grid, grid_overrides, run_grid, select_best, trial_name


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", required=True, help="YAML file of the EOPD variant")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument(
        "--output-dir", default=None, help="default: runs/tuning/<config folder>/<config name>"
    )
    parser.add_argument("--dry-run", action="store_true", help="list the trials and exit")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")

    base = load_config(args.config, args.overrides)
    if not base.objective.name.startswith("eopd"):
        raise SystemExit(f"{args.config} is not an EOPD configuration ({base.objective.name})")
    config_path = Path(args.config)
    default_dir = Path("runs") / "tuning" / config_path.parent.name / config_path.stem
    output_dir = Path(args.output_dir or default_dir)
    points = eopd_grid(base.tuning)

    def overrides(point: Mapping[str, float]) -> list[str]:
        folder = output_dir / trial_name(point)
        return [*args.overrides, *grid_overrides(point), f"train.output_dir={folder}"]

    if args.dry_run:
        print(f"{len(points)} development trials on {base.tuning.benchmark}:")
        for point in points:
            print("  " + " ".join(f"--set {item}" for item in overrides(point)))
        return

    def trial(point: Mapping[str, float]) -> dict[str, Any]:
        from vdopd.pipeline import run_tuning_trial

        record_path = output_dir / trial_name(point) / "trial.json"
        if record_path.is_file():
            with open(record_path, encoding="utf-8") as handle:
                return json.load(handle)
        record = run_tuning_trial(load_config(args.config, overrides(point)))
        with open(record_path, "w", encoding="utf-8") as handle:
            json.dump(record, handle, indent=2)
        return record

    records = run_grid(points, trial)
    best = select_best(records)
    print("tau    alpha  dev Avg@k (%)  total FLOPs")
    for record in records:
        marker = "  <- selected" if record is best else ""
        print(
            f"{record['eopd_entropy_threshold']:<6g} {record['eopd_fkl_weight']:<6g} "
            f"{100 * record['score']:<14.2f} {record['flops']:.3e}{marker}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "grid.json", "w", encoding="utf-8") as handle:
        json.dump({"trials": records, "selected": best}, handle, indent=2)


if __name__ == "__main__":
    main()
