"""Command-line interface: ``vdopd {train,eval,demo,flops}``."""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
from collections.abc import Sequence


def _add_config_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, help="YAML experiment file")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a config entry, e.g. --set train.seed=22 (repeatable)",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="directory with the dataset files (default: $VDOPD_DATA_DIR or ./data of the repo)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vdopd", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train", help="distil a student (Algorithm 1)")
    _add_config_arguments(train)

    evaluate = commands.add_parser("eval", help="evaluate a student over prompt views")
    _add_config_arguments(evaluate)
    evaluate.add_argument("--student", default=None, help="checkpoint dir (default: initial)")
    evaluate.add_argument("--output", default=None, help="write the results to this JSON file")

    demo = commands.add_parser("demo", help="CPU quick start on a synthetic arithmetic task")
    demo.add_argument("--objective", default="vdopd")
    demo.add_argument("--iterations", type=int, default=None)
    demo.add_argument("--seed", type=int, default=None)
    demo.add_argument("--output", default=None, help="write the metrics to this JSON file")

    flops = commands.add_parser("flops", help="print the iteration budget of a config")
    _add_config_arguments(flops)
    return parser


def _overrides(args: argparse.Namespace) -> list[str]:
    extra = [f"data.data_dir={args.data_dir}"] if args.data_dir else []
    return [*args.overrides, *extra]


def _command_train(args: argparse.Namespace) -> int:
    from vdopd.config import load_config
    from vdopd.pipeline import run_training

    trainer = run_training(load_config(args.config, _overrides(args)))
    print(json.dumps({"iterations": trainer.iteration, "flops": trainer.flops.state_dict()}))
    return 0


def _command_eval(args: argparse.Namespace) -> int:
    from vdopd.config import load_config
    from vdopd.pipeline import run_evaluation

    results = run_evaluation(load_config(args.config, _overrides(args)), args.student, args.output)
    print(json.dumps(results["macro"], indent=2))
    return 0


def _command_demo(args: argparse.Namespace) -> int:
    from vdopd.demo import DemoConfig, format_results, run_demo

    changes = {
        key: value
        for key, value in (
            ("objective", args.objective),
            ("iterations", args.iterations),
            ("seed", args.seed),
        )
        if value is not None
    }
    cfg = dataclasses.replace(DemoConfig(), **changes)
    results = run_demo(cfg)
    print()
    print(format_results(results, cfg.eval_samples))
    training = results["training"]
    print(
        f"\nloss {training['first_loss']:.4f} -> {training['last_loss']:.4f} over "
        f"{training['iterations']} iterations; mean retained mass Z = "
        + ("n/a" if training["mean_mass"] is None else f"{training['mean_mass']:.3f}")
        + f"; {results['seconds']:.0f} s"
    )
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)
    return 0


def _command_flops(args: argparse.Namespace) -> int:
    from vdopd.config import load_config
    from vdopd.optim import warmup_steps

    cfg = load_config(args.config, _overrides(args))
    print(
        json.dumps(
            {
                "name": cfg.name,
                "objective": cfg.objective.name,
                "rollout_batch_size": cfg.rollout.batch_size,
                "passes": cfg.train.passes,
                "iterations": cfg.train.iterations,
                "warmup_fraction": cfg.optim.warmup_fraction,
                "warmup_iterations_if_fixed": None
                if cfg.train.iterations is None
                else warmup_steps(cfg.train.iterations, cfg.optim.warmup_fraction),
                "max_total_flops": cfg.train.max_total_flops,
                "max_teacher_flops": cfg.train.max_teacher_flops,
            },
            indent=2,
        )
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point of the ``vdopd`` console script."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    args = build_parser().parse_args(argv)
    handlers = {
        "train": _command_train,
        "eval": _command_eval,
        "demo": _command_demo,
        "flops": _command_flops,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
