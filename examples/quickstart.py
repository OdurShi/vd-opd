#!/usr/bin/env python
"""Quick start: the whole VD-OPD loop on CPU with tiny models built in code.

Equivalent to ``vdopd demo``. Two tiny transformers are created with random weights; the
teacher (and, briefly, the initial student) is fitted to a synthetic arithmetic task, and
the student is then distilled with Algorithm 1: on-policy rollouts, two-view teacher scoring
at the same student prefix, the consensus loss, one update per rollout batch, evaluation
under the canonical prompt and the four held-out wrappers.

The numbers describe a toy problem; they are unrelated to the paper's experiments.
"""

from __future__ import annotations

import logging

from vdopd.demo import DemoConfig, format_results, run_demo


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    cfg = DemoConfig()  # try DemoConfig(objective="gkd_rkl") or objective="eopd_2v"
    results = run_demo(cfg)
    print()
    print(format_results(results, cfg.eval_samples))
    training = results["training"]
    print(
        f"\nloss {training['first_loss']:.4f} -> {training['last_loss']:.4f} over "
        f"{training['iterations']} iterations; mean retained mass Z = "
        f"{training['mean_mass']:.3f}; {results['seconds']:.0f} s"
    )


if __name__ == "__main__":
    main()
