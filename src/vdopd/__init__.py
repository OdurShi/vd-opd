"""View-Decomposed On-Policy Distillation (VD-OPD).

The package implements the training pipeline of the paper "View-Decomposed On-Policy
Distillation for Robust Reasoning in Large Language Models": student rollouts, two-view
teacher scoring at the same student-generated prefix, the unnormalised geometric consensus
target and its retained mass, the distillation baselines that share the same loop, and the
evaluation protocol (Avg@k, Pass@k, unseen-prompt robustness).
"""

from vdopd.consensus import (
    arithmetic_mean,
    bhattacharyya_affinity,
    entropy,
    generalized_kl,
    geometric_consensus,
    js_divergence,
    view_decomposition,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "arithmetic_mean",
    "bhattacharyya_affinity",
    "entropy",
    "generalized_kl",
    "geometric_consensus",
    "js_divergence",
    "view_decomposition",
]
