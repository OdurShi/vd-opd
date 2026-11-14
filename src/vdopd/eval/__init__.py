"""Evaluation: answer checking, Avg@k / Pass@k and unseen-view robustness."""

from vdopd.eval.answers import extract_answer, is_equivalent, normalize_answer
from vdopd.eval.metrics import avg_at_k, macro_average, pass_at_k, view_robustness

__all__ = [
    "avg_at_k",
    "extract_answer",
    "is_equivalent",
    "macro_average",
    "normalize_answer",
    "pass_at_k",
    "view_robustness",
]
