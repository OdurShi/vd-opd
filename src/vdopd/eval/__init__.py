"""Evaluation: answer checking, Avg@k / Pass@k, unseen-view robustness and bootstrap CIs."""

from vdopd.eval.answers import extract_answer, is_equivalent, normalize_answer
from vdopd.eval.bootstrap import (
    BootstrapCI,
    clustered_mean_bootstrap,
    paired_question_bootstrap,
    stratified_paired_bootstrap,
)
from vdopd.eval.metrics import avg_at_k, macro_average, pass_at_k, view_robustness

__all__ = [
    "BootstrapCI",
    "avg_at_k",
    "clustered_mean_bootstrap",
    "extract_answer",
    "is_equivalent",
    "macro_average",
    "normalize_answer",
    "paired_question_bootstrap",
    "pass_at_k",
    "stratified_paired_bootstrap",
    "view_robustness",
]
