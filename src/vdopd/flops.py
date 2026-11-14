"""Training-compute accounting for the compute-matched controls (Appendix B.5).

The paper reports "recorded" FLOPs that "include student generation, teacher
generation/scoring, and student forward/backward computation" but does not give the
counting rule. This module uses the standard analytic estimate for transformer language
models, ``2 * (matrix-product parameters)`` FLOPs per processed token for a forward pass and
twice that for the backward pass. Absolute values therefore need not match the paper's
figures; the *structure* of the accounting (three components, per-view teacher scoring, the
total- and teacher-budget stopping rules) follows Appendix B.5.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

FORWARD_FLOPS_PER_PARAMETER_TOKEN = 2.0
BACKWARD_FLOPS_PER_PARAMETER_TOKEN = 4.0


def forward_flops(num_parameters: int, num_tokens: int) -> float:
    """FLOPs of a forward pass over ``num_tokens`` tokens."""
    return FORWARD_FLOPS_PER_PARAMETER_TOKEN * num_parameters * num_tokens


def training_flops(
    num_parameters: int, num_tokens: int, gradient_checkpointing: bool = False
) -> float:
    """FLOPs of a forward and backward pass; checkpointing recomputes one forward pass."""
    per_token = FORWARD_FLOPS_PER_PARAMETER_TOKEN + BACKWARD_FLOPS_PER_PARAMETER_TOKEN
    if gradient_checkpointing:
        per_token += FORWARD_FLOPS_PER_PARAMETER_TOKEN
    return per_token * num_parameters * num_tokens


@dataclass
class FlopsMeter:
    """Cumulative FLOPs of the local training pipeline, split as in Appendix B.5.2.

    ``teacher_scoring`` counts the frozen teacher's scoring passes for all prompt views and
    is the horizontal axis of Figure 5; ``total`` is the quantity matched by VD-OPD-Budget
    in Table 2.
    """

    student_generation: float = 0.0
    teacher_scoring: float = 0.0
    student_training: float = 0.0

    @property
    def total(self) -> float:
        return self.student_generation + self.teacher_scoring + self.student_training

    def add_generation(self, num_parameters: int, num_tokens: int) -> None:
        self.student_generation += forward_flops(num_parameters, num_tokens)

    def add_teacher_scoring(self, num_parameters: int, num_tokens: int) -> None:
        self.teacher_scoring += forward_flops(num_parameters, num_tokens)

    def add_student_training(
        self, num_parameters: int, num_tokens: int, gradient_checkpointing: bool = False
    ) -> None:
        self.student_training += training_flops(num_parameters, num_tokens, gradient_checkpointing)

    def state_dict(self) -> dict[str, float]:
        return asdict(self)

    def load_state_dict(self, state: dict[str, float]) -> None:
        self.student_generation = float(state["student_generation"])
        self.teacher_scoring = float(state["teacher_scoring"])
        self.student_training = float(state["student_training"])


def budget_reached(
    meter: FlopsMeter, max_total_flops: float | None, max_teacher_flops: float | None
) -> bool:
    """Stopping rule of the compute controls (Appendix B.5.3).

    ``max_total_flops`` shortens training to a total-FLOPs budget (VD-OPD-Budget);
    ``max_teacher_flops`` stops at a teacher-scoring budget (Figure 5).
    """
    if max_total_flops is not None and meter.total >= max_total_flops:
        return True
    return max_teacher_flops is not None and meter.teacher_scoring >= max_teacher_flops
