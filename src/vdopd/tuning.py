"""Development-split grid search for baseline coefficients (supplementary E11).

The additional EOPD experiments of E11 search a 3 x 3 grid over the entropy threshold
``tau`` and the forward-KL coefficient ``alpha`` on the development split, separately for
EOPD (top-16), EOPD (full) and EOPD-2V. Model selection follows Appendix B.4: the
configuration with the highest development Avg@k wins, ties are resolved by lower training
cost. This module enumerates the grid and selects the winner; ``scripts/tune_eopd.py`` runs
the trials.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

DEVELOPMENT_SPLIT = "math_development"


@dataclass(frozen=True)
class TuningConfig:
    """Grid of the development search (``tuning:`` section of a YAML file).

    The grid values are not stated in the description of E11; the defaults are a 3 x 3 grid
    centred on the default coefficients (``tau = 0.8`` nats, ``alpha = 1.0``) that contains
    the three selected configurations reported there.
    """

    eopd_entropy_thresholds: tuple[float, ...] = (0.6, 0.8, 1.0)
    eopd_fkl_weights: tuple[float, ...] = (0.8, 1.0, 1.2)
    benchmark: str = DEVELOPMENT_SPLIT
    limit: int | None = None


def eopd_grid(tuning: TuningConfig) -> list[dict[str, float]]:
    """All ``(tau, alpha)`` combinations of the grid, ``tau`` varying slowest."""
    return [
        {"eopd_entropy_threshold": float(tau), "eopd_fkl_weight": float(alpha)}
        for tau in tuning.eopd_entropy_thresholds
        for alpha in tuning.eopd_fkl_weights
    ]


def grid_overrides(point: Mapping[str, float]) -> list[str]:
    """``--set`` overrides that select one grid point."""
    return [f"objective.{key}={value}" for key, value in point.items()]


def trial_name(point: Mapping[str, float]) -> str:
    """Folder name of one trial, e.g. ``tau0.6_alpha1.2``."""
    return f"tau{point['eopd_entropy_threshold']:g}_alpha{point['eopd_fkl_weight']:g}"


def run_grid(
    points: Sequence[Mapping[str, float]], trial: Callable[[Mapping[str, float]], Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Run ``trial`` on every grid point.

    ``trial`` trains one configuration and returns at least ``score`` (development Avg@k)
    and ``flops`` (total training FLOPs). The returned records add the grid point.
    """
    return [{**point, **trial(point)} for point in points]


def select_best(records: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    """Highest development score; ties go to the lower training cost, then to grid order."""
    if not records:
        raise ValueError("no trials")
    return min(records, key=lambda record: (-record["score"], record["flops"]))
