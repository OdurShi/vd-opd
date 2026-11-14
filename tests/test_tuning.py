"""Supplementary E11: development-split grid over EOPD's coefficients and tuned configs."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from conftest import make_trainer
from vdopd.config import config_from_dict, load_config
from vdopd.data.synthetic import SYNTHETIC_TEMPLATE
from vdopd.objectives import EOPD, UNSPECIFIED_BASELINES, build_objective
from vdopd.tuning import (
    DEVELOPMENT_SPLIT,
    TuningConfig,
    eopd_grid,
    grid_overrides,
    run_grid,
    select_best,
    trial_name,
)

REPOSITORY = Path(__file__).resolve().parents[1]
CONFIGS = REPOSITORY / "configs" / "qwen3_1p7b"
# (tau, alpha) selected in E11 for EOPD (top-16), EOPD (full) and EOPD-2V.
SELECTED = {"eopd_top16": (0.8, 1.0), "eopd_full": (0.6, 1.0), "eopd_2v": (0.8, 1.2)}


def test_default_grid_is_three_by_three_and_contains_the_selected_points():
    tuning = TuningConfig()
    points = eopd_grid(tuning)
    assert len(points) == 9 and tuning.benchmark == DEVELOPMENT_SPLIT == "math_development"
    pairs = [(p["eopd_entropy_threshold"], p["eopd_fkl_weight"]) for p in points]
    assert len(set(pairs)) == 9 and (0.8, 1.0) in pairs  # includes the default coefficients
    assert set(SELECTED.values()) <= set(pairs)
    assert pairs[:3] == [(0.6, 0.8), (0.6, 1.0), (0.6, 1.2)]  # tau varies slowest
    assert grid_overrides(points[1]) == [
        "objective.eopd_entropy_threshold=0.6",
        "objective.eopd_fkl_weight=1.0",
    ]
    assert trial_name(points[1]) == "tau0.6_alpha1"
    custom = TuningConfig(eopd_entropy_thresholds=(0.5,), eopd_fkl_weights=(1, 2))
    assert [trial_name(p) for p in eopd_grid(custom)] == ["tau0.5_alpha1", "tau0.5_alpha2"]


def test_grid_values_are_a_config_option():
    cfg = load_config(
        CONFIGS / "eopd_full.yaml",
        ["tuning.eopd_entropy_thresholds=[0.4,0.8]", "tuning.eopd_fkl_weights=[1.0]"],
    )
    assert cfg.tuning.eopd_entropy_thresholds == (0.4, 0.8) and len(eopd_grid(cfg.tuning)) == 2
    assert load_config(CONFIGS / "eopd_full.yaml").tuning == TuningConfig()
    point = eopd_grid(cfg.tuning)[0]
    trial = load_config(CONFIGS / "eopd_full.yaml", grid_overrides(point))
    objective = build_objective(trial.objective)
    assert isinstance(objective, EOPD) and objective.entropy_threshold == 0.4
    with pytest.raises(KeyError):
        config_from_dict({"tuning": {"taus": [0.8]}})


def test_selection_maximises_the_development_score_with_ties_to_lower_cost():
    calls = []

    def trial(point):
        calls.append(point)
        best = point["eopd_entropy_threshold"] == 0.6 and point["eopd_fkl_weight"] >= 1.0
        return {"score": 0.5 if best else 0.4, "flops": 100.0 * point["eopd_fkl_weight"]}

    records = run_grid(eopd_grid(TuningConfig()), trial)
    assert len(records) == len(calls) == 9 and all("score" in r and "flops" in r for r in records)
    best = select_best(records)
    # Two configurations tie at the top score; the cheaper one is selected (Appendix B.4).
    assert (best["eopd_entropy_threshold"], best["eopd_fkl_weight"]) == (0.6, 1.0)
    same = [{"score": 0.3, "flops": 1.0, "id": 0}, {"score": 0.3, "flops": 1.0, "id": 1}]
    assert select_best(same)["id"] == 0  # remaining ties keep the grid order
    with pytest.raises(ValueError):
        select_best([])


@pytest.mark.parametrize(("method", "selected"), SELECTED.items())
def test_tuned_configs_hold_the_selected_coefficients(method, selected):
    tuned = load_config(CONFIGS / f"{method}_tuned.yaml")
    original = load_config(CONFIGS / f"{method}.yaml")
    assert tuned.objective.name == original.objective.name == method
    assert (tuned.objective.eopd_entropy_threshold, tuned.objective.eopd_fkl_weight) == selected
    assert (original.objective.eopd_entropy_threshold, original.objective.eopd_fkl_weight) == (
        0.8,
        1.0,
    )
    assert tuned.optim == original.optim and tuned.rollout == original.rollout
    objective = build_objective(tuned.objective)
    terms = objective._terms if method == "eopd_2v" else (objective,)
    for term in terms:
        assert (term.entropy_threshold, term.fkl_weight) == selected
        assert term.top_k == (16 if method == "eopd_top16" else None)
    assert objective.num_views == (2 if method == "eopd_2v" else 1)


def test_other_adaptive_baselines_remain_registration_hooks():
    """ToDi-OP, TSD-KD and Veto are not specified by the paper; E11 only names their settings."""
    assert {"todi", "tsd_kd", "veto"} <= set(UNSPECIFIED_BASELINES)


def test_development_split_is_available_as_an_evaluation_set():
    from vdopd.pipeline import load_evaluation_problems, load_training_problems

    cfg = config_from_dict({})
    try:
        development = load_evaluation_problems(DEVELOPMENT_SPLIT, cfg)
    except Exception as error:  # the shipped MATH file is absent and there is no network
        pytest.skip(f"MATH training data not available: {error}")
    # 500 reserved problems; the few whose solution has no boxed answer cannot be scored.
    assert 495 <= len(development) <= 500 and all(p.answer for p in development)
    optimisation = {p.id for p in load_training_problems(cfg)}
    assert optimisation.isdisjoint(p.id for p in development)
    limited = config_from_dict({"evaluation": {"limit": 7}})
    assert len(load_evaluation_problems(DEVELOPMENT_SPLIT, limited)) == 7


def test_tuning_trial_scores_the_trained_student_on_the_development_set(
    make_lm, codec, monkeypatch
):
    import vdopd.pipeline as pipeline

    trainer = make_trainer(make_lm, codec, "eopd_full", iterations=1)
    monkeypatch.setattr(pipeline, "run_training", lambda cfg: (trainer.train(), trainer)[1])
    cfg = config_from_dict(
        {
            "data": {"source": "synthetic_arithmetic", "prompt_template": SYNTHETIC_TEMPLATE},
            "objective": {"name": "eopd_full"},
            "tuning": {"benchmark": "synthetic_arithmetic", "limit": 5},
            "evaluation": {
                "limit": 3,
                "decoding": {"num_samples": 2, "max_response_tokens": 6, "batch_size": 8},
            },
        }
    )
    record = pipeline.run_tuning_trial(cfg)
    assert record["num_questions"] == 5  # tuning.limit, not evaluation.limit
    assert 0.0 <= record["score"] <= record["pass_at_k"] <= 1.0
    assert record["flops"] == trainer.flops.total > 0 and record["iterations"] == 1


def test_tuning_script_lists_the_nine_trials(capsys):
    path = REPOSITORY / "scripts" / "tune_eopd.py"
    spec = importlib.util.spec_from_file_location("tune_eopd", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main(["--config", str(CONFIGS / "eopd_2v.yaml"), "--dry-run", "--output-dir", "runs/t"])
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "9 development trials on math_development:"
    assert len(lines) == 10 and all(
        "--set objective.eopd_fkl_weight=" in line for line in lines[1:]
    )
    assert "--set train.output_dir=runs/t/tau0.8_alpha1.2" in "\n".join(lines)
    with pytest.raises(SystemExit):
        module.main(["--config", str(CONFIGS / "vdopd.yaml"), "--dry-run"])
