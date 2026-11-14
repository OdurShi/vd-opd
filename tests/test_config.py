"""Configuration files reproduce the hyper-parameters stated in the paper."""

from __future__ import annotations

from pathlib import Path

import pytest

from vdopd.cli import build_parser
from vdopd.config import ExperimentConfig, apply_overrides, config_from_dict, load_config
from vdopd.objectives import build_objective

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
METHOD_FILES = sorted(
    path for path in CONFIG_DIR.glob("*/*.yaml") if not path.name.startswith("base")
)


def test_all_configs_load_and_build_an_objective():
    assert len(METHOD_FILES) >= 18
    for path in METHOD_FILES:
        cfg = load_config(path)
        objective = build_objective(cfg.objective)
        assert objective.num_views in (1, 2), path


def test_main_setting_matches_the_paper():
    cfg = load_config(CONFIG_DIR / "qwen3_1p7b" / "vdopd.yaml")
    assert cfg.objective.name == "vdopd"
    assert (cfg.model.teacher, cfg.model.student) == ("Qwen/Qwen3-8B", "Qwen/Qwen3-1.7B-Base")
    assert cfg.model.dtype == "bfloat16" and cfg.model.chat_template_kwargs == {
        "enable_thinking": False
    }
    # Appendix B.4: optimisation
    assert cfg.optim.learning_rate == 3e-6 and (cfg.optim.beta1, cfg.optim.beta2) == (0.9, 0.95)
    assert cfg.optim.eps == 1e-8 and cfg.optim.weight_decay == 0.0
    assert cfg.optim.max_grad_norm == 1.0 and cfg.optim.warmup_fraction == 0.03
    # Appendix A.2 / B.4: rollouts
    assert cfg.rollout.batch_size == 128 and cfg.rollout.max_response_tokens == 4096
    assert (cfg.rollout.temperature, cfg.rollout.top_p, cfg.rollout.top_k) == (1.0, 1.0, None)
    assert cfg.train.passes == 3.0 and cfg.train.position_block_size == 32
    assert cfg.train.seed == 11
    # Appendix B.1: data split
    assert (cfg.data.split_seed, cfg.data.num_diagnostic, cfg.data.num_development) == (
        42,
        500,
        500,
    )
    # Appendix B.2.1: decoding
    decoding = cfg.evaluation.decoding
    assert (decoding.num_samples, decoding.temperature, decoding.top_p) == (8, 1.0, 0.8)
    assert decoding.max_response_tokens == 8192
    assert len(cfg.evaluation.benchmarks) == 6 and cfg.evaluation.unseen_wrappers


@pytest.mark.parametrize(
    ("file", "objective", "views"),
    [
        ("eopd_full", "eopd_full", 1),
        ("eopd_top16", "eopd_top16", 1),
        ("eopd_2v", "eopd_2v", 2),
        ("gkd_fkl", "gkd_fkl", 1),
        ("gkd_rkl", "gkd_rkl", 1),
        ("gkd_jsd", "gkd_jsd", 1),
        ("sampled_opd", "sampled_opd", 1),
        ("ablation_arithmetic", "arithmetic", 2),
        ("ablation_weighted_arithmetic", "weighted_arithmetic", 2),
        ("ablation_geometric", "geometric", 2),
        ("ablation_shuffled_weight", "shuffled_weight", 2),
        ("ablation_batch_mean_weight", "batch_mean_weight", 2),
        ("vdopd_budget", "vdopd", 2),
    ],
)
def test_method_configs(file, objective, views):
    cfg = load_config(CONFIG_DIR / "qwen3_1p7b" / f"{file}.yaml")
    assert cfg.objective.name == objective
    assert build_objective(cfg.objective).num_views == views
    assert cfg.optim.learning_rate == 3e-6  # inherited common protocol
    if file.startswith("eopd"):
        assert cfg.objective.eopd_entropy_threshold == 0.8 and cfg.objective.eopd_fkl_weight == 1.0
    if file == "eopd_top16":
        assert build_objective(cfg.objective).top_k == 16
    if file == "gkd_jsd":
        assert cfg.objective.jsd_beta == 0.9
    if file == "vdopd_budget":
        assert cfg.train.max_total_flops == 4.7e18


def test_other_backbones_inherit_the_common_protocol():
    qwen4b = load_config(CONFIG_DIR / "qwen3_4b" / "vdopd.yaml")
    assert qwen4b.model.student == "Qwen/Qwen3-4B-Base" and qwen4b.model.teacher == "Qwen/Qwen3-8B"
    llama = load_config(CONFIG_DIR / "llama3p2_3b" / "vdopd.yaml")
    assert llama.model.teacher == "meta-llama/Llama-3.1-8B-Instruct"
    assert llama.model.student == "meta-llama/Llama-3.2-3B-Instruct"
    assert llama.rollout.batch_size == 128 and llama.optim.learning_rate == 3e-6


def test_overrides_and_validation(tmp_path):
    cfg = load_config(
        CONFIG_DIR / "qwen3_1p7b" / "vdopd.yaml",
        ["train.seed=22", "optim.learning_rate=1e-5", "train.max_teacher_flops=5.5e17"],
    )
    assert cfg.train.seed == 22 and cfg.optim.learning_rate == 1e-5
    assert cfg.train.max_teacher_flops == 5.5e17
    assert config_from_dict({}) == ExperimentConfig()
    with pytest.raises(KeyError):
        config_from_dict({"optim": {"learning_rat": 1e-5}})
    with pytest.raises(ValueError):
        apply_overrides({}, ["no_equals_sign"])
    child = tmp_path / "child.yaml"
    (tmp_path / "parent.yaml").write_text("optim:\n  learning_rate: 1.0e-4\n  beta1: 0.5\n")
    child.write_text("_base_: parent.yaml\noptim:\n  beta1: 0.8\n")
    merged = load_config(child)
    assert merged.optim.learning_rate == 1e-4 and merged.optim.beta1 == 0.8


def test_cli_parser():
    parser = build_parser()
    args = parser.parse_args(["train", "--config", "x.yaml", "--set", "train.seed=22"])
    assert args.command == "train" and args.overrides == ["train.seed=22"]
    args = parser.parse_args(["demo", "--objective", "eopd_full", "--iterations", "3"])
    assert args.objective == "eopd_full" and args.iterations == 3


def test_reduced_scale_config_is_marked_and_loads_exemplars():
    cfg = load_config(CONFIG_DIR / "reduced_scale" / "vdopd.yaml")
    assert cfg.model.teacher == "HuggingFaceTB/SmolLM2-360M-Instruct"
    assert cfg.data.source == "synthetic_arithmetic"
    assert cfg.data.exemplars[0] == ("What is 21+30?", "\\boxed{51}")
    assert len(cfg.data.exemplars) == 3 and cfg.evaluation.decoding.num_samples == 4


def test_data_directory_option():
    from vdopd.cli import _overrides

    args = build_parser().parse_args(["eval", "--config", "x.yaml", "--data-dir", "/tmp/d"])
    assert _overrides(args) == ["data.data_dir=/tmp/d"]
    cfg = load_config(CONFIG_DIR / "qwen3_1p7b" / "vdopd.yaml", _overrides(args))
    assert cfg.data.data_dir == "/tmp/d"
    assert load_config(CONFIG_DIR / "qwen3_1p7b" / "vdopd.yaml").data.data_dir is None
