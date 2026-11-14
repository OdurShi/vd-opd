"""Supplementary E10 (training view pools), E14 (StudentAug) and the structural views of E15."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from conftest import make_trainer
from vdopd.config import load_config
from vdopd.consensus import geometric_consensus
from vdopd.data.problems import build_messages
from vdopd.data.synthetic import SYNTHETIC_TEMPLATE, make_arithmetic_problems
from vdopd.objectives import build_objective
from vdopd.scoring import pack_prompt_and_response, response_hidden_states, score_responses
from vdopd.trainer import ViewConfig
from vdopd.view_pools import (
    ViewRecord,
    group_views,
    load_view_file,
    paraphrase_family,
    parse_view_record,
    rewritten_views,
    training_view_pool,
)
from vdopd.views import (
    EVAL_WRAPPERS,
    STRUCTURAL_VIEWS,
    TRAIN_WRAPPERS,
    RewrittenView,
    TemplateView,
    apply_wrapper,
    preserves_content_and_roles,
    select_views,
)

REPOSITORY = Path(__file__).resolve().parents[1]
CONFIGS = REPOSITORY / "configs" / "qwen3_1p7b"
FIXTURE = Path(__file__).parent / "fixtures" / "view_pool.jsonl"
PROBLEMS = make_arithmetic_problems(max_operand=4)
COVERED = ("synthetic/2+3", "synthetic/4-1", "synthetic/1+1")
MESSAGES = [
    {"role": "system", "content": "Be brief."},
    {"role": "user", "content": "What is 2+3? Answer in \\boxed{}."},
]
SELECT = {"max_response_tokens": 8, "teacher_context": None, "student_context": None}


# ------------------------------------------------------------------ view kinds
def test_template_views_keep_the_content_as_one_unchanged_block(codec):
    view = TemplateView("suffix", "{content}\n\nSolve the problem above.")
    wrapped = apply_wrapper(MESSAGES, view)
    assert wrapped[-1]["content"] == MESSAGES[-1]["content"] + "\n\nSolve the problem above."
    assert wrapped[0] == MESSAGES[0] and not view.is_identity
    assert preserves_content_and_roles(MESSAGES, wrapped, view, codec)
    # Literal braces of the prompt (\boxed{}) survive; the placeholder must occur once.
    assert "\\boxed{}" in wrapped[-1]["content"]
    assert TemplateView("same", "{content}").is_identity
    for template in ("no placeholder", "{content} and {content}"):
        with pytest.raises(ValueError):
            TemplateView("bad", template)
    changed = [wrapped[0], dict(wrapped[1], content=wrapped[1]["content"] + " ")]
    assert not preserves_content_and_roles(MESSAGES, changed, view, codec)


def test_structural_views_relocate_the_instruction_or_change_the_delimiter(codec):
    """Four held-out structural views, none of which is a prefix wrapper (E15)."""
    assert len(STRUCTURAL_VIEWS) == 4 and len({v.name for v in STRUCTURAL_VIEWS}) == 4
    content = MESSAGES[-1]["content"]
    relocated = [v for v in STRUCTURAL_VIEWS if v.name.startswith("relocated")]
    delimiter = [v for v in STRUCTURAL_VIEWS if v.name.startswith("delimiter")]
    assert len(relocated) == len(delimiter) == 2
    for view in relocated:  # the instruction follows the problem instead of preceding it
        assert view.transform(content).startswith(content + "\n\n")
    prefix_forms = {w.transform(content) for w in (*TRAIN_WRAPPERS, *EVAL_WRAPPERS)}
    for view in STRUCTURAL_VIEWS:
        text = view.transform(content)
        assert text.count(content) == 1 and text not in prefix_forms
        assert preserves_content_and_roles(MESSAGES, apply_wrapper(MESSAGES, view), view, codec)
        codec.encode(apply_wrapper(MESSAGES, view))  # representable by the toy tokenizer
    pair = select_views(MESSAGES, codec, rng=np.random.default_rng(0), wrappers=delimiter, **SELECT)
    assert pair.wrapper in delimiter


def test_rewritten_views_replace_the_content_but_keep_roles_and_context(codec):
    view = RewrittenView("paraphrase_1", "Compute 2+3. Answer in \\boxed{}.")
    wrapped = apply_wrapper(MESSAGES, view)
    assert wrapped[-1]["content"] == view.content and wrapped[:-1] == MESSAGES[:-1]
    assert preserves_content_and_roles(MESSAGES, wrapped, view, codec)
    other_system = [dict(wrapped[0], content="Be verbose."), wrapped[1]]
    assert not preserves_content_and_roles(MESSAGES, other_system, view, codec)
    other_role = [dict(wrapped[0], role="user"), wrapped[1]]
    assert not preserves_content_and_roles(MESSAGES, other_role, view, codec)
    pair = select_views(MESSAGES, codec, rng=np.random.default_rng(0), wrappers=[view], **SELECT)
    assert pair.wrapper == view and list(pair.view2_ids) == codec.encode(wrapped)


# ------------------------------------------------------------------ file schema
def test_view_file_fixture_loads():
    records = load_view_file(FIXTURE)
    assert len(records) == 15 and all(isinstance(r, ViewRecord) for r in records)
    assert Counter(r.pool for r in records) == {"validated_paraphrase": 12, "lower_quality": 3}
    assert records[0] == ViewRecord(
        "synthetic/2+3", "paraphrase_1", "Compute 2+3.", "validated_paraphrase", True
    )
    assert {r.problem_id for r in records} == set(COVERED)


@pytest.mark.parametrize(
    "record",
    [
        {"problem_id": "a", "view": "v"},  # missing question
        {"problem_id": "a", "view": "v", "question": "  "},  # empty text
        {"problem_id": "a", "view": "v", "question": "q", "validated": "yes"},
        {"problem_id": "a", "view": "v", "question": "q", "qestion": "typo"},  # unknown key
        {"problem_id": 7, "view": "v", "question": "q"},
        {"problem_id": "a", "view": "v", "question": "q", "pool": ""},
        ["not", "an", "object"],
    ],
)
def test_malformed_records_are_rejected(record, tmp_path):
    with pytest.raises(ValueError):
        parse_view_record(record)
    path = tmp_path / "views.jsonl"
    good = {"problem_id": "a", "view": "v", "question": "q"}
    path.write_text(json.dumps(good) + "\n\n" + json.dumps(record) + "\n")
    with pytest.raises(ValueError, match="views.jsonl:3"):  # the error names file and line
        load_view_file(path)


def test_invalid_json_names_the_line(tmp_path):
    path = tmp_path / "views.jsonl"
    path.write_text('{"problem_id": "a", "view": "v", "question": "q"}\n{not json}\n')
    with pytest.raises(ValueError, match="views.jsonl:2"):
        load_view_file(path)
    record = parse_view_record({"problem_id": "a", "view": "v", "question": "q", "notes": "n"})
    assert record.pool is None and record.validated is False


def test_grouping_validates_a_pool():
    records = load_view_file(FIXTURE)
    grouped = group_views(records, PROBLEMS, pool="validated_paraphrase", views_per_problem=4)
    assert set(grouped) == set(COVERED)
    assert [r.view for r in grouped["synthetic/4-1"]] == [f"paraphrase_{i}" for i in (1, 2, 3, 4)]
    # Views are ordered by name, whatever the line order of the file.
    assert group_views(records[::-1], PROBLEMS, pool="validated_paraphrase") == grouped
    # The lower-quality pool holds a view that nobody validated.
    with pytest.raises(ValueError, match="not marked as validated"):
        group_views(records, PROBLEMS, pool="lower_quality")
    relaxed = group_views(records, PROBLEMS, pool="lower_quality", require_validated=False)
    assert {len(v) for v in relaxed.values()} == {1}
    with pytest.raises(ValueError, match="no records"):
        group_views(records, PROBLEMS, pool="no_such_pool")
    with pytest.raises(ValueError, match="duplicate"):
        group_views([*records, records[0]], PROBLEMS, pool="validated_paraphrase")
    stranger = ViewRecord("synthetic/99+1", "paraphrase_1", "Compute 99+1.", None, True)
    with pytest.raises(ValueError, match="unknown problem"):
        group_views([stranger], PROBLEMS)
    echo = ViewRecord("synthetic/2+3", "copy", "What is 2+3?", None, True)
    with pytest.raises(ValueError, match="repeats the original"):
        group_views([echo], PROBLEMS)
    with pytest.raises(ValueError, match="same 4 view names"):
        group_views(records[:11], PROBLEMS, pool="validated_paraphrase", views_per_problem=4)
    with pytest.raises(ValueError, match="have no view"):
        group_views(records, PROBLEMS, pool="validated_paraphrase", require_all_problems=True)


def test_training_pool_and_paraphrase_family_use_the_common_prompt_template():
    pool = training_view_pool(PROBLEMS, FIXTURE, SYNTHETIC_TEMPLATE, pool="validated_paraphrase")
    assert len(pool) == len(PROBLEMS)
    by_id = dict(zip([p.id for p in PROBLEMS], pool, strict=True))
    assert sum(bool(views) for views in pool) == 3 and by_id["synthetic/0+0"] == ()
    views = by_id["synthetic/2+3"]
    assert [v.name for v in views] == [f"paraphrase_{i}" for i in (1, 2, 3, 4)]
    assert views[0].content == "Compute 2+3. Answer in \\boxed{}."  # instruction retained
    assert views == rewritten_views(
        group_views(load_view_file(FIXTURE), pool="validated_paraphrase")["synthetic/2+3"],
        SYNTHETIC_TEMPLATE,
    )
    covered = [p for p in PROBLEMS if p.id in COVERED]
    family = paraphrase_family(
        covered, load_view_file(FIXTURE), SYNTHETIC_TEMPLATE, pool="validated_paraphrase"
    )
    assert family.name == "paraphrase" and len(family.view_names) == 4
    assert family.views_for("synthetic/1+1")[3].content == "Evaluate 1+1. Answer in \\boxed{}."
    with pytest.raises(KeyError):
        family.views_for("synthetic/0+0")
    with pytest.raises(ValueError):  # every evaluated question needs its four paraphrases
        paraphrase_family(PROBLEMS[:5], load_view_file(FIXTURE), SYNTHETIC_TEMPLATE)


# ------------------------------------------------------------------ E10: training
def file_pool_trainer(make_lm, codec, **kwargs):
    pool = training_view_pool(PROBLEMS, FIXTURE, SYNTHETIC_TEMPLATE, pool="validated_paraphrase")
    return make_trainer(make_lm, codec, prompt_views=pool, **kwargs), pool


def test_trainer_samples_teacher_views_from_a_file_pool(make_lm, codec):
    trainer, pool = file_pool_trainer(make_lm, codec)
    names = Counter()
    for iteration in range(12):  # 72 of the 50 prompts: every covered prompt occurs
        batch = trainer.collect_rollouts(iteration)
        for index, pair in zip(batch.prompt_indices, batch.pairs, strict=True):
            if pool[index]:
                assert pair.wrapper in pool[index]
                expected = apply_wrapper(trainer.prompts[index], pair.wrapper)
                assert list(pair.view2_ids) == codec.encode(expected)
                names[pair.wrapper.name] += 1
            else:  # no view in the file: identity fallback of Appendix A.1
                assert pair.wrapper.is_identity and pair.view2_ids == pair.original_ids
    assert len(names) > 1  # different paraphrases are drawn
    stats = trainer.train_step()
    assert stats["status"] == "ok" and 0.0 < stats["mass_mean"] <= 1.0 + 1e-6
    batch = trainer.collect_rollouts(0)
    uncovered = np.mean([not pool[index] for index in batch.prompt_indices])
    assert stats["identity_view_fraction"] == pytest.approx(uncovered)
    with pytest.raises(ValueError):
        make_trainer(make_lm, codec, prompt_views=pool[:3])


def test_run_summary_reports_the_token_mean_of_z(make_lm, codec, tmp_path):
    trainer, _ = file_pool_trainer(make_lm, codec, output_dir=str(tmp_path))
    history = trainer.train()
    summary = trainer.summary()
    total = sum(h["mass_total"] for h in history)
    positions = sum(h["num_positions"] for h in history)
    assert summary["mean_mass"] == pytest.approx(total / positions)
    assert summary["mass_positions"] == positions and summary["iterations"] == 4
    assert summary["flops_total"] == trainer.flops.total
    written = json.loads((tmp_path / "summary.json").read_text())
    assert written == summary
    # The running totals survive a checkpoint round trip.
    trainer.save_checkpoint(tmp_path / "final")
    restored = make_trainer(make_lm, codec)
    restored.load_checkpoint(tmp_path / "final")
    assert restored.summary()["mean_mass"] == pytest.approx(summary["mean_mass"])
    single_view = make_trainer(make_lm, codec, "eopd_full")
    single_view.train()
    assert single_view.summary()["mean_mass"] is None  # one teacher view has no Z


def test_view_pool_configs():
    for method, objective in (("eopd_2v", "eopd_2v"), ("vdopd", "vdopd")):
        templates = load_config(CONFIGS / f"{method}.yaml")
        assert templates.views == ViewConfig() and templates.views.pool == "templates"
        for suffix, pool in (
            ("paraphrase", "validated_paraphrase"),
            ("lowquality", "lower_quality"),
        ):
            cfg = load_config(CONFIGS / f"{method}_{suffix}.yaml")
            assert cfg.objective.name == objective and cfg.views.pool == "file"
            assert cfg.views.pool_name == pool and cfg.views.require_validated
            assert cfg.views.pool_file and build_objective(cfg.objective).num_views == 2
            assert cfg.optim == templates.optim and cfg.rollout == templates.rollout
    for bad in ({"pool": "random"}, {"pool": "file"}, {"student_wrap_probability": 1.5}):
        with pytest.raises(ValueError):
            ViewConfig(**bad)
    with pytest.raises(ValueError):
        ViewConfig(student_aug_teacher_view="both")


def test_pipeline_resolves_the_training_view_pool():
    from vdopd.config import config_from_dict
    from vdopd.pipeline import training_prompt_views

    data = {"source": "synthetic_arithmetic", "prompt_template": SYNTHETIC_TEMPLATE}
    assert training_prompt_views(config_from_dict({"data": data}), PROBLEMS) is None
    views = {"pool": "file", "pool_file": str(FIXTURE), "pool_name": "validated_paraphrase"}
    resolved = training_prompt_views(config_from_dict({"data": data, "views": views}), PROBLEMS)
    assert len(resolved) == len(PROBLEMS) and sum(bool(v) for v in resolved) == 3
    views["pool_name"] = "lower_quality"
    with pytest.raises(ValueError):  # the pool contains an unvalidated view
        training_prompt_views(config_from_dict({"data": data, "views": views}), PROBLEMS)
    views["require_validated"] = False
    resolved = training_prompt_views(config_from_dict({"data": data, "views": views}), PROBLEMS)
    assert sum(len(v) for v in resolved) == 3


# ------------------------------------------------------------------ E14: StudentAug
def test_student_augmentation_samples_original_and_wrapped_prompts_equally(codec):
    view_rng, plain_rng = np.random.default_rng(3), np.random.default_rng(3)
    student_rng = np.random.default_rng(4)
    wrapped = 0
    for _ in range(600):
        pair = select_views(
            MESSAGES,
            codec,
            rng=view_rng,
            student_wrap_probability=0.5,
            student_rng=student_rng,
            **SELECT,
        )
        plain = select_views(MESSAGES, codec, rng=plain_rng, **SELECT)
        # The teacher views do not depend on the augmentation draw.
        assert (pair.wrapper, pair.view2_ids) == (plain.wrapper, plain.view2_ids)
        if pair.student_ids is not None:
            wrapped += 1
            assert pair.student_ids == pair.view2_ids == pair.student_prompt_ids
        else:
            assert pair.student_prompt_ids == pair.original_ids
    assert 250 < wrapped < 350  # probability one half
    always = select_views(
        MESSAGES,
        codec,
        rng=np.random.default_rng(0),
        student_wrap_probability=1.0,
        student_rng=np.random.default_rng(0),
        **SELECT,
    )
    assert always.student_ids == always.view2_ids
    with pytest.raises(ValueError):
        select_views(
            MESSAGES, codec, rng=np.random.default_rng(0), student_wrap_probability=0.5, **SELECT
        )
    # The wrapped prompt must fit the student context; otherwise the student keeps the
    # original prompt. With no valid alternative there is nothing to wrap.
    length = len(codec.encode(MESSAGES))
    for teacher_context, student_context in ((None, length + 8), (length + 8, None)):
        pair = select_views(
            MESSAGES,
            codec,
            rng=np.random.default_rng(0),
            max_response_tokens=8,
            teacher_context=teacher_context,
            student_context=student_context,
            student_wrap_probability=1.0,
            student_rng=np.random.default_rng(0),
        )
        assert pair.student_ids is None


def direct_eopd_loss(trainer, batch, teacher_prompts) -> float:
    """EOPD (full) on the student's prompts with one teacher view, evaluated in one piece."""
    student_prompts = [pair.student_prompt_ids for pair in batch.pairs]
    log_p = score_responses(trainer.teacher, teacher_prompts, batch.responses)
    packed = pack_prompt_and_response(
        student_prompts, batch.responses, trainer.student.pad_token_id
    )
    logits = trainer.student.project(response_hidden_states(trainer.student, packed)).float()
    log_q = F.log_softmax(logits, -1)
    index = packed.tokens.unsqueeze(-1)
    log_q_y, log_p_y = log_q.gather(-1, index).squeeze(-1), log_p.gather(-1, index).squeeze(-1)
    entropy = -(log_p.exp() * log_p).sum(-1)
    forward_kl = (log_p.exp() * (log_p - log_q)).sum(-1)
    loss = (log_q_y - log_p_y).detach() * log_q_y + (entropy > 0.8).float() * forward_kl
    return float(loss.detach().double().sum()) / batch.num_positions


def test_student_augmentation_in_the_trainer(make_lm, codec):
    aug = ViewConfig(student_wrap_probability=1.0)
    trainer = make_trainer(make_lm, codec, "eopd_full", views=aug)
    generated = []
    generate = trainer.student.generate

    def recording(prompts, **kwargs):
        generated.extend(tuple(p) for p in prompts)
        return generate(prompts, **kwargs)

    trainer.student.generate = recording
    batch = trainer.collect_rollouts(0)
    # Rollouts start from the wrapped prompts ...
    assert all(pair.student_ids == pair.view2_ids for pair in batch.pairs)
    assert generated == [pair.view2_ids for pair in batch.pairs]
    # ... and the loss is evaluated on them; the one teacher view is the student's prompt.
    stats = trainer.accumulate_gradients(batch)
    wrapped = [pair.view2_ids for pair in batch.pairs]
    assert stats["loss"] == pytest.approx(direct_eopd_loss(trainer, batch, wrapped), rel=1e-5)
    assert stats["student_wrapped_fraction"] == 1.0
    params = trainer.teacher.num_flop_parameters()
    tokens = sum(len(p) for p in wrapped) + batch.num_positions
    assert trainer.flops.teacher_scoring == pytest.approx(2.0 * params * tokens)
    # Alternative reading: the single teacher view stays on the original prompt.
    original_view = ViewConfig(student_wrap_probability=1.0, student_aug_teacher_view="original")
    other = make_trainer(make_lm, codec, "eopd_full", views=original_view)
    stats = other.accumulate_gradients(batch)
    originals = [pair.original_ids for pair in batch.pairs]
    assert stats["loss"] == pytest.approx(direct_eopd_loss(other, batch, originals), rel=1e-5)


def test_two_view_student_augmentation_scores_both_teacher_views(make_lm, codec):
    trainer = make_trainer(
        make_lm, codec, "eopd_2v", views=ViewConfig(student_wrap_probability=0.5)
    )
    plain = make_trainer(make_lm, codec, "eopd_2v")
    fractions = []
    for iteration in range(6):
        batch = trainer.collect_rollouts(iteration)
        reference = plain.collect_rollouts(iteration)
        # Same teacher views as EOPD-2V without augmentation (same view stream).
        assert [p.wrapper for p in batch.pairs] == [p.wrapper for p in reference.pairs]
        fractions.extend(pair.student_ids is not None for pair in batch.pairs)
    assert 0.2 < np.mean(fractions) < 0.8  # original and wrapped prompts are mixed
    before = trainer.flops.teacher_scoring
    stats = trainer.accumulate_gradients(batch)
    assert stats["status"] == "ok" and "mass_mean" in stats
    params = trainer.teacher.num_flop_parameters()
    tokens = sum(len(p.original_ids) + len(p.view2_ids) for p in batch.pairs)
    tokens += 2 * batch.num_positions  # original and wrapped view, whatever the student saw
    assert trainer.flops.teacher_scoring - before == pytest.approx(2.0 * params * tokens)
    # The student is scored on its own prompt.
    student_prompts = [pair.student_prompt_ids for pair in batch.pairs]
    log_q = score_responses(trainer.student, student_prompts, batch.responses)
    assert stats["entropy_q"] == pytest.approx(
        float(-(log_q.exp() * log_q).sum(-1).double().mean()), rel=1e-5
    )
    assert trainer.train_step()["stepped"]


def test_without_augmentation_the_student_always_sees_the_original_prompt(make_lm, codec):
    default = make_trainer(make_lm, codec, "eopd_full")
    explicit = make_trainer(make_lm, codec, "eopd_full", views=ViewConfig())
    first, second = default.collect_rollouts(1), explicit.collect_rollouts(1)
    assert first.pairs == second.pairs and first.responses == second.responses
    assert all(pair.student_ids is None for pair in first.pairs)
    assert "student_wrapped_fraction" not in default.accumulate_gradients(first)


def test_student_augmentation_configs():
    one_view = load_config(CONFIGS / "eopd_full_studentaug.yaml")
    two_view = load_config(CONFIGS / "eopd_2v_studentaug.yaml")
    assert one_view.objective.name == "eopd_full" and two_view.objective.name == "eopd_2v"
    assert build_objective(one_view.objective).num_views == 1  # one teacher-scoring view
    assert build_objective(two_view.objective).num_views == 2  # both teacher views
    for cfg in (one_view, two_view):
        assert cfg.views.student_wrap_probability == 0.5 and cfg.views.pool == "templates"
        assert cfg.objective.eopd_entropy_threshold == 0.8 and cfg.objective.eopd_fkl_weight == 1.0
    assert one_view.views.student_aug_teacher_view == "student"
    assert load_config(CONFIGS / "eopd_full.yaml").views.student_wrap_probability == 0.0


def test_matched_views_give_the_same_mass_as_a_direct_computation(make_lm, codec):
    """File-pool views enter the consensus exactly like template views."""
    trainer, _ = file_pool_trainer(make_lm, codec)
    batch = trainer.collect_rollouts(3)
    stats = trainer.accumulate_gradients(batch)
    log_p1 = score_responses(
        trainer.teacher, [p.original_ids for p in batch.pairs], batch.responses
    )
    log_p2 = score_responses(trainer.teacher, [p.view2_ids for p in batch.pairs], batch.responses)
    mass = geometric_consensus(log_p1, log_p2)[1]
    assert stats["mass_mean"] == pytest.approx(float(mass.double().mean()), rel=1e-5)
    messages = build_messages(PROBLEMS[0].question, SYNTHETIC_TEMPLATE)
    assert trainer.prompts[0] == messages
    assert torch.isfinite(torch.tensor(stats["loss"]))
