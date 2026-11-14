"""Algorithm 1: rollouts, blockwise gradient accumulation, skip rules and schedules."""

from __future__ import annotations

import copy
import math

import pytest
import torch
import torch.nn.functional as F

from conftest import PrefixSumLM, make_trainer
from vdopd.consensus import geometric_consensus
from vdopd.flops import FlopsMeter, budget_reached, forward_flops, training_flops
from vdopd.objectives import BlockLoss, ConsensusObjective
from vdopd.optim import OptimConfig, learning_rate_at, warmup_steps
from vdopd.scoring import pack_prompt_and_response, response_hidden_states, score_responses
from vdopd.seeding import derive_seed
from vdopd.trainer import PromptSchedule, RolloutBatch, Trainer
from vdopd.views import ORIGINAL_VIEW, TRAIN_WRAPPERS, ViewPair


def gradients(trainer: Trainer) -> list[torch.Tensor]:
    return [p.grad.clone() for p in trainer.parameters if p.grad is not None]


def direct_vdopd_loss(trainer: Trainer, batch: RolloutBatch) -> torch.Tensor:
    """Equation 7 evaluated in one piece with ordinary autograd (reference implementation)."""
    original = [pair.original_ids for pair in batch.pairs]
    wrapped = [pair.view2_ids for pair in batch.pairs]
    log_p1 = score_responses(trainer.teacher, original, batch.responses)
    log_p2 = score_responses(trainer.teacher, wrapped, batch.responses)
    packed = pack_prompt_and_response(original, batch.responses, trainer.student.pad_token_id)
    logits = trainer.student.project(response_hidden_states(trainer.student, packed)).float()
    c, _ = geometric_consensus(log_p1, log_p2)
    return -(c * F.log_softmax(logits, -1)).sum() / batch.num_positions


def test_blockwise_gradient_equals_direct_autograd(make_lm, codec):
    trainer = make_trainer(make_lm, codec)
    batch = trainer.collect_rollouts(0)
    stats = trainer.accumulate_gradients(batch)
    blockwise = gradients(trainer)
    trainer.optimizer.zero_grad(set_to_none=True)
    loss = direct_vdopd_loss(trainer, batch)
    direct = torch.autograd.grad(loss, trainer.parameters)
    assert stats["status"] == "ok"
    assert stats["loss"] == pytest.approx(loss.item(), rel=1e-5)
    assert len(blockwise) == len(direct)
    for a, b in zip(blockwise, direct, strict=True):
        assert torch.allclose(a, b, atol=1e-6, rtol=1e-4)


@pytest.mark.parametrize("objective", ["vdopd", "eopd_2v", "shuffled_weight", "gkd_rkl"])
def test_objective_is_invariant_to_block_and_microbatch_size(make_lm, codec, objective):
    """Changing the block size changes memory use while preserving the objective (B.6)."""
    reference = None
    for microbatch, block in ((1, 1), (2, 3), (6, 1000)):
        trainer = make_trainer(
            make_lm, codec, objective, microbatch_size=microbatch, position_block_size=block
        )
        batch = trainer.collect_rollouts(0)
        stats = trainer.accumulate_gradients(batch)
        current = (stats["loss"], stats["num_positions"], gradients(trainer))
        if reference is None:
            reference = current
            continue
        assert current[1] == reference[1]
        assert current[0] == pytest.approx(reference[0], rel=1e-5, abs=1e-7)
        for a, b in zip(current[2], reference[2], strict=True):
            assert torch.allclose(a, b, atol=1e-6, rtol=1e-4)


def test_denominator_counts_all_response_tokens_including_eos(make_lm, codec):
    trainer = make_trainer(make_lm, codec)
    eos = codec.tokenizer.eos_token_id
    prompts = [codec.encode(m) for m in trainer.prompts[:2]]
    wrapped = [
        codec.encode([dict(m[0], content="Problem:\n\n" + m[0]["content"])])
        for m in trainer.prompts[:2]
    ]
    batch = RolloutBatch(
        iteration=0,
        prompt_indices=[0, 1],
        pairs=[
            ViewPair(tuple(prompts[0]), tuple(wrapped[0]), TRAIN_WRAPPERS[2]),
            ViewPair(tuple(prompts[1]), tuple(prompts[1]), ORIGINAL_VIEW),
        ],
        responses=[[10, 11, 12, eos], [13, eos]],
    )
    assert batch.num_positions == 6  # N_k: prompt tokens and padding are excluded, EOS counts
    stats = trainer.accumulate_gradients(batch)
    assert stats["num_positions"] == 6
    assert stats["loss"] == pytest.approx(direct_vdopd_loss(trainer, batch).item(), rel=1e-5)
    assert stats["identity_view_fraction"] == 0.5 and stats["eos_fraction"] == 1.0
    # The identity-view response has Z = 1 at each of its positions.
    log_p = score_responses(trainer.teacher, [prompts[1]], [[13, eos]])
    assert torch.allclose(geometric_consensus(log_p, log_p)[1], torch.ones(2), atol=1e-5)


def test_train_step_updates_student_only(make_lm, codec):
    trainer = make_trainer(make_lm, codec)
    teacher_before = copy.deepcopy(trainer.teacher.model.state_dict())
    student_before = copy.deepcopy(trainer.student.model.state_dict())
    stats = trainer.train_step()
    assert stats["status"] == "ok" and stats["stepped"]
    assert all(not p.requires_grad and p.grad is None for p in trainer.teacher.parameters())
    for name, value in trainer.teacher.model.state_dict().items():
        assert torch.equal(value, teacher_before[name])
    changed = [
        not torch.equal(value, student_before[name])
        for name, value in trainer.student.model.state_dict().items()
    ]
    assert any(changed)
    assert all(p.grad is None for p in trainer.parameters)  # accumulated gradients discarded
    assert not trainer.student.model.training  # dropout stays disabled
    for key in ("mass_mean", "entropy_p1", "entropy_p2", "entropy_q", "js_divergence"):
        assert math.isfinite(stats[key])
    assert 0.0 < stats["mass_min"] <= stats["mass_median"] <= 1.0 + 1e-6
    assert stats["weight_mean"] == pytest.approx(stats["mass_mean"], rel=1e-5)


def sharp_teacher_batch(codec, same_view: bool) -> RolloutBatch:
    prompt = (7, 8, 9)
    other = prompt if same_view else (7, 8, 10)  # a different prefix sum -> disjoint support
    return RolloutBatch(0, [0], [ViewPair(prompt, other, TRAIN_WRAPPERS[0])], [[12, 13, 1]])


def test_zero_consensus_mass_skips_the_optimizer_step(make_lm, codec):
    teacher = PrefixSumLM(len(codec.tokenizer), sharp=True)
    trainer = make_trainer(make_lm, codec, teacher=teacher)
    before = copy.deepcopy(trainer.student.model.state_dict())
    optimizer_before = copy.deepcopy(trainer.optimizer.state_dict())
    stats = trainer.accumulate_gradients(sharp_teacher_batch(codec, same_view=False))
    assert stats["status"] == "zero_mass" and stats["mass_total"] == 0.0 and stats["loss"] == 0.0
    assert not trainer.apply_update(stats)
    for name, value in trainer.student.model.state_dict().items():
        assert torch.equal(value, before[name])
    assert trainer.optimizer.state_dict()["state"] == optimizer_before["state"] == {}
    # With identical (deterministic) views the mass is one and the step is taken.
    stats = trainer.accumulate_gradients(sharp_teacher_batch(codec, same_view=True))
    assert stats["status"] == "ok" and stats["mass_mean"] == pytest.approx(1.0)
    assert trainer.apply_update(stats)


def test_nonfinite_and_empty_batches_are_skipped(make_lm, codec):
    trainer = make_trainer(make_lm, codec)
    batch = trainer.collect_rollouts(0)

    class Poisoned(ConsensusObjective):
        def __call__(self, block):
            out = super().__call__(block)
            return BlockLoss(out.loss * float("nan"), out.weight)

    before = copy.deepcopy(trainer.student.model.state_dict())
    stats = trainer.accumulate_gradients(batch, Poisoned())
    assert stats["status"] == "nonfinite"
    assert not trainer.apply_update(stats)
    assert all(p.grad is None for p in trainer.parameters)
    for name, value in trainer.student.model.state_dict().items():
        assert torch.equal(value, before[name])
    assert trainer.optimizer.state_dict()["state"] == {}

    empty = RolloutBatch(0, [], [], [])
    stats = trainer.accumulate_gradients(empty)
    assert stats["status"] == "empty" and stats["num_positions"] == 0
    assert not trainer.apply_update(stats)


def test_weight_controls_preserve_multiset_and_total(make_lm, codec):
    for objective in ("shuffled_weight", "batch_mean_weight"):
        trainer = make_trainer(make_lm, codec, objective)
        stats = trainer.accumulate_gradients(trainer.collect_rollouts(0))
        assert stats["status"] == "ok"
        assert stats["weight_mean"] == pytest.approx(stats["mass_mean"], rel=1e-5)


def test_rollouts_are_reproducible_and_on_policy(make_lm, codec):
    first = make_trainer(make_lm, codec).collect_rollouts(2)
    second = make_trainer(make_lm, codec).collect_rollouts(2)
    assert first.responses == second.responses and first.pairs == second.pairs
    other_seed = make_trainer(make_lm, codec, seed=22).collect_rollouts(2)
    assert other_seed.responses != first.responses
    eos = codec.tokenizer.eos_token_id
    for response in first.responses:
        assert 1 <= len(response) <= 7  # response cap L_max includes EOS
        assert eos not in response[:-1]  # generation terminates at EOS
    assert {pair.wrapper.name for pair in first.pairs} <= {w.name for w in TRAIN_WRAPPERS}


def test_sampling_follows_the_full_vocabulary_distribution(make_lm):
    model = make_lm(seed=7)
    prompt = [9, 10, 11]
    ids = torch.tensor([prompt])
    probabilities = torch.softmax(model.logits(ids, torch.ones_like(ids))[0, -1].float(), -1)
    samples = model.generate([prompt] * 4000, max_new_tokens=1, seed=0)
    counts = torch.bincount(torch.tensor(samples).flatten(), minlength=model.vocab_size)
    total_variation = 0.5 * (counts / 4000 - probabilities).abs().sum().item()
    assert total_variation < 0.08
    assert counts.gt(0).sum().item() > 30  # no top-k / nucleus truncation at temperature one
    greedy = model.generate([prompt, [5, 6, 7, 8, 9, 10, 11]], max_new_tokens=1, temperature=0.0)
    assert greedy[0][0] == int(probabilities.argmax())  # left padding does not change outputs
    nucleus = model.generate([prompt] * 500, max_new_tokens=1, top_p=0.1, seed=1)
    assert len({r[0] for r in nucleus}) < counts.gt(0).sum().item()


def test_prompt_schedule_orders_and_budget():
    schedule = PromptSchedule(num_prompts=10, batch_size=4, seed=11)
    assert schedule.num_iterations(3.0) == math.ceil(30 / 4)
    visited = [i for k in range(5) for i in schedule.batch(k)]
    assert sorted(visited[:10]) == list(range(10)) and sorted(visited[10:20]) == list(range(10))
    assert visited[:10] != visited[10:20]  # a fresh order for every pass
    assert visited == [i for k in range(5) for i in PromptSchedule(10, 4, 11).batch(k)]
    assert visited != [i for k in range(5) for i in PromptSchedule(10, 4, 22).batch(k)]
    with pytest.raises(ValueError):
        PromptSchedule(0, 4, 11)
    assert derive_seed(11, "rollout", 3) != derive_seed(11, "views", 3)
    assert derive_seed(11, "rollout", 3) == derive_seed(11, "rollout", 3)


def test_learning_rate_schedule_warmup_and_cosine_decay():
    cfg = OptimConfig()
    assert (cfg.learning_rate, cfg.beta1, cfg.beta2, cfg.eps) == (3e-6, 0.9, 0.95, 1e-8)
    assert (cfg.weight_decay, cfg.max_grad_norm, cfg.warmup_fraction) == (0.0, 1.0, 0.03)
    total = 200
    assert warmup_steps(total, 0.03) == 6
    rates = [learning_rate_at(k, total, cfg) for k in range(total)]
    assert rates[0] == pytest.approx(3e-6 / 6) and rates[5] == pytest.approx(3e-6)
    assert all(a < b for a, b in zip(rates[:6], rates[1:6], strict=False))
    assert all(a >= b for a, b in zip(rates[5:], rates[6:], strict=False))
    middle = 6 + (total - 6) // 2
    assert rates[middle] == pytest.approx(1.5e-6, rel=1e-6)
    assert 0.0 <= rates[-1] < 1e-9


def test_optimizer_matches_the_paper_and_clips_gradients(make_lm, codec):
    trainer = make_trainer(make_lm, codec)
    group = trainer.optimizer.param_groups[0]
    assert isinstance(trainer.optimizer, torch.optim.AdamW)
    assert group["betas"] == (0.9, 0.95) and group["eps"] == 1e-8 and group["weight_decay"] == 0.0
    stats = trainer.train_step()
    assert stats["learning_rate"] == pytest.approx(learning_rate_at(0, 4, trainer.optim_cfg))
    assert trainer.optimizer.param_groups[0]["lr"] == stats["learning_rate"]
    assert trainer.optim_cfg.max_grad_norm == 1.0


def test_flops_accounting_and_budgets(make_lm, codec):
    assert forward_flops(10, 3) == 60.0
    assert training_flops(10, 3) == 180.0 and training_flops(10, 3, True) == 240.0
    meter = FlopsMeter()
    meter.add_generation(10, 1)
    meter.add_teacher_scoring(100, 2)
    meter.add_student_training(10, 1)
    assert meter.total == 20 + 400 + 60
    assert budget_reached(meter, 480.0, None) and not budget_reached(meter, 481.0, None)
    assert budget_reached(meter, None, 400.0) and not budget_reached(meter, None, None)

    two_view = make_trainer(make_lm, codec, "vdopd")
    one_view = make_trainer(make_lm, codec, "eopd_full")
    batch = two_view.collect_rollouts(0)
    two_view.accumulate_gradients(batch)
    one_view.accumulate_gradients(batch)
    response_tokens = batch.num_positions
    view1 = sum(len(pair.original_ids) for pair in batch.pairs) + response_tokens
    view2 = sum(len(pair.view2_ids) for pair in batch.pairs) + response_tokens
    params = two_view.teacher.num_flop_parameters()
    assert one_view.flops.teacher_scoring == pytest.approx(2.0 * params * view1)
    assert two_view.flops.teacher_scoring == pytest.approx(2.0 * params * (view1 + view2))
    assert two_view.flops.student_training == one_view.flops.student_training

    budgeted = make_trainer(make_lm, codec, iterations=50, max_teacher_flops=1.0)
    assert len(budgeted.train()) == 1  # stops as soon as the teacher-scoring budget is spent


def test_checkpoint_round_trip(make_lm, codec, tmp_path):
    trainer = make_trainer(make_lm, codec, output_dir=str(tmp_path), checkpoint_every=2)
    trainer.train()
    assert (tmp_path / "metrics.jsonl").read_text().count("\n") == 4
    assert (tmp_path / "checkpoint-2" / "trainer_state.pt").exists()
    trainer.save_checkpoint(tmp_path / "final")
    restored = make_trainer(make_lm, codec)
    restored.load_checkpoint(tmp_path / "final")
    assert restored.iteration == 4
    assert restored.flops.total == pytest.approx(trainer.flops.total)
    for a, b in zip(restored.parameters, trainer.parameters, strict=True):
        assert torch.equal(a, b)


def test_vocabulary_mismatch_is_rejected(make_lm, codec):
    with pytest.raises(ValueError):
        make_trainer(make_lm, codec, teacher=PrefixSumLM(len(codec.tokenizer) + 1))


def test_checkpoint_sampling_defaults_do_not_leak_into_generation(codec):
    """A model shipped with temperature/top-k defaults is still sampled at temperature one."""
    from transformers import LlamaConfig, LlamaForCausalLM

    from vdopd.models import CausalLM

    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=len(codec.tokenizer),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=64,
        pad_token_id=0,
        bos_token_id=None,
        eos_token_id=1,
    )
    model = LlamaForCausalLM(config)
    model.generation_config.do_sample = True
    model.generation_config.temperature = 0.01
    model.generation_config.top_k = 1
    model.generation_config.repetition_penalty = 1.3
    lm = CausalLM(model, eos_token_ids=[1], pad_token_id=0)
    prompt = [9, 10, 11]
    ids = torch.tensor([prompt])
    probabilities = torch.softmax(lm.logits(ids, torch.ones_like(ids))[0, -1].float(), -1)
    samples = lm.generate([prompt] * 3000, max_new_tokens=1, seed=0)
    counts = torch.bincount(torch.tensor(samples).flatten(), minlength=lm.vocab_size)
    assert counts.gt(0).sum().item() > 30
    assert 0.5 * (counts / 3000 - probabilities).abs().sum().item() < 0.1
