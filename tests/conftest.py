"""Shared fixtures: tiny models built in code and a hand-checkable stub language model."""

from __future__ import annotations

import pytest
import torch

from vdopd.chat import SimpleChatCodec
from vdopd.data.problems import build_messages
from vdopd.data.synthetic import SYNTHETIC_TEMPLATE, make_arithmetic_problems
from vdopd.models import CausalLM, build_tiny_causal_lm
from vdopd.objectives import ObjectiveConfig, build_objective
from vdopd.optim import OptimConfig
from vdopd.trainer import RolloutConfig, TrainConfig, Trainer

torch.set_num_threads(2)


class PrefixSumLM:
    """Stub LM whose next-token distribution is a known function of the prefix.

    After a prefix with token sum ``s`` the model puts all (``sharp=True``) or most of its
    mass on token ``s mod d``. Expected outputs can therefore be computed by hand, which makes
    off-by-one errors in the response-position alignment visible.
    """

    def __init__(self, vocab_size: int, *, sharp: bool = False, scale: float = 3.0) -> None:
        self.vocab_size = vocab_size
        self.sharp = sharp
        self.scale = scale
        self.pad_token_id = 0
        self.eos_token_ids = (1,)
        self.max_context = None
        self.device = torch.device("cpu")
        self.name = "prefix-sum-stub"

    def parameters(self):
        return iter(())

    def num_flop_parameters(self) -> int:
        return 1

    def hidden_states(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        sums = (input_ids * attention_mask).cumsum(dim=1) % self.vocab_size
        return torch.nn.functional.one_hot(sums, self.vocab_size).float()

    def project(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.sharp:
            return torch.where(hidden > 0, 0.0, float("-inf"))
        return hidden * self.scale

    def expected_log_probs(self, prefix_sum: int) -> torch.Tensor:
        hidden = torch.nn.functional.one_hot(
            torch.tensor(prefix_sum % self.vocab_size), self.vocab_size
        ).float()
        return torch.log_softmax(self.project(hidden), dim=-1)


@pytest.fixture(scope="session")
def codec() -> SimpleChatCodec:
    return SimpleChatCodec()


@pytest.fixture
def make_lm(codec):
    """Factory of tiny randomly initialised causal LMs sharing the synthetic vocabulary."""

    def factory(seed: int = 0, hidden_size: int = 32, num_layers: int = 2) -> CausalLM:
        return build_tiny_causal_lm(
            len(codec.tokenizer),
            hidden_size=hidden_size,
            num_layers=num_layers,
            eos_token_id=codec.tokenizer.eos_token_id,
            pad_token_id=codec.tokenizer.pad_token_id,
            seed=seed,
            name=f"tiny-{seed}",
        )

    return factory


def random_log_probs(shape: tuple[int, ...], seed: int = 0, scale: float = 2.0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.log_softmax(scale * torch.randn(*shape, generator=generator), dim=-1)


def make_trainer(make_lm, codec, objective="vdopd", teacher=None, **train_kwargs) -> Trainer:
    problems = make_arithmetic_problems(max_operand=4)
    defaults = {"seed": 11, "iterations": 4, "microbatch_size": 3, "position_block_size": 5}
    defaults.update(train_kwargs)
    return Trainer(
        student=make_lm(seed=1),
        teacher=teacher if teacher is not None else make_lm(seed=2, hidden_size=48),
        codec=codec,
        prompts=[build_messages(p.question, SYNTHETIC_TEMPLATE) for p in problems],
        objective=build_objective(ObjectiveConfig(objective)),
        rollout=RolloutConfig(batch_size=6, max_response_tokens=7, generation_batch_size=4),
        optim=OptimConfig(learning_rate=1e-3),
        train=TrainConfig(**defaults),
    )
