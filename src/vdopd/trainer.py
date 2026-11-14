"""Algorithm 1: View-Decomposed On-Policy Distillation.

One iteration samples prompts, selects the second teacher view, generates responses from
the current student, scores both teacher views and the student at the same response
positions, accumulates the gradient of the objective blockwise at fixed parameters, and
takes one optimizer step. The loop is shared by VD-OPD, its ablations and the loss-level
baselines; they differ only in the :class:`~vdopd.objectives.Objective`.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from vdopd.consensus import entropy, geometric_consensus, js_divergence
from vdopd.flops import FlopsMeter, budget_reached
from vdopd.models import CausalLM
from vdopd.objectives import Objective, PositionBlock, assign_weights
from vdopd.optim import OptimConfig, build_optimizer, learning_rate_at, set_learning_rate
from vdopd.scoring import pack_prompt_and_response, response_hidden_states
from vdopd.seeding import derive_rng, derive_seed
from vdopd.views import TRAIN_WRAPPERS, Message, PromptCodec, ViewPair, Wrapper, select_views

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RolloutConfig:
    """On-policy rollout settings (Appendix A.2 and B.4)."""

    batch_size: int = 128
    max_response_tokens: int = 4096
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int | None = None
    generation_batch_size: int = 32


@dataclass(frozen=True)
class TrainConfig:
    """Training-loop settings (Appendix A.2, B.4 and B.5.3)."""

    seed: int = 11
    passes: float = 3.0
    iterations: int | None = None
    microbatch_size: int = 8
    position_block_size: int = 32
    max_total_flops: float | None = None
    max_teacher_flops: float | None = None
    gradient_checkpointing: bool = False
    log_diagnostics: bool = True
    log_every: int = 1
    checkpoint_every: int = 0
    output_dir: str | None = None
    teacher_cache_device: str | None = None


class PromptSchedule:
    """Source-question order: seeded passes over the optimisation prompts.

    The order depends only on the run seed, so every method in a controlled comparison sees
    the same source questions in the same order (Appendix B.5.1).
    """

    def __init__(self, num_prompts: int, batch_size: int, seed: int) -> None:
        if num_prompts <= 0:
            raise ValueError("the optimisation set is empty")
        self.num_prompts = num_prompts
        self.batch_size = batch_size
        self.seed = seed

    def num_iterations(self, passes: float) -> int:
        """``J = ceil(passes * n / batch_size)`` rollout batches."""
        return math.ceil(passes * self.num_prompts / self.batch_size)

    def batch(self, iteration: int) -> list[int]:
        """Prompt indices of rollout batch ``k``."""
        start = iteration * self.batch_size
        indices: list[int] = []
        while len(indices) < self.batch_size:
            position = start + len(indices)
            epoch, offset = divmod(position, self.num_prompts)
            order = derive_rng(self.seed, "order", epoch).permutation(self.num_prompts)
            take = min(self.batch_size - len(indices), self.num_prompts - offset)
            indices.extend(int(i) for i in order[offset : offset + take])
        return indices


@dataclass
class RolloutBatch:
    """A fresh student rollout batch with its two teacher views (Algorithm 1, lines 2-4)."""

    iteration: int
    prompt_indices: list[int]
    pairs: list[ViewPair]
    responses: list[list[int]]
    num_excluded: int = 0

    @property
    def num_positions(self) -> int:
        """``N_k``: valid response positions, including generated EOS tokens."""
        return sum(len(r) for r in self.responses)


@dataclass
class _Accumulator:
    """Running sums of the per-position quantities logged for each iteration."""

    loss: float = 0.0
    weight: float = 0.0
    mass: list[Tensor] = field(default_factory=list)
    sums: dict[str, float] = field(default_factory=dict)

    def add(self, key: str, values: Tensor) -> None:
        self.sums[key] = self.sums.get(key, 0.0) + float(values.detach().cpu().double().sum())


class Trainer:
    """On-policy distillation loop (Algorithm 1).

    Args:
        student: trainable student ``pi_theta``; it always sees the original prompt.
        teacher: frozen teacher ``pi_T`` sharing the student's tokenizer and vocabulary.
        codec: the fixed chat serialisation used for both models.
        prompts: the optimisation prompts as chat messages (question text only; no
            reference solutions or answers are read).
        objective: per-position objective, e.g. ``ConsensusObjective()`` for VD-OPD.
        rollout, optim, train: configuration sections.
        wrappers: alternative teacher views to sample from (Table 5 by default).
    """

    def __init__(
        self,
        *,
        student: CausalLM,
        teacher: CausalLM,
        codec: PromptCodec,
        prompts: Sequence[Sequence[Message]],
        objective: Objective,
        rollout: RolloutConfig | None = None,
        optim: OptimConfig | None = None,
        train: TrainConfig | None = None,
        wrappers: Sequence[Wrapper] = TRAIN_WRAPPERS,
    ) -> None:
        if student.vocab_size != teacher.vocab_size:
            raise ValueError("teacher and student must share one vocabulary (Section 4.1)")
        self.student = student
        self.teacher = teacher
        self.codec = codec
        self.prompts = list(prompts)
        self.objective = objective
        self.rollout_cfg = rollout or RolloutConfig()
        self.optim_cfg = optim or OptimConfig()
        self.cfg = train or TrainConfig()
        self.wrappers = tuple(wrappers)

        for parameter in self.teacher.parameters():
            parameter.requires_grad_(False)
        self.parameters = [p for p in self.student.parameters() if p.requires_grad]
        self.optimizer = build_optimizer(self.parameters, self.optim_cfg)
        self.schedule = PromptSchedule(
            len(self.prompts), self.rollout_cfg.batch_size, self.cfg.seed
        )
        self.total_iterations = (
            self.cfg.iterations
            if self.cfg.iterations is not None
            else self.schedule.num_iterations(self.cfg.passes)
        )
        self.flops = FlopsMeter()
        self.iteration = 0
        self.history: list[dict[str, Any]] = []
        self._student_flop_params = student.num_flop_parameters()
        self._teacher_flop_params = teacher.num_flop_parameters()
        self._cache_device = torch.device(self.cfg.teacher_cache_device or str(student.device))

    # ------------------------------------------------------------------ rollouts
    def collect_rollouts(self, iteration: int) -> RolloutBatch:
        """Sample prompts, select views and generate student responses (lines 2-4)."""
        cfg = self.rollout_cfg
        view_rng = derive_rng(self.cfg.seed, "views", iteration)
        indices: list[int] = []
        pairs: list[ViewPair] = []
        excluded = 0
        for index in self.schedule.batch(iteration):
            pair = select_views(
                self.prompts[index],
                self.codec,
                max_response_tokens=cfg.max_response_tokens,
                teacher_context=self.teacher.max_context,
                student_context=self.student.max_context,
                rng=view_rng,
                wrappers=self.wrappers,
            )
            if pair is None:  # invalid original: excluded before generation
                excluded += 1
                continue
            indices.append(index)
            pairs.append(pair)

        responses: list[list[int]] = []
        for chunk, start in enumerate(range(0, len(pairs), cfg.generation_batch_size)):
            prompts = [p.original_ids for p in pairs[start : start + cfg.generation_batch_size]]
            responses.extend(
                self.student.generate(
                    prompts,
                    max_new_tokens=cfg.max_response_tokens,
                    temperature=cfg.temperature,
                    top_p=cfg.top_p,
                    top_k=cfg.top_k,
                    seed=derive_seed(self.cfg.seed, "rollout", iteration, chunk),
                )
            )
            tokens = sum(len(p) for p in prompts) + sum(len(r) for r in responses[start:])
            self.flops.add_generation(self._student_flop_params, tokens)
        return RolloutBatch(iteration, indices, pairs, responses, excluded)

    # ------------------------------------------------------------- teacher views
    @torch.no_grad()
    def _teacher_hidden(self, batch: RolloutBatch, rows: Sequence[int], views: int) -> list[Tensor]:
        """Teacher hidden states at the response positions of ``rows``, one tensor per view.

        Every view receives the same response token ids (Appendix A.1); only the serialised
        prompt in front of them differs.
        """
        responses = [batch.responses[i] for i in rows]
        hidden: list[Tensor] = []
        for view in range(views):
            prompts = [
                batch.pairs[i].original_ids if view == 0 else batch.pairs[i].view2_ids for i in rows
            ]
            packed = pack_prompt_and_response(prompts, responses, self.teacher.pad_token_id)
            hidden.append(response_hidden_states(self.teacher, packed).to(self._cache_device))
            self.flops.add_teacher_scoring(self._teacher_flop_params, packed.num_tokens)
        return hidden

    @torch.no_grad()
    def _teacher_log_probs(self, hidden: Sequence[Tensor], rows: slice) -> tuple[Tensor, ...]:
        """Full-vocabulary float32 log-probabilities of one position block, per view."""
        return tuple(
            F.log_softmax(self.teacher.project(h[rows]).float(), dim=-1).to(self.student.device)
            for h in hidden
        )

    def _microbatches(self, batch: RolloutBatch) -> list[list[int]]:
        size = self.cfg.microbatch_size
        rows = [i for i, response in enumerate(batch.responses) if len(response) > 0]
        return [rows[start : start + size] for start in range(0, len(rows), size)]

    # ------------------------------------------------------- gradient accumulation
    def accumulate_gradients(
        self, batch: RolloutBatch, objective: Objective | None = None
    ) -> dict[str, Any]:
        """Accumulate the gradient of the objective at the current parameters (lines 5-11).

        The loss is ``(1/N_k) sum_{(i,t)} l_{i,t}`` with the unweighted denominator ``N_k``
        (Equation 7). It is evaluated in blocks of response positions: the forward passes
        provide hidden states, the output heads materialise full-vocabulary logits for one
        block at a time, and each block contributes its gradient with the common denominator.
        Student parameters stay fixed and no optimizer step is taken here.

        Returns a statistics dictionary whose ``status`` is ``"ok"``, ``"empty"`` (no valid
        response positions), ``"nonfinite"`` or ``"zero_mass"``; only ``"ok"`` leaves usable
        gradients in ``.grad``.
        """
        objective = objective or self.objective
        self.optimizer.zero_grad(set_to_none=True)
        n_k = batch.num_positions
        stats: dict[str, Any] = {
            "iteration": batch.iteration,
            "num_positions": n_k,
            "num_responses": len(batch.responses),
            "num_excluded": batch.num_excluded,
        }
        if batch.responses:
            eos = set(self.student.eos_token_ids)
            stats["identity_view_fraction"] = float(
                np.mean([pair.wrapper.is_identity for pair in batch.pairs])
            )
            stats["eos_fraction"] = float(
                np.mean([bool(r) and r[-1] in eos for r in batch.responses])
            )
        if n_k == 0:
            stats["status"] = "empty"
            return stats

        block = self.cfg.position_block_size
        microbatches = self._microbatches(batch)
        teacher_hidden = [
            self._teacher_hidden(batch, rows, objective.num_views) for rows in microbatches
        ]

        assigned: Tensor | None = None
        if objective.weight_assignment is not None:
            masses = torch.cat(
                [
                    geometric_consensus(*self._teacher_log_probs(hidden, slice(s, s + block)))[1]
                    for hidden in teacher_hidden
                    for s in range(0, hidden[0].shape[0], block)
                ]
            )
            generator = torch.Generator().manual_seed(
                derive_seed(self.cfg.seed, "weights", batch.iteration)
            )
            assigned = assign_weights(masses.cpu(), objective.weight_assignment, generator)

        acc = _Accumulator()
        offset = 0
        for rows, hidden_views in zip(microbatches, teacher_hidden, strict=True):
            prompts = [batch.pairs[i].original_ids for i in rows]
            packed = pack_prompt_and_response(
                prompts, [batch.responses[i] for i in rows], self.student.pad_token_id
            )
            hidden = response_hidden_states(self.student, packed)
            hidden_grad = torch.zeros_like(hidden)
            tokens = packed.tokens.to(hidden.device)
            for start in range(0, hidden.shape[0], block):
                rows_slice = slice(start, start + block)
                # Detach the block so that only one block of logits is alive at a time; its
                # hidden-state gradient is back-propagated through the backbone afterwards.
                h_block = hidden[rows_slice].detach().requires_grad_(True)
                logits = self.student.project(h_block).float()
                teacher_lp = self._teacher_log_probs(hidden_views, rows_slice)
                weights = None
                if assigned is not None:
                    weights = assigned[offset + start : offset + start + h_block.shape[0]]
                output = objective(PositionBlock(logits, teacher_lp, tokens[rows_slice], weights))
                (output.loss.sum() / n_k).backward()
                hidden_grad[rows_slice] = h_block.grad
                acc.loss += float(output.loss.detach().cpu().double().sum())
                acc.weight += float(output.weight.detach().cpu().double().sum())
                self._record_diagnostics(acc, logits.detach(), teacher_lp)
            hidden.backward(hidden_grad)
            offset += hidden.shape[0]
            self.flops.add_student_training(
                self._student_flop_params, packed.num_tokens, self.cfg.gradient_checkpointing
            )

        stats["loss"] = acc.loss / n_k
        stats["weight_mean"] = acc.weight / n_k
        for key, value in acc.sums.items():
            stats[key] = value / n_k
        if acc.mass:
            mass = torch.cat(acc.mass)
            quantiles = torch.quantile(mass, torch.tensor([0.1, 0.5, 0.9]))
            stats.update(
                mass_total=float(mass.double().sum()),
                mass_mean=float(mass.double().mean()),
                mass_min=float(mass.min()),
                mass_p10=float(quantiles[0]),
                mass_median=float(quantiles[1]),
                mass_p90=float(quantiles[2]),
            )

        grad_norm = self._gradient_norm()
        stats["grad_norm"] = grad_norm
        if not (math.isfinite(stats["loss"]) and math.isfinite(grad_norm)):
            stats["status"] = "nonfinite"
        elif objective.uses_consensus_mass and stats.get("mass_total", acc.weight) <= 0.0:
            stats["status"] = "zero_mass"
        else:
            stats["status"] = "ok"
        return stats

    def _record_diagnostics(
        self, acc: _Accumulator, logits: Tensor, teacher_lp: tuple[Tensor, ...]
    ) -> None:
        """Token-level statistics analysed in the paper (Figure 2, Table 12(a))."""
        two_views = len(teacher_lp) >= 2
        if two_views:
            acc.mass.append(geometric_consensus(teacher_lp[0], teacher_lp[1])[1].cpu())
        if not self.cfg.log_diagnostics:
            return
        acc.add("entropy_p1", entropy(teacher_lp[0]))
        acc.add("entropy_q", entropy(F.log_softmax(logits, dim=-1)))
        if two_views:
            acc.add("entropy_p2", entropy(teacher_lp[1]))
            acc.add("js_divergence", js_divergence(teacher_lp[0], teacher_lp[1]))

    def _gradient_norm(self) -> float:
        total = 0.0
        for parameter in self.parameters:
            if parameter.grad is not None:
                total += float(parameter.grad.detach().float().pow(2).sum())
        return math.sqrt(total) if math.isfinite(total) else float("nan")

    # ------------------------------------------------------------------ updates
    def apply_update(self, stats: dict[str, Any]) -> bool:
        """Clip and step if the batch is finite with positive mass (lines 12-13).

        Otherwise the accumulated gradients are discarded and the parameters and optimizer
        state are left unchanged. Returns whether an optimizer step was taken.
        """
        stepped = stats["status"] == "ok"
        if stepped:
            torch.nn.utils.clip_grad_norm_(self.parameters, self.optim_cfg.max_grad_norm)
            self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        return stepped

    def train_step(self) -> dict[str, Any]:
        """One iteration of Algorithm 1: fresh rollout, gradient accumulation, one update."""
        iteration = self.iteration
        learning_rate = learning_rate_at(iteration, self.total_iterations, self.optim_cfg)
        set_learning_rate(self.optimizer, learning_rate)
        batch = self.collect_rollouts(iteration)
        stats = self.accumulate_gradients(batch)
        stats["stepped"] = self.apply_update(stats)
        stats["learning_rate"] = learning_rate
        stats.update(
            flops_total=self.flops.total,
            flops_teacher_scoring=self.flops.teacher_scoring,
            flops_student_generation=self.flops.student_generation,
            flops_student_training=self.flops.student_training,
        )
        self.iteration += 1
        self.history.append(stats)
        self._log(stats)
        return stats

    def train(
        self, callback: Callable[[Trainer, dict[str, Any]], None] | None = None
    ) -> list[dict[str, Any]]:
        """Run until the iteration budget ``J`` or a FLOPs budget is exhausted."""
        while self.iteration < self.total_iterations and not budget_reached(
            self.flops, self.cfg.max_total_flops, self.cfg.max_teacher_flops
        ):
            stats = self.train_step()
            if callback is not None:
                callback(self, stats)
            if (
                self.cfg.checkpoint_every
                and self.cfg.output_dir
                and self.iteration % self.cfg.checkpoint_every == 0
            ):
                self.save_checkpoint(Path(self.cfg.output_dir) / f"checkpoint-{self.iteration}")
        return self.history

    # ------------------------------------------------------ logging / checkpoints
    def _log(self, stats: dict[str, Any]) -> None:
        if self.cfg.output_dir:
            path = Path(self.cfg.output_dir)
            path.mkdir(parents=True, exist_ok=True)
            with open(path / "metrics.jsonl", "a", encoding="utf-8") as handle:
                handle.write(json.dumps(stats) + "\n")
        if self.cfg.log_every and stats["iteration"] % self.cfg.log_every == 0:
            logger.info(
                "iter %d/%d status=%s loss=%.4f N_k=%d mean_Z=%s lr=%.2e",
                stats["iteration"] + 1,
                self.total_iterations,
                stats["status"],
                stats.get("loss", float("nan")),
                stats["num_positions"],
                f"{stats['mass_mean']:.3f}" if "mass_mean" in stats else "n/a",
                stats["learning_rate"],
            )

    def save_checkpoint(self, directory: str | Path) -> None:
        """Save student weights, optimizer state, iteration counter and FLOPs meter."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.student.model.save_pretrained(directory / "student")
        torch.save(
            {
                "optimizer": self.optimizer.state_dict(),
                "iteration": self.iteration,
                "flops": self.flops.state_dict(),
                "seed": self.cfg.seed,
            },
            directory / "trainer_state.pt",
        )

    def load_checkpoint(self, directory: str | Path) -> None:
        """Restore a checkpoint written by :meth:`save_checkpoint`."""
        from safetensors.torch import load_file

        directory = Path(directory)
        weights: dict[str, Tensor] = {}
        for shard in sorted((directory / "student").glob("*.safetensors")):
            weights.update(load_file(str(shard)))
        if not weights:
            raise FileNotFoundError(f"no student weights found in {directory / 'student'}")
        # Tied output embeddings are stored once, hence strict=False.
        result = self.student.model.load_state_dict(weights, strict=False)
        if result.unexpected_keys:
            raise RuntimeError(f"unexpected checkpoint keys: {result.unexpected_keys[:5]}")
        state = torch.load(directory / "trainer_state.pt", weights_only=True)
        self.optimizer.load_state_dict(state["optimizer"])
        self.iteration = int(state["iteration"])
        self.flops.load_state_dict(state["flops"])
