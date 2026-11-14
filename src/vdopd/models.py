"""Causal-LM adapter exposing hidden states and the output head separately.

Appendix A.2 evaluates the full-vocabulary loss in blocks of response positions: "the model
forward passes provide the hidden states preceding vocabulary projection; the output heads
materialize logits for one block at a time". :class:`CausalLM` wraps a Hugging Face causal
language model accordingly (``hidden_states`` + ``project``) and adds batched sampling.
Tiny randomly initialised models for the quick start and the tests are built in code.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor, nn

logger = logging.getLogger(__name__)


class CausalLM:
    """Adapter around a Hugging Face ``*ForCausalLM`` model.

    Args:
        model: the wrapped model. It is kept in evaluation mode; dropout is disabled during
            scoring and gradient evaluation (Appendix A.2). Its ``generation_config`` is
            replaced by a neutral one so that checkpoint-specific sampling defaults cannot
            alter the rollout or evaluation distribution.
        eos_token_ids: token ids that terminate a response.
        pad_token_id: token id used for padding batches.
        max_context: context limit in tokens; defaults to ``config.max_position_embeddings``.
        name: label used in logs.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        eos_token_ids: Sequence[int],
        pad_token_id: int,
        max_context: int | None = None,
        name: str = "model",
    ) -> None:
        self.model = model.eval()
        self.eos_token_ids = tuple(int(i) for i in eos_token_ids)
        self.pad_token_id = int(pad_token_id)
        self.name = name
        config = getattr(model, "config", None)
        self.max_context = max_context or getattr(config, "max_position_embeddings", None)
        self._head = model.get_output_embeddings()
        if self._head is None:
            raise ValueError("the model has no output embedding (language-model head)")
        if hasattr(model, "generation_config"):
            # Pretrained checkpoints ship their own sampling defaults (e.g. a temperature
            # below one or a top-k limit), which `generate` would merge into the request.
            # Decoding is controlled exclusively by the arguments of `CausalLM.generate`.
            from transformers import GenerationConfig

            model.generation_config = GenerationConfig(
                eos_token_id=list(self.eos_token_ids), pad_token_id=self.pad_token_id
            )

    @property
    def device(self) -> torch.device:
        return self._head.weight.device

    @property
    def vocab_size(self) -> int:
        return int(self._head.weight.shape[0])

    def parameters(self):
        return self.model.parameters()

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    def num_flop_parameters(self) -> int:
        """Parameters that take part in matrix products for every token.

        The input embedding is a table lookup, so it is excluded unless it is tied to the
        output head (which is a matrix product). Used by :mod:`vdopd.flops`.
        """
        total = self.num_parameters()
        embedding = self.model.get_input_embeddings()
        if embedding is not None and embedding.weight is not self._head.weight:
            total -= embedding.weight.numel()
        return total

    def hidden_states(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        """Final hidden states preceding the vocabulary projection, ``[B, T, H]``."""
        outputs = self.model.base_model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state

    def project(self, hidden: Tensor) -> Tensor:
        """Apply the output head to hidden states, ``[..., H] -> [..., d]``."""
        return self._head(hidden.to(self._head.weight.device))

    def logits(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        """Full logits ``[B, T, d]``; convenience for small inputs."""
        return self.project(self.hidden_states(input_ids, attention_mask))

    @torch.no_grad()
    def verify_head_decomposition(self, atol: float = 1e-4) -> None:
        """Check that ``project(hidden_states(x))`` reproduces the model's own logits.

        Guards against architectures that apply further operations (scaling, bias) between
        the backbone output and the logits, for which blockwise evaluation would be wrong.
        """
        probe = torch.arange(1, 5, device=self.device).unsqueeze(0) % self.vocab_size
        mask = torch.ones_like(probe)
        reference = self.model(input_ids=probe, attention_mask=mask).logits.float()
        decomposed = self.logits(probe, mask).float()
        error = (reference - decomposed).abs().max().item()
        if error > atol * max(1.0, reference.abs().max().item()):
            raise RuntimeError(
                f"{self.name}: output head decomposition differs from model logits by {error:.3g}"
            )

    @torch.no_grad()
    def generate(
        self,
        prompts: Sequence[Sequence[int]],
        *,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int | None = None,
        seed: int | None = None,
    ) -> list[list[int]]:
        """Sample one response per prompt; ``temperature=0`` selects greedy decoding.

        Responses terminate at the first EOS (which is kept) or after ``max_new_tokens``
        tokens. With ``temperature=1``, ``top_p=1`` and no top-k restriction the tokens are
        drawn from the model's full-vocabulary distribution (Appendix A.2).
        """
        from transformers import GenerationConfig

        if not prompts:
            return []
        if seed is not None:
            torch.manual_seed(int(seed))
        width = max(len(p) for p in prompts)
        input_ids = torch.full((len(prompts), width), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((len(prompts), width), dtype=torch.long)
        for row, prompt in enumerate(prompts):  # left padding for decoder-only generation
            input_ids[row, width - len(prompt) :] = torch.as_tensor(prompt, dtype=torch.long)
            attention_mask[row, width - len(prompt) :] = 1

        sample = temperature > 0
        generation_config = GenerationConfig(
            do_sample=sample,
            temperature=float(temperature) if sample else None,
            top_p=float(top_p) if sample else None,
            top_k=(int(top_k) if top_k else 0) if sample else None,
            max_new_tokens=int(max_new_tokens),
            eos_token_id=list(self.eos_token_ids),
            pad_token_id=self.pad_token_id,
            repetition_penalty=1.0,
            num_beams=1,
        )
        output = self.model.generate(
            input_ids=input_ids.to(self.device),
            attention_mask=attention_mask.to(self.device),
            generation_config=generation_config,
        )
        responses: list[list[int]] = []
        for row in output[:, width:].tolist():
            response: list[int] = []
            for token in row:
                response.append(token)
                if token in self.eos_token_ids:
                    break
            responses.append(response)
        return responses


def build_tiny_causal_lm(
    vocab_size: int,
    *,
    hidden_size: int = 64,
    num_layers: int = 2,
    num_heads: int = 4,
    max_positions: int = 256,
    eos_token_id: int = 1,
    pad_token_id: int = 0,
    seed: int = 0,
    name: str = "tiny",
) -> CausalLM:
    """Build a small randomly initialised decoder-only transformer in code (no download).

    The architecture is a miniature Llama-style model (rotary positions, RMSNorm), i.e. the
    same family as the backbones used in the paper, so the quick start and the tests exercise
    exactly the code path used for real models.
    """
    from transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=2 * hidden_size,
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        num_key_value_heads=num_heads,
        max_position_embeddings=max_positions,
        pad_token_id=pad_token_id,
        bos_token_id=None,
        eos_token_id=eos_token_id,
        attention_dropout=0.0,
    )
    generator_state = torch.random.get_rng_state()
    torch.manual_seed(seed)
    model = LlamaForCausalLM(config)
    torch.random.set_rng_state(generator_state)
    return CausalLM(
        model,
        eos_token_ids=[eos_token_id],
        pad_token_id=pad_token_id,
        max_context=max_positions,
        name=name,
    )


def resolve_dtype(name: str) -> torch.dtype:
    """Map a config string (``"bfloat16"``, ``"float16"``, ``"float32"``) to a torch dtype."""
    try:
        return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[
            name
        ]
    except KeyError as error:
        raise ValueError(f"unsupported dtype {name!r}") from error


def resolve_device(name: str) -> torch.device:
    """Resolve ``"auto"`` to CUDA, then MPS, then CPU; otherwise return the named device."""
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_tokenizer(name_or_path: str, revision: str | None = None) -> Any:
    """Load a Hugging Face tokenizer."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(name_or_path, revision=revision)


def stop_token_ids(tokenizer: Any, stop_tokens: Sequence[str] = ()) -> list[int]:
    """Token ids that end a response: the tokenizer EOS plus any named stop tokens."""
    ids: list[int] = []
    if tokenizer.eos_token_id is not None:
        ids.append(int(tokenizer.eos_token_id))
    for token in stop_tokens:
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is None or token_id == getattr(tokenizer, "unk_token_id", None):
            raise ValueError(f"stop token {token!r} is not in the tokenizer vocabulary")
        if int(token_id) not in ids:
            ids.append(int(token_id))
    if not ids:
        raise ValueError("no end-of-sequence token is defined")
    return ids


def load_causal_lm(
    name_or_path: str,
    *,
    tokenizer: Any,
    stop_tokens: Sequence[str] = (),
    dtype: str = "bfloat16",
    device: str = "auto",
    revision: str | None = None,
    max_context: int | None = None,
    gradient_checkpointing: bool = False,
    trainable: bool = False,
) -> CausalLM:
    """Load a pretrained causal LM from the Hugging Face Hub or a local directory."""
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        name_or_path, revision=revision, dtype=resolve_dtype(dtype)
    )
    model.to(resolve_device(device))
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
    for parameter in model.parameters():
        parameter.requires_grad_(trainable)
    pad_token_id = tokenizer.pad_token_id
    eos_ids = stop_token_ids(tokenizer, stop_tokens)
    lm = CausalLM(
        model,
        eos_token_ids=eos_ids,
        pad_token_id=eos_ids[0] if pad_token_id is None else pad_token_id,
        max_context=max_context,
        name=name_or_path,
    )
    lm.verify_head_decomposition()
    return lm


def check_pair_compatibility(
    teacher: CausalLM, student: CausalLM, teacher_tokenizer: Any, student_tokenizer: Any
) -> None:
    """Verify vocabulary indices, special-token ids and output-head sizes (Appendix B.4)."""
    if teacher.vocab_size != student.vocab_size:
        raise ValueError(
            f"output heads differ: teacher {teacher.vocab_size} vs student {student.vocab_size}"
        )
    if teacher_tokenizer.get_vocab() != student_tokenizer.get_vocab():
        raise ValueError("teacher and student tokenizers map tokens to different indices")
    if teacher_tokenizer.all_special_ids != student_tokenizer.all_special_ids:
        raise ValueError("teacher and student tokenizers disagree on special-token ids")
