"""Teacher-forced scoring of student responses with response-token alignment.

Appendix A.1: after generation the *same* response token ids are attached to each serialised
prompt. The distribution for ``y_{i,t}`` is read from the output immediately preceding that
token, so it conditions on ``y_{i,<t}``; causal masking lets the complete response be scored
in one forward pass per view, and the alignment holds although the views' prompts differ in
length.

With 0-based indices, a prompt of ``P`` tokens followed by a response ``y_0 .. y_{L-1}``
occupies input positions ``0 .. P+L-1``; the distribution of ``y_t`` is the model output at
input position ``P - 1 + t`` (the last prompt token for ``t = 0``).
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from vdopd.models import CausalLM


@dataclass
class PackedBatch:
    """Right-padded ``prompt + response`` sequences and the positions that predict responses.

    Attributes:
        input_ids: ``[B, T]`` token ids, padded on the right.
        attention_mask: ``[B, T]`` with ones on real tokens.
        gather_index: ``[n]`` indices into the flattened ``[B * T]`` positions; entry ``j``
            is the position whose output is the distribution of the ``j``-th response token,
            enumerated sequence by sequence and, within a sequence, in response order.
        tokens: ``[n]`` the response token ids in the same order.
        num_tokens: number of non-padding input tokens (for FLOPs accounting).
    """

    input_ids: Tensor
    attention_mask: Tensor
    gather_index: Tensor
    tokens: Tensor
    num_tokens: int


def pack_prompt_and_response(
    prompts: Sequence[Sequence[int]], responses: Sequence[Sequence[int]], pad_token_id: int
) -> PackedBatch:
    """Attach each response to its prompt and record the aligned prediction positions.

    Prompt tokens and padding never enter ``gather_index``; every generated token, including
    a generated EOS, does (Appendix A.2).
    """
    if len(prompts) != len(responses):
        raise ValueError("prompts and responses must have the same length")
    if any(len(p) == 0 for p in prompts):
        raise ValueError("every prompt needs at least one token")
    lengths = [len(p) + len(r) for p, r in zip(prompts, responses, strict=True)]
    width = max(lengths) if lengths else 0
    input_ids = torch.full((len(prompts), width), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(prompts), width), dtype=torch.long)
    gather: list[Tensor] = []
    tokens: list[Tensor] = []
    for row, (prompt, response) in enumerate(zip(prompts, responses, strict=True)):
        response_ids = torch.as_tensor(list(response), dtype=torch.long)
        input_ids[row, : len(prompt)] = torch.as_tensor(list(prompt), dtype=torch.long)
        input_ids[row, len(prompt) : lengths[row]] = response_ids
        attention_mask[row, : lengths[row]] = 1
        first = row * width + len(prompt) - 1
        gather.append(torch.arange(first, first + len(response), dtype=torch.long))
        tokens.append(response_ids)
    empty = torch.empty(0, dtype=torch.long)
    return PackedBatch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        gather_index=torch.cat(gather) if gather else empty,
        tokens=torch.cat(tokens) if tokens else empty,
        num_tokens=int(sum(lengths)),
    )


def response_hidden_states(model: CausalLM, packed: PackedBatch) -> Tensor:
    """Hidden states at the positions that predict each response token, ``[n, H]``.

    Gradients flow if the caller has them enabled; one forward pass scores the complete
    responses of the batch.
    """
    hidden = model.hidden_states(
        packed.input_ids.to(model.device), packed.attention_mask.to(model.device)
    )
    flat = hidden.reshape(-1, hidden.shape[-1])
    return flat.index_select(0, packed.gather_index.to(flat.device))


@torch.no_grad()
def iter_log_prob_blocks(
    model: CausalLM,
    prompts: Sequence[Sequence[int]],
    responses: Sequence[Sequence[int]],
    *,
    block_size: int = 32,
) -> Iterator[Tensor]:
    """Yield full-vocabulary log-probabilities ``log pi(. | x_i, y_{i,<t})`` block by block.

    One forward pass provides the hidden states of all response positions; the output head
    is then applied to ``block_size`` positions at a time, so at most ``block_size * d``
    probabilities are alive at once (Appendix A.2). Blocks are float32 at temperature one.
    """
    packed = pack_prompt_and_response(prompts, responses, model.pad_token_id)
    hidden = response_hidden_states(model, packed)
    for start in range(0, hidden.shape[0], block_size):
        yield F.log_softmax(model.project(hidden[start : start + block_size]).float(), dim=-1)


def score_responses(
    model: CausalLM,
    prompts: Sequence[Sequence[int]],
    responses: Sequence[Sequence[int]],
    *,
    block_size: int = 32,
) -> Tensor:
    """All blocks of :func:`iter_log_prob_blocks` concatenated into one ``[n, d]`` tensor.

    Convenient for small inputs and tests; large-vocabulary code should consume the blocks.
    """
    blocks = list(iter_log_prob_blocks(model, prompts, responses, block_size=block_size))
    if not blocks:
        return torch.empty(0, model.vocab_size)
    return torch.cat(blocks, dim=0)
