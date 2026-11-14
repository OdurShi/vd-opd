"""Appendix A.1: response-token alignment across views of different prompt length."""

from __future__ import annotations

import pytest
import torch

from conftest import PrefixSumLM
from vdopd.models import build_tiny_causal_lm
from vdopd.scoring import pack_prompt_and_response, response_hidden_states, score_responses


def test_pack_records_the_position_preceding_each_response_token():
    prompts = [[5, 6, 7], [8, 9, 10, 11, 12]]
    responses = [[20, 21, 1], [30, 1]]
    packed = pack_prompt_and_response(prompts, responses, pad_token_id=0)
    assert packed.input_ids.tolist() == [[5, 6, 7, 20, 21, 1, 0], [8, 9, 10, 11, 12, 30, 1]]
    assert packed.attention_mask.tolist() == [[1, 1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1, 1]]
    # Row 0: y_0 is predicted at the last prompt token (index 2), then indices 3 and 4.
    # Row 1 (offset 7): y_0 at index 4, y_1 at index 5.
    assert packed.gather_index.tolist() == [2, 3, 4, 7 + 4, 7 + 5]
    assert packed.tokens.tolist() == [20, 21, 1, 30, 1]  # generated EOS tokens are included
    assert packed.num_tokens == 6 + 7
    flat = packed.input_ids.reshape(-1)
    # The token *following* each gathered position is exactly the response token.
    assert flat[packed.gather_index + 1].tolist() == packed.tokens.tolist()
    with pytest.raises(ValueError):
        pack_prompt_and_response([[1]], [[2], [3]], 0)
    with pytest.raises(ValueError):
        pack_prompt_and_response([[]], [[2]], 0)


def test_alignment_with_hand_checkable_stub_under_two_views():
    """Each view's distribution for y_t conditions on its own prompt and on y_<t only."""
    model = PrefixSumLM(vocab_size=13)
    original, wrapped = [2, 3], [7, 7, 7, 2, 3]  # the second view has a longer prompt
    response = [4, 5, 6, 1]
    for prompt in (original, wrapped):
        log_probs = score_responses(model, [prompt], [response])
        assert log_probs.shape == (len(response), 13)
        for t in range(len(response)):
            prefix_sum = sum(prompt) + sum(response[:t])
            assert torch.allclose(log_probs[t], model.expected_log_probs(prefix_sum), atol=1e-6)
            assert log_probs[t].argmax().item() == prefix_sum % 13
    # The two views differ at every position because their prompts differ.
    a = score_responses(model, [original], [response])
    b = score_responses(model, [wrapped], [response])
    assert (a.argmax(-1) != b.argmax(-1)).all()


def test_alignment_matches_incremental_decoding_of_a_real_model(make_lm):
    model = make_lm(seed=3)
    generator = torch.Generator().manual_seed(0)
    views = [
        torch.randint(6, 60, (4,), generator=generator).tolist(),
        torch.randint(6, 60, (11,), generator=generator).tolist(),
    ]
    response = torch.randint(6, 60, (5,), generator=generator).tolist() + [1]
    for prompt in views:
        scored = score_responses(model, [prompt], [response], block_size=2)
        for t in range(len(response)):
            prefix = torch.tensor([prompt + response[:t]])
            logits = model.logits(prefix, torch.ones_like(prefix))[0, -1]
            assert torch.allclose(scored[t], torch.log_softmax(logits.float(), -1), atol=1e-5)


def test_scoring_is_invariant_to_padding_and_batch_composition(make_lm):
    model = make_lm(seed=4)
    generator = torch.Generator().manual_seed(1)
    prompts = [torch.randint(6, 60, (n,), generator=generator).tolist() for n in (3, 12, 7)]
    responses = [torch.randint(6, 60, (n,), generator=generator).tolist() for n in (6, 2, 9)]
    together = score_responses(model, prompts, responses, block_size=4)
    separately = torch.cat(
        [score_responses(model, [p], [r]) for p, r in zip(prompts, responses, strict=True)]
    )
    assert together.shape == (17, model.vocab_size)
    assert torch.allclose(together, separately, atol=1e-5)
    reordered = score_responses(model, prompts[::-1], responses[::-1], block_size=64)
    expected = torch.cat([together[8:17], together[6:8], together[0:6]])
    assert torch.allclose(reordered, expected, atol=1e-5)
    assert torch.allclose(together.exp().sum(-1), torch.ones(17), atol=1e-5)


def test_response_hidden_states_shape_and_empty_response(make_lm):
    model = make_lm(seed=5)
    packed = pack_prompt_and_response([[7, 8], [9]], [[10, 11], []], model.pad_token_id)
    hidden = response_hidden_states(model, packed)
    assert hidden.shape == (2, 32)
    assert score_responses(model, [[7, 8]], [[]]).shape == (0, model.vocab_size)


@pytest.mark.parametrize("architecture", ["llama", "gpt2"])
def test_head_decomposition_reproduces_model_logits(architecture, codec):
    """hidden_states + project equals the model's own logits (blockwise evaluation, A.2)."""
    if architecture == "llama":
        lm = build_tiny_causal_lm(len(codec.tokenizer), hidden_size=32, seed=6)
    else:
        from transformers import GPT2Config, GPT2LMHeadModel

        from vdopd.models import CausalLM

        config = GPT2Config(
            vocab_size=50,
            n_embd=32,
            n_layer=2,
            n_head=4,
            n_positions=64,
            bos_token_id=1,
            eos_token_id=1,
        )
        lm = CausalLM(GPT2LMHeadModel(config), eos_token_ids=[1], pad_token_id=0)
    lm.verify_head_decomposition()
    ids = torch.tensor([[3, 4, 5, 6, 7]])
    mask = torch.ones_like(ids)
    reference = lm.model(input_ids=ids, attention_mask=mask).logits
    assert torch.allclose(lm.logits(ids, mask), reference, atol=1e-5)
    assert lm.vocab_size == reference.shape[-1]
