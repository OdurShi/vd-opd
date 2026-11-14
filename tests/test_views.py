"""Section 4.1, Appendix A.1 and B.2.2: prompt views, validity checks and selection."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from vdopd.chat import HFChatCodec
from vdopd.views import (
    EVAL_WRAPPERS,
    ORIGINAL_VIEW,
    TRAIN_WRAPPERS,
    Wrapper,
    apply_wrapper,
    fits_context,
    preserves_content_and_roles,
    select_views,
)

MESSAGES = [
    {"role": "system", "content": "Be brief."},
    {"role": "user", "content": "What is 2+3? Answer in \\boxed{}."},
]


def test_wrapper_strings_are_those_of_the_paper():
    assert [w.prefix for w in TRAIN_WRAPPERS] == [
        "Solve the following problem.",
        "Find a solution to the problem below.",
        "Problem:",
    ]
    assert [w.prefix for w in EVAL_WRAPPERS] == [
        "Please solve this problem.",
        "Determine the answer to the following problem.",
        "Problem statement:",
        "Question:",
    ]
    train, held_out = {w.prefix for w in TRAIN_WRAPPERS}, {w.prefix for w in EVAL_WRAPPERS}
    assert train.isdisjoint(held_out)  # evaluation wrappers are never used in training
    assert ORIGINAL_VIEW.is_identity


def test_apply_wrapper_prepends_prefix_and_two_newlines():
    wrapped = apply_wrapper(MESSAGES, TRAIN_WRAPPERS[2])
    assert wrapped[-1]["content"] == "Problem:\n\n" + MESSAGES[-1]["content"]
    assert wrapped[-1]["content"].endswith(MESSAGES[-1]["content"])  # unchanged suffix
    assert wrapped[0] == MESSAGES[0] and wrapped[-1]["role"] == "user"
    assert MESSAGES[-1]["content"].startswith("What is")  # the input is not mutated
    assert apply_wrapper(MESSAGES, ORIGINAL_VIEW) == MESSAGES  # the original view adds no text
    with pytest.raises(ValueError):
        apply_wrapper([{"role": "assistant", "content": "hi"}], TRAIN_WRAPPERS[0])


def test_preservation_check(codec):
    wrapper = TRAIN_WRAPPERS[0]
    wrapped = apply_wrapper(MESSAGES, wrapper)
    assert preserves_content_and_roles(MESSAGES, wrapped, wrapper)
    assert preserves_content_and_roles(MESSAGES, wrapped, wrapper, codec)
    changed_role = [dict(wrapped[0], role="user"), wrapped[1]]
    assert not preserves_content_and_roles(MESSAGES, changed_role, wrapper)
    changed_system = [dict(wrapped[0], content="Be verbose."), wrapped[1]]
    assert not preserves_content_and_roles(MESSAGES, changed_system, wrapper)
    one_newline = [wrapped[0], dict(wrapped[1], content=wrapper.prefix + "\n" + "What is 2+3?")]
    assert not preserves_content_and_roles(MESSAGES, one_newline, wrapper)
    assert not preserves_content_and_roles(MESSAGES, wrapped[1:], wrapper)


def test_serialised_views_share_everything_but_the_inserted_prefix(codec):
    original = codec.encode(MESSAGES)
    for wrapper in (*TRAIN_WRAPPERS, *EVAL_WRAPPERS):
        ids = codec.encode(apply_wrapper(MESSAGES, wrapper))
        inserted = len(codec.tokenizer.encode_text(wrapper.prefix + "\n\n"))
        assert len(ids) == len(original) + inserted
        tail = len(codec.tokenizer.encode_text(MESSAGES[-1]["content"])) + 2
        assert ids[-tail:] == original[-tail:]  # question, end-of-turn, assistant prefix
        assert ids[: len(original) - tail] == original[: len(original) - tail]  # system turn


def test_select_views_samples_uniformly_from_valid_alternatives(codec):
    rng = np.random.default_rng(0)
    counts: Counter[str] = Counter()
    for _ in range(600):
        pair = select_views(
            MESSAGES,
            codec,
            max_response_tokens=8,
            teacher_context=None,
            student_context=None,
            rng=rng,
        )
        counts[pair.wrapper.name] += 1
        assert list(pair.original_ids) == codec.encode(MESSAGES)
        assert list(pair.view2_ids) == codec.encode(apply_wrapper(MESSAGES, pair.wrapper))
    assert set(counts) == {w.name for w in TRAIN_WRAPPERS}
    assert min(counts.values()) > 150  # expected 200 each


def test_identity_fallback_and_invalid_originals(codec):
    length = len(codec.encode(MESSAGES))
    rng = np.random.default_rng(0)
    # The original fits, but no wrapped prompt does: the original is used for both views.
    pair = select_views(
        MESSAGES,
        codec,
        max_response_tokens=8,
        teacher_context=length + 8,
        student_context=length + 8,
        rng=rng,
    )
    assert pair.wrapper.is_identity and pair.view2_ids == pair.original_ids
    # Only the shortest wrapper ("Problem:" + two newlines = 10 characters) fits.
    pair = select_views(
        MESSAGES,
        codec,
        max_response_tokens=8,
        teacher_context=length + 8 + 10,
        student_context=length + 8,
        rng=rng,
    )
    assert pair.wrapper.name == "layout"
    # The original itself does not fit one of the context limits: excluded before generation.
    for teacher_context, student_context in ((length + 7, None), (None, length + 7)):
        assert (
            select_views(
                MESSAGES,
                codec,
                max_response_tokens=8,
                teacher_context=teacher_context,
                student_context=student_context,
                rng=rng,
            )
            is None
        )
    assert fits_context(10, 5, 15) and not fits_context(10, 6, 15) and fits_context(10, 6, None)


def test_wrappers_that_break_the_serialisation_are_rejected(codec):
    class TrimmingCodec:
        """A template that rewrites user content, so wrapped views cannot be validated."""

        def render(self, messages):
            return codec.render([dict(m, content=m["content"].upper()) for m in messages])

        def encode(self, messages):
            return codec.encode(messages)

        def decode(self, token_ids):
            return codec.decode(token_ids)

    pair = select_views(
        MESSAGES,
        TrimmingCodec(),
        max_response_tokens=4,
        teacher_context=None,
        student_context=None,
        rng=np.random.default_rng(0),
    )
    assert pair.wrapper.is_identity


def test_hf_chat_codec_uses_the_template_and_generation_prefix():
    class FakeTokenizer:
        chat_template = "fake"

        def apply_chat_template(self, messages, tokenize, add_generation_prompt, **kwargs):
            assert tokenize is False and add_generation_prompt is True
            assert kwargs == {"enable_thinking": False}
            body = "".join(f"<{m['role']}>{m['content']}</{m['role']}>" for m in messages)
            return body + "<assistant><think>\n\n</think>\n\n"

        def __call__(self, text, add_special_tokens):
            assert add_special_tokens is False
            return {"input_ids": [ord(ch) for ch in text]}

        def decode(self, token_ids, skip_special_tokens):
            return "".join(chr(i) for i in token_ids)

    hf_codec = HFChatCodec(FakeTokenizer(), {"enable_thinking": False})
    text = hf_codec.render(MESSAGES)
    assert text.endswith("<assistant><think>\n\n</think>\n\n")
    assert hf_codec.decode(hf_codec.encode(MESSAGES)) == text
    wrapper = Wrapper("layout", "Problem:")
    assert preserves_content_and_roles(
        MESSAGES, apply_wrapper(MESSAGES, wrapper), wrapper, hf_codec
    )

    class NoTemplate:
        chat_template = None

    with pytest.raises(ValueError):
        HFChatCodec(NoTemplate())


def test_wrapper_only_touches_the_final_user_message_of_a_dialogue(codec):
    dialogue = [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "What is 1+1?"},
        {"role": "assistant", "content": "\\boxed{2}"},
        {"role": "user", "content": "What is 2+3?"},
    ]
    wrapper = TRAIN_WRAPPERS[1]
    wrapped = apply_wrapper(dialogue, wrapper)
    assert wrapped[:3] == dialogue[:3]  # system message and earlier turns are identical
    assert wrapped[3]["content"] == "Find a solution to the problem below.\n\nWhat is 2+3?"
    assert preserves_content_and_roles(dialogue, wrapped, wrapper, codec)
    pair = select_views(
        dialogue,
        codec,
        max_response_tokens=4,
        teacher_context=None,
        student_context=None,
        rng=np.random.default_rng(1),
    )
    assert len(pair.view2_ids) > len(pair.original_ids)
