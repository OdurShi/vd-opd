"""Task-preserving prompt views (Section 4.1, Appendix A.1 and B.2.2).

A *wrapper* prepends a fixed string followed by exactly two newline characters to the content
of the final user message. All original characters are retained as an unchanged suffix and
system messages, earlier turns, roles and the assistant generation prefix are identical
across views. Three wrappers are used for training (Table 5) and four further wrappers are
reserved exclusively for evaluation (Appendix B.2.2).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

Message = dict[str, str]

WRAPPER_SEPARATOR = "\n\n"  # exactly two U+000A characters (Appendix A.1)


@dataclass(frozen=True)
class Wrapper:
    """A named prompt view; ``prefix=None`` is the original view, which adds no text."""

    name: str
    prefix: str | None

    @property
    def is_identity(self) -> bool:
        return self.prefix is None


ORIGINAL_VIEW = Wrapper("original", None)

# Table 5: fixed prefixes of the three alternative teacher views used in training.
TRAIN_WRAPPERS: tuple[Wrapper, ...] = (
    Wrapper("instruction_1", "Solve the following problem."),
    Wrapper("instruction_2", "Find a solution to the problem below."),
    Wrapper("layout", "Problem:"),
)

# Appendix B.2.2: four prefixes reserved exclusively for evaluation.
EVAL_WRAPPERS: tuple[Wrapper, ...] = (
    Wrapper("unseen_instruction_1", "Please solve this problem."),
    Wrapper("unseen_instruction_2", "Determine the answer to the following problem."),
    Wrapper("unseen_layout_1", "Problem statement:"),
    Wrapper("unseen_layout_2", "Question:"),
)


class PromptCodec(Protocol):
    """Fixed chat serialisation shared by teacher and student (Appendix A.1, B.4)."""

    def render(self, messages: Sequence[Message]) -> str:
        """Serialise the messages, ending with the fixed assistant generation prefix."""

    def encode(self, messages: Sequence[Message]) -> list[int]:
        """Token ids of :meth:`render`."""

    def decode(self, token_ids: Sequence[int]) -> str:
        """Text of a generated response (special tokens removed)."""


def apply_wrapper(messages: Sequence[Message], wrapper: Wrapper) -> list[Message]:
    """Return the messages of the view defined by ``wrapper``.

    The prefix and two newline characters are prepended to the final user-message content;
    everything else is copied unchanged. The original view returns an unchanged copy.
    """
    if not messages or messages[-1]["role"] != "user":
        raise ValueError("a prompt must end with a user message (Appendix A.1)")
    copied = [dict(m) for m in messages]
    if not wrapper.is_identity:
        copied[-1]["content"] = f"{wrapper.prefix}{WRAPPER_SEPARATOR}{messages[-1]['content']}"
    return copied


def preserves_content_and_roles(
    original: Sequence[Message],
    wrapped: Sequence[Message],
    wrapper: Wrapper,
    codec: PromptCodec | None = None,
) -> bool:
    """Check that a wrapped view preserves the original content and roles (Appendix A.1).

    Message level: same number of messages, identical roles, identical earlier messages, and
    final user content equal to ``prefix + "\\n\\n" + original``. If a codec is given, the
    check is repeated on the *serialised* prompts: the wrapped serialisation must equal the
    original one with the prefix inserted immediately before the final user content, which
    also verifies that the template left the assistant generation prefix untouched.
    """
    if len(original) != len(wrapped):
        return False
    if any(o["role"] != w["role"] for o, w in zip(original, wrapped, strict=True)):
        return False
    if any(o["content"] != w["content"] for o, w in zip(original[:-1], wrapped[:-1], strict=True)):
        return False
    inserted = "" if wrapper.is_identity else f"{wrapper.prefix}{WRAPPER_SEPARATOR}"
    content = original[-1]["content"]
    if wrapped[-1]["content"] != inserted + content:
        return False
    if codec is None:
        return True
    text = codec.render(original)
    start = text.rfind(content)
    if start < 0:
        return False
    return codec.render(wrapped) == text[:start] + inserted + text[start:]


def fits_context(prompt_length: int, response_allowance: int, context_limit: int | None) -> bool:
    """Whether ``prompt + L_max`` tokens fit a context limit (``None`` = unlimited)."""
    return context_limit is None or prompt_length + response_allowance <= context_limit


@dataclass(frozen=True)
class ViewPair:
    """The two teacher views of one prompt, ``x^(1) = x`` and ``x^(2) = T(x)`` (Section 4.1)."""

    original_ids: tuple[int, ...]
    view2_ids: tuple[int, ...]
    wrapper: Wrapper


def select_views(
    messages: Sequence[Message],
    codec: PromptCodec,
    *,
    max_response_tokens: int,
    teacher_context: int | None,
    student_context: int | None,
    rng: np.random.Generator,
    wrappers: Sequence[Wrapper] = TRAIN_WRAPPERS,
) -> ViewPair | None:
    """Validate a prompt and sample its second teacher view (Algorithm 1, lines 2-3).

    The original is valid when its encoded prompt plus ``L_max`` response tokens fits both
    model context limits; invalid originals return ``None`` and are excluded before
    generation. An alternative is valid when its prompt plus the same allowance fits the
    teacher context limit and its serialisation preserves the original content and roles.
    One valid alternative is sampled uniformly; an empty valid set selects the identity
    transformation, i.e. the original prompt is used for both views. Reference answers are
    never read.
    """
    original_ids = codec.encode(messages)
    if not (
        fits_context(len(original_ids), max_response_tokens, teacher_context)
        and fits_context(len(original_ids), max_response_tokens, student_context)
    ):
        return None

    valid: list[tuple[Wrapper, list[int]]] = []
    for wrapper in wrappers:
        wrapped = apply_wrapper(messages, wrapper)
        if not preserves_content_and_roles(messages, wrapped, wrapper, codec):
            continue
        ids = codec.encode(wrapped)
        if fits_context(len(ids), max_response_tokens, teacher_context):
            valid.append((wrapper, ids))

    if not valid:
        return ViewPair(tuple(original_ids), tuple(original_ids), ORIGINAL_VIEW)
    wrapper, ids = valid[int(rng.integers(len(valid)))]
    return ViewPair(tuple(original_ids), tuple(ids), wrapper)
