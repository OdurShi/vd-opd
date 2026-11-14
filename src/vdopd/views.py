"""Task-preserving prompt views (Section 4.1, Appendix A.1 and B.2.2).

A *wrapper* prepends a fixed string followed by exactly two newline characters to the content
of the final user message. All original characters are retained as an unchanged suffix and
system messages, earlier turns, roles and the assistant generation prefix are identical
across views. Three wrappers are used for training (Table 5) and four further wrappers are
reserved exclusively for evaluation (Appendix B.2.2).

The supplementary experiments add two further kinds of view, which share the validity checks
and the selection code of the wrappers:

* :class:`TemplateView` places the unchanged content inside a fixed template; it expresses
  the held-out *structural changes* of E15 (a relocated instruction, a changed delimiter).
* :class:`RewrittenView` replaces the content by text produced outside this repository, e.g.
  a validated paraphrase of the question (the training pools of E10, the held-out paraphrase
  family of E15). Such views are loaded with :mod:`vdopd.view_pools`.

:func:`select_views` also covers the joint ``K``-view target of E08 and the student-side
prompt augmentation of E14.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

Message = dict[str, str]

WRAPPER_SEPARATOR = "\n\n"  # exactly two U+000A characters (Appendix A.1)
CONTENT_FIELD = "{content}"  # placeholder of the original final user content in a template


@dataclass(frozen=True)
class Wrapper:
    """A named prompt view; ``prefix=None`` is the original view, which adds no text."""

    name: str
    prefix: str | None

    @property
    def is_identity(self) -> bool:
        return self.prefix is None

    def transform(self, content: str) -> str:
        """Final user-message content of this view: ``prefix + two newlines + content``."""
        return content if self.is_identity else f"{self.prefix}{WRAPPER_SEPARATOR}{content}"


@dataclass(frozen=True)
class TemplateView:
    """A view that embeds the unchanged content in a template (supplementary E15).

    ``template`` contains the placeholder ``{content}`` exactly once, so every original
    character is retained as one contiguous, unchanged block; only its surroundings (a
    relocated instruction, a different delimiter) change.
    """

    name: str
    template: str

    def __post_init__(self) -> None:
        if self.template.count(CONTENT_FIELD) != 1:
            raise ValueError(
                f"view {self.name!r}: the template must contain {CONTENT_FIELD} exactly once"
            )

    @property
    def is_identity(self) -> bool:
        return self.template == CONTENT_FIELD

    def transform(self, content: str) -> str:
        # str.replace rather than str.format: prompts contain literal braces (\boxed{}).
        return self.template.replace(CONTENT_FIELD, content)


@dataclass(frozen=True)
class RewrittenView:
    """A view of one prompt whose final user content was written outside this repository.

    Used for paraphrases (supplementary E10 and E15). Whether the rewritten text preserves
    the task is established by the external validation recorded in the view file; the checks
    here still require identical roles, earlier turns and assistant generation prefix.
    """

    name: str
    content: str

    @property
    def is_identity(self) -> bool:
        return False

    def transform(self, content: str) -> str:
        return self.content


View = Wrapper | TemplateView | RewrittenView

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

# Supplementary E15: four held-out structural views, "instruction relocation and
# layout/delimiter changes". The experiment description does not give the strings; these are
# this repository's choice (docs/implementation_notes.md) and can be replaced through
# `evaluation.structural_views`. Two views move a solving instruction behind the problem,
# two change the delimiter between a layout heading and the problem.
STRUCTURAL_VIEWS: tuple[TemplateView, ...] = (
    TemplateView("relocated_instruction_1", "{content}\n\nPlease solve this problem."),
    TemplateView(
        "relocated_instruction_2", "{content}\n\nDetermine the answer to the problem above."
    ),
    TemplateView("delimiter_heading", "Question: {content}"),
    TemplateView("delimiter_block", "---\n{content}\n---"),
)

PREFIX_FAMILY = "prefix"
STRUCTURAL_FAMILY = "structural"
PARAPHRASE_FAMILY = "paraphrase"
VIEW_FAMILIES = (PREFIX_FAMILY, STRUCTURAL_FAMILY, PARAPHRASE_FAMILY)


@dataclass(frozen=True)
class ViewFamily:
    """A named family of held-out evaluation views, fixed before evaluation (E14, E15).

    ``views`` apply to every question (prefix wrappers, structural templates).
    ``per_question`` maps a problem id to that problem's own views (verified paraphrases);
    every problem then has the same number of views with the same names, so that view ``m``
    of the family is well defined as a "template" across questions.
    """

    name: str
    views: tuple[View, ...] = ()
    per_question: Mapping[str, tuple[View, ...]] | None = None

    def views_for(self, problem_id: str) -> tuple[View, ...]:
        if self.per_question is None:
            return self.views
        if problem_id not in self.per_question:
            raise KeyError(f"view family {self.name!r} has no views for problem {problem_id!r}")
        return self.per_question[problem_id]

    @property
    def view_names(self) -> tuple[str, ...]:
        if self.per_question is None:
            return tuple(view.name for view in self.views)
        first = next(iter(self.per_question.values()), ())
        return tuple(view.name for view in first)


class PromptCodec(Protocol):
    """Fixed chat serialisation shared by teacher and student (Appendix A.1, B.4)."""

    def render(self, messages: Sequence[Message]) -> str:
        """Serialise the messages, ending with the fixed assistant generation prefix."""

    def encode(self, messages: Sequence[Message]) -> list[int]:
        """Token ids of :meth:`render`."""

    def decode(self, token_ids: Sequence[int]) -> str:
        """Text of a generated response (special tokens removed)."""


def apply_wrapper(messages: Sequence[Message], wrapper: View) -> list[Message]:
    """Return the messages of the view defined by ``wrapper``.

    For a :class:`Wrapper` the prefix and two newline characters are prepended to the final
    user-message content; a :class:`TemplateView` or :class:`RewrittenView` sets that content
    through its ``transform``. Everything else is copied unchanged. The original view
    returns an unchanged copy.
    """
    if not messages or messages[-1]["role"] != "user":
        raise ValueError("a prompt must end with a user message (Appendix A.1)")
    copied = [dict(m) for m in messages]
    if not wrapper.is_identity:
        copied[-1]["content"] = wrapper.transform(messages[-1]["content"])
    return copied


apply_view = apply_wrapper


def preserves_content_and_roles(
    original: Sequence[Message],
    wrapped: Sequence[Message],
    wrapper: View,
    codec: PromptCodec | None = None,
) -> bool:
    """Check that a view preserves the original content and roles (Appendix A.1).

    Message level: same number of messages, identical roles, identical earlier messages, and
    final user content equal to the view's transform of the original content (for a wrapper:
    ``prefix + "\\n\\n" + original``). If a codec is given, the check is repeated on the
    *serialised* prompts: the serialisation of the view must equal the original one with
    only the final user content exchanged, which also verifies that the template left the
    assistant generation prefix untouched.
    """
    if len(original) != len(wrapped):
        return False
    if any(o["role"] != w["role"] for o, w in zip(original, wrapped, strict=True)):
        return False
    if any(o["content"] != w["content"] for o, w in zip(original[:-1], wrapped[:-1], strict=True)):
        return False
    content = original[-1]["content"]
    expected = wrapper.transform(content)
    if wrapped[-1]["content"] != expected:
        return False
    if codec is None:
        return True
    text = codec.render(original)
    start = text.rfind(content)
    if start < 0:
        return False
    return codec.render(wrapped) == text[:start] + expected + text[start + len(content) :]


def fits_context(prompt_length: int, response_allowance: int, context_limit: int | None) -> bool:
    """Whether ``prompt + L_max`` tokens fit a context limit (``None`` = unlimited)."""
    return context_limit is None or prompt_length + response_allowance <= context_limit


@dataclass(frozen=True)
class ViewPair:
    """The teacher views of one prompt, ``x^(1) = x`` and ``x^(2) = T(x)`` (Section 4.1).

    ``extra_view_ids`` / ``extra_wrappers`` hold the views ``x^(3), ..., x^(K)`` of the joint
    ``K``-view target (supplementary E08) and are empty in the paper's two-view setting.
    ``student_ids`` is the prompt given to the student for rollout and loss evaluation when
    student-side augmentation selected the wrapped prompt (supplementary E14); ``None``
    means the original prompt, as in the paper.
    """

    original_ids: tuple[int, ...]
    view2_ids: tuple[int, ...]
    wrapper: View
    extra_view_ids: tuple[tuple[int, ...], ...] = ()
    extra_wrappers: tuple[View, ...] = ()
    student_ids: tuple[int, ...] | None = None

    @property
    def teacher_view_ids(self) -> tuple[tuple[int, ...], ...]:
        """Encoded prompts of all teacher views, the original first."""
        return (self.original_ids, self.view2_ids, *self.extra_view_ids)

    @property
    def wrappers(self) -> tuple[View, ...]:
        """The alternative views ``x^(2), ..., x^(K)`` in the order of their prompts."""
        return (self.wrapper, *self.extra_wrappers)

    @property
    def student_prompt_ids(self) -> tuple[int, ...]:
        """Encoded prompt the student is rolled out and scored on."""
        return self.original_ids if self.student_ids is None else self.student_ids


def select_views(
    messages: Sequence[Message],
    codec: PromptCodec,
    *,
    max_response_tokens: int,
    teacher_context: int | None,
    student_context: int | None,
    rng: np.random.Generator,
    wrappers: Sequence[View] = TRAIN_WRAPPERS,
    num_views: int = 2,
    student_wrap_probability: float = 0.0,
    student_rng: np.random.Generator | None = None,
) -> ViewPair | None:
    """Validate a prompt and select its alternative teacher views (Algorithm 1, lines 2-3).

    The original is valid when its encoded prompt plus ``L_max`` response tokens fits both
    model context limits; invalid originals return ``None`` and are excluded before
    generation. An alternative is valid when its prompt plus the same allowance fits the
    teacher context limit and its serialisation preserves the original content and roles.
    One valid alternative is sampled uniformly; an empty valid set selects the identity
    transformation, i.e. the original prompt is used for both views. Reference answers are
    never read.

    ``num_views = K > 2`` selects ``K - 1`` alternatives for the joint target of
    supplementary E08: all valid ones if there are at most ``K - 1`` (``K = 4`` with the
    three training wrappers uses the original prompt and all three wrappers), otherwise a
    uniform sample without replacement, kept in pool order. Missing views fall back to the
    identity transformation. For ``K <= 2`` the single draw from ``rng`` is unchanged.

    ``student_wrap_probability = p > 0`` is the student-side augmentation of supplementary
    E14: with probability ``p`` (drawn from ``student_rng``) the student is given the first
    alternative view instead of the original prompt, provided that view also fits the
    student context limit.
    """
    original_ids = codec.encode(messages)
    if not (
        fits_context(len(original_ids), max_response_tokens, teacher_context)
        and fits_context(len(original_ids), max_response_tokens, student_context)
    ):
        return None

    valid: list[tuple[View, list[int]]] = []
    for wrapper in wrappers:
        wrapped = apply_wrapper(messages, wrapper)
        if not preserves_content_and_roles(messages, wrapped, wrapper, codec):
            continue
        ids = codec.encode(wrapped)
        if fits_context(len(ids), max_response_tokens, teacher_context):
            valid.append((wrapper, ids))

    alternatives = max(num_views, 2) - 1
    if not valid:
        chosen: list[tuple[View, list[int]]] = []
    elif alternatives == 1:
        chosen = [valid[int(rng.integers(len(valid)))]]
    elif len(valid) <= alternatives:
        chosen = valid
    else:
        picks = sorted(int(i) for i in rng.choice(len(valid), size=alternatives, replace=False))
        chosen = [valid[i] for i in picks]
    chosen = chosen + [(ORIGINAL_VIEW, original_ids)] * (alternatives - len(chosen))

    student_ids: tuple[int, ...] | None = None
    if student_wrap_probability > 0.0:
        if student_rng is None:
            raise ValueError("student_wrap_probability > 0 needs student_rng")
        first_view, first_ids = chosen[0]
        wrapped_student = student_rng.random() < student_wrap_probability
        if (
            wrapped_student
            and not first_view.is_identity
            and fits_context(len(first_ids), max_response_tokens, student_context)
        ):
            student_ids = tuple(first_ids)

    return ViewPair(
        original_ids=tuple(original_ids),
        view2_ids=tuple(chosen[0][1]),
        wrapper=chosen[0][0],
        extra_view_ids=tuple(tuple(ids) for _, ids in chosen[1:]),
        extra_wrappers=tuple(view for view, _ in chosen[1:]),
        student_ids=student_ids,
    )
