"""Problem records and prompt construction."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from vdopd.views import Message

# The paper does not print its task instruction; this is the conventional zero-shot wording
# for boxed final answers. The same template is used for training and evaluation prompts.
DEFAULT_PROMPT_TEMPLATE = (
    "{question}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."
)


@dataclass(frozen=True)
class Problem:
    """A question with an optional reference answer.

    ``answer`` is used only for scoring evaluations; the distillation update never reads it.
    """

    id: str
    question: str
    answer: str | None = None
    subject: str | None = None
    level: str | None = None


def build_messages(
    question: str,
    template: str = DEFAULT_PROMPT_TEMPLATE,
    system_prompt: str | None = None,
    exemplars: Sequence[Sequence[str]] = (),
) -> list[Message]:
    """Chat messages of the original prompt view.

    The prompt consists of an optional system turn, optional few-shot exemplar turns given
    as ``(question, assistant reply)`` pairs, and the final user turn. Exemplars are earlier
    dialogue turns and therefore identical across prompt views (Appendix A.1); the paper's
    mathematics protocol is zero-shot, i.e. uses none.
    """
    messages: list[Message] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    for exemplar_question, exemplar_reply in exemplars:
        messages.append({"role": "user", "content": template.format(question=exemplar_question)})
        messages.append({"role": "assistant", "content": exemplar_reply})
    messages.append({"role": "user", "content": template.format(question=question)})
    return messages


def load_jsonl(path: str | Path) -> list[Problem]:
    """Load problems from a JSON-lines file with ``id``, ``question`` and optional
    ``answer``, ``subject``, ``level`` fields (offline or custom data)."""
    problems: list[Problem] = []
    with open(path, encoding="utf-8") as handle:
        for number, line in enumerate(handle):
            if not line.strip():
                continue
            record = json.loads(line)
            problems.append(
                Problem(
                    id=str(record.get("id", number)),
                    question=record["question"],
                    answer=None if record.get("answer") is None else str(record["answer"]),
                    subject=record.get("subject"),
                    level=None if record.get("level") is None else str(record["level"]),
                )
            )
    return problems
