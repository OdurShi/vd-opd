"""Chat serialisation codecs (Appendix A.1 and B.4).

Teacher and student share one fixed serialisation: the teacher's chat template with the
assistant generation prefix appended. For Qwen3 this is the non-thinking serialisation,
which includes the fixed empty thinking block. A small character-level codec built in code
serves the synthetic quick start and the offline tests.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from vdopd.views import Message


class HFChatCodec:
    """Codec backed by a Hugging Face tokenizer and its chat template.

    Args:
        tokenizer: tokenizer of the teacher (shared with the student).
        template_kwargs: extra keyword arguments for ``apply_chat_template``. For Qwen3 use
            ``{"enable_thinking": False}`` to obtain the non-thinking serialisation with the
            fixed empty thinking block (Appendix B.4).
    """

    def __init__(self, tokenizer: Any, template_kwargs: dict[str, Any] | None = None) -> None:
        if getattr(tokenizer, "chat_template", None) is None:
            raise ValueError("the tokenizer has no chat template; pass the teacher's tokenizer")
        self.tokenizer = tokenizer
        self.template_kwargs = dict(template_kwargs or {})

    def render(self, messages: Sequence[Message]) -> str:
        return self.tokenizer.apply_chat_template(
            list(messages), tokenize=False, add_generation_prompt=True, **self.template_kwargs
        )

    def encode(self, messages: Sequence[Message]) -> list[int]:
        # The template already contains every special token (e.g. BOS) it needs.
        return list(self.tokenizer(self.render(messages), add_special_tokens=False)["input_ids"])

    def encode_text(self, text: str) -> list[int]:
        """Token ids of plain text without special tokens (candidate answers, Equation 15)."""
        return list(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def decode(self, token_ids: Sequence[int]) -> str:
        return self.tokenizer.decode(list(token_ids), skip_special_tokens=True)


class CharTokenizer:
    """Deterministic character-level tokenizer with a few special tokens (synthetic task)."""

    PAD, EOS, SYSTEM, USER, ASSISTANT, END = (
        "<pad>",
        "<eos>",
        "<|system|>",
        "<|user|>",
        "<|assistant|>",
        "<|end|>",
    )
    SPECIALS = (PAD, EOS, SYSTEM, USER, ASSISTANT, END)
    ALPHABET = (
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        " \n.,:;?!+-*/=()[]{}\\_^'\"<>|%$&#@~"
    )

    def __init__(self) -> None:
        self.vocab: list[str] = list(self.SPECIALS) + list(self.ALPHABET)
        self._index = {token: i for i, token in enumerate(self.vocab)}
        self.pad_token_id = self._index[self.PAD]
        self.eos_token_id = self._index[self.EOS]

    def __len__(self) -> int:
        return len(self.vocab)

    def token_id(self, token: str) -> int:
        return self._index[token]

    def encode_text(self, text: str) -> list[int]:
        unknown = sorted({ch for ch in text if ch not in self._index})
        if unknown:
            raise ValueError(f"characters outside the synthetic alphabet: {unknown}")
        return [self._index[ch] for ch in text]

    def decode(self, token_ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        pieces = []
        for token_id in token_ids:
            token = self.vocab[int(token_id)]
            if skip_special_tokens and token in self.SPECIALS:
                continue
            pieces.append(token)
        return "".join(pieces)


class SimpleChatCodec:
    """Minimal chat serialisation over :class:`CharTokenizer`.

    Every message is rendered as ``<|role|>content<|end|>`` and the prompt ends with the fixed
    assistant generation prefix ``<|assistant|>``.
    """

    _ROLE_TOKENS = {
        "system": CharTokenizer.SYSTEM,
        "user": CharTokenizer.USER,
        "assistant": CharTokenizer.ASSISTANT,
    }

    def __init__(self, tokenizer: CharTokenizer | None = None) -> None:
        self.tokenizer = tokenizer or CharTokenizer()

    def render(self, messages: Sequence[Message]) -> str:
        parts = [
            f"{self._ROLE_TOKENS[m['role']]}{m['content']}{CharTokenizer.END}" for m in messages
        ]
        return "".join(parts) + CharTokenizer.ASSISTANT

    def encode(self, messages: Sequence[Message]) -> list[int]:
        tok = self.tokenizer
        ids: list[int] = []
        for message in messages:
            ids.append(tok.token_id(self._ROLE_TOKENS[message["role"]]))
            ids.extend(tok.encode_text(message["content"]))
            ids.append(tok.token_id(CharTokenizer.END))
        ids.append(tok.token_id(CharTokenizer.ASSISTANT))
        return ids

    def encode_text(self, text: str) -> list[int]:
        """Token ids of plain text without special tokens (candidate answers, Equation 15)."""
        return self.tokenizer.encode_text(text)

    def encode_response(self, text: str) -> list[int]:
        """Token ids of an assistant response followed by EOS (used for toy pretraining)."""
        return self.tokenizer.encode_text(text) + [self.tokenizer.eos_token_id]

    def decode(self, token_ids: Sequence[int]) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)
