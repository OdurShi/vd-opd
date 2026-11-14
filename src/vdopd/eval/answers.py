r"""Final-answer extraction and mathematical equivalence (Appendix B.2.1).

The paper freezes "the symbolic-equivalence rules for mathematical answers" but does not
list them. This module implements the conventional pipeline used for these benchmarks: the
final answer is the content of the last ``\boxed{...}`` in a completion; answers are compared
after a MATH-style string normalisation, then numerically, then symbolically with SymPy.
Missing or malformed final answers receive zero credit.

Model output is never passed to ``eval`` unchecked: expressions are converted from a small
LaTeX subset and must consist solely of whitelisted characters and names before parsing.
"""

from __future__ import annotations

import re
from fractions import Fraction

_ALLOWED_NAMES = {"sqrt", "pi", "oo"}
_MAX_EXPRESSION_LENGTH = 160


def last_boxed_string(text: str) -> str | None:
    r"""Return the last ``\boxed{...}`` / ``\fbox{...}`` including braces, or ``None``."""
    start = max(text.rfind("\\boxed"), text.rfind("\\fbox"))
    if start < 0:
        return None
    brace = text.find("{", start)
    if brace < 0 or text[start:brace].strip() not in ("\\boxed", "\\fbox"):
        return None
    depth = 0
    for position in range(brace, len(text)):
        if text[position] == "{":
            depth += 1
        elif text[position] == "}":
            depth -= 1
            if depth == 0:
                return text[start : position + 1]
    return None  # unbalanced braces: malformed


def extract_answer(text: str) -> str | None:
    r"""Final answer of a completion: the content of its last ``\boxed{...}``.

    Returns ``None`` when no well-formed boxed answer exists or the box is empty.
    """
    boxed = last_boxed_string(text)
    if boxed is None:
        return None
    content = boxed[boxed.index("{") + 1 : -1].strip()
    return content or None


def _strip_text_commands(string: str) -> str:
    for command in ("\\text", "\\textbf", "\\mathrm", "\\mbox", "\\textit", "\\mathbf"):
        pattern = re.compile(re.escape(command) + r"\s*\{([^{}]*)\}")
        while pattern.search(string):
            string = pattern.sub(lambda m: m.group(1), string)
    return string


def _fix_fracs(string: str) -> str:
    r"""``\frac12`` -> ``\frac{1}{2}`` and ``\frac1{72}`` -> ``\frac{1}{72}``."""
    pieces = string.split("\\frac")
    fixed = pieces[0]
    for piece in pieces[1:]:
        fixed += "\\frac"
        if piece.startswith("{") or len(piece) < 2:
            fixed += piece
            continue
        numerator, rest = piece[0], piece[1:]
        if rest.startswith("{"):
            fixed += "{" + numerator + "}" + rest
        else:
            fixed += "{" + numerator + "}{" + rest[0] + "}" + rest[1:]
    return fixed


def _fix_sqrt(string: str) -> str:
    r"""``\sqrt3`` -> ``\sqrt{3}``."""
    return re.sub(r"\\sqrt(\w)", r"\\sqrt{\1}", string)


def _fix_simple_slash(string: str) -> str:
    """``a/b`` with integer ``a`` and ``b`` -> ``\\frac{a}{b}``."""
    match = re.fullmatch(r"(-?\d+)/(\d+)", string)
    return f"\\frac{{{match.group(1)}}}{{{match.group(2)}}}" if match else string


def normalize_answer(answer: str) -> str:
    """MATH-style normalisation of a final answer string."""
    string = answer.strip().replace("\n", "")
    string = string.replace("\\!", "").replace("\\,", "").replace("\\;", "").replace("\\ ", "")
    string = string.replace("\\\\", "\\")
    string = string.replace("tfrac", "frac").replace("dfrac", "frac")
    string = string.replace("\\left", "").replace("\\right", "")
    string = string.replace("^{\\circ}", "").replace("^\\circ", "")
    string = string.replace("\\$", "").replace("$", "")
    string = string.replace("\\%", "").replace("%", "")
    unit = re.fullmatch(r"(.*\d.*?)\\text\s*\{\s*[a-zA-Z ]+\}\s*", string)
    if unit:  # trailing units after a number, e.g. "5\\text{ cm}"
        string = unit.group(1)
    string = _strip_text_commands(string)
    string = string.rstrip(".")
    if re.fullmatch(r"-?\d{1,3}(,\d{3})+(\.\d+)?", string):
        string = string.replace(",", "")  # thousands separators
    string = string.replace(" .", " 0.").replace("{.", "{0.")
    if string.startswith("."):
        string = "0" + string
    if string.startswith("-."):
        string = "-0" + string[1:]
    parts = string.split("=")
    if len(parts) == 2 and len(parts[0].strip()) <= 2:  # "x = 5" -> "5"
        string = parts[1]
    string = _fix_sqrt(string)
    string = string.replace(" ", "")
    string = _fix_fracs(string)
    string = _fix_simple_slash(string)
    return string


def _matching_brace(string: str, start: int) -> int:
    depth = 0
    for position in range(start, len(string)):
        if string[position] == "{":
            depth += 1
        elif string[position] == "}":
            depth -= 1
            if depth == 0:
                return position
    return -1


def _latex_to_expression(string: str) -> str | None:
    """Convert a small LaTeX subset to a plain arithmetic expression, or ``None``."""
    for _ in range(64):
        index = string.find("\\frac{")
        if index < 0:
            break
        end_num = _matching_brace(string, index + 5)
        if end_num < 0 or end_num + 1 >= len(string) or string[end_num + 1] != "{":
            return None
        end_den = _matching_brace(string, end_num + 1)
        if end_den < 0:
            return None
        numerator = string[index + 6 : end_num]
        denominator = string[end_num + 2 : end_den]
        string = f"{string[:index]}(({numerator})/({denominator})){string[end_den + 1 :]}"
    for _ in range(64):
        index = string.find("\\sqrt{")
        if index < 0:
            break
        end = _matching_brace(string, index + 5)
        if end < 0:
            return None
        string = f"{string[:index]}sqrt({string[index + 6 : end]}){string[end + 1 :]}"
    string = string.replace("\\pi", "pi").replace("\\infty", "oo")
    string = string.replace("\\cdot", "*").replace("\\times", "*").replace("\\div", "/")
    string = string.replace("^", "**").replace("{", "(").replace("}", ")")
    return None if "\\" in string else string


def _is_safe_expression(expression: str) -> bool:
    if not expression or len(expression) > _MAX_EXPRESSION_LENGTH:
        return False
    if not re.fullmatch(r"[0-9a-zA-Z+\-*/(). ]+", expression):
        return False
    if re.search(r"\.(?!\d)", expression) or re.search(r"(?<!\d)\.", expression):
        return False  # a dot may only appear inside a decimal number
    for name in re.findall(r"[a-zA-Z]+", expression):
        if len(name) > 1 and name not in _ALLOWED_NAMES:
            return False
    if expression.count("**") > 2 or re.search(r"\*\*\s*\(?\s*-?\d{4,}", expression):
        return False  # guard against enormous powers
    return True


def _parse_expression(answer: str):
    """Parse a normalised answer into a SymPy expression, or ``None`` if not possible."""
    from sympy.parsing.sympy_parser import (
        convert_xor,
        implicit_multiplication_application,
        parse_expr,
        standard_transformations,
    )

    expression = _latex_to_expression(answer)
    if expression is None or not _is_safe_expression(expression):
        return None
    transformations = standard_transformations + (
        implicit_multiplication_application,
        convert_xor,
    )
    try:
        return parse_expr(expression, transformations=transformations, evaluate=True)
    except Exception:  # malformed model output must never crash the evaluator
        return None


def _as_fraction(answer: str) -> Fraction | None:
    """Exact rational value of plain numerals such as ``-3``, ``0.25`` or ``\\frac{1}{4}``."""
    match = re.fullmatch(r"(-?)\\frac\{(-?\d+)\}\{(-?\d+)\}", answer)
    try:
        if match:
            value = Fraction(int(match.group(2)), int(match.group(3)))
            return -value if match.group(1) else value
        if re.fullmatch(r"-?\d+(\.\d+)?", answer):
            return Fraction(answer)
    except (ValueError, ZeroDivisionError):
        return None
    return None


def _split_top_level(string: str) -> tuple[str, list[str], str] | None:
    """Split ``(a,b)`` / ``[a,b]`` / ``a,b`` at top-level commas; ``None`` if no comma."""
    opening, closing, body = "", "", string
    if len(string) >= 2 and string[0] in "([" and string[-1] in ")]":
        opening, closing, body = string[0], string[-1], string[1:-1]
    parts, depth, current = [], 0, ""
    for char in body:
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        if char == "," and depth == 0:
            parts.append(current)
            current = ""
        else:
            current += char
    parts.append(current)
    return (opening, parts, closing) if len(parts) > 1 else None


def is_equivalent(prediction: str | None, reference: str, rel_tol: float = 1e-9) -> bool:
    """Whether a predicted final answer is mathematically equivalent to the reference.

    The checks are, in order: equality after :func:`normalize_answer`; exact rational
    equality; elementwise equivalence of tuples, intervals and lists; numeric equality within
    ``rel_tol``; and symbolic equality (``simplify(a - b) == 0``). ``None`` (missing or
    malformed answer) is never correct.
    """
    if prediction is None:
        return False
    a, b = normalize_answer(prediction), normalize_answer(reference)
    if not a:
        return False
    if a == b:
        return True
    fa, fb = _as_fraction(a), _as_fraction(b)
    if fa is not None and fb is not None:
        return fa == fb or (
            rel_tol > 0
            and abs(float(fa) - float(fb)) <= rel_tol * max(abs(float(fa)), abs(float(fb)))
        )
    split_a, split_b = _split_top_level(a), _split_top_level(b)
    if split_a is not None or split_b is not None:
        if split_a is None or split_b is None:
            return False
        if (split_a[0], split_a[2]) != (split_b[0], split_b[2]):
            return False
        if len(split_a[1]) != len(split_b[1]):
            return False
        return all(
            is_equivalent(x, y, rel_tol) for x, y in zip(split_a[1], split_b[1], strict=True)
        )
    return _symbolically_equal(a, b, rel_tol)


def _symbolically_equal(a: str, b: str, rel_tol: float) -> bool:
    import sympy

    expr_a, expr_b = _parse_expression(a), _parse_expression(b)
    if expr_a is None or expr_b is None:
        return False
    try:
        difference = expr_a - expr_b
        if not difference.free_symbols:
            value = complex(sympy.N(difference, 30))
            scale = max(abs(complex(sympy.N(expr_a, 30))), abs(complex(sympy.N(expr_b, 30))), 1.0)
            return abs(value) <= max(rel_tol, 1e-12) * scale
        return sympy.simplify(difference) == 0
    except Exception:  # malformed model output must never crash the evaluator
        return False


def gsm8k_reference(solution: str) -> str:
    """Reference answer of a GSM8K item: the text after the final ``####`` marker."""
    return solution.split("####")[-1].strip().replace(",", "")


def extract_choice(text: str, num_options: int) -> str | None:
    r"""Option letter of a multiple-choice completion.

    Uses the last ``\boxed{X}`` if present, otherwise the last "answer is (X)" pattern.
    """
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:num_options]
    boxed = extract_answer(text)
    if boxed is not None:
        candidate = _strip_text_commands(boxed).strip().strip("()").strip().upper()
        if len(candidate) == 1 and candidate in letters:
            return candidate
    matches = re.findall(r"answer is\s*:?\s*\(?([A-Za-z])\)?", text)
    if matches and matches[-1].upper() in letters:
        return matches[-1].upper()
    return None
