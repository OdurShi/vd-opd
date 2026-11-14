"""Appendix B.2: answer extraction, equivalence, Avg@k / Pass@k and view robustness."""

from __future__ import annotations

import numpy as np
import pytest

from vdopd.data.benchmarks import BENCHMARKS, SIX_MATH, records_to_problems
from vdopd.data.problems import Problem
from vdopd.eval.answers import (
    extract_answer,
    extract_choice,
    gsm8k_reference,
    is_equivalent,
    last_boxed_string,
    normalize_answer,
)
from vdopd.eval.evaluator import EvalConfig, evaluate, sample_correctness
from vdopd.eval.metrics import (
    avg_at_k,
    macro_average,
    pass_at_k,
    per_question_accuracy,
    seed_mean_and_sd,
    view_robustness,
)
from vdopd.views import EVAL_WRAPPERS, ORIGINAL_VIEW


def test_extract_last_boxed_answer():
    assert extract_answer("So \\boxed{3}, wait, actually \\boxed{42}.") == "42"
    assert extract_answer("The answer is \\boxed{\\frac{1}{2}}") == "\\frac{1}{2}"
    assert extract_answer("\\boxed{\\text{(A) } \\{1, 2\\}}") == "\\text{(A) } \\{1, 2\\}"
    assert last_boxed_string("x \\boxed{a{b}c} y") == "\\boxed{a{b}c}"
    assert extract_answer("\\fbox{7}") == "7"


@pytest.mark.parametrize(
    "text", ["no final answer here", "\\boxed{unbalanced", "\\boxed{}", "\\boxed 5", ""]
)
def test_missing_or_malformed_answers_are_none(text):
    assert extract_answer(text) is None
    assert not is_equivalent(extract_answer(text), "5")


@pytest.mark.parametrize(
    ("prediction", "reference"),
    [
        ("42", "42"),
        (" 42 ", "42."),
        ("\\dfrac{1}{2}", "\\frac{1}{2}"),
        ("\\frac12", "\\frac{1}{2}"),
        ("1/2", "\\frac{1}{2}"),
        ("0.5", "\\frac{1}{2}"),
        ("\\frac{2}{4}", "\\frac{1}{2}"),
        ("1,000", "1000"),
        ("x = 5", "5"),
        ("5\\text{ cm}", "5"),
        ("30^\\circ", "30"),
        ("\\$12.50", "12.5"),
        ("50\\%", "50"),
        ("\\left( 3, \\frac{\\pi}{2} \\right)", "(3,\\frac{\\pi}{2})"),
        ("(0.5, 3)", "(\\frac{1}{2},3)"),
        ("2\\sqrt{2}", "\\sqrt{8}"),
        ("\\sqrt3", "\\sqrt{3}"),
        ("x^2+2x+1", "(x+1)^2"),
        ("\\frac{\\sqrt{3}}{3}", "\\frac{1}{\\sqrt{3}}"),
        ("\\text{east}", "\\text{east}"),
        ("-\\frac{3}{4}", "-0.75"),
    ],
)
def test_equivalent_answers(prediction, reference):
    assert is_equivalent(prediction, reference)


@pytest.mark.parametrize(
    ("prediction", "reference"),
    [
        ("41", "42"),
        ("\\frac{1}{3}", "\\frac{1}{2}"),
        ("(1,2)", "(2,1)"),
        ("[1,2]", "(1,2)"),
        ("(1,2,3)", "(1,2)"),
        ("x+1", "x+2"),
        ("\\text{west}", "\\text{east}"),
        ("", "5"),
        ("\\sqrt{2}", "2"),
        ("__import__('os').system('true')", "0"),
        ("9**9**9**9", "1"),
    ],
)
def test_non_equivalent_answers(prediction, reference):
    assert not is_equivalent(prediction, reference)


def test_normalisation_and_auxiliary_extractors():
    assert normalize_answer("\\tfrac{3}{4}") == "\\frac{3}{4}"
    assert normalize_answer(".5") == "0.5"
    assert normalize_answer("\\text{Monday}") == "Monday"
    assert gsm8k_reference("He has 3 + 4 = <<3+4=7>>7 apples.\n#### 1,234") == "1234"
    assert extract_choice("Therefore \\boxed{C}", 4) == "C"
    assert extract_choice("Therefore \\boxed{\\text{(B)}}", 4) == "B"
    assert extract_choice("so the answer is (D).", 4) == "D"
    assert extract_choice("so the answer is (E).", 4) is None
    assert extract_choice("no idea", 4) is None


def test_avg_and_pass_at_k_hand_example():
    """Equation 10 with n = 3 questions and k = 4 samples: c = (4, 1, 0)."""
    correct = np.array([[1, 1, 1, 1], [0, 1, 0, 0], [0, 0, 0, 0]], dtype=bool)
    assert avg_at_k(correct) == pytest.approx(5 / 12)
    assert pass_at_k(correct) == pytest.approx(2 / 3)
    assert per_question_accuracy(correct).tolist() == [1.0, 0.25, 0.0]
    assert macro_average([0.5, 0.25, 0.75]) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        avg_at_k(np.zeros((0, 8)))


def test_view_robustness_hand_example():
    """Equations 11-12 with M = 2 wrappers, n = 2 questions, a = [[1, 0], [0.5, 0.5]]."""
    correct = np.array([[[1, 1], [0, 0]], [[1, 0], [0, 1]]], dtype=bool)
    metrics = view_robustness(correct)
    assert metrics["mean_view"] == pytest.approx(0.5)
    assert metrics["worst_view"] == pytest.approx((0.5 + 0.0) / 2)  # per-question minimum
    assert metrics["worst_template"] == pytest.approx(0.5)  # minimum of per-template means


def test_view_metric_ordering_holds():
    """WorstView <= WorstTemplate <= MeanView (Appendix B.2.2)."""
    rng = np.random.default_rng(0)
    for _ in range(50):
        correct = rng.random((4, 30, 8)) < rng.random((4, 1, 1))
        metrics = view_robustness(correct)
        assert metrics["worst_view"] <= metrics["worst_template"] + 1e-12
        assert metrics["worst_template"] <= metrics["mean_view"] + 1e-12


def test_seed_statistics_use_the_sample_standard_deviation():
    mean, sd = seed_mean_and_sd([66.0, 67.0, 68.0])  # Equation 14
    assert mean == pytest.approx(67.0) and sd == pytest.approx(1.0)
    assert seed_mean_and_sd([5.0]) == (5.0, 0.0)


class ScriptedModel:
    """Stand-in for a language model: answers correctly only under selected wrappers."""

    max_context = None

    def __init__(self, codec, answers: dict[str, str], broken_prefixes: tuple[str, ...]) -> None:
        self.codec, self.answers, self.broken = codec, answers, broken_prefixes
        self.calls: list[dict] = []

    def generate(self, prompts, *, max_new_tokens, temperature, top_p, seed):
        self.calls.append(
            {
                "n": len(prompts),
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "seed": seed,
            }
        )
        responses = []
        for prompt in prompts:
            text = self.codec.tokenizer.decode(prompt)
            question = next(q for q in self.answers if q in text)
            wrong = any(prefix in text for prefix in self.broken)
            answer = "0" if wrong else self.answers[question]
            responses.append(self.codec.encode_response(f"\\boxed{{{answer}}}"))
        return responses


def test_evaluator_scores_each_view_independently(codec):
    problems = [Problem("a", "What is 2+3?", "5"), Problem("b", "What is 4+4?", "8")]
    model = ScriptedModel(
        codec, {"What is 2+3?": "5", "What is 4+4?": "8"}, broken_prefixes=("Question:",)
    )
    cfg = EvalConfig(num_samples=3, max_response_tokens=12, batch_size=4, seed=11)
    result = evaluate(model, codec, problems, cfg)
    assert result["avg_at_k"] == 1.0 and result["pass_at_k"] == 1.0
    assert result["views"]["unseen_layout_2"] == 0.0  # the "Question:" wrapper
    assert result["views"]["unseen_instruction_1"] == 1.0
    assert set(result["views"]) == {ORIGINAL_VIEW.name, *(w.name for w in EVAL_WRAPPERS)}
    assert result["unseen"] == {"mean_view": 0.75, "worst_view": 0.0, "worst_template": 0.0}
    # Paper decoding defaults: 8 samples, temperature 1.0, top-p 0.8, 8,192-token cap.
    default = EvalConfig()
    assert (default.num_samples, default.temperature, default.top_p) == (8, 1.0, 0.8)
    assert default.max_response_tokens == 8192
    assert all(call["temperature"] == 1.0 and call["top_p"] == 0.8 for call in model.calls)
    assert sum(call["n"] for call in model.calls) == 5 * 2 * 3  # views x questions x samples
    # Seeds depend on the view and batch only, so checkpoints share random streams.
    again = ScriptedModel(codec, model.answers, ())
    sample_correctness(again, codec, problems, ORIGINAL_VIEW, cfg, view_index=0)
    assert [c["seed"] for c in again.calls] == [c["seed"] for c in model.calls[:2]]


def test_benchmark_registry_and_record_conversion():
    assert SIX_MATH == ("math500", "amc23", "aime24", "aime25", "minerva", "olympiadbench")
    assert set(SIX_MATH) < set(BENCHMARKS) and "gsm8k" in BENCHMARKS
    plain = records_to_problems(
        [{"problem": "1+1?", "answer": "2", "unique_id": "t/1", "subject": "Algebra", "level": 2}],
        BENCHMARKS["math500"],
        "math500",
    )
    assert plain[0] == Problem("math500/t/1", "1+1?", "2", "Algebra", "2")
    gsm = records_to_problems(
        [{"question": "q", "answer": "work\n#### 18"}], BENCHMARKS["gsm8k"], "gsm8k"
    )
    assert gsm[0].answer == "18" and gsm[0].id == "gsm8k/0"
    olympiad = records_to_problems(
        [{"id": 7, "question": "q", "final_answer": "['$\\\\frac{1}{2}$']"}],
        BENCHMARKS["olympiadbench"],
        "olympiadbench",
    )
    assert is_equivalent("\\frac{1}{2}", olympiad[0].answer)
