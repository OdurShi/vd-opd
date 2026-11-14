"""Supplementary E02-E05 (common evaluation protocol) and E12-E15 (robustness metrics)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vdopd.config import config_from_dict, load_config
from vdopd.data.problems import Problem
from vdopd.data.synthetic import SYNTHETIC_TEMPLATE
from vdopd.eval.evaluator import (
    EvalConfig,
    evaluate,
    greedy_answers,
    same_final_answer,
    sample_completions,
)
from vdopd.eval.metrics import (
    avg_at_k,
    greedy_answer_flip_rate,
    parse_failure_rate,
    pass_at_k,
    sensitivity_gaps,
    view_robustness,
)
from vdopd.pipeline import check_generation_suffix, evaluate_student, evaluation_families
from vdopd.view_pools import load_view_file, paraphrase_family
from vdopd.views import (
    EVAL_WRAPPERS,
    ORIGINAL_VIEW,
    STRUCTURAL_VIEWS,
    TemplateView,
    ViewFamily,
)

REPOSITORY = Path(__file__).resolve().parents[1]
CONFIGS = REPOSITORY / "configs"
FIXTURE = Path(__file__).parent / "fixtures" / "view_pool.jsonl"
PROBLEMS = [
    Problem("synthetic/2+3", "What is 2+3?", "5"),
    Problem("synthetic/4-1", "What is 4-1?", "3"),
    Problem("synthetic/1+1", "What is 1+1?", "2"),
]
ANSWERS = {"2+3": "5", "4-1": "3", "1+1": "2"}
CFG = EvalConfig(
    num_samples=2, max_response_tokens=12, batch_size=4, seed=11, prompt_template=SYNTHETIC_TEMPLATE
)


class RuleModel:
    """Stand-in for a language model whose answer depends on marker strings in the prompt.

    ``rules`` maps a marker to ``"wrong"`` (a wrong boxed answer), ``"none"`` (no boxed
    answer) or ``"other"`` (a different wrong answer); prompts without a marker are answered
    correctly. Every call is recorded.
    """

    max_context = None

    def __init__(self, codec, rules: dict[str, str] | None = None) -> None:
        self.codec, self.rules = codec, rules or {}
        self.calls: list[dict] = []

    def generate(self, prompts, *, max_new_tokens, temperature, top_p, seed):
        self.calls.append(
            {
                "n": len(prompts),
                "temperature": temperature,
                "top_p": top_p,
                "seed": seed,
                "max_new_tokens": max_new_tokens,
            }
        )
        responses = []
        for prompt in prompts:
            text = self.codec.tokenizer.decode(prompt)
            answer = ANSWERS[next(q for q in ANSWERS if q in text)]
            behaviour = next((b for marker, b in self.rules.items() if marker in text), "right")
            reply = {
                "right": f"\\boxed{{{answer}}}",
                "wrong": "\\boxed{0}",
                "other": "\\boxed{99}",
                "none": "no final answer",
            }[behaviour]
            responses.append(self.codec.encode_response(reply))
        return responses


# ------------------------------------------------------------------ metrics
def test_sensitivity_gaps():
    gaps = sensitivity_gaps(canonical=0.75, mean_view=0.5, worst_view=0.25)
    assert gaps == {"canonical_minus_mean": 0.25, "mean_minus_worst": 0.25}


def test_greedy_answer_flip_rate_hand_example():
    """Three questions, two views: answers change for (q2, view 1) and (q2, q3 of view 2)."""
    canonical = ["5", "3", None]
    views = [["5", "4", None], ["5.0", "7", "2"]]
    same = same_final_answer
    assert greedy_answer_flip_rate(canonical, views, same) == pytest.approx(3 / 6)  # pairs
    assert greedy_answer_flip_rate(canonical, views, same, mode="question") == pytest.approx(2 / 3)
    assert greedy_answer_flip_rate(canonical, [canonical], same) == 0.0
    assert same_final_answer("0.5", "\\frac{1}{2}") and same_final_answer(None, None)
    assert not same_final_answer(None, "2") and not same_final_answer("2", "3")
    with pytest.raises(ValueError):
        greedy_answer_flip_rate(canonical, views, same, mode="any")
    with pytest.raises(ValueError):
        greedy_answer_flip_rate(canonical, [["5"]], same)
    with pytest.raises(ValueError):
        greedy_answer_flip_rate([], [], same)


def test_parse_failure_rate():
    parsed = np.array([[[1, 1], [0, 1]], [[1, 0], [1, 1]]], dtype=bool)
    assert parse_failure_rate(parsed) == pytest.approx(2 / 8)
    with pytest.raises(ValueError):
        parse_failure_rate(np.zeros((0, 2)))


def test_worst_view_and_worst_template_definitions():
    """WorstView takes the minimum over views per question and then averages; WorstTemplate
    is the minimum of the template-level mean accuracies (E12)."""
    accuracy = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])  # [views, questions]
    correct = np.repeat(accuracy[:, :, None], 2, axis=2).astype(bool)
    metrics = view_robustness(correct)
    assert metrics["mean_view"] == pytest.approx(4 / 6)
    assert metrics["worst_view"] == pytest.approx((0 + 0 + 1) / 3)
    assert metrics["worst_template"] == pytest.approx(2 / 3)


# ------------------------------------------------------------------ evaluator
def test_sampled_completions_record_parse_failures(codec):
    model = RuleModel(codec, {"Question:": "none", "Problem statement:": "wrong"})
    scored = sample_completions(model, codec, PROBLEMS, EVAL_WRAPPERS[3], CFG, view_index=4)
    assert not scored.correct.any() and not scored.parsed.any()  # nothing to extract
    scored = sample_completions(model, codec, PROBLEMS, EVAL_WRAPPERS[2], CFG, view_index=3)
    assert not scored.correct.any() and scored.parsed.all()  # wrong but parseable
    scored = sample_completions(model, codec, PROBLEMS, ORIGINAL_VIEW, CFG)
    assert scored.correct.all() and scored.correct.shape == (3, 2)
    # A problem without a reference answer is rejected before anything is generated.
    calls = len(model.calls)
    unanswered = [*PROBLEMS, Problem("synthetic/3+3", "What is 3+3?", None)]
    with pytest.raises(ValueError, match="no reference answer"):
        sample_completions(model, codec, unanswered, ORIGINAL_VIEW, CFG)
    assert len(model.calls) == calls


def test_evaluation_over_three_held_out_families(codec):
    structural = ViewFamily("structural", STRUCTURAL_VIEWS)
    paraphrase = paraphrase_family(
        PROBLEMS, load_view_file(FIXTURE), SYNTHETIC_TEMPLATE, pool="validated_paraphrase"
    )
    rules = {
        "Question:": "wrong",  # prefix view 4 and the structural view "Question: ..."
        "Evaluate": "none",  # the fourth paraphrase of every question
        "Compute 4-1": "wrong",  # the first paraphrase of one question
    }
    model = RuleModel(codec, rules)
    result = evaluate(model, codec, PROBLEMS, CFG, families=[structural, paraphrase], greedy=True)
    assert result["avg_at_k"] == 1.0 and result["pass_at_k"] == 1.0
    assert result["parse_failures"] == 0.0
    prefix, struct, para = (result["families"][n] for n in ("prefix", "structural", "paraphrase"))
    # Prefix family: one of four templates fails on every question.
    assert result["unseen"] == {"mean_view": 0.75, "worst_view": 0.0, "worst_template": 0.0}
    assert {k: prefix[k] for k in result["unseen"]} == result["unseen"]
    assert prefix["canonical_minus_mean"] == 0.25 and prefix["mean_minus_worst"] == 0.75
    assert prefix["greedy_answer_flips"] == 0.25 and prefix["parse_failures"] == 0.0
    assert prefix["views"] == [w.name for w in EVAL_WRAPPERS]
    # Structural family: the view "Question: {content}" fails.
    assert (struct["mean_view"], struct["worst_view"], struct["worst_template"]) == (0.75, 0.0, 0.0)
    assert struct["views"] == [v.name for v in STRUCTURAL_VIEWS]
    # Paraphrases: view 4 has no final answer (3 of 12 pairs), view 1 fails for one question.
    assert para["mean_view"] == pytest.approx(8 / 12)
    assert para["worst_view"] == 0.0 and para["worst_template"] == 0.0
    assert para["parse_failures"] == pytest.approx(3 / 12)
    assert para["greedy_answer_flips"] == pytest.approx(4 / 12)
    assert para["views"] == [f"paraphrase_{i}" for i in (1, 2, 3, 4)]
    assert result["views"]["paraphrase_1"] == pytest.approx(2 / 3)
    assert result["views"]["paraphrase_4"] == 0.0 and result["views"]["delimiter_heading"] == 0.0
    # Per-question results for the paired question bootstrap.
    questions = result["questions"]
    assert questions["ids"] == [p.id for p in PROBLEMS]
    assert set(questions["correct_counts"]) == set(result["views"])
    assert questions["correct_counts"]["original"] == [2, 2, 2]
    assert questions["correct_counts"]["paraphrase_1"] == [2, 0, 2]
    by_question = evaluate(
        model, codec, PROBLEMS, CFG, families=[paraphrase], greedy=True, flip_mode="question"
    )
    assert by_question["families"]["paraphrase"]["greedy_answer_flips"] == 1.0
    assert by_question["families"]["prefix"]["greedy_answer_flips"] == 1.0


def test_added_families_do_not_change_the_random_streams_of_earlier_views(codec):
    plain, extended = RuleModel(codec), RuleModel(codec)
    evaluate(plain, codec, PROBLEMS, CFG)
    result = evaluate(extended, codec, PROBLEMS, CFG, families=[ViewFamily("s", STRUCTURAL_VIEWS)])
    seeds = [call["seed"] for call in plain.calls]
    assert [call["seed"] for call in extended.calls[: len(seeds)]] == seeds
    assert len(extended.calls) > len(seeds) and len(set(c["seed"] for c in extended.calls)) > 4
    assert "greedy_answer_flips" not in result["families"]["s"]  # no greedy pass requested
    lean = evaluate(
        RuleModel(codec), codec, PROBLEMS, CFG, unseen_wrappers=(), keep_questions=False
    )
    assert "questions" not in lean and "unseen" not in lean and "families" not in lean
    duplicate = ViewFamily("copy", (TemplateView("unseen_layout_2", "Q: {content}"),))
    with pytest.raises(ValueError):
        evaluate(RuleModel(codec), codec, PROBLEMS, CFG, families=[duplicate])


def test_paraphrases_must_keep_the_serialised_context(codec):
    """Externally written views are checked with the preservation rule of Appendix A.1."""

    class RewritingCodec:
        """A template that alters user content, so that views cannot be validated."""

        tokenizer = codec.tokenizer

        def render(self, messages):
            return codec.render([dict(m, content=m["content"].upper()) for m in messages])

        def encode(self, messages):
            return codec.encode(messages)

        def decode(self, token_ids):
            return codec.decode(token_ids)

        def encode_response(self, text):
            return codec.encode_response(text)

    paraphrase = paraphrase_family(
        PROBLEMS, load_view_file(FIXTURE), SYNTHETIC_TEMPLATE, pool="validated_paraphrase"
    )
    broken = RewritingCodec()
    with pytest.raises(ValueError, match="does not preserve"):
        evaluate(
            RuleModel(broken), broken, PROBLEMS, CFG, unseen_wrappers=(), families=[paraphrase]
        )
    ok = evaluate(RuleModel(codec), codec, PROBLEMS, CFG, unseen_wrappers=(), families=[paraphrase])
    assert ok["families"]["paraphrase"]["mean_view"] == 1.0 and "unseen" not in ok


def test_greedy_decoding_is_one_deterministic_completion_per_question(codec):
    model = RuleModel(codec, {"Problem statement:": "other"})
    answers = greedy_answers(model, codec, PROBLEMS, ORIGINAL_VIEW, CFG)
    assert answers == ["5", "3", "2"]
    assert all(call["temperature"] == 0.0 for call in model.calls)
    assert sum(call["n"] for call in model.calls) == len(PROBLEMS)
    assert greedy_answers(model, codec, PROBLEMS, EVAL_WRAPPERS[2], CFG) == ["99", "99", "99"]


# ------------------------------------------------------------------ common protocol
def test_common_protocol_defaults():
    """E05: temperature 1.0, top-p 0.8, 8,192-token cap, eight samples per question."""
    default = EvalConfig()
    assert (default.num_samples, default.temperature, default.top_p) == (8, 1.0, 0.8)
    assert default.max_response_tokens == 8192
    for group in ("qwen3_1p7b", "qwen3_4b", "llama3p2_3b"):
        decoding = load_config(CONFIGS / group / "vdopd.yaml").evaluation.decoding
        assert (decoding.num_samples, decoding.temperature, decoding.top_p) == (8, 1.0, 0.8)
        assert decoding.max_response_tokens == 8192
    qwen = load_config(CONFIGS / "qwen3_4b" / "vdopd.yaml")
    assert qwen.model.chat_template_kwargs == {"enable_thinking": False}
    assert qwen.model.expected_generation_suffix == "<think>\n\n</think>\n\n"
    assert qwen.model.stop_tokens == ("<|im_end|>",)
    assert qwen.evaluation.view_families == ("prefix",) and not qwen.evaluation.greedy_answers
    llama = load_config(CONFIGS / "llama3p2_3b" / "vdopd.yaml")
    assert llama.model.expected_generation_suffix is None


def test_teacher_and_students_are_evaluated_identically(codec):
    """Any checkpoint goes through the same prompts, decoding settings and answer checker."""
    cfg = EvalConfig(num_samples=8, max_response_tokens=8192, batch_size=8, seed=22)
    teacher, student = RuleModel(codec), RuleModel(codec, {"Question:": "wrong"})
    results = [evaluate(model, codec, PROBLEMS, cfg) for model in (teacher, student)]
    assert teacher.calls == student.calls  # same batches, seeds, temperature, top-p and cap
    assert all(
        (c["temperature"], c["top_p"], c["max_new_tokens"]) == (1.0, 0.8, 8192)
        for c in teacher.calls
    )
    assert sum(c["n"] for c in teacher.calls) == 5 * len(PROBLEMS) * 8
    counts = np.array(results[1]["questions"]["correct_counts"]["original"])
    # Avg@8 comes from the sampled correctness, Pass@8 from question-level success.
    correct = np.arange(8)[None, :] < counts[:, None]
    assert results[1]["avg_at_k"] == avg_at_k(correct) == counts.sum() / (8 * len(PROBLEMS))
    assert results[1]["pass_at_k"] == pass_at_k(correct) == np.mean(counts > 0)
    assert results[0]["unseen"]["mean_view"] == 1.0 and results[1]["unseen"]["mean_view"] == 0.75


def test_serialisation_guard_requires_the_empty_thinking_block(codec):
    class ThinkingCodec:
        def __init__(self, suffix: str) -> None:
            self.suffix = suffix

        def render(self, messages):
            return "<|im_start|>user\n" + messages[-1]["content"] + "<|im_end|>\n" + self.suffix

    expected = "<think>\n\n</think>\n\n"
    check_generation_suffix(ThinkingCodec("<|im_start|>assistant\n" + expected), expected)
    check_generation_suffix(codec, None)  # disabled
    with pytest.raises(ValueError, match="expected_generation_suffix"):
        check_generation_suffix(ThinkingCodec("<|im_start|>assistant\n"), expected)
    with pytest.raises(ValueError):
        check_generation_suffix(codec, expected)


# ------------------------------------------------------------------ pipeline
def synthetic_config(**evaluation):
    section = {"benchmarks": ["synthetic_arithmetic"], "limit": 4, **evaluation}
    section["decoding"] = {"num_samples": 2, "max_response_tokens": 12, "batch_size": 8}
    data = {"source": "synthetic_arithmetic", "prompt_template": SYNTHETIC_TEMPLATE}
    return config_from_dict({"data": data, "evaluation": section})


def test_evaluation_families_from_the_config():
    default = synthetic_config()
    assert evaluation_families(default, PROBLEMS) == []  # the prefix family is built in
    cfg = synthetic_config(view_families=["prefix", "structural"])
    (family,) = evaluation_families(cfg, PROBLEMS)
    assert family.name == "structural" and family.views == STRUCTURAL_VIEWS
    custom = synthetic_config(
        view_families=["structural"], structural_views=[["tail", "{content}\n\nGo."]]
    )
    assert evaluation_families(custom, PROBLEMS)[0].views == (
        TemplateView("tail", "{content}\n\nGo."),
    )
    with pytest.raises(ValueError, match="paraphrase_file"):
        evaluation_families(synthetic_config(view_families=["paraphrase"]), PROBLEMS)
    with pytest.raises(ValueError, match="unknown view families"):
        evaluation_families(synthetic_config(view_families=["prefix", "typo"]), PROBLEMS)
    para = synthetic_config(
        view_families=["paraphrase"],
        paraphrase_file=str(FIXTURE),
        paraphrase_pool="validated_paraphrase",
    )
    (family,) = evaluation_families(para, PROBLEMS)
    assert family.name == "paraphrase" and len(family.views_for("synthetic/2+3")) == 4
    off = synthetic_config(view_families=["structural"], unseen_wrappers=False)
    assert evaluation_families(off, PROBLEMS) == []


def test_evaluate_student_macro_averages_every_family(codec):
    class AnyQuestion(RuleModel):
        def generate(self, prompts, **kwargs):
            text = [self.codec.tokenizer.decode(p) for p in prompts]
            wrong = ["Question:" in t for t in text]
            self.calls.append({"n": len(prompts), **kwargs})
            return [self.codec.encode_response("\\boxed{0}" if w else "no answer") for w in wrong]

    cfg = synthetic_config(view_families=["prefix", "structural"], greedy_answers=True)
    results = evaluate_student(cfg, AnyQuestion(codec), codec)
    benchmark = results["benchmarks"]["synthetic_arithmetic"]
    assert benchmark["num_questions"] == 4 and set(benchmark["families"]) == {
        "prefix",
        "structural",
    }
    macro = results["macro"]
    assert set(macro["families"]) == {"prefix", "structural"}
    for name, family in macro["families"].items():
        assert family == {
            key: value for key, value in benchmark["families"][name].items() if key != "views"
        }
        assert family["parse_failures"] == 0.75  # three of four views give no boxed answer
    assert macro["mean_view"] == macro["families"]["prefix"]["mean_view"]
    assert len(benchmark["questions"]["ids"]) == 4
    lean = evaluate_student(synthetic_config(unseen_wrappers=False), AnyQuestion(codec), codec)
    assert "families" not in lean["macro"] and "mean_view" not in lean["macro"]
