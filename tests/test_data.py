"""Appendix B.1 and B.7: training-split construction, and the synthetic task."""

from __future__ import annotations

import json
from collections import Counter

import pytest

from vdopd.data.math_train import (
    deduplicate,
    question_key,
    remove_overlaps,
    split_diagnostic,
    stratified_split,
)
from vdopd.data.problems import DEFAULT_PROMPT_TEMPLATE, Problem, build_messages, load_jsonl
from vdopd.data.synthetic import make_arithmetic_problems, reference_response, split_problems
from vdopd.eval.answers import extract_answer, is_equivalent


def fake_math_train(n_per_stratum: int = 150) -> list[Problem]:
    problems = []
    for subject in ("Algebra", "Geometry", "Number Theory"):
        for level in ("Level 1", "Level 3", "Level 5"):
            count = n_per_stratum * (2 if subject == "Algebra" else 1)
            for i in range(count):
                problems.append(
                    Problem(
                        f"{subject}/{level}/{i}", f"{subject} {level} q{i}", "1", subject, level
                    )
                )
    return problems


def test_deduplication_and_overlap_removal():
    problems = [
        Problem("a", "What is  2+2?"),
        Problem("b", "what is 2+2?"),  # duplicate up to case and whitespace
        Problem("c", "What is 3+3?"),
    ]
    unique = deduplicate(problems)
    assert [p.id for p in unique] == ["a", "c"]
    assert question_key(" A \n B ") == "a b"
    assert [p.id for p in remove_overlaps(unique, ["WHAT IS 3+3?"])] == ["a"]


def test_stratified_split_sizes_disjointness_and_determinism():
    problems = fake_math_train()
    splits = stratified_split(problems, num_diagnostic=500, num_development=500, seed=42)
    assert len(splits.diagnostic) == 500 and len(splits.development) == 500
    assert len(splits.optimization) == len(problems) - 1000
    ids = [
        p.id for part in (splits.diagnostic, splits.development, splits.optimization) for p in part
    ]
    assert len(ids) == len(set(ids)) == len(problems)  # every problem is in exactly one split
    again = stratified_split(problems, num_diagnostic=500, num_development=500, seed=42)
    assert [p.id for p in again.diagnostic] == [p.id for p in splits.diagnostic]
    other = stratified_split(problems, num_diagnostic=500, num_development=500, seed=7)
    assert [p.id for p in other.diagnostic] != [p.id for p in splits.diagnostic]
    with pytest.raises(ValueError):
        stratified_split(problems[:100], num_diagnostic=500)


def test_split_is_stratified_by_subject_and_difficulty():
    problems = fake_math_train()
    splits = stratified_split(problems, num_diagnostic=500, num_development=500, seed=42)
    population = Counter((p.subject, p.level) for p in problems)
    for part in (splits.diagnostic, splits.development):
        counts = Counter((p.subject, p.level) for p in part)
        for stratum, size in population.items():
            expected = 500 * size / len(problems)
            assert abs(counts[stratum] - expected) <= 1.0


def test_diagnostic_split_into_update_and_measurement_problems():
    diagnostic = stratified_split(fake_math_train(), seed=42).diagnostic
    update, measurement = split_diagnostic(diagnostic, num_update=250, seed=42)
    assert len(update) == 250 and len(measurement) == 250
    assert {p.id for p in update}.isdisjoint({p.id for p in measurement})
    assert split_diagnostic(diagnostic, seed=42)[0] == update


def test_prompt_construction_and_jsonl_loading(tmp_path):
    messages = build_messages("What is 1+1?")
    assert [m["role"] for m in messages] == ["user"]
    assert messages[0]["content"] == DEFAULT_PROMPT_TEMPLATE.format(question="What is 1+1?")
    assert "\\boxed{}" in messages[0]["content"]
    with_system = build_messages("q", "{question}", system_prompt="sys")
    assert with_system == [{"role": "system", "content": "sys"}, {"role": "user", "content": "q"}]
    path = tmp_path / "problems.jsonl"
    path.write_text(
        json.dumps({"id": "x", "question": "q1", "answer": 3})
        + "\n\n"
        + json.dumps({"question": "q2"})
        + "\n"
    )
    loaded = load_jsonl(path)
    assert loaded == [Problem("x", "q1", "3"), Problem("2", "q2", None)]  # default id = line number


def test_synthetic_arithmetic_task():
    problems = make_arithmetic_problems(max_operand=9)
    assert len(problems) == 200 and len({p.id for p in problems}) == 200
    by_id = {p.id: p for p in problems}
    assert by_id["synthetic/7+8"].answer == "15" and by_id["synthetic/3-9"].answer == "-6"
    train, test = split_problems(problems, test_fraction=0.2, seed=0)
    assert len(test) == 40 and {p.id for p in train}.isdisjoint({p.id for p in test})
    for problem in problems[:20]:
        assert is_equivalent(extract_answer(reference_response(problem)), problem.answer)


def test_exemplar_turns_precede_the_final_user_message():
    messages = build_messages(
        "What is 4+16?", "{question}", "sys", exemplars=[("What is 21+30?", "\\boxed{51}")]
    )
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "user"]
    assert messages[1]["content"] == "What is 21+30?" and messages[2]["content"] == "\\boxed{51}"
    assert messages[-1]["content"] == "What is 4+16?"
