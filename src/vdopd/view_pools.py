"""Externally produced prompt views: file schema, loader and validation.

Supplementary experiment E10 trains with view pools other than the simple templates of
Table 5 (validated paraphrases; lower-quality but task-preserving views) and E15 evaluates on
a held-out family of verified paraphrases. Such views are written and checked by people
outside this repository; nothing here generates text. This module defines the file they are
delivered in and turns it into :class:`~vdopd.views.RewrittenView` objects for the trainer
and the evaluator.

File schema
-----------
A view file is UTF-8 JSON Lines, one record per view of one problem::

    {"problem_id": "math_train/algebra/12", "view": "paraphrase_1",
     "question": "<rewritten question text>", "pool": "validated_paraphrase",
     "validated": true}

``problem_id``  (required) id of the source problem, as in the problem files.
``view``        (required) name of the view; unique within a problem. In a held-out
                evaluation family every problem has the same view names.
``question``    (required) the rewritten *question* text. It is inserted into the same prompt
                template as the original question, so the task instruction and the
                answer-format requirement are retained verbatim.
``pool``        (optional) label of the pool a record belongs to, so that one file can hold
                several pools (e.g. ``validated_paraphrase`` and ``lower_quality``).
``validated``   (optional, default false) whether a person confirmed that the view preserves
                the question's conditions, variables, answer and roles.
``source``, ``notes``  (optional) free text, ignored.

Other keys are rejected, so that a misspelt field cannot be dropped silently.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from vdopd.data.problems import Problem
from vdopd.views import PARAPHRASE_FAMILY, RewrittenView, ViewFamily

REQUIRED_FIELDS = ("problem_id", "view", "question")
OPTIONAL_FIELDS = ("pool", "validated", "source", "notes")


@dataclass(frozen=True)
class ViewRecord:
    """One externally produced view of one problem (a line of a view file)."""

    problem_id: str
    view: str
    question: str
    pool: str | None = None
    validated: bool = False


def parse_view_record(record: object, where: str = "record") -> ViewRecord:
    """Validate one decoded JSON object against the schema and convert it."""
    if not isinstance(record, dict):
        raise ValueError(f"{where}: expected a JSON object")
    unknown = sorted(set(record) - set(REQUIRED_FIELDS) - set(OPTIONAL_FIELDS))
    if unknown:
        raise ValueError(f"{where}: unknown field(s) {unknown}")
    for name in REQUIRED_FIELDS:
        value = record.get(name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{where}: field {name!r} must be a non-empty string")
    pool = record.get("pool")
    if pool is not None and (not isinstance(pool, str) or not pool.strip()):
        raise ValueError(f"{where}: field 'pool' must be a non-empty string")
    validated = record.get("validated", False)
    if not isinstance(validated, bool):
        raise ValueError(f"{where}: field 'validated' must be true or false")
    return ViewRecord(record["problem_id"], record["view"], record["question"], pool, validated)


def load_view_file(path: str | Path) -> list[ViewRecord]:
    """Read a view file; malformed lines raise ``ValueError`` naming the line."""
    records: list[ViewRecord] = []
    with open(path, encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            where = f"{Path(path).name}:{number}"
            try:
                decoded = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{where}: invalid JSON ({error.msg})") from error
            records.append(parse_view_record(decoded, where))
    return records


def group_views(
    records: Sequence[ViewRecord],
    problems: Sequence[Problem] | None = None,
    *,
    pool: str | None = None,
    require_validated: bool = True,
    views_per_problem: int | None = None,
    require_all_problems: bool = False,
) -> dict[str, tuple[ViewRecord, ...]]:
    """Validate the records of one pool and group them by problem id.

    Args:
        records: records of a view file.
        problems: the problems the views belong to. If given, every record must name one of
            them and its text must differ from that problem's question (an unchanged
            question is the original view, not an alternative).
        pool: keep only records with this ``pool`` label (``None`` keeps all records).
        require_validated: reject the pool if any kept record lacks ``"validated": true``.
        views_per_problem: required number of views of every covered problem; all problems
            must then also use the same view names (held-out evaluation families).
        require_all_problems: require at least one view for every problem in ``problems``.

    Returns:
        ``{problem_id: records}`` with each problem's records sorted by view name, which
        makes later uniform sampling independent of the line order of the file.
    """
    kept = [r for r in records if pool is None or r.pool == pool]
    if not kept:
        raise ValueError("the view file has no records" + (f" for pool {pool!r}" if pool else ""))
    if require_validated:
        unvalidated = [r for r in kept if not r.validated]
        if unvalidated:
            first = unvalidated[0]
            raise ValueError(
                f"{len(unvalidated)} view(s) are not marked as validated, e.g. "
                f"{first.problem_id!r}/{first.view!r}; set require_validated to false to "
                "use views without human validation"
            )
    questions = None if problems is None else {p.id: p.question for p in problems}
    grouped: dict[str, dict[str, ViewRecord]] = {}
    for record in kept:
        if questions is not None:
            if record.problem_id not in questions:
                raise ValueError(f"view for unknown problem {record.problem_id!r}")
            if record.question.strip() == questions[record.problem_id].strip():
                raise ValueError(
                    f"view {record.view!r} of {record.problem_id!r} repeats the original question"
                )
        views = grouped.setdefault(record.problem_id, {})
        if record.view in views:
            raise ValueError(f"duplicate view {record.view!r} for problem {record.problem_id!r}")
        views[record.view] = record
    result = {
        problem_id: tuple(views[name] for name in sorted(views))
        for problem_id, views in grouped.items()
    }
    if require_all_problems and questions is not None:
        missing = [problem_id for problem_id in questions if problem_id not in result]
        if missing:
            raise ValueError(f"{len(missing)} problem(s) have no view, e.g. {missing[0]!r}")
    if views_per_problem is not None:
        names = {tuple(r.view for r in views) for views in result.values()}
        if len(names) != 1 or len(next(iter(names))) != views_per_problem:
            raise ValueError(
                f"every problem needs the same {views_per_problem} view names; found "
                f"{len(names)} different name set(s)"
            )
    return result


def rewritten_views(records: Sequence[ViewRecord], template: str) -> tuple[RewrittenView, ...]:
    """Views of one problem: each rewritten question inside the common prompt template."""
    return tuple(
        RewrittenView(record.view, template.format(question=record.question)) for record in records
    )


def training_view_pool(
    problems: Sequence[Problem],
    path: str | Path,
    template: str,
    *,
    pool: str | None = None,
    require_validated: bool = True,
) -> list[tuple[RewrittenView, ...]]:
    """Alternative teacher views of every training problem from a view file (E10).

    Returns one tuple per problem, in the order of ``problems``. Records of other problems
    are ignored (one file may serve several splits). A problem without a view in the file
    gets an empty tuple; the identity fallback of Appendix A.1 then uses its original prompt
    for both views (the trainer logs the fraction of such responses).
    """
    identifiers = {problem.id for problem in problems}
    records = [record for record in load_view_file(path) if record.problem_id in identifiers]
    grouped = group_views(records, problems, pool=pool, require_validated=require_validated)
    return [rewritten_views(grouped.get(problem.id, ()), template) for problem in problems]


def paraphrase_family(
    problems: Sequence[Problem],
    records: Sequence[ViewRecord],
    template: str,
    *,
    num_views: int = 4,
    pool: str | None = None,
    require_validated: bool = True,
    name: str = PARAPHRASE_FAMILY,
) -> ViewFamily:
    """Held-out family of verified paraphrases for evaluation (E14, E15).

    ``records`` may cover more problems than ``problems`` (one file for several benchmarks);
    every evaluated problem must have exactly ``num_views`` views with the same names. The
    family is fixed by the file before any model is evaluated.
    """
    identifiers = {problem.id for problem in problems}
    grouped = group_views(
        [record for record in records if record.problem_id in identifiers],
        problems,
        pool=pool,
        require_validated=require_validated,
        views_per_problem=num_views,
        require_all_problems=True,
    )
    per_question = {
        problem_id: rewritten_views(views, template) for problem_id, views in grouped.items()
    }
    return ViewFamily(name, per_question=per_question)
