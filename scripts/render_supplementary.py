#!/usr/bin/env python
"""Render the supplementary experiment tables (E01-E18) as Markdown.

The numbers live in ``results/supplementary/*/*.csv`` and their layout in
``results/supplementary/index.yaml``. This script writes ``docs/supplementary_results.md``
and refreshes the tables that ``README.md`` embeds between ``<!-- supplementary:E06 -->``
markers.

    python scripts/render_supplementary.py            # rewrite the document
    python scripts/render_supplementary.py --check    # verify it is current and consistent
    python scripts/render_supplementary.py --show E07 # print one table

``--check`` exits with status 1 if the document differs from what the CSV files produce or
if tables that report the same quantity disagree (see ``vdopd.supplementary``).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from vdopd.supplementary import check_consistency, embed_tables, load_index, render_markdown

REPOSITORY = Path(__file__).resolve().parents[1]
DOCUMENT = REPOSITORY / "docs" / "supplementary_results.md"
README = REPOSITORY / "README.md"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=None, help="directory with index.yaml")
    parser.add_argument("--output", type=Path, default=DOCUMENT, help="Markdown file to write")
    parser.add_argument("--check", action="store_true", help="verify instead of writing")
    parser.add_argument("--show", metavar="ID", help="print the table of one experiment")
    args = parser.parse_args(argv)

    index = load_index(args.root)
    if args.show:
        experiment = index[args.show.upper()]
        print(f"{experiment.id}. {experiment.title}\n")
        print(experiment.to_markdown())
        return 0

    problems = check_consistency(index)
    for problem in problems:
        print(f"inconsistent: {problem}", file=sys.stderr)

    documents = {args.output: render_markdown(index)}
    if README.is_file():
        documents[README] = embed_tables(README.read_text(encoding="utf-8"), index)
    if args.check:
        stale = [
            path
            for path, text in documents.items()
            if not path.is_file() or path.read_text(encoding="utf-8") != text
        ]
        for path in stale:
            print(f"{path} is out of date; run {Path(__file__).name}", file=sys.stderr)
        return 1 if stale or problems else 0

    for path, text in documents.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.is_file() or path.read_text(encoding="utf-8") != text:
            path.write_text(text, encoding="utf-8")
            print(f"wrote {path}")
    print(f"{len(index)} experiments, {len(problems)} inconsistencies")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
