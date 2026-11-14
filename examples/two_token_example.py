#!/usr/bin/env python
"""Two-token example: the same normalised target with different consensus mass.

Appendix D.1 of the paper and supplementary experiment E18. With two teacher views ``p1`` and
``p2`` over a two-token vocabulary, ``c(a) = sqrt(p1(a) p2(a))``, ``Z = sum_a c(a)`` and
``r = c / Z``. Two views that agree on (0.5, 0.5) and two confident views that select
opposite tokens both give ``r = (0.5, 0.5)``; the retained mass ``Z`` tells them apart
(1 versus 0.199) and sets the supervision strength of VD-OPD.

    python examples/two_token_example.py
    python examples/two_token_example.py --csv out/E18_two_token_example.csv
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from vdopd.eval.tables import write_csv
from vdopd.theory import two_token_example

COLUMNS = ("case", "p1", "p2", "r", "z")


def vector(values: Sequence[float]) -> str:
    """``(0.99, 0.01)``: shortest decimal form of each entry."""
    return "(" + ", ".join(f"{value:g}" for value in values) + ")"


def table() -> list[dict[str, str]]:
    """The example as text cells; ``Z`` is rounded to three decimals."""
    return [
        {
            "case": row["case"],
            "p1": vector(row["p1"]),
            "p2": vector(row["p2"]),
            "r": vector(row["r"]),
            "z": f"{round(row['z'], 3):g}",
        }
        for row in two_token_example()
    ]


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--csv", default=None, help="write the table to this CSV file")
    args = parser.parse_args(argv)

    rows = table()
    header = ("Case", "Teacher view p1", "Teacher view p2", "Normalized consensus r", "Mass Z")
    widths = [
        max(len(h), *(len(row[c]) for row in rows)) for h, c in zip(header, COLUMNS, strict=True)
    ]
    print("  ".join(h.ljust(w) for h, w in zip(header, widths, strict=True)))
    for row in rows:
        print("  ".join(row[c].ljust(w) for c, w in zip(COLUMNS, widths, strict=True)))

    if args.csv:
        write_csv(args.csv, COLUMNS, rows)


if __name__ == "__main__":
    main()
