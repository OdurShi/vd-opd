#!/usr/bin/env python
"""Extract the numbered figures of the paper from its PDF into ``assets/paper_figures/``.

    pip install -e ".[figures]"          # PyMuPDF
    python scripts/extract_paper_figures.py /path/to/paper.pdf

Each figure is rendered from the PDF with a fixed crop rectangle (PDF points, origin at the
top-left corner of the page). The rectangles contain the figure only: no caption, no page
header and no margin line numbers. The PDF is read, never modified or copied.
"""

from __future__ import annotations

import argparse
from pathlib import Path

# figure number -> (1-based page, (x0, y0, x1, y1) in PDF points)
CROP_BOXES: dict[int, tuple[int, tuple[float, float, float, float]]] = {
    1: (2, (127.0, 80.0, 485.5, 220.5)),
    2: (3, (119.5, 81.5, 493.0, 196.0)),
    3: (4, (107.5, 80.5, 504.5, 194.0)),
    4: (7, (107.0, 297.0, 505.0, 415.5)),
    5: (8, (107.5, 251.0, 505.0, 352.5)),
    6: (9, (308.0, 407.5, 502.0, 504.5)),
}

DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "assets" / "paper_figures"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("pdf", type=Path, help="path to the paper PDF (read-only)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT, help="output directory")
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    try:
        import pymupdf as fitz  # PyMuPDF >= 1.24.3
    except ImportError:
        import fitz  # older PyMuPDF releases

    args.out.mkdir(parents=True, exist_ok=True)
    document = fitz.open(args.pdf)
    for number, (page_number, box) in sorted(CROP_BOXES.items()):
        page = document[page_number - 1]
        pixmap = page.get_pixmap(clip=fitz.Rect(*box), dpi=args.dpi)
        target = args.out / f"fig{number:02d}.png"
        pixmap.save(target)
        print(f"Figure {number}: page {page_number} -> {target} ({pixmap.width}x{pixmap.height})")


if __name__ == "__main__":
    main()
