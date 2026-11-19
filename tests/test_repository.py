"""Repository hygiene: README links resolve and the paper figures are in place."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parents[1]
LINK = re.compile(r"\]\(([^)#\s]+)(?:#[^)]*)?\)")


@pytest.mark.parametrize(
    "document", ["README.md", "data/README.md", "assets/paper_figures/README.md"]
)
def test_relative_links_resolve(document):
    path = REPOSITORY / document
    if not path.is_file():
        pytest.skip(f"{document} is not present")
    for target in LINK.findall(path.read_text(encoding="utf-8")):
        if target.startswith(("http://", "https://", "mailto:")):
            continue
        assert (path.parent / target).exists(), f"{document}: broken link {target}"


def test_readme_commands_reference_existing_files():
    text = (REPOSITORY / "README.md").read_text(encoding="utf-8")
    for match in set(re.findall(r"(?:scripts|configs|examples)/[\w./-]+\.(?:py|sh|yaml)", text)):
        assert (REPOSITORY / match).is_file(), f"README mentions missing file {match}"


def test_paper_figures_are_present_and_small():
    folder = REPOSITORY / "assets" / "paper_figures"
    if not folder.is_dir():
        pytest.skip("paper figures are not present")
    figures = sorted(folder.glob("fig*.png"))
    assert [f.name for f in figures] == [f"fig{n:02d}.png" for n in range(1, 7)]
    assert all(f.stat().st_size < 1_500_000 for f in figures)
    listed = (folder / "README.md").read_text(encoding="utf-8")
    assert all(f.name in listed for f in figures)


def test_readme_images_exist_and_wording_stays_anonymous():
    text = (REPOSITORY / "README.md").read_text(encoding="utf-8")
    for source in re.findall(r'<img src="([^"]+)"', text):
        if not source.startswith(("http://", "https://")):
            assert (REPOSITORY / source).is_file(), f"README: missing image {source}"
    lowered = text.lower()
    assert not [w for w in ("openreview", "rebuttal", "reviewer", "submission") if w in lowered]


def test_readme_highlights_quote_the_reported_tables():
    """Every decimal number of the Highlights section is a value of a shipped table."""
    from vdopd.supplementary import load_index, parse_number

    text = (REPOSITORY / "README.md").read_text(encoding="utf-8")
    section = text[text.index("## Highlights") : text.index("## Main results")]
    reported = set()
    for experiment in load_index():
        for row in experiment.rows:
            for cell in row.values():
                value = parse_number(cell)
                if value is not None:
                    reported |= {f"{abs(value):.1f}", f"{abs(value):.2f}"}
    quoted = re.findall(r"\d+\.\d+", section)
    assert quoted and not [number for number in quoted if number not in reported]
