"""Location of the dataset files shipped with the repository (``data/``).

Loaders read the files under the data directory when they exist and fall back to the
Hugging Face Hub otherwise. The directory is resolved in this order:

1. an explicit override (``--data-dir`` on the command line, ``data.data_dir`` in a config,
   or :func:`set_data_dir`);
2. the ``VDOPD_DATA_DIR`` environment variable;
3. ``data/`` at the root of a source checkout (next to ``src/``).
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable
from pathlib import Path

ENV_VAR = "VDOPD_DATA_DIR"
_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_override: Path | None = None


def set_data_dir(path: str | Path | None) -> None:
    """Set (or with ``None`` clear) the explicit data-directory override."""
    global _override
    _override = None if path is None else Path(path)


def data_dir() -> Path:
    """The data directory (it need not exist; missing files trigger the Hub fallback)."""
    if _override is not None:
        return _override
    if os.environ.get(ENV_VAR):
        return Path(os.environ[ENV_VAR])
    return _REPOSITORY_ROOT / "data"


def local_file(relative: str) -> Path | None:
    """Path of a shipped data file, or ``None`` if it is absent."""
    path = data_dir() / relative
    return path if path.is_file() else None


def read_jsonl(path: str | Path) -> list[dict]:
    """Read a JSON-lines file into a list of dictionaries."""
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: str | Path, records: Iterable[dict]) -> None:
    """Write dictionaries as JSON lines (UTF-8, one record per line, stable key order)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_lines(path: str | Path) -> list[str]:
    """Non-empty lines of a text file (index lists)."""
    with open(path, encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def write_lines(path: str | Path, lines: Iterable[str]) -> None:
    """Write one entry per line."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")


def sha256(path: str | Path) -> str:
    """SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
