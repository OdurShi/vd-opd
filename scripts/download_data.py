#!/usr/bin/env python
"""Fetch the datasets of the paper into ``data/`` (same layout as the shipped files).

    python scripts/download_data.py                          # fetch whatever is missing
    python scripts/download_data.py --only amc23 aime24 minerva
    python scripts/download_data.py --force                  # fetch everything again
    python scripts/download_data.py --data-dir /path/to/data

Datasets come from the Hugging Face Hub at pinned revisions (``pip install -e ".[data]"``).
Every fetched file is checked against the SHA-256 recorded in ``data/manifest.json``;
``--update-manifest`` records new checksums instead. Sources, licences and sizes are listed
in ``data/README.md``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vdopd.data import paths
from vdopd.data.benchmarks import BENCHMARKS, benchmark_file, fetch_benchmark_records
from vdopd.data.math_train import (
    MATH_SUBJECTS,
    MATH_TRAIN_FILE,
    MATH_TRAIN_PATH,
    MATH_TRAIN_REVISION,
    fetch_math_train_records,
)

# Licence as declared by the dataset's authors or on its dataset card, and whether the file
# is distributed with this repository.
LICENSES: dict[str, tuple[str, bool]] = {
    "math_train": ("MIT", True),
    "math500": ("MIT (subset of the MATH test split)", True),
    "amc23": ("none stated on the dataset card", False),
    "aime24": ("none stated on the dataset card", False),
    "aime25": ("Apache-2.0", True),
    "minerva": ("none stated on the dataset card", False),
    "olympiadbench": ("Apache-2.0 (dataset card), MIT (upstream repository)", True),
    "gsm8k": ("MIT", True),
}
MANIFEST = "manifest.json"


def describe(name: str) -> dict:
    """Source description of a dataset (without checksum)."""
    license_text, shipped = LICENSES[name]
    if name == "math_train":
        entry = {
            "file": MATH_TRAIN_FILE,
            "source": MATH_TRAIN_PATH,
            "config": ", ".join(MATH_SUBJECTS),
            "split": "train",
            "revision": MATH_TRAIN_REVISION,
        }
    else:
        spec = BENCHMARKS[name]
        entry = {
            "file": benchmark_file(name),
            "source": spec.path,
            "config": spec.config or "default",
            "split": spec.split,
            "revision": spec.revision,
        }
    return dict(entry, license=license_text, shipped=shipped)


def load_manifest(directory: Path) -> dict:
    path = directory / MANIFEST
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {"files": {}}


def save_manifest(directory: Path, manifest: dict) -> None:
    manifest["files"] = dict(sorted(manifest["files"].items()))
    (directory / MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def fetch(name: str) -> list[dict]:
    return fetch_math_train_records() if name == "math_train" else fetch_benchmark_records(name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--only", nargs="+", choices=sorted(LICENSES), help="datasets to fetch")
    parser.add_argument("--force", action="store_true", help="fetch files that already exist")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--update-manifest", action="store_true", help="record new checksums")
    args = parser.parse_args()

    if args.data_dir is not None:
        paths.set_data_dir(args.data_dir)
    directory = paths.data_dir()
    directory.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(directory)

    for name in args.only or sorted(LICENSES):
        entry = describe(name)
        target = directory / entry["file"]
        if target.is_file() and not args.force:
            print(f"{name}: present ({entry['file']})")
            continue
        records = fetch(name)
        paths.write_jsonl(target, records)
        digest = paths.sha256(target)
        recorded = manifest["files"].get(entry["file"], {}).get("sha256")
        if args.update_manifest or recorded is None:
            manifest["files"][entry["file"]] = dict(entry, num_examples=len(records), sha256=digest)
            status = "recorded"
        else:
            status = "checksum ok" if digest == recorded else "CHECKSUM MISMATCH"
        print(f"{name}: {len(records)} examples -> {entry['file']} ({status})")
    save_manifest(directory, manifest)


if __name__ == "__main__":
    main()
