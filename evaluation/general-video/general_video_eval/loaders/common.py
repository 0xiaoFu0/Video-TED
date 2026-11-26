"""Small local annotation readers; never download datasets during evaluation."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from ..schema import normalize_answer, normalize_choices


def read_rows(path):
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError("Parquet annotations require pyarrow: pip install pyarrow") from exc
        return pq.read_table(path).to_pylist()
    if path.suffix.lower() in {".tsv", ".csv"}:
        with path.open(encoding="utf-8-sig", newline="") as stream:
            return list(csv.DictReader(stream, delimiter="\t" if path.suffix.lower() == ".tsv" else ","))
    if path.suffix.lower() == ".jsonl":
        with path.open(encoding="utf-8") as stream:
            return [json.loads(line) for line in stream if line.strip()]
    with path.open(encoding="utf-8-sig") as stream:
        return json.load(stream)


def find_file(root, candidates, description, override=None):
    root = Path(root)
    if override:
        path = Path(override).expanduser()
        if not path.is_absolute():
            path = root / path
        if not path.is_file():
            raise FileNotFoundError(f"{description}: explicit annotation file does not exist: {path}")
        return path
    for candidate in candidates:
        path = root / candidate
        if path.is_file():
            return path
    raise FileNotFoundError(f"{description} not found in {root}; checked: {', '.join(candidates)}")
