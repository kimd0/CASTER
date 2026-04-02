"""JSONL and path helpers."""

from __future__ import annotations

import json
import re
from pathlib import Path


def ensure_dir(path: str | Path) -> Path:
    output = Path(path)
    output.mkdir(parents=True, exist_ok=True)
    return output


def slugify_model_name(model_name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", model_name.strip().lower())
    return slug.strip("-")


def read_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    rows: list[dict] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def append_jsonl(path: str | Path, row: dict) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def sort_jsonl(path: str | Path, key: str = "seg_index") -> None:
    path = Path(path)
    if not path.exists():
        return
    rows = read_jsonl(path)
    rows.sort(key=lambda item: item.get(key, 0))
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_processed_seg_indices(path: str | Path) -> set[int]:
    processed: set[int] = set()
    for row in read_jsonl(path):
        try:
            processed.add(int(row["seg_index"]))
        except (KeyError, TypeError, ValueError):
            continue
    return processed


def collect_match_prediction_rows(predictions_dir: str | Path) -> dict[tuple[int, int], dict]:
    predictions_dir = Path(predictions_dir)
    rows: dict[tuple[int, int], dict] = {}
    for path in sorted(predictions_dir.glob("match_*.jsonl")):
        match_name = path.stem
        match_id = int(match_name.split("_")[1])
        for row in read_jsonl(path):
            try:
                seg_index = int(row["seg_index"])
            except (KeyError, TypeError, ValueError):
                continue
            rows[(match_id, seg_index)] = row
    return rows


def write_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
