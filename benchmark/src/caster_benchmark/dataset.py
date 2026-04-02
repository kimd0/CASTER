"""Dataset helpers for released CASTER files on Hugging Face."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError, HfHubHTTPError

from .constants import HF_DATASET_REPO, HF_DATASET_REVISION, HF_REPO_TYPE, SPLITS


@dataclass(frozen=True)
class SegmentRecord:
    match_id: int
    seg_index: int
    clip_path: str
    speech: str
    speech_tag: str
    time: str
    events: list[dict]

    @property
    def match_name(self) -> str:
        return f"match_{self.match_id:03d}"

    @property
    def clip_filename(self) -> str:
        return f"{self.clip_path}.mp4"


def parse_match_ids(raw_match_ids: str | None, split_name: str) -> list[int]:
    if raw_match_ids:
        values = []
        for token in raw_match_ids.split(","):
            token = token.strip()
            if not token:
                continue
            values.append(int(token))
        return sorted(set(values))
    if split_name not in SPLITS:
        valid = ", ".join(sorted(SPLITS))
        raise ValueError(f"Unknown split '{split_name}'. Valid options: {valid}")
    return list(SPLITS[split_name])


class CasterDataset:
    """Released benchmark loader backed by Hugging Face cache downloads."""

    def __init__(
        self,
        repo_id: str = HF_DATASET_REPO,
        revision: str = HF_DATASET_REVISION,
        local_root: str | Path | None = None,
        cache_dir: str | Path | None = None,
    ) -> None:
        self.repo_id = repo_id
        self.revision = revision
        self.local_root = Path(local_root).expanduser().resolve() if local_root else None
        self.cache_dir = Path(cache_dir).expanduser().resolve() if cache_dir else None

    def _resolve_path(self, relative_path: str) -> Path:
        if self.local_root is not None:
            path = self.local_root / relative_path
            if not path.exists():
                raise FileNotFoundError(f"Missing local dataset file: {path}")
            return path
        return Path(
            hf_hub_download(
                repo_id=self.repo_id,
                repo_type=HF_REPO_TYPE,
                revision=self.revision,
                filename=relative_path,
                cache_dir=str(self.cache_dir) if self.cache_dir else None,
            )
        )

    def resolve_optional_path(self, relative_path: str) -> Path | None:
        try:
            return self._resolve_path(relative_path)
        except (FileNotFoundError, EntryNotFoundError, HfHubHTTPError):
            return None

    def load_context(self, match_id: int) -> list[SegmentRecord]:
        path = self._resolve_path(f"match_{match_id:03d}/context.json")
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        if isinstance(payload, dict):
            rows = payload.get("records", [])
        elif isinstance(payload, list):
            rows = payload
        else:
            raise ValueError(f"Unsupported context payload type for match {match_id}: {type(payload)!r}")

        records: list[SegmentRecord] = []
        for row in rows:
            seg_index = int(row["seg_index"])
            records.append(
                SegmentRecord(
                    match_id=match_id,
                    seg_index=seg_index,
                    clip_path=str(
                        row.get(
                            "clip_path",
                            f"match_{match_id:03d}/clip/match_{match_id:03d}_{seg_index:03d}",
                        )
                    ),
                    speech=str(row.get("speech", "")).strip(),
                    speech_tag=str(row.get("speech_tag", "UNTAGGED")).strip() or "UNTAGGED",
                    time=str(row.get("time", "")).strip(),
                    events=list(row.get("events", [])),
                )
            )

        records.sort(key=lambda item: item.seg_index)
        return records

    def load_metadata(self, match_id: int) -> dict:
        path = self._resolve_path(f"match_{match_id:03d}/metadata.json")
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def ensure_clip(self, record: SegmentRecord) -> Path | None:
        return self.resolve_optional_path(record.clip_filename)

    def iter_segments(self, match_ids: Iterable[int]) -> Iterable[SegmentRecord]:
        for match_id in match_ids:
            yield from self.load_context(match_id)
