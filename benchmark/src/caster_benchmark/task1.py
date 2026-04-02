"""Task 1 generation runner."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from .constants import DEFAULT_TASK1_FRAME_COUNT, REPLAY_FPS
from .dataset import CasterDataset, SegmentRecord
from .io_utils import append_jsonl, ensure_dir, load_processed_seg_indices, sort_jsonl
from .openrouter import OpenRouterPool, request_json_completion
from .prompts import build_task1_messages, parse_task1_events
from .video import extract_frames_base64


@dataclass(frozen=True)
class Task1Job:
    record: SegmentRecord
    output_path: Path
    slot: int


def build_jobs(
    *,
    dataset: CasterDataset,
    match_ids: list[int],
    output_dir: Path,
    overwrite: bool,
    limit: int | None,
) -> list[Task1Job]:
    jobs: list[Task1Job] = []
    slot = 0

    for match_id in match_ids:
        output_path = output_dir / f"match_{match_id:03d}.jsonl"
        processed = set() if overwrite else load_processed_seg_indices(output_path)
        for record in dataset.load_context(match_id):
            if record.seg_index in processed:
                continue
            if not record.events:
                continue
            jobs.append(Task1Job(record=record, output_path=output_path, slot=slot))
            slot += 1
            if limit is not None and len(jobs) >= limit:
                return jobs
    return jobs


def run(
    *,
    dataset: CasterDataset,
    match_ids: list[int],
    output_dir: str | Path,
    model: str,
    api_pool: OpenRouterPool,
    workers: int,
    overwrite: bool = False,
    limit: int | None = None,
) -> dict:
    output_root = ensure_dir(output_dir)
    jobs = build_jobs(
        dataset=dataset,
        match_ids=match_ids,
        output_dir=output_root,
        overwrite=overwrite,
        limit=limit,
    )

    stats = {"queued": len(jobs), "written": 0, "skipped": 0, "errors": 0}
    if not jobs:
        return stats

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        future_map = {
            executor.submit(_process_job, dataset, api_pool, model, job): job
            for job in jobs
        }
        for future in as_completed(future_map):
            job = future_map[future]
            try:
                outcome = future.result()
            except Exception as exc:
                print(
                    f"[task1] match_{job.record.match_id:03d} seg_{job.record.seg_index:03d} failed: {exc}"
                )
                stats["errors"] += 1
                continue
            if outcome["status"] == "ok":
                append_jsonl(outcome["output_path"], outcome["row"])
                stats["written"] += 1
            elif outcome["status"] == "skip":
                stats["skipped"] += 1
            else:
                stats["errors"] += 1

    for match_id in match_ids:
        sort_jsonl(output_root / f"match_{match_id:03d}.jsonl")

    return stats


def _process_job(dataset: CasterDataset, api_pool: OpenRouterPool, model: str, job: Task1Job) -> dict:
    clip_path = dataset.ensure_clip(job.record)
    if clip_path is None:
        return {"status": "skip", "reason": "missing_clip"}

    frames = extract_frames_base64(
        clip_path,
        fps=REPLAY_FPS,
        fixed_frames=DEFAULT_TASK1_FRAME_COUNT,
    )
    if not frames:
        return {"status": "skip", "reason": "frame_extraction_failed"}

    raw_text = request_json_completion(
        client=api_pool.client_for_slot(job.slot),
        model=model,
        messages=build_task1_messages(job.record, frames, REPLAY_FPS),
    )
    generated_events = parse_task1_events(raw_text)
    row = {
        "match_id": job.record.match_name,
        "seg_index": job.record.seg_index,
        "time": job.record.time,
        "model": model,
        "generated_events": generated_events,
    }
    return {"status": "ok", "output_path": job.output_path, "row": row}
