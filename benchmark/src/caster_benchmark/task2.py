"""Task 2 generation runner."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from .constants import DEFAULT_TASK2_MAX_FRAMES, DEFAULT_TASK2_TARGET_FRAMES_PER_SECOND, REPLAY_FPS
from .dataset import CasterDataset, SegmentRecord
from .io_utils import append_jsonl, ensure_dir, load_processed_seg_indices, sort_jsonl
from .openrouter import OpenRouterPool, request_json_completion
from .prompts import build_task2_messages, parse_task2_speech
from .video import extract_frames_base64


@dataclass(frozen=True)
class Task2Job:
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
) -> list[Task2Job]:
    jobs: list[Task2Job] = []
    slot = 0
    for match_id in match_ids:
        output_path = output_dir / f"match_{match_id:03d}.jsonl"
        processed = set() if overwrite else load_processed_seg_indices(output_path)
        for record in dataset.load_context(match_id):
            if record.seg_index in processed:
                continue
            jobs.append(Task2Job(record=record, output_path=output_path, slot=slot))
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
    conditioning: str,
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
            executor.submit(_process_job, dataset, api_pool, model, conditioning, job): job
            for job in jobs
        }
        for future in as_completed(future_map):
            job = future_map[future]
            try:
                outcome = future.result()
            except Exception as exc:
                print(
                    f"[task2:{conditioning}] match_{job.record.match_id:03d} seg_{job.record.seg_index:03d} failed: {exc}"
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


def _process_job(
    dataset: CasterDataset,
    api_pool: OpenRouterPool,
    model: str,
    conditioning: str,
    job: Task2Job,
) -> dict:
    frames: list[str] | None = None
    if conditioning in {"video", "multimodal"}:
        clip_path = dataset.ensure_clip(job.record)
        if clip_path is None:
            return {"status": "skip", "reason": "missing_clip"}
        frames = extract_frames_base64(
            clip_path,
            fps=REPLAY_FPS,
            target_frames_per_second=DEFAULT_TASK2_TARGET_FRAMES_PER_SECOND,
            max_frames=DEFAULT_TASK2_MAX_FRAMES,
        )
        if not frames:
            return {"status": "skip", "reason": "frame_extraction_failed"}

    raw_text = request_json_completion(
        client=api_pool.client_for_slot(job.slot),
        model=model,
        messages=build_task2_messages(job.record, conditioning, frames),
    )
    speech = parse_task2_speech(raw_text)
    row = {
        "match_id": job.record.match_name,
        "seg_index": job.record.seg_index,
        "time": job.record.time,
        "conditioning": conditioning,
        "model": model,
        "speech": speech,
    }
    return {"status": "ok", "output_path": job.output_path, "row": row}
