"""Task 2 LLM-judge evaluation for SC/CN/TC metrics only."""

from __future__ import annotations

import json
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from openai import OpenAI

from .constants import OPENROUTER_BASE_URL
from .dataset import CasterDataset, SegmentRecord, parse_match_ids
from .io_utils import collect_match_prediction_rows, write_json
from .openrouter import parse_json_response, request_json_completion

DEFAULT_JUDGE_MODEL = "gpt-5-mini"
DEFAULT_JUDGE_WORKERS = 2
DEFAULT_JUDGE_VOTES = 1
ALLOWED_ERROR_TYPES = {
    "hallucinated_entity",
    "wrong_player_attribution",
    "unsupported_strategy_claim",
    "unnatural_phrasing",
    "tag_mismatch",
    "insufficient_information",
    "none",
}

JUDGE_SYSTEM_PROMPT = """You are an expert evaluator for StarCraft: Brood War esports commentary.

You will receive:
1. A discourse tag.
2. A synchronized structured observation from context.json.
3. A candidate commentary utterance.

Evaluate the candidate on exactly three metrics from 1 to 5:

1. strategic_correctness
- Does the utterance capture the strategic meaning or likely implications of the situation?
- Reward strategically sound interpretation grounded in the provided observation.
- If the evidence is insufficient for a strong strategic claim, lower the score.

2. caster_naturalness
- Does it sound like fluent, engaging live esports commentary?
- Reward natural, energetic, and immersive casting style.
- Penalize flat, awkward, overly mechanical, or list-like phrasing.

3. tag_consistency
- Does the utterance fulfill the discourse role indicated by the tag?

Important rules:
- Base strategic judgments on the provided observation.
- Do not output or infer any factual_grounding field.
- Return JSON only with exactly these keys:
  strategic_correctness
  caster_naturalness
  tag_consistency
  short_rationale
  error_types
"""


@dataclass(frozen=True)
class JudgeJob:
    match_id: int
    seg_index: int
    speech_tag: str
    reference: str
    candidate: str
    observation: dict


@dataclass
class JudgeClientPool:
    api_keys: list[str]
    base_url: str | None = None
    _clients: dict[int, OpenAI] = field(default_factory=dict, init=False)

    def client_for_slot(self, slot: int) -> OpenAI:
        slot_key = slot % len(self.api_keys)
        if slot_key not in self._clients:
            kwargs = {"api_key": self.api_keys[slot_key]}
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._clients[slot_key] = OpenAI(**kwargs)
        return self._clients[slot_key]


def evaluate(
    *,
    dataset: CasterDataset,
    predictions_dir: str | Path,
    split: str,
    match_ids_arg: str | None,
    model: str,
    workers: int = DEFAULT_JUDGE_WORKERS,
    votes: int = DEFAULT_JUDGE_VOTES,
    summary_path: str | Path | None,
    raw_output_path: str | Path | None,
    base_url: str | None = None,
) -> dict:
    if workers < 1:
        raise ValueError("--workers must be at least 1.")
    if votes < 1:
        raise ValueError("--votes must be at least 1.")

    match_ids = parse_match_ids(match_ids_arg, split)
    predictions = collect_match_prediction_rows(predictions_dir)
    jobs = _build_jobs(dataset, predictions, match_ids)
    if not jobs:
        raise RuntimeError("No matched Task 2 predictions were found.")

    api_keys, resolved_base_url = _resolve_judge_credentials(base_url)
    pool = JudgeClientPool(api_keys=api_keys, base_url=resolved_base_url)

    raw_path = Path(raw_output_path) if raw_output_path else None
    items: list[dict] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_judge_job, pool, slot, model, votes, job): job
            for slot, job in enumerate(jobs)
        }
        for future in as_completed(futures):
            job = futures[future]
            try:
                item = future.result()
            except Exception as exc:  # pragma: no cover - network/runtime failures
                raise RuntimeError(
                    f"Task 2 judge evaluation failed for match_{job.match_id:03d} seg_{job.seg_index:03d}."
                ) from exc
            items.append(item)

    items.sort(key=lambda item: (item["match_id"], item["seg_index"]))
    if raw_path:
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        with raw_path.open("w", encoding="utf-8") as handle:
            for item in items:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    summary = _summarize_items(
        items,
        split=split,
        match_ids=match_ids,
        model=model,
        votes=votes,
        base_url=resolved_base_url,
        raw_output_path=str(raw_path) if raw_path else None,
    )
    if summary_path:
        write_json(summary_path, summary)
    return summary


def _build_jobs(
    dataset: CasterDataset,
    predictions: dict[tuple[int, int], dict],
    match_ids: list[int],
) -> list[JudgeJob]:
    jobs: list[JudgeJob] = []
    for match_id in match_ids:
        for record in dataset.load_context(match_id):
            prediction = predictions.get((match_id, record.seg_index))
            if not prediction:
                continue
            candidate = str(prediction.get("speech", "")).strip()
            reference = record.speech.strip()
            if not candidate or not reference:
                continue
            jobs.append(
                JudgeJob(
                    match_id=match_id,
                    seg_index=record.seg_index,
                    speech_tag=record.speech_tag,
                    reference=reference,
                    candidate=candidate,
                    observation=_build_observation_payload(record),
                )
            )
    return jobs


def _build_observation_payload(record: SegmentRecord) -> dict:
    return {
        "time": record.time,
        "events": record.events,
    }


def _load_env_keys(single_name: str, multi_name: str) -> list[str]:
    values: list[str] = []
    single = os.getenv(single_name, "").strip()
    if single:
        values.append(single)
    for token in os.getenv(multi_name, "").split(","):
        token = token.strip()
        if token:
            values.append(token)
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        deduped.append(value)
        seen.add(value)
    return deduped


def _resolve_judge_credentials(base_url_override: str | None) -> tuple[list[str], str | None]:
    judge_keys = _load_env_keys("JUDGE_API_KEY", "JUDGE_API_KEYS")
    if judge_keys:
        return judge_keys, base_url_override or os.getenv("JUDGE_BASE_URL") or None

    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    if openai_key:
        return [openai_key], base_url_override or None

    openrouter_keys = _load_env_keys("OPENROUTER_API_KEY", "OPENROUTER_API_KEYS")
    if openrouter_keys:
        return openrouter_keys, base_url_override or OPENROUTER_BASE_URL

    raise RuntimeError(
        "Missing judge API credentials. Set JUDGE_API_KEY(S), OPENAI_API_KEY, "
        "or OPENROUTER_API_KEY(S)."
    )


def _judge_job(
    pool: JudgeClientPool,
    slot: int,
    model: str,
    votes: int,
    job: JudgeJob,
) -> dict:
    client = pool.client_for_slot(slot)
    ballots = [_judge_once(client=client, model=model, job=job) for _ in range(votes)]
    strategic_scores = [vote["strategic_correctness"] for vote in ballots]
    naturalness_scores = [vote["caster_naturalness"] for vote in ballots]
    consistency_scores = [vote["tag_consistency"] for vote in ballots]

    error_union: list[str] = []
    seen_errors: set[str] = set()
    for vote in ballots:
        for label in vote["error_types"]:
            if label in seen_errors:
                continue
            seen_errors.add(label)
            error_union.append(label)
    if len(error_union) > 1 and "none" in error_union:
        error_union = [label for label in error_union if label != "none"]

    return {
        "match_id": job.match_id,
        "seg_index": job.seg_index,
        "speech_tag": job.speech_tag,
        "candidate": job.candidate,
        "reference": job.reference,
        "observation": job.observation,
        "votes": ballots,
        "final": {
            "SC_mean": _mean(strategic_scores),
            "CN_mean": _mean(naturalness_scores),
            "TC_mean": _mean(consistency_scores),
            "SC_median": _median_int(strategic_scores),
            "CN_median": _median_int(naturalness_scores),
            "TC_median": _median_int(consistency_scores),
        },
        "error_types_union": error_union or ["none"],
    }


def _judge_once(*, client: OpenAI, model: str, job: JudgeJob) -> dict:
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": _build_judge_user_prompt(job),
        },
    ]

    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            raw = request_json_completion(client=client, model=model, messages=messages)
            payload = parse_json_response(raw)
            return _normalize_judge_payload(payload)
        except Exception as exc:  # pragma: no cover - depends on remote model behavior
            last_error = exc
            if attempt == 3:
                break
    raise RuntimeError(f"Judge model did not return a valid payload: {last_error}")


def _build_judge_user_prompt(job: JudgeJob) -> str:
    return (
        "[Discourse Tag]\n"
        f"{job.speech_tag}\n\n"
        "[Structured Observation]\n"
        f"{json.dumps(job.observation, ensure_ascii=False, indent=2)}\n\n"
        "[Candidate Commentary]\n"
        f"{job.candidate}\n\n"
        "Score exactly these metrics from 1 to 5:\n"
        "- strategic_correctness\n"
        "- caster_naturalness\n"
        "- tag_consistency\n\n"
        "Also return:\n"
        "- short_rationale\n"
        "- error_types\n\n"
        "Allowed error_types:\n"
        "- hallucinated_entity\n"
        "- wrong_player_attribution\n"
        "- unsupported_strategy_claim\n"
        "- unnatural_phrasing\n"
        "- tag_mismatch\n"
        "- insufficient_information\n"
        "- none\n\n"
        "Return JSON only. Do not include any factual_grounding field."
    )


def _normalize_judge_payload(payload: dict | list) -> dict:
    if not isinstance(payload, dict):
        raise ValueError("Judge response payload must be a JSON object.")

    return {
        "strategic_correctness": _coerce_score(payload, "strategic_correctness"),
        "caster_naturalness": _coerce_score(payload, "caster_naturalness"),
        "tag_consistency": _coerce_score(payload, "tag_consistency"),
        "short_rationale": str(payload.get("short_rationale", "")).strip(),
        "error_types": _normalize_error_types(payload.get("error_types")),
    }


def _coerce_score(payload: dict, key: str) -> int:
    value = payload.get(key)
    if isinstance(value, str):
        value = value.strip()
        if not value:
            raise ValueError(f"Missing score for {key}.")
    try:
        score = int(round(float(value)))
    except Exception as exc:
        raise ValueError(f"Invalid score for {key}: {value!r}") from exc
    if score < 1 or score > 5:
        raise ValueError(f"Score for {key} must be in [1, 5], got {score}.")
    return score


def _normalize_error_types(value: object) -> list[str]:
    if not isinstance(value, list):
        return ["none"]
    normalized: list[str] = []
    for item in value:
        label = str(item).strip().lower().replace(" ", "_")
        if label not in ALLOWED_ERROR_TYPES:
            continue
        if label in normalized:
            continue
        normalized.append(label)
    if not normalized:
        return ["none"]
    if len(normalized) > 1 and "none" in normalized:
        normalized.remove("none")
    return normalized


def _summarize_items(
    items: list[dict],
    *,
    split: str,
    match_ids: list[int],
    model: str,
    votes: int,
    base_url: str | None,
    raw_output_path: str | None,
) -> dict:
    by_tag: dict[str, list[dict]] = defaultdict(list)
    for item in items:
        by_tag[item["speech_tag"]].append(item)

    summary = {
        "split": split,
        "match_ids": match_ids,
        "samples": len(items),
        "judge": {
            "model": model,
            "votes": votes,
            "base_url": base_url,
            "metrics": ["SC", "CN", "TC"],
        },
        "overall": _aggregate_group(items),
        "by_tag": {},
    }
    if raw_output_path:
        summary["raw_output_path"] = raw_output_path

    for tag in sorted(by_tag):
        summary["by_tag"][tag] = _aggregate_group(by_tag[tag])
    return summary


def _aggregate_group(items: list[dict]) -> dict:
    if not items:
        return {"samples": 0, "SC": 0.0, "CN": 0.0, "TC": 0.0}

    return {
        "samples": len(items),
        "SC": _mean(item["final"]["SC_mean"] for item in items),
        "CN": _mean(item["final"]["CN_mean"] for item in items),
        "TC": _mean(item["final"]["TC_mean"] for item in items),
    }


def _mean(values) -> float:
    numeric = [float(value) for value in values]
    if not numeric:
        return 0.0
    return round(sum(numeric) / len(numeric), 4)


def _median_int(values: list[int]) -> int:
    ordered = sorted(values)
    if not ordered:
        return 0
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[mid]
    return int(round((ordered[mid - 1] + ordered[mid]) / 2))
