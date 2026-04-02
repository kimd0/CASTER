"""Legacy-compatible Task 1 evaluation."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

from .dataset import CasterDataset, parse_match_ids
from .io_utils import collect_match_prediction_rows, write_json

FRAME_PATTERN = re.compile(r"(\d+)~(\d+)")
COORD_PATTERN = re.compile(r"\[(\d+),(\d+)\](?:->\[(\d+),(\d+)\])?")
UNIT_PATTERN = re.compile(r"([A-Za-z0-9_]+)\*(\d+)\s+\[(\d+),(\d+)\](?:->\[(\d+),(\d+)\])?")


@dataclass
class ParsedEvent:
    frames: tuple[int, int] | None
    viewport: dict | None
    units: dict[str, list[dict]]


def evaluate(
    *,
    dataset: CasterDataset,
    predictions_dir: str | Path,
    split: str,
    match_ids_arg: str | None,
    summary_path: str | Path | None,
) -> dict:
    match_ids = parse_match_ids(match_ids_arg, split)
    predictions = collect_match_prediction_rows(predictions_dir)

    total_segments = 0
    valid_format_segments = 0
    sum_tiou = 0.0
    sum_vp_l2 = 0.0
    sum_unit_f1 = 0.0
    sum_count_mae = 0.0
    sum_unit_l2 = 0.0

    for match_id in match_ids:
        for record in dataset.load_context(match_id):
            total_segments += 1
            clip_range = parse_frames(record.time)
            clip_start = clip_range[0] if clip_range else 0
            gt_events = [parse_event(item, clip_start=clip_start) for item in record.events]

            prediction_row = predictions.get((match_id, record.seg_index))
            raw_pred_events = _get_prediction_events(prediction_row)
            if raw_pred_events is None:
                sum_vp_l2 += 800.0
                continue

            try:
                pred_events = [parse_event(item, clip_start=clip_start) for item in raw_pred_events]
            except Exception:
                sum_vp_l2 += 800.0
                continue

            valid_format_segments += 1
            scores = evaluate_segment(gt_events, pred_events)
            sum_tiou += scores["tiou"]
            sum_vp_l2 += scores["vp_l2"]
            sum_unit_f1 += scores["f1"]
            sum_count_mae += scores["mae"]
            sum_unit_l2 += scores["unit_l2"]

    if total_segments == 0:
        raise RuntimeError("No reference segments were found for the requested split.")

    summary = {
        "split": split,
        "match_ids": match_ids,
        "segments": total_segments,
        "formatAcc": round((valid_format_segments / total_segments) * 100.0, 4),
        "unitF1": round(sum_unit_f1 / total_segments, 6),
        "countMae": round(sum_count_mae / total_segments, 6),
        "vpL2": round(sum_vp_l2 / total_segments, 6),
        "unitL2": round(sum_unit_l2 / total_segments, 6),
        "temporalIoU": round(sum_tiou / total_segments, 6),
    }

    if summary_path:
        write_json(summary_path, summary)
    return summary


def _get_prediction_events(row: dict | None) -> list[dict] | None:
    if not row:
        return None
    if isinstance(row.get("generated_events"), list):
        return row["generated_events"]
    if isinstance(row.get("events"), list):
        return row["events"]
    return None


def parse_frames(frame_str: str) -> tuple[int, int] | None:
    match = FRAME_PATTERN.fullmatch(str(frame_str).strip())
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def parse_viewport(viewport_str: str) -> dict | None:
    match = COORD_PATTERN.search(str(viewport_str).strip())
    if not match:
        return None
    start_x, start_y = int(match.group(1)), int(match.group(2))
    end_x = int(match.group(3)) if match.group(3) else start_x
    end_y = int(match.group(4)) if match.group(4) else start_y
    return {"start": (start_x, start_y), "end": (end_x, end_y)}


def parse_unit(unit_str: str) -> dict | None:
    match = UNIT_PATTERN.search(str(unit_str).strip())
    if not match:
        return None
    start_x, start_y = int(match.group(3)), int(match.group(4))
    end_x = int(match.group(5)) if match.group(5) else start_x
    end_y = int(match.group(6)) if match.group(6) else start_y
    return {
        "name": match.group(1),
        "count": int(match.group(2)),
        "start": (start_x, start_y),
        "end": (end_x, end_y),
    }


def parse_event(event_dict: dict, *, clip_start: int) -> ParsedEvent:
    parsed_units = {"[PLAYER_1]": [], "[PLAYER_2]": []}
    for player_key in parsed_units:
        for item in event_dict.get("units", {}).get(player_key, []):
            parsed = parse_unit(item)
            if parsed:
                parsed_units[player_key].append(parsed)

    frames = parse_frames(event_dict.get("frames", ""))
    if frames is not None:
        frames = (frames[0] - clip_start, frames[1] - clip_start)

    return ParsedEvent(
        frames=frames,
        viewport=parse_viewport(event_dict.get("viewport", "")),
        units=parsed_units,
    )


def calc_tiou(frames_gt: tuple[int, int] | None, frames_pred: tuple[int, int] | None) -> float:
    if not frames_gt or not frames_pred:
        return 0.0
    start_gt, end_gt = frames_gt
    start_pred, end_pred = frames_pred
    intersection = max(0, min(end_gt, end_pred) - max(start_gt, start_pred))
    union = max(end_gt, end_pred) - min(start_gt, start_pred)
    if union == 0:
        return 1.0 if start_gt == start_pred else 0.0
    return intersection / union


def get_center(point_dict: dict | None) -> tuple[float, float]:
    if not point_dict:
        return (0.0, 0.0)
    start_x, start_y = point_dict["start"]
    end_x, end_y = point_dict["end"]
    return ((start_x + end_x) / 2.0, (start_y + end_y) / 2.0)


def calc_l2_distance(point_a: tuple[float, float], point_b: tuple[float, float]) -> float:
    return math.sqrt((point_a[0] - point_b[0]) ** 2 + (point_a[1] - point_b[1]) ** 2)


def evaluate_segment(gt_events: list[ParsedEvent], pred_events: list[ParsedEvent]) -> dict:
    total_tiou = 0.0
    total_viewport_l2 = 0.0
    tp = 0
    fp = 0
    fn = 0
    count_errors: list[float] = []
    unit_l2_distances: list[float] = []

    for gt_event in gt_events:
        best_pred = None
        best_iou = 0.0
        for pred_event in pred_events:
            iou = calc_tiou(gt_event.frames, pred_event.frames)
            if iou > best_iou:
                best_iou = iou
                best_pred = pred_event

        total_tiou += best_iou

        if best_pred and gt_event.viewport and best_pred.viewport:
            total_viewport_l2 += calc_l2_distance(get_center(gt_event.viewport), get_center(best_pred.viewport))
        else:
            total_viewport_l2 += 800.0

        gt_units_flat = [{"player": player, **unit} for player in gt_event.units for unit in gt_event.units[player]]
        pred_units_flat = []
        if best_pred:
            pred_units_flat = [{"player": player, **unit} for player in best_pred.units for unit in best_pred.units[player]]

        matched_pred_indices: set[int] = set()
        for gt_unit in gt_units_flat:
            matched = False
            for index, pred_unit in enumerate(pred_units_flat):
                if index in matched_pred_indices:
                    continue
                if gt_unit["player"] == pred_unit["player"] and gt_unit["name"] == pred_unit["name"]:
                    tp += 1
                    count_errors.append(abs(gt_unit["count"] - pred_unit["count"]))
                    unit_l2_distances.append(calc_l2_distance(get_center(gt_unit), get_center(pred_unit)))
                    matched_pred_indices.add(index)
                    matched = True
                    break
            if not matched:
                fn += 1

        fp += len(pred_units_flat) - len(matched_pred_indices)

    num_gt = len(gt_events) if gt_events else 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0

    return {
        "tiou": total_tiou / num_gt,
        "vp_l2": total_viewport_l2 / num_gt,
        "f1": f1_score,
        "mae": sum(count_errors) / len(count_errors) if count_errors else 0.0,
        "unit_l2": sum(unit_l2_distances) / len(unit_l2_distances) if unit_l2_distances else 0.0,
    }
