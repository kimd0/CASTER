"""Video frame extraction helpers."""

from __future__ import annotations

import base64
from pathlib import Path

import cv2


def extract_frames_base64(
    video_path: str | Path,
    *,
    fps: float,
    fixed_frames: int | None = None,
    target_frames_per_second: int | None = None,
    max_frames: int | None = None,
    resize_width: int = 640,
    resize_height: int = 360,
) -> list[str]:
    video_path = Path(video_path)
    if not video_path.exists():
        return []

    capture = cv2.VideoCapture(str(video_path))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        capture.release()
        return []

    frame_indices = _choose_frame_indices(
        total_frames=total_frames,
        fps=fps,
        fixed_frames=fixed_frames,
        target_frames_per_second=target_frames_per_second,
        max_frames=max_frames,
    )

    frames: list[str] = []
    for index in frame_indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        if not ok:
            continue
        resized = cv2.resize(frame, (resize_width, resize_height))
        ok, encoded = cv2.imencode(".jpg", resized)
        if not ok:
            continue
        frames.append(base64.b64encode(encoded).decode("utf-8"))

    capture.release()
    return frames


def _choose_frame_indices(
    *,
    total_frames: int,
    fps: float,
    fixed_frames: int | None,
    target_frames_per_second: int | None,
    max_frames: int | None,
) -> list[int]:
    if fixed_frames is not None:
        count = max(1, fixed_frames)
    else:
        duration_seconds = total_frames / fps
        count = int(duration_seconds * max(1, target_frames_per_second or 1))
        count = max(1, count)
        if max_frames is not None:
            count = min(count, max_frames)

    step = max(total_frames // count, 1)
    return [min(index * step, total_frames - 1) for index in range(count)]
