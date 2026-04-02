"""Prompt builders and response parsers."""

from __future__ import annotations

import json

from .dataset import SegmentRecord
from .openrouter import parse_json_response


TASK1_SYSTEM_PROMPT = """You are an expert StarCraft gameplay vision analyzer.

Your task is to analyze the provided clip frames and generate a structured JSON object with one key: "events".

Instructions:
1. Break the clip into one or more events based on meaningful camera-view changes.
2. Do not create a new event for every tiny viewport jitter.
3. For each event, output:
   - "frames": frame range for that event
   - "viewport": observer viewport anchor or transition
   - "units": visible units grouped under "[PLAYER_1]" and "[PLAYER_2]"
4. Format units strictly as:
   - "Unit_Name*Count [x,y]"
   - "Unit_Name*Count [start_x,start_y]->[end_x,end_y]"
5. Output only valid JSON:
{
  "events": [
    {
      "frames": "...",
      "viewport": "...",
      "units": {
        "[PLAYER_1]": [],
        "[PLAYER_2]": []
      }
    }
  ]
}
"""

TASK2_TEXT_SYSTEM_PROMPT = """You are an expert StarCraft esports commentator.

You must generate a concise, professional commentary utterance from the provided structured observation only.

Instructions:
1. Use only the supplied event log.
2. Refer to players only as "[PLAYER_1]" and "[PLAYER_2]".
3. Keep the output to one to three natural commentary sentences.
4. Output only valid JSON:
{
  "speech": "..."
}
"""

TASK2_VIDEO_SYSTEM_PROMPT = """You are an expert StarCraft esports commentator.

You must generate a concise, professional commentary utterance from the provided clip frames only.

Instructions:
1. Use the clip frames as your only evidence.
2. Refer to players only as "[PLAYER_1]" and "[PLAYER_2]".
3. Keep the output to one to three natural commentary sentences.
4. Output only valid JSON:
{
  "speech": "..."
}
"""

TASK2_MULTIMODAL_SYSTEM_PROMPT = """You are an expert StarCraft esports commentator.

You must generate a concise, professional commentary utterance from both the clip frames and the synchronized structured observation.

Instructions:
1. Use the event log as factual grounding for units and movement.
2. Use the clip frames for scene intensity, positioning, and local visual context.
3. Refer to players only as "[PLAYER_1]" and "[PLAYER_2]".
4. Keep the output to one to three natural commentary sentences.
5. Output only valid JSON:
{
  "speech": "..."
}
"""


def build_task1_messages(record: SegmentRecord, base64_frames: list[str], fps: float) -> list[dict]:
    first_event = record.events[0] if record.events else {}
    user_text = (
        f"Video FPS: {fps}\n"
        f"Overall clip time: {record.time}\n"
        f"Initial event frames: {first_event.get('frames', '')}\n"
        f"Initial event viewport: {first_event.get('viewport', '')}\n\n"
        "Generate the full events JSON object for this clip."
    )
    content = [{"type": "text", "text": user_text}]
    for frame in base64_frames:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{frame}"},
            }
        )
    return [
        {"role": "system", "content": TASK1_SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


def build_task2_messages(
    record: SegmentRecord,
    conditioning: str,
    base64_frames: list[str] | None = None,
) -> list[dict]:
    if conditioning == "text":
        system_prompt = TASK2_TEXT_SYSTEM_PROMPT
        user_text = (
            f"Time range: {record.time}\n"
            f"Structured observation:\n{json.dumps(record.events, indent=2)}\n\n"
            "Generate the commentary JSON object."
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [{"type": "text", "text": user_text}]},
        ]

    if conditioning == "video":
        system_prompt = TASK2_VIDEO_SYSTEM_PROMPT
        user_text = f"Time range: {record.time}\nGenerate the commentary JSON object from the clip frames."
        content = [{"type": "text", "text": user_text}]
        for frame in base64_frames or []:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{frame}"},
                }
            )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]

    system_prompt = TASK2_MULTIMODAL_SYSTEM_PROMPT
    user_text = (
        f"Time range: {record.time}\n"
        f"Structured observation:\n{json.dumps(record.events, indent=2)}\n\n"
        "Generate the commentary JSON object using both the frames and the structured observation."
    )
    content = [{"type": "text", "text": user_text}]
    for frame in base64_frames or []:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{frame}"},
            }
        )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content},
    ]


def parse_task1_events(raw_text: str) -> list[dict]:
    payload = parse_json_response(raw_text)
    if isinstance(payload, dict):
        if isinstance(payload.get("events"), list):
            return payload["events"]
        for value in payload.values():
            if isinstance(value, list):
                return value
    if isinstance(payload, list):
        return payload
    raise ValueError("Task 1 response did not contain an events list.")


def parse_task2_speech(raw_text: str) -> str:
    payload = parse_json_response(raw_text)
    if isinstance(payload, dict) and isinstance(payload.get("speech"), str):
        return payload["speech"].strip()
    raise ValueError("Task 2 response did not contain a speech field.")
