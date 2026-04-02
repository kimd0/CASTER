"""OpenRouter-compatible chat completion helpers."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass

from openai import OpenAI

from .constants import DEFAULT_MAX_RETRIES, OPENROUTER_BASE_URL


def load_openrouter_api_keys() -> list[str]:
    keys: list[str] = []

    single = os.getenv("OPENROUTER_API_KEY", "").strip()
    if single:
        keys.append(single)

    multi = os.getenv("OPENROUTER_API_KEYS", "")
    for token in multi.split(","):
        token = token.strip()
        if token:
            keys.append(token)

    deduped: list[str] = []
    seen: set[str] = set()
    for key in keys:
        if key in seen:
            continue
        deduped.append(key)
        seen.add(key)

    if not deduped:
        raise RuntimeError(
            "Missing OpenRouter credentials. Set OPENROUTER_API_KEY or OPENROUTER_API_KEYS."
        )
    return deduped


@dataclass
class OpenRouterPool:
    api_keys: list[str]
    base_url: str = OPENROUTER_BASE_URL

    def __post_init__(self) -> None:
        self._clients: dict[str, OpenAI] = {}

    def client_for_slot(self, slot: int) -> OpenAI:
        key = self.api_keys[slot % len(self.api_keys)]
        if key not in self._clients:
            self._clients[key] = OpenAI(base_url=self.base_url, api_key=key)
        return self._clients[key]


def request_json_completion(
    *,
    client: OpenAI,
    model: str,
    messages: list[dict],
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> str:
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content
            if not content:
                raise ValueError("Model returned an empty response.")
            return content
        except Exception as exc:  # pragma: no cover
            last_error = exc
            if attempt == max_retries:
                break
            time.sleep(2 * attempt)
    raise RuntimeError(f"Chat completion failed after {max_retries} attempts: {last_error}")


def strip_code_fences(raw_text: str) -> str:
    text = raw_text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:].strip()
    return text.strip()


def parse_json_response(raw_text: str) -> dict | list:
    return json.loads(strip_code_fences(raw_text))
