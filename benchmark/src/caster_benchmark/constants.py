"""Shared benchmark constants."""

from __future__ import annotations

HF_DATASET_REPO = "kimd00/CASTER"
HF_DATASET_REVISION = "main"
HF_REPO_TYPE = "dataset"

REPLAY_FPS = 23.81

LEGACY_TEST_MATCH_IDS = (
    2,
    16,
    18,
    21,
    27,
    36,
    43,
    49,
    53,
    62,
    102,
    104,
    105,
    108,
    114,
    121,
    139,
    161,
    170,
    176,
    182,
    208,
    211,
    235,
)

ALL_PUBLIC_MATCH_IDS = tuple(range(1, 240))

SPLITS = {
    "legacy_test": LEGACY_TEST_MATCH_IDS,
    "all_public": ALL_PUBLIC_MATCH_IDS,
}

TASK1_MODEL_PRESETS = {
    "gemini": "google/gemini-3-flash-preview",
    "gpt4o": "openai/gpt-4o",
    "mimo": "xiaomi/mimo-v2-omni",
    "qwen": "qwen/qwen3.5-397b-a17b",
}

TASK2_MODEL_PRESETS = {
    "gemini": "google/gemini-3-flash-preview",
    "gpt4o": "openai/gpt-4o",
}

TASK2_CONDITIONINGS = ("text", "video", "multimodal")

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

DEFAULT_TASK1_FRAME_COUNT = 5
DEFAULT_TASK2_TARGET_FRAMES_PER_SECOND = 2
DEFAULT_TASK2_MAX_FRAMES = 20
DEFAULT_MAX_RETRIES = 3
