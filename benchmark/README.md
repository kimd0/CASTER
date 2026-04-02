<div align="center">

# CASTER Benchmark Suite

Benchmark runners and evaluation tools for the
**CASTER: A Multimodal Dataset and Benchmark for Observation-Grounded StarCraft Commentary Generation** release.

[![Dataset: Hugging Face](https://img.shields.io/badge/Dataset-Hugging%20Face-f6c945?style=flat-square)](https://huggingface.co/datasets/kimd00/CASTER)
[![Presentation Page](https://img.shields.io/badge/Page-GitHub%20Pages-24292f?style=flat-square)](https://kimd0.github.io/CASTER/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776ab?style=flat-square)](https://www.python.org/)

<p>
  <a href="https://huggingface.co/datasets/kimd00/CASTER"><strong>Dataset</strong></a>
  &middot;
  <a href="https://kimd0.github.io/CASTER/"><strong>Presentation Page</strong></a>
  &middot;
  <a href="../README.md"><strong>Main Repository</strong></a>
</p>

</div>

This package reads released files directly from the public Hugging Face dataset and provides generation and evaluation utilities for:

- `Task 1: Clip-to-Observation`
- `Task 2: Clip+Observation-to-Commentary`

## Installation

From this directory:

```bash
pip install -e .
```

## Credentials

Generation commands use OpenRouter-compatible chat completions through the `openai` client.

Set one of:

```bash
OPENROUTER_API_KEY=...
```

or

```bash
OPENROUTER_API_KEYS=key1,key2,key3
```

`task2 eval-judge` can use either `OPENAI_API_KEY` directly or the same OpenRouter credentials.

## Quick Start

Run Task 1:

```bash
caster-benchmark task1 run --preset gpt4o --output-dir outputs/task1/gpt4o
caster-benchmark task1 eval --predictions outputs/task1/gpt4o
```

Run Task 2:

```bash
caster-benchmark task2 run --preset gpt4o --conditioning multimodal --output-dir outputs/task2/gpt4o_multimodal
caster-benchmark task2 eval --predictions outputs/task2/gpt4o_multimodal
```

Optional Task 2 judge evaluation:

```bash
caster-benchmark task2 eval-judge --predictions outputs/task2/gpt4o_multimodal --model gpt-5-mini --summary-path outputs/task2/gpt4o_multimodal/judge_summary.json
```

## Evaluation

- `task1 eval` reports `formatAcc`, `unitF1`, `countMae`, `vpL2`, `unitL2`, and `temporalIoU`
- `task2 eval` reports `BLEU-4`, `ROUGE-L`, and `BERTScore`, overall and by `speech_tag`
- `task2 eval-judge` reports `SC`, `CN`, and `TC`, overall and by `speech_tag`
- `task2 eval --skip-bertscore` is available for environments where local `torch` / `bert-score` setup is unavailable

## Released Files Used

- `context.json`: core supervision file used by both tasks
- `clip/*.mp4`: used by Task 1 and the `video` / `multimodal` Task 2 settings
- `metadata.json`: optional match-level metadata
- `state.csv` and `viewport.csv`: released with the dataset, but not required by the baseline runners in this package

## Notes

- The default split is `legacy_test`
- `--match-ids 2,16,18` overrides split selection
- `--local-dataset-root PATH` points the suite to a local dataset mirror instead of Hugging Face
- Prediction commands write one JSONL shard per match under the chosen output directory
- Some released `context.json` rows correspond to post-game commentary and may not have a matching released clip; those rows are skipped automatically for video-backed settings
