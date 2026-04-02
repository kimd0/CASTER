<div align="center">

# CASTER

**CASTER: A Multimodal Dataset and Benchmark for Observation-Grounded StarCraft Commentary Generation**

[![Dataset: Hugging Face](https://img.shields.io/badge/Dataset-Hugging%20Face-f6c945?style=flat-square)](https://huggingface.co/datasets/kimd00/CASTER)
[![Presentation Page](https://img.shields.io/badge/Page-GitHub%20Pages-24292f?style=flat-square)](https://kimd0.github.io/CASTER/)
[![Benchmark Suite](https://img.shields.io/badge/Benchmark-Python-3776ab?style=flat-square)](benchmark/)

<p>
  <a href="https://huggingface.co/datasets/kimd00/CASTER"><strong>Dataset</strong></a>
  &middot;
  <a href="https://kimd0.github.io/CASTER/"><strong>Presentation Page</strong></a>
  &middot;
  <a href="./benchmark/"><strong>Benchmark Suite</strong></a>
  &middot;
  <a href="./benchmark/README.md"><strong>Benchmark Docs</strong></a>
</p>

</div>

This repository accompanies the public CASTER release for academic use. The benchmark package lives in [`benchmark/`](benchmark/), and the presentation page source lives in [`docs/`](docs/).

## Quick Start

```bash
cd benchmark
pip install -e .
```

Run examples:

```bash
caster-benchmark task1 run --preset gpt4o --output-dir outputs/task1/gpt4o
caster-benchmark task1 eval --predictions outputs/task1/gpt4o

caster-benchmark task2 run --preset gpt4o --conditioning multimodal --output-dir outputs/task2/gpt4o_multimodal
caster-benchmark task2 eval --predictions outputs/task2/gpt4o_multimodal
```

Generation commands use OpenRouter credentials through environment variables:

```bash
OPENROUTER_API_KEY=...
```

or

```bash
OPENROUTER_API_KEYS=key1,key2,key3
```

See [`benchmark/README.md`](benchmark/README.md) for benchmark details and evaluation notes.
