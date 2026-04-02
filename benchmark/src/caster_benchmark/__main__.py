"""CLI entrypoint for the CASTER benchmark package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .constants import (
    HF_DATASET_REPO,
    HF_DATASET_REVISION,
    SPLITS,
    TASK1_MODEL_PRESETS,
    TASK2_CONDITIONINGS,
    TASK2_MODEL_PRESETS,
)
from .dataset import CasterDataset, parse_match_ids


def main() -> None:
    parser = argparse.ArgumentParser(prog="caster-benchmark")
    subparsers = parser.add_subparsers(dest="task", required=True)

    task1_parser = subparsers.add_parser("task1")
    task1_sub = task1_parser.add_subparsers(dest="action", required=True)
    _add_run_parser(task1_sub.add_parser("run"), TASK1_MODEL_PRESETS, include_conditioning=False)
    _add_eval_parser(task1_sub.add_parser("eval"), include_skip_bertscore=False)

    task2_parser = subparsers.add_parser("task2")
    task2_sub = task2_parser.add_subparsers(dest="action", required=True)
    _add_run_parser(task2_sub.add_parser("run"), TASK2_MODEL_PRESETS, include_conditioning=True)
    _add_eval_parser(task2_sub.add_parser("eval"), include_skip_bertscore=True)
    _add_eval_judge_parser(task2_sub.add_parser("eval-judge"))

    args = parser.parse_args()
    if args.task == "task1" and args.action == "run":
        _run_task1_command(args)
        return
    if args.task == "task1" and args.action == "eval":
        _eval_task1_command(args)
        return
    if args.task == "task2" and args.action == "run":
        _run_task2_command(args)
        return
    if args.task == "task2" and args.action == "eval":
        _eval_task2_command(args)
        return
    if args.task == "task2" and args.action == "eval-judge":
        _eval_task2_judge_command(args)
        return
    parser.error("Unhandled command.")


def _add_common_dataset_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repo-id", default=HF_DATASET_REPO)
    parser.add_argument("--revision", default=HF_DATASET_REVISION)
    parser.add_argument("--local-dataset-root")
    parser.add_argument("--cache-dir")
    parser.add_argument("--split", default="legacy_test", choices=sorted(SPLITS))
    parser.add_argument("--match-ids", help="Comma-separated override list, e.g. 2,16,18")


def _add_run_parser(
    parser: argparse.ArgumentParser,
    presets: dict[str, str],
    *,
    include_conditioning: bool,
) -> None:
    _add_common_dataset_flags(parser)
    parser.add_argument("--preset", choices=sorted(presets))
    parser.add_argument("--model")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    if include_conditioning:
        parser.add_argument("--conditioning", choices=TASK2_CONDITIONINGS, required=True)


def _add_eval_parser(parser: argparse.ArgumentParser, *, include_skip_bertscore: bool) -> None:
    _add_common_dataset_flags(parser)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--summary-path")
    if include_skip_bertscore:
        parser.add_argument("--skip-bertscore", action="store_true")


def _add_eval_judge_parser(parser: argparse.ArgumentParser) -> None:
    _add_common_dataset_flags(parser)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument("--summary-path")
    parser.add_argument("--raw-output-path")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--votes", type=int, default=1)
    parser.add_argument("--base-url")


def _build_dataset(args: argparse.Namespace) -> CasterDataset:
    return CasterDataset(
        repo_id=args.repo_id,
        revision=args.revision,
        local_root=args.local_dataset_root,
        cache_dir=args.cache_dir,
    )


def _resolve_model(args: argparse.Namespace, presets: dict[str, str]) -> str:
    if args.model:
        return args.model
    if args.preset:
        return presets[args.preset]
    raise SystemExit("Provide --model or --preset.")


def _run_task1_command(args: argparse.Namespace) -> None:
    from .openrouter import OpenRouterPool, load_openrouter_api_keys
    from .task1 import run as run_task1

    dataset = _build_dataset(args)
    match_ids = parse_match_ids(args.match_ids, args.split)
    model = _resolve_model(args, TASK1_MODEL_PRESETS)
    pool = OpenRouterPool(load_openrouter_api_keys())
    stats = run_task1(
        dataset=dataset,
        match_ids=match_ids,
        output_dir=Path(args.output_dir),
        model=model,
        api_pool=pool,
        workers=args.workers,
        overwrite=args.overwrite,
        limit=args.limit,
    )
    print(json.dumps(stats, indent=2, ensure_ascii=False))


def _run_task2_command(args: argparse.Namespace) -> None:
    from .openrouter import OpenRouterPool, load_openrouter_api_keys
    from .task2 import run as run_task2

    dataset = _build_dataset(args)
    match_ids = parse_match_ids(args.match_ids, args.split)
    model = _resolve_model(args, TASK2_MODEL_PRESETS)
    pool = OpenRouterPool(load_openrouter_api_keys())
    stats = run_task2(
        dataset=dataset,
        match_ids=match_ids,
        output_dir=Path(args.output_dir),
        model=model,
        conditioning=args.conditioning,
        api_pool=pool,
        workers=args.workers,
        overwrite=args.overwrite,
        limit=args.limit,
    )
    print(json.dumps(stats, indent=2, ensure_ascii=False))


def _eval_task1_command(args: argparse.Namespace) -> None:
    from .eval_task1 import evaluate as evaluate_task1

    dataset = _build_dataset(args)
    summary = evaluate_task1(
        dataset=dataset,
        predictions_dir=Path(args.predictions),
        split=args.split,
        match_ids_arg=args.match_ids,
        summary_path=args.summary_path,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def _eval_task2_command(args: argparse.Namespace) -> None:
    from .eval_task2 import evaluate as evaluate_task2

    dataset = _build_dataset(args)
    summary = evaluate_task2(
        dataset=dataset,
        predictions_dir=Path(args.predictions),
        split=args.split,
        match_ids_arg=args.match_ids,
        summary_path=args.summary_path,
        skip_bertscore=bool(getattr(args, "skip_bertscore", False)),
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def _eval_task2_judge_command(args: argparse.Namespace) -> None:
    from .eval_task2_judge import evaluate as evaluate_task2_judge

    dataset = _build_dataset(args)
    summary = evaluate_task2_judge(
        dataset=dataset,
        predictions_dir=Path(args.predictions),
        split=args.split,
        match_ids_arg=args.match_ids,
        model=args.model,
        workers=args.workers,
        votes=args.votes,
        summary_path=args.summary_path,
        raw_output_path=args.raw_output_path,
        base_url=args.base_url,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
