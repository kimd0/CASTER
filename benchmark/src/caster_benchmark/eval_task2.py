"""Task 2 evaluation for the released paper-level reference metrics."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import nltk
import numpy as np
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge_score import rouge_scorer

from .dataset import CasterDataset, parse_match_ids
from .io_utils import collect_match_prediction_rows, write_json


def evaluate(
    *,
    dataset: CasterDataset,
    predictions_dir: str | Path,
    split: str,
    match_ids_arg: str | None,
    summary_path: str | Path | None,
    skip_bertscore: bool = False,
) -> dict:
    _ensure_nltk()
    match_ids = parse_match_ids(match_ids_arg, split)
    predictions = collect_match_prediction_rows(predictions_dir)

    overall_references: list[str] = []
    overall_candidates: list[str] = []
    refs_by_tag: dict[str, list[str]] = defaultdict(list)
    cands_by_tag: dict[str, list[str]] = defaultdict(list)

    for match_id in match_ids:
        for record in dataset.load_context(match_id):
            prediction = predictions.get((match_id, record.seg_index))
            if not prediction:
                continue
            candidate = str(prediction.get("speech", "")).strip()
            reference = record.speech.strip()
            if not candidate or not reference:
                continue

            overall_references.append(reference)
            overall_candidates.append(candidate)
            refs_by_tag[record.speech_tag].append(reference)
            cands_by_tag[record.speech_tag].append(candidate)

    if not overall_candidates:
        raise RuntimeError("No matched Task 2 predictions were found.")

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    smoother = SmoothingFunction()

    summary = {
        "split": split,
        "match_ids": match_ids,
        "samples": len(overall_candidates),
        "overall": calculate_metrics(
            overall_references,
            overall_candidates,
            scorer,
            smoother,
            skip_bertscore=skip_bertscore,
        ),
        "by_tag": {},
    }

    for tag in sorted(cands_by_tag):
        summary["by_tag"][tag] = calculate_metrics(
            refs_by_tag[tag],
            cands_by_tag[tag],
            scorer,
            smoother,
            skip_bertscore=skip_bertscore,
        )
        summary["by_tag"][tag]["samples"] = len(cands_by_tag[tag])

    if summary_path:
        write_json(summary_path, summary)
    return summary


def calculate_metrics(
    references: list[str],
    candidates: list[str],
    scorer: rouge_scorer.RougeScorer,
    smoother: SmoothingFunction,
    *,
    skip_bertscore: bool,
) -> dict:
    bleu_scores: list[float] = []
    rouge_scores: list[float] = []

    for reference, candidate in zip(references, candidates):
        ref_tokens = nltk.word_tokenize(reference.lower())
        cand_tokens = nltk.word_tokenize(candidate.lower())
        bleu_scores.append(sentence_bleu([ref_tokens], cand_tokens, smoothing_function=smoother.method1) * 100.0)
        rouge_scores.append(scorer.score(reference, candidate)["rougeL"].fmeasure * 100.0)

    bertscore_value: float | None = None
    if not skip_bertscore:
        try:
            from bert_score import score as bert_score_fn
        except Exception as exc:  # pragma: no cover - depends on local torch/runtime setup
            raise RuntimeError(
                "BERTScore dependencies could not be loaded. "
                "Check the local torch installation or rerun with `--skip-bertscore`."
            ) from exc
        _, _, bert_f1 = bert_score_fn(candidates, references, lang="en", verbose=False)
        bertscore_value = round(float(bert_f1.mean().item() * 100.0), 4)

    return {
        "BLEU-4": round(float(np.mean(bleu_scores)), 4),
        "ROUGE-L": round(float(np.mean(rouge_scores)), 4),
        "BERTScore": bertscore_value,
    }


def _ensure_nltk() -> None:
    nltk.download("wordnet", quiet=True)
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
