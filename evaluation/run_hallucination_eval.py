"""
evaluation/run_hallucination_eval.py
Evaluates VLM hallucination using POPE (binary) and CHAIR (caption) metrics.
"""

import json
import re
import os
from dataclasses import dataclass, field
from typing import Any


# ── POPE Evaluator ─────────────────────────────────────────────────────────────

@dataclass
class POPEResult:
    accuracy:  float = 0.0
    precision: float = 0.0
    recall:    float = 0.0
    f1:        float = 0.0
    yes_ratio: float = 0.0
    n_samples: int   = 0


class POPEEvaluator:
    """
    POPE (Polling-based Object Probing Evaluation).

    Measures object hallucination via binary yes/no questions:
        "Is there a [object] in the image?"

    Reference: Li et al., "Evaluating Object Hallucination in LVLMs" (EMNLP 2023)
    """

    def __init__(self, label_file: str):
        """
        Args:
            label_file: Path to JSONL with {'label': 'yes'/'no', ...} per line.
        """
        self.labels = []
        with open(label_file) as f:
            for line in f:
                self.labels.append(json.loads(line.strip()))

    @staticmethod
    def parse_answer(response: str) -> str:
        """Extract yes/no from model response."""
        resp = response.strip().lower()
        if resp.startswith("yes"):
            return "yes"
        if resp.startswith("no"):
            return "no"
        if "yes" in resp:
            return "yes"
        return "no"

    def evaluate(self, predictions: list[str]) -> POPEResult:
        """
        Args:
            predictions: List of raw model response strings.

        Returns:
            POPEResult with accuracy, precision, recall, F1.
        """
        assert len(predictions) == len(self.labels), \
            f"Length mismatch: {len(predictions)} preds vs {len(self.labels)} labels"

        tp = fp = tn = fn = yes_count = 0
        for pred, item in zip(predictions, self.labels):
            answer = self.parse_answer(pred)
            label  = item["label"].lower()
            if answer == "yes":
                yes_count += 1
            if label == "yes" and answer == "yes":
                tp += 1
            elif label == "no" and answer == "yes":
                fp += 1
            elif label == "no" and answer == "no":
                tn += 1
            else:
                fn += 1

        n          = len(predictions)
        accuracy   = (tp + tn) / n
        precision  = tp / (tp + fp) if (tp + fp) else 0.0
        recall     = tp / (tp + fn) if (tp + fn) else 0.0
        f1         = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        yes_ratio  = yes_count / n

        return POPEResult(
            accuracy=round(accuracy * 100, 2),
            precision=round(precision * 100, 2),
            recall=round(recall * 100, 2),
            f1=round(f1 * 100, 2),
            yes_ratio=round(yes_ratio * 100, 2),
            n_samples=n,
        )


# ── CHAIR Evaluator ────────────────────────────────────────────────────────────

@dataclass
class CHAIRResult:
    chair_s: float = 0.0   # sentence-level hallucination rate
    chair_i: float = 0.0   # instance-level hallucination rate
    n_captions:   int = 0
    n_hallucinated_sentences: int = 0
    n_hallucinated_objects:   int = 0
    n_total_objects:          int = 0


class CHAIREvaluator:
    """
    CHAIR (Caption Hallucination Assessment with Image Relevance).

    Measures what fraction of generated caption objects are NOT in the image.

    CHAIR_S = hallucinated_sentences / total_sentences
    CHAIR_I = hallucinated_objects   / total_objects

    Reference: Rohrbach et al. (2018)
    """

    def __init__(self, coco_objects_file: str | None = None):
        """
        Args:
            coco_objects_file: Path to JSON mapping image_id → list of GT object names.
                               If None, runs in mock mode.
        """
        self.gt_objects: dict[str, list[str]] = {}
        if coco_objects_file and os.path.exists(coco_objects_file):
            with open(coco_objects_file) as f:
                self.gt_objects = json.load(f)

    @staticmethod
    def _extract_nouns(text: str) -> list[str]:
        """Naive noun extraction via stop-word removal (replace with SpaCy in production)."""
        stop = {"a", "an", "the", "is", "are", "there", "in", "on", "at",
                "of", "and", "or", "with", "has", "have", "this", "that"}
        words = re.findall(r"\b[a-z]+\b", text.lower())
        return [w for w in words if w not in stop and len(w) > 2]

    def evaluate(self,
                 image_ids:   list[str],
                 captions:    list[str]) -> CHAIRResult:
        """
        Args:
            image_ids: List of image identifiers (must match gt_objects keys).
            captions:  List of model-generated captions.

        Returns:
            CHAIRResult with CHAIR_S and CHAIR_I scores.
        """
        hall_sentences = 0
        hall_objects   = 0
        total_objects  = 0
        n              = len(captions)

        for img_id, cap in zip(image_ids, captions):
            gt = set(self.gt_objects.get(str(img_id), []))
            nouns = self._extract_nouns(cap)
            total_objects += len(nouns)
            sentence_hallucinated = False
            for noun in nouns:
                if gt and noun not in gt:
                    hall_objects += 1
                    sentence_hallucinated = True
            if sentence_hallucinated:
                hall_sentences += 1

        chair_s = hall_sentences / n               if n else 0.0
        chair_i = hall_objects   / total_objects   if total_objects else 0.0

        return CHAIRResult(
            chair_s=round(chair_s * 100, 2),
            chair_i=round(chair_i * 100, 2),
            n_captions=n,
            n_hallucinated_sentences=hall_sentences,
            n_hallucinated_objects=hall_objects,
            n_total_objects=total_objects,
        )


# ── CLI Entry Point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse, sys

    parser = argparse.ArgumentParser(description="Hallucination Evaluation")
    parser.add_argument("--mode", choices=["pope", "chair", "both"], default="both")
    parser.add_argument("--pope-labels",   default="data/pope_labels.jsonl")
    parser.add_argument("--pope-preds",    default="data/pope_predictions.jsonl")
    parser.add_argument("--chair-gt",      default="data/coco_gt_objects.json")
    parser.add_argument("--chair-captions",default="data/model_captions.jsonl")
    args = parser.parse_args()

    if args.mode in ("pope", "both"):
        if not os.path.exists(args.pope_labels):
            print(f"[POPE] Label file not found: {args.pope_labels}. Skipping.")
        else:
            preds = [json.loads(l)["response"]
                     for l in open(args.pope_preds)]
            result = POPEEvaluator(args.pope_labels).evaluate(preds)
            print(f"\n── POPE Results ──────────────────────")
            print(f"  Accuracy   : {result.accuracy}%")
            print(f"  F1 Score   : {result.f1}%")
            print(f"  Yes Ratio  : {result.yes_ratio}%")
            print(f"  Samples    : {result.n_samples}")

    if args.mode in ("chair", "both"):
        if not os.path.exists(args.chair_captions):
            print(f"[CHAIR] Captions file not found: {args.chair_captions}. Skipping.")
        else:
            data   = [json.loads(l) for l in open(args.chair_captions)]
            ids    = [d["image_id"] for d in data]
            caps   = [d["caption"]  for d in data]
            result = CHAIREvaluator(args.chair_gt).evaluate(ids, caps)
            print(f"\n── CHAIR Results ─────────────────────")
            print(f"  CHAIR_S (sentence): {result.chair_s}%")
            print(f"  CHAIR_I (instance): {result.chair_i}%")
            print(f"  Hallucinated objs : {result.n_hallucinated_objects}/{result.n_total_objects}")
