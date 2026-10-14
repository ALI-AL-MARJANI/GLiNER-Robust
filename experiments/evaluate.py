"""Run a GLiNER model (PyTorch or ONNX-backed) over a dataset and score it."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from gliner.evaluation.evaluator import BaseNEREvaluator

from .data import Example, load
from .metrics import prf, bootstrap_ci, f1_from_counts, per_type_counts, sentence_counts


def _normalize(evaluator: BaseNEREvaluator, gold: List[Any], pred: List[Any]):
    g = [(s, e, lab) for lab, (s, e) in evaluator.get_ground_truth(gold)]
    p = [(s, e, lab) for lab, (s, e) in evaluator.get_predictions(pred)]
    return g, p


@torch.no_grad()
def predict(
    model: Any,
    examples: List[Example],
    labels: List[str],
    threshold: float = 0.5,
    batch_size: int = 8,
    flat_ner: bool = True,
) -> Dict[str, Any]:
    """Predict every sentence and return per-sentence counts.

    Uses the same collate / run / decode path as ``GLiNER.evaluate`` so that
    the summed counts reproduce its micro-F1 exactly, but keeps the
    per-sentence breakdown needed for bootstrap tests.
    """
    model.eval()
    collator = model.create_collator()
    loader = torch.utils.data.DataLoader(
        examples,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: model.collate_batch(batch, labels, collator),
    )
    evaluator = BaseNEREvaluator([], [])
    counts: List[List[int]] = []
    by_type: Dict[str, List[int]] = {}
    t0 = time.perf_counter()
    for batch in loader:
        output = model.run_batch(batch, threshold=threshold, move_to_device=True)
        decoded = model.decode_batch(output, batch, threshold=threshold, flat_ner=flat_ner, multi_label=False)
        for gold, pred in zip(batch["entities"], decoded):
            g, p = _normalize(evaluator, gold, pred)
            counts.append(list(sentence_counts(g, p)))
            for typ, c in per_type_counts(g, p).items():
                acc = by_type.setdefault(typ, [0, 0, 0])
                for i in range(3):
                    acc[i] += c[i]
    elapsed = time.perf_counter() - t0
    arr = np.asarray(counts, dtype=np.int64)
    return {
        "counts": arr,
        "per_type": {t: {**prf(*c), "support": c[2]} for t, c in sorted(by_type.items())},
        "seconds": elapsed,
    }


def evaluate_dataset(
    model: Any,
    name: str,
    split: str = "test",
    threshold: float = 0.5,
    batch_size: int = 8,
    max_examples: Optional[int] = None,
) -> Dict[str, Any]:
    """Score ``model`` on one dataset split; returns a JSON-serialisable dict."""
    examples, labels = load(name, split)
    if max_examples:
        examples = examples[:max_examples]
    out = predict(model, examples, labels, threshold=threshold, batch_size=batch_size)
    counts = out["counts"]
    tp, n_pred, n_gold = counts.sum(axis=0).tolist()
    return {
        "dataset": name,
        "split": split,
        "n_sentences": len(examples),
        "labels": labels,
        "threshold": threshold,
        **prf(tp, n_pred, n_gold),
        "tp": tp,
        "n_pred": n_pred,
        "n_gold": n_gold,
        "f1_bootstrap": bootstrap_ci(counts),
        "per_type": out["per_type"],
        "seconds": out["seconds"],
        "sentence_counts": counts.tolist(),
    }


__all__ = ["evaluate_dataset", "f1_from_counts", "predict"]
