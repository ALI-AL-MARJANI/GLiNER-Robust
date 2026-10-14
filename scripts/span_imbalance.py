"""Count positive vs negative (span, entity type) pairs that a span-based GLiNER scores.

For a sentence of ``n`` words, a span model with maximum width ``K`` scores every
span of width 1..K (``sum_w max(0, n - w + 1)`` spans) against every prompted
entity type ``C``. A pair is positive when that exact span carries that type
in the gold annotation. Gold entities wider than ``K`` words cannot be
predicted at all and are reported separately.

This is what the training loss sees when all of a dataset's types are
prompted (``negatives=1.0``, no type sampling); it is a property of the data
and of ``K``, independent of any model weights.

Usage:
    python -m scripts.span_imbalance --max_width 12
"""

from __future__ import annotations

import json
import argparse
from typing import Dict
from pathlib import Path

from experiments.env import collect_env
from experiments.data import ZERO_SHOT_BENCHMARKS, load


def n_spans(n_words: int, max_width: int) -> int:
    return sum(max(0, n_words - w + 1) for w in range(1, max_width + 1))


def imbalance(name: str, split: str, max_width: int) -> Dict[str, float]:
    examples, labels = load(name, split)
    spans = sum(n_spans(len(ex["tokenized_text"]), max_width) for ex in examples)
    pairs = spans * len(labels)
    entities = [e for ex in examples for e in ex["ner"]]
    reachable = sum(1 for s, e, _ in entities if e - s + 1 <= max_width)
    return {
        "dataset": name,
        "split": split,
        "n_sentences": len(examples),
        "n_types": len(labels),
        "n_entities": len(entities),
        "n_entities_wider_than_max_width": len(entities) - reachable,
        "n_candidate_pairs": pairs,
        "n_positive_pairs": reachable,
        "negatives_per_positive": (pairs - reachable) / reachable if reachable else float("inf"),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--max_width", type=int, default=12)
    p.add_argument("--output", default="results/span_imbalance.json")
    a = p.parse_args()

    rows = [imbalance("conll2003", "train", a.max_width)]
    rows += [imbalance(name, "test", a.max_width) for name in ["conll2003", *ZERO_SHOT_BENCHMARKS]]
    print(f"{'dataset':22s} {'split':6s} {'types':>5s} {'entities':>9s} {'pairs':>12s} {'neg:pos':>9s}")
    for r in rows:
        print(
            f"{r['dataset']:22s} {r['split']:6s} {r['n_types']:5d} {r['n_entities']:9d} "
            f"{r['n_candidate_pairs']:12d} {r['negatives_per_positive']:8.0f}:1"
        )
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"max_width": a.max_width, "rows": rows, "env": collect_env()}, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
