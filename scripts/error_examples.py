"""Dump concrete prediction errors (deterministic: first N test sentences with an error).

For each dataset, writes the sentence, the gold entities, the predicted entities,
and the false positives / false negatives, to ``results/zero_shot/error_examples.json``.

Usage:
    python -m scripts.error_examples --datasets wnut17 crossner_ai mit_movie --n 5
"""

from __future__ import annotations

import json
import argparse
from pathlib import Path

import torch

from gliner import GLiNER
from experiments.env import collect_env
from experiments.data import load
from gliner.evaluation.evaluator import BaseNEREvaluator


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="gliner-community/gliner_small-v2.5")
    p.add_argument("--revision", default="f227d3cd637bd4e6757ae143935316d062393341")
    p.add_argument("--datasets", nargs="+", default=["wnut17", "crossner_ai", "mit_movie"])
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--output", default="results/zero_shot/error_examples.json")
    a = p.parse_args()

    model = GLiNER.from_pretrained(a.model, revision=a.revision).eval()
    evaluator = BaseNEREvaluator([], [])
    out = {"model": a.model, "revision": a.revision, "threshold": a.threshold, "datasets": {}, "env": collect_env()}
    for name in a.datasets:
        examples, labels = load(name, "test")
        collator = model.create_collator()
        found = []
        for i in range(0, len(examples), 8):
            batch = model.collate_batch(examples[i : i + 8], labels, collator)
            with torch.no_grad():
                raw = model.run_batch(batch, threshold=a.threshold)
            decoded = model.decode_batch(raw, batch, threshold=a.threshold, flat_ner=True, multi_label=False)
            for j, (gold, pred) in enumerate(zip(batch["entities"], decoded)):
                g = {(s, e, lab) for lab, (s, e) in evaluator.get_ground_truth(gold)}
                pr = {(s, e, lab) for lab, (s, e) in evaluator.get_predictions(pred)}
                if g != pr:
                    words = examples[i + j]["tokenized_text"]

                    def show(spans, words=words):
                        return [f"{' '.join(words[s : e + 1])} [{lab}]" for s, e, lab in sorted(spans)]

                    found.append(
                        {
                            "index": i + j,
                            "text": " ".join(words),
                            "gold": show(g),
                            "pred": show(pr),
                            "false_positives": show(pr - g),
                            "false_negatives": show(g - pr),
                        }
                    )
                if len(found) >= a.n:
                    break
            if len(found) >= a.n:
                break
        out["datasets"][name] = {"labels": labels, "examples": found}
        print(f"== {name}")
        for ex in found:
            print(f"  {ex['text']}\n    FP: {ex['false_positives']}\n    FN: {ex['false_negatives']}")
    path = Path(a.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
