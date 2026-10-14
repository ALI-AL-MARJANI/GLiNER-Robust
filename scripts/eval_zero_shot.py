r"""Zero-shot evaluation of a GLiNER checkpoint (no fine-tuning) on the benchmark suite.

Writes ``<output_dir>/<model-slug>.json`` with per-dataset micro-F1, a 95%
bootstrap interval, per-type scores and per-sentence counts.

Usage:
    python -m scripts.eval_zero_shot --model gliner-community/gliner_small-v2.5 \
        --revision f227d3cd637bd4e6757ae143935316d062393341
"""

from __future__ import annotations

import re
import json
import argparse
from pathlib import Path

import numpy as np
import torch

from gliner import GLiNER
from experiments.env import collect_env
from experiments.data import ZERO_SHOT_BENCHMARKS
from experiments.evaluate import evaluate_dataset


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="gliner-community/gliner_small-v2.5")
    p.add_argument("--revision", default="f227d3cd637bd4e6757ae143935316d062393341")
    p.add_argument("--datasets", nargs="+", default=["conll2003", *ZERO_SHOT_BENCHMARKS])
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--output_dir", default="results/zero_shot")
    a = p.parse_args()

    torch.set_num_threads(a.threads)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"loading {a.model} on {device} ...", flush=True)
    model = GLiNER.from_pretrained(a.model, revision=a.revision).to(device).eval()

    results = {}
    for name in a.datasets:
        print(f"evaluating {name} ...", flush=True)
        r = evaluate_dataset(model, name, threshold=a.threshold, batch_size=a.batch_size)
        results[name] = r
        ci = r["f1_bootstrap"]
        print(
            f"{name:22s} F1={100 * r['f1']:6.2f}  95% CI [{100 * ci['ci95_low']:.2f}, {100 * ci['ci95_high']:.2f}]",
            flush=True,
        )
    zs = [n for n in a.datasets if n in ZERO_SHOT_BENCHMARKS]
    avg = float(np.mean([results[n]["f1"] for n in zs])) if zs else None
    if avg is not None:
        print(f"{'zero-shot average':22s} F1={100 * avg:6.2f}  ({len(zs)} datasets)")

    out = {
        "model": a.model,
        "revision": a.revision,
        "threshold": a.threshold,
        "batch_size": a.batch_size,
        "device": device,
        "zero_shot_datasets": zs,
        "zero_shot_avg_f1": avg,
        "results": results,
        "env": collect_env(),
    }
    out_dir = Path(a.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / (re.sub(r"[^A-Za-z0-9.]+", "_", a.model) + ".json")
    path.write_text(json.dumps(out))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
