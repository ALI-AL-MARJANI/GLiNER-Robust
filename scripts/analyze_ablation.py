"""Aggregate ablation runs into a summary table with seed spread and paired bootstrap tests.

Reads ``<ablation_dir>/runs/*.json`` (from ``scripts/train_ablation.py``) and writes
``<ablation_dir>/summary.json`` and ``<ablation_dir>/summary.md``.

Statistics
----------
- mean ± sample std over seeds of entity-level micro-F1, per dataset;
- zero-shot average = unweighted mean over the zero-shot datasets, computed per seed;
- paired bootstrap (10,000 resamples of test sentences, seeds kept) of
  ``variant - baseline`` on the zero-shot average, with Holm-Bonferroni
  correction across variants; per-dataset bootstrap deltas are reported uncorrected;
- rare-type F1: macro-F1 over entity types that make up < 5% of the gold
  entities of their test set, averaged over seeds.

Usage:
    python -m scripts.analyze_ablation --ablation_dir results/ablation
"""

from __future__ import annotations

import json
import argparse
from typing import Any, Dict, List
from pathlib import Path
from collections import defaultdict

import numpy as np

from experiments.metrics import holm, mean_std, paired_bootstrap, paired_bootstrap_average

RARE_SHARE = 0.05


def load_runs(run_dir: Path) -> List[Dict[str, Any]]:
    runs = [json.loads(p.read_text()) for p in sorted(run_dir.glob("*.json"))]
    runs = [r for r in runs if not r.get("smoke")]
    if not runs:
        raise SystemExit(f"No run files in {run_dir}")
    return runs


def rare_types(result: Dict[str, Any]) -> List[str]:
    total = sum(v["support"] for v in result["per_type"].values())
    return [t for t, v in result["per_type"].items() if total and v["support"] / total < RARE_SHARE]


def summarize(runs: List[Dict[str, Any]], mode: str, baseline: str, n_resamples: int) -> Dict[str, Any]:
    by_variant: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in runs:
        by_variant[r["variant"]].append(r)
    for v in by_variant.values():
        v.sort(key=lambda r: r["seed"])

    cfg = runs[0]["config"]["data"]
    zero_shot = cfg["eval_zero_shot"]
    in_domain = cfg["eval_in_domain"]
    seeds = {v: [r["seed"] for r in rs] for v, rs in by_variant.items()}

    table: Dict[str, Any] = {}
    for variant, rs in by_variant.items():
        row: Dict[str, Any] = {"seeds": seeds[variant], "datasets": {}}
        for name in in_domain + zero_shot:
            row["datasets"][name] = mean_std([r["results"][mode][name]["f1"] for r in rs])
        row["zero_shot_avg"] = mean_std([np.mean([r["results"][mode][n]["f1"] for n in zero_shot]) for r in rs])
        rare_f1 = []
        for r in rs:
            vals = [
                r["results"][mode][n]["per_type"][t]["f1"] for n in zero_shot for t in rare_types(r["results"][mode][n])
            ]
            rare_f1.append(float(np.mean(vals)) if vals else float("nan"))
        row["zero_shot_rare_type_macro_f1"] = mean_std(rare_f1)
        row["tuned_thresholds"] = [r["thresholds"]["tuned"] for r in rs]
        row["best_steps"] = [r["selection"]["best_step"] for r in rs]
        table[variant] = row

    tests: Dict[str, Any] = {}
    if baseline in by_variant:
        base = by_variant[baseline]

        def counts(rs, name):
            return [np.asarray(r["results"][mode][name]["sentence_counts"]) for r in rs]

        for variant, rs in by_variant.items():
            if variant == baseline:
                continue
            per_ds = {
                name: paired_bootstrap(counts(base, name), counts(rs, name), n_resamples=n_resamples)
                for name in in_domain + zero_shot
            }
            avg = paired_bootstrap_average(
                {n: counts(base, n) for n in zero_shot},
                {n: counts(rs, n) for n in zero_shot},
                n_resamples=n_resamples,
            )
            tests[variant] = {"zero_shot_avg": avg, "per_dataset": per_ds}
        adjusted = holm({v: t["zero_shot_avg"]["p_value"] for v, t in tests.items()})
        for v, p in adjusted.items():
            tests[v]["zero_shot_avg"]["p_holm"] = p
    return {
        "mode": mode,
        "baseline": baseline,
        "zero_shot": zero_shot,
        "in_domain": in_domain,
        "table": table,
        "tests_vs_baseline": tests,
    }


def _fmt(ms: Dict[str, float]) -> str:
    return f"{100 * ms['mean']:.2f} ± {100 * ms['std']:.2f}"


def to_markdown(summary: Dict[str, Any], order: List[str]) -> str:
    zs, ind = summary["zero_shot"], summary["in_domain"]
    base = summary["baseline"]
    head = ["variant", "n seeds", *ind, *zs, "zero-shot avg", "Δ avg vs " + base + " [95% CI]", "p (Holm)"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for v in order:
        if v not in summary["table"]:
            continue
        row = summary["table"][v]
        cells = [v, str(len(row["seeds"]))]
        cells += [_fmt(row["datasets"][n]) for n in ind + zs]
        cells.append(_fmt(row["zero_shot_avg"]))
        t = summary["tests_vs_baseline"].get(v)
        if t:
            a = t["zero_shot_avg"]
            cells.append(f"{100 * a['delta_f1']:+.2f} [{100 * a['ci95_low']:+.2f}, {100 * a['ci95_high']:+.2f}]")
            cells.append(f"{a['p_holm']:.3f}")
        else:
            cells += ["—", "—"]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ablation_dir", default="results/ablation")
    p.add_argument("--baseline", default="bce")
    p.add_argument("--n_resamples", type=int, default=10_000)
    a = p.parse_args()

    root = Path(a.ablation_dir)
    runs = load_runs(root / "runs")
    order = list(runs[0]["config"]["variants"])
    out = {mode: summarize(runs, mode, a.baseline, a.n_resamples) for mode in ("tuned", "fixed")}
    (root / "summary.json").write_text(json.dumps(out, indent=2))
    md = [
        "<!-- generated by scripts/analyze_ablation.py; do not edit -->",
        "## Threshold tuned on CoNLL-2003 dev (primary)",
        "",
        to_markdown(out["tuned"], order),
        "",
        "## Fixed threshold 0.5",
        "",
        to_markdown(out["fixed"], order),
        "",
        "F1 in %, mean ± sample std over seeds. Δ: paired bootstrap over test sentences "
        f"({a.n_resamples} resamples, stratified by dataset), p-values Holm-corrected across variants.",
    ]
    (root / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
