"""Generate every figure in ``results/figures/`` from the JSON files in ``results/``.

A figure is skipped (with a message) when its input file does not exist yet,
so this script can be re-run after each experiment.

Usage:
    python -m scripts.make_figures
"""

from __future__ import annotations

import json
from typing import Any, Dict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE = "#2a78d6"  # categorical slot 1
GRAY = "#8a8985"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e6e5e1"
SURFACE = "#fcfcfb"

RESULTS = Path("results")
FIGURES = RESULTS / "figures"


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _save(fig, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.patch.set_facecolor(SURFACE)
    fig.savefig(FIGURES / name, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {FIGURES / name}")


def ablation_figure(summary: Dict[str, Any]) -> None:
    """Zero-shot average F1 per loss variant: seed mean, ±1 std, and the baseline in gray."""
    tuned = summary["tuned"]
    table = tuned["table"]
    variants = list(table)
    fig, ax = plt.subplots(figsize=(6.4, 0.45 * len(variants) + 1.2))
    _style(ax)
    for i, v in enumerate(variants[::-1]):
        ms = table[v]["zero_shot_avg"]
        color = GRAY if v == tuned["baseline"] else BLUE
        ax.errorbar(
            100 * ms["mean"], i, xerr=100 * ms["std"], fmt="o", ms=7, color=color, ecolor=color, elinewidth=2, capsize=0
        )
    ax.set_yticks(range(len(variants)), variants[::-1], color=INK)
    base = table[tuned["baseline"]]["zero_shot_avg"]["mean"] * 100
    ax.axvline(base, color=GRAY, linewidth=1, linestyle=(0, (3, 3)))
    ax.set_xlabel("Zero-shot average micro-F1 (%), mean ± std over seeds", color=INK_2)
    n = len(table[variants[0]]["seeds"])
    ax.set_title(
        f"Loss ablation, {len(tuned['zero_shot'])} zero-shot benchmarks, {n} seeds", color=INK, fontsize=11, loc="left"
    )
    _save(fig, "ablation_zero_shot.png")


def deploy_figures(path: Path) -> None:
    """Accuracy vs latency (Pareto view) and latency by input length, one file per machine."""
    data = json.loads(path.read_text())
    ok = {k: v for k, v in data["backends"].items() if "latency" in v}
    if not ok:
        return
    tag = path.parent.name
    key = "w64_b1"

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    _style(ax)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    points = [
        (rec["latency"][key]["p50_ms"], 100 * rec["accuracy_avg_f1"], name, rec["size_mb"]) for name, rec in ok.items()
    ]
    xs, ys = [pt[0] for pt in points], [pt[1] for pt in points]
    ax.set_ylim(min(ys) - 6, max(ys) + 8)
    ax.set_xlim(0, max(xs) * 1.6)
    # Labels: labels whose text would overlap horizontally (points closer than `near` on x) are stacked
    # at least `gap` apart and joined to their point by a thin leader line.
    near, gap = 0.45 * max(xs), 0.075 * (max(ys) - min(ys) + 14)  # label text is ~0.4 * max(xs) wide
    placed: list = []
    for x, y, name, size in sorted(points, key=lambda pt: -pt[1]):
        ly = y
        for px, py in placed:
            if abs(px - x) < near and abs(py - ly) < gap:
                ly = py - gap
        placed.append((x, ly))
        ax.scatter(x, y, s=60, color=BLUE, edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.annotate(
            f"{name} ({size:.0f} MB)",
            (x, y),
            xytext=(x + 0.04 * max(xs), ly),
            textcoords="data",
            va="center",
            fontsize=8,
            color=INK_2,
            arrowprops={"arrowstyle": "-", "color": GRID, "linewidth": 0.8},
        )
    ax.set_xlabel("p50 latency, 64 words, batch 1 (ms)", color=INK_2)
    ax.set_ylabel("Average micro-F1 (%)", color=INK_2)
    ax.set_title(
        f"Accuracy vs latency — {data['env']['cpu']}, {data['config']['threads']} threads",
        color=INK,
        fontsize=11,
        loc="left",
    )
    _save(fig, f"deploy_pareto_{tag}.png")

    words = data["config"]["latency"]["words"]
    fig, axes = plt.subplots(1, len(words), figsize=(3.2 * len(words), 0.42 * len(ok) + 1.4), sharey=True)
    names = list(ok)[::-1]
    for ax, w in zip(axes, words):
        _style(ax)
        vals = [ok[n]["latency"][f"w{w}_b1"]["p50_ms"] for n in names]
        ax.barh(range(len(names)), vals, color=BLUE, height=0.6)
        for i, v in enumerate(vals):
            ax.text(v, i, f" {v:.0f}", va="center", fontsize=8, color=INK_2)
        ax.set_title(f"{w} words", color=INK, fontsize=10, loc="left")
        ax.set_xlabel("p50 ms, batch 1", color=INK_2)
    axes[0].set_yticks(range(len(names)), names, color=INK)
    _save(fig, f"deploy_latency_{tag}.png")


def main() -> None:
    summary = RESULTS / "ablation" / "summary.json"
    if summary.exists():
        ablation_figure(json.loads(summary.read_text()))
    else:
        print(f"skip ablation figure: {summary} not found (run scripts/analyze_ablation.py)")
    deploy = sorted((RESULTS / "deploy").glob("*/results.json"))
    if not deploy:
        print("skip deployment figures: no results/deploy/*/results.json")
    for path in deploy:
        deploy_figures(path)


if __name__ == "__main__":
    main()
