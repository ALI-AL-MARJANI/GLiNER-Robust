"""Entity-level micro-F1 (exact span + type match) and resampling statistics.

Predictions are stored as per-sentence counts so that F1 can be recomputed on
bootstrap resamples of the test sentences without re-running the model.
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Iterable, Sequence

import numpy as np

Span = Tuple[int, int, str]


def sentence_counts(gold: Iterable[Span], pred: Iterable[Span]) -> Tuple[int, int, int]:
    """Return ``(tp, n_pred, n_gold)`` for one sentence (exact match on start, end and type)."""
    gold_set = {(int(s), int(e), str(t)) for s, e, t in gold}
    pred_set = {(int(s), int(e), str(t)) for s, e, t in pred}
    return len(gold_set & pred_set), len(pred_set), len(gold_set)


def per_type_counts(gold: Iterable[Span], pred: Iterable[Span]) -> Dict[str, List[int]]:
    """``{type: [tp, n_pred, n_gold]}`` for one sentence."""
    gold_set = {(int(s), int(e), str(t)) for s, e, t in gold}
    pred_set = {(int(s), int(e), str(t)) for s, e, t in pred}
    out: Dict[str, List[int]] = {}
    for s in gold_set | pred_set:
        c = out.setdefault(s[2], [0, 0, 0])
        c[0] += s in gold_set and s in pred_set
        c[1] += s in pred_set
        c[2] += s in gold_set
    return out


def prf(tp: float, n_pred: float, n_gold: float) -> Dict[str, float]:
    """Micro precision / recall / F1 from summed counts (0 when undefined)."""
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_gold if n_gold else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": p, "recall": r, "f1": f}


def f1_from_counts(counts: np.ndarray) -> float:
    """Micro-F1 from an ``(n_sentences, 3)`` array of ``(tp, n_pred, n_gold)``."""
    tp, n_pred, n_gold = counts.sum(axis=0)
    return prf(tp, n_pred, n_gold)["f1"]


def _f1_rows(counts: np.ndarray) -> np.ndarray:
    """Vectorised micro-F1 over the last two axes ``(..., n_sentences, 3)``."""
    tp, n_pred, n_gold = counts[..., 0], counts[..., 1], counts[..., 2]
    tp, n_pred, n_gold = tp.sum(-1), n_pred.sum(-1), n_gold.sum(-1)
    denom = n_pred + n_gold
    return np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0)


def mean_std(values: Sequence[float]) -> Dict[str, float]:
    """Mean and sample standard deviation (ddof=1; 0 for a single value)."""
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "n": int(arr.size),
    }


def paired_bootstrap(
    counts_a: Sequence[np.ndarray],
    counts_b: Sequence[np.ndarray],
    n_resamples: int = 10_000,
    seed: int = 0,
) -> Dict[str, float]:
    """Paired bootstrap over test sentences for ``mean_F1(B) - mean_F1(A)``.

    ``counts_a`` / ``counts_b`` hold one ``(n_sentences, 3)`` array per seed of
    method A and B, all on the same sentences in the same order. Each resample
    draws sentence indices with replacement, recomputes micro-F1 of every run on
    those sentences, averages over seeds within each method and takes the
    difference. Seed-to-seed variance is therefore kept in the comparison, and
    sentence-level variance is resampled.

    Returns the observed difference, a 95% percentile interval and a two-sided
    p-value (twice the smaller tail mass of the bootstrap distribution around 0).
    """
    a = np.stack(counts_a)  # (seeds_a, n, 3)
    b = np.stack(counts_b)
    if a.shape[1:] != b.shape[1:]:
        raise ValueError("Both methods must be evaluated on the same sentences.")
    n = a.shape[1]
    rng = np.random.default_rng(seed)
    observed = float(_f1_rows(b).mean() - _f1_rows(a).mean())
    diffs = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        diffs[i] = _f1_rows(b[:, idx]).mean() - _f1_rows(a[:, idx]).mean()
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return {
        "delta_f1": observed,
        "ci95_low": float(lo),
        "ci95_high": float(hi),
        "p_value": float(min(p, 1.0)),
        "n_resamples": n_resamples,
    }


def bootstrap_ci(counts: np.ndarray, n_resamples: int = 10_000, seed: int = 0) -> Dict[str, float]:
    """95% percentile interval of micro-F1 for a single run (sentence resampling)."""
    rng = np.random.default_rng(seed)
    n = counts.shape[0]
    f1s = np.concatenate(
        [
            _f1_rows(counts[rng.integers(0, n, size=(min(chunk, n_resamples - start), n))])
            for start, chunk in ((s, 256) for s in range(0, n_resamples, 256))
        ]
    )
    lo, hi = np.percentile(f1s, [2.5, 97.5])
    return {"f1": f1_from_counts(counts), "ci95_low": float(lo), "ci95_high": float(hi)}


def paired_bootstrap_average(
    per_dataset_a: Dict[str, Sequence[np.ndarray]],
    per_dataset_b: Dict[str, Sequence[np.ndarray]],
    n_resamples: int = 10_000,
    seed: int = 0,
) -> Dict[str, float]:
    """Paired bootstrap of the macro-average over datasets of seed-mean micro-F1.

    Each dataset's sentences are resampled independently (stratified
    bootstrap), so every dataset keeps its weight of ``1 / n_datasets`` as in
    the reported average.
    """
    names = sorted(per_dataset_a)
    if names != sorted(per_dataset_b):
        raise ValueError("Both methods must cover the same datasets.")
    a = {k: np.stack(per_dataset_a[k]) for k in names}
    b = {k: np.stack(per_dataset_b[k]) for k in names}
    rng = np.random.default_rng(seed)
    observed = float(np.mean([_f1_rows(b[k]).mean() - _f1_rows(a[k]).mean() for k in names]))
    diffs = np.zeros(n_resamples)
    for k in names:
        n = a[k].shape[1]
        for i in range(n_resamples):
            idx = rng.integers(0, n, size=n)
            diffs[i] += _f1_rows(b[k][:, idx]).mean() - _f1_rows(a[k][:, idx]).mean()
    diffs /= len(names)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return {
        "delta_f1": observed,
        "ci95_low": float(lo),
        "ci95_high": float(hi),
        "p_value": float(min(p, 1.0)),
        "n_resamples": n_resamples,
    }


def holm(p_values: Dict[str, float]) -> Dict[str, float]:
    """Holm-Bonferroni adjusted p-values (family-wise error control)."""
    items = sorted(p_values.items(), key=lambda kv: kv[1])
    m = len(items)
    adjusted: Dict[str, float] = {}
    running = 0.0
    for rank, (key, p) in enumerate(items):
        running = max(running, min(1.0, (m - rank) * p))
        adjusted[key] = running
    return adjusted
