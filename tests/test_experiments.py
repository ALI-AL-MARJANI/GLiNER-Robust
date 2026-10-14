"""Fast tests for the experiment harness (no network, no model download)."""

import json

import numpy as np
import pytest

from experiments.data import DATASETS, parse_bio, bio_to_spans
from experiments.metrics import (
    prf,
    holm,
    mean_std,
    bootstrap_ci,
    f1_from_counts,
    per_type_counts,
    sentence_counts,
    paired_bootstrap,
    paired_bootstrap_average,
)
from gliner.evaluation.evaluator import BaseNEREvaluator

# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #


def test_parse_bio_token_first_and_tag_first():
    token_first = "Paris\tB-location\nis\tO\n\nBob\tB-person\n"
    tag_first = "B-ACTOR\ttom\nI-ACTOR\thanks\n\nO\tmovies\n"
    assert parse_bio(token_first, tag_first=False) == [(["Paris", "is"], ["B-location", "O"]), (["Bob"], ["B-person"])]
    assert parse_bio(tag_first, tag_first=True) == [(["tom", "hanks"], ["B-ACTOR", "I-ACTOR"]), (["movies"], ["O"])]


def test_parse_bio_rejects_malformed_lines():
    with pytest.raises(ValueError):
        parse_bio("token B-person\n", tag_first=False)


def test_bio_to_spans_conventions():
    tags = ["B-per", "I-per", "O", "I-loc", "I-loc", "B-loc", "B-org", "I-per"]
    assert bio_to_spans(tags) == [(0, 1, "per"), (3, 4, "loc"), (5, 5, "loc"), (6, 6, "org"), (7, 7, "per")]


def test_every_dataset_has_a_pinned_test_split():
    for name, spec in DATASETS.items():
        assert "test" in spec.splits, name
        for split in spec.splits.values():
            assert len(split.sha256) == 64
            assert split.url.startswith("https://")


def test_label_verbalization():
    assert DATASETS["crossner_ai"].verbalize("programlang") == "programming language"
    assert DATASETS["mit_restaurant"].verbalize("Restaurant_Name") == "restaurant name"
    assert DATASETS["wnut17"].verbalize("person") == "person"


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #


def _random_corpus(n=60, seed=0):
    rng = np.random.default_rng(seed)
    gold, pred = [], []
    for _ in range(n):
        g = {(int(s), int(s + rng.integers(0, 3)), str(rng.choice(["a", "b", "c"]))) for s in rng.integers(0, 20, 4)}
        p = {x for x in g if rng.random() < 0.7}
        p |= {(int(s), int(s), "a") for s in rng.integers(0, 20, 2)}
        gold.append(sorted(g))
        pred.append(sorted(p))
    return gold, pred


def test_micro_f1_matches_upstream_evaluator():
    gold, pred = _random_corpus()
    counts = np.array([sentence_counts(g, p) for g, p in zip(gold, pred)])
    _, upstream_f1 = BaseNEREvaluator(gold, pred).evaluate()
    assert f1_from_counts(counts) == pytest.approx(float(upstream_f1), abs=1e-9)


def test_per_type_counts_sum_to_sentence_counts():
    gold, pred = _random_corpus(n=5)
    for g, p in zip(gold, pred):
        per_type = per_type_counts(g, p)
        assert np.sum(list(per_type.values()), axis=0).tolist() == list(sentence_counts(g, p))


def test_prf_edge_cases():
    assert prf(0, 0, 0) == {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    assert prf(1, 2, 1)["f1"] == pytest.approx(2 / 3)


def test_mean_std_uses_sample_std():
    assert mean_std([1.0, 3.0]) == {"mean": 2.0, "std": pytest.approx(np.sqrt(2.0)), "n": 2}
    assert mean_std([5.0])["std"] == 0.0


def test_bootstrap_ci_contains_point_estimate():
    gold, pred = _random_corpus()
    counts = np.array([sentence_counts(g, p) for g, p in zip(gold, pred)])
    ci = bootstrap_ci(counts, n_resamples=500)
    assert ci["ci95_low"] <= ci["f1"] <= ci["ci95_high"]


def test_paired_bootstrap_identical_methods():
    gold, pred = _random_corpus()
    counts = np.array([sentence_counts(g, p) for g, p in zip(gold, pred)])
    res = paired_bootstrap([counts, counts], [counts, counts], n_resamples=300)
    assert res["delta_f1"] == 0.0 and res["p_value"] == 1.0


def test_paired_bootstrap_detects_clear_improvement():
    gold, pred = _random_corpus(n=200)
    worse = np.array([sentence_counts(g, p) for g, p in zip(gold, pred)])
    better = np.array([sentence_counts(g, g) for g in gold])  # perfect predictions
    res = paired_bootstrap([worse], [better], n_resamples=300)
    assert res["delta_f1"] > 0 and res["ci95_low"] > 0 and res["p_value"] < 0.01
    avg = paired_bootstrap_average({"x": [worse], "y": [worse]}, {"x": [better], "y": [worse]}, n_resamples=300)
    assert avg["delta_f1"] == pytest.approx(res["delta_f1"] / 2)


def test_holm_adjustment():
    adj = holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adj == {"a": pytest.approx(0.03), "c": pytest.approx(0.06), "b": pytest.approx(0.06)}


# --------------------------------------------------------------------------- #
# aggregation script on synthetic runs
# --------------------------------------------------------------------------- #


def _fake_run(variant, seed, counts_by_ds):
    def result(c):
        c = np.asarray(c)
        tp, n_pred, n_gold = c.sum(0).tolist()
        return {
            **prf(tp, n_pred, n_gold),
            "per_type": {"x": {**prf(tp, n_pred, n_gold), "support": n_gold}},
            "sentence_counts": c.tolist(),
        }

    res = {name: result(c) for name, c in counts_by_ds.items()}
    return {
        "variant": variant,
        "seed": seed,
        "smoke": False,
        "config": {
            "data": {"eval_in_domain": ["ind"], "eval_zero_shot": ["z1", "z2"]},
            "variants": {"bce": {}, "dice": {}},
        },
        "thresholds": {"fixed": 0.5, "tuned": 0.5},
        "selection": {"best_step": 10},
        "results": {"fixed": res, "tuned": res},
    }


def test_analyze_ablation_end_to_end(tmp_path):
    from scripts.analyze_ablation import load_runs, summarize, to_markdown  # noqa: PLC0415

    rng = np.random.default_rng(0)
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    for variant in ("bce", "dice"):
        for seed in range(3):
            counts = {
                n: np.stack([rng.integers(0, 3, 40), np.full(40, 3), np.full(40, 3)], 1) for n in ("ind", "z1", "z2")
            }
            (runs_dir / f"{variant}__seed{seed}.json").write_text(json.dumps(_fake_run(variant, seed, counts)))
    summary = summarize(load_runs(runs_dir), "tuned", "bce", n_resamples=200)
    assert set(summary["table"]) == {"bce", "dice"}
    assert summary["table"]["bce"]["zero_shot_avg"]["n"] == 3
    assert "p_holm" in summary["tests_vs_baseline"]["dice"]["zero_shot_avg"]
    md = to_markdown(summary, ["bce", "dice"])
    assert md.count("\n") == 3
