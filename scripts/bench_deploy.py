"""CPU deployment benchmark: accuracy, latency and size of six inference backends.

Backends (all run GLiNER's own pre-processing and span decoding):
    pytorch_fp32           PyTorch eager, FP32
    onnx_fp32              ONNX Runtime, FP32
    onnx_int8_dynamic      ONNX Runtime dynamic quantization (QUInt8 weights, activations quantized at run time)
    openvino_fp32          OpenVINO IR, FP32 (INFERENCE_PRECISION_HINT=f32)
    openvino_int8_weights  NNCF compress_weights INT8_ASYM: INT8 weights, FP32 activations
    openvino_int8_static   NNCF quantize (PTQ, 300 CoNLL-2003 dev sentences): INT8 weights and activations

Latency is end-to-end per call (collate + forward + decode) on inputs of exactly
N words, after warmup, with a fixed thread count. A backend that fails to build
or compile is recorded with its error instead of a number.

Writes ``<output_dir>/<hardware-slug>/results.json`` (one file per machine).

Usage:
    python -m scripts.bench_deploy --config configs/experiments/deploy.yaml
    python -m scripts.bench_deploy --quick   # 2 datasets x 50 sentences, 5 repetitions
"""

from __future__ import annotations

import re
import json
import time
import random
import argparse
import tempfile
import traceback
from typing import Any, Dict, List
from pathlib import Path

import yaml
import numpy as np
import torch

from gliner import GLiNER
from experiments import deploy
from experiments.env import collect_env
from experiments.data import load
from experiments.evaluate import evaluate_dataset


def make_inputs(source: List[Dict], n_words: int, batch_size: int) -> List[Dict]:
    """Batch of examples of exactly ``n_words`` words, cut from consecutive test sentences."""
    words = [w for ex in source for w in ex["tokenized_text"]]
    return [{"tokenized_text": words[i * n_words : (i + 1) * n_words], "ner": []} for i in range(batch_size)]


def time_calls(model: Any, examples: List[Dict], labels: List[str], warmup: int, reps: int, threshold: float):
    collator = model.create_collator()

    def call():
        batch = model.collate_batch(examples, labels, collator)
        out = model.run_batch(batch, threshold=threshold, move_to_device=True)
        model.decode_batch(out, batch, threshold=threshold, flat_ner=True, multi_label=False)

    with torch.no_grad():
        for _ in range(warmup):
            call()
        times = []
        for _ in range(reps):
            t0 = time.perf_counter()
            call()
            times.append((time.perf_counter() - t0) * 1000)
    arr = np.asarray(times)
    return {
        "p50_ms": float(np.percentile(arr, 50)),
        "p90_ms": float(np.percentile(arr, 90)),
        "mean_ms": float(arr.mean()),
        "std_ms": float(arr.std(ddof=1)),
        "repetitions": reps,
        "warmup": warmup,
    }


def probe_fidelity(reference: Any, candidate: Any, examples: List[Dict], labels: List[str]) -> Dict[str, float]:
    """Max / mean absolute difference of span probabilities vs the PyTorch model, valid spans only."""
    collator = reference.create_collator()
    batch = reference.collate_batch(examples, labels, collator)
    with torch.no_grad():
        ref = torch.sigmoid(reference.run_batch(batch, move_to_device=True).logits).cpu().numpy()
        cand = candidate.run_batch(batch, move_to_device=True).logits
    cand = torch.sigmoid(torch.as_tensor(np.asarray(cand))).numpy()
    valid = batch["span_mask"].cpu().numpy().reshape(ref.shape[:3]).astype(bool)
    diff = np.abs(ref - cand)[valid]
    return {"max_abs_dprob": float(diff.max()), "mean_abs_dprob": float(diff.mean())}


def build_backends(cfg: Dict, work: Path, pytorch_model: Any) -> Dict[str, Dict[str, Any]]:
    """Export/convert every backend; returns ``{name: {"model": ..., "files": [...]} or {"error": ...}}``."""
    threads = cfg["threads"]
    ov_opts = deploy.OpenVINOOptions(num_threads=threads, performance_hint=cfg["openvino_hint"])
    built: Dict[str, Dict[str, Any]] = {}
    wanted = cfg["backends"]

    built["pytorch_fp32"] = {"model": pytorch_model, "files": []}  # size read from the Hub file in main()

    export_dir = work / "onnx"
    paths: Dict[str, Path] = {}
    try:
        paths = deploy.export_onnx(pytorch_model, export_dir)
        if_nodes = deploy.count_if_nodes(paths["onnx_fp32"])
        built["onnx_fp32"] = {
            "model": deploy.load_onnxruntime(export_dir, paths["onnx_fp32"].name, threads),
            "files": [paths["onnx_fp32"]],
            "notes": {"if_nodes": if_nodes},
        }
        built["onnx_int8_dynamic"] = {
            "model": deploy.load_onnxruntime(export_dir, paths["onnx_int8_dynamic"].name, threads),
            "files": [paths["onnx_int8_dynamic"]],
        }
    except Exception:
        err = traceback.format_exc(limit=3)
        for name in wanted:
            if name != "pytorch_fp32":
                built[name] = {"error": err}
        return {k: v for k, v in built.items() if k in wanted}

    steps = {
        "openvino_fp32": lambda: deploy.convert_openvino(paths["onnx_fp32"], work / "ov_fp32"),
        "openvino_int8_weights": lambda: deploy.compress_weights_int8(work / "ov_fp32" / "model.xml", work / "ov_w8"),
        "openvino_int8_static": lambda: deploy.quantize_static_int8(
            work / "ov_fp32" / "model.xml",
            _calibration(cfg, pytorch_model),
            work / "ov_static",
            cfg["calibration"].get("ignored_patterns"),
            cfg["calibration"].get("ignored_types"),
        ),
    }
    for name, step in steps.items():
        if name not in wanted:
            continue
        try:
            t0 = time.perf_counter()
            xml = step()
            build_s = time.perf_counter() - t0
            built[name] = {
                "model": deploy.load_openvino(export_dir, xml, ov_opts),
                "files": [xml],
                "notes": {"build_seconds": build_s, "openvino_config": ov_opts.as_config()},
            }
        except Exception:
            built[name] = {"error": traceback.format_exc(limit=3)}
    return {k: v for k, v in built.items() if k in wanted}


def _calibration(cfg: Dict, model: Any) -> List[Dict[str, np.ndarray]]:
    c = cfg["calibration"]
    examples, labels = load(c["dataset"], c["split"])
    random.Random(c["seed"]).shuffle(examples)
    return deploy.calibration_feeds(model, examples[: c["n_sentences"]], labels)


def pytorch_size_mb(model_id: str, revision: str) -> float:
    from huggingface_hub import hf_hub_download  # noqa: PLC0415

    path = hf_hub_download(model_id, "pytorch_model.bin", revision=revision)
    return Path(path).stat().st_size / 2**20


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/experiments/deploy.yaml")
    p.add_argument("--output_dir", default="results/deploy")
    p.add_argument("--quick", action="store_true", help="Tiny accuracy subset and few repetitions (pipeline check).")
    p.add_argument("--backends", nargs="+", default=None)
    p.add_argument("--threads", type=int, default=None, help="Override the config thread count.")
    a = p.parse_args()

    cfg = yaml.safe_load(Path(a.config).read_text())
    if a.backends:
        cfg["backends"] = a.backends
    if a.threads:
        cfg["threads"] = a.threads
    if a.quick:
        cfg["accuracy"]["datasets"] = cfg["accuracy"]["datasets"][:2]
        cfg["latency"].update(warmup=2, repetitions=5)
    torch.set_num_threads(cfg["threads"])

    env = collect_env()
    slug = re.sub(r"[^a-z0-9]+", "-", env["cpu"].lower()).strip("-")
    out_dir = Path(a.output_dir) / (slug + ("-quick" if a.quick else ""))
    out_dir.mkdir(parents=True, exist_ok=True)

    model = GLiNER.from_pretrained(cfg["model"], revision=cfg["revision"]).to("cpu").eval()
    lat = cfg["latency"]
    source, _ = load(lat["source"]["dataset"], lat["source"]["split"])
    probe = make_inputs(source, 64, 4)

    results: Dict[str, Any] = {"config": cfg, "quick": a.quick, "env": env, "backends": {}}
    with tempfile.TemporaryDirectory() as tmp:
        backends = build_backends(cfg, Path(tmp), model)
        for name in cfg["backends"]:
            entry = backends.get(name, {"error": "not built"})
            rec: Dict[str, Any] = {}
            if "error" in entry:
                rec["error"] = entry["error"]
                print(f"[{name}] FAILED\n{entry['error']}")
                results["backends"][name] = rec
                continue
            m = entry["model"]
            rec["notes"] = entry.get("notes", {})
            rec["size_mb"] = (
                pytorch_size_mb(cfg["model"], cfg["revision"])
                if name == "pytorch_fp32"
                else deploy.disk_size_mb(entry["files"])
            )
            rec["fidelity_vs_pytorch"] = probe_fidelity(model, m, probe, lat["labels"])
            rec["accuracy"] = {}
            for ds in cfg["accuracy"]["datasets"]:
                r = evaluate_dataset(
                    m,
                    ds,
                    threshold=cfg["accuracy"]["threshold"],
                    batch_size=cfg["accuracy"]["batch_size"],
                    max_examples=50 if a.quick else None,
                )
                rec["accuracy"][ds] = {k: r[k] for k in ("f1", "precision", "recall", "n_sentences", "f1_bootstrap")}
                rec["accuracy"][ds]["sentence_counts"] = r["sentence_counts"]
                print(f"[{name}] {ds:22s} F1={100 * r['f1']:.2f}")
            rec["accuracy_avg_f1"] = float(np.mean([v["f1"] for v in rec["accuracy"].values()]))
            rec["latency"] = {}
            for words in lat["words"]:
                for bs in lat["batch_sizes"]:
                    t = time_calls(
                        m,
                        make_inputs(source, words, bs),
                        lat["labels"],
                        lat["warmup"],
                        lat["repetitions"],
                        cfg["accuracy"]["threshold"],
                    )
                    rec["latency"][f"w{words}_b{bs}"] = {"words": words, "batch_size": bs, **t}
                    print(f"[{name}] words={words:3d} batch={bs} p50={t['p50_ms']:.1f}ms p90={t['p90_ms']:.1f}ms")
            results["backends"][name] = rec
            (out_dir / "results.json").write_text(json.dumps(results, indent=1))

    (out_dir / "results.json").write_text(json.dumps(results, indent=1))
    print(f"\nwrote {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
