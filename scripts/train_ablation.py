"""Loss ablation: fine-tune on CoNLL-2003, select on CoNLL-2003 dev, evaluate zero-shot.

One JSON per (variant, seed) is written to ``<output_dir>/runs/``. Existing
run files are skipped, so an interrupted grid can be resumed by re-running the
same command. ``scripts/analyze_ablation.py`` aggregates the runs.

Usage:
    python -m scripts.train_ablation --config configs/experiments/ablation.yaml
    python -m scripts.train_ablation --variants bce dice --seeds 0 1 2
    python -m scripts.train_ablation --smoke   # 20 steps, tiny eval, writes to /tmp
"""

from __future__ import annotations

import copy
import json
import time
import argparse
import tempfile
from typing import Any, Dict, List, Optional
from pathlib import Path

import yaml
import torch
from transformers import TrainerCallback
from transformers.trainer_utils import set_seed

from gliner import GLiNER
from experiments.env import collect_env
from gliner.training import Trainer, TrainingArguments
from experiments.data import load
from experiments.metrics import f1_from_counts
from experiments.evaluate import predict, evaluate_dataset


class SelectOnDev(TrainerCallback):
    """Every ``eval_every`` steps, score the model on a dev set and keep the best weights in memory."""

    def __init__(self, gliner_model: GLiNER, dev: List[Dict], labels: List[str], eval_every: int, eval_cfg: Dict):
        self.gliner_model = gliner_model
        self.dev = dev
        self.labels = labels
        self.eval_every = eval_every
        self.eval_cfg = eval_cfg
        self.history: List[Dict[str, float]] = []
        self.best_f1 = -1.0
        self.best_step: Optional[int] = None
        self.best_state: Optional[Dict[str, torch.Tensor]] = None

    def _score(self, step: int) -> None:
        was_training = self.gliner_model.training
        out = predict(
            self.gliner_model,
            self.dev,
            self.labels,
            threshold=self.eval_cfg["threshold"],
            batch_size=self.eval_cfg["batch_size"],
        )
        f1 = f1_from_counts(out["counts"])
        self.history.append({"step": step, "dev_f1": f1})
        if f1 > self.best_f1:
            self.best_f1, self.best_step = f1, step
            self.best_state = {k: v.detach().to("cpu", copy=True) for k, v in self.gliner_model.state_dict().items()}
        if was_training:
            self.gliner_model.train()

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.eval_every == 0 or state.global_step == state.max_steps:
            self._score(state.global_step)
        return control


def _as_gliner(examples: List[Dict]) -> List[Dict]:
    return [{"tokenized_text": ex["tokenized_text"], "ner": [list(s) for s in ex["ner"]]} for ex in examples]


def run_one(cfg: Dict[str, Any], variant: str, seed: int, out_path: Path, smoke: bool) -> Dict[str, Any]:
    tcfg = dict(cfg["training"])
    ecfg = cfg["evaluation"]
    vcfg = cfg["variants"][variant]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    set_seed(seed)
    model = GLiNER.from_pretrained(cfg["base_model"], revision=cfg.get("base_revision")).to(device)

    train, _ = load(cfg["data"]["train"]["dataset"], cfg["data"]["train"]["split"])
    sel = cfg["data"]["select"]
    dev, dev_labels = load(sel["dataset"], sel["split"])
    dev = dev[: sel.get("max_examples") or None]

    max_steps, eval_every = tcfg.pop("max_steps"), tcfg.pop("eval_every")
    if smoke:
        max_steps, eval_every, dev = 20, 10, dev[:32]
    fp16 = tcfg.pop("fp16")
    fp16 = torch.cuda.is_available() if fp16 == "auto" else bool(fp16)

    callback = SelectOnDev(model, dev, dev_labels, eval_every, ecfg)
    with tempfile.TemporaryDirectory() as tmp:
        args = TrainingArguments(
            output_dir=tmp,
            max_steps=max_steps,
            seed=seed,
            data_seed=seed,
            save_strategy="no",
            eval_strategy="no",
            logging_steps=50,
            report_to="none",
            dataloader_num_workers=0,
            fp16=fp16,
            use_cpu=device == "cpu",
            **tcfg,
            **vcfg,
        )
        # Same construction as GLiNER.train_model, plus the selection callback.
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=_as_gliner(train),
            data_collator=model._create_data_collator(),
            processing_class=model.data_processor.transformer_tokenizer,
            callbacks=[callback],
        )
        t0 = time.perf_counter()
        trainer.train()
        train_seconds = time.perf_counter() - t0
        loss_log = [
            {"step": e["step"], "loss": e["loss"]} for e in trainer.state.log_history if "loss" in e and "step" in e
        ]
        trainer.callback_handler.callbacks.clear()
    del trainer

    if callback.best_state is None:
        callback._score(max_steps)
    model.load_state_dict(callback.best_state)
    model.eval()

    threshold_sweep = {}
    for t in ecfg["threshold_grid"]:
        out = predict(model, dev, dev_labels, threshold=t, batch_size=ecfg["batch_size"])
        threshold_sweep[str(t)] = f1_from_counts(out["counts"])
    tuned = float(max(threshold_sweep, key=lambda k: (threshold_sweep[k], -abs(float(k) - 0.5))))

    datasets = cfg["data"]["eval_in_domain"] + cfg["data"]["eval_zero_shot"]
    results: Dict[str, Dict[str, Any]] = {"fixed": {}, "tuned": {}}
    thresholds = {"fixed": ecfg["threshold"], "tuned": tuned}
    for mode, threshold in thresholds.items():
        for name in datasets:
            r = evaluate_dataset(
                model,
                name,
                threshold=threshold,
                batch_size=ecfg["batch_size"],
                max_examples=32 if smoke else None,
            )
            results[mode][name] = r
            print(f"  [{variant} seed={seed} {mode} t={threshold}] {name:22s} F1={r['f1'] * 100:6.2f}")

    record = {
        "variant": variant,
        "seed": seed,
        "smoke": smoke,
        "config": {"base_model": cfg["base_model"], "base_revision": cfg.get("base_revision"), **cfg},
        "variant_config": vcfg,
        "max_steps": max_steps,
        "selection": {
            "dataset": sel["dataset"],
            "split": sel["split"],
            "n": len(dev),
            "history": callback.history,
            "best_step": callback.best_step,
            "best_dev_f1": callback.best_f1,
            "threshold_sweep": threshold_sweep,
            "tuned_threshold": tuned,
        },
        "thresholds": thresholds,
        "train_loss": loss_log,
        "train_seconds": train_seconds,
        "results": results,
        "env": collect_env(),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record))
    return record


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/experiments/ablation.yaml")
    p.add_argument("--output_dir", default="results/ablation")
    p.add_argument("--variants", nargs="+", default=None, help="Subset of variants (default: all in config).")
    p.add_argument("--seeds", nargs="+", type=int, default=None, help="Subset of seeds (default: config).")
    p.add_argument("--smoke", action="store_true", help="20 steps + 32-sentence evals, written to a temp dir.")
    a = p.parse_args()

    cfg = yaml.safe_load(Path(a.config).read_text())
    variants = a.variants or list(cfg["variants"])
    seeds = a.seeds if a.seeds is not None else cfg["seeds"]
    out_dir = Path(tempfile.mkdtemp(prefix="ablation_smoke_")) if a.smoke else Path(a.output_dir)
    unknown = set(variants) - set(cfg["variants"])
    if unknown:
        raise SystemExit(f"Unknown variants: {sorted(unknown)}")

    for variant in variants:
        for seed in seeds:
            path = out_dir / "runs" / f"{variant}__seed{seed}.json"
            if path.exists():
                print(f"skip {path} (exists)")
                continue
            print(f"\n=== {variant} seed={seed} -> {path}")
            run_one(copy.deepcopy(cfg), variant, seed, path, a.smoke)
    print(f"\nRun files in {out_dir / 'runs'}")


if __name__ == "__main__":
    main()
