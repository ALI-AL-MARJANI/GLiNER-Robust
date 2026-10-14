# GLiNER-Robust

A fork of [GLiNER](https://github.com/urchade/GLiNER) that adds span-level losses for the extreme negative/positive imbalance of span-based NER, a multi-seed zero-shot evaluation harness, and a measured CPU deployment comparison (PyTorch, ONNX Runtime, OpenVINO).

## TL;DR

**Zero-shot entity-level micro-F1 of the base checkpoint**, measured with this repository's harness (`results/zero_shot/gliner_community_gliner_small_v2.5.json`):

| Dataset (test) | Sentences | F1 (%) | 95% CI (bootstrap) |
|---|---:|---:|---|
| WNUT-17 | 1,287 | 46.65 | [44.00, 49.25] |
| CrossNER AI | 431 | 52.18 | [49.03, 55.25] |
| CrossNER literature | 416 | 63.67 | [61.36, 65.94] |
| CrossNER music | 465 | 74.90 | [72.88, 76.94] |
| CrossNER politics | 651 | 64.51 | [62.20, 66.71] |
| CrossNER science | 543 | 64.11 | [61.37, 66.76] |
| MIT Movie | 2,443 | 53.92 | [52.59, 55.27] |
| MIT Restaurant | 1,521 | 40.77 | [38.98, 42.56] |
| **Average (8 zero-shot sets)** | | **57.59** | |
| CoNLL-2003 (in-domain for the ablation, not zero-shot) | 3,453 | 62.77 | [61.37, 64.16] |

Setup: `gliner-community/gliner_small-v2.5` @ `f227d3c`, no fine-tuning, threshold 0.5, flat NER, Apple M4 CPU (env block in the JSON). Deterministic inference, single run; intervals come from resampling test sentences.

| Study | Status |
|---|---|
| Loss ablation (7 variants × 5 seeds, fine-tune on CoNLL-2003, evaluate on the 8 sets above) | **TBD (run pending)**. Code, config and statistics are in place; see [RUNBOOK.md](RUNBOOK.md#2-loss-ablation). |
| CPU deployment (6 backends: F1, p50/p90 latency, size) | **TBD (run pending)**. See [RUNBOOK.md](RUNBOOK.md#3-cpu-deployment-benchmark). |

## What this fork adds

Everything else is upstream GLiNER at commit `dfc0061` (see `git log`).

| Path | Content |
|---|---|
| `gliner/modeling/loss_functions.py` | `span_dice_loss` (self-adjusting Dice, Li et al. 2020), `span_width_weights`, `apply_positive_weights` |
| `gliner/modeling/base.py` | `BaseModel._span_loss`: shared span loss for `UniEncoderSpanModel` and `BiEncoderSpanModel`, with `loss_type`, Dice and width-weighting options. Also fixes an upstream bug where `BiEncoderSpanModel.loss` passed `label_smoothing` positionally into `prob_margin`. |
| `gliner/training/trainer.py` | `TrainingArguments.loss_type / dice_alpha / dice_smooth / use_span_width_weight` (defaults reproduce upstream exactly); the trainer refuses these options for model types that would silently ignore them |
| `experiments/` | Dataset download with pinned revisions and SHA-256 checks, BIO parsing, entity-level F1, bootstrap / paired bootstrap / Holm correction, run metadata, deployment backends |
| `scripts/` | `eval_zero_shot.py`, `span_imbalance.py`, `train_ablation.py`, `analyze_ablation.py`, `bench_deploy.py`, `error_examples.py`, `make_figures.py` |
| `configs/experiments/` | `ablation.yaml`, `deploy.yaml`: every hyperparameter used by the scripts |
| `tests/test_span_losses.py`, `tests/test_experiments.py`, `tests/test_deploy.py` | 35 fast tests (CPU, no download) |
| `results/` | Raw JSON outputs; figures are generated from them by `scripts/make_figures.py` |

## Motivation

A span model with maximum width K = 12 scores every span of 1–12 words against every prompted entity type, and almost all of these pairs are negatives (`results/span_imbalance.json`, `scripts/span_imbalance.py`):

| Dataset | Split | Types | Gold entities | Candidate (span, type) pairs | Negatives per positive |
|---|---|---:|---:|---:|---:|
| CoNLL-2003 | train | 4 | 23,429 | 6,655,148 | 283 |
| WNUT-17 | test | 6 | 1,079 | 1,211,826 | 1,124 |
| CrossNER science | test | 17 | 3,089 | 3,366,340 | 1,089 |
| MIT Restaurant | test | 8 | 3,151 | 647,544 | 205 |

Losses designed for class imbalance (focal, Dice) are a natural candidate. The question this repository tests is whether they change zero-shot transfer after fine-tuning, measured with enough seeds to tell a real effect from noise.

## Method

**BCE / focal** (upstream, Lin et al. 2017). Per (span, type) pair with probability *p* and label *y*:
`FL = −α_t (1 − p_t)^γ log p_t`. With `focal_loss_alpha=-1, focal_loss_gamma=0` it is exactly BCE (tested in `tests/test_span_losses.py`).

**Self-adjusting Dice** (`span_dice_loss`, Li et al. 2020). Set-level over all valid pairs of the batch, as in the authors' implementation:

```
p~   = (1 − p)^α · p
loss = 1 − (2 Σ w·p~·y + s) / (Σ w·p~ + Σ w·y + s)
```

Padded spans and padded entity types are excluded through `w` (tested). Deviations from the reference: no OHEM; Dice is not combined with the GLiNER negative-type sub-sampling (`negatives`, `masking` apply to BCE/focal only).

**Span-width weighting** (`use_span_width_weight`). Positive pairs of width *k* words are weighted by `1 + ln k`; negatives keep weight 1. This is a heuristic of this repository, not a published method: long entities are rarer, and up-weighting all long spans would mostly up-weight negatives.

**Threshold.** Dice does not produce calibrated probabilities, so every ablation run reports two results: GLiNER's default threshold 0.5, and a threshold picked on CoNLL-2003 dev from {0.2, …, 0.8}.

**OpenVINO export.** The ONNX graph exported by upstream GLiNER contains six `If` nodes coming from DeBERTa-v2's `@torch.jit.script` `build_rpos` in `transformers`. The OpenVINO CPU plugin refuses to compile them ("If operation with dynamic rank"). In GLiNER's encoder, attention is always self-attention, so the branch is never taken. `experiments.deploy.traceable_deberta_rpos` replaces the helper with its taken branch during export only. On the real model, the patched and original ONNX graphs give identical logits (max abs difference 0.0 on a probe batch). `tests/test_deploy.py` checks the same property on a random tiny DeBERTa-v2 with a different input shape than the export.

## Experimental protocol

**Data.** Original distributions, pinned (commit or SHA-256) in `experiments/data.py`, downloaded on first use into `data/raw/`:

| Dataset | Source | Test sentences / entities | Types |
|---|---|---:|---:|
| CoNLL-2003 (train 14,041 / dev 3,250 / test 3,453) | CrossNER repository copy | 3,453 / 5,628 | 4 |
| CrossNER AI, literature, music, politics, science | [zliucr/CrossNER](https://github.com/zliucr/CrossNER) @ `2e7ba2a` | 431–651 / 1,809–4,209 | 9–17 |
| MIT Movie (`engtest.bio`), MIT Restaurant | [MIT CSAIL SLS](https://groups.csail.mit.edu/sls/downloads/) | 2,443 / 5,339 and 1,521 / 3,151 | 12, 8 |
| WNUT-17 | [leondz/emerging_entities_17](https://github.com/leondz/emerging_entities_17) @ `e52a2d2` | 1,287 / 1,079 | 6 |

Entity types are prompted as natural-language names (e.g. `programlang` → "programming language"). The mapping is defined in `experiments/data.py`.

**Metric.** Entity-level micro-F1 with exact match on start, end and type. It is computed from per-sentence (TP, #pred, #gold) counts, and the result is identical to `gliner.evaluation.BaseNEREvaluator` (unit test, and checked on real predictions). For a single run, a 95% interval comes from 10,000 bootstrap resamples of test sentences. Method comparisons use a paired bootstrap over sentences that keeps all seeds, stratified by dataset for the average, with Holm–Bonferroni correction across variants.

**Ablation** (`configs/experiments/ablation.yaml`):
- Base checkpoint: `gliner_small-v2.5` @ `f227d3c`.
- Training: fine-tuning on the full CoNLL-2003 train split for 2,000 steps at batch 8. Optimizer values follow upstream `configs/config_span.yaml`.
- Selection: the checkpoint is chosen on 1,000 CoNLL-2003 dev sentences every 500 steps.
- Variants:
  - `bce` (baseline);
  - `bce_width`;
  - `focal_a025`, `focal_a050`, `focal_a075` (γ = 2);
  - `dice`;
  - `dice_width`.
- Seeds: 0–4.
- Evaluation: CoNLL-2003 test plus the 8 zero-shot sets. Rare-type F1 is the macro-F1 over types that make up less than 5% of a test set's gold entities.

**Deployment** (`configs/experiments/deploy.yaml`):
- Backends: PyTorch FP32, ONNX Runtime FP32, ONNX Runtime dynamic INT8 (upstream `export_to_onnx(quantize=True)`), OpenVINO FP32, OpenVINO INT8 weight-only (`nncf.compress_weights`, INT8_ASYM), and OpenVINO static INT8 (`nncf.quantize`, weights and activations, 300 CoNLL-2003 dev calibration sentences).
- Pipeline: the same GLiNER pre-processing and decoding for every backend.
- Measurements: F1 on the 8 zero-shot sets; end-to-end latency with 10 warmup calls and 50 timed calls, inputs of 16 / 64 / 256 words at batch 1 and 8, 4 threads, reporting p50 / p90 / mean ± std; size on disk; and the max |Δp| from PyTorch on a probe batch.

## Results

### Zero-shot baseline

See the TL;DR table. For context only, the GLiNER paper reports these numbers for its own checkpoints. They are not comparable with ours: the checkpoints differ (paper v1 models), the label strings differ (the paper uses the InstructUIE/UniNER JSON release), and WNUT-17 is not in the paper's table.

| Paper (Zaratiana et al., 2024), Table 1 | Movie | Restaurant | AI | Literature | Music | Politics | Science | Avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GLiNER-S (50M) | 46.9 | 33.3 | 50.7 | 60.0 | 60.9 | 61.5 | 55.6 | 52.7 |
| GLiNER-L (0.3B) | 57.2 | 42.9 | 57.2 | 64.4 | 69.6 | 72.6 | 62.6 | 60.9 |

**Error analysis.** Per-type scores are in the result JSON. Concrete failures come from `scripts/error_examples.py` and are stored in `results/zero_shot/error_examples.json`.

- **Precision, not recall, limits WNUT-17.** P = 36.5 and R = 64.5. The model tags common nouns that the WNUT guidelines leave unannotated, such as "soldier" [person], "Police" [group] and "army" [group].
- **Tokenization artifacts in CrossNER.** Plurals are split ("naive Bayes classifier s"). The model includes the stray "s" and loses the exact match: "Gaussian mixture model s" is predicted where the gold entity is "Gaussian mixture model".
- **Vague types.** `miscellaneous` reaches 11.8 F1 on CrossNER AI and `task` reaches 26.3. On MIT Movie, `review` is at 5.5, `plot` at 13.2 and `title` at 25.1, while `actor` and `director` exceed 82. Annotation conventions also cost exact matches: "right now" is gold [year], and the gold for "1980s action movies" is "action", not "action movies".

### Loss ablation

TBD (run pending). `scripts/analyze_ablation.py` will write `results/ablation/summary.md` (mean ± std over seeds, Δ vs `bce` with 95% CI and Holm-corrected p), and `scripts/make_figures.py` will draw `results/figures/ablation_zero_shot.png`. A result where no variant beats BCE significantly will be reported as such.

### CPU deployment

TBD (run pending). Output: `results/deploy/<cpu>/results.json`, `results/figures/deploy_pareto_<cpu>.png`, `results/figures/deploy_latency_<cpu>.png`.

## Limitations and threats to validity

- **One checkpoint, one size.** The ablation fine-tunes a single small model on a single source dataset (CoNLL-2003). It says nothing about training from scratch on Pile-NER as in the paper, or about larger backbones.
- **Fine-tuning on CoNLL-2003 can reduce zero-shot F1 relative to the base checkpoint**, whatever the loss. The base-checkpoint row above is the reference for that effect.
- **Label verbalization is ours.** Changing a prompt string (e.g. "miscellaneous") changes F1. Absolute numbers are therefore not comparable to published tables.
- **Selection uses CoNLL-2003 dev only.** No target-domain data is used for checkpoint or threshold selection. That is the intended zero-shot setting, but the selected threshold may be far from optimal on a target domain.
- **Scale of the losses.** Upstream sums BCE over pairs (`loss_reduction: sum`), whereas Dice lies in [0, 1]. With AdamW and gradient clipping at 10 the effective step size differs between variants. No per-variant learning-rate search was run.
- **Latency.** Results apply to the measured CPU only. OpenVINO is designed for Intel CPUs, so numbers from Apple silicon do not transfer.

## Reproduce

Commands, expected run time and hardware for every table are in [RUNBOOK.md](RUNBOOK.md):
- **Zero-shot baseline:** about 10 minutes on an Apple M4 CPU.
- **Loss ablation:** 35 fine-tuning runs, GPU recommended.
- **CPU benchmark:** about 1 hour per machine.

```bash
uv sync --extra experiments --extra training --group dev   # Python >= 3.11
uv run pytest -q
```

## Using GLiNER itself

Installation, inference API, training and ONNX conversion are unchanged from upstream: see the [upstream README](https://github.com/urchade/GLiNER) and `docs/`. The losses added here are opt-in training arguments:

```python
from gliner.training import TrainingArguments

args = TrainingArguments(output_dir="out", loss_type="dice", dice_alpha=1.0, use_span_width_weight=True)
```

## References

- Zaratiana, Tomeh, Holat, Charnois. *GLiNER: Generalist Model for Named Entity Recognition using Bidirectional Transformer.* NAACL 2024. [arXiv:2311.08526](https://arxiv.org/abs/2311.08526), [code](https://github.com/urchade/GLiNER)
- Lin, Goyal, Girshick, He, Dollár. *Focal Loss for Dense Object Detection.* ICCV 2017. [arXiv:1708.02002](https://arxiv.org/abs/1708.02002)
- Li, Sun, Meng, Liang, Wu, Li. *Dice Loss for Data-imbalanced NLP Tasks.* ACL 2020. [arXiv:1911.02855](https://arxiv.org/abs/1911.02855), [code](https://github.com/ShannonAI/dice_loss_for_NLP)
- Liu et al. *CrossNER: Evaluating Cross-Domain Named Entity Recognition.* AAAI 2021. [arXiv:2012.04373](https://arxiv.org/abs/2012.04373), [data](https://github.com/zliucr/CrossNER)
- Liu, Pasupat, Cyphers, Glass. *ASGARD: A Portable Architecture for Multilingual Dialogue Systems.* ICASSP 2013 (MIT Movie / Restaurant). [data](https://groups.csail.mit.edu/sls/downloads/)
- Derczynski, Nichols, van Erp, Limsopatham. *Results of the WNUT2017 Shared Task on Novel and Emerging Entity Recognition.* W-NUT 2017. [data](https://github.com/leondz/emerging_entities_17)
- He, Gao, Chen. *DeBERTaV3.* ICLR 2023. [arXiv:2111.09543](https://arxiv.org/abs/2111.09543)
- OpenVINO weight compression and post-training quantization: [docs](https://docs.openvino.ai/), [NNCF](https://github.com/openvinotoolkit/nncf)
- ONNX Runtime quantization: [docs](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)
- Efron, Tibshirani. *An Introduction to the Bootstrap.* 1993. Koehn. *Statistical Significance Tests for Machine Translation Evaluation.* EMNLP 2004 (paired bootstrap).

## License

Apache-2.0, inherited from upstream GLiNER (`LICENSE`). The added code is released under the same license.
