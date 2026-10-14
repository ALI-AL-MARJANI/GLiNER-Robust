# Development log

## 2026-10 — rebuild of the experiments on `main`

### Numbers withdrawn from the previous README

| Previous claim | Status | Reason |
|---|---|---|
| Loss ablation table (BCE 50.09, Focal α=0.25 51.08, Focal α=0.70 50.27, Dice 50.79, Dice+Width 50.79 F1 on WNUT-17) | Withdrawn, not reproducible | `scripts/train_ablation.py` passed `loss_type`, `dice_gamma` and `use_span_width_weight` to `TrainingArguments`, which had no such fields on `main` (the script raised `TypeError`). The package contained no Dice loss or width weighting. The surviving local training logs of the "dice" and "dice_width" runs show identical loss curves (16.47, 11.84, 6.93, 7.24 at steps 50–200), on the scale of a summed BCE; a Dice loss is bounded in [0, 1]. Those runs were BCE runs. In addition: 1 seed, 200 steps, 1,500 training sentences, one evaluation set. |
| "+0.99 pp" focal-loss gain | Withdrawn | Single seed, no interval; superseded by the 5-seed ablation (pending). |
| "We tune with α=0.7" | Withdrawn | Contradicted by the table it came with (α=0.25 scored higher). |
| OpenVINO INT8 "2.35× faster, 4× smaller", "no accuracy degradation" | Withdrawn | The conversion script never measured F1 for OpenVINO models (`"f1_wnut17": None`); it described weight-only compression while calling `nncf.quantize` (static W+A); latency was measured on one input length on unspecified Apple hardware. Replaced by `scripts/bench_deploy.py` (pending run). |
| "187× more negative spans than positive" on WNUT-17 | Replaced | Not produced by any script in the repository. `scripts/span_imbalance.py` now computes it with an explicit definition: 1,124 negatives per positive on WNUT-17 test (K=12, all 6 types prompted). |
| Baseline 44.12% (WNUT-17), 53.22% (CoNLL-2003) for `knowledgator/gliner-bi-small-v1.0` | Not carried over | No result file in the repository; the evaluation harness now uses `gliner_small-v2.5` throughout. |

### Changes

- Losses implemented in the package and tested; trainer options with backward-compatible defaults.
- Upstream bug fixed: `BiEncoderSpanModel.loss` passed `label_smoothing` into the `prob_margin` slot of `_loss`.
- ONNX export made compilable by the OpenVINO CPU plugin (`If` nodes from DeBERTa-v2's scripted `build_rpos`).
- Dataset loading moved off script-based Hugging Face datasets (unsupported by `datasets>=4`) to pinned original files with checksums.
- The ablation runner originally built the selection callback without registering it with the Trainer (only the last step was ever scored); caught in a smoke run before any real run, fixed by constructing the Trainer explicitly.
- Static INT8 (OpenVINO, `nncf.quantize`) gives near-zero F1 on Apple M4 in pipeline checks; documented as an open issue in `RUNBOOK.md`, not hidden.
