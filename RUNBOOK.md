# RUNBOOK

Exact commands to regenerate every number and figure in the README. Every
script writes JSON into `results/` with an `env` block (git SHA, dirty flag,
CPU/GPU, RAM, library versions, date). Commit the code first, then run, so
that the recorded SHA contains the code that produced the file.

Durations below are measured where stated, otherwise estimated (marked "est.").

## 0. Setup

```bash
uv sync --extra experiments --extra training --group dev   # Python >= 3.11
uv run pytest -q                                            # ~40 s on Apple M4, no GPU, no download for the new tests
```

Datasets are downloaded on first use into `data/raw/` (about 6 MB) and
checked against SHA-256 hashes in `experiments/data.py`. The base checkpoint
(`gliner-community/gliner_small-v2.5`, ~630 MB) is downloaded from the Hugging
Face Hub at a pinned revision.

Disk: the deployment benchmark needs about 2 GB of free space for the exported
models (temporary directory, deleted at the end).

## 1. Zero-shot baseline and data statistics

| Command | Hardware | Time | Output |
|---|---|---|---|
| `uv run python -m scripts.span_imbalance` | any CPU | < 10 s (measured) | `results/span_imbalance.json` |
| `uv run python -m scripts.eval_zero_shot` | CPU, 8 threads | ~10 min on Apple M4 (measured) | `results/zero_shot/gliner_community_gliner_small_v2.5.json` |
| `uv run python -m scripts.error_examples` | CPU | ~1 min (measured) | `results/zero_shot/error_examples.json` |

Inference is deterministic. A re-run must reproduce the committed F1 values
exactly; any difference means the environment changed (compare the `env` blocks).

## 2. Loss ablation

7 variants × 5 seeds = 35 runs (`configs/experiments/ablation.yaml`). Each run
fine-tunes for 2,000 steps, scores CoNLL-2003 dev 4 times, sweeps 7
thresholds, then evaluates 9 test sets at 2 thresholds.

```bash
# full grid; resumable: finished runs (results/ablation/runs/<variant>__seed<k>.json) are skipped
uv run python -m scripts.train_ablation --config configs/experiments/ablation.yaml

# or split across sessions / machines, e.g. one seed at a time
uv run python -m scripts.train_ablation --seeds 0
uv run python -m scripts.train_ablation --seeds 1
# ...

# aggregate + figures (CPU, ~5 min for 10,000 bootstrap resamples)
uv run python -m scripts.analyze_ablation --ablation_dir results/ablation
uv run python -m scripts.make_figures
```

| Hardware | Per run | Full grid | Notes |
|---|---|---|---|
| Apple M4 CPU | ~35 min (est. from a 20-step smoke run at 1.6 it/s) | ~20 h | Feasible overnight in two sessions; laptop must stay plugged in |
| NVIDIA T4 16 GB (school JupyterHub, Colab, Kaggle) | ~10–15 min (est.) | ~6–9 h | `fp16: auto` enables fp16 on CUDA. If the loss becomes NaN (DeBERTa-v3 is known to overflow in fp16), set `fp16: false` and re-run that run only |
| A100 (rented) | ~4–6 min (est.) | ~3 h | ~$5–10 at typical hourly prices (est.) |

Only CPU was tested end to end (smoke runs); the CUDA path uses the same code
with `device="cuda"`. Pipeline check before a long run (~2 min, writes to a temp dir):

```bash
uv run python -m scripts.train_ablation --smoke --variants bce dice_width --seeds 0
```

Reduced grid if budget is short: `--variants bce focal_a025 dice dice_width --seeds 0 1 2`
(12 runs). Report the reduced grid as such.

Copy `results/ablation/runs/*.json` back into the repository if the runs happen on another machine.
Each file is ~0.5 MB; no checkpoints are saved.

## 3. CPU deployment benchmark

```bash
# pipeline check (~10 min): 2 datasets x 50 sentences, 5 repetitions -> results/deploy/<cpu>-quick/
uv run python -m scripts.bench_deploy --quick --output_dir /tmp/deploy_check

# full run (~1 h on Apple M4, est.)
uv run python -m scripts.bench_deploy --config configs/experiments/deploy.yaml
uv run python -m scripts.make_figures
```

Before the full run: plug in the laptop, close other applications, and do not
use the machine during the latency section. The `env` block records the CPU model.

**Intel CPU (recommended, OpenVINO's target hardware).** On a free Colab CPU
runtime (2 vCPUs, Intel Xeon), clone the repo, then:

```bash
pip install uv && uv sync --extra experiments --extra training
uv run python -m scripts.bench_deploy --threads 2
```

Expect ~2–3 h (est.). Download `results/deploy/<intel-cpu-slug>/results.json`
and commit it next to the Apple M4 file. Every machine gets its own sub-directory and figures.

**Known issue.** In pipeline checks on Apple M4, `openvino_int8_static`
(`nncf.quantize`, weights + activations) produced near-zero F1. The same
happened with the span/prompt heads, LSTM and einsum kept in FP32
(`calibration.ignored_patterns/types` in `deploy.yaml`). The root cause is
not established: candidates are activation outliers in DeBERTa-v3, or INT8
kernels of the ARM CPU plugin. Running on an Intel CPU separates the two.
The benchmark records whatever F1 it gets; it does not skip the backend.

## 4. Commit results

```bash
git add results/
git commit -m "exp: <what was run>, <hardware>"
```
