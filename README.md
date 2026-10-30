# AI Engineer Journey

Production inference engineering, built from scratch on a laptop.

Each phase takes one concrete problem — make a model fast, serve it, containerize it, then do the same for LLMs — and solves it with measurements instead of guesses. No training, no notebooks. The focus is everything that happens to a model *after* it has been trained.

**Status:** Phase 0 and 1 complete.

---

## Phase 0 + 1 — ResNet-18 Inference Optimization

I took a pretrained ResNet-18 and made it **4.4x faster with no measurable accuracy loss**, then verified every claim against ground-truth labels with confidence intervals.

| Metric | Baseline | Optimized | Change |
|---|---|---|---|
| Latency (p50, batch 1) | 16.9 ms | 3.8 ms | **4.4x faster** (77% cut) |
| Model size | 45 MB | 11 MB | **4x smaller** |
| Throughput (batch 1) | 57 img/s | 258 img/s | **4.5x higher** |
| Top-1 accuracy (500 labeled images) | 67.6% | 68.6% | **no measurable loss** |

Baseline is PyTorch FP32 on CPU. Optimized is static INT8 quantized ONNX. Hardware: MacBook Pro M1 Pro, no NVIDIA GPU.

**→ [Read the full write-up](phase0-1-inference/observation1.md)** — including why each optimization worked, and why one of them didn't.

### Three findings worth reading

**1. INT8 quantization was the entire win.** Roughly 4x faster at every batch size, measured against the best FP32 ONNX configuration so the comparison is fair. Accuracy cost was statistically zero: 9 predictions changed out of 500, the confidence intervals overlap almost completely, and the small apparent gain is noise rather than an improvement.

**2. Graph optimization did nothing, and I can explain why.** Enabling ONNX Runtime's highest optimization level cut the graph from 49 nodes to 24, with no measurable speedup. Inspecting the graphs showed the exporter had *already* folded BatchNorm into Conv during export, and the remaining Conv+ReLU fusion only removes per-node overhead, not arithmetic. For a compute-bound CNN, that barely matters. A negative result I can account for is worth more than a positive one I can't.

**3. PyTorch on CPU does not scale with batch size.** Throughput collapsed from 86 img/s at batch 8 to 37 img/s at batch 32 — worse than batch 1. I re-ran the entire benchmark suite to confirm it was a real behaviour and not system noise; it reproduced. ONNX Runtime held a flat ~68 img/s at every batch size.

### The method

Every number in this repo follows the same discipline:

- **Baseline first.** No optimization claim means anything without one.
- **Correctness before speed.** A broken export is extremely fast, because wrong answers are cheap to compute. Parity against the reference model is checked before any benchmark is trusted.
- **One variable at a time.** Precision, graph optimization level, batch size, and device are each isolated so a speedup can be attributed to a cause.
- **p50, p95 and p99 — not averages.** Production SLAs live in the tail.
- **Accuracy with statistics.** Wilson confidence intervals and prediction-flip counts, not a single percentage.
- **Re-run anything surprising.** Two independent runs before believing a strange result.
- **Controlled environment.** Pinned dependencies, fixed random seeds, fixed thread count for both frameworks, warmup runs excluded from timing.

### The evidence

Every number above is reproducible from the raw measurement files committed here:

| File | What it holds |
|---|---|
| `results/summary.md` | Latency and throughput across all 7 configurations × 3 batch sizes |
| `results/accuracy.md` | Top-1 accuracy with 95% Wilson confidence intervals and flip counts |
| `results/bench_*.json` | Raw timing output per configuration |
| `results/parity_*.json` | Numerical agreement between PyTorch and each converted model |
| `results/stability/` | Two independent benchmark runs, used to confirm the batch-32 anomaly was real |

### Reproduce it

```bash
conda create -n ai-journey python=3.11
conda activate ai-journey
pip install -r requirements.txt

cd phase0-1-inference
python -m src.prepare_data                                            # download + split dataset
python -m src.bench_pytorch                                           # FP32 CPU baseline
python -m src.export_onnx                                             # export to ONNX
python -m src.verify_correctness --strict                             # parity check BEFORE benchmarking
python -m src.quantize                                                # static INT8 + calibration
python -m src.verify_correctness --model results/resnet18_quantized.onnx
python -m src.bench_onnx                                              # 4 optimization levels + INT8
python -m src.bench_pytorch_mps                                       # Apple GPU
python -m src.inspect_graphs                                          # node counts per graph
python -m src.make_summary                                            # build summary.md
python -m src.eval_accuracy --mps                                     # labeled accuracy + CIs
```

Datasets and `.onnx` artifacts are not committed — the commands above regenerate them.

### Project layout

```
phase0-1-inference/
├── observation1.md          # full write-up: results, analysis, limitations
├── src/
│   ├── config.py            # all settings in one place
│   ├── logging_utils.py     # shared logger
│   ├── prepare_data.py      # dataset download, disjoint calib/eval splits, labels
│   ├── data_utils.py        # preprocessing + label loading
│   ├── labels.py            # Imagenette WordNet IDs → ImageNet class indices
│   ├── benchmark_utils.py   # one shared timing function (p50/p95/p99/throughput)
│   ├── bench_pytorch.py     # FP32 CPU baseline
│   ├── bench_pytorch_mps.py # Apple GPU (with explicit sync)
│   ├── export_onnx.py       # PyTorch → ONNX FP32
│   ├── quantize.py          # static INT8 with calibration
│   ├── verify_correctness.py# parity vs the PyTorch reference
│   ├── bench_onnx.py        # ONNX Runtime across 4 optimization levels + INT8
│   ├── inspect_graphs.py    # node and operator counts per graph
│   ├── eval_accuracy.py     # labeled accuracy, Wilson CIs, flip counts
│   └── make_summary.py      # builds the results table
└── results/                 # committed measurements
```

---

## Roadmap

| Phase | Topic | Status |
|---|---|---|
| 0 | Production mindset: reproducible environments, config, logging, benchmark hygiene | Done |
| 1 | Inference fundamentals: ONNX, quantization, graph optimization, benchmarking | Done |
| 2 | Serving: FastAPI, model lifecycle, request validation, dynamic batching | Next |
| 3 | Deployment: Docker, load testing, p95/p99 under concurrency | Planned |
| 4 | LLM internals: attention cost, autoregressive decoding, KV-cache | Planned |
| 5 | LLM quantization: GGUF, llama.cpp, accuracy vs speed trade-offs | Planned |
| 6 | LLM serving: vLLM, PagedAttention, continuous batching, TTFT | Planned |
| 7 | Capstone: a RAG-based customer operations service, quantized and load tested | Planned |

---

## Stack

PyTorch · ONNX · ONNX Runtime · static INT8 quantization (QDQ, per-channel) · Apple MPS · NumPy · statsmodels

---

## Honest limitations

Stated here rather than left for someone else to find:

- The batch-32 speedup numbers are inflated, because the PyTorch CPU baseline degrades badly at that batch size. The fair comparison there is INT8 against FP32 ONNX (~4x), not against PyTorch (7.8x).
- Absolute accuracy is optimistic — the evaluation images likely overlap with ResNet-18's original training data, so only the *differences between models* are meaningful here, which is what was being measured.
- 500 evaluation images is enough to rule out a large accuracy drop, not a sub-1-point one.
- CPU and Apple GPU only. None of these numbers transfer to CUDA deployment.
