# AI Engineer Journey

Production inference engineering, built from scratch on a laptop.

Each phase takes one concrete problem — make a model fast, serve it, containerize it, then do the same for LLMs — and solves it with measurements instead of guesses. No training, no notebooks. The focus is everything that happens to a model *after* it has been trained.

**Status:** Phase 0, 1 and 2 complete.

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

## Phase 2 — Serving It With FastAPI

A fast model sitting in a benchmark script is not a product. This phase wraps the Phase 1 INT8 model in a real HTTP service, and checks whether the speed survives the trip.

| Metric | Result |
|---|---|
| Inference time inside the service | 4.01 ms (Phase 1 standalone benchmark: 3.8 ms — essentially the same) |
| Model's share of total request time | 72.4% (the rest is decode + preprocessing) |
| Cold-start vs steady-state decode time | 26 ms → 0.25 ms (100x), a one-time thread-pool warm-up cost |
| Concurrent request latency, load doubled (10 → 20) | +21% for the correct endpoint, **+107% for the broken one** |

**→ [Read the full write-up](phase2-serving/observation2.md)** — including the measurement that initially pointed the wrong way, and what fixed it.

### The finding that mattered most

The real experiment here was `def` vs `async def`: does it matter whether a FastAPI endpoint is declared as `async def` when the work inside it is CPU-bound model inference rather than something waiting on a network call? The theory says yes — blocking CPU work inside `async def` ties up the single event-loop thread, so concurrent requests stop overlapping and start queuing.

My first measurement said the opposite. At 10 concurrent requests, the "correct" endpoint (plain `def`, dispatched to a thread pool) measured *slower* than the deliberately wrong one (`async def`, blocking the event loop). Taken at face value, that would have meant the theory was wrong.

It wasn't. The first run included a one-time thread-pool warm-up cost that inflated that single number. Instead of trusting it, I reran both endpoints at two load levels, 10 and 20 concurrent requests, and looked at how each one *scaled*:

- The broken endpoint's latency nearly doubled when load doubled (69 ms → 143 ms) — the signature of requests being processed one at a time.
- The correct endpoint's latency barely moved (63 ms → 76 ms) — the signature of requests actually overlapping across threads.

A single measurement said one thing. A scaling trend across two load levels said the true thing. That is the same discipline Phase 1 used to confirm the batch-32 anomaly was real and not noise, now applied to a web service instead of a benchmark script.

### Other things worth knowing

- **The model loads once, at startup**, via FastAPI's `lifespan` hook — not per request. Loading it per request would have quietly erased the entire Phase 1 result by paying the model-load cost on every single call.
- **The service has zero dependency on PyTorch.** Preprocessing (resize, crop, normalize) is reimplemented using only PIL and NumPy, because torch is roughly 800 MB and is not needed to run an already-exported ONNX model. The risk of a hand-rolled reimplementation is a silent mismatch with the original pipeline, so it's tested directly against torchvision's output and matches to five decimal places. Torch now appears only in that one test, never in the deployed service.
- **`/health` and `/ready` are two different checks.** The process can be alive before the model has finished loading, and a load balancer needs to know the difference before it sends real traffic.

### Reproduce it

```bash
cd phase2-serving
python -c "
import json, torchvision.models as m
json.dump(m.ResNet18_Weights.IMAGENET1K_V1.meta['categories'], open('app/imagenet_classes.json','w'))
"                                              # one-time: class index -> class name
python -m pytest tests/ -v                    # 13 tests, including preprocessing parity vs torchvision
uvicorn app.main:app --port 8000              # then open http://localhost:8000/docs
```

### Project layout

```
phase2-serving/
├── observation2.md       # full write-up: results, analysis, limitations
├── app/
│   ├── config.py         # all settings in one place
│   ├── schemas.py        # Pydantic request/response models
│   ├── inference.py      # model wrapper + torch-free preprocessing
│   ├── main.py           # FastAPI app, lifespan, routes
│   └── logging_utils.py
└── tests/
    └── test_api.py       # API tests + preprocessing-parity check
```

---

## Roadmap

| Phase | Topic | Status |
|---|---|---|
| 0 | Production mindset: reproducible environments, config, logging, benchmark hygiene | Done |
| 1 | Inference fundamentals: ONNX, quantization, graph optimization, benchmarking | Done |
| 2 | Serving: FastAPI, model lifecycle, request validation, concurrency | Done |
| 3 | Deployment: Docker, load testing, p95/p99 under concurrency | Next |
| 4 | LLM internals: attention cost, autoregressive decoding, KV-cache | Planned |
| 5 | LLM quantization: GGUF, llama.cpp, accuracy vs speed trade-offs | Planned |
| 6 | LLM serving: vLLM, PagedAttention, continuous batching, TTFT | Planned |
| 7 | Capstone: a RAG-based customer operations service, quantized and load tested | Planned |

---

## Stack

PyTorch · ONNX · ONNX Runtime · static INT8 quantization (QDQ, per-channel) · Apple MPS · NumPy · statsmodels · FastAPI · Pydantic · Uvicorn · Pillow

---

## Honest limitations

Stated here rather than left for someone else to find:

**Phase 0 + 1:**
- The batch-32 speedup numbers are inflated, because the PyTorch CPU baseline degrades badly at that batch size. The fair comparison there is INT8 against FP32 ONNX (~4x), not against PyTorch (7.8x).
- Absolute accuracy is optimistic — the evaluation images likely overlap with ResNet-18's original training data, so only the *differences between models* are meaningful here, which is what was being measured.
- 500 evaluation images is enough to rule out a large accuracy drop, not a sub-1-point one.
- CPU and Apple GPU only. None of these numbers transfer to CUDA deployment.

**Phase 2:**
- Concurrency was tested up to 20 requests on one process, on one laptop — the scaling trend is clear, but these are not production-scale numbers.
- No authentication, rate limiting, or load balancer in front of the service. This is a serving-fundamentals exercise, not a hardened production API.
- Tested client-to-server on the same machine. Real network latency and proxies aren't part of this measurement — that's Phase 3's job, with Docker and a proper load-testing tool.
