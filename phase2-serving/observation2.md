# Phase 2: Serving the Model with FastAPI

My notes from the second phase of my AI engineering journey. Phase 1 made a model fast. This phase turns that model into something another program can actually call over HTTP, and measures what happens to the speed once it is wrapped in a real service.

**Short version:** I wrapped the Phase 1 INT8 model in a FastAPI service, and the model still runs in about 4 ms inside the service, almost exactly matching the 3.8 ms I measured in Phase 1. But the two framework choices I had to make (how the model loads, and whether an endpoint is written as `def` or `async def`) turned out to matter more than I expected. One of them I got right on paper but still had to prove with numbers before I believed it myself.

---

## 1. Goal of this phase

A fast model sitting in a Python script is not a product. Someone has to be able to send it a request and get an answer back, and the service around the model has to not throw away the speed I worked for in Phase 1.

So this phase answers three questions:

1. How do I turn the Phase 1 model into something callable over the network, without reloading it on every request?
2. What actually happens when more than one request arrives at the same time?
3. Does the 3.8 ms I measured in Phase 1 survive once there is a real request path around it (decoding an image, validating input, returning JSON)?

The third question turned out to have the most interesting answer, and the second one is where I was most surprised by my own first measurement.

**Hardware:** same as Phase 1, MacBook Pro M1 Pro, everything local, no GPU involved in this phase.

---

## 2. What I built

A small FastAPI service with four endpoints:

- `GET /health` — is the process alive
- `GET /ready` — is the model actually loaded and able to serve a request
- `POST /predict` — send a base64-encoded image, get back the top-k predicted classes
- `GET /metrics` — running request count and latency percentiles

Two design decisions I made on purpose, worth recording because they are not obvious to someone new to serving:

**The model loads once, at startup, not per request.** FastAPI has a `lifespan` hook that runs code once when the server starts and once when it shuts down. I load the ONNX session there, not inside the `/predict` function. If I had loaded it per request, every single prediction would pay the cost of reading an 11 MB file off disk and constructing a new inference session, which would have quietly destroyed the entire Phase 1 result.

**The service does not depend on PyTorch.** Phase 1's preprocessing used `torchvision.transforms`, but torch is roughly 800 MB. Pulling that into a service that only needs to run a quantized ONNX model made no sense, so I rewrote the exact same resize, center-crop, and normalize steps using only PIL and NumPy. The risk with rewriting it by hand is that a small mistake would silently change every prediction, so I wrote a test that runs both pipelines on the same image and checks the output tensors match to five decimal places. They do. Torch only appears in the test file now, never in the service itself.

`/health` and `/ready` are deliberately two separate checks. Right after the server process starts, it is alive but the model has not finished loading yet, so it cannot serve a correct prediction. A load balancer or orchestrator needs to know that difference before it sends real traffic to the instance.

---

## 3. Results

### 3.1 Where a request's time actually goes

I sent 500 random test requests before measuring — the standard warm-up discipline from Phase 1 — then averaged 50 timed requests through the real `/predict` endpoint with a real evaluation image:

| Stage | Mean time | Share of total |
|---|---|---|
| Decode (base64 → image) | 0.25 ms | 4.5% |
| Preprocess (resize, crop, normalize) | 1.21 ms | 21.8% |
| **Model inference** | **4.01 ms** | **72.4%** |
| **Total** | **5.54 ms** | 100% |

The inference number, 4.01 ms, lines up almost exactly with the 3.8 ms I measured for the same INT8 model back in Phase 1. That match matters: it means the serving layer (FastAPI routing, Pydantic validation, JSON parsing) is not adding meaningful overhead of its own. The model is still the model.

What is new here is the other 1.5 ms. The model is 72% of the request, not 100%. Preprocessing alone costs more than half of what decoding and JSON handling cost combined. On its own this is a small number, but it is the kind of thing that stops being small the moment someone tries to make inference even faster without touching the code around it.

### 3.2 The first request is not representative

The very first request after startup reported a decode time of 26 ms. Fifty requests later, the steady-state average for the same step was 0.25 ms — a 100x difference for the exact same code path on the exact same kind of input.

This is not a bug, it is a one-time cost: the first call into FastAPI's thread pool pays for thread creation and some lazy initialization that every later call skips. I nearly took that first number at face value before running the averaged test. It is the same lesson Phase 1 taught with warm-up runs, just showing up in a new place.

### 3.3 `def` vs `async def` under concurrent load

This was the actual experiment of the phase: does writing a FastAPI endpoint as `async def` instead of plain `def` matter when the work inside it is CPU-bound model inference, not waiting on a network call?

I built two identical endpoints, one correct (`/predict`, plain `def`, so FastAPI runs it in a worker thread) and one deliberately wrong (`/predict-blocking`, `async def`, so it runs directly on the single event loop thread with nothing to hand control back during the blocking inference call). Then I fired concurrent requests at both and measured total wall-clock time, first at 10 concurrent requests, then at 20, to see how each one scales with load:

| Concurrent requests | `/predict` (def) | `/predict-blocking` (async def) |
|---|---|---|
| 10 | 63 ms | 69 ms |
| 20 | 76 ms | 143 ms |
| **Time when load doubled** | **+21%** | **+107%** |

My very first single measurement at 10 requests actually showed the *wrong* endpoint looking faster (80 ms vs 66 ms), which would have led me to the wrong conclusion if I had trusted it. Looking closer, that run included a 95 ms outlier from the same thread-pool warm-up cost described above. Once I repeated the test and looked at how each endpoint scaled from 10 to 20 requests instead of trusting a single number, the real pattern appeared: `/predict-blocking`'s time almost exactly doubled when I doubled the load, while `/predict`'s barely moved.

Doubling load and getting double the time is the signature of serial, one-at-a-time processing. Doubling load and getting almost no change is the signature of genuine concurrency. The numbers confirm it exactly: the "wrong" endpoint processes requests one after another regardless of how many arrive together, and the "correct" one spreads them across threads and actual CPU cores.

### 3.4 One real prediction, for a sanity check

I ran one actual evaluation image through the live service. It predicted "monastery" as the top class; the true label was "church," which showed up fourth in the top-5 list at 1.9% confidence, behind monastery, vault, and organ. All four of those classes are architecturally similar (arches, columns, interior halls), and this kind of confusion between visually close ImageNet classes is expected, not a bug — it is consistent with the 67.6% top-1 accuracy I already measured with a 500-image labeled test in Phase 1.

---

## 4. What I found, and why

### Finding 1: the model's speed survived being wrapped in a service

Inference inside the FastAPI service measured 4.01 ms, against 3.8 ms for the same model running directly in Phase 1's benchmark script. That is close enough to call it the same number once normal measurement noise is accounted for. The work of Phase 1 did not get undone by Phase 2 — the service is a thin, low-overhead wrapper around it, which is exactly what a serving layer should be.

### Finding 2: the model is not the whole request

72% of the request's time is the model. The other 28% is decode and preprocessing, mostly preprocessing. I had read this as a general warning in textbooks before, but seeing the actual 1.21 ms number sitting next to the 4.01 ms model number made it concrete: if I ever squeeze another millisecond out of the model, I will have made the total request only marginally faster, because a quarter of the cost was never in the model to begin with.

### Finding 3: a single measurement nearly led me to the wrong conclusion, and scaling trends fixed it

This is the finding I think matters most from this phase, more than the final concurrency numbers themselves. My first test of `def` vs `async def` showed the theoretically-wrong endpoint performing *better*, which is the opposite of what the underlying theory predicts. It would have been easy to report that result and move on.

Instead of trusting one number, I reran the same test at two different load levels and looked at how each endpoint's latency scaled, not just what it measured once. That is what exposed the truth: one endpoint's latency doubled when load doubled (serial execution), and the other's barely moved (concurrent execution). The scaling trend was the real signal; the single-point measurement was mostly noise from a one-time warm-up cost. This is the same discipline from Phase 1 — re-run anything surprising before believing it — applied to a completely different kind of system.

### Finding 4: `def` vs `async def` is invisible until you load-test it

Under light or single-request load, both endpoints look identical, because there is no contention for either of them to reveal. The difference only exists under concurrent load, and it only grows with more concurrent load. A service that is never load-tested could ship this exact bug — an `async def` endpoint quietly serializing every CPU-bound request — and nobody would notice until real traffic arrived.

---

## 5. What I achieved

- **A working inference service**, with the Phase 1 quantized model loaded once at startup and served over HTTP, model inference time inside the service matching Phase 1's standalone benchmark to within normal noise.
- **A torch-free serving path**, numerically verified against the original torchvision preprocessing pipeline, so the deployed service does not carry an 800 MB dependency it does not need.
- **A full request-latency breakdown**, not just a model number — proving the model is 72% of request time, not 100%, with the remainder measured and accounted for instead of assumed.
- **A reproduced, load-tested proof** that blocking CPU work inside `async def` serializes requests under concurrency, shown through scaling behavior across two load levels rather than a single comparison that could have been misread.
- **Boundary validation** that rejects malformed input with a clear error before it reaches the model, instead of letting bad input crash or silently misbehave inside the service.

The deeper lesson carries forward from Phase 1: don't trust a number you only measured once, especially when it surprises you. In Phase 1 that meant re-running a benchmark. Here it meant re-running a concurrency test at a second load level before accepting what the first one seemed to say.

---

## 6. Limitations I am aware of

- **Concurrency was tested up to 20 requests on one process, on one laptop.** The scaling trend is clear, but these are not production-scale numbers, and I have not tested what happens when the thread pool itself is exhausted.
- **The timing breakdown used one image repeatedly.** Decode and preprocessing time will vary with image size and format in ways this test does not cover.
- **No authentication, rate limiting, or request size enforcement beyond a basic byte limit.** This is a serving-fundamentals exercise, not a hardened production API.
- **Tested with `TestClient` and a local `httpx` client, both on the same machine as the server.** Real network latency, proxies, and load balancers are not part of this measurement.

---

## 7. What I would carry into the next phase

- Load-test any endpoint before assuming `async def` is automatically correct. It only pays off for I/O-bound work; CPU-bound work needs the thread pool.
- When a measurement surprises me, check how it scales before trusting it, not just whether it repeats.
- Preprocessing cost is worth tracking as its own line item going forward, not folded into "everything that isn't the model."

---

*Next phase: Docker (containerizing this service) and load testing it properly under real concurrent traffic with a tool built for it, comparing the quantized model against the unquantized one under load.*
