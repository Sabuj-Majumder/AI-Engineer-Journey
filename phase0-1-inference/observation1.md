# Phase 0 + 1: Making ResNet-18 Inference 4x Faster

My notes from the first two phases of my AI engineering journey. I wrote this after finishing the work, to record what I did, what I measured, and what I actually learned from the numbers.

**Short version:** I took a pretrained ResNet-18 and cut its inference latency by 77% (16.9 ms to 3.8 ms per image) and its file size by 4x (45 MB to 11 MB), with no measurable loss in accuracy. More importantly, I now have a repeatable method for doing this to any model, and I can explain why each optimization worked or did not work.

---

## 1. Goal of this phase

I already knew how to use a model inside a notebook. What I did not know was what happens to a model *after* the notebook. How do you make it fast? How do you prove it is still correct after you change it? How do you decide which optimization is actually worth doing?

So the goal here was not to build a classifier. The goal was to take one model I did not train (ResNet-18, pretrained on ImageNet) and answer three questions using real measurements instead of guesses:

1. How fast is it, really?
2. Can I make it significantly faster?
3. Does it still give the right answers after I make it faster?

The target I set at the start was to cut inference time in half. I ended up cutting it by 77%.

**Hardware:** MacBook Pro with M1 Pro chip. No NVIDIA GPU. Everything here runs on my personal laptop.

---

## 2. Phase 0: the rules I set before measuring anything

This part produced no results, and that was the point. Before I benchmarked anything, I made sure the numbers would actually mean something.

- **One reproducible environment.** A Conda env on Python 3.11, arm64 build (verified, because an x86 build running under emulation would have silently made every number wrong). Dependencies in `requirements.txt`, exact versions pinned in `requirements.lock.txt`.
- **All settings in one file.** `config.py` holds the batch sizes, warmup count, timed-run count, thread count, and paths. Nothing hardcoded in the middle of a script where I would forget about it.
- **Logging instead of print.** Every script logs through one shared `get_logger()`.
- **Fixed thread count.** `NUM_THREADS = 4` for both PyTorch and ONNX Runtime. If one framework quietly used more cores than the other, the comparison would be meaningless.
- **Benchmark hygiene.** Laptop plugged in (on battery macOS throttles the CPU), other apps closed, 50 warmup runs before every measurement, then 500 timed runs.
- **Report p50, p95 and p99, not just the average.** In production nobody is paged about the average. They are paged about the slow 5%.
- **Change one variable at a time.** This turned out to be the single most useful habit of the whole phase.

The warmup runs matter more than I expected. The first few inference calls are always slower because of lazy initialization and cold caches. If I had included them in the timing, every result would have been polluted.

---

## 3. What I did, step by step

The order here is deliberate. A couple of these steps are in this position because doing them in a different order gives you wrong answers.

1. **Downloaded the dataset.** Imagenette2-160, a 10-class subset of ImageNet. I split it into 100 calibration images and 500 evaluation images, kept strictly separate, selected with a fixed random seed so the split is identical every time.
2. **Downloaded the pretrained ResNet-18 weights.** I did not train anything. Somebody else already trained this on ImageNet, and the whole phase is about serving it, not teaching it.
3. **Benchmarked PyTorch FP32 on CPU.** This is the baseline. Every later number is meaningless without it.
4. **Exported the model to ONNX** (opset 17, dynamic batch axis).
5. **Verified the ONNX output matched PyTorch, before benchmarking it.** This is the step people skip. A broken export can be extremely fast, because wrong answers are cheap to compute. Speed means nothing until correctness is confirmed.
6. **Benchmarked ONNX Runtime at four graph optimization levels** (disable all, basic, extended, all).
7. **Quantized the ONNX model to static INT8** (QDQ format, per-channel, calibrated on the 100 calibration images).
8. **Verified the INT8 model again**, this time expecting some drift instead of an exact match.
9. **Benchmarked PyTorch on MPS** (the Apple GPU). This needed `torch.mps.synchronize()` inside the timing function, because GPU calls are asynchronous and without it I would have been timing how fast Python can queue work, not how fast the GPU finishes it.
10. **Inspected the graphs** to count nodes and operator types, so I could explain the results instead of just reporting them.
11. **Measured real accuracy against ground-truth labels** on the 500 evaluation images, with confidence intervals.
12. **Re-ran the benchmarks a second time** to check that the strange results were real and not noise.

Two ordering lessons I got wrong at first and had to correct:

- **Quantization happens after the ONNX export, and it operates on the ONNX file.** Not before, not on the PyTorch model.
- **Correctness check comes before the speed check.** Always.

---

## 4. Results

### 4.1 Latency, p50 in milliseconds (lower is better)

| Config | Batch 1 | Batch 8 | Batch 32 |
|---|---|---|---|
| PyTorch CPU (FP32) | 16.9 | 91.0 | 843.5 |
| ONNX, no optimization | 15.6 | 126.0 | 447.2 |
| ONNX, basic | 15.6 | 114.3 | 439.1 |
| ONNX, extended | 14.8 | 112.4 | 453.2 |
| ONNX, all | 14.6 | 111.4 | 444.9 |
| **ONNX INT8 (static)** | **3.8** | **27.3** | **108.4** |
| PyTorch MPS (Apple GPU) | 4.1 | 16.0 | 54.6 |

### 4.2 Throughput, samples per second (higher is better)

| Config | Batch 1 | Batch 8 | Batch 32 |
|---|---|---|---|
| PyTorch CPU (FP32) | 57.4 | 86.3 | 36.8 |
| ONNX, all | 68.4 | 68.5 | 69.8 |
| **ONNX INT8 (static)** | **258.3** | **292.7** | **293.0** |
| PyTorch MPS (Apple GPU) | 242.3 | 499.7 | 582.8 |

### 4.3 Accuracy on 500 labeled images

Top-1 accuracy against the true ImageNet label, with a 95% Wilson confidence interval. "Agreement" is how often the model picked the same class as the PyTorch FP32 reference. "Flips" shows how many predictions changed from right to wrong, and from wrong to right.

| Model | Accuracy | 95% CI | Agreement with FP32 | Flips (right to wrong / wrong to right) |
|---|---|---|---|---|
| PyTorch FP32 (reference) | 67.6% | 63.4% - 71.6% | - | - |
| ONNX FP32 | 67.6% | 63.4% - 71.6% | 100% | 0 / 0 |
| ONNX INT8 (static) | 68.6% | 64.4% - 72.5% | 96.6% | 2 / 7 |
| PyTorch MPS | 67.6% | 63.4% - 71.6% | 100% | 0 / 0 |

### 4.4 Model size

| Model | Size |
|---|---|
| ONNX FP32 | 45 MB |
| ONNX INT8 | 11 MB |

### 4.5 What the graph looks like inside

| Graph | Nodes | Main operators |
|---|---|---|
| Raw ONNX export | 49 | Conv 20, Relu 17, Add 8 |
| After "basic" optimization | 49 | identical to raw |
| After "extended" | 40 | Conv 11, FusedConv 9, Relu 8, Add 8 |
| After "all" | 24 | FusedConv 17, Conv 3, Gemm 1 |
| INT8 static | 36 | QLinearConv 20, QLinearAdd 8 |

---

## 5. What I found, and why it happened

### Finding 1: INT8 quantization was the entire win

At batch 1, static INT8 took 3.8 ms against the PyTorch baseline's 16.9 ms. That is **4.4x faster, a 77% cut in latency**, and the model file went from 45 MB to 11 MB. Compared fairly against the best FP32 ONNX config, INT8 is still about **4x faster at every single batch size**.

Why it works: storing weights and activations as 8-bit integers instead of 32-bit floats means four times less data has to move between memory and the CPU for the same computation, and integer arithmetic is cheaper than floating-point arithmetic. For a convolutional network on a CPU, data movement is a serious part of the cost, so cutting it by 4x shows up directly in the latency.

### Finding 2: Graph optimization did almost nothing, and I found out exactly why

This was the result I did not expect. All four optimization levels landed within noise of each other, even though the node count dropped from 49 to 24 at the highest level. Cutting the graph in half and getting no speedup looks like something is broken.

Inspecting the graphs explained it. **The raw exported model has no BatchNorm nodes at all.** PyTorch's ONNX exporter already folds BatchNorm into the preceding Conv layer during export, because in eval mode BatchNorm is just a fixed linear transform that can be merged into the convolution's weights. So the biggest, most famous graph optimization had already happened before ONNX Runtime ever saw the model. That is also why the "basic" level produced a byte-identical node count to the raw export. It had nothing left to do.

What the higher levels did find was Conv + ReLU fusion, merging those pairs into single `FusedConv` nodes (9 of them at "extended", all 17 at "all"). But fusion mainly removes **per-node overhead**, things like function call cost and memory allocation between operators. It does not reduce the actual number of multiply-add operations. ResNet-18 is dominated by convolution arithmetic, so removing overhead from a workload whose cost is almost entirely arithmetic changes very little.

This is the thing I am happiest about learning in this phase: a negative result that I can fully explain is more valuable than a positive result I cannot. Graph fusion and precision reduction are two completely different kinds of optimization, and for this model only one of them mattered.

### Finding 3: PyTorch on CPU does not scale with batch size

PyTorch's throughput went 57 samples/sec at batch 1, up to 86 at batch 8, then **collapsed to 37 at batch 32**, which is worse than batch 1. Tail latency went bad too, with p99 reaching about 1.2 seconds.

I thought this was a fluke, so I re-ran the whole benchmark a second time with the laptop plugged in and other apps closed. It reproduced almost exactly (p50 843 ms then 856 ms, p99 1181 ms then 1282 ms). So it is a real, consistent behaviour and not a random system hiccup. The likely cause is that the fixed 4-thread setting stops being enough at batch 32, or that batch 32 saturates memory bandwidth. Confirming that needs a thread sweep, which is on my list.

Meanwhile ONNX Runtime held a flat 68 to 70 samples/sec across every batch size. Its execution engine degrades gracefully where PyTorch's does not.

### Finding 4: Converting to ONNX by itself is not a speedup

Plain ONNX FP32 was only about 1.1x faster than PyTorch at batch 1, and actually **slower** than PyTorch at batch 8 (68 samples/sec against 86). This also reproduced across both runs, so it is a real result.

The lesson is that "export to ONNX" is not an optimization on its own. It is the step that makes real optimizations possible, because quantization and graph-level work operate on that format. The value of the ONNX conversion here was the 4x INT8 model it unlocked, not the conversion itself.

### Finding 5: the conversions were numerically exact

Both ONNX FP32 and MPS agreed with the PyTorch reference on **100% of 500 images, with zero flipped predictions**. The maximum absolute difference in raw output was 5.5e-05. This is the boring result, and it is the one that makes every other number in this document trustworthy.

### Finding 6: INT8 cost me no measurable accuracy

INT8 scored 68.6% against FP32's 67.6%. It would be very easy, and wrong, to write "quantization improved accuracy" here.

The confidence intervals are 63.4% - 71.6% and 64.4% - 72.5%. They overlap almost completely, so with 500 samples this difference cannot be distinguished from random variation. Looking at the individual predictions tells the real story: 9 images out of 500 changed their answer, 2 going from right to wrong and 7 going from wrong to right. Quantization adds small numerical noise, which pushes borderline predictions around in both directions. This time slightly more of them landed on the correct side.

The honest conclusion is: **INT8 gave a 4x speedup at no measurable accuracy cost on this evaluation set.** Not that it made the model better.

### Finding 7: the Apple GPU scales the best, but it is not a deployment path

MPS throughput grew with batch size the way you want it to: 242, then 500, then 583 samples/sec. At batch 1 it roughly ties INT8 on CPU, and at larger batches it is about 2x ahead. It is also exactly as accurate as the CPU reference. But it only runs on Apple hardware, so for me this is a learning comparison rather than something I would ship.

---

## 6. What I achieved

- **77% latency reduction** (16.9 ms to 3.8 ms per image), which is 4.4x faster. The goal I set was 2x.
- **4x smaller model** (45 MB to 11 MB), which matters for container image size, cold start time, and memory footprint.
- **No measurable accuracy loss**, demonstrated with ground-truth labels and confidence intervals rather than a single number.
- **A verified-correct conversion pipeline**, with 100% prediction agreement on 500 images before any speed claim was made.
- **An explanation for every result, including the ones that did not work.** I can say why graph fusion gave nothing on this model, and when it would have helped.
- **Reproducible measurements**, with fixed seeds, pinned dependencies, fixed thread counts, warmup runs, and a full second run to confirm the surprising results held up.

The deeper gain is the method, not the number. The sequence I used here does not depend on ResNet at all:

> establish a baseline, convert the model, **prove correctness before trusting any speed number**, change one variable at a time, measure with percentiles rather than averages, validate accuracy against real labels with statistics, and re-run anything surprising before believing it.

That applies to a CNN, a transformer, or an LLM. The specific tools change. The discipline does not.

---

## 7. Limitations I am aware of

I would rather state these than have someone find them.

- **The batch-32 speedup numbers are inflated.** Because the PyTorch CPU baseline collapses at batch 32, anything compared against it there looks better than it is. The fair comparison at that batch size is INT8 against FP32 ONNX, which is about 4x, not the 7.8x that a naive comparison against PyTorch suggests.
- **The absolute accuracy figure is optimistic.** The Imagenette validation images are very likely part of the data ResNet-18 was originally trained on, so 67.6% should not be read as a clean generalization score. Only the differences between models are meaningful here, and the differences are what I was measuring.
- **500 evaluation images is a small sample.** It is enough to show that INT8 did not cause a large accuracy drop. It is not enough to detect a drop smaller than roughly one percentage point.
- **This is CPU and Apple GPU only.** No NVIDIA GPU was involved, so none of these numbers transfer to CUDA deployment.
- **The batch-32 PyTorch bottleneck is confirmed but not root-caused.** I know it is real and reproducible. I have not yet proven whether it is thread count or memory bandwidth.

---

## 8. What I would do differently on the next model

- Check whether the exporter already applied the obvious graph optimizations **before** spending time benchmarking optimization levels.
- Sweep the thread count instead of fixing it at 4, since that one setting plausibly explains an entire bad result.
- Measure the full request path, including image preprocessing, not just the model call. Preprocessing can easily be a larger cost than a 3.8 ms inference.
- Keep the correctness check in the pipeline permanently, not as a one-time step.

---

*Next phase: wrapping this optimized model in a real serving layer (FastAPI), then containerizing and load testing it, before moving on to LLM-specific inference work.*
