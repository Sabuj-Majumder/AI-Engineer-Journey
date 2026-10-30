import time
import numpy as np

def measure(fn, batch_size: int, warmup: int, runs: int) -> dict:
    for _ in range(warmup):
        fn()
    lat = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        lat.append((time.perf_counter() - t0) * 1000)
    a = np.array(lat)
    return {
        "batch_size": batch_size,
        "mean_ms": float(a.mean()),
        "p50_ms": float(np.percentile(a, 50)),
        "p95_ms": float(np.percentile(a, 95)),
        "p99_ms": float(np.percentile(a, 99)),
        "throughput_samples_per_sec": float(batch_size / (a.mean() / 1000)),
    }