import json
import numpy as np
import onnxruntime as ort
from src.config import (INPUT_SIZE, BATCH_SIZES, WARMUP_RUNS, TIMED_RUNS,
                        NUM_THREADS, RESULTS_DIR, ONNX_PATH, ONNX_QUANT_PATH)
from src.benchmark_utils import measure
from src.logging_utils import get_logger

logger = get_logger(__name__)
GO = ort.GraphOptimizationLevel
GRAPH_DIR = RESULTS_DIR / "graphs"
GRAPH_DIR.mkdir(exist_ok=True)

CONFIGS = {
    "onnx_no_opt":      (ONNX_PATH, GO.ORT_DISABLE_ALL),
    "onnx_basic":       (ONNX_PATH, GO.ORT_ENABLE_BASIC),
    "onnx_extended":    (ONNX_PATH, GO.ORT_ENABLE_EXTENDED),
    "onnx_all":         (ONNX_PATH, GO.ORT_ENABLE_ALL),
    "onnx_int8_static": (ONNX_QUANT_PATH, GO.ORT_ENABLE_ALL),
}

def make_session(path, level, name):
    so = ort.SessionOptions()
    so.graph_optimization_level = level
    so.intra_op_num_threads = NUM_THREADS
    so.inter_op_num_threads = 1
    if level != GO.ORT_DISABLE_ALL:
        so.optimized_model_filepath = str(GRAPH_DIR / f"{name}.onnx")
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])

def main():
    for name, (path, level) in CONFIGS.items():
        if not path.exists():
            logger.warning("Skipping %s (missing %s)", name, path.name)
            continue
        sess = make_session(path, level, name)
        results = []
        for bs in BATCH_SIZES:
            x = np.random.randn(bs, *INPUT_SIZE[1:]).astype(np.float32)
            r = measure(lambda: sess.run(None, {"input": x}), bs, WARMUP_RUNS, TIMED_RUNS)
            logger.info("%s %s", name, r)
            results.append(r)
        (RESULTS_DIR / f"bench_{name}.json").write_text(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()