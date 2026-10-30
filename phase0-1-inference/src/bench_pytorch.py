import json
import torch
import torchvision.models as models
from src.config import (INPUT_SIZE, BATCH_SIZES, WARMUP_RUNS, TIMED_RUNS,
                        NUM_THREADS, RESULTS_DIR)
from src.benchmark_utils import measure
from src.logging_utils import get_logger

logger = get_logger(__name__)

def main():
    torch.set_num_threads(NUM_THREADS)
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1).eval()
    results = []
    with torch.inference_mode():
        for bs in BATCH_SIZES:
            x = torch.randn(bs, *INPUT_SIZE[1:])
            r = measure(lambda: model(x), bs, WARMUP_RUNS, TIMED_RUNS)
            logger.info(r)
            results.append(r)
    (RESULTS_DIR / "bench_pytorch.json").write_text(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()