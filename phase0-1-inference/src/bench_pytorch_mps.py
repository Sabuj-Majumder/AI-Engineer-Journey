import json
import torch
import torchvision.models as models
from src.config import INPUT_SIZE, BATCH_SIZES, WARMUP_RUNS, TIMED_RUNS, RESULTS_DIR
from src.benchmark_utils import measure
from src.logging_utils import get_logger

logger = get_logger(__name__)

def main():
    if not torch.backends.mps.is_available():
        raise SystemExit("MPS not available in this PyTorch build.")
    device = torch.device("mps")
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1).eval().to(device)

    results = []
    with torch.inference_mode():
        for bs in BATCH_SIZES:
            x = torch.randn(bs, *INPUT_SIZE[1:], device=device)

            def run():
                model(x)
                torch.mps.synchronize()   # without this, you time only the launch

            r = measure(run, bs, WARMUP_RUNS, TIMED_RUNS)
            logger.info(r)
            results.append(r)
    (RESULTS_DIR / "bench_pytorch_mps.json").write_text(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()