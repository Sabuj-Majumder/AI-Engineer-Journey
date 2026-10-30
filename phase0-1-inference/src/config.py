from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

MODEL_NAME = "resnet18"
INPUT_SIZE = (1, 3, 224, 224)
BATCH_SIZES = [1, 8, 32]
WARMUP_RUNS = 50
TIMED_RUNS = 500
NUM_THREADS = 4          # fixed so PyTorch and ONNX Runtime are compared fairly

ONNX_PATH = RESULTS_DIR / "resnet18.onnx"
ONNX_QUANT_PATH = RESULTS_DIR / "resnet18_quantized.onnx"