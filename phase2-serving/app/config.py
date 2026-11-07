from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent   # phase2-serving/
REPO_ROOT = ROOT.parent                         # ai-engineer-journey/

# The Phase 1 artifact this service serves. INT8, not MPS: MPS only runs on a Mac,
# and this service is meant to be deployable.
MODEL_PATH = REPO_ROOT / "phase0-1-inference" / "results" / "resnet18_quantized.onnx"
LABELS_PATH = ROOT / "app" / "imagenet_classes.json"

RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

NUM_THREADS = 4          # same as Phase 1, so latency stays comparable

# Preprocessing. These must match the Phase 1 pipeline exactly or predictions drift.
RESIZE_SIZE = 256
CROP_SIZE = 224
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

# Boundary limits
MAX_IMAGE_BYTES = 5 * 1024 * 1024
DEFAULT_TOP_K = 5
MAX_TOP_K = 10

METRICS_WINDOW = 1000    # how many recent requests to keep for percentiles
