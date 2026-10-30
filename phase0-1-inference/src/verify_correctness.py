import argparse, json
from pathlib import Path
import numpy as np
import onnxruntime as ort
import torch
import torchvision.models as models
from src.config import ONNX_PATH, RESULTS_DIR
from src.data_utils import load_split
from src.logging_utils import get_logger

logger = get_logger(__name__)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ONNX_PATH))
    ap.add_argument("--atol", type=float, default=1e-4)
    ap.add_argument("--strict", action="store_true",
                    help="fail if outputs differ more than atol (use for FP32 export)")
    args = ap.parse_args()

    x = load_split("eval")
    ref_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1).eval()
    with torch.inference_mode():
        ref = ref_model(x).numpy()

    sess = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    out = sess.run(None, {"input": x.numpy()})[0]

    diff = np.abs(ref - out)
    result = {
        "model": Path(args.model).name,
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "top1_agreement_with_fp32": float((ref.argmax(1) == out.argmax(1)).mean()),
        "num_images": int(len(x)),
    }
    logger.info(result)
    (RESULTS_DIR / f"parity_{Path(args.model).stem}.json").write_text(json.dumps(result, indent=2))
    if args.strict:
        assert np.allclose(ref, out, atol=args.atol), "ONNX output diverges from PyTorch!"
        logger.info("PASSED strict parity check")

if __name__ == "__main__":
    main()