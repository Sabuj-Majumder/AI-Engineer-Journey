import argparse, json
import numpy as np
import onnxruntime as ort
import torch
import torchvision.models as models
from statsmodels.stats.proportion import proportion_confint
from src.config import ONNX_PATH, ONNX_QUANT_PATH, RESULTS_DIR
from src.data_utils import load_split, load_labels
from src.logging_utils import get_logger

logger = get_logger(__name__)


def predict_pytorch(x, device="cpu"):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1).eval().to(device)
    with torch.inference_mode():
        out = model(x.to(device))
    return out.cpu().numpy().argmax(1)


def predict_onnx(x, path):
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    out = sess.run(None, {"input": x.numpy()})[0]
    return out.argmax(1)


def wilson_interval(k, n):
    lo, hi = proportion_confint(k, n, alpha=0.05, method="wilson")
    return float(lo), float(hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mps", action="store_true", help="also evaluate PyTorch on Apple GPU")
    args = ap.parse_args()

    x = load_split("eval")
    y = load_labels("eval").numpy()
    n = len(y)

    preds = {"pytorch_fp32": predict_pytorch(x)}
    preds["onnx_fp32"] = predict_onnx(x, ONNX_PATH)
    preds["onnx_int8"] = predict_onnx(x, ONNX_QUANT_PATH)
    if args.mps:
        preds["pytorch_mps"] = predict_pytorch(x, device="mps")

    report = {}
    for name, p in preds.items():
        correct = (p == y)
        k = int(correct.sum())
        lo, hi = wilson_interval(k, n)
        report[name] = {
            "accuracy": k / n,
            "correct": k,
            "n": n,
            "wilson_95ci": [lo, hi],
        }

    ref = preds["pytorch_fp32"]
    for name, p in preds.items():
        if name == "pytorch_fp32":
            continue
        ref_correct = (ref == y)
        this_correct = (p == y)
        report[name]["right_to_wrong"] = int(((ref_correct) & (~this_correct)).sum())
        report[name]["wrong_to_right"] = int(((~ref_correct) & (this_correct)).sum())
        report[name]["agree_with_fp32"] = float((ref == p).mean())

    logger.info(json.dumps(report, indent=2))
    (RESULTS_DIR / "accuracy.json").write_text(json.dumps(report, indent=2))

    lines = ["| model | accuracy | 95% CI | agree w/ fp32 | right\u2192wrong | wrong\u2192right |",
             "|---|---|---|---|---|---|"]
    for name, r in report.items():
        ci = f"[{r['wilson_95ci'][0]:.3f}, {r['wilson_95ci'][1]:.3f}]"
        agree = f"{r.get('agree_with_fp32', 1.0):.3f}"
        lines.append(f"| {name} | {r['accuracy']:.3f} | {ci} | {agree} | "
                      f"{r.get('right_to_wrong', '-')} | {r.get('wrong_to_right', '-')} |")
    (RESULTS_DIR / "accuracy.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
