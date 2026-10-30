import torch
import torchvision.models as models
from src.config import INPUT_SIZE, ONNX_PATH
from src.logging_utils import get_logger

logger = get_logger(__name__)

def main():
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1).eval()
    dummy = torch.randn(*INPUT_SIZE)
    torch.onnx.export(
        model, dummy, str(ONNX_PATH),
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=17,
        dynamo=False,
    )
    logger.info("Exported to %s", ONNX_PATH)

if __name__ == "__main__":
    main()