import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import (CalibrationDataReader, QuantFormat,
                                      QuantType, quantize_dynamic, quantize_static)
from src.config import ONNX_PATH, ONNX_QUANT_PATH, RESULTS_DIR
from src.data_utils import load_split
from src.logging_utils import get_logger

logger = get_logger(__name__)

class CalibReader(CalibrationDataReader):
    def __init__(self, x: np.ndarray):
        self._it = iter([{"input": x[i:i + 1]} for i in range(len(x))])
    def get_next(self):
        return next(self._it, None)

def static_int8():
    x = load_split("calib").numpy()
    quantize_static(
        str(ONNX_PATH), str(ONNX_QUANT_PATH), CalibReader(x),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
    )
    logger.info("Static INT8 model saved: %s", ONNX_QUANT_PATH)

def dynamic_int8_experiment():
    out = RESULTS_DIR / "resnet18_dynamic_int8.onnx"
    try:
        quantize_dynamic(str(ONNX_PATH), str(out), weight_type=QuantType.QUInt8)
        sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
        sess.run(None, {"input": np.random.randn(1, 3, 224, 224).astype(np.float32)})
        logger.info("Dynamic INT8 ran OK. Benchmark it manually and compare.")
    except Exception as e:
        logger.warning("Dynamic INT8 failed (this is a finding, record it): %s", e)

if __name__ == "__main__":
    static_int8()
    dynamic_int8_experiment()