import base64
import binascii
import io
import json
import time

import numpy as np
import onnxruntime as ort
from PIL import Image, UnidentifiedImageError

from app.config import (CROP_SIZE, LABELS_PATH, MAX_IMAGE_BYTES, MEAN, MODEL_PATH,
                        NUM_THREADS, RESIZE_SIZE, STD)
from app.logging_utils import get_logger

logger = get_logger(__name__)

_MEAN = np.array(MEAN, dtype=np.float32)
_STD = np.array(STD, dtype=np.float32)


class BadImageError(ValueError):
    """Raised when the client sent something that is not a usable image."""


def decode_image(image_base64: str) -> Image.Image:
    try:
        raw = base64.b64decode(image_base64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise BadImageError("image_base64 is not valid base64") from exc

    if len(raw) > MAX_IMAGE_BYTES:
        raise BadImageError(f"image exceeds {MAX_IMAGE_BYTES} bytes")

    try:
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise BadImageError("bytes could not be decoded as an image") from exc


def _resize_shorter_side(img: Image.Image, size: int) -> Image.Image:
    """Match torchvision.transforms.Resize(size) for a PIL input."""
    w, h = img.size
    short, long = (w, h) if w <= h else (h, w)
    if short == size:
        return img
    new_short, new_long = size, int(size * long / short)
    new_w, new_h = (new_short, new_long) if w <= h else (new_long, new_short)
    return img.resize((new_w, new_h), Image.BILINEAR)


def _center_crop(img: Image.Image, crop: int) -> Image.Image:
    w, h = img.size
    left = int(round((w - crop) / 2.0))
    top = int(round((h - crop) / 2.0))
    return img.crop((left, top, left + crop, top + crop))


def preprocess(img: Image.Image) -> np.ndarray:
    """PIL image -> normalized NCHW float32 batch of 1.

    Deliberately reimplements the Phase 1 torchvision pipeline using only PIL and
    NumPy, so that serving does not depend on torch. tests/test_api.py asserts
    this produces the same tensor torchvision would.
    """
    img = _resize_shorter_side(img, RESIZE_SIZE)
    img = _center_crop(img, CROP_SIZE)
    arr = np.asarray(img, dtype=np.float32) / 255.0      # HWC in [0, 1]
    arr = (arr - _MEAN) / _STD
    arr = arr.transpose(2, 0, 1)[np.newaxis, ...]        # NCHW
    return np.ascontiguousarray(arr, dtype=np.float32)


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max()
    exp = np.exp(shifted)
    return exp / exp.sum()


class Classifier:
    """Owns the ONNX session. Created once at startup, reused by every request."""

    def __init__(self) -> None:
        self.session: ort.InferenceSession | None = None
        self.input_name: str | None = None
        self.categories: list[str] = []

    @property
    def is_loaded(self) -> bool:
        return self.session is not None

    def load(self) -> None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. Build it first:\n"
                f"  cd {MODEL_PATH.parents[1]} && python -m src.export_onnx && python -m src.quantize"
            )
        if not LABELS_PATH.exists():
            raise FileNotFoundError(
                f"Class names not found at {LABELS_PATH}. Generate them once with the "
                f"python -c snippet in the Phase 2 setup steps."
            )

        options = ort.SessionOptions()
        options.intra_op_num_threads = NUM_THREADS
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(MODEL_PATH), options, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.categories = json.loads(LABELS_PATH.read_text())
        logger.info("Loaded %s (%d classes)", MODEL_PATH.name, len(self.categories))

    def predict(self, image_base64: str, top_k: int) -> tuple[list[dict], dict]:
        """Run one request end to end and report where the time went."""
        t_start = time.perf_counter()

        img = decode_image(image_base64)
        t_decoded = time.perf_counter()

        batch = preprocess(img)
        t_preprocessed = time.perf_counter()

        logits = self.session.run(None, {self.input_name: batch})[0][0]
        t_inferred = time.perf_counter()

        probs = softmax(logits)
        top_indices = np.argsort(probs)[::-1][:top_k]
        predictions = [
            {
                "class_index": int(i),
                "class_name": self.categories[int(i)],
                "score": float(probs[int(i)]),
            }
            for i in top_indices
        ]

        timing = {
            "decode_ms": (t_decoded - t_start) * 1000,
            "preprocess_ms": (t_preprocessed - t_decoded) * 1000,
            "inference_ms": (t_inferred - t_preprocessed) * 1000,
            "total_ms": (time.perf_counter() - t_start) * 1000,
        }
        return predictions, timing
