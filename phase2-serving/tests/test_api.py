import base64
import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.inference import preprocess
from app.main import app


def make_image_b64(width: int = 500, height: int = 375, seed: int = 0) -> str:
    rng = np.random.default_rng(seed)
    pixels = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode()


@pytest.fixture(scope="module")
def client():
    # The `with` block is what triggers lifespan, so the model actually loads.
    with TestClient(app) as test_client:
        yield test_client


def test_health_does_not_require_model(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_ready_reports_model_loaded(client):
    response = client.get("/ready")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is True


def test_predict_returns_sorted_top_k(client):
    response = client.post("/predict", json={"image_base64": make_image_b64(), "top_k": 3})
    assert response.status_code == 200

    body = response.json()
    assert len(body["predictions"]) == 3

    scores = [p["score"] for p in body["predictions"]]
    assert scores == sorted(scores, reverse=True)
    assert all(0.0 <= s <= 1.0 for s in scores)
    assert all(p["class_name"] for p in body["predictions"])


def test_predict_reports_timing_breakdown(client):
    response = client.post("/predict", json={"image_base64": make_image_b64()})
    timing = response.json()["timing"]

    parts = timing["decode_ms"] + timing["preprocess_ms"] + timing["inference_ms"]
    assert parts <= timing["total_ms"] + 1e-6
    assert all(timing[key] > 0 for key in ("decode_ms", "preprocess_ms", "inference_ms"))


def test_invalid_base64_is_rejected(client):
    response = client.post("/predict", json={"image_base64": "not base64 at all!!"})
    assert response.status_code == 422


def test_non_image_bytes_are_rejected(client):
    payload = base64.b64encode(b"this is plain text, not an image").decode()
    response = client.post("/predict", json={"image_base64": payload})
    assert response.status_code == 422


def test_top_k_above_limit_is_rejected_before_the_model(client):
    response = client.post("/predict", json={"image_base64": make_image_b64(), "top_k": 999})
    assert response.status_code == 422


def test_missing_field_is_rejected(client):
    response = client.post("/predict", json={"top_k": 5})
    assert response.status_code == 422


def test_metrics_counts_requests(client):
    before = client.get("/metrics").json()["requests_total"]
    client.post("/predict", json={"image_base64": make_image_b64()})
    after = client.get("/metrics").json()

    assert after["requests_total"] == before + 1
    assert after["p50_ms"] > 0


@pytest.mark.parametrize("size", [(500, 375), (375, 500), (256, 256), (1024, 300)])
def test_preprocess_matches_phase1_torchvision_pipeline(size):
    """The serving path reimplements preprocessing without torch. Prove it matches.

    torch/torchvision are test-only dependencies here. The service itself must never
    import them, so the deployed image stays small.
    """
    torch = pytest.importorskip("torch")
    transforms = pytest.importorskip("torchvision.transforms")

    width, height = size
    rng = np.random.default_rng(1)
    pixels = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    image = Image.fromarray(pixels)

    reference_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    reference = reference_transform(image).unsqueeze(0).numpy()
    ours = preprocess(image)

    assert ours.shape == reference.shape == (1, 3, 224, 224)
    np.testing.assert_allclose(ours, reference, atol=1e-5)
