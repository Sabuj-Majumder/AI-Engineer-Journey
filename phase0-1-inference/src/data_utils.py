import json
import torch
import torchvision.transforms as T
from PIL import Image
from src.config import ROOT
from src.labels import IMAGENETTE_WNID_TO_IMAGENET_IDX

DATA_DIR = ROOT / "data" / "images"
_tf = T.Compose([
    T.Resize(256), T.CenterCrop(224), T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def load_split(split: str) -> torch.Tensor:
    files = sorted((DATA_DIR / split).glob("*.jpg"))
    if not files:
        raise FileNotFoundError("No images found. Run: python -m src.prepare_data")
    return torch.stack([_tf(Image.open(f).convert("RGB")) for f in files])

def load_labels(split: str) -> torch.Tensor:
    label_file = DATA_DIR / split / "labels.json"
    if not label_file.exists():
        raise FileNotFoundError("No labels found. Re-run: python -m src.prepare_data")
    labels = json.loads(label_file.read_text())
    ordered_names = sorted(labels.keys())
    return torch.tensor([IMAGENETTE_WNID_TO_IMAGENET_IDX[labels[name]] for name in ordered_names])
