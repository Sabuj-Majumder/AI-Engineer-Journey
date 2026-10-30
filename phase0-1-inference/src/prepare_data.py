import json, random, shutil, tarfile, urllib.request
from src.config import ROOT
from src.logging_utils import get_logger

logger = get_logger(__name__)
URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz"
DATA = ROOT / "data"

def main():
    DATA.mkdir(exist_ok=True)
    archive = DATA / "imagenette2-160.tgz"
    raw = DATA / "raw"
    if not archive.exists():
        logger.info("Downloading Imagenette (~100 MB)...")
        urllib.request.urlretrieve(URL, archive)
    if not raw.exists():
        with tarfile.open(archive) as tar:
            try:
                tar.extractall(raw, filter="data")
            except TypeError:
                tar.extractall(raw)
    files = sorted(raw.rglob("val/*/*.JPEG"))
    random.Random(0).shuffle(files)
    for split, subset in (("calib", files[:100]), ("eval", files[100:600])):
        out = DATA / "images" / split
        out.mkdir(parents=True, exist_ok=True)
        labels = {}
        for i, f in enumerate(subset):
            name = f"{i:03d}.jpg"
            shutil.copy(f, out / name)
            labels[name] = f.parent.name  # WordNet ID folder name, e.g. n01440764
        (out / "labels.json").write_text(json.dumps(labels, indent=2))
        logger.info("%s: %d images", split, len(subset))

if __name__ == "__main__":
    main()
