"""Create a tiny abstract-image dataset for validator tests, never model training."""
from pathlib import Path

from PIL import Image, ImageDraw


def create_dataset(root: Path) -> Path:
    root = Path(root).resolve()
    for split in ("train", "val", "test"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
    (root / ".synthetic-fixture").write_text(
        "Abstract generated test images only. Not real lanternfly photos. Never train a detector on this fixture.\n",
        encoding="utf-8",
    )
    # Simple geometric color patches exercise image and normalized box parsing;
    # they are not intended to resemble lanternflies.
    for index, split in enumerate(("train", "val", "test")):
        image_path = root / "images" / split / f"abstract_{split}.png"
        image = Image.new("RGB", (48, 40), (245, 245, 245))
        draw = ImageDraw.Draw(image)
        draw.rectangle((8 + index, 8, 23 + index, 24), fill=(25, 70 + index * 20, 140))
        image.save(image_path)
        # One arbitrary rectangle label only tests the YOLO file syntax.
        (root / "labels" / split / f"abstract_{split}.txt").write_text("0 0.333333 0.4 0.333333 0.4\n", encoding="utf-8")
    # Confirmed-negative example syntax: the image still has an empty label file.
    negative = Image.new("RGB", (40, 32), (225, 235, 225))
    negative.save(root / "images" / "train" / "abstract_negative.png")
    (root / "labels" / "train" / "abstract_negative.txt").write_text("", encoding="utf-8")

    config = root / "dataset.yaml"
    config.write_text(
        "path: " + root.as_posix() + "\n"
        "train: images/train\nval: images/val\ntest: images/test\n"
        "names:\n  0: lanternfly\n",
        encoding="utf-8",
    )
    return config
