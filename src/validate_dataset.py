"""Validate a YOLOv8 detection dataset before training or evaluation."""
from __future__ import annotations

import argparse
import hashlib
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError
import yaml

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
SPLITS = ("train", "val", "test")


@dataclass
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    images: Counter = field(default_factory=Counter)
    boxes: Counter = field(default_factory=Counter)
    empty_images: Counter = field(default_factory=Counter)
    names: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def format(self) -> str:
        lines = ["Dataset validation: " + ("PASS" if self.ok else "FAIL")]
        for split in SPLITS:
            lines.append(
                f"  {split}: {self.images[split]} images, {self.boxes[split]} boxes, "
                f"{self.empty_images[split]} empty-label negatives"
            )
        lines.extend(f"ERROR: {item}" for item in self.errors)
        lines.extend(f"WARNING: {item}" for item in self.warnings)
        return "\n".join(lines)


def _class_names(raw: Any) -> list[str]:
    if isinstance(raw, list):
        return [str(value) for value in raw]
    if isinstance(raw, dict):
        try:
            return [str(raw[key]) for key in sorted(raw, key=lambda value: int(value))]
        except (TypeError, ValueError):
            return []
    return []


def load_dataset_config(config_path: Path) -> tuple[dict[str, Any], Path]:
    config_path = config_path.resolve()
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    base = Path(config.get("path", "."))
    if not base.is_absolute():
        base = (config_path.parent / base).resolve()
    return config, base


def is_synthetic_dataset(config_path: Path) -> bool:
    """True only for a fixture explicitly marked as synthetic."""
    try:
        _config, dataset_root = load_dataset_config(config_path)
    except (OSError, yaml.YAMLError, TypeError, ValueError):
        return False
    return (dataset_root / ".synthetic-fixture").is_file()


def validate_dataset(config_path: Path) -> ValidationReport:
    report = ValidationReport()
    try:
        config, dataset_root = load_dataset_config(config_path)
    except (OSError, yaml.YAMLError, TypeError, ValueError) as error:
        report.errors.append(f"Cannot read dataset config {config_path}: {error}")
        return report

    report.names = _class_names(config.get("names"))
    if not report.names:
        report.errors.append("Config must define class names as a list or numeric-keyed mapping.")
    if len(report.names) != len(set(report.names)):
        report.errors.append("Class names must be unique.")

    split_hashes: dict[str, dict[str, Path]] = {}
    for split in SPLITS:
        image_entry = config.get(split)
        label_entry = config.get("labels", {}).get(split) if isinstance(config.get("labels"), dict) else None
        image_dir = (dataset_root / str(image_entry)).resolve() if image_entry else dataset_root / "images" / split
        label_dir = (dataset_root / str(label_entry)).resolve() if label_entry else dataset_root / "labels" / split
        if not image_dir.is_dir():
            report.errors.append(f"Missing {split} image directory: {image_dir}")
            continue
        if not label_dir.is_dir():
            report.errors.append(f"Missing {split} label directory: {label_dir}")
            continue

        images = sorted(path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
        labels = {path.stem.casefold(): path for path in label_dir.glob("*.txt") if path.is_file()}
        if len(labels) != sum(1 for path in label_dir.glob("*.txt") if path.is_file()):
            report.errors.append(f"{split}: multiple annotation files use the same name with different capitalization.")
        stems: dict[str, Path] = {}
        expected: set[str] = set()
        split_hashes[split] = {}
        for image_path in images:
            key = image_path.stem.casefold()
            if key in stems:
                report.errors.append(f"{split}: duplicate image stem {stems[key].name} and {image_path.name}.")
            stems[key] = image_path
            expected.add(key)
            try:
                with Image.open(image_path) as image:
                    image.verify()
                digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
                split_hashes[split][digest] = image_path
            except (OSError, ImportError, UnidentifiedImageError, Image.DecompressionBombError) as error:
                report.errors.append(f"{split}: unreadable/corrupted image {image_path}: {error}")
                continue

            report.images[split] += 1
            label_path = labels.get(key)
            if label_path is None:
                report.errors.append(f"{split}: missing annotation file for {image_path.name}; use an empty .txt only when confirmed negative.")
                continue
            try:
                rows = label_path.read_text(encoding="utf-8").splitlines()
            except OSError as error:
                report.errors.append(f"Cannot read label file {label_path}: {error}")
                continue
            valid_rows = 0
            for line_number, row in enumerate(rows, start=1):
                fields = row.split()
                if not fields:
                    continue
                if len(fields) != 5:
                    report.errors.append(f"{label_path}:{line_number}: expected 5 YOLO fields (class x_center y_center width height), got {len(fields)}.")
                    continue
                try:
                    class_id_value = float(fields[0])
                    coords = [float(value) for value in fields[1:]]
                except ValueError:
                    report.errors.append(f"{label_path}:{line_number}: all fields must be numeric.")
                    continue
                if not class_id_value.is_integer() or not 0 <= int(class_id_value) < len(report.names):
                    report.errors.append(f"{label_path}:{line_number}: class id {fields[0]} is outside 0..{len(report.names) - 1}.")
                    continue
                cx, cy, width, height = coords
                if not math.isfinite(class_id_value) or not all(math.isfinite(value) for value in coords):
                    report.errors.append(f"{label_path}:{line_number}: class and coordinates must be finite numbers.")
                    continue
                if any(value < 0 or value > 1 for value in coords) or width == 0 or height == 0:
                    report.errors.append(f"{label_path}:{line_number}: normalized coordinates must be in [0,1] and box size must be positive.")
                    continue
                if cx - width / 2 < -1e-6 or cx + width / 2 > 1 + 1e-6 or cy - height / 2 < -1e-6 or cy + height / 2 > 1 + 1e-6:
                    report.errors.append(f"{label_path}:{line_number}: box extends outside the image.")
                    continue
                valid_rows += 1
            report.boxes[split] += valid_rows
            if valid_rows == 0:
                report.empty_images[split] += 1

        for label_key, label_path in labels.items():
            if label_key not in expected:
                report.errors.append(f"{split}: label has no matching image: {label_path.name}")
        if not images:
            message = f"{split}: no images found in {image_dir}."
            if split in ("train", "val"):
                report.errors.append(message)
            else:
                report.warnings.append(message)

    for left_index, left in enumerate(SPLITS):
        for right in SPLITS[left_index + 1:]:
            overlap = set(split_hashes.get(left, {})) & set(split_hashes.get(right, {}))
            for digest in sorted(overlap):
                report.errors.append(f"Exact image leakage across {left} and {right}: {split_hashes[left][digest].name} / {split_hashes[right][digest].name}.")

    if any(name.casefold() != "lanternfly" for name in report.names) and report.names:
        report.warnings.append("The current Flask API expects one class named 'lanternfly' (or a name containing 'lantern'); update its class filter if you use other labels.")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(__file__).resolve().parents[1] / "data.yaml", help="Ultralytics dataset YAML")
    parser.add_argument("--require-test", action="store_true", help="Fail if the test split has no images; use before evaluation")
    args = parser.parse_args()
    report = validate_dataset(args.data)
    if args.require_test and report.images["test"] == 0:
        report.errors.append("Test split is empty; add unseen test images before evaluation.")
    print(report.format())
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
