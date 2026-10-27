"""Convert COCO object-detection JSON to single-class YOLO lanternfly labels.

Images are not copied or modified. Only categories explicitly selected by name
are retained; negatives receive empty label files.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from PIL import Image


def convert(annotation_json: Path, image_root: Path, labels_out: Path, category_names: set[str], *, strict: bool = True) -> dict:
    data = json.loads(annotation_json.read_text(encoding="utf-8"))
    categories = {int(c["id"]): str(c["name"]).strip().casefold() for c in data.get("categories", [])}
    keep = {cid for cid, name in categories.items() if name in category_names}
    if not keep:
        raise ValueError(f"No COCO category matched: {sorted(category_names)}")
    annotations: dict[int, list[dict]] = {}
    invalid_annotations: list[dict] = []
    for ann in data.get("annotations", []):
        if int(ann.get("category_id", -1)) in keep:
            annotations.setdefault(int(ann["image_id"]), []).append(ann)
    labels_out.mkdir(parents=True, exist_ok=True)
    images = data.get("images", [])
    output_stems: set[str] = set()
    for image in images:
        image_id = int(image["id"])
        width, height = float(image["width"]), float(image["height"])
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid image dimensions for {image.get('file_name')}")
        image_path = image_root / image["file_name"]
        if not image_path.is_file():
            raise ValueError(f"COCO image missing: {image_path}")
        try:
            with Image.open(image_path) as opened:
                actual_dimensions = opened.size
                opened.verify()
        except (OSError, Image.DecompressionBombError) as error:
            raise ValueError(f"COCO image is unreadable: {image_path}: {error}") from error
        if actual_dimensions != (int(width), int(height)):
            raise ValueError(f"COCO dimensions {int(width)}x{int(height)} do not match image {actual_dimensions[0]}x{actual_dimensions[1]} for {image_path}")
        stem = Path(image["file_name"]).stem
        if stem.casefold() in output_stems:
            raise ValueError(f"Duplicate image stem cannot be represented by this flat YOLO label directory: {stem}")
        output_stems.add(stem.casefold())
        rows = []
        for ann in annotations.get(image_id, []):
            try:
                values = tuple(map(float, ann["bbox"]))
                if len(values) != 4:
                    raise ValueError("bbox must contain four values")
                x, y, box_w, box_h = values
                if not all(math.isfinite(v) for v in values) or box_w <= 0 or box_h <= 0:
                    raise ValueError("non-finite or non-positive bbox")
                if x < 0 or y < 0 or x + box_w > width + 1e-4 or y + box_h > height + 1e-4:
                    raise ValueError("Out-of-image bbox extends outside image")
            except (KeyError, TypeError, ValueError) as error:
                issue = {"annotation_id": ann.get("id"), "image_id": image_id,
                         "category_id": ann.get("category_id"), "category": categories.get(int(ann.get("category_id", -1)), "unknown"),
                         "bbox": ann.get("bbox"), "reason": str(error), "file_name": image["file_name"]}
                invalid_annotations.append(issue)
                if strict:
                    raise ValueError(f"Invalid COCO box in annotation {ann.get('id')}: {error}") from error
                continue
            rows.append(f"0 {(x + box_w / 2) / width:.8f} {(y + box_h / 2) / height:.8f} {box_w / width:.8f} {box_h / height:.8f}")
        (labels_out / f"{stem}.txt").write_text("\n".join(rows), encoding="utf-8")
    stats = {"images": len(images), "boxes": sum(len((labels_out / f"{Path(i['file_name']).stem}.txt").read_text(encoding="utf-8").splitlines()) for i in images),
             "empty_images": sum(not (labels_out / f"{Path(i['file_name']).stem}.txt").read_text(encoding="utf-8").strip() for i in images)}
    if invalid_annotations:
        stats["invalid_annotations"] = invalid_annotations
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True, type=Path, help="COCO instances JSON")
    parser.add_argument("--images", required=True, type=Path, help="Root containing COCO file_name paths")
    parser.add_argument("--labels-out", required=True, type=Path)
    parser.add_argument("--category", action="append", default=[], help="Exact category name to map to class 0; repeatable")
    args = parser.parse_args()
    try:
        stats = convert(args.annotations, args.images, args.labels_out, {name.casefold() for name in args.category})
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        parser.error(str(error))
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
