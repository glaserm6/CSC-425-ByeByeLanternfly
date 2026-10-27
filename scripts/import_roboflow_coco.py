"""Stage a Roboflow COCO export for human review; never trains or downloads data.

Only the exact adult class is retained. Source test images remain marked test,
and every image stays pending until the existing review gates are completed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from coco_to_yolo import convert

ROOT = Path(__file__).resolve().parents[1]
FIELDS = [
    "asset_id", "source_repo", "source_revision", "source_license", "source_split",
    "classifier_label", "original_filename", "original_group", "image_path", "pixel_sha256",
    "review_state", "identity_verified", "adult_verified", "negative_verified",
    "bbox_verified", "near_duplicate_reviewed", "annotation_file", "assigned_split", "exclusion_reason", "duplicate_of",
]


def pixel_hash(path: Path) -> str:
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        digest = hashlib.sha256()
        digest.update(f"{image.width}x{image.height}:".encode())
        digest.update(image.tobytes())
        return digest.hexdigest()


def visual_hashes(path: Path) -> tuple[int, int]:
    """Return small average/difference hashes for review-only near-duplicate flags."""
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("L")
        diff = list(image.resize((9, 8)).getdata())
        avg_values = list(image.resize((8, 8)).getdata())
    average = sum(avg_values) / 64
    average_hash = sum((int(value > average) << bit) for bit, value in enumerate(avg_values))
    difference_hash = 0
    for y in range(8):
        for x in range(8):
            difference_hash = (difference_hash << 1) | int(diff[y * 9 + x] > diff[y * 9 + x + 1])
    return average_hash, difference_hash


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def render_contact_sheet(items: list[tuple[Path, str]], destination: Path, title: str) -> None:
    if not items:
        return
    thumb_w, thumb_h, label_h = 320, 250, 42
    columns = 3
    rows = (len(items) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * thumb_w, rows * (thumb_h + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (path, label) in enumerate(items):
        with Image.open(path) as opened:
            thumb = ImageOps.exif_transpose(opened).convert("RGB")
            thumb.thumbnail((thumb_w - 8, thumb_h - 8))
        x, y = (index % columns) * thumb_w, (index // columns) * (thumb_h + label_h)
        sheet.paste(thumb, (x + (thumb_w - thumb.width) // 2, y + (thumb_h - thumb.height) // 2))
        draw.text((x + 4, y + thumb_h + 2), label[:76], fill="black")
    sheet.save(destination)


def render_near_duplicate_sheet(pairs: list[dict], rows: list[dict], destination: Path, limit: int = 10) -> None:
    row_by_asset = {row.get("asset_id"): row for row in rows}
    selected = pairs[:limit]
    if not selected:
        return
    panel_w, panel_h, header_h = 320, 240, 46
    sheet = Image.new("RGB", (panel_w * 2, len(selected) * (panel_h + header_h)), "white")
    draw = ImageDraw.Draw(sheet)
    for index, pair in enumerate(selected):
        y = index * (panel_h + header_h)
        draw.text((4, y + 2), f"Pair {index + 1}: {pair['split_a']} ↔ {pair['split_b']} (hash distances {pair['average_hash_distance']}/{pair['difference_hash_distance']})", fill="black")
        for side, asset_key in enumerate(("asset_a", "asset_b")):
            row = row_by_asset.get(pair[asset_key])
            if not row or not row.get("image_path"):
                continue
            path = ROOT / row["image_path"]
            with Image.open(path) as opened:
                image = ImageOps.exif_transpose(opened).convert("RGB")
            labels = ROOT / row["annotation_file"]
            width, height = image.size
            painter = ImageDraw.Draw(image)
            if labels.is_file():
                for line in labels.read_text(encoding="utf-8").splitlines():
                    fields = line.split()
                    if len(fields) != 5:
                        continue
                    _, cx, cy, bw, bh = map(float, fields)
                    x1, y1 = (cx - bw / 2) * width, (cy - bh / 2) * height
                    x2, y2 = (cx + bw / 2) * width, (cy + bh / 2) * height
                    painter.rectangle((x1, y1, x2, y2), outline="lime", width=3)
            image.thumbnail((panel_w - 6, panel_h - 6))
            x = side * panel_w + (panel_w - image.width) // 2
            sheet.paste(image, (x, y + header_h + (panel_h - image.height) // 2))
            draw.text((side * panel_w + 4, y + header_h - 17), row["original_filename"][:44], fill="black")
    sheet.save(destination)


def normalize_export_images(image_root: Path, data: dict) -> dict[str, str]:
    """Match image pixel orientation to COCO dimensions, recording all changes."""
    changes = {}
    for item in data.get("images", []):
        path = image_root / item["file_name"]
        target = (int(item["width"]), int(item["height"]))
        with Image.open(path) as source:
            raw_size = source.size
            transposed = ImageOps.exif_transpose(source)
            transposed_size = transposed.size
            if raw_size == target:
                continue
            if transposed_size == target:
                normalized = transposed.convert("RGB")
                change = f"EXIF orientation normalized from {raw_size} to {target}"
            elif raw_size == (target[1], target[0]):
                normalized = source.convert("RGB").rotate(90, expand=True)
                if normalized.size != target:
                    raise ValueError(f"Could not rotate dimension-mismatched image to COCO size: {path}")
                change = f"rotated 90 degrees counter-clockwise from {raw_size} to {target}"
            else:
                raise ValueError(f"COCO dimensions {target} do not match image dimensions {raw_size} for {path}")
        normalized.save(path, format="JPEG", quality=95)
        changes[str(item["file_name"])] = change
    return changes


def stage(export: Path, output: Path, version: str, source_url: str,
          license_name: str, attribution: str, adult_class: str, archive_sha256: str = "") -> Path:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite staging directory: {output}")
    if not version.strip() or not source_url.startswith("https://") or not attribution.strip():
        raise ValueError("A version identifier, HTTPS source URL, and attribution are required.")
    splits = {"train": "train", "valid": "val", "val": "val", "test": "test"}
    found = []
    for folder, assigned in splits.items():
        split_root = export / folder
        annotation = split_root / "_annotations.coco.json"
        if annotation.is_file():
            found.append((folder, assigned, annotation, split_root))
    if not found:
        raise ValueError(f"No Roboflow COCO split folders found under {export}.")

    output.mkdir(parents=True)
    rows: list[dict[str, str]] = []
    seen: dict[str, str] = {}
    summary = []
    review_sheets = {"flagged": [], "samples": [], "no_adult_candidates": []}
    invalid_records = []
    dimension_changes = []
    near_duplicate_pairs = []
    randomizer = random.Random(425)
    try:
        for source_split, assigned, annotation, image_root in found:
            data = json.loads(annotation.read_text(encoding="utf-8"))
            source_annotations = output / "source_annotations"
            source_annotations.mkdir(exist_ok=True)
            shutil.copy2(annotation, source_annotations / f"{source_split}.coco.json")
            changed_files = normalize_export_images(image_root, data)
            dimension_changes.extend({"split": source_split, "file_name": name, "change": change}
                                    for name, change in changed_files.items())
            category_ids = {int(item["id"]): str(item["name"]).strip() for item in data.get("categories", [])}
            if adult_class not in category_ids.values():
                raise ValueError(f"Exact adult category {adult_class!r} absent from {annotation}; found {sorted(category_ids.values())}.")
            split_out = output / source_split
            images_out, labels_out = split_out / "images", split_out / "labels"
            images_out.mkdir(parents=True)
            stats = convert(annotation, image_root, labels_out, {adult_class.casefold()}, strict=False)
            issues_by_file: dict[str, list[dict]] = {}
            for issue in stats["invalid_annotations"]:
                issues_by_file.setdefault(issue["file_name"], []).append(issue)
                invalid_records.append({"split": source_split, **issue})
            for name, change in changed_files.items():
                issues_by_file.setdefault(name, []).append({"annotation_id": "dimensions", "reason": change})
            annotations_by_image: dict[int, list[dict]] = {}
            for ann in data.get("annotations", []):
                annotations_by_image.setdefault(int(ann["image_id"]), []).append(ann)
            sample_candidates = [im for im in data.get("images", []) if any(category_ids.get(int(a.get("category_id", -1))) == adult_class for a in annotations_by_image.get(int(im["id"]), []))]
            sample_candidates = randomizer.sample(sample_candidates, min(6, len(sample_candidates)))
            sample_names = {str(im["file_name"]) for im in sample_candidates}
            negative_candidates = [im for im in data.get("images", []) if not any(category_ids.get(int(a.get("category_id", -1))) == adult_class for a in annotations_by_image.get(int(im["id"]), []))]
            negative_candidates = randomizer.sample(negative_candidates, min(6, len(negative_candidates)))
            negative_names = {str(im["file_name"]) for im in negative_candidates}
            for image_info in data.get("images", []):
                source_image = image_root / image_info["file_name"]
                digest = pixel_hash(source_image)
                filename = Path(image_info["file_name"]).name
                stem = Path(filename).stem
                source_label = labels_out / f"{stem}.txt"
                if digest in seen:
                    source_label.unlink(missing_ok=True)
                    rows.append({"asset_id": digest[:20], "source_repo": source_url, "source_revision": version,
                                 "source_license": license_name, "source_split": source_split,
                                 "classifier_label": "unknown_candidate", "original_filename": filename,
                                 "original_group": stem, "image_path": "", "pixel_sha256": digest,
                                 "review_state": "duplicate", "identity_verified": "false", "adult_verified": "false",
                                 "negative_verified": "false", "bbox_verified": "false", "annotation_file": "",
                                 "near_duplicate_reviewed": "false",
                                 "assigned_split": assigned, "exclusion_reason": "Exact decoded-pixel duplicate; canonical image retained.",
                                 "duplicate_of": seen[digest]})
                    continue
                if any(p.name.casefold() == filename.casefold() for p in images_out.iterdir()):
                    raise ValueError(f"Duplicate image filename in split {source_split}: {filename}")
                asset_id = digest[:20]
                dest_image = images_out / filename
                shutil.copy2(source_image, dest_image)
                if not source_label.is_file():
                    raise ValueError(f"Converter did not produce a label for {source_image}")
                # Prefix labels and images by pixel hash so separate source folders cannot collide.
                dest_label = labels_out / f"{asset_id}.txt"
                source_label.rename(dest_label)
                dest_image.rename(images_out / f"{asset_id}{dest_image.suffix.lower()}")
                dest_image = images_out / f"{asset_id}{dest_image.suffix.lower()}"
                recorded_image = dest_image.resolve().relative_to(ROOT).as_posix()
                recorded_label = dest_label.resolve().relative_to(ROOT).as_posix()
                positive = bool(dest_label.read_text(encoding="utf-8").strip())
                issues = issues_by_file.get(str(image_info["file_name"]), [])
                if issues:
                    review_state = "needs_annotation_review"
                    exclusion_reason = "Suspicious source annotation(s): " + ", ".join(str(issue["annotation_id"]) for issue in issues)
                else:
                    review_state = "pending"
                    exclusion_reason = ""
                rows.append({"asset_id": asset_id, "source_repo": source_url, "source_revision": version,
                             "source_license": license_name, "source_split": source_split,
                             "classifier_label": "positive_candidate" if positive else "negative_candidate",
                             "original_filename": filename, "original_group": stem,
                             "image_path": recorded_image, "pixel_sha256": digest, "review_state": review_state,
                             "identity_verified": "false", "adult_verified": "false",
                             "negative_verified": "false", "bbox_verified": "false",
                             "near_duplicate_reviewed": "false",
                             "annotation_file": recorded_label, "assigned_split": assigned,
                             "exclusion_reason": exclusion_reason, "duplicate_of": ""})
                seen[digest] = recorded_image
                if issues or str(image_info["file_name"]) in sample_names or str(image_info["file_name"]) in negative_names:
                    # Overlay all original classes so reviewers can see omitted nymph/egg boxes too.
                    with Image.open(dest_image) as opened:
                        preview = ImageOps.exif_transpose(opened).convert("RGB")
                    painter = ImageDraw.Draw(preview)
                    width, height = preview.size
                    issue_ids = {item.get("annotation_id") for item in issues}
                    for ann in annotations_by_image.get(int(image_info["id"]), []):
                        bbox = ann.get("bbox", [])
                        if len(bbox) != 4:
                            continue
                        x, y, bw, bh = map(float, bbox)
                        color = "lime" if category_ids.get(int(ann.get("category_id", -1))) == adult_class else "orange"
                        if ann.get("id") in issue_ids:
                            color = "red"
                        x1, y1 = max(0, min(width - 1, int(x))), max(0, min(height - 1, int(y)))
                        x2, y2 = max(0, min(width - 1, int(x + bw))), max(0, min(height - 1, int(y + bh)))
                        painter.rectangle((x1, y1, x2, y2), outline=color, width=max(2, min(width, height) // 200))
                    preview_path = output / "review_previews" / f"{source_split}_{asset_id}.jpg"
                    preview_path.parent.mkdir(parents=True, exist_ok=True)
                    preview.save(preview_path, quality=90)
                    label = f"{source_split}: {filename}; adult boxes={sum(category_ids.get(int(a.get('category_id', -1))) == adult_class for a in annotations_by_image.get(int(image_info['id']), []))}; flags={len(issues)}"
                    if issues:
                        review_sheets["flagged"].append((preview_path, label))
                    if str(image_info["file_name"]) in sample_names:
                        review_sheets["samples"].append((preview_path, label))
                    if str(image_info["file_name"]) in negative_names:
                        review_sheets["no_adult_candidates"].append((preview_path, label + "; unverified adult-free candidate"))
            # The COCO conversion itself verifies image readability, dimensions, and box geometry.
            summary.append({"source_split": source_split, "assigned_split": assigned, **stats,
                            "categories": sorted(category_ids.values()), "retained_category": adult_class})

        # Coarse visual hashes only flag likely cross-split leakage for review.
        importable = [row for row in rows if row.get("image_path") and row.get("review_state") != "duplicate"]
        hashed = [(row, *visual_hashes(ROOT / row["image_path"])) for row in importable]
        for index, (left, left_avg, left_diff) in enumerate(hashed):
            for right, right_avg, right_diff in hashed[index + 1:]:
                if left["source_split"] == right["source_split"]:
                    continue
                avg_distance = (left_avg ^ right_avg).bit_count()
                diff_distance = (left_diff ^ right_diff).bit_count()
                if avg_distance <= 2 and diff_distance <= 2:
                    near_duplicate_pairs.append({
                        "image_a": left["image_path"], "split_a": left["source_split"], "asset_a": left["asset_id"],
                        "image_b": right["image_path"], "split_b": right["source_split"], "asset_b": right["asset_id"],
                        "average_hash_distance": avg_distance, "difference_hash_distance": diff_distance,
                    })
                    for row, other in ((left, right), (right, left)):
                        if row["review_state"] not in {"needs_annotation_review", "duplicate"}:
                            row["review_state"] = "needs_duplicate_review"
                        notice = f"Potential cross-split near-duplicate with {other['image_path']} (aHash/dHash <= 2)."
                        row["exclusion_reason"] = (row.get("exclusion_reason", "") + " " + notice).strip()
        near_fields = ["image_a", "split_a", "asset_a", "image_b", "split_b", "asset_b", "average_hash_distance", "difference_hash_distance"]
        with (output / "near_duplicate_pairs.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=near_fields)
            writer.writeheader()
            writer.writerows(near_duplicate_pairs)
        render_near_duplicate_sheet(near_duplicate_pairs, rows,
                                    output / "review_previews" / "near_duplicate_pairs.jpg")

        with (output / "review_manifest.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        manifest = {
            "created_utc": datetime.now(timezone.utc).isoformat(), "source_url": source_url,
            "version": version, "license": license_name,
            "source_archive_sha256": archive_sha256,
            "attribution": attribution,
            "export_format": "COCO object detection", "adult_category_retained": adult_class,
            "class_mapping": {adult_class: "lanternfly"}, "split_summary": summary,
            "invalid_annotation_records": invalid_records, "image_dimension_changes": dimension_changes,
            "potential_cross_split_near_duplicate_pairs": len(near_duplicate_pairs),
            "policy": "Every image is pending human review; rows containing invalid source boxes or possible cross-split visual duplicates are blocked from immediate acceptance. Source test remains test. Exact pixel duplicates are represented once. See review_previews for flagged labels and deterministic representative samples.",
        }
        (output / "source_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        for readme_name in ("README.dataset.txt", "README.roboflow.txt"):
            readme = export / readme_name
            if readme.is_file():
                shutil.copy2(readme, output / f"source_{readme_name}")
        (output / "split_counts.json").write_text(json.dumps({"source_splits": summary, "review_rows": len(rows)}, indent=2), encoding="utf-8")
        for key, items in review_sheets.items():
            render_contact_sheet(items, output / "review_previews" / f"{key}.jpg", key)
        return output
    except Exception:
        # Keep partial data visible for inspection; never silently erase a failed import.
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--export", type=Path, help="Already unzipped Roboflow COCO export")
    source.add_argument("--zip", type=Path, help="Already downloaded Roboflow COCO ZIP; unpacked temporarily")
    parser.add_argument("--output", required=True, type=Path, help="New directory under the repository data/external/roboflow area")
    parser.add_argument("--version", required=True, help="Roboflow dataset version identifier, e.g. project-slug/7")
    parser.add_argument("--source-url", required=True)
    parser.add_argument("--license", default="CC BY 4.0")
    parser.add_argument("--attribution", required=True, help="Creator/project credit and required license notice")
    parser.add_argument("--adult-class", default="Adult Spotted Lanternfly")
    args = parser.parse_args()
    try:
        output = args.output.resolve()
        if not output.is_relative_to((ROOT / "data" / "external" / "roboflow").resolve()):
            raise ValueError("--output must stay under data/external/roboflow.")
        if args.zip:
            with tempfile.TemporaryDirectory(prefix="roboflow-coco-", dir=ROOT / "data") as temp:
                temp_root = Path(temp).resolve()
                with zipfile.ZipFile(args.zip.resolve()) as archive:
                    for member in archive.infolist():
                        target = (temp_root / Path(member.filename)).resolve()
                        if not target.is_relative_to(temp_root) or member.is_dir():
                            if member.is_dir():
                                continue
                            raise ValueError(f"Unsafe ZIP path: {member.filename}")
                    archive.extractall(temp_root)
                digest = file_sha256(args.zip.resolve())
                result = stage(temp_root, output, args.version, args.source_url, args.license, args.attribution, args.adult_class, digest)
        else:
            result = stage(args.export.resolve(), output, args.version, args.source_url, args.license, args.attribution, args.adult_class)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        parser.error(str(error))
    print(f"Staged pending review: {result}")
    print(f"Review CSV: {result / 'review_manifest.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
