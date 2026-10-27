"""Build a YOLO candidate only from reviewed Roboflow import rows.

Unlike the HF builder, this preserves Roboflow's reviewed train/valid/test
assignments so the source test split remains held out for final evaluation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.validate_dataset import validate_dataset  # noqa: E402


def truth(value: str) -> bool:
    return value.strip().casefold() in {"true", "yes", "1", "y"}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def average_hash(path: Path) -> int:
    from PIL import Image, ImageOps
    with Image.open(path) as source:
        pixels = list(ImageOps.exif_transpose(source).convert("L").resize((8, 8)).getdata())
    threshold = sum(pixels) / len(pixels)
    return sum((int(value > threshold) << bit) for bit, value in enumerate(pixels))


def difference_hash(path: Path) -> int:
    from PIL import Image, ImageOps
    with Image.open(path) as source:
        pixels = list(ImageOps.exif_transpose(source).convert("L").resize((9, 8)).getdata())
    result = 0
    for row in range(8):
        for col in range(8):
            result = (result << 1) | int(pixels[row * 9 + col] > pixels[row * 9 + col + 1])
    return result


def build(review_csv: Path, name: str, test_history: str = "untouched", source_adult_screening: bool = False) -> Path:
    if not name or Path(name).name != name or name in {".", ".."}:
        raise ValueError("Candidate name must be one simple directory name.")
    review_csv = review_csv.resolve()
    source_manifest_path = review_csv.parent / "source_manifest.json"
    if not source_manifest_path.is_file():
        raise ValueError("Roboflow source_manifest.json is required beside the review CSV.")
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if source_manifest.get("export_format") != "COCO object detection":
        raise ValueError("Source manifest is not a Roboflow COCO import.")
    if source_adult_screening and source_manifest.get("adult_category_retained") != "Adult Spotted Lanternfly":
        raise ValueError("Automated source screening is limited to the exact Adult Spotted Lanternfly category.")
    out = ROOT / "data" / "candidates" / name
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite existing candidate: {out}")
    out.mkdir(parents=True)
    splitmap = {"train": "train", "valid": "val", "test": "test"}
    copied = Counter()
    rejected = Counter()
    accepted = Counter()
    accepted_negatives = Counter()
    near_hashes: dict[str, list[tuple[int, int, Path, bool]]] = defaultdict(list)
    exact_hashes: dict[str, str] = {}
    group_splits: dict[str, set[str]] = defaultdict(set)
    try:
        with review_csv.open(newline="", encoding="utf-8-sig") as stream:
            rows = list(csv.DictReader(stream))
        for row in rows:
            if row.get("review_state", "").strip().casefold() == "accepted":
                source_split = row.get("source_split", "").strip().casefold()
                target_split = splitmap.get(source_split)
                if target_split is None or row.get("assigned_split", "").strip().casefold() != target_split:
                    raise ValueError(f"Accepted item {row.get('asset_id')} changed its source split; preserve train/valid/test assignments.")
                group_splits[row.get("original_group") or row.get("asset_id", "")].add(target_split)
        leakage = [key for key, splits in group_splits.items() if len(splits) > 1]
        if leakage:
            raise ValueError(f"Source image groups cross splits: {leakage[:8]}")

        for split in ("train", "val", "test"):
            (out / "images" / split).mkdir(parents=True)
            (out / "labels" / split).mkdir(parents=True)
        flagged_pairs = set()
        pairs_file = review_csv.parent / "near_duplicate_pairs.csv"
        if pairs_file.is_file():
            with pairs_file.open(newline="", encoding="utf-8-sig") as stream:
                for pair in csv.DictReader(stream):
                    flagged_pairs.add(pair.get("asset_a", ""))
                    flagged_pairs.add(pair.get("asset_b", ""))
        for row in rows:
            state = row.get("review_state", "").strip().casefold()
            if state != "accepted":
                rejected[state or "unreviewed"] += 1
                continue
            if row.get("exclusion_reason", "").strip():
                raise ValueError(f"Accepted item {row.get('asset_id')} still has unresolved review flags: {row['exclusion_reason']}")
            if row.get("asset_id", "") in flagged_pairs and not truth(row.get("near_duplicate_reviewed", "false")):
                raise ValueError(f"Accepted item {row.get('asset_id')} has an unresolved cross-split near-duplicate flag.")
            kind = row.get("classifier_label", "").strip().casefold()
            if kind not in {"positive_candidate", "negative_candidate"}:
                raise ValueError(f"Accepted item {row.get('asset_id')} has an unknown source label.")
            positive = kind == "positive_candidate"
            if not positive:
                accepted_negatives[splitmap[row["source_split"].strip().casefold()]] += 1
            if source_adult_screening and not positive:
                raise ValueError("Automated source screening never accepts empty or negative images.")
            if not source_adult_screening and not truth(row.get("bbox_verified", "false")):
                raise ValueError(f"Accepted item {row.get('asset_id')} needs reviewed bounding boxes/empty label.")
            if positive and not source_adult_screening and (not truth(row.get("identity_verified", "false")) or not truth(row.get("adult_verified", "false"))):
                raise ValueError(f"Positive {row.get('asset_id')} needs verified adult SLF identity and stage.")
            if not positive and not truth(row.get("negative_verified", "false")):
                raise ValueError(f"Negative {row.get('asset_id')} is not confirmed adult-free.")
            source_image = ROOT / row.get("image_path", "")
            source_label = ROOT / row.get("annotation_file", "")
            if not source_image.is_file() or not source_label.is_file():
                raise ValueError(f"Accepted asset {row.get('asset_id')} is missing its image or label.")
            label_lines = [line.strip() for line in source_label.read_text(encoding="utf-8").splitlines() if line.strip()]
            if positive and not label_lines:
                raise ValueError(f"Positive {row.get('asset_id')} has an empty label file.")
            if not positive and label_lines:
                raise ValueError(f"Negative {row.get('asset_id')} must have an empty label file.")
            digest = sha(source_image)
            if digest in exact_hashes:
                raise ValueError(f"Exact image duplicate across candidate data: {source_image} / {exact_hashes[digest]}")
            split = splitmap[row["source_split"].strip().casefold()]
            image_hash = difference_hash(source_image)
            image_avg_hash = average_hash(source_image)
            duplicate_reviewed = truth(row.get("near_duplicate_reviewed", "false"))
            for prior_split, records in near_hashes.items():
                if prior_split == split:
                    continue
                for prior_avg, prior_diff, prior_path, prior_reviewed in records:
                    if (image_avg_hash ^ prior_avg).bit_count() <= 2 and (image_hash ^ prior_diff).bit_count() <= 2:
                        if duplicate_reviewed and prior_reviewed:
                            continue
                        raise ValueError(f"Near-duplicate leakage across {prior_split}/{split}: {source_image} / {prior_path}")
            stem = f"rf_{row.get('asset_id') or digest[:20]}"
            image_dest = out / "images" / split / f"{stem}{source_image.suffix.lower()}"
            label_dest = out / "labels" / split / f"{stem}.txt"
            shutil.copy2(source_image, image_dest)
            shutil.copy2(source_label, label_dest)
            exact_hashes[digest] = str(source_image)
            near_hashes[split].append((image_avg_hash, image_hash, source_image, duplicate_reviewed))
            accepted[split] += 1
            copied[split] += len(label_lines)

        if not accepted["train"] or not accepted["val"] or not copied["train"] or not copied["val"]:
            raise ValueError("Reviewed training and validation splits must both contain adult boxes.")
        if not accepted["test"] or not copied["test"]:
            raise ValueError("The unchanged source test split must contain reviewed adult boxes.")
        # Ultralytics resolves `path` relative to the current process directory,
        # while this repository's validator resolves it beside the YAML. Use an
        # absolute candidate root so both tools consume the same image folders.
        config = {"path": str(out.resolve()), "train": "images/train", "val": "images/val", "test": "images/test", "names": {0: "lanternfly"}}
        config_path = out / "dataset.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        manifest = {
            "created_utc": datetime.now(timezone.utc).isoformat(), "source_manifest": str(source_manifest_path),
            "review_manifest": str(review_csv),
            "review_manifest_sha256": sha(review_csv),
            "screening_report": str(review_csv.parent / "screening_report.json") if (review_csv.parent / "screening_report.json").is_file() else None,
            "screening_report_sha256": sha(review_csv.parent / "screening_report.json") if (review_csv.parent / "screening_report.json").is_file() else None,
            "source_url": source_manifest.get("source_url"), "source_version": source_manifest.get("version"),
            "source_license": source_manifest.get("license"), "source_attribution": source_manifest.get("attribution"),
            "source_archive_sha256": source_manifest.get("source_archive_sha256"),
            "accepted_images_by_split": dict(accepted), "accepted_boxes_by_split": dict(copied),
            "rejected_by_review_state": dict(rejected), "test_history": test_history,
            "annotation_review_status": "machine_structural_screen_only; adult labels not individually human verified" if source_adult_screening else "human_review_required",
            "verified_negative_images_by_split": dict(accepted_negatives),
            "test_policy": "Roboflow source test images retain their original test split; no source test image is copied into train or validation.",
        }
        (out / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        report = validate_dataset(config_path)
        (out / "validation.txt").write_text(report.format() + "\n", encoding="utf-8")
        if not report.ok:
            raise ValueError("Candidate failed YOLO validation:\n" + report.format())
        return out
    except Exception:
        # Preserve failed candidates so the user can inspect the partial build.
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-csv", required=True, type=Path)
    parser.add_argument("--name", required=True)
    parser.add_argument("--test-history", choices=("untouched", "unknown", "used-for-model-selection"), default="untouched")
    parser.add_argument("--source-adult-screening", action="store_true", help="Allow structurally valid positives from the exact Roboflow adult category without claiming individual human verification; negatives remain excluded.")
    args = parser.parse_args()
    try:
        result = build(args.review_csv, args.name, args.test_history, args.source_adult_screening)
    except (OSError, ValueError, yaml.YAMLError) as error:
        parser.error(str(error))
    print(f"Candidate dataset: {result}")
    print((result / "validation.txt").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
