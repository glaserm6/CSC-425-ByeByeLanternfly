"""Create a machine-screened positive-only review manifest for the Roboflow v7 import.

This applies only structural checks and the source's exact adult category label.
It does not claim that every insect identity, life stage, or box was reviewed by
a person. Empty adult-only labels and all unresolved source records are excluded.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


SPLIT_ORDER = {"test": 0, "train": 1, "valid": 2}


def screen(source_csv: Path, output_csv: Path) -> Path:
    source_csv = source_csv.resolve()
    root = source_csv.parent
    manifest = json.loads((root / "source_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("adult_category_retained") != "Adult Spotted Lanternfly":
        raise ValueError("Expected the exact Adult Spotted Lanternfly category.")
    with source_csv.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
        fields = list(rows[0]) if rows else []
    by_id = {row["asset_id"]: row for row in rows}

    # Build connected components from every detected perceptual near-duplicate
    # pair. Keeping one member per component prevents transitive leakage too.
    parent = {asset_id: asset_id for asset_id in by_id}

    def find(key: str) -> str:
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    pairs_path = root / "near_duplicate_pairs.csv"
    pairs = []
    with pairs_path.open(newline="", encoding="utf-8-sig") as stream:
        for pair in csv.DictReader(stream):
            a, b = pair["asset_a"], pair["asset_b"]
            pairs.append(pair)
            if a in parent and b in parent:
                union(a, b)

    eligible: dict[str, list[dict[str, str]]] = defaultdict(list)
    reasons: dict[str, str] = {}
    for row in rows:
        asset_id = row["asset_id"]
        reason = row.get("exclusion_reason", "").strip()
        if row.get("classifier_label") != "positive_candidate":
            reasons[asset_id] = "excluded_unverified_empty_or_negative"
            continue
        if reason and not reason.startswith("Potential cross-split near-duplicate"):
            # The importer uses this field for invalid boxes and dimension flags.
            # Any non-duplicate reason therefore blocks automatic acceptance.
            reasons[asset_id] = "excluded_unresolved_annotation_or_orientation_issue"
            continue
        image = Path(row["image_path"])
        label = Path(row["annotation_file"])
        image_path = (Path.cwd() / image).resolve() if not image.is_absolute() else image
        label_path = (Path.cwd() / label).resolve() if not label.is_absolute() else label
        if not image_path.is_file() or not label_path.is_file():
            reasons[asset_id] = "excluded_missing_image_or_label"
            continue
        label_rows = [line for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not label_rows:
            reasons[asset_id] = "excluded_positive_without_retained_adult_box"
            continue
        eligible[find(asset_id)].append(row)

    # Test examples win. Otherwise retain a train member over validation so
    # source assignments are never changed and no duplicate crosses splits.
    retained: set[str] = set()
    for component in eligible.values():
        component.sort(key=lambda row: (SPLIT_ORDER.get(row["source_split"], 99), row["asset_id"]))
        winner = component[0]
        retained.add(winner["asset_id"])
        for duplicate in component[1:]:
            reasons[duplicate["asset_id"]] = f"excluded_near_duplicate_of_{winner['asset_id']}"

    # Write a derivative manifest with machine status in dedicated fields.
    extra = ["machine_screen_status", "machine_screen_reason", "human_identity_verified", "human_adult_verified", "human_bbox_verified"]
    output_csv = output_csv.resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    with output_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields + [field for field in extra if field not in fields])
        writer.writeheader()
        for row in rows:
            asset_id = row["asset_id"]
            accepted = asset_id in retained
            row.update({
                "review_state": "accepted" if accepted else "excluded",
                "exclusion_reason": "" if accepted else reasons.get(asset_id, "excluded_unresolved_review"),
                "assigned_split": {"train": "train", "valid": "val", "test": "test"}.get(row["source_split"], "") if accepted else "",
                "near_duplicate_reviewed": "true" if accepted else row.get("near_duplicate_reviewed", "false"),
                "machine_screen_status": "structurally_valid_source_adult_positive" if accepted else "excluded",
                "machine_screen_reason": "source_category_and_boxes_passed_automated_checks; not human verified" if accepted else reasons.get(asset_id, "excluded_unresolved_review"),
                "human_identity_verified": "false",
                "human_adult_verified": "false",
                "human_bbox_verified": "false",
            })
            counts[(row["source_split"], "accepted" if accepted else "excluded")] += 1
            writer.writerow(row)

    reason_counts = Counter(reasons.values())
    report = {
        "policy": "Retain only non-empty source Adult Spotted Lanternfly positives with no importer annotation/orientation issue; exclude all unverified empty labels; retain one original-split item per near-duplicate connected component, preferring source test then train then valid.",
        "human_verification": False,
        "near_duplicate_pairs_considered": len(pairs),
        "counts_by_split_and_status": {f"{key[0]}_{key[1]}": value for key, value in sorted(counts.items())},
        "excluded_empty_or_unverified": reason_counts["excluded_unverified_empty_or_negative"],
        "excluded_annotation_or_orientation_flags": reason_counts["excluded_unresolved_annotation_or_orientation_issue"],
        "excluded_as_near_duplicate_copies": sum(count for reason, count in reason_counts.items() if reason.startswith("excluded_near_duplicate_of_")),
        "excluded_for_other_reasons": sum(count for reason, count in reason_counts.items() if not reason.startswith(("excluded_unverified_empty_or_negative", "excluded_unresolved_annotation_or_orientation_issue", "excluded_near_duplicate_of_"))),
        "exclusion_reasons_by_asset": dict(reasons),
        "screened_manifest": str(output_csv),
    }
    (output_csv.parent / "screening_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return output_csv


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-csv", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(screen(args.review_csv, args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
