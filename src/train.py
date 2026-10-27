"""Fine-tune YOLOv8n on a real, validated lanternfly detection dataset."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.validate_dataset import is_synthetic_dataset, validate_dataset  # noqa: E402


def numeric_metrics(metrics: object) -> dict[str, float]:
    values = getattr(metrics, "results_dict", {})
    output = {}
    for key, value in values.items():
        if hasattr(value, "item"):
            value = value.item()
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            output[str(key)] = float(value)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data.yaml", help="YOLOv8 dataset YAML")
    parser.add_argument("--model", default="yolov8n.pt", help="Ultralytics YOLOv8 starting checkpoint (downloaded on first use)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8, help="Lower this if GPU/CPU memory runs out")
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--device", default=None, help="For example 0 (CUDA GPU), cpu, or omit for auto")
    parser.add_argument("--name", default=None, help="Run name; defaults to a timestamped unique name")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", type=Path, default=None, help="Resume an interrupted candidate run from its weights/last.pt checkpoint")
    args = parser.parse_args()
    if args.epochs < 1 or args.imgsz < 32 or args.batch < 1:
        parser.error("--epochs and --batch must be positive, and --imgsz must be at least 32.")
    data = args.data.resolve()
    report = validate_dataset(data)
    print(report.format())
    if not report.ok:
        parser.error("Fix dataset validation errors before training.")
    if is_synthetic_dataset(data):
        parser.error("This dataset is marked synthetic and is for validator tests only. Add real annotated photographs before training.")
    for split in ("train", "val"):
        if report.images[split] == 0:
            parser.error(f"The {split} split has no images.")
        if report.boxes[split] == 0:
            parser.error(f"The {split} split has no valid lanternfly boxes; training/validation needs positive examples.")
    if report.images["test"] == 0 or report.boxes["test"] == 0:
        parser.error("A held-out test split with real images and lanternfly boxes is required for the documented fair comparison workflow.")
    if report.names != ["lanternfly"]:
        parser.error("The current Flask endpoint is configured for one class named exactly 'lanternfly'. Keep that class until you intentionally update inference for multiple classes.")

    try:
        import ultralytics
        from ultralytics import YOLO
    except ImportError as error:
        parser.error(f"Ultralytics is not installed in this Python environment: {error}. Run python -m pip install -r requirements.txt.")

    candidate_root = (ROOT / "runs" / "candidates" / "lanternfly_yolov8").resolve()
    if args.resume is not None:
        resume_checkpoint = args.resume.resolve()
        if not resume_checkpoint.is_file() or resume_checkpoint.name.casefold() != "last.pt":
            parser.error("--resume must point to an existing candidate weights/last.pt checkpoint.")
        output_dir = resume_checkpoint.parent.parent
        if not output_dir.is_relative_to(candidate_root):
            parser.error(f"--resume checkpoint must be inside {candidate_root}.")
        run_name = output_dir.name
        if args.name and args.name != run_name:
            parser.error("When resuming, --name must match the existing candidate run directory.")
        model = YOLO(str(resume_checkpoint))
        starting_weights_path = resume_checkpoint
    else:
        run_name = args.name or datetime.now().strftime("lanternfly_%Y%m%d_%H%M%S")
        output_dir = candidate_root / run_name
        if output_dir.exists():
            parser.error(f"Run directory already exists; choose another --name: {output_dir}")
        model = YOLO(args.model)
        starting_weights_path = Path(args.model)
        if not starting_weights_path.is_absolute():
            starting_weights_path = (ROOT / starting_weights_path).resolve()
    starting_weights_sha = hashlib.sha256(starting_weights_path.read_bytes()).hexdigest() if starting_weights_path.is_file() else None
    metrics = model.train(
        data=str(data), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        patience=args.patience, device=args.device, workers=args.workers,
        project=str(output_dir.parent), name=run_name, exist_ok=args.resume is not None,
        seed=42, deterministic=True, plots=True, save=True, resume=args.resume is not None,
    )
    best = output_dir / "weights" / "best.pt"
    if not best.is_file():
        raise RuntimeError(f"Training finished but Ultralytics did not produce the expected best checkpoint: {best}")
    source_manifest_file = data.parent / "dataset_manifest.json"
    effective_args = getattr(getattr(model, "trainer", None), "args", None)
    def effective(name: str, fallback):
        return getattr(effective_args, name, fallback)
    config_hash = hashlib.sha256(data.read_bytes()).hexdigest()
    import yaml
    config = yaml.safe_load(data.read_text(encoding="utf-8")) or {}
    dataset_root = Path(config.get("path", "."))
    if not dataset_root.is_absolute():
        dataset_root = (data.parent / dataset_root).resolve()
    data_fingerprints = {}
    for split in ("train", "val", "test"):
        # The config can point outside the repository; use its resolved root.
        image_dir = dataset_root / "images" / split
        image_entry = config.get(split)
        if image_entry:
            image_dir = Path(image_entry)
            if not image_dir.is_absolute():
                image_dir = dataset_root / image_dir
        label_config = config.get("labels", {})
        label_entry = label_config.get(split) if isinstance(label_config, dict) else None
        label_dir = Path(label_entry) if label_entry else dataset_root / "labels" / split
        if not label_dir.is_absolute():
            label_dir = dataset_root / label_dir
        records = {}
        for directory in (image_dir, label_dir):
            if directory.is_dir():
                for file_path in sorted(item for item in directory.iterdir() if item.is_file()):
                    try:
                        key = file_path.resolve().relative_to(dataset_root).as_posix()
                    except ValueError:
                        key = str(file_path.resolve())
                    records[key] = hashlib.sha256(file_path.read_bytes()).hexdigest()
        data_fingerprints[split] = records
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "ultralytics_version": ultralytics.__version__,
        "starting_weights": args.model,
        "starting_weights_path": str(starting_weights_path),
        "starting_weights_sha256": starting_weights_sha,
        "resume_from": str(args.resume.resolve()) if args.resume else None,
        "candidate_checkpoint_sha256": hashlib.sha256(best.read_bytes()).hexdigest(),
        "dataset_yaml": str(data),
        "dataset_yaml_sha256": config_hash,
        "dataset_manifest": str(source_manifest_file) if source_manifest_file.is_file() else None,
        "dataset_manifest_sha256": hashlib.sha256(source_manifest_file.read_bytes()).hexdigest() if source_manifest_file.is_file() else None,
        "dataset_files_sha256": data_fingerprints,
        "class_names": report.names,
        "train_images": report.images["train"],
        "train_boxes": report.boxes["train"],
        "validation_images": report.images["val"],
        "validation_boxes": report.boxes["val"],
        "best_checkpoint": str(best),
        "training_metrics_csv": str(output_dir / "results.csv"),
        "training_arguments": {"epochs": effective("epochs", args.epochs), "imgsz": effective("imgsz", args.imgsz),
                               "batch": effective("batch", args.batch), "patience": effective("patience", args.patience),
                               "device": effective("device", args.device or "auto"),
                               "workers": effective("workers", args.workers), "seed": effective("seed", 42),
                               "deterministic": effective("deterministic", True)},
        "metrics": numeric_metrics(metrics),
        "test_evaluation_completed": False,
    }
    (output_dir / "training_manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Best checkpoint: {best}")
    print(f"Training metrics: {output_dir / 'results.csv'}")
    print(f"Training manifest: {output_dir / 'training_manifest.json'}")
    print("This is a candidate checkpoint. Compare it with the production model before considering manual promotion.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
