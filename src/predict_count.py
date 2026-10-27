"""Run YOLO detection and counting on one image."""
import argparse
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Path to an image")
    parser.add_argument("--weights", default=str(ROOT / "models" / "best.pt"))
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()
    weights = Path(args.weights)
    if not weights.is_file():
        parser.error(f"Weights not found: {weights}. Train a model first (see README.md).")
    if not Path(args.source).is_file():
        parser.error(f"Image not found: {args.source}")

    result = YOLO(str(weights)).predict(source=args.source, conf=args.conf, verbose=False)[0]
    count = len(result.boxes)
    print(f"Lanternflies detected: {count}")
    output = ROOT / "runs" / "predict"
    output.mkdir(parents=True, exist_ok=True)
    destination = output / f"{Path(args.source).stem}_detected.jpg"
    result.save(str(destination))
    print(f"Annotated result saved to: {destination}")


if __name__ == "__main__":
    main()
