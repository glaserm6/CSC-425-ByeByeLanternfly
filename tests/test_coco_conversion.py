import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from scripts.coco_to_yolo import convert


class CocoConversionTests(unittest.TestCase):
    def test_coco_boxes_convert_and_negatives_get_empty_files(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as tmp:
            root = Path(tmp)
            (root / "images").mkdir()
            Image.new("RGB", (100, 80), "white").save(root / "images" / "positive.jpg")
            Image.new("RGB", (100, 80), "black").save(root / "images" / "negative.jpg")
            ann = {
                "images": [{"id": 1, "file_name": "positive.jpg", "width": 100, "height": 80},
                           {"id": 2, "file_name": "negative.jpg", "width": 100, "height": 80}],
                "categories": [{"id": 8, "name": "spotted lanternfly"}],
                "annotations": [{"id": 1, "image_id": 1, "category_id": 8, "bbox": [10, 20, 30, 40]}],
            }
            path = root / "instances.json"
            path.write_text(json.dumps(ann), encoding="utf-8")
            out = root / "labels"
            stats = convert(path, root / "images", out, {"spotted lanternfly"})
            self.assertEqual(stats, {"images": 2, "boxes": 1, "empty_images": 1})
            self.assertEqual((out / "positive.txt").read_text(), "0 0.25000000 0.50000000 0.30000000 0.50000000")
            self.assertEqual((out / "negative.txt").read_text(), "")

    def test_converter_rejects_invalid_or_out_of_bounds_box(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as tmp:
            root = Path(tmp)
            (root / "images").mkdir()
            Image.new("RGB", (20, 20)).save(root / "images" / "bad.png")
            data = {"images": [{"id": 1, "file_name": "bad.png", "width": 20, "height": 20}],
                    "categories": [{"id": 1, "name": "lanternfly"}],
                    "annotations": [{"image_id": 1, "category_id": 1, "bbox": [15, 4, 10, 5]}]}
            path = root / "instances.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Out-of-image"):
                convert(path, root / "images", root / "labels", {"lanternfly"})


if __name__ == "__main__":
    unittest.main()
