import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
from tests.fixtures.create_synthetic_dataset import create_dataset
from src.validate_dataset import is_synthetic_dataset, load_dataset_config, validate_dataset


class DatasetValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / "tests")
        self.root = Path(self.temp.name)
        self.config = create_dataset(self.root / "fixture")

    def tearDown(self):
        self.temp.cleanup()

    def test_valid_synthetic_fixture_and_empty_negative(self):
        report = validate_dataset(self.config)
        self.assertTrue(report.ok, report.format())
        self.assertEqual(report.images["train"], 2)
        self.assertEqual(report.boxes["test"], 1)
        self.assertEqual(report.empty_images["train"], 1)
        self.assertTrue(is_synthetic_dataset(self.config))
        config, _ = load_dataset_config(self.config)
        self.assertEqual(config["names"], {0: "lanternfly"})

    def test_training_refuses_synthetic_fixture_before_model_loading(self):
        result = subprocess.run(
            [sys.executable, str(ROOT / "src" / "train.py"), "--data", str(self.config)],
            cwd=ROOT, capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("marked synthetic", result.stderr)

    def test_invalid_normalized_box_and_class_id_are_reported(self):
        label = self.root / "fixture" / "labels" / "train" / "abstract_train.txt"
        label.write_text("2 1.5 0.4 -0.1 0.4\n", encoding="utf-8")
        report = validate_dataset(self.config)
        self.assertFalse(report.ok)
        self.assertTrue(any("class id" in error for error in report.errors))

    def test_out_of_bounds_and_non_finite_coordinates_are_reported(self):
        label = self.root / "fixture" / "labels" / "train" / "abstract_train.txt"
        label.write_text("0 0.98 0.4 0.2 0.4\n0 NaN 0.4 0.1 0.1\n", encoding="utf-8")
        report = validate_dataset(self.config)
        self.assertTrue(any("extends outside" in error for error in report.errors))
        self.assertTrue(any("finite numbers" in error for error in report.errors))

    def test_missing_annotation_and_unmatched_label_are_reported(self):
        (self.root / "fixture" / "labels" / "val" / "abstract_val.txt").unlink()
        (self.root / "fixture" / "labels" / "val" / "orphan.txt").write_text("", encoding="utf-8")
        report = validate_dataset(self.config)
        self.assertTrue(any("missing annotation" in error for error in report.errors))
        self.assertTrue(any("no matching image" in error for error in report.errors))

    def test_corrupt_image_and_exact_split_leakage_are_reported(self):
        train_image = self.root / "fixture" / "images" / "train" / "abstract_train.png"
        val_image = self.root / "fixture" / "images" / "val" / "abstract_val.png"
        val_image.write_bytes(train_image.read_bytes())
        corrupt = self.root / "fixture" / "images" / "test" / "broken.png"
        corrupt.write_bytes(b"not an image")
        (self.root / "fixture" / "labels" / "test" / "broken.txt").write_text("", encoding="utf-8")
        report = validate_dataset(self.config)
        self.assertTrue(any("corrupted image" in error for error in report.errors))
        self.assertTrue(any("leakage" in error for error in report.errors))


if __name__ == "__main__":
    unittest.main()
