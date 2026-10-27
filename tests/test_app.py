"""Small Flask input/count contract tests; no checkpoint is required."""
import io
import unittest
from unittest.mock import patch

from PIL import Image

import app


class Values(list):
    def cpu(self):
        return self

    def tolist(self):
        return list(self)


class FakeBoxes:
    xyxy = Values([[2, 3, 12, 14], [20, 22, 35, 36]])
    conf = Values([0.9, 0.8])
    cls = Values([0, 1])


class FakeResult:
    names = {0: "lanternfly", 1: "bird"}
    boxes = FakeBoxes()


class FakeModel:
    def predict(self, *_args, **_kwargs):
        return [FakeResult()]


def jpeg():
    stream = io.BytesIO()
    Image.new("RGB", (64, 48), "white").save(stream, format="JPEG")
    stream.seek(0)
    return stream


class FlaskInputTests(unittest.TestCase):
    def setUp(self):
        self.client = app.app.test_client()

    def test_single_target_class_detection_returns_matching_box_count_and_image(self):
        with patch.object(app, "model", FakeModel()):
            response = self.client.post("/predict", data={"image": (jpeg(), "photo.jpg")})
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["count"], 1)
        self.assertEqual(data["count"], len(data["detections"]))
        self.assertTrue(data["annotated_image"].startswith("data:image/jpeg;base64,"))

    def test_zero_is_a_successful_count(self):
        class NoBoxes(FakeResult):
            boxes = None

        class NoDetectionModel:
            def predict(self, *_args, **_kwargs):
                return [NoBoxes()]

        with patch.object(app, "model", NoDetectionModel()):
            response = self.client.post("/predict", data={"image": (jpeg(), "photo.jpg")})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["count"], 0)

    def test_bad_image_is_rejected(self):
        response = self.client.post("/predict", data={"image": (io.BytesIO(b"not an image"), "bad.jpg")})
        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
