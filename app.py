"""Small Flask interface for counting adult spotted lanternflies in a photo."""
from __future__ import annotations

import base64
import io
import os
from pathlib import Path

from flask import Flask, jsonify, render_template, request
from PIL import Image, ImageDraw, ImageOps
_pil_open = Image.open

ROOT = Path(__file__).resolve().parent
WEIGHTS = Path(os.environ.get("LANTERNFLY_WEIGHTS", ROOT / "models" / "best.pt"))
MAX_UPLOAD_BYTES = 12 * 1024 * 1024
MAX_IMAGE_PIXELS = 30_000_000
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}

try:
    CONFIDENCE = float(os.environ.get("LANTERNFLY_CONFIDENCE", "0.20"))
except ValueError:
    CONFIDENCE = 0.20
if not 0 <= CONFIDENCE <= 1:
    CONFIDENCE = 0.20

Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES


def load_detector():
    """Load one compatible model at server startup, or retain a readable error."""
    if not WEIGHTS.is_file():
        return None, f"Trained adult lanternfly weights are missing: {WEIGHTS}"
    try:
        from ultralytics import YOLO
        model = YOLO(str(WEIGHTS))
        names = model.names
        names = names.values() if isinstance(names, dict) else names
        if not any("lanternfly" in str(name).lower() for name in names):
            return None, "The selected model has no spotted lanternfly class."
        return model, None
    except Exception as exc:  # model errors should not prevent the upload page loading
        app.logger.exception("Could not load lanternfly weights")
        return None, f"The trained model could not be loaded: {exc}"


model, model_error = load_detector()


@app.get("/")
def index():
    return render_template("index.html", model_ready=model is not None)


@app.post("/predict")
def predict():
    upload = request.files.get("image")
    if upload is None or not upload.filename:
        return jsonify(error="Choose or take a photo first."), 400
    payload = upload.read(MAX_UPLOAD_BYTES + 1)
    if len(payload) > MAX_UPLOAD_BYTES:
        return jsonify(error="That photo is too large. Choose one under 12 MB."), 413
    try:
        with _pil_open(io.BytesIO(payload)) as source:
            if source.format not in ALLOWED_FORMATS:
                return jsonify(error="Use a JPEG, PNG, or WebP photo."), 400
            if source.width * source.height > MAX_IMAGE_PIXELS:
                return jsonify(error="That photo has too many pixels. Choose a smaller image."), 400
            source.load()
            image = ImageOps.exif_transpose(source).convert("RGB")
    except Exception:
        return jsonify(error="The selected file could not be read as a safe image."), 400

    if model is None:
        return jsonify(error=model_error or "The detector is not ready. Train a model first."), 503
    try:
        # Keep inference at the same 640-pixel size used to select the demo
        # confidence threshold and produce the evaluation report.
        result = model.predict(image, imgsz=640, conf=CONFIDENCE, iou=0.45, max_det=1000, verbose=False)[0]
        names = result.names
        target_ids = {int(i) for i, name in names.items() if "lanternfly" in str(name).lower()} if isinstance(names, dict) else {i for i, name in enumerate(names) if "lanternfly" in str(name).lower()}
        retained = []
        if result.boxes is not None:
            for box, score, class_id in zip(result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist(), result.boxes.cls.cpu().tolist()):
                if int(class_id) in target_ids:
                    retained.append((box, float(score)))
        annotated = image.copy()
        draw = ImageDraw.Draw(annotated)
        line_width = max(2, round(max(image.size) / 400))
        for box, score in retained:
            coords = tuple(round(value) for value in box)
            draw.rectangle(coords, outline="#79d74b", width=line_width)
            label = f"Spotted lanternfly {score:.0%}"
            text_box = draw.textbbox((coords[0], coords[1]), label)
            draw.rectangle((text_box[0] - 3, text_box[1] - 3, text_box[2] + 3, text_box[3] + 3), fill="#193b2c")
            draw.text((coords[0], coords[1]), label, fill="white")
        output = io.BytesIO()
        annotated.save(output, format="JPEG", quality=88)
        encoded = base64.b64encode(output.getvalue()).decode("ascii")
    except Exception:
        app.logger.exception("Lanternfly inference failed")
        return jsonify(error="The detector could not analyze this image. Try another photo."), 500
    return jsonify(count=len(retained), detections=[{"box": [round(float(v), 2) for v in box], "confidence": round(score, 4)} for box, score in retained], annotated_image=f"data:image/jpeg;base64,{encoded}")


@app.errorhandler(413)
def too_large(_error):
    return jsonify(error="That photo is too large. Choose one under 12 MB."), 413


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")), debug=False)
