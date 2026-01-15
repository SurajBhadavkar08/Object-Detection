import os
from datetime import datetime
from typing import Optional

from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash, send_file
from werkzeug.utils import secure_filename
import cv2

from detector import get_default_detector

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")

# Ensure folders exist
for d in (UPLOAD_DIR, RESULTS_DIR, TEMPLATES_DIR, STATIC_DIR):
    os.makedirs(d, exist_ok=True)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp"}

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

app = Flask(__name__, template_folder=TEMPLATES_DIR, static_folder=STATIC_DIR)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-key")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/detect", methods=["POST"])
def detect():
    if "image" not in request.files:
        flash("No file part in request")
        return redirect(url_for("index"))

    file = request.files["image"]
    if file.filename == "":
        flash("No selected file")
        return redirect(url_for("index"))

    if not allowed_file(file.filename):
        flash("Unsupported file type. Please upload png, jpg, jpeg, or bmp.")
        return redirect(url_for("index"))

    filename = secure_filename(file.filename)
    ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S-%f")
    filename_ts = f"{ts}-{filename}"
    upload_path = os.path.join(UPLOAD_DIR, filename_ts)
    file.save(upload_path)

    # Read image with OpenCV
    image_bgr = cv2.imread(upload_path)
    if image_bgr is None:
        flash("Failed to read image. Ensure it's a valid image file.")
        return redirect(url_for("index"))

    # Thresholds via form or defaults
    try:
        conf = float(request.form.get("confidence", 0.5))
    except Exception:
        conf = 0.5
    try:
        nms = float(request.form.get("nms", 0.4))
    except Exception:
        nms = 0.4

    detector = get_default_detector()
    detector.set_thresholds(conf, nms)
    detections = detector.detect(image_bgr)
    drawn = detector.draw_detections(image_bgr, detections)

    result_filename = filename_ts
    result_path = os.path.join(RESULTS_DIR, result_filename)
    cv2.imwrite(result_path, drawn)

    # Prepare simple JSON-like stats for display
    indices = detections["indices"].tolist()
    classes = [detector.class_names[int(detections["class_ids"][i])]
               for i in indices] if len(indices) else []
    confidences = [float(detections["confidences"][i]) for i in indices]

    return render_template(
        "index.html",
        uploaded_image=url_for("uploaded_file", filename=filename_ts),
        result_image=url_for("result_file", filename=result_filename),
        classes=classes,
        confidences=confidences,
        conf=conf,
        nms=nms,
    )

@app.route("/uploads/<path:filename>")
def uploaded_file(filename: str):
    return send_from_directory(UPLOAD_DIR, filename)

@app.route("/results/<path:filename>")
def result_file(filename: str):
    return send_from_directory(RESULTS_DIR, filename)

@app.route("/speak", methods=["POST"])
def speak():
    # Always serve the static person.mp3 file
    audio_path = os.path.join(BASE_DIR, "person.mp3")
    if not os.path.exists(audio_path):
        return "Audio file not found.", 404
    return send_file(audio_path, mimetype="audio/mpeg")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=True)