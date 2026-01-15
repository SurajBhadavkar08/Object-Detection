YOLOv4 Web UI
==============

Run a simple Flask app to upload images and visualize YOLOv4 detections using the provided weights/config and class labels in `object  detection/`.

Project layout
--------------

- `object  detection/` — put `yolov4.cfg`, `yolov4.weights`, `labels.txt` here (already present)
- `detector.py` — reusable YOLOv4 detector wrapper
- `app.py` — Flask web server with upload and detect endpoints
- `templates/index.html` — UI
- `static/styles.css` — styles
- `uploads/`, `results/` — runtime folders created automatically

Setup

1) Requirements
- Windows 10/11, Python 3.10+ installed and on PATH
- Model files in `object  detection/`:
  - `yolov4.cfg`
  - `yolov4.weights`
  - `labels.txt`

2) Create and activate a virtual environment (PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

3) Install dependencies
```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

4) Run the server (development)
```powershell
python app.py
```
You should see it running on `http://127.0.0.1:5000`.

5) Open the app
- Visit `http://127.0.0.1:5000` in Chrome/Edge/Firefox.
- Upload an image and click Detect, or use the webcam section:
  - Click Start Camera, allow permission, then click Capture to run detection.

Optional: HTTPS locally
-----------------------
Browsers require HTTPS for camera access when not on `localhost`.

If you must access over LAN or want HTTPS:
```powershell
pip install cryptography
python -m flask run --cert=adhoc --host 0.0.0.0 --port 5001
```
Then open `https://127.0.0.1:5001` (accept the certificate warning) and allow camera.

Notes
-----

- The detector uses OpenCV DNN backend and does not require Darknet at runtime.
- For CPU-only environments this may be slow for large images; consider resizing before upload.
- If you want audio announcements like the original script, you can extend `app.py` to generate speech using `gTTS` after detection.

Troubleshooting
---------------

- Camera blocked or error like "Could not access camera":
  - Use `http://127.0.0.1:5000` (localhost). Do not use your LAN IP without HTTPS.
  - In the browser, Site settings → Camera → Allow for `127.0.0.1`.
  - If using LAN, run with HTTPS (see above) and open `https://<your-ip>:5001`.

- Template error about `zip` being undefined:
  - Hard refresh (Ctrl+F5). The template has been updated to avoid `zip`.

- Large or repeated CPU usage on capture:
  - Reduce `confidence` or `nms` thresholds, or capture smaller frames.

Endpoints (for reference)
-------------------------
- `GET /` — main UI page
- `POST /detect` — upload or webcam snapshot detection
- `GET /uploads/<filename>` — original uploads
- `GET /results/<filename>` — images annotated with detections



