import os, time
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from ocr_core import extract_with_confidence

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
history = []


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/extract", methods=["POST"])
def extract():
    if "image" not in request.files:
        return jsonify({"error": "no image uploaded"}), 400

    file = request.files["image"]
    if not file.filename:
        return jsonify({"error": "no image uploaded"}), 400
    filename = f"{int(time.time()*1000)}_{secure_filename(file.filename)}"
    path = os.path.join(UPLOAD_DIR, filename)
    file.save(path)

    try:
        start = time.time()
        result = extract_with_confidence(path)
        result["filename"] = file.filename
        result["time_s"] = round(time.time() - start, 2)
        history.insert(0, result)
        history[:] = history[:50]
        return jsonify(result)
    except RuntimeError as e:
        # Model/quota failures: 503 (not 500) so clients can retry later.
        msg = str(e)
        status = 503 if ("quota" in msg.lower() or "models failed" in msg.lower()) else 500
        return jsonify({"error": msg}), status
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/history")
def get_history():
    return jsonify(history[:20])


if __name__ == "__main__":
    print("\n  VisionRead portal running at: http://127.0.0.1:5000\n")
    app.run(debug=True, port=5000)
