# SMS Spam Detector — Web Version
# Backend = Jawad's original code (spam_detector.py).
# This file only serves it over HTTP; all prediction logic lives in his module.

import os

# Make sure relative paths (spam.csv, .pkl files) resolve to this folder
# no matter where the app is started from.
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import spam_detector  # noqa: E402  (trains the model on import, exactly as his file does)
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(silent=True) or {}
    message = (data.get("message") or "").strip()
    if not message:
        return jsonify({"error": "message is required"}), 400
    result = spam_detector.predict_message(message)
    label = "SPAM" if result == "SPAM" else "HAM"
    return jsonify({"result": label, "message": message})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
