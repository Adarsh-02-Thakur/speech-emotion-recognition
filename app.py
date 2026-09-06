"""
app.py
------
Simple Flask web demo: upload a .wav file in the browser and get back
the predicted emotion with a probability breakdown.

Usage
-----
    python app.py --model_dir models
Then open http://127.0.0.1:5000 in your browser.
"""

import os
import argparse
from flask import Flask, request, render_template

from predict import load_artifacts, predict_emotion

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model, encoder, norm_stats = None, None, None  # populated in main()


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    probs = None
    error = None

    if request.method == "POST":
        file = request.files.get("audio_file")
        if file and file.filename.lower().endswith(".wav"):
            save_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(save_path)
            try:
                result, probs = predict_emotion(save_path, model, encoder, norm_stats)
            except Exception as e:
                error = str(e)
        else:
            error = "Please upload a valid .wav file."

    return render_template("index.html", result=result, probs=probs, error=error)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="models")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    model, encoder, norm_stats = load_artifacts(args.model_dir)
    app.run(debug=True, port=args.port)
