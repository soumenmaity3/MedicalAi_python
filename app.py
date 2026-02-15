import json
import requests
from flask import Flask, request, jsonify
from pathlib import Path

# ----------------------------------
# Hugging Face Space Endpoint
# ----------------------------------
SPACE_API_URL = "https://sm89-symptom2disease-app.hf.space/run/predict"

# ----------------------------------
# Load Label Mapping
# ----------------------------------
BASE_DIR = Path(__file__).resolve().parent
LABEL_PATH = BASE_DIR / "Data" / "process" / "label_mappings.json"

with open(LABEL_PATH, "r") as f:
    mappings = json.load(f)

id_to_label = mappings["id_to_label"]

# ----------------------------------
# Flask App
# ----------------------------------
app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Medical Department Prediction API Running"})

@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    if not data or "text" not in data:
        return jsonify({"error": "Please provide symptom text"}), 400

    # Gradio expects "data" list format
    payload = {
        "data": [data["text"]]
    }

    response = requests.post(SPACE_API_URL, json=payload)

    if response.status_code != 200:
        return jsonify({
            "error": "Inference failed",
            "details": response.text
        }), 500

    result = response.json()

    # Gradio returns output inside "data"
    predictions_text = result["data"][0]

    return jsonify({
        "input_text": data["text"],
        "space_response": predictions_text
    })

# ----------------------------------
# Run
# ----------------------------------
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
