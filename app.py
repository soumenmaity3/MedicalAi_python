import os
import json
import requests
from flask import Flask, request, jsonify
from pathlib import Path

# ----------------------------------
# Hugging Face Router Endpoint
# ----------------------------------
HF_MODEL = "sm89/Symptom2Disease"
HF_API_URL = f"https://router.huggingface.co/hf-inference/models/{HF_MODEL}"

HF_TOKEN = os.getenv("HF_TOKEN")

HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json"
}

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

    payload = {
        "inputs": data["text"]
    }

    response = requests.post(HF_API_URL, headers=HEADERS, json=payload)

    if response.status_code != 200:
        return jsonify({
            "error": "Inference failed",
            "details": response.text
        }), 500

    result = response.json()

    # HF returns list of predictions
    predictions = result[0] if isinstance(result, list) else result

    # Sort top 3
    predictions = sorted(predictions, key=lambda x: x["score"], reverse=True)[:3]

    formatted = []

    for p in predictions:
        label_index = p["label"].split("_")[-1]
        formatted.append({
            "department": id_to_label[label_index],
            "confidence": round(p["score"], 4)
        })

    return jsonify({
        "input_text": data["text"],
        "top_predictions": formatted,
        "final_prediction": formatted[0]
    })

# ----------------------------------
# Run
# ----------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
