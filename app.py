import torch
import json
from pathlib import Path
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "Model" / "trained_model"
LABEL_PATH = BASE_DIR / "Data" / "process" / "label_mappings.json"

# -----------------------------
# Load label mappings
# -----------------------------
with open(LABEL_PATH, "r") as f:
    mappings = json.load(f)

id_to_label = {int(k): v for k, v in mappings["id_to_label"].items()}

# -----------------------------
# Load model + tokenizer
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

model.to(device)
model.eval()

print("Model loaded successfully.")

# -----------------------------
# Flask App
# -----------------------------
app = Flask(__name__)


# -----------------------------
# Prediction Function
# -----------------------------
def predict_department(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)

    probs = probabilities.cpu().numpy()[0]
    top_indices = probs.argsort()[-3:][::-1]

    results = []
    for idx in top_indices:
        results.append({
            "department": id_to_label[idx],
            "confidence": round(float(probs[idx]), 4)
        })

    return results


# -----------------------------
# Routes
# -----------------------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Medical Department Prediction API Running"})


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if not data or "text" not in data:
        return jsonify({"error": "Please provide symptom text"}), 400

    text = data["text"]

    results = predict_department(text)

    return jsonify({
        "input_text": text,
        "top_predictions": results,
        "final_prediction": results[0]
    })


# -----------------------------
# Run Server
# -----------------------------
if __name__ == "__main__":
    app.run()
