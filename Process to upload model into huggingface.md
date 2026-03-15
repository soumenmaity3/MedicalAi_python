Perfect. I’ll write it clean and structured so you can copy-paste directly as a `.md` file.

---

# 🚀 Symptom2Disease — Full Deployment Guide

## 📌 Overview

This guide covers the complete process:

1. Upload fine-tuned model to Hugging Face
2. Create Docker Space
3. Build FastAPI backend
4. Deploy model API
5. Test using Postman
6. Use in mobile app

---

# 🧠 Part 1 — Upload Fine-Tuned Model to Hugging Face

## Step 1 — Login to Hugging Face

Create a file `hf_login.py`:

```python
from huggingface_hub import login

login()
```

Run:

```bash
python hf_login.py
```

Paste your **Write Token** when asked.

---

## Step 2 — Create Repository

Run in terminal:

```bash
python -c "from huggingface_hub import HfApi; HfApi().create_repo(repo_id='sm89/Symptom2Disease', exist_ok=True)"
```

---

## Step 3 — Upload Model Folder

Make sure your trained model folder contains:

```
config.json
model.safetensors
tokenizer.json
tokenizer_config.json
```

Upload:

```bash
python -c "from huggingface_hub import HfApi; HfApi().upload_folder(folder_path='Model/trained_model', repo_id='sm89/Symptom2Disease')"
```

---

## Step 4 — Verify Upload

Test locally:

```bash
python -c "from transformers import AutoModelForSequenceClassification; AutoModelForSequenceClassification.from_pretrained('sm89/Symptom2Disease'); print('Loaded successfully')"
```

If it loads → upload successful.

---

# 🚀 Part 2 — Create Docker Space API

## Step 1 — Create New Space

Go to:

[https://huggingface.co/new-space](https://huggingface.co/new-space)

Fill:

* Owner: sm89
* Space name: symptom2disease-api
* SDK: **Docker**
* Hardware: CPU Basic

Click **Create Space**

---

# 📁 Add Required Files

Your Space must contain:

```
app.py
requirements.txt
Dockerfile
```

---

# 📄 app.py (Final Version)

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = FastAPI(title="Medical Symptom Prediction API")

MODEL_NAME = "sm89/Symptom2Disease"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()

id_to_label = {
    0: "Dermatology",
    1: "Neurology",
    2: "Cardiology",
    3: "Gastroenterology",
    4: "Orthopedics",
    5: "ENT",
    6: "Pulmonology",
    7: "Urology",
    8: "General Medicine"
}

class PredictionRequest(BaseModel):
    text: str

@app.get("/")
def health_check():
    return {"message": "Medical Symptom API Running"}

@app.post("/predict")
def predict(request: PredictionRequest):

    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text input cannot be empty")

    inputs = tokenizer(
        request.text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=1)

    top_probs, top_indices = torch.topk(probabilities, 3)

    results = []

    for prob, idx in zip(top_probs[0], top_indices[0]):
        label_index = int(idx.item())
        results.append({
            "department": id_to_label.get(label_index, f"LABEL_{label_index}"),
            "confidence": round(float(prob.item()), 4)
        })

    return {
        "input_text": request.text,
        "top_predictions": results,
        "final_prediction": results[0]
    }
```

---

# 📄 requirements.txt

```
fastapi
uvicorn
torch
transformers
```

---

# 📄 Dockerfile

```
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
```

---

# 🚀 Part 3 — Deployment

After committing files:

* Space will start building
* Dependencies will install
* Model will download from Hugging Face
* API will start

Wait until status = **Running**

---

# 🌍 Part 4 — API URL

Your API Base URL:

```
https://sm89-symptom2disease-api.hf.space
```

---

# 🧪 Part 5 — Test with Postman

## Health Check

Method: GET

```
https://sm89-symptom2disease-api.hf.space/
```

Response:

```json
{
  "message": "Medical Symptom API Running"
}
```

---

## Prediction Endpoint

Method: POST

```
https://sm89-symptom2disease-api.hf.space/predict
```

Headers:

```
Content-Type: application/json
```

Body (Raw JSON):

```json
{
  "text": "I have fever and headache"
}
```

---

## Expected Response

```json
{
  "input_text": "I have fever and headache",
  "top_predictions": [
    {
      "department": "Orthopedics",
      "confidence": 0.94
    },
    {
      "department": "ENT",
      "confidence": 0.02
    }
  ],
  "final_prediction": {
    "department": "Orthopedics",
    "confidence": 0.94
  }
}
```

---

# 📱 Part 6 — Use in Mobile App

Call:

```
POST https://sm89-symptom2disease-api.hf.space/predict
```

Send:

```json
{
  "text": "I have chest pain"
}
```

Parse:

```
response.final_prediction.department
response.final_prediction.confidence
```

---

# 🏗 Final Architecture

```
Fine-Tune Model
        ↓
Upload to HF Model Hub
        ↓
Docker Space (FastAPI)
        ↓
Public REST API
        ↓
Mobile App / Web App
```

---

# 🎉 You Now Have

* Custom fine-tuned model
* Hosted on Hugging Face
* Docker-based API
* Production REST endpoint
* Mobile-ready backend

---

If you want next level:

* Add Swagger docs customization
* Add logging
* Add authentication
* Add rate limiting
* Deploy to AWS

Tell me how far you want to go 🚀
