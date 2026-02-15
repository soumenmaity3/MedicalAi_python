Perfect 👍 I’ll write the full clean Markdown here so you can copy–paste directly into your project as `DEPLOYMENT_GUIDE.md` or update your `README.md`.

Just copy everything below.

---

````md
# Symptom2Disease – Complete Fine-Tuned Model Deployment Guide

This document explains the complete end-to-end process:

1. Fine-tuning a Hugging Face model  
2. Uploading the model to Hugging Face Hub  
3. Verifying the upload  
4. Integrating the model into a Flask API  
5. Running and testing the API  

---

# 1️⃣ Save the Fine-Tuned Model

After training completes, save both the model and tokenizer.

```python
model.save_pretrained("./trained_model")
tokenizer.save_pretrained("./trained_model")
````

Your folder should contain:

```
trained_model/
 ├── config.json
 ├── model.safetensors  (or pytorch_model.bin)
 ├── tokenizer.json
 ├── tokenizer_config.json
 └── special_tokens_map.json (optional)
```

---

# 2️⃣ Install Hugging Face Hub

Inside your activated virtual environment:

```bash
pip install --upgrade huggingface_hub
```

Verify installation:

```bash
pip show huggingface_hub
```

---

# 3️⃣ Login to Hugging Face

Create a token:

👉 [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
Create token with **Write** permission.

Login using Python:

```python
from huggingface_hub import login
login(token="YOUR_HF_TOKEN")
```

Verify login:

```bash
python -c "from huggingface_hub import HfApi; print(HfApi().whoami())"
```

If it prints your username, login is successful.

---

# 4️⃣ Create Model Repository

```bash
python -c "from huggingface_hub import HfApi; HfApi().create_repo(repo_id='sm89/Symptom2Disease', exist_ok=True)"
```

Replace `sm89` with your username if different.

---

# 5️⃣ Upload Model to Hugging Face

From your project root:

```bash
python -c "from huggingface_hub import HfApi; HfApi().upload_folder(folder_path='Model/trained_model', repo_id='sm89/Symptom2Disease')"
```

Wait until upload completes.

If successful, your model is now live at:

```
https://huggingface.co/sm89/Symptom2Disease
```

---

# 6️⃣ Test Loading From Hugging Face

```bash
python -c "from transformers import AutoModelForSequenceClassification; AutoModelForSequenceClassification.from_pretrained('sm89/Symptom2Disease'); print('Loaded successfully')"
```

If it prints success, the model is working.

---

# 7️⃣ Integrate Model into Flask API

Replace local model path with Hugging Face model name.

## Old (Local Path)

```python
MODEL_PATH = BASE_DIR / "Model" / "trained_model"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
```

## New (Hugging Face Hub)

```python
MODEL_NAME = "sm89/Symptom2Disease"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
```

Keep your label mapping file to convert prediction index → department name.

---

# 8️⃣ Prediction Logic

Use softmax to get probabilities:

```python
with torch.no_grad():
    outputs = model(**inputs)
    probabilities = torch.softmax(outputs.logits, dim=1)
```

Sort top predictions:

```python
probs = probabilities.cpu().numpy()[0]
top_indices = probs.argsort()[-3:][::-1]
```

Map to department name using:

```python
id_to_label[idx]
```

---

# 9️⃣ Run Flask Server

```bash
python app.py
```

Test API:

```
POST http://localhost:5000/predict
```

Request Body:

```json
{
  "text": "I have fever and headache"
}
```

Example Response:

```json
{
  "final_prediction": {
    "confidence": 0.5264,
    "department": "Dermatology"
  },
  "input_text": "I have fever and headache",
  "top_predictions": [...]
}
```

---

# 🔟 Production Notes

* Always save label mappings during training
* Check class imbalance before fine-tuning
* Use weighted loss if dataset is imbalanced
* Use Gunicorn for production deployment
* Consider Docker for portability
* Add proper README to Hugging Face repo
* Enable Developer Mode on Windows to remove symlink warning (optional)

---

# Final Architecture

```
Train Model
    ↓
Save Model
    ↓
Upload to Hugging Face
    ↓
Load From Hugging Face in Flask
    ↓
Serve Predictions via API
```

This ensures:

✔ Portability
✔ Reproducibility
✔ Cloud deployment readiness
✔ Clean ML engineering workflow

---

# Status

Model successfully:

* Fine-tuned
* Uploaded
* Loaded from Hub
* Integrated into Flask
* Returning predictions

Deployment pipeline completed successfully.

```

---

If you want, I can now:

- Rewrite this as a professional GitHub README  
- Add architecture diagram  
- Or convert this into a portfolio-ready project documentation  

Your pipeline is officially production structured now 🔥
```
