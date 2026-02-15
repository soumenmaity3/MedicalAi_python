# Symptom2Disease

Symptom2Disease is an NLP-based medical triage project that predicts the most likely **department/condition category** from free-text symptom descriptions.

It includes:
- A training pipeline based on `dmis-lab/biobert-v1.1`
- A Flask inference API (`app.py`)
- A Streamlit UI (`app2.py`)
- Optional automatic pretrained model download from Google Drive

## What This Project Does

The pipeline converts symptom-level disease data into user-style text (for example: "i have headache nausea dizziness") and maps diseases into clinical departments such as:
- Cardiology
- Neurology
- Pulmonology
- Gastroenterology
- Dermatology
- Infectious
- Orthopedics
- Endocrinology
- Urology

A BioBERT classifier is then fine-tuned on these labels.

## Project Structure

```text
Symptom2Disease/
|-- app.py                               # Flask API
|-- app2.py                              # Streamlit app
|-- download_model.py                    # Download pretrained model zip
|-- requirements.txt
|-- Procfile
|-- README.md
|-- src/
|   |-- Component/
|   |   |-- setup_kaggle.py              # Kaggle auth + paths
|   |   |-- download_data.py             # Download/extract dataset
|   |   |-- data_load.py                 # Merge dataset + description
|   |   |-- data_transform.py            # Text generation + label mapping
|   |   |-- data_ingestion.py            # Split + label encoding save
|   |   |-- training_complete_data.py    # DataLoaders + model prep
|   |-- Pipeline/
|   |   |-- load_model.py                # Load BioBERT classifier
|   |   |-- train_and_save_model.py      # Fine-tune + save model
|   |-- main.py
|-- Data/
|   |-- extract/                         # Extracted Kaggle files
|   |-- process/                         # process_data.csv, refine_data.csv, label_mappings.json
|-- Model/
|   |-- trained_model/                   # Saved tokenizer + model weights
```

## Installation

1. Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Install additional runtime packages used by the apps/scripts:

```powershell
pip install flask gdown gunicorn
```

## Training Pipeline

### 1) Kaggle credentials (required for dataset download)
Place your Kaggle API token at:

```text
C:\Users\<your-user>\.kaggle\kaggle.json
```

### 2) Train model

```powershell
python src\Pipeline\train_and_save_model.py
```

This will:
- Download/prepare dataset from Kaggle (`itachi9604/disease-symptom-description-dataset`) if needed
- Generate processed data files in `Data/process/`
- Create `label_mappings.json`
- Fine-tune BioBERT with partial unfreezing + early stopping
- Save model artifacts to `Model/trained_model/`

## Run Inference

### Option A: Streamlit Web App

```powershell
streamlit run app2.py
```

Behavior:
- Loads local model from `Model/trained_model/`
- If missing, tries to download pretrained model from Google Drive
- Accepts natural language symptom text and shows top predictions + confidence

### Option B: Flask API

```powershell
python app.py
```

Server starts at `http://localhost:5000`.

Endpoints:
- `GET /` health message
- `POST /predict` prediction endpoint

Example request:

```powershell
Invoke-RestMethod -Uri "http://localhost:5000/predict" -Method Post -ContentType "application/json" -Body '{"text":"i have chest pain and shortness of breath"}'
```

## Optional: Download Pretrained Model Only

If you want inference without training:

```powershell
python download_model.py
```

This downloads and extracts the pretrained model to `Model/trained_model/`.

## Main Outputs

- `Data/process/process_data.csv` - transformed training text + labels
- `Data/process/refine_data.csv` - encoded labels (`label_id`)
- `Data/process/label_mappings.json` - `label_to_id` and `id_to_label`
- `Model/trained_model/` - tokenizer + classifier files

## Notes

- `requirements.txt` currently contains core ML dependencies; Flask/Gunicorn/Gdown are used by runtime scripts and should also be installed.
- This project is a triage-assist prototype and not a medical diagnosis system.
