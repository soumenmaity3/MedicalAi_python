import os
import gdown
import zipfile
from pathlib import Path

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "Model"
MODEL_PATH = MODEL_DIR / "trained_model"

FILE_ID = "1A3diiWfUX30I9jGwkmgwDzqzsgzSfd5W"
ZIP_PATH = BASE_DIR / "trained_model.zip"

# -----------------------------
# Download Model (Build Phase)
# -----------------------------
if not MODEL_PATH.exists():
    print("Downloading model from Google Drive...")

    os.makedirs(MODEL_DIR, exist_ok=True)

    url = f"https://drive.google.com/uc?id={FILE_ID}"
    gdown.download(url, str(ZIP_PATH), quiet=False)

    print("Extracting model...")
    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(MODEL_DIR)

    os.remove(ZIP_PATH)

    print("Model downloaded and ready.")
else:
    print("Model already exists. Skipping download.")
