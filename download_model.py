import os
import gdown
import zipfile
from pathlib import Path

from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "sm89/Symptom2Disease"

def download_model():
    print("Downloading model from Hugging Face (if not cached)...")

    AutoTokenizer.from_pretrained(MODEL_NAME)
    AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

    print("Model ready.")