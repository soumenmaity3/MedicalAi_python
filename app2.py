import streamlit as st
from spellchecker import SpellChecker
import torch
import json
import os
import zipfile
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# -----------------------------
# Streamlit Config
# -----------------------------
st.set_page_config(
    page_title="Medical AI - Symptom Checker",
    page_icon="🩺",
    layout="centered"
)

# -----------------------------
# Paths & Constants
# -----------------------------
PROJECT_ROOT = Path(__file__).parent
LOCAL_MODEL_PATH = PROJECT_ROOT / "Model" / "trained_model"
MAPPING_PATH = PROJECT_ROOT / "Data" / "process" / "label_mappings.json"

HF_MODEL_NAME = "sm89/Symptom2Disease"
GOOGLE_DRIVE_FILE_ID = "1A3diiWfUX30I9jGwkmgwDzqzsgzSfd5W"
ZIP_PATH = PROJECT_ROOT / "trained_model.zip"

spell = SpellChecker()

# -----------------------------
# Spell Correction
# -----------------------------
def correct_spelling(text):
    words = text.lower().split()
    corrected = []
    for w in words:
        if w.isalpha():
            corrected.append(spell.correction(w) or w)
        else:
            corrected.append(w)
    return " ".join(corrected)

# -----------------------------
# Google Drive Download
# -----------------------------
def download_from_google_drive():
    try:
        import gdown
    except:
        st.error("Install gdown using: pip install gdown")
        return False

    try:
        st.info("Downloading model from Google Drive...")
        os.makedirs(LOCAL_MODEL_PATH.parent, exist_ok=True)

        url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}"
        gdown.download(url, str(ZIP_PATH), quiet=False)

        with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
            zip_ref.extractall(LOCAL_MODEL_PATH.parent)

        os.remove(ZIP_PATH)

        st.success("Model downloaded successfully.")
        return True

    except Exception as e:
        st.error(f"Download failed: {e}")
        return False

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model(source):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if source == "Local Model":
        if not LOCAL_MODEL_PATH.exists():
            st.error("Local model not found.")
            return None, None
        model_name = LOCAL_MODEL_PATH

    elif source == "Hugging Face":
        model_name = HF_MODEL_NAME

    else:  # Google Drive
        if not LOCAL_MODEL_PATH.exists():
            if not download_from_google_drive():
                return None, None
        model_name = LOCAL_MODEL_PATH

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)

        model.to(device)
        model.eval()

        return model, tokenizer

    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

# -----------------------------
# Load Label Mapping
# -----------------------------
@st.cache_data
def load_label_mapping():
    if not MAPPING_PATH.exists():
        st.error("Label mapping file not found.")
        return None

    with open(MAPPING_PATH, "r") as f:
        mappings = json.load(f)

    return mappings["id_to_label"]

# -----------------------------
# Main UI
# -----------------------------
def main():
    st.title("🩺 Symptom to Disease Predictor")
    st.markdown("Describe your symptoms and let AI suggest the department.")

    # Sidebar Options
    st.sidebar.header("Model Source")
    model_source = st.sidebar.radio(
        "Select Model Source:",
        ["Local Model", "Hugging Face", "Google Drive"]
    )

    # Load Model
    model, tokenizer = load_model(model_source)
    id_to_label = load_label_mapping()

    if not model or not id_to_label:
        st.stop()

    user_input = st.text_area(
        "Enter your symptoms:",
        height=120,
        placeholder="E.g., I have fever and headache..."
    )

    if st.button("Analyze Symptoms"):
        if not user_input.strip():
            st.warning("Please enter symptoms.")
            return

        with st.spinner("Analyzing..."):

            device = model.device

            inputs = tokenizer(
                correct_spelling(user_input),
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=128
            )

            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)
                probabilities = torch.softmax(outputs.logits, dim=1)

            confidence, predicted_class_id = torch.max(probabilities, dim=1)

            pred_idx = str(predicted_class_id.item())
            predicted_label = id_to_label.get(pred_idx, "Unknown")
            conf_score = confidence.item() * 100

            st.success("Analysis Complete")
            st.metric("Predicted Department", predicted_label)
            st.progress(int(conf_score), text=f"Confidence: {conf_score:.2f}%")

            top_probs, top_indices = torch.topk(probabilities, 3)

            with st.expander("See Top 3 Predictions"):
                for prob, idx in zip(top_probs[0], top_indices[0]):
                    idx_str = str(idx.item())
                    label = id_to_label.get(idx_str, "Unknown")
                    st.write(f"- **{label}**: {prob.item()*100:.2f}%")

if __name__ == "__main__":
    main()
