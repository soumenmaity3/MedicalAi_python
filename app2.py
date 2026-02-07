import streamlit as st
from spellchecker import SpellChecker
import torch
import json
import os
import sys
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Set page config
st.set_page_config(
    page_title="Medical AI - Symptom Checker",
    page_icon="🩺",
    layout="centered"
)

# --- Path Setup ---
# Project root is current directory if running from Symptom2Disease
PROJECT_ROOT = Path(__file__).parent
MODEL_PATH = PROJECT_ROOT / 'Model' / 'trained_model'
MAPPING_PATH = PROJECT_ROOT / 'Data' / 'process' / 'label_mappings.json'

spell = SpellChecker()

def correct_spelling(text):
    words = text.lower().split()
    corrected = []

    for w in words:
        if w.isalpha():
            corrected.append(spell.correction(w) or w)
        else:
            corrected.append(w)

    return " ".join(corrected)


# --- Load Resources (Cached) ---

@st.cache_resource
def load_model_and_tokenizer():
    if not MODEL_PATH.exists():
        st.error(f"Model not found at {MODEL_PATH}. Please train the model first.")
        return None, None
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        model.to(device)
        model.eval()
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

@st.cache_data
def load_label_mapping():
    if not MAPPING_PATH.exists():
        st.error(f"Label mapping not found at {MAPPING_PATH}.")
        return None
    
    try:
        with open(MAPPING_PATH, 'r') as f:
            mappings = json.load(f)
        return mappings['id_to_label']
    except Exception as e:
        st.error(f"Error loading mappings: {e}")
        return None

# --- UI & Logic ---

def main():
    st.title("🩺 Symptom to Disease Predictor")
    st.markdown("Describe your symptoms in plain English, and the AI will suggest the likely department/condition.")

    # Load resources
    model, tokenizer = load_model_and_tokenizer()
    id_to_label = load_label_mapping()
    
    if not model or not id_to_label:
        st.stop()
        
    # Input
    user_input = st.text_area("Enter your symptoms:", height=100, placeholder="E.g., I have a bad headache and feel dizzy...")
    
    if st.button("Analyze Symptoms"):
        if not user_input.strip():
            st.warning("Please enter some text.")
        else:
            with st.spinner("Analyzing..."):
                # Prediction
                device = model.device
                inputs = tokenizer(
                    correct_spelling(user_input), 
                    return_tensors="pt", 
                    truncation=True, 
                    max_length=128, 
                    padding=True
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = model(**inputs)
                    logits = outputs.logits
                    probabilities = torch.softmax(logits, dim=1)
                    confidence, predicted_class_id = torch.max(probabilities, dim=1)
                
                # Get Result
                pred_idx = str(predicted_class_id.item())
                predicted_label = id_to_label.get(pred_idx, "Unknown")
                conf_score = confidence.item() * 100
                
                # Display Result
                st.success("Analysis Complete")
                st.metric(label="Predicted Condition/Department", value=predicted_label)
                st.progress(int(conf_score), text=f"Confidence: {conf_score:.2f}%")
                
                # Setup details (Optional: Show top 3)
                top_k = 3
                top_probs, top_indices = torch.topk(probabilities, top_k)
                
                with st.expander("See Top 3 Predictions"):
                    for prob, idx in zip(top_probs[0], top_indices[0]):
                        idx_str = str(idx.item())
                        label = id_to_label.get(idx_str, "Unknown")
                        st.write(f"- **{label}**: {prob.item()*100:.2f}%")

if __name__ == "__main__":
    main()
