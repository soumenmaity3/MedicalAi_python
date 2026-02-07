import os
import sys
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Add project root to path (MedicalAi)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..','..','..')))

from utils.logger import logging
from utils.exception import CustomException

class LoadModel:
    def __init__(self, num_labels, model_name="dmis-lab/biobert-v1.1"):
        self.num_labels = num_labels
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        logging.info(f"Loading Model: {model_name} with {num_labels} labels")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
            hidden_dropout_prob=0.3,
            attention_probs_dropout_prob=0.3,
            classifier_dropout=0.4
        )
        
    def run(self):
        return self.model, self.tokenizer
