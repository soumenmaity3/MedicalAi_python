import os
import sys
from transformers import AutoTokenizer, DistilBertForSequenceClassification

# Add project root to path (MedicalAi)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..','..','..')))

from utils.logger import logging
from utils.exception import CustomException

class LoadModel:
    def __init__(self, num_labels, model_name="distilbert-base-uncased"):
        self.num_labels = num_labels
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        logging.info(f"Loading Model: {model_name} with {num_labels} labels")
        self.model = DistilBertForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
            dropout=0.3,
            attention_dropout=0.3  
        )
        
    def run(self):
        return self.model, self.tokenizer
