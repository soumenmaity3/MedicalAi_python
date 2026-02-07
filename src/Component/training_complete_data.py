import os
import sys
import torch
from torch.utils.data import DataLoader, Dataset

# Add project root to path (MedicalAi)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..','..','..')))


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..', 'Pipeline')))

from utils.logger import logging
from utils.exception import CustomException
from data_ingestion import DataIngestion
from Pipeline.load_model import LoadModel

# Dataset class
class TextDataset(Dataset):
    def __init__(self, dataframe):
        self.df = dataframe

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return {
            "text": self.df.iloc[idx]["text"],
            "label": self.df.iloc[idx]["label_id"]
        }

class TrainingCompleteData:
    def __init__(self):
        try:
            # 1. Ingest Data First
            logging.info("Starting Data Ingestion in Training Pipeline...")
            self.ingestion = DataIngestion()
            self.train_df, self.val_df = self.ingestion.run()
            
            # 2. Get Label Count from Ingestion
            # DataIngestion creates a 'label_to_id' dict. we can use its length.
            self.num_labels = len(self.ingestion.label_to_id)
            logging.info(f"Unique labels found: {self.num_labels}")
            
            # 3. Load Model with correct label count
            logging.info("Loading Model...")
            self.model_loader = LoadModel(num_labels=self.num_labels)
            self.model, self.tokenizer = self.model_loader.run()
            
        except Exception as e:
            logging.error(f"Error in TrainingCompleteData init: {e}")
            raise CustomException(e, sys)
            
    def collate_fn(self, batch):
        texts = [item["text"] for item in batch]
        labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)

        encoding = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )

        encoding["labels"] = labels
        return encoding
    
    def data_loader(self):
        try:
            logging.info("Creating DataLoaders...")
            train_loader = DataLoader(
                TextDataset(self.train_df),
                batch_size=16,
                shuffle=True,
                collate_fn=self.collate_fn
            )

            val_loader = DataLoader(
                TextDataset(self.val_df),
                batch_size=16,
                shuffle=False,
                collate_fn=self.collate_fn
            )
            
            return train_loader, val_loader,self.tokenizer
            
        except Exception as e:
            logging.error(f"Error creating dataloaders: {e}")
            raise CustomException(e, sys)

# # For testing
# if __name__ == "__main__":
#     trainer = TrainingCompleteData()
#     tr_loader, val_loader = trainer.data_loader()
    
#     # Test a batch
#     batch = next(iter(tr_loader))
#     print("Batch Keys:", batch.keys())
#     print("Labels Shape:", batch['labels'].shape)
#     print("Input IDs Shape:", batch['input_ids'].shape)
