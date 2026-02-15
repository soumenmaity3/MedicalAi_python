import os
import sys
import time
import errno
import torch
from pathlib import Path
from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score
from transformers import get_linear_schedule_with_warmup

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..','..','..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..', 'Component')))

from utils.logger import logging
from utils.exception import CustomException
from Component.training_complete_data import TrainingCompleteData


class TrainAndSaveModel:
    def __init__(self):
        try:
            logging.info("Initializing Training Pipeline...")

            # Load data + model
            self.data_module = TrainingCompleteData()
            self.train_loader, self.val_loader, self.tokenizer = self.data_module.data_loader()
            self.model = self.data_module.model

            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.model.to(self.device)

            logging.info(f"Using device: {self.device}")

            # -----------------------
            # Freeze lower layers
            # -----------------------
            # Freeze all distilbert layers
            for param in self.model.distilbert.parameters():
                param.requires_grad = False

            # Unfreeze last 2 transformer layers
            for name, param in self.model.distilbert.named_parameters():
                if any(layer in name for layer in ["transformer.layer.4", "transformer.layer.5"]):
                    param.requires_grad = True

            # Ensure classifier is trainable
            for param in self.model.classifier.parameters():
                param.requires_grad = True


            logging.info("Partial fine-tuning enabled (last 3 layers + classifier).")

            # -----------------------
            # Optimizer (Differential LR)
            # -----------------------
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters()
                               if "bert" in n and p.requires_grad],
                    "lr": 2e-5,
                    "weight_decay": 0.01
                },
                {
                    "params": [p for n, p in self.model.named_parameters()
                               if "classifier" in n],
                    "lr": 5e-5,
                    "weight_decay": 0.01
                }
            ]

            self.optimizer = torch.optim.AdamW(optimizer_grouped_parameters)

            self.epochs = 4
            self.patience = 2
            self.max_grad_norm = 1.0

            total_steps = len(self.train_loader) * self.epochs
            warmup_steps = int(0.1 * total_steps)

            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps
            )

            # Save paths
            self.project_root = Path(__file__).parent.parent.parent
            self.model_save_path = self.project_root / 'Model' / 'trained_model'
            self.checkpoints_path = self.project_root / 'Model' / 'trained_model_checkpoints'
            self.save_retries = 3
            self.save_retry_wait_sec = 1.5

        except Exception as e:
            logging.error(f"Initialization error: {e}")
            raise CustomException(e, sys)

    # -------------------------------------------------------
    # TRAINING FUNCTION
    # -------------------------------------------------------
    def train(self):
        logging.info("Training started...")
        best_val_loss = float("inf")
        epochs_without_improvement = 0

        loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

        for epoch in range(self.epochs):

            # -----------------------
            # TRAIN
            # -----------------------
            self.model.train()
            total_train_loss = 0.0

            for batch in self.train_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}

                outputs = self.model(**batch)
                logits = outputs.logits
                loss = loss_fn(logits, batch["labels"])

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                self.scheduler.step()

                total_train_loss += loss.item()

            avg_train_loss = total_train_loss / len(self.train_loader)

            # -----------------------
            # VALIDATION
            # -----------------------
            self.model.eval()
            total_val_loss = 0.0
            preds, true_labels = [], []

            with torch.no_grad():
                for batch in self.val_loader:
                    batch = {k: v.to(self.device) for k, v in batch.items()}

                    outputs = self.model(**batch)
                    logits = outputs.logits
                    loss = loss_fn(logits, batch["labels"])
                    total_val_loss += loss.item()

                    predictions = torch.argmax(logits, dim=1)
                    preds.extend(predictions.cpu().numpy())
                    true_labels.extend(batch["labels"].cpu().numpy())

            avg_val_loss = total_val_loss / len(self.val_loader)
            val_acc = accuracy_score(true_labels, preds)
            val_f1 = f1_score(true_labels, preds, average='weighted')

            logging.info(
                f"Epoch {epoch+1}/{self.epochs} | "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {avg_val_loss:.4f} | "
                f"Val Acc: {val_acc:.4f} | "
                f"Val F1: {val_f1:.4f}"
            )

            print(
                f"Epoch {epoch+1}/{self.epochs} | "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {avg_val_loss:.4f} | "
                f"Val Acc: {val_acc:.4f} | "
                f"Val F1: {val_f1:.4f}"
            )

            # -----------------------
            # Early Stopping on VAL LOSS
            # -----------------------
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_without_improvement = 0
                self.save_model(epoch + 1, val_f1)
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= self.patience:
                logging.info(f"Early stopping triggered at epoch {epoch+1}")
                break

        logging.info("Training completed.")

    # -------------------------------------------------------
    # SAVE MODEL
    # -------------------------------------------------------
    def save_model(self, epoch, val_f1):
        try:
            logging.info(f"Saving model (epoch {epoch})...")

            os.makedirs(self.model_save_path, exist_ok=True)

            self.model.save_pretrained(self.model_save_path)
            self.tokenizer.save_pretrained(self.model_save_path)

            logging.info("Model saved successfully.")

        except Exception as e:
            logging.error(f"Error saving model: {e}")
            raise CustomException(e, sys)


if __name__ == "__main__":
    trainer = TrainAndSaveModel()
    trainer.train()
