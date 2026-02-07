import os
import sys
import json
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit, train_test_split

# Add project root to path (MedicalAi)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..','..','..')))

from utils.logger import logging
from utils.exception import CustomException
from data_transform import DataTransformation

class DataIngestion:
    def __init__(self):
        try:
            logging.info("Initializing Data Ingestion...")
            self.data_transformation = DataTransformation()
            
            logging.info('Load the preprocessed Data')
            # DataTransformation.run() returns the dataframe with 'text' and 'lable' (sic)
            self.merge_df = self.data_transformation.run()
            
            # Filter out any rows where label might be NaN (due to mapping issues)
            if self.merge_df['label'].isnull().any():
                logging.warning(f"Found {self.merge_df['label'].isnull().sum()} rows with NaN labels. Dropping them.")
                self.merge_df = self.merge_df.dropna(subset=['label'])

            # Remove duplicate training examples to reduce memorization risk.
            before_dedup = len(self.merge_df)
            self.merge_df = self.merge_df.drop_duplicates(subset=['text', 'label']).reset_index(drop=True)
            after_dedup = len(self.merge_df)
            logging.info(
                f"Deduplicated text-label rows: removed {before_dedup - after_dedup} "
                f"(from {before_dedup} to {after_dedup})."
            )

            self.process_label_encoding()
            self.train_validation()
            
        except Exception as e:
            logging.error(f"Error in DataIngestion init: {e}")
            raise CustomException(e, sys)
        
    def process_label_encoding(self):
        try:
            logging.info("Creating label to index mapping...")
            # Get unique sorted labels
            unique_labels = sorted(self.merge_df['label'].unique())
            
            # Create mappings
            self.label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
            self.id_to_label = {idx: label for label, idx in self.label_to_id.items()}
            
            # Map labels in dataframe
            self.merge_df['label_id'] = self.merge_df['label'].map(self.label_to_id)
            
            # Define path to save mappings
            self.artifacts_path = Path(__file__).parent.parent.parent / 'Data' / 'process'
            os.makedirs(self.artifacts_path, exist_ok=True)
            
            logging.info('Save refine dataset...')
            self.merge_df.to_csv(self.artifacts_path/'refine_data.csv')
            
            mappings = {
                'label_to_id': self.label_to_id,
                'id_to_label': self.id_to_label
            }
            
            mapping_file = self.artifacts_path / 'label_mappings.json'
            logging.info(f"Saving label mappings to {mapping_file}")
            
            with open(mapping_file, 'w') as f:
                json.dump(mappings, f, indent=4)
                
            logging.info("Label encoding completed and saved.")
            
        except Exception as e:
            logging.error(f"Error in process_label_encoding: {e}")
            raise CustomException(e, sys)
        
    def train_validation(self):
        # Prevent text leakage across splits by grouping on exact text.
        groups = self.merge_df['text'].astype(str)
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
        train_idx, val_idx = next(splitter.split(self.merge_df, groups=groups))

        self.train_df = self.merge_df.iloc[train_idx].reset_index(drop=True)
        self.val_df = self.merge_df.iloc[val_idx].reset_index(drop=True)

        # If group split produces an invalid val set, fall back to stratified split.
        if self.val_df.empty or self.train_df.empty:
            logging.warning("Group split failed; falling back to stratified split.")
            self.train_df, self.val_df = train_test_split(
                self.merge_df,
                test_size=0.3,
                stratify=self.merge_df['label_id'],
                random_state=42
            )

    def run(self):
        return self.train_df,self.val_df

# if __name__ == "__main__":
#     ingestion = DataIngestion()
#     print(ingestion.merge_df.head())
#     print(ingestion.merge_df.shape)
#     print(f"Unique Labels: {len(ingestion.label_to_id)}")
