import os
import sys
import pandas as pd
import random
from pathlib import Path

# Add project root to path (MedicalAi)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..','..','..')))

from utils.logger import logging
from utils.exception import CustomException
from data_load import DataLoader

class DataTransformation:
    def __init__(self):
        try:
            self.DATA_LOADER = DataLoader()
            
            self.merge_df = self.DATA_LOADER.load_data()
            
            if self.merge_df is None:
                raise CustomException("Merge DataFrame is None", sys)

            self.symptom_cols = [col for col in self.merge_df.columns if col.startswith("Symptom")]
            
            self.disease_to_department = {
                'heart attack':'Cardiology',
                'hypertension':'Cardiology',
                'varicose veins':'Cardiology',
                'migraine':'Neurology',
                'paralysis (brain hemorrhage)':'Neurology',
                '(vertigo) paroymsal positional vertigo':'Neurology',
                'cervical spondylosis':'Neurology',
                'bronchial asthma':'Pulmonology',
                'pneumonia':'Pulmonology',
                'tuberculosis':'Pulmonology',
                'common cold':'Pulmonology',
                'gerd':'Gastroenterology',
                'peptic ulcer diseae':'Gastroenterology',
                'gastroenteritis':'Gastroenterology',
                'jaundice':'Gastroenterology',
                'alcoholic hepatitis':'Gastroenterology',
                'hepatitis a':'Gastroenterology',
                'hepatitis b':'Gastroenterology',
                'hepatitis c':'Gastroenterology',
                'hepatitis d':'Gastroenterology',
                'hepatitis e':'Gastroenterology',
                'fungal infection':'Dermatology',
                'acne':'Dermatology',
                'psoriasis':'Dermatology',
                'impetigo':'Dermatology',
                'allergy':'Dermatology',
                'drug reaction':'Dermatology',
                'malaria':'Infectious',
                'dengue':'Infectious',
                'typhoid':'Infectious',
                'chicken pox':'Infectious',
                'aids':'Infectious',
                'osteoarthristis':'Orthopedics',
                'arthritis':'Orthopedics',
                'diabetes':'Endocrinology',
                'hypothyroidism':'Endocrinology',
                'hyperthyroidism':'Endocrinology',
                'hypoglycemia':'Endocrinology',
                'urinary tract infection':'Urology',
                }
            
            logging.info("Combining symptoms into a single text column...")
            self.merge_df['Symptom_text'] = self.merge_df.apply(self.combine_symptoms, axis=1)
            logging.info("Make Symptoms to user style text...")
            self.merge_df['text'] = self.merge_df.apply(lambda row: self.to_user_style(row['Symptom_text']), axis=1)
            logging.info('Make disease to depertment...')
            self.merge_df['Disease'] = self.merge_df['Disease'].str.lower().str.strip()
            self.merge_df['label'] = self.merge_df['Disease'].map(self.disease_to_department)
            logging.info('Make randomise to all data...')
            self.merge_df = self.merge_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
            try:
                logging.info("Dropping original symptom columns...")
                self.merge_df = self.merge_df.drop(columns=self.symptom_cols)
                logging.info("Dropping Description and Disease column...")
                self.merge_df = self.merge_df.drop(columns=['Description','Symptom_text','Disease'])
            except Exception as e:
                logging.error(f"Dropping Error: {e}",sys)
                raise CustomException(e,sys)
            
            logging.info(f"Data Transformation completed. New shape: {self.merge_df.shape}")
            
            # Fix path handling
            
            self.process_data_path = Path(__file__).parent.parent.parent / 'Data' / 'process'
            os.makedirs(self.process_data_path, exist_ok=True)
            self.merge_df.to_csv(self.process_data_path / 'process_data.csv', index=False)
            # Sticking to valid class definition. The user can call access .merge_df
            
        except Exception as e:
            logging.error(f"Error in DataTransformation init: {e}")
            raise CustomException(e, sys)
        
    def combine_symptoms(self, row):
        symptoms = []
        for col in self.symptom_cols:
            val = row[col]
            if pd.notna(val) and val != 0 and str(val).lower() != 'nan':
                # Basic cleaning: replace underscores
                symptoms.append(str(val).replace("_", " "))
        return " ".join(symptoms)
    
    def to_user_style(self,text):
        words = text.split()
        random.shuffle(words)
        text = " ".join(words)
        templates = [
        "i have {}",
        "i am having {}",
        "suffering from {}",
        "having {}",
        "feeling {}",
        "experiencing {}",
        "dealing with {}",
        "i think it's {}",
        "i feel {} lately",
        "my main issue is {}"
        ]
        return random.choice(templates).format(text)
    
    def run(self):
        return self.merge_df

# if __name__ == "__main__":
#     dt = DataTransformation()
#     print(dt.merge_df.head())