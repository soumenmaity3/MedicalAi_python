import os
import sys
from pathlib import Path

#Project parent path (MedicalAi)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))


from kaggle.api.kaggle_api_extended import KaggleApi
import stat

from utils.logger import logging
from utils.exception import CustomException

class SetupKaggle:
    def __init__(self):
        self.KAGGLE_CONFIG_DIR = Path.home()/'.kaggle'
        self.KAGGLE_JSON_FILE = self.KAGGLE_CONFIG_DIR/'kaggle.json'
        self.DATASET_NAME = 'itachi9604/disease-symptom-description-dataset'
        
        #project absolute path
        project_root = Path(__file__).parent.parent.parent
        
        # make download and extract path
        os.makedirs(project_root/'Data'/'raw',exist_ok=True)
        os.makedirs(project_root/'Data'/'extract',exist_ok=True)
                
        # store data on this path
        self.DOWNLOAD_PATH = project_root/'Data'/'raw'
        self.EXTRACT_PATH = project_root/'Data'/'extract'
        
    def check_kaggle_json(self):
        # Check kaggle file exist or not
        if not self.KAGGLE_JSON_FILE.exists():
            logging.error(f'kaggle.json not found. Expected Location: {self.KAGGLE_JSON_FILE}',sys)
        else:
            logging.info(f'File found at: {self.KAGGLE_JSON_FILE}')
            try:
                self.KAGGLE_JSON_FILE.chmod(stat.S_IRUSR | stat.S_IWUSR)
                logging.info('File permission fixed')
            except Exception as e:
                logging.error('File permission error',sys)
                raise CustomException(e,sys)
            
    def authenticate_kaggle(self):
        # Authenticate with kaggle api
        os.environ['KAGGLE_CONFIG_DIR'] = str(self.KAGGLE_CONFIG_DIR)
        
        try:
            self.api = KaggleApi()
            self.api.authenticate()
            logging.info('API successfully authenticated')
        except Exception as e:
            logging.error(f"Authenticate failed: {e}",sys)
            raise CustomException(e,sys)
        
        
    def run(self):
        self.check_kaggle_json()
        self.authenticate_kaggle()
        return self.EXTRACT_PATH,self.DATASET_NAME,self.DOWNLOAD_PATH,self.api