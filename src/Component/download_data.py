import os
import sys
from pathlib import Path
import stat

# #Project parent path (MedicalAi)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from utils.logger import logging
from utils.exception import CustomException
from setup_kaggle import SetupKaggle

class DownloaAndExtractData:
    def __init__(self):
        self.setup_kaggle = SetupKaggle()
        self.EXTRACT_PATH,self.DATASET_NAME,self.DOWNLOAD_PATH,self.api = self.setup_kaggle.run()
        
        
    def is_dataset_complete(self):
        logging.info('Checking if dataset exist...')
        
        # Check if extract path exists
        logging.info(f'Checking: {self.EXTRACT_PATH}')
        if not self.EXTRACT_PATH.exists():
            logging.info('Dataset folder not found..')
            return False , 'Dataset folder not found'

        # Check if folder is empty (if it has items, any(iterdir()) is True)
        if not any(self.EXTRACT_PATH.iterdir()):
            logging.info('Dataset folder is empty..')
            return False, 'Dataset folder is empty'

        return True, f'Dataset is found at: {self.EXTRACT_PATH}'
    
    def download_dataset(self):
        is_complete,msg = self.is_dataset_complete()
        
        if is_complete:
            logging.info(f'Dataset already downloaded!')
            logging.info(f'Location: {self.EXTRACT_PATH.absolute()}')
            logging.info(f'{msg}')
            logging.info("Download skipped - using existing dataset")
        else:
            logging.info(f'Downloading: {self.DATASET_NAME}')
            print(f'  Destination: {self.DOWNLOAD_PATH.absolute()}')
            
            self.DOWNLOAD_PATH.mkdir(parents=True,exist_ok=True)
            try:
                self.api.dataset_download_files(
                    self.DATASET_NAME,
                    path=str(self.EXTRACT_PATH),
                    unzip=True
                )
                logging.info(f'Download Complete!')
                
                # verify download
                if self.EXTRACT_PATH.exists():
                    file_count = sum([len(files) for r, d,files in os.walk(self.EXTRACT_PATH)])
                    logging.info(f'Dataset verified - {file_count} files found')
                    print(f'Dataset verified - {file_count} files found')
                else:
                    logging.error("Dataset folder not found after extraction")
            except Exception as e:
                logging.error(f'Error Downloading: {e}',sys)
                raise CustomException(f'Error Downloading: {e}',sys)
            
    def run(self):
        self.download_dataset()
        return self.EXTRACT_PATH
    
# if __name__ == "__main__":
#     downloader = DownloaAndExtractData()
#     downloader.run()