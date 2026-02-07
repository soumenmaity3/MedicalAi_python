import os
import sys
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..','..','..')))

from utils.logger import logging
from utils.exception import CustomException
from download_data import DownloaAndExtractData

class DataLoader:
    def __init__(self):
        self.DOWNLOAD_DATA = DownloaAndExtractData()
        self.EXTRACT_DATA_PATH = self.DOWNLOAD_DATA.run()
    
    def load_data(self):
        logging.info('Load all data by pandas..')
        self.og_df = pd.read_csv(self.EXTRACT_DATA_PATH/'dataset.csv')
        self.sym_df = pd.read_csv(self.EXTRACT_DATA_PATH/'symptom_Description.csv')
        if not self.og_df.empty and not self.sym_df.empty:
            logging.info('Data loaded successfully')
            logging.info('Now merge them into one DataFrame..')
            self.merge_df = pd.merge(self.og_df,self.sym_df,on='Disease',how='inner')
            logging.info(f'After merge info: og_df - {self.og_df.shape}, sym_df - {self.sym_df.shape} and merge_df - {self.merge_df.shape}')
            return self.merge_df
        else:
            logging.error('Data not loaded',sys)
            raise CustomException('Data not loaded',sys)