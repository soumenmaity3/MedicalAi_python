import sys
import os

# Add the parent directory (MedicalAi) to sys.path to access utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from utils.logger import logging
from utils.exception import CustomException

def main():
    try:
        logging.info("Starting the main execution")
        print("Welcome to Symptom2Disease Project")
        # Add your pipeline execution logic here
        
    except Exception as e:
        logging.error("An error occurred in main")
        raise CustomException(e, sys)

if __name__ == "__main__":
    main()
