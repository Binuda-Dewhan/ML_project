import sys
import os
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        logging.info("Data Ingestion component initialized with config: %s", self.ingestion_config)

    def initiate_data_ingestion(self):
        logging.info("Starting data ingestion process.")
        try:
            df = pd.read_csv('notebook/data/stud.csv')
            logging.info("Data loaded successfully from 'notebook/data/stud.csv'.")

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False)
            logging.info("Raw data saved to %s", self.ingestion_config.raw_data_path)

            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            logging.info("Data split into train and test sets.")

            train_set.to_csv(self.ingestion_config.train_data_path, index=False)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False)
            logging.info("Train and test data saved to %s and %s", 
                         self.ingestion_config.train_data_path, 
                         self.ingestion_config.test_data_path)
            
        except Exception as e:
            raise CustomException(e, sys)
        

if __name__ == "__main__":
    data_ingestion = DataIngestion()
    data_ingestion.initiate_data_ingestion()
    logging.info("Data ingestion process completed successfully.")