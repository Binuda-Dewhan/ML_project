import os
import sys

import numpy as np
import pandas as pd
import dill

from src.exception import CustomException
from src.logger import logging

from sklearn.metrics import r2_score

def save_object(file_path, obj):
    """
    Saves an object to a file using joblib.
    
    Parameters:
    - file_path (str): The path where the object will be saved.
    - obj: The object to be saved.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file:
            dill.dump(obj, file)
        

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_model(X_train,y_train,X_test,y_test , models):
    """
    Evaluates the performance of multiple models on the given dataset.
    
    Parameters:
    - X (np.ndarray): Feature matrix.
    - y (np.ndarray): Target vector.
    - models (dict): Dictionary of model names and their instances.
    
    Returns:
    - dict: A dictionary containing model names and their respective scores.
    """
    try:
        report = {}

        for i in range(len(list(models))):

            model = list(models.values())[i]
            
            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

            logging.info(f"{list(models.keys())[i]}: {test_model_score}")
            # logging.info(f"Train Score: {train_model_score}, Test Score: {test_model_score}")

        return report  

    except Exception as e:
        raise CustomException(e, sys)