import os 
import sys

import numpy as np
import pandas as pd
import dill
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException

def save_object(file_path, obj):
    """
    Saves an object to a specified file path using dill.

    Parameters:
    - file_path (str): Path where the object will be saved.
    - obj: Python object to be saved.

    Raises:
    - CustomException: If there is an error during saving.
    """
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e,sys)
    
def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    """
    Trains and evaluates multiple models with hyperparameter tuning using GridSearchCV.

    Parameters:
    - X_train, y_train: Training data and labels.
    - X_test, y_test: Testing data and labels.
    - models (dict): Dictionary of models to train and evaluate.
    - param (dict): Dictionary of hyperparameter grids for each model.

    Returns:
    - report (dict): Test R^2 scores for each model.
    
    Raises:
    - CustomException: If there is an error during model training/evaluation.
    """
    try:
        report={}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para=param[list(models.keys())[i]]

            # Perform grid search with cross-validation
            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

            # model.fit(X_train, y_train) # Train model

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = r2_score (y_train, y_train_pred)
            
            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score
        return report
    
    except Exception as e:
        raise CustomException(e,sys)
    
def load_object(file_path):
    """
    Loads an object from a specified file path using dill.

    Parameters:
    - file_path (str): Path from where the object will be loaded.

    Returns:
    - Loaded Python object.
    
    Raises:
    - CustomException: If there is an error during loading.
    """
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
        
    except Exception as e:
        raise CustomException(e,sys)