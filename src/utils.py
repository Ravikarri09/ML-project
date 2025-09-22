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
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train, X_test, y_test, models, param=None):
    try:
        report = {}

        for i, (name, model) in enumerate(models.items()):
            # If hyperparameters are provided for this model → use GridSearchCV
            if param is not None and name in param:
                para = param[name]
                gs = GridSearchCV(model, para, cv=3)
                gs.fit(X_train, y_train)
                model.set_params(**gs.best_params_)
                model.fit(X_train, y_train)
            else:
                # Train without hyperparameter tuning
                model.fit(X_train, y_train)

            # Predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Scores (using r2_score since this looks like regression)
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            report[name] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)
