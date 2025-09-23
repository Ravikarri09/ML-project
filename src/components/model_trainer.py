import os
import sys
from dataclasses import dataclass
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score
from src.utils import save_object, evaluate_models
from src.exception import CustomException

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor()
            }

            model_report = evaluate_models(X_train, y_train, X_test, y_test, models, None)

            best_model_name = max(model_report, key=model_report.get)
            best_model_score = model_report[best_model_name]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No suitable model found with R2 > 0.6", sys)

            save_object(self.config.trained_model_file_path, best_model)
            print(f"Best Model: {best_model_name}, R2 Score: {best_model_score}")
            return best_model_score
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    from data_ingestion import DataIngestion
    from data_transformation import DataTransformation

    ingestion = DataIngestion()
    train_path, test_path = ingestion.initiate_data_ingestion(r"D:\\mlproject\\notebook\\data\\stud.csv")

    transformation = DataTransformation()
    train_arr, test_arr, _ = transformation.initiate_data_transformation(train_path, test_path)

    trainer = ModelTrainer()
    trainer.initiate_model_trainer(train_arr, test_arr)
