import os
import sys
from dataclasses import dataclass
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from src.exception import CustomException
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()

    def get_preprocessor(self):
        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender", "race_ethnicity", "parental_level_of_education",
                "lunch", "test_preparation_course"
            ]

            num_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])
            cat_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("one_hot", OneHotEncoder()),
                ("scaler", StandardScaler(with_mean=False))
            ])

            preprocessor = ColumnTransformer([
                ("num", num_pipeline, numerical_columns),
                ("cat", cat_pipeline, categorical_columns)
            ])

            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            target_col = "math_score"

            X_train = train_df.drop(columns=[target_col])
            y_train = train_df[target_col]
            X_test = test_df.drop(columns=[target_col])
            y_test = test_df[target_col]

            preprocessor = self.get_preprocessor()
            X_train_arr = preprocessor.fit_transform(X_train)
            X_test_arr = preprocessor.transform(X_test)

            train_arr = np.c_[X_train_arr, y_train.to_numpy()]
            test_arr = np.c_[X_test_arr, y_test.to_numpy()]

            save_object(self.config.preprocessor_obj_file_path, preprocessor)

            return train_arr, test_arr, self.config.preprocessor_obj_file_path
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    from data_ingestion import DataIngestion
    ingestion = DataIngestion()
    train_path, test_path = ingestion.initiate_data_ingestion(r"D:\\mlproject\\notebook\\data\\stud.csv")

    transformation = DataTransformation()
    train_arr, test_arr, _ = transformation.initiate_data_transformation(train_path, test_path)
