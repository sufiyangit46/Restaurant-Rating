import sys
import os
import numpy as np
import pandas as pd
from dataclasses import dataclass

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join(
        "artifacts", "preprocessor.pkl"
    )


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            num_features = [
                "Longitude",
                "Latitude",
                "Average Cost for two",
                "Votes",
                "Price range"
            ]

            cat_features = [
                "Country Code",
                "City",
                "Locality",
                "Cuisines",
                "Currency",
                "Has Table booking",
                "Has Online delivery",
                "Is delivering now",
                "Rating color"
            ]

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False
                    ))
                ]
            )

            preprocessor = ColumnTransformer(
                [
                    ("num", num_pipeline, num_features),
                    ("cat", cat_pipeline, cat_features)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Train and test data loaded")

            target_column_name = "Aggregate rating"

            X_train = train_df.drop(columns=[target_column_name])
            y_train = train_df[target_column_name]

            X_test = test_df.drop(columns=[target_column_name])
            y_test = test_df[target_column_name]

            preprocessing_object = self.get_data_transformer_object()

            X_train_processed = preprocessing_object.fit_transform(X_train)
            X_test_processed = preprocessing_object.transform(X_test)

            train_arr = np.c_[X_train_processed, y_train.to_numpy()]
            test_arr = np.c_[X_test_processed, y_test.to_numpy()]

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_object
            )

            logging.info("Preprocessor saved successfully")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)






