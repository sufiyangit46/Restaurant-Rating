import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_ob_file_path=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        this is responsible for data transformation
        """
        try:
            num_features = [
                'Longitude',
                'Latitude',
                'Average Cost for two',
                'Votes'
            ]

            low_cat_features = [
                'Country Code',
                'Currency',
                'Has Table booking',
                'Has Online delivery',
                'Is delivering now',
                'Rating color'
            ]

            high_cat_features = [
                'City',
                'Locality',
                'Cuisines'
            ]

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), num_features),
                    ('low_cat', OneHotEncoder(handle_unknown='ignore', drop='first'), low_cat_features),
                    ('high_cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1),
                     high_cat_features)
                ],
                remainder='drop'
            )
            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)

    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info('Obtaining preprocessing object')

            preprocessing_object=self.get_data_transformer_object()

            target_column_name='Aggregate rating'
            numerical_columns=['Country Code','Longitude','Latitude','Average Cost for two','Votes','Price range']

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info('Applying preprocessing object on training dataframe and testing dataframe')

            input_feature_train_arr=preprocessing_object.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_object.fit_transform(input_feature_test_df)

            train_arr=np.c_[
                input_feature_train_arr,np.array(target_feature_train_df)
            ]
            test_arr=np.c_[input_feature_test_arr,np.array(target_feature_test_df)]

            logging.info('Saved preprocessing object completed')

            save_object(
                file_path=self.data_transformation_config.preprocessor_ob_file_path,
                obj=preprocessing_object,
            )

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_ob_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)

