import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path='artifacts\model.pkl'
            preprocessor_path='artifacts\preprocessor.pkl'
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            data_scaled=preprocessor.transform(features)
            assume=model.predict(data_scaled)
            return assume

        except Exception as e:
            raise CustomException(e,sys)



class CustomData:
    def __init__(
        self,
        country_code: str,
        city: str,
        cuisines: str,
        average_cost_for_two: int,
        currency: str,
        has_table_booking: str,
        has_online_delivery: str,
        is_delivering_now: str,
        price_range: int,
        votes:int
        ):
        self.country_code = country_code
        self.city = city
        self.cuisines = cuisines
        self.average_cost_for_two = average_cost_for_two
        self.currency = currency
        self.has_table_booking = has_table_booking
        self.has_online_delivery = has_online_delivery
        self.is_delivering_now = is_delivering_now
        self.price_range = price_range
        self.votes = votes

    def get_data_as_frame(self):
        try:
            custom_data_input_dict={
                "Country Code":[self.country_code],
                "City":[self.city],
                "Cuisines":[self.cuisines],
                "Average Cost for two":[self.average_cost_for_two],
                "Currency":[self.currency],
                "Has Table booking":[self.has_table_booking],
                "Is delivering now":[self.is_delivering_now],
                "Price range":[self.price_range],
                "Votes":[self.votes],
                "Has Online delivery":[self.has_online_delivery]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e,sys)