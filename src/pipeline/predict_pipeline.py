import sys
import os
import pandas as pd

from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            data_scaled = preprocessor.transform(features)
            predictions = model.predict(data_scaled)

            return predictions

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(
        self,
        country_code: int,
        city: str,
        locality: str,
        cuisines: str,
        longitude: float,
        latitude: float,
        average_cost_for_two: int,
        currency: str,
        has_table_booking: str,
        has_online_delivery: str,
        is_delivering_now: str,
        price_range: int,
        votes: int,
        rating_color: str,
    ):
        self.country_code = country_code
        self.city = city
        self.locality = locality
        self.cuisines = cuisines
        self.longitude = longitude
        self.latitude = latitude
        self.average_cost_for_two = average_cost_for_two
        self.currency = currency
        self.has_table_booking = has_table_booking
        self.has_online_delivery = has_online_delivery
        self.is_delivering_now = is_delivering_now
        self.price_range = price_range
        self.votes = votes
        self.rating_color = rating_color

    def get_data_as_frame(self):
        try:
            custom_data_input_dict = {
                "Country Code": [self.country_code],
                "City": [self.city],
                "Locality": [self.locality],
                "Cuisines": [self.cuisines],
                "Longitude": [self.longitude],
                "Latitude": [self.latitude],
                "Average Cost for two": [self.average_cost_for_two],
                "Currency": [self.currency],
                "Has Table booking": [self.has_table_booking],
                "Has Online delivery": [self.has_online_delivery],
                "Is delivering now": [self.is_delivering_now],
                "Price range": [self.price_range],
                "Votes": [self.votes],
                "Rating color": [self.rating_color]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)


