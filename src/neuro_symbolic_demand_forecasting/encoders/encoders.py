import datetime as dt

import numpy as np
import pytz
from astral import LocationInfo
from astral.sun import sun

AMS_TZ = pytz.timezone('Europe/Amsterdam')


def create_encoders(model_type: str) -> dict:
    city = LocationInfo("Amsterdam", "Netherlands", "Europe/Amsterdam")
    part_of_day_encoder = PartOfDayEncoder(city, AMS_TZ)

    if model_type == 'TFT':
        encoders = {
            # 'cyclic': {'future': ['month', 'day']},
            'datetime_attribute': {'future': ['dayofweek'],
                                   'past': ['dayofweek']},
            'custom': {'past': [encode_ptu], 'future': [part_of_day_encoder.encode]},
            # 'position': {'past': ['relative'], 'future': ['relative']},
            # 'transformer': Scaler(),
            'tz': 'Europe/Amsterdam'
        }
    else:
        encoders = {
            'datetime_attribute': {'future': ['dayofweek'],
                                   },
            'custom': {
                'future': [encode_ptu, part_of_day_encoder.encode]
            },
            # 'position': {'past': ['relative'], 'future': ['relative']},
            # 'transformer': Scaler(),
            'tz': 'Europe/Amsterdam'
        }
    return encoders


LSTM_MAPPING = {
    "smart_meter": [0, 0],
    # past future_covariates
    "past_wind_speed": [1, 0],
    "past_global_radiation": [1, 1],
    "past_air_pressure": [1, 2],
    "past_air_temperature": [1, 3],
    "past_relative_humidity": [1, 4],
    "past_day_of_week": [1, 5],
    "past_ptu": [1, 6],
    "past_part_of_day": [1, 7],
    # future_covariates
    "future_wind_speed": [2, 0],
    "future_global_radiation": [2, 1],
    "future_air_pressure": [2, 2],
    "future_air_temperature": [2, 3],
    "future_relative_humidity": [2, 4],
    "future_day_of_week": [2, 5],
    "future_ptu": [2, 6],
    "future_part_of_day": [2, 7],
}


def encode_ptu(idx):
    total_minutes = idx.hour * 60 + idx.minute
    ptu = total_minutes // 15
    return ptu + 1


class PartOfDayEncoder:
    def __init__(self, city: LocationInfo, timezone):
        self.city = city
        self.timezone = timezone

    def encode(self, idx):
        results = []
        for timestamp in idx:
            sun_info = sun(self.city.observer, date=timestamp.date())
            sunrise = sun_info['sunrise'].astimezone(self.timezone).time()
            sunset = sun_info['sunset'].astimezone(self.timezone).time()

            if timestamp.time() < sunrise or timestamp.time() > sunset:
                results.append(0)  # Night
            elif sunrise <= timestamp.time() < dt.time(9, 0, 0):
                results.append(0.25)  # Morning
            elif dt.time(9, 0, 0) <= timestamp.time() < dt.time(13, 0, 0):
                results.append(0.5)  # Midday
            elif dt.time(13, 0, 0) <= timestamp.time() < dt.time(17, 0, 0):
                results.append(0.75)  # Afternoon
            else:
                results.append(1)  # Evening

        return np.array(results)
