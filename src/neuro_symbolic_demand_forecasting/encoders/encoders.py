import datetime as dt

import holidays
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
            'custom': {
                'past': [part_of_day_encoder.encode],
                'future': [part_of_day_encoder.encode]
            },
            # 'position': {'past': ['relative'], 'future': ['relative']},
            # 'transformer': Scaler(),
            'tz': 'Europe/Amsterdam'
        }
    else:  # LSTM
        encoders = {
            'datetime_attribute': {'future': ['dayofweek'],
                                   'past': ['dayofweek'],
                                   },
            'custom': {
                'future': [encode_ptu, part_of_day_encoder.encode, encode_is_holiday],
                'past': [encode_ptu, part_of_day_encoder.encode, encode_is_holiday],
            },
            # 'position': {'past': ['relative'], 'future': ['relative']},
            # 'transformer': Scaler(),
            'tz': 'Europe/Amsterdam'
        }
    return encoders

TFT_MAPPING = {
    "smart_meter": [0, 0],
    "is_holiday_input": [0, 1],
    # past future_covariates, corresponding to the order of the features in timeseries
    "past_wind_speed": [1, 0],
    "past_global_radiation": [1, 1],
    "past_air_pressure": [1, 2],
    "past_air_temperature": [1, 3],
    "past_relative_humidity": [1, 4],
    "past_day_of_week": [1, 5],
    # "past_ptu": [1, 6],
    # "past_part_of_day": [1, 7],
    # past future_covariates, corresponding to the order of the features in timeseries
    "past_future_wind_speed": [2, 0],
    "past_future_global_radiation": [2, 1],
    "past_future_air_pressure": [2, 2],
    "past_future_air_temperature": [2, 3],
    "past_future_relative_humidity": [2, 4],
    "past_future_day_of_week": [2, 5],
    "past_future_part_of_day": [2, 6],
    # future covariates, corresponding to the order of the features in timeseries
    "future_wind_speed": [3, 0],
    "future_global_radiation": [3, 1],
    "future_air_pressure": [3, 2],
    "future_air_temperature": [3, 3],
    "future_relative_humidity": [3, 4],
    "future_day_of_week": [3, 5],
    "future_part_of_day": [3, 6],
    # static covariates
    "is_pv": [4, 0],
    "is_holiday_target": [5, 1],
}

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


def encode_holiday(idx):
    hl = holidays.country_holidays('NL', years=idx.year)
    # if f'{id.year}-{id.month}-{id.day}' in hl: # would only encode holidays
    if not hl.is_workday(f'{idx.year}-{idx.month}-{idx.day}'):  # includes weekends
        return 1
    else:
        return 0


def encode_is_holiday(idx):
    return idx.map(encode_holiday)


class PartOfDayEncoder:
    def __init__(self, city: LocationInfo, timezone):
        self.city = city
        self.timezone = timezone

    def _encode_part_of_day(self, timestamp):
        sun_info = sun(self.city.observer, date=timestamp.date())
        sunrise = sun_info['sunrise'].astimezone(self.timezone)
        sunset = sun_info['sunset'].astimezone(self.timezone).time()

        morning_start = (sunrise - dt.timedelta(hours=2)).time()
        if timestamp.time() < morning_start or timestamp.time() > sunset:
            return 0  # Night
        elif morning_start <= timestamp.time() < dt.time(10, 0, 0):
            return 0.25  # Morning
        elif dt.time(10, 0, 0) <= timestamp.time() < dt.time(13, 0, 0):
            return 0.5  # Midday
        elif dt.time(13, 0, 0) <= timestamp.time() < dt.time(16, 0, 0):
            return 0.75  # Afternoon
        else:
            return 1  # Evening

    def encode(self, idx):
        return idx.map(self._encode_part_of_day)
