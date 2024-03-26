import datetime

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from src.neuro_symbolic_demand_forecasting.helpers.weather_utils import GridPoints, get_gridpoint


class PostgresDao:
    engine: Engine = None

    features = [
        'wind_speed_10m',  # wind speed in m/s
        'wind_speed_50m',  # wind speed in m/s
        # 'wind_speed_100m', # wind speed in m/s
        'wind_blast_speed_10m',  # wind blast speed in m/s
        't_instant_ground_2m',  # TEMPERATURE in K
        # 't_instant_ground_100m', # TEMPERATURE in K
        'td_instant_ground_2m',  # Dewpoint temperature in K
        'pres_instant_sea_0m',  # PRESSURE at sea level in Pa
        'pres_instant_ground_0m',  # PRESSURE on ground level in Pa
        'r_instant_ground_2m',  # HUMIDITY in %
        'rain_instant_ground_0m',  # RAIN in kg/m**2
        'grad_accum_ground_0m',  # GLOBAL_RADIATION in J/m**2
        'tcc_instant_ground_0m',  # TOTAL CLOUD COVERAGE in %
        'vis_instant_ground_0m'  # visibility in m
    ]

    def __init__(self, url: str, uid: str, pwd: str, database: str, schema: str, port: int):
        self.url = url
        self.uid = uid
        self.pwd = pwd
        self.database = database
        self.schema = schema
        self.port = port

    def connect(self):
        self.engine = create_engine(
            f"postgresql://{self.uid}:{self.pwd}@{self.url}:{self.port}/{self.database}"
        )
        print("Connected to postgres host!")

    def fetch_weather_forecast_by_grid_point(self, grid_point: int, datetime_start: datetime.datetime,
                                             datetime_end: datetime.datetime) -> pd.DataFrame:
        columns = ", ".join(self.features)
        with self.engine.connect() as conn:
            query = f"""
            SELECT grid_point, latitude, longitude, valid_datetime, batch_datetime, {columns} FROM {self.schema}.harmonie_40_forecast 
            WHERE grid_point={grid_point} 
            AND valid_datetime < '{datetime_end}'
            AND valid_datetime >= '{datetime_start}'
            """
            result = conn.execute(query)
            df = pd.DataFrame(result.fetchall())
            df.columns = result.keys()
            return df

    def fetch_weather_forecast(self, lat: float, lon: float, datetime_start: datetime.datetime,
                               datetime_end: datetime.datetime) -> pd.DataFrame:
        gridpoint = get_gridpoint(lat, lon, GridPoints)
        print(lat, lon, "have gridpoint id", gridpoint)
        return self.fetch_weather_forecast_by_grid_point(gridpoint, datetime_start, datetime_end)
