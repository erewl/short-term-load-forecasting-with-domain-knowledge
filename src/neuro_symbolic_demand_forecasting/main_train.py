import os
from dotenv import load_dotenv
import datetime as dt

from src.neuro_symbolic_demand_forecasting.database.weather_postgres import WeatherDao
from src.neuro_symbolic_demand_forecasting.database.snowflake import SnowflakeDao


def main_train():
    load_dotenv()
    sf_dao = SnowflakeDao(
        url=os.getenv("SNOWFLAKE_URL"),
        uid=os.getenv("SNOWFLAKE_UID"),
        pwd=os.getenv("SNOWFLAKE_PASSWORD"),
        database=os.getenv("SNOWFLAKE_DATABASE"),
        schema=os.getenv("SNOWFLAKE_SCHEMA"),
    )
    sf_dao.connect()
    pg_dao = WeatherDao(
        url=os.getenv("POSTGRES_HOST_WEATHER_DATA"),
        uid=os.getenv("POSTGRES_UID_WEATHER_DATA"),
        pwd=os.getenv("POSTGRES_PASSWORD_WEATHER_DATA"),
        database=os.getenv("POSTGRES_DATABASE_WEATHER_DATA"),
        schema=os.getenv("POSTGRES_SCHEMA_WEATHER_DATA"),
        port=int(os.getenv("POSTGRES_PORT_WEATHER_DATA"))
    )
    pg_dao.connect()

    # df = pg_dao.fetch_weather_forecast(52.1, 5.18,
    #                                    to_ams_local_tz(dt.datetime(2024, 3, 12, 0, 0)),
    #                                    to_ams_local_tz(dt.datetime(2024, 3, 13, 0, 0)))
    # print(df.columns)


if __name__ == "__main__":
    main_train()
