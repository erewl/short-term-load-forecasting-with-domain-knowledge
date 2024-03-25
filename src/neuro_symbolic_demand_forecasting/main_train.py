import os
from dotenv import load_dotenv

from src.neuro_symbolic_demand_forecasting.database.snowflake import SnowflakeDao


def main_train():
    load_dotenv()
    dao = SnowflakeDao(
        url=os.getenv("SNOWFLAKE_URL"),
        uid=os.getenv("SNOWFLAKE_UID"),
        pwd=os.getenv("SNOWFLAKE_PASSWORD"),
    )
    dao.connect()


if __name__ == "__main__":
    main_train()
