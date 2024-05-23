import pandas as pd
import polars as pl
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from snowflake.sqlalchemy import URL


class SnowflakeDao:
    engine: Engine = None

    def __init__(self, url: str, uid: str, pwd: str, database: str, schema: str, role: str = "DATA_ENGINEER",
                 warehouse: str = "USER_WH"):
        self.url = url
        self.uid = uid
        self.pwd = pwd
        self.database = database
        self.schema = schema
        self.role = role
        self.warehouse = warehouse

    def connect(self):
        self.engine = create_engine(URL(
            account=self.url,
            user=self.uid,
            password=self.pwd,
            database=self.database,
            schema=self.schema,
            warehouse=self.warehouse,
            role=self.role,
            numpy=True,
        ))

        print("Connected to snowflake instance!")

    def fetch_list(self, query: str) -> pd.DataFrame:
        with self.engine.connect() as conn:
            print("Executing query")
            result = conn.execute(query)
            print("Fetching all")
            fetched_result= result.fetchall()
            print("Writing data to df")
            df = pd.DataFrame(fetched_result)
            # df.columns = result.keys()
            # df = pl.from_pandas(df)
            return df

    def fetch_one(self, query: str) -> pd.DataFrame:
        with self.engine.connect() as conn:
            result = conn.execute(query)
            df = pd.DataFrame(result.fetchone())
            df.columns = result.keys()
            return df
