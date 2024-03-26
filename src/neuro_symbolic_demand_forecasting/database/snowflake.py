import pandas as pd
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
            result = conn.execute(query)
            df = pd.DataFrame(result.fetchall())
            df.columns = result.keys()
            return df

    def fetch_one(self, query: str) -> pd.DataFrame:
        with self.engine.connect() as conn:
            result = conn.execute(query)
            df = pd.DataFrame(result.fetchone())
            df.columns = result.keys()
            return df
