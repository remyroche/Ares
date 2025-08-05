# src/database/influxdb_manager.py

import influxdb_client
import pandas as pd
from influxdb_client.client.write_api import SYNCHRONOUS

from src.config import INFLUXDB_BUCKET, INFLUXDB_ORG, INFLUXDB_TOKEN, INFLUXDB_URL
from src.utils.logger import logger


class InfluxDBManager:
    """
    Manages connections and data operations with an InfluxDB database.
    This class is optimized for handling time-series financial data.
    """

    def __init__(
        self,
        url=INFLUXDB_URL,
        token=INFLUXDB_TOKEN,
        org=INFLUXDB_ORG,
        bucket=INFLUXDB_BUCKET,
    ):
        """
        Initializes the InfluxDB client.

        Args:
            url (str): The URL of the InfluxDB instance.
            token (str): The authentication token for InfluxDB.
            org (str): The organization to use in InfluxDB.
            bucket (str): The bucket to store data in.
        """
        self.url = url
        self.token = token
        self.org = org
        self.bucket = bucket
        self.client = influxdb_client.InfluxDBClient(
            url=self.url,
            token=self.token,
            org=self.org,
        )
        self.write_api = self.client.write_api(write_options=SYNCHRONOUS)
        self.query_api = self.client.query_api()
        self.logger = logger.getChild("InfluxDBManager")
        self.logger.info("InfluxDBManager initialized with synchronous client.")

    def write_kline_data(
        self,
        df: pd.DataFrame,
        measurement_name: str,
        symbol: str,
        interval: str,
    ):
        """
        Writes a DataFrame of kline data to InfluxDB.

        Args:
            df (pd.DataFrame): DataFrame containing kline data. Must have a 'timestamp' column.
            measurement_name (str): The name of the measurement in InfluxDB (e.g., 'kline_data').
            symbol (str): The trading symbol (e.g., 'BTCUSDT').
            interval (str): The data interval (e.g., '1h').
        """
        df_copy = df.copy()
        df_copy["timestamp"] = pd.to_datetime(df_copy["timestamp"], unit="ms")
        df_copy = df_copy.set_index("timestamp")

        # Add symbol and interval as tags
        df_copy["symbol"] = symbol
        df_copy["interval"] = interval

        # Convert all data columns to float to avoid type conflicts in InfluxDB
        for col in ["open", "high", "low", "close", "volume"]:
            if col in df_copy.columns:
                df_copy[col] = pd.to_numeric(df_copy[col], errors="coerce")

        self.write_api.write(
            bucket=self.bucket,
            record=df_copy,
            data_frame_measurement_name=measurement_name,
            data_frame_tag_columns=["symbol", "interval"],
        )
        self.logger.info(
            f"Successfully wrote {len(df)} rows for {symbol}/{interval} to InfluxDB.",
        )

    def query_kline_data(
        self,
        symbol: str,
        interval: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """
        Queries kline data from InfluxDB for a specific symbol and date range.

        Args:
            symbol (str): The trading symbol (e.g., 'BTCUSDT').
            interval (str): The data interval (e.g., '1h').
            start_date (str): The start date for the query (e.g., '2020-01-01T00:00:00Z').
            end_date (str): The end date for the query (e.g., '2023-01-01T00:00:00Z').

        Returns:
            pd.DataFrame: A DataFrame containing the queried kline data.
        """
        query = f"""
        from(bucket: "{self.bucket}")
          |> range(start: {start_date}, stop: {end_date})
          |> filter(fn: (r) => r["_measurement"] == "kline_data")
          |> filter(fn: (r) => r["symbol"] == "{symbol}")
          |> filter(fn: (r) => r["interval"] == "{interval}")
          |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
          |> keep(columns: ["_time", "open", "high", "low", "close", "volume"])
          |> rename(columns: {{_time: "timestamp"}})
        """
        df = self.query_api.query_data_frame(query=query, org=self.org)
        if isinstance(df, list):
            if not df:
                return pd.DataFrame()
            df = pd.concat(df, ignore_index=True)

        if df.empty:
            return pd.DataFrame()

        # Convert timestamp to milliseconds integer, as expected by the rest of the application
        if "timestamp" in df.columns:
            df["timestamp"] = (
                pd.to_datetime(df["timestamp"]).astype("int64") // 1_000_000
            )

        # Ensure correct data types
        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        return df

    def close(self):
        """Closes the InfluxDB client."""
        self.client.close()
        self.logger.info("InfluxDB client closed.")
