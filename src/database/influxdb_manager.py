# src/database/influxdb_manager.py


import numpy as np
import pandas as pd
import influxdb_client
from influxdb_client.client.write_api import SYNCHRONOUS

from src.config import (
    INFLUXDB_BUCKET,
    INFLUXDB_ORG,
    INFLUXDB_TOKEN,
    INFLUXDB_URL,
)
from src.utils.logger import logger


class InfluxDBManager:
    """
    Manages connections and data operations with an InfluxDB database.
    This class is optimized for handling time-series financial data.
    """

    def __init__(
        self,
        url: str = INFLUXDB_URL,
        token: str = INFLUXDB_TOKEN,
        org: str = INFLUXDB_ORG,
        bucket: str = INFLUXDB_BUCKET,
    ) -> None:
        """
        Initializes the InfluxDB client.

        Args:
            url: The URL of the InfluxDB instance.
            token: The authentication token for InfluxDB.
            org: The organization to use in InfluxDB.
            bucket: The bucket to store data in.
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

    def close(self) -> None:
        """Closes the InfluxDB client."""
        self.client.close()
        self.logger.info("InfluxDB client closed.")
