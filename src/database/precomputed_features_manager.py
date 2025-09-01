# src/database/precomputed_features_manager.py


from datetime import datetime
import json

import pandas as pd

from src.utils.logger import system_logger
from src.utils.error_handler import handle_errors

try:
    from src.database.influxdb_manager import InfluxDBManager
    INFLUXDB_AVAILABLE = True
except Exception:
    InfluxDBManager = None  # type: ignore
    INFLUXDB_AVAILABLE = False


