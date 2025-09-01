# src/database/precomputed_features_manager.py





try:
    from src.database.influxdb_manager import InfluxDBManager
    INFLUXDB_AVAILABLE = True
except Exception:
    InfluxDBManager = None  # type: ignore
    INFLUXDB_AVAILABLE = False


