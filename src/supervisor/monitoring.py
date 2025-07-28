# src/supervisor/monitoring.py

import json
import logging
import time
from datetime import datetime

from google.cloud.firestore import Client

from src.database.firestore_manager import FirestoreManager

logger = logging.getLogger(__name__)


class Monitoring:
    def __init__(self, firestore_manager: FirestoreManager, log_file="monitoring_log.json"):
        self.firestore_manager = firestore_manager
        self.log_file = log_file
        self.start_time = time.time()

    def record_heartbeat(self):
        """Records a heartbeat to indicate the bot is alive and running."""
        uptime = time.time() - self.start_time
        heartbeat_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "status": "alive",
            "uptime_seconds": uptime,
        }
        self._log_to_file(heartbeat_data)
        self.firestore_manager.db.collection("monitoring").document("heartbeat").set(heartbeat_data)
        logger.info("Heartbeat recorded.")

    def record_trade(self, trade_data: dict):
        """Records the details of a trade."""
        self._log_to_file(trade_data)
        self.firestore_manager.db.collection("trades").add(trade_data)
        logger.info(f"Trade recorded: {trade_data}")

    def record_error(self, error_message: str):
        """Records an error."""
        error_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "error": error_message,
        }
        self._log_to_file(error_data)
        self.firestore_manager.db.collection("errors").add(error_data)
        logger.error(f"Error recorded: {error_message}")

    def record_performance_metrics(self, metrics: dict):
        """Records performance metrics."""
        performance_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "metrics": metrics,
        }
        self._log_to_file(performance_data)
        self.firestore_manager.db.collection("performance").add(performance_data)
        logger.info(f"Performance metrics recorded: {metrics}")

    def _log_to_file(self, data: dict):
        """Logs data to a local JSON file."""
        with open(self.log_file, "a") as f:
            f.write(json.dumps(data) + "\n")
