# src/supervisor/monitoring.py

import json
import logging
import time
from datetime import datetime

# from google.cloud.firestore import Client # Not directly used, remove if not needed

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
        # Fixed: Access _db directly
        if self.firestore_manager._db:
            self.firestore_manager._db.collection("monitoring").document("heartbeat").set(heartbeat_data)
        else:
            self.logger.warning("Firestore not initialized, cannot record heartbeat to Firestore.")
        logger.info("Heartbeat recorded.")

    def record_trade(self, trade_data: dict):
        """Records the details of a trade."""
        self._log_to_file(trade_data)
        # Fixed: Access _db directly
        if self.firestore_manager._db:
            self.firestore_manager._db.collection("trades").add(trade_data)
        else:
            self.logger.warning("Firestore not initialized, cannot record trade to Firestore.")
        logger.info(f"Trade recorded: {trade_data}")

    def record_error(self, error_message: str):
        """Records an error."""
        error_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "error": error_message,
        }
        self._log_to_file(error_data)
        # Fixed: Access _db directly
        if self.firestore_manager._db:
            self.firestore_manager._db.collection("errors").add(error_data)
        else:
            self.logger.warning("Firestore not initialized, cannot record error to Firestore.")
        logger.error(f"Error recorded: {error_message}")

    def record_performance_metrics(self, metrics: dict):
        """Records performance metrics."""
        performance_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "metrics": metrics,
        }
        self._log_to_file(performance_data)
        # Fixed: Access _db directly
        if self.firestore_manager._db:
            self.firestore_manager._db.collection("performance").add(performance_data)
        else:
            self.logger.warning("Firestore not initialized, cannot record performance metrics to Firestore.")
        logger.info(f"Performance metrics recorded: {metrics}")

    def _log_to_file(self, data: dict):
        """Logs data to a local JSON file."""
        try:
            with open(self.log_file, "a") as f:
                f.write(json.dumps(data) + "\n")
        except Exception as e:
            logger.error(f"Failed to write to monitoring log file {self.log_file}: {e}", exc_info=True)
