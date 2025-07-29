# src/supervisor/monitoring.py

import json
import logging
import time
from datetime import datetime
from typing import Union # Added import for Union

# Import both managers for type hinting, but use the one passed in __init__
from src.database.firestore_manager import FirestoreManager
from src.database.sqlite_manager import SQLiteManager

logger = logging.getLogger(__name__)


class Monitoring:
    def __init__(self, db_manager: Union[FirestoreManager, SQLiteManager, None], log_file="monitoring_log.json"): # Fixed: Accept generic db_manager
        self.db_manager = db_manager # Use the passed db_manager
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
        # Fixed: Use db_manager.set_document
        if self.db_manager:
            # Firestore uses doc_id, SQLite might use auto-increment.
            # For heartbeat, set_document is appropriate as it's a single, updated document.
            asyncio.create_task(self.db_manager.set_document("monitoring", "heartbeat", heartbeat_data))
        else:
            self.logger.warning("DB Manager not initialized, cannot record heartbeat to DB.")
        logger.info("Heartbeat recorded.")

    def record_trade(self, trade_data: dict):
        """Records the details of a trade."""
        self._log_to_file(trade_data)
        # Fixed: Use db_manager.add_document
        if self.db_manager:
            asyncio.create_task(self.db_manager.add_document("trades", trade_data))
        else:
            self.logger.warning("DB Manager not initialized, cannot record trade to DB.")
        logger.info(f"Trade recorded: {trade_data}")

    def record_error(self, error_message: str):
        """Records an error."""
        error_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "error": error_message,
        }
        self._log_to_file(error_data)
        # Fixed: Use db_manager.add_document
        if self.db_manager:
            asyncio.create_task(self.db_manager.add_document("errors", error_data))
        else:
            self.logger.warning("DB Manager not initialized, cannot record error to DB.")
        logger.error(f"Error recorded: {error_message}")

    def record_performance_metrics(self, metrics: dict):
        """Records performance metrics."""
        performance_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "metrics": metrics,
        }
        self._log_to_file(performance_data)
        # Fixed: Use db_manager.add_document
        if self.db_manager:
            asyncio.create_task(self.db_manager.add_document("performance", performance_data))
        else:
            self.logger.warning("DB Manager not initialized, cannot record performance metrics to DB.")
        logger.info(f"Performance metrics recorded: {metrics}")

    def _log_to_file(self, data: dict):
        """Logs data to a local JSON file."""
        try:
            with open(self.log_file, "a") as f:
                f.write(json.dumps(data) + "\n")
        except Exception as e:
            logger.error(f"Failed to write to monitoring log file {self.log_file}: {e}", exc_info=True)
