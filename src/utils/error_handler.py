# src/utils/error_handler.py
import sys
import logging
import json
import traceback
from datetime import datetime
import os
import asyncio
from typing import List, Dict, Any

from src.utils.logger import system_logger # Use the existing system logger
from src.database.firestore_manager import db_manager # Import the global Firestore manager
from src.config import CONFIG # Import CONFIG for error log file path

class AresExceptionHandler:
    """
    A centralized exception handler for the Ares trading bot.
    It logs unhandled exceptions to the system logger, a dedicated JSON file,
    and optionally to Firestore.
    """
    def __init__(self):
        self.logger = system_logger.getChild('ExceptionHandler')
        self.error_log_file = os.path.join("logs", CONFIG.get("ERROR_LOG_FILE", "ares_errors.jsonl")) # Using .jsonl for JSON Lines format
        os.makedirs(os.path.dirname(self.error_log_file), exist_ok=True)
        self._original_excepthook = sys.excepthook

    def _log_exception(self, exc_type, exc_value, exc_traceback):
        """
        Internal method to log the exception details.
        """
        timestamp = datetime.utcnow().isoformat()
        traceback_str = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
        
        error_record = {
            "timestamp": timestamp,
            "level": "CRITICAL",
            "type": exc_type.__name__,
            "message": str(exc_value),
            "traceback": traceback_str,
            "context": "Unhandled Exception" # Default context, can be expanded
        }

        # Log to system logger (which goes to console and main log file)
        self.logger.critical(f"Unhandled exception caught: {exc_type.__name__}: {exc_value}", exc_info=True)

        # Log to dedicated JSON Lines file
        try:
            with open(self.error_log_file, 'a') as f:
                f.write(json.dumps(error_record) + "\n")
            self.logger.info(f"Unhandled exception logged to {self.error_log_file}")
        except Exception as e:
            self.logger.error(f"Failed to write exception to dedicated log file: {e}", exc_info=True)

        # Optionally log to Firestore
        if db_manager and db_manager.firestore_enabled:
            try:
                # Firestore has document size limits (1MB), so truncate traceback if very long
                firestore_record = error_record.copy()
                if len(firestore_record["traceback"]) > 50000: # Arbitrary limit, adjust as needed
                    firestore_record["traceback"] = firestore_record["traceback"][:50000] + "\n... (truncated)"
                
                # Run Firestore operation in a non-blocking way (e.g., in a separate task)
                asyncio.create_task(db_manager.add_document("ares_unhandled_exceptions", firestore_record, is_public=False))
                self.logger.info("Unhandled exception sent to Firestore.")
            except Exception as e:
                self.logger.error(f"Failed to send exception to Firestore: {e}", exc_info=True)

        # Call the original excepthook to maintain default Python behavior (e.g., for IDEs)
        self._original_excepthook(exc_type, exc_value, exc_traceback)

    def register(self):
        """Registers this handler as the default sys.excepthook."""
        sys.excepthook = self._log_exception
        self.logger.info("Ares custom exception handler registered.")

    def unregister(self):
        """Unregisters the custom handler and restores the original."""
        sys.excepthook = self._original_excepthook
        self.logger.info("Ares custom exception handler unregistered.")

# Global instance of the exception handler
ares_exception_handler = AresExceptionHandler()

def register_global_exception_handler():
    """Convenience function to register the global exception handler."""
    ares_exception_handler.register()

def unregister_global_exception_handler():
    """Convenience function to unregister the global exception handler."""
    ares_exception_handler.unregister()

# Function to get a list of logged exceptions (e.g., for a report)
def get_logged_exceptions(limit: int = 100) -> List[Dict[str, Any]]:
    """
    Reads recent exceptions from the dedicated error log file.
    """
    exceptions = []
    try:
        with open(ares_exception_handler.error_log_file, 'r') as f:
            # Read lines in reverse to get most recent first, then reverse again
            for line in reversed(f.readlines()):
                try:
                    exceptions.append(json.loads(line))
                    if len(exceptions) >= limit:
                        break
                except json.JSONDecodeError:
                    ares_exception_handler.logger.warning(f"Skipping malformed line in error log: {line.strip()}")
        return list(reversed(exceptions)) # Return in chronological order
    except FileNotFoundError:
        ares_exception_handler.logger.info("Error log file not found.")
        return []
    except Exception as e:
        ares_exception_handler.logger.error(f"Error reading error log file: {e}", exc_info=True)
        return []

