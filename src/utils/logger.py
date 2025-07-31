# src/utils/logger.py
import logging
import logging.handlers
import os
import sys
from datetime import datetime
from pythonjsonlogger import jsonlogger

# Assume CONFIG is imported from your project's config file.
# This might need to be adjusted based on your exact import structure.
# For example: from src.config import CONFIG
try:
    from src.config import CONFIG
except ImportError:
    # Provide a default config if the main one isn't available,
    # which can happen when running scripts from different locations.
    CONFIG = {"logging": {}}


def setup_logging():
    """
    Configures the root logger for the entire application.
    This should be called once at the very beginning of the application's entry point.

    Features:
    - Structured (JSON) formatting for all logs.
    - Configurable log level from the main config file.
    - Logs to both the console (for real-time feedback) and a rotating file.
    - Log rotation to prevent log files from becoming too large.
    - Centralized configuration.
    """
    # Get logging configuration or use sensible defaults
    log_config = CONFIG.get("logging", {})
    log_level = log_config.get("level", "INFO").upper()
    log_to_file = log_config.get("log_to_file", True)
    log_directory = log_config.get("directory", "logs")
    log_filename_base = log_config.get("filename", "ares.log")

    # Get the root logger. All other loggers in the application
    # will inherit this configuration.
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # IMPORTANT: Clear any existing handlers to prevent duplicate log entries
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    # Define a readable format for console output
    console_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Define a standard JSON format for file logs.
    # This includes standard log fields plus module and line number for easy debugging.
    json_formatter = jsonlogger.JsonFormatter(
        "%(asctime)s %(name)s %(levelname)s %(module)s %(funcName)s %(lineno)d %(message)s"
    )

    # --- Console Handler ---
    # This handler prints logs to the standard output (your terminal).
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # --- Rotating File Handler ---
    # This handler writes logs to a file and automatically rotates the logs
    # when they reach a certain size.
    if log_to_file:
        try:
            os.makedirs(log_directory, exist_ok=True)

            # Add timestamp to the log filename for unique logs per run
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_filename_name, log_filename_ext = os.path.splitext(log_filename_base)
            timestamped_log_filename = f"{log_filename_name}_{timestamp}{log_filename_ext}"
            log_path = os.path.join(log_directory, timestamped_log_filename)

            # Rotate logs when they reach 5MB. Keep the last 5 log files as backups.
            # With timestamped filenames, a new log is created for each run.
            file_handler = logging.handlers.RotatingFileHandler(
                log_path, maxBytes=5 * 1024 * 1024, backupCount=5
            )
            file_handler.setFormatter(json_formatter)
            root_logger.addHandler(file_handler)
        except Exception as e:
            root_logger.error(f"Failed to configure file logging: {e}", exc_info=True)

    root_logger.info(
        "Logging has been successfully configured.", extra={"config": log_config}
    )


# --- How to use the logger throughout your application ---

# 1. At the very start of your application (e.g., in main_launcher.py), call the setup function once:
#
#    from src.utils.logger import setup_logging
#    setup_logging()

# 2. In any other module (e.g., strategist.py, analyst.py), get a logger instance for that specific module:
#
#    import logging
#    logger = logging.getLogger(__name__)
#
#    logger.info("Strategist is making a decision.", extra={'current_bias': 'LONG'})
#    logger.warning("Market volatility is unusually high.")
#
#    try:
#        # some operation that might fail
#        result = 1 / 0
#    except Exception:
#        # logger.exception automatically includes the full traceback in the log
#        logger.exception("A critical error occurred during calculation.")

# Create a system logger instance that can be imported by other modules
system_logger = logging.getLogger("ares.system")

__all__ = ["system_logger", "setup_logging"]
