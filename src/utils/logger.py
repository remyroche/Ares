# src/utils/logger.py
import logging
import logging.handlers
import os
# Import the main CONFIG dictionary
from config import CONFIG 

def setup_logger(name='ares_system', config=None): # Changed default config to None
    """
    Sets up a centralized logger for the Ares system.
    Logs to console and a rotating file.
    """
    logger = logging.getLogger(name)
    
    # Prevent adding multiple handlers if logger is already configured
    if logger.handlers:
        return logger

    # Use the logging configuration from the main CONFIG dictionary
    if config is None:
        config = CONFIG['logging']

    log_level_str = config.get("log_level", "INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    logger.setLevel(log_level)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Console Handler
    if config.get("log_to_console", True):
        ch = logging.StreamHandler()
        ch.setLevel(log_level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    # File Handler (Rotating)
    log_file = config.get("log_file", "logs/ares_system.log")
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    fh = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=config.get("max_bytes", 10485760), # 10 MB default
        backupCount=config.get("backup_count", 5) # Keep 5 backup files
    )
    fh.setLevel(log_level)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger

# Initialize the main system logger
# Pass CONFIG['logging'] explicitly to ensure correct configuration is used
system_logger = setup_logger(config=CONFIG['logging'])
