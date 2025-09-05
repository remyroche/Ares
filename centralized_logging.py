"""
Centralized Logging System for Ares Project

This module provides a centralized logging configuration that ensures:
1. All log outputs are in the log/ directory
2. Each run has a centralized log file with datetime in filename
3. Consistent logging format across all modules
4. Proper log rotation and management
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
import threading


class CentralizedLogger:
    """
    Centralized logging system that manages all logging across the application.
    
    Features:
    - Single log file per run with datetime in filename
    - All logs go to log/ directory
    - Consistent formatting
    - Thread-safe logging
    - Automatic log rotation
    """
    
    _instance = None
    _lock = threading.Lock()
    _run_id = None
    _log_dir = None
    _main_logger = None
    _configured = False
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(CentralizedLogger, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._configured:
            self._setup_logging()
    
    def _setup_logging(self):
        """Initialize the centralized logging system."""
        # Create run ID with timestamp
        if self._run_id is None:
            self._run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Ensure log directory exists
        self._log_dir = Path("log")
        self._log_dir.mkdir(exist_ok=True)
        
        # Create main log file path
        main_log_file = self._log_dir / f"ares_run_{self._run_id}.log"
        
        # Configure root logger
        self._main_logger = logging.getLogger()
        self._main_logger.setLevel(logging.INFO)
        
        # Clear any existing handlers
        self._main_logger.handlers.clear()
        
        # Create formatter
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        
        # File handler with rotation
        file_handler = logging.handlers.RotatingFileHandler(
            main_log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        
        # Add handlers to root logger
        self._main_logger.addHandler(file_handler)
        self._main_logger.addHandler(console_handler)
        
        # Log initialization
        self._main_logger.info(f"Centralized logging initialized - Run ID: {self._run_id}")
        self._main_logger.info(f"Main log file: {main_log_file}")
        
        self._configured = True
    
    def get_logger(self, name: str) -> logging.Logger:
        """
        Get a logger instance for a specific module.
        
        Args:
            name: Logger name (typically __name__ of the calling module)
            
        Returns:
            Configured logger instance
        """
        if not self._configured:
            self._setup_logging()
        
        logger = logging.getLogger(name)
        return logger
    
    def get_run_id(self) -> str:
        """Get the current run ID."""
        return self._run_id
    
    def get_log_file_path(self) -> Path:
        """Get the path to the main log file."""
        return self._log_dir / f"ares_run_{self._run_id}.log"
    
    def set_level(self, level: int):
        """
        Set the logging level for all handlers.
        
        Args:
            level: Logging level (e.g., logging.DEBUG, logging.INFO)
        """
        if not self._configured:
            self._setup_logging()
        
        self._main_logger.setLevel(level)
        for handler in self._main_logger.handlers:
            handler.setLevel(level)
    
    def add_module_logger(self, module_name: str, log_file_suffix: Optional[str] = None) -> logging.Logger:
        """
        Create a module-specific logger with its own log file.
        
        Args:
            module_name: Name of the module
            log_file_suffix: Optional suffix for the log file name
            
        Returns:
            Logger instance with dedicated log file
        """
        if not self._configured:
            self._setup_logging()
        
        # Create module-specific logger
        logger = logging.getLogger(module_name)
        
        # Create module-specific log file
        if log_file_suffix:
            log_filename = f"ares_{module_name}_{log_file_suffix}_{self._run_id}.log"
        else:
            log_filename = f"ares_{module_name}_{self._run_id}.log"
        
        log_file_path = self._log_dir / log_filename
        
        # Create file handler for this module
        file_handler = logging.handlers.RotatingFileHandler(
            log_file_path,
            maxBytes=5*1024*1024,  # 5MB
            backupCount=3,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.INFO)
        
        # Use same formatter as main logger
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(formatter)
        
        # Add handler to module logger
        logger.addHandler(file_handler)
        logger.setLevel(logging.INFO)
        
        # Log the creation
        logger.info(f"Module logger created - Log file: {log_file_path}")
        
        return logger


# Global instance
_centralized_logger = CentralizedLogger()


def get_logger(name: str) -> logging.Logger:
    """
    Convenience function to get a logger instance.
    
    Args:
        name: Logger name (typically __name__ of the calling module)
        
    Returns:
        Configured logger instance
    """


def get_run_id() -> str:
    """Get the current run ID."""


def get_log_file_path() -> Path:
    """Get the path to the main log file."""


def set_log_level(level: int):
    """
    Set the logging level for all handlers.
    
    Args:
        level: Logging level (e.g., logging.DEBUG, logging.INFO)
    """
    _centralized_logger.set_level(level)


def add_module_logger(module_name: str, log_file_suffix: Optional[str] = None) -> logging.Logger:
    """
    Create a module-specific logger with its own log file.
    
    Args:
        module_name: Name of the module
        log_file_suffix: Optional suffix for the log file name
        
    Returns:
        Logger instance with dedicated log file
    """
    return _centralized_logger.add_module_logger(module_name, log_file_suffix)


# Initialize logging on import
if not _centralized_logger._configured:
    _centralized_logger._setup_logging()