"""
Comprehensive logging utility for the Ares trading bot.

This module provides a unified logging system that ensures all logs are stored
in the log/ directory with proper file rotation, component-specific logging,
and comprehensive error tracking.
"""

import logging
import logging.handlers
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


class ComprehensiveLogger:
    """
    Comprehensive logger that ensures all logs are stored in the log/ directory.
    """

    def __init__(self, config: dict[str, Any]):
        """
        Initialize comprehensive logger.

        Args:
            config: Configuration dictionary with logging settings
        """
        self.config = config
        self.log_config = config.get("logging", {})
        self.log_dir = Path(self.log_config.get("log_directory", "log"))
        self.log_dir.mkdir(exist_ok=True)

        # Initialize loggers
        self.system_logger = None
        self.error_logger = None
        self.trade_logger = None
        self.performance_logger = None
        self.global_logger = None  # Global logger for all logs

        self._setup_loggers()

    def _setup_loggers(self):
        """Setup all loggers with file handlers."""
        # Create timestamp for log files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Setup global logger (captures ALL logs)
        if self.log_config.get("enable_global_logging", True):
            self.global_logger = self._create_logger(
                "AresGlobal",
                self.log_dir / f"ares_global_{timestamp}.log",
                self.log_config.get("level", "INFO"),
            )
        else:
            self.global_logger = None

        # Setup system logger
        self.system_logger = self._create_logger(
            "AresSystem",
            self.log_dir / f"ares_system_{timestamp}.log",
            self.log_config.get("level", "INFO"),
        )

        # Setup error logger
        if self.log_config.get("enable_error_logging", True):
            self.error_logger = self._create_logger(
                "AresErrors",
                self.log_dir / f"ares_errors_{timestamp}.log",
                "ERROR",
            )

        # Setup trade logger
        if self.log_config.get("enable_trade_logging", True):
            self.trade_logger = self._create_logger(
                "AresTrades",
                self.log_dir / f"ares_trades_{timestamp}.log",
                "INFO",
            )

        # Setup performance logger
        if self.log_config.get("enable_performance_logging", True):
            self.performance_logger = self._create_logger(
                "AresPerformance",
                self.log_dir / f"ares_performance_{timestamp}.log",
                "INFO",
            )

    def _create_logger(self, name: str, log_file: Path, level: str) -> logging.Logger:
        """
        Create a logger with file and console handlers.

        Args:
            name: Logger name
            log_file: Log file path
            level: Log level

        Returns:
            logging.Logger: Configured logger
        """
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, level))

        # Clear existing handlers
        logger.handlers.clear()

        # Create formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

        # Add console handler if enabled
        if self.log_config.get("console_output", True):
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

        # Add file handler
        if self.log_config.get("file_output", True):
            # Create rotating file handler
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=self.log_config.get("max_file_size", 10 * 1024 * 1024),
                backupCount=self.log_config.get("backup_count", 5),
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        return logger

    def get_global_logger(self) -> logging.Logger:
        """Get the global logger that captures all logs."""
        return self.global_logger

    def get_system_logger(self) -> logging.Logger:
        """Get the system logger."""
        return self.system_logger

    def get_error_logger(self) -> logging.Logger | None:
        """Get the error logger."""
        return self.error_logger

    def get_trade_logger(self) -> logging.Logger | None:
        """Get the trade logger."""
        return self.trade_logger

    def get_performance_logger(self) -> logging.Logger | None:
        """Get the performance logger."""
        return self.performance_logger

    def get_component_logger(self, component_name: str) -> logging.Logger:
        """
        Get a component-specific logger.

        Args:
            component_name: Name of the component

        Returns:
            logging.Logger: Component logger
        """
        if self.global_logger:
            return self.global_logger.getChild(component_name)
        return logging.getLogger(component_name)

    def log_global(self, message: str, level: str = "INFO"):
        """
        Log to the global logger with specified level.

        Args:
            message: Log message
            level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        if self.global_logger:
            log_method = getattr(
                self.global_logger,
                level.lower(),
                self.global_logger.info,
            )
            log_method(message)

    def log_system_info(self, message: str):
        """Log system information."""
        if self.system_logger:
            self.system_logger.info(message)
        if self.global_logger:
            self.global_logger.info(f"[SYSTEM] {message}")

    def log_error(self, message: str, exc_info: bool = False):
        """Log error messages."""
        if self.error_logger:
            self.error_logger.error(message, exc_info=exc_info)
        if self.system_logger:
            self.system_logger.error(message, exc_info=exc_info)
        if self.global_logger:
            self.global_logger.error(f"[ERROR] {message}", exc_info=exc_info)

    def log_trade(self, message: str):
        """Log trade information."""
        if self.trade_logger:
            self.trade_logger.info(message)
        if self.system_logger:
            self.system_logger.info(f"[TRADE] {message}")
        if self.global_logger:
            self.global_logger.info(f"[TRADE] {message}")

    def log_performance(self, message: str):
        """Log performance information."""
        if self.performance_logger:
            self.performance_logger.info(message)
        if self.system_logger:
            self.system_logger.info(f"[PERFORMANCE] {message}")
        if self.global_logger:
            self.global_logger.info(f"[PERFORMANCE] {message}")

    def log_launcher_start(self, mode: str, symbol: str = None, exchange: str = None):
        """Log launcher startup information."""
        start_info = f"🚀 ARES LAUNCHER STARTED - Mode: {mode}"
        if symbol and exchange:
            start_info += f" - Symbol: {symbol} - Exchange: {exchange}"

        self.log_system_info("=" * 80)
        self.log_system_info(start_info)
        self.log_system_info(
            f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        )
        self.log_system_info(f"Log directory: {self.log_dir}")
        self.log_system_info("=" * 80)

    def log_launcher_end(self, exit_code: int = 0):
        """Log launcher shutdown information."""
        self.log_system_info("=" * 80)
        self.log_system_info(f"🛑 ARES LAUNCHER ENDED - Exit code: {exit_code}")
        self.log_system_info(
            f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        )
        self.log_system_info("=" * 80)

    def log_session_summary(self):
        """Log a session summary to the global logger."""
        if self.global_logger:
            self.global_logger.info("=" * 80)
            self.global_logger.info("📊 SESSION SUMMARY")
            self.global_logger.info(
                f"Session started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            )
            self.global_logger.info(
                f"Global log file: ares_global_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
            )
            self.global_logger.info("This file contains ALL logs for this session")
            self.global_logger.info("=" * 80)


# Global comprehensive logger instance
comprehensive_logger: ComprehensiveLogger | None = None


def setup_comprehensive_logging(config: dict[str, Any]) -> ComprehensiveLogger:
    """
    Setup comprehensive logging system.

    Args:
        config: Configuration dictionary

    Returns:
        ComprehensiveLogger: Configured logger instance
    """
    global comprehensive_logger
    comprehensive_logger = ComprehensiveLogger(config)

    # Log session summary to global logger
    comprehensive_logger.log_session_summary()

    return comprehensive_logger


def get_comprehensive_logger() -> ComprehensiveLogger | None:
    """Get the global comprehensive logger instance."""
    return comprehensive_logger


def get_component_logger(component_name: str) -> logging.Logger:
    """
    Get a component-specific logger.

    Args:
        component_name: Name of the component

    Returns:
        logging.Logger: Component logger
    """
    if comprehensive_logger:
        return comprehensive_logger.get_component_logger(component_name)
    return logging.getLogger(component_name)


def get_global_logger() -> logging.Logger | None:
    """
    Get the global logger that captures all logs.

    Returns:
        Optional[logging.Logger]: Global logger instance
    """
    if comprehensive_logger:
        return comprehensive_logger.get_global_logger()
    return None
