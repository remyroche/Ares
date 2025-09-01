"""
Comprehensive logging utility for the Ares trading bot.

This module provides a unified logging system that ensures all logs are stored
in the log/ directory with proper file rotation, component-specific logging,
and comprehensive error tracking.
"""

import logging
import logging.handlers
import sys
import errno
from datetime import datetime
from pathlib import Path
from typing import Any

from .structured_logging import CorrelationIdFilter, get_json_formatter


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
        self.backtest_logger = None
        self.performance_logger = None
        self.global_logger = None  # Global logger for all logs

        self._setup_loggers()

        # Prepare a single, unified "full run" log that aggregates all output
        # across all loggers (including legacy ones) to a single file.
        self._setup_full_run_log()

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

        # Create structured JSON formatter by default
        formatter = get_json_formatter()

        # Add console handler if enabled
        if self.log_config.get("console_output", True):
            console_handler = _SafeStreamHandler(sys.stdout)
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

        # Add correlation filter to enrich records
        correlation_filter = CorrelationIdFilter()
        logger.addFilter(correlation_filter)
        for handler in logger.handlers:
            handler.addFilter(correlation_filter)

        return logger

    def _setup_full_run_log(self) -> None:
        """Attach a unified file handler that captures all log records.

        - Creates `ares_full_<timestamp>.log` under the configured log dir
        - Attaches the handler to the root logger to capture most records
        - Also attaches to the legacy `AresTradingSystem` logger to ensure
          records with `propagate=False` are captured as well
        """
        try:
            # Route Python warnings through logging system so they get captured too
            logging.captureWarnings(True)

            # Resolve path and handler
            full_log_path = (
                self.log_dir
                / f"ares_full_{getattr(self, '_timestamp', datetime.now().strftime('%Y%m%d_%H%M%S'))}.log"
            )
            full_handler = logging.handlers.RotatingFileHandler(
                full_log_path,
                maxBytes=self.log_config.get("max_file_size", 10 * 1024 * 1024),
                backupCount=self.log_config.get("backup_count", 5),
            )

            formatter = get_json_formatter()
            full_handler.setFormatter(formatter)

            # Enrich with correlation IDs
            correlation_filter = CorrelationIdFilter()
            full_handler.addFilter(correlation_filter)

            # Attach to root logger to aggregate everything that propagates
            root_logger = logging.getLogger()
            # Avoid duplicate handler attachment
            if not any(
                isinstance(h, logging.handlers.RotatingFileHandler)
                and getattr(h, "baseFilename", None) == str(full_log_path)
                for h in root_logger.handlers
            ):
                root_logger.addHandler(full_handler)
                # Ensure root level is permissive so we don't miss records
                if root_logger.level > logging.DEBUG:
                    root_logger.setLevel(logging.DEBUG)

            # Also attach directly to the legacy enhanced logger if present,
            # because it sets propagate=False by design.
            legacy_logger = logging.getLogger("AresTradingSystem")
            if not any(
                isinstance(h, logging.handlers.RotatingFileHandler)
                and getattr(h, "baseFilename", None) == str(full_log_path)
                for h in legacy_logger.handlers
            ):
                legacy_logger.addHandler(full_handler)

            # Stash path for external access (e.g., launcher banner)
            self._full_log_path = full_log_path
        except Exception:
            # Never fail logging setup due to aggregation handler issues
            self._full_log_path = None

    def get_backtest_logger(self) -> logging.Logger | None:
        """Get the backtest logger."""
        return self.backtest_logger

    def get_backtest_log_path(self) -> str | None:
        """Return the absolute path to the backtest log file, if set."""
        try:
            return (
                str(self._backtest_log_path)
                if getattr(self, "_backtest_log_path", None)
                else None
            )
        except Exception:
            return None

    def log_system_info(self, message: str):
        """Log system information."""
        try:
            if self.system_logger:
                self.system_logger.info(message)
            if self.global_logger:
                self.global_logger.info(f"[SYSTEM] {message}")
        except (BrokenPipeError, OSError) as e:
            # Safely ignore broken pipe during shutdown/piped output
            if not (
                isinstance(e, OSError) and getattr(e, "errno", None) == errno.EPIPE
            ):
                raise

    def log_backtest(self, message: str):
        """Log backtesting information to a dedicated backtest log as well as system/global logs."""
        if self.backtest_logger:
            self.backtest_logger.info(message)
        if self.system_logger:
            self.system_logger.info(f"[BACKTEST] {message}")
        if self.global_logger:
            self.global_logger.info(f"[BACKTEST] {message}")

    def log_session_summary(self) -> None:
        """Log a session summary to the global logger."""
        if not self.global_logger:
            return
        try:
            self.global_logger.info("=" * 80)
            self.global_logger.info("📊 SESSION SUMMARY")
            self.global_logger.info(
                f"Session started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            )
            self.global_logger.info("This file contains ALL logs for this session")
            self.global_logger.info("=" * 80)
        except Exception:
            pass

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


class _SafeStreamHandler(logging.StreamHandler):
    """StreamHandler that suppresses BrokenPipeError during emit/flush."""

    def emit(self, record):
        try:
            super().emit(record)
        except (BrokenPipeError, OSError) as e:
            if not (
                isinstance(e, OSError) and getattr(e, "errno", None) == errno.EPIPE
            ):
                raise

    def flush(self):
        try:
            super().flush()
        except (BrokenPipeError, OSError) as e:
            if not (
                isinstance(e, OSError) and getattr(e, "errno", None) == errno.EPIPE
            ):
                raise

    # moved to ComprehensiveLogger


# Global comprehensive logger instance
comprehensive_logger: ComprehensiveLogger | None = None




