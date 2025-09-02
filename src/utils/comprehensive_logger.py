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
from typing import Any, Optional, Dict, Union

from .structured_logging import CorrelationIdFilter, get_json_formatter


class ComprehensiveLogger:
    """
    Comprehensive logger that ensures all logs are stored in the log/ directory.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize comprehensive logger.

        Args:
            config: Configuration dictionary with logging settings
        """
        self.config = config or {}
        self.log_config = config.get("logging", {}) if config else {}
        self.log_dir = Path(self.log_config.get("log_directory", "log"))
        self.log_dir.mkdir(exist_ok=True)

        # Initialize loggers
        self.system_logger: Optional[logging.Logger] = None
        self.error_logger: Optional[logging.Logger] = None
        self.trade_logger: Optional[logging.Logger] = None
        self.backtest_logger: Optional[logging.Logger] = None
        self.performance_logger: Optional[logging.Logger] = None
        self.global_logger: Optional[logging.Logger] = None  # Global logger for all logs

        self._setup_loggers()

        # Prepare a single, unified "full run" log that aggregates all output
        # across all loggers (including legacy ones) to a single file.
        self._setup_full_run_log()

    def _setup_loggers(self):
        """Setup all loggers with file handlers."""
        # Prevent logging from raising exceptions on broken pipes
        logging.raiseExceptions = False
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
            trade_path = self.log_dir / f"ares_trades_{timestamp}.log"
            self.trade_logger = self._create_logger(
                "AresTrades",
                trade_path,
                "INFO",
            )
            # Expose path for external usage
            self._trades_log_path = trade_path

        # Setup performance logger
        if self.log_config.get("enable_performance_logging", True):
            self.performance_logger = self._create_logger(
                "AresPerformance",
                self.log_dir / f"ares_performance_{timestamp}.log",
                "INFO",
            )

        # Setup backtest logger (dedicated per-run backtesting log)
        if self.log_config.get("enable_backtest_logging", True):
            backtest_path = self.log_dir / f"ares_backtest_{timestamp}.log"
            self.backtest_logger = self._create_logger(
                "AresBacktest",
                backtest_path,
                "INFO",
            )
            # Expose path for external usage
            self._backtest_log_path = backtest_path

        # Persist timestamp for unified log path computation
        self._timestamp = timestamp

    def _create_logger(self, name: str, log_file: Path, level: str) -> logging.Logger:
        """Create a logger with file and console handlers."""
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

    def _setup_full_run_log(self):
        """Setup unified log that captures all output."""
        try:
            # Route Python warnings through logging system so they get captured too
            logging.captureWarnings(True)

            # Resolve path and handler
            full_log_path = (
                self.log_dir / f"ares_full_{getattr(self, '_timestamp', datetime.now().strftime('%Y%m%d_%H%M%S'))}.log"
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
            # because it sets propagate = False by design.
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

    def get_global_logger(self) -> Optional[logging.Logger]:
        """Get the global logger instance."""
        return self.global_logger

    def get_system_logger(self) -> Optional[logging.Logger]:
        """Get the system logger instance."""
        return self.system_logger

    def get_error_logger(self) -> Optional[logging.Logger]:
        """Get the error logger instance."""
        return self.error_logger

    def get_trade_logger(self) -> Optional[logging.Logger]:
        """Get the trade logger instance."""
        return self.trade_logger

    def get_backtest_logger(self) -> Optional[logging.Logger]:
        """Get the backtest logger instance."""
        return self.backtest_logger

    def get_performance_logger(self) -> Optional[logging.Logger]:
        """Get the performance logger instance."""
        return self.performance_logger

    def get_component_logger(self, component_name: str) -> logging.Logger:
        """Get a component-specific logger."""
        if self.global_logger:
            return self.global_logger.getChild(component_name)
        return logging.getLogger(component_name)

    def get_full_log_path(self) -> Optional[str]:
        """Get the path to the unified full run log."""
        try:
            return (
                str(self._full_log_path)
                if getattr(self, "_full_log_path", None)
                else None
            )
        except Exception:
            return None

    def get_trades_log_path(self) -> Optional[str]:
        """Get the path to the trades log."""
        try:
            return (
                str(self._trades_log_path)
                if getattr(self, "_trades_log_path", None)
                else None
            )
        except Exception:
            return None

    def get_backtest_log_path(self) -> Optional[str]:
        """Get the path to the backtest log."""
        try:
            return (
                str(self._backtest_log_path)
                if getattr(self, "_backtest_log_path", None)
                else None
            )
        except Exception:
            return None

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

    def log_error(self, message: str, exc_info: bool = False):
        """Log error messages."""
        if self.error_logger:
            self.error_logger.error(message, exc_info=exc_info)
        if self.system_logger:
            self.system_logger.error(message, exc_info=exc_info)
        if self.global_logger:
            self.global_logger.error(message, exc_info=exc_info)

    def log_trade(self, message: str):
        """Log trade information."""
        if self.trade_logger:
            self.trade_logger.info(message)
        if self.system_logger:
            self.system_logger.info(f"[TRADE] {message}")
        if self.global_logger:
            self.global_logger.info(f"[TRADE] {message}")

    def log_backtest(self, message: str):
        """Log backtesting information to a dedicated backtest log as well as system/global logs."""
        if self.backtest_logger:
            self.backtest_logger.info(message)
        if self.system_logger:
            self.system_logger.info(f"[BACKTEST] {message}")
        if self.global_logger:
            self.global_logger.info(f"[BACKTEST] {message}")

    def log_performance(self, message: str):
        """Log performance information."""
        if self.performance_logger:
            self.performance_logger.info(message)
        if self.system_logger:
            self.system_logger.info(f"[PERFORMANCE] {message}")
        if self.global_logger:
            self.global_logger.info(f"[PERFORMANCE] {message}")

    def log_session_summary(self) -> None:
        """Log session summary information."""
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

    def log_launcher_start(self, mode: str, symbol: Optional[str] = None, exchange: Optional[str] = None):
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

    def log_launcher_end(self, exit_code: int):
        """Log launcher shutdown information."""
        try:
            self.log_system_info("=" * 80)
            self.log_system_info(f"🛑 ARES LAUNCHER ENDED - Exit code: {exit_code}")
            self.log_system_info(
                f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            )
            self.log_system_info("=" * 80)
        except (BrokenPipeError, OSError) as e:
            # Safely ignore broken pipe during shutdown
            if not (
                isinstance(e, OSError) and getattr(e, "errno", None) == errno.EPIPE
            ):
                raise


class _SafeStreamHandler(logging.StreamHandler):
    """Safe stream handler that handles broken pipes gracefully."""

    def emit(self, record: logging.LogRecord) -> None:
        """Emit a log record safely."""
        try:
            super().emit(record)
        except (BrokenPipeError, OSError) as e:
            # Safely ignore broken pipe during shutdown/piped output
            if not (
                isinstance(e, OSError) and getattr(e, "errno", None) == errno.EPIPE
            ):
                raise

    def flush(self) -> None:
        """Flush the stream safely."""
        try:
            super().flush()
        except (BrokenPipeError, OSError) as e:
            # Safely ignore broken pipe during shutdown/piped output
            if not (
                isinstance(e, OSError) and getattr(e, "errno", None) == errno.EPIPE
            ):
                raise


# Global comprehensive logger instance
comprehensive_logger: Optional[ComprehensiveLogger] = None


def setup_comprehensive_logging(config: Optional[Dict[str, Any]] = None) -> ComprehensiveLogger:
    """Setup comprehensive logging system."""
    global comprehensive_logger
    comprehensive_logger = ComprehensiveLogger(config)

    # Log session summary to global logger
    comprehensive_logger.log_session_summary()

    return comprehensive_logger


def get_comprehensive_logger() -> Optional[ComprehensiveLogger]:
    """Get the global comprehensive logger instance."""
    return comprehensive_logger


def get_component_logger(component_name: str) -> logging.Logger:
    """Get a component-specific logger."""
    if comprehensive_logger:
        return comprehensive_logger.get_component_logger(component_name)
    return logging.getLogger(component_name)


def get_global_logger() -> Optional[logging.Logger]:
    """Get the global logger instance."""
    if comprehensive_logger:
        return comprehensive_logger.get_global_logger()
    return None
