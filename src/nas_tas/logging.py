"""
Unified Logging for NAS/TAS Systems

This module provides standardized logging capabilities across both NAS and TAS
implementations, consolidating logging logic and ensuring consistent output.
"""

import logging
import sys
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import json
import threading
from contextlib import contextmanager

from src.utils.tprint import (
    tprint, tprint_debug, tprint_info, tprint_warning, tprint_error,
    tprint_success, tprint_progress, tprint_performance, tprint_timer
)

@dataclass
class LoggingConfig:
    """Configuration for unified logging."""

    # Basic settings
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"

    # Output settings
    console_logging: bool = True
    file_logging: bool = True
    log_directory: str = "logs"

    # File settings
    log_filename: str = "nas_tas.log"
    max_file_size_mb: int = 100
    backup_count: int = 5

    # Component-specific logging
    enable_component_logging: bool = True
    log_components: List[str] = field(default_factory=lambda: [
        "training", "evaluation", "data", "search", "results"
    ])

    # Performance logging
    enable_performance_logging: bool = True
    log_execution_times: bool = True
    log_memory_usage: bool = True

    # Structured logging
    enable_structured_logging: bool = True
    structured_log_file: str = "nas_tas_structured.json"

    # Custom fields
    custom_fields: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'log_level': self.log_level,
            'log_format': self.log_format,
            'date_format': self.date_format,
            'console_logging': self.console_logging,
            'file_logging': self.file_logging,
            'log_directory': self.log_directory,
            'log_filename': self.log_filename,
            'max_file_size_mb': self.max_file_size_mb,
            'backup_count': self.backup_count,
            'enable_component_logging': self.enable_component_logging,
            'log_components': self.log_components,
            'enable_performance_logging': self.enable_performance_logging,
            'log_execution_times': self.log_execution_times,
            'log_memory_usage': self.log_memory_usage,
            'enable_structured_logging': self.enable_structured_logging,
            'structured_log_file': self.structured_log_file,
            'custom_fields': self.custom_fields
        }

class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured JSON logging."""

    def format(self, record):
        """Format log record as JSON."""
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }

        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)

        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                          'filename', 'module', 'exc_info', 'exc_text', 'stack_info',
                          'lineno', 'funcName', 'created', 'msecs', 'relativeCreated',
                          'thread', 'threadName', 'processName', 'process', 'getMessage']:
                log_entry[key] = value

        return json.dumps(log_entry)

class PerformanceFilter(logging.Filter):
    """Filter for performance-related log messages."""

    def filter(self, record):
        """Filter performance log messages."""
        return hasattr(record, 'performance') and record.performance

class UnifiedLogger:
    """
    Unified logger for NAS/TAS systems.

    This class consolidates logging logic that was previously scattered
    across NAS and TAS implementations, providing consistent logging
    capabilities with structured output and performance tracking.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, config: Optional[LoggingConfig] = None):
        """Singleton pattern implementation."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(UnifiedLogger, cls).__new__(cls)
        return cls._instance

    def __init__(self, config: Optional[LoggingConfig] = None):
        """
        Initialize unified logger.

        Args:
            config: Logging configuration
        """
        if hasattr(self, '_initialized'):
            return

        self.config = config or LoggingConfig()
        self._initialized = True

        # Setup logging
        self._setup_logging()

        # Performance tracking
        self.performance_data = []
        self.execution_times = {}

        # Component loggers
        self.component_loggers = {}

        tprint_info("Unified logger initialized")

    def _setup_logging(self):
        """Setup logging configuration."""
        # Create log directory
        log_dir = Path(self.config.log_directory)
        log_dir.mkdir(parents=True, exist_ok=True)

        # Get root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, self.config.log_level.upper()))

        # Clear existing handlers
        root_logger.handlers.clear()

        # Console handler
        if self.config.console_logging:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(getattr(logging, self.config.log_level.upper()))
            console_formatter = logging.Formatter(
                self.config.log_format,
                datefmt=self.config.date_format
            )
            console_handler.setFormatter(console_formatter)
            root_logger.addHandler(console_handler)

        # File handler
        if self.config.file_logging:
            from logging.handlers import RotatingFileHandler

            log_file = log_dir / self.config.log_filename
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=self.config.max_file_size_mb * 1024 * 1024,
                backupCount=self.config.backup_count
            )
            file_handler.setLevel(getattr(logging, self.config.log_level.upper()))
            file_formatter = logging.Formatter(
                self.config.log_format,
                datefmt=self.config.date_format
            )
            file_handler.setFormatter(file_formatter)
            root_logger.addHandler(file_handler)

        # Structured logging handler
        if self.config.enable_structured_logging:
            structured_file = log_dir / self.config.structured_log_file
            structured_handler = RotatingFileHandler(
                structured_file,
                maxBytes=self.config.max_file_size_mb * 1024 * 1024,
                backupCount=self.config.backup_count
            )
            structured_handler.setLevel(getattr(logging, self.config.log_level.upper()))
            structured_formatter = StructuredFormatter()
            structured_handler.setFormatter(structured_formatter)

            # Add performance filter
            if self.config.enable_performance_logging:
                performance_filter = PerformanceFilter()
                structured_handler.addFilter(performance_filter)

            root_logger.addHandler(structured_handler)

    def get_component_logger(self, component: str) -> logging.Logger:
        """Get logger for specific component."""
        if component not in self.component_loggers:
            logger_name = f"nas_tas.{component}"
            logger = logging.getLogger(logger_name)
            self.component_loggers[component] = logger

        return self.component_loggers[component]

    def log_training_progress(
        self,
        stage: str,
        metrics: Dict[str, Any],
        component: str = "training"
    ):
        """Log training progress with metrics."""
        logger = self.get_component_logger(component)

        log_message = f"Training stage: {stage}"
        logger.info(log_message, extra={
            'component': component,
            'stage': stage,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        })

        # Also use tprint for console output
        tprint_progress(f"{component.upper()} - {stage}: {metrics}")

    def log_evaluation_results(
        self,
        results: Dict[str, Any],
        component: str = "evaluation"
    ):
        """Log evaluation results."""
        logger = self.get_component_logger(component)

        log_message = "Evaluation results"
        logger.info(log_message, extra={
            'component': component,
            'results': results,
            'timestamp': datetime.now().isoformat()
        })

        # Extract key metrics for console output
        key_metrics = {}
        for key in ['accuracy', 'f1_score', 'sharpe_ratio', 'max_drawdown']:
            if key in results:
                key_metrics[key] = results[key]

        if key_metrics:
            tprint_success(f"{component.upper()} Results: {key_metrics}")

    def log_performance_metrics(
        self,
        metrics: Dict[str, float],
        component: str = "performance"
    ):
        """Log performance metrics."""
        logger = self.get_component_logger(component)

        log_message = "Performance metrics"
        logger.info(log_message, extra={
            'component': component,
            'performance': True,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        })

        # Store performance data
        if self.config.enable_performance_logging:
            self.performance_data.append({
                'timestamp': datetime.now().isoformat(),
                'component': component,
                'metrics': metrics
            })

        # Console output
        tprint_performance(f"{component.upper()} Metrics: {metrics}")

    def log_search_progress(
        self,
        iteration: int,
        total_iterations: int,
        best_score: float,
        component: str = "search"
    ):
        """Log search progress."""
        logger = self.get_component_logger(component)

        progress_percent = (iteration / total_iterations) * 100
        log_message = f"Search progress: {iteration}/{total_iterations} ({progress_percent:.1f}%)"

        logger.info(log_message, extra={
            'component': component,
            'iteration': iteration,
            'total_iterations': total_iterations,
            'progress_percent': progress_percent,
            'best_score': best_score,
            'timestamp': datetime.now().isoformat()
        })

        # Console output
        tprint_progress(f"{component.upper()} Progress: {iteration}/{total_iterations} "
                       f"({progress_percent:.1f}%) - Best: {best_score:.4f}")

    def log_data_quality(
        self,
        quality_metrics: Dict[str, Any],
        component: str = "data"
    ):
        """Log data quality metrics."""
        logger = self.get_component_logger(component)

        log_message = "Data quality assessment"
        logger.info(log_message, extra={
            'component': component,
            'quality_metrics': quality_metrics,
            'timestamp': datetime.now().isoformat()
        })

        # Console output
        overall_score = quality_metrics.get('overall_quality_score', 0)
        tprint_info(f"{component.upper()} Quality Score: {overall_score:.3f}")

    def log_error(
        self,
        error: Exception,
        context: Dict[str, Any],
        component: str = "system"
    ):
        """Log error with context."""
        logger = self.get_component_logger(component)

        log_message = f"Error in {component}: {str(error)}"
        logger.error(log_message, extra={
            'component': component,
            'error_type': type(error).__name__,
            'error_message': str(error),
            'context': context,
            'timestamp': datetime.now().isoformat()
        }, exc_info=True)

        # Console output
        tprint_error(f"{component.upper()} Error: {str(error)}")

    def log_system_info(
        self,
        info: Dict[str, Any],
        component: str = "system"
    ):
        """Log system information."""
        logger = self.get_component_logger(component)

        log_message = "System information"
        logger.info(log_message, extra={
            'component': component,
            'system_info': info,
            'timestamp': datetime.now().isoformat()
        })

        # Console output
        tprint_info(f"{component.upper()} Info: {info}")

    @contextmanager
    def log_execution_time(self, operation: str, component: str = "system"):
        """Context manager for logging execution time."""
        start_time = datetime.now()

        logger = self.get_component_logger(component)
        logger.info(f"Starting {operation}", extra={
            'component': component,
            'operation': operation,
            'start_time': start_time.isoformat()
        })

        tprint_timer(f"Starting {component.upper()} - {operation}")

        try:
            yield
        finally:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            logger.info(f"Completed {operation} in {duration:.2f}s", extra={
                'component': component,
                'operation': operation,
                'duration_seconds': duration,
                'end_time': end_time.isoformat()
            })

            # Store execution time
            if self.config.log_execution_times:
                self.execution_times[f"{component}.{operation}"] = duration

            tprint_timer(f"Completed {component.upper()} - {operation} in {duration:.2f}s")

    def log_configuration(
        self,
        config: Dict[str, Any],
        component: str = "config"
    ):
        """Log configuration information."""
        logger = self.get_component_logger(component)

        log_message = "Configuration loaded"
        logger.info(log_message, extra={
            'component': component,
            'configuration': config,
            'timestamp': datetime.now().isoformat()
        })

        # Console output
        tprint_info(f"{component.upper()} Configuration loaded")

    def log_model_info(
        self,
        model_info: Dict[str, Any],
        component: str = "model"
    ):
        """Log model information."""
        logger = self.get_component_logger(component)

        log_message = "Model information"
        logger.info(log_message, extra={
            'component': component,
            'model_info': model_info,
            'timestamp': datetime.now().isoformat()
        })

        # Console output
        model_type = model_info.get('model_type', 'Unknown')
        model_size = model_info.get('model_size_mb', 0)
        tprint_info(f"{component.upper()} Model: {model_type} ({model_size:.1f}MB)")

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance logging summary."""
        return {
            'performance_data_count': len(self.performance_data),
            'execution_times': dict(self.execution_times),
            'component_loggers': list(self.component_loggers.keys()),
            'config': self.config.to_dict()
        }

    def export_performance_data(self, filepath: Union[str, Path]) -> bool:
        """Export performance data to file."""
        try:
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)

            performance_summary = {
                'export_timestamp': datetime.now().isoformat(),
                'performance_data': self.performance_data,
                'execution_times': self.execution_times,
                'summary': self.get_performance_summary()
            }

            with open(filepath, 'w') as f:
                json.dump(performance_summary, f, indent=2)

            tprint_success(f"Performance data exported to {filepath}")
            return True

        except Exception as e:
            tprint_error(f"Failed to export performance data: {e}")
            return False

    def clear_performance_data(self):
        """Clear performance data."""
        self.performance_data.clear()
        self.execution_times.clear()
        tprint_info("Performance data cleared")

    def set_log_level(self, level: str):
        """Set logging level."""
        self.config.log_level = level.upper()

        # Update all loggers
        for logger in self.component_loggers.values():
            logger.setLevel(getattr(logging, level.upper()))

        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, level.upper()))

        tprint_info(f"Log level set to {level.upper()}")

    def add_custom_field(self, key: str, value: Any):
        """Add custom field to all log messages."""
        self.config.custom_fields[key] = value

    def remove_custom_field(self, key: str):
        """Remove custom field."""
        if key in self.config.custom_fields:
            del self.config.custom_fields[key]

# Global logger instance
_global_logger = None

def get_unified_logger(config: Optional[LoggingConfig] = None) -> UnifiedLogger:
    """Get global unified logger instance."""
    global _global_logger
    if _global_logger is None:
        _global_logger = UnifiedLogger(config)
    return _global_logger

def setup_logging(config: Optional[LoggingConfig] = None) -> UnifiedLogger:
    """Setup unified logging system."""
    return get_unified_logger(config)

# Convenience functions for common logging operations
def log_training_progress(stage: str, metrics: Dict[str, Any], component: str = "training"):
    """Log training progress."""
    logger = get_unified_logger()
    logger.log_training_progress(stage, metrics, component)

def log_evaluation_results(results: Dict[str, Any], component: str = "evaluation"):
    """Log evaluation results."""
    logger = get_unified_logger()
    logger.log_evaluation_results(results, component)

def log_performance_metrics(metrics: Dict[str, float], component: str = "performance"):
    """Log performance metrics."""
    logger = get_unified_logger()
    logger.log_performance_metrics(metrics, component)

def log_search_progress(iteration: int, total_iterations: int, best_score: float, component: str = "search"):
    """Log search progress."""
    logger = get_unified_logger()
    logger.log_search_progress(iteration, total_iterations, best_score, component)

@contextmanager
def log_execution_time(operation: str, component: str = "system"):
    """Context manager for execution time logging."""
    logger = get_unified_logger()
    with logger.log_execution_time(operation, component):
        yield
