"""
Pipeline components for Ares trading bot.

This module provides reusable components for pipeline implementations,
including lifecycle management, signal handling, data management,
and monitoring capabilities.
"""

from src.utils.warning_symbols import (
    connection_error,
    critical,
    error,
    execution_error,
    failed,
    initialization_error,
    invalid,
    missing,
    problem,
    timeout,
    validation_error,
    warning,
)

from .checkpoint_manager import PipelineCheckpointManager
from .config_manager import ConfigManager
from .data_manager import DataManager
from .lifecycle_manager import LifecycleManager
from .monitoring_manager import MonitoringManager
from .notification_manager import NotificationManager
from .signal_handler import PipelineSignalHandler

__all__ = [
    "LifecycleManager",
    "PipelineSignalHandler",
    "ConfigManager",
    "DataManager",
    "PipelineCheckpointManager",
    "NotificationManager",
    "MonitoringManager",
]
