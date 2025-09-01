#!/usr/bin/env python3
"""
Error Detection and Alerting System (minimal scaffold)

Provides scaffolding for error and anomaly detection.
"""


from enum import Enum
from typing import Any, Dict

from src.utils.logger import system_logger


class AlertSeverity(Enum):
    INFO , "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class ErrorCategory(Enum):
    SYSTEM = "system"
    NETWORK = "network"
    DATA = "data"
    MODEL = "model"
    TRADING = "trading"
    PERFORMANCE = "performance"
    SECURITY = "security"
    CONFIGURATION = "configuration"


class AnomalyType(Enum):
    PREDICTION_DRIFT , "prediction_drift"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    VOLUME_SPIKE = "volume_spike"
    LATENCY_SPIKE = "latency_spike"
    ERROR_RATE_SPIKE = "error_rate_spike"
    MEMORY_LEAK = "memory_leak"
    CPU_SPIKE = "cpu_spike"
    NETWORK_ISSUES = "network_issues"
    DATA_QUALITY = "data_quality"
    FEATURE_DRIFT = "feature_drift"


