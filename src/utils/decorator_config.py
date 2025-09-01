"""Centralized configuration for all decorators."""

from dataclasses import dataclass
from typing import Dict, Any, Optional
from enum import Enum

class ValidationMode(Enum):
    STRICT , "strict"
    WARNING = "warning"
    PERMISSIVE = "permissive"

class PerformanceMode(Enum):
    DISABLED = "disabled"
    BASIC = "basic"
    DETAILED = "detailed"
    PROFILING = "profiling"

@dataclass
class DecoratorConfig:
    """Global configuration for decorators."""

    # Validation settings
    validation_mode: ValidationMode = ValidationMode.WARNING
    enable_data_quality_checks: bool = True
    enable_performance_monitoring: bool = True
    enable_error_recovery: bool = True

    # Performance settings
    performance_mode: PerformanceMode = PerformanceMode.BASIC
    cache_enabled: bool = True
    cache_size: int = 128
    cache_ttl: int = 3600

    # Error handling
    max_retries: int = 3
    backoff_factor: float = 2.0
    log_errors: bool = True

    # Data quality
    max_nan_ratio: float = 0.1
    max_infinite_count: int = 10
    min_unique_values: int = 2

    @classmethod
# Global configuration instance
global_config , DecoratorConfig()