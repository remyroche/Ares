"""Centralized configuration for all decorators."""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional

from src.utils.pipeline_standards import PipelineStandards, pipeline_standards


class ValidationMode(Enum):
    STRICT = "strict"
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
    def from_dict(cls, config_dict: Dict[str, Any]) -> "DecoratorConfig":
        """Create config from dictionary."""
        return cls(**config_dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "validation_mode": self.validation_mode.value,
            "enable_data_quality_checks": self.enable_data_quality_checks,
            "enable_performance_monitoring": self.enable_performance_monitoring,
            "enable_error_recovery": self.enable_error_recovery,
            "performance_mode": self.performance_mode.value,
            "cache_enabled": self.cache_enabled,
            "cache_size": self.cache_size,
            "cache_ttl": self.cache_size,
            "max_retries": self.max_retries,
            "backoff_factor": self.backoff_factor,
            "log_errors": self.log_errors,
            "max_nan_ratio": self.max_nan_ratio,
            "max_infinite_count": self.max_infinite_count,
            "min_unique_values": self.min_unique_values,
        }


# Global configuration instance
global_config = DecoratorConfig()
