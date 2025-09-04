from __future__ import annotations

"""Centralized configuration for all decorators with comprehensive error handling."""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

# Import enhanced logging functions
try:
    from .logger import log_error_with_context, log_system_status
    from .warning_symbols import error, warning, info, success
except ImportError:
    # Fallback if imports fail
    def log_error_with_context(logger, error, context=None, operation="", recovery_attempted=False):
        logger.error(f"Error in {operation}: {error}")
    
    def log_system_status(logger, component, status, details="", health_metrics=None):
        logger.info(f"System Status | {component} | {status}")
    
    def error(msg): return f"❌ {msg}"
    def warning(msg): return f"⚠️ {msg}"
    def info(msg): return f"ℹ️ {msg}"
    def success(msg): return f"✅ {msg}"

logger = logging.getLogger(__name__)


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
    def from_dict(cls, config_dict: dict[str, Any]) -> "DecoratorConfig":
        """Create config from dictionary with comprehensive error handling."""
        try:
            logger.info(f"🔧 Creating DecoratorConfig from dictionary with {len(config_dict)} keys")
            
            # Validate required fields
            required_fields = ['validation_mode', 'performance_mode']
            for field in required_fields:
                if field not in config_dict:
                    logger.warning(f"⚠️ Missing required field '{field}' in config dictionary")
            
            # Create config with error handling
            config = cls(**config_dict)
            logger.info(f"✅ DecoratorConfig created successfully")
            return config
            
        except Exception as e:
            logger.error(f"❌ Failed to create DecoratorConfig from dictionary: {e}")
            log_error_with_context(
                logger, e, 
                context={"config_keys": list(config_dict.keys()) if config_dict else []},
                operation="DecoratorConfig.from_dict"
            )
            # Return default config as fallback
            logger.info("🔄 Returning default DecoratorConfig as fallback")
            return cls()

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary with comprehensive error handling."""
        try:
            logger.info("🔧 Converting DecoratorConfig to dictionary")
            
            result = {
                "validation_mode": self.validation_mode.value,
                "enable_data_quality_checks": self.enable_data_quality_checks,
                "enable_performance_monitoring": self.enable_performance_monitoring,
                "enable_error_recovery": self.enable_error_recovery,
                "performance_mode": self.performance_mode.value,
                "cache_enabled": self.cache_enabled,
                "cache_size": self.cache_size,
                "cache_ttl": self.cache_ttl,  # Fixed: was using cache_size instead of cache_ttl
                "max_retries": self.max_retries,
                "backoff_factor": self.backoff_factor,
                "log_errors": self.log_errors,
                "max_nan_ratio": self.max_nan_ratio,
                "max_infinite_count": self.max_infinite_count,
                "min_unique_values": self.min_unique_values,
            }
            
            logger.info(f"✅ DecoratorConfig converted to dictionary with {len(result)} fields")
            return result
            
        except Exception as e:
            logger.error(f"❌ Failed to convert DecoratorConfig to dictionary: {e}")
            log_error_with_context(
                logger, e,
                operation="DecoratorConfig.to_dict"
            )
            # Return minimal fallback dictionary
            return {"error": "Failed to convert config to dictionary"}

    def validate_config(self) -> tuple[bool, list[str]]:
        """
        Validate the current configuration and return validation results.
        
        Returns:
            tuple[bool, list[str]]: (is_valid, list_of_issues)
        """
        issues = []
        
        try:
            logger.info("🔍 Validating DecoratorConfig")
            
            # Validate numeric ranges
            if self.cache_size <= 0:
                issues.append("cache_size must be positive")
                
            if self.cache_ttl <= 0:
                issues.append("cache_ttl must be positive")
                
            if self.max_retries < 0:
                issues.append("max_retries must be non-negative")
                
            if self.backoff_factor <= 0:
                issues.append("backoff_factor must be positive")
                
            if not 0 <= self.max_nan_ratio <= 1:
                issues.append("max_nan_ratio must be between 0 and 1")
                
            if self.max_infinite_count < 0:
                issues.append("max_infinite_count must be non-negative")
                
            if self.min_unique_values < 1:
                issues.append("min_unique_values must be at least 1")
            
            is_valid = len(issues) == 0
            
            if is_valid:
                logger.info("✅ DecoratorConfig validation passed")
            else:
                logger.warning(f"⚠️ DecoratorConfig validation failed with {len(issues)} issues")
                for issue in issues:
                    logger.warning(f"  - {issue}")
                    
            return is_valid, issues
            
        except Exception as e:
            logger.error(f"❌ Error during DecoratorConfig validation: {e}")
            log_error_with_context(
                logger, e,
                operation="DecoratorConfig.validate_config"
            )
            return False, [f"Validation error: {e}"]

    def get_health_status(self) -> dict[str, Any]:
        """
        Get health status of the configuration.
        
        Returns:
            dict[str, Any]: Health status information
        """
        try:
            logger.info("🏥 Getting DecoratorConfig health status")
            
            is_valid, issues = self.validate_config()
            
            # Determine status based on validation
            if is_valid and len(issues) == 0:
                status = "excellent"
            elif is_valid and len(issues) <= 2:
                status = "good"
            elif is_valid:
                status = "fair"
            else:
                status = "poor"
            
            health_status = {
                "status": status,
                "is_valid": is_valid,
                "issues": issues,
                "issue_count": len(issues),
                "validation_mode": self.validation_mode.value,
                "performance_mode": self.performance_mode.value,
                "error_recovery_enabled": self.enable_error_recovery,
                "data_quality_checks_enabled": self.enable_data_quality_checks,
                "performance_monitoring_enabled": self.enable_performance_monitoring,
                "cache_enabled": self.cache_enabled,
                "log_errors_enabled": self.log_errors
            }
            
            # Only log health status if there are issues (fair/poor)
            if not is_valid or len(issues) > 0:
                status = "degraded" if is_valid else "failed"
                try:
                    log_system_status(
                        logger, "DecoratorConfig", status,
                        details=f"Validation: {'PASSED' if is_valid else 'FAILED'}",
                        health_metrics=health_status
                    )
                except Exception:
                    # Fallback if log_system_status fails
                    pass
            
            return health_status
            
        except Exception as e:
            logger.error(f"❌ Error getting DecoratorConfig health status: {e}")
            log_error_with_context(
                logger, e,
                operation="DecoratorConfig.get_health_status"
            )
            return {
                "is_valid": False,
                "issues": [f"Health check error: {e}"],
                "issue_count": 1,
                "error": str(e)
            }


# Global configuration instance
global_config = DecoratorConfig()
