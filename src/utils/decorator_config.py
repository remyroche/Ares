"""Centralized configuration for all decorators."""

from dataclasses import dataclass
from typing import Dict, Any, Optional, List
from enum import Enum
import logging
from pathlib import Path
import json
import yaml

class ValidationMode(Enum):
    """Validation modes for decorators."""
    STRICT = "strict"
    WARNING = "warning"
    PERMISSIVE = "permissive"

class PerformanceMode(Enum):
    """Performance monitoring modes."""
    DISABLED = "disabled"
    BASIC = "basic"
    DETAILED = "detailed"
    PROFILING = "profiling"

class LoggingMode(Enum):
    """Logging modes for decorators."""
    SILENT = "silent"
    MINIMAL = "minimal"
    VERBOSE = "verbose"
    DEBUG = "debug"

@dataclass
class DecoratorConfig:
    """Global configuration for decorators."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize DecoratorConfig."""
        self.config = config or {}
        self.logger = logging.getLogger("DecoratorConfig")
        self.is_initialized = False
        
        # Default configuration
        self.defaults = {
            "validation_mode": ValidationMode.STRICT,
            "performance_mode": PerformanceMode.BASIC,
            "logging_mode": LoggingMode.MINIMAL,
            "enable_caching": True,
            "cache_ttl": 300,  # 5 minutes
            "max_cache_size": 1000,
            "enable_metrics": True,
            "enable_tracing": False,
            "timeout": 30,
            "retry_attempts": 3,
            "retry_delay": 1.0,
            "enable_async": True,
            "max_concurrent": 10,
            "enable_validation_cache": True,
            "validation_cache_ttl": 600,  # 10 minutes
            "enable_performance_monitoring": True,
            "performance_threshold_ms": 100,
            "enable_error_recovery": True,
            "error_recovery_strategies": ["retry", "fallback", "circuit_breaker"],
            "enable_security_checks": True,
            "security_level": "medium",
            "enable_audit_logging": False,
            "audit_log_path": "/workspace/logs/decorator_audit.log"
        }
        
        # Load configuration
        self._load_config()
        self.is_initialized = True
    
    def _load_config(self) -> None:
        """Load configuration from various sources."""
        # Start with defaults
        self.current_config = self.defaults.copy()
        
        # Load from environment variables
        self._load_from_env()
        
        # Load from config files
        self._load_from_files()
        
        # Override with provided config
        if self.config:
            self.current_config.update(self.config)
        
        # Validate configuration
        self._validate_config()
    
    def _load_from_env(self) -> None:
        """Load configuration from environment variables."""
        import os
        
        env_mappings = {
            "DECORATOR_VALIDATION_MODE": ("validation_mode", self._parse_validation_mode),
            "DECORATOR_PERFORMANCE_MODE": ("performance_mode", self._parse_performance_mode),
            "DECORATOR_LOGGING_MODE": ("logging_mode", self._parse_logging_mode),
            "DECORATOR_ENABLE_CACHING": ("enable_caching", self._parse_bool),
            "DECORATOR_CACHE_TTL": ("cache_ttl", int),
            "DECORATOR_MAX_CACHE_SIZE": ("max_cache_size", int),
            "DECORATOR_ENABLE_METRICS": ("enable_metrics", self._parse_bool),
            "DECORATOR_ENABLE_TRACING": ("enable_tracing", self._parse_bool),
            "DECORATOR_TIMEOUT": ("timeout", int),
            "DECORATOR_RETRY_ATTEMPTS": ("retry_attempts", int),
            "DECORATOR_RETRY_DELAY": ("retry_delay", float),
            "DECORATOR_ENABLE_ASYNC": ("enable_async", self._parse_bool),
            "DECORATOR_MAX_CONCURRENT": ("max_concurrent", int),
            "DECORATOR_ENABLE_VALIDATION_CACHE": ("enable_validation_cache", self._parse_bool),
            "DECORATOR_VALIDATION_CACHE_TTL": ("validation_cache_ttl", int),
            "DECORATOR_ENABLE_PERFORMANCE_MONITORING": ("enable_performance_monitoring", self._parse_bool),
            "DECORATOR_PERFORMANCE_THRESHOLD_MS": ("performance_threshold_ms", int),
            "DECORATOR_ENABLE_ERROR_RECOVERY": ("enable_error_recovery", self._parse_bool),
            "DECORATOR_ENABLE_SECURITY_CHECKS": ("enable_security_checks", self._parse_bool),
            "DECORATOR_SECURITY_LEVEL": ("security_level", str),
            "DECORATOR_ENABLE_AUDIT_LOGGING": ("enable_audit_logging", self._parse_bool),
            "DECORATOR_AUDIT_LOG_PATH": ("audit_log_path", str)
        }
        
        for env_var, (config_key, parser) in env_mappings.items():
            if env_var in os.environ:
                try:
                    value = parser(os.environ[env_var])
                    self.current_config[config_key] = value
                except (ValueError, TypeError) as e:
                    self.logger.warning(f"Failed to parse environment variable {env_var}: {e}")
    
    def _load_from_files(self) -> None:
        """Load configuration from config files."""
        config_paths = [
            Path("/workspace/config/decorator_config.yaml"),
            Path("/workspace/config/decorator_config.yml"),
            Path("/workspace/config/decorator_config.json"),
            Path("/workspace/.decorator_config.yaml"),
            Path("/workspace/.decorator_config.yml"),
            Path("/workspace/.decorator_config.json")
        ]
        
        for config_path in config_paths:
            if config_path.exists():
                try:
                    if config_path.suffix in ['.yaml', '.yml']:
                        with open(config_path, 'r') as f:
                            file_config = yaml.safe_load(f)
                    elif config_path.suffix == '.json':
                        with open(config_path, 'r') as f:
                            file_config = json.load(f)
                    else:
                        continue
                    
                    if file_config:
                        self.current_config.update(file_config)
                        self.logger.info(f"Loaded configuration from {config_path}")
                        break
                        
                except Exception as e:
                    self.logger.warning(f"Failed to load config from {config_path}: {e}")
    
    def _validate_config(self) -> None:
        """Validate configuration values."""
        # Validate numeric values
        numeric_fields = ["cache_ttl", "max_cache_size", "timeout", "retry_attempts", 
                         "retry_delay", "max_concurrent", "validation_cache_ttl", 
                         "performance_threshold_ms"]
        
        for field in numeric_fields:
            if field in self.current_config:
                value = self.current_config[field]
                if not isinstance(value, (int, float)) or value < 0:
                    self.logger.warning(f"Invalid value for {field}: {value}, using default")
                    self.current_config[field] = self.defaults[field]
        
        # Validate boolean values
        boolean_fields = ["enable_caching", "enable_metrics", "enable_tracing", 
                         "enable_async", "enable_validation_cache", 
                         "enable_performance_monitoring", "enable_error_recovery",
                         "enable_security_checks", "enable_audit_logging"]
        
        for field in boolean_fields:
            if field in self.current_config:
                value = self.current_config[field]
                if not isinstance(value, bool):
                    self.logger.warning(f"Invalid value for {field}: {value}, using default")
                    self.current_config[field] = self.defaults[field]
    
    def _parse_validation_mode(self, value: str) -> ValidationMode:
        """Parse validation mode from string."""
        try:
            return ValidationMode(value.lower())
        except ValueError:
            return ValidationMode.STRICT
    
    def _parse_performance_mode(self, value: str) -> PerformanceMode:
        """Parse performance mode from string."""
        try:
            return PerformanceMode(value.lower())
        except ValueError:
            return PerformanceMode.BASIC
    
    def _parse_logging_mode(self, value: str) -> LoggingMode:
        """Parse logging mode from string."""
        try:
            return LoggingMode(value.lower())
        except ValueError:
            return LoggingMode.MINIMAL
    
    def _parse_bool(self, value: str) -> bool:
        """Parse boolean from string."""
        if value.lower() in ['true', '1', 'yes', 'on']:
            return True
        elif value.lower() in ['false', '0', 'no', 'off']:
            return False
        else:
            raise ValueError(f"Cannot parse boolean from: {value}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self.current_config.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value."""
        self.current_config[key] = value
        self.logger.info(f"Configuration updated: {key} = {value}")
    
    def update(self, config: Dict[str, Any]) -> None:
        """Update configuration with new values."""
        self.current_config.update(config)
        self.logger.info(f"Configuration updated with {len(config)} values")
    
    def reset(self) -> None:
        """Reset configuration to defaults."""
        self.current_config = self.defaults.copy()
        self.logger.info("Configuration reset to defaults")
    
    def get_config(self) -> Dict[str, Any]:
        """Get current configuration as dictionary."""
        return self.current_config.copy()
    
    def get_validation_mode(self) -> ValidationMode:
        """Get current validation mode."""
        return self.current_config["validation_mode"]
    
    def get_performance_mode(self) -> PerformanceMode:
        """Get current performance mode."""
        return self.current_config["performance_mode"]
    
    def get_logging_mode(self) -> LoggingMode:
        """Get current logging mode."""
        return self.current_config["logging_mode"]
    
    def is_caching_enabled(self) -> bool:
        """Check if caching is enabled."""
        return self.current_config["enable_caching"]
    
    def is_metrics_enabled(self) -> bool:
        """Check if metrics are enabled."""
        return self.current_config["enable_metrics"]
    
    def is_tracing_enabled(self) -> bool:
        """Check if tracing is enabled."""
        return self.current_config["enable_tracing"]
    
    def is_async_enabled(self) -> bool:
        """Check if async is enabled."""
        return self.current_config["enable_async"]
    
    def is_performance_monitoring_enabled(self) -> bool:
        """Check if performance monitoring is enabled."""
        return self.current_config["enable_performance_monitoring"]
    
    def is_error_recovery_enabled(self) -> bool:
        """Check if error recovery is enabled."""
        return self.current_config["enable_error_recovery"]
    
    def is_security_checks_enabled(self) -> bool:
        """Check if security checks are enabled."""
        return self.current_config["enable_security_checks"]
    
    def is_audit_logging_enabled(self) -> bool:
        """Check if audit logging is enabled."""
        return self.current_config["enable_audit_logging"]
    
    def get_cache_config(self) -> Dict[str, Any]:
        """Get cache-related configuration."""
        return {
            "enabled": self.current_config["enable_caching"],
            "ttl": self.current_config["cache_ttl"],
            "max_size": self.current_config["max_cache_size"],
            "validation_cache_enabled": self.current_config["enable_validation_cache"],
            "validation_cache_ttl": self.current_config["validation_cache_ttl"]
        }
    
    def get_performance_config(self) -> Dict[str, Any]:
        """Get performance-related configuration."""
        return {
            "mode": self.current_config["performance_mode"],
            "monitoring_enabled": self.current_config["enable_performance_monitoring"],
            "threshold_ms": self.current_config["performance_threshold_ms"],
            "timeout": self.current_config["timeout"],
            "retry_attempts": self.current_config["retry_attempts"],
            "retry_delay": self.current_config["retry_delay"],
            "max_concurrent": self.current_config["max_concurrent"]
        }
    
    def get_security_config(self) -> Dict[str, Any]:
        """Get security-related configuration."""
        return {
            "checks_enabled": self.current_config["enable_security_checks"],
            "level": self.current_config["security_level"],
            "audit_logging_enabled": self.current_config["enable_audit_logging"],
            "audit_log_path": self.current_config["audit_log_path"]
        }
    
    def export_config(self, file_path: str, format: str = "yaml") -> bool:
        """Export configuration to file."""
        try:
            config_data = self.get_config()
            
            # Convert enums to strings for serialization
            for key, value in config_data.items():
                if isinstance(value, Enum):
                    config_data[key] = value.value
            
            if format.lower() == "yaml":
                with open(file_path, 'w') as f:
                    yaml.dump(config_data, f, default_flow_style=False, indent=2)
            elif format.lower() == "json":
                with open(file_path, 'w') as f:
                    json.dump(config_data, f, indent=2)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            self.logger.info(f"Configuration exported to {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export configuration: {e}")
            return False
    
    def __repr__(self) -> str:
        """String representation of configuration."""
        return f"DecoratorConfig(validation_mode={self.get_validation_mode()}, " \
               f"performance_mode={self.get_performance_mode()}, " \
               f"logging_mode={self.get_logging_mode()})"

# Global configuration instance
global_config = DecoratorConfig()