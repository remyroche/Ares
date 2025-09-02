"""
Enhanced Configuration Management Utilities

This module provides structured configuration management for the training pipeline.
"""

import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import logging

try:
    from src.utils.logger import system_logger
    from src.utils.pipeline_standards import PipelineStandards, pipeline_standards
except ImportError:
    system_logger = logging.getLogger("EnhancedConfigManagement")


@dataclass
class PlaceholderDataClass:
    """Placeholder data class for configuration management."""
    
    def __init__(self):
        self.is_initialized = False
        self.logger = logging.getLogger(__name__)

    async def initialize(self) -> bool:
        """Initialize PlaceholderDataClass."""
        try:
            self.logger.info(f"🚀 Initializing {self.__class__.__name__}...")
            self.is_initialized = True
            self.logger.info(f"✅ {self.__class__.__name__} initialized successfully")
            return True
        except Exception as e:
            self.logger.exception(f"❌ Error initializing {self.__class__.__name__}: {e}")
            return False


@dataclass
class Step1Config:
    """Enhanced configuration for Step1 data collection."""

    # Basic parameters
    symbol: str = "ETHUSDT"
    exchange: str = "BINANCE"
    timeframe: str = "1m"
    lookback_days: int = 1095

    # Performance parameters
    max_retries: int = 3
    retry_backoff_factor: float = 2.0
    chunk_size: int = 10000
    max_memory_mb: int = 1024
    max_workers: int = 4

    # Quality thresholds
    max_nan_ratio: float = 0.0  # Zero tolerance for NaN
    max_infinite_count: int = 0  # Zero tolerance for infinite values
    min_unique_values: int = 2
    max_gap_hours: int = 48
    price_tolerance: float = 0.001
    volume_tolerance: float = 0.001

    # Data directories
    data_dir: str = "data_cache"
    backup_dir: str = "data_cache/backup"
    temp_dir: str = "data_cache/temp"

    # Error handling
    enable_retry: bool = True
    enable_fallback: bool = True
    log_level: str = "INFO"
    
    # Validation settings
    validate_data_quality: bool = True
    validate_schema: bool = True
    strict_mode: bool = False
    
    # Caching settings
    enable_cache: bool = True
    cache_ttl_seconds: int = 3600
    cache_max_size: int = 1000
    
    # Monitoring settings
    enable_metrics: bool = True
    enable_tracing: bool = True
    performance_threshold_ms: int = 5000

    def __post_init__(self):
        """Post-initialization validation and setup."""
        self._validate_config()
        self._setup_directories()
        self._setup_logging()

    def _validate_config(self):
        """Validate configuration parameters."""
        if self.lookback_days <= 0:
            raise ValueError("lookback_days must be positive")
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.max_memory_mb <= 0:
            raise ValueError("max_memory_mb must be positive")
        if self.max_workers <= 0:
            raise ValueError("max_workers must be positive")
        if not 0 <= self.max_nan_ratio <= 1:
            raise ValueError("max_nan_ratio must be between 0 and 1")
        if self.price_tolerance <= 0:
            raise ValueError("price_tolerance must be positive")
        if self.volume_tolerance <= 0:
            raise ValueError("volume_tolerance must be positive")

    def _setup_directories(self):
        """Create necessary directories if they don't exist."""
        for directory in [self.data_dir, self.backup_dir, self.temp_dir]:
            Path(directory).mkdir(parents=True, exist_ok=True)

    def _setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=getattr(logging, self.log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Convert configuration to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    def save_to_file(self, file_path: str):
        """Save configuration to file."""
        with open(file_path, 'w') as f:
            f.write(self.to_json())

    @classmethod
    def load_from_file(cls, file_path: str) -> 'Step1Config':
        """Load configuration from file."""
        with open(file_path, 'r') as f:
            config_data = json.load(f)
        return cls(**config_data)

    def update(self, **kwargs):
        """Update configuration with new values."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown configuration key: {key}")
        self._validate_config()

    def get_data_path(self, filename: str) -> str:
        """Get full path for data file."""
        return os.path.join(self.data_dir, filename)

    def get_backup_path(self, filename: str) -> str:
        """Get full path for backup file."""
        return os.path.join(self.backup_dir, filename)

    def get_temp_path(self, filename: str) -> str:
        """Get full path for temporary file."""
        return os.path.join(self.temp_dir, filename)

    def is_valid_symbol(self, symbol: str) -> bool:
        """Check if symbol is valid for the exchange."""
        # Basic validation - could be enhanced with exchange-specific logic
        return isinstance(symbol, str) and len(symbol) > 0 and '/' in symbol

    def is_valid_timeframe(self, timeframe: str) -> bool:
        """Check if timeframe is valid."""
        valid_timeframes = ['1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w']
        return timeframe in valid_timeframes

    def get_chunk_size_for_memory(self, available_memory_mb: int) -> int:
        """Calculate optimal chunk size based on available memory."""
        # Reserve 20% of memory for other operations
        usable_memory = int(available_memory_mb * 0.8)
        # Estimate memory per row (conservative estimate)
        memory_per_row = 0.001  # MB per row
        optimal_chunk_size = int(usable_memory / memory_per_row)
        return min(optimal_chunk_size, self.chunk_size)

    def get_retry_delay(self, attempt: int) -> float:
        """Calculate retry delay using exponential backoff."""
        if attempt <= 0:
            return 0
        return min(self.retry_backoff_factor ** attempt, 60)  # Cap at 60 seconds

    def should_retry(self, attempt: int, error: Exception) -> bool:
        """Determine if operation should be retried."""
        if not self.enable_retry:
            return False
        if attempt >= self.max_retries:
            return False
        # Add specific error type checks here if needed
        return True

    def get_quality_thresholds(self) -> Dict[str, Any]:
        """Get data quality thresholds."""
        return {
            'max_nan_ratio': self.max_nan_ratio,
            'max_infinite_count': self.max_infinite_count,
            'min_unique_values': self.min_unique_values,
            'max_gap_hours': self.max_gap_hours,
            'price_tolerance': self.price_tolerance,
            'volume_tolerance': self.volume_tolerance
        }

    def get_performance_settings(self) -> Dict[str, Any]:
        """Get performance-related settings."""
        return {
            'chunk_size': self.chunk_size,
            'max_memory_mb': self.max_memory_mb,
            'max_workers': self.max_workers,
            'enable_cache': self.enable_cache,
            'cache_ttl_seconds': self.cache_ttl_seconds,
            'cache_max_size': self.cache_max_size
        }

    def get_monitoring_settings(self) -> Dict[str, Any]:
        """Get monitoring and observability settings."""
        return {
            'enable_metrics': self.enable_metrics,
            'enable_tracing': self.enable_tracing,
            'performance_threshold_ms': self.performance_threshold_ms,
            'log_level': self.log_level
        }

    def validate_data_quality(self, data: Any) -> bool:
        """Validate data quality against configured thresholds."""
        if not self.validate_data_quality:
            return True
        
        # This is a placeholder implementation
        # In a real implementation, you would check the actual data
        # against the quality thresholds
        return True

    def validate_schema(self, data: Any) -> bool:
        """Validate data schema against expected structure."""
        if not self.validate_schema:
            return True
        
        # This is a placeholder implementation
        # In a real implementation, you would check the actual data
        # against the expected schema
        return True

    def get_fallback_config(self) -> 'Step1Config':
        """Get fallback configuration with conservative settings."""
        fallback = Step1Config()
        fallback.chunk_size = min(self.chunk_size, 1000)
        fallback.max_workers = min(self.max_workers, 2)
        fallback.max_memory_mb = min(self.max_memory_mb, 512)
        fallback.strict_mode = False
        fallback.enable_retry = True
        fallback.max_retries = 5
        return fallback

    def __str__(self) -> str:
        """String representation of configuration."""
        return f"Step1Config(symbol={self.symbol}, exchange={self.exchange}, timeframe={self.timeframe})"

    def __repr__(self) -> str:
        """Detailed string representation of configuration."""
        return f"Step1Config({self.to_dict()})"


@dataclass
class EnhancedConfigManager:
    """Enhanced configuration manager with validation and fallback support."""
    
    primary_config: Step1Config
    fallback_config: Optional[Step1Config] = None
    config_history: List[Dict[str, Any]] = field(default_factory=list)
    validation_enabled: bool = True
    auto_fallback: bool = True
    
    def __post_init__(self):
        """Post-initialization setup."""
        if self.fallback_config is None:
            self.fallback_config = self.primary_config.get_fallback_config()
        
        # Initialize logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(self.primary_config.log_level)

    def get_config(self, use_fallback: bool = False) -> Step1Config:
        """Get configuration, optionally using fallback."""
        if use_fallback and self.fallback_config:
            self.logger.warning("Using fallback configuration")
            return self.fallback_config
        return self.primary_config

    def validate_config(self, config: Step1Config) -> bool:
        """Validate configuration."""
        if not self.validation_enabled:
            return True
        
        try:
            config._validate_config()
            return True
        except Exception as e:
            self.logger.error(f"Configuration validation failed: {e}")
            return False

    def update_config(self, **kwargs) -> bool:
        """Update configuration with validation."""
        try:
            # Store current config in history
            self.config_history.append({
                'timestamp': self._get_timestamp(),
                'config': self.primary_config.to_dict()
            })
            
            # Update configuration
            self.primary_config.update(**kwargs)
            
            # Validate updated configuration
            if not self.validate_config(self.primary_config):
                # Rollback to previous configuration
                self._rollback_config()
                return False
            
            self.logger.info("Configuration updated successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update configuration: {e}")
            return False

    def _rollback_config(self):
        """Rollback to previous configuration."""
        if self.config_history:
            previous_config = self.config_history.pop()
            self.primary_config = Step1Config(**previous_config['config'])
            self.logger.info("Configuration rolled back to previous state")

    def _get_timestamp(self) -> str:
        """Get current timestamp string."""
        from datetime import datetime
        return datetime.now().isoformat()

    def export_config(self, file_path: str) -> bool:
        """Export current configuration to file."""
        try:
            self.primary_config.save_to_file(file_path)
            self.logger.info(f"Configuration exported to {file_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to export configuration: {e}")
            return False

    def import_config(self, file_path: str) -> bool:
        """Import configuration from file."""
        try:
            new_config = Step1Config.load_from_file(file_path)
            if self.validate_config(new_config):
                self.primary_config = new_config
                self.logger.info(f"Configuration imported from {file_path}")
                return True
            else:
                self.logger.error("Imported configuration validation failed")
                return False
        except Exception as e:
            self.logger.error(f"Failed to import configuration: {e}")
            return False

    def get_config_summary(self) -> Dict[str, Any]:
        """Get configuration summary."""
        return {
            'primary_config': self.primary_config.to_dict(),
            'fallback_config': self.fallback_config.to_dict() if self.fallback_config else None,
            'validation_enabled': self.validation_enabled,
            'auto_fallback': self.auto_fallback,
            'config_history_count': len(self.config_history),
            'last_updated': self.config_history[-1]['timestamp'] if self.config_history else None
        }

    def reset_to_defaults(self) -> bool:
        """Reset configuration to default values."""
        try:
            self.primary_config = Step1Config()
            self.logger.info("Configuration reset to defaults")
            return True
        except Exception as e:
            self.logger.error(f"Failed to reset configuration: {e}")
            return False

    def get_environment_overrides(self) -> Dict[str, Any]:
        """Get configuration overrides from environment variables."""
        overrides = {}
        env_mapping = {
            'SYMBOL': 'symbol',
            'EXCHANGE': 'exchange',
            'TIMEFRAME': 'timeframe',
            'LOOKBACK_DAYS': 'lookback_days',
            'MAX_RETRIES': 'max_retries',
            'CHUNK_SIZE': 'chunk_size',
            'MAX_MEMORY_MB': 'max_memory_mb',
            'MAX_WORKERS': 'max_workers'
        }
        
        for env_var, config_key in env_mapping.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                try:
                    # Try to convert to appropriate type
                    if config_key in ['lookback_days', 'max_retries', 'chunk_size', 'max_memory_mb', 'max_workers']:
                        overrides[config_key] = int(env_value)
                    elif config_key in ['retry_backoff_factor', 'max_nan_ratio', 'price_tolerance', 'volume_tolerance']:
                        overrides[config_key] = float(env_value)
                    else:
                        overrides[config_key] = env_value
                except ValueError:
                    self.logger.warning(f"Invalid environment variable value for {env_var}: {env_value}")
        
        return overrides

    def apply_environment_overrides(self) -> bool:
        """Apply configuration overrides from environment variables."""
        overrides = self.get_environment_overrides()
        if overrides:
            return self.update_config(**overrides)
        return True


# Default configuration instance
default_config = Step1Config()

# Enhanced configuration manager instance
config_manager = EnhancedConfigManager(
    primary_config=default_config,
    validation_enabled=True,
    auto_fallback=True
)

# Export main classes and instances
__all__ = [
    'Step1Config',
    'EnhancedConfigManager',
    'default_config',
    'config_manager'
]