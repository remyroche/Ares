"""
Configuration Management System

This module provides centralized configuration management for all exchange
OHLCV data processing operations.

Features:
- Centralized configuration management
- Environment-specific configurations
- Runtime configuration updates
- Configuration validation
- Secure credential management
- Performance tuning parameters
"""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Type
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging
from datetime import datetime, timezone
import threading
from contextlib import contextmanager

# Import our unified components
from .unified_ohlcv_standardizer import ExchangeType, DataQualityLevel
from .unified_exchange_interface import UnifiedExchangeManager

# Import src/utils/data utilities
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.logger import system_logger

logger = logging.getLogger(__name__)


class ConfigEnvironment(Enum):
    """Configuration environments"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


class ConfigSource(Enum):
    """Configuration sources"""
    FILE = "file"
    ENVIRONMENT = "environment"
    DATABASE = "database"
    API = "api"
    DEFAULT = "default"


@dataclass
class ExchangeConfig:
    """Configuration for a specific exchange"""
    name: str
    enabled: bool = True
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    base_url: str = ""
    sandbox: bool = False
    rate_limits: Dict[str, int] = field(default_factory=dict)
    timeout: int = 30
    max_retries: int = 3
    data_quality_level: str = "standard"
    custom_headers: Dict[str, str] = field(default_factory=dict)
    proxy_settings: Optional[Dict[str, str]] = None


@dataclass
class DataProcessingConfig:
    """Configuration for data processing operations"""
    batch_size: int = 1000
    max_memory_usage_mb: int = 1000
    enable_caching: bool = True
    cache_ttl_seconds: int = 3600
    enable_compression: bool = True
    compression_level: int = 6
    parallel_processing: bool = True
    max_workers: int = 4
    chunk_size: int = 10000


@dataclass
class QualityConfig:
    """Configuration for data quality validation"""
    validation_level: str = "standard"
    enable_anomaly_detection: bool = True
    enable_outlier_detection: bool = True
    enable_pattern_analysis: bool = True
    quality_threshold: float = 75.0
    auto_fix_issues: bool = False
    enable_cross_validation: bool = True
    validation_timeout: int = 30


@dataclass
class PerformanceConfig:
    """Configuration for performance monitoring and optimization"""
    enable_monitoring: bool = True
    monitoring_interval: float = 1.0
    max_metrics_history: int = 1000
    enable_auto_optimization: bool = False
    performance_thresholds: Dict[str, float] = field(default_factory=dict)
    enable_profiling: bool = False
    profiling_sample_rate: float = 0.1


@dataclass
class LoggingConfig:
    """Configuration for logging"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    enable_file_logging: bool = True
    log_file_path: str = "logs/exchange_ohlcv.log"
    max_file_size_mb: int = 10
    backup_count: int = 5
    enable_console_logging: bool = True
    enable_structured_logging: bool = True


@dataclass
class SecurityConfig:
    """Configuration for security settings"""
    enable_encryption: bool = True
    encryption_key: Optional[str] = None
    enable_audit_logging: bool = True
    mask_sensitive_data: bool = True
    enable_rate_limiting: bool = True
    max_requests_per_minute: int = 100
    enable_ip_whitelisting: bool = False
    allowed_ips: List[str] = field(default_factory=list)


@dataclass
class SystemConfig:
    """System-wide configuration"""
    environment: str = "development"
    debug_mode: bool = True
    enable_metrics: bool = True
    metrics_endpoint: Optional[str] = None
    enable_health_checks: bool = True
    health_check_interval: int = 60
    enable_auto_scaling: bool = False
    min_instances: int = 1
    max_instances: int = 10


@dataclass
class ExchangeOHLCVConfig:
    """Complete configuration for exchange OHLCV processing"""
    system: SystemConfig = field(default_factory=SystemConfig)
    exchanges: Dict[str, ExchangeConfig] = field(default_factory=dict)
    data_processing: DataProcessingConfig = field(default_factory=DataProcessingConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    
    # Metadata
    version: str = "1.0.0"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    config_source: str = "default"


class ConfigurationManager:
    """
    Centralized configuration manager for exchange OHLCV processing.
    
    Provides configuration loading, validation, updating, and management
    across all components of the system.
    """
    
    def __init__(self, config_dir: str = "config"):
        """Initialize the configuration manager"""
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        
        self.logger = system_logger.getChild("ConfigurationManager")
        
        # Configuration state
        self._config: Optional[ExchangeOHLCVConfig] = None
        self._config_lock = threading.RLock()
        self._config_watchers: List[callable] = []
        
        # Configuration sources
        self._config_sources: Dict[str, ConfigSource] = {}
        self._config_files: Dict[str, Path] = {}
        
        # Load default configuration
        self._load_default_config()
        
        self.logger.info("✅ ConfigurationManager initialized")
    
    def _load_default_config(self):
        """Load default configuration"""
        self._config = ExchangeOHLCVConfig()
        
        # Add default exchange configurations
        self._config.exchanges = {
            "binance": ExchangeConfig(
                name="binance",
                base_url="https://api.binance.com",
                rate_limits={"requests_per_minute": 1200, "weight_per_minute": 6000},
                data_quality_level="standard"
            ),
            "bingx": ExchangeConfig(
                name="bingx",
                base_url="https://open-api.bingx.com",
                rate_limits={"requests_per_minute": 600, "weight_per_minute": 3000},
                data_quality_level="standard"
            ),
            "okx": ExchangeConfig(
                name="okx",
                base_url="https://www.okx.com",
                rate_limits={"requests_per_minute": 300, "weight_per_minute": 1500},
                data_quality_level="standard"
            ),
            "mexc": ExchangeConfig(
                name="mexc",
                base_url="https://api.mexc.com",
                rate_limits={"requests_per_minute": 1200, "weight_per_minute": 6000},
                data_quality_level="standard"
            )
        }
        
        # Set default performance thresholds
        self._config.performance.performance_thresholds = {
            "max_operation_time": 30.0,
            "max_memory_usage_mb": 1000.0,
            "max_cpu_percent": 80.0,
            "min_success_rate": 0.95
        }
    
    def load_config(self, config_path: Union[str, Path], source: ConfigSource = ConfigSource.FILE) -> bool:
        """
        Load configuration from file.
        
        Args:
            config_path: Path to configuration file
            source: Configuration source type
            
        Returns:
            True if configuration loaded successfully
        """
        try:
            config_path = Path(config_path)
            
            if not config_path.exists():
                self.logger.error(f"Configuration file not found: {config_path}")
                return False
            
            # Load configuration based on file extension
            if config_path.suffix.lower() == '.json':
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
            elif config_path.suffix.lower() in ['.yml', '.yaml']:
                with open(config_path, 'r') as f:
                    config_data = yaml.safe_load(f)
            else:
                self.logger.error(f"Unsupported configuration file format: {config_path.suffix}")
                return False
            
            # Convert to configuration object
            new_config = self._dict_to_config(config_data)
            
            # Validate configuration
            if not self._validate_config(new_config):
                self.logger.error("Configuration validation failed")
                return False
            
            # Update configuration
            with self._config_lock:
                self._config = new_config
                self._config.updated_at = datetime.now(timezone.utc)
                self._config.config_source = source.value
            
            # Store configuration source info
            self._config_sources[source.value] = source
            self._config_files[source.value] = config_path
            
            # Notify watchers
            self._notify_config_watchers()
            
            self.logger.info(f"✅ Configuration loaded from {config_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load configuration from {config_path}: {e}")
            return False
    
    def save_config(self, config_path: Union[str, Path], format: str = "json") -> bool:
        """
        Save current configuration to file.
        
        Args:
            config_path: Path to save configuration
            format: File format (json, yaml)
            
        Returns:
            True if configuration saved successfully
        """
        try:
            config_path = Path(config_path)
            config_path.parent.mkdir(parents=True, exist_ok=True)
            
            with self._config_lock:
                if self._config is None:
                    self.logger.error("No configuration to save")
                    return False
                
                config_dict = asdict(self._config)
                
                if format.lower() == 'json':
                    with open(config_path, 'w') as f:
                        json.dump(config_dict, f, indent=2, default=str)
                elif format.lower() in ['yml', 'yaml']:
                    with open(config_path, 'w') as f:
                        yaml.dump(config_dict, f, default_flow_style=False)
                else:
                    self.logger.error(f"Unsupported format: {format}")
                    return False
            
            self.logger.info(f"✅ Configuration saved to {config_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save configuration to {config_path}: {e}")
            return False
    
    def get_config(self) -> ExchangeOHLCVConfig:
        """Get current configuration"""
        with self._config_lock:
            if self._config is None:
                self._load_default_config()
            return self._config
    
    def update_config(self, updates: Dict[str, Any]) -> bool:
        """
        Update configuration with new values.
        
        Args:
            updates: Dictionary of configuration updates
            
        Returns:
            True if configuration updated successfully
        """
        try:
            with self._config_lock:
                if self._config is None:
                    self._load_default_config()
                
                # Apply updates
                self._apply_config_updates(self._config, updates)
                self._config.updated_at = datetime.now(timezone.utc)
                
                # Validate updated configuration
                if not self._validate_config(self._config):
                    self.logger.error("Configuration validation failed after update")
                    return False
            
            # Notify watchers
            self._notify_config_watchers()
            
            self.logger.info("✅ Configuration updated successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update configuration: {e}")
            return False
    
    def get_exchange_config(self, exchange_name: str) -> Optional[ExchangeConfig]:
        """Get configuration for specific exchange"""
        config = self.get_config()
        return config.exchanges.get(exchange_name)
    
    def update_exchange_config(self, exchange_name: str, updates: Dict[str, Any]) -> bool:
        """Update configuration for specific exchange"""
        try:
            with self._config_lock:
                if self._config is None:
                    self._load_default_config()
                
                if exchange_name not in self._config.exchanges:
                    self.logger.error(f"Exchange {exchange_name} not found in configuration")
                    return False
                
                # Apply updates to exchange config
                exchange_config = self._config.exchanges[exchange_name]
                self._apply_config_updates(exchange_config, updates)
                
                self._config.updated_at = datetime.now(timezone.utc)
            
            # Notify watchers
            self._notify_config_watchers()
            
            self.logger.info(f"✅ Configuration updated for exchange {exchange_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update exchange configuration: {e}")
            return False
    
    def add_config_watcher(self, callback: callable):
        """Add configuration change watcher"""
        self._config_watchers.append(callback)
    
    def remove_config_watcher(self, callback: callable):
        """Remove configuration change watcher"""
        if callback in self._config_watchers:
            self._config_watchers.remove(callback)
    
    def _notify_config_watchers(self):
        """Notify all configuration watchers"""
        for callback in self._config_watchers:
            try:
                callback(self._config)
            except Exception as e:
                self.logger.error(f"Error in config watcher: {e}")
    
    def _dict_to_config(self, config_dict: Dict[str, Any]) -> ExchangeOHLCVConfig:
        """Convert dictionary to configuration object"""
        # Handle nested configurations
        system_config = SystemConfig(**config_dict.get('system', {}))
        
        exchanges = {}
        for name, exchange_data in config_dict.get('exchanges', {}).items():
            exchanges[name] = ExchangeConfig(**exchange_data)
        
        data_processing = DataProcessingConfig(**config_dict.get('data_processing', {}))
        quality = QualityConfig(**config_dict.get('quality', {}))
        performance = PerformanceConfig(**config_dict.get('performance', {}))
        logging = LoggingConfig(**config_dict.get('logging', {}))
        security = SecurityConfig(**config_dict.get('security', {}))
        
        return ExchangeOHLCVConfig(
            system=system_config,
            exchanges=exchanges,
            data_processing=data_processing,
            quality=quality,
            performance=performance,
            logging=logging,
            security=security,
            version=config_dict.get('version', '1.0.0'),
            created_at=datetime.fromisoformat(config_dict.get('created_at', datetime.now(timezone.utc).isoformat())),
            updated_at=datetime.fromisoformat(config_dict.get('updated_at', datetime.now(timezone.utc).isoformat())),
            config_source=config_dict.get('config_source', 'file')
        )
    
    def _apply_config_updates(self, config_obj: Any, updates: Dict[str, Any]):
        """Apply updates to configuration object"""
        for key, value in updates.items():
            if hasattr(config_obj, key):
                if isinstance(value, dict) and hasattr(getattr(config_obj, key), '__dict__'):
                    # Nested object update
                    nested_obj = getattr(config_obj, key)
                    self._apply_config_updates(nested_obj, value)
                else:
                    # Direct attribute update
                    setattr(config_obj, key, value)
    
    def _validate_config(self, config: ExchangeOHLCVConfig) -> bool:
        """Validate configuration"""
        try:
            # Validate system configuration
            if not config.system.environment in [e.value for e in ConfigEnvironment]:
                self.logger.error(f"Invalid environment: {config.system.environment}")
                return False
            
            # Validate exchange configurations
            for name, exchange_config in config.exchanges.items():
                if not exchange_config.name:
                    self.logger.error(f"Exchange {name} missing name")
                    return False
                
                if not exchange_config.base_url:
                    self.logger.error(f"Exchange {name} missing base_url")
                    return False
            
            # Validate data processing configuration
            if config.data_processing.batch_size <= 0:
                self.logger.error("Invalid batch_size")
                return False
            
            if config.data_processing.max_memory_usage_mb <= 0:
                self.logger.error("Invalid max_memory_usage_mb")
                return False
            
            # Validate quality configuration
            if not config.quality.validation_level in [l.value for l in DataQualityLevel]:
                self.logger.error(f"Invalid validation_level: {config.quality.validation_level}")
                return False
            
            if not 0 <= config.quality.quality_threshold <= 100:
                self.logger.error(f"Invalid quality_threshold: {config.quality.quality_threshold}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Configuration validation error: {e}")
            return False
    
    def load_from_environment(self) -> bool:
        """Load configuration from environment variables"""
        try:
            updates = {}
            
            # System configuration
            if os.getenv('EXCHANGE_ENVIRONMENT'):
                updates['system'] = {'environment': os.getenv('EXCHANGE_ENVIRONMENT')}
            
            if os.getenv('EXCHANGE_DEBUG'):
                updates['system'] = updates.get('system', {})
                updates['system']['debug_mode'] = os.getenv('EXCHANGE_DEBUG').lower() == 'true'
            
            # Exchange configurations
            for exchange in ['binance', 'bingx', 'okx', 'mexc']:
                exchange_updates = {}
                
                if os.getenv(f'{exchange.upper()}_API_KEY'):
                    exchange_updates['api_key'] = os.getenv(f'{exchange.upper()}_API_KEY')
                
                if os.getenv(f'{exchange.upper()}_API_SECRET'):
                    exchange_updates['api_secret'] = os.getenv(f'{exchange.upper()}_API_SECRET')
                
                if os.getenv(f'{exchange.upper()}_BASE_URL'):
                    exchange_updates['base_url'] = os.getenv(f'{exchange.upper()}_BASE_URL')
                
                if os.getenv(f'{exchange.upper()}_ENABLED'):
                    exchange_updates['enabled'] = os.getenv(f'{exchange.upper()}_ENABLED').lower() == 'true'
                
                if exchange_updates:
                    updates['exchanges'] = updates.get('exchanges', {})
                    updates['exchanges'][exchange] = exchange_updates
            
            # Data processing configuration
            if os.getenv('BATCH_SIZE'):
                updates['data_processing'] = updates.get('data_processing', {})
                updates['data_processing']['batch_size'] = int(os.getenv('BATCH_SIZE'))
            
            if os.getenv('MAX_MEMORY_MB'):
                updates['data_processing'] = updates.get('data_processing', {})
                updates['data_processing']['max_memory_usage_mb'] = int(os.getenv('MAX_MEMORY_MB'))
            
            # Quality configuration
            if os.getenv('QUALITY_LEVEL'):
                updates['quality'] = updates.get('quality', {})
                updates['quality']['validation_level'] = os.getenv('QUALITY_LEVEL')
            
            if os.getenv('QUALITY_THRESHOLD'):
                updates['quality'] = updates.get('quality', {})
                updates['quality']['quality_threshold'] = float(os.getenv('QUALITY_THRESHOLD'))
            
            if updates:
                return self.update_config(updates)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load configuration from environment: {e}")
            return False
    
    def export_config(self, filepath: Union[str, Path], format: str = "json") -> bool:
        """Export current configuration to file"""
        return self.save_config(filepath, format)
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get configuration summary"""
        config = self.get_config()
        
        return {
            'version': config.version,
            'environment': config.system.environment,
            'debug_mode': config.system.debug_mode,
            'exchanges': {
                name: {
                    'enabled': exchange.enabled,
                    'base_url': exchange.base_url,
                    'has_credentials': bool(exchange.api_key and exchange.api_secret)
                }
                for name, exchange in config.exchanges.items()
            },
            'data_processing': {
                'batch_size': config.data_processing.batch_size,
                'max_memory_mb': config.data_processing.max_memory_usage_mb,
                'parallel_processing': config.data_processing.parallel_processing
            },
            'quality': {
                'validation_level': config.quality.validation_level,
                'quality_threshold': config.quality.quality_threshold,
                'auto_fix_issues': config.quality.auto_fix_issues
            },
            'performance': {
                'monitoring_enabled': config.performance.enable_monitoring,
                'auto_optimization': config.performance.enable_auto_optimization
            },
            'last_updated': config.updated_at.isoformat()
        }


# Global configuration manager instance
config_manager = ConfigurationManager()


# Convenience functions
def get_config() -> ExchangeOHLCVConfig:
    """Get current configuration"""
    return config_manager.get_config()


def update_config(updates: Dict[str, Any]) -> bool:
    """Update configuration"""
    return config_manager.update_config(updates)


def get_exchange_config(exchange_name: str) -> Optional[ExchangeConfig]:
    """Get exchange configuration"""
    return config_manager.get_exchange_config(exchange_name)


def load_config_from_file(config_path: Union[str, Path]) -> bool:
    """Load configuration from file"""
    return config_manager.load_config(config_path)


def save_config_to_file(config_path: Union[str, Path], format: str = "json") -> bool:
    """Save configuration to file"""
    return config_manager.save_config(config_path, format)