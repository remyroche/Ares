"""
Configuration Management System

Comprehensive configuration management for multi-exchange trading.
Supports loading from multiple sources, validation, and runtime updates.
"""

import json
import os
import yaml
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Union, Set
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import asyncio
from contextlib import asynccontextmanager

from .config import TradingConfig, TradingMode, OrderType, OrderSide
from .error_handler import ErrorHandler, ErrorCategory, ErrorSeverity


class ConfigSource(Enum):
    """Configuration source types"""
    ENVIRONMENT = "environment"
    JSON_FILE = "json_file"
    YAML_FILE = "yaml_file"
    DATABASE = "database"
    API = "api"
    DEFAULT = "default"


class ConfigValidationLevel(Enum):
    """Configuration validation levels"""
    NONE = "none"
    BASIC = "basic"
    STRICT = "strict"
    COMPREHENSIVE = "comprehensive"


@dataclass
class ExchangeConfig:
    """Configuration for a single exchange"""
    name: str
    api_key: str
    api_secret: str
    password: Optional[str] = None
    sandbox: bool = False
    rate_limit: int = 1200
    timeout: int = 30
    enabled: bool = True
    symbols: List[str] = field(default_factory=list)
    risk_limits: Dict[str, Any] = field(default_factory=dict)
    custom_settings: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MultiExchangeConfig:
    """Configuration for multiple exchanges"""
    primary_exchange: str
    exchanges: Dict[str, ExchangeConfig] = field(default_factory=dict)
    default_symbols: List[str] = field(default_factory=lambda: ["BTCUSDT"])
    enable_failover: bool = True
    failover_exchanges: List[str] = field(default_factory=list)
    load_balancing: bool = False
    load_balancing_strategy: str = "round_robin"  # round_robin, least_loaded, weighted


@dataclass
class SystemConfig:
    """System-wide configuration"""
    trading_config: TradingConfig
    exchange_config: MultiExchangeConfig
    logging_config: Dict[str, Any] = field(default_factory=dict)
    database_config: Dict[str, Any] = field(default_factory=dict)
    monitoring_config: Dict[str, Any] = field(default_factory=dict)
    security_config: Dict[str, Any] = field(default_factory=dict)
    custom_config: Dict[str, Any] = field(default_factory=dict)


class ConfigurationManager:
    """Centralized configuration management system"""

    def __init__(self, validation_level: ConfigValidationLevel = ConfigValidationLevel.STRICT):
        self.validation_level = validation_level
        self.logger = logging.getLogger(__name__)
        self.error_handler = ErrorHandler()

        # Configuration storage
        self._current_config: Optional[SystemConfig] = None
        self._config_history: List[Dict[str, Any]] = []
        self._config_sources: Dict[str, Any] = {}

        # Configuration file paths
        self._config_paths = {
            ConfigSource.JSON_FILE: "config/trading_config.json",
            ConfigSource.YAML_FILE: "config/trading_config.yaml",
            ConfigSource.ENVIRONMENT: "env",
            ConfigSource.DEFAULT: "default"
        }

        # Validation rules
        self._validation_rules = {
            "required_fields": [
                "trading_config.mode",
                "exchange_config.primary_exchange",
                "exchange_config.exchanges"
            ],
            "exchange_fields": [
                "name", "api_key", "api_secret"
            ],
            "valid_modes": [mode.value for mode in TradingMode],
            "valid_order_types": [order_type.value for order_type in OrderType],
            "valid_order_sides": [order_side.value for order_side in OrderSide]
        }

        # Change listeners
        self._change_listeners: List[Callable[[Dict[str, Any]], Awaitable[None]]] = []

    async def load_configuration(self, sources: List[ConfigSource]) -> SystemConfig:
        """Load configuration from multiple sources with priority order"""
        config_data = {}

        # Load in priority order (later sources override earlier ones)
        for source in sources:
            try:
                source_data = await self._load_from_source(source)
                if source_data:
                    config_data.update(source_data)
                    self._config_sources[source.value] = source_data
                    self.logger.info(f"Loaded configuration from {source.value}")
            except Exception as e:
                self.logger.warning(f"Failed to load from {source.value}: {e}")

        # Apply defaults for missing values
        config_data = self._apply_defaults(config_data)

        # Validate configuration
        await self._validate_configuration(config_data)

        # Create system config object
        system_config = self._create_system_config(config_data)

        # Store configuration
        self._current_config = system_config
        self._config_history.append({
            "config": config_data,
            "timestamp": datetime.now().isoformat(),
            "sources": [source.value for source in sources]
        })

        # Limit history size
        if len(self._config_history) > 50:
            self._config_history = self._config_history[-25:]

        # Notify listeners
        await self._notify_change_listeners(config_data)

        self.logger.info("Configuration loaded successfully")
        return system_config

    async def _load_from_source(self, source: ConfigSource) -> Dict[str, Any]:
        """Load configuration from a specific source"""
        if source == ConfigSource.ENVIRONMENT:
            return self._load_from_environment()
        elif source == ConfigSource.JSON_FILE:
            return await self._load_from_json_file()
        elif source == ConfigSource.YAML_FILE:
            return await self._load_from_yaml_file()
        elif source == ConfigSource.DEFAULT:
            return self._get_default_config()
        else:
            raise ValueError(f"Unsupported configuration source: {source}")

    def _load_from_environment(self) -> Dict[str, Any]:
        """Load configuration from environment variables"""
        config = {}

        # Trading configuration
        trading_env = {
            "mode": os.getenv("TRADING_MODE", "paper"),
            "exchange_name": os.getenv("EXCHANGE_NAME", "binance"),
            "symbols": os.getenv("TRADING_SYMBOLS", "BTCUSDT").split(","),
            "max_position_size": float(os.getenv("MAX_POSITION_SIZE", "1000.0")),
            "max_daily_loss": float(os.getenv("MAX_DAILY_LOSS", "100.0")),
            "max_leverage": float(os.getenv("MAX_LEVERAGE", "10.0")),
        }
        config["trading_config"] = trading_env

        # Exchange configuration
        exchanges = {}
        exchange_names = os.getenv("EXCHANGE_NAMES", "binance").split(",")

        for exchange_name in exchange_names:
            exchange_config = {
                "name": exchange_name,
                "api_key": os.getenv(f"{exchange_name.upper()}_API_KEY", ""),
                "api_secret": os.getenv(f"{exchange_name.upper()}_API_SECRET", ""),
                "password": os.getenv(f"{exchange_name.upper()}_PASSWORD"),
                "sandbox": os.getenv(f"{exchange_name.upper()}_SANDBOX", "false").lower() == "true",
                "rate_limit": int(os.getenv(f"{exchange_name.upper()}_RATE_LIMIT", "1200")),
                "timeout": int(os.getenv(f"{exchange_name.upper()}_TIMEOUT", "30")),
                "enabled": os.getenv(f"{exchange_name.upper()}_ENABLED", "true").lower() == "true"
            }
            exchanges[exchange_name] = exchange_config

        config["exchange_config"] = {
            "primary_exchange": os.getenv("PRIMARY_EXCHANGE", "binance"),
            "exchanges": exchanges,
            "enable_failover": os.getenv("ENABLE_FAILOVER", "true").lower() == "true",
            "failover_exchanges": os.getenv("FAILOVER_EXCHANGES", "").split(",") if os.getenv("FAILOVER_EXCHANGES") else [],
            "load_balancing": os.getenv("LOAD_BALANCING", "false").lower() == "true",
            "load_balancing_strategy": os.getenv("LOAD_BALANCING_STRATEGY", "round_robin")
        }

        return config

    async def _load_from_json_file(self) -> Dict[str, Any]:
        """Load configuration from JSON file"""
        config_path = self._config_paths[ConfigSource.JSON_FILE]

        try:
            with open(config_path, 'r', encoding='utf-8') as file:
                return json.load(file)
        except FileNotFoundError:
            self.logger.warning(f"Configuration file not found: {config_path}")
            return {}
        except Exception as e:
            self.logger.error(f"Error loading JSON configuration: {e}")
            return {}

    async def _load_from_yaml_file(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        config_path = self._config_paths[ConfigSource.YAML_FILE]

        try:
            with open(config_path, 'r', encoding='utf-8') as file:
                return yaml.safe_load(file) or {}
        except FileNotFoundError:
            self.logger.warning(f"Configuration file not found: {config_path}")
            return {}
        except Exception as e:
            self.logger.error(f"Error loading YAML configuration: {e}")
            return {}

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            "trading_config": {
                "mode": TradingMode.PAPER.value,
                "exchange_name": "binance",
                "symbols": ["BTCUSDT"],
                "max_position_size": 1000.0,
                "max_daily_loss": 100.0,
                "max_leverage": 10.0,
                "stop_loss_percentage": 2.0,
                "take_profit_percentage": 4.0,
                "order_timeout": 30,
                "max_retries": 3,
                "retry_delay": 1.0,
                "data_update_interval": 1.0,
                "reconnect_attempts": 5,
                "reconnect_delay": 5.0,
                "performance_log_interval": 60,
                "trade_log_enabled": True,
                "metrics_enabled": True,
                "api_rate_limit": 1200,
                "api_timeout": 30
            },
            "exchange_config": {
                "primary_exchange": "binance",
                "exchanges": {
                    "binance": {
                        "name": "binance",
                        "api_key": "",
                        "api_secret": "",
                        "sandbox": False,
                        "rate_limit": 1200,
                        "timeout": 30,
                        "enabled": True,
                        "symbols": ["BTCUSDT", "ETHUSDT"],
                        "risk_limits": {},
                        "custom_settings": {}
                    }
                },
                "enable_failover": False,
                "failover_exchanges": [],
                "load_balancing": False,
                "load_balancing_strategy": "round_robin"
            }
        }

    def _apply_defaults(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply default values for missing configuration items"""
        defaults = self._get_default_config()

        def apply_recursive_defaults(target: Dict[str, Any], defaults: Dict[str, Any]) -> None:
            for key, value in defaults.items():
                if key not in target:
                    target[key] = value
                elif isinstance(value, dict) and isinstance(target[key], dict):
                    apply_recursive_defaults(target[key], value)

        apply_recursive_defaults(config, defaults)
        return config

    async def _validate_configuration(self, config: Dict[str, Any]) -> None:
        """Validate configuration based on validation level"""
        if self.validation_level == ConfigValidationLevel.NONE:
            return

        validation_errors = []

        # Basic validation
        for field in self._validation_rules["required_fields"]:
            if not self._get_nested_value(config, field):
                validation_errors.append(f"Missing required field: {field}")

        # Exchange validation
        exchanges = config.get("exchange_config", {}).get("exchanges", {})
        for exchange_name, exchange_config in exchanges.items():
            if isinstance(exchange_config, dict):
                for field in self._validation_rules["exchange_fields"]:
                    if field not in exchange_config:
                        validation_errors.append(f"Missing field '{field}' in exchange '{exchange_name}'")

        # Mode validation
        mode = config.get("trading_config", {}).get("mode")
        if mode and mode not in self._validation_rules["valid_modes"]:
            validation_errors.append(f"Invalid trading mode: {mode}")

        if validation_errors:
            if self.validation_level in [ConfigValidationLevel.STRICT, ConfigValidationLevel.COMPREHENSIVE]:
                raise ValueError(f"Configuration validation failed: {'; '.join(validation_errors)}")
            else:
                for error in validation_errors:
                    self.logger.warning(f"Configuration warning: {error}")

        # Comprehensive validation (if enabled)
        if self.validation_level == ConfigValidationLevel.COMPREHENSIVE:
            await self._comprehensive_validation(config)

    async def _comprehensive_validation(self, config: Dict[str, Any]) -> None:
        """Perform comprehensive configuration validation"""
        # Validate API keys format
        exchanges = config.get("exchange_config", {}).get("exchanges", {})
        for exchange_name, exchange_config in exchanges.items():
            if isinstance(exchange_config, dict):
                api_key = exchange_config.get("api_key", "")
                if api_key and not self._validate_api_key_format(api_key):
                    self.logger.warning(f"API key format may be invalid for exchange {exchange_name}")

        # Validate symbol formats
        symbols = config.get("trading_config", {}).get("symbols", [])
        for symbol in symbols:
            if not self._validate_symbol_format(symbol):
                self.logger.warning(f"Symbol format may be invalid: {symbol}")

    def _validate_api_key_format(self, api_key: str) -> bool:
        """Validate API key format (basic validation)"""
        if not api_key or len(api_key) < 10:
            return False
        return True  # Could add more sophisticated validation

    def _validate_symbol_format(self, symbol: str) -> bool:
        """Validate trading symbol format"""
        if not symbol or len(symbol) < 5:
            return False
        # Basic format check (should contain base and quote currencies)
        if not any(char.isdigit() for char in symbol) and not any(char.isalpha() for char in symbol):
            return False
        return True

    def _get_nested_value(self, config: Dict[str, Any], path: str) -> Any:
        """Get nested value from configuration using dot notation"""
        keys = path.split('.')
        current = config

        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None

        return current

    def _create_system_config(self, config_data: Dict[str, Any]) -> SystemConfig:
        """Create SystemConfig object from configuration data"""
        # Create trading config
        trading_data = config_data.get("trading_config", {})
        trading_config = TradingConfig.from_dict(trading_data)

        # Create exchange config
        exchange_data = config_data.get("exchange_config", {})
        exchanges = {}

        for name, ex_config in exchange_data.get("exchanges", {}).items():
            if isinstance(ex_config, dict):
                exchanges[name] = ExchangeConfig(
                    name=ex_config.get("name", name),
                    api_key=ex_config.get("api_key", ""),
                    api_secret=ex_config.get("api_secret", ""),
                    password=ex_config.get("password"),
                    sandbox=ex_config.get("sandbox", False),
                    rate_limit=ex_config.get("rate_limit", 1200),
                    timeout=ex_config.get("timeout", 30),
                    enabled=ex_config.get("enabled", True),
                    symbols=ex_config.get("symbols", []),
                    risk_limits=ex_config.get("risk_limits", {}),
                    custom_settings=ex_config.get("custom_settings", {})
                )

        exchange_config = MultiExchangeConfig(
            primary_exchange=exchange_data.get("primary_exchange", "binance"),
            exchanges=exchanges,
            default_symbols=exchange_data.get("default_symbols", ["BTCUSDT"]),
            enable_failover=exchange_data.get("enable_failover", True),
            failover_exchanges=exchange_data.get("failover_exchanges", []),
            load_balancing=exchange_data.get("load_balancing", False),
            load_balancing_strategy=exchange_data.get("load_balancing_strategy", "round_robin")
        )

        return SystemConfig(
            trading_config=trading_config,
            exchange_config=exchange_config,
            logging_config=config_data.get("logging_config", {}),
            database_config=config_data.get("database_config", {}),
            monitoring_config=config_data.get("monitoring_config", {}),
            security_config=config_data.get("security_config", {}),
            custom_config=config_data.get("custom_config", {})
        )

    async def update_configuration(self, updates: Dict[str, Any]) -> SystemConfig:
        """Update configuration with changes"""
        if not self._current_config:
            raise RuntimeError("No current configuration to update")

        # Apply updates to current config data
        current_data = self._get_current_config_data()
        self._apply_updates_recursive(current_data, updates)

        # Re-validate and recreate config
        await self._validate_configuration(current_data)
        new_config = self._create_system_config(current_data)

        # Update stored config
        self._current_config = new_config
        self._config_history.append({
            "config": current_data,
            "timestamp": datetime.now().isoformat(),
            "update": True
        })

        # Notify listeners
        await self._notify_change_listeners(updates)

        self.logger.info("Configuration updated successfully")
        return new_config

    def _apply_updates_recursive(self, target: Dict[str, Any], updates: Dict[str, Any]) -> None:
        """Apply updates recursively to configuration"""
        for key, value in updates.items():
            if isinstance(value, dict) and key in target and isinstance(target[key], dict):
                self._apply_updates_recursive(target[key], value)
            else:
                target[key] = value

    def _get_current_config_data(self) -> Dict[str, Any]:
        """Get current configuration as dictionary"""
        if not self._current_config:
            return {}

        return {
            "trading_config": self._current_config.trading_config.to_dict(),
            "exchange_config": {
                "primary_exchange": self._current_config.exchange_config.primary_exchange,
                "exchanges": {name: {
                    "name": ex.name,
                    "api_key": ex.api_key,
                    "api_secret": ex.api_secret,
                    "password": ex.password,
                    "sandbox": ex.sandbox,
                    "rate_limit": ex.rate_limit,
                    "timeout": ex.timeout,
                    "enabled": ex.enabled,
                    "symbols": ex.symbols,
                    "risk_limits": ex.risk_limits,
                    "custom_settings": ex.custom_settings
                } for name, ex in self._current_config.exchange_config.exchanges.items()},
                "default_symbols": self._current_config.exchange_config.default_symbols,
                "enable_failover": self._current_config.exchange_config.enable_failover,
                "failover_exchanges": self._current_config.exchange_config.failover_exchanges,
                "load_balancing": self._current_config.exchange_config.load_balancing,
                "load_balancing_strategy": self._current_config.exchange_config.load_balancing_strategy
            }
        }

    async def _notify_change_listeners(self, changes: Dict[str, Any]) -> None:
        """Notify configuration change listeners"""
        for listener in self._change_listeners:
            try:
                await listener(changes)
            except Exception as e:
                self.logger.error(f"Error in configuration change listener: {e}")

    def register_change_listener(self, listener: Callable[[Dict[str, Any]], Awaitable[None]]) -> None:
        """Register a configuration change listener"""
        self._change_listeners.append(listener)
        self.logger.info(f"Registered configuration change listener: {listener.__name__}")

    async def save_configuration(self, file_path: str, format: str = "json") -> None:
        """Save current configuration to file"""
        if not self._current_config:
            raise RuntimeError("No current configuration to save")

        config_data = self._get_current_config_data()

        try:
            if format.lower() == "json":
                with open(file_path, 'w', encoding='utf-8') as file:
                    json.dump(config_data, file, indent=2, ensure_ascii=False)
            elif format.lower() == "yaml":
                with open(file_path, 'w', encoding='utf-8') as file:
                    yaml.dump(config_data, file, default_flow_style=False, allow_unicode=True)
            else:
                raise ValueError(f"Unsupported format: {format}")

            self.logger.info(f"Configuration saved to {file_path}")

        except Exception as e:
            self.logger.error(f"Error saving configuration: {e}")
            raise

    async def get_configuration_status(self) -> Dict[str, Any]:
        """Get configuration status and metadata"""
        return {
            "loaded": self._current_config is not None,
            "validation_level": self.validation_level.value,
            "sources": list(self._config_sources.keys()),
            "history_count": len(self._config_history),
            "last_updated": self._config_history[-1]["timestamp"] if self._config_history else None,
            "exchange_count": len(self._current_config.exchange_config.exchanges) if self._current_config else 0,
            "enabled_exchanges": [
                name for name, ex in self._current_config.exchange_config.exchanges.items()
                if ex.enabled
            ] if self._current_config else [],
            "primary_exchange": self._current_config.exchange_config.primary_exchange if self._current_config else None
        }


# Factory function to create configuration manager with common settings
async def create_config_manager(
    validation_level: ConfigValidationLevel = ConfigValidationLevel.STRICT
) -> ConfigurationManager:
    """Create a configuration manager with common settings"""
    manager = ConfigurationManager(validation_level)

    # Load from common sources in priority order
    sources = [
        ConfigSource.ENVIRONMENT,  # Environment variables (highest priority)
        ConfigSource.JSON_FILE,    # Configuration file
        ConfigSource.YAML_FILE,    # YAML configuration file
        ConfigSource.DEFAULT       # Default values (lowest priority)
    ]

    await manager.load_configuration(sources)
    return manager


# Example usage
async def example_config_manager():
    """Example of how to use the configuration manager"""
    try:
        # Create configuration manager
        config_manager = await create_config_manager()

        # Get current configuration
        status = await config_manager.get_configuration_status()
        print(f"Configuration status: {status}")

        # Update configuration
        updates = {
            "trading_config": {
                "max_position_size": 2000.0,
                "symbols": ["BTCUSDT", "ETHUSDT", "ADAUSDT"]
            },
            "exchange_config": {
                "primary_exchange": "binance",
                "enable_failover": True
            }
        }

        await config_manager.update_configuration(updates)

        # Save configuration
        await config_manager.save_configuration("updated_config.json")

    except Exception as e:
        print(f"Error in configuration manager example: {e}")


if __name__ == "__main__":
    asyncio.run(example_config_manager())