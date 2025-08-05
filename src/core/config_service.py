# src/core/config_service.py

import asyncio
import json
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

# Try to import watchdog for file watching
try:
    from watchdog.events import FileSystemEventHandler
    from watchdog.observers import Observer

    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False
    Observer = None
    FileSystemEventHandler = None

from src.utils.error_handler import (
    handle_errors,
    handle_file_operations,
    handle_specific_errors,
)
from src.utils.logger import system_logger


@dataclass
class DatabaseConfig:
    """Database configuration dataclass."""

    database_path: str = "data/ares.db"
    auto_backup: bool = True
    backup_interval: int = 3600
    max_connections: int = 10
    enable_foreign_keys: bool = True
    journal_mode: str = "WAL"
    max_recovery_attempts: int = 3
    recovery_cooldown: int = 60


@dataclass
class ExchangeConfig:
    """Exchange configuration dataclass."""

    exchange_name: str = "BINANCE"
    api_key: str = ""
    api_secret: str = ""
    testnet: bool = True
    rate_limit: int = 1200
    timeout: int = 30
    retry_attempts: int = 3
    retry_delay: int = 1


@dataclass
class ModelTrainingConfig:
    """Model training configuration dataclass."""

    enable_advanced_training: bool = True
    enable_ensemble_training: bool = True
    enable_multi_timeframe_training: bool = True
    enable_adaptive_training: bool = True
    training_interval: int = 3600
    max_training_history: int = 100
    lookback_days: int = 730
    min_data_points: int = 100000


@dataclass
class RiskConfig:
    """Risk management configuration dataclass."""

    max_position_size: float = 0.1
    max_portfolio_risk: float = 0.02
    stop_loss_percentage: float = 0.05
    take_profit_percentage: float = 0.15
    max_drawdown: float = 0.20
    risk_free_rate: float = 0.02


if WATCHDOG_AVAILABLE:

    class ConfigurationWatcher(FileSystemEventHandler):
        """Watch for configuration file changes and reload automatically."""

        def __init__(self, config_service: "ConfigurationService"):
            self.config_service = config_service
            self.logger = system_logger.getChild("ConfigurationWatcher")

        def on_modified(self, event):
            """Handle file modification events."""
            if event.src_path.endswith((".yaml", ".yml", ".json")):
                self.logger.info(f"Configuration file changed: {event.src_path}")
                asyncio.create_task(self.config_service._reload_configuration())
else:

    class ConfigurationWatcher:
        """Dummy configuration watcher when watchdog is not available."""

        def __init__(self, config_service: "ConfigurationService"):
            self.config_service = config_service
            self.logger = system_logger.getChild("ConfigurationWatcher")

        def on_modified(self, event):
            """Handle file modification events."""
            # No-op when watchdog is not available


class ConfigurationService:
    """
    Enhanced Configuration Service with hot-reload, environment-specific configs,
    and dynamic configuration management.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config: dict[str, Any] = config
        self.logger = system_logger.getChild("ConfigurationService")

        # Configuration state
        self.is_initialized: bool = False
        self.config_data: dict[str, Any] = {}
        self.config_sections: dict[str, Any] = {}
        self.config_history: list[dict[str, Any]] = []
        self.max_history: int = 100

        # Environment-specific configuration
        self.environment: str = os.getenv("TRADING_ENV", "development")
        self.config_files: list[str] = []
        self.config_directories: list[str] = ["config"]

        # Hot-reload settings
        self.enable_hot_reload: bool = self.config.get("enable_hot_reload", True)
        self.watcher: Observer | None = None
        self.watched_files: set[str] = set()

        # Configuration validation
        self.validation_rules: dict[str, Any] = {}
        self.validation_errors: list[str] = []

        # Configuration encryption
        self.encryption_enabled: bool = self.config.get("encryption_enabled", False)
        self.encryption_key: str | None = None

        # Performance monitoring
        self.load_times: list[float] = []
        self.last_load_time: float = 0

    @handle_specific_errors(
        error_handlers={
            ValueError: (False, "Invalid configuration service setup"),
            AttributeError: (False, "Missing required configuration parameters"),
            KeyError: (False, "Missing configuration keys"),
        },
        default_return=False,
        context="configuration service initialization",
    )
    async def initialize(self) -> bool:
        """Initialize configuration service with enhanced capabilities."""
        try:
            self.logger.info("Initializing Configuration Service...")

            # Load configuration
            await self._load_configuration()

            # Validate configuration
            if not await self._validate_configuration():
                self.logger.error("Configuration validation failed")
                return False

            # Setup configuration sections
            await self._setup_configuration_sections()

            # Setup hot-reload if enabled
            if self.enable_hot_reload:
                await self._setup_hot_reload()

            # Setup encryption if enabled
            if self.encryption_enabled:
                await self._setup_encryption()

            self.is_initialized = True
            self.logger.info(
                "✅ Configuration Service initialization completed successfully",
            )
            return True

        except Exception as e:
            self.logger.error(f"❌ Configuration Service initialization failed: {e}")
            return False

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="configuration loading",
    )
    async def _load_configuration(self) -> None:
        """Load configuration from multiple sources."""
        try:
            start_time = time.time()

            # Determine environment-specific config files
            self.config_files = [
                "config/base.yaml",
                f"config/{self.environment}.yaml",
                f"config/{self.environment}_local.yaml",  # Optional local overrides
            ]

            # Load configuration from files
            for config_file in self.config_files:
                if os.path.exists(config_file):
                    await self._load_config_file(config_file)

            # Load from environment variables
            await self._load_from_environment()

            # Load from command line arguments
            await self._load_from_arguments()

            # Record load time
            load_time = time.time() - start_time
            self.load_times.append(load_time)
            self.last_load_time = load_time

            # Keep only recent load times
            if len(self.load_times) > 10:
                self.load_times = self.load_times[-10:]

            self.logger.info(f"Configuration loaded successfully in {load_time:.3f}s")

        except Exception as e:
            self.logger.error(f"Error loading configuration: {e}")

    @handle_file_operations(
        default_return=None,
        context="config file loading",
    )
    async def _load_config_file(self, config_file: str) -> None:
        """Load configuration from a specific file."""
        try:
            file_path = Path(config_file)

            if file_path.suffix.lower() in [".yaml", ".yml"]:
                with open(file_path) as f:
                    file_config = yaml.safe_load(f)
            elif file_path.suffix.lower() == ".json":
                with open(file_path) as f:
                    file_config = json.load(f)
            else:
                self.logger.warning(
                    f"Unsupported config file format: {file_path.suffix}",
                )
                return

            # Merge configuration
            self._merge_configuration(file_config)

            # Add to watched files for hot-reload
            if self.enable_hot_reload:
                self.watched_files.add(str(file_path))

            self.logger.info(f"Loaded configuration from: {config_file}")

        except Exception as e:
            self.logger.error(f"Error loading config file {config_file}: {e}")

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="environment variable loading",
    )
    async def _load_from_environment(self) -> None:
        """Load configuration from environment variables."""
        try:
            env_config = {}

            # Load database configuration
            env_config["database"] = {
                "database_path": os.getenv("DB_PATH", "data/ares.db"),
                "auto_backup": os.getenv("DB_AUTO_BACKUP", "true").lower() == "true",
                "backup_interval": int(os.getenv("DB_BACKUP_INTERVAL", "3600")),
                "max_connections": int(os.getenv("DB_MAX_CONNECTIONS", "10")),
            }

            # Load exchange configuration
            env_config["exchange"] = {
                "exchange_name": os.getenv("EXCHANGE_NAME", "binance"),
                "api_key": os.getenv("EXCHANGE_API_KEY", ""),
                "api_secret": os.getenv("EXCHANGE_API_SECRET", ""),
                "testnet": os.getenv("EXCHANGE_TESTNET", "true").lower() == "true",
                "rate_limit": int(os.getenv("EXCHANGE_RATE_LIMIT", "1200")),
            }

            # Load training configuration
            env_config["training"] = {
                "enable_advanced_training": os.getenv(
                    "ENABLE_ADVANCED_TRAINING",
                    "true",
                ).lower()
                == "true",
                "enable_ensemble_training": os.getenv(
                    "ENABLE_ENSEMBLE_TRAINING",
                    "true",
                ).lower()
                == "true",
                "training_interval": int(os.getenv("TRAINING_INTERVAL", "3600")),
                "lookback_days": int(os.getenv("TRAINING_LOOKBACK_DAYS", "730")),
            }

            # Load risk configuration
            env_config["risk"] = {
                "max_position_size": float(os.getenv("MAX_POSITION_SIZE", "0.1")),
                "max_portfolio_risk": float(os.getenv("MAX_PORTFOLIO_RISK", "0.02")),
                "stop_loss_percentage": float(
                    os.getenv("STOP_LOSS_PERCENTAGE", "0.05"),
                ),
                "take_profit_percentage": float(
                    os.getenv("TAKE_PROFIT_PERCENTAGE", "0.15"),
                ),
            }

            # Merge environment configuration
            self._merge_configuration(env_config)

            self.logger.info("Configuration loaded from environment variables")

        except Exception as e:
            self.logger.error(f"Error loading from environment: {e}")

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="command line arguments loading",
    )
    async def _load_from_arguments(self) -> None:
        """Load configuration from command line arguments."""
        try:
            # This would be implemented to parse command line arguments
            # For now, we'll use a mock implementation
            arg_config = {}

            # Add any command line specific overrides here
            # arg_config["debug"] = True  # Example

            if arg_config:
                self._merge_configuration(arg_config)
                self.logger.info("Configuration loaded from command line arguments")

        except Exception as e:
            self.logger.error(f"Error loading from arguments: {e}")

    def _merge_configuration(self, new_config: dict[str, Any]) -> None:
        """Merge new configuration with existing configuration."""
        try:

            def deep_merge(base: dict, update: dict) -> dict:
                """Deep merge two dictionaries."""
                for key, value in update.items():
                    if (
                        key in base
                        and isinstance(base[key], dict)
                        and isinstance(value, dict)
                    ):
                        deep_merge(base[key], value)
                    else:
                        base[key] = value
                return base

            self.config_data = deep_merge(self.config_data, new_config)

        except Exception as e:
            self.logger.error(f"Error merging configuration: {e}")

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="configuration validation",
    )
    async def _validate_configuration(self) -> bool:
        """Validate configuration using defined rules."""
        try:
            self.validation_errors.clear()

            # Validate database configuration
            if "database" in self.config_data:
                db_config = self.config_data["database"]
                if not db_config.get("database_path"):
                    self.validation_errors.append("Database path is required")
                if db_config.get("backup_interval", 0) <= 0:
                    self.validation_errors.append("Backup interval must be positive")
                if db_config.get("max_connections", 0) <= 0:
                    self.validation_errors.append("Max connections must be positive")

            # Validate exchange configuration
            if "exchange" in self.config_data:
                exchange_config = self.config_data["exchange"]
                if not exchange_config.get("exchange_name"):
                    self.validation_errors.append("Exchange name is required")
                if not exchange_config.get("api_key"):
                    self.validation_errors.append("Exchange API key is required")
                if not exchange_config.get("api_secret"):
                    self.validation_errors.append("Exchange API secret is required")

            # Validate training configuration
            if "training" in self.config_data:
                training_config = self.config_data["training"]
                if training_config.get("training_interval", 0) <= 0:
                    self.validation_errors.append("Training interval must be positive")
                if training_config.get("lookback_days", 0) <= 0:
                    self.validation_errors.append("Lookback days must be positive")

            # Validate risk configuration
            if "risk" in self.config_data:
                risk_config = self.config_data["risk"]
                if not (0 < risk_config.get("max_position_size", 0) <= 1):
                    self.validation_errors.append(
                        "Max position size must be between 0 and 1",
                    )
                if not (0 < risk_config.get("max_portfolio_risk", 0) <= 1):
                    self.validation_errors.append(
                        "Max portfolio risk must be between 0 and 1",
                    )

            if self.validation_errors:
                self.logger.error(
                    f"Configuration validation failed: {self.validation_errors}",
                )
                return False

            self.logger.info("Configuration validation successful")
            return True

        except Exception as e:
            self.logger.error(f"Error validating configuration: {e}")
            return False

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="configuration sections setup",
    )
    async def _setup_configuration_sections(self) -> None:
        """Setup typed configuration sections."""
        try:
            # Setup database configuration
            db_config_data = self.config_data.get("database", {})
            self.config_sections["database"] = DatabaseConfig(**db_config_data)

            # Setup exchange configuration
            exchange_config_data = self.config_data.get("exchange", {})
            self.config_sections["exchange"] = ExchangeConfig(**exchange_config_data)

            # Setup training configuration
            training_config_data = self.config_data.get("training", {})
            self.config_sections["training"] = ModelTrainingConfig(
                **training_config_data,
            )

            # Setup risk configuration
            risk_config_data = self.config_data.get("risk", {})
            self.config_sections["risk"] = RiskConfig(**risk_config_data)

            self.logger.info("Configuration sections setup complete")

        except Exception as e:
            self.logger.error(f"Error setting up configuration sections: {e}")

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="hot-reload setup",
    )
    async def _setup_hot_reload(self) -> None:
        """Setup hot-reload for configuration files."""
        try:
            if not WATCHDOG_AVAILABLE:
                self.logger.warning("Watchdog not available, hot-reload disabled")
                return

            self.watcher = Observer()

            # Watch config directories
            for config_dir in self.config_directories:
                if os.path.exists(config_dir):
                    self.watcher.schedule(
                        ConfigurationWatcher(self),
                        config_dir,
                        recursive=True,
                    )

            self.watcher.start()
            self.logger.info("Hot-reload setup complete")

        except Exception as e:
            self.logger.error(f"Error setting up hot-reload: {e}")

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="encryption setup",
    )
    async def _setup_encryption(self) -> None:
        """Setup configuration encryption."""
        try:
            # In a real implementation, you would setup encryption keys here
            self.encryption_key = os.getenv("CONFIG_ENCRYPTION_KEY")

            if not self.encryption_key:
                self.logger.warning("No encryption key provided, encryption disabled")
                self.encryption_enabled = False
            else:
                self.logger.info("Configuration encryption setup complete")

        except Exception as e:
            self.logger.error(f"Error setting up encryption: {e}")

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="configuration reload",
    )
    async def _reload_configuration(self) -> None:
        """Reload configuration from files."""
        try:
            self.logger.info("🔄 Reloading configuration...")

            # Store current configuration in history
            self.config_history.append(
                {
                    "timestamp": datetime.now().isoformat(),
                    "config": self.config_data.copy(),
                },
            )

            # Limit history size
            if len(self.config_history) > self.max_history:
                self.config_history.pop(0)

            # Clear current configuration
            self.config_data.clear()
            self.config_sections.clear()

            # Reload configuration
            await self._load_configuration()

            # Re-validate configuration
            if not await self._validate_configuration():
                self.logger.error("Configuration validation failed after reload")
                # Rollback to previous configuration
                if self.config_history:
                    previous_config = self.config_history[-1]["config"]
                    self.config_data = previous_config
                    await self._setup_configuration_sections()
                return

            # Re-setup configuration sections
            await self._setup_configuration_sections()

            self.logger.info("✅ Configuration reloaded successfully")

        except Exception as e:
            self.logger.error(f"Error reloading configuration: {e}")

    def get_config(self, section: str = None) -> Any:
        """Get configuration data."""
        try:
            if section:
                return self.config_sections.get(section)
            return self.config_data.copy()
        except Exception as e:
            self.logger.error(f"Error getting configuration: {e}")
            return None

    def get_database_config(self) -> DatabaseConfig:
        """Get database configuration."""
        return self.config_sections.get("database", DatabaseConfig())

    def get_exchange_config(self) -> ExchangeConfig:
        """Get exchange configuration."""
        return self.config_sections.get("exchange", ExchangeConfig())

    def get_training_config(self) -> ModelTrainingConfig:
        """Get training configuration."""
        return self.config_sections.get("training", ModelTrainingConfig())

    def get_risk_config(self) -> RiskConfig:
        """Get risk configuration."""
        return self.config_sections.get("risk", RiskConfig())

    def update_config(self, section: str, updates: dict[str, Any]) -> bool:
        """Update configuration dynamically."""
        try:
            if section not in self.config_sections:
                self.logger.error(f"Unknown configuration section: {section}")
                return False

            # Update the configuration section
            current_config = asdict(self.config_sections[section])
            current_config.update(updates)

            # Recreate the dataclass instance
            if section == "database":
                self.config_sections[section] = DatabaseConfig(**current_config)
            elif section == "exchange":
                self.config_sections[section] = ExchangeConfig(**current_config)
            elif section == "training":
                self.config_sections[section] = ModelTrainingConfig(**current_config)
            elif section == "risk":
                self.config_sections[section] = RiskConfig(**current_config)

            # Update the main config data
            self.config_data[section] = current_config

            self.logger.info(f"Configuration section '{section}' updated successfully")
            return True

        except Exception as e:
            self.logger.error(f"Error updating configuration: {e}")
            return False

    def get_config_status(self) -> dict[str, Any]:
        """Get configuration service status."""
        try:
            status = {
                "is_initialized": self.is_initialized,
                "environment": self.environment,
                "config_files": self.config_files,
                "enable_hot_reload": self.enable_hot_reload,
                "encryption_enabled": self.encryption_enabled,
                "validation_errors": self.validation_errors.copy(),
                "config_sections": list(self.config_sections.keys()),
                "history_count": len(self.config_history),
                "last_load_time": self.last_load_time,
                "average_load_time": sum(self.load_times) / len(self.load_times)
                if self.load_times
                else 0,
                "watched_files_count": len(self.watched_files),
            }

            return status

        except Exception as e:
            self.logger.error(f"Error getting configuration status: {e}")
            return {}

    def get_config_history(self, limit: int = None) -> list[dict[str, Any]]:
        """Get configuration history."""
        try:
            history = self.config_history.copy()
            if limit:
                history = history[-limit:]
            return history
        except Exception as e:
            self.logger.error(f"Error getting configuration history: {e}")
            return []

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="configuration service cleanup",
    )
    async def stop(self) -> None:
        """Stop the configuration service."""
        self.logger.info("🛑 Stopping Configuration Service...")

        try:
            # Stop hot-reload watcher
            if self.watcher:
                self.watcher.stop()
                self.watcher.join()
                self.watcher = None

            # Clear configuration data
            self.config_data.clear()
            self.config_sections.clear()
            self.config_history.clear()

            self.is_initialized = False
            self.logger.info("✅ Configuration Service stopped successfully")

        except Exception as e:
            self.logger.error(f"Error stopping configuration service: {e}")


# Global configuration service instance
config_service: ConfigurationService | None = None


@handle_errors(
    exceptions=(Exception,),
    default_return=None,
    context="configuration service setup",
)
async def setup_configuration_service(
    config: dict[str, Any] | None = None,
) -> ConfigurationService | None:
    """
    Setup global configuration service.

    Args:
        config: Optional configuration dictionary

    Returns:
        Optional[ConfigurationService]: Global configuration service instance
    """
    try:
        global config_service

        if config is None:
            config = {
                "enable_hot_reload": True,
                "encryption_enabled": False,
                "max_history": 100,
            }

        # Create configuration service
        config_service = ConfigurationService(config)

        # Initialize configuration service
        success = await config_service.initialize()
        if success:
            return config_service
        return None

    except Exception as e:
        print(f"Error setting up configuration service: {e}")
        return None
