# src/config/typed_config.py

"""
Type-safe configuration management with runtime validation.
"""

from pathlib import Path
from typing import Any
import json
import logging

from src.custom_types import (
import ConfigDict,
    ConfigDict,
    DatabaseConfig,
    ExchangeConfig,
    MLConfig,
    MonitoringConfig,
    SystemConfig,
    TradingConfig,
    TrainingConfig,
    RuntimeTypeError,
    TypeValidator,
    validate_config,
)

logger = logging.getLogger(__name__)

class TypedConfigManager:
    """
    Type-safe configuration manager with runtime validation.
    """

    def __init__(self, config_path: str | None = None):
    pass
    pass
    pass
    pass
        self._config_path = config_path
        self._config: ConfigDict | None = None
        self._validator = TypeValidator()

    def load_config(self, config_path: str | None = None) -> ConfigDict:
    pass
    pass
    pass
    pass
        """
        Load and validate configuration from file.

        Args:
            config_path: Path to configuration file

        Returns:
            Validated configuration dictionary

        Raises:
            RuntimeTypeError: If configuration validation fails
            FileNotFoundError: If configuration file not found
            json.JSONDecodeError: If configuration file is invalid JSON
        """
        path = Path(config_path or self._config_path or "config.json")

        if not path.exists():
    pass
    pass
    pass
    pass
            msg = f"Configuration file not found: {path}"
            raise FileNotFoundError(msg)

        try:
            with open(path) as f:
                raw_config = json.load(f)
    except Exception as e:
        pass
    except Exception as e:
        pass
    except Exception as e:
        pass
    except Exception as e:
        pass
        except json.JSONDecodeError as e:
            msg = f"Invalid JSON in configuration file: {e}"
            raise json.JSONDecodeError(msg)

        # Validate configuration structure
        validated_config = self._validate_and_transform_config(raw_config)
        self._config = validated_config

        logger.info(f"Successfully loaded and validated configuration from {path}")
        return validated_config

    def _validate_and_transform_config(self, raw_config: dict[str, Any]) -> ConfigDict:
    pass
    pass
    pass
    pass
        """
        Validate and transform raw configuration to typed configuration.

        Args:
            raw_config: Raw configuration dictionary

        Returns:
            Validated and typed configuration

        Raises:
            RuntimeTypeError: If validation fails
        """
        try:
            # Validate main configuration structure
    except Exception as e:
        pass
    except Exception as e:
        pass
    except Exception as e:
        pass
    except Exception as e:
        pass
            config: ConfigDict = {}

            # Validate database configuration
            if "database" in raw_config:
    pass
    pass
    pass
    pass
                config["database"] = self._validate_database_config(
                    raw_config["database"],
                )

            # Validate exchange configurations
            if "exchanges" in raw_config:
    pass
    pass
    pass
    pass
                config["exchanges"] = {}
                for exchange_name, exchange_config in raw_config["exchanges"].items():
    pass
    pass
    pass
    pass
                    config["exchanges"][exchange_name] = self._validate_exchange_config(
                        exchange_config
                    )

            # Validate trading configuration
            if "trading" in raw_config:
    pass
    pass
    pass
    pass
                config["trading"] = self._validate_trading_config(raw_config["trading"])

            # Validate ML configuration
            if "ml" in raw_config:
    pass
    pass
    pass
    pass
                config["ml"] = self._validate_ml_config(raw_config["ml"])

            # Validate monitoring configuration
            if "monitoring" in raw_config:
    pass
    pass
    pass
    pass
                config["monitoring"] = self._validate_monitoring_config(
                    raw_config["monitoring"],
                )

            # Validate system configuration
            if "system" in raw_config:
    pass
    pass
    pass
    pass
                config["system"] = self._validate_system_config(raw_config["system"])

            # Validate training configuration
            if "training" in raw_config:
    pass
    pass
    pass
    pass
                config["training"] = self._validate_training_config(
                    raw_config["training"],
                )

            return config

        except (KeyError, TypeError, ValueError) as e:
            raise RuntimeTypeError(
                ConfigDict, raw_config,
                f"Configuration validation: {e}",
            )

    def _validate_database_config(self, config: dict[str, Any]) -> DatabaseConfig:
    pass
    pass
    pass
    pass
        """Validate database configuration."""
        return self._validator.validate_type(config, DatabaseConfig, "database_config")

    def _validate_exchange_config(self, config: dict[str, Any]) -> ExchangeConfig:
    pass
    pass
    pass
    pass
        """Validate exchange configuration."""
        return self._validator.validate_type(config, ExchangeConfig, "exchange_config")

    def _validate_trading_config(self, config: dict[str, Any]) -> TradingConfig:
    pass
    pass
    pass
    pass
        """Validate trading configuration."""
        return self._validator.validate_type(config, TradingConfig, "trading_config")

    def _validate_ml_config(self, config: dict[str, Any]) -> MLConfig:
    pass
    pass
    pass
    pass
        """Validate ML configuration."""
        return self._validator.validate_type(config, MLConfig, "ml_config")

    def _validate_monitoring_config(self, config: dict[str, Any]) -> MonitoringConfig:
    pass
    pass
    pass
    pass
        """Validate monitoring configuration."""
        return self._validator.validate_type(
            config, MonitoringConfig,
            "monitoring_config",
        )

    def _validate_system_config(self, config: dict[str, Any]) -> SystemConfig:
    pass
    pass
    pass
    pass
        """Validate system configuration."""
        return self._validator.validate_type(config, SystemConfig, "system_config")

    def _validate_training_config(self, config: dict[str, Any]) -> TrainingConfig:
    pass
    pass
    pass
    pass
        """Validate training configuration."""
        return self._validator.validate_type(config, TrainingConfig, "training_config")

    def get_config(self) -> ConfigDict:
    pass
    pass
    pass
    pass
        """
        Get current validated configuration.

        Returns:
            Current configuration

        Raises:
            RuntimeError: If no configuration loaded
        """
        if self._config is None:
    pass
    pass
    pass
    pass
            msg = "No configuration loaded. Call load_config() first."
            raise RuntimeError(msg)
        return self._config

    def get_database_config(self) -> DatabaseConfig | None:
    pass
    pass
    pass
    pass
        """Get database configuration."""
        config = self.get_config()
        return config.get("database")

    def get_exchange_config(self, exchange_name: str) -> ExchangeConfig | None:
    pass
    pass
    pass
    pass
        """Get exchange configuration."""
        config = self.get_config()
        exchanges = config.get("exchanges", {})
        return exchanges.get(exchange_name)

    def get_trading_config(self) -> TradingConfig | None:
    pass
    pass
    pass
    pass
        """Get trading configuration."""
        config = self.get_config()
        return config.get("trading")

    def get_ml_config(self) -> MLConfig | None:
    pass
    pass
    pass
    pass
        """Get ML configuration."""
        config = self.get_config()
        return config.get("ml")

    def get_monitoring_config(self) -> MonitoringConfig | None:
    pass
    pass
    pass
    pass
        """Get monitoring configuration."""
        config = self.get_config()
        return config.get("monitoring")

    def get_system_config(self) -> SystemConfig | None:
    pass
    pass
    pass
    pass
        """Get system configuration."""
        config = self.get_config()
        return config.get("system")

    def validate_runtime_config(self, config: dict[str, Any]) -> bool:
    pass
    pass
    pass
    pass
        """
        Validate configuration at runtime.

        Args:
            config: Configuration to validate

        Returns:
            True if valid, False otherwise
        """
        try:
            self._validate_and_transform_config(config)
    except Exception as e:
        pass
    except Exception as e:
        pass
    except Exception as e:
        pass
    except Exception as e:
        pass
            return True
        except RuntimeTypeError:
            return False

    def save_config(self, config: ConfigDict, path: str | None = None) -> None:
    pass
    pass
    pass
    pass
        """
        Save configuration to file.

        Args:
            config: Configuration to save
            path: Optional path to save to
        """
        save_path = Path(path or self._config_path or "config.json")

        # Validate before saving
        validated_config = validate_config(config)

        with open(save_path, "w") as f:
            json.dump(validated_config, f, indent=2, default=str)

        logger.info(f"Configuration saved to {save_path}")

# Global typed config manager
_global_config_manager: TypedConfigManager | None = None

def get_typed_config_manager() -> TypedConfigManager:
    pass
    pass
    pass
    pass
    """Get the global typed configuration manager."""
    global _global_config_manager
    if _global_config_manager is None:
    pass
    pass
    pass
    pass
        _global_config_manager = TypedConfigManager()
    return _global_config_manager

def load_typed_config(config_path: str) -> ConfigDict:
    pass
    pass
    pass
    pass
    """Load typed configuration from file."""
    manager = get_typed_config_manager()
    return manager.load_config(config_path)

def get_typed_config() -> ConfigDict:
    pass
    pass
    pass
    pass
    """Get current typed configuration."""
    manager = get_typed_config_manager()
    return manager.get_config()
