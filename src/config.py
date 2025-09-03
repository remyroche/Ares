# src/config.py
"""Legacy configuration module for backward compatibility.

This module now uses the new modular configuration structure.
"""

# Import the new modular configuration
import asyncio
from dataclasses import dataclass
from typing import Any

from src.config.environment import get_env_settings, get_environment_settings
from src.config.modular_config import (
    CONFIG,
    AresConfig,
    get_complete_config,
    get_dual_model_config,
    get_enhanced_training_config,
    get_environment_config,
    get_leverage_sizing_config,
    get_lookback_window,
    get_ml_confidence_predictor_config,
    get_position_closing_config,
    get_position_division_config,
    get_position_monitoring_config,
    get_position_sizing_config,
    get_system_config_section,
    get_trading_config_section,
    get_training_config_section,
)
from src.utils.logger import system_logger

# Re-export all the functions and classes for backward compatibility
__all__ = [
    "get_complete_config",
    "get_environment_config",
    "get_system_config_section",
    "get_trading_config_section",
    "get_training_config_section",
    "get_lookback_window",
    "AresConfig",
    "CONFIG",
    "get_dual_model_config",
    "get_ml_confidence_predictor_config",
    "get_position_sizing_config",
    "get_leverage_sizing_config",
    "get_position_closing_config",
    "get_position_division_config",
    "get_position_monitoring_config",
    "get_enhanced_training_config",
]

# Legacy compatibility - maintain the old CONFIG structure


def get_config() -> dict[str, Any]:
    """Get the complete configuration (legacy function).

    Returns:
        dict: Complete configuration dictionary
    """
    return get_complete_config()


def get_environment_settings():
    """Get environment settings (legacy function).

    Returns:
        EnvironmentSettings: Environment settings instance
    """
    return get_env_settings()


# Legacy dataclass definitions for backward compatibility


@dataclass
class DatabaseConfig:
    """Database configuration settings."""

    host: str = "localhost"
    port: int = 5432
    database: str = "ares_trading"
    username: str = "postgres"
    password: str = ""
    max_connections: int = 10
    connection_timeout: int = 30


@dataclass
class ExchangeConfig:
    """Exchange configuration settings."""

    name: str = "binance"
    api_key: str = ""
    api_secret: str = ""
    testnet: bool = True
    rate_limit: int = 1200
    timeout: int = 30


@dataclass
class ModelTrainingConfig:
    """Model training configuration settings."""

    lookback_days: int = 180  # Exactly 6 months for consistent data range
    training_split: float = 0.8
    validation_split: float = 0.1
    test_split: float = 0.1
    batch_size: int = 64
    epochs: int = 100
    learning_rate: float = 0.001

    # Enhanced optimization settings
    enhanced_lm_optimizer: dict[str, Any] = None

    def __post_init__(self):
        if self.enhanced_lm_optimizer is None:
            self.enhanced_lm_optimizer = {
                "feature_selection": {
                    "enable": True,
                    "methods": ["mutual_info", "lasso", "random_forest", "shap"],
                    "target_features": {"step06": 80, "step6_5": 100, "step09": 90},
                    "vif_threshold": 10.0,
                    "correlation_threshold": 0.95,
                    "variance_threshold": 0.01,
                    "mutual_info_threshold": 0.001,
                    "shap_threshold": 0.001,
                },
                "regularization": {
                    "enable": True,
                    "l1_alpha_range": [0.001, 0.1],
                    "l2_alpha_range": [0.0001, 0.01],
                    "dropout_range": [0.1, 0.5],
                    "model_specific": {
                        "lightgbm": {
                            "reg_alpha_range": [0.001, 0.1],
                            "reg_lambda_range": [0.0001, 0.01],
                        },
                        "neural_networks": {
                            "weight_decay_range": [1e-6, 1e-3],
                            "dropout_range": [0.1, 0.5],
                        },
                    },
                },
                "optuna": {
                    "enable": True,
                    "n_trials_per_batch": 50,
                    "n_batches": 3,
                    "timeout_per_batch": 300,  # 5 minutes per batch
                    "sampler": "tpe",
                    "pruner": "median",
                    "storage": None,  # Can be set to database URL
                },
                "vectorization": {
                    "enable": True,
                    "batch_size": 1024,
                    "use_gpu": True,
                    "memory_efficient": True,
                },
            }


@dataclass
class RiskConfig:
    """Risk management configuration settings."""

    max_position_size: float = 0.1
    max_drawdown: float = 0.15
    stop_loss_pct: float = 0.05
    take_profit_pct: float = 0.1
    max_leverage: int = 10

# Legacy ConfigurationManager class for backward compatibility
from src.core.decorators import handles_errors
from src.utils.warning_symbols import invalid, warning, failed
import copy

# Legacy ConfigurationManager class for backward compatibility
from src.utils.error_handler import handle_errors, handle_specific_errors
from src.utils.warning_symbols import failed, invalid, warning


class ConfigurationManager:
    """Legacy configuration manager for backward compatibility.

    This class now uses the new modular configuration structure.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize configuration manager.

        Args:
            config: Configuration dictionary
        """
        self.config: dict[str, Any] = config
        self.logger = system_logger.getChild("ConfigurationManager")

        # Configuration manager state
        self.is_initialized: bool = False
        self.config_history: list[dict[str, Any]] = []
        self.config_sections: dict[str, Any] = {}

        # Configuration
        self.config_manager_config: dict[str, Any] = self.config.get(
            "config_manager",
            {},
        )
        self.max_config_history: int = self.config_manager_config.get(
            "max_config_history",
            100,
        )

    @handles_errors(
        error_handlers={
            ValueError: (False, "Invalid configuration manager configuration"),
            AttributeError: (
                False,
                "Missing required configuration manager parameters",
            ),
            KeyError: (False, "Missing configuration keys"),
        },
        default_return=False,
        context="configuration manager initialization",
    )
    async def initialize(self) -> bool:
        """Initialize configuration manager.

        Returns:
            bool: True if initialization successful=False otherwise
        """
        try:
            self.logger.info("Initializing Configuration Manager...")

            # Load configuration manager configuration
            await self._load_config_manager_configuration()

            # Validate configuration
            if not self._validate_configuration():
                self.print(invalid("Invalid configuration for configuration manager"))
                return False

            # Initialize configuration sections
            await self._initialize_config_sections()

            # Initialize configuration service
            await self._initialize_config_service()

            self.is_initialized = True
            self.logger.info("✅ Configuration Manager initialized successfully")
            return True

        except (ValueError, KeyError) as e:
            self.logger.exception(
                f"❌ Configuration Manager initialization failed - Invalid configuration: {e}",
            )
            return False
        except OSError as e:
            self.logger.exception(
                f"❌ Configuration Manager initialization failed - File system error: {e}",
            )
            return False
        except Exception as e:
            self.logger.exception(
                f"❌ Configuration Manager initialization failed - Unexpected error: {e}",
            )
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="config manager configuration loading",
    )
    async def _load_config_manager_configuration(self) -> None:
        """Load configuration manager specific configuration."""
        try:
            # Configuration manager specific settings are already loaded
            self.logger.info("✅ Configuration manager configuration loaded")

        except (ValueError, KeyError) as e:
            self.logger.exception(
                f"❌ Failed to load configuration manager configuration - Invalid config: {e}",
            )
            raise
        except Exception as e:
            self.logger.exception(
                f"❌ Failed to load configuration manager configuration - Unexpected error: {e}",
            )
            raise

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=False,
        context="configuration validation",
    )
    def _validate_configuration(self) -> bool:
        """Validate configuration manager configuration.

        Returns:
            bool: True if configuration is valid=False otherwise
        """
        try:
            # Validate configuration manager specific settings
            if self.max_config_history <= 0:
                self.print(invalid("Invalid max_config_history configuration"))
                return False

            return True

        except (ValueError, TypeError) as e:
            self.print(failed(f"Configuration validation failed - Invalid value: {e}"))
            return False
        except Exception as e:
            self.print(
                failed(f"Configuration validation failed - Unexpected error: {e}"),
            )
            return False

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="config sections initialization",
    )
    async def _initialize_config_sections(self) -> None:
        """Initialize configuration sections."""
        try:
            # Initialize all configuration sections
            self.config_sections = {
                "environment": get_environment_config(),
                "system": get_system_config_section(),
                "trading": get_trading_config_section(),
                "training": get_training_config_section(),
            }

            self.logger.info("✅ All configuration sections initialized")

        except Exception as e:
            self.logger.exception(
                f"❌ Failed to initialize configuration sections: {e}",
            )
            raise

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="config service initialization",
    )
    async def _initialize_config_service(self) -> None:
        """Initialize configuration service."""
        try:
            # Configuration service is handled by the new modular structure
            self.logger.info("✅ Configuration service initialized")

        except Exception:
            self.print(failed("❌ Failed to initialize configuration service: {e}"))
            raise

    @handles_errors(
        error_handlers={
            Exception: (False, "Configuration manager run failed"),
        },
        default_return=False,
        context="configuration manager run",
    )
    async def run(self) -> bool:
        """Run the configuration manager.

        Returns:
            bool: True if successful=False otherwise
        """
        try:
            self.logger.info("🚀 Starting Configuration Manager...")

            # Update configuration
            await self._update_configuration()

            # Validate configuration sections
            await self._validate_configuration_sections()

            # Update configuration service
            await self._update_config_service()

            self.logger.info("✅ Configuration Manager run completed successfully")
            return True

        except Exception:
            self.print(failed("❌ Configuration Manager run failed: {e}"))
            return False

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="configuration update",
    )
    async def _update_configuration(self) -> None:
        """Update configuration."""
        try:
            # Add to configuration history
            history_entry = {
                "timestamp": "2024-01-01T00:00:00",  # Placeholder timestamp
                "config_sections": self.config_sections.copy(),
            }

            self.config_history.append(history_entry)

            # Limit history size
            if len(self.config_history) > self.max_config_history:
                self.config_history = self.config_history[-self.max_config_history :]

            self.logger.info(
                f"📁 Updated configuration (history: {len(self.config_history)} entries)",
            )

        except Exception:
            self.print(failed("❌ Failed to update configuration: {e}"))

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="configuration reload",
    )
    async def _reload_configuration(self) -> None:
        """Reload configuration."""
        try:
            # Reinitialize configuration sections
            await self._initialize_config_sections()

            self.logger.info("✅ Configuration reloaded successfully")

        except Exception:
            self.print(failed("❌ Failed to reload configuration: {e}"))

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="configuration sections validation",
    )
    async def _validate_configuration_sections(self) -> None:
        """Validate configuration sections."""
        try:
            # Validate each configuration section
            for section_name, section_config in self.config_sections.items():
                if not section_config:
                    self.print(warning("Empty configuration section: {section_name}"))
                else:
                    self.logger.info(
                        f"✅ Validated configuration section: {section_name}",
                    )

            self.logger.info("✅ All configuration sections validated")

        except Exception:
            self.print(failed("❌ Failed to validate configuration sections: {e}"))

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="config service update",
    )
    async def _update_config_service(self) -> None:
        """Update configuration service."""
        try:
            # Configuration service updates are handled by the new modular structure
            self.logger.info("✅ Configuration service updated")

        except Exception:
            self.print(failed("❌ Failed to update configuration service: {e}"))

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="configuration manager stop",
    )
    async def stop(self) -> None:
        """Stop the configuration manager and cleanup resources."""
        try:
            self.logger.info("🛑 Stopping Configuration Manager...")
            self.is_initialized = False
            self.logger.info("✅ Configuration Manager stopped successfully")

        except Exception:
            self.print(failed("❌ Failed to stop Configuration Manager: {e}"))

    def get_status(self) -> dict[str, Any]:
        """Get configuration manager status."""
        return {
            "is_initialized": self.is_initialized,
            "config_sections_count": len(self.config_sections),
            "history_count": len(self.config_history),
        }

    def get_history(self, limit: int | None = None) -> list[dict[str, Any]]:
        """Get configuration history."""
        history = self.config_history.copy()
        if limit:
            history = history[-limit:]
        return history

    def get_config_sections(self) -> dict[str, Any]:
        """Get configuration sections."""
        return self.config_sections.copy()

    def get_config_service(self):
        """Get configuration service."""
        # This would return the actual configuration service if needed
        return

    def get_dual_model_config(self) -> dict[str, Any]:
        """Get dual model configuration."""
        return get_dual_model_config()

    def get_ml_confidence_predictor_config(self) -> dict[str, Any]:
        """Get ML confidence predictor configuration."""
        return get_ml_confidence_predictor_config()

    def get_position_sizing_config(self) -> dict[str, Any]:
        """Get position sizing configuration."""
        return get_position_sizing_config()

    def get_leverage_sizing_config(self) -> dict[str, Any]:
        """Get leverage sizing configuration."""
        return get_leverage_sizing_config()

    def get_position_closing_config(self) -> dict[str, Any]:
        """Get position closing configuration."""
        return get_position_closing_config()

    def get_position_division_config(self) -> dict[str, Any]:
        """Get position division configuration."""
        return get_position_division_config()

    def get_position_monitoring_config(self) -> dict[str, Any]:
        """Get position monitoring configuration."""
        return get_position_monitoring_config()

    def get_enhanced_training_config(self) -> dict[str, Any]:
        """Get enhanced training configuration."""
        return get_enhanced_training_config()

    def get_complete_config(self) -> dict[str, Any]:
        """Get complete configuration."""
        return get_complete_config()


# Legacy setup function
@handle_errors(
    exceptions=(Exception,),
    default_return=None,
    context="configuration manager setup",
)
async def setup_configuration_manager(
    config: dict[str, Any] | None = None,
) -> ConfigurationManager | None:
    """Setup and return a configured ConfigurationManager instance.

    Args:
        config: Configuration dictionary

    Returns:
        ConfigurationManager: Configured configuration manager instance
    """
    try:
        if config is None:
            config = get_complete_config()

        manager = ConfigurationManager(config)
        if await manager.initialize():
            return manager
        return None
    except Exception as e:
        system_logger.exception(f"Failed to setup configuration manager: {e}")
        return None
