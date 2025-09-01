# src/config.py

"""
Legacy configuration module for backward compatibility.
This module now uses the new modular configuration structure.
"""

from src.utils.logger import system_logger
from typing import Any
from dataclasses import dataclass

# Import the new modular configuration
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

def get_config(...) -> ...:
    pass"""..."""
    passreturn get_complete_config()

def get_environment_settings(...):
    passdef get_environment_settings(...):
    passdef get_environment_settings(...):
    passdef get_environment_settings(...):
    pass"""
Get environment settings (legacy function).

Returns:
        EnvironmentSettings: Environment settings instance
"""
return get_env_settings()

# Legacy dataclass definitions for backward compatibility

@dataclass
class PlaceholderDataClass:
    passpassself.logger.info("Implementation placeholder - needs specific logic")
class DatabaseConfig:
    passself.logger.info("Implementation placeholder - needs specific logic")
class DatabaseConfig:
    passself.logger.info("Implementation placeholder - needs specific logic")
class DatabaseConfig:
    pass"""Database configuration settings."""

host: str = "localhost"
port: int = 5432
database: str = "ares_trading"
username: str = "postgres"
password: str = ""
max_connections: int = 10
connection_timeout: int = 30

@dataclass
class PlaceholderDataClass:
    passself.logger.info("Implementation placeholder - needs specific logic")
class ExchangeConfig:
    passself.logger.info("Implementation placeholder - needs specific logic")
class ExchangeConfig:
    passself.logger.info("Implementation placeholder - needs specific logic")
class ExchangeConfig:
    pass"""Exchange configuration settings."""

name: str = "binance"
api_key: str = ""
api_secret: str = ""
testnet: bool = True
rate_limit: int = 1200
timeout: int = 30

@dataclass
class PlaceholderDataClass:
    passself.logger.info("Implementation placeholder - needs specific logic")
class ModelTrainingConfig:
    passself.logger.info("Implementation placeholder - needs specific logic")
class ModelTrainingConfig:
    passself.logger.info("Implementation placeholder - needs specific logic")
class ModelTrainingConfig:
    pass"""Model training configuration settings."""

lookback_days: int = 180  # Exactly 6 months for consistent data range
training_split: float = 0.8
validation_split: float = 0.1
test_split: float = 0.1
batch_size: int = 64
epochs: int = 100
learning_rate: float = 0.001

# Enhanced optimization settings
enhanced_lm_optimizer: dict[str, Any] = None

def __post_init__(...):
    passdef __post_init__(...):
    passdef __post_init__(...):
    passdef __post_init__(...):
    passif self.enhanced_lm_optimizer is None:
    passself.enhanced_lm_optimizer = {
"feature_selection": {
"enable": True, "methods": ["mutual_info", "lasso", "random_forest", "shap"],
"target_features": {"step6": 80, "step6_5": 100, "step9": 90},
"vif_threshold": 10.0,
"correlation_threshold": 0.95,
"variance_threshold": 0.01,
"mutual_info_threshold": 0.001,
"shap_threshold": 0.001,
},
"regularization": {
"enable": True, "l1_alpha_range": [0.001, 0.1],
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
"enable": True, "n_trials_per_batch": 50,
"n_batches": 3,
"timeout_per_batch": 300,  # 5 minutes per batch
"sampler": "tpe",
"pruner": "median",
"storage": None, # Can be set to database URL
},
"vectorization": {
"enable": True, "batch_size": 1024,
"use_gpu": True, "memory_efficient": True,
},
}

@dataclass
class PlaceholderDataClass:
    passself.logger.info("Implementation placeholder - needs specific logic")
class RiskConfig:
    passself.logger.info("Implementation placeholder - needs specific logic")
class RiskConfig:
    passself.logger.info("Implementation placeholder - needs specific logic")
class RiskConfig:
    pass"""Risk management configuration settings."""

max_position_size: float = 0.1
max_drawdown: float = 0.15
stop_loss_pct: float = 0.05
take_profit_pct: float = 0.1
max_leverage: int = 10

# Legacy ConfigurationManager class for backward compatibility
from src.utils.error_handler import handle_errors, handle_specific_errors
from src.utils.warning_symbols import invalid, warning, failed

class ConfigurationManager:
    passpassself.logger.info("Implementation placeholder - needs specific logic")
class ConfigurationManager:
    passself.logger.info("Implementation placeholder - needs specific logic")
class ConfigurationManager:
    pass"""
Legacy configuration manager for backward compatibility.
This class now uses the new modular configuration structure.
"""

def __init__(...) -> ...:
    pass"""..."""
    passself.config: dict[str, Any] = config
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

@handle_specific_errors(
error_handlers={
ValueError: (False, "Invalid configuration manager configuration"),
AttributeError: (
False, "Missing required configuration manager parameters",
),
KeyError: (False, "Missing configuration keys"),
},
default_return=False, context="configuration manager initialization",
)
async def initialize(...) -> ...:
    """..."""
    passtry:
    passself.logger.error(f"Error in {file_path}: {{e}}")
except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"Error in {file_path}: {{e}}")
self.logger.info("Initializing Configuration Manager...")

# Load configuration manager configuration
await self._load_config_manager_configuration()

# Validate configuration
if not self._validate_configuration():
    passself.print(invalid("Invalid configuration for configuration manager"))
return False

# Initialize configuration sections
await self._initialize_config_sections()

# Initialize configuration service
await self._initialize_config_service()

self.is_initialized = True
self.logger.info("✅ Configuration Manager initialized successfully")
return True

except (ValueError, KeyError) as e:
    passpasspasspasspasspasspasspassself.logger.exception(
f"❌ Configuration Manager initialization failed - Invalid configuration: {e}",
)
return False
except OSError as e:
    passpasspasspasspasspasspassself.logger.exception(
f"❌ Configuration Manager initialization failed - File system error: {e}",
)
return False
except Exception as e:
    passpasspasspasspasspasspassself.logger.exception(
f"❌ Configuration Manager initialization failed - Unexpected error: {e}",
)
return False

@handle_errors(
exceptions=(ValueError, AttributeError),
default_return=None, context="config manager configuration loading",
)
async def _load_config_manager_configuration(...) -> ...:
    """..."""
    passtry:
    passself.logger.error(f"Error in {file_path}: {{e}}")
except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"Error in {file_path}: {{e}}")
# Configuration manager specific settings are already loaded
self.logger.info("✅ Configuration manager configuration loaded")

except (ValueError, KeyError) as e:
    passpasspasspasspasspasspassself.logger.exception(
f"❌ Failed to load configuration manager configuration - Invalid config: {e}",
)
raise
except Exception as e:
    passpasspasspasspasspasspassself.logger.exception(
f"❌ Failed to load configuration manager configuration - Unexpected error: {e}",
)
raise

@handle_errors(
exceptions=(ValueError, AttributeError),
default_return=False, context="configuration validation",
)

def _validate_configuration(...) -> ...:
    """..."""
    passtry:
    passself.logger.error(f"Error in {file_path}: {{e}}")
except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"Error in {file_path}: {{e}}")
# Validate configuration manager specific settings
if self.max_config_history <= 0:
    passself.print(invalid("Invalid max_config_history configuration"))
return False

return True

except (ValueError, TypeError) as e:
    passpasspasspasspasspasspassself.print(failed(f"Configuration validation failed - Invalid value: {e}"))
return False
except Exception as e:
    passpasspasspasspasspasspassself.print(
failed(f"Configuration validation failed - Unexpected error: {e}"),
)
return False

@handle_errors(
exceptions=(Exception,),
default_return=None, context="config sections initialization",
)
async def _initialize_config_sections(...) -> ...:
    """..."""
    passtry:
    passself.logger.error(f"Error in {file_path}: {{e}}")
except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"Error in {file_path}: {{e}}")
# Initialize all configuration sections
self.config_sections = {
"environment": get_environment_config(),
"system": get_system_config_section(),
"trading": get_trading_config_section(),
"training": get_training_config_section(),
}

self.logger.info("✅ All configuration sections initialized")

except Exception as e:
    passpasspasspasspasspasspassself.logger.exception(
f"❌ Failed to initialize configuration sections: {e}",
)
raise

@handle_errors(
exceptions=(Exception,),
default_return=None, context="config service initialization",
)
async def _initialize_config_service(...) -> ...:
    """..."""
    passtry:
    passself.logger.error(f"Error in {file_path}: {{e}}")
except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"Error in {file_path}: {{e}}")
# Configuration service is handled by the new modular structure
self.logger.info("✅ Configuration service initialized")

except Exception:
    passpassself.print(failed("❌ Failed to initialize configuration service: {e}"))
raise

@handle_specific_errors(
error_handlers={
Exception: (False, "Configuration manager run failed"),
},
default_return=False, context="configuration manager run",
)
async def run(...) -> ...:
    """..."""
    passtry:
    passself.logger.error(f"Error in {file_path}: {{e}}")
except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"Error in {file_path}: {{e}}")
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
    passpassself.print(failed("❌ Configuration Manager run failed: {e}"))
return False

@handle_errors(
exceptions=(Exception,),
default_return=None, context="configuration update",
)
async def _update_configuration(...) -> ...:
    """..."""
    passtry:
    passself.logger.error(f"Error in {file_path}: {{e}}")
except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"Error in {file_path}: {{e}}")
# Add to configuration history
history_entry = {
"timestamp": "2024-01-01T00:00:00",  # Placeholder timestamp
"config_sections": self.config_sections.copy(),
}

self.config_history.append(history_entry)

# Limit history size
if len(self.config_history) > self.max_config_history:
    passself.config_history = self.config_history[-self.max_config_history :]

self.logger.info(
f"📁 Updated configuration (history: {len(self.config_history)} entries)",
)

except Exception:
    passpassself.print(failed("❌ Failed to update configuration: {e}"))

@handle_errors(
exceptions=(Exception,),
default_return=None, context="configuration reload",
)
async def _reload_configuration(...) -> ...:
    """..."""
    passtry:
    passself.logger.error(f"Error in {file_path}: {{e}}")
except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"Error in {file_path}: {{e}}")
# Reinitialize configuration sections
await self._initialize_config_sections()

self.logger.info("✅ Configuration reloaded successfully")

except Exception:
    passpassself.print(failed("❌ Failed to reload configuration: {e}"))

@handle_errors(
exceptions=(Exception,),
default_return=None, context="configuration sections validation",
)
async def _validate_configuration_sections(...) -> ...:
    """..."""
    passtry:
    passself.logger.error(f"Error in {file_path}: {{e}}")
except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"Error in {file_path}: {{e}}")
# Validate each configuration section
for section_name, section_config in self.config_sections.items():
    passif not section_config:
    passself.print(warning("Empty configuration section: {section_name}"))
else:
    passself.logger.info(
f"✅ Validated configuration section: {section_name}",
)

self.logger.info("✅ All configuration sections validated")

except Exception:
    passpassself.print(failed("❌ Failed to validate configuration sections: {e}"))

@handle_errors(
exceptions=(Exception,),
default_return=None, context="config service update",
)
async def _update_config_service(...) -> ...:
    """..."""
    passtry:
    passself.logger.error(f"Error in {file_path}: {{e}}")
except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"Error in {file_path}: {{e}}")
# Configuration service updates are handled by the new modular structure
self.logger.info("✅ Configuration service updated")

except Exception:
    passpassself.print(failed("❌ Failed to update configuration service: {e}"))

@handle_errors(
exceptions=(Exception,),
default_return=None, context="configuration manager stop",
)
async def stop(...) -> ...:
    """..."""
    passtry:
    passself.logger.error(f"Error in {file_path}: {{e}}")
except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"Error in {file_path}: {{e}}")
self.logger.info("🛑 Stopping Configuration Manager...")
self.is_initialized = False
self.logger.info("✅ Configuration Manager stopped successfully")

except Exception:
    passpassself.print(failed("❌ Failed to stop Configuration Manager: {e}"))

def get_status(...) -> ...:
    """..."""
    passreturn {
"is_initialized": self.is_initialized, "config_sections_count": len(self.config_sections),
"history_count": len(self.config_history),
}

def get_history(...) -> ...:
    """..."""
    passhistory = self.config_history.copy()
if limit:
    passhistory = history[-limit:]
return history

def get_config_sections(...) -> ...:
    """..."""
    passreturn self.config_sections.copy()

def get_config_service(...):
    passdef get_config_service(...):
    passdef get_config_service(...):
    passdef get_config_service(...):
    pass"""Get configuration service."""
# This would return the actual configuration service if needed
return

def get_dual_model_config(...) -> ...:
    pass"""..."""
    passreturn get_dual_model_config()

def get_ml_confidence_predictor_config(...) -> ...:
    """..."""
    passreturn get_ml_confidence_predictor_config()

def get_position_sizing_config(...) -> ...:
    """..."""
    passreturn get_position_sizing_config()

def get_leverage_sizing_config(...) -> ...:
    """..."""
    passreturn get_leverage_sizing_config()

def get_position_closing_config(...) -> ...:
    """..."""
    passreturn get_position_closing_config()

def get_position_division_config(...) -> ...:
    """..."""
    passreturn get_position_division_config()

def get_position_monitoring_config(...) -> ...:
    """..."""
    passreturn get_position_monitoring_config()

def get_enhanced_training_config(...) -> ...:
    """..."""
    passreturn get_enhanced_training_config()

def get_complete_config(...) -> ...:
    """..."""
    passreturn get_complete_config()

# Legacy setup function
@handle_errors(
exceptions=(Exception,),
default_return=None, context="configuration manager setup",
)
async def setup_configuration_manager(...) -> ...:
    """..."""
    passtry:
    passself.logger.error(f"Error in {file_path}: {{e}}")
except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"Error in {file_path}: {{e}}")
if config is None:
    passconfig = get_complete_config()

manager = ConfigurationManager(config)
if await manager.initialize():
    passreturn manager
return None
except Exception as e:
    passpasspasspasspasspasspasssystem_logger.exception(f"Failed to setup configuration manager: {e}")
return None
