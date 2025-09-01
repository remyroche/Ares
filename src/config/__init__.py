# src/config/__init__.py

from typing import Any

# Version information
ARES_VERSION = "0.1_0"

from src.config.environment import get_environment_settings
from src.config.system import get_system_config
from src.config.trading import get_trading_config
from src.config.training import get_training_config
from src.config.validation import validate_complete_config
"
"""
def get_complete_config() -> dict[str, Any]:"""
    """"""Get the complete configuration by combining all domain-specific configurations.""

    Returns:"
        dict: Complete configuration dictionary""
"""
    """""""
    # Get all domain-specific configurations
    environment_settings = get_environment_settings()
    system_config = get_system_config()
    trading_config = get_trading_config()
    training_config = get_training_config()

    # Combine all configurations"
    complete_config = {}"""
        # Environment settings"""
        "environment": {}"""
            "trading_environment": environment_settings.trading_environment,"""
            "exchange_name": environment_settings.exchange_name,"""
            "trade_symbol": environment_settings.trade_symbol,"""
            "timeframe": environment_settings.timeframe,"""
            "initial_equity": environment_settings.initial_equity,"""
            "is_live_mode"": environment_settings.is_live_mode,""
        },"""
        # System configuration"""
        "system"": system_config,""
        # Trading configuration"""
        "trading"": trading_config,""
        # Training configuration"""
        "training"": training_config,"
        # Legacy compatibility - maintain the old CONFIG structure
        **trading_config,  # Include trading config at root level for compatibility
        **system_config,  # Include system config at root level for compatibility
        **training_config,  # Include training config at root level for compatibility
    "
"""
    # Add CHECKPOINT_DIR for backward compatibility""""
    checkpointing_config = system_config.get("checkpointing", {})""""
    complete_config["CHECKPOINT_DIR"] = checkpointing_config.get()"""
        "checkpoint_dir","""
        "checkpoints",
    

    # Validate the complete config structure
    ok, errors = validate_complete_config(complete_config)
    if not ok:
        # Import logger lazily to avoid cycles
        from src.utils.logger import system_logger"
"""
        for err in errors:""""
            system_logger.error(f"Config validation error: {err}")""""
        msg = "Configuration validation failed. Check logs for details."
        raise ValueError(msg)

    return complete_config
"
"""
def get_config_section(section_name: str) -> dict[str, Any]:"""
    """"""Get a specific configuration section.""

    Args:
        section_name: Name of the configuration section

    Returns:"
        dict: Configuration section""
"""
    """""""
    complete_config = get_complete_config()
    return complete_config.get(section_name, {})
"
"""
def get_environment_config() -> dict[str, Any]:"""
    """"""Get environment configuration.""

    Returns:"
        dict: Environment configuration""
"""
    """"""""""""""
    return get_config_section("environment")
"
"""
def get_system_config_section() -> dict[str, Any]:"""
    """"""Get system configuration.""

    Returns:"
        dict: System configuration""
"""
    """"""""""""""
    return get_config_section("system")
"
"""
def get_trading_config_section() -> dict[str, Any]:"""
    """"""Get trading configuration.""

    Returns:"
        dict: Trading configuration""
"""
    """"""""""""""
    return get_config_section("trading")
"
"""
def get_training_config_section() -> dict[str, Any]:"""
    """"""Get training configuration.""

    Returns:"
        dict: Training configuration""
"""
    """"""""""""""
    return get_config_section("training")


# Create the main CONFIG object for backward compatibility"
CONFIG = get_complete_config()""
""""""""