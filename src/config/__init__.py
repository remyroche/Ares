# src/config/__init__.py

from typing import Any

from src.config.environment import get_environment_settings
from src.config.system import get_system_config
from src.config.trading import get_trading_config
from src.config.training import get_training_config
from src.config.validation import validate_complete_config


def get_complete_config() -> dict[str, Any]:
    """
    Get the complete configuration by combining all domain-specific configurations.
    
    Returns:
        dict: Complete configuration dictionary
    """
    # Get all domain-specific configurations
    environment_settings = get_environment_settings()
    system_config = get_system_config()
    trading_config = get_trading_config()
    training_config = get_training_config()
    
    # Combine all configurations
    complete_config = {
        # Environment settings
        "environment": {
            "trading_environment": environment_settings.trading_environment,
            "exchange_name": environment_settings.exchange_name,
            "trade_symbol": environment_settings.trade_symbol,
            "timeframe": environment_settings.timeframe,
            "initial_equity": environment_settings.initial_equity,
            "is_live_mode": environment_settings.is_live_mode,
        },
        
        # System configuration
        "system": system_config,
        
        # Trading configuration
        "trading": trading_config,
        
        # Training configuration
        "training": training_config,
        
        # Legacy compatibility - maintain the old CONFIG structure
        **trading_config,  # Include trading config at root level for compatibility
        **system_config,   # Include system config at root level for compatibility
        **training_config, # Include training config at root level for compatibility
    }
    
    # Add CHECKPOINT_DIR for backward compatibility
    checkpointing_config = system_config.get("checkpointing", {})
    complete_config["CHECKPOINT_DIR"] = checkpointing_config.get("checkpoint_dir", "checkpoints")

    # Validate the complete config structure
    ok, errors = validate_complete_config(complete_config)
    if not ok:
        # Import logger lazily to avoid cycles
        from src.utils.logger import system_logger
        for err in errors:
            system_logger.error(f"Config validation error: {err}")
        raise ValueError("Configuration validation failed. Check logs for details.")
    
    return complete_config


def get_config_section(section_name: str) -> dict[str, Any]:
    """
    Get a specific configuration section.
    
    Args:
        section_name: Name of the configuration section
        
    Returns:
        dict: Configuration section
    """
    complete_config = get_complete_config()
    return complete_config.get(section_name, {})


def get_environment_config() -> dict[str, Any]:
    """
    Get environment configuration.
    
    Returns:
        dict: Environment configuration
    """
    return get_config_section("environment")


def get_system_config_section() -> dict[str, Any]:
    """
    Get system configuration.
    
    Returns:
        dict: System configuration
    """
    return get_config_section("system")


def get_trading_config_section() -> dict[str, Any]:
    """
    Get trading configuration.
    
    Returns:
        dict: Trading configuration
    """
    return get_config_section("trading")


def get_training_config_section() -> dict[str, Any]:
    """
    Get training configuration.
    
    Returns:
        dict: Training configuration
    """
    return get_config_section("training")


# Legacy compatibility functions
def get_lookback_window(config: dict[str, Any] | None = None) -> int:
    """
    Get the lookback window for data collection.
    
    Args:
        config: Configuration dictionary (optional)
        
    Returns:
        int: Lookback window in days
    """
    if config is None:
        config = get_complete_config()
    
    # Try to get from training config first
    training_config = get_training_config_section()
    data_config = training_config.get("DATA_CONFIG", {})
    lookback_days = data_config.get("default_lookback_days", 730)
    
    # Fallback to legacy config
    if lookback_days is None:
        lookback_days = config.get("lookback_years", 2) * 365
    
    return lookback_days


# Legacy AresConfig class for backward compatibility
class AresConfig:
    """
    Legacy configuration class for backward compatibility.
    """
    
    def __init__(self):
        self.config = get_complete_config()
        self.settings = get_environment_settings()
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value."""
        return self.config.get(key, default)
    
    def get_setting(self, key: str, default: Any = None) -> Any:
        """Get a setting value."""
        return getattr(self.settings, key, default)
    
    @property
    def trading_environment(self) -> str:
        """Get the trading environment."""
        return self.settings.trading_environment
    
    @property
    def trade_symbol(self) -> str:
        """Get the trade symbol."""
        return self.settings.trade_symbol
    
    @property
    def exchange_name(self) -> str:
        """Get the exchange name."""
        return self.settings.exchange_name
    
    @property
    def timeframe(self) -> str:
        """Get the timeframe."""
        return self.settings.timeframe
    
    @property
    def initial_equity(self) -> float:
        """Get the initial equity."""
        return self.settings.initial_equity
    
    @property
    def is_live_mode(self) -> bool:
        """Check if running in live mode."""
        return self.settings.is_live_mode
    
    @property
    def exchange_config(self) -> dict:
        """Get the exchange configuration."""
        trading_config = get_trading_config_section()
        exchanges = trading_config.get("exchanges", {})
        exchange_name = self.exchange_name.lower()
        return exchanges.get(exchange_name, {})
    
    @property
    def api_key(self) -> str | None:
        """Get the API key for the current exchange."""
        exchange_config = self.exchange_config
        return exchange_config.get("api_key")
    
    @property
    def api_secret(self) -> str | None:
        """Get the API secret for the current exchange."""
        exchange_config = self.exchange_config
        return exchange_config.get("api_secret")
    
    @property
    def password(self) -> str | None:
        """Get the password for the current exchange."""
        exchange_config = self.exchange_config
        return exchange_config.get("password")
    
    @property
    def symbols(self) -> list[str]:
        """Get the symbols for the current exchange."""
        exchange_config = self.exchange_config
        return exchange_config.get("symbols", [])


# Legacy configuration functions for backward compatibility
def get_dual_model_config() -> dict[str, Any]:
    """Get dual model configuration."""
    # This would be implemented based on the specific dual model requirements
    return {}


def get_ml_confidence_predictor_config() -> dict[str, Any]:
    """Get ML confidence predictor configuration."""
    # This would be implemented based on the specific ML confidence predictor requirements
    return {}


def get_position_sizing_config() -> dict[str, Any]:
    """Get position sizing configuration."""
    from src.config.trading import get_position_sizing_config as get_pos_sizing_config
    return get_pos_sizing_config()


def get_leverage_sizing_config() -> dict[str, Any]:
    """Get leverage sizing configuration."""
    # This would be implemented based on the specific leverage sizing requirements
    return {}


def get_position_closing_config() -> dict[str, Any]:
    """Get position closing configuration."""
    from src.config.trading import get_position_closing_config as get_pos_closing_config
    return get_pos_closing_config()


def get_position_division_config() -> dict[str, Any]:
    """Get position division configuration."""
    # This would be implemented based on the specific position division requirements
    return {}


def get_position_monitoring_config() -> dict[str, Any]:
    """Get position monitoring configuration."""
    from src.config.trading import get_position_monitoring_config as get_pos_monitoring_config
    return get_pos_monitoring_config()


def get_enhanced_training_config() -> dict[str, Any]:
    """Get enhanced training configuration."""
    from src.config.training import get_enhanced_training_config as get_enh_training_config
    return get_enh_training_config()


# Global configuration instance for backward compatibility
CONFIG = get_complete_config() 