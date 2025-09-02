# src/config/__init__.py

from typing import Any, Dict

# Version information
ARES_VERSION = "0_2_3"


def get_complete_config() -> Dict[str, Any]:
    """Get complete configuration for the system."""
    # Return a basic configuration structure
    return {
        "environment": {
            "trading_environment": "paper",
            "exchange_name": "binance",
            "trade_symbol": "BTCUSDT",
            "timeframe": "1h",
            "initial_equity": 10000.0,
            "is_live_mode": False,
        },
        "system": {
            "checkpointing": {
                "checkpoint_dir": "checkpoints",
                "save_interval": 3600,
            },
            "logging": {
                "level": "INFO",
                "file": "ares.log",
            },
        },
        "trading": {
            "max_position_size": 0.1,
            "stop_loss_pct": 0.02,
            "take_profit_pct": 0.04,
        },
        "training": {
            "model_dir": "models",
            "batch_size": 32,
            "epochs": 100,
        },
        "analyst": {
            "analysis_interval": 3600,
            "max_analysis_history": 100,
        },
        "training": {
            "training_interval": 86400,
            "max_training_history": 50,
        },
        # Legacy compatibility
        "CHECKPOINT_DIR": "checkpoints",
    }


def get_config_section(section_name: str) -> Dict[str, Any]:
    """Get a specific configuration section."""
    complete_config = get_complete_config()
    return complete_config.get(section_name, {})


def get_environment_config() -> Dict[str, Any]:
    """Get environment configuration."""
    return get_config_section("environment")


def get_system_config_section() -> Dict[str, Any]:
    """Get system configuration section."""
    return get_config_section("system")


def get_trading_config_section() -> Dict[str, Any]:
    """Get trading configuration section."""
    return get_config_section("trading")


def get_training_config_section() -> Dict[str, Any]:
    """Get training configuration section."""
    return get_config_section("training")


# Create the main CONFIG object for backward compatibility
CONFIG = get_complete_config()
