# src/config/__init__.py

from typing import Any

# Version information
ARES_VERSION = "0.1.0"

from src.config.environment import get_environment_settings
from src.config.system import get_system_config
from src.config.trading import get_trading_config
from src.config.training import get_training_config
from src.config.validation import validate_complete_config







def get_training_config_section() -> dict[str, Any]:
    """Get training configuration.

    Returns:
        dict: Training configuration

    """
    return get_config_section("training")


# Create the main CONFIG object for backward compatibility
CONFIG = get_complete_config()
