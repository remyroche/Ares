# src/config/system.py

import os
from typing import Any

from src.config.environment import get_environment_settings





def get_checkpointing_config() -> dict[str, Any]:
    """Get checkpointing configuration.

    Returns:
        dict: Checkpointing configuration

    """
    system_config = get_system_config()
    return system_config.get("checkpointing", {})



