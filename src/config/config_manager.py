from __future__ import annotations

# src/config/config_manager.py

"""
Unified configuration manager for the Ares trading system.
This module organizes all configurable and optimizable parameters.
"""

from dataclasses import asdict
from typing import Any

from .config_confidence import get_confidence_config, get_confidence_search_space
from .config_ensemble import get_ensemble_config, get_ensemble_search_space
from .config_leverage import get_leverage_config, get_leverage_search_space
from .config_position_sizing import (
    get_position_sizing_config,
    get_position_sizing_search_space,
)
from .config_regime_transitions import (
    get_regime_transition_config,
    get_regime_transition_search_space,
)
from .config_sr import get_sr_config, get_sr_search_space
from .config_system_monitoring import (
    get_system_monitoring_config,
    get_system_monitoring_search_space,
)
from .config_technical_indicators import (
    get_technical_indicators_config,
    get_technical_indicators_search_space,
)
from .config_tpsl import get_tpsl_config, get_tpsl_search_space
from .config_training_optimization import (
    get_training_optimization_config,
    get_training_optimization_search_space,
)
from .config_two_tier import get_two_tier_config, get_two_tier_search_space


class ConfigManager:
    """Unified configuration manager for the Ares trading system."""

    def __init__(self):
        """Initialize the configuration manager."""
        self._static_config = None
        self._optimizable_configs = {}
        self._search_spaces = {}
        self._load_configurations()

    def _load_configurations(self):
        """Load all configurations."""
        # Load static (non-optimizable) configuration
        self._static_config = self.get_static_config()

        # Load optimizable configurations
        self._optimizable_configs = {
            "confidence": get_confidence_config(),
            "position_sizing": get_position_sizing_config(),
            "leverage": get_leverage_config(),
            "tpsl": get_tpsl_config(),
            "ensemble": get_ensemble_config(),
            "sr": get_sr_config(),
            "two_tier": get_two_tier_config(),
            "technical_indicators": get_technical_indicators_config(),
            "system_monitoring": get_system_monitoring_config(),
            "training_optimization": get_training_optimization_config(),
            "regime_transitions": get_regime_transition_config(),
        }

        # Load search spaces for optimization
        self._search_spaces = {
            "confidence": get_confidence_search_space(),
            "position_sizing": get_position_sizing_search_space(),
            "leverage": get_leverage_search_space(),
            "tpsl": get_tpsl_search_space(),
            "ensemble": get_ensemble_search_space(),
            "sr": get_sr_search_space(),
            "two_tier": get_two_tier_search_space(),
            "technical_indicators": get_technical_indicators_search_space(),
            "system_monitoring": get_system_monitoring_search_space(),
            "training_optimization": get_training_optimization_search_space(),
            "regime_transitions": get_regime_transition_search_space(),
        }

    def get_static_config(self) -> dict[str, Any]:
        """Get static (non-optimizable) configuration."""
        return self._static_config

    def get_optimizable_config(self, category: str) -> Any:
        """Get optimizable configuration for a specific category."""
        return self._optimizable_configs.get(category)

    def get_all_optimizable_configs(self) -> dict[str, Any]:
        """Get all optimizable configurations."""
        return self._optimizable_configs.copy()

    def get_search_space(self, category: str) -> dict[str, dict[str, Any]]:
        """Get search space for a specific category."""
        return self._search_spaces.get(category, {})

    def get_all_search_spaces(self) -> dict[str, dict[str, dict[str, Any]]]:
        """Get all search spaces."""
        return self._search_spaces.copy()

    def get_complete_config(self) -> dict[str, Any]:
        """Get complete configuration including both static and optimizable parameters."""
        complete_config = self._static_config.copy()

        # Add optimizable configurations
        for category, config in self._optimizable_configs.items():
            if hasattr(config, "__dict__"):
                complete_config[category] = asdict(config)
            else:
                complete_config[category] = config

        return complete_config

    def update_optimizable_config(self, category: str, updates: dict[str, Any]) -> bool:
        """Update optimizable configuration for a specific category."""
        if category not in self._optimizable_configs:
            return False

        config = self._optimizable_configs[category]

        for key, value in updates.items():
            if hasattr(config, key):
                setattr(config, key, value)

        return True

    def get_parameter_value(self, param_path: str) -> Any:
        """Get parameter value using dot notation (e.g., 'confidence.base_entry_threshold')."""
        if "." not in param_path:
            return None

        category, param_name = param_path.split(".", 1)

        # Check static config first
        if category in self._static_config:
            config = self._static_config[category]
            if hasattr(config, param_name):
                return getattr(config, param_name)
            if isinstance(config, dict) and param_name in config:
                return config[param_name]

        # Check optimizable configs
        if category in self._optimizable_configs:
            config = self._optimizable_configs[category]
            if hasattr(config, param_name):
                return getattr(config, param_name)

        return None

    def get_optimizable_parameters(self) -> dict[str, dict[str, dict[str, Any]]]:
        """Get all optimizable parameters organized by category."""
        return self._search_spaces.copy()

    def validate_config(self) -> tuple[bool, list[str]]:
        """Validate the complete configuration."""
        errors = []

        # Validate static config
        if not self._static_config:
            errors.append("Static configuration is missing")

        # Validate optimizable configs
        for category, config in self._optimizable_configs.items():
            if config is None:
                errors.append(f"Optimizable configuration for {category} is missing")

        # Validate search spaces
        for category, search_space in self._search_spaces.items():
            if not search_space:
                errors.append(f"Search space for {category} is missing")

        return len(errors) == 0, errors


# Global configuration manager instance
_config_manager = None


def get_config_manager() -> ConfigManager:
    """Get the global configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


def get_static_config_global() -> dict[str, Any]:
    """Get static configuration."""
    return get_config_manager().get_static_config()


def get_optimizable_config(category: str) -> Any:
    """Get optimizable configuration for a specific category."""
    return get_config_manager().get_optimizable_config(category)


def get_all_optimizable_configs() -> dict[str, Any]:
    """Get all optimizable configurations."""
    return get_config_manager().get_all_optimizable_configs()


def get_search_space(category: str) -> dict[str, dict[str, Any]]:
    """Get search space for a specific category."""
    return get_config_manager().get_search_space(category)


def get_all_search_spaces() -> dict[str, dict[str, dict[str, Any]]]:
    """Get all search spaces."""
    return get_config_manager().get_all_search_spaces()


def get_complete_config() -> dict[str, Any]:
    """Get complete configuration."""
    return get_config_manager().get_complete_config()


def get_parameter_value(param_path: str) -> Any:
    """Get parameter value using dot notation."""
    return get_config_manager().get_parameter_value(param_path)


def get_optimizable_parameters() -> dict[str, dict[str, dict[str, Any]]]:
    """Get all optimizable parameters."""
    return get_config_manager().get_optimizable_parameters()


def update_optimizable_config(category: str, updates: dict[str, Any]) -> bool:
    """Update optimizable configuration for a specific category."""
    return get_config_manager().update_optimizable_config(category, updates)


def validate_config() -> tuple[bool, list[str]]:
    """Validate the complete configuration."""
    return get_config_manager().validate_config()
