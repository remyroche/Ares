"""
Centralized Pipeline Mode Configuration

This module provides a single source of truth for all pipeline mode definitions,
ensuring consistency across the entire system.
"""

from typing import Final, Dict, Any
from dataclasses import dataclass
from enum import Enum

class PipelineMode(Enum):
    """Pipeline execution modes."""
    FULL = "full"
    LIGHT = "light"
    BLANK = "blank"

@dataclass
class ModeConfiguration:
    """Configuration for a specific pipeline mode."""
    name: str
    description: str
    lookback_days: int
    lookback_years: int
    intensity_percentage: float
    computational_intensity: str
    estimated_duration_minutes: int
    max_trials: int
    n_trials: int
    monte_carlo_samples: int
    ab_test_rounds: int
    optuna_trials: int
    optuna_timeout: int
    batch_size: int
    epochs: int
    early_stopping_patience: int
    cross_validation_folds: int
    enable_parallelization: bool
    enable_caching: bool
    enable_advanced_features: bool
    enable_ensemble_training: bool
    enable_multi_timeframe_training: bool
    enable_adaptive_training: bool

# Centralized mode definitions
FULL_MODE_CONFIG: Final[ModeConfiguration] = ModeConfiguration(
    name="full",
    description="Production mode - Complete training with full dataset",
    lookback_days=1460,  # 4 years
    lookback_years=4,
    intensity_percentage=1.0,  # 100% intensity
    computational_intensity="high",
    estimated_duration_minutes=240,
    max_trials=200,
    n_trials=100,
    monte_carlo_samples=10000,
    ab_test_rounds=10,
    optuna_trials=200,
    optuna_timeout=3600,
    batch_size=4096,
    epochs=100,
    early_stopping_patience=20,
    cross_validation_folds=5,
    enable_parallelization=True,
    enable_caching=True,
    enable_advanced_features=True,
    enable_ensemble_training=True,
    enable_multi_timeframe_training=True,
    enable_adaptive_training=True
)

LIGHT_MODE_CONFIG: Final[ModeConfiguration] = ModeConfiguration(
    name="light",
    description="Development mode - Minimal data with all features/models",
    lookback_days=20,  # 20 days (increased from 10 for better regime diversity)
    lookback_years=0,  # Less than a year
    intensity_percentage=0.025,  # 2.5% intensity
    computational_intensity="minimal",
    estimated_duration_minutes=10,  # Updated estimate for 20 days
    max_trials=10,
    n_trials=5,
    monte_carlo_samples=100,
    ab_test_rounds=2,
    optuna_trials=10,
    optuna_timeout=300,
    batch_size=512,
    epochs=10,
    early_stopping_patience=5,
    cross_validation_folds=2,
    enable_parallelization=False,
    enable_caching=True,
    enable_advanced_features=True,
    enable_ensemble_training=True,
    enable_multi_timeframe_training=True,
    enable_adaptive_training=True
)

BLANK_MODE_CONFIG: Final[ModeConfiguration] = ModeConfiguration(
    name="blank",
    description="Quick testing mode - All features/models with shorter lookback",
    lookback_days=180,  # 6 months
    lookback_years=0,  # Less than a year
    intensity_percentage=0.1,  # 10% intensity
    computational_intensity="medium",
    estimated_duration_minutes=60,
    max_trials=50,
    n_trials=25,
    monte_carlo_samples=1000,
    ab_test_rounds=3,
    optuna_trials=50,
    optuna_timeout=900,
    batch_size=2048,
    epochs=50,
    early_stopping_patience=10,
    cross_validation_folds=3,
    enable_parallelization=True,
    enable_caching=True,
    enable_advanced_features=True,
    enable_ensemble_training=True,
    enable_multi_timeframe_training=True,
    enable_adaptive_training=True
)

# Mode registry for easy access
MODE_REGISTRY: Final[Dict[str, ModeConfiguration]] = {
    "full": FULL_MODE_CONFIG,
    "light": LIGHT_MODE_CONFIG,
    "blank": BLANK_MODE_CONFIG
}

def get_mode_config(mode: str) -> ModeConfiguration:
    """
    Get configuration for a specific pipeline mode.

    Args:
        mode: Pipeline mode name ("full", "light", "blank")

    Returns:
        ModeConfiguration object for the specified mode

    Raises:
        ValueError: If mode is not recognized
    """
    if mode not in MODE_REGISTRY:
        raise ValueError(f"Unknown pipeline mode: {mode}. Available modes: {list(MODE_REGISTRY.keys())}")

    return MODE_REGISTRY[mode]

def get_full_mode_config() -> ModeConfiguration:
    """Get the full mode configuration (convenience function)."""
    return get_mode_config("full")

def get_light_mode_config() -> ModeConfiguration:
    """Get the light mode configuration (convenience function)."""
    return get_mode_config("light")

def get_blank_mode_config() -> ModeConfiguration:
    """Get the blank mode configuration (convenience function)."""
    return get_mode_config("blank")

def get_mode_lookback_days(mode: str) -> int:
    """Get lookback days for a specific mode."""
    return get_mode_config(mode).lookback_days

def get_mode_lookback_years(mode: str) -> int:
    """Get lookback years for a specific mode."""
    return get_mode_config(mode).lookback_years

def get_mode_intensity_percentage(mode: str) -> float:
    """Get intensity percentage for a specific mode."""
    return get_mode_config(mode).intensity_percentage

def get_mode_estimated_duration(mode: str) -> int:
    """Get estimated duration in minutes for a specific mode."""
    return get_mode_config(mode).estimated_duration_minutes

def get_all_mode_configs() -> Dict[str, ModeConfiguration]:
    """Get all mode configurations."""
    return MODE_REGISTRY.copy()

def get_mode_summary() -> Dict[str, Dict[str, Any]]:
    """Get a summary of all modes for display purposes."""
    summary = {}
    for mode_name, config in MODE_REGISTRY.items():
        summary[mode_name] = {
            "name": config.name,
            "description": config.description,
            "lookback_days": config.lookback_days,
            "lookback_years": config.lookback_years,
            "intensity_percentage": config.intensity_percentage,
            "computational_intensity": config.computational_intensity,
            "estimated_duration_minutes": config.estimated_duration_minutes,
            "max_trials": config.max_trials,
            "n_trials": config.n_trials
        }
    return summary

# Legacy compatibility - maintain existing constants
DEFAULT_LOOKBACK_DAYS: Final[int] = FULL_MODE_CONFIG.lookback_days
DEFAULT_LOOKBACK_YEARS: Final[int] = FULL_MODE_CONFIG.lookback_years
DEFAULT_INTENSITY_PERCENTAGE: Final[float] = FULL_MODE_CONFIG.intensity_percentage
