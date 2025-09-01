"""
Training Mode Configuration

This module centralizes all training mode configurations to provide a single source of truth
for the three modes: light, blank, and full. Each mode has specific lookback periods and
training parameters optimized for different use cases.
"""

from typing import Final, Dict, Any
from dataclasses import dataclass


@dataclass
class TrainingModeConfig:
    """Configuration for a specific training mode."""
    name: str
    description: str
    lookback_days: int
    max_trials: int
    n_trials: int
    exclude_recent_days: int
    enable_advanced_model_training: bool
    enable_ensemble_training: bool
    enable_multi_timeframe_training: bool
    enable_adaptive_training: bool
    enhanced_training_interval: int
    max_enhanced_training_history: int
    min_data_points: int  # Minimum data points required for training
    computational_intensity: str  # "low", "medium", "high"
    estimated_duration_minutes: int  # Estimated training duration


# Training Mode Configurations
LIGHT_MODE = TrainingModeConfig(
    name="light",
    description="Light training mode for quick testing and development (30 days) - 2% of full intensity",
    lookback_days=30,
    max_trials=4,  # 2% of 200 = 4, minimum 3
    n_trials=3,   # 2% of 100 = 2, but minimum 3
    exclude_recent_days=1,
    enable_advanced_model_training=False,
    enable_ensemble_training=False,
    enable_multi_timeframe_training=False,
    enable_adaptive_training=False,
    enhanced_training_interval=1800,  # 30 minutes
    max_enhanced_training_history=10,
    min_data_points=50,
    computational_intensity="low",
    estimated_duration_minutes=5
)

BLANK_MODE = TrainingModeConfig(
    name="blank",
    description="Blank training mode for moderate testing and validation (180 days) - 10% of full intensity",
    lookback_days=180,
    max_trials=20,  # 10% of 200 = 20
    n_trials=10,   # 10% of 100 = 10 (already above minimum 3)
    exclude_recent_days=2,
    enable_advanced_model_training=True,
    enable_ensemble_training=True,
    enable_multi_timeframe_training=False,
    enable_adaptive_training=False,
    enhanced_training_interval=3600,  # 1 hour
    max_enhanced_training_history=50,
    min_data_points=100,
    computational_intensity="medium",
    estimated_duration_minutes=15
)

FULL_MODE = TrainingModeConfig(
    name="full",
    description="Full training mode for production-ready models (730 days) - 100% intensity",
    lookback_days=730,
    max_trials=200,
    n_trials=100,
    exclude_recent_days=2,
    enable_advanced_model_training=True,
    enable_ensemble_training=True,
    enable_multi_timeframe_training=True,
    enable_adaptive_training=True,
    enhanced_training_interval=7200,  # 2 hours
    max_enhanced_training_history=100,
    min_data_points=500,
    computational_intensity="high",
    estimated_duration_minutes=120
)


# Intensity percentages for each mode
INTENSITY_PERCENTAGES = {
    "light": 0.02,  # 2% of full intensity
    "blank": 0.10,  # 10% of full intensity
    "full": 1.00,   # 100% intensity
}

# Mode mapping for easy access
TRAINING_MODES: Dict[str, TrainingModeConfig] = {
    "light": LIGHT_MODE,
    "blank": BLANK_MODE,
    "full": FULL_MODE,
}

# Backward compatibility constants
LIGHT_TRAINING_LOOKBACK_DAYS: Final[int] = LIGHT_MODE.lookback_days
BLANK_TRAINING_LOOKBACK_DAYS: Final[int] = BLANK_MODE.lookback_days
FULL_TRAINING_LOOKBACK_DAYS: Final[int] = FULL_MODE.lookback_days
SHORT_BLANK_LOOKBACK_DAYS: Final[int] = LIGHT_MODE.lookback_days  # Alias for backward compatibility


def get_training_mode_config(mode: str) -> TrainingModeConfig:
    """
    Get the configuration for a specific training mode.

    Args:
        mode: The training mode ("light", "blank", or "full")

    Returns:
        TrainingModeConfig for the specified mode

    Raises:
        ValueError: If the mode is not supported
    """
    if mode not in TRAINING_MODES:
        raise ValueError(f"Unsupported training mode: {mode}. Supported modes: {list(TRAINING_MODES.keys())}")
    return TRAINING_MODES[mode]


def get_training_config_dict(mode: str) -> Dict[str, Any]:
    """
    Get the training configuration dictionary for a specific mode.

    Args:
        mode: The training mode ("light", "blank", or "full")

    Returns:
        Dictionary containing the training configuration
    """
    config = get_training_mode_config(mode)

    return {
        "enhanced_training_manager": {
            "enhanced_training_interval": config.enhanced_training_interval,
            "max_enhanced_training_history": config.max_enhanced_training_history,
            "enable_advanced_model_training": config.enable_advanced_model_training,
            "enable_ensemble_training": config.enable_ensemble_training,
            "enable_multi_timeframe_training": config.enable_multi_timeframe_training,
            "enable_adaptive_training": config.enable_adaptive_training,
            "max_trials": config.max_trials,
            "n_trials": config.n_trials,
            "lookback_days": config.lookback_days,
            "exclude_recent_days": config.exclude_recent_days,
            "min_data_points": config.min_data_points,
            "computational_intensity": config.computational_intensity,
            "estimated_duration_minutes": config.estimated_duration_minutes,
            # Mode flags for backward compatibility
            "blank_training_mode": mode == "blank",
            "full_training_mode": mode == "full",
            "light_training_mode": mode == "light",
        }
    }


def get_training_input_dict(mode: str, symbol: str, exchange: str, **kwargs) -> Dict[str, Any]:
    """
    Get the training input dictionary for a specific mode.

    Args:
        mode: The training mode ("light", "blank", or "full")
        symbol: Trading symbol
        exchange: Exchange name
        **kwargs: Additional parameters to override defaults

    Returns:
        Dictionary containing the training input configuration
    """
    config = get_training_mode_config(mode)

    base_input = {
        "enhanced_training_type": f"{mode}_training",
        "model_architecture": "enhanced_ensemble",
        "symbol": symbol,
        "exchange": exchange,
        "timeframe": "1m",
        "lookback_days": config.lookback_days,
        "training_mode": mode,
        "exclude_recent_days": config.exclude_recent_days,
        "min_data_points": config.min_data_points,
        "computational_intensity": config.computational_intensity,
        "estimated_duration_minutes": config.estimated_duration_minutes,
    }

    # Override with any provided kwargs
    base_input.update(kwargs)

    return base_input



def validate_mode_parameters(mode: str, **kwargs) -> bool:
    """
    Validate that the provided parameters are appropriate for the specified mode.

    Args:
        mode: The training mode to validate
        **kwargs: Parameters to validate

    Returns:
        True if parameters are valid, False otherwise
    """
    try:
        config = get_training_mode_config(mode)

        # Validate lookback_days if provided
        if "lookback_days" in kwargs:
            provided_lookback = kwargs["lookback_days"]
            if provided_lookback < 7:  # Minimum 1 week
                return False
            if mode == "light" and provided_lookback > 60:  # Light mode should be short
                return False
            if mode == "full" and provided_lookback < 365:  # Full mode should be substantial
                return False

        # Validate max_trials if provided
        if "max_trials" in kwargs:
            provided_max_trials = kwargs["max_trials"]
            if provided_max_trials < 3:  # Minimum 3 trials for all modes
                return False
            if mode == "light" and provided_max_trials > 5:  # Light mode should be quick
                return False

        # Validate n_trials if provided
        if "n_trials" in kwargs:
            provided_n_trials = kwargs["n_trials"]
            if provided_n_trials < 3:  # Minimum 3 trials for all modes
                return False
            if mode == "light" and provided_n_trials > 5:  # Light mode should be quick
                return False

        return True

    except ValueError:
        return False







