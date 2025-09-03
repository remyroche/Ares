# src/config/multi_timeframe_hmm_ensemble_config.py
"""Multi-Timeframe HMM Ensemble Configuration.

Configuration settings for the multi-timeframe HMM cluster ensemble system that combines
predictions from HMM clusters across multiple timeframes.
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class TimeframeConfig:
    """Configuration for each timeframe in the ensemble."""

    timeframe: str
    weight: float
    min_samples: int = 50
    enable_hazard_model: bool = True
    enable_price_prediction: bool = (
        False  # Hazard models are for regime transitions only
    )


@dataclass
class EnsembleConfig:
    """Configuration for the multi-timeframe ensemble."""

    timeframes: list[TimeframeConfig]
    meta_learner_type: str = "lgbm"  # "lgbm", "random_forest", "logistic"
    enable_dynamic_weighting: bool = True
    weight_update_frequency: int = 100  # Update weights every N predictions
    min_confidence_threshold: float = 0.6
    ensemble_method: str = (
        "meta_learner"  # "weighted_average", "meta_learner", "stacking"
    )


def get_multi_timeframe_hmm_ensemble_config() -> dict[str, Any]:
    """Get multi-timeframe HMM ensemble configuration.

    NOTE: This system predicts REGIME TRANSITIONS only = not price direction.
    Price direction predictions (BUY/SELL/HOLD) are made in:
    - src/interfaces/base_interfaces.py (AnalysisResult.signal)
    - src/analyst/predictive_ensembles/ensemble_orchestrator.py (global meta-learner)
    - src/training/steps/step4_analyst_labeling_feature_engineering_components/ (triple barrier labeling)

    Returns:
        dict: Configuration dictionary
    """
    return {
        "MULTI_TIMEFRAME_HMM_ENSEMBLE": {
            "enabled": True,
            "timeframes": {
                "1m": {
                    "weight": 0.20,  # High frequency signals for quick reactions
                    "min_samples": 50,
                    "enable_hazard_model": True,
                    "enable_price_prediction": False,
                },
                "5m": {
                    "weight": 0.30,  # Primary timeframe for high leverage trading
                    "min_samples": 50,
                    "enable_hazard_model": True,
                    "enable_price_prediction": False,
                },
                "15m": {
                    "weight": 0.35,  # Higher weight for medium-term trends and stability
                    "min_samples": 50,
                    "enable_hazard_model": True,
                    "enable_price_prediction": False,
                },
                "1h": {
                    "weight": 0.15,  # Lower weight but higher quality signals for trend confirmation
                    "min_samples": 50,
                    "enable_hazard_model": True,
                    "enable_price_prediction": False,  # Hazard models are for regime transitions only
                },
            },
            "meta_learner": {
                "type": "lgbm",  # "lgbm", "random_forest", "logistic"
                "n_estimators": 100,
                "learning_rate": 0.1,
                "max_depth": 6,
                "random_state": 42,
                "verbose": -1,
            },
            "ensemble_method": "stacking",  # "meta_learner", "stacking" (weighted_average is fallback only)
            "dynamic_weighting": {
                "enabled": True,
                "update_frequency": 100,  # Update weights every N predictions
                "performance_window": 1000,  # Keep last N predictions for performance tracking
                "min_weight": 0.1,  # Minimum weight for any timeframe
                "max_weight": 0.5,  # Maximum weight for any timeframe
            },
            "prediction": {
                "min_confidence_threshold": 0.6,
                "default_prediction": "REGIME_CONTINUE",
                "regime_change_threshold": 0.7,
            },
            "training": {
                "cross_validation_folds": 3,
                "test_size": 0.2,
                "random_state": 42,
                "enable_early_stopping": True,
                "patience": 10,
            },
            "model_storage": {
                "base_dir": "models/multi_timeframe_hmm_ensemble",
                "save_metadata": True,
                "save_models": True,
                "compression": "gzip",
            },
            "logging": {
                "level": "INFO",
                "enable_performance_tracking": True,
                "log_predictions": True,
                "log_weight_updates": True,
            },
        },
    }


def get_default_timeframe_configs() -> list[TimeframeConfig]:
    """Get default timeframe configurations.

    Returns:
        List[TimeframeConfig]: List of timeframe configurations
    """
    return [
        TimeframeConfig(
            timeframe="1m",
            weight=0.20,
            min_samples=50,
            enable_hazard_model=True,
            enable_price_prediction=False,
        ),
        TimeframeConfig(
            timeframe="5m",
            weight=0.30,
            min_samples=50,
            enable_hazard_model=True,
            enable_price_prediction=False,
        ),
        TimeframeConfig(
            timeframe="15m",
            weight=0.35,
            min_samples=50,
            enable_hazard_model=True,
            enable_price_prediction=False,
        ),
        TimeframeConfig(
            timeframe="1h",
            weight=0.15,
            min_samples=50,
            enable_hazard_model=True,
            enable_price_prediction=False,
        ),
    ]


def get_default_ensemble_config() -> EnsembleConfig:
    """Get default ensemble configuration.

    Returns:
        EnsembleConfig: Default ensemble configuration
    """
    return EnsembleConfig(
        timeframes=get_default_timeframe_configs(),
        meta_learner_type="lgbm",
        enable_dynamic_weighting=True,
        weight_update_frequency=100,
        min_confidence_threshold=0.6,
        ensemble_method="meta_learner",
    )


def validate_ensemble_config(config: dict[str, Any]) -> bool:
    """Validate ensemble configuration.

    Args:
        config: Configuration dictionary

    Returns:
        bool: True if valid = False otherwise
    """
    try:
        ensemble_config = config.get("MULTI_TIMEFRAME_HMM_ENSEMBLE", {})

        # Check if enabled
        if not ensemble_config.get("enabled", False):
            return False

        # Check timeframes
        timeframes = ensemble_config.get("timeframes", {})
        if not timeframes:
            return False

        # Validate timeframe weights sum to 1.0
        total_weight = sum(tf.get("weight", 0) for tf in timeframes.values())
        if abs(total_weight - 1.0) > 0.01:
            return False

        # Check ensemble method
        ensemble_method = ensemble_config.get("ensemble_method", "")
        valid_methods = ["weighted_average", "meta_learner", "stacking"]
        if ensemble_method not in valid_methods:
            return False

        # Check meta-learner type
        meta_learner_type = ensemble_config.get("meta_learner", {}).get("type", "")
        valid_learner_types = ["lgbm", "random_forest", "logistic"]
        return meta_learner_type in valid_learner_types

    except Exception:
        return False


def get_optimized_timeframe_weights() -> dict[str, float]:
    """Get optimized timeframe weights based on typical market behavior.

    Returns:
        Dict[str, float]: Optimized weights for each timeframe
    """
    return {
        "1m": 0.20,  # Lower weight due to noise
        "5m": 0.25,  # Good balance of signal and noise
        "15m": 0.30,  # Higher weight for medium-term trends
        "30m": 0.25,  # Good for longer-term regime changes
    }


def get_adaptive_weighting_config() -> dict[str, Any]:
    """Get adaptive weighting configuration for dynamic weight updates.

    Returns:
        Dict[str = Any]: Adaptive weighting configuration
    """
    return {
        "enabled": True,
        "update_frequency": 100,
        "performance_window": 1000,
        "min_weight": 0.1,
        "max_weight": 0.5,
        "learning_rate": 0.01,
        "momentum": 0.9,
        "decay_factor": 0.95,
    }
