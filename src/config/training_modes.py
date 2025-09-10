from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass
import yaml
from pathlib import Path

"""Training modes configuration."""

@dataclass
class TrainingModeConfig:
    """Configuration for a training mode."""
    description: str
    lookback_days: int
    training_mode: str
    enable_blank_training_mode: bool
    enable_light_training_mode: bool
    enable_full_training_mode: bool
    data_collection: Dict[str, Any]
    feature_engineering: Dict[str, Any]
    model_training: Dict[str, Any]
    validation: Dict[str, Any]
    optimization: Dict[str, Any]
    computational_intensity: str
    estimated_duration_minutes: int
    exclude_recent_days: int = 0
    min_data_points: int = 1000
    enable_advanced_model_training: bool = True
    enable_ensemble_training: bool = True
    enable_multi_timeframe_training: bool = True
    enable_adaptive_training: bool = True

def _load_training_modes_config() -> Dict[str, Any]:
    """Load training modes configuration from YAML file."""
    config_path = Path(__file__).parent.parent.parent / "config" / "training_modes.yaml"
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        # Fallback to default configuration
        return _get_default_training_modes_config()

def _get_default_training_modes_config() -> Dict[str, Any]:
    """Get default training modes configuration."""
    return {
        "training_modes": {
            "full": {
                "description": "Production mode - Complete training with full dataset",
                "lookback_days": 730,
                "training_mode": "full",
                "enable_blank_training_mode": False,
                "enable_light_training_mode": False,
                "enable_full_training_mode": True,
                "computational_intensity": "high",
                "estimated_duration_minutes": 240,
                "data_collection": {"enable_all_exchanges": True},
                "feature_engineering": {"enable_all_features": True},
                "model_training": {"max_trials": 200, "n_trials": 100},
                "validation": {"monte_carlo_samples": 10000, "ab_test_rounds": 10},
                "optimization": {"optuna_trials": 200, "optuna_timeout": 3600}
            },
            "blank": {
                "description": "Quick testing mode - All features/models with shorter lookback",
                "lookback_days": 180,
                "training_mode": "blank",
                "enable_blank_training_mode": True,
                "enable_light_training_mode": False,
                "enable_full_training_mode": False,
                "computational_intensity": "medium",
                "estimated_duration_minutes": 60,
                "data_collection": {"enable_all_exchanges": False},
                "feature_engineering": {"enable_all_features": True},
                "model_training": {"max_trials": 50, "n_trials": 25},
                "validation": {"monte_carlo_samples": 1000, "ab_test_rounds": 3},
                "optimization": {"optuna_trials": 50, "optuna_timeout": 900}
            },
            "light": {
                "description": "Development mode - Minimal data with all features/models",
                "lookback_days": 10,
                "training_mode": "light",
                "enable_blank_training_mode": False,
                "enable_light_training_mode": True,
                "enable_full_training_mode": False,
                "computational_intensity": "low",
                "estimated_duration_minutes": 15,
                "data_collection": {"enable_all_exchanges": False},
                "feature_engineering": {"enable_all_features": True},
                "model_training": {"max_trials": 10, "n_trials": 5},
                "validation": {"monte_carlo_samples": 100, "ab_test_rounds": 2},
                "optimization": {"optuna_trials": 10, "optuna_timeout": 300}
            }
        }
    }

def get_intensity_percentage(training_mode: str) -> float:
    """Get intensity percentage for a training mode."""
    intensity_map = {
        "full": 1.0,    # 100% intensity
        "blank": 0.1,   # 10% intensity
        "light": 0.05   # 5% intensity
    }
    return intensity_map.get(training_mode, 1.0)

def get_training_mode_config(training_mode: str) -> TrainingModeConfig:
    """Get training mode configuration."""
    config_data = _load_training_modes_config()
    modes = config_data.get("training_modes", {})
    
    if training_mode not in modes:
        raise ValueError(f"Unknown training mode: {training_mode}")
    
    mode_data = modes[training_mode]
    
    return TrainingModeConfig(
        description=mode_data.get("description", ""),
        lookback_days=mode_data.get("lookback_days", 30),
        training_mode=mode_data.get("training_mode", training_mode),
        enable_blank_training_mode=mode_data.get("enable_blank_training_mode", False),
        enable_light_training_mode=mode_data.get("enable_light_training_mode", False),
        enable_full_training_mode=mode_data.get("enable_full_training_mode", False),
        data_collection=mode_data.get("data_collection", {}),
        feature_engineering=mode_data.get("feature_engineering", {}),
        model_training=mode_data.get("model_training", {}),
        validation=mode_data.get("validation", {}),
        optimization=mode_data.get("optimization", {}),
        computational_intensity=mode_data.get("computational_intensity", "medium"),
        estimated_duration_minutes=mode_data.get("estimated_duration_minutes", 60),
        exclude_recent_days=mode_data.get("exclude_recent_days", 0),
        min_data_points=mode_data.get("min_data_points", 1000),
        enable_advanced_model_training=mode_data.get("enable_advanced_model_training", True),
        enable_ensemble_training=mode_data.get("enable_ensemble_training", True),
        enable_multi_timeframe_training=mode_data.get("enable_multi_timeframe_training", True),
        enable_adaptive_training=mode_data.get("enable_adaptive_training", True)
    )

def get_intensity_comparison() -> Dict[str, Dict[str, Any]]:
    """Get intensity comparison data for all modes."""
    modes = ["full", "blank", "light"]
    comparison = {}
    
    for mode in modes:
        try:
            config = get_training_mode_config(mode)
            intensity_pct = get_intensity_percentage(mode)
            
            comparison[mode] = {
                "intensity_percentage": intensity_pct,
                "max_trials": config.model_training.get("max_trials", 100),
                "n_trials": config.model_training.get("n_trials", 50),
                "estimated_duration_minutes": config.estimated_duration_minutes,
                "lookback_days": config.lookback_days,
                "monte_carlo_samples": config.validation.get("monte_carlo_samples", 1000),
                "ab_test_rounds": config.validation.get("ab_test_rounds", 5),
                "optuna_trials": config.optimization.get("optuna_trials", 50)
            }
        except ValueError:
            continue
    
    return comparison

def get_mode_recommendations() -> Dict[str, str]:
    """Get mode recommendations."""
    return {
        "full": "Use for production training with complete datasets. Provides highest accuracy but longest training time.",
        "blank": "Use for testing and validation. Good balance between speed and accuracy with 10% intensity.",
        "light": "Use for development and quick iterations. Fastest training with 5% intensity."
    }

def list_available_modes() -> Dict[str, str]:
    """List available training modes."""
    return {
        "full": "Production mode - Complete training with full dataset (100% intensity)",
        "blank": "Testing mode - All features with reduced data (10% intensity)",
        "light": "Development mode - Minimal data for quick iterations (5% intensity)"
    }

def apply_mode_parameters_to_config(config: Dict[str, Any], training_mode: str) -> Dict[str, Any]:
    """Apply mode parameters to configuration."""
    mode_config = get_training_mode_config(training_mode)
    intensity_pct = get_intensity_percentage(training_mode)
    
    # Apply intensity scaling to key parameters
    if "model_training" in config:
        model_config = config["model_training"]
        if "max_trials" in model_config:
            model_config["max_trials"] = int(model_config["max_trials"] * intensity_pct)
        if "n_trials" in model_config:
            model_config["n_trials"] = int(model_config["n_trials"] * intensity_pct)
        if "epochs" in model_config:
            model_config["epochs"] = int(model_config["epochs"] * intensity_pct)
    
    if "validation" in config:
        validation_config = config["validation"]
        if "monte_carlo_samples" in validation_config:
            validation_config["monte_carlo_samples"] = int(validation_config["monte_carlo_samples"] * intensity_pct)
        if "ab_test_rounds" in validation_config:
            validation_config["ab_test_rounds"] = int(validation_config["ab_test_rounds"] * intensity_pct)
    
    if "optimization" in config:
        opt_config = config["optimization"]
        if "optuna_trials" in opt_config:
            opt_config["optuna_trials"] = int(opt_config["optuna_trials"] * intensity_pct)
        if "optuna_timeout" in opt_config:
            opt_config["optuna_timeout"] = int(opt_config["optuna_timeout"] * intensity_pct)
    
    return config

def get_training_config_dict(training_mode: str) -> Dict[str, Any]:
    """Get training configuration dictionary for a mode."""
    mode_config = get_training_mode_config(training_mode)
    return {
        "lookback_days": mode_config.lookback_days,
        "data_collection": mode_config.data_collection,
        "feature_engineering": mode_config.feature_engineering,
        "model_training": mode_config.model_training,
        "validation": mode_config.validation,
        "optimization": mode_config.optimization,
        "intensity_percentage": get_intensity_percentage(training_mode)
    }

def get_training_input_dict(training_mode: str) -> Dict[str, Any]:
    """Get training input dictionary for a mode."""
    mode_config = get_training_mode_config(training_mode)
    return {
        "training_mode": training_mode,
        "lookback_days": mode_config.lookback_days,
        "intensity_percentage": get_intensity_percentage(training_mode),
        "computational_intensity": mode_config.computational_intensity,
        "estimated_duration_minutes": mode_config.estimated_duration_minutes
    }

def get_step_specific_parameters(step_name: str, mode: str='blank') -> dict:
    """Get step specific parameters."""
    base_params = {'timeout': 1800, 'memory_limit_gb': 8.0, 'cpu_limit_percent': 90.0, 'retry_attempts': 3, 'validation_enabled': True, 'logging_level': 'INFO'}
    step_overrides = {'step15_tactician_specialist_training': {'timeout': 5400, 'memory_limit_gb': 16.0, 'cpu_limit_percent': 95.0, 'retry_attempts': 2, 'validation_enabled': True, 'logging_level': 'INFO', 'model_training': {'enable_lightgbm': True, 'enable_xgboost': True, 'enable_random_forest': True, 'enable_calibrated_logistic': True, 'cross_validation_folds': 5, 'test_size': 0.2, 'random_state': 42}, 'regime_aware_training': {'enabled': True, 'min_regime_samples': 500, 'regime_validation_split': 0.2, 'regime_sr_integration': True, 'regime_parallel_processing': True}, 'sr_integration': {'enabled': True, 'use_optimized_params': True, 'lookback_bars': 200, 'min_bars_for_analysis': 20}}}
    step_params = base_params.copy()
    if step_name in step_overrides:
        step_params.update(step_overrides[step_name])
    return step_params