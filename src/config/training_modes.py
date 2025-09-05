"""Training modes configuration."""

def apply_mode_parameters_to_config(*args, **kwargs):
    """Apply mode parameters to config."""
    return {}

def get_step_specific_parameters(*args, **kwargs):
    """Get step specific parameters."""
    return {}

def get_intensity_comparison():
    """Get intensity comparison data."""
    return {}

def get_intensity_percentage():
    """Get intensity percentage data."""
    return {}

def get_mode_recommendations():
    """Get mode recommendations."""
    return {}

def get_training_config_dict():
    """Get training config dictionary."""
    return {}

def get_training_input_dict():
    """Get training input dictionary."""
    return {}

def get_training_mode_config():
    """Get training mode config."""
    return {}

def list_available_modes():
    """List available training modes."""
    return []

def get_step_specific_parameters(step_name: str, mode: str = "blank") -> dict:
    """Get step specific parameters."""
    # Base parameters for all steps
    base_params = {
        "timeout": 1800,  # 30 minutes
        "memory_limit_gb": 8.0,
        "cpu_limit_percent": 90.0,
        "retry_attempts": 3,
        "validation_enabled": True,
        "logging_level": "INFO"
    }
    
    # Step-specific parameters
    step_overrides = {
        "step15_tactician_specialist_training": {
            "timeout": 5400,  # 90 minutes for training
            "memory_limit_gb": 16.0,  # More memory for model training
            "cpu_limit_percent": 95.0,
            "retry_attempts": 2,
            "validation_enabled": True,
            "logging_level": "INFO",
            "model_training": {
                "enable_lightgbm": True,
                "enable_xgboost": True,
                "enable_random_forest": True,
                "enable_calibrated_logistic": True,
                "cross_validation_folds": 5,
                "test_size": 0.2,
                "random_state": 42
            },
            "regime_aware_training": {
                "enabled": True,
                "min_regime_samples": 500,
                "regime_validation_split": 0.2,
                "regime_sr_integration": True,
                "regime_parallel_processing": True
            },
            "sr_integration": {
                "enabled": True,
                "use_optimized_params": True,
                "lookback_bars": 200,
                "min_bars_for_analysis": 20
            }
        }
    }
    
    # Merge base parameters with step-specific overrides
    step_params = base_params.copy()
    if step_name in step_overrides:
        step_params.update(step_overrides[step_name])
    
    return step_params