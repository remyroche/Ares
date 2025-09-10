#!/usr/bin/env python3
"""
Intensity Scaler Utility

This module provides utilities for scaling ML training parameters based on execution mode intensity.
Supports 5% (light), 10% (blank), and 100% (full) intensity configurations.
"""

import os
import logging
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class IntensityConfig:
    """Configuration for intensity scaling."""
    intensity_percentage: float
    training_mode: str
    max_trials: int
    n_trials: int
    epochs: int
    batch_size: int
    monte_carlo_samples: int
    ab_test_rounds: int
    optuna_trials: int
    optuna_timeout: int
    cross_validation_folds: int
    ensemble_models: int
    early_stopping_patience: int

def get_intensity_from_environment() -> float:
    """Get intensity percentage from environment variables."""
    if os.getenv("LIGHT_TRAINING_MODE") == "1":
        return 0.05  # 5% intensity
    elif os.getenv("BLANK_TRAINING_MODE") == "1":
        return 0.10  # 10% intensity
    elif os.getenv("FULL_TRAINING_MODE") == "1":
        return 1.0   # 100% intensity
    else:
        return 1.0   # Default to full intensity

def get_training_mode_from_environment() -> str:
    """Get training mode from environment variables."""
    if os.getenv("LIGHT_TRAINING_MODE") == "1":
        return "light"
    elif os.getenv("BLANK_TRAINING_MODE") == "1":
        return "blank"
    elif os.getenv("FULL_TRAINING_MODE") == "1":
        return "full"
    else:
        return "full"

def scale_parameter(value: Union[int, float], intensity_percentage: float, min_value: Optional[int] = None) -> Union[int, float]:
    """Scale a parameter based on intensity percentage."""
    if isinstance(value, int):
        scaled = int(value * intensity_percentage)
        if min_value is not None:
            scaled = max(scaled, min_value)
        return scaled
    else:
        return value * intensity_percentage

def apply_intensity_scaling(config: Dict[str, Any], intensity_percentage: Optional[float] = None) -> Dict[str, Any]:
    """Apply intensity scaling to a configuration dictionary."""
    if intensity_percentage is None:
        intensity_percentage = get_intensity_from_environment()
    
    if intensity_percentage >= 1.0:
        # Full intensity, no scaling needed
        return config
    
    scaled_config = config.copy()
    
    # Scale model training parameters
    if "model_training" in scaled_config:
        model_config = scaled_config["model_training"]
        
        # Scale trials and epochs
        if "max_trials" in model_config:
            model_config["max_trials"] = scale_parameter(model_config["max_trials"], intensity_percentage, min_value=1)
        if "n_trials" in model_config:
            model_config["n_trials"] = scale_parameter(model_config["n_trials"], intensity_percentage, min_value=1)
        if "epochs" in model_config:
            model_config["epochs"] = scale_parameter(model_config["epochs"], intensity_percentage, min_value=1)
        if "batch_size" in model_config:
            model_config["batch_size"] = scale_parameter(model_config["batch_size"], intensity_percentage, min_value=32)
        if "early_stopping_patience" in model_config:
            model_config["early_stopping_patience"] = scale_parameter(model_config["early_stopping_patience"], intensity_percentage, min_value=1)
        if "cross_validation_folds" in model_config:
            model_config["cross_validation_folds"] = scale_parameter(model_config["cross_validation_folds"], intensity_percentage, min_value=2)
        if "ensemble_models" in model_config:
            model_config["ensemble_models"] = scale_parameter(model_config["ensemble_models"], intensity_percentage, min_value=1)
        
        # Scale neural network parameters
        if "nn_hidden_layers" in model_config:
            # Reduce hidden layer sizes
            original_layers = model_config["nn_hidden_layers"]
            if isinstance(original_layers, list):
                scaled_layers = [max(int(layer * intensity_percentage), 16) for layer in original_layers]
                model_config["nn_hidden_layers"] = scaled_layers
        
        # Scale gradient boosting parameters
        if "gb_max_depth" in model_config:
            model_config["gb_max_depth"] = scale_parameter(model_config["gb_max_depth"], intensity_percentage, min_value=2)
        if "gb_n_estimators" in model_config:
            model_config["gb_n_estimators"] = scale_parameter(model_config["gb_n_estimators"], intensity_percentage, min_value=10)
        
        # Scale random forest parameters
        if "rf_max_depth" in model_config:
            model_config["rf_max_depth"] = scale_parameter(model_config["rf_max_depth"], intensity_percentage, min_value=2)
        if "rf_n_estimators" in model_config:
            model_config["rf_n_estimators"] = scale_parameter(model_config["rf_n_estimators"], intensity_percentage, min_value=10)
    
    # Scale validation parameters
    if "validation" in scaled_config:
        validation_config = scaled_config["validation"]
        
        if "monte_carlo_samples" in validation_config:
            validation_config["monte_carlo_samples"] = scale_parameter(validation_config["monte_carlo_samples"], intensity_percentage, min_value=10)
        if "ab_test_rounds" in validation_config:
            validation_config["ab_test_rounds"] = scale_parameter(validation_config["ab_test_rounds"], intensity_percentage, min_value=1)
        if "validation_splits" in validation_config:
            validation_config["validation_splits"] = scale_parameter(validation_config["validation_splits"], intensity_percentage, min_value=2)
    
    # Scale optimization parameters
    if "optimization" in scaled_config:
        opt_config = scaled_config["optimization"]
        
        if "optuna_trials" in opt_config:
            opt_config["optuna_trials"] = scale_parameter(opt_config["optuna_trials"], intensity_percentage, min_value=1)
        if "optuna_timeout" in opt_config:
            opt_config["optuna_timeout"] = scale_parameter(opt_config["optuna_timeout"], intensity_percentage, min_value=60)
        if "max_optimization_time" in opt_config:
            opt_config["max_optimization_time"] = scale_parameter(opt_config["max_optimization_time"], intensity_percentage, min_value=60)
    
    # Scale feature engineering parameters
    if "feature_engineering" in scaled_config:
        fe_config = scaled_config["feature_engineering"]
        
        if "max_interactions" in fe_config:
            fe_config["max_interactions"] = scale_parameter(fe_config["max_interactions"], intensity_percentage, min_value=1)
        if "feature_selection_samples" in fe_config:
            fe_config["feature_selection_samples"] = scale_parameter(fe_config["feature_selection_samples"], intensity_percentage, min_value=1000)
    
    logger.info(f"🔧 Applied intensity scaling: {intensity_percentage*100:.0f}% intensity")
    return scaled_config

def get_intensity_config(intensity_percentage: Optional[float] = None) -> IntensityConfig:
    """Get intensity configuration for the current mode."""
    if intensity_percentage is None:
        intensity_percentage = get_intensity_from_environment()
    
    training_mode = get_training_mode_from_environment()
    
    # Base parameters (full intensity)
    base_max_trials = 200
    base_n_trials = 100
    base_epochs = 100
    base_batch_size = 4096
    base_monte_carlo_samples = 10000
    base_ab_test_rounds = 10
    base_optuna_trials = 200
    base_optuna_timeout = 3600
    base_cv_folds = 5
    base_ensemble_models = 10
    base_early_stopping_patience = 20
    
    return IntensityConfig(
        intensity_percentage=intensity_percentage,
        training_mode=training_mode,
        max_trials=scale_parameter(base_max_trials, intensity_percentage, min_value=1),
        n_trials=scale_parameter(base_n_trials, intensity_percentage, min_value=1),
        epochs=scale_parameter(base_epochs, intensity_percentage, min_value=1),
        batch_size=scale_parameter(base_batch_size, intensity_percentage, min_value=32),
        monte_carlo_samples=scale_parameter(base_monte_carlo_samples, intensity_percentage, min_value=10),
        ab_test_rounds=scale_parameter(base_ab_test_rounds, intensity_percentage, min_value=1),
        optuna_trials=scale_parameter(base_optuna_trials, intensity_percentage, min_value=1),
        optuna_timeout=scale_parameter(base_optuna_timeout, intensity_percentage, min_value=60),
        cross_validation_folds=scale_parameter(base_cv_folds, intensity_percentage, min_value=2),
        ensemble_models=scale_parameter(base_ensemble_models, intensity_percentage, min_value=1),
        early_stopping_patience=scale_parameter(base_early_stopping_patience, intensity_percentage, min_value=1)
    )

def log_intensity_info(intensity_percentage: Optional[float] = None):
    """Log intensity information for debugging."""
    if intensity_percentage is None:
        intensity_percentage = get_intensity_from_environment()
    
    training_mode = get_training_mode_from_environment()
    config = get_intensity_config(intensity_percentage)
    
    logger.info("=" * 60)
    logger.info(f"🎯 INTENSITY CONFIGURATION: {training_mode.upper()} MODE")
    logger.info("=" * 60)
    logger.info(f"Intensity Percentage: {intensity_percentage*100:.0f}%")
    logger.info(f"Max Trials: {config.max_trials}")
    logger.info(f"N Trials: {config.n_trials}")
    logger.info(f"Epochs: {config.epochs}")
    logger.info(f"Batch Size: {config.batch_size}")
    logger.info(f"Monte Carlo Samples: {config.monte_carlo_samples}")
    logger.info(f"A/B Test Rounds: {config.ab_test_rounds}")
    logger.info(f"Optuna Trials: {config.optuna_trials}")
    logger.info(f"Optuna Timeout: {config.optuna_timeout}s")
    logger.info(f"Cross Validation Folds: {config.cross_validation_folds}")
    logger.info(f"Ensemble Models: {config.ensemble_models}")
    logger.info(f"Early Stopping Patience: {config.early_stopping_patience}")
    logger.info("=" * 60)

# Convenience functions for common use cases
def get_scaled_hpo_trials(base_trials: int = 100, intensity_percentage: Optional[float] = None) -> int:
    """Get scaled HPO trials based on intensity."""
    if intensity_percentage is None:
        intensity_percentage = get_intensity_from_environment()
    return scale_parameter(base_trials, intensity_percentage, min_value=1)

def get_scaled_hpo_timeout(base_timeout: int = 3600, intensity_percentage: Optional[float] = None) -> int:
    """Get scaled HPO timeout based on intensity."""
    if intensity_percentage is None:
        intensity_percentage = get_intensity_from_environment()
    return scale_parameter(base_timeout, intensity_percentage, min_value=60)

def get_scaled_epochs(base_epochs: int = 100, intensity_percentage: Optional[float] = None) -> int:
    """Get scaled epochs based on intensity."""
    if intensity_percentage is None:
        intensity_percentage = get_intensity_from_environment()
    return scale_parameter(base_epochs, intensity_percentage, min_value=1)

def get_scaled_batch_size(base_batch_size: int = 4096, intensity_percentage: Optional[float] = None) -> int:
    """Get scaled batch size based on intensity."""
    if intensity_percentage is None:
        intensity_percentage = get_intensity_from_environment()
    return scale_parameter(base_batch_size, intensity_percentage, min_value=32)

def get_scaled_monte_carlo_samples(base_samples: int = 10000, intensity_percentage: Optional[float] = None) -> int:
    """Get scaled Monte Carlo samples based on intensity."""
    if intensity_percentage is None:
        intensity_percentage = get_intensity_from_environment()
    return scale_parameter(base_samples, intensity_percentage, min_value=10)