"""
Configuration for ML Common Optimization Module

This module provides centralized configuration for all optimization components.
"""

from __future__ import annotations

from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

# Default configuration for optimization
CONFIG = {
    # Grid search settings
    'grid_points': 5,
    'max_combinations': 1000,
    'random_samples': 100,

    # Bayesian optimization settings
    'bayesian_trials': 100,
    'bayesian_timeout': 300,
    'bayesian_init_points': 10,

    # Cross-validation settings
    'cv_folds': 5,
    'cv_repeats': 3,

    # Parallel processing settings
    'max_workers': 4,
    'chunk_size': 100,

    # Memory optimization
    'memory_limit_mb': 1024,
    'cleanup_interval': 60,

    # Early stopping
    'early_stopping_patience': 10,
    'early_stopping_min_delta': 0.001,

    # Logging
    'log_level': 'INFO',
    'save_results': True,
    'results_dir': 'optimization_results',

    # Feature importance
    'importance_threshold': 0.01,
    'feature_selection_method': 'boruta',

    # Multi-objective optimization
    'pareto_front_size': 20,
    'diversity_weight': 0.3,

    # Regime-specific settings
    'regime_detection_method': 'hybrid',
    'tpsl_optimization_method': 'bayesian',
    'regime_count': 8,
}

def get_config() -> Dict[str, Any]:
    """Get the current configuration."""
    return CONFIG.copy()

def update_config(updates: Dict[str, Any]) -> None:
    """Update configuration with new values."""
    CONFIG.update(updates)
    logger.info(f"Configuration updated: {updates}")

def get_config_value(key: str, default: Any = None) -> Any:
    """Get a specific configuration value."""
    return CONFIG.get(key, default)

__all__ = [
    'CONFIG',
    'get_config',
    'update_config',
    'get_config_value'
]
