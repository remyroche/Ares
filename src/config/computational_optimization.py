# src/config/computational_optimization.py

"""Computational Optimization Configuration for Enhanced Training Pipeline."""

from typing import Any




def get_backtesting_optimization_config() -> dict[str, Any]:
    """Get backtesting optimization configuration.

    Returns:
        dict: Backtesting optimization configuration

    """
    config = get_computational_optimization_config()
    return config["computational_optimization"]["backtesting"]


def get_model_training_optimization_config() -> dict[str, Any]:
    """Get model training optimization configuration.

    Returns:
        dict: Model training optimization configuration

    """
    config = get_computational_optimization_config()
    return config["computational_optimization"]["model_training"]







