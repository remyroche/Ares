# src/config/computational_optimization.py

"""
Computational Optimization Configuration for Enhanced Training Pipeline.
"""

from typing import Any


def get_computational_optimization_config() -> dict[str, Any]:
    """
    Get computational optimization configuration.

    Returns:
        dict: Computational optimization configuration
    """
    return {
        "computational_optimization": {
            # Caching configuration
            "enable_caching": True,
            "max_cache_size": 1000,
            "cache_ttl_hours": 24,
            # Parallel processing configuration
            "enable_parallelization": True,
            "max_workers": None,  # Auto-detect if None
            "chunk_size": 1000,
            # Early stopping configuration
            "enable_early_stopping": True,
            "patience": 10,
            "min_trials": 20,
            # Surrogate models configuration
            "enable_surrogate_models": True,
            "expensive_trials": 50,
            "update_frequency": 10,
            # Memory management configuration
            "enable_memory_management": True,
            "memory_threshold": 0.8,
            "cleanup_frequency": 100,
            # Progressive evaluation configuration
            "enable_progressive_evaluation": True,
            "evaluation_stages": [
                (0.1, 0.3),  # 10% data, 30% weight
                (0.3, 0.5),  # 30% data, 50% weight
                (1.0, 1.0),  # 100% data, 100% weight
            ],
            # Model complexity scaling
            "enable_adaptive_complexity": True,
            "complexity_levels": {
                "light": {"n_estimators": 50, "max_depth": 3},
                "medium": {"n_estimators": 100, "max_depth": 6},
                "heavy": {"n_estimators": 200, "max_depth": 10},
            },
            # Backtesting optimization
            "backtesting": {
                "enable_cached_backtesting": True,
                "enable_progressive_evaluation": True,
                "enable_parallel_backtesting": True,
                "max_backtest_workers": 4,
                "backtest_timeout_seconds": 300,
            },
            # Model training optimization
            "model_training": {
                "enable_incremental_training": True,
                "enable_adaptive_complexity": True,
                "model_cache_size": 100,
                "warm_start_threshold": 0.8,
            },
            # Feature engineering optimization
            "feature_engineering": {
                "enable_precomputed_features": True,
                "enable_feature_caching": True,
                "feature_cache_size": 500,
                "enable_memory_efficient_data": True,
            },
            # Multi-objective optimization
            "multi_objective": {
                "enable_surrogate_models": True,
                "enable_adaptive_sampling": True,
                "surrogate_model_type": "gaussian_process",
                "expensive_evaluation_ratio": 0.2,
            },
            # Memory management
            "memory_management": {
                "enable_memory_monitoring": True,
                "memory_threshold": 0.8,
                "cleanup_frequency": 100,
                "enable_garbage_collection": True,
                "max_memory_usage_mb": 8000,
            },
            # Performance monitoring
            "performance_monitoring": {
                "enable_performance_tracking": True,
                "track_cache_hits": True,
                "track_memory_usage": True,
                "track_optimization_time": True,
                "performance_log_interval": 100,
            },
            # Phase-specific configurations
            "phases": {
                "phase_1_quick_wins": {
                    "enable_caching": True,
                    "enable_early_stopping": True,
                    "enable_memory_optimization": True,
                },
                "phase_2_medium_term": {
                    "enable_parallel_processing": True,
                    "enable_surrogate_models": True,
                    "enable_progressive_evaluation": True,
                },
                "phase_3_advanced": {
                    "enable_adaptive_complexity": True,
                    "enable_advanced_surrogates": True,
                    "enable_multi_objective_optimization": True,
                },
            },
            # Expected performance improvements
            "expected_improvements": {
                "backtesting_time_reduction": 0.75,  # 75% reduction
                "model_training_time_reduction": 0.60,  # 60% reduction
                "feature_engineering_time_reduction": 0.90,  # 90% reduction
                "overall_time_reduction": 0.70,  # 70% reduction
                "memory_usage_reduction": 0.40,  # 40% reduction
            },
        },
    }


def get_optimization_phase_config(phase: str) -> dict[str, Any]:
    """
    Get configuration for a specific optimization phase.

    Args:
        phase: The optimization phase ('phase_1_quick_wins', 'phase_2_medium_term', 'phase_3_advanced')

    Returns:
        dict: Phase-specific configuration
    """
    config = get_computational_optimization_config()
    phases = config["computational_optimization"]["phases"]
    return phases.get(phase, {})


def get_backtesting_optimization_config() -> dict[str, Any]:
    """
    Get backtesting optimization configuration.

    Returns:
        dict: Backtesting optimization configuration
    """
    config = get_computational_optimization_config()
    return config["computational_optimization"]["backtesting"]


def get_model_training_optimization_config() -> dict[str, Any]:
    """
    Get model training optimization configuration.

    Returns:
        dict: Model training optimization configuration
    """
    config = get_computational_optimization_config()
    return config["computational_optimization"]["model_training"]


def get_feature_engineering_optimization_config() -> dict[str, Any]:
    """
    Get feature engineering optimization configuration.

    Returns:
        dict: Feature engineering optimization configuration
    """
    config = get_computational_optimization_config()
    return config["computational_optimization"]["feature_engineering"]


def get_multi_objective_optimization_config() -> dict[str, Any]:
    """
    Get multi-objective optimization configuration.

    Returns:
        dict: Multi-objective optimization configuration
    """
    config = get_computational_optimization_config()
    return config["computational_optimization"]["multi_objective"]


def get_memory_management_config() -> dict[str, Any]:
    """
    Get memory management configuration.

    Returns:
        dict: Memory management configuration
    """
    config = get_computational_optimization_config()
    return config["computational_optimization"]["memory_management"]


def get_performance_monitoring_config() -> dict[str, Any]:
    """
    Get performance monitoring configuration.

    Returns:
        dict: Performance monitoring configuration
    """
    config = get_computational_optimization_config()
    return config["computational_optimization"]["performance_monitoring"]


def get_expected_improvements() -> dict[str, float]:
    """
    Get expected performance improvements.

    Returns:
        dict: Expected performance improvements
    """
    config = get_computational_optimization_config()
    return config["computational_optimization"]["expected_improvements"]


def is_optimization_enabled(optimization_type: str) -> bool:
    """
    Check if a specific optimization type is enabled.

    Args:
        optimization_type: The optimization type to check

    Returns:
        bool: True if enabled, False otherwise
    """
    config = get_computational_optimization_config()
    optimization_config = config["computational_optimization"]

    if optimization_type == "caching":
        return optimization_config.get("enable_caching", False)
    if optimization_type == "parallelization":
        return optimization_config.get("enable_parallelization", False)
    if optimization_type == "early_stopping":
        return optimization_config.get("enable_early_stopping", False)
    if optimization_type == "surrogate_models":
        return optimization_config.get("enable_surrogate_models", False)
    if optimization_type == "memory_management":
        return optimization_config.get("enable_memory_management", False)
    if optimization_type == "progressive_evaluation":
        return optimization_config.get("enable_progressive_evaluation", False)
    if optimization_type == "adaptive_complexity":
        return optimization_config.get("enable_adaptive_complexity", False)
    return False


def get_optimization_statistics() -> dict[str, Any]:
    """
    Get optimization statistics and status.

    Returns:
        dict: Optimization statistics
    """
    config = get_computational_optimization_config()
    optimization_config = config["computational_optimization"]

    enabled_optimizations = []
    for key, value in optimization_config.items():
        if key.startswith("enable_") and value:
            enabled_optimizations.append(key.replace("enable_", ""))

    return {
        "enabled_optimizations": enabled_optimizations,
        "total_optimizations": len(enabled_optimizations),
        "expected_improvements": get_expected_improvements(),
        "configuration": optimization_config,
    }
