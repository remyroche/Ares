# src/config/computational_optimization_config.py

"""
Computational optimization configuration for the enhanced training manager.
Based on the computational_optimization_strategies.md document.
"""

from typing import Any, Dict

# Default computational optimization configuration
COMPUTATIONAL_OPTIMIZATION_CONFIG = {
    "caching": {
        "enabled": True,
        "max_cache_size": 1000,
        "cache_ttl": 3600,  # 1 hour in seconds
        "cache_types": {
            "backtest_results": True,
            "technical_indicators": True,
            "feature_engineering": True,
            "model_predictions": True
        }
    },
    
    "parallelization": {
        "enabled": True,
        "max_workers": 8,
        "chunk_size": 1000,
        "parallel_components": {
            "backtesting": True,
            "feature_engineering": True,
            "model_training": True,
            "regime_classification": True,
            "ensemble_creation": True
        }
    },
    
    "early_stopping": {
        "enabled": True,
        "patience": 10,
        "min_trials": 20,
        "performance_threshold": -0.5,
        "evaluation_stages": [
            {"data_ratio": 0.1, "weight": 0.3},
            {"data_ratio": 0.3, "weight": 0.5},
            {"data_ratio": 1.0, "weight": 1.0}
        ]
    },
    
    "surrogate_models": {
        "enabled": True,
        "expensive_trials": 50,
        "update_frequency": 10,
        "model_types": ["gaussian_process", "random_forest"]
    },
    
    "memory_management": {
        "enabled": True,
        "memory_threshold": 0.8,
        "cleanup_frequency": 100,
        "optimization_strategies": {
            "dataframe_optimization": True,
            "garbage_collection": True,
            "cache_size_limits": True,
            "memory_profiling": True
        }
    },
    
    "data_streaming": {
        "enabled": True,
        "chunk_size": 10000,
        "compression": "snappy",
        "file_formats": {
            "prefer_parquet": True,
            "fallback_csv": True,
            "arrow_support": True
        }
    },
    
    "adaptive_sampling": {
        "enabled": True,
        "initial_samples": 100,
        "top_quartile_sampling": True,
        "perturbation_factor": 0.1,
        "exploration_exploitation_ratio": 0.7
    },
    
    "incremental_training": {
        "enabled": True,
        "model_caching": True,
        "state_reuse": True,
        "checkpoint_frequency": 50
    },
    
    "feature_optimization": {
        "precompute_indicators": True,
        "cache_feature_selection": True,
        "parallel_computation": True,
        "memory_efficient_storage": True
    },
    
    "monitoring": {
        "continuous_monitoring": True,
        "monitoring_interval": 30,  # seconds
        "memory_leak_detection": True,
        "performance_tracking": True,
        "alert_thresholds": {
            "memory_usage_percent": 80,
            "memory_growth_mb": 100,
            "execution_time_factor": 2.0
        }
    }
}

def get_optimization_config(custom_config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Get computational optimization configuration.
    
    Args:
        custom_config: Optional custom configuration to override defaults
        
    Returns:
        Dict containing optimization configuration
    """
    config = COMPUTATIONAL_OPTIMIZATION_CONFIG.copy()
    
    if custom_config:
        # Deep merge custom configuration
        config = _deep_merge_config(config, custom_config)
    
    return config

def _deep_merge_config(base_config: Dict[str, Any], custom_config: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two configuration dictionaries."""
    result = base_config.copy()
    
    for key, value in custom_config.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge_config(result[key], value)
        else:
            result[key] = value
    
    return result

def get_memory_optimization_config() -> Dict[str, Any]:
    """Get memory-specific optimization configuration."""
    return COMPUTATIONAL_OPTIMIZATION_CONFIG["memory_management"]

def get_parallelization_config() -> Dict[str, Any]:
    """Get parallelization-specific configuration."""
    return COMPUTATIONAL_OPTIMIZATION_CONFIG["parallelization"]

def get_caching_config() -> Dict[str, Any]:
    """Get caching-specific configuration."""
    return COMPUTATIONAL_OPTIMIZATION_CONFIG["caching"]

def get_early_stopping_config() -> Dict[str, Any]:
    """Get early stopping configuration."""
    return COMPUTATIONAL_OPTIMIZATION_CONFIG["early_stopping"]

def get_adaptive_sampling_config() -> Dict[str, Any]:
    """Get adaptive sampling configuration."""
    return COMPUTATIONAL_OPTIMIZATION_CONFIG["adaptive_sampling"]

def is_optimization_enabled(optimization_type: str) -> bool:
    """
    Check if a specific optimization is enabled.
    
    Args:
        optimization_type: Type of optimization to check
        
    Returns:
        True if optimization is enabled, False otherwise
    """
    optimization_map = {
        "caching": COMPUTATIONAL_OPTIMIZATION_CONFIG["caching"]["enabled"],
        "parallelization": COMPUTATIONAL_OPTIMIZATION_CONFIG["parallelization"]["enabled"],
        "early_stopping": COMPUTATIONAL_OPTIMIZATION_CONFIG["early_stopping"]["enabled"],
        "memory_management": COMPUTATIONAL_OPTIMIZATION_CONFIG["memory_management"]["enabled"],
        "data_streaming": COMPUTATIONAL_OPTIMIZATION_CONFIG["data_streaming"]["enabled"],
        "adaptive_sampling": COMPUTATIONAL_OPTIMIZATION_CONFIG["adaptive_sampling"]["enabled"],
        "incremental_training": COMPUTATIONAL_OPTIMIZATION_CONFIG["incremental_training"]["enabled"],
        "monitoring": COMPUTATIONAL_OPTIMIZATION_CONFIG["monitoring"]["continuous_monitoring"]
    }
    
    return optimization_map.get(optimization_type, False)

# Performance expectations based on the optimization strategies document
EXPECTED_PERFORMANCE_IMPROVEMENTS = {
    "computational_time_reduction": {
        "backtesting": {"min": 70, "max": 80, "unit": "percent"},
        "model_training": {"min": 50, "max": 60, "unit": "percent"},
        "feature_engineering": {"min": 90, "max": 95, "unit": "percent"},
        "overall": {"min": 60, "max": 70, "unit": "percent"}
    },
    
    "memory_usage_reduction": {
        "data_storage": {"min": 40, "max": 50, "unit": "percent"},
        "model_storage": {"min": 30, "max": 40, "unit": "percent"},
        "overall": {"min": 35, "max": 45, "unit": "percent"}
    },
    
    "quality_metrics": {
        "accuracy": "maintained",
        "robustness": "improved",
        "reliability": "enhanced"
    }
}

def get_performance_expectations() -> Dict[str, Any]:
    """Get expected performance improvements."""
    return EXPECTED_PERFORMANCE_IMPROVEMENTS

# Configuration validation
def validate_optimization_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate optimization configuration.
    
    Args:
        config: Configuration to validate
        
    Returns:
        Validation results with any errors or warnings
    """
    validation_results = {
        "valid": True,
        "errors": [],
        "warnings": []
    }
    
    # Check required sections
    required_sections = ["caching", "parallelization", "memory_management"]
    for section in required_sections:
        if section not in config:
            validation_results["errors"].append(f"Missing required section: {section}")
            validation_results["valid"] = False
    
    # Validate parallelization settings
    if "parallelization" in config:
        parallel_config = config["parallelization"]
        if parallel_config.get("max_workers", 0) <= 0:
            validation_results["errors"].append("max_workers must be greater than 0")
            validation_results["valid"] = False
        
        if parallel_config.get("max_workers", 0) > 32:
            validation_results["warnings"].append("max_workers > 32 may not provide additional benefits")
    
    # Validate memory management settings
    if "memory_management" in config:
        memory_config = config["memory_management"]
        threshold = memory_config.get("memory_threshold", 0.8)
        if not 0.5 <= threshold <= 0.95:
            validation_results["warnings"].append("memory_threshold should be between 0.5 and 0.95")
    
    # Validate caching settings
    if "caching" in config:
        cache_config = config["caching"]
        cache_size = cache_config.get("max_cache_size", 1000)
        if cache_size <= 0:
            validation_results["errors"].append("max_cache_size must be greater than 0")
            validation_results["valid"] = False
    
    return validation_results