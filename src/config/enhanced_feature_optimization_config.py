# src/config/enhanced_feature_optimization_config.py

"""
Enhanced Feature Engineering Optimization Configuration

Configuration settings for the enhanced feature engineering optimization system that
optimizes the optimization process itself using RF, SHAP, MI, and multi-objective optimization.
"""

from typing import Any


def get_enhanced_feature_optimization_config() -> dict[str, Any]:
    """
    Get enhanced feature engineering optimization configuration.

    Returns:
        dict: Configuration dictionary
    """
    return {
        "enhanced_feature_optimization": {
            "meta_optimization": {
                "enabled": True,
                "n_trials": 200,
                "cv_folds": 5,
                "random_state": 42,
                "early_stopping_patience": 20,
                "performance_threshold": 0.8,
                "sampler": "tpe",  # "tpe", "random", "cmaes"
                "pruner": "median",  # "median", "hyperband", "percentile"
                "storage": "sqlite:///enhanced_optimization.db",
            },
            "parameter_space_optimization": {
                "enabled": True,
                "mi_threshold": 0.1,
                "correlation_threshold": 0.8,
                "adaptive_sampling": True,
                "space_reduction_factor": 0.5,
                "min_parameter_values": 2,
                "max_parameter_values": 10,
                "importance_weighting": True,
                "diversity_preservation": True,
            },
            "multi_objective": {
                "enabled": True,
                "objectives": ["importance", "stability", "diversity", "efficiency"],
                "weights": [0.4, 0.2, 0.2, 0.2],
                "pareto_front_size": 10,
                "diversity_penalty": 0.1,
                "efficiency_threshold": 0.5,
            },
            "shap_analysis": {
                "n_samples": 1000,
                "max_display": 20,
                "interaction_analysis": True,
                "feature_interactions": True,
                "background_samples": 100,
                "nsamples": 100,
                "l1_reg": "auto",
            },
            "mutual_information": {
                "n_neighbors": 3,
                "random_state": 42,
                "discrete_features": "auto",
                "mi_threshold": 0.05,
                "correlation_threshold": 0.7,
            },
            "performance_tracking": {
                "enabled": True,
                "track_optimization_history": True,
                "track_parameter_importance": True,
                "track_convergence": True,
                "save_intermediate_results": True,
            },
            "regime_optimization": {
                "enabled": True,
                "min_regime_samples": 100,
                "regime_weight_decay": 0.95,
                "cross_regime_validation": True,
                "regime_specific_thresholds": True,
                "regime_importance_weighting": True,
            },
            "output_settings": {
                "save_optimization_results": True,
                "save_parameter_space_analysis": True,
                "save_meta_optimization_history": True,
                "save_multi_objective_results": True,
                "save_performance_analysis": True,
                "output_format": "json",
                "compression": "gzip",
            },
        },
        "enhanced_parameter_ranges": {
            "RSI": {
                "lookback_period": {
                    "range": [5, 60],
                    "step": 5,
                    "importance_weight": 0.8,
                    "efficiency_weight": 0.6,
                },
                "overbought_threshold": {
                    "range": [65, 90],
                    "step": 5,
                    "importance_weight": 0.4,
                    "efficiency_weight": 0.8,
                },
                "oversold_threshold": {
                    "range": [10, 35],
                    "step": 5,
                    "importance_weight": 0.4,
                    "efficiency_weight": 0.8,
                },
            },
            "MACD": {
                "fast_period": {
                    "range": [5, 25],
                    "step": 1,
                    "importance_weight": 0.7,
                    "efficiency_weight": 0.7,
                },
                "slow_period": {
                    "range": [20, 40],
                    "step": 2,
                    "importance_weight": 0.7,
                    "efficiency_weight": 0.7,
                },
                "signal_period": {
                    "range": [5, 15],
                    "step": 1,
                    "importance_weight": 0.5,
                    "efficiency_weight": 0.8,
                },
            },
            "Bollinger_Bands": {
                "lookback_period": {
                    "range": [10, 60],
                    "step": 5,
                    "importance_weight": 0.8,
                    "efficiency_weight": 0.6,
                },
                "std_dev": {
                    "range": [1.0, 3.5],
                    "step": 0.5,
                    "importance_weight": 0.6,
                    "efficiency_weight": 0.9,
                },
                "squeeze_threshold": {
                    "range": [0.05, 0.4],
                    "step": 0.05,
                    "importance_weight": 0.5,
                    "efficiency_weight": 0.9,
                },
            },
        },
        "meta_optimization_strategies": {
            "adaptive_sampling": {
                "enabled": True,
                "initial_samples": 50,
                "adaptive_threshold": 0.1,
                "exploration_factor": 0.3,
                "exploitation_factor": 0.7,
            },
            "parameter_importance_learning": {
                "enabled": True,
                "learning_rate": 0.1,
                "update_frequency": 10,
                "importance_decay": 0.95,
            },
            "multi_objective_balancing": {
                "enabled": True,
                "pareto_front_optimization": True,
                "diversity_maintenance": True,
                "efficiency_constraints": True,
            },
        },
        "performance_metrics": {
            "importance_metrics": {
                "shap_importance": True,
                "permutation_importance": True,
                "feature_importance": True,
                "correlation_importance": True,
            },
            "stability_metrics": {
                "cross_validation_stability": True,
                "temporal_stability": True,
                "regime_stability": True,
                "parameter_sensitivity": True,
            },
            "diversity_metrics": {
                "feature_diversity": True,
                "parameter_diversity": True,
                "regime_diversity": True,
                "correlation_diversity": True,
            },
            "efficiency_metrics": {
                "computational_efficiency": True,
                "memory_efficiency": True,
                "parameter_efficiency": True,
                "convergence_efficiency": True,
            },
        },
        "optimization_constraints": {
            "computational_limits": {
                "max_trials_per_feature": 500,
                "max_time_per_feature": 3600,  # 1 hour
                "max_memory_usage": 8.0,  # GB
                "max_cpu_usage": 0.8,  # 80%
            },
            "quality_constraints": {
                "min_importance_score": 0.1,
                "min_stability_score": 0.6,
                "min_diversity_score": 0.3,
                "min_efficiency_score": 0.5,
            },
            "regime_constraints": {
                "min_regime_coverage": 0.7,
                "min_regime_samples": 100,
                "max_regime_imbalance": 0.3,
            },
        },
    }


def get_meta_optimization_objectives() -> dict[str, Any]:
    """
    Get meta-optimization objective definitions.

    Returns:
        dict: Objective definitions
    """
    return {
        "objectives": {
            "importance": {
                "description": "Feature importance using SHAP analysis",
                "direction": "maximize",
                "weight": 0.4,
                "normalization": "min_max",
                "threshold": 0.1,
            },
            "stability": {
                "description": "Feature stability across cross-validation folds",
                "direction": "maximize",
                "weight": 0.2,
                "normalization": "min_max",
                "threshold": 0.6,
            },
            "diversity": {
                "description": "Feature diversity (inverse correlation)",
                "direction": "maximize",
                "weight": 0.2,
                "normalization": "min_max",
                "threshold": 0.3,
            },
            "efficiency": {
                "description": "Computational efficiency",
                "direction": "maximize",
                "weight": 0.2,
                "normalization": "min_max",
                "threshold": 0.5,
            },
        },
        "constraints": {
            "importance_min": 0.1,
            "stability_min": 0.6,
            "diversity_min": 0.3,
            "efficiency_min": 0.5,
        },
    }


def get_parameter_importance_weights() -> dict[str, float]:
    """
    Get parameter importance weights for different features.

    Returns:
        dict: Parameter importance weights
    """
    return {
        "RSI": {
            "lookback_period": 0.8,
            "overbought_threshold": 0.4,
            "oversold_threshold": 0.4,
        },
        "MACD": {
            "fast_period": 0.7,
            "slow_period": 0.7,
            "signal_period": 0.5,
        },
        "Bollinger_Bands": {
            "lookback_period": 0.8,
            "std_dev": 0.6,
            "squeeze_threshold": 0.5,
        },
        "SMA": {
            "short_period": 0.7,
            "long_period": 0.7,
        },
        "EMA": {
            "short_period": 0.7,
            "long_period": 0.7,
        },
        "ATR": {
            "lookback_period": 0.8,
        },
        "Stochastic": {
            "k_period": 0.7,
            "d_period": 0.5,
            "overbought": 0.4,
            "oversold": 0.4,
        },
        "ADX": {
            "lookback_period": 0.8,
            "threshold": 0.5,
        },
        "CCI": {
            "lookback_period": 0.8,
            "constant": 0.6,
        },
    }


def get_optimization_strategies() -> dict[str, Any]:
    """
    Get optimization strategies for different scenarios.

    Returns:
        dict: Optimization strategies
    """
    return {
        "high_leverage_trading": {
            "objectives": ["importance", "stability", "efficiency"],
            "weights": [0.5, 0.3, 0.2],
            "constraints": {
                "min_importance": 0.2,
                "min_stability": 0.7,
                "min_efficiency": 0.6,
            },
            "parameter_focus": ["lookback_period", "fast_period", "slow_period"],
        },
        "regime_specific": {
            "objectives": ["importance", "diversity", "stability"],
            "weights": [0.4, 0.3, 0.3],
            "constraints": {
                "min_importance": 0.15,
                "min_diversity": 0.4,
                "min_stability": 0.6,
            },
            "parameter_focus": ["lookback_period", "threshold", "std_dev"],
        },
        "multi_timeframe": {
            "objectives": ["importance", "efficiency", "diversity"],
            "weights": [0.4, 0.3, 0.3],
            "constraints": {
                "min_importance": 0.15,
                "min_efficiency": 0.7,
                "min_diversity": 0.3,
            },
            "parameter_focus": ["lookback_period", "signal_period", "k_period"],
        },
    }


def get_enhanced_output_schema() -> dict[str, Any]:
    """
    Get enhanced optimization output schema.

    Returns:
        dict: Output schema
    """
    return {
        "output_schema": {
            "enhanced_optimization_results": {
                "timestamp": "string",
                "symbol": "string",
                "exchange": "string",
                "timeframe": "string",
                "meta_optimization_results": "object",
                "parameter_space_optimization": "object",
                "multi_objective_results": "object",
                "enhanced_optimizations": "object",
                "performance_analysis": "object",
            },
            "meta_optimization_history": {
                "feature_name": "string",
                "trial_number": "integer",
                "parameters": "object",
                "objectives": "object",
                "constraints": "object",
                "status": "string",
            },
            "parameter_importance_analysis": {
                "feature_name": "string",
                "parameter_importance": "object",
                "shap_analysis": "object",
                "correlation_analysis": "object",
                "mi_analysis": "object",
            },
        },
        "file_naming": {
            "enhanced_optimization": "{exchange}_{symbol}_{timeframe}_enhanced_feature_optimization.json",
            "meta_optimization_history": "{exchange}_{symbol}_{timeframe}_meta_optimization_history.json",
            "parameter_importance": "{exchange}_{symbol}_{timeframe}_parameter_importance.json",
            "performance_analysis": "{exchange}_{symbol}_{timeframe}_performance_analysis.json",
        },
    }
