# src/config/feature_engineering_optimization_config.py
"""Feature Engineering Optimization Configuration.

Configuration settings for the feature engineering optimization system that uses Random
Forest + SHAP for correlation analysis and mutual importance matrix.
"""

from typing import Any, Dict, List


def get_feature_engineering_optimization_config() -> dict[str, Any]:
    """Get feature engineering optimization configuration.

    Returns:
        dict: Configuration dictionary
    """
    return {
        "feature_engineering_optimization": {
            "enabled": True,
            "optimization_settings": {
                "n_trials": 100,
                "cv_folds": 5,
                "random_state": 42,
                "correlation_threshold": 0.8,
                "mi_threshold": 0.1,
                "top_k_parameters": 3,
                "min_data_points": 1000,
                "regime_min_samples": 100,
            },
            "feature_parameters": {
                "RSI": {
                    "lookback_period": [7, 14, 21, 30, 50],
                    "overbought_threshold": [70, 75, 80, 85],
                    "oversold_threshold": [15, 20, 25, 30],
                },
                "MACD": {
                    "fast_period": [8, 12, 16, 20],
                    "slow_period": [20, 26, 30, 34],
                    "signal_period": [7, 9, 11, 13],
                },
                "Bollinger_Bands": {
                    "lookback_period": [10, 20, 30, 50],
                    "std_dev": [1.5, 2.0, 2.5, 3.0],
                    "squeeze_threshold": [0.1, 0.2, 0.3, 0.4],
                },
                "SMA": {
                    "short_period": [5, 10, 15, 20],
                    "long_period": [20, 30, 50, 100],
                },
                "EMA": {
                    "short_period": [5, 10, 15, 20],
                    "long_period": [20, 30, 50, 100],
                },
                "ATR": {"lookback_period": [7, 14, 21, 30]},
                "Stochastic": {
                    "k_period": [7, 14, 21, 30],
                    "d_period": [3, 5, 7, 9],
                    "overbought": [70, 75, 80, 85],
                    "oversold": [15, 20, 25, 30],
                },
                "ADX": {
                    "lookback_period": [7, 14, 21, 30],
                    "threshold": [20, 25, 30, 35],
                },
                "CCI": {
                    "lookback_period": [7, 14, 21, 30],
                    "constant": [0.015, 0.02, 0.025, 0.03],
                },
            },
            "shap_settings": {
                "n_samples": 1000,
                "max_display": 20,
                "plot_type": "bar",
                "feature_importance_method": "mean_abs",
            },
            "mutual_information_settings": {
                "n_neighbors": 3,
                "random_state": 42,
                "discrete_features": "auto",
            },
            "correlation_analysis": {
                "method": "pearson",
                "min_periods": 1,
                "correlation_threshold": 0.8,
                "multicollinearity_threshold": 0.95,
            },
            "regime_optimization": {
                "enabled": True,
                "min_regime_samples": 100,
                "regime_weight_decay": 0.95,
                "cross_regime_validation": True,
            },
            "output_settings": {
                "save_optimization_results": True,
                "save_correlation_matrix": True,
                "save_shap_plots": True,
                "save_mutual_information": True,
                "output_format": "json",
            },
        },
        "high_leverage_trading": {
            "min_leverage": 10,
            "max_leverage": 100,
            "target_leverage": 25,
            "max_drawdown_threshold": 0.05,  # 5% max drawdown for high leverage
            "volatility_threshold": 0.02,  # 2% daily volatility threshold
            "signal_quality_threshold": 0.6,
            "timeframe_analysis_window": 30,  # days
        },
        "timeframe_analysis": {
            "min_data_points": 1000,
            "volatility_lookback": 20,
            "correlation_threshold": 0.7,
            "signal_decay_factor": 0.95,
            "ensemble_weight_min": 0.05,
            "ensemble_weight_max": 0.5,
            "timeframes_to_analyze": ["1m", "5m", "15m", "30m", "1h"],
            "leverage_ranges": {
                "conservative": (10, 25),
                "moderate": (25, 50),
                "aggressive": (50, 100),
            },
        },
    }


def get_optimized_timeframe_config() -> dict[str, Any]:
    """Get optimized timeframe configuration for high leverage trading.

    Returns:
        dict: Optimized timeframe configuration
    """
    return {
        "optimized_timeframes": {
            "1m": {
                "weight": 0.20,
                "description": "High frequency signals for quick reactions",
                "leverage_suitability": "high",
                "volatility_tolerance": "low",
            },
            "5m": {
                "weight": 0.30,
                "description": "Primary timeframe for high leverage trading",
                "leverage_suitability": "very_high",
                "volatility_tolerance": "medium",
            },
            "15m": {
                "weight": 0.35,
                "description": "Higher weight for medium-term trends and stability",
                "leverage_suitability": "high",
                "volatility_tolerance": "high",
            },
            "1h": {
                "weight": 0.15,
                "description": "Lower weight but higher quality signals for trend confirmation",
                "leverage_suitability": "medium",
                "volatility_tolerance": "very_high",
            },
        },
        "excluded_timeframes": {
            "30m": {
                "reason": "Not relevant for high leverage trading (10x-100x)",
                "issues": [
                    "Too slow for high leverage position management",
                    "Poor signal-to-noise ratio for quick trades",
                    "Inadequate volatility capture for high leverage scenarios",
                ],
                "alternatives": ["5m", "15m"],
            }
        },
        "leverage_specific_settings": {
            "10x_25x": {
                "primary_timeframes": ["5m", "15m"],
                "secondary_timeframes": ["1m", "1h"],
                "position_holding_time": "short",
                "stop_loss_tightness": "tight",
            },
            "25x_50x": {
                "primary_timeframes": ["5m"],
                "secondary_timeframes": ["15m"],
                "position_holding_time": "very_short",
                "stop_loss_tightness": "very_tight",
            },
            "50x_100x": {
                "primary_timeframes": ["1m", "5m"],
                "secondary_timeframes": [],
                "position_holding_time": "instant",
                "stop_loss_tightness": "instant",
            },
        },
    }


def get_feature_optimization_validation_rules() -> dict[str, Any]:
    """Get validation rules for feature optimization results.

    Returns:
        dict: Validation rules
    """
    return {
        "validation_rules": {
            "min_importance_score": 0.1,
            "max_correlation": 0.8,
            "min_mutual_information": 0.05,
            "max_multicollinearity": 0.95,
            "min_data_quality": 0.8,
            "max_parameter_combinations": 1000,
            "min_regime_coverage": 0.7,
        },
        "quality_checks": {
            "feature_stability": {
                "enabled": True,
                "threshold": 0.8,
                "method": "cross_validation",
            },
            "parameter_sensitivity": {
                "enabled": True,
                "threshold": 0.1,
                "method": "shap_analysis",
            },
            "regime_consistency": {
                "enabled": True,
                "threshold": 0.6,
                "method": "regime_cross_validation",
            },
        },
    }


def get_optimization_output_schema() -> dict[str, Any]:
    """Get schema for optimization output files.

    Returns:
        dict: Output schema
    """
    return {
        "output_schema": {
            "optimization_results": {
                "timestamp": "string",
                "symbol": "string",
                "exchange": "string",
                "timeframe": "string",
                "global_optimizations": "object",
                "regime_optimizations": "object",
                "correlation_analysis": "object",
                "top_parameters": "object",
            },
            "timeframe_analysis": {
                "timestamp": "string",
                "symbol": "string",
                "exchange": "string",
                "leverage_range": "array",
                "timeframe_analysis": "object",
                "volatility_analysis": "object",
                "signal_quality_analysis": "object",
                "ensemble_optimization": "object",
                "recommendations": "object",
            },
        },
        "file_naming": {
            "feature_optimization": "{exchange}_{symbol}_{timeframe}_feature_optimization.json",
            "timeframe_analysis": "{exchange}_{symbol}_timeframe_analysis.json",
            "correlation_matrix": "{exchange}_{symbol}_{timeframe}_correlation_matrix.json",
            "shap_analysis": "{exchange}_{symbol}_{timeframe}_shap_analysis.json",
        },
    }
