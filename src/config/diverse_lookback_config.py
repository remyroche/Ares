from __future__ import annotations
# src/config/diverse_lookback_config.py

"""
Diverse Lookback Period Configuration

Configuration settings for finding 2-3 lookback periods for each feature that deliver
meaningful yet significantly different information.
"""

from typing import Any


def get_diverse_lookback_config() -> dict[str, Any]:
    """
    Get diverse lookback period optimization configuration.

    Returns:
        dict: Configuration dictionary
    """
    return {
        "diverse_lookback_optimization": {
            "target_periods_per_feature": 3,
            "min_periods_per_feature": 2,
            "max_periods_per_feature": 3,
            "diversity_threshold": 0.3,  # Minimum correlation difference
            "meaningful_threshold": 0.1,  # Minimum SHAP importance
            "correlation_threshold": 0.7,  # Maximum correlation between periods
            "information_diversity_weight": 0.4,
            "signal_strength_weight": 0.4,
            "correlation_penalty_weight": 0.2,
            "min_data_points": 1000,
            "regime_min_samples": 100,
            "early_stopping_patience": 10,
            "performance_threshold": 0.8,
        },
        "lookback_ranges": {
            "RSI": {
                "min": 5,
                "max": 50,
                "step": 2,
                "description": "RSI lookback periods for momentum analysis",
                "expected_insights": ["Short-term momentum", "Medium-term trend", "Long-term trend"],
            },
            "MACD_fast": {
                "min": 5,
                "max": 25,
                "step": 1,
                "description": "MACD fast period for quick signal generation",
                "expected_insights": ["Quick momentum", "Fast trend changes", "Short-term signals"],
            },
            "MACD_slow": {
                "min": 20,
                "max": 40,
                "step": 2,
                "description": "MACD slow period for trend confirmation",
                "expected_insights": ["Trend confirmation", "Medium-term trend", "Signal filtering"],
            },
            "Bollinger_Bands": {
                "min": 10,
                "max": 50,
                "step": 2,
                "description": "Bollinger Bands lookback for volatility analysis",
                "expected_insights": ["Volatility regime", "Price extremes", "Mean reversion"],
            },
            "SMA_short": {
                "min": 3,
                "max": 20,
                "step": 1,
                "description": "Short SMA for immediate trend detection",
                "expected_insights": ["Immediate trend", "Quick reversals", "Short-term support/resistance"],
            },
            "SMA_long": {
                "min": 20,
                "max": 100,
                "step": 5,
                "description": "Long SMA for major trend identification",
                "expected_insights": ["Major trend", "Long-term support/resistance", "Trend strength"],
            },
            "EMA_short": {
                "min": 3,
                "max": 20,
                "step": 1,
                "description": "Short EMA for responsive trend detection",
                "expected_insights": ["Responsive trend", "Quick signals", "Short-term momentum"],
            },
            "EMA_long": {
                "min": 20,
                "max": 100,
                "step": 5,
                "description": "Long EMA for major trend confirmation",
                "expected_insights": ["Major trend confirmation", "Long-term momentum", "Trend persistence"],
            },
            "ATR": {
                "min": 5,
                "max": 30,
                "step": 1,
                "description": "ATR for volatility measurement",
                "expected_insights": ["Volatility regime", "Risk assessment", "Position sizing"],
            },
            "Stochastic_k": {
                "min": 5,
                "max": 30,
                "step": 1,
                "description": "Stochastic %K for momentum analysis",
                "expected_insights": ["Momentum extremes", "Overbought/oversold", "Divergence detection"],
            },
            "Stochastic_d": {
                "min": 3,
                "max": 10,
                "step": 1,
                "description": "Stochastic %D for signal confirmation",
                "expected_insights": ["Signal confirmation", "Momentum smoothing", "Trend confirmation"],
            },
            "ADX": {
                "min": 5,
                "max": 30,
                "step": 1,
                "description": "ADX for trend strength measurement",
                "expected_insights": ["Trend strength", "Market regime", "Directional movement"],
            },
            "CCI": {
                "min": 5,
                "max": 30,
                "step": 1,
                "description": "CCI for cyclical analysis",
                "expected_insights": ["Cyclical extremes", "Mean reversion", "Momentum cycles"],
            },
        },
        "diversity_metrics": {
            "correlation_analysis": {
                "enabled": True,
                "threshold": 0.7,
                "penalty_weight": 0.3,
            },
            "information_diversity": {
                "enabled": True,
                "shap_analysis": True,
                "mutual_information": True,
                "feature_importance": True,
            },
            "signal_complementarity": {
                "enabled": True,
                "frequency_analysis": True,
                "regime_capture": True,
                "market_insight_diversity": True,
            },
        },
        "market_insight_categories": {
            "short_term": {
                "period_range": [3, 10],
                "insights": ["Immediate momentum", "Quick reversals", "Short-term signals"],
                "weight": 0.3,
            },
            "medium_term": {
                "period_range": [11, 25],
                "insights": ["Trend detection", "Momentum confirmation", "Medium-term patterns"],
                "weight": 0.4,
            },
            "long_term": {
                "period_range": [26, 100],
                "insights": ["Major trends", "Long-term support/resistance", "Market regime"],
                "weight": 0.3,
            },
        },
        "regime_specific_optimization": {
            "enabled": True,
            "min_regime_samples": 100,
            "regime_weight_decay": 0.95,
            "cross_regime_validation": True,
            "regime_specific_thresholds": True,
            "regime_insight_mapping": {
                "trending_up": {
                    "preferred_periods": ["medium_term", "long_term"],
                    "insights": ["Trend continuation", "Momentum strength", "Support levels"],
                },
                "trending_down": {
                    "preferred_periods": ["medium_term", "long_term"],
                    "insights": ["Trend continuation", "Momentum weakness", "Resistance levels"],
                },
                "high_volatility": {
                    "preferred_periods": ["short_term", "medium_term"],
                    "insights": ["Volatility spikes", "Quick reversals", "Risk management"],
                },
                "low_volatility": {
                    "preferred_periods": ["medium_term", "long_term"],
                    "insights": ["Range-bound markets", "Mean reversion", "Breakout detection"],
                },
            },
        },
        "output_settings": {
            "save_diverse_periods": True,
            "save_diversity_analysis": True,
            "save_information_content": True,
            "save_regime_specific": True,
            "output_format": "json",
            "include_visualizations": True,
        },
    }


def get_diverse_period_selection_strategy() -> dict[str, Any]:
    """
    Get strategy for selecting diverse periods.

    Returns:
        dict: Selection strategy
    """
    return {
        "selection_strategy": {
            "greedy_algorithm": {
                "enabled": True,
                "start_with_best": True,
                "diversity_optimization": True,
                "information_preservation": True,
            },
            "pareto_optimization": {
                "enabled": True,
                "objectives": ["information_score", "diversity_score", "complementarity_score"],
                "weights": [0.4, 0.4, 0.2],
            },
            "clustering_approach": {
                "enabled": False,
                "n_clusters": 3,
                "cluster_selection": "best_per_cluster",
            },
        },
        "diversity_measures": {
            "correlation_diversity": {
                "measure": "inverse_correlation",
                "weight": 0.4,
                "threshold": 0.7,
            },
            "information_diversity": {
                "measure": "shap_importance_difference",
                "weight": 0.3,
                "threshold": 0.1,
            },
            "temporal_diversity": {
                "measure": "period_spacing",
                "weight": 0.2,
                "threshold": 0.3,
            },
            "regime_diversity": {
                "measure": "regime_capture_difference",
                "weight": 0.1,
                "threshold": 0.2,
            },
        },
        "complementarity_analysis": {
            "frequency_complementarity": {
                "enabled": True,
                "analysis_method": "fourier_transform",
                "complementarity_threshold": 0.3,
            },
            "regime_complementarity": {
                "enabled": True,
                "regime_capture_analysis": True,
                "regime_switching_detection": True,
            },
            "signal_complementarity": {
                "enabled": True,
                "signal_timing_analysis": True,
                "signal_confirmation_analysis": True,
            },
        },
    }


def get_period_insight_mapping() -> dict[str, Any]:
    """
    Get mapping of periods to market insights.

    Returns:
        dict: Period to insight mapping
    """
    return {
        "period_insights": {
            "RSI": {
                "short_periods": {
                    "range": [5, 10],
                    "insights": ["Quick momentum changes", "Short-term overbought/oversold", "Immediate reversals"],
                    "use_cases": ["Scalping", "Quick entries", "Momentum trading"],
                },
                "medium_periods": {
                    "range": [11, 20],
                    "insights": ["Trend momentum", "Medium-term cycles", "Divergence detection"],
                    "use_cases": ["Swing trading", "Trend following", "Mean reversion"],
                },
                "long_periods": {
                    "range": [21, 50],
                    "insights": ["Major trend confirmation", "Long-term cycles", "Market regime"],
                    "use_cases": ["Position trading", "Trend confirmation", "Market analysis"],
                },
            },
            "MACD": {
                "fast_periods": {
                    "range": [5, 15],
                    "insights": ["Quick signal generation", "Fast trend changes", "Momentum shifts"],
                    "use_cases": ["Quick entries", "Momentum trading", "Signal generation"],
                },
                "slow_periods": {
                    "range": [20, 40],
                    "insights": ["Trend confirmation", "Signal filtering", "Major trend detection"],
                    "use_cases": ["Trend confirmation", "Signal filtering", "Major trend analysis"],
                },
            },
            "Bollinger_Bands": {
                "short_periods": {
                    "range": [10, 20],
                    "insights": ["Quick volatility changes", "Short-term extremes", "Immediate mean reversion"],
                    "use_cases": ["Volatility trading", "Quick reversals", "Range trading"],
                },
                "medium_periods": {
                    "range": [21, 35],
                    "insights": ["Volatility regime", "Medium-term extremes", "Trend volatility"],
                    "use_cases": ["Volatility analysis", "Trend volatility", "Risk management"],
                },
                "long_periods": {
                    "range": [36, 50],
                    "insights": ["Major volatility cycles", "Long-term extremes", "Market regime volatility"],
                    "use_cases": ["Market analysis", "Long-term volatility", "Regime detection"],
                },
            },
        },
        "complementary_insights": {
            "momentum_trend": {
                "description": "Combining momentum and trend indicators",
                "period_combinations": [
                    {"RSI": "short", "SMA": "long"},
                    {"MACD_fast": "short", "MACD_slow": "long"},
                    {"Stochastic": "short", "EMA": "long"},
                ],
                "insights": ["Momentum within trends", "Trend confirmation with momentum", "Divergence detection"],
            },
            "volatility_trend": {
                "description": "Combining volatility and trend indicators",
                "period_combinations": [
                    {"ATR": "short", "SMA": "long"},
                    {"Bollinger_Bands": "medium", "EMA": "long"},
                    {"ATR": "long", "ADX": "medium"},
                ],
                "insights": ["Trend strength with volatility", "Volatility regime trends", "Risk-adjusted trends"],
            },
            "cycle_trend": {
                "description": "Combining cyclical and trend indicators",
                "period_combinations": [
                    {"CCI": "short", "SMA": "long"},
                    {"Stochastic": "medium", "EMA": "long"},
                    {"RSI": "medium", "ADX": "long"},
                ],
                "insights": ["Cycles within trends", "Trend cycle analysis", "Cycle-based entries"],
            },
        },
    }


def get_high_leverage_period_priorities() -> dict[str, Any]:
    """
    Get period priorities for high leverage trading.

    Returns:
        dict: High leverage period priorities
    """
    return {
        "high_leverage_priorities": {
            "risk_management": {
                "priority": "high",
                "preferred_periods": ["short", "medium"],
                "insights": ["Quick risk assessment", "Immediate volatility", "Fast position sizing"],
                "indicators": ["ATR", "Bollinger_Bands", "Stochastic"],
            },
            "signal_speed": {
                "priority": "high",
                "preferred_periods": ["short", "medium"],
                "insights": ["Quick signal generation", "Fast momentum detection", "Immediate reversals"],
                "indicators": ["RSI", "MACD_fast", "Stochastic_k"],
            },
            "trend_confirmation": {
                "priority": "medium",
                "preferred_periods": ["medium", "long"],
                "insights": ["Trend confirmation", "Signal filtering", "Major trend detection"],
                "indicators": ["MACD_slow", "SMA_long", "EMA_long"],
            },
            "volatility_analysis": {
                "priority": "medium",
                "preferred_periods": ["short", "medium"],
                "insights": ["Volatility regime", "Risk assessment", "Position sizing"],
                "indicators": ["ATR", "Bollinger_Bands", "ADX"],
            },
        },
        "leverage_specific_settings": {
            "conservative": {
                "leverage_range": [10, 25],
                "period_preferences": ["medium", "long"],
                "diversity_weight": 0.5,
                "stability_weight": 0.5,
            },
            "moderate": {
                "leverage_range": [25, 50],
                "period_preferences": ["short", "medium"],
                "diversity_weight": 0.4,
                "stability_weight": 0.6,
            },
            "aggressive": {
                "leverage_range": [50, 100],
                "period_preferences": ["short", "medium"],
                "diversity_weight": 0.3,
                "stability_weight": 0.7,
            },
        },
    }
