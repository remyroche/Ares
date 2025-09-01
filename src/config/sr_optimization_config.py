# src/config/sr_optimization_config.py

"""
Configuration for S/R Detection Optimization System

This module provides comprehensive configuration options for the S/R detection
optimization system, including all optimization strategies and parameters.
"""

from typing import Any, Dict, List
from dataclasses import dataclass, field


import @dataclass
@dataclass
class SROptimizationConfig:
    """Configuration for S/R detection optimization."""

    # Basic optimization settings
    n_trials: int = 100
    cv_folds: int = 5
    test_size: float = 0.2
    optimization_timeout: int = 3600  # 1 hour

    # Performance thresholds
    performance_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "min_sharpe_ratio": 0.5,
        "max_drawdown": -0.15,
        "min_win_rate": 0.55,
        "min_profit_factor": 1.3,
        "min_signal_clarity": 0.1,
    })

    # Method weight optimization ranges
    method_weight_ranges: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "fractal_weight": {"min": 0.1, "max": 0.6},
        "volume_weight": {"min": 0.1, "max": 0.5},
        "pivot_weight": {"min": 0.1, "max": 0.4},
        "atr_weight": {"min": 0.05, "max": 0.3},
    })

    # Strength weight optimization ranges
    strength_weight_ranges: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "touch_count_weight": {"min": 0.2, "max": 0.5},
        "total_volume_weight": {"min": 0.1, "max": 0.4},
        "level_age_weight": {"min": 0.1, "max": 0.4},
        "bounce_rate_weight": {"min": 0.1, "max": 0.4},
        "isolation_score_weight": {"min": 0.05, "max": 0.3},
    })

    # DBSCAN optimization ranges
    dbscan_ranges: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "dbscan_eps": {"min": 0.005, "max": 0.02},
        "dbscan_min_samples": {"min": 2, "max": 6, "type": "int"},
    })

    # Timeframe weight optimization ranges
    timeframe_weight_ranges: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "tf_1m_weight": {"min": 0.05, "max": 0.2},
        "tf_5m_weight": {"min": 0.1, "max": 0.25},
        "tf_15m_weight": {"min": 0.15, "max": 0.3},
        "tf_1h_weight": {"min": 0.2, "max": 0.35},
        "tf_4h_weight": {"min": 0.15, "max": 0.3},
        "tf_1d_weight": {"min": 0.05, "max": 0.2},
    })

    # Advanced method optimization ranges
    advanced_ranges: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "fibonacci_sensitivity": {"min": 0.5, "max": 0.9},
        "elliott_confidence_threshold": {"min": 0.4, "max": 0.8},
        "order_flow_hvn_threshold": {"min": 1.2, "max": 2.0},
    })

    # Optimization strategies
    enable_optuna: bool = True
    enable_basic_optimization: bool = True
    enable_cross_validation: bool = True
    enable_out_of_sample_validation: bool = True

    # Market regime optimization
    enable_regime_optimization: bool = False
    regime_periods: List[str] = field(default_factory=lambda: [
        "bull_market", "bear_market", "sideways_market", "volatile_market"
    ])

    # Multi-timeframe optimization
    enable_multi_timeframe_optimization: bool = True
    timeframes: List[str] = field(default_factory=lambda: [
        "1m", "5m", "15m", "1h", "4h", "1d"
    ])

    # Advanced optimization features
    enable_advanced_methods: bool = True
    enable_fibonacci_optimization: bool = True
    enable_elliott_wave_optimization: bool = True
    enable_order_flow_optimization: bool = True

    # Clustering optimization
    enable_dbscan_optimization: bool = True
    enable_silhouette_optimization: bool = True
    enable_noise_filtering_optimization: bool = True

    # Real-time optimization
    enable_real_time_adaptation: bool = False
    adaptation_frequency: str = "1h"  # How often to adapt parameters
    learning_rate: float = 0.01

    # Output and logging
    save_optimization_results: bool = True
    optimization_results_file: str = "optimization_results.json"
    enable_detailed_logging: bool = True
    log_optimization_progress: bool = True

    def to_dict(self) -> Dict[str, Any]:
    pass
    pass
        """Convert configuration to dictionary."""
        return {
            "n_trials": self.n_trials,
            "cv_folds": self.cv_folds,
            "test_size": self.test_size,
            "optimization_timeout": self.optimization_timeout,
            "performance_thresholds": self.performance_thresholds,
            "method_weight_ranges": self.method_weight_ranges,
            "strength_weight_ranges": self.strength_weight_ranges,
            "dbscan_ranges": self.dbscan_ranges,
            "timeframe_weight_ranges": self.timeframe_weight_ranges,
            "advanced_ranges": self.advanced_ranges,
            "enable_optuna": self.enable_optuna,
            "enable_basic_optimization": self.enable_basic_optimization,
            "enable_cross_validation": self.enable_cross_validation,
            "enable_out_of_sample_validation": self.enable_out_of_sample_validation,
            "enable_regime_optimization": self.enable_regime_optimization,
            "regime_periods": self.regime_periods,
            "enable_multi_timeframe_optimization": self.enable_multi_timeframe_optimization,
            "timeframes": self.timeframes,
            "enable_advanced_methods": self.enable_advanced_methods,
            "enable_fibonacci_optimization": self.enable_fibonacci_optimization,
            "enable_elliott_wave_optimization": self.enable_elliott_wave_optimization,
            "enable_order_flow_optimization": self.enable_order_flow_optimization,
            "enable_dbscan_optimization": self.enable_dbscan_optimization,
            "enable_silhouette_optimization": self.enable_silhouette_optimization,
            "enable_noise_filtering_optimization": self.enable_noise_filtering_optimization,
            "enable_real_time_adaptation": self.enable_real_time_adaptation,
            "adaptation_frequency": self.adaptation_frequency,
            "learning_rate": self.learning_rate,
            "save_optimization_results": self.save_optimization_results,
            "optimization_results_file": self.optimization_results_file,
            "enable_detailed_logging": self.enable_detailed_logging,
            "log_optimization_progress": self.log_optimization_progress,
        }


def get_sr_optimization_config() -> SROptimizationConfig:
    pass
    pass
    """Get default S/R optimization configuration."""
    return SROptimizationConfig()


def get_light_optimization_config() -> SROptimizationConfig:
    pass
    pass
    """Get light optimization configuration for quick testing."""
    config = SROptimizationConfig()
    config.n_trials = 20
    config.cv_folds = 3
    config.optimization_timeout = 300  # 5 minutes
    config.enable_advanced_methods = False
    config.enable_multi_timeframe_optimization = False
    config.enable_regime_optimization = False
    config.enable_real_time_adaptation = False
    return config


def get_comprehensive_optimization_config() -> SROptimizationConfig:
    pass
    pass
    """Get comprehensive optimization configuration for production."""
    config = SROptimizationConfig()
    config.n_trials = 500
    config.cv_folds = 10
    config.optimization_timeout = 7200  # 2 hours
    config.enable_advanced_methods = True
    config.enable_multi_timeframe_optimization = True
    config.enable_regime_optimization = True
    config.enable_real_time_adaptation = True
    config.performance_thresholds.update({
        "min_sharpe_ratio": 0.7,
        "max_drawdown": -0.1,
        "min_win_rate": 0.6,
        "min_profit_factor": 1.5,
        "min_signal_clarity": 0.15,
    })
    return config


def get_market_specific_config(market_type: str) -> SROptimizationConfig:
    pass
    pass
    """Get market-specific optimization configuration."""
    config = SROptimizationConfig()

    if market_type == "crypto":
    pass
    pass
        # Crypto markets are more volatile
        config.performance_thresholds.update({
            "min_sharpe_ratio": 0.4,
            "max_drawdown": -0.2,
            "min_win_rate": 0.5,
            "min_profit_factor": 1.2,
        })
        config.dbscan_ranges["dbscan_eps"]["max"] = 0.03  # Higher volatility
        config.timeframe_weight_ranges["tf_1h_weight"]["max"] = 0.4  # More weight on hourly

    elif market_type == "forex":
        # Forex markets are more stable
        config.performance_thresholds.update({
            "min_sharpe_ratio": 0.6,
            "max_drawdown": -0.1,
            "min_win_rate": 0.6,
            "min_profit_factor": 1.4,
        })
        config.dbscan_ranges["dbscan_eps"]["max"] = 0.015  # Lower volatility
        config.timeframe_weight_ranges["tf_4h_weight"]["max"] = 0.4  # More weight on 4h

    elif market_type == "stocks":
        # Stock markets are moderate
        config.performance_thresholds.update({
            "min_sharpe_ratio": 0.5,
            "max_drawdown": -0.15,
            "min_win_rate": 0.55,
            "min_profit_factor": 1.3,
        })
        config.timeframe_weight_ranges["tf_1d_weight"]["max"] = 0.3  # More weight on daily

    return config


def create_optimization_config(
    optimization_level: str = "standard",
    market_type: str = "general",
    custom_settings: Dict[str, Any] = None
) -> SROptimizationConfig:
    """
    Create optimization configuration based on level and market type.

    Args:
        optimization_level: "light", "standard", "comprehensive"
        market_type: "crypto", "forex", "stocks", "general"
        custom_settings: Custom settings to override defaults

    Returns:
        SROptimizationConfig: Configured optimization settings
    """
    # Get base configuration
    if optimization_level == "light":
    pass
    pass
        config = get_light_optimization_config()
    elif optimization_level == "comprehensive":
        config = get_comprehensive_optimization_config()
    else:
        config = get_sr_optimization_config()

    # Apply market-specific settings
    if market_type != "general":
    pass
    pass
        market_config = get_market_specific_config(market_type)
        # Merge relevant settings
        config.performance_thresholds.update(market_config.performance_thresholds)
        config.dbscan_ranges.update(market_config.dbscan_ranges)
        config.timeframe_weight_ranges.update(market_config.timeframe_weight_ranges)

    # Apply custom settings
    if custom_settings:
    pass
    pass
        for key, value in custom_settings.items():
    pass
    pass
            if hasattr(config, key):
    pass
    pass
                setattr(config, key, value)

    return config