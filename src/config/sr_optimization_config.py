# src/config/sr_optimization_config.py

"""
Configuration for S/R Detection Optimization System

This module provides comprehensive configuration options for the S/R detection
optimization system, including all optimization strategies and parameters.
"""

from typing import Any, Dict, List
from dataclasses import dataclass, field


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





