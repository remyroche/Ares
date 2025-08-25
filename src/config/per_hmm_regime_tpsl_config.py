#!/usr/bin/env python3
"""
Configuration for Per-HMM Regime Triple Barrier Thresholds and TPSL Parameters Optimization

This module provides comprehensive configuration settings for the per-HMM regime
TPSL optimization system, including parameter bounds, optimization settings,
and regime-specific configurations.
"""

from typing import Dict, Any, Tuple

# Per-HMM Regime TPSL Optimizer Configuration
PER_HMM_REGIME_TPSL_CONFIG = {
    # Optimization settings
    "optimization": {
        "n_trials": 200,  # Number of optimization trials per regime
        "min_trades_per_regime": 30,  # Minimum trades required for valid optimization
        "cv_folds": 5,  # Number of cross-validation folds
        "optimization_metric": "sharpe_ratio",  # Optimization target metric
        "optimization_timeout": 3600,  # Timeout in seconds for optimization
        "parallel_trials": 4,  # Number of parallel trials (if supported)
    },
    
    # Parameter bounds for triple barrier optimization
    "triple_barrier_bounds": {
        "profit_take_multiplier": (0.001, 0.01),  # 0.1% to 1%
        "stop_loss_multiplier": (0.0005, 0.005),  # 0.05% to 0.5%
        "time_barrier_minutes": (15, 120),  # 15 minutes to 2 hours
        "max_lookahead": (50, 200),  # 50 to 200 bars
    },
    
    # Parameter bounds for TPSL optimization
    "tpsl_bounds": {
        "target_pct": (0.002, 0.02),  # 0.2% to 2%
        "stop_pct": (0.001, 0.01),  # 0.1% to 1%
        "risk_reward_ratio": (1.5, 4.0),  # 1.5:1 to 4:1
        "position_sizing_pct": (0.01, 0.05),  # 1% to 5% of capital
    },
    
    # Regime-specific adjustments
    "regime_adjustment_bounds": {
        "volatility_multiplier": (0.5, 2.0),  # Volatility-based scaling
        "momentum_multiplier": (0.8, 1.5),  # Momentum-based scaling
        "regime_confidence_threshold": (0.3, 0.8),  # Minimum confidence for regime
    },
    
    # Regime-specific default parameters
    "regime_defaults": {
        "hmm_cluster_0": {
            "name": "Low Volatility Sideways",
            "description": "Low volatility sideways market with frequent small moves",
            "triple_barrier": {
                "profit_take_multiplier": 0.003,
                "stop_loss_multiplier": 0.002,
                "time_barrier_minutes": 45,
                "max_lookahead": 100
            },
            "tpsl": {
                "target_pct": 0.005,
                "stop_pct": 0.003,
                "risk_reward_ratio": 1.67,
                "position_sizing_pct": 0.02
            },
            "characteristics": {
                "volatility": "low",
                "trend": "sideways",
                "frequency": "high",
                "expected_return": "low",
                "risk_level": "low"
            }
        },
        "hmm_cluster_1": {
            "name": "Moderate Volatility Trending",
            "description": "Moderate volatility trending market with clear direction",
            "triple_barrier": {
                "profit_take_multiplier": 0.005,
                "stop_loss_multiplier": 0.003,
                "time_barrier_minutes": 60,
                "max_lookahead": 120
            },
            "tpsl": {
                "target_pct": 0.008,
                "stop_pct": 0.004,
                "risk_reward_ratio": 2.0,
                "position_sizing_pct": 0.03
            },
            "characteristics": {
                "volatility": "moderate",
                "trend": "trending",
                "frequency": "medium",
                "expected_return": "medium",
                "risk_level": "medium"
            }
        },
        "hmm_cluster_2": {
            "name": "High Volatility Breakout",
            "description": "High volatility breakout market with explosive moves",
            "triple_barrier": {
                "profit_take_multiplier": 0.008,
                "stop_loss_multiplier": 0.004,
                "time_barrier_minutes": 30,
                "max_lookahead": 80
            },
            "tpsl": {
                "target_pct": 0.012,
                "stop_pct": 0.006,
                "risk_reward_ratio": 2.0,
                "position_sizing_pct": 0.025
            },
            "characteristics": {
                "volatility": "high",
                "trend": "breakout",
                "frequency": "low",
                "expected_return": "high",
                "risk_level": "high"
            }
        },
        "hmm_cluster_3": {
            "name": "Extreme Volatility Crisis",
            "description": "Extreme volatility crisis market with panic moves",
            "triple_barrier": {
                "profit_take_multiplier": 0.015,
                "stop_loss_multiplier": 0.008,
                "time_barrier_minutes": 20,
                "max_lookahead": 50
            },
            "tpsl": {
                "target_pct": 0.02,
                "stop_pct": 0.01,
                "risk_reward_ratio": 2.0,
                "position_sizing_pct": 0.015
            },
            "characteristics": {
                "volatility": "extreme",
                "trend": "crisis",
                "frequency": "very_low",
                "expected_return": "very_high",
                "risk_level": "very_high"
            }
        },
        "hmm_cluster_4": {
            "name": "Low Volatility Trending",
            "description": "Low volatility trending market with steady direction",
            "triple_barrier": {
                "profit_take_multiplier": 0.004,
                "stop_loss_multiplier": 0.002,
                "time_barrier_minutes": 90,
                "max_lookahead": 150
            },
            "tpsl": {
                "target_pct": 0.006,
                "stop_pct": 0.003,
                "risk_reward_ratio": 2.0,
                "position_sizing_pct": 0.035
            },
            "characteristics": {
                "volatility": "low",
                "trend": "trending",
                "frequency": "medium",
                "expected_return": "medium_high",
                "risk_level": "low"
            }
        },
        "hmm_cluster_5": {
            "name": "Moderate Volatility Sideways",
            "description": "Moderate volatility sideways market with range-bound moves",
            "triple_barrier": {
                "profit_take_multiplier": 0.004,
                "stop_loss_multiplier": 0.003,
                "time_barrier_minutes": 60,
                "max_lookahead": 100
            },
            "tpsl": {
                "target_pct": 0.007,
                "stop_pct": 0.004,
                "risk_reward_ratio": 1.75,
                "position_sizing_pct": 0.025
            },
            "characteristics": {
                "volatility": "moderate",
                "trend": "sideways",
                "frequency": "high",
                "expected_return": "medium",
                "risk_level": "medium"
            }
        },
        "hmm_cluster_6": {
            "name": "High Volatility Sideways",
            "description": "High volatility sideways market with wide ranges",
            "triple_barrier": {
                "profit_take_multiplier": 0.006,
                "stop_loss_multiplier": 0.004,
                "time_barrier_minutes": 45,
                "max_lookahead": 80
            },
            "tpsl": {
                "target_pct": 0.01,
                "stop_pct": 0.005,
                "risk_reward_ratio": 2.0,
                "position_sizing_pct": 0.02
            },
            "characteristics": {
                "volatility": "high",
                "trend": "sideways",
                "frequency": "medium",
                "expected_return": "medium",
                "risk_level": "high"
            }
        },
        "hmm_cluster_7": {
            "name": "Moderate Volatility Breakout",
            "description": "Moderate volatility breakout market with controlled moves",
            "triple_barrier": {
                "profit_take_multiplier": 0.006,
                "stop_loss_multiplier": 0.004,
                "time_barrier_minutes": 40,
                "max_lookahead": 90
            },
            "tpsl": {
                "target_pct": 0.009,
                "stop_pct": 0.005,
                "risk_reward_ratio": 1.8,
                "position_sizing_pct": 0.03
            },
            "characteristics": {
                "volatility": "moderate",
                "trend": "breakout",
                "frequency": "low",
                "expected_return": "high",
                "risk_level": "medium_high"
            }
        }
    },
    
    # Performance metrics configuration
    "performance_metrics": {
        "primary_metric": "sharpe_ratio",
        "secondary_metrics": ["total_return", "win_rate", "calmar_ratio", "max_drawdown"],
        "risk_free_rate": 0.02,  # 2% annual risk-free rate
        "min_sharpe_ratio": 0.5,  # Minimum acceptable Sharpe ratio
        "max_drawdown_threshold": 0.15,  # Maximum acceptable drawdown (15%)
    },
    
    # Backtesting configuration
    "backtesting": {
        "min_data_points": 1000,  # Minimum data points required for backtesting
        "warmup_period": 100,  # Warmup period for indicators
        "transaction_costs": 0.001,  # 0.1% transaction costs
        "slippage": 0.0005,  # 0.05% slippage
        "commission": 0.001,  # 0.1% commission
    },
    
    # Regime transition handling
    "regime_transitions": {
        "transition_threshold": 0.3,  # Confidence threshold for regime transitions
        "transition_cooldown": 300,  # Cooldown period in seconds between transitions
        "parameter_smoothing": True,  # Smooth parameter changes during transitions
        "smoothing_window": 5,  # Number of periods for parameter smoothing
    },
    
    # Optimization constraints
    "constraints": {
        "min_risk_reward_ratio": 1.2,  # Minimum risk-reward ratio
        "max_position_size": 0.1,  # Maximum position size (10% of capital)
        "min_trade_frequency": 0.1,  # Minimum trade frequency (10% of opportunities)
        "max_trade_frequency": 0.8,  # Maximum trade frequency (80% of opportunities)
        "min_win_rate": 0.4,  # Minimum acceptable win rate (40%)
        "max_win_rate": 0.9,  # Maximum expected win rate (90%)
    },
    
    # Model persistence
    "persistence": {
        "save_optimization_results": True,
        "save_regime_statistics": True,
        "save_performance_history": True,
        "auto_backup": True,
        "backup_interval": 3600,  # Backup every hour
        "max_backups": 24,  # Keep 24 backups
    },
    
    # Logging and monitoring
    "logging": {
        "log_level": "INFO",
        "log_optimization_progress": True,
        "log_regime_transitions": True,
        "log_performance_metrics": True,
        "log_parameter_changes": True,
    },
    
    # Validation settings
    "validation": {
        "validate_parameters": True,
        "validate_regime_consistency": True,
        "validate_performance_metrics": True,
        "cross_validation_enabled": True,
        "out_of_sample_testing": True,
        "walk_forward_analysis": True,
    }
}


def get_regime_config(regime: str) -> Dict[str, Any]:
    """Get configuration for a specific regime.
    
    Args:
        regime: Regime name (e.g., 'hmm_cluster_0')
        
    Returns:
        Dict[str, Any]: Regime-specific configuration
    """
    return PER_HMM_REGIME_TPSL_CONFIG["regime_defaults"].get(regime, {})


def get_optimization_bounds() -> Dict[str, Tuple[float, float]]:
    """Get all optimization parameter bounds.
    
    Returns:
        Dict[str, Tuple[float, float]]: Parameter bounds dictionary
    """
    bounds = {}
    bounds.update(PER_HMM_REGIME_TPSL_CONFIG["triple_barrier_bounds"])
    bounds.update(PER_HMM_REGIME_TPSL_CONFIG["tpsl_bounds"])
    bounds.update(PER_HMM_REGIME_TPSL_CONFIG["regime_adjustment_bounds"])
    return bounds


def get_performance_metrics_config() -> Dict[str, Any]:
    """Get performance metrics configuration.
    
    Returns:
        Dict[str, Any]: Performance metrics configuration
    """
    return PER_HMM_REGIME_TPSL_CONFIG["performance_metrics"]


def get_backtesting_config() -> Dict[str, Any]:
    """Get backtesting configuration.
    
    Returns:
        Dict[str, Any]: Backtesting configuration
    """
    return PER_HMM_REGIME_TPSL_CONFIG["backtesting"]


def get_regime_transition_config() -> Dict[str, Any]:
    """Get regime transition configuration.
    
    Returns:
        Dict[str, Any]: Regime transition configuration
    """
    return PER_HMM_REGIME_TPSL_CONFIG["regime_transitions"]


def get_optimization_constraints() -> Dict[str, Any]:
    """Get optimization constraints.
    
    Returns:
        Dict[str, Any]: Optimization constraints
    """
    return PER_HMM_REGIME_TPSL_CONFIG["constraints"]


def validate_config() -> bool:
    """Validate the configuration for consistency and completeness.
    
    Returns:
        bool: True if configuration is valid
    """
    try:
        # Check required sections
        required_sections = [
            "optimization", "triple_barrier_bounds", "tpsl_bounds",
            "regime_adjustment_bounds", "regime_defaults", "performance_metrics",
            "backtesting", "regime_transitions", "constraints"
        ]
        
        for section in required_sections:
            if section not in PER_HMM_REGIME_TPSL_CONFIG:
                raise ValueError(f"Missing required configuration section: {section}")
        
        # Validate parameter bounds
        bounds = get_optimization_bounds()
        for param, (min_val, max_val) in bounds.items():
            if min_val >= max_val:
                raise ValueError(f"Invalid bounds for {param}: min={min_val}, max={max_val}")
        
        # Validate regime defaults
        for regime, config in PER_HMM_REGIME_TPSL_CONFIG["regime_defaults"].items():
            if "name" not in config:
                raise ValueError(f"Missing name for regime: {regime}")
            if "triple_barrier" not in config:
                raise ValueError(f"Missing triple_barrier config for regime: {regime}")
            if "tpsl" not in config:
                raise ValueError(f"Missing tpsl config for regime: {regime}")
        
        return True
        
    except Exception as e:
        print(f"Configuration validation failed: {e}")
        return False


# Validate configuration on import
if not validate_config():
    raise ValueError("Per-HMM regime Tpsl configuration validation failed")