# src/config/training.py

from typing import Any

from src.config.environment import get_environment_settings
from src.config.constants import DEFAULT_LOOKBACK_DAYS


def get_training_config() -> dict[str, Any]:
    """
    Get the complete training configuration.

    Returns:
        dict: Complete training configuration
    """
    get_environment_settings()

    return {
        # --- Model Training Configuration ---
        "training_pipeline": {
            "n_splits": 5,  # Number of folds for walk-forward validation
            "test_size": 0.2,  # Test set size for each fold
            "validation_size": 0.2,  # Validation set size for each fold
            "min_train_size": 1000,  # Minimum training samples required
            "max_train_size": 50000,  # Maximum training samples to use
        },
        # --- Model Training Parameters ---
        "MODEL_TRAINING": {
            "regularization": {
                "lightgbm": {"l1_alpha": 0.01, "l2_alpha": 0.1, "dropout_rate": 0.1},
                "tensorflow": {
                    "l1_alpha": 0.001,
                    "l2_alpha": 0.01,
                    "dropout_rate": 0.2,
                },
                "sklearn": {
                    "l1_alpha": 0.01,
                    "l2_alpha": 0.1,
                    "elastic_net_ratio": 0.5,
                },
                "tabnet": {
                    "lambda_sparse": 0.001,
                    "reg_lambda": 0.01,
                    "dropout_rate": 0.15,
                },
            },
            "optimization": {
                "hyperparameter_trials": 500,
                "cross_validation_folds": 5,
                "early_stopping_patience": 20,
                "ensemble_weight_optimization": True,
                "feature_selection_method": "recursive_feature_elimination",
                "model_selection_criteria": "sharpe_ratio",
            },
            "advanced_features": {
                "enable_market_regime_detection": True,
                "enable_volatility_regime_modeling": True,
                "enable_correlation_analysis": True,
                "enable_momentum_analysis": True,
                "enable_liquidity_analysis": True,
            },
        },
        # --- Global Data Configuration ---
        "DATA_CONFIG": {
            "default_lookback_days": DEFAULT_LOOKBACK_DAYS,  # Default lookback period for all timeframes (3 years)
            "exclude_recent_days": 2,  # Exclude the most recent N days from the lookback window
        },
        # --- Enhanced Training Configuration ---
        "ENHANCED_TRAINING": {
            "enable_efficiency_optimizations": True,
            "segment_days": 30,  # Days per segment for large datasets
            "chunk_size": 10000,  # Chunk size for memory-efficient processing
            "enable_feature_caching": True,  # Cache computed features in database
            "memory_threshold": 0.8,  # Memory usage threshold for cleanup (80%)
            "cache_expiry_hours": 24,  # Cache expiry time in hours
            "database_cleanup_threshold_mb": 1000,  # Database size threshold for cleanup
            "enable_checkpointing": True,  # Enable training checkpoint/resume
            "max_segment_size": 50000,  # Maximum rows per segment
            "enable_computational_optimization": True,  # Enable computational optimization strategies
            "enable_validators": True,  # Enable step validators
        },
        # --- Labeling & Feature Pipeline Parameters ---
        "vectorized_labelling_orchestrator": {
            # Optimize Triple Barrier parameters before labeling (grid search)
            "optimize_triple_barrier_params": False,
            # Default Triple Barrier parameters
            "profit_take_multiplier": 0.002,
            "stop_loss_multiplier": 0.001,
            "time_barrier_minutes": 30,
            "max_lookahead": 100,
            # Ensure binary labels and parquet saving are enabled by default
            "enable_parquet_saving": True,
            "enable_feature_selection": True,
            "enable_data_normalization": True,
            "enable_stationary_checks": True,
            # Optional search spaces when optimization is enabled
            "pt_candidates": [0.0015, 0.002, 0.003],
            "sl_candidates": [0.001, 0.0015, 0.002],
            "time_barrier_candidates": [15, 30, 60],
            "max_lookahead_candidates": [50, 100, 150]
        },
        # --- Method A: Mixture of Experts Pipeline Controls ---
        "pipeline": {
            "method_a": {
                # If True, Step2 runs Step4 early to materialize L0/L1/L2/L3 before splitting
                "step2_is_leveling": True,
                # What to use for regime splitting in Step3: 'bull_bear_sideways' or 'meta_labels'
                "regime_basis": "meta_labels",
            }
        },
        # --- Method A: Expert Training Configuration ---
        "method_a_mixture_of_experts": {
            "enabled": True,
            # Regime source for expert datasets: 'step2_bull_bear_sideways' or 'meta_labels'
            "regime_source": "meta_labels",
            # When using meta_labels, which columns to use as regimes
            "meta_label_columns": [
                # Example defaults (adjust per asset):
                "sr_breakout_up",
                "sr_breakout_down",
                "sr_bounce_up",
                "sr_bounce_down",
            ],
            # Minimum rows required to train a given expert
            "min_rows_per_expert": 5000,
            # Whether to use strength-weighted combining in live dispatcher
            "use_strength_weighting": True,
            # Mapping from regime/meta to strength column name (if available)
            "strength_columns": {
                "sr_breakout_up": "sr_zone_strength",
                "sr_breakout_down": "sr_zone_strength",
                "sr_bounce_up": "sr_zone_strength",
                "sr_bounce_down": "sr_zone_strength",
            },
        },
        # --- Multi-Timeframe Training Configuration ---
        "MULTI_TIMEFRAME_TRAINING": {
            "enable_parallel_training": True,  # Train timeframes in parallel
            "enable_ensemble": True,  # Create ensemble models across timeframes
            "enable_cross_validation": True,  # Perform cross-timeframe validation
            "ensemble_method": "meta_learner",  # Use meta-learner for optimal weights
            "validation_split": 0.2,  # Validation data split
            "max_parallel_workers": 3,  # Maximum parallel workers
            # Meta-learner configuration for high leverage trading
            "meta_learner": {
                "algorithm": "gradient_boosting",  # Meta-learner algorithm
                "optimization_objective": "sharpe_ratio",  # Optimize for Sharpe ratio
                "high_leverage_mode": True,  # Optimize for high leverage trading
                "short_timeframe_priority": True,  # Prioritize shorter timeframes
                "weight_constraints": {
                    "min_weight": 0.05,  # Minimum weight per timeframe
                    "max_weight": 0.40,  # Maximum weight per timeframe
                    "short_timeframe_bonus": 0.1,  # Bonus weight for short timeframes
                },
                "optimization_trials": 100,  # Meta-learner optimization trials
                "cross_validation_folds": 5,  # Cross-validation folds for meta-learner
            },
            # High leverage trading preferences
            "high_leverage_settings": {
                "prioritize_short_timeframes": True,  # Shorter timeframes more important
                "risk_management": "aggressive",  # Aggressive risk management
                "position_sizing": "dynamic",  # Dynamic position sizing
                "stop_loss_tightness": "tight",  # Tight stop losses
            },
        },
        # --- Timeframe Definitions and Purposes ---
        "TIMEFRAMES": {
            # Short-term timeframes (Intraday Trading)
            "1m": {
                "purpose": "Ultra-short-term scalping and high-frequency trading",
                "trading_style": "scalping",
                "feature_set": "ultra_short_term",
                "optimization_trials": 20,  # Fewer trials for speed
                "description": "Captures micro-movements and immediate market reactions",
            },
            "5m": {
                "purpose": "Short-term scalping and momentum trading",
                "trading_style": "scalping",
                "feature_set": "short_term",
                "optimization_trials": 25,
                "description": "Identifies short-term momentum and breakout patterns",
            },
            "15m": {
                "purpose": "Intraday swing trading and momentum analysis",
                "trading_style": "intraday_swing",
                "feature_set": "intraday",
                "optimization_trials": 30,
                "description": "Balances noise reduction with responsiveness to intraday moves",
            },
            # Medium-term timeframes (Swing Trading)
            "1h": {
                "purpose": "Swing trading and medium-term trend identification",
                "trading_style": "swing_trading",
                "feature_set": "swing",
                "optimization_trials": 40,
                "description": "Primary timeframe for swing trading, captures daily cycles",
            },
            "4h": {
                "purpose": "Medium-term trend analysis and position trading",
                "trading_style": "position_trading",
                "feature_set": "medium_term",
                "optimization_trials": 50,
                "description": "Excellent for trend identification and reducing noise",
            },
            "6h": {
                "purpose": "Extended swing trading and trend confirmation",
                "trading_style": "position_trading",
                "feature_set": "medium_term",
                "optimization_trials": 45,
                "description": "Good for trend confirmation and reducing false signals",
            },
            # Long-term timeframes (Position Trading)
            "1d": {
                "purpose": "Long-term trend analysis and position trading",
                "trading_style": "position_trading",
                "feature_set": "long_term",
                "optimization_trials": 50,
                "description": "Primary timeframe for long-term trend identification",
            },
            "3d": {
                "purpose": "Extended position trading and major trend analysis",
                "trading_style": "position_trading",
                "feature_set": "long_term",
                "optimization_trials": 40,
                "description": "Captures major market cycles and long-term trends",
            },
            "1w": {
                "purpose": "Major trend analysis and long-term investment decisions",
                "trading_style": "investment",
                "feature_set": "investment",
                "optimization_trials": 30,
                "description": "For major market cycle analysis and long-term positioning",
            },
        },
        # --- Predefined Timeframe Sets ---
        "TIMEFRAME_SETS": {
            "scalping": {
                "timeframes": ["1m", "5m", "15m"],
                "description": "Ultra-short-term trading with high frequency",
                "use_case": "High-frequency trading and scalping strategies",
            },
            "intraday": {
                "timeframes": ["1m", "5m", "15m", "1h"],
                "description": "Intraday trading with ultra-short to short-term confirmation levels",
                "use_case": "High-frequency day trading and intraday swing trading",
            },
            "swing": {
                "timeframes": ["1h", "4h", "1d"],
                "description": "Swing trading with trend confirmation",
                "use_case": "Swing trading and medium-term position trading",
            },
            "position": {
                "timeframes": ["4h", "1d", "3d"],
                "description": "Position trading with long-term trend analysis",
                "use_case": "Position trading and long-term trend following",
            },
            "investment": {
                "timeframes": ["1d", "3d", "1w"],
                "description": "Long-term investment and major trend analysis",
                "use_case": "Long-term investment and major market cycle analysis",
            },
            "comprehensive": {
                "timeframes": ["15m", "1h", "4h", "1d"],
                "description": "Comprehensive analysis across multiple time horizons",
                "use_case": "Multi-timeframe analysis for robust trading decisions",
            },
        },
        # --- Default Timeframe Configuration ---
        "DEFAULT_TIMEFRAME_SET": "intraday",  # Use intraday timeframes by default for high leverage
        # --- Two-Tier Decision System Configuration ---
        "TWO_TIER_DECISION": {
            "tier1_timeframes": [
                "1m",
                "5m",
                "15m",
                "1h",
            ],  # All timeframes for direction
            "tier2_timeframes": ["1m", "5m"],  # Only shortest for timing
            "direction_threshold": 0.7,  # Threshold for trade direction
            "timing_threshold": 0.8,  # Threshold for precise timing
            "high_leverage_mode": True,
            "enable_two_tier": True,  # Enable two-tier decision system
        },
        # --- Enhanced Ensemble Configuration ---
        "ENHANCED_ENSEMBLE": {
            "enable_enhanced_ensembles": True,
            "model_types": ["xgboost", "lstm", "random_forest"],
            "multi_timeframe_integration": True,
            "confidence_integration": True,
            "liquidation_risk_integration": True,
            "meta_learner_config": {
                "model_type": "lightgbm",
                "n_estimators": 100,
                "learning_rate": 0.1,
                "max_depth": 6,
                "random_state": 42,
            },
        },
        # --- Meta-Labeling System Defaults ---
        "meta_labeling": {
            "enable_analyst_labels": True,
            "enable_tactician_labels": True,
            "pattern_detection": {
                "volatility_threshold": 0.02,
                "momentum_threshold": 0.01,
                "volume_threshold": 1.5,
                "bb_edge_low": 0.2,
                "bb_edge_high": 0.8,
                "bb_mid_low": 0.3,
                "bb_mid_high": 0.7,
                "bb_width_compression": 0.05,
                "bb_width_triangle": 0.03,
                "trend_momentum_strong": 0.02,
                "breakout_momentum": 0.01,
                "failed_break_momentum": 0.005,
                "rsi_overbought": 70,
                "rsi_oversold": 30,
                "sr_lookback": 50,
                "sr_near_pct": 0.003,
                "sr_break_pct": 0.0005,
                "default_activation_threshold": 0.5,
            },
            "entry_prediction": {
                "prediction_horizon": 5,
                "max_adverse_excursion": 0.02,
            },
        },
        # --- Regime Integration via Meta-Labels ---
        "multi_timeframe_regime_integration": {
            "enable_propagation": True,
            "analysis_timeframe": "1h",
            "smoothing_window": 5,
            "candidate_labels": [
                "STRONG_TREND_CONTINUATION",
                "EXHAUSTION_REVERSAL",
                "RANGE_MEAN_REVERSION",
                "BREAKOUT_SUCCESS",
                "BREAKOUT_FAILURE",
                "MOMENTUM_IGNITION",
                "VOLATILITY_COMPRESSION",
                "VOLATILITY_EXPANSION",
                "SR_TOUCH",
                "SR_BOUNCE",
                "SR_BREAK",
                "IGNITION_BAR",
            ],
        },
        # --- Regime-Specific TP/SL Optimizer (Meta-Label Driven) ---
        "regime_specific_tpsl_optimizer": {
            "n_trials": 100,
            "min_trades": 20,
            "optimization_metric": "sharpe_ratio",
            "analysis_timeframe": "30m",
            "candidate_labels": [
                "STRONG_TREND_CONTINUATION",
                "EXHAUSTION_REVERSAL",
                "RANGE_MEAN_REVERSION",
                "BREAKOUT_SUCCESS",
                "BREAKOUT_FAILURE",
                "MOMENTUM_IGNITION",
                "VOLATILITY_COMPRESSION",
                "VOLATILITY_EXPANSION",
                "SR_TOUCH",
                "SR_BOUNCE",
                "SR_BREAK",
                "IGNITION_BAR",
            ],
        },
    }


def get_model_training_config() -> dict[str, Any]:
    """
    Get model training configuration.

    Returns:
        dict: Model training configuration
    """
    training_config = get_training_config()
    return training_config.get("MODEL_TRAINING", {})


def get_enhanced_training_config() -> dict[str, Any]:
    """
    Get enhanced training configuration.

    Returns:
        dict: Enhanced training configuration
    """
    training_config = get_training_config()
    return training_config.get("ENHANCED_TRAINING", {})


def get_multi_timeframe_training_config() -> dict[str, Any]:
    """
    Get multi-timeframe training configuration.

    Returns:
        dict: Multi-timeframe training configuration
    """
    training_config = get_training_config()
    return training_config.get("MULTI_TIMEFRAME_TRAINING", {})


def get_timeframes_config() -> dict[str, Any]:
    """
    Get timeframes configuration.

    Returns:
        dict: Timeframes configuration
    """
    training_config = get_training_config()
    return training_config.get("TIMEFRAMES", {})


def get_timeframe_sets_config() -> dict[str, Any]:
    """
    Get timeframe sets configuration.

    Returns:
        dict: Timeframe sets configuration
    """
    training_config = get_training_config()
    return training_config.get("TIMEFRAME_SETS", {})


def get_two_tier_decision_config() -> dict[str, Any]:
    """
    Get two-tier decision configuration.

    Returns:
        dict: Two-tier decision configuration
    """
    training_config = get_training_config()
    return training_config.get("TWO_TIER_DECISION", {})


def get_enhanced_ensemble_config() -> dict[str, Any]:
    """
    Get enhanced ensemble configuration.

    Returns:
        dict: Enhanced ensemble configuration
    """
    training_config = get_training_config()
    return training_config.get("ENHANCED_ENSEMBLE", {})


def get_training_pipeline_config() -> dict[str, Any]:
    """
    Get training pipeline configuration.

    Returns:
        dict: Training pipeline configuration
    """
    training_config = get_training_config()
    return training_config.get("training_pipeline", {})


def get_data_config() -> dict[str, Any]:
    """
    Get data configuration.

    Returns:
        dict: Data configuration
    """
    training_config = get_training_config()
    return training_config.get("DATA_CONFIG", {})


def get_default_timeframe_set() -> str:
    """
    Get the default timeframe set.

    Returns:
        str: Default timeframe set name
    """
    training_config = get_training_config()
    return training_config.get("DEFAULT_TIMEFRAME_SET", "intraday")


def get_timeframe_config(timeframe: str) -> dict[str, Any]:
    """
    Get configuration for a specific timeframe.

    Args:
        timeframe: The timeframe to get configuration for

    Returns:
        dict: Timeframe configuration
    """
    timeframes_config = get_timeframes_config()
    return timeframes_config.get(timeframe, {})


def get_timeframe_set_config(set_name: str) -> dict[str, Any]:
    """
    Get configuration for a specific timeframe set.

    Args:
        set_name: The timeframe set name

    Returns:
        dict: Timeframe set configuration
    """
    timeframe_sets_config = get_timeframe_sets_config()
    return timeframe_sets_config.get(set_name, {})
