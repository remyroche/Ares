# src/config.py
import asyncio
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

from dotenv import load_dotenv
from pydantic import Field, validator
from pydantic_settings import BaseSettings

from src.core.config_service import ConfigurationService
from src.utils.error_handler import (
    handle_errors,
    handle_specific_errors,
)
from src.utils.logger import system_logger

# --- Environment Loading ---
dotenv_path = os.path.join(os.path.dirname(__file__), "..", ".env")
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path)
    print(".env file loaded.")
else:
    print(".env file not found. Using environment variables or defaults.")


# ==============================================================================
# Pydantic Settings for Environment Variables
# ==============================================================================
class Settings(BaseSettings):
    """
    Manages all environment-specific settings using Pydantic.
    """

    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    trading_environment: Literal["LIVE", "TESTNET", "PAPER"] = Field(
        default="PAPER",
        env="TRADING_ENVIRONMENT",
    )
    initial_equity: float = Field(default=100.0, env="INITIAL_EQUITY")
    trade_symbol: str = Field(default="ETHUSDT", env="TRADE_SYMBOL")
    exchange_name: str = Field(default="BINANCE", env="EXCHANGE_NAME")
    timeframe: str = Field(default="15m", env="TIMEFRAME")

    

    # --- Gate.io Credentials (loaded from .env) ---
    gateio_api_key: str | None = Field(default=None, env="GATEIO_API_KEY")
    gateio_api_secret: str | None = Field(default=None, env="GATEIO_API_SECRET")

    # --- MEXC Credentials (loaded from .env) ---
    mexc_api_key: str | None = Field(default=None, env="MEXC_API_KEY")
    mexc_api_secret: str | None = Field(default=None, env="MEXC_API_SECRET")

    # --- OKX Credentials (loaded from .env) ---
    okx_api_key: str | None = Field(default=None, env="OKX_API_KEY")
    okx_api_secret: str | None = Field(default=None, env="OKX_API_SECRET")
    okx_password: str | None = Field(default=None, env="OKX_PASSWORD")

    # --- Firestore Credentials (loaded from .env) ---
    google_application_credentials: str | None = Field(
        default=None,
        env="GOOGLE_APPLICATION_CREDENTIALS",
    )
    firestore_project_id: str | None = Field(
        default=None,
        env="FIRESTORE_PROJECT_ID",
    )

    # --- Emailer Credentials (loaded from .env) ---
    email_sender_address: str | None = Field(
        default=None,
        env="EMAIL_SENDER_ADDRESS",
    )
    email_sender_password: str | None = Field(
        default=None,
        env="EMAIL_SENDER_PASSWORD",
    )
    email_recipient_address: str | None = Field(
        default=None,
        env="EMAIL_RECIPIENT_ADDRESS",
    )

    # --- Derived Properties for Convenience ---
    @property
    def is_live_mode(self) -> bool:
        return self.trading_environment == "LIVE"

    

    # --- Validators ---
    

    class Config:
        case_sensitive = True


# --- Global Settings Instance ---
try:
    settings = Settings()
    print(
        f"Configuration loaded successfully. Trading Environment: {settings.trading_environment}, Exchange: {settings.exchange_name}",
    )
except Exception as e:
    print(f"Failed to load or validate configuration: {e}")
    exit(1)


ARES_VERSION = "2.0.0"
LOG_LEVEL = "INFO"


# ==============================================================================
# Main Configuration Dictionary
# ==============================================================================
CONFIG: dict[str, Any] = {
    # --- General System & Trading Parameters ---
    "trading_symbol": settings.trade_symbol,
    "exchange_name": settings.exchange_name,
    "trading_interval": settings.timeframe,
    "initial_equity": settings.initial_equity,
    "taker_fee": 0.0004,
    "maker_fee": 0.0002,
    "state_file": "ares_state.json",
    "lookback_years": 2,  # 2 years of historical data

    # --- Logging Configuration ---
    "logging": {
        "level": "INFO",
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "console_output": True,
        "file_output": True,
        "max_file_size": 10 * 1024 * 1024,  # 10MB
        "backup_count": 5,
        "log_directory": "log",
        "enable_rotation": True,
        "enable_timestamped_files": True,
        "enable_global_logging": True,  # Global log file per session
        "enable_error_logging": True,
        "enable_performance_logging": True,
        "enable_trade_logging": True,
        "enable_system_logging": True,
    },

    # --- Exchange Configurations ---
    "exchanges": {
        "binance": {
            "symbols": ["BTCUSDT", "ETHUSDT"],
            "api_key": None,  # Will be set after ares_config is created
            "api_secret": None,  # Will be set after ares_config is created
        },
        "gateio": {
            "symbols": ["BTCUSDT", "ETHUSDT"],
            "api_key": settings.gateio_api_key,
            "api_secret": settings.gateio_api_secret,
        },
        "mexc": {
            "symbols": ["BTCUSDT", "ETHUSDT"],
            "api_key": settings.mexc_api_key,
            "api_secret": settings.mexc_api_secret,
        },
        "okx": {
            "symbols": ["BTCUSDT", "ETHUSDT"],
            "api_key": settings.okx_api_key,
            "api_secret": settings.okx_api_secret,
            "password": settings.okx_password,
        },
    },

    # --- Data Caching Configuration (filenames are set dynamically below) ---
    "klines_filename": "",
    "agg_trades_filename": "",
    "futures_filename": "",
    "prepared_data_filename": "",
    # --- Script Names ---
    "downloader_script_name": "backtesting/ares_data_downloader.py",
    "preparer_script_name": "backtesting/ares_data_preparer.py",
    "pipeline_script_name": "src/ares_pipeline.py",
    "pipeline_pid_file": "ares_pipeline.pid",
    "restart_flag_file": "restart_pipeline.flag",
    # --- Checkpointing Configuration ---
    "CHECKPOINT_DIR": "checkpoints",
    "OPTIMIZER_CHECKPOINT_FILE": "optimizer_state.pkl",  # Relative to CHECKPOINT_DIR
    "PIPELINE_PROGRESS_FILE": "pipeline_progress.json",  # Relative to CHECKPOINT_DIR
    "WALK_FORWARD_REPORTS_FILE": "walk_forward_reports.json",  # Relative to CHECKPOINT_DIR
    "PREPARED_DATA_CHECKPOINT_FILE": "full_prepared_data.parquet",  # Relative to CHECKPOINT_DIR
    "REGIME_CLASSIFIER_MODEL_PREFIX": "regime_classifier_fold_",  # Prefix for fold-specific models
    "ENSEMBLE_MODEL_PREFIX": "ensemble_fold_",  # Prefix for fold-specific ensemble models
    # --- Reporting Configuration ---
    "DETAILED_TRADE_LOG_FILE": "reports/detailed_trade_log.csv",
    "DAILY_SUMMARY_LOG_FILENAME_FORMAT": "reports/daily_summary_log_%Y-%m.csv",  # New: Monthly filenames for summary reports
    "STRATEGY_PERFORMANCE_LOG_FILENAME_FORMAT": "reports/strategy_performance_log_%Y-%m.csv",  # New: Monthly filenames for strategy reports
    "ERROR_LOG_FILE": "ares_errors.jsonl",  # New: Dedicated error log file
    # --- Database Configuration ---
    "DATABASE_TYPE": "sqlite",  # 'sqlite' only - Firebase removed
    "SQLITE_DB_PATH": "data/ares_local_db.sqlite",  # Path for SQLite database file
    "BACKUP_INTERVAL_HOURS": 24,  # Automatic backup interval
    "BACKUP_RETENTION_DAYS": 30,  # How long to keep backups
    "DB_TYPE": os.getenv("DB_TYPE", "influxdb"),  # Moved inside CONFIG
    # --- InfluxDB Configuration ---
    "INFLUXDB_URL": os.getenv(
        "INFLUXDB_URL",
        "http://localhost:8086",
    ),  # Moved inside CONFIG
    "INFLUXDB_TOKEN": os.getenv(
        "INFLUXDB_TOKEN",
        "your_influxdb_token",
    ),  # Moved inside CONFIG
    "INFLUXDB_ORG": os.getenv("INFLUXDB_ORG", "your_org"),  # Moved inside CONFIG
    "INFLUXDB_BUCKET": os.getenv(
        "INFLUXDB_BUCKET",
        "ares_market_data",
    ),  # Moved inside CONFIG
    # --- MLflow Configuration ---
    "MLFLOW_TRACKING_URI": os.getenv(
        "MLFLOW_TRACKING_URI",
        "file:./mlruns",
    ),  # Moved inside CONFIG
    "MLFLOW_EXPERIMENT_NAME": "Ares_Trading_Models",  # Moved inside CONFIG
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
            "lightgbm": {
                "l1_alpha": 0.01,
                "l2_alpha": 0.1,
                "dropout_rate": 0.1
            },
            "tensorflow": {
                "l1_alpha": 0.001,
                "l2_alpha": 0.01,
                "dropout_rate": 0.2
            },
            "sklearn": {
                "l1_alpha": 0.01,
                "l2_alpha": 0.1,
                "elastic_net_ratio": 0.5
            },
            "tabnet": {
                "lambda_sparse": 0.001,
                "reg_lambda": 0.01,
                "dropout_rate": 0.15
            }
        },
        "optimization": {
            "hyperparameter_trials": 500,
            "cross_validation_folds": 5,
            "early_stopping_patience": 20,
            "ensemble_weight_optimization": True,
            "feature_selection_method": "recursive_feature_elimination",
            "model_selection_criteria": "sharpe_ratio"
        },
        "advanced_features": {
            "enable_market_regime_detection": True,
            "enable_volatility_regime_modeling": True,
            "enable_correlation_analysis": True,
            "enable_momentum_analysis": True,
            "enable_liquidity_analysis": True
        }
    },
    # --- Global Data Configuration ---
    "DATA_CONFIG": {
        "default_lookback_days": 730,  # Default lookback period for all timeframes (2 years)
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
        "tier1_timeframes": ["1m", "5m", "15m", "1h"],  # All timeframes for direction
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
    # --- Pipeline Configuration ---
    "pipeline": {
        "loop_interval_seconds": 10,  # Main loop interval for live trading
        "max_retries": 3,  # Maximum retries for failed operations
        "timeout_seconds": 30,  # Timeout for operations
    },
    # --- Risk Management Configuration ---
    "risk_management": {
        "max_position_size": 0.3,  # Maximum position size as fraction of portfolio (30%) -> across multiple positions if we use several positions for a same movement as our confidence increases
        "max_daily_loss": 0.1,  # Maximum daily loss as fraction of portfolio (10%)
        "max_drawdown": 0.50,  # Maximum drawdown before stopping (50%)
        "kill_switch_threshold": 0.50,  # Loss threshold for kill switch (50%)
        "position_sizing": {
            "confidence_based_scaling": True,  # Enable confidence-based position sizing
            "base_position_size": 0.05,  # Base position size (5% of portfolio)
            "max_positions_per_signal": 5,  # Maximum number of positions for same signal
            "confidence_thresholds": {
                "low_confidence": 0.6,  # Confidence threshold for low confidence
                "medium_confidence": 0.75,  # Confidence threshold for medium confidence
                "high_confidence": 0.85,  # Confidence threshold for high confidence
                "very_high_confidence": 0.95,  # Confidence threshold for very high confidence
            },
            "position_size_multipliers": {
                "low_confidence": 0.5,  # 50% of base size for low confidence
                "medium_confidence": 1.0,  # 100% of base size for medium confidence
                "high_confidence": 1.5,  # 150% of base size for high confidence
                "very_high_confidence": 2.0,  # 200% of base size for very high confidence
            },
            "successive_position_rules": {
                "enable_successive_positions": True,  # Enable multiple positions for high confidence
                "min_confidence_for_successive": 0.85,  # Minimum confidence for successive positions
                "max_successive_positions": 3,  # Maximum successive positions
                "position_spacing_minutes": 15,  # Minutes between successive positions
                "size_reduction_factor": 0.8,  # Each successive position is 80% of previous
                "max_total_exposure": 0.3,  # Maximum total exposure across all positions (30%)
            },
            "volatility_adjustment": {
                "enable_volatility_scaling": True,
                "atr_multiplier": 1.0,
                "volatility_thresholds": {
                    "low_volatility": 0.02,  # 2% ATR for low volatility
                    "medium_volatility": 0.05,  # 5% ATR for medium volatility
                    "high_volatility": 0.10,  # 10% ATR for high volatility
                },
                "volatility_multipliers": {
                    "low_volatility": 1.2,  # Increase size by 20% in low volatility
                    "medium_volatility": 1.0,  # Normal size in medium volatility
                    "high_volatility": 0.7,  # Reduce size by 30% in high volatility
                },
            },
            "regime_based_adjustment": {
                "enable_regime_adjustment": True,
                "regime_multipliers": {
                    "BULL_TREND": 1.2,  # Increase size by 20% in bull trend
                    "BEAR_TREND": 0.8,  # Reduce size by 20% in bear trend
                    "SIDEWAYS_RANGE": 0.9,  # Reduce size by 10% in sideways
                    "HIGH_IMPACT_CANDLE": 0.6,  # Reduce size by 40% in high impact
                    "SR_ZONE_ACTION": 1.1,  # Increase size by 10% in SR zones
                },
            },
            "risk_limits": {
                "max_single_position": 0.15,  # Maximum single position (15%)
                "max_total_exposure": 0.3,  # Maximum total exposure (30%)
                "max_correlation_exposure": 0.2,  # Maximum exposure to correlated assets
                "min_position_size": 0.01,  # Minimum position size (1%)
                "max_leverage": 10.0,  # Maximum leverage allowed
            },
        },
        "dynamic_risk_management": {
            "enable_dynamic_risk": True,
            "drawdown_adjustment": {
                "enable_drawdown_scaling": True,
                "drawdown_thresholds": {
                    "warning": 0.1,  # 10% drawdown - warning
                    "reduction": 0.2,  # 20% drawdown - reduce position sizes
                    "aggressive": 0.3,  # 30% drawdown - aggressive reduction
                    "emergency": 0.4,  # 40% drawdown - emergency mode
                },
                "size_reduction_factors": {
                    "warning": 0.9,  # Reduce by 10% at warning
                    "reduction": 0.7,  # Reduce by 30% at reduction threshold
                    "aggressive": 0.5,  # Reduce by 50% at aggressive threshold
                    "emergency": 0.2,  # Reduce by 80% at emergency threshold
                },
            },
            "daily_loss_adjustment": {
                "enable_daily_loss_scaling": True,
                "daily_loss_thresholds": {
                    "warning": 0.05,  # 5% daily loss - warning
                    "reduction": 0.08,  # 8% daily loss - reduce sizes
                    "emergency": 0.10,  # 10% daily loss - emergency mode
                },
                "size_reduction_factors": {
                    "warning": 0.8,  # Reduce by 20% at warning
                    "reduction": 0.5,  # Reduce by 50% at reduction
                    "emergency": 0.2,  # Reduce by 80% at emergency
                },
            },
        },
    },
    # --- Enhanced Risk Management Configuration ---
    "RISK_MANAGEMENT": {
        "position_sizing": {
            "kelly_criterion_enabled": True,
            "volatility_targeting": True,
            "max_position_size": 0.3,  # 30% max per position
            "dynamic_sizing": {
                "confidence_threshold": 0.7,
                "volatility_multiplier": 0.8,
                "market_regime_adjustment": True
            }
        },
        "stop_loss": {
            "trailing_stop": True,
            "atr_multiplier": 2.0,
            "confidence_based_stop": True,
            "max_loss_per_trade": 0.02,  # 2% max loss per trade
        },
        "take_profit": {
            "dynamic_tp": True,
            "risk_reward_ratio": 2.0,
            "partial_profit_taking": True,
            "profit_targets": [0.01, 0.02, 0.03]  # 1%, 2%, 3%
        }
    },
    # --- Best Parameters (from previous optimization) ---
    "best_params": {
        "lookback_period": 20,
        "rsi_period": 14,
        "rsi_overbought": 70,
        "rsi_oversold": 30,
        "macd_fast": 12,
        "macd_slow": 26,
        "macd_signal": 9,
        "bollinger_period": 20,
        "bollinger_std": 2,
        "atr_period": 14,
        "stop_loss_atr_multiplier": 2.0,
        "take_profit_atr_multiplier": 3.0,
        "position_size_atr_multiplier": 1.0,
    },
    # --- Feature Engineering Configuration ---
    "feature_engineering": {
        "technical_indicators": [
            "rsi",
            "macd",
            "bollinger_bands",
            "atr",
            "stochastic",
            "williams_r",
        ],
        "price_features": [
            "returns",
            "log_returns",
            "volatility",
            "momentum",
        ],
        "volume_features": [
            "volume_sma",
            "volume_ratio",
            "volume_momentum",
        ],
        "regime_features": [
            "regime_label",
            "regime_probability",
            "regime_transition",
        ],
    },
    # --- Model Configuration ---
    "models": {
        "regime_classifier": {
            "type": "hmm",
            "n_components": 3,
            "covariance_type": "full",
            "random_state": 42,
        },
        # --- Unified Regime Classifier Configuration ---
        "unified_regime_classifier": {
            "n_states": 4,  # Number of hidden states
            "n_iter": 100,  # Maximum iterations for HMM training
            "random_state": 42,
            "target_timeframe": "1h",  # Target timeframe for regime classification
            "volatility_period": 10,  # Volatility period for feature calculation
            "min_data_points": 1000,  # Minimum data points for training
        },
        
        # --- Candle Analyzer Configuration ---
        "candle_analyzer": {
            "size_thresholds": {
                "small": 0.5,      # 0.5x average
                "normal": 1.0,      # 1.0x average
                "large": 2.0,       # 2.0x average
                "huge": 3.0,        # 3.0x average
                "extreme": 5.0      # 5.0x average
            },
            "volatility_period": 20,  # Period for volatility calculation
            "volatility_multiplier": 2.0,  # Multiplier for volatility-based thresholds
            "doji_threshold": 0.1,  # 10% of range for doji detection
            "hammer_ratio": 0.3,    # 30% body for hammer pattern
            "shooting_star_ratio": 0.3,  # 30% body for shooting star
            "outlier_threshold": 2.5,  # Standard deviations for outlier detection
            "min_candle_count": 100,  # Minimum candles for analysis
            "use_adaptive_thresholds": True,  # Use volatility-based adaptive thresholds
            "use_volume_confirmation": True,  # Use volume for confirmation
            "use_multi_timeframe": True,  # Enable multi-timeframe analysis
        },
        # --- Regime-Specific TP/SL Optimizer Configuration ---
        "regime_specific_tpsl_optimizer": {
            "n_trials": 100,  # Number of optimization trials
            "min_trades": 20,  # Minimum trades for optimization
            "optimization_metric": "sharpe_ratio",  # Optimization target
            "cache_duration_minutes": 60,  # Cache optimization results
        },
        # --- Multi-Timeframe Regime Integration Configuration ---
        "multi_timeframe_regime_integration": {
            "enable_propagation": True,  # Enable regime propagation across timeframes
            "smoothing_window": 5,  # Smoothing window for regime changes
            "regime_cache_duration_minutes": 15,  # Cache regime for 15 minutes
            "strategic_timeframe": "1h",  # Strategic timeframe for regime classification
        },
        # --- Multi-Timeframe Feature Engineering Configuration ---
        "multi_timeframe_feature_engineering": {
            "enable_mtf_features": True,  # Enable multi-timeframe features
            "enable_timeframe_adaptation": True,  # Adapt indicators to timeframes
            "cache_duration_minutes": 5,  # Cache features for 5 minutes
            "enable_feature_caching": True,  # Enable feature caching
            "max_cache_size": 50,  # Maximum cache entries
            "timeframe_adaptation_rules": {
                "execution_timeframes": ["1m", "5m"],  # Ultra-short-term
                "tactical_timeframes": ["15m"],  # Tactical decision making
                "strategic_timeframes": ["1h"],  # Strategic macro trend
                "additional_timeframes": ["4h", "1d"],  # Additional timeframes
            },
        },
        # --- Advanced Feature Engineering Configuration ---
        "advanced_feature_engineering": {
            "enable_divergence_detection": True,  # Enable divergence detection
            "enable_pattern_recognition": True,  # Enable pattern recognition
            "enable_volume_profile": True,  # Enable volume profile analysis
            "enable_market_microstructure": True,  # Enable market microstructure analysis
            "enable_volatility_targeting": True,  # Enable volatility targeting
            "target_volatility": 0.15,  # Target volatility for position sizing
            "divergence_detection": {
                "rsi_period": 14,  # RSI period for divergence
                "macd_period": 12,  # MACD period for divergence
                "obv_period": 20,  # OBV period for divergence
                "min_peak_distance": 10,  # Minimum distance between peaks
            },
            "pattern_recognition": {
                "double_pattern_tolerance": 0.02,  # 2% tolerance for double patterns
                "head_shoulders_tolerance": 0.05,  # 5% tolerance for H&S
                "triangle_window": 20,  # Window for triangle detection
                "flag_pennant_window": 10,  # Window for flag/pennant detection
            },
            "volume_profile": {
                "price_bins": 50,  # Number of price bins for volume profile
                "hvn_threshold": 0.75,  # High volume node threshold
                "lvn_threshold": 0.25,  # Low volume node threshold
                "poc_calculation_method": "max_volume",  # POC calculation method
            },
            "market_microstructure": {
                "spread_calculation": "bid_ask",  # Spread calculation method
                "imbalance_calculation": "volume_weighted",  # Imbalance calculation
                "depth_levels": 10,  # Number of depth levels to analyze
                "liquidity_threshold": 0.1,  # Liquidity threshold
            },
            "momentum_indicators": {
                "roc_period": 10,  # Rate of Change period
                "willr_period": 14,  # Williams %R period
                "cci_period": 20,  # CCI period
                "mfi_period": 14,  # MFI period
            },
            "sr_zones": {
                "bb_period": 20,  # Bollinger Bands period
                "bb_std": 2,  # Bollinger Bands standard deviation
                "kc_period": 20,  # Keltner Channel period
                "vwap_std_multiplier": 2,  # VWAP standard deviation multiplier
                "sr_peak_distance": 20,  # Distance for S/R peak detection
            },
        },
        "ensemble": {
            "type": "voting",
            "voting_method": "soft",
            "base_models": ["random_forest", "xgboost", "lightgbm"],
        },
        "base_models": {
            "random_forest": {
                "n_estimators": 100,
                "max_depth": 10,
                "random_state": 42,
            },
            "xgboost": {
                "n_estimators": 100,
                "max_depth": 6,
                "learning_rate": 0.1,
                "random_state": 42,
            },
            "lightgbm": {
                "n_estimators": 100,
                "max_depth": 6,
                "learning_rate": 0.1,
                "random_state": 42,
            },
        },
    },
    # --- Backtesting Configuration ---
    "backtesting": {
        "commission": 0.001,  # Commission rate
        "slippage": 0.0005,  # Slippage rate
        "initial_capital": 10000,  # Initial capital for backtesting
        "position_sizing": "fixed",  # Position sizing method
        "risk_per_trade": 0.02,  # Risk per trade as fraction of capital
    },
    # --- ML Target Updater Configuration ---
    "ml_target_updater": {
        "enabled": True,  # Enable continuous target adjustment
        "update_interval_seconds": 30,  # Check for updates every 30 seconds (high leverage)
        "min_time_between_updates_seconds": 10,  # Minimum 10 seconds between updates
        "confidence_threshold_for_update": 0.5,  # Lower confidence threshold for faster response
        "max_target_change_percent": 0.5,  # Allow 50% changes for high leverage responsiveness
        "enable_stop_loss_updates": True,  # Allow stop loss adjustments
        "enable_take_profit_updates": True,  # Allow take profit adjustments
        "trailing_stop_enabled": True,  # Enable trailing stop functionality
        "max_sl_distance_from_entry": 0.03,  # Tighter 3% stop loss for high leverage
        "min_tp_distance_from_current": 0.005,  # Tighter 0.5% minimum TP distance
        "dynamic_adjustment_enabled": True,  # Enable dynamic target adjustment
        "market_condition_adaptation": True,  # Adapt to changing market conditions
        "volatility_based_adjustment": True,  # Adjust based on volatility changes
        "momentum_based_adjustment": True,  # Adjust based on momentum changes
        "regime_based_adjustment": True,  # Adjust based on market regime
        "high_leverage_mode": True,  # Enable high leverage optimizations
        "emergency_update_threshold": 0.02,  # Emergency update if price moves 2%
        "volatility_scaling": True,  # Scale update frequency with volatility
    },
    # --- Global Lookback Configuration ---
    "lookback_config": {
        "default_days": 730,  # 2 years default
        "blank_mode_days": 30,  # 30 days for blank mode
        "min_days": 5,  # Minimum allowed
        "max_days": 2000,  # Maximum allowed
    },
    # --- ML Dynamic Target Predictor Configuration ---
    "ml_dynamic_target_predictor": {
        "enabled": True,  # Enable ML-based dynamic target prediction
        "retrain_interval_hours": 24,  # Retrain models every 24 hours
        "min_samples_for_training": 500,  # Minimum samples required for training
        "validation_split": 0.2,  # Validation split for model training
        "min_tp_multiplier": 0.5,  # Minimum take profit multiplier
        "max_tp_multiplier": 6.0,  # Maximum take profit multiplier
        "min_sl_multiplier": 0.2,  # Minimum stop loss multiplier
        "max_sl_multiplier": 2.0,  # Maximum stop loss multiplier
        "fallback_tp_multiplier": 2.0,  # Fallback take profit multiplier
        "fallback_sl_multiplier": 0.5,  # Fallback stop loss multiplier
        "continuous_learning_enabled": True,  # Enable continuous model learning
        "adaptive_thresholds": True,  # Enable adaptive confidence thresholds
        "market_regime_adaptation": True,  # Adapt to market regime changes
    },
    # --- Email Configuration ---
    "email": {
        "enabled": True,
        "smtp_server": "smtp.gmail.com",
        "smtp_port": 587,
        "use_tls": True,
    },
    # --- Logging Configuration ---
    "logging": {
        "level": "INFO",
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "file_rotation": "1 day",
        "file_retention": "30 days",
    },
    # --- Enhanced Hyperparameter Optimization Configuration ---
    "hyperparameter_optimization": {
        "multi_objective": {
            "enabled": True,
            "objectives": ["sharpe_ratio", "win_rate", "profit_factor"],
            "weights": {
                "sharpe_ratio": 0.50,
                "win_rate": 0.30,
                "profit_factor": 0.20
            },
            "risk_constraints": {
                "max_drawdown_threshold": 0.20,
                "min_win_rate": 0.40,
                "min_profit_factor": 1.2
            }
        },
        "bayesian_optimization": {
            "enabled": True,
            "sampling_strategy": "tpe",  # "tpe", "random", "cmaes", "nsga2"
            "max_trials": 500,
            "patience": 50,
            "min_trials": 20,
            "pruning_threshold": 0.1,
            "acquisition_function": "ei",  # "ei", "pi", "ucb"
            "n_startup_trials": 10,
            "n_ei_candidates": 24
        },
        "adaptive_optimization": {
            "enabled": True,
            "regime_detection": {
                "lookback_window": 50,
                "volatility_threshold": 0.02,
                "trend_threshold": 0.01,
                "regime_stability_threshold": 0.7
            },
            "regime_specific_constraints": {
                "bull": {
                    "tp_multiplier_range": [2.5, 5.0],
                    "sl_multiplier_range": [1.2, 2.5],
                    "position_size_range": [0.10, 0.25]
                },
                "bear": {
                    "tp_multiplier_range": [2.0, 4.5],
                    "sl_multiplier_range": [1.0, 2.2],
                    "position_size_range": [0.08, 0.20]
                },
                "sideways": {
                    "tp_multiplier_range": [1.5, 3.0],
                    "sl_multiplier_range": [0.8, 1.5],
                    "position_size_range": [0.05, 0.15]
                },
                "sr": {
                    "tp_multiplier_range": [1.8, 3.5],
                    "sl_multiplier_range": [0.9, 1.8],
                    "position_size_range": [0.06, 0.18]
                },
                "candle": {
                    "tp_multiplier_range": [1.2, 2.5],
                    "sl_multiplier_range": [0.6, 1.2],
                    "position_size_range": [0.03, 0.12]
                }
            }
        },
        "advanced_features": {
            "early_stopping": True,
            "pruning": True,
            "parameter_importance_analysis": True,
            "convergence_monitoring": True,
            "adaptive_sampling": True,
            "multi_start_optimization": True
        },
        "search_spaces": {
            "model_hyperparameters": {
                "learning_rate": {"type": "float", "low": 1e-4, "high": 1e-1, "log": True},
                "max_depth": {"type": "int", "low": 3, "high": 15},
                "n_estimators": {"type": "int", "low": 50, "high": 1000},
                "subsample": {"type": "float", "low": 0.6, "high": 1.0},
                "colsample_bytree": {"type": "float", "low": 0.6, "high": 1.0},
                "reg_alpha": {"type": "float", "low": 1e-8, "high": 10.0, "log": True},
                "reg_lambda": {"type": "float", "low": 1e-8, "high": 10.0, "log": True},
                "model_type": {"type": "categorical", "choices": ["xgboost", "lightgbm", "catboost", "random_forest", "gradient_boosting", "tabnet", "transformer"]}
            },
            "feature_engineering": {
                "lookback_window": {"type": "int", "low": 5, "high": 200},
                "feature_selection_threshold": {"type": "float", "low": 0.001, "high": 0.1},
                "technical_indicator_periods": {"type": "int", "low": 5, "high": 50}
            },
            "trading_parameters": {
                "tp_multiplier": {"type": "float", "low": 1.2, "high": 10.0},
                "sl_multiplier": {"type": "float", "low": 0.5, "high": 5.0},
                "position_size": {"type": "float", "low": 0.01, "high": 0.5},
                "confidence_threshold": {"type": "float", "low": 0.6, "high": 0.95},
                "reg_alpha": {"type": "float", "low": 1e-8, "high": 10.0, "log": True},
                "reg_lambda": {"type": "float", "low": 1e-8, "high": 10.0, "log": True}
            },
            "ensemble_parameters": {
                "ensemble_size": {"type": "int", "low": 3, "high": 10},
                "ensemble_weight": {"type": "float", "low": 0.1, "high": 0.9},
                "meta_learner_type": {"type": "categorical", "choices": ["lightgbm", "xgboost", "random_forest"]}
            }
        },
        "optimization_schedules": {
            "daily": {
                "enabled": True,
                "time": "02:00",
                "max_trials": 100,
                "focus": "quick_adaptation"
            },
            "weekly": {
                "enabled": True,
                "day": "sunday",
                "time": "03:00",
                "max_trials": 500,
                "focus": "comprehensive_optimization"
            },
            "monthly": {
                "enabled": True,
                "day": 1,
                "time": "04:00",
                "max_trials": 1000,
                "focus": "deep_optimization"
            }
        },
        "performance_metrics": {
            "primary": ["sharpe_ratio", "total_return", "max_drawdown"],
            "secondary": ["win_rate", "profit_factor", "calmar_ratio", "sortino_ratio"],
            "risk_metrics": ["var_95", "cvar_95", "volatility", "beta"],
            "custom_metrics": ["regime_adaptation_score", "parameter_stability_score"]
        },
        "constraints_and_penalties": {
            "hard_constraints": {
                "max_drawdown": 0.25,
                "min_win_rate": 0.35,
                "max_position_size": 0.3
            },
            "soft_constraints": {
                "target_sharpe_ratio": 1.0,
                "target_profit_factor": 1.5,
                "target_calmar_ratio": 2.0
            },
            "penalty_functions": {
                "drawdown_penalty": "exponential",
                "volatility_penalty": "linear",
                "complexity_penalty": "l1_norm"
            }
        }
    },
    
    # --- Computational Optimization Configuration ---
    "computational_optimization": {
        "caching": {
            "enabled": True,
            "max_cache_size": 1000,
            "cache_ttl": 3600  # 1 hour
        },
        "parallelization": {
            "enabled": True,
            "max_workers": 8,
            "chunk_size": 1000
        },
        "early_stopping": {
            "enabled": True,
            "patience": 10,
            "min_trials": 20
        },
        "surrogate_models": {
            "enabled": True,
            "expensive_trials": 50,
            "update_frequency": 10
        },
        "memory_management": {
            "enabled": True,
            "memory_threshold": 0.8,
            "cleanup_frequency": 100
        },
        "progressive_evaluation": {
            "enabled": True,
            "stages": [
                {"data_ratio": 0.1, "weight": 0.3},
                {"data_ratio": 0.3, "weight": 0.5},
                {"data_ratio": 1.0, "weight": 1.0}
            ]
        }
    },
    # --- Feature Integration Configuration ---
    "FEATURE_INTEGRATION": {
        "enable_liquidity_features": True,
        "enable_advanced_features": True,
        "feature_selection_method": "correlation",
        "pca_variance_threshold": 0.95,
        "correlation_threshold": 0.95,
        "min_features": 10,
        "max_features": 100,
        "liquidity_feature_weights": {
            "volume_liquidity": 1.0,
            "price_impact": 1.0,
            "spread_liquidity": 1.0,
            "liquidity_regime": 1.0,
            "liquidity_percentile": 1.0,
            "kyle_lambda": 1.0,
            "amihud_illiquidity": 1.0,
            "order_flow_imbalance": 1.0,
            "large_order_ratio": 1.0,
            "vwap": 1.0,
            "volume_roc": 1.0,
            "volume_ma_ratio": 1.0,
            "liquidity_stress": 1.0,
            "liquidity_health": 1.0
        }
    },
}


# ==============================================================================
# Lookback Window Utility Functions
# ==============================================================================
def get_lookback_window(config: dict[str, Any] | None = None) -> int:
    """
    Get the appropriate lookback window based on current mode.
    
    Args:
        config: Configuration dictionary. If None, uses global CONFIG.
        
    Returns:
        int: Lookback window in days
    """
    if config is None:
        config = CONFIG
    
    # Get lookback configuration
    lookback_config = config.get("lookback_config", {})
    default_days = lookback_config.get("default_days", 730)  # 2 years default
    blank_mode_days = lookback_config.get("blank_mode_days", 30)  # 30 days for blank mode
    
    # Check if we're in blank training mode
    import os
    blank_training_mode = os.environ.get("BLANK_TRAINING_MODE", "0") == "1"
    
    if blank_training_mode:
        return blank_mode_days
    else:
        return default_days

# ==============================================================================
# Unified Configuration Interface
# ==============================================================================
class AresConfig:
    """
    Unified configuration interface that provides access to both settings and CONFIG.
    This ensures consistent configuration access patterns across the codebase.
    """

    def __init__(self):
        self.settings = settings
        self.config = CONFIG

    def get(self, key: str, default: Any = None) -> Any:
        """Get value from CONFIG dictionary."""
        return self.config.get(key, default)

    def get_setting(self, key: str, default: Any = None) -> Any:
        """Get value from settings object."""
        return getattr(self.settings, key, default)

    @property
    def trading_environment(self) -> str:
        return self.settings.trading_environment

    @property
    def trade_symbol(self) -> str:
        return self.settings.trade_symbol

    @property
    def exchange_name(self) -> str:
        return self.settings.exchange_name.lower()

    @property
    def timeframe(self) -> str:
        return self.settings.timeframe

    @property
    def initial_equity(self) -> float:
        return self.settings.initial_equity

    @property
    def is_live_mode(self) -> bool:
        return self.settings.is_live_mode

    @property
    def exchange_config(self) -> dict:
        """Gets the configuration for the currently selected exchange."""
        return self.config.get("exchanges", {}).get(self.exchange_name.lower(), {})

    @property
    def api_key(self) -> str | None:
        """Gets the API key for the currently configured exchange."""
        return self.exchange_config.get("api_key")

    @property
    def api_secret(self) -> str | None:
        """Gets the API secret for the currently configured exchange."""
        return self.exchange_config.get("api_secret")

    @property
    def password(self) -> str | None:
        """Gets the API password for the currently configured exchange (e.g., for OKX)."""
        return self.exchange_config.get("password")
    
    @property
    def symbols(self) -> list[str]:
        """Gets the list of symbols for the currently configured exchange."""
        return self.exchange_config.get("symbols", [])


# Create a global instance for easy access
ares_config = AresConfig()

# Update CONFIG with ares_config values for all exchanges
CONFIG["exchanges"]["binance"]["api_key"] = ares_config.api_key
CONFIG["exchanges"]["binance"]["api_secret"] = ares_config.api_secret

# Update other exchanges with their respective API keys
CONFIG["exchanges"]["gateio"]["api_key"] = settings.gateio_api_key
CONFIG["exchanges"]["gateio"]["api_secret"] = settings.gateio_api_secret
CONFIG["exchanges"]["mexc"]["api_key"] = settings.mexc_api_key
CONFIG["exchanges"]["mexc"]["api_secret"] = settings.mexc_api_secret
CONFIG["exchanges"]["okx"]["api_key"] = settings.okx_api_key
CONFIG["exchanges"]["okx"]["api_secret"] = settings.okx_api_secret
CONFIG["exchanges"]["okx"]["password"] = settings.okx_password

# ==============================================================================
# Export Configuration for Backward Compatibility
# ==============================================================================
# These exports ensure existing code continues to work
__all__ = ["settings", "CONFIG", "ares_config"]


@dataclass
class DatabaseConfig:
    """Database configuration settings."""

    host: str = "localhost"
    port: int = 5432
    database: str = "ares_trading"
    username: str = "postgres"
    password: str = ""
    max_connections: int = 10
    connection_timeout: int = 30


@dataclass
class ExchangeConfig:
    """Exchange configuration settings."""

    name: str = "binance"
    api_key: str = ""
    api_secret: str = ""
    testnet: bool = True
    rate_limit: int = 1200
    timeout: int = 30


@dataclass
class ModelTrainingConfig:
    """Model training configuration settings."""

    lookback_days: int = 30
    training_split: float = 0.8
    validation_split: float = 0.1
    test_split: float = 0.1
    batch_size: int = 64
    epochs: int = 100
    learning_rate: float = 0.001


@dataclass
class RiskConfig:
    """Risk management configuration settings."""

    max_position_size: float = 0.1
    max_drawdown: float = 0.15
    stop_loss_pct: float = 0.05
    take_profit_pct: float = 0.1
    max_leverage: int = 10


class ConfigurationManager:
    """
    Enhanced Configuration Manager component with DI, type hints, and robust error handling.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config: dict[str, Any] = config
        self.logger = system_logger.getChild("ConfigurationManager")
        self.is_running: bool = False
        self.status: dict[str, Any] = {}
        self.history: list[dict[str, Any]] = []
        self.config_manager_config: dict[str, Any] = self.config.get(
            "config_manager",
            {},
        )
        self.update_interval: int = self.config_manager_config.get(
            "update_interval",
            300,
        )
        self.max_history: int = self.config_manager_config.get("max_history", 100)
        self.config_sections: dict[str, Any] = {}
        self.config_service: ConfigurationService | None = None

    @handle_specific_errors(
        error_handlers={
            ValueError: (False, "Invalid configuration manager configuration"),
            AttributeError: (
                False,
                "Missing required configuration manager parameters",
            ),
            KeyError: (False, "Missing configuration keys"),
        },
        default_return=False,
        context="configuration manager initialization",
    )
    async def initialize(self) -> bool:
        try:
            self.logger.info("Initializing Configuration Manager...")
            await self._load_config_manager_configuration()
            if not self._validate_configuration():
                self.logger.error("Invalid configuration for configuration manager")
                return False
            await self._initialize_config_sections()
            await self._initialize_config_service()
            self.logger.info(
                "✅ Configuration Manager initialization completed successfully",
            )
            return True
        except Exception as e:
            self.logger.error(f"❌ Configuration Manager initialization failed: {e}")
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="config manager configuration loading",
    )
    async def _load_config_manager_configuration(self) -> None:
        try:
            self.config_manager_config.setdefault("update_interval", 300)
            self.config_manager_config.setdefault("max_history", 100)
            self.update_interval = self.config_manager_config["update_interval"]
            self.max_history = self.config_manager_config["max_history"]
            self.logger.info("Configuration manager configuration loaded successfully")
        except Exception as e:
            self.logger.error(f"Error loading config manager configuration: {e}")

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=False,
        context="configuration validation",
    )
    def _validate_configuration(self) -> bool:
        try:
            if self.update_interval <= 0:
                self.logger.error("Invalid update interval")
                return False
            if self.max_history <= 0:
                self.logger.error("Invalid max history")
                return False
            self.logger.info("Configuration validation successful")
            return True
        except Exception as e:
            self.logger.error(f"Error validating configuration: {e}")
            return False

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="config sections initialization",
    )
    async def _initialize_config_sections(self) -> None:
        try:
            # Initialize configuration sections
            self.config_sections = {
                "database": DatabaseConfig(),
                "exchange": ExchangeConfig(),
                "model_training": ModelTrainingConfig(),
                "risk": RiskConfig(),
            }
            self.logger.info("Configuration sections initialized successfully")
        except Exception as e:
            self.logger.error(f"Error initializing config sections: {e}")

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="config service initialization",
    )
    async def _initialize_config_service(self) -> None:
        try:
            # Initialize configuration service
            self.config_service = ConfigurationService(self.config)
            await self.config_service.initialize()
            self.logger.info("Configuration service initialized successfully")
        except Exception as e:
            self.logger.error(f"Error initializing config service: {e}")

    @handle_specific_errors(
        error_handlers={
            Exception: (False, "Configuration manager run failed"),
        },
        default_return=False,
        context="configuration manager run",
    )
    async def run(self) -> bool:
        try:
            self.is_running = True
            self.logger.info("🚦 Configuration Manager started.")
            while self.is_running:
                await self._update_configuration()
                await asyncio.sleep(self.update_interval)
            return True
        except Exception as e:
            self.logger.error(f"Error in configuration manager run: {e}")
            self.is_running = False
            return False

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="configuration update",
    )
    async def _update_configuration(self) -> None:
        try:
            now = datetime.now().isoformat()
            self.status = {"timestamp": now, "status": "running"}
            self.history.append(self.status.copy())
            if len(self.history) > self.max_history:
                self.history.pop(0)
            await self._reload_configuration()
            await self._validate_configuration_sections()
            await self._update_config_service()
            self.logger.info(f"Configuration update tick at {now}")
        except Exception as e:
            self.logger.error(f"Error in configuration update: {e}")

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="configuration reload",
    )
    async def _reload_configuration(self) -> None:
        try:
            # Reload configuration from files/environment
            if self.config_service:
                await self.config_service.reload_configuration()
            self.logger.info("Configuration reloaded successfully")
        except Exception as e:
            self.logger.error(f"Error reloading configuration: {e}")

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="configuration sections validation",
    )
    async def _validate_configuration_sections(self) -> None:
        try:
            # Validate all configuration sections
            for section_name, section_config in self.config_sections.items():
                if hasattr(section_config, "__dataclass_fields__"):
                    for field_name in section_config.__dataclass_fields__:
                        field_value = getattr(section_config, field_name)
                        if field_value is None:
                            self.logger.warning(
                                f"Missing value for {section_name}.{field_name}",
                            )
            self.logger.info("Configuration sections validation completed")
        except Exception as e:
            self.logger.error(f"Error validating configuration sections: {e}")

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="config service update",
    )
    async def _update_config_service(self) -> None:
        try:
            # Update configuration service with new values
            if self.config_service:
                await self.config_service.update_configuration(self.config)
            self.logger.info("Configuration service updated successfully")
        except Exception as e:
            self.logger.error(f"Error updating config service: {e}")

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="configuration manager stop",
    )
    async def stop(self) -> None:
        self.logger.info("🛑 Stopping Configuration Manager...")
        try:
            self.is_running = False
            self.status = {"timestamp": datetime.now().isoformat(), "status": "stopped"}
            if self.config_service:
                await self.config_service.stop()
            self.logger.info("✅ Configuration Manager stopped successfully")
        except Exception as e:
            self.logger.error(f"Error stopping configuration manager: {e}")

    def get_status(self) -> dict[str, Any]:
        return self.status.copy()

    def get_history(self, limit: int | None = None) -> list[dict[str, Any]]:
        history = self.history.copy()
        if limit:
            history = history[-limit:]
        return history

    def get_config_sections(self) -> dict[str, Any]:
        return self.config_sections.copy()

    def get_config_service(self) -> ConfigurationService | None:
        return self.config_service


configuration_manager: ConfigurationManager | None = None


@handle_errors(
    exceptions=(Exception,),
    default_return=None,
    context="configuration manager setup",
)
async def setup_configuration_manager(
    config: dict[str, Any] | None = None,
) -> ConfigurationManager | None:
    try:
        global configuration_manager
        if config is None:
            config = {"config_manager": {"update_interval": 300, "max_history": 100}}
        configuration_manager = ConfigurationManager(config)
        success = await configuration_manager.initialize()
        if success:
            return configuration_manager
        return None
    except Exception as e:
        print(f"Error setting up configuration manager: {e}")
        return None


# Legacy CONFIG for backward compatibility - REMOVED to avoid conflicts
# The main CONFIG is defined at the top of this file