import os
from typing import Literal, Dict, Any, Optional
from pydantic import BaseSettings, Field, validator
from loguru import logger
from dotenv import load_dotenv

# --- Environment Loading ---
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path)
    logger.info(".env file loaded.")
else:
    logger.warning(".env file not found. Using environment variables or defaults.")

# ==============================================================================
# Pydantic Settings for Environment Variables
# ==============================================================================
class Settings(BaseSettings):
    """
    Manages all environment-specific settings using Pydantic.
    """
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    trading_environment: Literal["LIVE", "TESTNET", "PAPER"] = Field(default="TESTNET", env="TRADING_ENVIRONMENT")
    initial_equity: float = Field(default=10000.0, env="INITIAL_EQUITY")
    trade_symbol: str = Field(default="BTCUSDT", env="TRADE_SYMBOL")
    timeframe: str = Field(default="15m", env="TIMEFRAME")


    # --- Binance Credentials (loaded from .env) ---
    binance_live_api_key: Optional[str] = Field(default=None, env="BINANCE_LIVE_API_KEY")
    binance_live_api_secret: Optional[str] = Field(default=None, env="BINANCE_LIVE_API_SECRET")
    binance_testnet_api_key: Optional[str] = Field(default=None, env="BINANCE_TESTNET_API_KEY")
    binance_testnet_api_secret: Optional[str] = Field(default=None, env="BINANCE_TESTNET_API_SECRET")

    # --- Firestore Credentials (loaded from .env) ---
    google_application_credentials: Optional[str] = Field(default=None, env="GOOGLE_APPLICATION_CREDENTIALS")
    firestore_project_id: Optional[str] = Field(default=None, env="FIRESTORE_PROJECT_ID")

    # --- Emailer Credentials (loaded from .env) ---
    email_sender_address: Optional[str] = Field(default=None, env="EMAIL_SENDER_ADDRESS")
    email_sender_password: Optional[str] = Field(default=None, env="EMAIL_SENDER_PASSWORD")
    email_recipient_address: Optional[str] = Field(default=None, env="EMAIL_RECIPIENT_ADDRESS")


    # --- Derived Properties for Convenience ---
    @property
    def is_live_mode(self) -> bool:
        return self.trading_environment == "LIVE"

    @property
    def binance_api_key(self) -> Optional[str]:
        return self.binance_live_api_key if self.is_live_mode else self.binance_testnet_api_key

    @property
    def binance_api_secret(self) -> Optional[str]:
        return self.binance_live_api_secret if self.is_live_mode else self.binance_testnet_api_secret

    # --- Validators ---
    @validator('trading_environment')
    def check_keys_for_environment(cls, v, values):
        if v == "LIVE" and (not values.get('binance_live_api_key') or not values.get('binance_live_api_secret')):
            raise ValueError("For LIVE environment, BINANCE_LIVE_API_KEY and BINANCE_LIVE_API_SECRET must be set.")
        if v == "TESTNET" and (not values.get('binance_testnet_api_key') or not values.get('binance_testnet_api_secret')):
            raise ValueError("For TESTNET environment, BINANCE_TESTNET_API_KEY and BINANCE_TESTNET_API_SECRET must be set.")
        return v
        
    class Config:
        case_sensitive = True

# --- Global Settings Instance ---
try:
    settings = Settings()
    logger.info(f"Configuration loaded successfully. Trading Environment: {settings.trading_environment}")
except Exception as e:
    logger.critical(f"Failed to load or validate configuration: {e}")
    exit(1) # Exit if essential env vars are missing/invalid

# ==============================================================================
# Main Configuration Dictionary
# ==============================================================================
CONFIG: Dict[str, Any] = {
    # --- General System & Trading Parameters ---
    "trading_symbol": settings.trade_symbol,
    "trading_interval": settings.timeframe,
    "initial_equity": settings.initial_equity,
    "taker_fee": 0.0004,
    "maker_fee": 0.0002,
    "state_file": "ares_state.json",
    "lookback_years": 2,

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
    "OPTIMIZER_CHECKPOINT_FILE": "optimizer_state.pkl", # Relative to CHECKPOINT_DIR
    "PIPELINE_PROGRESS_FILE": "pipeline_progress.json", # Relative to CHECKPOINT_DIR
    "WALK_FORWARD_REPORTS_FILE": "walk_forward_reports.json", # Relative to CHECKPOINT_DIR
    "PREPARED_DATA_CHECKPOINT_FILE": "full_prepared_data.parquet", # Relative to CHECKPOINT_DIR
    "REGIME_CLASSIFIER_MODEL_PREFIX": "regime_classifier_fold_", # Prefix for fold-specific models
    "ENSEMBLE_MODEL_PREFIX": "ensemble_fold_", # Prefix for fold-specific ensemble models

    # --- Reporting Configuration ---
    "DETAILED_TRADE_LOG_FILE": "reports/detailed_trade_log.csv",
    "DAILY_SUMMARY_LOG_FILENAME_FORMAT": "reports/daily_summary_log_%Y-%m.csv", # New: Monthly filenames for summary reports
    "STRATEGY_PERFORMANCE_LOG_FILENAME_FORMAT": "reports/strategy_performance_log_%Y-%m.csv", # New: Monthly filenames for strategy reports
    "ERROR_LOG_FILE": "ares_errors.jsonl", # New: Dedicated error log file

    # --- Database Configuration ---
    "DATABASE_TYPE": "sqlite", # New: 'firestore' or 'sqlite'
    "SQLITE_DB_PATH": "data/ares_local_db.sqlite", # New: Path for SQLite database file

    # --- Firestore Configuration ---
    "firestore": {
        "enabled": settings.firestore_project_id is not None,
        "project_id": settings.firestore_project_id,
        "optimized_params_collection": "ares_optimized_params",
        "live_metrics_collection": "ares_live_metrics",
        "alerts_collection": "ares_alerts"
    },

    # --- Email Configuration ---
    "email_config": {
        "enabled": settings.email_sender_address is not None and settings.email_recipient_address is not None,
        "sender_email": settings.email_sender_address,
        "app_password": settings.email_sender_password,
        "recipient_email": settings.email_recipient_address,
        "smtp_server": "smtp.gmail.com",
        "smtp_port": 587
    },
    
    # --- Command Email Listener Configuration ---
    "command_email_config": {
        "enabled": True,
        "imap_server": "imap.gmail.com",
        "imap_port": 993,
        "email_address": settings.email_sender_address,
        "app_password": settings.email_sender_password,
        "allowed_sender": settings.email_recipient_address,
        "polling_interval_seconds": 60,
    },

    # --- Analyst Component ---
    "analyst": {
        "sr_analyzer": {
            "peak_prominence": 0.005,
            "peak_width": 5,
            "level_tolerance_pct": 0.002,
            "min_touches": 2,
            "max_age_days": 90
        },
        "market_regime_classifier": {
            "model_storage_path": "models/analyst/", # This will be overridden by checkpoint dir
            "adx_period": 14,
            "macd_fast_period": 12,
            "macd_slow_period": 26,
            "macd_signal_period": 9,
            "trend_scaling_factor": 100,
            "trend_threshold": 20,
            "max_strength_threshold": 60
        },
        "ml_targets": {
            "profit_take_multiplier": 2.0,
            "stop_loss_multiplier": 1.5,
            "max_hold_periods": 20
        },
        "feature_engineering": {
            "autoencoder_latent_dim": 16,
            "proximity_multiplier": 0.25,
            "resample_interval": "1T",
            "adx_period": 14,
            "macd_fast_period": 12,
            "macd_slow_period": 26,
            "macd_signal_period": 9,
            "rsi_period": 14,
            "stoch_period": 14,
            "bb_period": 20,
            "cmf_period": 20,
            "kc_period": 20,
            "atr_period": 14,
            "model_storage_path": "models/analyst/feature_engineering/" # This will be overridden by checkpoint dir
        }
    },

    # --- Strategist Component ---
    "strategist": {
        "timeframe": "1D",
        "long_threshold": 0.6,
        "short_threshold": 0.6,
    },

    # --- Tactician Component ---
    "tactician": {
        "initial_leverage": 25,
        "min_lss_for_entry": 60,
    },
    
    # --- Supervisor Component ---
    "supervisor": {
        "check_interval_seconds": 300,
        "risk_reduction_drawdown_pct": 0.10,
        "decay_threshold_profit_factor": 0.80,
        "decay_threshold_sharpe_ratio": 0.70,
        "decay_threshold_max_drawdown_multiplier": 1.50,
        "min_trades_for_monitoring": 50,
        "retrain_interval_days": 30
    },

    # --- Risk Management (Global Portfolio Level) ---
    "risk_management": {
        "global_max_allocated_capital_usd": 50000,
        "max_allocation_per_pair_usd": 5000,
        "pause_trading_drawdown_pct": 0.20,
    },

    # --- Backtesting & Optimization ---
    "backtesting": {
        "fee_rate": 0.0005,
        "optimization": {
            "enabled": True,
            "n_trials": 100,
            "study_name": "ares_strategy_optimization",
            "storage": "sqlite:///ares_optimization.db",
            "direction": "maximize",
            "objective_metric": "Sharpe Ratio",
            "bayesian_opt_n_calls": 20, # Default for Bayesian Optimization
            "bayesian_opt_n_initial_points": 5,
            "bayesian_opt_random_state": 42
        },
        "results": {
            "best_params_file": "models/best_strategy_params.json",
            "report_file": "backtests/performance_report.txt"
        }
    },

    # --- Walk-Forward Analysis Parameters ---
    "training_pipeline": {
        "walk_forward": {
            "n_splits": 5, # Number of walk-forward folds
            "train_months": 12, # Training window size in months
            "test_months": 3 # Testing window size in months
        }
    },

    # --- Optimal Indicator Parameters (To be updated by the optimizer) ---
    "best_params": {
        # --- Confidence Score Weights ---
        'weight_trend': 0.4,
        'weight_reversion': 0.3,
        'weight_sentiment': 0.3,
        
        # --- Trade Execution Parameters ---
        'trade_entry_threshold': 0.6,
        'sl_atr_multiplier': 1.5,
        'take_profit_rr': 2.0,
        
        # --- Underlying Indicator Settings ---
        'adx_period': 20, 
        'trend_threshold': 25,
        'max_strength_threshold': 60, 
        'atr_period': 14, 
        'proximity_multiplier': 0.25,
        'sma_period': 50, 
        'volume_multiplier': 3, 
        'volatility_multiplier': 2,
        'zscore_threshold': 1.5,
        'obv_lookback': 20,
        'bband_length': 20,
        'bband_std': 2.0,
        'bband_squeeze_threshold': 0.01,
        'scaling_factor': 100, 
        'trend_strength_threshold': 25
    }
}

# --- Dynamically set filenames based on other config values ---
CONFIG["klines_filename"] = f"data_cache/{CONFIG['trading_symbol']}_{CONFIG['trading_interval']}_{CONFIG['lookback_years']}y_klines.csv"
CONFIG["agg_trades_filename"] = f"data_cache/{CONFIG['trading_symbol']}_{CONFIG['lookback_years']}y_aggtrades.csv"
CONFIG["futures_filename"] = f"data_cache/{CONFIG['trading_symbol']}_futures_{CONFIG['lookback_years']}y_data.csv"
CONFIG["prepared_data_filename"] = f"data_cache/{CONFIG['trading_symbol']}_{CONFIG['trading_interval']}_{CONFIG['lookback_years']}y_prepared_data.csv"

