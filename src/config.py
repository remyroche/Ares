import os
from typing import Literal, Dict, Any, Optional
from pydantic import BaseSettings, Field, validator
from loguru import logger
from dotenv import load_dotenv

# --- Environment Loading ---
# Load .env file from the project's root directory
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path)
    logger.info(".env file loaded.")
else:
    logger.warning(".env file not found. Using environment variables or defaults.")

# ==============================================================================
# Pydantic Settings for Environment Variables
# ==============================================================================
# This class handles loading and validating all settings that come from the
# environment (e.g., .env file or system variables), such as API keys.

class Settings(BaseSettings):
    """
    Manages all environment-specific settings using Pydantic.
    """
    # --- Top-Level Environment-Specific Settings ---
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    trading_environment: Literal["LIVE", "TESTNET"] = Field(default="TESTNET", env="TRADING_ENVIRONMENT")

    # --- Binance Credentials (loaded from .env) ---
    binance_live_api_key: Optional[str] = Field(default=None, env="BINANCE_LIVE_API_KEY")
    binance_live_api_secret: Optional[str] = Field(default=None, env="BINANCE_LIVE_API_SECRET")
    binance_testnet_api_key: Optional[str] = Field(default=None, env="BINANCE_TESTNET_API_KEY")
    binance_testnet_api_secret: Optional[str] = Field(default=None, env="BINANCE_TESTNET_API_SECRET")

    # --- Firestore Credentials (loaded from .env) ---
    google_application_credentials: Optional[str] = Field(default=None, env="GOOGLE_APPLICATION_CREDENTIALS")
    firestore_project_id: Optional[str] = Field(default=None, env="FIRESTORE_PROJECT_ID")

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
# This dictionary holds all the strategy, component, and backtesting parameters.
# It's kept separate from the Pydantic settings for easier manipulation during
# optimization and backtesting.

CONFIG: Dict[str, Any] = {
    # --- General System & Trading Parameters ---
    "trading_symbol": "BTCUSDT",
    "trading_interval": "15m",
    "initial_equity": 10000.0,
    "taker_fee": 0.0004,
    "maker_fee": 0.0002,
    "state_file": "ares_state.json",

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
            "model_storage_path": "models/analyst/",
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
    
    # --- Backtesting & Optimization ---
    # This section integrates the Optuna-based optimization configuration.
    "backtesting": {
        "data": {
            "klines_path": "data/historical_klines.csv",
            "symbol": "BTCUSDT",
            "interval": "15m",
            "start_date": "2023-01-01",
            "end_date": "2024-01-01",
        },
        "optimization": {
            "enabled": True,
            "n_trials": 100,
            "study_name": "ares_strategy_optimization",
            "storage": "sqlite:///ares_optimization.db",
            "direction": "maximize",
            "objective_metric": "Sharpe Ratio",
            "params": {
                "atr_period": {"type": "int", "low": 10, "high": 30},
                "atr_multiplier_tp": {"type": "float", "low": 1.0, "high": 5.0},
                "atr_multiplier_sl": {"type": "float", "low": 0.5, "high": 3.0},
                "trailing_sl_enabled": {"type": "categorical", "choices": [True, False]},
                "atr_multiplier_tsl": {
                    "type": "float", "low": 1.0, "high": 4.0, "condition": ("trailing_sl_enabled", "==", True)
                }
            }
        },
        "results": {
            "best_params_file": "models/best_strategy_params.json",
            "report_file": "backtests/performance_report.txt"
        }
    }
}
