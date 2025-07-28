import os
import json
from typing import Literal, List, Dict, Any, Optional
from pydantic import BaseSettings, Field, BaseModel, validator
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

# --- Sub-Models for Structured Configuration ---
# This approach provides validation and clear structure for all parameters.

class SRAnalyzerConfig(BaseModel):
    peak_prominence: float = 0.005
    peak_width: int = 5
    level_tolerance_pct: float = 0.002
    min_touches: int = 2
    max_age_days: int = 90

class AnalystMarketRegimeClassifierConfig(BaseModel):
    model_storage_path: str = "models/analyst/"
    adx_period: int = 14
    macd_fast_period: int = 12
    macd_slow_period: int = 26
    macd_signal_period: int = 9
    trend_scaling_factor: int = 100
    trend_threshold: int = 20
    max_strength_threshold: int = 60

class AnalystConfig(BaseModel):
    sr_analyzer: SRAnalyzerConfig = SRAnalyzerConfig()
    market_regime_classifier: AnalystMarketRegimeClassifierConfig = AnalystMarketRegimeClassifierConfig()
    # Add other analyst sub-modules here if needed

class TacticianConfig(BaseModel):
    initial_leverage: int = 25
    min_lss_for_entry: int = 60

class StrategistConfig(BaseModel):
    timeframe: str = "1D"
    long_threshold: float = 0.6
    short_threshold: float = 0.6

class BacktestingOptimizationParams(BaseModel):
    atr_period: Dict[str, Any] = {"type": "int", "low": 10, "high": 30}
    atr_multiplier_tp: Dict[str, Any] = {"type": "float", "low": 1.0, "high": 5.0}
    atr_multiplier_sl: Dict[str, Any] = {"type": "float", "low": 0.5, "high": 3.0}
    trailing_sl_enabled: Dict[str, Any] = {"type": "categorical", "choices": [True, False]}
    atr_multiplier_tsl: Dict[str, Any] = {
        "type": "float", "low": 1.0, "high": 4.0, "condition": ("trailing_sl_enabled", "==", True)
    }

class BacktestingOptimizationConfig(BaseModel):
    enabled: bool = True
    n_trials: int = 100
    study_name: str = "ares_strategy_optimization"
    storage: str = "sqlite:///ares_optimization.db"
    direction: Literal["maximize", "minimize"] = "maximize"
    objective_metric: str = "Sharpe Ratio"
    params: BacktestingOptimizationParams = BacktestingOptimizationParams()

class BacktestingConfig(BaseModel):
    data: Dict[str, Any] = {
        "klines_path": "data/historical_klines.csv",
        "symbol": "BTCUSDT",
        "interval": "15m",
        "start_date": "2023-01-01",
        "end_date": "2024-01-01",
    }
    optimization: BacktestingOptimizationConfig = BacktestingOptimizationConfig()
    results: Dict[str, str] = {
        "best_params_file": "models/best_strategy_params.json",
        "report_file": "backtests/performance_report.txt"
    }

# --- Main Settings Class ---
# This class orchestrates the entire configuration.

class Settings(BaseSettings):
    """
    Manages all application settings and parameters using a structured,
    validated model. Loads sensitive/environment-specific values from .env.
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

    # --- General Trading Parameters ---
    trading_symbol: str = Field(default="BTCUSDT", alias="SYMBOL")
    trading_interval: str = Field(default="15m", alias="INTERVAL")
    initial_equity: float = 10000.0
    taker_fee: float = 0.0004
    maker_fee: float = 0.0002

    # --- Nested Configuration Models for Components ---
    analyst: AnalystConfig = AnalystConfig()
    tactician: TacticianConfig = TacticianConfig()
    strategist: StrategistConfig = StrategistConfig()
    backtesting: BacktestingConfig = BacktestingConfig()
    
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
        # Allow loading settings with aliases (e.g., SYMBOL -> trading_symbol)
        allow_population_by_field_name = True

# --- Global Settings Instance ---
# Instantiate the settings object to be imported by other modules.
try:
    settings = Settings()
    logger.info(f"Configuration loaded successfully. Trading Environment: {settings.trading_environment}")
    # You can now access any setting via `settings.component.parameter`
    # e.g., `settings.analyst.market_regime_classifier.adx_period`
except Exception as e:
    logger.critical(f"Failed to load or validate configuration: {e}")
    # Exit if config is invalid, as the bot cannot run without it.
    exit(1)
