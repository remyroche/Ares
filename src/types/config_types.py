# src/types/config_types.py

"""
Configuration type definitions for type-safe configuration management.
"""

from typing import Dict, List, Literal, Optional, TypedDict, Union

from .base_types import Interval, Percentage, Symbol


class DatabaseConfig(TypedDict, total=False):
    """Type-safe database configuration."""
    type: Literal["sqlite", "firestore", "mongodb"]
    path: str
    host: Optional[str]
    port: Optional[int]
    username: Optional[str]
    password: Optional[str]
    database_name: Optional[str]
    connection_timeout: Optional[int]
    max_connections: Optional[int]


class ExchangeConfig(TypedDict, total=False):
    """Type-safe exchange configuration."""
    name: Literal["binance", "coinbase", "kraken", "bybit"]
    api_key: str
    api_secret: str
    sandbox: bool
    testnet: bool
    rate_limit: Optional[int]
    timeout: Optional[int]
    max_retries: Optional[int]


class TradingConfig(TypedDict, total=False):
    """Type-safe trading configuration."""
    symbols: List[Symbol]
    intervals: List[Interval]
    max_position_size: float
    max_leverage: float
    stop_loss_percentage: Percentage
    take_profit_percentage: Percentage
    max_drawdown: Percentage
    risk_per_trade: Percentage
    enable_trailing_stop: bool
    paper_trading: bool


class MLConfig(TypedDict, total=False):
    """Type-safe ML configuration."""
    model_type: Literal["xgboost", "lightgbm", "neural_network", "ensemble"]
    lookback_days: int
    prediction_horizon: int
    feature_engineering: Dict[str, Union[bool, int, float]]
    hyperparameters: Dict[str, Union[int, float, str, bool]]
    validation_split: Percentage
    early_stopping_rounds: Optional[int]
    max_iterations: Optional[int]


class MonitoringConfig(TypedDict, total=False):
    """Type-safe monitoring configuration."""
    enable_prometheus: bool
    prometheus_port: Optional[int]
    enable_health_checks: bool
    health_check_interval: int
    enable_performance_tracking: bool
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    log_file_path: Optional[str]
    max_log_file_size: Optional[int]


class SystemConfig(TypedDict, total=False):
    """Type-safe system configuration."""
    environment: Literal["development", "staging", "production"]
    debug_mode: bool
    max_threads: Optional[int]
    memory_limit_mb: Optional[int]
    enable_profiling: bool
    data_cache_size_mb: Optional[int]


class TrainingConfig(TypedDict, total=False):
    """Type-safe training configuration."""
    training_pipeline: Dict[str, Union[int, float]]
    MODEL_TRAINING: Dict[str, Union[int, float, str, bool, Dict[str, Union[int, float, str, bool]]]]
    DATA_CONFIG: Dict[str, Union[int, float, str]]
    ENHANCED_TRAINING: Dict[str, Union[int, float, str, bool]]
    MULTI_TIMEFRAME_TRAINING: Dict[str, Union[int, float, str, bool, Dict[str, Union[int, float, str, bool]]]]
    TIMEFRAMES: Dict[str, Dict[str, Union[int, float, str]]]
    TIMEFRAME_SETS: Dict[str, Dict[str, Union[List[str], str]]]
    DEFAULT_TIMEFRAME_SET: str
    TWO_TIER_DECISION: Dict[str, Union[int, float, str, bool, List[str]]]
    ENHANCED_ENSEMBLE: Dict[str, Union[int, float, str, bool, Dict[str, Union[int, float, str]]]]


# Main configuration type
class ConfigDict(TypedDict, total=False):
    """Complete type-safe configuration dictionary."""
    database: DatabaseConfig
    exchanges: Dict[str, ExchangeConfig]
    trading: TradingConfig
    ml: MLConfig
    monitoring: MonitoringConfig
    system: SystemConfig
    training: TrainingConfig