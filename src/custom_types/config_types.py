# src/types/config_types.py

"""
Configuration type definitions for type-safe configuration management.
"""

from typing import Literal, TypedDict

from .base_types import Interval, Percentage, Symbol


class DatabaseConfig(TypedDict, total=False):
    """Type-safe database configuration."""

    type: Literal["sqlite", "firestore", "mongodb"]
    path: str
    host: str | None
    port: int | None
    username: str | None
    password: str | None
    database_name: str | None
    connection_timeout: int | None
    max_connections: int | None


class ExchangeConfig(TypedDict, total=False):
    """Type-safe exchange configuration."""

    name: Literal["binance", "gateio", "mexc", "okx", "coinbase", "kraken", "bybit"]
    api_key: str
    api_secret: str
    password: str | None
    sandbox: bool
    testnet: bool
    rate_limit: int | None
    timeout: int | None
    max_retries: int | None


class TradingConfig(TypedDict, total=False):
    """Type-safe trading configuration."""

    symbols: list[Symbol]
    intervals: list[Interval]
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
    feature_engineering: dict[str, bool | int | float]
    hyperparameters: dict[str, int | float | str | bool]
    validation_split: Percentage
    early_stopping_rounds: int | None
    max_iterations: int | None


class MonitoringConfig(TypedDict, total=False):
    """Type-safe monitoring configuration."""

    enable_prometheus: bool
    prometheus_port: int | None
    enable_health_checks: bool
    health_check_interval: int
    enable_performance_tracking: bool
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    log_file_path: str | None
    max_log_file_size: int | None


class SystemConfig(TypedDict, total=False):
    """Type-safe system configuration."""

    environment: Literal["development", "staging", "production"]
    debug_mode: bool
    max_threads: int | None
    memory_limit_mb: int | None
    enable_profiling: bool
    data_cache_size_mb: int | None


class TrainingConfig(TypedDict, total=False):
    """Type-safe training configuration."""

    training_pipeline: dict[str, int | float]
    MODEL_TRAINING: dict[
        str,
        int | float | str | bool | dict[str, int | float | str | bool],
    ]
    DATA_CONFIG: dict[str, int | float | str]
    ENHANCED_TRAINING: dict[str, int | float | str | bool]
    MULTI_TIMEFRAME_TRAINING: dict[
        str,
        int | float | str | bool | dict[str, int | float | str | bool],
    ]
    TIMEFRAMES: dict[str, dict[str, int | float | str]]
    TIMEFRAME_SETS: dict[str, dict[str, list[str] | str]]
    DEFAULT_TIMEFRAME_SET: str
    TWO_TIER_DECISION: dict[str, int | float | str | bool | list[str]]
    ENHANCED_ENSEMBLE: dict[
        str,
        int | float | str | bool | dict[str, int | float | str],
    ]


# Main configuration type
class ConfigDict(TypedDict, total=False):
    """Complete type-safe configuration dictionary."""

    database: DatabaseConfig
    exchanges: dict[str, ExchangeConfig]
    trading: TradingConfig
    ml: MLConfig
    monitoring: MonitoringConfig
    system: SystemConfig
    training: TrainingConfig
