# src/types/config_types.py

"""
Configuration type definitions for type-safe configuration management.
"""

from typing import Literal, TypedDict, Any, Dict, List, Optional, Union
from datetime import datetime


class DatabaseConfig(TypedDict):
    """Database configuration."""
    host: str
    port: int
    database: str
    username: str
    password: str
    ssl_mode: Optional[str]
    connection_pool_size: int
    max_connections: int
    timeout: int


class ExchangeConfig(TypedDict):
    """Exchange configuration."""
    name: str
    api_key: str
    api_secret: str
    passphrase: Optional[str]
    sandbox: bool
    rate_limit: int
    timeout: int
    retry_attempts: int


class MLConfig(TypedDict):
    """Machine learning configuration."""
    model_path: str
    feature_columns: List[str]
    target_column: str
    test_size: float
    random_state: int
    hyperparameters: Dict[str, Any]
    model_type: str
    training_interval: str
    retraining_threshold: float


class MonitoringConfig(TypedDict):
    """Monitoring configuration."""
    log_level: str
    metrics_interval: int
    alert_thresholds: Dict[str, float]
    notification_channels: List[str]
    health_check_interval: int
    performance_tracking: bool
    error_reporting: bool


class TradingConfig(TypedDict):
    """Trading configuration."""
    default_order_type: str
    default_time_in_force: str
    max_position_size: float
    max_leverage: float
    risk_per_trade: float
    stop_loss_pct: float
    take_profit_pct: float
    trailing_stop_pct: float
    position_sizing_method: str
    rebalancing_interval: str


class SystemConfig(TypedDict):
    """System configuration."""
    environment: str
    debug_mode: bool
    data_dir: str
    log_dir: str
    temp_dir: str
    max_memory_usage: int
    cpu_limit: int
    backup_enabled: bool
    backup_interval: str


class TrainingConfig(TypedDict):
    """Training configuration."""
    data_source: str
    data_start_date: str
    data_end_date: str
    validation_split: float
    batch_size: int
    epochs: int
    learning_rate: float
    early_stopping_patience: int
    model_checkpoint_path: str


class ConfigDict(TypedDict):
    """Main configuration dictionary."""
    database: DatabaseConfig
    exchange: ExchangeConfig
    ml: MLConfig
    monitoring: MonitoringConfig
    trading: TradingConfig
    system: SystemConfig
    training: TrainingConfig
    version: str
    last_updated: datetime
