# src/config/config.py

"""
Main configuration file containing all non-optimizable parameters.
These are static configuration parameters that should not be optimized.
"""

from typing import Any
from dataclasses import dataclass


@dataclass class DatabaseConfig: pass  # TODO: Add implementation class DatabaseConfig: pass  # TODO: Add implementation class DatabaseConfig: """Database configuration settings.""" host: str = "localhost" port: int = 5432 database: str = "ares_trading" username: str = "postgres" password: str = "" max_connections: int = 10 connection_timeout: int = 30   @dataclass class ExchangeConfig: pass  # TODO: Add implementation class ExchangeConfig: pass  # TODO: Add implementation class ExchangeConfig: """Exchange configuration settings.""" name: str = "binance" api_key: str = "" api_secret: str = "" testnet: bool = True rate_limit: int = 1200 timeout: int = 30   @dataclass class SystemConfig: pass  # TODO: Add implementation class SystemConfig: pass  # TODO: Add implementation class SystemConfig: """System-level configuration settings.""" # Checkpointing checkpoint_dir: str = "checkpoints" save_interval: int = 1000 max_checkpoints: int = 10  # Logging log_level: str = "INFO" log_file: str = "ares.log" max_log_size: int = 100 * 1024 * 1024  # 100MB backup_count: int = 5  # Performance max_workers: int = 4 batch_size: int = 1000 memory_limit_gb: float = 8.0  # Data data_dir: str = "data" cache_dir: str = "cache" max_cache_size_gb: float = 10.0   @dataclass class EnvironmentConfig: pass  # TODO: Add implementation class EnvironmentConfig: pass  # TODO: Add implementation class EnvironmentConfig: """Environment-specific configuration.""" trading_environment: str = "paper"  # paper, live, backtest exchange_name: str = "binance" trade_symbol: str = "ETHUSDT" timeframe: str = "1m" initial_equity: float = 10000.0 is_live_mode: bool = False  # API Keys (will be loaded from environment)
binance_api_key: str = ""
binance_api_secret: str = ""
gateio_api_key: str = ""
gateio_api_secret: str = ""
mexc_api_key: str = ""
mexc_api_secret: str = ""
okx_api_key: str = ""
okx_api_secret: str = ""
okx_password: str = ""


@dataclass class TradingConfig: pass  # TODO: Add implementation class TradingConfig: pass  # TODO: Add implementation class TradingConfig: """Trading-specific configuration (non-optimizable).""" # Basic trading parameters taker_fee: float = 0.0004 maker_fee: float = 0.0002 state_file: str = "ares_state.json" lookback_years: int = 2  # Time-based exit configuration enable_time_exit: bool = True max_holding_time_hours: int = 24 profit_lock_time_hours: int = 4 loss_cut_time_hours: int = 2  # Stop loss configuration enable_stop_loss: bool = True stop_loss_type: str = "trailing"  # 'fixed' or 'trailing' fixed_stop_loss_pct: float = 0.02 trailing_stop_activation_threshold: float = 0.01 trailing_stop_distance: float = 0.005 lock_profit_threshold: float = 0.03  # Take profit configuration enable_take_profit: bool = True take_profit_type: str = "dynamic"  # 'fixed' or 'dynamic' fixed_take_profit_pct: float = 0.05 base_take_profit: float = 0.03 volatility_multiplier: float = 1.5 max_take_profit: float = 0.15   @dataclass class TrainingConfig: pass  # TODO: Add implementation class TrainingConfig: pass  # TODO: Add implementation class TrainingConfig: """Training-specific configuration (non-optimizable).""" # Data configuration train_split: float = 0.7 val_split: float = 0.15 test_split: float = 0.15  # Model configuration model_type: str = "lightgbm" random_state: int = 42  # Training configuration early_stopping_patience: int = 10 max_epochs: int = 1000 batch_size: int = 1024  # Feature engineering feature_window: int = 100 target_window: int = 10  # Validation cv_folds: int = 5 cv_strategy: str = "time_series_split"   def get_static_config() -> dict[str, Any]: """Get the complete non-optimizable configuration.""" return { "database": DatabaseConfig(), "exchange": ExchangeConfig(), "system": SystemConfig(), "environment": EnvironmentConfig(), "trading": TradingConfig(), "training": TrainingConfig(), }   def get_config_section(section_name: str) -> dict[str, Any]: """Get a specific configuration section.""" config = get_static_config()
section = config.get(section_name)
if section is None:
        return {}

if hasattr(section, '__dict__'):
        return section.__dict__
    return section