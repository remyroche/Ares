"""
Unified Configuration for NAS/TAS Backtesting Engine

This module provides a unified configuration system for the backtesting engine,
combining all necessary settings for data, hardware, validation, and backtesting.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from enum import Enum


class ExecutionMode(Enum):
    """Execution modes for backtesting."""
    VECTORIZED = "vectorized"
    PARALLEL = "parallel"
    GPU_ACCELERATED = "gpu_accelerated"
    HYBRID = "hybrid"


@dataclass
class DataConfig:
    """Data configuration for backtesting."""
    symbol: str
    exchange: str
    timeframe: str
    data_dir: str
    data_type: str = "klines"
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    min_data_points: int = 1000
    max_data_points: int = 10000


@dataclass
class HardwareConfig:
    """Hardware optimization configuration."""
    enable_gpu_acceleration: bool = True
    enable_memory_optimization: bool = True
    enable_parallel_processing: bool = True
    memory_limit_gb: float = 8.0
    max_workers: Optional[int] = None
    enable_caching: bool = True
    cache_size_mb: int = 100


@dataclass
class ValidationConfig:
    """Validation configuration."""
    enable_cv_validation: bool = True
    enable_hpo: bool = True
    enable_walk_forward: bool = True
    min_trades_for_validation: int = 10
    confidence_level: float = 0.95


@dataclass
class BacktestingConfig:
    """Backtesting specific configuration."""
    initial_capital: float = 100000.0
    commission_rate: float = 0.001
    slippage_rate: float = 0.0005
    min_position_size: float = 0.01
    max_position_size: float = 0.1
    max_drawdown_threshold: float = 0.2
    stop_loss_threshold: float = 0.05
    take_profit_threshold: float = 0.1


@dataclass
class OutputConfig:
    """Output configuration."""
    save_detailed_results: bool = True
    generate_plots: bool = True
    output_format: str = "parquet"  # parquet, csv, json
    results_directory: str = "backtesting_results"


@dataclass
class UnifiedBacktestingConfig:
    """Unified configuration for the backtesting engine."""
    data: DataConfig
    hardware: HardwareConfig
    validation: ValidationConfig
    backtesting: BacktestingConfig
    output: OutputConfig
    execution_mode: ExecutionMode = ExecutionMode.HYBRID
    
    # Custom parameters
    custom_params: Dict[str, Any] = field(default_factory=dict)


class ConfigBuilder:
    """Builder pattern for creating unified configuration."""
    
    def __init__(self):
        self._data = DataConfig("BTCUSDT", "binance", "1h", "data")
        self._hardware = HardwareConfig()
        self._validation = ValidationConfig()
        self._backtesting = BacktestingConfig()
        self._output = OutputConfig()
        self._execution_mode = ExecutionMode.HYBRID
        self._custom_params = {}
    
    def set_symbol(self, symbol: str) -> 'ConfigBuilder':
        """Set the trading symbol."""
        self._data.symbol = symbol
        return self
    
    def set_exchange(self, exchange: str) -> 'ConfigBuilder':
        """Set the exchange."""
        self._data.exchange = exchange
        return self
    
    def set_timeframe(self, timeframe: str) -> 'ConfigBuilder':
        """Set the timeframe."""
        self._data.timeframe = timeframe
        return self
    
    def set_data_dir(self, data_dir: str) -> 'ConfigBuilder':
        """Set the data directory."""
        self._data.data_dir = data_dir
        return self
    
    def set_date_range(self, start_date: str, end_date: str) -> 'ConfigBuilder':
        """Set the date range."""
        self._data.start_date = start_date
        self._data.end_date = end_date
        return self
    
    def set_execution_mode(self, mode: ExecutionMode) -> 'ConfigBuilder':
        """Set the execution mode."""
        self._execution_mode = mode
        return self
    
    def set_hardware_config(self, **kwargs) -> 'ConfigBuilder':
        """Set hardware configuration."""
        for key, value in kwargs.items():
            if hasattr(self._hardware, key):
                setattr(self._hardware, key, value)
        return self
    
    def set_validation_config(self, **kwargs) -> 'ConfigBuilder':
        """Set validation configuration."""
        for key, value in kwargs.items():
            if hasattr(self._validation, key):
                setattr(self._validation, key, value)
        return self
    
    def set_backtesting_config(self, **kwargs) -> 'ConfigBuilder':
        """Set backtesting configuration."""
        for key, value in kwargs.items():
            if hasattr(self._backtesting, key):
                setattr(self._backtesting, key, value)
        return self
    
    def set_output_config(self, **kwargs) -> 'ConfigBuilder':
        """Set output configuration."""
        for key, value in kwargs.items():
            if hasattr(self._output, key):
                setattr(self._output, key, value)
        return self
    
    def set_custom_params(self, **kwargs) -> 'ConfigBuilder':
        """Set custom parameters."""
        self._custom_params.update(kwargs)
        return self
    
    def build(self) -> UnifiedBacktestingConfig:
        """Build the unified configuration."""
        return UnifiedBacktestingConfig(
            data=self._data,
            hardware=self._hardware,
            validation=self._validation,
            backtesting=self._backtesting,
            output=self._output,
            execution_mode=self._execution_mode,
            custom_params=self._custom_params
        )


def create_config() -> ConfigBuilder:
    """Create a new configuration builder."""
    return ConfigBuilder()


def create_default_config() -> UnifiedBacktestingConfig:
    """Create a default configuration."""
    return (create_config()
            .set_symbol("BTCUSDT")
            .set_exchange("binance")
            .set_timeframe("1h")
            .set_data_dir("data")
            .set_date_range("2024-01-01", "2024-01-31")
            .build())


def create_optimized_config() -> UnifiedBacktestingConfig:
    """Create an optimized configuration for performance."""
    return (create_config()
            .set_symbol("BTCUSDT")
            .set_exchange("binance")
            .set_timeframe("1h")
            .set_data_dir("data")
            .set_date_range("2024-01-01", "2024-01-31")
            .set_execution_mode(ExecutionMode.GPU_ACCELERATED)
            .set_hardware_config(
                enable_gpu_acceleration=True,
                enable_memory_optimization=True,
                enable_parallel_processing=True,
                memory_limit_gb=16.0,
                max_workers=8
            )
            .set_validation_config(
                enable_cv_validation=True,
                enable_hpo=True,
                enable_walk_forward=True
            )
            .set_backtesting_config(
                initial_capital=100000.0,
                commission_rate=0.001,
                slippage_rate=0.0005
            )
            .build())