"""
Execution Configuration

Configuration for order execution, exchange interfaces, and execution-related parameters.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from enum import Enum

class ExchangeType(Enum):
    """Supported exchange types."""
    BINANCE = "binance"
    BINANCE_TESTNET = "binance_testnet"
    SIMULATED = "simulated"

class OrderType(Enum):
    """Order types."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"

class OrderSide(Enum):
    """Order sides."""
    BUY = "buy"
    SELL = "sell"

@dataclass
class ExecutionConfig:
    """Execution configuration."""
    
    # Exchange settings
    exchange: ExchangeType = ExchangeType.BINANCE_TESTNET
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    testnet: bool = True
    
    # Order execution
    default_order_type: OrderType = OrderType.MARKET
    order_timeout: int = 30  # seconds
    max_retries: int = 3
    retry_delay: float = 1.0  # seconds
    
    # Slippage and fees
    slippage_tolerance: float = 0.001  # 0.1%
    commission_rate: float = 0.001  # 0.1%
    funding_rate: float = 0.0001  # 0.01%
    
    # Risk limits
    max_order_size: float = 10000.0  # USD
    min_order_size: float = 10.0  # USD
    max_daily_volume: float = 100000.0  # USD
    
    # Execution timing
    execution_delay: float = 0.1  # seconds
    batch_execution: bool = False
    batch_size: int = 5
    
    # Error handling
    enable_circuit_breaker: bool = True
    circuit_breaker_threshold: int = 5  # consecutive failures
    circuit_breaker_timeout: int = 300  # seconds
    
    # Monitoring
    enable_execution_logging: bool = True
    log_all_orders: bool = True
    execution_metrics_interval: int = 60  # seconds
    
    # Advanced settings
    enable_smart_routing: bool = False
    enable_partial_fills: bool = True
    enable_iceberg_orders: bool = False
    
    # Custom parameters
    custom_params: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Enum):
                result[key] = value.value
            elif isinstance(value, dict):
                result[key] = value.copy()
            elif isinstance(value, list):
                result[key] = value.copy()
            else:
                result[key] = value
        return result
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ExecutionConfig":
        """Create configuration from dictionary."""
        # Handle enum conversions
        if "exchange" in config_dict and isinstance(config_dict["exchange"], str):
            config_dict["exchange"] = ExchangeType(config_dict["exchange"])
        if "default_order_type" in config_dict and isinstance(config_dict["default_order_type"], str):
            config_dict["default_order_type"] = OrderType(config_dict["default_order_type"])
        
        return cls(**config_dict)
    
    def validate(self) -> bool:
        """Validate configuration parameters."""
        if self.order_timeout <= 0:
            return False
        if self.max_retries < 0:
            return False
        if self.slippage_tolerance < 0 or self.slippage_tolerance > 1:
            return False
        if self.commission_rate < 0 or self.commission_rate > 1:
            return False
        if self.max_order_size <= 0:
            return False
        if self.min_order_size <= 0:
            return False
        
        return True