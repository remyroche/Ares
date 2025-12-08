"""
Trading Configuration

Configuration management for live trading operations.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
from src.config.leverage_constants import MAX_LEVERAGE, validate_leverage


class TradingMode(Enum):
    """Trading mode enumeration"""
    PAPER = "paper"
    LIVE = "live"
    BACKTEST = "backtest"


class OrderType(Enum):
    """Order type enumeration"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    """Order side enumeration"""
    BUY = "buy"
    SELL = "sell"


@dataclass
class TradingConfig:
    """Configuration for live trading operations"""
    
    # Trading Mode
    mode: TradingMode = TradingMode.PAPER
    
    # Exchange Configuration
    exchange_name: str = "binance"
    symbols: List[str] = field(default_factory=lambda: ["BTCUSDT"])
    direction: str = "long"  # "long", "short", or "both"
    
    # Risk Management
    max_position_size: float = 1000.0
    max_daily_loss: float = 100.0
    max_leverage: float = MAX_LEVERAGE  # Using centralized max leverage
    stop_loss_percentage: float = 2.0
    take_profit_percentage: float = 4.0
    
    # Order Management
    order_timeout: int = 30  # seconds
    max_retries: int = 3
    retry_delay: float = 1.0  # seconds
    
    # Data Streaming
    data_update_interval: float = 1.0  # seconds
    reconnect_attempts: int = 5
    reconnect_delay: float = 5.0  # seconds
    
    # Performance Monitoring
    performance_log_interval: int = 60  # seconds
    trade_log_enabled: bool = True
    metrics_enabled: bool = True
    
    # API Configuration
    api_rate_limit: int = 1200  # requests per minute
    api_timeout: int = 30  # seconds
    
    # Additional Parameters
    custom_parameters: Dict[str, Any] = field(default_factory=dict)
    
    def validate(self) -> bool:
        """Validate configuration parameters"""
        if self.max_position_size <= 0:
            raise ValueError("max_position_size must be positive")
        if self.max_daily_loss <= 0:
            raise ValueError("max_daily_loss must be positive")
        # Validate leverage using centralized validation
        self.max_leverage = validate_leverage(self.max_leverage)
        if self.max_leverage <= 0:
            raise ValueError("max_leverage must be positive")
        if self.stop_loss_percentage <= 0:
            raise ValueError("stop_loss_percentage must be positive")
        if self.take_profit_percentage <= 0:
            raise ValueError("take_profit_percentage must be positive")
        if not self.symbols:
            raise ValueError("At least one symbol must be specified")
        if self.direction not in ["long", "short", "both"]:
            raise ValueError("direction must be 'long', 'short', or 'both'")
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "mode": self.mode.value,
            "exchange_name": self.exchange_name,
            "symbols": self.symbols,
            "direction": self.direction,
            "max_position_size": self.max_position_size,
            "max_daily_loss": self.max_daily_loss,
            "max_leverage": self.max_leverage,
            "stop_loss_percentage": self.stop_loss_percentage,
            "take_profit_percentage": self.take_profit_percentage,
            "order_timeout": self.order_timeout,
            "max_retries": self.max_retries,
            "retry_delay": self.retry_delay,
            "data_update_interval": self.data_update_interval,
            "reconnect_attempts": self.reconnect_attempts,
            "reconnect_delay": self.reconnect_delay,
            "performance_log_interval": self.performance_log_interval,
            "trade_log_enabled": self.trade_log_enabled,
            "metrics_enabled": self.metrics_enabled,
            "api_rate_limit": self.api_rate_limit,
            "api_timeout": self.api_timeout,
            "custom_parameters": self.custom_parameters
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TradingConfig":
        """Create configuration from dictionary"""
        config = cls()
        
        if "mode" in data:
            config.mode = TradingMode(data["mode"])
        if "exchange_name" in data:
            config.exchange_name = data["exchange_name"]
        if "symbols" in data:
            config.symbols = data["symbols"]
        if "direction" in data:
            config.direction = data["direction"]
        if "max_position_size" in data:
            config.max_position_size = data["max_position_size"]
        if "max_daily_loss" in data:
            config.max_daily_loss = data["max_daily_loss"]
        if "max_leverage" in data:
            config.max_leverage = data["max_leverage"]
        if "stop_loss_percentage" in data:
            config.stop_loss_percentage = data["stop_loss_percentage"]
        if "take_profit_percentage" in data:
            config.take_profit_percentage = data["take_profit_percentage"]
        if "order_timeout" in data:
            config.order_timeout = data["order_timeout"]
        if "max_retries" in data:
            config.max_retries = data["max_retries"]
        if "retry_delay" in data:
            config.retry_delay = data["retry_delay"]
        if "data_update_interval" in data:
            config.data_update_interval = data["data_update_interval"]
        if "reconnect_attempts" in data:
            config.reconnect_attempts = data["reconnect_attempts"]
        if "reconnect_delay" in data:
            config.reconnect_delay = data["reconnect_delay"]
        if "performance_log_interval" in data:
            config.performance_log_interval = data["performance_log_interval"]
        if "trade_log_enabled" in data:
            config.trade_log_enabled = data["trade_log_enabled"]
        if "metrics_enabled" in data:
            config.metrics_enabled = data["metrics_enabled"]
        if "api_rate_limit" in data:
            config.api_rate_limit = data["api_rate_limit"]
        if "api_timeout" in data:
            config.api_timeout = data["api_timeout"]
        if "custom_parameters" in data:
            config.custom_parameters = data["custom_parameters"]
        
        return config