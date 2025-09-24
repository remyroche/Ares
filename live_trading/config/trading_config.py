"""
Trading Configuration

Configuration classes and utilities for trading operations.
"""

from dataclasses import dataclass
from typing import List, Dict, Any
import json
import os


@dataclass
class TradingConfig:
    """Configuration for live trading operations"""
    exchange_name: str
    symbols: List[str]
    max_position_size: float
    max_daily_trades: int
    risk_per_trade: float
    enable_data_streaming: bool = True
    enable_order_execution: bool = True
    api_key: str = ""
    api_secret: str = ""

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TradingConfig':
        """Create TradingConfig from dictionary."""
        return cls(
            exchange_name=config_dict.get('exchange_name', 'binance'),
            symbols=config_dict.get('symbols', ['BTCUSDT']),
            max_position_size=config_dict.get('max_position_size', 10000.0),
            max_daily_trades=config_dict.get('max_daily_trades', 10),
            risk_per_trade=config_dict.get('risk_per_trade', 0.02),
            enable_data_streaming=config_dict.get('enable_data_streaming', True),
            enable_order_execution=config_dict.get('enable_order_execution', True),
            api_key=config_dict.get('api_key', ''),
            api_secret=config_dict.get('api_secret', '')
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'exchange_name': self.exchange_name,
            'symbols': self.symbols,
            'max_position_size': self.max_position_size,
            'max_daily_trades': self.max_daily_trades,
            'risk_per_trade': self.risk_per_trade,
            'enable_data_streaming': self.enable_data_streaming,
            'enable_order_execution': self.enable_order_execution,
            'api_key': self.api_key,
            'api_secret': self.api_secret
        }

    @classmethod
    def load_from_file(cls, config_path: str) -> 'TradingConfig':
        """Load configuration from JSON file."""
        try:
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            return cls.from_dict(config_dict)
        except Exception as e:
            raise ValueError(f"Failed to load config from {config_path}: {e}")

    def save_to_file(self, config_path: str) -> None:
        """Save configuration to JSON file."""
        try:
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            with open(config_path, 'w') as f:
                json.dump(self.to_dict(), f, indent=2)
        except Exception as e:
            raise ValueError(f"Failed to save config to {config_path}: {e}")


# Default configurations for different exchanges (futures/perp trading)
DEFAULT_CONFIGS = {
    'binance': TradingConfig(
        exchange_name='binance',
        symbols=['BTCUSDT', 'ETHUSDT', 'BNBUSDT'],  # Futures symbols
        max_position_size=10000.0,
        max_daily_trades=20,
        risk_per_trade=0.02,
        enable_data_streaming=True,
        enable_order_execution=True
    ),
    'okx': TradingConfig(
        exchange_name='okx',
        symbols=['BTC-USDT', 'ETH-USDT'],
        max_position_size=5000.0,
        max_daily_trades=15,
        risk_per_trade=0.03,
        enable_data_streaming=True,
        enable_order_execution=True
    ),
    'gateio': TradingConfig(
        exchange_name='gateio',
        symbols=['BTC_USDT', 'ETH_USDT'],
        max_position_size=3000.0,
        max_daily_trades=10,
        risk_per_trade=0.025,
        enable_data_streaming=True,
        enable_order_execution=True
    ),
    'mexc': TradingConfig(
        exchange_name='mexc',
        symbols=['BTCUSDT', 'ETHUSDT'],
        max_position_size=2000.0,
        max_daily_trades=8,
        risk_per_trade=0.03,
        enable_data_streaming=True,
        enable_order_execution=True
    )
}

def get_default_config(exchange_name: str) -> TradingConfig:
    """Get default configuration for an exchange."""
    return DEFAULT_CONFIGS.get(exchange_name, DEFAULT_CONFIGS['binance'])