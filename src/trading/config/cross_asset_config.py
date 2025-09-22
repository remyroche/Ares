"""
Cross-Asset Trading Configuration

Configuration for multi-cryptocurrency trading with consolidated reporting
and cross-asset risk management.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from enum import Enum
from .trading_config import TradingConfig, TradingMode, RiskLevel

class CrossAssetStrategy(Enum):
    """Cross-asset trading strategies."""
    EQUAL_WEIGHT = "equal_weight"
    MARKET_CAP_WEIGHT = "market_cap_weight"
    VOLATILITY_ADJUSTED = "volatility_adjusted"
    CORRELATION_MINIMIZED = "correlation_minimized"

class RebalancingFrequency(Enum):
    """Rebalancing frequency options."""
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"

@dataclass
class SymbolConfiguration:
    """Configuration for individual trading symbols."""
    symbol: str
    exchange: str = "binance"
    trading_mode: TradingMode = TradingMode.PAPER
    account_balance: float = 1000.0

    # Symbol-specific parameters
    volatility_adjustment: float = 1.0
    liquidity_factor: float = 1.0
    correlation_factor: float = 1.0

    # Position sizing
    base_position_size: float = 0.1
    max_position_size: float = 0.25
    min_position_size: float = 0.01

    # Risk parameters
    risk_level: RiskLevel = RiskLevel.MODERATE
    max_portfolio_risk: float = 0.02
    max_symbol_risk: float = 0.05

    # Trading parameters
    confidence_threshold: float = 0.6
    max_daily_trades: int = 10
    min_trade_interval: int = 60  # seconds

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
            else:
                result[key] = value
        return result

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "SymbolConfiguration":
        """Create configuration from dictionary."""
        # Handle enum conversions
        if "trading_mode" in config_dict and isinstance(config_dict["trading_mode"], str):
            config_dict["trading_mode"] = TradingMode(config_dict["trading_mode"])
        if "risk_level" in config_dict and isinstance(config_dict["risk_level"], str):
            config_dict["risk_level"] = RiskLevel(config_dict["risk_level"])

        return cls(**config_dict)

@dataclass
class CrossAssetConfig:
    """Configuration for cross-asset trading manager."""

    # Basic configuration
    symbols: List[str] = field(default_factory=lambda: ["ETHUSDT", "BTCUSDT", "ADAUSDT"])
    primary_symbol: str = "ETHUSDT"
    exchange: str = "binance"
    trading_mode: TradingMode = TradingMode.PAPER
    total_account_balance: float = 10000.0

    # Cross-asset strategy
    strategy: CrossAssetStrategy = CrossAssetStrategy.EQUAL_WEIGHT
    rebalancing_frequency: RebalancingFrequency = RebalancingFrequency.DAILY
    rebalancing_threshold: float = 0.1  # 10% deviation triggers rebalancing

    # Risk management
    max_concurrent_symbols: int = 3
    max_portfolio_risk: float = 0.05  # 5% total portfolio risk
    max_symbol_concentration: float = 0.3  # 30% max per symbol
    risk_per_trade: float = 0.02  # 2% risk per trade

    # Trade execution control
    max_trades_per_minute: int = 5
    max_trades_per_hour: int = 50
    trade_timeout_seconds: int = 30

    # Symbol-specific configurations
    symbol_configs: Dict[str, SymbolConfiguration] = field(default_factory=dict)

    # Reporting and monitoring
    consolidated_reporting: bool = True
    real_time_monitoring: bool = True
    export_directory: str = "cross_asset_reports"

    # Advanced settings
    enable_dynamic_allocation: bool = True
    enable_correlation_monitoring: bool = True
    enable_liquidity_monitoring: bool = True

    # Custom parameters
    custom_params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize symbol configurations if not provided."""
        if not self.symbol_configs:
            self._create_default_symbol_configs()

    def _create_default_symbol_configs(self):
        """Create default symbol configurations."""
        base_config = SymbolConfiguration(
            exchange=self.exchange,
            trading_mode=self.trading_mode,
            account_balance=self.total_account_balance / len(self.symbols)
        )

        for symbol in self.symbols:
            config = SymbolConfiguration(
                symbol=symbol,
                **base_config.__dict__
            )

            # Apply symbol-specific adjustments
            config.volatility_adjustment = self._get_volatility_adjustment(symbol)
            config.liquidity_factor = self._get_liquidity_factor(symbol)
            config.correlation_factor = self._get_correlation_factor(symbol)

            self.symbol_configs[symbol] = config

    def _get_volatility_adjustment(self, symbol: str) -> float:
        """Get volatility adjustment for symbol."""
        adjustments = {
            'BTCUSDT': 1.0,    # Bitcoin - baseline
            'ETHUSDT': 1.2,    # Ethereum - more volatile
            'BNBUSDT': 0.8,    # BNB - less volatile
            'ADAUSDT': 1.5,    # Cardano - more volatile
            'SOLUSDT': 1.3,    # Solana - volatile
            'DOTUSDT': 1.4,    # Polkadot - volatile
            'LINKUSDT': 1.6,   # Chainlink - very volatile
            'LTCUSDT': 1.1,    # Litecoin - slightly more volatile
        }
        return adjustments.get(symbol, 1.0)

    def _get_liquidity_factor(self, symbol: str) -> float:
        """Get liquidity factor for symbol."""
        factors = {
            'BTCUSDT': 1.0,    # Bitcoin - highest liquidity
            'ETHUSDT': 0.9,    # Ethereum - very high liquidity
            'BNBUSDT': 0.7,    # BNB - high liquidity
            'ADAUSDT': 0.5,    # Cardano - moderate liquidity
            'SOLUSDT': 0.6,    # Solana - moderate liquidity
            'DOTUSDT': 0.4,    # Polkadot - moderate liquidity
            'LINKUSDT': 0.3,   # Chainlink - moderate liquidity
            'LTCUSDT': 0.7,    # Litecoin - high liquidity
        }
        return factors.get(symbol, 0.5)

    def _get_correlation_factor(self, symbol: str) -> float:
        """Get correlation factor for symbol."""
        factors = {
            'BTCUSDT': 1.0,    # Bitcoin - highly correlated with market
            'ETHUSDT': 0.8,    # Ethereum - correlated with BTC
            'BNBUSDT': 0.6,    # BNB - moderately correlated
            'ADAUSDT': 0.4,    # Cardano - less correlated
            'SOLUSDT': 0.7,    # Solana - correlated with ETH
            'DOTUSDT': 0.5,    # Polkadot - moderately correlated
            'LINKUSDT': 0.3,   # Chainlink - less correlated
            'LTCUSDT': 0.8,    # Litecoin - correlated with BTC
        }
        return factors.get(symbol, 0.5)

    def get_symbol_config(self, symbol: str) -> Optional[SymbolConfiguration]:
        """Get configuration for specific symbol."""
        return self.symbol_configs.get(symbol)

    def update_symbol_config(self, symbol: str, config: SymbolConfiguration):
        """Update configuration for specific symbol."""
        self.symbol_configs[symbol] = config

    def add_symbol(self, symbol: str, config: Optional[SymbolConfiguration] = None):
        """Add a new symbol to the configuration."""
        if config is None:
            # Create default config
            config = SymbolConfiguration(
                symbol=symbol,
                exchange=self.exchange,
                trading_mode=self.trading_mode,
                account_balance=self.total_account_balance / (len(self.symbols) + 1)
            )

        self.symbol_configs[symbol] = config
        if symbol not in self.symbols:
            self.symbols.append(symbol)

    def remove_symbol(self, symbol: str):
        """Remove a symbol from the configuration."""
        if symbol in self.symbol_configs:
            del self.symbol_configs[symbol]
        if symbol in self.symbols:
            self.symbols.remove(symbol)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        result = {}
        for key, value in self.__dict__.items():
            if key == 'symbol_configs':
                result[key] = {
                    symbol: config.to_dict()
                    for symbol, config in value.items()
                }
            elif isinstance(value, Enum):
                result[key] = value.value
            elif isinstance(value, dict):
                result[key] = value.copy()
            elif isinstance(value, list):
                result[key] = value.copy()
            else:
                result[key] = value
        return result

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "CrossAssetConfig":
        """Create configuration from dictionary."""
        # Handle enum conversions
        if "trading_mode" in config_dict and isinstance(config_dict["trading_mode"], str):
            config_dict["trading_mode"] = TradingMode(config_dict["trading_mode"])
        if "strategy" in config_dict and isinstance(config_dict["strategy"], str):
            config_dict["strategy"] = CrossAssetStrategy(config_dict["strategy"])
        if "rebalancing_frequency" in config_dict and isinstance(config_dict["rebalancing_frequency"], str):
            config_dict["rebalancing_frequency"] = RebalancingFrequency(config_dict["rebalancing_frequency"])

        # Handle symbol configurations
        if "symbol_configs" in config_dict:
            symbol_configs = {}
            for symbol, config_data in config_dict["symbol_configs"].items():
                symbol_configs[symbol] = SymbolConfiguration.from_dict(config_data)
            config_dict["symbol_configs"] = symbol_configs

        return cls(**config_dict)

    def validate(self) -> bool:
        """Validate configuration parameters."""
        if not self.symbols:
            return False

        if self.total_account_balance <= 0:
            return False

        if self.max_concurrent_symbols <= 0:
            return False

        if self.max_portfolio_risk <= 0 or self.max_portfolio_risk > 1:
            return False

        if self.max_symbol_concentration <= 0 or self.max_symbol_concentration > 1:
            return False

        if self.risk_per_trade <= 0 or self.risk_per_trade > 0.1:  # Max 10% per trade
            return False

        return True

    def get_trading_config(self, symbol: str) -> Optional[TradingConfig]:
        """Convert symbol configuration to TradingConfig format."""
        symbol_config = self.get_symbol_config(symbol)
        if not symbol_config:
            return None

        return TradingConfig(
            mode=symbol_config.trading_mode,
            risk_level=symbol_config.risk_level,
            max_portfolio_risk=symbol_config.max_portfolio_risk,
            max_drawdown=0.15,  # Default max drawdown
            max_leverage=3.0,   # Default max leverage
            base_position_size=symbol_config.base_position_size,
            max_position_size=symbol_config.max_position_size,
            min_position_size=symbol_config.min_position_size,
            symbols=[symbol],
            primary_symbol=symbol,
            custom_params={
                'volatility_adjustment': symbol_config.volatility_adjustment,
                'liquidity_factor': symbol_config.liquidity_factor,
                'correlation_factor': symbol_config.correlation_factor,
                'confidence_threshold': symbol_config.confidence_threshold,
                'max_daily_trades': symbol_config.max_daily_trades,
                'symbol_config': symbol_config.to_dict()
            }
        )