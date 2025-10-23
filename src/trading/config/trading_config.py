"""
Trading Configuration

Central configuration for the trading system including general trading parameters,
risk limits, and system-wide settings.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from enum import Enum
from src.config.leverage_constants import MAX_LEVERAGE, validate_leverage

class TradingMode(Enum):
    """Trading execution modes."""
    PAPER = "paper"
    LIVE = "live"
    BACKTEST = "backtest"
    SIMULATION = "simulation"

class RiskLevel(Enum):
    """Risk tolerance levels."""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"

@dataclass
class TradingConfig:
    """Main trading configuration."""

    # Trading mode
    mode: TradingMode = TradingMode.PAPER

    # Risk management
    risk_level: RiskLevel = RiskLevel.MODERATE
    max_portfolio_risk: float = 0.02  # 2% max portfolio risk per trade
    max_drawdown: float = 0.15  # 15% max drawdown
    max_leverage: float = MAX_LEVERAGE  # Maximum leverage allowed (centralized)

    # Position sizing
    base_position_size: float = 0.1  # 10% base position size
    max_position_size: float = 0.25  # 25% max position size
    min_position_size: float = 0.01  # 1% min position size

    # Trading parameters
    symbols: List[str] = field(default_factory=lambda: ["ETHUSDT", "BTCUSDT"])
    primary_symbol: str = "ETHUSDT"
    trading_hours: Dict[str, Any] = field(default_factory=dict)

    # Regime-based parameters
    regime_confidence_threshold: float = 0.7  # Minimum confidence for regime-based decisions
    regime_transition_threshold: float = 0.3  # Threshold for regime change detection

    # Execution parameters
    slippage_tolerance: float = 0.001  # 0.1% slippage tolerance
    commission_rate: float = 0.001  # 0.1% commission rate
    order_timeout: int = 30  # Order timeout in seconds

    # Monitoring parameters
    performance_report_interval: int = 3600  # 1 hour
    trade_log_level: str = "INFO"
    enable_real_time_monitoring: bool = True

    # Advanced settings
    enable_regime_switching: bool = True
    enable_dynamic_sizing: bool = True
    enable_risk_scaling: bool = True

    # Cross-timeframe confirmation settings
    cross_timeframe_confirmation: Dict[str, Any] = field(default_factory=lambda: {
        'enabled': True,
        'max_regime_difference': 0,
        'max_confidence_delta': 0.2,
        'downgrade_confidence_factor': 0.6,
        'reject_on_disagreement': False,
        'rejection_confidence': 0.0,
    })

    # Exit strategy configuration (optimized trailing settings)
    exit_strategy: Dict[str, Any] = field(default_factory=dict)

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
    def from_dict(cls, config_dict: Dict[str, Any]) -> "TradingConfig":
        """Create configuration from dictionary."""
        # Handle enum conversions
        if "mode" in config_dict and isinstance(config_dict["mode"], str):
            config_dict["mode"] = TradingMode(config_dict["mode"])
        if "risk_level" in config_dict and isinstance(config_dict["risk_level"], str):
            config_dict["risk_level"] = RiskLevel(config_dict["risk_level"])

        return cls(**config_dict)

    def validate(self) -> bool:
        """Validate configuration parameters."""
        if self.max_portfolio_risk <= 0 or self.max_portfolio_risk > 1:
            return False
        if self.max_drawdown <= 0 or self.max_drawdown > 1:
            return False
        # Validate leverage using centralized validation
        self.max_leverage = validate_leverage(self.max_leverage)
        if self.max_leverage <= 0:
            return False
        if self.base_position_size <= 0 or self.base_position_size > 1:
            return False
        if self.regime_confidence_threshold <= 0 or self.regime_confidence_threshold > 1:
            return False

        return True
