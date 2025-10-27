"""
Simulator Configuration

Configuration management for paper trading simulation including fees,
slippage models, latency simulation, and position management constraints.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Tuple
from enum import Enum


class SlippageModel(Enum):
    """Slippage calculation models"""
    ORDERBOOK = "orderbook"  # Use real order book depth
    PERCENTAGE = "percentage"  # Fixed percentage


@dataclass
class SimulatorConfig:
    """
    Configuration for paper trading simulator.
    
    Attributes:
        fee_structure: Exchange-specific fee rates {exchange: {maker: float, taker: float}}
        default_taker_fee: Default taker fee (0.0008 = 0.08%)
        default_maker_fee: Default maker fee (0.0006 = 0.06%)
        use_maker_taker_distinction: Whether to distinguish maker/taker fees
        slippage_model: Model for calculating slippage
        max_slippage_pct: Maximum slippage percentage allowed
        orderbook_depth_limit: Number of price levels to use from order book
        enable_latency_simulation: Whether to simulate network latency
        latency_range_ms: Range of latency in milliseconds (min, max)
        allow_multiple_positions: Allow multiple positions per symbol
        allow_pyramiding: Allow scaling into positions
        max_positions_per_symbol: Maximum concurrent positions per symbol
        allow_partial_closes: Allow partial position closes
        max_position_size_usd: Maximum position size in USD
        max_total_exposure_usd: Maximum total exposure across all positions
        orderbook_staleness_threshold_sec: Reject order book if older than this
        price_deviation_threshold_pct: Reject orders if price deviates more than this
    """
    
    # Fee configuration (exchange-specific)
    fee_structure: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "binance": {"maker": 0.0006, "taker": 0.001},
        "okx": {"maker": 0.0008, "taker": 0.001},
        "gateio": {"maker": 0.0006, "taker": 0.001},
        "mexc": {"maker": 0.0007, "taker": 0.001},
        "phemex": {"maker": 0.0005, "taker": 0.001},
    })
    
    default_taker_fee: float = 0.0008  # 0.08%
    default_maker_fee: float = 0.0006  # 0.06%
    use_maker_taker_distinction: bool = True
    
    # Slippage configuration
    slippage_model: SlippageModel = SlippageModel.ORDERBOOK
    max_slippage_pct: float = 0.01  # 1%
    orderbook_depth_limit: int = 20
    
    # Latency simulation
    enable_latency_simulation: bool = True
    latency_range_ms: Tuple[int, int] = (50, 200)
    
    # Position management
    allow_multiple_positions: bool = True
    allow_pyramiding: bool = True
    max_positions_per_symbol: int = 3
    allow_partial_closes: bool = True
    
    # Risk limits
    max_position_size_usd: float = 50000.0
    max_total_exposure_usd: float = 100000.0
    
    # Data validation
    orderbook_staleness_threshold_sec: float = 5.0
    price_deviation_threshold_pct: float = 0.05  # 5%
    
    def get_fee_rates(self, exchange: str) -> Tuple[float, float]:
        """
        Get maker and taker fee rates for an exchange.
        
        Args:
            exchange: Exchange name (e.g., "binance")
            
        Returns:
            Tuple of (maker_fee, taker_fee)
        """
        fees = self.fee_structure.get(exchange.lower(), {})
        maker_fee = fees.get("maker", self.default_maker_fee)
        taker_fee = fees.get("taker", self.default_taker_fee)
        return maker_fee, taker_fee
    
    def validate(self) -> bool:
        """Validate configuration parameters."""
        if self.default_taker_fee < 0 or self.default_maker_fee < 0:
            raise ValueError("Fee rates must be non-negative")
        if self.max_slippage_pct < 0 or self.max_slippage_pct > 1:
            raise ValueError("max_slippage_pct must be between 0 and 1")
        if self.max_positions_per_symbol < 1:
            raise ValueError("max_positions_per_symbol must be at least 1")
        if self.max_position_size_usd <= 0 or self.max_total_exposure_usd <= 0:
            raise ValueError("Position size limits must be positive")
        if self.latency_range_ms[0] < 0 or self.latency_range_ms[1] < self.latency_range_ms[0]:
            raise ValueError("Invalid latency range")
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "fee_structure": self.fee_structure,
            "default_taker_fee": self.default_taker_fee,
            "default_maker_fee": self.default_maker_fee,
            "use_maker_taker_distinction": self.use_maker_taker_distinction,
            "slippage_model": self.slippage_model.value,
            "max_slippage_pct": self.max_slippage_pct,
            "orderbook_depth_limit": self.orderbook_depth_limit,
            "enable_latency_simulation": self.enable_latency_simulation,
            "latency_range_ms": self.latency_range_ms,
            "allow_multiple_positions": self.allow_multiple_positions,
            "allow_pyramiding": self.allow_pyramiding,
            "max_positions_per_symbol": self.max_positions_per_symbol,
            "allow_partial_closes": self.allow_partial_closes,
            "max_position_size_usd": self.max_position_size_usd,
            "max_total_exposure_usd": self.max_total_exposure_usd,
            "orderbook_staleness_threshold_sec": self.orderbook_staleness_threshold_sec,
            "price_deviation_threshold_pct": self.price_deviation_threshold_pct,
        }
