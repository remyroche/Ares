"""
Simulator Configuration

Configuration management for paper trading simulation including fees,
slippage models, latency simulation, and position management constraints.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Tuple
from enum import Enum
from src.utils.tprint import tprint


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
        base_spread_bps: Base spread in basis points (2.0 = 0.02%)
        spread_multiplier_by_exchange: Exchange-specific spread multipliers
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
    
    # Spread configuration
    base_spread_bps: float = 2.0  # 0.02% base spread
    spread_multiplier_by_exchange: Dict[str, float] = field(default_factory=lambda: {
        "binance": 1.0,      # Lowest spread (most liquid)
        "okx": 1.2,          # Slightly wider
        "gateio": 1.5,       # Moderate
        "mexc": 1.8,         # Wider spread
        "phemex": 1.3,       # Moderate
        "bingx": 1.6,        # Moderate-wide
    })
    
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
        tprint(f"[CONFIG] get_fee_rates: exchange={exchange}")
        fees = self.fee_structure.get(exchange.lower(), {})
        maker_fee = fees.get("maker", self.default_maker_fee)
        taker_fee = fees.get("taker", self.default_taker_fee)
        tprint(f"[CONFIG] get_fee_rates -> maker={maker_fee:.4%}, taker={taker_fee:.4%}")
        return maker_fee, taker_fee
    
    def get_spread_pct(self, exchange: str) -> float:
        """
        Get spread percentage for an exchange.

        Args:
            exchange: Exchange name (e.g., "binance")

        Returns:
            Spread as decimal (e.g., 0.0002 for 0.02%)
        """
        tprint(f"[CONFIG] get_spread_pct: exchange={exchange}, base_spread_bps={self.base_spread_bps}")
        multiplier = self.spread_multiplier_by_exchange.get(exchange.lower(), 1.0)
        # Convert basis points to decimal: 2.0 bps = 0.0002
        spread_pct = (self.base_spread_bps * multiplier) / 10000.0
        tprint(f"[CONFIG] get_spread_pct -> spread={spread_pct:.4%} (multiplier={multiplier})")
        return spread_pct
    
    def validate(self) -> bool:
        """Validate configuration parameters."""
        tprint("[CONFIG] validate: Starting configuration validation")
        if self.default_taker_fee < 0 or self.default_maker_fee < 0:
            tprint(f"[CONFIG] validate -> FAILED: Fee rates must be non-negative (maker={self.default_maker_fee}, taker={self.default_taker_fee})", color="red")
            raise ValueError("Fee rates must be non-negative")
        if self.max_slippage_pct < 0 or self.max_slippage_pct > 1:
            tprint(f"[CONFIG] validate -> FAILED: max_slippage_pct={self.max_slippage_pct} must be between 0 and 1", color="red")
            raise ValueError("max_slippage_pct must be between 0 and 1")
        if self.max_positions_per_symbol < 1:
            tprint(f"[CONFIG] validate -> FAILED: max_positions_per_symbol={self.max_positions_per_symbol} must be at least 1", color="red")
            raise ValueError("max_positions_per_symbol must be at least 1")
        if self.max_position_size_usd <= 0 or self.max_total_exposure_usd <= 0:
            tprint(f"[CONFIG] validate -> FAILED: Position size limits must be positive (max_position={self.max_position_size_usd}, max_exposure={self.max_total_exposure_usd})", color="red")
            raise ValueError("Position size limits must be positive")
        if self.latency_range_ms[0] < 0 or self.latency_range_ms[1] < self.latency_range_ms[0]:
            tprint(f"[CONFIG] validate -> FAILED: Invalid latency range {self.latency_range_ms}", color="red")
            raise ValueError("Invalid latency range")
        tprint("[CONFIG] validate -> SUCCESS: All configuration parameters valid", color="green")
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        tprint("[CONFIG] to_dict: Converting configuration to dictionary")
        result = {
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
        tprint(f"[CONFIG] to_dict -> dictionary with {len(result)} fields")
        return result
