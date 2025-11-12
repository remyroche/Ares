"""
Fee Calculator

Calculates trading fees based on exchange-specific rates and order characteristics.
"""

from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass
import logging

from src.utils.tprint import (
    tprint, tprint_logged, tprint_timer, tprint_performance,
    tprint_data_preview, tprint_data_format, tprint_feature_counts, LogLevel
)
from .config import SimulatorConfig


@dataclass
class FeeResult:
    """Result of fee calculation"""
    fee_amount: float
    fee_percentage: float
    fee_type: str  # "maker" or "taker"
    exchange: str
    is_maker: bool


@tprint_logged(LogLevel.INFO, include_args=True)
class FeeCalculator:
    """
    Calculate trading fees based on exchange, order type, and quantity.
    
    Supports exchange-specific fee structures and maker/taker distinctions.
    """
    
    def __init__(self, config: SimulatorConfig):
        """
        Initialize fee calculator.

        Args:
            config: Simulator configuration with fee structure
        """
        tprint(f"[FEE_CALC] __init__: Initializing fee calculator")
        self.config = config
        self.logger = logging.getLogger(__name__)
        tprint(f"[FEE_CALC] __init__ -> initialized with use_maker_taker_distinction={config.use_maker_taker_distinction}")
    
    @tprint_logged(LogLevel.INFO, include_args=True)
    def calculate_fee(
        self,
        exchange: str,
        quantity: float,
        price: float,
        order_type: str,
        is_maker: Optional[bool] = None
    ) -> FeeResult:
        """
        Calculate fee for an order.

        Args:
            exchange: Exchange name (e.g., "binance")
            quantity: Order quantity
            price: Order price
            order_type: Order type ("market", "limit", etc.)
            is_maker: Whether order is maker (adds liquidity)
                     If None, determines based on order_type

        Returns:
            FeeResult with fee amount and metadata
        """
        tprint(f"[FEE_CALC] calculate_fee: exchange={exchange}, qty={quantity}, price={price}, order_type={order_type}, is_maker={is_maker}")

        # Preview order parameters
        order_params = {
            "exchange": exchange,
            "quantity": quantity,
            "price": price,
            "order_type": order_type,
            "is_maker": is_maker
        }
        tprint_data_preview(order_params, "Order Parameters", max_rows=5)

        # Determine if maker or taker with timer
        with tprint_timer("Fee determination"):
            if is_maker is None:
                is_maker = order_type.lower() == "limit"  # Limit orders are typically makers
                tprint(f"[FEE_CALC] calculate_fee: Determined is_maker={is_maker} based on order_type")

            # Get fee rates for exchange
            maker_fee, taker_fee = self.config.get_fee_rates(exchange)

            # Select appropriate fee rate
            if self.config.use_maker_taker_distinction and is_maker:
                fee_rate = maker_fee
                fee_type = "maker"
            else:
                fee_rate = taker_fee
                fee_type = "taker"

            # Calculate fee amount
            notional_value = quantity * price
            fee_amount = notional_value * fee_rate

        tprint(f"[FEE_CALC] calculate_fee -> {fee_type} fee={fee_amount:.6f} ({fee_rate*100:.4f}%), notional={notional_value:.2f}")
        
        # Performance logging for fee calculation
        tprint_performance("Fee calculation", 0.001)

        self.logger.debug(
            f"Fee calculated: {exchange} {order_type} "
            f"qty={quantity} price={price} {fee_type} fee={fee_amount:.6f} ({fee_rate*100:.4f}%)"
        )

        result = FeeResult(
            fee_amount=fee_amount,
            fee_percentage=fee_rate,
            fee_type=fee_type,
            exchange=exchange,
            is_maker=is_maker
        )
        return result
    
    @tprint_logged(LogLevel.INFO, include_args=True)
    def calculate_total_fee(
        self,
        entry_fee: FeeResult,
        exit_fee: FeeResult
    ) -> Dict[str, Any]:
        """
        Calculate total fees for round trip (entry + exit).

        Args:
            entry_fee: FeeResult from entry trade
            exit_fee: FeeResult from exit trade

        Returns:
            Dictionary with total fees and breakdown
        """
        tprint(f"[FEE_CALC] calculate_total_fee: entry_fee={entry_fee.fee_amount:.6f} ({entry_fee.fee_type}), exit_fee={exit_fee.fee_amount:.6f} ({exit_fee.fee_type})")

        # Preview fee calculation data
        fee_data = {
            "entry_fee_amount": entry_fee.fee_amount,
            "entry_fee_type": entry_fee.fee_type,
            "exit_fee_amount": exit_fee.fee_amount,
            "exit_fee_type": exit_fee.fee_type
        }
        tprint_data_preview(fee_data, "Fee Calculation Data", max_rows=5)

        with tprint_timer("Total fee calculation"):
            total_fee = entry_fee.fee_amount + exit_fee.fee_amount
            total_fee_pct = entry_fee.fee_percentage + exit_fee.fee_percentage

            result = {
                "entry_fee": entry_fee.fee_amount,
                "exit_fee": exit_fee.fee_amount,
                "total_fee": total_fee,
                "entry_fee_pct": entry_fee.fee_percentage * 100,
                "exit_fee_pct": exit_fee.fee_percentage * 100,
                "total_fee_pct": total_fee_pct * 100,
                "fee_type": f"{entry_fee.fee_type}/{exit_fee.fee_type}"
            }

        tprint(f"[FEE_CALC] calculate_total_fee -> total_fee={total_fee:.6f} ({total_fee_pct*100:.4f}%)")
        
        # Performance logging for total fee calculation
        tprint_performance("Total fee calculation", 0.001)
        return result
