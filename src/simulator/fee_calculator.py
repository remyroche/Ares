"""
Fee Calculator

Calculates trading fees based on exchange-specific rates and order characteristics.
"""

from typing import Dict, Any, Tuple
from dataclasses import dataclass
import logging

from .config import SimulatorConfig


@dataclass
class FeeResult:
    """Result of fee calculation"""
    fee_amount: float
    fee_percentage: float
    fee_type: str  # "maker" or "taker"
    exchange: str
    is_maker: bool


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
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def calculate_fee(
        self,
        exchange: str,
        quantity: float,
        price: float,
        order_type: str,
        is_maker: bool = None
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
        # Determine if maker or taker
        if is_maker is None:
            is_maker = order_type.lower() == "limit"  # Limit orders are typically makers
        
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
        
        self.logger.debug(
            f"Fee calculated: {exchange} {order_type} "
            f"qty={quantity} price={price} {fee_type} fee={fee_amount:.6f} ({fee_rate*100:.4f}%)"
        )
        
        return FeeResult(
            fee_amount=fee_amount,
            fee_percentage=fee_rate,
            fee_type=fee_type,
            exchange=exchange,
            is_maker=is_maker
        )
    
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
        total_fee = entry_fee.fee_amount + exit_fee.fee_amount
        total_fee_pct = entry_fee.fee_percentage + exit_fee.fee_percentage
        
        return {
            "entry_fee": entry_fee.fee_amount,
            "exit_fee": exit_fee.fee_amount,
            "total_fee": total_fee,
            "entry_fee_pct": entry_fee.fee_percentage * 100,
            "exit_fee_pct": exit_fee.fee_percentage * 100,
            "total_fee_pct": total_fee_pct * 100,
            "fee_type": f"{entry_fee.fee_type}/{exit_fee.fee_type}"
        }
