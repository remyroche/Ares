"""
Slippage Calculator

Calculates realistic fill prices based on order book depth, handling
partial fills across multiple price levels.
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import logging
import random

from .config import SimulatorConfig, SlippageModel


@dataclass
class FillResult:
    """Result of slippage calculation"""
    avg_fill_price: float
    filled_quantity: float
    slippage_pct: float
    price_levels_used: List[Tuple[float, float]]  # [(price, qty), ...]
    remaining_quantity: float


class SlippageCalculator:
    """
    Calculate realistic fill prices using order book data.
    
    Supports both order book-based and percentage-based slippage models.
    Handles partial fills across multiple price levels.
    """
    
    def __init__(self, config: SimulatorConfig):
        """
        Initialize slippage calculator.
        
        Args:
            config: Simulator configuration with slippage settings
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def calculate_fill_price(
        self,
        order_book: Dict[str, Any],
        side: str,
        quantity: float,
        order_type: str,
        limit_price: Optional[float] = None
    ) -> FillResult:
        """
        Calculate fill price based on order book.
        
        Args:
            order_book: Order book data with 'bids' and 'asks' arrays
            side: Order side ("buy" or "sell")
            quantity: Order quantity
            order_type: Order type ("market" or "limit")
            limit_price: Limit price (if limit order)
            
        Returns:
            FillResult with average fill price and slippage details
        """
        if self.config.slippage_model == SlippageModel.ORDERBOOK:
            return self._calculate_orderbook_slippage(
                order_book, side, quantity, order_type, limit_price
            )
        else:
            return self._calculate_percentage_slippage(
                order_book, side, quantity, order_type, limit_price
            )
    
    def _calculate_orderbook_slippage(
        self,
        order_book: Dict[str, Any],
        side: str,
        quantity: float,
        order_type: str,
        limit_price: Optional[float] = None
    ) -> FillResult:
        """Calculate slippage using order book depth."""
        
        # Determine which side of the book to use
        is_buy = side.lower() in ["buy", "long"]
        
        # Get appropriate price levels from order book
        if is_buy:
            price_levels = order_book.get("asks", [])  # Buying from asks
            reference_price = self._get_best_price(order_book, "bids")
        else:
            price_levels = order_book.get("bids", [])  # Selling to bids
            reference_price = self._get_best_price(order_book, "asks")
        
        if not price_levels or reference_price == 0:
            self.logger.warning("Invalid order book data, using percentage model")
            return self._calculate_percentage_slippage(
                order_book, side, quantity, order_type, limit_price
            )
        
        # For limit orders, check if price is favorable
        if limit_price and order_type.lower() == "limit":
            if is_buy and limit_price < reference_price:
                # Can't fill at limit (price too low)
                return FillResult(
                    avg_fill_price=0.0,
                    filled_quantity=0.0,
                    slippage_pct=0.0,
                    price_levels_used=[],
                    remaining_quantity=quantity
                )
            elif not is_buy and limit_price > reference_price:
                # Can't fill at limit (price too high)
                return FillResult(
                    avg_fill_price=0.0,
                    filled_quantity=0.0,
                    slippage_pct=0.0,
                    price_levels_used=[],
                    remaining_quantity=quantity
                )
        
        # Calculate weighted average fill price
        remaining_qty = quantity
        total_cost = 0.0
        filled_qty = 0.0
        price_levels_used = []
        
        for level in price_levels[:self.config.orderbook_depth_limit]:
            if remaining_qty <= 0:
                break
            
            # Parse level (format may vary by exchange)
            if isinstance(level, (list, tuple)) and len(level) >= 2:
                price = float(level[0])
                available_qty = float(level[1])
            elif isinstance(level, dict):
                price = float(level.get("price", level.get("p", 0)))
                available_qty = float(level.get("quantity", level.get("q", 0)))
            else:
                self.logger.warning(f"Unexpected order book level format: {level}")
                continue
            
            # Check limit price constraint
            if limit_price:
                if is_buy and price > limit_price:
                    break
                elif not is_buy and price < limit_price:
                    break
            
            # Fill at this price level
            qty_at_level = min(remaining_qty, available_qty)
            total_cost += price * qty_at_level
            filled_qty += qty_at_level
            remaining_qty -= qty_at_level
            
            price_levels_used.append((price, qty_at_level))
            
            # Cap slippage
            slippage = abs(price - reference_price) / reference_price
            if slippage > self.config.max_slippage_pct:
                self.logger.warning(f"Slippage {slippage:.2%} exceeds max {self.config.max_slippage_pct:.2%}")
                break
        
        # Calculate average fill price and slippage
        if filled_qty > 0:
            avg_fill_price = total_cost / filled_qty
            slippage_pct = abs(avg_fill_price - reference_price) / reference_price
        else:
            avg_fill_price = reference_price
            slippage_pct = 0.0
        
        self.logger.debug(
            f"Fill calculated: {side} qty={quantity} filled={filled_qty} "
            f"price={avg_fill_price:.6f} slippage={slippage_pct:.4%}"
        )
        
        return FillResult(
            avg_fill_price=avg_fill_price,
            filled_quantity=filled_qty,
            slippage_pct=slippage_pct,
            price_levels_used=price_levels_used,
            remaining_quantity=remaining_qty
        )
    
    def _calculate_percentage_slippage(
        self,
        order_book: Dict[str, Any],
        side: str,
        quantity: float,
        order_type: str,
        limit_price: Optional[float] = None
    ) -> FillResult:
        """Calculate slippage using fixed percentage model."""
        
        is_buy = side.lower() in ["buy", "long"]
        reference_price = self._get_best_price(order_book, "bids" if not is_buy else "asks")
        
        if reference_price == 0:
            reference_price = 50000.0  # Fallback price
        
        # Apply slippage percentage (higher for market orders)
        slippage_pct = self.config.max_slippage_pct
        if order_type.lower() == "market":
            slippage_pct *= 0.8  # Market orders get slightly worse fills
        
        # Calculate fill price (unfavorable for trader)
        if is_buy:
            fill_price = reference_price * (1 + slippage_pct)
        else:
            fill_price = reference_price * (1 - slippage_pct)
        
        # Check limit price
        if limit_price and order_type.lower() == "limit":
            if is_buy and fill_price > limit_price:
                return FillResult(
                    avg_fill_price=0.0,
                    filled_quantity=0.0,
                    slippage_pct=0.0,
                    price_levels_used=[],
                    remaining_quantity=quantity
                )
            elif not is_buy and fill_price < limit_price:
                return FillResult(
                    avg_fill_price=0.0,
                    filled_quantity=0.0,
                    slippage_pct=0.0,
                    price_levels_used=[],
                    remaining_quantity=quantity
                )
        
        return FillResult(
            avg_fill_price=fill_price,
            filled_quantity=quantity,
            slippage_pct=slippage_pct,
            price_levels_used=[(fill_price, quantity)],
            remaining_quantity=0.0
        )
    
    def _get_best_price(self, order_book: Dict[str, Any], side: str) -> float:
        """Get best (highest bid or lowest ask) price from order book."""
        levels = order_book.get(side, [])
        if not levels:
            return 0.0
        
        first_level = levels[0]
        if isinstance(first_level, (list, tuple)):
            return float(first_level[0])
        elif isinstance(first_level, dict):
            return float(first_level.get("price", first_level.get("p", 0)))
        return 0.0
    
    def validate_orderbook_freshness(
        self,
        order_book: Dict[str, Any]
    ) -> Tuple[bool, float]:
        """
        Validate order book timestamp freshness.
        
        Returns:
            Tuple of (is_fresh, age_seconds)
        """
        timestamp = order_book.get("timestamp", order_book.get("ts", 0))
        if timestamp == 0:
            return True, 0.0
        
        from time import time
        age_seconds = time() - (timestamp / 1000 if timestamp > 1e12 else timestamp)
        is_fresh = age_seconds <= self.config.orderbook_staleness_threshold_sec
        
        if not is_fresh:
            self.logger.warning(
                f"Order book is stale: {age_seconds:.2f}s old "
                f"(threshold: {self.config.orderbook_staleness_threshold_sec}s)"
            )
        
        return is_fresh, age_seconds
