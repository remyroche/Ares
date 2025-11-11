"""
Slippage Calculator

Calculates realistic fill prices based on order book depth, handling
partial fills across multiple price levels.
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import logging
import random

from src.utils.tprint import tprint
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
        tprint(f"[SLIP_CALC] __init__: Initializing slippage calculator with model={config.slippage_model.value}")
        self.config = config
        self.logger = logging.getLogger(__name__)
        tprint(f"[SLIP_CALC] __init__ -> initialized (max_slippage={config.max_slippage_pct:.2%}, depth_limit={config.orderbook_depth_limit})")
    
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
        tprint(f"[SLIP_CALC] calculate_fill_price: side={side}, qty={quantity}, type={order_type}, limit={limit_price}, model={self.config.slippage_model.value}")

        if self.config.slippage_model == SlippageModel.ORDERBOOK:
            result = self._calculate_orderbook_slippage(
                order_book, side, quantity, order_type, limit_price
            )
        else:
            result = self._calculate_percentage_slippage(
                order_book, side, quantity, order_type, limit_price
            )

        tprint(f"[SLIP_CALC] calculate_fill_price -> filled={result.filled_quantity}, avg_price={result.avg_fill_price:.6f}, slippage={result.slippage_pct:.4%}")
        return result
    
    def _calculate_orderbook_slippage(
        self,
        order_book: Dict[str, Any],
        side: str,
        quantity: float,
        order_type: str,
        limit_price: Optional[float] = None
    ) -> FillResult:
        """Calculate slippage using order book depth."""
        tprint(f"[SLIP_CALC] _calculate_orderbook_slippage: side={side}, qty={quantity}")

        # Determine which side of the book to use
        is_buy = side.lower() in ["buy", "long"]

        # Get appropriate price levels from order book
        if is_buy:
            price_levels = order_book.get("asks", [])  # Buying from asks
            reference_price = self._get_best_price(order_book, "bids")
        else:
            price_levels = order_book.get("bids", [])  # Selling to bids
            reference_price = self._get_best_price(order_book, "asks")

        tprint(f"[SLIP_CALC] _calculate_orderbook_slippage: is_buy={is_buy}, price_levels={len(price_levels)}, ref_price={reference_price:.6f}")

        if not price_levels or reference_price == 0:
            tprint(f"[SLIP_CALC] _calculate_orderbook_slippage: Invalid order book, falling back to percentage model", color="yellow")
            self.logger.warning("Invalid order book data, using percentage model")
            return self._calculate_percentage_slippage(
                order_book, side, quantity, order_type, limit_price
            )
        
        # For limit orders, check if price is favorable
        if limit_price and order_type.lower() == "limit":
            if is_buy and limit_price < reference_price:
                # Can't fill at limit (price too low)
                tprint(f"[SLIP_CALC] _calculate_orderbook_slippage -> UNFILLED: limit price {limit_price:.6f} too low (ref={reference_price:.6f})", color="yellow")
                return FillResult(
                    avg_fill_price=0.0,
                    filled_quantity=0.0,
                    slippage_pct=0.0,
                    price_levels_used=[],
                    remaining_quantity=quantity
                )
            elif not is_buy and limit_price > reference_price:
                # Can't fill at limit (price too high)
                tprint(f"[SLIP_CALC] _calculate_orderbook_slippage -> UNFILLED: limit price {limit_price:.6f} too high (ref={reference_price:.6f})", color="yellow")
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

        tprint(f"[SLIP_CALC] _calculate_orderbook_slippage -> filled={filled_qty:.6f}/{quantity:.6f}, avg_price={avg_fill_price:.6f}, slippage={slippage_pct:.4%}, levels_used={len(price_levels_used)}")

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
        tprint(f"[SLIP_CALC] _calculate_percentage_slippage: side={side}, qty={quantity}, type={order_type}")

        is_buy = side.lower() in ["buy", "long"]
        reference_price = self._get_best_price(order_book, "bids" if not is_buy else "asks")

        if reference_price == 0:
            reference_price = 50000.0  # Fallback price
            tprint(f"[SLIP_CALC] _calculate_percentage_slippage: Using fallback price={reference_price}", color="yellow")
        
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
                tprint(f"[SLIP_CALC] _calculate_percentage_slippage -> UNFILLED: fill_price {fill_price:.6f} > limit {limit_price:.6f}", color="yellow")
                return FillResult(
                    avg_fill_price=0.0,
                    filled_quantity=0.0,
                    slippage_pct=0.0,
                    price_levels_used=[],
                    remaining_quantity=quantity
                )
            elif not is_buy and fill_price < limit_price:
                tprint(f"[SLIP_CALC] _calculate_percentage_slippage -> UNFILLED: fill_price {fill_price:.6f} < limit {limit_price:.6f}", color="yellow")
                return FillResult(
                    avg_fill_price=0.0,
                    filled_quantity=0.0,
                    slippage_pct=0.0,
                    price_levels_used=[],
                    remaining_quantity=quantity
                )

        tprint(f"[SLIP_CALC] _calculate_percentage_slippage -> filled={quantity}, price={fill_price:.6f}, slippage={slippage_pct:.4%}")
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
            tprint(f"[SLIP_CALC] _get_best_price: No {side} levels in order book", color="yellow")
            return 0.0

        first_level = levels[0]
        if isinstance(first_level, (list, tuple)):
            price = float(first_level[0])
        elif isinstance(first_level, dict):
            price = float(first_level.get("price", first_level.get("p", 0)))
        else:
            price = 0.0

        tprint(f"[SLIP_CALC] _get_best_price: {side} best={price:.6f}")
        return price
    
    def validate_orderbook_freshness(
        self,
        order_book: Dict[str, Any]
    ) -> Tuple[bool, float]:
        """
        Validate order book timestamp freshness.

        Returns:
            Tuple of (is_fresh, age_seconds)
        """
        tprint(f"[SLIP_CALC] validate_orderbook_freshness: Checking order book freshness")
        timestamp = order_book.get("timestamp", order_book.get("ts", 0))
        if timestamp == 0:
            tprint(f"[SLIP_CALC] validate_orderbook_freshness -> no timestamp, assuming fresh")
            return True, 0.0

        from time import time
        age_seconds = time() - (timestamp / 1000 if timestamp > 1e12 else timestamp)
        is_fresh = age_seconds <= self.config.orderbook_staleness_threshold_sec

        if not is_fresh:
            tprint(f"[SLIP_CALC] validate_orderbook_freshness -> STALE: {age_seconds:.2f}s old (threshold={self.config.orderbook_staleness_threshold_sec}s)", color="red")
            self.logger.warning(
                f"Order book is stale: {age_seconds:.2f}s old "
                f"(threshold: {self.config.orderbook_staleness_threshold_sec}s)"
            )
        else:
            tprint(f"[SLIP_CALC] validate_orderbook_freshness -> FRESH: {age_seconds:.2f}s old", color="green")

        return is_fresh, age_seconds
    
    def generate_synthetic_orderbook(
        self,
        mid_price: float,
        exchange: str,
        depth_levels: int = 20
    ) -> Dict[str, Any]:
        """
        Generate a synthetic order book with realistic spread and depth.

        Args:
            mid_price: Mid market price
            exchange: Exchange name for spread calculation
            depth_levels: Number of price levels to generate

        Returns:
            Synthetic order book with bids and asks
        """
        tprint(f"[SLIP_CALC] generate_synthetic_orderbook: mid_price={mid_price:.6f}, exchange={exchange}, depth={depth_levels}")

        # Get exchange-specific spread
        spread_pct = self.config.get_spread_pct(exchange)
        
        # Calculate bid and ask prices at best level
        half_spread = mid_price * (spread_pct / 2)
        best_bid = mid_price - half_spread
        best_ask = mid_price + half_spread
        
        # Generate bids (decreasing price, increasing size)
        bids = []
        for i in range(depth_levels):
            # Price decreases with each level
            price_offset = (i * 0.0001 * mid_price)  # 0.01% per level
            price = best_bid - price_offset
            
            # Quantity increases slightly with worse prices (liquidity aggregation)
            base_qty = random.uniform(0.5, 2.0)
            qty = base_qty * (1 + i * 0.1)  # 10% more per level
            
            bids.append([price, qty])
        
        # Generate asks (increasing price, increasing size)
        asks = []
        for i in range(depth_levels):
            # Price increases with each level
            price_offset = (i * 0.0001 * mid_price)  # 0.01% per level
            price = best_ask + price_offset
            
            # Quantity increases slightly with worse prices
            base_qty = random.uniform(0.5, 2.0)
            qty = base_qty * (1 + i * 0.1)  # 10% more per level
            
            asks.append([price, qty])
        
        tprint(f"[SLIP_CALC] generate_synthetic_orderbook -> mid={mid_price:.2f}, spread={spread_pct:.4%}, best_bid={best_bid:.2f}, best_ask={best_ask:.2f}, levels={depth_levels}")

        self.logger.debug(
            f"Generated synthetic order book: mid={mid_price:.2f}, "
            f"spread={spread_pct:.4%}, best_bid={best_bid:.2f}, best_ask={best_ask:.2f}"
        )

        return {
            "bids": bids,
            "asks": asks,
            "timestamp": int(random.random() * 1000),  # Synthetic timestamp
            "synthetic": True
        }