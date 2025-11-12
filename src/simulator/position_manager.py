"""
Position Manager

Manages trading positions with support for multiple positions per symbol,
pyramiding, partial closes, and direction constraints.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import uuid
import logging

from src.utils.tprint import (
    tprint, tprint_logged, tprint_timer, tprint_performance,
    tprint_data_preview, tprint_data_format, tprint_feature_counts, LogLevel
)
from .config import SimulatorConfig


@dataclass
class Position:
    """Represents a trading position"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    symbol: str = ""
    direction: str = ""  # "long" or "short"
    quantity: float = 0.0
    avg_entry_price: float = 0.0
    entry_time: datetime = field(default_factory=datetime.now)
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def notional_value(self) -> float:
        """Calculate notional value of position"""
        return abs(self.quantity) * self.avg_entry_price


@tprint_logged(LogLevel.INFO, include_args=True)
class PositionManager:
    """
    Manage trading positions with flexible position sizing.
    
    Supports:
    - Multiple positions per symbol (if config allows)
    - Pyramiding (scaling into positions)
    - Partial position closes
    - Position averaging
    """
    
    def __init__(self, config: SimulatorConfig):
        """
        Initialize position manager.

        Args:
            config: Simulator configuration
        """
        tprint(f"[POS_MGR] __init__: Initializing position manager")
        self.config = config
        self.positions: Dict[str, List[Position]] = {}  # {symbol: [positions]}
        self.logger = logging.getLogger(__name__)
        tprint(f"[POS_MGR] __init__ -> initialized (allow_multiple={config.allow_multiple_positions}, allow_pyramiding={config.allow_pyramiding}, max_per_symbol={config.max_positions_per_symbol})")
    
    @tprint_logged(LogLevel.INFO, include_args=True)
    def add_position(
        self,
        symbol: str,
        direction: str,
        quantity: float,
        price: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Position:
        """
        Add or extend a position.

        Args:
            symbol: Trading symbol
            direction: Position direction ("long" or "short")
            quantity: Position quantity (positive)
            price: Entry price
            metadata: Additional position metadata

        Returns:
            Position object
        """
        tprint(f"[POS_MGR] add_position: symbol={symbol}, direction={direction}, qty={quantity}, price={price}")

        if quantity <= 0:
            tprint(f"[POS_MGR] add_position -> ERROR: quantity must be positive ({quantity})", color="red")
            raise ValueError("Quantity must be positive")

        direction_lower = direction.lower()
        if direction_lower not in ["long", "short"]:
            tprint(f"[POS_MGR] add_position -> ERROR: invalid direction {direction}", color="red")
            raise ValueError(f"Invalid direction: {direction}")
        
        # Check if we can add more positions for this symbol
        if symbol not in self.positions:
            self.positions[symbol] = []
        
        existing_positions = self.positions[symbol]
        
        # Check position limits
        if not self.config.allow_multiple_positions and existing_positions:
            # Replace existing position
            existing_positions.clear()
        elif len(existing_positions) >= self.config.max_positions_per_symbol:
            raise ValueError(
                f"Maximum {self.config.max_positions_per_symbol} positions "
                f"per symbol allowed for {symbol}"
            )
        
        # Check if there's a similar position we can add to (pyramiding)
        same_dir_pos = None
        if self.config.allow_pyramiding:
            for pos in existing_positions:
                if pos.direction == direction_lower:
                    same_dir_pos = pos
                    break
        
        if same_dir_pos:
            # Average into existing position
            total_qty = same_dir_pos.quantity + quantity
            total_cost = (same_dir_pos.quantity * same_dir_pos.avg_entry_price) + (quantity * price)
            same_dir_pos.quantity = total_qty
            same_dir_pos.avg_entry_price = total_cost / total_qty
            same_dir_pos.metadata.update(metadata or {})

            tprint(f"[POS_MGR] add_position -> PYRAMIDED: {symbol} {direction_lower}, qty={quantity:.6f} @ {price:.6f}, total_qty={total_qty:.6f}, avg_price={same_dir_pos.avg_entry_price:.6f}", color="green")

            self.logger.debug(
                f"Pyramided {symbol} {direction_lower}: "
                f"qty={quantity:.6f} price={price:.6f}, "
                f"total_qty={total_qty:.6f} avg_price={same_dir_pos.avg_entry_price:.6f}"
            )

            return same_dir_pos
        else:
            # Create new position
            position = Position(
                symbol=symbol,
                direction=direction_lower,
                quantity=quantity,
                avg_entry_price=price,
                metadata=metadata or {}
            )
            existing_positions.append(position)

            tprint(f"[POS_MGR] add_position -> OPENED: {symbol} {direction_lower}, qty={quantity:.6f} @ {price:.6f}, position_id={position.id}", color="green")

            self.logger.debug(
                f"Opened {symbol} {direction_lower}: "
                f"qty={quantity:.6f} price={price:.6f}"
            )

            return position
    
    @tprint_logged(LogLevel.INFO, include_args=True)
    def reduce_position(
        self,
        symbol: str,
        quantity: float,
        price: float
    ) -> List[Position]:
        """
        Partially close position(s).

        Args:
            symbol: Trading symbol
            quantity: Quantity to close (positive)
            price: Closing price

        Returns:
            List of updated Position objects
        """
        tprint(f"[POS_MGR] reduce_position: symbol={symbol}, qty={quantity}, price={price}")

        if symbol not in self.positions:
            tprint(f"[POS_MGR] reduce_position -> ERROR: No positions found for {symbol}", color="red")
            raise ValueError(f"No positions found for {symbol}")

        remaining_qty = quantity
        updated_positions = []
        
        for position in self.positions[symbol]:
            if remaining_qty <= 0:
                break
            
            # Can only close same direction positions
            if position.quantity == 0:
                continue
            
            # Determine closing quantity
            close_qty = min(abs(position.quantity), remaining_qty)
            position.quantity -= close_qty  # Reduce position size
            remaining_qty -= close_qty
            
            # Store partial close info in metadata
            if "partial_closes" not in position.metadata:
                position.metadata["partial_closes"] = []
            
            position.metadata["partial_closes"].append({
                "quantity": close_qty,
                "price": price,
                "timestamp": datetime.now().isoformat()
            })
            
            updated_positions.append(position)
        
        if remaining_qty > 0:
            tprint(f"[POS_MGR] reduce_position: Could not close {remaining_qty:.6f} of {symbol}, insufficient position size", color="yellow")
            self.logger.warning(
                f"Could not close {remaining_qty:.6f} of {symbol}, "
                f"insufficient position size"
            )

        tprint(f"[POS_MGR] reduce_position -> reduced {len(updated_positions)} positions for {symbol}")
        return updated_positions
    
    @tprint_logged(LogLevel.INFO, include_args=True)
    def close_position(
        self,
        position_id: str
    ) -> Optional[Position]:
        """
        Close a specific position by ID.

        Args:
            position_id: Position ID to close

        Returns:
            Closed Position object or None if not found
        """
        tprint(f"[POS_MGR] close_position: position_id={position_id}")

        for symbol, positions in self.positions.items():
            for pos in positions:
                if pos.id == position_id:
                    self.positions[symbol].remove(pos)
                    tprint(f"[POS_MGR] close_position -> CLOSED: position {position_id} for {symbol}, qty={pos.quantity}, price={pos.avg_entry_price:.6f}", color="green")
                    self.logger.debug(f"Closed position {position_id} for {symbol}")
                    return pos

        tprint(f"[POS_MGR] close_position -> NOT FOUND: position {position_id}", color="yellow")
        return None
    
    @tprint_logged(LogLevel.INFO, include_args=True)
    def close_all_positions(self, symbol: Optional[str] = None) -> List[Position]:
        """
        Close all positions, optionally filtered by symbol.

        Args:
            symbol: If provided, only close positions for this symbol

        Returns:
            List of closed Position objects
        """
        tprint(f"[POS_MGR] close_all_positions: symbol={symbol or 'ALL'}")

        closed = []

        symbols_to_close = [symbol] if symbol else list(self.positions.keys())

        for sym in symbols_to_close:
            if sym in self.positions:
                closed.extend(self.positions[sym])
                self.positions[sym].clear()

        tprint(f"[POS_MGR] close_all_positions -> CLOSED: {len(closed)} positions", color="green")
        return closed
    
    @tprint_logged(LogLevel.DEBUG, include_args=True)
    def get_positions(
        self,
        symbol: Optional[str] = None,
        direction: Optional[str] = None,
        status: str = "open"
    ) -> List[Position]:
        """
        Get positions matching criteria.

        Args:
            symbol: Filter by symbol (None = all)
            direction: Filter by direction ("long" or "short", None = all)
            status: Filter by status ("open", "closed", "all")

        Returns:
            List of Position objects
        """
        tprint(f"[POS_MGR] get_positions: symbol={symbol or 'ALL'}, direction={direction or 'ALL'}, status={status}")

        all_positions = []

        for sym, positions in self.positions.items():
            if symbol and sym != symbol:
                continue

            for pos in positions:
                if direction and pos.direction != direction.lower():
                    continue

                all_positions.append(pos)

        tprint(f"[POS_MGR] get_positions -> found {len(all_positions)} positions")
        return all_positions
    
    def get_position_by_id(self, position_id: str) -> Optional[Position]:
        """Get position by ID."""
        for positions in self.positions.values():
            for pos in positions:
                if pos.id == position_id:
                    return pos
        return None
    
    @tprint_logged(LogLevel.INFO, include_args=True)
    def calculate_unrealized_pnl(
        self,
        current_prices: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Calculate unrealized PnL for all positions.

        Args:
            current_prices: Current market prices {symbol: price}

        Returns:
            Dict with total unrealized PnL and per-symbol breakdown
        """
        tprint(f"[POS_MGR] calculate_unrealized_pnl: prices for {len(current_prices)} symbols")

        total_pnl = 0.0
        symbol_pnl = {}

        for symbol, positions in self.positions.items():
            current_price = current_prices.get(symbol, 0.0)
            if current_price == 0:
                tprint(f"[POS_MGR] calculate_unrealized_pnl: No price for {symbol}, skipping", color="yellow")
                continue

            symbol_total = 0.0
            for pos in positions:
                if pos.direction == "long":
                    pnl = (current_price - pos.avg_entry_price) * pos.quantity
                else:  # short
                    pnl = (pos.avg_entry_price - current_price) * pos.quantity

                symbol_total += pnl

            symbol_pnl[symbol] = symbol_total
            total_pnl += symbol_total

        tprint(f"[POS_MGR] calculate_unrealized_pnl -> total_pnl={total_pnl:.2f}, symbols={len(symbol_pnl)}")
        return {
            "total": total_pnl,
            "by_symbol": symbol_pnl
        }
    
    @tprint_logged(LogLevel.DEBUG, include_args=True)
    def validate_direction(
        self,
        symbol: str,
        direction: str
    ) -> bool:
        """
        Validate direction constraint.
        
        For "both" direction, allows any direction.
        For "long" or "short", checks if existing positions allow the new direction.
        """
        # If no positions, any direction is allowed
        if symbol not in self.positions or not self.positions[symbol]:
            return True
        
        # Get existing positions
        existing = self.positions[symbol]
        
        # Check if there are positions in opposite direction
        opposite_direction = "short" if direction.lower() == "long" else "long"
        
        for pos in existing:
            if pos.direction == opposite_direction:
                return False  # Can't have opposite positions
        
        return True
    
    @tprint_logged(LogLevel.INFO)
    def get_position_summary(self) -> Dict[str, Any]:
        """Get summary of all positions."""
        tprint(f"[POS_MGR] get_position_summary: Generating position summary")

        summary = {
            "total_positions": 0,
            "total_notional": 0.0,
            "by_symbol": {}
        }

        for symbol, positions in self.positions.items():
            symbol_qty = sum(abs(p.quantity) for p in positions)
            symbol_notional = sum(p.notional_value for p in positions)

            summary["by_symbol"][symbol] = {
                "count": len(positions),
                "total_quantity": symbol_qty,
                "total_notional": symbol_notional,
                "positions": [
                    {
                        "id": p.id,
                        "direction": p.direction,
                        "quantity": p.quantity,
                        "entry_price": p.avg_entry_price
                    }
                    for p in positions
                ]
            }

            summary["total_positions"] += len(positions)
            summary["total_notional"] += symbol_notional

        tprint(f"[POS_MGR] get_position_summary -> total_positions={summary['total_positions']}, total_notional={summary['total_notional']:.2f}, symbols={len(summary['by_symbol'])}")
        return summary
