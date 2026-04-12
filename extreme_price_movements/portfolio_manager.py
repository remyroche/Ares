"""Portfolio Manager for position-level risk and constraint enforcement.

Enforces portfolio-level constraints:
- Max 4 positions open simultaneously
- Max 30% of portfolio invested at any time
- Dynamic entry threshold based on current position count
- 24h cooldown after losing trades per asset
- Max 75% same-side concentration (3 long OR 3 short max)
- Max 50% same-strategy concentration (2 per strategy max, different assets)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
import numpy as np

from extreme_price_movements.utils import tprint


@dataclass
class Position:
    """Represents an open position."""
    symbol: str
    side: str  # "long" or "short"
    strategy_id: str
    entry_time: pd.Timestamp
    position_size: float  # As fraction of portfolio or absolute USDT
    entry_price: float
    is_open: bool = True


@dataclass
class CooldownRecord:
    """Tracks cooldown state for an asset."""
    cooldown_until: pd.Timestamp
    last_trade_was_loss: bool
    last_trade_time: pd.Timestamp


class PortfolioManager:
    """Manages portfolio state and enforces position constraints.
    
    Constraints:
    - MAX_POSITIONS: Maximum simultaneous open positions (default: 4)
    - MAX_PORTFOLIO_PCT: Maximum % of portfolio invested (default: 30%)
    - MAX_POSITION_USDT: Maximum absolute position size (default: 5000 USDT)
    - COOLDOWN_HOURS: Hours to wait after losing trade (default: 24)
    - MAX_SAME_SIDE_PCT: Max % of positions on same side (default: 75% -> 3 of 4)
    - MAX_SAME_STRATEGY_PCT: Max % from same strategy_id (default: 50% -> 2 of 4)
    """
    
    def __init__(
        self,
        max_positions: int = 4,
        max_portfolio_pct: float = 0.30,
        max_position_usdt: float = 5000.0,
        cooldown_hours: float = 24.0,
        max_same_side_pct: float = 0.75,
        max_same_strategy_pct: float = 0.50,
        portfolio_value: float = 10000.0,  # Default assumed portfolio value
    ):
        self.max_positions = max_positions
        self.max_portfolio_pct = max_portfolio_pct
        self.max_position_usdt = max_position_usdt
        self.cooldown_hours = cooldown_hours
        self.max_same_side = int(max_positions * max_same_side_pct)  # 3 for defaults
        self.max_same_strategy = int(max_positions * max_same_strategy_pct)  # 2 for defaults
        self.portfolio_value = portfolio_value
        
        # State tracking
        self.positions: Dict[str, Position] = {}  # symbol -> Position
        self.cooldowns: Dict[str, CooldownRecord] = {}  # symbol -> CooldownRecord
        self.closed_positions: List[Position] = []
        
    def get_portfolio_state(self) -> Dict[str, Any]:
        """Return current portfolio state summary."""
        open_positions = [p for p in self.positions.values() if p.is_open]
        n_open = len(open_positions)
        
        # Calculate invested percentage
        total_invested = sum(p.position_size for p in open_positions)
        invested_pct = total_invested / self.portfolio_value if self.portfolio_value > 0 else 0.0
        
        # Count by side
        long_count = sum(1 for p in open_positions if p.side == "long")
        short_count = sum(1 for p in open_positions if p.side == "short")
        
        # Count by strategy
        strategy_counts: Dict[str, int] = {}
        for p in open_positions:
            strategy_counts[p.strategy_id] = strategy_counts.get(p.strategy_id, 0) + 1
        
        # Active cooldowns
        now = pd.Timestamp.now(tz="UTC")
        active_cooldowns = {
            sym: {
                "cooldown_until": rec.cooldown_until.isoformat(),
                "expires_in_hours": max(0, (rec.cooldown_until - now).total_seconds() / 3600),
            }
            for sym, rec in self.cooldowns.items()
            if rec.cooldown_until > now
        }
        
        return {
            "n_positions": n_open,
            "max_positions": self.max_positions,
            "invested_usdt": total_invested,
            "invested_pct": invested_pct,
            "max_invested_pct": self.max_portfolio_pct,
            "remaining_pct": self.max_portfolio_pct - invested_pct,
            "long_count": long_count,
            "short_count": short_count,
            "max_same_side": self.max_same_side,
            "strategy_counts": strategy_counts,
            "max_same_strategy": self.max_same_strategy,
            "active_cooldowns": active_cooldowns,
            "open_symbols": list(self.positions.keys()),
        }
    
    def calculate_dynamic_threshold(self, initial_threshold: float) -> float:
        """Calculate adjusted entry threshold based on current position count.
        
        Formula: final_threshold = initial_threshold + (n_positions * (1 - initial_threshold)) / max_positions
        
        As more positions are open, threshold increases making entry harder.
        """
        n_positions = len([p for p in self.positions.values() if p.is_open])
        if n_positions == 0:
            return initial_threshold
        
        # Formula: initial + (n_positions * (1 - initial)) / max_positions
        adjustment = (n_positions * (1.0 - initial_threshold)) / self.max_positions
        final_threshold = initial_threshold + adjustment
        
        return min(final_threshold, 1.0)  # Cap at 1.0
    
    def calculate_position_size_cap(self, requested_size: float) -> float:
        """Calculate allowed position size considering portfolio constraints.
        
        Returns: min(requested_size, remaining_portfolio_capacity, max_position_usdt)
        """
        open_positions = [p for p in self.positions.values() if p.is_open]
        total_invested = sum(p.position_size for p in open_positions)
        
        # Remaining capacity
        max_total_invested = self.portfolio_value * self.max_portfolio_pct
        remaining_capacity = max_total_invested - total_invested
        
        # Cap at multiple constraints
        allowed_size = min(
            requested_size,
            remaining_capacity,
            self.max_position_usdt
        )
        
        return max(0.0, allowed_size)
    
    def can_enter_position(
        self,
        symbol: str,
        side: str,
        strategy_id: str,
        confidence_score: float,
        initial_threshold: float,
        current_time: Optional[pd.Timestamp] = None,
    ) -> Tuple[bool, Dict[str, Any]]:
        """Check if a new position can be entered.
        
        Args:
            symbol: Asset symbol
            side: "long" or "short"
            strategy_id: Strategy identifier
            confidence_score: Current confidence score (0-1)
            initial_threshold: Base threshold for entry
            current_time: Timestamp for cooldown checks (default: now)
            
        Returns:
            (allowed: bool, info: dict with details)
        """
        if current_time is None:
            current_time = pd.Timestamp.now(tz="UTC")
        
        info = {
            "reason": "",
            "final_threshold": 0.0,
            "position_size_cap": 0.0,
            "n_positions_before": 0,
            "constraints_checked": [],
        }
        
        # Get current state
        open_positions = [p for p in self.positions.values() if p.is_open]
        n_positions = len(open_positions)
        info["n_positions_before"] = n_positions
        
        # 1. Check max positions
        if n_positions >= self.max_positions:
            info["reason"] = f"max_positions_reached ({n_positions}/{self.max_positions})"
            info["constraints_checked"].append("max_positions")
            return False, info
        
        # 2. Check if symbol already has position (max 1 per asset)
        if symbol in self.positions and self.positions[symbol].is_open:
            info["reason"] = f"symbol_already_has_position ({symbol})"
            info["constraints_checked"].append("max_one_per_symbol")
            return False, info
        
        # 3. Check cooldown
        if symbol in self.cooldowns:
            cd = self.cooldowns[symbol]
            if cd.cooldown_until > current_time:
                hours_remaining = (cd.cooldown_until - current_time).total_seconds() / 3600
                info["reason"] = f"cooldown_active ({hours_remaining:.1f}h remaining)"
                info["constraints_checked"].append("cooldown")
                info["cooldown_until"] = cd.cooldown_until.isoformat()
                return False, info
        
        # 4. Check same-side concentration (max 75% -> 3 of 4)
        side_count = sum(1 for p in open_positions if p.side == side)
        if side_count >= self.max_same_side:
            info["reason"] = f"max_same_side_reached ({side_count} {side}, max {self.max_same_side})"
            info["constraints_checked"].append("max_same_side")
            return False, info
        
        # 5. Check same-strategy concentration (max 50% -> 2 of 4)
        strategy_count = sum(1 for p in open_positions if p.strategy_id == strategy_id)
        if strategy_count >= self.max_same_strategy:
            info["reason"] = f"max_same_strategy_reached ({strategy_count} {strategy_id}, max {self.max_same_strategy})"
            info["constraints_checked"].append("max_same_strategy")
            return False, info
        
        # 6. Calculate dynamic threshold
        final_threshold = self.calculate_dynamic_threshold(initial_threshold)
        info["final_threshold"] = final_threshold
        info["initial_threshold"] = initial_threshold
        info["constraints_checked"].append("dynamic_threshold")
        
        # 7. Check confidence against dynamic threshold
        if confidence_score < final_threshold:
            info["reason"] = f"confidence_below_threshold ({confidence_score:.4f} < {final_threshold:.4f})"
            info["constraints_checked"].append("confidence_threshold")
            return False, info
        
        # 8. Calculate position size cap
        position_size_cap = self.calculate_position_size_cap(self.max_position_usdt)
        info["position_size_cap"] = position_size_cap
        info["constraints_checked"].append("position_size_cap")
        
        if position_size_cap <= 0:
            info["reason"] = "no_remaining_portfolio_capacity"
            return False, info
        
        # All checks passed
        info["reason"] = "allowed"
        return True, info
    
    def record_position_open(
        self,
        symbol: str,
        side: str,
        strategy_id: str,
        position_size: float,
        entry_price: float,
        entry_time: Optional[pd.Timestamp] = None,
    ) -> Position:
        """Record a new position opening."""
        if entry_time is None:
            entry_time = pd.Timestamp.now(tz="UTC")
        
        position = Position(
            symbol=symbol,
            side=side,
            strategy_id=strategy_id,
            entry_time=entry_time,
            position_size=position_size,
            entry_price=entry_price,
            is_open=True,
        )
        
        self.positions[symbol] = position
        tprint(f"[PortfolioManager] Opened {side} position on {symbol} via {strategy_id} "
               f"(size: {position_size:.2f} USDT, price: {entry_price:.4f})")
        
        return position
    
    def record_position_close(
        self,
        symbol: str,
        exit_price: float,
        exit_time: Optional[pd.Timestamp] = None,
        exit_reason: str = "",
    ) -> Optional[Dict[str, Any]]:
        """Record position closure and update cooldowns if needed.
        
        Returns dict with position info or None if position not found.
        """
        if exit_time is None:
            exit_time = pd.Timestamp.now(tz="UTC")
        
        if symbol not in self.positions:
            return None
        
        position = self.positions[symbol]
        if not position.is_open:
            return None
        
        # Mark as closed
        position.is_open = False
        
        # Calculate PnL
        if position.side == "long":
            pnl_pct = (exit_price - position.entry_price) / position.entry_price
        else:
            pnl_pct = (position.entry_price - exit_price) / position.entry_price
        
        pnl_usdt = pnl_pct * position.position_size
        was_win = pnl_usdt > 0
        
        result = {
            "symbol": symbol,
            "side": position.side,
            "strategy_id": position.strategy_id,
            "entry_price": position.entry_price,
            "exit_price": exit_price,
            "pnl_pct": pnl_pct,
            "pnl_usdt": pnl_usdt,
            "was_win": was_win,
            "exit_reason": exit_reason,
            "holding_time": (exit_time - position.entry_time).total_seconds() / 3600,
        }
        
        # Update cooldown if it was a loss
        if not was_win:
            cooldown_until = exit_time + timedelta(hours=self.cooldown_hours)
            self.cooldowns[symbol] = CooldownRecord(
                cooldown_until=cooldown_until,
                last_trade_was_loss=True,
                last_trade_time=exit_time,
            )
            tprint(f"[PortfolioManager] Closed {symbol} with loss ({pnl_usdt:.2f} USDT), "
                   f"cooldown until {cooldown_until}")
        else:
            tprint(f"[PortfolioManager] Closed {symbol} with win (+{pnl_usdt:.2f} USDT)")
        
        # Move to closed list
        self.closed_positions.append(position)
        del self.positions[symbol]
        
        return result
    
    def get_open_positions_summary(self) -> pd.DataFrame:
        """Return DataFrame summary of open positions."""
        open_positions = [p for p in self.positions.values() if p.is_open]
        if not open_positions:
            return pd.DataFrame()
        
        return pd.DataFrame([
            {
                "symbol": p.symbol,
                "side": p.side,
                "strategy_id": p.strategy_id,
                "entry_time": p.entry_time,
                "position_size": p.position_size,
                "entry_price": p.entry_price,
            }
            for p in open_positions
        ])
    
    def save_state(self, filepath: str) -> None:
        """Save portfolio state to JSON file."""
        state = {
            "positions": [
                {
                    "symbol": p.symbol,
                    "side": p.side,
                    "strategy_id": p.strategy_id,
                    "entry_time": p.entry_time.isoformat(),
                    "position_size": p.position_size,
                    "entry_price": p.entry_price,
                    "is_open": p.is_open,
                }
                for p in self.positions.values()
            ],
            "cooldowns": {
                sym: {
                    "cooldown_until": rec.cooldown_until.isoformat(),
                    "last_trade_was_loss": rec.last_trade_was_loss,
                    "last_trade_time": rec.last_trade_time.isoformat(),
                }
                for sym, rec in self.cooldowns.items()
            },
            "portfolio_value": self.portfolio_value,
        }
        
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(state, indent=2))
        tprint(f"[PortfolioManager] State saved to {filepath}")
    
    def load_state(self, filepath: str) -> None:
        """Load portfolio state from JSON file."""
        path = Path(filepath)
        if not path.exists():
            tprint(f"[PortfolioManager] No state file found at {filepath}")
            return
        
        data = json.loads(path.read_text())
        
        # Restore positions
        self.positions = {}
        for p_data in data.get("positions", []):
            pos = Position(
                symbol=p_data["symbol"],
                side=p_data["side"],
                strategy_id=p_data["strategy_id"],
                entry_time=pd.Timestamp(p_data["entry_time"]),
                position_size=p_data["position_size"],
                entry_price=p_data["entry_price"],
                is_open=p_data["is_open"],
            )
            self.positions[pos.symbol] = pos
        
        # Restore cooldowns
        self.cooldowns = {}
        for sym, cd_data in data.get("cooldowns", {}).items():
            self.cooldowns[sym] = CooldownRecord(
                cooldown_until=pd.Timestamp(cd_data["cooldown_until"]),
                last_trade_was_loss=cd_data["last_trade_was_loss"],
                last_trade_time=pd.Timestamp(cd_data["last_trade_time"]),
            )
        
        self.portfolio_value = data.get("portfolio_value", self.portfolio_value)
        tprint(f"[PortfolioManager] State loaded from {filepath} "
               f"({len(self.positions)} positions, {len(self.cooldowns)} cooldowns)")


__all__ = ["PortfolioManager", "Position", "CooldownRecord"]
