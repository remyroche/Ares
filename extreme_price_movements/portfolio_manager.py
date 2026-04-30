"""Portfolio Manager for position-level risk and constraint enforcement.

Enforces portfolio-level constraints:
- Max 4 positions open simultaneously
- No max total capital allocation by default
- Max 15% of portfolio per position
- Dynamic entry threshold based on current position count
- 24h cooldown after losing trades per asset
- Max 50% same-strategy concentration (2 per strategy max, different assets)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from extreme_price_movements.utils import tprint


def _classify_api_error(exc: Exception) -> str:
    """Classify private exchange API errors for risk-gate logging."""
    text = f"{exc.__class__.__name__} {exc}".lower()
    if "429" in text or "rate limit" in text or "too many requests" in text:
        return "rate_limited"
    if "timeout" in text or "timed out" in text:
        return "timeout"
    if "network" in text or "connection" in text:
        return "network"
    if "auth" in text or "permission" in text or "forbidden" in text or "401" in text:
        return "auth_or_permission"
    return "api_error"


@dataclass
class Position:
    """Represents an open position."""

    symbol: str
    side: str  # "long" or "short"
    strategy_id: str
    entry_time: pd.Timestamp
    position_size: float  # As fraction of portfolio or absolute quote notional
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
    - MAX_PORTFOLIO_PCT: Optional maximum % of portfolio invested (default: disabled)
    - MAX_POSITION_PCT: Maximum % of portfolio per position (default: 15%)
    - MAX_POSITION_USDT: Optional absolute quote-notional cap (default: disabled)
    - COOLDOWN_HOURS: Hours to wait after losing trade (default: 24)
    - MAX_SAME_SIDE_PCT: Max % of positions on same side (default: 100% -> disabled)
    - MAX_SAME_STRATEGY_PCT: Max % from same strategy_id (default: 50% -> 2 of 4)
    """

    def __init__(
        self,
        max_positions: int = 4,
        max_portfolio_pct: Optional[float] = None,
        max_position_usdt: Optional[float] = None,
        max_position_pct: float = 0.15,
        cooldown_hours: float = 24.0,
        max_same_side_pct: float = 1.0,
        max_same_strategy_pct: float = 0.50,
        portfolio_value: float = 10000.0,  # Default assumed portfolio value
        max_daily_loss_pct: float = 0.10,
        max_weekly_loss_pct: float = 0.20,
        max_consecutive_losing_trades: int = 5,
        max_failed_api_calls_5m: int = 10,
        max_consecutive_order_rejections: int = 5,
    ):
        self.max_positions = max_positions
        self.max_portfolio_pct = max_portfolio_pct
        self.max_position_usdt = max_position_usdt
        self.max_position_pct = max_position_pct
        self.cooldown_hours = cooldown_hours
        self.max_same_side = max(1, int(max_positions * max_same_side_pct))
        self.max_same_strategy = max(1, int(max_positions * max_same_strategy_pct))
        self.portfolio_value = portfolio_value
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_weekly_loss_pct = max_weekly_loss_pct
        self.max_consecutive_losing_trades = max_consecutive_losing_trades
        self.max_failed_api_calls_5m = max_failed_api_calls_5m
        self.max_consecutive_order_rejections = max_consecutive_order_rejections

        # State tracking
        self.positions: Dict[str, Position] = {}  # symbol -> Position
        self.cooldowns: Dict[str, CooldownRecord] = {}  # symbol -> CooldownRecord
        self.closed_positions: List[Position] = []
        self.pnl_events: List[Dict[str, Any]] = []
        self.failed_api_events: List[pd.Timestamp] = []
        self.consecutive_losing_trades = 0
        self.consecutive_order_rejections = 0
        self.order_rejection_backoff_until: Optional[pd.Timestamp] = None
        self.manual_reset_required = False
        self.hard_limit_reason = ""

    def get_portfolio_state(self) -> Dict[str, Any]:
        """Return current portfolio state summary."""
        open_positions = [p for p in self.positions.values() if p.is_open]
        n_open = len(open_positions)

        # Calculate invested percentage
        total_invested = sum(p.position_size for p in open_positions)
        invested_pct = (
            total_invested / self.portfolio_value if self.portfolio_value > 0 else 0.0
        )

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
                "expires_in_hours": max(
                    0, (rec.cooldown_until - now).total_seconds() / 3600
                ),
            }
            for sym, rec in self.cooldowns.items()
            if rec.cooldown_until > now
        }

        if self.max_portfolio_pct is None or not np.isfinite(self.max_portfolio_pct):
            max_invested_pct = None
            remaining_pct = None
        else:
            max_invested_pct = self.max_portfolio_pct
            remaining_pct = self.max_portfolio_pct - invested_pct

        return {
            "n_positions": n_open,
            "max_positions": self.max_positions,
            "invested_usdt": total_invested,
            "invested_pct": invested_pct,
            "max_invested_pct": max_invested_pct,
            "remaining_pct": remaining_pct,
            "max_position_pct": self.max_position_pct,
            "max_position_usdt": self.max_position_usdt,
            "long_count": long_count,
            "short_count": short_count,
            "max_same_side": self.max_same_side,
            "strategy_counts": strategy_counts,
            "max_same_strategy": self.max_same_strategy,
            "active_cooldowns": active_cooldowns,
            "open_symbols": list(self.positions.keys()),
            "hard_limits": self.get_hard_limit_status(),
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

        Returns the minimum of the requested size, the per-position equity cap,
        and any optional absolute or total portfolio allocation caps.
        """
        open_positions = [p for p in self.positions.values() if p.is_open]
        total_invested = sum(p.position_size for p in open_positions)

        caps = [
            float(requested_size),
            float(self.portfolio_value) * float(self.max_position_pct),
        ]
        if self.max_position_usdt is not None and np.isfinite(self.max_position_usdt):
            caps.append(float(self.max_position_usdt))
        if self.max_portfolio_pct is not None and np.isfinite(self.max_portfolio_pct):
            max_total_invested = self.portfolio_value * float(self.max_portfolio_pct)
            caps.append(max_total_invested - total_invested)

        return max(0.0, min(caps))

    def _rolling_loss_pct(
        self, current_time: pd.Timestamp, window: pd.Timedelta
    ) -> float:
        cutoff = current_time - window
        loss_usdt = 0.0
        for event in self.pnl_events:
            ts = pd.Timestamp(event.get("timestamp"))
            pnl = float(event.get("pnl_usdt", 0.0) or 0.0)
            if ts >= cutoff and pnl < 0.0:
                loss_usdt += -pnl
        return loss_usdt / max(float(self.portfolio_value), 1e-12)

    def _trip_hard_limit(self, reason: str) -> None:
        if not self.manual_reset_required:
            tprint(f"[PortfolioManager] Hard risk limit tripped: {reason}")
        self.manual_reset_required = True
        self.hard_limit_reason = reason

    def evaluate_hard_limits(
        self, current_time: Optional[pd.Timestamp] = None
    ) -> Tuple[bool, Dict[str, Any]]:
        """Return whether opening new positions is allowed by hard risk gates."""
        if current_time is None:
            current_time = pd.Timestamp.now(tz="UTC")
        current_time = pd.Timestamp(current_time)

        cutoff_5m = current_time - pd.Timedelta(minutes=5)
        self.failed_api_events = [
            pd.Timestamp(ts)
            for ts in self.failed_api_events
            if pd.Timestamp(ts) >= cutoff_5m
        ]
        daily_loss_pct = self._rolling_loss_pct(current_time, pd.Timedelta(days=1))
        weekly_loss_pct = self._rolling_loss_pct(current_time, pd.Timedelta(days=7))

        if daily_loss_pct >= self.max_daily_loss_pct:
            self._trip_hard_limit(
                f"rolling_daily_loss_pct {daily_loss_pct:.4f} >= {self.max_daily_loss_pct:.4f}"
            )
        elif weekly_loss_pct >= self.max_weekly_loss_pct:
            self._trip_hard_limit(
                f"rolling_weekly_loss_pct {weekly_loss_pct:.4f} >= {self.max_weekly_loss_pct:.4f}"
            )
        elif self.consecutive_losing_trades >= self.max_consecutive_losing_trades:
            self._trip_hard_limit(
                f"consecutive_losing_trades {self.consecutive_losing_trades} >= {self.max_consecutive_losing_trades}"
            )
        elif len(self.failed_api_events) >= self.max_failed_api_calls_5m:
            self._trip_hard_limit(
                f"failed_api_calls_5m {len(self.failed_api_events)} >= {self.max_failed_api_calls_5m}"
            )
        elif self.consecutive_order_rejections >= self.max_consecutive_order_rejections:
            self._trip_hard_limit(
                f"consecutive_order_rejections {self.consecutive_order_rejections} >= {self.max_consecutive_order_rejections}"
            )

        if (
            self.order_rejection_backoff_until is not None
            and self.order_rejection_backoff_until > current_time
        ):
            return False, {
                "reason": "order_rejection_backoff",
                "backoff_until": self.order_rejection_backoff_until.isoformat(),
                "manual_reset_required": self.manual_reset_required,
                "hard_limit_reason": self.hard_limit_reason,
                "daily_loss_pct": daily_loss_pct,
                "weekly_loss_pct": weekly_loss_pct,
                "failed_api_calls_5m": len(self.failed_api_events),
                "consecutive_order_rejections": self.consecutive_order_rejections,
                "consecutive_losing_trades": self.consecutive_losing_trades,
            }

        allowed = not self.manual_reset_required
        return allowed, {
            "reason": "allowed" if allowed else "manual_reset_required",
            "manual_reset_required": self.manual_reset_required,
            "hard_limit_reason": self.hard_limit_reason,
            "daily_loss_pct": daily_loss_pct,
            "weekly_loss_pct": weekly_loss_pct,
            "failed_api_calls_5m": len(self.failed_api_events),
            "consecutive_order_rejections": self.consecutive_order_rejections,
            "consecutive_losing_trades": self.consecutive_losing_trades,
        }

    def get_hard_limit_status(self) -> Dict[str, Any]:
        """Return current hard-risk gate status without mutating counters."""
        allowed, status = self.evaluate_hard_limits()
        status["allowed_to_open"] = allowed
        return status

    def record_api_call(
        self,
        success: bool,
        *,
        timestamp: Optional[pd.Timestamp] = None,
        error: str = "",
    ) -> None:
        """Record API-call health for hard failed-call gates."""
        if success:
            return
        ts = (
            pd.Timestamp(timestamp)
            if timestamp is not None
            else pd.Timestamp.now(tz="UTC")
        )
        self.failed_api_events.append(ts)
        self.evaluate_hard_limits(ts)
        if error:
            tprint(f"[PortfolioManager] API failure recorded: {error}")

    def record_order_result(
        self,
        success: bool,
        *,
        rejected: bool = False,
        timestamp: Optional[pd.Timestamp] = None,
        error: str = "",
    ) -> None:
        """Record order acceptance/rejection state with exponential backoff."""
        ts = (
            pd.Timestamp(timestamp)
            if timestamp is not None
            else pd.Timestamp.now(tz="UTC")
        )
        if success:
            self.consecutive_order_rejections = 0
            self.order_rejection_backoff_until = None
            return
        if rejected:
            self.consecutive_order_rejections += 1
            backoff_seconds = min(
                60.0 * (2.0 ** max(self.consecutive_order_rejections - 1, 0)),
                900.0,
            )
            self.order_rejection_backoff_until = ts + pd.Timedelta(
                seconds=backoff_seconds
            )
        self.evaluate_hard_limits(ts)
        if error:
            tprint(f"[PortfolioManager] Order failure recorded: {error}")

    def manual_reset_hard_limits(self) -> None:
        """Manual reset hook after a hard gate has stopped new entries."""
        self.manual_reset_required = False
        self.hard_limit_reason = ""
        self.failed_api_events = []
        self.consecutive_order_rejections = 0
        self.order_rejection_backoff_until = None
        self.consecutive_losing_trades = 0

    def fetch_exchange_snapshot(
        self,
        exchange: Any,
        *,
        quote_currency: str = "USDC",
        execution_account: str = "margin",
        margin_mode: str = "cross",
    ) -> Dict[str, Any]:
        """Fetch wallet and exchange-side open-position state.

        The method intentionally accepts any ccxt-like exchange so tests can
        cover the private API path without live credentials.
        """
        quote = str(quote_currency).upper()
        account = str(execution_account or "margin").lower()
        mode = str(margin_mode or "cross").lower()
        balance_params: Dict[str, Any] = {}
        position_params: Dict[str, Any] = {}
        if account == "margin":
            balance_params = {"type": "margin", "marginMode": mode}
            position_params = {"type": "margin", "marginMode": mode}
        snapshot: Dict[str, Any] = {
            "quote_currency": quote,
            "execution_account": account,
            "margin_mode": mode if account == "margin" else None,
            "total_balance": np.nan,
            "free_balance": np.nan,
            "used_balance": np.nan,
            "exchange_positions": [],
            "exchange_open_positions": 0,
            "local_open_positions": len(
                [p for p in self.positions.values() if p.is_open]
            ),
            "errors": [],
            "error_categories": [],
        }

        tprint("[PortfolioManager] Fetching exchange wallet/position snapshot")
        try:
            balance = exchange.fetch_balance(balance_params)
            self.record_api_call(True)
            total = balance.get("total", {}) if isinstance(balance, dict) else {}
            free = balance.get("free", {}) if isinstance(balance, dict) else {}
            used = balance.get("used", {}) if isinstance(balance, dict) else {}
            snapshot["total_balance"] = float(total.get(quote, np.nan))
            snapshot["free_balance"] = float(free.get(quote, np.nan))
            snapshot["used_balance"] = float(used.get(quote, np.nan))
            if np.isfinite(float(snapshot["total_balance"])):
                self.portfolio_value = float(snapshot["total_balance"])
        except Exception as exc:
            category = _classify_api_error(exc)
            self.record_api_call(
                False, error=f"fetch_balance failed: {category}: {exc}"
            )
            snapshot["errors"].append(f"fetch_balance: {exc}")
            snapshot["error_categories"].append(category)

        try:
            positions = []
            fetch_positions = getattr(exchange, "fetch_positions", None)
            if callable(fetch_positions):
                raw_positions = fetch_positions(position_params)
                for pos in raw_positions or []:
                    if not isinstance(pos, dict):
                        continue
                    contracts = float(
                        pos.get(
                            "contracts",
                            pos.get("contractSize", pos.get("positionAmt", 0.0)),
                        )
                        or 0.0
                    )
                    info = pos.get("info", {})
                    if isinstance(info, dict) and not contracts:
                        contracts = float(info.get("positionAmt", 0.0) or 0.0)
                    if abs(contracts) > 0.0:
                        positions.append(pos)
            snapshot["exchange_positions"] = positions
            snapshot["exchange_open_positions"] = len(positions)
            self.record_api_call(True)
        except Exception as exc:
            category = _classify_api_error(exc)
            self.record_api_call(
                False, error=f"fetch_positions failed: {category}: {exc}"
            )
            snapshot["errors"].append(f"fetch_positions: {exc}")
            snapshot["error_categories"].append(category)

        tprint(
            "[PortfolioManager] Exchange snapshot complete: "
            f"balance_ok={np.isfinite(float(snapshot['total_balance']))} "
            f"exchange_open_positions={snapshot['exchange_open_positions']} "
            f"errors={len(snapshot['errors'])} "
            f"error_categories={snapshot['error_categories']}"
        )
        return snapshot

    def can_enter_position(
        self,
        symbol: str,
        side: str,
        strategy_id: str,
        confidence_score: float,
        initial_threshold: float,
        current_time: Optional[pd.Timestamp] = None,
        requested_position_size: Optional[float] = None,
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

        hard_allowed, hard_status = self.evaluate_hard_limits(current_time)
        info["hard_limits"] = hard_status
        if not hard_allowed:
            info["reason"] = hard_status.get("reason", "hard_limit_block")
            info["constraints_checked"].append("hard_limits")
            return False, info

        # 1. Check max positions
        if n_positions >= self.max_positions:
            info[
                "reason"
            ] = f"max_positions_reached ({n_positions}/{self.max_positions})"
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
                hours_remaining = (
                    cd.cooldown_until - current_time
                ).total_seconds() / 3600
                info["reason"] = f"cooldown_active ({hours_remaining:.1f}h remaining)"
                info["constraints_checked"].append("cooldown")
                info["cooldown_until"] = cd.cooldown_until.isoformat()
                return False, info

        # 4. Check same-side concentration (max 75% -> 3 of 4)
        side_count = sum(1 for p in open_positions if p.side == side)
        if side_count >= self.max_same_side:
            info[
                "reason"
            ] = f"max_same_side_reached ({side_count} {side}, max {self.max_same_side})"
            info["constraints_checked"].append("max_same_side")
            return False, info

        # 5. Check same-strategy concentration (max 50% -> 2 of 4)
        strategy_count = sum(1 for p in open_positions if p.strategy_id == strategy_id)
        if strategy_count >= self.max_same_strategy:
            info[
                "reason"
            ] = f"max_same_strategy_reached ({strategy_count} {strategy_id}, max {self.max_same_strategy})"
            info["constraints_checked"].append("max_same_strategy")
            return False, info

        # 6. Calculate dynamic threshold
        final_threshold = self.calculate_dynamic_threshold(initial_threshold)
        info["final_threshold"] = final_threshold
        info["initial_threshold"] = initial_threshold
        info["constraints_checked"].append("dynamic_threshold")

        # 7. Check confidence against dynamic threshold
        if confidence_score < final_threshold:
            info[
                "reason"
            ] = f"confidence_below_threshold ({confidence_score:.4f} < {final_threshold:.4f})"
            info["constraints_checked"].append("confidence_threshold")
            return False, info

        # 8. Calculate position size cap
        requested_size = (
            float(requested_position_size)
            if requested_position_size is not None
            else self.portfolio_value * self.max_position_pct
        )
        position_size_cap = self.calculate_position_size_cap(requested_size)
        info["position_size_cap"] = position_size_cap
        info["requested_position_size"] = requested_size
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
        tprint(
            f"[PortfolioManager] Opened {side} position on {symbol} via {strategy_id} "
            f"(size: {position_size:.2f} quote, price: {entry_price:.4f})"
        )

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
        self.pnl_events.append(
            {
                "timestamp": exit_time,
                "symbol": symbol,
                "strategy_id": position.strategy_id,
                "pnl_usdt": float(pnl_usdt),
            }
        )
        if was_win:
            self.consecutive_losing_trades = 0
        else:
            self.consecutive_losing_trades += 1
        self.evaluate_hard_limits(exit_time)

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
            tprint(
                f"[PortfolioManager] Closed {symbol} with loss ({pnl_usdt:.2f} USDT), "
                f"cooldown until {cooldown_until}"
            )
        else:
            tprint(
                f"[PortfolioManager] Closed {symbol} with win (+{pnl_usdt:.2f} USDT)"
            )

        # Move to closed list
        self.closed_positions.append(position)
        del self.positions[symbol]

        return result

    def get_open_positions_summary(self) -> pd.DataFrame:
        """Return DataFrame summary of open positions."""
        open_positions = [p for p in self.positions.values() if p.is_open]
        if not open_positions:
            return pd.DataFrame()

        return pd.DataFrame(
            [
                {
                    "symbol": p.symbol,
                    "side": p.side,
                    "strategy_id": p.strategy_id,
                    "entry_time": p.entry_time,
                    "position_size": p.position_size,
                    "entry_price": p.entry_price,
                }
                for p in open_positions
            ]
        )

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
        tprint(
            f"[PortfolioManager] State loaded from {filepath} "
            f"({len(self.positions)} positions, {len(self.cooldowns)} cooldowns)"
        )


__all__ = ["PortfolioManager", "Position", "CooldownRecord"]
