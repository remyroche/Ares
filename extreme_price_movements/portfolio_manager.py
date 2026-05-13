"""Portfolio Manager for position-level risk and constraint enforcement.

Enforces portfolio-level constraints:
- Max 10 positions open simultaneously
- Max 75% total wallet allocation by default
- Max 15% of portfolio per position
- Dynamic entry threshold based on current position count
- 24h cooldown after losing trades per asset
- Max 6 positions per side and per strategy by default
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

try:
    from extreme_price_movements.inference.portfolio_policy import PortfolioPolicyConfig
except Exception:  # pragma: no cover - keeps legacy imports resilient.
    PortfolioPolicyConfig = Any  # type: ignore


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
    - MAX_POSITIONS: Maximum simultaneous open positions (default: 10)
    - MAX_PORTFOLIO_PCT: Maximum % of portfolio invested (default: 75%)
    - MAX_POSITION_PCT: Maximum % of portfolio per position (default: 15%)
    - MAX_POSITION_USDT: Optional absolute quote-notional cap (default: 5000)
    - COOLDOWN_HOURS: Hours to wait after losing trade (default: 24)
    - MAX_SAME_SIDE: Max positions on same side (default: 6)
    - MAX_SAME_STRATEGY: Max positions from same strategy_id (default: 6)
    """

    def __init__(
        self,
        max_positions: int = 10,
        max_portfolio_pct: Optional[float] = 0.75,
        max_position_usdt: Optional[float] = 5000.0,
        max_position_pct: float = 0.15,
        cooldown_hours: float = 24.0,
        max_same_side: Optional[int] = 6,
        max_same_strategy: Optional[int] = 6,
        max_same_side_pct: Optional[float] = None,
        max_same_strategy_pct: Optional[float] = None,
        portfolio_value: float = 10000.0,  # Default assumed portfolio value
        book_notional_multiplier: float = 1.0,
        leverage_wallet_multiplier: float = 1.0,
        min_margin_level_after_entry: float = 2.5,
        max_daily_loss_pct: float = 0.10,
        max_weekly_loss_pct: float = 0.20,
        max_consecutive_losing_trades: int = 5,
        max_failed_api_calls_5m: int = 10,
        max_consecutive_order_rejections: int = 5,
    ):
        self.max_positions = max(1, int(max_positions))
        self.max_portfolio_pct = max_portfolio_pct
        self.max_position_usdt = max_position_usdt
        self.max_position_pct = max_position_pct
        self.book_notional_multiplier = max(float(book_notional_multiplier), 0.0)
        self.leverage_wallet_multiplier = max(float(leverage_wallet_multiplier), 1.0)
        self.min_margin_level_after_entry = max(
            float(min_margin_level_after_entry), 1.0
        )
        self.margin_total_assets_quote: Optional[float] = None
        self.margin_total_liabilities_quote: Optional[float] = None
        self.margin_level: Optional[float] = None
        self.cooldown_hours = cooldown_hours
        if max_same_side is not None:
            self.max_same_side = max(1, int(max_same_side))
        elif max_same_side_pct is not None:
            self.max_same_side = max(1, int(self.max_positions * max_same_side_pct))
        else:
            self.max_same_side = max(1, int(0.75 * self.max_positions))
        if max_same_strategy is not None:
            self.max_same_strategy = max(1, int(max_same_strategy))
        elif max_same_strategy_pct is not None:
            self.max_same_strategy = max(
                1, int(self.max_positions * max_same_strategy_pct)
            )
        else:
            self.max_same_strategy = max(1, int(0.75 * self.max_positions))
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

    @classmethod
    def from_policy_config(
        cls,
        policy: PortfolioPolicyConfig,
        *,
        portfolio_value: float = 10000.0,
        cooldown_hours: float = 24.0,
        **kwargs: Any,
    ) -> "PortfolioManager":
        max_same_side = kwargs.pop(
            "max_same_side", policy.resolved_max_concurrent_per_side()
        )
        max_same_strategy = kwargs.pop(
            "max_same_strategy", policy.resolved_max_concurrent_per_strategy()
        )
        return cls(
            max_positions=policy.max_concurrent_positions,
            max_portfolio_pct=policy.max_total_wallet_allocation_pct,
            max_position_usdt=policy.max_position_quote_notional,
            max_position_pct=policy.max_position_wallet_pct,
            book_notional_multiplier=policy.book_notional_multiplier,
            leverage_wallet_multiplier=policy.leverage_wallet_multiplier,
            min_margin_level_after_entry=policy.min_margin_level_after_entry,
            max_same_side=max_same_side,
            max_same_strategy=max_same_strategy,
            portfolio_value=portfolio_value,
            cooldown_hours=cooldown_hours,
            **kwargs,
        )

    def update_margin_account_metrics(
        self,
        *,
        total_assets_quote: Optional[float],
        total_liabilities_quote: Optional[float],
    ) -> None:
        """Update cross-margin account metrics used for reserved slot sizing."""
        assets = (
            float(total_assets_quote)
            if total_assets_quote is not None and np.isfinite(float(total_assets_quote))
            else np.nan
        )
        liabilities = (
            float(total_liabilities_quote)
            if total_liabilities_quote is not None
            and np.isfinite(float(total_liabilities_quote))
            else np.nan
        )
        if not np.isfinite(assets) or not np.isfinite(liabilities):
            return
        self.margin_total_assets_quote = max(assets, 0.0)
        self.margin_total_liabilities_quote = max(liabilities, 0.0)
        equity = max(
            self.margin_total_assets_quote - self.margin_total_liabilities_quote,
            0.0,
        )
        if equity > 0.0:
            self.portfolio_value = equity
        self.margin_level = (
            self.margin_total_assets_quote
            / max(self.margin_total_liabilities_quote, 1e-12)
            if self.margin_total_liabilities_quote > 0.0
            else float("inf")
        )

    def get_portfolio_state(self) -> Dict[str, Any]:
        """Return current portfolio state summary."""
        open_positions = [p for p in self.positions.values() if p.is_open]
        n_open = len(open_positions)

        capacity = self.get_portfolio_capacity(side="", strategy_id="")

        # Calculate invested percentage against the current margin-safe book,
        # not against a static leverage multiplier.
        total_invested = sum(p.position_size for p in open_positions)
        max_total_notional = float(capacity.get("max_total_notional") or 0.0)
        invested_pct = (
            total_invested / max(max_total_notional, 1e-12)
            if max_total_notional > 0.0
            else 0.0
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
            remaining_pct = (
                float(capacity.get("remaining_total_notional") or 0.0)
                / max(max_total_notional, 1e-12)
                if max_total_notional > 0.0
                else 0.0
            )

        return {
            "n_positions": n_open,
            "max_positions": self.max_positions,
            "invested_usdt": total_invested,
            "invested_pct": invested_pct,
            "max_invested_pct": max_invested_pct,
            "remaining_pct": remaining_pct,
            "max_position_pct": self.max_position_pct,
            "max_position_usdt": self.max_position_usdt,
            "book_notional_multiplier": self.book_notional_multiplier,
            "leverage_wallet_multiplier": self.leverage_wallet_multiplier,
            "min_margin_level_after_entry": self.min_margin_level_after_entry,
            "max_total_notional": max_total_notional,
            "configured_book_notional": capacity.get("configured_book_notional"),
            "margin_surplus_notional": capacity.get("margin_surplus_notional"),
            "total_assets_quote": self.margin_total_assets_quote,
            "total_liabilities_quote": self.margin_total_liabilities_quote,
            "current_margin_level": self.margin_level,
            "long_count": long_count,
            "short_count": short_count,
            "max_same_side": self.max_same_side,
            "strategy_counts": strategy_counts,
            "max_same_strategy": self.max_same_strategy,
            "active_cooldowns": active_cooldowns,
            "open_symbols": list(self.positions.keys()),
            "hard_limits": self.get_hard_limit_status(),
        }

    def calculate_dynamic_threshold(
        self,
        initial_threshold: float,
        *,
        side: Optional[str] = None,
        strategy_id: Optional[str] = None,
        side_penalty: float = 0.0,
        strategy_penalty: float = 0.0,
        max_threshold: float = 0.99,
    ) -> float:
        """Calculate adjusted entry threshold based on current position count.

        Formula: final_threshold = initial_threshold + (n_positions * (1 - initial_threshold)) / max_positions

        As more positions are open, threshold increases making entry harder.
        """
        n_positions = len([p for p in self.positions.values() if p.is_open])

        # Formula: initial + (n_positions * (1 - initial)) / max_positions
        adjustment = (n_positions * (1.0 - initial_threshold)) / self.max_positions
        final_threshold = (
            initial_threshold
            + adjustment
            + float(side_penalty)
            + float(strategy_penalty)
        )

        return min(float(max_threshold), final_threshold)

    def calculate_position_size_cap(self, requested_size: float) -> float:
        """Calculate allowed position size considering portfolio constraints.

        Returns the minimum of the requested size, the per-position equity cap,
        and any optional absolute or total portfolio allocation caps.
        """
        open_positions = [p for p in self.positions.values() if p.is_open]
        total_invested = sum(p.position_size for p in open_positions)
        max_total_notional = float(
            self.get_portfolio_capacity(side="", strategy_id="")["max_total_notional"]
        )

        caps = [
            float(requested_size),
            float(self.portfolio_value)
            * float(self.max_position_pct)
            * float(self.book_notional_multiplier),
        ]
        if self.max_position_usdt is not None and np.isfinite(self.max_position_usdt):
            caps.append(float(self.max_position_usdt) * self.book_notional_multiplier)
        if self.max_portfolio_pct is not None and np.isfinite(self.max_portfolio_pct):
            caps.append(max_total_notional - total_invested)

        return max(0.0, min(caps))

    def get_portfolio_capacity(
        self,
        *,
        side: str,
        strategy_id: str,
        wallet_value: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Return deterministic portfolio capacity without mutating state."""
        wallet = (
            float(wallet_value)
            if wallet_value is not None and np.isfinite(float(wallet_value))
            else float(self.portfolio_value)
        )
        open_positions = [p for p in self.positions.values() if p.is_open]
        n_open = len(open_positions)
        side_open = sum(1 for p in open_positions if p.side == side)
        strategy_open = sum(1 for p in open_positions if p.strategy_id == strategy_id)
        open_notional = float(sum(p.position_size for p in open_positions))

        configured_book_notional = (
            float(self.max_portfolio_pct)
            * wallet
            * float(self.book_notional_multiplier)
            if self.max_portfolio_pct is not None
            and np.isfinite(float(self.max_portfolio_pct))
            else float("inf")
        )
        margin_surplus_notional = float("inf")
        if (
            self.margin_total_assets_quote is not None
            and self.margin_total_liabilities_quote is not None
            and np.isfinite(float(self.margin_total_assets_quote))
            and np.isfinite(float(self.margin_total_liabilities_quote))
        ):
            margin_surplus_notional = max(
                float(self.margin_total_assets_quote)
                - float(self.min_margin_level_after_entry)
                * float(self.margin_total_liabilities_quote),
                0.0,
            )
        max_total_notional = min(configured_book_notional, margin_surplus_notional)
        if not np.isfinite(max_total_notional):
            max_total_notional = configured_book_notional
        remaining_total = max(max_total_notional - open_notional, 0.0)
        per_position_caps = [
            float(self.max_position_pct) * wallet * float(self.book_notional_multiplier)
        ]
        if self.max_position_usdt is not None and np.isfinite(
            float(self.max_position_usdt)
        ):
            per_position_caps.append(
                float(self.max_position_usdt) * self.book_notional_multiplier
            )
        max_position_notional = max(0.0, min(per_position_caps))
        remaining_position_slots = self.max_positions - n_open
        remaining_side_slots = self.max_same_side - side_open
        remaining_strategy_slots = self.max_same_strategy - strategy_open
        return {
            "wallet_value": wallet,
            "open_positions": n_open,
            "side_open_positions": side_open,
            "strategy_open_positions": strategy_open,
            "open_notional": open_notional,
            "max_concurrent_positions": self.max_positions,
            "max_concurrent_per_side": self.max_same_side,
            "max_concurrent_per_strategy": self.max_same_strategy,
            "max_total_notional": max_total_notional,
            "configured_book_notional": configured_book_notional,
            "margin_surplus_notional": (
                margin_surplus_notional
                if np.isfinite(margin_surplus_notional)
                else None
            ),
            "remaining_total_notional": remaining_total,
            "max_position_notional": max_position_notional,
            "book_notional_multiplier": float(self.book_notional_multiplier),
            "leverage_wallet_multiplier": float(self.leverage_wallet_multiplier),
            "min_margin_level_after_entry": float(self.min_margin_level_after_entry),
            "total_assets_quote": self.margin_total_assets_quote,
            "total_liabilities_quote": self.margin_total_liabilities_quote,
            "current_margin_level": self.margin_level,
            "remaining_position_slots": remaining_position_slots,
            "remaining_side_slots": remaining_side_slots,
            "remaining_strategy_slots": remaining_strategy_slots,
            "allowed_by_count_caps": bool(
                remaining_position_slots > 0
                and remaining_side_slots > 0
                and remaining_strategy_slots > 0
            ),
            "allowed_by_allocation_caps": bool(
                remaining_total > 0.0 and max_position_notional > 0.0
            ),
        }

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

    def trip_hard_limit(self, reason: str) -> None:
        """Block new entries until the hard-risk gate is manually reset."""
        self._trip_hard_limit(reason)

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
                try:
                    raw_positions = fetch_positions([], position_params)
                except TypeError:
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
        confidence_score: Optional[float] = None,
        initial_threshold: float = 0.90,
        current_time: Optional[pd.Timestamp] = None,
        requested_position_size: Optional[float] = None,
        rank_score: Optional[float] = None,
        wallet_value: Optional[float] = None,
        side_penalty: float = 0.0,
        strategy_penalty: float = 0.0,
    ) -> Tuple[bool, Dict[str, Any]]:
        """Check if a new position can be entered.

        Args:
            symbol: Asset symbol
            side: "long" or "short"
            strategy_id: Strategy identifier
            confidence_score: Legacy confidence score (0-1)
            initial_threshold: Base threshold for entry
            current_time: Timestamp for cooldown checks (default: now)

        Returns:
            (allowed: bool, info: dict with details)
        """
        if current_time is None:
            current_time = pd.Timestamp.now(tz="UTC")
        effective_score = (
            float(rank_score)
            if rank_score is not None
            else float(confidence_score) if confidence_score is not None else np.nan
        )

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
            info["reason"] = "hard_limit_block"
            info["hard_limit_reason"] = hard_status.get("reason", "hard_limit_block")
            info["constraints_checked"].append("hard_limits")
            return False, info

        capacity = self.get_portfolio_capacity(
            side=side,
            strategy_id=strategy_id,
            wallet_value=wallet_value,
        )
        info["capacity"] = capacity

        if symbol in self.positions and self.positions[symbol].is_open:
            info["reason"] = "symbol_already_has_position"
            info["constraints_checked"].append("max_one_per_symbol")
            return False, info

        if symbol in self.cooldowns:
            cd = self.cooldowns[symbol]
            if cd.cooldown_until > current_time:
                hours_remaining = (
                    cd.cooldown_until - current_time
                ).total_seconds() / 3600
                info["reason"] = "cooldown_active"
                info["constraints_checked"].append("cooldown")
                info["cooldown_until"] = cd.cooldown_until.isoformat()
                info["cooldown_hours_remaining"] = float(hours_remaining)
                return False, info

        if n_positions >= self.max_positions:
            info["reason"] = "max_concurrent_positions_reached"
            info["constraints_checked"].append("max_positions")
            return False, info

        side_count = int(capacity["side_open_positions"])
        if side_count >= self.max_same_side:
            info["reason"] = "max_concurrent_per_side_reached"
            info["constraints_checked"].append("max_same_side")
            return False, info

        strategy_count = int(capacity["strategy_open_positions"])
        if strategy_count >= self.max_same_strategy:
            info["reason"] = "max_concurrent_per_strategy_reached"
            info["constraints_checked"].append("max_same_strategy")
            return False, info

        final_threshold = self.calculate_dynamic_threshold(
            initial_threshold,
            side=side,
            strategy_id=strategy_id,
            side_penalty=side_penalty,
            strategy_penalty=strategy_penalty,
        )
        info["final_threshold"] = final_threshold
        info["initial_threshold"] = initial_threshold
        info["rank_score"] = effective_score
        info["side_penalty"] = float(side_penalty)
        info["strategy_penalty"] = float(strategy_penalty)
        info["constraints_checked"].append("dynamic_threshold")

        if not np.isfinite(effective_score) or effective_score < final_threshold:
            info["reason"] = "rank_below_dynamic_threshold"
            info["constraints_checked"].append("rank_threshold")
            return False, info

        requested_size = (
            float(requested_position_size)
            if requested_position_size is not None
            else self.portfolio_value
            * self.max_position_pct
            * self.book_notional_multiplier
        )
        if not np.isfinite(requested_size) or requested_size <= 0.0:
            info["reason"] = "invalid_requested_position_size"
            info["requested_position_size"] = requested_size
            return False, info
        position_size_cap = self.calculate_position_size_cap(requested_size)
        info["position_size_cap"] = position_size_cap
        info["requested_position_size"] = requested_size
        info["constraints_checked"].append("position_size_cap")

        if position_size_cap <= 0:
            info["reason"] = "no_remaining_portfolio_capacity"
            return False, info
        if requested_size > float(capacity["remaining_total_notional"]):
            info["reason"] = "requested_size_exceeds_remaining_total_notional"
            return False, info

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
            "book_notional_multiplier": self.book_notional_multiplier,
            "leverage_wallet_multiplier": self.leverage_wallet_multiplier,
            "min_margin_level_after_entry": self.min_margin_level_after_entry,
            "margin_total_assets_quote": self.margin_total_assets_quote,
            "margin_total_liabilities_quote": self.margin_total_liabilities_quote,
            "margin_level": self.margin_level,
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
        self.book_notional_multiplier = max(
            0.0,
            float(
                data.get(
                    "book_notional_multiplier",
                    self.book_notional_multiplier,
                )
            ),
        )
        self.leverage_wallet_multiplier = max(
            1.0,
            float(
                data.get(
                    "leverage_wallet_multiplier",
                    self.leverage_wallet_multiplier,
                )
            ),
        )
        self.min_margin_level_after_entry = max(
            1.0,
            float(
                data.get(
                    "min_margin_level_after_entry",
                    self.min_margin_level_after_entry,
                )
            ),
        )
        assets = data.get("margin_total_assets_quote")
        liabilities = data.get("margin_total_liabilities_quote")
        if assets is not None and liabilities is not None:
            self.update_margin_account_metrics(
                total_assets_quote=float(assets),
                total_liabilities_quote=float(liabilities),
            )
        tprint(
            f"[PortfolioManager] State loaded from {filepath} "
            f"({len(self.positions)} positions, {len(self.cooldowns)} cooldowns)"
        )


__all__ = ["PortfolioManager", "Position", "CooldownRecord"]
