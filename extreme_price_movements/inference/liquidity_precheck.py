"""Ticker, book-walk liquidity, and signal-gap checks for live entries."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from extreme_price_movements.inference.portfolio_policy import PortfolioPolicyConfig


@dataclass(frozen=True)
class ExecutionSnapshot:
    symbol: str
    timestamp: pd.Timestamp
    bid: Optional[float] = None
    ask: Optional[float] = None
    last: Optional[float] = None
    mid: Optional[float] = None
    spread_bps: Optional[float] = None
    orderbook_side: Optional[str] = None
    best_touch: Optional[float] = None
    max_walk_price: Optional[float] = None
    orderbook_capacity_quote_within_slippage: Optional[float] = None
    intended_quote_size: Optional[float] = None
    expected_fill_price: Optional[float] = None
    expected_fill_slippage_bps: Optional[float] = None
    expected_total_entry_friction_bps: Optional[float] = None
    spread_weight: float = 1.0
    depth_weight: float = 1.0
    liquidity_capacity_weight: float = 1.0
    hard_reject: bool = False
    reject_reason: Optional[str] = None
    details: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        out = asdict(self)
        out["timestamp"] = pd.Timestamp(self.timestamp).isoformat()
        return out


def _safe_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def _is_live_mode(mode: str) -> bool:
    return str(mode or "").lower() in {"live", "live-test", "live_test"}


def fetch_ticker_snapshot(
    *,
    exchange: Any,
    symbol: str,
    side: str,
    policy: PortfolioPolicyConfig,
    mode: str,
    now: Optional[pd.Timestamp] = None,
) -> ExecutionSnapshot:
    """Fetch ticker and enforce touch/spread/freshness requirements."""
    request_started_ts = pd.Timestamp(
        now if now is not None else pd.Timestamp.now(tz="UTC")
    )
    now_ts = request_started_ts
    if now_ts.tzinfo is None:
        now_ts = now_ts.tz_localize("UTC")
    else:
        now_ts = now_ts.tz_convert("UTC")
    ticker = exchange.fetch_ticker(symbol)
    received_ts = (
        now_ts
        if now is not None
        else pd.Timestamp.now(tz="UTC").tz_convert("UTC")
    )
    bid = _safe_float(ticker.get("bid"))
    ask = _safe_float(ticker.get("ask"))
    last = _safe_float(ticker.get("last"))
    ts_raw = ticker.get("timestamp")
    exchange_ticker_ts = None
    if ts_raw is not None:
        try:
            exchange_ticker_ts = pd.to_datetime(float(ts_raw), unit="ms", utc=True)
        except Exception:
            exchange_ticker_ts = None

    reject = None
    mid = spread_bps = None
    if not (np.isfinite(bid) and np.isfinite(ask)) or bid <= 0 or ask <= 0:
        reject = "missing_or_invalid_bid_ask"
    elif ask < bid:
        reject = "crossed_ticker_bid_ask"
    else:
        mid = 0.5 * (bid + ask)
        spread_bps = (ask - bid) / max(mid, 1e-12) * 10000.0

    if reject is None and _is_live_mode(mode):
        local_age = (received_ts - now_ts).total_seconds()
        if local_age > float(policy.max_ticker_age_seconds):
            reject = "stale_ticker"

    spread_weight = 1.0
    if reject is None and spread_bps is not None:
        if spread_bps > policy.max_spread_bps:
            spread_weight = 1.0 - (
                (spread_bps - policy.max_spread_bps)
                / max(policy.hard_max_spread_bps - policy.max_spread_bps, 1e-12)
            )
            spread_weight = float(np.clip(spread_weight, 0.0, 1.0))

    return ExecutionSnapshot(
        symbol=symbol,
        timestamp=now_ts,
        bid=float(bid) if np.isfinite(bid) else None,
        ask=float(ask) if np.isfinite(ask) else None,
        last=float(last) if np.isfinite(last) else None,
        mid=float(mid) if mid is not None and np.isfinite(mid) else None,
        spread_bps=(
            float(spread_bps)
            if spread_bps is not None and np.isfinite(spread_bps)
            else None
        ),
        spread_weight=spread_weight,
        liquidity_capacity_weight=spread_weight,
        hard_reject=reject is not None,
        reject_reason=reject,
        details={
            "ticker_request_started_at": now_ts.isoformat(),
            "ticker_received_at": received_ts.isoformat(),
            "ticker_fetch_latency_seconds": float(
                max((received_ts - now_ts).total_seconds(), 0.0)
            ),
            "exchange_ticker_timestamp": (
                exchange_ticker_ts.isoformat()
                if exchange_ticker_ts is not None
                else None
            ),
            "exchange_ticker_age_seconds": (
                float(max((received_ts - exchange_ticker_ts).total_seconds(), 0.0))
                if exchange_ticker_ts is not None
                else None
            ),
            "side": side,
        },
    )


def _walk_levels(
    levels: Any,
    *,
    side: str,
    best_touch: float,
    max_walk_price: float,
    intended_quote_size: float,
) -> Tuple[float, float, float]:
    filled_quote = 0.0
    filled_base = 0.0
    total_cost = 0.0
    for raw in levels or []:
        if not isinstance(raw, (list, tuple)) or len(raw) < 2:
            continue
        price = _safe_float(raw[0])
        amount = _safe_float(raw[1])
        if (
            not (np.isfinite(price) and np.isfinite(amount))
            or price <= 0
            or amount <= 0
        ):
            continue
        if side == "long" and price > max_walk_price:
            break
        if side == "short" and price < max_walk_price:
            break
        level_quote = price * amount
        take_quote = min(level_quote, max(intended_quote_size - filled_quote, 0.0))
        if take_quote <= 0:
            break
        take_base = take_quote / price
        filled_quote += take_quote
        filled_base += take_base
        total_cost += take_base * price
        if filled_quote >= intended_quote_size:
            break
    expected_price = total_cost / filled_base if filled_base > 0 else float("nan")
    return filled_quote, filled_base, expected_price


def evaluate_orderbook_liquidity(
    *,
    exchange: Any,
    symbol: str,
    side: str,
    intended_quote_size: float,
    ticker_snapshot: ExecutionSnapshot,
    policy: PortfolioPolicyConfig,
    mode: str,
) -> ExecutionSnapshot:
    """Measure executable quote capacity within slippage and entry-friction caps."""
    if ticker_snapshot.hard_reject:
        return ticker_snapshot
    bid = _safe_float(ticker_snapshot.bid)
    ask = _safe_float(ticker_snapshot.ask)
    spread = float(ticker_snapshot.spread_bps or 0.0)
    half_spread_bps = max(0.0, spread / 2.0)
    effective_slippage_cap_bps = max(0.0, float(policy.max_orderbook_slippage_bps))
    if side == "long":
        best_touch = ask
        max_walk_price = ask * (1.0 + effective_slippage_cap_bps / 10000.0)
        levels_key = "asks"
    else:
        best_touch = bid
        max_walk_price = bid * (1.0 - effective_slippage_cap_bps / 10000.0)
        levels_key = "bids"
    if not np.isfinite(best_touch) or best_touch <= 0:
        return replace(
            ticker_snapshot,
            hard_reject=True,
            reject_reason="missing_best_touch",
        )

    orderbook = exchange.fetch_order_book(symbol)
    filled_quote, _filled_base, expected_price = _walk_levels(
        orderbook.get(levels_key, []),
        side=side,
        best_touch=best_touch,
        max_walk_price=max_walk_price,
        intended_quote_size=float(intended_quote_size),
    )
    depth_weight = min(1.0, filled_quote / max(float(intended_quote_size), 1e-9))
    if np.isfinite(expected_price) and expected_price > 0:
        if side == "long":
            fill_slip = (expected_price / best_touch - 1.0) * 10000.0
        else:
            fill_slip = (1.0 - expected_price / best_touch) * 10000.0
    else:
        fill_slip = float("nan")
    total_friction = half_spread_bps + (
        float(fill_slip) if np.isfinite(fill_slip) else 0.0
    )
    liq_weight = min(float(ticker_snapshot.spread_weight), float(depth_weight))

    reject = None
    if filled_quote <= 0:
        reject = "no_orderbook_capacity_within_slippage"
    elif np.isfinite(fill_slip) and fill_slip > policy.max_orderbook_slippage_bps:
        reject = "orderbook_slippage_above_cap"
    elif liq_weight < policy.min_liquidity_capacity_weight:
        reject = "liquidity_capacity_weight_below_min"

    return replace(
        ticker_snapshot,
        orderbook_side=levels_key,
        best_touch=float(best_touch),
        max_walk_price=float(max_walk_price),
        orderbook_capacity_quote_within_slippage=float(filled_quote),
        intended_quote_size=float(intended_quote_size),
        expected_fill_price=(
            float(expected_price) if np.isfinite(expected_price) else None
        ),
        expected_fill_slippage_bps=(
            float(fill_slip) if np.isfinite(fill_slip) else None
        ),
        expected_total_entry_friction_bps=float(total_friction),
        depth_weight=float(depth_weight),
        liquidity_capacity_weight=float(liq_weight),
        hard_reject=reject is not None,
        reject_reason=reject,
        details={
            **(ticker_snapshot.details or {}),
            "mode": mode,
            "entry_friction_formula": "expected_fill_slippage_bps + spread_bps / 2",
            "half_spread_bps": float(half_spread_bps),
            "max_entry_friction_bps": float(policy.max_entry_friction_bps),
            "effective_orderbook_slippage_cap_bps": float(effective_slippage_cap_bps),
            "entry_friction_gate": "ev_haircut",
        },
    )


def compute_price_gap_rank_penalty(
    *,
    strategy_id: str,
    side: str,
    signal_price: float,
    decision_mid: float,
    policy: PortfolioPolicyConfig,
) -> Tuple[float, Dict[str, Any]]:
    """Compute rank-space penalty from OHLCV signal price to execution mid."""
    signal = _safe_float(signal_price)
    mid = _safe_float(decision_mid)
    if not (np.isfinite(signal) and np.isfinite(mid)) or signal <= 0 or mid <= 0:
        return policy.price_gap_penalty_max, {
            "hard_reject": True,
            "reason": "invalid_price_gap_inputs",
        }
    direction = 1.0 if str(side).lower() == "long" else -1.0
    signal_gap_bps = direction * (mid / signal - 1.0) * 10000.0
    max_gap = float(policy.max_signal_gap_bps_default)
    deadband = max(float(policy.price_gap_deadband_bps), 0.0)
    favorable_multiplier = float(
        np.clip(policy.price_gap_favorable_penalty_multiplier, 0.0, 1.0)
    )
    # Positive signed gap is a worse entry for either side. A favorable move can
    # still indicate a stale signal, but receives only a small freshness penalty.
    adverse_gap = max(signal_gap_bps - deadband, 0.0)
    favorable_gap = max(-signal_gap_bps - deadband, 0.0)
    penalized_gap = adverse_gap + favorable_multiplier * favorable_gap
    penalty = policy.price_gap_penalty_max * min(
        penalized_gap / max(max_gap, 1e-9), 1.0
    )
    return float(penalty), {
        "hard_reject": False,
        "signal_gap_bps": float(signal_gap_bps),
        "price_gap_deadband_bps": float(deadband),
        "price_gap_adverse_excess_bps": float(adverse_gap),
        "price_gap_favorable_excess_bps": float(favorable_gap),
        "price_gap_favorable_penalty_multiplier": float(favorable_multiplier),
        "penalized_gap_bps": float(penalized_gap),
    }


def marketable_limit_price(
    *,
    side: str,
    decision_mid: float,
    policy: PortfolioPolicyConfig,
) -> float:
    """Return marketable limit price constrained by chase bps."""
    mid = float(decision_mid)
    chase = float(policy.max_order_chase_bps) / 10000.0
    return mid * (1.0 + chase) if str(side).lower() == "long" else mid * (1.0 - chase)
