"""Trade executor for inference.

Live mode places an entry order plus a STOP_LOSS order and updates that stop
with cancel-replace. Shadow mode records the same lifecycle without sending
exchange orders. The inference caller passes USDT quote notional; live orders
convert that notional to base quantity before touching the exchange.
"""

import hashlib
import hmac
import json
import os
import threading
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.parse import urlencode

import numpy as np
import pandas as pd

from extreme_price_movements import hf_data_loader
from extreme_price_movements.data_store import _resolve_perp_symbol
from extreme_price_movements.inference.parity import strategy_core_id
from extreme_price_movements.inference.simple_policy_stop import (
    SimplePolicyStopDecision,
    SimplePolicyStopParamsError,
    compute_initial_simple_policy_stop_decision,
    compute_simple_policy_stop_decision,
    extract_simple_policy_stop_params_by_strategy,
    is_concrete_simple_policy_params_source,
    validate_simple_policy_stop_params,
)
from extreme_price_movements.utils import tprint

LIVE_EXECUTION_MODES = {"live", "live-test", "live_test", "livetest"}
DUST_TO_BNB_METHOD_VERSION = "margin_dust_v2_repeated_params"
SMALL_LIABILITY_EXCHANGE_METHOD_VERSION = "small_liability_v5_target_guard"
NON_STOP_COMPAT_BUCKET_KEYS = {
    "max_hold_hours",
    "cooldown_hours",
}
CANONICAL_STOP_POSITION_FIELDS = {
    "stop_price",
    "barrier_frac",
    "barrier_pct",
    "sl_mult",
    "strategy_id",
    "decision_module",
    "stop_policy_params_source",
    "stop_policy_params_hash",
    "stop_policy_schema",
    "stop_trigger_signal",
    "stop_trigger_reference_source",
}


def _shadow_stop_order_id(symbol: str, state: Dict[str, Any]) -> str:
    """Return a deterministic synthetic stop id for shadow monitoring."""
    raw = "|".join(
        [
            str(symbol),
            str(state.get("side") or ""),
            str(state.get("entry_time") or state.get("timestamp") or ""),
            str(state.get("stop_price") or ""),
            str(state.get("strategy_id") or state.get("bucket_key") or ""),
        ]
    )
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
    return f"shadow-stop-{digest}"


def _validate_policy_stop_provenance(state: Dict[str, Any]) -> List[str]:
    stop_price = _safe_float(state.get("stop_price"), default=np.nan)
    barrier_frac = _safe_float(state.get("barrier_frac"), default=np.nan)
    sl_mult = _safe_float(state.get("sl_mult"), default=np.nan)
    required_text = (
        "stop_policy_params_source",
        "stop_policy_params_hash",
        "stop_policy_schema",
        "strategy_id",
    )
    missing = [key for key in required_text if not str(state.get(key) or "").strip()]
    if str(state.get("stop_policy_schema") or "") != "simple_policy_v1":
        missing.append("stop_policy_schema=simple_policy_v1")
    if not np.isfinite(barrier_frac) or barrier_frac <= 0.0:
        missing.append("barrier_frac")
    if not np.isfinite(sl_mult) or sl_mult <= 0.0:
        missing.append("sl_mult")
    if not np.isfinite(stop_price) or stop_price <= 0.0:
        missing.append("stop_price")
    return missing


MODEL_AND_POLICY_CONTEXT_KEYS = (
    "base_pred",
    "base_rank_pct",
    "base_train_rank_pct",
    "base_gate_top_frac",
    "meta_pred",
    "meta_train_rank_pct",
    "rank_score_source",
    "calibrated_score",
    "rank_percentile",
    "effective_threshold",
    "deployment_rank_threshold",
    "quote_size",
    "requested_base_amount",
    "entry_notional_quote",
    "wallet_value",
    "wallet_value_at_entry",
    "open_notional",
    "open_notional_at_entry",
    "leverage_wallet_multiplier",
    "book_notional_multiplier",
    "safe_book_notional",
    "target_slot_notional",
    "current_margin_level",
    "market_mode",
    "perp_rank_number",
    "perp_rank_x",
    "perp_rank_leverage",
    "perp_risk_cap_leverage",
    "perp_effective_leverage",
    "perp_stop_loss_pct",
    "perp_full_wallet",
    "perp_available_wallet",
    "position_size_before_liquidity",
    "position_size_after_liquidity",
    "policy_artifact_run_id",
    "policy_schema_version",
)
TRIGGER_PRICE_REJECT_TOKENS = (
    "order_would_immediately_trigger",
    "order would immediately trigger",
    "orderimmediatelyfillable",
    "immediately fillable",
    "strategy_invalid_trigger_price",
    "invalid trigger price",
    "conditional_order_trigger_reject",
    "trigger reject",
    "trigger price",
)

STOP_MIN_CURRENT_DISTANCE_PCT = 0.003

EXECUTION_AUDIT_KEYS = (
    "signal_price",
    "decision_mid",
    "theoretical_entry_price",
    "policy_entry_price",
    "signal_gap_bps",
    "signal_bar_close_ts",
    "signal_close_to_decision_seconds",
    "signal_to_decision_seconds",
    "max_signal_close_to_entry_seconds",
    "signal_to_entry_alert_seconds",
    "stale_signal_age_gate_enabled",
    "stale_signal_age_gate_exceeded",
    "bid",
    "ask",
    "mid",
    "spread_bps",
    "ticker_bid",
    "ticker_ask",
    "ticker_mid",
    "ticker_spread_bps",
    "expected_fill_price",
    "expected_entry_price",
    "expected_fill_slippage_bps",
    "orderbook_slippage_bps",
    "slippage_bps",
    "entry_gap_bps",
    "entry_slippage_proxy_bps",
    "hourly_close_to_latest_decision_price_bps",
    "decision_price_to_fill_bps",
    "latest_decision_price",
    "entry_price_attribution_schema",
    "spread_proxy_bps",
    "orderbook_live_slippage_bps",
    "fee_bps",
    "adverse_signal_gap_bps",
    "expected_total_entry_friction_bps",
    "expected_friction_drag_bps",
    "entry_delay_effect_bps",
    "entry_delay_adverse_bps",
    "entry_delay_abs_bps",
    "decision_to_entry_seconds",
    "signal_to_entry_seconds",
    "gross_to_net_friction_drag_bps",
    "orderbook_side",
    "best_touch",
    "max_walk_price",
    "orderbook_capacity_quote_within_slippage",
    "intended_quote_size",
    "spread_weight",
    "depth_weight",
    "liquidity_capacity_weight",
    "price_gap_penalty",
    "adjusted_rank_score",
    "final_threshold",
    "position_size_before_liquidity",
    "position_size_after_liquidity",
    "wallet_value",
    "wallet_value_at_entry",
    "open_notional",
    "open_notional_at_entry",
    "leverage_wallet_multiplier",
    "book_notional_multiplier",
    "safe_book_notional",
    "target_slot_notional",
    "current_margin_level",
    "market_mode",
    "perp_rank_number",
    "perp_rank_x",
    "perp_rank_leverage",
    "perp_risk_cap_leverage",
    "perp_effective_leverage",
    "perp_stop_loss_pct",
    "perp_full_wallet",
    "perp_available_wallet",
    "max_chase_bps",
    "entry_limit_price",
    "decision_audit_schema",
    "model_prediction_audit",
    "raw_data_audit",
    "model_feature_audit",
)


def _is_live_execution_mode(mode: str) -> bool:
    return str(mode or "").strip().lower() in LIVE_EXECUTION_MODES


def _config_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "y", "on"}:
            return True
        if text in {"0", "false", "no", "n", "off"}:
            return False
    return bool(default)


def _default_cross_margin_dust_quote_threshold(mode: str) -> float:
    """Return mode-specific dust threshold for margin reconciliation."""
    mode_l = str(mode or "").strip().lower()
    if mode_l in {"live-test", "live_test", "livetest"}:
        return 2.5
    if mode_l == "live":
        return 5.0
    return 5.0


def _safe_float(value: Any, default: float = np.nan) -> float:
    """Convert exchange payload values to float without raising."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def _first_finite_price(source: Optional[Dict[str, Any]], keys: Sequence[str]) -> float:
    if not isinstance(source, dict):
        return np.nan
    for key in keys:
        value = _safe_float(source.get(key), default=np.nan)
        if np.isfinite(value) and value > 0.0:
            return float(value)
    return np.nan


def _execution_audit_fields(source: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Return execution-quality fields safe to persist in trade logs/emails."""
    if not isinstance(source, dict):
        return {}
    out = {key: source.get(key) for key in EXECUTION_AUDIT_KEYS if key in source}
    if "bid" in source and "ticker_bid" not in out:
        out["ticker_bid"] = source.get("bid")
    if "ask" in source and "ticker_ask" not in out:
        out["ticker_ask"] = source.get("ask")
    if "mid" in source and "ticker_mid" not in out:
        out["ticker_mid"] = source.get("mid")
    if "spread_bps" in source and "ticker_spread_bps" not in out:
        out["ticker_spread_bps"] = source.get("spread_bps")
    if "spread_bps" in source and "spread_proxy_bps" not in out:
        out["spread_proxy_bps"] = source.get("spread_bps")
    if "expected_fill_price" in source and "expected_entry_price" not in out:
        out["expected_entry_price"] = source.get("expected_fill_price")
    if (
        "expected_fill_slippage_bps" in source
        and "orderbook_live_slippage_bps" not in out
    ):
        out["orderbook_live_slippage_bps"] = source.get("expected_fill_slippage_bps")
    return out


def _exchange_error_text(exc: Exception) -> str:
    return f"{exc.__class__.__name__} {exc}".lower()


def _exchange_reject_reason(exc: Exception) -> str:
    """Extract the most useful stable reject reason from a Binance/ccxt error."""
    text = _exchange_error_text(exc)
    for token in TRIGGER_PRICE_REJECT_TOKENS:
        if token in text:
            return token.upper().replace(" ", "_")
    for token in (
        "insufficient",
        "balance",
        "min_notional",
        "lot_size",
        "precision",
        "rate limit",
        "too many requests",
        "timeout",
        "network",
        "permission",
        "unauthorized",
    ):
        if token in text:
            return token.upper().replace(" ", "_")
    return exc.__class__.__name__


def _is_trigger_price_reject(exc: Exception) -> bool:
    text = _exchange_error_text(exc)
    return any(token in text for token in TRIGGER_PRICE_REJECT_TOKENS)


def _stop_min_distance_boundary(
    side: str,
    current_price: float,
    min_distance_pct: float = STOP_MIN_CURRENT_DISTANCE_PCT,
) -> float:
    if not np.isfinite(current_price) or current_price <= 0.0:
        return np.nan
    side_l = str(side or "").lower()
    if side_l == "long":
        return float(current_price) * (1.0 - float(min_distance_pct))
    if side_l == "short":
        return float(current_price) * (1.0 + float(min_distance_pct))
    return np.nan


def _stop_side_is_valid(
    side: str,
    stop_price: float,
    current_price: float,
    min_distance_pct: float = STOP_MIN_CURRENT_DISTANCE_PCT,
) -> bool:
    if not (np.isfinite(stop_price) and np.isfinite(current_price)):
        return False
    if current_price <= 0.0 or stop_price <= 0.0:
        return False
    side_l = str(side or "").lower()
    boundary = _stop_min_distance_boundary(
        side_l, current_price, min_distance_pct=min_distance_pct
    )
    if not np.isfinite(boundary):
        return False
    return stop_price <= boundary if side_l == "long" else stop_price >= boundary


def _adjust_stop_to_min_current_distance(
    side: str,
    stop_price: float,
    current_price: float,
    min_distance_pct: float = STOP_MIN_CURRENT_DISTANCE_PCT,
) -> Tuple[float, bool, float]:
    """Move a stop outward so Binance will not reject an immediate trigger."""
    if not (np.isfinite(stop_price) and np.isfinite(current_price)):
        return stop_price, False, np.nan
    boundary = _stop_min_distance_boundary(
        side, current_price, min_distance_pct=min_distance_pct
    )
    if not np.isfinite(boundary):
        return stop_price, False, np.nan
    side_l = str(side or "").lower()
    if side_l == "long" and stop_price > boundary:
        return float(boundary), True, float(boundary)
    if side_l == "short" and stop_price < boundary:
        return float(boundary), True, float(boundary)
    return float(stop_price), False, float(boundary)


def _stop_improves_existing(side: str, candidate: float, existing: float) -> bool:
    if not (np.isfinite(candidate) and np.isfinite(existing)):
        return False
    side_l = str(side or "").lower()
    return candidate > existing if side_l == "long" else candidate < existing


def _exchange_valid_giveback_fallback_stop(
    exchange: Any,
    symbol: str,
    *,
    side: str,
    policy_stop: float,
    existing_stop: float,
    current_price: float,
    config: Optional[Dict[str, Any]],
) -> Tuple[float, float, float, str]:
    """Return a monotonic fallback stop clipped to an exchange-valid trigger side."""
    if not (
        np.isfinite(policy_stop)
        and policy_stop > 0.0
        and np.isfinite(existing_stop)
        and existing_stop > 0.0
        and np.isfinite(current_price)
        and current_price > 0.0
    ):
        return np.nan, np.nan, np.nan, ""
    if not _stop_improves_existing(side, policy_stop, existing_stop):
        return np.nan, np.nan, np.nan, ""

    distance_modes: List[Tuple[float, str]] = [
        (STOP_MIN_CURRENT_DISTANCE_PCT, "standard_min_distance")
    ]
    # Kraken Futures stops use executable side-aware triggers. For very coarse
    # ticks, the conservative min-distance boundary can round back to the invalid
    # side even though a trigger-side stop is placeable and still reduces risk.
    if (
        _exchange_id(exchange) == "krakenfutures"
        or _configured_exchange_id(config) == "krakenfutures"
    ):
        distance_modes.append((0.0, "kraken_trigger_side"))

    for min_distance_pct, mode in distance_modes:
        adjusted, _, boundary = _adjust_stop_to_min_current_distance(
            side,
            float(policy_stop),
            float(current_price),
            min_distance_pct=float(min_distance_pct),
        )
        if not np.isfinite(adjusted) or adjusted <= 0.0:
            continue
        candidate = _exchange_precision(exchange, symbol, float(adjusted), kind="price")
        if not (
            np.isfinite(candidate)
            and candidate > 0.0
            and _stop_side_is_valid(
                side,
                float(candidate),
                float(current_price),
                min_distance_pct=float(min_distance_pct),
            )
            and _stop_improves_existing(side, float(candidate), float(existing_stop))
        ):
            continue
        return (
            float(candidate),
            float(boundary) if np.isfinite(boundary) else np.nan,
            float(min_distance_pct),
            mode,
        )
    return np.nan, np.nan, np.nan, ""


def _validate_policy_stop_decision(
    decision: Any,
    *,
    require_should_replace: bool = True,
    position_state: Optional[Dict[str, Any]] = None,
    artifact_params: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, str]:
    """Validate policy-stop decision metadata before state/exchange mutation."""
    if not isinstance(decision, SimplePolicyStopDecision):
        return False, "policy_stop_decision must be SimplePolicyStopDecision"
    if require_should_replace and not decision.should_replace:
        return False, "policy_stop_decision does not request replacement"
    stop_price = _safe_float(decision.stop_price, default=np.nan)
    barrier_frac = _safe_float(decision.barrier_frac, default=np.nan)
    sl_mult = _safe_float(decision.sl_mult, default=np.nan)
    if decision.params_schema != "simple_policy_v1":
        return False, "invalid simple-policy decision schema"
    if not str(decision.strategy_id or "").strip():
        return False, "missing simple-policy decision strategy_id"
    params_source = str(decision.params_source or "").strip()
    if not params_source:
        return False, "missing simple-policy decision params_source"
    if not is_concrete_simple_policy_params_source(params_source):
        return False, "invalid simple-policy decision params_source"
    if not str(decision.params_hash or "").strip():
        return False, "missing simple-policy decision params_hash"
    if require_should_replace and not np.isfinite(stop_price):
        return False, "missing finite simple-policy decision stop_price"
    if not np.isfinite(barrier_frac) or barrier_frac <= 0.0:
        return False, "invalid simple-policy decision barrier_frac"
    if not np.isfinite(sl_mult) or sl_mult <= 0.0:
        return False, "invalid simple-policy decision sl_mult"
    if position_state is not None:
        expected_strategy = str(
            position_state.get("strategy_id") or position_state.get("bucket_key") or ""
        ).strip()
        if expected_strategy and decision.strategy_id != expected_strategy:
            return (
                False,
                "simple-policy decision strategy_id does not match active position",
            )
        if (
            str(position_state.get("stop_policy_params_source") or "").strip()
            != params_source
        ):
            return (
                False,
                "simple-policy decision params_source does not match active position",
            )
        if (
            str(position_state.get("stop_policy_params_hash") or "").strip()
            != str(decision.params_hash or "").strip()
        ):
            return (
                False,
                "simple-policy decision params_hash does not match active position",
            )
    if artifact_params is not None:
        if (
            str(artifact_params.get("strategy_id") or "").strip()
            != decision.strategy_id
        ):
            return (
                False,
                "simple-policy decision strategy_id does not match latest artifact",
            )
        if str(artifact_params.get("params_source") or "").strip() != params_source:
            return (
                False,
                "simple-policy decision params_source does not match latest artifact",
            )
        if (
            str(artifact_params.get("params_hash") or "").strip()
            != str(decision.params_hash or "").strip()
        ):
            return (
                False,
                "simple-policy decision params_hash does not match latest artifact",
            )
    return True, ""


def _append_position_event(
    state: Dict[str, Any],
    event: str,
    **payload: Any,
) -> None:
    """Append a compact per-position audit event for close recaps."""
    events = state.setdefault("trade_recap_events", [])
    if not isinstance(events, list):
        events = []
        state["trade_recap_events"] = events
    if len(events) >= 500:
        del events[: len(events) - 499]
    clean_payload: Dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, (np.floating, np.integer)):
            value = value.item()
        clean_payload[key] = value
    events.append(
        {
            "ts": pd.Timestamp.now(tz="UTC").isoformat(),
            "event": str(event),
            **clean_payload,
        }
    )


def _order_filled_amount(order: Optional[Dict[str, Any]]) -> float:
    """Extract filled amount from common ccxt/exchange order payload fields."""
    if not isinstance(order, dict):
        return np.nan
    info = order.get("info") if isinstance(order.get("info"), dict) else {}
    for value in (
        order.get("filled"),
        info.get("filled"),
        info.get("filledSize"),
        info.get("filled_size"),
    ):
        amount = _safe_float(value, default=np.nan)
        if np.isfinite(amount):
            return float(amount)
    return np.nan


def _position_absent_reconciliation_mode(
    state: Optional[Dict[str, Any]],
    *,
    stop_order: Optional[Dict[str, Any]] = None,
) -> str:
    """Classify why a locally tracked perp position disappeared from exchange state."""
    state = state if isinstance(state, dict) else {}
    explicit_values = (
        state.get("exit_reason"),
        state.get("closed_via"),
        state.get("reconciliation_mode"),
        state.get("close_mode"),
    )
    if any("liquidat" in str(value or "").lower() for value in explicit_values):
        return "liquidated"

    order_statuses: List[str] = []
    if isinstance(stop_order, dict):
        order_statuses.append(str(stop_order.get("status") or "").lower())
        info = stop_order.get("info") if isinstance(stop_order.get("info"), dict) else {}
        order_statuses.append(str(info.get("status") or "").lower())
    order_statuses.append(str(state.get("last_order_status") or "").lower())
    order_statuses = [status for status in order_statuses if status]
    filled = _order_filled_amount(stop_order)
    if not np.isfinite(filled):
        filled = _safe_float(state.get("last_order_filled"), default=np.nan)
    terminal_stop_fill = any(status in {"closed", "filled"} for status in order_statuses)
    active_stop = any(status in {"open", "new", "untouched"} for status in order_statuses)
    if active_stop and not terminal_stop_fill and (not np.isfinite(filled) or filled <= 0.0):
        return "suspected_liquidation"
    return "exchange_position_absent"


def _format_trade_recap(events: Any) -> str:
    """Return a readable one-line-per-event recap for email/reporting."""
    if not isinstance(events, list) or not events:
        return ""
    lines: List[str] = []
    for event in events[-120:]:
        if not isinstance(event, dict):
            continue
        ts = event.get("ts", "")
        name = event.get("event", "event")
        rest = " ".join(
            f"{k}={v}"
            for k, v in event.items()
            if k not in {"ts", "event"} and v is not None
        )
        lines.append(f"{ts} {name} {rest}".strip())
    return "\n".join(lines)


def _classify_exchange_error(exc: Exception) -> str:
    """Return a stable exchange error category for logs and risk gates."""
    text = _exchange_error_text(exc)
    if _is_trigger_price_reject(exc):
        return "trigger_price_rejected"
    if "429" in text or "rate limit" in text or "too many requests" in text:
        return "rate_limited"
    if "authentication" in text or "unauthorized" in text or "permission" in text:
        return "auth_or_permission"
    if "max pledged collateral" in text or "max transfer in quantity is 0" in text:
        return "asset_collateral_limit"
    if (
        "maximum borrow amount" in text
        or "exceed maximum borrow" in text
        or "max borrow" in text
        or "maxborrowable" in text
    ):
        return "borrow_limit"
    if "insufficient" in text or "balance" in text or "margin" in text:
        return "insufficient_balance"
    if "not supported" in text or "not_supported" in text:
        return "unsupported_exchange_method"
    if (
        "precision" in text
        or "lot_size" in text
        or "min_notional" in text
        or "notional below" in text
        or "exchange minimum" in text
    ):
        return "invalid_precision_or_filter"
    if "halt" in text or "suspend" in text or "inactive" in text or "closed" in text:
        return "symbol_halted"
    if "unsupported target asset" in text:
        return "unsupported_liability_target"
    if "duplicate" in text or "clientorderid" in text or "client order id" in text:
        return "duplicate_client_order_id"
    if "timeout" in text or "timed out" in text or "network" in text:
        return "network_timeout"
    if "cancel" in text:
        return "cancel_failed"
    if "reject" in text or "invalidorder" in text:
        return "order_rejected"
    return "exchange_error"


def _load_market(exchange: Any, symbol: str) -> Dict[str, Any]:
    """Load a ccxt market definition if available."""
    if exchange is None:
        return {}
    try:
        if hasattr(exchange, "load_markets") and not getattr(exchange, "markets", None):
            exchange.load_markets()
    except Exception as exc:
        tprint(f"Warning: could not load markets before trading {symbol}: {exc}")
    try:
        if hasattr(exchange, "market"):
            market = exchange.market(symbol)
            if isinstance(market, dict):
                return market
    except Exception:
        pass
    markets = getattr(exchange, "markets", None)
    if isinstance(markets, dict):
        market = markets.get(symbol, {})
        if isinstance(market, dict):
            return market
    return {}


def _market_is_active(market: Dict[str, Any]) -> bool:
    """Return whether a market definition looks tradeable."""
    if not market:
        return True
    if market.get("active") is False:
        return False
    info = market.get("info", {})
    if isinstance(info, dict):
        status = str(info.get("status", "")).upper()
        if status and status not in {"TRADING", "ENABLED"}:
            return False
    return True


def _exchange_precision(
    exchange: Any,
    symbol: str,
    value: float,
    *,
    kind: str,
) -> float:
    """Apply exchange amount or price precision when supported."""
    if not np.isfinite(value):
        return value
    method_name = f"{kind}_to_precision"
    method = getattr(exchange, method_name, None)
    if callable(method):
        try:
            return float(method(symbol, value))
        except Exception:
            return float(value)
    return float(value)


def _validate_order_filters(
    symbol: str,
    market: Dict[str, Any],
    *,
    amount: float,
    price: float,
) -> None:
    """Raise if amount/cost violates exchange market limits."""
    if not _market_is_active(market):
        raise ValueError(f"symbol halted or inactive: {symbol}")
    if not np.isfinite(amount) or amount <= 0:
        raise ValueError(f"invalid base amount for {symbol}: {amount}")
    if not np.isfinite(price) or price <= 0:
        raise ValueError(f"invalid reference price for {symbol}: {price}")
    limits = market.get("limits", {}) if isinstance(market, dict) else {}
    amount_limits = limits.get("amount", {}) if isinstance(limits, dict) else {}
    cost_limits = limits.get("cost", {}) if isinstance(limits, dict) else {}
    min_amount = _safe_float(amount_limits.get("min"), default=np.nan)
    max_amount = _safe_float(amount_limits.get("max"), default=np.nan)
    min_cost = _safe_float(cost_limits.get("min"), default=np.nan)
    max_cost = _safe_float(cost_limits.get("max"), default=np.nan)
    cost = amount * price
    if np.isfinite(min_amount) and amount < min_amount:
        raise ValueError(
            f"amount below exchange minimum for {symbol}: {amount} < {min_amount}"
        )
    if np.isfinite(max_amount) and amount > max_amount:
        raise ValueError(
            f"amount above exchange maximum for {symbol}: {amount} > {max_amount}"
        )
    if np.isfinite(min_cost) and cost < min_cost:
        raise ValueError(
            f"notional below exchange minimum for {symbol}: {cost} < {min_cost}"
        )
    if np.isfinite(max_cost) and cost > max_cost:
        raise ValueError(
            f"notional above exchange maximum for {symbol}: {cost} > {max_cost}"
        )


def _market_min_notional(market: Dict[str, Any]) -> float:
    """Return exchange minimum notional when the market definition exposes it."""
    limits = market.get("limits", {}) if isinstance(market, dict) else {}
    cost_limits = limits.get("cost", {}) if isinstance(limits, dict) else {}
    return _safe_float(cost_limits.get("min"), default=np.nan)


def _extract_order_fill(
    order: Dict[str, Any], fallback_price: float
) -> Tuple[float, float, bool]:
    """Return average price, filled amount, and partial-fill flag."""
    avg = _safe_float(order.get("average"), default=np.nan)
    price = _safe_float(order.get("price"), default=np.nan)
    filled = _safe_float(order.get("filled"), default=np.nan)
    amount = _safe_float(order.get("amount"), default=np.nan)
    realized_price = avg if np.isfinite(avg) and avg > 0 else price
    if not np.isfinite(realized_price) or realized_price <= 0:
        realized_price = fallback_price
    partial = bool(
        np.isfinite(filled) and np.isfinite(amount) and filled > 0 and filled < amount
    )
    return float(realized_price), float(filled), partial


def _filled_base_fee(order: Dict[str, Any], symbol: str) -> float:
    """Return base-asset fees charged by the entry fill."""
    base_asset = str(symbol).split("/", 1)[0].upper()
    fees = order.get("fees")
    if not isinstance(fees, list) or not fees:
        fee = order.get("fee")
        fees = [fee] if isinstance(fee, dict) else []
    total = 0.0
    for fee in fees:
        if not isinstance(fee, dict):
            continue
        if str(fee.get("currency", "")).upper() != base_asset:
            continue
        total += _safe_float(fee.get("cost"), default=0.0)
    return float(max(total, 0.0))


def _non_stop_bucket_fields(source: Any) -> Dict[str, Any]:
    """Return only compatibility bucket fields that cannot drive stop policy."""
    if not isinstance(source, dict):
        return {}
    return {k: source[k] for k in NON_STOP_COMPAT_BUCKET_KEYS if k in source}


def _canonical_stop_fields_from_oco_result(oco_result: Any) -> Dict[str, Any]:
    """Return canonical stop fields safe for mirror position state."""
    if not isinstance(oco_result, dict):
        return {}
    aliases = {
        "stop_policy_params_source": "params_source",
        "stop_policy_params_hash": "params_hash",
        "stop_policy_schema": "params_schema",
    }
    out: Dict[str, Any] = {}
    for key in CANONICAL_STOP_POSITION_FIELDS:
        value = oco_result.get(key)
        if value is None and key in aliases:
            value = oco_result.get(aliases[key])
        if value is not None:
            out[key] = value
    return out


def _execution_account(config: Optional[Dict[str, Any]]) -> str:
    """Return configured execution account type for exchange routing."""
    cfg = config or {}
    raw = str(
        cfg.get(
            "execution_account",
            cfg.get("account_type", cfg.get("market_mode", "margin")),
        )
        or "margin"
    ).lower()
    if raw in {"perp", "perps", "future", "futures", "swap"}:
        return "perps"
    if raw in {"margin", "cross_margin", "isolated_margin"}:
        return "margin"
    return "spot"


def _exchange_id(exchange: Any) -> str:
    return str(getattr(exchange, "id", "") or "").strip().lower()


def _stop_trigger_reference_price(
    exchange: Any,
    ticker: Dict[str, Any],
    config: Optional[Dict[str, Any]],
    *,
    position_side: Optional[str] = None,
) -> Tuple[float, str]:
    """Return the executable-side live price used by software stop risk checks."""
    ticker = ticker if isinstance(ticker, dict) else {}
    if (
        _execution_account(config) == "perps"
        and _exchange_id(exchange) == "krakenfutures"
    ):
        side = str(position_side or "").strip().lower()
        trigger_key = "bid" if side == "long" else "ask" if side == "short" else ""
        if trigger_key:
            value = _safe_float(ticker.get(trigger_key), default=np.nan)
            if np.isfinite(value) and value > 0.0:
                return float(value), trigger_key
    for key in ("last", "close"):
        value = _safe_float(ticker.get(key), default=np.nan)
        if np.isfinite(value) and value > 0.0:
            return float(value), key
    bid = _safe_float(ticker.get("bid"), default=np.nan)
    ask = _safe_float(ticker.get("ask"), default=np.nan)
    if np.isfinite(bid) and bid > 0.0 and np.isfinite(ask) and ask > 0.0:
        return float((bid + ask) / 2.0), "bid_ask_mid"
    return np.nan, "unavailable"


def _kraken_futures_last_stop_from_executable_stop(
    ticker: Dict[str, Any],
    config: Optional[Dict[str, Any]],
    *,
    position_side: str,
    policy_stop_price: float,
) -> Tuple[float, Dict[str, Any]]:
    """Translate executable bid/ask stop into a conservative last-trigger stop."""
    policy_stop = _safe_float(policy_stop_price, default=np.nan)
    if not np.isfinite(policy_stop) or policy_stop <= 0.0:
        return np.nan, {"status": "invalid_policy_stop"}

    ticker = ticker if isinstance(ticker, dict) else {}
    bid = _safe_float(ticker.get("bid"), default=np.nan)
    ask = _safe_float(ticker.get("ask"), default=np.nan)
    last = _safe_float(ticker.get("last") or ticker.get("close"), default=np.nan)
    spread = (
        float(ask - bid)
        if np.isfinite(bid) and bid > 0.0 and np.isfinite(ask) and ask > bid
        else np.nan
    )
    synthetic_last = False
    if not (np.isfinite(last) and last > 0.0):
        if np.isfinite(spread):
            last = float((bid + ask) / 2.0)
            synthetic_last = True

    side = str(position_side or "").strip().lower()
    gap = np.nan
    gap_source = "unavailable"
    if side == "short":
        if np.isfinite(ask) and ask > 0.0 and np.isfinite(last) and last > 0.0:
            gap = max(float(ask - last), 0.0)
            gap_source = "ask_minus_last"
        elif np.isfinite(spread):
            gap = float(spread / 2.0)
            gap_source = "half_spread"
        else:
            gap = 0.0
            gap_source = "zero_no_spread"
        exchange_stop = float(policy_stop - gap)
    elif side == "long":
        if np.isfinite(bid) and bid > 0.0 and np.isfinite(last) and last > 0.0:
            gap = max(float(last - bid), 0.0)
            gap_source = "last_minus_bid"
        elif np.isfinite(spread):
            gap = float(spread / 2.0)
            gap_source = "half_spread"
        else:
            gap = 0.0
            gap_source = "zero_no_spread"
        exchange_stop = float(policy_stop + gap)
    else:
        exchange_stop = float(policy_stop)
        gap = 0.0
        gap_source = "unknown_side"

    meta = {
        "status": "ok",
        "policy_stop_price": float(policy_stop),
        "exchange_stop_price": float(exchange_stop),
        "position_side": side,
        "bid": float(bid) if np.isfinite(bid) else None,
        "ask": float(ask) if np.isfinite(ask) else None,
        "last": float(last) if np.isfinite(last) else None,
        "last_synthetic_mid": bool(synthetic_last),
        "spread": float(spread) if np.isfinite(spread) else None,
        "last_to_executable_gap": float(gap) if np.isfinite(gap) else None,
        "gap_source": gap_source,
        "trigger_signal": "last",
    }
    return exchange_stop, meta


def _configured_exchange_id(config: Optional[Dict[str, Any]] = None) -> str:
    cfg = config or {}
    raw = (
        cfg.get("exchange")
        or cfg.get("exchange_id")
        or cfg.get("ccxt_exchange")
        or os.environ.get("EPM_EXCHANGE")
        or os.environ.get("EXCHANGE_NAME")
        or os.environ.get("PRIMARY_EXCHANGE")
        or "binance"
    )
    exchange_id = str(raw or "binance").strip().lower()
    if exchange_id in {"okx", "okex"}:
        return "okx"
    if exchange_id in {"kraken", "krakenfutures", "kraken_futures"}:
        return "kraken"
    return "binance"


def _exchange_symbol_for_config(
    exchange: Any, config: Optional[Dict[str, Any]], symbol: str
) -> str:
    if _execution_account(config) != "perps" or ":" in str(symbol):
        return symbol
    return _resolve_perp_symbol(exchange, symbol) or symbol


def _margin_mode(config: Optional[Dict[str, Any]]) -> str:
    """Return configured margin mode for ccxt order params."""
    cfg = config or {}
    raw = str(cfg.get("margin_mode", "cross") or "cross").lower()
    return "isolated" if raw == "isolated" else "cross"


def _margin_side_effect(config: Optional[Dict[str, Any]]) -> str:
    """Return Binance margin sideEffectType for opening orders."""
    cfg = config or {}
    raw = str(cfg.get("margin_side_effect_type", "AUTO_BORROW_REPAY") or "")
    allowed = {"NO_SIDE_EFFECT", "MARGIN_BUY", "AUTO_REPAY", "AUTO_BORROW_REPAY"}
    return raw if raw in allowed else "AUTO_BORROW_REPAY"


def _is_one_x_margin_policy(config: Optional[Dict[str, Any]]) -> bool:
    """Return True when live sizing is configured not to use wallet leverage."""
    cfg = config or {}
    try:
        book_multiplier = float(cfg.get("book_notional_multiplier", 1.0))
    except Exception:
        book_multiplier = 1.0
    try:
        leverage_multiplier = float(cfg.get("leverage_wallet_multiplier", 1.0))
    except Exception:
        leverage_multiplier = 1.0
    return book_multiplier <= 1.000001 and leverage_multiplier <= 1.000001


def _opening_margin_side_effect(
    config: Optional[Dict[str, Any]],
    *,
    side: Optional[str] = None,
) -> str:
    """Return Binance sideEffectType for opening margin orders.

    At 1x, long entries should spend available quote balance rather than invoke
    Binance borrow/pledge mechanics. Shorts still need AUTO_BORROW_REPAY unless
    the account already holds the base asset, which is not the normal policy.
    """
    side_l = str(side or "").lower()
    if side_l == "buy" and _is_one_x_margin_policy(config):
        return "NO_SIDE_EFFECT"
    return _margin_side_effect(config)


def _order_params(
    config: Optional[Dict[str, Any]],
    *,
    reduce_only: bool = False,
    side: Optional[str] = None,
    leverage: Optional[float] = None,
) -> Dict[str, Any]:
    """Build ccxt params for spot or margin order placement."""
    account = _execution_account(config)
    if account == "perps":
        params: Dict[str, Any] = {}
        if _configured_exchange_id(config) == "okx":
            params["tdMode"] = _margin_mode(config)
        if not reduce_only and _configured_exchange_id(config) == "krakenfutures":
            lev = _safe_float(leverage, default=np.nan)
            if np.isfinite(lev) and lev > 1.0:
                params["leverage"] = float(lev)
        if reduce_only:
            params["reduceOnly"] = True
        return params
    if account != "margin":
        return {}
    params: Dict[str, Any] = {"marginMode": _margin_mode(config)}
    if reduce_only:
        params["sideEffectType"] = "AUTO_REPAY"
    else:
        params["sideEffectType"] = _opening_margin_side_effect(config, side=side)
    return params


def _perp_entry_leverage_from_context(
    trade_context: Optional[Dict[str, Any]],
    config: Optional[Dict[str, Any]],
) -> float:
    """Return the intended perps entry leverage from sizing/audit context."""
    ctx = trade_context or {}
    for key in ("perp_effective_leverage", "leverage_wallet_multiplier"):
        lev = _safe_float(ctx.get(key), default=np.nan)
        if np.isfinite(lev) and lev > 0.0:
            return float(lev)
    lev = _safe_float((config or {}).get("leverage_wallet_multiplier"), default=np.nan)
    return float(lev) if np.isfinite(lev) and lev > 0.0 else 1.0


def _set_perp_leverage_best_effort(
    exchange: Any,
    *,
    symbol: str,
    leverage: float,
    config: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Set exchange leverage where ccxt supports it, without blocking implicit-margin venues."""
    if _execution_account(config) != "perps":
        return {"attempted": False, "reason": "not_perps"}
    lev = _safe_float(leverage, default=np.nan)
    if not np.isfinite(lev) or lev <= 1.0:
        return {"attempted": False, "reason": "leverage_lte_1", "leverage": lev}
    set_leverage = getattr(exchange, "set_leverage", None)
    if not callable(set_leverage):
        return {"attempted": False, "reason": "set_leverage_unavailable", "leverage": lev}
    params: Dict[str, Any] = {}
    if _configured_exchange_id(config) == "okx":
        params["mgnMode"] = _margin_mode(config)
    try:
        result = set_leverage(float(lev), symbol, params)
        return {
            "attempted": True,
            "success": True,
            "leverage": float(lev),
            "result": result,
        }
    except Exception as exc:
        return {
            "attempted": True,
            "success": False,
            "leverage": float(lev),
            "error": f"{type(exc).__name__}: {exc}",
        }


def _cancel_params(config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Build ccxt params for spot or margin order cancellation."""
    if (
        _execution_account(config) == "perps"
        and _configured_exchange_id(config) == "okx"
    ):
        return {"tdMode": _margin_mode(config)}
    if _execution_account(config) != "margin":
        return {}
    return {"marginMode": _margin_mode(config)}


def _dedupe_param_variants(variants: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return order-preserving unique exchange param dictionaries."""
    seen = set()
    out: List[Dict[str, Any]] = []
    for params in variants:
        clean = {k: v for k, v in dict(params or {}).items() if v is not None}
        key = tuple(sorted(clean.items()))
        if key in seen:
            continue
        seen.add(key)
        out.append(clean)
    return out


def _margin_fetch_param_variants(
    config: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Return robust ccxt param variants for Binance margin order reads."""
    base = _cancel_params(config)
    if _execution_account(config) != "margin":
        return [base]
    margin_mode = _margin_mode(config)
    return _dedupe_param_variants(
        [
            base,
            {**base, "type": "margin"},
            {"type": "margin", "marginMode": margin_mode},
            {"type": "margin"},
            {},
        ]
    )


def _reduce_order_param_variants(
    config: Optional[Dict[str, Any]],
    *,
    side: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Return robust ccxt param variants for margin reduce/close orders."""
    base = _order_params(config, reduce_only=True)
    if _execution_account(config) == "perps":
        return _dedupe_param_variants([base, _cancel_params(config), {}])
    if _execution_account(config) != "margin":
        return [base]
    margin_mode = _margin_mode(config)
    variants: List[Dict[str, Any]] = []
    side_effects = ["AUTO_REPAY", "NO_SIDE_EFFECT"]
    if str(side or "").lower() == "buy":
        side_effects.append("MARGIN_BUY")
    side_effects.append("AUTO_BORROW_REPAY")
    for side_effect in side_effects:
        params = {"marginMode": margin_mode, "sideEffectType": side_effect}
        variants.append(params)
        variants.append({**params, "type": "margin"})
    variants.append(base)
    variants.extend(
        [
            {"marginMode": margin_mode},
            {"type": "margin", "marginMode": margin_mode},
        ]
    )
    return _dedupe_param_variants(variants)


def _create_reduce_stop_loss_order(
    exchange: Any,
    *,
    symbol: str,
    side: str,
    amount: float,
    stop_price: float,
    config: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Create a reduce/close STOP_LOSS order, retrying safe margin param variants."""
    if _execution_account(config) == "perps":
        if _exchange_id(exchange) == "okx":
            params = {
                "reduceOnly": True,
                "tdMode": _margin_mode(config),
            }
            create_stop_loss = getattr(exchange, "create_stop_loss_order", None)
            if callable(create_stop_loss):
                return create_stop_loss(
                    symbol=symbol,
                    type="market",
                    side=side,
                    amount=amount,
                    price=None,
                    stopLossPrice=stop_price,
                    params=params,
                )
            return exchange.create_order(
                symbol=symbol,
                type="market",
                side=side,
                amount=amount,
                price=None,
                params={
                    **params,
                    "stopLossPrice": stop_price,
                    "slTriggerPx": stop_price,
                    "slOrdPx": "-1",
                },
            )
        if _exchange_id(exchange) == "krakenfutures":
            native_order = _create_kraken_futures_native_reduce_stop_loss_order(
                exchange,
                symbol=symbol,
                side=side,
                amount=amount,
                stop_price=stop_price,
                config=config,
            )
            if native_order is not None:
                return native_order
            trigger_signal = _kraken_futures_stop_trigger_signal_for_reduce_side(side)
            return exchange.create_order(
                symbol=symbol,
                type="market",
                side=side,
                amount=amount,
                price=None,
                params={
                    "stopLossPrice": stop_price,
                    "reduceOnly": True,
                    "triggerSignal": trigger_signal,
                },
            )
        return exchange.create_order(
            symbol=symbol,
            type="STOP_MARKET",
            side=side,
            amount=amount,
            price=None,
            params={"stopPrice": stop_price, "reduceOnly": True},
        )
    last_exc: Optional[Exception] = None
    for params in _reduce_order_param_variants(config, side=side):
        try:
            return exchange.create_order(
                symbol=symbol,
                type="STOP_LOSS",
                side=side,
                amount=amount,
                price=stop_price,
                params={**params, "stopPrice": stop_price},
            )
        except Exception as exc:
            last_exc = exc
            category = _classify_exchange_error(exc)
            if category not in {
                "insufficient_balance",
                "auth_or_permission",
                "invalid_precision_or_filter",
            }:
                raise
    if last_exc is not None:
        raise last_exc
    raise RuntimeError(f"STOP_LOSS creation did not run for {symbol}")


def _create_kraken_futures_native_reduce_stop_loss_order(
    exchange: Any,
    *,
    symbol: str,
    side: str,
    amount: float,
    stop_price: float,
    config: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Submit a Kraken Futures reduce stop through the native /sendorder method."""
    cfg = config or {}
    if not bool(cfg.get("kraken_futures_native_stop_orders", True)):
        return None
    send_order = getattr(exchange, "privatePostSendorder", None) or getattr(
        exchange, "private_post_sendorder", None
    )
    if not callable(send_order):
        return None
    market = _load_market(exchange, symbol)
    market_id = str(market.get("id") or symbol)
    reduce_side = str(side or "").strip().lower()
    trigger_signal = _kraken_futures_stop_trigger_signal_for_reduce_side(reduce_side)
    amount_s = str(amount)
    price_s = str(stop_price)
    amount_to_precision = getattr(exchange, "amount_to_precision", None)
    price_to_precision = getattr(exchange, "price_to_precision", None)
    if callable(amount_to_precision):
        try:
            amount_s = str(amount_to_precision(symbol, amount))
        except Exception:
            amount_s = str(amount)
    if callable(price_to_precision):
        try:
            price_s = str(price_to_precision(symbol, stop_price))
        except Exception:
            price_s = str(stop_price)
    request = {
        "symbol": market_id,
        "side": reduce_side,
        "size": amount_s,
        "orderType": "stp",
        "stopPrice": price_s,
        "reduceOnly": True,
        "triggerSignal": trigger_signal,
    }
    response = send_order(request)
    info = response if isinstance(response, dict) else {"response": response}
    send_status = info.get("sendStatus") if isinstance(info.get("sendStatus"), dict) else {}
    order_id = (
        info.get("order_id")
        or info.get("orderId")
        or info.get("id")
        or send_status.get("order_id")
        or send_status.get("orderId")
    )
    status = str(info.get("status") or info.get("result") or "open").lower()
    return {
        "id": order_id,
        "symbol": symbol,
        "type": "stop",
        "side": reduce_side,
        "amount": float(amount),
        "price": None,
        "stopPrice": float(stop_price),
        "status": status,
        "reduceOnly": True,
        "triggerSignal": trigger_signal,
        "info": info,
    }


def _create_reduce_market_order(
    exchange: Any,
    *,
    symbol: str,
    side: str,
    amount: float,
    price: Optional[float] = None,
    config: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Create a reduce/close market order, retrying safe margin param variants."""
    last_exc: Optional[Exception] = None
    order_price = float(price) if price is not None and np.isfinite(price) else None
    for params in _reduce_order_param_variants(config, side=side):
        try:
            return exchange.create_order(
                symbol=symbol,
                type="market",
                side=side,
                amount=amount,
                price=order_price,
                params=params,
            )
        except Exception as exc:
            last_exc = exc
            category = _classify_exchange_error(exc)
            if category not in {
                "insufficient_balance",
                "auth_or_permission",
                "invalid_precision_or_filter",
            }:
                raise
    if last_exc is not None:
        raise last_exc
    raise RuntimeError(f"Market reduce order creation did not run for {symbol}")


def _create_margin_market_order_variants(
    exchange: Any,
    *,
    symbol: str,
    side: str,
    amount: float,
    price: Optional[float],
    side_effects: Sequence[str],
    config: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Create a margin market order using explicit sideEffectType variants."""
    if exchange is None:
        raise RuntimeError("exchange is required")
    margin_mode = _margin_mode(config)
    order_price = float(price) if price is not None and np.isfinite(price) else None
    variants: List[Dict[str, Any]] = []
    for side_effect in side_effects:
        params = {"marginMode": margin_mode, "sideEffectType": str(side_effect)}
        variants.append(params)
        variants.append({**params, "type": "margin"})
    variants.extend(
        [
            {"marginMode": margin_mode},
            {"type": "margin", "marginMode": margin_mode},
        ]
    )
    last_exc: Optional[Exception] = None
    for params in _dedupe_param_variants(variants):
        try:
            return exchange.create_order(
                symbol=symbol,
                type="market",
                side=side,
                amount=amount,
                price=order_price,
                params=params,
            )
        except Exception as exc:
            last_exc = exc
            category = _classify_exchange_error(exc)
            if category not in {
                "insufficient_balance",
                "auth_or_permission",
                "invalid_precision_or_filter",
                "platform_collateral_limit",
            }:
                raise
    if last_exc is not None:
        raise last_exc
    raise RuntimeError(f"Margin market order creation did not run for {symbol}")


def _order_stop_price(order: Dict[str, Any]) -> float:
    """Extract stop price from a ccxt exchange order payload."""
    info = order.get("info") if isinstance(order.get("info"), dict) else {}
    return _safe_float(
        order.get("stopPrice")
        or order.get("stop_price")
        or order.get("stopLossPrice")
        or info.get("stopPrice")
        or info.get("stopLossPrice")
        or info.get("slTriggerPx")
        or info.get("slOrdPx")
        or info.get("triggerPx")
        or info.get("triggerPrice")
        or info.get("activatePrice")
        or order.get("triggerPrice")
        or order.get("price"),
        default=np.nan,
    )


def _stop_is_at_least_as_protective(
    position_side: str,
    existing_stop: float,
    candidate_stop: float,
    *,
    rel_tol: float = 1e-4,
) -> bool:
    """Return whether an existing stop is no looser than the candidate stop."""
    existing = _safe_float(existing_stop, np.nan)
    candidate = _safe_float(candidate_stop, np.nan)
    if not (
        np.isfinite(existing)
        and existing > 0.0
        and np.isfinite(candidate)
        and candidate > 0.0
    ):
        return False
    tol = max(abs(candidate) * float(rel_tol), 1e-8)
    side = str(position_side or "").strip().lower()
    if side == "short":
        return existing <= candidate + tol
    if side == "long":
        return existing >= candidate - tol
    return abs(existing - candidate) <= tol


def _order_trigger_signal(order: Dict[str, Any]) -> str:
    """Extract the exchange stop trigger source when present."""
    info = order.get("info") if isinstance(order.get("info"), dict) else {}
    raw = (
        order.get("triggerSignal")
        or order.get("trigger_signal")
        or order.get("triggerSource")
        or order.get("trigger_source")
        or info.get("triggerSignal")
        or info.get("trigger_signal")
        or info.get("triggerSource")
        or info.get("trigger_source")
    )
    return str(raw or "").strip().lower()


def _verify_exchange_stop_trigger_signal(
    exchange: Any,
    *,
    symbol: str,
    order_id: Any,
    order: Dict[str, Any],
    config: Optional[Dict[str, Any]],
) -> Tuple[str, Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Return the exchange-reported stop trigger, fetching the order when possible."""
    actual = _order_trigger_signal(order)
    if order_id in (None, ""):
        return actual, None, None
    try:
        fetched, meta = _fetch_order_with_list_fallback(
            exchange,
            order_id=order_id,
            symbol=symbol,
            config=config,
        )
        fetched_signal = _order_trigger_signal(fetched)
        if fetched_signal:
            return fetched_signal, fetched, meta
        return actual, fetched, meta
    except Exception as exc:
        return actual, None, {
            "trigger_verify_error_category": _classify_exchange_error(exc),
            "trigger_verify_error": str(exc),
        }


def _kraken_futures_stop_trigger_signal_for_reduce_side(side: str) -> str:
    """Return the Kraken-hosted trigger for a Futures reduce stop."""
    return "last"


def _kraken_futures_stop_trigger_signal_for_position_side(side: str) -> str:
    """Return the Kraken-hosted trigger for a Futures position stop."""
    return "last"


def _protective_stop_trigger_matches_policy(
    exchange: Any,
    order: Dict[str, Any],
    config: Optional[Dict[str, Any]],
    *,
    position_side: Optional[str] = None,
) -> bool:
    """Return whether an existing protective stop uses the configured trigger."""
    if (
        _execution_account(config) != "perps"
        or _exchange_id(exchange) != "krakenfutures"
    ):
        return True
    expected = _kraken_futures_stop_trigger_signal_for_position_side(
        str(position_side or "")
    )
    if expected == "last":
        side = str(order.get("side") or "").lower()
        expected = _kraken_futures_stop_trigger_signal_for_reduce_side(side)
    return _order_trigger_signal(order) == expected


def _is_stop_loss_order(order: Dict[str, Any]) -> bool:
    """Return whether an exchange order payload looks like a protective stop."""
    if not isinstance(order, dict):
        return False
    info = order.get("info") if isinstance(order.get("info"), dict) else {}
    tokens = " ".join(
        str(v or "").upper()
        for v in (
            order.get("type"),
            order.get("orderType"),
            order.get("origType"),
            info.get("type"),
            info.get("orderType"),
            info.get("origType"),
            info.get("ordType"),
            info.get("algoOrdType"),
        )
    )
    if "STOP" in tokens:
        return True
    return np.isfinite(_order_stop_price(order))


def _order_identity_values(order: Dict[str, Any]) -> List[str]:
    """Return exchange/client ids that can identify an order across ccxt shapes."""
    if not isinstance(order, dict):
        return []
    info = order.get("info") if isinstance(order.get("info"), dict) else {}
    values = [
        order.get("id"),
        order.get("clientOrderId"),
        order.get("client_order_id"),
        order.get("orderId"),
        info.get("id"),
        info.get("orderId"),
        info.get("order_id"),
        info.get("cliOrdId"),
        info.get("clientOrderId"),
        info.get("client_order_id"),
    ]
    out: List[str] = []
    for value in values:
        if value in (None, ""):
            continue
        out.append(str(value))
    return out


def _order_matches_id(order: Dict[str, Any], order_id: Any) -> bool:
    """Return whether an exchange order payload matches the tracked order id."""
    if order_id in (None, ""):
        return False
    return str(order_id) in set(_order_identity_values(order))


def _order_remaining_amount(order: Dict[str, Any]) -> float:
    """Extract remaining/open order amount from common ccxt/exchange fields."""
    if not isinstance(order, dict):
        return np.nan
    info = order.get("info") if isinstance(order.get("info"), dict) else {}
    for value in (
        order.get("remaining"),
        info.get("unfilledSize"),
        info.get("unfilled_size"),
        order.get("amount"),
        info.get("size"),
        info.get("amount"),
    ):
        amount = _safe_float(value, default=np.nan)
        if np.isfinite(amount) and amount > 0.0:
            return float(amount)
    return np.nan


def _fetch_order_from_open_closed_lists(
    exchange: Any,
    *,
    order_id: Any,
    symbol: str,
    config: Optional[Dict[str, Any]],
) -> Optional[Tuple[Dict[str, Any], str]]:
    """Locate an order on exchanges that do not support fetch_order/fetch_orders."""
    for method_name, default_status in (
        ("fetch_open_orders", "open"),
        ("fetch_closed_orders", "closed"),
    ):
        fetch_orders = getattr(exchange, method_name, None)
        if not callable(fetch_orders):
            continue
        for fetch_params in _margin_fetch_param_variants(config):
            try:
                orders = fetch_orders(symbol, None, 50, fetch_params) or []
            except Exception:
                continue
            for candidate in reversed(list(orders)):
                if not isinstance(candidate, dict):
                    continue
                if not _order_matches_id(candidate, order_id):
                    continue
                order = dict(candidate)
                if not order.get("status"):
                    order["status"] = default_status
                return order, method_name
    return None


def _fetch_order_with_list_fallback(
    exchange: Any,
    *,
    order_id: Any,
    symbol: str,
    config: Optional[Dict[str, Any]],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Fetch an order, falling back to open/closed order lists for Kraken Futures."""
    fetch_order = getattr(exchange, "fetch_order", None)
    original_exc: Optional[Exception] = None
    if callable(fetch_order):
        try:
            return fetch_order(order_id, symbol, _cancel_params(config)), {}
        except Exception as exc:
            original_exc = exc

    listed = _fetch_order_from_open_closed_lists(
        exchange,
        order_id=order_id,
        symbol=symbol,
        config=config,
    )
    if listed is not None:
        order, method_name = listed
        meta = {"resolved_via": method_name}
        if original_exc is not None:
            meta.update(
                {
                    "reconciled_after_error": True,
                    "fetch_order_error_category": _classify_exchange_error(
                        original_exc
                    ),
                    "fetch_order_error": str(original_exc),
                }
            )
        else:
            meta["fetch_order_unavailable"] = True
        return order, meta

    if original_exc is not None:
        raise original_exc
    raise RuntimeError("fetch_order_unavailable")


def _position_info(position: Dict[str, Any]) -> Dict[str, Any]:
    return position.get("info") if isinstance(position.get("info"), dict) else {}


def _position_symbol(position: Dict[str, Any], exchange: Any) -> str:
    """Extract a ccxt unified symbol from a position payload."""
    symbol = str(position.get("symbol") or "").strip()
    if symbol:
        return symbol
    info = _position_info(position)
    market_id = str(
        info.get("symbol")
        or info.get("instrument")
        or info.get("instrument_name")
        or info.get("market")
        or ""
    ).strip()
    if market_id and hasattr(exchange, "markets_by_id"):
        markets_by_id = getattr(exchange, "markets_by_id", {}) or {}
        market = markets_by_id.get(market_id)
        if isinstance(market, list) and market:
            market = market[0]
        if isinstance(market, dict) and market.get("symbol"):
            return str(market["symbol"])
    return market_id


def _position_contracts(position: Dict[str, Any]) -> float:
    """Return signed position contracts/amount when the exchange exposes it."""
    info = _position_info(position)
    for key in (
        "contracts",
        "positionAmt",
        "size",
        "currentQty",
        "qty",
        "amount",
    ):
        value = position.get(key)
        if value in (None, ""):
            value = info.get(key)
        amount = _safe_float(value, default=np.nan)
        if np.isfinite(amount) and amount != 0.0:
            return float(amount)
    return 0.0


def _position_side(position: Dict[str, Any], contracts: float) -> str:
    """Return long/short from ccxt position side or signed contracts."""
    info = _position_info(position)
    raw_side = str(
        position.get("side")
        or info.get("side")
        or info.get("positionSide")
        or info.get("type")
        or ""
    ).lower()
    if raw_side in {"long", "buy"}:
        return "long"
    if raw_side in {"short", "sell"}:
        return "short"
    if contracts < 0.0:
        return "short"
    return "long"


def _position_entry_price(position: Dict[str, Any]) -> float:
    """Extract an entry price from common ccxt/exchange position fields."""
    info = _position_info(position)
    for key in (
        "entryPrice",
        "entry_price",
        "avgEntryPrice",
        "average",
        "avgPrice",
        "price",
    ):
        value = position.get(key)
        if value in (None, ""):
            value = info.get(key)
        price = _safe_float(value, default=np.nan)
        if np.isfinite(price) and price > 0.0:
            return float(price)
    return np.nan


def _position_quote_value(
    position: Dict[str, Any], amount: float, entry_price: float
) -> float:
    """Return absolute notional quote value for sizing/reconciliation reports."""
    info = _position_info(position)
    for key in ("notional", "notionalUsd", "cost", "value", "quoteValue"):
        value = position.get(key)
        if value in (None, ""):
            value = info.get(key)
        quote_value = abs(_safe_float(value, default=np.nan))
        if np.isfinite(quote_value) and quote_value > 0.0:
            return float(quote_value)
    contract_size = _safe_float(
        position.get("contractSize") or info.get("contractSize"),
        default=1.0,
    )
    if not np.isfinite(contract_size) or contract_size <= 0.0:
        contract_size = 1.0
    if np.isfinite(entry_price) and entry_price > 0.0:
        return float(abs(amount) * entry_price * contract_size)
    return np.nan


def _fetch_open_exchange_positions(
    exchange: Any,
    config: Optional[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Return currently open exchange positions keyed by unified symbol."""
    fetch_positions = getattr(exchange, "fetch_positions", None)
    if not callable(fetch_positions):
        return {}
    try:
        raw_positions = fetch_positions([], _cancel_params(config))
    except TypeError:
        raw_positions = fetch_positions(_cancel_params(config))
    out: Dict[str, Dict[str, Any]] = {}
    for position in raw_positions or []:
        if not isinstance(position, dict):
            continue
        signed_amount = _position_contracts(position)
        if abs(float(signed_amount)) <= 0.0:
            continue
        symbol = _position_symbol(position, exchange)
        if symbol:
            out[str(symbol)] = position
    return out


def _fetch_open_protective_stop_orders(
    exchange: Any,
    *,
    symbol: str,
    position_side: str,
    config: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Fetch open STOP_LOSS orders that reduce the tracked position side."""
    fetch_open_orders = getattr(exchange, "fetch_open_orders", None)
    if not callable(fetch_open_orders):
        return []
    expected_side = "sell" if str(position_side).lower() == "long" else "buy"
    out: List[Dict[str, Any]] = []
    for fetch_params in _margin_fetch_param_variants(config):
        try:
            open_orders = fetch_open_orders(symbol, None, None, fetch_params) or []
        except Exception:
            continue
        for order in open_orders:
            if not isinstance(order, dict):
                continue
            if str(order.get("side") or "").lower() != expected_side:
                continue
            if not _is_stop_loss_order(order):
                continue
            out.append(order)
        if out:
            break
    return out


def _cancel_open_protective_stop_orders(
    exchange: Any,
    *,
    symbol: str,
    position_side: str,
    config: Optional[Dict[str, Any]],
    keep_order_id: Any = None,
) -> int:
    """Cancel open protective stops to free reserved position quantity."""
    cancel_order = getattr(exchange, "cancel_order", None)
    if not callable(cancel_order):
        return 0
    cancelled = 0
    keep = str(keep_order_id) if keep_order_id not in (None, "") else ""
    for order in _fetch_open_protective_stop_orders(
        exchange, symbol=symbol, position_side=position_side, config=config
    ):
        order_id = order.get("id")
        if not order_id or (keep and str(order_id) == keep):
            continue
        for cancel_params in _margin_fetch_param_variants(config):
            try:
                cancel_order(order_id, symbol, cancel_params)
                cancelled += 1
                break
            except Exception as exc:
                category = _classify_exchange_error(exc)
                if category in {"order_not_found", "already_closed_or_cancelled"}:
                    cancelled += 1
                    break
    return cancelled


def _symbol_from_asset_quote(asset: str, quote: str) -> str:
    """Return the canonical ccxt symbol for an asset/quote pair."""
    return f"{str(asset).upper()}/{str(quote).upper()}"


def _fee_to_quote(
    symbol: str,
    fee: Any,
    *,
    price: float,
) -> Tuple[float, float, str, str]:
    """Return fee as quote currency, original cost/currency, and source status."""
    fee_dict = fee if isinstance(fee, dict) else {}
    if not fee_dict:
        return np.nan, np.nan, "", "missing"
    fee_cost = _safe_float(fee_dict.get("cost"), np.nan)
    fee_currency = str(fee_dict.get("currency") or "").upper()
    if not np.isfinite(fee_cost):
        return np.nan, np.nan, fee_currency, "missing"
    base_asset = str(symbol).split("/", 1)[0].upper()
    quote_asset = (
        str(symbol).split("/", 1)[1].split(":", 1)[0].upper()
        if "/" in str(symbol)
        else ""
    )
    if fee_cost <= 0.0:
        return 0.0, float(fee_cost), fee_currency, "order_fee"
    if fee_currency == quote_asset or not fee_currency:
        return float(fee_cost), float(fee_cost), fee_currency, "order_fee"
    if fee_currency == base_asset and np.isfinite(price):
        return float(fee_cost * price), float(fee_cost), fee_currency, "order_fee"
    return np.nan, float(fee_cost), fee_currency, "unconverted_order_fee"


def _combine_fee_quotes(
    entry_fee_quote: float,
    exit_fee_quote: float,
) -> Tuple[float, bool]:
    """Combine finite fee quotes without silently treating missing fees as zero."""
    parts = [v for v in (entry_fee_quote, exit_fee_quote) if np.isfinite(v)]
    if not parts:
        return np.nan, False
    return float(sum(parts)), len(parts) == 2


def _directional_price_gap_bps(
    *,
    side: str,
    actual_price: float,
    reference_price: float,
) -> float:
    """Positive means actual fill is more favorable than the reference stop."""
    if (
        not np.isfinite(actual_price)
        or actual_price <= 0.0
        or not np.isfinite(reference_price)
        or reference_price <= 0.0
    ):
        return np.nan
    side_norm = str(side or "").lower()
    if side_norm == "short":
        return float((1.0 - actual_price / reference_price) * 10000.0)
    return float((actual_price / reference_price - 1.0) * 10000.0)


def _timestamp_delta_seconds(start: Any, end: Any) -> float:
    start_ts = pd.to_datetime(start, utc=True, errors="coerce")
    end_ts = pd.to_datetime(end, utc=True, errors="coerce")
    if pd.isna(start_ts) or pd.isna(end_ts):
        return np.nan
    out = float((pd.Timestamp(end_ts) - pd.Timestamp(start_ts)).total_seconds())
    return out if np.isfinite(out) else np.nan


def _signal_bar_close_ts_from_context(ctx: Dict[str, Any]) -> pd.Timestamp | None:
    close_ts = pd.to_datetime(ctx.get("signal_bar_close_ts"), utc=True, errors="coerce")
    if not pd.isna(close_ts):
        return pd.Timestamp(close_ts)
    signal_ts = pd.to_datetime(ctx.get("signal_bar_ts"), utc=True, errors="coerce")
    if pd.isna(signal_ts):
        return None
    return pd.Timestamp(signal_ts) + pd.Timedelta(hours=1)


def _hard_stale_signal_entry_gate_enabled(
    config: Optional[Dict[str, Any]],
    *,
    mode: str,
) -> bool:
    cfg = config or {}
    default = _is_live_execution_mode(mode)
    raw = os.environ.get(
        "EPM_HARD_STALE_SIGNAL_ENTRY_GATE_ENABLED",
        cfg.get("hard_stale_signal_entry_gate_enabled", default),
    )
    return _config_bool(raw, default=default)


def _max_signal_close_to_entry_seconds(config: Optional[Dict[str, Any]]) -> float:
    cfg = config or {}
    raw = os.environ.get(
        "EPM_MAX_SIGNAL_CLOSE_TO_ENTRY_SECONDS",
        cfg.get("max_signal_close_to_entry_seconds", 900.0),
    )
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 900.0


def _stale_signal_entry_reject(
    *,
    trade_context: Dict[str, Any],
    now: pd.Timestamp,
    config: Optional[Dict[str, Any]],
    mode: str,
) -> Dict[str, Any] | None:
    if not _hard_stale_signal_entry_gate_enabled(config, mode=mode):
        return None
    limit_seconds = _max_signal_close_to_entry_seconds(config)
    if not np.isfinite(limit_seconds) or limit_seconds < 0.0:
        return None
    close_ts = _signal_bar_close_ts_from_context(trade_context)
    if close_ts is None:
        return None
    age_seconds = _timestamp_delta_seconds(close_ts, now)
    if not (np.isfinite(age_seconds) and age_seconds > limit_seconds):
        return None
    signal_to_entry_seconds = _timestamp_delta_seconds(
        trade_context.get("signal_bar_ts"),
        now,
    )
    return {
        "success": False,
        "status": "rejected",
        "error": "stale_signal_age_exceeded",
        "error_category": "stale_signal_age_exceeded",
        "portfolio_reject_reason": "stale_signal_age_exceeded",
        "signal_bar_ts": trade_context.get("signal_bar_ts"),
        "signal_bar_close_ts": close_ts.isoformat(),
        "signal_close_to_decision_seconds": age_seconds,
        "signal_close_to_entry_seconds": age_seconds,
        "signal_to_entry_seconds": signal_to_entry_seconds,
        "max_signal_close_to_entry_seconds": limit_seconds,
        "stale_signal_age_gate_enabled": True,
        "stale_signal_age_gate_exceeded": True,
    }


def _entry_price_attribution_fields(
    *,
    side: str,
    theoretical_entry_price: float,
    decision_price: float,
    fill_price: float,
    trade_context: Optional[Dict[str, Any]],
    entry_fee_quote: float = np.nan,
    entry_notional_quote: float = np.nan,
) -> Dict[str, Any]:
    hourly_to_decision_bps = _directional_price_gap_bps(
        side=side,
        actual_price=float(decision_price),
        reference_price=float(theoretical_entry_price),
    )
    decision_to_fill_bps = _directional_price_gap_bps(
        side=side,
        actual_price=float(fill_price),
        reference_price=float(decision_price),
    )
    ctx = dict(trade_context or {})
    spread_proxy_bps = _safe_float(
        ctx.get("ticker_spread_bps", ctx.get("spread_bps")),
        default=np.nan,
    )
    orderbook_live_slippage_bps = _safe_float(
        ctx.get("orderbook_slippage_bps", ctx.get("expected_fill_slippage_bps")),
        default=np.nan,
    )
    fee_bps = (
        float(entry_fee_quote) / max(abs(float(entry_notional_quote)), 1e-12) * 10000.0
        if np.isfinite(entry_fee_quote)
        and np.isfinite(entry_notional_quote)
        and abs(float(entry_notional_quote)) > 0.0
        else _safe_float(ctx.get("fee_bps", ctx.get("realized_fee_bps")), default=np.nan)
    )
    return {
        "latest_decision_price": (
            float(decision_price)
            if np.isfinite(decision_price) and decision_price > 0.0
            else np.nan
        ),
        "hourly_close_to_latest_decision_price_bps": hourly_to_decision_bps,
        "decision_price_to_fill_bps": decision_to_fill_bps,
        "spread_proxy_bps": spread_proxy_bps,
        "orderbook_live_slippage_bps": orderbook_live_slippage_bps,
        "fee_bps": fee_bps,
        "entry_price_attribution_schema": (
            "side_aware_bps: hourly_close_to_latest_decision_price + "
            "decision_price_to_fill + spread_proxy + orderbook_live_slippage + fees"
        ),
    }


def _wallet_scaled_pnl_fields(
    *,
    state: Dict[str, Any],
    notional: float,
    gross_pnl: float,
    net_pnl: float,
) -> Dict[str, Any]:
    """Return effective notional-to-wallet leverage and wallet-scaled PnL."""
    wallet_value = _safe_float(
        state.get("wallet_value_at_entry"),
        default=_safe_float(state.get("wallet_value"), np.nan),
    )
    open_notional = _safe_float(
        state.get("open_notional_at_entry"),
        default=_safe_float(state.get("open_notional"), np.nan),
    )
    effective_leverage = (
        float(notional / wallet_value)
        if np.isfinite(notional)
        and notional > 0.0
        and np.isfinite(wallet_value)
        and wallet_value > 0.0
        else np.nan
    )
    gross_pnl_pct_wallet = (
        float(gross_pnl / wallet_value)
        if np.isfinite(gross_pnl) and np.isfinite(wallet_value) and wallet_value > 0.0
        else np.nan
    )
    net_pnl_pct_wallet = (
        float(net_pnl / wallet_value)
        if np.isfinite(net_pnl) and np.isfinite(wallet_value) and wallet_value > 0.0
        else np.nan
    )
    return {
        "wallet_value_at_entry": wallet_value,
        "open_notional_at_entry": open_notional,
        "leverage_wallet_multiplier": _safe_float(
            state.get("leverage_wallet_multiplier"), np.nan
        ),
        "effective_position_leverage": effective_leverage,
        "gross_pnl_pct_wallet": gross_pnl_pct_wallet,
        "net_pnl_pct_wallet": net_pnl_pct_wallet,
        "leverage_adjusted_gross_pnl_pct": gross_pnl_pct_wallet,
        "leverage_adjusted_net_pnl_pct": net_pnl_pct_wallet,
    }


def _closed_trade_metrics(
    symbol: str,
    state: Dict[str, Any],
    order: Optional[Dict[str, Any]],
    *,
    reason: str,
) -> Dict[str, Any]:
    """Build close metrics from tracked state and an exchange close order."""
    order = order or {}
    side = str(state.get("side", "") or "").lower()
    entry_price = _safe_float(state.get("entry_price"))
    exit_price = _safe_float(
        order.get("average"),
        _safe_float(order.get("price"), _safe_float(order.get("stopPrice"))),
    )
    filled = _safe_float(order.get("filled"), _safe_float(order.get("amount")))
    if not np.isfinite(filled) or filled <= 0:
        filled = _safe_float(state.get("size"))
    gross_pnl = np.nan
    gross_pnl_pct = np.nan
    notional = np.nan
    exit_notional = np.nan
    if (
        np.isfinite(entry_price)
        and entry_price > 0
        and np.isfinite(exit_price)
        and np.isfinite(filled)
        and filled > 0
    ):
        direction = 1.0 if side == "long" else -1.0
        notional = entry_price * filled
        exit_notional = exit_price * filled
        gross_pnl = direction * (exit_price - entry_price) * filled
        gross_pnl_pct = gross_pnl / max(notional, 1e-12)
    exit_fee_quote, fee_cost, fee_currency, exit_fee_source = _fee_to_quote(
        symbol,
        order.get("fee"),
        price=exit_price,
    )
    entry_fee_quote = _safe_float(state.get("entry_fee_quote"), np.nan)
    entry_fee_source = str(state.get("entry_fee_source") or "")
    fee_quote, fees_verified = _combine_fee_quotes(entry_fee_quote, exit_fee_quote)
    net_pnl = (
        gross_pnl - fee_quote
        if np.isfinite(gross_pnl) and np.isfinite(fee_quote)
        else np.nan
    )
    net_pnl_pct = net_pnl / max(notional, 1e-12) if np.isfinite(notional) else np.nan
    gross_to_net_cost = fee_quote if np.isfinite(fee_quote) else np.nan
    gross_to_net_cost_pct = (
        gross_to_net_cost / max(notional, 1e-12)
        if np.isfinite(gross_to_net_cost) and np.isfinite(notional)
        else np.nan
    )
    stop_price = _safe_float(state.get("stop_price"), default=np.nan)
    placed_stop_price = _safe_float(
        state.get("final_placed_stop"),
        default=_safe_float(state.get("exchange_stop_price"), default=stop_price),
    )
    requested_policy_stop = _safe_float(
        state.get("requested_policy_stop"),
        default=_safe_float(state.get("policy_stop_price"), default=stop_price),
    )
    exit_vs_policy_stop_bps = _directional_price_gap_bps(
        side=side,
        actual_price=exit_price,
        reference_price=requested_policy_stop,
    )
    mfe_value = _safe_float(state.get("mfe"), np.nan)
    mae_value = _safe_float(state.get("mae"), np.nan)
    exit_vs_peak_giveback_pct = np.nan
    if np.isfinite(mfe_value) and mfe_value > 0.0 and np.isfinite(gross_pnl_pct):
        exit_vs_peak_giveback_pct = float(
            max(mfe_value - gross_pnl_pct, 0.0) / mfe_value
        )
    policy_parity_ok = (
        bool(
            np.isfinite(exit_vs_policy_stop_bps)
            and abs(exit_vs_policy_stop_bps) <= 75.0
        )
        if np.isfinite(requested_policy_stop)
        else False
    )
    wallet_pnl_fields = _wallet_scaled_pnl_fields(
        state=state,
        notional=notional,
        gross_pnl=gross_pnl,
        net_pnl=net_pnl,
    )
    stop_origin = str(state.get("stop_reason") or "original_stop_loss")
    reason_detail = str(state.get("stop_reason_detail") or stop_origin)
    reason_out = (
        f"{reason}:{stop_origin}"
        if str(reason) == "stop_loss_filled" and stop_origin
        else reason
    )
    entry_time = state.get("entry_time")
    exit_time = pd.Timestamp.now(tz="UTC")
    holding_time_hours = np.nan
    try:
        if entry_time is not None:
            entry_ts = pd.Timestamp(entry_time)
            if entry_ts.tzinfo is None:
                entry_ts = entry_ts.tz_localize("UTC")
            else:
                entry_ts = entry_ts.tz_convert("UTC")
            holding_time_hours = float((exit_time - entry_ts).total_seconds() / 3600.0)
    except Exception:
        holding_time_hours = np.nan
    shadow = (
        state.get("simple_policy_shadow")
        if isinstance(state.get("simple_policy_shadow"), dict)
        else {}
    )
    if (
        isinstance(shadow, dict)
        and str(shadow.get("status") or "open") == "open"
        and str(reason) == "stop_loss_filled"
    ):
        shadow_stop_price = _safe_float(
            shadow.get("shadow_stop_price"),
            default=_safe_float(shadow.get("initial_shadow_stop_price"), default=stop_price),
        )
        shadow_exit_price = shadow_stop_price if np.isfinite(shadow_stop_price) else exit_price
        if np.isfinite(shadow_exit_price) and shadow_exit_price > 0.0:
            shadow_exit_reason = (
                "shadow_stop_loss_filled:"
                f"{shadow.get('shadow_stop_reason') or stop_origin}"
            )
            shadow.update(
                {
                    "status": "shadow_exit_triggered",
                    "shadow_exit_time": exit_time.isoformat(),
                    "shadow_exit_price": float(shadow_exit_price),
                    "shadow_exit_reason": shadow_exit_reason,
                    "shadow_exit_return": (
                        (float(shadow_exit_price) - float(entry_price))
                        / max(abs(float(entry_price)), 1e-12)
                        if side == "long"
                        else (float(entry_price) - float(shadow_exit_price))
                        / max(abs(float(entry_price)), 1e-12)
                    )
                    if np.isfinite(entry_price) and entry_price > 0.0
                    else None,
                    "shadow_exit_vs_live_stop_bps": _directional_price_gap_bps(
                        side=side,
                        actual_price=float(shadow_exit_price),
                        reference_price=stop_price,
                    ),
                }
            )
            events = shadow.setdefault("events", [])
            if isinstance(events, list):
                events.append(
                    {
                        "ts": exit_time.isoformat(),
                        "event": "shadow_exchange_stop_filled",
                        "shadow_exit_price": float(shadow_exit_price),
                        "live_exit_price": float(exit_price)
                        if np.isfinite(exit_price)
                        else None,
                        "live_stop_price": float(stop_price)
                        if np.isfinite(stop_price)
                        else None,
                        "stop_reason": stop_origin,
                    }
                )
                if len(events) > 200:
                    del events[:-200]
    return {
        "symbol": symbol,
        "side": side,
        "strategy_id": state.get("bucket_key"),
        "reason": reason_out,
        "exit_reason_detail": reason_detail,
        "stop_origin": stop_origin,
        "entry_time": entry_time,
        "exit_time": exit_time,
        "decision_to_entry_seconds": state.get("decision_to_entry_seconds"),
        "signal_to_entry_seconds": state.get("signal_to_entry_seconds"),
        "holding_time_hours": holding_time_hours,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "entry_order_type": state.get("entry_order_type"),
        "ohlcv_entry_price": state.get("ohlcv_entry_price"),
        "entry_price_delta_vs_ohlcv": state.get("entry_price_delta_vs_ohlcv"),
        "entry_price_delta_vs_ohlcv_pct": state.get("entry_price_delta_vs_ohlcv_pct"),
        "base_pred": state.get("base_pred"),
        "base_rank_pct": state.get("base_rank_pct"),
        "base_train_rank_pct": state.get("base_train_rank_pct"),
        "base_gate_top_frac": state.get("base_gate_top_frac"),
        "meta_pred": state.get("meta_pred"),
        "estimated_hit_rate": state.get("estimated_hit_rate"),
        "estimated_hit_rate_source": state.get("estimated_hit_rate_source"),
        "estimated_hit_rate_calibration_n": state.get(
            "estimated_hit_rate_calibration_n"
        ),
        "estimated_ev_gross_return": state.get("estimated_ev_gross_return"),
        "estimated_ev_net_return": state.get("estimated_ev_net_return"),
        "estimated_ev_cost_bps": state.get("estimated_ev_cost_bps"),
        "estimated_ev_hit_rate": state.get("estimated_ev_hit_rate"),
        "estimated_ev_source": state.get("estimated_ev_source"),
        "estimated_ev_calibration_n": state.get("estimated_ev_calibration_n"),
        "meta_train_rank_pct": state.get("meta_train_rank_pct"),
        "rank_score_source": state.get("rank_score_source"),
        "calibrated_score": state.get("calibrated_score"),
        "rank_percentile": state.get("rank_percentile"),
        "effective_threshold": state.get("effective_threshold"),
        "deployment_rank_threshold": state.get("deployment_rank_threshold"),
        "filled": filled,
        "entry_notional_quote": notional,
        "exit_notional_quote": exit_notional,
        "quote_size": state.get("quote_size"),
        "requested_base_amount": state.get("requested_base_amount"),
        **wallet_pnl_fields,
        "gross_pnl": gross_pnl,
        "gross_pnl_amount": gross_pnl,
        "gross_pnl_pct": gross_pnl_pct,
        "net_pnl": net_pnl,
        "net_pnl_amount": net_pnl,
        "net_pnl_pct": net_pnl_pct,
        "entry_fee_quote": entry_fee_quote,
        "exit_fee_quote": exit_fee_quote,
        "entry_fee_source": entry_fee_source,
        "exit_fee_source": exit_fee_source,
        "fee_source": (
            "verified_order_fees"
            if fees_verified
            else f"partial_or_missing(entry={entry_fee_source or 'missing'},exit={exit_fee_source})"
        ),
        "fees_verified": fees_verified,
        "gross_to_net_cost_quote": gross_to_net_cost,
        "gross_to_net_cost_pct": gross_to_net_cost_pct,
        "gross_to_net_friction_drag_bps": (
            float(gross_to_net_cost_pct) * 10000.0
            if np.isfinite(gross_to_net_cost_pct)
            else np.nan
        ),
        "pnl_scope": "position_notional_excluding_wallet_equity_borrow_interest",
        "mfe": mfe_value,
        "mae": mae_value,
        "requested_policy_stop": requested_policy_stop,
        "final_placed_stop": placed_stop_price,
        "exchange_stop_price": state.get("exchange_stop_price"),
        "exchange_stop_trigger_reference_source": state.get(
            "exchange_stop_trigger_reference_source"
        ),
        "exchange_stop_adjustment": state.get("exchange_stop_adjustment"),
        "exit_vs_policy_stop_bps": exit_vs_policy_stop_bps,
        "exit_vs_peak_giveback_pct": exit_vs_peak_giveback_pct,
        "policy_parity_ok": policy_parity_ok,
        "stop_price": state.get("stop_price"),
        "stop_order_id": state.get("stop_order_id"),
        "stop_trigger_signal": state.get("stop_trigger_signal"),
        "stop_trigger_reference_source": state.get("stop_trigger_reference_source"),
        "close_order_id": order.get("id"),
        "close_order_status": order.get("status"),
        "close_order_type": order.get("type"),
        "close_order_cost": order.get("cost"),
        "fee_cost": fee_cost,
        "fee_currency": fee_currency,
        "fees_amount": fee_quote,
        "decision_module": state.get("decision_module"),
        "stop_policy_params_source": state.get("stop_policy_params_source"),
        "stop_policy_params_hash": state.get("stop_policy_params_hash"),
        "stop_policy_schema": state.get("stop_policy_schema"),
        "simple_policy_shadow": state.get("simple_policy_shadow"),
        "shadow_policy_schema": shadow.get("schema"),
        "shadow_policy_params_source": shadow.get("params_source"),
        "shadow_policy_params_hash": shadow.get("params_hash"),
        "shadow_policy_entry_price": shadow.get("policy_entry_price"),
        "shadow_realized_entry_price": shadow.get("realized_entry_price"),
        "shadow_entry_gap_bps": shadow.get("entry_gap_bps"),
        "shadow_initial_stop_price": shadow.get("initial_shadow_stop_price"),
        "shadow_latest_stop_price": shadow.get("shadow_stop_price"),
        "shadow_live_stop_price": shadow.get("latest_live_stop_price"),
        "shadow_stop_gap_bps": shadow.get("latest_stop_gap_bps"),
        "shadow_exit_time": shadow.get("shadow_exit_time"),
        "shadow_exit_price": shadow.get("shadow_exit_price"),
        "shadow_exit_reason": shadow.get("shadow_exit_reason"),
        "shadow_exit_return": shadow.get("shadow_exit_return"),
        "shadow_status": shadow.get("status"),
        "trade_recap": _format_trade_recap(state.get("trade_recap_events")),
        **_execution_audit_fields(state),
    }


class OCOExecutor:
    """STOP_LOSS executor for simple-policy-governed protective orders.

    This class places initial STOP_LOSS orders from validated
    simple_policy_optimiser params and only replaces them from
    SimplePolicyStopDecision objects produced by simple_policy_stop.py.

    TODO: Rename this class after downstream callers no longer use the OCO name.
    """

    def __init__(
        self,
        exchange: Any,
        bucket_params: Dict[str, Dict[str, Any]],
        config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize OCOExecutor.

        Args:
            exchange: ccxt exchange instance
            bucket_params: Parameters from ridge position sizer per bucket
            config: Additional configuration options
        """
        self.exchange = exchange
        self.bucket_params = bucket_params
        self.config = config or {}
        self.simple_policy_stop_params_by_strategy = (
            extract_simple_policy_stop_params_by_strategy(bucket_params)
        )
        self.active_positions: Dict[str, Dict[str, Any]] = {}
        self._positions_lock = threading.RLock()

    def get_simple_policy_stop_params(
        self, bucket_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """Return exact-strategy simple_policy_optimiser stop params."""
        if bucket_key is None:
            return {}
        key = str(bucket_key)
        params = self.simple_policy_stop_params_by_strategy.get(key)
        return dict(params) if isinstance(params, dict) else {}

    def resolve_simple_policy_strategy_id(
        self, strategy_id: Optional[str], side: Optional[str]
    ) -> str:
        """Return the exact side-prefixed strategy id present in policy artifacts."""
        sid = str(strategy_id or "").strip()
        side_l = str(side or "").lower().strip()
        candidates: List[str] = []
        if sid:
            candidates.append(sid)
            core = strategy_core_id(sid)
            if core and side_l in {"long", "short"}:
                candidates.append(f"{side_l}_{core}")
        for candidate in candidates:
            if candidate in self.simple_policy_stop_params_by_strategy:
                return candidate
        return self._fallback_simple_policy_strategy_id(side) or sid

    def _fallback_simple_policy_strategy_id(self, side: Optional[str]) -> str:
        """Pick a deterministic deployed simple-policy strategy for unknown imports."""
        side_l = str(side or "").lower().strip()
        candidates: List[str] = []
        for sid, params in self.simple_policy_stop_params_by_strategy.items():
            sid_s = str(sid)
            param_side = str((params or {}).get("side") or "").lower().strip()
            if side_l in {"long", "short"} and (
                param_side == side_l or sid_s.startswith(f"{side_l}_")
            ):
                candidates.append(sid_s)
        if not candidates:
            candidates = [
                str(sid) for sid in self.simple_policy_stop_params_by_strategy
            ]
        return sorted(candidates)[0] if candidates else ""

    def get_cooldown_hours(self, bucket_key: Optional[str] = None) -> float:
        """Return zero: live inference cooldown is handled on losing closes only."""
        return 0.0

    def _fetch_aggtrades(
        self, symbol: str, since: int = None, limit: int = None
    ) -> List[Dict[str, Any]]:
        """Fetch aggregated trades for a symbol.

        Args:
            symbol: Trading symbol (e.g., "BTC/USDT")
            since: Start time in milliseconds (optional)
            limit: Number of trades to fetch (optional)

        Returns:
            List of aggtrade dictionaries
        """
        try:
            # Try different method names for aggtrades
            if hasattr(self.exchange, "fetch_aggregated_trades"):
                return self.exchange.fetch_aggregated_trades(
                    symbol, since=since, limit=limit
                )
            elif hasattr(self.exchange, "fetch_agg_trades"):
                return self.exchange.fetch_agg_trades(symbol, since=since, limit=limit)
            elif hasattr(self.exchange, "fetch_trades"):
                # Fallback to regular trades
                return self.exchange.fetch_trades(symbol, since=since, limit=limit)
            else:
                tprint(f"Exchange does not support fetching trades for {symbol}")
                return []
        except Exception as e:
            tprint(
                f"Error fetching aggtrades for {symbol}: "
                f"{_classify_exchange_error(e)}: {e}"
            )
            return []

    def _get_aggtrades_at_entry(
        self, symbol: str, entry_time: pd.Timestamp
    ) -> List[Dict[str, Any]]:
        """Fetch aggtrades around the entry time.

        Fetches aggtrades for the minute surrounding the entry time.

        Args:
            symbol: Trading symbol
            entry_time: Entry timestamp

        Returns:
            List of aggtrade dictionaries
        """
        # Convert entry_time to milliseconds
        if entry_time.tz is not None:
            entry_time = entry_time.tz_convert("UTC")

        # Get trades for 1 minute around entry time
        start_ms = int(entry_time.timestamp() * 1000) - 60000  # 1 minute before
        end_ms = int(entry_time.timestamp() * 1000) + 60000  # 1 minute after

        return self._fetch_aggtrades(symbol, since=start_ms, limit=1000)

    def place_oco_order(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        size: float,
        bucket_key: str,
        barrier_frac: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Place the initial STOP_LOSS order after entry.

        Args:
            symbol: Trading symbol (e.g., "BTC/USDT")
            side: "long" or "short"
            entry_price: Entry price for the position
            size: Position size in base asset units
            bucket_key: Bucket key for getting stop parameters

        Returns:
            Dictionary with order details
        """
        params = self.get_simple_policy_stop_params(bucket_key)

        require_metadata = True
        try:
            initial_stop_decision = compute_initial_simple_policy_stop_decision(
                entry_price=float(entry_price),
                policy_params=params,
                side=side,
                strategy_id=bucket_key,
                barrier_frac=barrier_frac,
                require_metadata=require_metadata,
            )
            valid_decision, invalid_reason = _validate_policy_stop_decision(
                initial_stop_decision, require_should_replace=True
            )
            if not valid_decision:
                raise SimplePolicyStopParamsError(invalid_reason)
        except SimplePolicyStopParamsError as exc:
            error = str(exc)
            tprint(f"Refusing STOP_LOSS for {symbol}: {error}")
            return {
                "success": False,
                "symbol": symbol,
                "side": side,
                "entry_price": entry_price,
                "size": size,
                "bucket_key": bucket_key,
                "error": error,
                "error_category": "invalid_simple_policy_stop_params",
            }
        barrier_frac = initial_stop_decision.barrier_frac
        sl_mult = initial_stop_decision.sl_mult
        stop_price = float(initial_stop_decision.stop_price)
        limit_price = None

        # Track position state
        position_state = {
            "side": side,
            "entry_price": entry_price,
            "size": size,
            "bucket_key": bucket_key,
            "stop_price": stop_price,
            "limit_price": limit_price,
            "barrier_frac": barrier_frac,
            "barrier_pct": barrier_frac,
            "sl_mult": sl_mult,
            "trailing_activation_mult": initial_stop_decision.trailing_activation_mult,
            "trailing_power": initial_stop_decision.trailing_power,
            "trailing_squash_divisor": initial_stop_decision.trailing_squash_divisor,
            "giveback_beta": initial_stop_decision.giveback_beta,
            "atr_power": initial_stop_decision.atr_power,
            "atr_multiplier": initial_stop_decision.atr_multiplier,
            "hard_tp_abs_pct": initial_stop_decision.hard_tp_abs_pct,
            "capital_protect_mfe_mult": initial_stop_decision.capital_protect_mfe_mult,
            "capital_protect_regression_frac": initial_stop_decision.capital_protect_regression_frac,
            "adverse_exit_enabled": initial_stop_decision.adverse_exit_enabled,
            "adverse_exit_theta": initial_stop_decision.adverse_exit_theta,
            "adverse_exit_theta_quantile": initial_stop_decision.adverse_exit_theta_quantile,
            "adverse_exit_min_mae_atr": initial_stop_decision.adverse_exit_min_mae_atr,
            "adverse_exit_min_speed": initial_stop_decision.adverse_exit_min_speed,
            "adverse_exit_fast_bars": initial_stop_decision.adverse_exit_fast_bars,
            "adverse_exit_max_mfe_atr": initial_stop_decision.adverse_exit_max_mfe_atr,
            "decision_module": initial_stop_decision.decision_module,
            "stop_policy_params_source": initial_stop_decision.params_source,
            "stop_policy_params_hash": initial_stop_decision.params_hash,
            "stop_policy_schema": initial_stop_decision.params_schema,
            "strategy_id": initial_stop_decision.strategy_id,
            "initial_stop_price": stop_price,
            "stop_reason": "original_stop_loss",
            "stop_reason_detail": initial_stop_decision.reason_detail,
            "peak_price": entry_price,
            "mfe": 0.0,
            "mae": 0.0,
            "entry_time": pd.Timestamp.now(tz="UTC"),
            "last_update": pd.Timestamp.now(tz="UTC"),
            "oco_order_id": None,
            "stop_order_id": None,
            "take_profit_order_id": None,
        }
        _append_position_event(
            position_state,
            "entry_stop_created",
            entry_price=float(entry_price),
            stop_price=float(stop_price),
            stop_dev_pct=(
                (float(stop_price) - float(entry_price))
                / max(float(entry_price), 1e-12)
            ),
            sl_mult=float(sl_mult),
            barrier_frac=float(barrier_frac),
            stop_reason="original_stop_loss",
            params_source=initial_stop_decision.params_source,
            params_hash=initial_stop_decision.params_hash,
            schema=initial_stop_decision.params_schema,
            decision_module=initial_stop_decision.decision_module,
            trailing_activation_mult=initial_stop_decision.trailing_activation_mult,
            trailing_power=initial_stop_decision.trailing_power,
            trailing_squash_divisor=initial_stop_decision.trailing_squash_divisor,
            giveback_beta=initial_stop_decision.giveback_beta,
            atr_power=initial_stop_decision.atr_power,
            atr_multiplier=initial_stop_decision.atr_multiplier,
            hard_tp_abs_pct=initial_stop_decision.hard_tp_abs_pct,
            capital_protect_mfe_mult=initial_stop_decision.capital_protect_mfe_mult,
            capital_protect_regression_frac=initial_stop_decision.capital_protect_regression_frac,
            adverse_exit_enabled=initial_stop_decision.adverse_exit_enabled,
            adverse_exit_theta=initial_stop_decision.adverse_exit_theta,
            adverse_exit_theta_quantile=initial_stop_decision.adverse_exit_theta_quantile,
            adverse_exit_min_mae_atr=initial_stop_decision.adverse_exit_min_mae_atr,
            adverse_exit_min_speed=initial_stop_decision.adverse_exit_min_speed,
            adverse_exit_fast_bars=initial_stop_decision.adverse_exit_fast_bars,
            adverse_exit_max_mfe_atr=initial_stop_decision.adverse_exit_max_mfe_atr,
        )

        market = _load_market(self.exchange, symbol)
        amount = _exchange_precision(self.exchange, symbol, float(size), kind="amount")
        stop_price = _exchange_precision(
            self.exchange, symbol, float(stop_price), kind="price"
        )
        current_price = np.nan
        current_price_source = "unavailable"
        ticker: Dict[str, Any] = {}
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            current_price, current_price_source = _stop_trigger_reference_price(
                self.exchange, ticker, self.config, position_side=side
            )
        except Exception as price_exc:
            _append_position_event(
                position_state,
                "entry_stop_price_fetch_failed",
                error_category=_classify_exchange_error(price_exc),
                error=str(price_exc),
            )
        if np.isfinite(current_price) and current_price > 0.0:
            adjusted_stop, adjusted, boundary = _adjust_stop_to_min_current_distance(
                side, float(stop_price), float(current_price)
            )
            if adjusted:
                adjusted_stop = _exchange_precision(
                    self.exchange, symbol, adjusted_stop, kind="price"
                )
                _append_position_event(
                    position_state,
                    "entry_stop_min_current_distance_adjusted",
                    original_stop=float(stop_price),
                    adjusted_stop=float(adjusted_stop),
                    current_price=float(current_price),
                    current_price_source=current_price_source,
                    min_distance_pct=STOP_MIN_CURRENT_DISTANCE_PCT,
                    boundary=float(boundary),
                )
                stop_price = float(adjusted_stop)
            if not _stop_side_is_valid(side, float(stop_price), float(current_price)):
                error = (
                    "entry stop does not satisfy min current-price distance: "
                    f"side={side} stop={float(stop_price):.8g} "
                    f"current={float(current_price):.8g} "
                    f"current_source={current_price_source} "
                    f"min_distance_pct={STOP_MIN_CURRENT_DISTANCE_PCT:.6g}"
                )
                tprint(f"Refusing STOP_LOSS for {symbol}: {error}")
                return {
                    "success": False,
                    "symbol": symbol,
                    "side": side,
                    "entry_price": entry_price,
                    "size": size,
                    "bucket_key": bucket_key,
                    "error": error,
                    "error_category": "local_stop_min_distance_invalid",
                }
        position_state["size"] = amount
        position_state["stop_price"] = stop_price
        position_state["policy_stop_price"] = stop_price
        position_state["requested_policy_stop"] = stop_price
        exchange_stop_price = float(stop_price)
        exchange_stop_meta: Dict[str, Any] = {}
        if (
            _execution_account(self.config) == "perps"
            and _exchange_id(self.exchange) == "krakenfutures"
        ):
            exchange_stop_price, exchange_stop_meta = (
                _kraken_futures_last_stop_from_executable_stop(
                    ticker,
                    self.config,
                    position_side=side,
                    policy_stop_price=float(stop_price),
                )
            )
            if np.isfinite(exchange_stop_price) and exchange_stop_price > 0.0:
                exchange_stop_price = _exchange_precision(
                    self.exchange, symbol, float(exchange_stop_price), kind="price"
                )
                last_trigger_price = _safe_float(
                    exchange_stop_meta.get("last"), default=np.nan
                )
                if np.isfinite(last_trigger_price) and last_trigger_price > 0.0:
                    (
                        exchange_stop_adjusted,
                        exchange_adjusted,
                        exchange_boundary,
                    ) = _adjust_stop_to_min_current_distance(
                        side, float(exchange_stop_price), float(last_trigger_price)
                    )
                    if exchange_adjusted:
                        exchange_stop_adjusted = _exchange_precision(
                            self.exchange,
                            symbol,
                            float(exchange_stop_adjusted),
                            kind="price",
                        )
                        _append_position_event(
                            position_state,
                            "exchange_last_stop_min_current_distance_adjusted",
                            policy_stop=float(stop_price),
                            original_exchange_stop=float(exchange_stop_price),
                            adjusted_exchange_stop=float(exchange_stop_adjusted),
                            last_price=float(last_trigger_price),
                            min_distance_pct=STOP_MIN_CURRENT_DISTANCE_PCT,
                            boundary=float(exchange_boundary),
                        )
                        exchange_stop_price = float(exchange_stop_adjusted)
            else:
                exchange_stop_price = float(stop_price)
        position_state["exchange_stop_price"] = float(exchange_stop_price)
        position_state["exchange_stop_trigger_reference_source"] = "last"
        position_state["exchange_stop_adjustment"] = exchange_stop_meta
        position_state["final_placed_stop"] = float(exchange_stop_price)
        reduce_side = "sell" if side == "long" else "buy"
        stop_trigger_signal = (
            _kraken_futures_stop_trigger_signal_for_reduce_side(reduce_side)
            if _execution_account(self.config) == "perps"
            and _exchange_id(self.exchange) == "krakenfutures"
            else None
        )
        if stop_trigger_signal:
            position_state["stop_trigger_signal"] = stop_trigger_signal
        if current_price_source and current_price_source != "unavailable":
            position_state["stop_trigger_reference_source"] = current_price_source

        stop_order_error = None
        stop_order_error_category = None
        try:
            _validate_order_filters(
                symbol, market, amount=amount, price=float(entry_price)
            )
            stop_order = _create_reduce_stop_loss_order(
                self.exchange,
                symbol=symbol,
                side=reduce_side,
                amount=amount,
                stop_price=exchange_stop_price,
                config=self.config,
            )
            position_state["stop_order_id"] = stop_order.get("id")
            if position_state["stop_order_id"]:
                position_state["stop_order_ids"] = [position_state["stop_order_id"]]
            requested_trigger_signal = position_state.get("stop_trigger_signal")
            actual_trigger_signal, _verified_order, trigger_meta = (
                _verify_exchange_stop_trigger_signal(
                    self.exchange,
                    symbol=symbol,
                    order_id=position_state.get("stop_order_id"),
                    order=stop_order,
                    config=self.config,
                )
            )
            if actual_trigger_signal:
                position_state["stop_trigger_signal"] = actual_trigger_signal
            if (
                requested_trigger_signal
                and actual_trigger_signal
                and str(actual_trigger_signal) != str(requested_trigger_signal)
            ):
                position_state["stop_trigger_signal_requested"] = str(
                    requested_trigger_signal
                )
                position_state["stop_exchange_trigger_mismatch"] = True
                _append_position_event(
                    position_state,
                    "stop_trigger_signal_mismatch",
                    requested_trigger_signal=str(requested_trigger_signal),
                    exchange_trigger_signal=str(actual_trigger_signal),
                    stop_order_id=position_state.get("stop_order_id"),
                    trigger_verify_meta=trigger_meta,
                )

        except Exception as e:
            stop_order_error = str(e)
            stop_order_error_category = _classify_exchange_error(e)
            stop_order_reject_reason = _exchange_reject_reason(e)
            position_state["stop_order_error"] = stop_order_error
            position_state["stop_order_error_category"] = stop_order_error_category
            position_state["stop_order_reject_reason"] = stop_order_reject_reason
            _append_position_event(
                position_state,
                "entry_stop_failed",
                error_category=stop_order_error_category,
                reject_reason=stop_order_reject_reason,
                error=stop_order_error,
            )
            tprint(
                f"Error placing STOP_LOSS for {symbol}: "
                f"{stop_order_error_category} reason={stop_order_reject_reason}: {e}"
            )
            # Continue with tracking even if order placement fails
            # This allows for manual intervention or retry

        # Store position
        with self._positions_lock:
            self.active_positions[symbol] = position_state

        # Fetch aggtrades and 5m OHLCV for analysis (OCOExecutor is only used in live mode)
        aggtrades_data = None
        ohlcv_5m_data = None

        # Only fetch data if order was placed successfully
        if position_state.get("stop_order_id"):
            try:
                # Fetch aggtrades around entry time
                entry_time = pd.Timestamp.now(tz="UTC")
                aggtrades_data = self._get_aggtrades_at_entry(symbol, entry_time)
                position_state["aggtrades"] = aggtrades_data

                # Fetch 5m OHLCV data using hf_data_loader
                try:
                    ohlcv_5m_data = hf_data_loader.fetch_ohlcv_5m(
                        self.exchange,
                        symbol,
                        entry_time - pd.Timedelta(hours=1),  # 1 hour before
                        entry_time + pd.Timedelta(hours=12),  # 12 hours after
                    )
                    position_state["ohlcv_5m"] = ohlcv_5m_data
                except Exception as e:
                    tprint(
                        f"Error fetching 5m OHLCV for {symbol}: "
                        f"{_classify_exchange_error(e)}: {e}"
                    )

            except Exception as e:
                tprint(
                    f"Error fetching entry diagnostics for {symbol}: "
                    f"{_classify_exchange_error(e)}: {e}"
                )

        if position_state.get("stop_order_id"):
            tprint(
                f"Placed STOP_LOSS for {symbol}: SL={stop_price:.8g} "
                f"trigger={position_state.get('stop_trigger_signal') or 'unknown'} "
                f"barrier_frac={barrier_frac:.6g} sl_mult={sl_mult:.6g}"
            )

        return {
            "success": position_state.get("stop_order_id") is not None,
            "symbol": symbol,
            "side": side,
            "entry_price": entry_price,
            "stop_price": stop_price,
            "limit_price": limit_price,
            "size": amount,
            "bucket_key": bucket_key,
            "stop_order_id": position_state.get("stop_order_id"),
            "stop_trigger_signal": position_state.get("stop_trigger_signal"),
            "stop_trigger_reference_source": position_state.get(
                "stop_trigger_reference_source"
            ),
            "barrier_frac": float(barrier_frac),
            "barrier_pct": float(barrier_frac),
            "sl_mult": float(sl_mult),
            "trailing_activation_mult": initial_stop_decision.trailing_activation_mult,
            "trailing_power": initial_stop_decision.trailing_power,
            "trailing_squash_divisor": initial_stop_decision.trailing_squash_divisor,
            "giveback_beta": initial_stop_decision.giveback_beta,
            "atr_power": initial_stop_decision.atr_power,
            "atr_multiplier": initial_stop_decision.atr_multiplier,
            "hard_tp_abs_pct": initial_stop_decision.hard_tp_abs_pct,
            "capital_protect_mfe_mult": initial_stop_decision.capital_protect_mfe_mult,
            "capital_protect_regression_frac": initial_stop_decision.capital_protect_regression_frac,
            "adverse_exit_enabled": initial_stop_decision.adverse_exit_enabled,
            "adverse_exit_theta": initial_stop_decision.adverse_exit_theta,
            "adverse_exit_theta_quantile": initial_stop_decision.adverse_exit_theta_quantile,
            "adverse_exit_min_mae_atr": initial_stop_decision.adverse_exit_min_mae_atr,
            "adverse_exit_min_speed": initial_stop_decision.adverse_exit_min_speed,
            "adverse_exit_fast_bars": initial_stop_decision.adverse_exit_fast_bars,
            "adverse_exit_max_mfe_atr": initial_stop_decision.adverse_exit_max_mfe_atr,
            "decision_module": initial_stop_decision.decision_module,
            "strategy_id": initial_stop_decision.strategy_id,
            "stop_policy_params_source": initial_stop_decision.params_source,
            "stop_policy_params_hash": initial_stop_decision.params_hash,
            "stop_policy_schema": initial_stop_decision.params_schema,
            "policy_stop_price": position_state.get("policy_stop_price", stop_price),
            "requested_policy_stop": position_state.get(
                "requested_policy_stop", stop_price
            ),
            "exchange_stop_price": position_state.get("exchange_stop_price"),
            "exchange_stop_trigger_reference_source": position_state.get(
                "exchange_stop_trigger_reference_source"
            ),
            "exchange_stop_adjustment": position_state.get("exchange_stop_adjustment"),
            "final_placed_stop": position_state.get("final_placed_stop"),
            "error": stop_order_error,
            "error_category": stop_order_error_category,
            "aggtrades": aggtrades_data,
            "ohlcv_5m": ohlcv_5m_data,
        }

    def _update_stop_loss_from_policy_decision(
        self,
        symbol: str,
        state: Dict[str, Any],
        decision: Any,
    ):
        """Replace a stop only from a validated simple-policy decision."""
        artifact_params = self.get_simple_policy_stop_params(
            str(state.get("strategy_id") or state.get("bucket_key") or "")
        )
        valid_decision, invalid_reason = _validate_policy_stop_decision(
            decision,
            require_should_replace=False,
            position_state=state,
            artifact_params=artifact_params,
        )
        if not valid_decision:
            state["stop_update_error"] = invalid_reason
            state["stop_update_error_category"] = "unauthorised_stop_update"
            _append_position_event(
                state,
                "stop_replace_skipped",
                reason="invalid_simple_policy_decision_metadata",
                error=invalid_reason,
            )
            return
        if not decision.should_replace:
            _append_position_event(
                state,
                "stop_replace_skipped",
                reason="simple_policy_decision_no_replace",
                strategy_id=decision.strategy_id,
                params_source=decision.params_source,
                params_hash=decision.params_hash,
            )
            return
        valid_decision, invalid_reason = _validate_policy_stop_decision(
            decision,
            require_should_replace=True,
            position_state=state,
            artifact_params=artifact_params,
        )
        stop_price_value = _safe_float(decision.stop_price, default=np.nan)
        if not valid_decision:
            state["stop_update_error"] = invalid_reason
            state["stop_update_error_category"] = "unauthorised_stop_update"
            _append_position_event(
                state,
                "stop_replace_skipped",
                reason="invalid_simple_policy_decision_metadata",
                candidate_stop=decision.stop_price,
                error=invalid_reason,
            )
            return
        current_stop = _safe_float(state.get("stop_price"), default=np.nan)
        side = str(state.get("side", "long")).lower()
        improved = (
            stop_price_value > current_stop
            if side == "long"
            else stop_price_value < current_stop
        )
        if not (np.isfinite(current_stop) and improved):
            state["stop_update_error"] = "simple-policy stop is not an improvement"
            state["stop_update_error_category"] = "policy_stop_not_improved"
            _append_position_event(
                state,
                "stop_replace_skipped",
                reason="simple_policy_stop_not_improved",
                previous_stop=current_stop,
                candidate_stop=stop_price_value,
                strategy_id=decision.strategy_id,
                params_source=decision.params_source,
                params_hash=decision.params_hash,
            )
            return
        state["stop_reason"] = decision.reason
        state["stop_reason_detail"] = decision.reason_detail
        state["stop_policy_params_source"] = decision.params_source
        state["stop_policy_params_hash"] = decision.params_hash
        state["stop_policy_schema"] = decision.params_schema
        state["strategy_id"] = decision.strategy_id
        state["barrier_frac"] = decision.barrier_frac
        state["barrier_pct"] = decision.barrier_frac
        state["sl_mult"] = decision.sl_mult
        state["trailing_activation_mult"] = decision.trailing_activation_mult
        state["trailing_power"] = decision.trailing_power
        state["trailing_squash_divisor"] = decision.trailing_squash_divisor
        state["giveback_beta"] = decision.giveback_beta
        state["atr_power"] = decision.atr_power
        state["atr_multiplier"] = decision.atr_multiplier
        state["hard_tp_abs_pct"] = decision.hard_tp_abs_pct
        state["capital_protect_mfe_mult"] = decision.capital_protect_mfe_mult
        state["capital_protect_regression_frac"] = (
            decision.capital_protect_regression_frac
        )
        state["adverse_exit_enabled"] = decision.adverse_exit_enabled
        state["adverse_exit_theta"] = decision.adverse_exit_theta
        state["adverse_exit_theta_quantile"] = decision.adverse_exit_theta_quantile
        state["adverse_exit_min_mae_atr"] = decision.adverse_exit_min_mae_atr
        state["adverse_exit_min_speed"] = decision.adverse_exit_min_speed
        state["adverse_exit_fast_bars"] = decision.adverse_exit_fast_bars
        state["adverse_exit_max_mfe_atr"] = decision.adverse_exit_max_mfe_atr
        state["decision_module"] = decision.decision_module
        state["requested_policy_stop"] = stop_price_value
        self._replace_stop_order_from_decision(symbol, state, decision)

    def _replace_stop_order_from_decision(
        self,
        symbol: str,
        state: Dict[str, Any],
        decision: SimplePolicyStopDecision,
    ):
        """Cancel/replace a stop from a validated simple-policy decision only."""
        artifact_params = self.get_simple_policy_stop_params(
            str(state.get("strategy_id") or state.get("bucket_key") or "")
        )
        valid_decision, invalid_reason = _validate_policy_stop_decision(
            decision,
            require_should_replace=True,
            position_state=state,
            artifact_params=artifact_params,
        )
        requested_stop = _safe_float(
            (
                decision.stop_price
                if isinstance(decision, SimplePolicyStopDecision)
                else None
            ),
            default=np.nan,
        )
        if not valid_decision:
            state["stop_update_error"] = invalid_reason
            state["stop_update_error_category"] = "unauthorised_stop_update"
            _append_position_event(
                state,
                "stop_replace_skipped",
                reason="invalid_or_missing_policy_decision",
                candidate_stop=(
                    float(requested_stop) if np.isfinite(requested_stop) else None
                ),
                error=invalid_reason,
            )
            return

        old_stop_price = _safe_float(
            state.get("policy_stop_price", state.get("stop_price")), default=np.nan
        )
        old_exchange_stop_price = _safe_float(
            state.get(
                "exchange_stop_price",
                state.get("final_placed_stop", state.get("stop_price")),
            ),
            default=np.nan,
        )
        side = str(state.get("side", "long")).lower()
        canceled_existing = False
        try:
            stop_price = _exchange_precision(
                self.exchange, symbol, float(decision.stop_price), kind="price"
            )
            current_price = np.nan
            current_price_source = "unavailable"
            ticker: Dict[str, Any] = {}
            try:
                ticker = self.exchange.fetch_ticker(symbol)
                current_price, current_price_source = _stop_trigger_reference_price(
                    self.exchange, ticker, self.config, position_side=side
                )
            except Exception as price_exc:
                _append_position_event(
                    state,
                    "stop_replace_price_fetch_failed",
                    attempt=0,
                    error_category=_classify_exchange_error(price_exc),
                    error=str(price_exc),
                )
            if np.isfinite(current_price) and current_price > 0.0:
                adjusted_stop, adjusted, boundary = (
                    _adjust_stop_to_min_current_distance(
                        side, float(stop_price), float(current_price)
                    )
                )
                if adjusted:
                    adjusted_stop = _exchange_precision(
                        self.exchange, symbol, adjusted_stop, kind="price"
                    )
                    _append_position_event(
                        state,
                        "simple_policy_stop_min_current_distance_adjusted",
                        attempt=0,
                        original_stop=float(stop_price),
                        adjusted_stop=float(adjusted_stop),
                        current_price=float(current_price),
                        current_price_source=current_price_source,
                        min_distance_pct=STOP_MIN_CURRENT_DISTANCE_PCT,
                        boundary=float(boundary),
                        stop_reason=getattr(decision, "reason", None),
                        reason_detail=getattr(decision, "reason_detail", None),
                        params_source=getattr(decision, "params_source", None),
                        params_hash=getattr(decision, "params_hash", None),
                    )
                    stop_price = float(adjusted_stop)
                    state["final_placed_stop"] = float(stop_price)
                if _stop_side_is_valid(side, stop_price, current_price):
                    pass
                else:
                    original_stop_price = float(stop_price)
                    recompute_stop_price = np.nan
                    recompute_reason = None
                    recompute_error = None
                    recompute_decision: Optional[SimplePolicyStopDecision] = None
                    try:
                        live_bar = pd.DataFrame(
                            [
                                {
                                    "open": float(current_price),
                                    "high": float(current_price),
                                    "low": float(current_price),
                                    "close": float(current_price),
                                }
                            ],
                            index=[pd.Timestamp.utcnow()],
                        )
                        recompute_state = dict(state)
                        recompute_state["current_price"] = float(current_price)
                        recompute_state["last_price"] = float(current_price)
                        recompute_decision = compute_simple_policy_stop_decision(
                            state=recompute_state,
                            latest_market_state=live_bar,
                            policy_params=artifact_params,
                            side=side,
                            require_metadata=True,
                        )
                        recompute_valid, recompute_reason = (
                            _validate_policy_stop_decision(
                                recompute_decision,
                                require_should_replace=True,
                                position_state=state,
                                artifact_params=artifact_params,
                            )
                        )
                        if recompute_valid:
                            recompute_stop_price = _exchange_precision(
                                self.exchange,
                                symbol,
                                float(recompute_decision.stop_price),
                                kind="price",
                            )
                            (
                                recompute_stop_price,
                                recompute_adjusted,
                                recompute_boundary,
                            ) = _adjust_stop_to_min_current_distance(
                                side,
                                float(recompute_stop_price),
                                float(current_price),
                            )
                            if recompute_adjusted:
                                recompute_stop_price = _exchange_precision(
                                    self.exchange,
                                    symbol,
                                    float(recompute_stop_price),
                                    kind="price",
                                )
                                _append_position_event(
                                    state,
                                    "simple_policy_recomputed_stop_min_current_distance_adjusted",
                                    attempt=0,
                                    adjusted_stop=float(recompute_stop_price),
                                    current_price=float(current_price),
                                    current_price_source=current_price_source,
                                    min_distance_pct=STOP_MIN_CURRENT_DISTANCE_PCT,
                                    boundary=float(recompute_boundary),
                                )
                    except Exception as recompute_exc:
                        recompute_error = str(recompute_exc)

                    if (
                        recompute_decision is not None
                        and np.isfinite(recompute_stop_price)
                        and recompute_stop_price > 0.0
                        and _stop_side_is_valid(
                            side, recompute_stop_price, current_price
                        )
                    ):
                        _append_position_event(
                            state,
                            "simple_policy_stop_recomputed_after_invalid_candidate",
                            attempt=0,
                            original_candidate_stop=original_stop_price,
                            recomputed_stop=float(recompute_stop_price),
                            current_price=float(current_price),
                            current_price_source=current_price_source,
                            stop_reason=recompute_decision.reason,
                            reason_detail=recompute_decision.reason_detail,
                            params_source=recompute_decision.params_source,
                            params_hash=recompute_decision.params_hash,
                        )
                        decision = recompute_decision
                        stop_price = float(recompute_stop_price)
                    else:
                        state["stop_update_error"] = "LOCAL_STOP_SIDE_INVALID"
                        state["stop_update_error_category"] = (
                            "policy_stop_rejected_by_exchange"
                        )
                        state["stop_update_reject_reason"] = "LOCAL_STOP_SIDE_INVALID"
                        _append_position_event(
                            state,
                            "simple_policy_stop_rejected_by_exchange",
                            attempt=0,
                            error_category="policy_stop_rejected_by_exchange",
                            reject_reason="LOCAL_STOP_SIDE_INVALID",
                            current_price=float(current_price),
                            current_price_source=current_price_source,
                            candidate_stop=original_stop_price,
                            recomputed_stop=(
                                float(recompute_stop_price)
                                if np.isfinite(recompute_stop_price)
                                else None
                            ),
                            recompute_validation_error=recompute_reason,
                            recompute_error=recompute_error,
                        )
                        existing_stop_price = _safe_float(
                            state.get("stop_price"), np.nan
                        )
                        existing_stop_order_id = state.get("stop_order_id")
                        fallback_policy_stop = _safe_float(
                            getattr(decision, "requested_policy_stop", None),
                            default=np.nan,
                        )
                        if not np.isfinite(fallback_policy_stop):
                            fallback_policy_stop = _safe_float(
                                requested_stop, default=np.nan
                            )
                        if not np.isfinite(fallback_policy_stop):
                            fallback_policy_stop = original_stop_price
                        (
                            fallback_stop_price,
                            fallback_boundary,
                            fallback_min_distance_pct,
                            fallback_mode,
                        ) = _exchange_valid_giveback_fallback_stop(
                            self.exchange,
                            symbol,
                            side=side,
                            policy_stop=float(fallback_policy_stop),
                            existing_stop=float(existing_stop_price),
                            current_price=float(current_price),
                            config=self.config,
                        )
                        if np.isfinite(fallback_stop_price):
                            entry_price = _safe_float(
                                state.get("entry_price"), default=np.nan
                            )
                            profit_locking = False
                            if np.isfinite(entry_price) and entry_price > 0.0:
                                profit_locking = (
                                    fallback_stop_price > entry_price
                                    if side == "long"
                                    else fallback_stop_price < entry_price
                                )
                            fallback_detail = (
                                "exchange_valid_giveback_fallback: "
                                f"policy_stop={float(fallback_policy_stop):.8g} "
                                f"fallback_stop={float(fallback_stop_price):.8g} "
                                f"existing_stop={float(existing_stop_price):.8g} "
                                f"current={float(current_price):.8g} "
                                f"boundary={float(fallback_boundary):.8g} "
                                f"min_distance_pct={float(fallback_min_distance_pct):.6g} "
                                f"mode={fallback_mode} "
                                f"profit_locking={bool(profit_locking)}"
                            )
                            _append_position_event(
                                state,
                                "exchange_valid_giveback_fallback_stop",
                                original_candidate_stop=original_stop_price,
                                policy_stop=float(fallback_policy_stop),
                                fallback_stop=float(fallback_stop_price),
                                existing_stop=float(existing_stop_price),
                                current_price=float(current_price),
                                current_price_source=current_price_source,
                                boundary=(
                                    float(fallback_boundary)
                                    if np.isfinite(fallback_boundary)
                                    else None
                                ),
                                min_distance_pct=(
                                    float(fallback_min_distance_pct)
                                    if np.isfinite(fallback_min_distance_pct)
                                    else None
                                ),
                                fallback_mode=fallback_mode,
                                profit_locking=bool(profit_locking),
                                recomputed_stop=(
                                    float(recompute_stop_price)
                                    if np.isfinite(recompute_stop_price)
                                    else None
                                ),
                                recompute_validation_error=recompute_reason,
                                recompute_error=recompute_error,
                            )
                            tprint(
                                f"Using exchange-valid giveback fallback for {symbol}: "
                                f"policy={float(fallback_policy_stop):.8g} "
                                f"fallback={float(fallback_stop_price):.8g} "
                                f"existing={float(existing_stop_price):.8g} "
                                f"current={float(current_price):.8g} "
                                f"mode={fallback_mode} "
                                f"profit_locking={bool(profit_locking)}"
                            )
                            decision = replace(
                                decision,
                                should_replace=True,
                                stop_price=float(fallback_stop_price),
                                requested_policy_stop=float(fallback_policy_stop),
                                reason="exchange_valid_giveback_fallback",
                                reason_detail=fallback_detail,
                            )
                            stop_price = float(fallback_stop_price)
                            state["stop_reason"] = decision.reason
                            state["stop_reason_detail"] = decision.reason_detail
                            state["requested_policy_stop"] = float(
                                fallback_policy_stop
                            )
                            state["final_placed_stop"] = float(fallback_stop_price)
                        elif (
                            existing_stop_order_id
                            and np.isfinite(existing_stop_price)
                            and _stop_side_is_valid(
                                side, existing_stop_price, current_price
                            )
                        ):
                            tprint(
                                f"Rejected simple-policy SL for {symbol} before cancel: "
                                f"candidate={original_stop_price:.8g} "
                                f"current={current_price:.8g}; "
                                "canonical recompute did not produce a valid replacement; "
                                "keeping existing exchange stop unchanged"
                            )
                            _append_position_event(
                                state,
                                "simple_policy_stop_existing_stop_preserved",
                                reason="candidate_crossed_existing_stop_valid",
                                existing_stop=float(existing_stop_price),
                                existing_stop_order_id=str(existing_stop_order_id),
                                original_candidate_stop=original_stop_price,
                                current_price=float(current_price),
                                current_price_source=current_price_source,
                                recomputed_stop=(
                                    float(recompute_stop_price)
                                    if np.isfinite(recompute_stop_price)
                                    else None
                                ),
                                recompute_validation_error=recompute_reason,
                                recompute_error=recompute_error,
                            )
                            return
                        else:
                            tprint(
                                f"Rejected simple-policy SL for {symbol} before cancel: "
                                f"candidate={original_stop_price:.8g} "
                                f"current={current_price:.8g}; "
                                "canonical recompute did not produce a valid replacement "
                                "and no valid existing exchange stop is known; "
                                "closing position with software policy stop"
                            )
                            _append_position_event(
                                state,
                                "software_policy_stop_close",
                                reason="policy_stop_crossed_no_valid_exchange_stop",
                                original_candidate_stop=original_stop_price,
                                current_price=float(current_price),
                                current_price_source=current_price_source,
                                recomputed_stop=(
                                    float(recompute_stop_price)
                                    if np.isfinite(recompute_stop_price)
                                    else None
                                ),
                                recompute_validation_error=recompute_reason,
                                recompute_error=recompute_error,
                            )
                            self._close_position(
                                symbol,
                                state,
                                float(current_price),
                                "software_policy_stop_close",
                            )
                            return

            policy_stop_price = float(stop_price)
            exchange_stop_price = float(policy_stop_price)
            exchange_stop_meta: Dict[str, Any] = {}
            if (
                _execution_account(self.config) == "perps"
                and _exchange_id(self.exchange) == "krakenfutures"
            ):
                exchange_stop_price, exchange_stop_meta = (
                    _kraken_futures_last_stop_from_executable_stop(
                        ticker,
                        self.config,
                        position_side=side,
                        policy_stop_price=float(policy_stop_price),
                    )
                )
                if np.isfinite(exchange_stop_price) and exchange_stop_price > 0.0:
                    exchange_stop_price = _exchange_precision(
                        self.exchange,
                        symbol,
                        float(exchange_stop_price),
                        kind="price",
                    )
                else:
                    exchange_stop_price = float(policy_stop_price)

            if np.isfinite(old_stop_price) and not _stop_is_at_least_as_protective(
                side,
                float(policy_stop_price),
                float(old_stop_price),
            ):
                state["stop_update_error"] = (
                    "simple-policy stop would loosen current protection"
                )
                state["stop_update_error_category"] = "policy_stop_not_improved"
                _append_position_event(
                    state,
                        "stop_replace_skipped",
                        reason="replacement_stop_would_loosen_current_protection",
                        previous_stop=float(old_stop_price),
                        candidate_stop=float(policy_stop_price),
                        requested_policy_stop=(
                            float(requested_stop) if np.isfinite(requested_stop) else None
                        ),
                        stop_reason=getattr(decision, "reason", None),
                        reason_detail=getattr(decision, "reason_detail", None),
                )
                tprint(
                    f"Skipped SL update for {symbol}: candidate={float(policy_stop_price):.8g} "
                    f"would loosen current stop={float(old_stop_price):.8g}"
                )
                return
            if (
                np.isfinite(old_exchange_stop_price)
                and not _stop_is_at_least_as_protective(
                    side,
                    float(exchange_stop_price),
                    float(old_exchange_stop_price),
                )
            ):
                state["stop_update_error"] = (
                    "spread-adjusted exchange stop would loosen current protection"
                )
                state["stop_update_error_category"] = "exchange_stop_not_improved"
                _append_position_event(
                    state,
                    "stop_replace_skipped",
                    reason="exchange_last_stop_would_loosen_current_protection",
                    previous_exchange_stop=float(old_exchange_stop_price),
                    candidate_exchange_stop=float(exchange_stop_price),
                    candidate_policy_stop=float(policy_stop_price),
                    exchange_stop_adjustment=exchange_stop_meta,
                )
                tprint(
                    f"Skipped SL update for {symbol}: exchange candidate="
                    f"{float(exchange_stop_price):.8g} would loosen current "
                    f"exchange stop={float(old_exchange_stop_price):.8g}"
                )
                return

            existing_stops = _fetch_open_protective_stop_orders(
                self.exchange, symbol=symbol, position_side=side, config=self.config
            )
            for existing in existing_stops:
                existing_id = existing.get("id")
                existing_stop = _order_stop_price(existing)
                if np.isfinite(existing_stop) and abs(
                    float(existing_stop) - float(exchange_stop_price)
                ) <= max(abs(float(exchange_stop_price)) * 1e-4, 1e-8):
                    state["stop_price"] = float(policy_stop_price)
                    state["policy_stop_price"] = float(policy_stop_price)
                    state["exchange_stop_price"] = float(existing_stop)
                    state["final_placed_stop"] = float(existing_stop)
                    state["stop_order_id"] = existing_id
                    state["stop_order_ids"] = [existing_id] if existing_id else []
                    state.pop("stop_update_error", None)
                    state.pop("stop_update_error_category", None)
                    _append_position_event(
                        state,
                        "existing_stop_adopted",
                        stop_order_id=existing_id,
                        stop_price=float(policy_stop_price),
                        exchange_stop_price=float(existing_stop),
                        requested_policy_stop=float(policy_stop_price),
                        exchange_stop_adjustment=exchange_stop_meta,
                    )
                    tprint(
                        f"Adopted existing protective SL for {symbol}: "
                        f"stop_order_id={existing_id} stop={existing_stop:.8g}"
                    )
                    return

            cancel_ids = []
            for key in (
                "stop_order_id",
                "take_profit_order_id",
                "limit_order_id",
                "oco_order_id",
            ):
                order_id = state.get(key)
                if order_id and order_id not in cancel_ids:
                    cancel_ids.append(order_id)
            for order_id in cancel_ids:
                try:
                    self.exchange.cancel_order(
                        order_id, symbol, _cancel_params(self.config)
                    )
                except Exception as exc:
                    category = _classify_exchange_error(exc)
                    text = str(exc).lower()
                    already_done = any(
                        token in text
                        for token in (
                            "not found",
                            "notfound",
                            "unknown order",
                            "filled",
                            "closed",
                        )
                    ) or str(state.get("last_order_status") or "").lower() in {
                        "canceled",
                        "cancelled",
                        "expired",
                        "rejected",
                    }
                    if not already_done:
                        raise RuntimeError(
                            f"cancel failed before stop replace for {symbol}: "
                            f"{category}: {exc}"
                        ) from exc
            for key in ("stop_order_id", "take_profit_order_id", "limit_order_id"):
                state[key] = None
            canceled_existing = bool(cancel_ids)
            extra_cancelled = _cancel_open_protective_stop_orders(
                self.exchange, symbol=symbol, position_side=side, config=self.config
            )
            if extra_cancelled:
                canceled_existing = True
                _append_position_event(
                    state,
                    "stale_protective_stops_cancelled",
                    count=int(extra_cancelled),
                    requested_policy_stop=float(stop_price),
                )

            market = _load_market(self.exchange, symbol)
            amount = _exchange_precision(
                self.exchange, symbol, float(state["size"]), kind="amount"
            )
            _validate_order_filters(
                symbol,
                market,
                amount=amount,
                price=max(float(state.get("entry_price", stop_price)), stop_price),
            )
            new_stop_order = _create_reduce_stop_loss_order(
                self.exchange,
                symbol=symbol,
                side="sell" if state["side"] == "long" else "buy",
                amount=amount,
                stop_price=exchange_stop_price,
                config=self.config,
            )
            state["stop_price"] = policy_stop_price
            state["policy_stop_price"] = policy_stop_price
            state["exchange_stop_price"] = float(exchange_stop_price)
            state["exchange_stop_trigger_reference_source"] = "last"
            state["exchange_stop_adjustment"] = exchange_stop_meta
            state["final_placed_stop"] = float(exchange_stop_price)
            state["stop_order_id"] = new_stop_order.get("id")
            state["stop_order_ids"] = (
                [state["stop_order_id"]] if state.get("stop_order_id") else []
            )
            requested_trigger_signal = (
                _kraken_futures_stop_trigger_signal_for_reduce_side(
                    "sell" if state["side"] == "long" else "buy"
                )
                if _execution_account(self.config) == "perps"
                and _exchange_id(self.exchange) == "krakenfutures"
                else state.get("stop_trigger_signal")
            )
            actual_trigger_signal, _verified_order, trigger_meta = (
                _verify_exchange_stop_trigger_signal(
                    self.exchange,
                    symbol=symbol,
                    order_id=state.get("stop_order_id"),
                    order=new_stop_order,
                    config=self.config,
                )
            )
            if actual_trigger_signal:
                state["stop_trigger_signal"] = actual_trigger_signal
            if (
                requested_trigger_signal
                and actual_trigger_signal
                and str(actual_trigger_signal) != str(requested_trigger_signal)
            ):
                state["stop_trigger_signal_requested"] = str(requested_trigger_signal)
                state["stop_exchange_trigger_mismatch"] = True
                _append_position_event(
                    state,
                    "stop_trigger_signal_mismatch",
                    requested_trigger_signal=str(requested_trigger_signal),
                    exchange_trigger_signal=str(actual_trigger_signal),
                    stop_order_id=state.get("stop_order_id"),
                    trigger_verify_meta=trigger_meta,
                )
            if current_price_source and current_price_source != "unavailable":
                state["stop_trigger_reference_source"] = current_price_source
            state["size"] = amount
            stop_reason = state.get("stop_reason", "stop_replaced")
            stop_detail = state.get("stop_reason_detail", stop_reason)
            state["stop_reason"] = stop_reason
            state["stop_reason_detail"] = stop_detail
            state.pop("stop_update_error", None)
            state.pop("stop_update_error_category", None)
            entry_price = _safe_float(state.get("entry_price"), default=np.nan)
            _append_position_event(
                state,
                "stop_replaced",
                stop_reason=stop_reason,
                previous_stop=old_stop_price,
                previous_exchange_stop=(
                    float(old_exchange_stop_price)
                    if np.isfinite(old_exchange_stop_price)
                    else None
                ),
                new_stop=float(policy_stop_price),
                new_exchange_stop=float(exchange_stop_price),
                requested_policy_stop=state.get("requested_policy_stop"),
                final_placed_stop=float(exchange_stop_price),
                exchange_stop_adjustment=exchange_stop_meta,
                stop_order_id=state.get("stop_order_id"),
                stop_trigger_signal=state.get("stop_trigger_signal"),
                stop_trigger_reference_source=state.get(
                    "stop_trigger_reference_source"
                ),
                strategy_id=state.get("strategy_id"),
                params_source=state.get("stop_policy_params_source"),
                params_hash=state.get("stop_policy_params_hash"),
                schema=state.get("stop_policy_schema"),
                decision_module=state.get("decision_module"),
                trailing_activation_mult=state.get("trailing_activation_mult"),
                trailing_power=state.get("trailing_power"),
                trailing_squash_divisor=state.get("trailing_squash_divisor"),
                giveback_beta=state.get("giveback_beta"),
                capital_protect_mfe_mult=state.get("capital_protect_mfe_mult"),
                capital_protect_regression_frac=state.get(
                    "capital_protect_regression_frac"
                ),
                entry_price=entry_price,
                stop_dev_pct=(
                    (float(stop_price) - entry_price) / max(abs(entry_price), 1e-12)
                    if np.isfinite(entry_price)
                    else np.nan
                ),
                mfe=_safe_float(state.get("mfe"), 0.0),
                mae=_safe_float(state.get("mae"), 0.0),
            )
            tprint(
                f"Updated SL for {symbol} to {stop_price:.8g} "
                f"reason={stop_reason} detail={stop_detail}"
            )
        except Exception as e:
            category = _classify_exchange_error(e)
            reject_reason = _exchange_reject_reason(e)
            if reject_reason == "LOCAL_STOP_SIDE_INVALID" or _is_trigger_price_reject(
                e
            ):
                category = "policy_stop_rejected_by_exchange"
            state["stop_update_error"] = str(e)
            state["stop_update_error_category"] = category
            state["stop_update_reject_reason"] = reject_reason
            _append_position_event(
                state,
                "stop_replace_failed",
                previous_stop=old_stop_price,
                candidate_stop=(
                    float(requested_stop) if np.isfinite(requested_stop) else None
                ),
                error_category=category,
                reject_reason=reject_reason,
                error=str(e),
            )
            tprint(
                f"Error updating SL for {symbol}: category={category} "
                f"reason={reject_reason}: {e}"
            )
            if reject_reason == "LOCAL_STOP_SIDE_INVALID" or _is_trigger_price_reject(
                e
            ):
                try:
                    ticker = self.exchange.fetch_ticker(symbol)
                    current_price, current_price_source = _stop_trigger_reference_price(
                        self.exchange, ticker, self.config, position_side=side
                    )
                    if np.isfinite(current_price) and current_price > 0.0:
                        live_bar = pd.DataFrame(
                            [
                                {
                                    "open": float(current_price),
                                    "high": float(current_price),
                                    "low": float(current_price),
                                    "close": float(current_price),
                                }
                            ],
                            index=[pd.Timestamp.utcnow()],
                        )
                        recompute_state = dict(state)
                        recompute_state["current_price"] = float(current_price)
                        recompute_state["last_price"] = float(current_price)
                        recompute_decision = compute_simple_policy_stop_decision(
                            state=recompute_state,
                            latest_market_state=live_bar,
                            policy_params=artifact_params,
                            side=side,
                            require_metadata=True,
                        )
                        recompute_valid, recompute_reason = (
                            _validate_policy_stop_decision(
                                recompute_decision,
                                require_should_replace=True,
                                position_state=state,
                                artifact_params=artifact_params,
                            )
                        )
                        recompute_stop_price = (
                            _exchange_precision(
                                self.exchange,
                                symbol,
                                float(recompute_decision.stop_price),
                                kind="price",
                            )
                            if recompute_valid
                            else np.nan
                        )
                        if recompute_valid and np.isfinite(recompute_stop_price):
                            (
                                recompute_stop_price,
                                recompute_adjusted,
                                recompute_boundary,
                            ) = _adjust_stop_to_min_current_distance(
                                side,
                                float(recompute_stop_price),
                                float(current_price),
                            )
                            if recompute_adjusted:
                                recompute_stop_price = _exchange_precision(
                                    self.exchange,
                                    symbol,
                                    float(recompute_stop_price),
                                    kind="price",
                                )
                                _append_position_event(
                                    state,
                                    "stop_recompute_after_exchange_reject_min_current_distance_adjusted",
                                    current_price=float(current_price),
                                    current_price_source=current_price_source,
                                    adjusted_stop=float(recompute_stop_price),
                                    min_distance_pct=STOP_MIN_CURRENT_DISTANCE_PCT,
                                    boundary=float(recompute_boundary),
                                )
                        if (
                            recompute_valid
                            and np.isfinite(recompute_stop_price)
                            and recompute_stop_price > 0.0
                            and _stop_side_is_valid(
                                side, recompute_stop_price, current_price
                            )
                            and (
                                not np.isfinite(old_stop_price)
                                or _stop_is_at_least_as_protective(
                                    side,
                                    float(recompute_stop_price),
                                    float(old_stop_price),
                                )
                            )
                        ):
                            amount = _exchange_precision(
                                self.exchange,
                                symbol,
                                float(state["size"]),
                                kind="amount",
                            )
                            retry_order = _create_reduce_stop_loss_order(
                                self.exchange,
                                symbol=symbol,
                                side="sell" if side == "long" else "buy",
                                amount=amount,
                                stop_price=float(recompute_stop_price),
                                config=self.config,
                            )
                            state["stop_price"] = float(recompute_stop_price)
                            state["stop_order_id"] = retry_order.get("id")
                            state["stop_order_ids"] = (
                                [state["stop_order_id"]]
                                if state.get("stop_order_id")
                                else []
                            )
                            actual_trigger_signal = _order_trigger_signal(retry_order)
                            if actual_trigger_signal:
                                state["stop_trigger_signal"] = actual_trigger_signal
                            if current_price_source and current_price_source != "unavailable":
                                state["stop_trigger_reference_source"] = (
                                    current_price_source
                                )
                            state["size"] = amount
                            state["stop_reason"] = recompute_decision.reason
                            state["stop_reason_detail"] = (
                                recompute_decision.reason_detail
                            )
                            state["requested_policy_stop"] = float(recompute_stop_price)
                            state["stop_policy_params_source"] = (
                                recompute_decision.params_source
                            )
                            state["stop_policy_params_hash"] = (
                                recompute_decision.params_hash
                            )
                            state["stop_policy_schema"] = (
                                recompute_decision.params_schema
                            )
                            state["decision_module"] = "simple_policy_stop"
                            state.pop("stop_update_error", None)
                            state.pop("stop_update_error_category", None)
                            state.pop("stop_update_reject_reason", None)
                            _append_position_event(
                                state,
                                "stop_recomputed_and_replaced_after_exchange_reject",
                                previous_stop=old_stop_price,
                                rejected_candidate_stop=(
                                    float(requested_stop)
                                    if np.isfinite(requested_stop)
                                    else None
                                ),
                                recomputed_stop=float(recompute_stop_price),
                                current_price=float(current_price),
                                current_price_source=current_price_source,
                                stop_order_id=state.get("stop_order_id"),
                                stop_trigger_signal=state.get("stop_trigger_signal"),
                                stop_trigger_reference_source=state.get(
                                    "stop_trigger_reference_source"
                                ),
                                stop_reason=recompute_decision.reason,
                                reason_detail=recompute_decision.reason_detail,
                                params_source=recompute_decision.params_source,
                                params_hash=recompute_decision.params_hash,
                                schema=recompute_decision.params_schema,
                            )
                            tprint(
                                f"Recomputed and replaced SL for {symbol} after "
                                f"exchange reject: stop={recompute_stop_price:.8g} "
                                f"reason={recompute_decision.reason}"
                            )
                            return
                        _append_position_event(
                            state,
                            "stop_recompute_after_exchange_reject_skipped",
                            current_price=float(current_price),
                            current_price_source=current_price_source,
                            recomputed_stop=(
                                float(recompute_stop_price)
                                if np.isfinite(recompute_stop_price)
                                else None
                            ),
                            recompute_validation_error=recompute_reason,
                        )
                except Exception as recompute_exc:
                    _append_position_event(
                        state,
                        "stop_recompute_after_exchange_reject_failed",
                        error_category=_classify_exchange_error(recompute_exc),
                        error=str(recompute_exc),
                    )
            if canceled_existing and np.isfinite(old_stop_price):
                try:
                    ticker = self.exchange.fetch_ticker(symbol)
                    current_price, current_price_source = _stop_trigger_reference_price(
                        self.exchange, ticker, self.config, position_side=side
                    )
                    restore_ok = not np.isfinite(current_price)
                    if side == "long" and np.isfinite(current_price):
                        restore_ok = _stop_side_is_valid(
                            side, old_stop_price, current_price
                        )
                    if side == "short" and np.isfinite(current_price):
                        restore_ok = _stop_side_is_valid(
                            side, old_stop_price, current_price
                        )
                    if restore_ok:
                        restored_stop = _exchange_precision(
                            self.exchange, symbol, old_stop_price, kind="price"
                        )
                        restore_order = _create_reduce_stop_loss_order(
                            self.exchange,
                            symbol=symbol,
                            side="sell" if side == "long" else "buy",
                            amount=_exchange_precision(
                                self.exchange,
                                symbol,
                                float(state["size"]),
                                kind="amount",
                            ),
                            stop_price=restored_stop,
                            config=self.config,
                        )
                        state["stop_price"] = old_stop_price
                        state["stop_order_id"] = restore_order.get("id")
                        state["stop_order_ids"] = (
                            [state["stop_order_id"]]
                            if state.get("stop_order_id")
                            else []
                        )
                        actual_trigger_signal = _order_trigger_signal(restore_order)
                        if actual_trigger_signal:
                            state["stop_trigger_signal"] = actual_trigger_signal
                        if current_price_source and current_price_source != "unavailable":
                            state["stop_trigger_reference_source"] = (
                                current_price_source
                            )
                        state["stop_reason_detail"] = (
                            str(state.get("stop_reason_detail") or "")
                            + "; restored_previous_stop_after_replace_failure"
                        )
                        _append_position_event(
                            state,
                            "stop_restored",
                            restored_stop=float(old_stop_price),
                            stop_order_id=state.get("stop_order_id"),
                            stop_trigger_signal=state.get("stop_trigger_signal"),
                            stop_trigger_reference_source=state.get(
                                "stop_trigger_reference_source"
                            ),
                        )
                except Exception as restore_exc:
                    state["stop_restore_error"] = str(restore_exc)
                    state["stop_restore_error_category"] = _classify_exchange_error(
                        restore_exc
                    )
                    tprint(
                        f"CRITICAL: failed to restore previous SL for {symbol}: "
                        f"{state['stop_restore_error_category']}: {restore_exc}"
                    )

    def _reattach_protective_stop(
        self,
        symbol: str,
        state: Dict[str, Any],
        *,
        previous_status: str,
    ) -> Dict[str, Any]:
        """Try to reattach a policy-derived protective STOP_LOSS."""
        if str(previous_status or "").lower() in {"open", "new"} and state.get(
            "stop_order_id"
        ):
            return {
                "success": False,
                "error_category": "stop_order_still_active",
                "error": "reattach skipped because tracked stop order is still active",
            }
        stop_price = _safe_float(state.get("stop_price"), default=np.nan)
        barrier_frac = _safe_float(state.get("barrier_frac"), default=np.nan)
        sl_mult = _safe_float(state.get("sl_mult"), default=np.nan)
        required_text = (
            "stop_policy_params_source",
            "stop_policy_params_hash",
            "stop_policy_schema",
            "strategy_id",
        )
        missing_provenance = [
            key for key in required_text if not str(state.get(key) or "").strip()
        ]
        if str(state.get("stop_policy_schema") or "") != "simple_policy_v1":
            missing_provenance.append("stop_policy_schema=simple_policy_v1")
        if not np.isfinite(barrier_frac) or barrier_frac <= 0.0:
            missing_provenance.append("barrier_frac")
        if not np.isfinite(sl_mult) or sl_mult <= 0.0:
            missing_provenance.append("sl_mult")
        if not np.isfinite(stop_price) or stop_price <= 0.0:
            missing_provenance.append("stop_price")
        if missing_provenance:
            return {
                "success": False,
                "error_category": "missing_policy_provenance",
                "error": (
                    "cannot reattach stop without policy provenance: "
                    + ",".join(missing_provenance)
                ),
            }

        before_id = state.get("stop_order_id")
        _append_position_event(
            state,
            "stop_reattach_attempt",
            previous_stop_order_id=before_id,
            previous_status=previous_status,
            candidate_stop=float(stop_price),
        )
        reattach_decision = SimplePolicyStopDecision(
            should_replace=True,
            stop_price=float(stop_price),
            reason="policy_stop_reattach",
            reason_detail="reattach policy-derived protective stop",
            strategy_id=str(state.get("strategy_id") or ""),
            params_source=str(state.get("stop_policy_params_source") or ""),
            params_hash=str(state.get("stop_policy_params_hash") or ""),
            barrier_frac=float(barrier_frac),
            sl_mult=float(sl_mult),
            requested_policy_stop=float(stop_price),
            params_schema=str(state.get("stop_policy_schema") or ""),
        )
        for existing in _fetch_open_protective_stop_orders(
            self.exchange,
            symbol=symbol,
            position_side=str(state.get("side", "long")).lower(),
            config=self.config,
        ):
            existing_id = existing.get("id")
            existing_stop = _order_stop_price(existing)
            trigger_matches_policy = _protective_stop_trigger_matches_policy(
                self.exchange,
                existing,
                self.config,
                position_side=str(state.get("side", "long")).lower(),
            )
            if (
                not trigger_matches_policy
                and not _stop_is_at_least_as_protective(
                    str(state.get("side", "long")).lower(),
                    float(existing_stop),
                    float(stop_price),
                )
            ):
                continue
            if _stop_is_at_least_as_protective(
                str(state.get("side", "long")).lower(),
                float(existing_stop),
                float(stop_price),
            ):
                state["stop_order_id"] = existing_id
                state["stop_order_ids"] = [existing_id] if existing_id else []
                state["stop_price"] = float(existing_stop)
                actual_trigger_signal = _order_trigger_signal(existing)
                if actual_trigger_signal:
                    state["stop_trigger_signal"] = actual_trigger_signal
                state.pop("stop_update_error", None)
                state.pop("stop_update_error_category", None)
                _append_position_event(
                    state,
                    "existing_stop_adopted_on_reattach",
                    previous_stop_order_id=before_id,
                    stop_order_id=existing_id,
                    stop_price=float(existing_stop),
                    stop_trigger_signal=state.get("stop_trigger_signal"),
                )
                return {
                    "success": True,
                    "stop_order_id": existing_id,
                    "stop_price": float(existing_stop),
                    "stop_trigger_signal": state.get("stop_trigger_signal"),
                    "stop_trigger_reference_source": state.get(
                        "stop_trigger_reference_source"
                    ),
                    "adopted_existing_stop": True,
                }
        cancelled = _cancel_open_protective_stop_orders(
            self.exchange,
            symbol=symbol,
            position_side=str(state.get("side", "long")).lower(),
            config=self.config,
        )
        if cancelled:
            _append_position_event(
                state,
                "stale_protective_stops_cancelled_before_reattach",
                count=int(cancelled),
                candidate_stop=float(stop_price),
            )
        self._replace_stop_order_from_decision(symbol, state, reattach_decision)
        after_id = state.get("stop_order_id")
        success = bool(after_id) and str(after_id) != str(before_id)
        _append_position_event(
            state,
            "stop_reattach_result",
            previous_stop_order_id=before_id,
            stop_order_id=after_id,
            previous_status=previous_status,
            success=success,
            stop_price=state.get("stop_price"),
            error_category=state.get("stop_update_error_category"),
            error=state.get("stop_update_error"),
        )
        return {
            "success": success,
            "stop_order_id": after_id,
            "stop_price": state.get("stop_price"),
            "error_category": state.get("stop_update_error_category"),
            "error": state.get("stop_update_error"),
        }

    def _close_position(
        self, symbol: str, state: Dict[str, Any], current_price: float, reason: str
    ):
        """Close position and remove from tracking.

        Args:
            symbol: Trading symbol
            state: Position state dictionary
            current_price: Current market price
            reason: Reason for closing (e.g., "stop_loss")
        """
        close_success = False
        try:
            # Cancel any existing orders
            if state.get("stop_order_id"):
                try:
                    self.exchange.cancel_order(
                        state["stop_order_id"], symbol, _cancel_params(self.config)
                    )
                except Exception:
                    pass
            cancelled_extra = _cancel_open_protective_stop_orders(
                self.exchange,
                symbol=symbol,
                position_side=str(state.get("side", "long")).lower(),
                config=self.config,
            )
            if cancelled_extra:
                _append_position_event(
                    state,
                    "protective_stops_cancelled_before_close",
                    count=int(cancelled_extra),
                )

            # Market close
            amount = _exchange_precision(
                self.exchange,
                symbol,
                float(state["size"]),
                kind="amount",
            )
            close_order = _create_reduce_market_order(
                self.exchange,
                symbol=symbol,
                side="sell" if state["side"] == "long" else "buy",
                amount=amount,
                price=float(current_price),
                config=self.config,
            )
            exit_price, filled_amount, _partial_fill = _extract_order_fill(
                close_order, current_price
            )
            if not np.isfinite(exit_price) or exit_price <= 0.0:
                exit_price = float(current_price)
            if not np.isfinite(filled_amount) or filled_amount <= 0.0:
                filled_amount = _safe_float(state.get("size"), 0.0)
            close_order["average"] = exit_price
            close_order["price"] = exit_price
            close_order.setdefault("filled", filled_amount)

            tprint(f"Closed {symbol} at {exit_price:.4f}, reason: {reason}")

            # Log trade result
            pnl = 0
            if state["side"] == "long":
                pnl = (exit_price - state["entry_price"]) * filled_amount
            else:
                pnl = (state["entry_price"] - exit_price) * filled_amount

            tprint(f"  PnL: {pnl:.2f}, MFE: {state['mfe']*100:.2f}%")
            _append_position_event(
                state,
                "position_closed",
                reason=reason,
                close_price=float(exit_price),
                gross_pnl=float(pnl),
                mfe=_safe_float(state.get("mfe"), 0.0),
                mae=_safe_float(state.get("mae"), 0.0),
                stop_reason=state.get("stop_reason"),
            )
            state["last_close_metrics"] = _closed_trade_metrics(
                symbol, state, close_order, reason=reason
            )
            close_success = True

        except Exception as e:
            category = _classify_exchange_error(e)
            state["close_error"] = str(e)
            state["close_error_category"] = category
            tprint(f"Error closing {symbol}: {category}: {e}")
        finally:
            # Keep failed closes tracked so the next monitor cycle can retry.
            if close_success:
                with self._positions_lock:
                    if symbol in self.active_positions:
                        del self.active_positions[symbol]

    def get_active_positions(self) -> Dict[str, Dict[str, Any]]:
        """Get all active positions."""
        with self._positions_lock:
            return self.active_positions.copy()

    def fetch_5m_ohlcv_for_positions(self) -> Dict[str, pd.DataFrame]:
        """Fetch 5m OHLCV data for all active positions.

        Uses hf_data_loader.fetch_ohlcv_5m() to get high-frequency
        data for trailing profit analysis.

        Returns:
            Dictionary mapping symbol to 5m OHLCV DataFrame
        """
        results = {}
        current_time = pd.Timestamp.now(tz="UTC")

        with self._positions_lock:
            items = list(self.active_positions.items())
        for symbol, state in items:
            try:
                entry_time = state.get("entry_time", current_time)

                # Fetch 5m OHLCV from 1 hour before entry to now + 1 hour
                ohlcv_5m = hf_data_loader.fetch_ohlcv_5m(
                    self.exchange,
                    symbol,
                    entry_time - pd.Timedelta(hours=1),
                    current_time + pd.Timedelta(hours=1),
                )

                # Safely check ohlcv_5m
                try:
                    ohlcv_not_empty = (
                        ohlcv_5m is not None
                        and isinstance(ohlcv_5m, (pd.DataFrame, pd.Series))
                        and not (hasattr(ohlcv_5m, "empty") and ohlcv_5m.empty)
                    )
                except Exception:
                    ohlcv_not_empty = False

                if ohlcv_not_empty:
                    results[symbol] = ohlcv_5m
                    # Also store in position state
                    state["ohlcv_5m"] = ohlcv_5m
                else:
                    tprint(f"Warning: No 5m OHLCV data fetched for {symbol}")

            except Exception as e:
                tprint(
                    f"Error fetching 5m OHLCV for {symbol}: "
                    f"{_classify_exchange_error(e)}: {e}"
                )

        return results

    def monitor_order_statuses(self) -> Dict[str, Dict[str, Any]]:
        """Fetch current stop-order status for all tracked positions."""
        statuses: Dict[str, Dict[str, Any]] = {}
        with self._positions_lock:
            items = list(self.active_positions.items())
        for symbol, state in items:
            stop_order_id = state.get("stop_order_id")
            raw_stop_order_ids = state.get("stop_order_ids")
            stop_order_ids = [
                str(order_id)
                for order_id in (
                    raw_stop_order_ids
                    if isinstance(raw_stop_order_ids, (list, tuple, set))
                    else ([stop_order_id] if stop_order_id else [])
                )
                if order_id not in (None, "")
            ]
            if not stop_order_ids:
                statuses[symbol] = {"status": "missing_stop_order"}
                continue
            if len(stop_order_ids) > 1:
                try:
                    fetched_orders: List[Dict[str, Any]] = []
                    fetch_metas: List[Dict[str, Any]] = []
                    for order_id in stop_order_ids:
                        order, fetch_meta = _fetch_order_with_list_fallback(
                            self.exchange,
                            order_id=order_id,
                            symbol=symbol,
                            config=self.config,
                        )
                        fetched_orders.append(order)
                        fetch_metas.append(fetch_meta)
                    order_statuses = [
                        str(order.get("status", "") or "").lower()
                        for order in fetched_orders
                    ]
                    terminal_filled = {"closed", "filled"}
                    terminal_bad = {"canceled", "cancelled", "expired", "rejected"}
                    state["last_order_status"] = ",".join(order_statuses)
                    state["last_order_check_ts"] = pd.Timestamp.now(tz="UTC")
                    stop_coverage = 0.0
                    for order in fetched_orders:
                        remaining = _order_remaining_amount(order)
                        if np.isfinite(remaining):
                            stop_coverage += float(remaining)
                    if all(status in terminal_filled for status in order_statuses):
                        representative_order = fetched_orders[0]
                        state["exit_reason"] = "stop_loss_filled"
                        exit_price, filled_amount, _partial_fill = _extract_order_fill(
                            representative_order,
                            _safe_float(state.get("stop_price"), np.nan),
                        )
                        _append_position_event(
                            state,
                            "stop_order_filled",
                            stop_order_id=",".join(stop_order_ids),
                            stop_price=state.get("stop_price"),
                            fill_price=exit_price,
                            filled=filled_amount,
                            stop_reason=state.get("stop_reason"),
                            order_status=",".join(order_statuses),
                        )
                        statuses[symbol] = {
                            "status": "closed",
                            "orders": fetched_orders,
                            "stop_order_ids": stop_order_ids,
                            "stop_order_coverage": float(stop_coverage),
                            "closed_trade": _closed_trade_metrics(
                                symbol,
                                state,
                                representative_order,
                                reason="stop_loss_filled",
                            ),
                        }
                        with self._positions_lock:
                            self.active_positions.pop(symbol, None)
                        continue
                    if any(status in terminal_bad for status in order_statuses):
                        state["stop_order_error"] = "multi_stop_order_" + ",".join(
                            order_statuses
                        )
                        repair = self._reattach_protective_stop(
                            symbol, state, previous_status="multi_stop_order_terminal"
                        )
                        statuses[symbol] = {
                            "status": (
                                "open"
                                if repair.get("success")
                                else "unprotected_stop_multi_terminal"
                            ),
                            "orders": fetched_orders,
                            "stop_order_ids": stop_order_ids,
                            "stop_order_coverage": float(stop_coverage),
                            "stop_repair": repair,
                        }
                        continue
                    status = "open" if "open" in order_statuses else "unknown"
                    statuses[symbol] = {
                        "status": status,
                        "orders": fetched_orders,
                        "order_statuses": order_statuses,
                        "stop_order_ids": stop_order_ids,
                        "stop_order_coverage": float(stop_coverage),
                    }
                    if any(
                        bool(meta.get("reconciled_after_error")) for meta in fetch_metas
                    ):
                        statuses[symbol]["reconciled_after_error"] = True
                    resolved_via = sorted(
                        {
                            str(meta.get("resolved_via"))
                            for meta in fetch_metas
                            if meta.get("resolved_via")
                        }
                    )
                    if resolved_via:
                        statuses[symbol]["resolved_via"] = ",".join(resolved_via)
                    continue
                except Exception as exc:
                    category = _classify_exchange_error(exc)
                    statuses[symbol] = {
                        "status": "error",
                        "error_category": category,
                        "error": str(exc),
                        "stop_order_ids": stop_order_ids,
                    }
                    tprint(
                        f"Error monitoring stop orders for {symbol}: "
                        f"{category}: {exc}"
                    )
                    continue
            stop_order_id = stop_order_ids[0]
            try:
                order, fetch_meta = _fetch_order_with_list_fallback(
                    self.exchange,
                    order_id=stop_order_id,
                    symbol=symbol,
                    config=self.config,
                )
                status = str(order.get("status", "") or "").lower()
                state["last_order_status"] = status
                state["last_order_check_ts"] = pd.Timestamp.now(tz="UTC")
                statuses[symbol] = {
                    "status": status or "unknown",
                    "order": order,
                }
                statuses[symbol].update(fetch_meta)
                if status in {"closed", "filled"}:
                    state["exit_reason"] = "stop_loss_filled"
                    exit_price, filled_amount, _partial_fill = _extract_order_fill(
                        order, _safe_float(state.get("stop_price"), np.nan)
                    )
                    _append_position_event(
                        state,
                        "stop_order_filled",
                        stop_order_id=stop_order_id,
                        stop_price=state.get("stop_price"),
                        fill_price=exit_price,
                        filled=filled_amount,
                        stop_reason=state.get("stop_reason"),
                        order_status=status,
                    )
                    statuses[symbol]["closed_trade"] = _closed_trade_metrics(
                        symbol, state, order, reason="stop_loss_filled"
                    )
                    with self._positions_lock:
                        self.active_positions.pop(symbol, None)
                elif status in {"canceled", "cancelled", "expired", "rejected"}:
                    state["stop_order_error"] = f"stop_order_{status}"
                    repair = self._reattach_protective_stop(
                        symbol, state, previous_status=status
                    )
                    statuses[symbol]["stop_repair"] = repair
                    if repair.get("success"):
                        statuses[symbol]["status"] = "open"
                        statuses[symbol]["repaired_after_status"] = status
                        statuses[symbol]["stop_order_id"] = repair.get("stop_order_id")
                    else:
                        statuses[symbol]["status"] = f"unprotected_stop_{status}"
            except Exception as exc:
                category = _classify_exchange_error(exc)
                if category == "unsupported_exchange_method":
                    try:
                        open_positions = _fetch_open_exchange_positions(
                            self.exchange, self.config
                        )
                    except Exception as pos_exc:
                        open_positions = {}
                        statuses[symbol] = {
                            "status": "error",
                            "error_category": category,
                            "error": str(exc),
                            "position_reconcile_error_category": _classify_exchange_error(
                                pos_exc
                            ),
                            "position_reconcile_error": str(pos_exc),
                        }
                        tprint(
                            f"Error monitoring stop order for {symbol}: "
                            f"{category}: {exc}; position reconciliation failed: "
                            f"{_classify_exchange_error(pos_exc)}: {pos_exc}"
                        )
                        continue
                    if symbol not in open_positions:
                        close_mode = _position_absent_reconciliation_mode(state)
                        state["exit_reason"] = close_mode
                        state["reconciliation_mode"] = close_mode
                        _append_position_event(
                            state,
                            "exchange_position_absent_after_stop_lookup_failure",
                            stop_order_id=stop_order_id,
                            fetch_order_error=str(exc),
                            fetch_order_error_category=category,
                            reason=close_mode,
                            reconciliation_mode=close_mode,
                        )
                        statuses[symbol] = {
                            "status": "closed",
                            "closed_via": close_mode,
                            "reconciliation_mode": close_mode,
                            "fetch_order_error_category": category,
                            "fetch_order_error": str(exc),
                            "stop_order_id": stop_order_id,
                        }
                        with self._positions_lock:
                            self.active_positions.pop(symbol, None)
                        continue
                    repair = self._reattach_protective_stop(
                        symbol, state, previous_status="stop_order_missing_from_lists"
                    )
                    statuses[symbol] = {
                        "status": (
                            "open"
                            if repair.get("success")
                            else "unprotected_stop_missing_from_lists"
                        ),
                        "stop_repair": repair,
                        "stop_order_id": stop_order_id,
                    }
                    if repair.get("success"):
                        statuses[symbol]["reconciled_after_error"] = True
                        statuses[symbol]["fetch_order_error_category"] = category
                        statuses[symbol]["fetch_order_error"] = str(exc)
                    else:
                        statuses[symbol]["error_category"] = category
                        statuses[symbol]["error"] = str(exc)
                    continue
                statuses[symbol] = {
                    "status": "error",
                    "error_category": category,
                    "error": str(exc),
                }
                tprint(f"Error monitoring stop order for {symbol}: {category}: {exc}")
        return statuses

    def close_all_positions(self):
        """Close all active positions."""
        with self._positions_lock:
            symbols = list(self.active_positions.keys())
        for symbol in symbols:
            try:
                ticker = self.exchange.fetch_ticker(symbol)
                current_price = float(ticker["last"])
                with self._positions_lock:
                    state = self.active_positions.get(symbol)
                if state is not None:
                    self._close_position(
                        symbol, state, current_price, "emergency_close"
                    )
            except Exception as e:
                tprint(
                    f"Error emergency closing {symbol}: "
                    f"{_classify_exchange_error(e)}: {e}"
                )


class TradeExecutor:
    """Handles trade execution in live or shadow mode with OCO support."""

    def __init__(
        self,
        mode: str = "shadow",
        exchange: Optional[Any] = None,
        capital: float = 10000.0,
        max_position_size: float = 0.1,
        bucket_params: Optional[Dict[str, Dict[str, Any]]] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize trade executor.

        Args:
            mode: "live" or "shadow"
            exchange: ccxt exchange instance (required for live mode)
            capital: Starting capital
            max_position_size: Maximum position size as fraction of capital
            bucket_params: Parameters from ridge position sizer for SL/TP
            config: Additional configuration options
        """
        self.mode = mode
        self.exchange = exchange
        self.capital = capital
        self.max_position_size = max_position_size
        self.bucket_params = bucket_params or {}
        self.config = config or {}
        self.simple_policy_stop_params_by_strategy = (
            extract_simple_policy_stop_params_by_strategy(self.bucket_params)
        )

        # Track positions
        self.positions = {}
        self._state_lock = threading.RLock()
        self._last_trade_timestamps: Dict[str, pd.Timestamp] = {}

        # Initialize OCO executor for live mode
        self.oco_executor: Optional[OCOExecutor] = None
        if _is_live_execution_mode(mode) and exchange is not None:
            self.oco_executor = OCOExecutor(
                exchange=exchange, bucket_params=self.bucket_params, config=self.config
            )
            tprint(
                "OCOExecutor background monitor disabled; inference loop owns "
                "simple_policy_optimiser 15m stop-policy updates"
            )

        tprint(f"TradeExecutor initialized in {mode} mode")

    def _fetch_margin_balance(self) -> Dict[str, Any]:
        """Fetch cross-margin balances through ccxt with Binance fallbacks."""
        if self.exchange is None:
            return {}
        if _exchange_id(self.exchange) == "krakenfutures":
            attempts = ({"type": "flex"}, {})
        else:
            attempts = (
                {"type": "margin", "marginMode": "cross"},
                {"type": "margin"},
                {},
            )
        last_error: Optional[Exception] = None
        for params in attempts:
            try:
                balance = self.exchange.fetch_balance(params)
                if isinstance(balance, dict):
                    return balance
            except Exception as exc:
                last_error = exc
        if last_error is not None:
            raise last_error
        return {}

    def _asset_quote_value(
        self,
        asset: str,
        amount: float,
        quote: str,
    ) -> Tuple[float, Optional[float], Optional[str]]:
        """Convert an asset quantity to quote value using current tickers."""
        asset_u = str(asset or "").upper()
        quote_u = str(quote or "").upper()
        amount_f = abs(float(amount))
        if amount_f <= 0.0:
            return 0.0, None, None
        if asset_u == quote_u:
            return amount_f, 1.0, quote_u
        symbol = _symbol_from_asset_quote(asset_u, quote_u)
        try:
            _load_market(self.exchange, symbol)
            ticker = self.exchange.fetch_ticker(symbol)
            px = _safe_float(ticker.get("last"), default=np.nan)
            if np.isfinite(px) and px > 0.0:
                return amount_f * float(px), float(px), symbol
        except Exception:
            pass
        return np.nan, None, symbol

    def _parse_margin_assets(
        self, balance: Dict[str, Any]
    ) -> Dict[str, Dict[str, float]]:
        """Normalize ccxt and Binance raw margin balance payloads by asset."""
        assets: Dict[str, Dict[str, float]] = {}

        def _slot(asset: str) -> Dict[str, float]:
            asset_u = str(asset or "").upper()
            return assets.setdefault(
                asset_u,
                {"free": 0.0, "used": 0.0, "total": 0.0, "debt": 0.0, "interest": 0.0},
            )

        for asset, payload in balance.items():
            if asset in {"info", "free", "used", "total"} or not isinstance(
                payload, dict
            ):
                continue
            slot = _slot(asset)
            slot["free"] = max(slot["free"], _safe_float(payload.get("free"), 0.0))
            slot["used"] = max(slot["used"], _safe_float(payload.get("used"), 0.0))
            slot["total"] = max(slot["total"], _safe_float(payload.get("total"), 0.0))
            slot["debt"] = max(
                slot["debt"],
                _safe_float(
                    payload.get("debt", payload.get("borrowed", payload.get("loan"))),
                    0.0,
                ),
            )
            slot["interest"] = max(
                slot["interest"], _safe_float(payload.get("interest"), 0.0)
            )

        info = balance.get("info")
        user_assets = None
        if isinstance(info, dict):
            user_assets = info.get("userAssets") or info.get("userassets")
        if isinstance(user_assets, list):
            for row in user_assets:
                if not isinstance(row, dict):
                    continue
                asset = row.get("asset")
                if not asset:
                    continue
                slot = _slot(asset)
                free = _safe_float(row.get("free"), 0.0)
                locked = _safe_float(row.get("locked"), 0.0)
                borrowed = _safe_float(row.get("borrowed"), 0.0)
                interest = _safe_float(row.get("interest"), 0.0)
                net_asset = _safe_float(row.get("netAsset"), np.nan)
                slot["free"] = max(slot["free"], free)
                slot["used"] = max(slot["used"], locked)
                slot["total"] = max(
                    slot["total"],
                    (
                        free + locked
                        if not np.isfinite(net_asset)
                        else max(free + locked, 0.0)
                    ),
                )
                slot["debt"] = max(slot["debt"], borrowed + interest)
                slot["interest"] = max(slot["interest"], interest)
        return assets

    def _binance_sapi_post(self, path: str, params: Dict[str, Any]) -> Any:
        """Call a Binance SAPI POST endpoint through ccxt implicit methods."""
        if self.exchange is None:
            raise RuntimeError("exchange is not configured")
        explicit_methods = {
            "asset/dust-btc": (
                "sapiPostAssetDustBtc",
                "sapi_post_asset_dust_btc",
            ),
            "asset/dust": ("sapiPostAssetDust", "sapi_post_asset_dust"),
            "margin/dust": ("sapiPostMarginDust", "sapi_post_margin_dust"),
            "margin/exchange-small-liability": (
                "sapiPostMarginExchangeSmallLiability",
                "sapi_post_margin_exchange_small_liability",
            ),
        }
        for explicit_name in explicit_methods.get(str(path), ()):
            explicit_method = getattr(self.exchange, explicit_name, None)
            if callable(explicit_method):
                return explicit_method(params)
        method_name = "sapiPost" + "".join(
            part[:1].upper() + part[1:]
            for chunk in str(path).split("/")
            for part in chunk.split("-")
            if part
        )
        method = getattr(self.exchange, method_name, None)
        if callable(method):
            return method(params)
        request = getattr(self.exchange, "request", None)
        if callable(request):
            return request(path, "sapi", "POST", params)
        raise RuntimeError(f"exchange does not expose Binance SAPI POST {path}")

    def _binance_sapi_post_repeated_params(
        self, path: str, params: Dict[str, Any]
    ) -> Any:
        """Call Binance SAPI POST preserving repeated array keys in signature."""
        if self.exchange is None:
            raise RuntimeError("exchange is not configured")
        fetch = getattr(self.exchange, "fetch", None)
        api_key = str(getattr(self.exchange, "apiKey", "") or "")
        secret = str(getattr(self.exchange, "secret", "") or "")
        if not callable(fetch) or not api_key or not secret:
            return self._binance_sapi_post(path, params)
        milliseconds = getattr(self.exchange, "milliseconds", None)
        timestamp = int(
            milliseconds()
            if callable(milliseconds)
            else pd.Timestamp.now(tz="UTC").timestamp() * 1000
        )
        recv_window = int(
            getattr(self.exchange, "options", {}).get("recvWindow", 10000)
        )
        pairs: List[Tuple[str, Any]] = [("timestamp", timestamp)]
        for key, value in params.items():
            if isinstance(value, (list, tuple, set)):
                for item in value:
                    pairs.append((str(key), item))
            else:
                pairs.append((str(key), value))
        pairs.append(("recvWindow", recv_window))
        query = urlencode(pairs, doseq=True)
        signature = hmac.new(
            secret.encode("utf-8"),
            query.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        base_url = str(
            getattr(self.exchange, "urls", {})
            .get("api", {})
            .get("sapi", "https://api.binance.com/sapi/v1")
        )
        signed = {
            "url": f"{base_url}/{path}?{query}&signature={signature}",
            "method": "POST",
            "headers": {
                "X-MBX-APIKEY": api_key,
                "Content-Type": "application/x-www-form-urlencoded",
            },
            "body": None,
        }
        return fetch(
            signed["url"],
            signed.get("method", "POST"),
            signed.get("headers"),
            signed.get("body"),
        )

    def _binance_sapi_get(
        self, path: str, params: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Call a Binance SAPI GET endpoint through ccxt implicit methods."""
        if self.exchange is None:
            raise RuntimeError("exchange is not configured")
        params = dict(params or {})
        explicit_methods = {
            "margin/exchange-small-liability": (
                "sapiGetMarginExchangeSmallLiability",
                "sapi_get_margin_exchange_small_liability",
            ),
            "margin/exchange-small-liability-history": (
                "sapiGetMarginExchangeSmallLiabilityHistory",
                "sapi_get_margin_exchange_small_liability_history",
            ),
            "margin/dust": ("sapiGetMarginDust", "sapi_get_margin_dust"),
        }
        for explicit_name in explicit_methods.get(str(path), ()):
            explicit_method = getattr(self.exchange, explicit_name, None)
            if callable(explicit_method):
                return explicit_method(params)
        method_name = "sapiGet" + "".join(
            part[:1].upper() + part[1:]
            for chunk in str(path).split("/")
            for part in chunk.split("-")
            if part
        )
        method = getattr(self.exchange, method_name, None)
        if callable(method):
            return method(params)
        request = getattr(self.exchange, "request", None)
        if callable(request):
            return request(path, "sapi", "GET", params)
        raise RuntimeError(f"exchange does not expose Binance SAPI GET {path}")

    @staticmethod
    def _extract_small_liability_assets(response: Any) -> Tuple[List[str], List[str]]:
        """Extract exchangeable source assets and liability targets from Binance."""
        rows: List[Any]
        if isinstance(response, list):
            rows = response
        elif isinstance(response, dict):
            data = response.get("data")
            if isinstance(data, list):
                rows = data
            elif isinstance(response.get("rows"), list):
                rows = response.get("rows", [])
            elif isinstance(response.get("assets"), list):
                rows = response.get("assets", [])
            else:
                rows = []
        else:
            rows = []

        assets: List[str] = []
        liability_assets: List[str] = []
        for row in rows:
            if isinstance(row, str):
                asset = row
                liability_asset = row
            elif isinstance(row, dict):
                asset = (
                    row.get("asset")
                    or row.get("liabilityAsset")
                    or row.get("liability_asset")
                    or row.get("coin")
                )
                liability_asset = (
                    row.get("liabilityAsset")
                    or row.get("liability_asset")
                    or row.get("targetAsset")
                    or row.get("target_asset")
                    or row.get("asset")
                )
                if row.get("exchangeable") is False or row.get("convertible") is False:
                    continue
            else:
                continue
            asset_u = str(asset or "").strip().upper()
            if asset_u and asset_u not in assets:
                assets.append(asset_u)
            liability_u = str(liability_asset or "").strip().upper()
            if liability_u and liability_u not in liability_assets:
                liability_assets.append(liability_u)
        return assets, liability_assets

    def _fetch_exchangeable_small_liability_assets(
        self,
    ) -> Tuple[List[str], List[str], Any]:
        """Return assets Binance currently accepts for small-liability exchange."""
        response = self._binance_sapi_get("margin/exchange-small-liability", {})
        assets, liability_assets = self._extract_small_liability_assets(response)
        return assets, liability_assets, response

    def _margin_repay_amount_variants(self, asset: str, amount: float) -> List[str]:
        """Return safe repay amount strings from finest to coarsest precision."""
        asset_u = str(asset or "").upper()
        amount_f = float(amount)
        if not asset_u or not np.isfinite(amount_f) or amount_f <= 0.0:
            return []
        quote = str(
            self.config.get("quote_currency")
            or self.config.get("live_quote_currency")
            or "USDC"
        ).upper()
        candidates: List[str] = []

        def _add(value: Any) -> None:
            try:
                value_f = float(value)
            except Exception:
                return
            if not np.isfinite(value_f) or value_f <= 0.0 or value_f > amount_f:
                return
            text = f"{value_f:.12f}".rstrip("0").rstrip(".")
            if text and text not in candidates:
                candidates.append(text)

        for symbol in (f"{asset_u}/{quote}", f"{asset_u}/USDC", f"{asset_u}/USDT"):
            try:
                _load_market(self.exchange, symbol)
                amount_to_precision = getattr(
                    self.exchange, "amount_to_precision", None
                )
                if callable(amount_to_precision):
                    _add(amount_to_precision(symbol, amount_f))
            except Exception:
                pass
        for decimals in (8, 6, 5, 4, 3, 2, 1, 0):
            scale = 10**decimals
            floored = np.floor(amount_f * scale) / scale
            _add(floored)
        return candidates

    def _repay_cross_margin_asset(self, asset: str, amount: float) -> Dict[str, Any]:
        """Repay a cross-margin liability for one asset."""
        asset_u = str(asset or "").upper()
        amount_f = float(amount)
        if not asset_u or not np.isfinite(amount_f) or amount_f <= 0.0:
            raise ValueError(f"invalid margin repay request: {asset} {amount}")
        amount_variants = self._margin_repay_amount_variants(asset_u, amount_f)
        if not amount_variants:
            amount_variants = [f"{amount_f:.8f}".rstrip("0").rstrip(".")]
        errors: List[str] = []
        for amount_text in amount_variants:
            params = {"asset": asset_u, "amount": amount_text}
            borrow_repay_params = {
                "asset": asset_u,
                "amount": amount_text,
                "type": "REPAY",
                "isIsolated": "FALSE",
            }
            for method_name in (
                "sapiPostMarginBorrowRepay",
                "sapi_post_margin_borrow_repay",
            ):
                method = getattr(self.exchange, method_name, None)
                if not callable(method):
                    continue
                try:
                    response = method(borrow_repay_params)
                    if isinstance(response, dict):
                        response.setdefault("attempted_repay_amount", amount_text)
                    return response
                except TypeError:
                    try:
                        response = method(**borrow_repay_params)
                        if isinstance(response, dict):
                            response.setdefault("attempted_repay_amount", amount_text)
                        return response
                    except Exception as exc:
                        errors.append(f"{method_name}({amount_text}): {exc}")
                        continue
                except Exception as exc:
                    errors.append(f"{method_name}({amount_text}): {exc}")
                    continue
            for method_name in (
                "repayCrossMargin",
                "repay_cross_margin",
                "repayMargin",
                "repay_margin",
                "sapiPostMarginRepay",
                "sapi_post_margin_repay",
            ):
                method = getattr(self.exchange, method_name, None)
                if not callable(method):
                    continue
                try:
                    if method_name in {
                        "repayCrossMargin",
                        "repay_cross_margin",
                        "repayMargin",
                        "repay_margin",
                    }:
                        response = method(asset_u, float(amount_text), params)
                    else:
                        response = method(params)
                    if isinstance(response, dict):
                        response.setdefault("attempted_repay_amount", amount_text)
                    return response
                except TypeError:
                    try:
                        response = method(params)
                        if isinstance(response, dict):
                            response.setdefault("attempted_repay_amount", amount_text)
                        return response
                    except Exception as exc:
                        errors.append(f"{method_name}({amount_text}): {exc}")
                        continue
                except Exception as exc:
                    errors.append(f"{method_name}({amount_text}): {exc}")
                    continue
            try:
                response = self._binance_sapi_post("margin/repay", params)
                if isinstance(response, dict):
                    response.setdefault("attempted_repay_amount", amount_text)
                return response
            except Exception as exc:
                errors.append(f"margin/repay({amount_text}): {exc}")
        raise RuntimeError("; ".join(errors[-5:]))

    def _try_repay_available_margin_debt(
        self,
        *,
        asset: str,
        free: float,
        debt: float,
        quote_value: float,
        report: Dict[str, Any],
        reason: str,
    ) -> Dict[str, Any]:
        """Repay same-asset debt using currently free asset balance."""
        result: Dict[str, Any] = {
            "asset": str(asset).upper(),
            "reason": reason,
            "free": float(free),
            "debt": float(debt),
            "quote_value": float(quote_value) if np.isfinite(quote_value) else None,
            "attempted": False,
            "success": False,
        }
        report.setdefault("margin_repay_attempts", []).append(result)
        if not bool(
            self.config.get(
                "cross_margin_auto_repay_same_asset_enabled",
                _is_live_execution_mode(self.mode),
            )
        ):
            result["skip_reason"] = "disabled"
            return result
        repay_amount = min(max(float(free), 0.0), max(float(debt), 0.0))
        if repay_amount <= 0.0:
            result["skip_reason"] = "no_free_balance_to_repay"
            return result
        # Leave a small precision buffer so Binance does not reject due rounding.
        repay_amount *= 0.999
        if repay_amount <= 0.0:
            result["skip_reason"] = "repay_amount_after_buffer_zero"
            return result
        result["attempted"] = True
        result["repay_amount"] = float(repay_amount)
        try:
            response = self._repay_cross_margin_asset(asset, repay_amount)
            result["success"] = True
            result["response"] = response
            tprint(
                "Repaid same-asset cross-margin liability: "
                f"asset={asset} amount={repay_amount:.12g} reason={reason}"
            )
        except Exception as exc:
            result["error_category"] = _classify_exchange_error(exc)
            result["error"] = str(exc)
            tprint(
                "Same-asset margin repay skipped/failed: "
                f"asset={asset} category={result['error_category']}: {exc}"
            )
        return result

    def _try_admin_close_external_margin_position(
        self,
        *,
        symbol: str,
        side: str,
        amount: float,
        price: float,
        quote_value: float,
        report: Dict[str, Any],
        reason: str,
    ) -> Dict[str, Any]:
        """Close an unimportable external margin position without fallback stops."""
        result: Dict[str, Any] = {
            "symbol": symbol,
            "side": side,
            "amount": float(amount),
            "price": float(price),
            "quote_value": float(quote_value),
            "reason": reason,
            "attempted": False,
            "success": False,
        }
        report.setdefault("external_position_cleanup_attempts", []).append(result)
        if not bool(
            self.config.get(
                "cross_margin_cleanup_unimported_external_positions_enabled",
                _is_live_execution_mode(self.mode),
            )
        ):
            result["skip_reason"] = "disabled"
            return result
        if not symbol or not np.isfinite(amount) or amount <= 0.0:
            result["skip_reason"] = "invalid_symbol_or_amount"
            return result
        close_side = "sell" if side == "long" else "buy"
        result["attempted"] = True
        result["close_side"] = close_side
        try:
            if side == "long":
                order = _create_margin_market_order_variants(
                    self.exchange,
                    symbol=symbol,
                    side=close_side,
                    amount=float(amount),
                    price=float(price),
                    side_effects=("AUTO_REPAY", "NO_SIDE_EFFECT"),
                    config=self.config,
                )
            else:
                order = _create_margin_market_order_variants(
                    self.exchange,
                    symbol=symbol,
                    side=close_side,
                    amount=float(amount),
                    price=float(price),
                    side_effects=("AUTO_REPAY", "MARGIN_BUY", "AUTO_BORROW_REPAY"),
                    config=self.config,
                )
            result["success"] = True
            result["order_id"] = order.get("id") if isinstance(order, dict) else None
            result["order"] = order
            tprint(
                "Submitted admin cleanup order for unimported external margin "
                f"position: {symbol} side={side} close_side={close_side} "
                f"amount={float(amount):.12g} reason={reason}"
            )
        except Exception as exc:
            result["error_category"] = _classify_exchange_error(exc)
            result["error"] = str(exc)
            tprint(
                "Admin cleanup close failed for unimported external margin "
                f"position {symbol}: {result['error_category']}: {exc}"
            )
        return result

    def _maybe_cleanup_small_margin_positions(
        self,
        *,
        candidate_positions: List[Dict[str, Any]],
        report: Dict[str, Any],
    ) -> None:
        """Try to settle dust-sized margin positions that official dust APIs missed."""
        threshold = float(
            self.config.get("cross_margin_small_position_cleanup_quote_threshold", 1.0)
        )
        result: Dict[str, Any] = {
            "enabled": bool(
                self.config.get(
                    "cross_margin_cleanup_small_positions_enabled",
                    _is_live_execution_mode(self.mode),
                )
            ),
            "threshold_quote": threshold,
            "candidate_count": len(candidate_positions),
            "attempts": [],
        }
        report["small_position_cleanup"] = result
        if not result["enabled"]:
            result["reason"] = "disabled"
            return

        dust_result = report.get("dust_to_bnb", {})
        dust_converted_assets = {
            str(asset).upper()
            for asset in (
                dust_result.get("assets")
                or dust_result.get("converted_assets")
                or dust_result.get("last_assets")
                or []
            )
        }
        liability_result = report.get("small_liability_exchange", {})
        exchanged_assets = {
            str(asset).upper()
            for asset in (
                liability_result.get("assets")
                if liability_result.get("exchanged")
                else []
            )
        }
        max_attempts = int(
            self.config.get("cross_margin_small_position_cleanup_max_assets", 20)
        )
        seen: set[tuple[str, str]] = set()
        for pos in sorted(
            candidate_positions,
            key=lambda row: float(row.get("quote_value") or 0.0),
            reverse=True,
        ):
            if len(result["attempts"]) >= max_attempts:
                result["truncated"] = True
                break
            asset = str(pos.get("asset") or "").upper()
            side = str(pos.get("side") or "").lower()
            symbol = str(pos.get("symbol") or "")
            quote_value = _safe_float(pos.get("quote_value"), default=np.nan)
            amount = _safe_float(pos.get("amount"), default=np.nan)
            price = _safe_float(pos.get("price"), default=np.nan)
            key = (asset, side)
            if key in seen:
                continue
            seen.add(key)
            attempt: Dict[str, Any] = {
                "asset": asset,
                "symbol": symbol,
                "side": side,
                "amount": float(amount) if np.isfinite(amount) else None,
                "price": float(price) if np.isfinite(price) else None,
                "quote_value": float(quote_value) if np.isfinite(quote_value) else None,
                "classification": pos.get("classification"),
                "reason": pos.get("reason"),
                "attempted": False,
                "success": False,
            }
            result["attempts"].append(attempt)
            if not asset or asset in {
                "BNB",
                str(report.get("quote_currency", "")).upper(),
            }:
                attempt["skip_reason"] = "quote_or_fee_asset"
                continue
            if side not in {"long", "short"} or not symbol:
                attempt["skip_reason"] = "missing_symbol_or_side"
                continue
            if not np.isfinite(quote_value) or quote_value <= 0.0:
                attempt["skip_reason"] = "invalid_quote_value"
                continue
            if quote_value > threshold:
                attempt["skip_reason"] = "above_small_position_threshold"
                continue
            if not np.isfinite(amount) or amount <= 0.0:
                attempt["skip_reason"] = "invalid_amount"
                continue
            if side == "long" and asset in dust_converted_assets:
                attempt["skip_reason"] = "already_submitted_to_dust_conversion"
                continue
            if side == "short" and asset in exchanged_assets:
                attempt["skip_reason"] = "already_submitted_to_small_liability_exchange"
                continue
            market = _load_market(self.exchange, symbol)
            min_notional = _market_min_notional(market)
            if np.isfinite(min_notional) and quote_value < min_notional:
                attempt["skip_reason"] = "below_exchange_min_notional_use_liability_api"
                attempt["exchange_min_notional"] = float(min_notional)
                continue
            close_side = "sell" if side == "long" else "buy"
            # Sell slightly less than the free dust balance to avoid precision/balance rejects.
            order_amount = float(amount) * (0.999 if side == "long" else 1.0)
            if order_amount <= 0.0:
                attempt["skip_reason"] = "order_amount_after_buffer_zero"
                continue
            attempt["attempted"] = True
            attempt["close_side"] = close_side
            attempt["order_amount"] = order_amount
            try:
                side_effects = (
                    ("AUTO_REPAY", "NO_SIDE_EFFECT")
                    if side == "long"
                    else ("AUTO_REPAY", "MARGIN_BUY", "AUTO_BORROW_REPAY")
                )
                order = _create_margin_market_order_variants(
                    self.exchange,
                    symbol=symbol,
                    side=close_side,
                    amount=order_amount,
                    price=float(price) if np.isfinite(price) and price > 0.0 else None,
                    side_effects=side_effects,
                    config=self.config,
                )
                attempt["success"] = True
                attempt["order_id"] = (
                    order.get("id") if isinstance(order, dict) else None
                )
                attempt["order"] = order
                tprint(
                    "Submitted small cross-margin position cleanup order: "
                    f"{symbol} side={side} close_side={close_side} "
                    f"quote_value={quote_value:.6g}"
                )
            except Exception as exc:
                attempt["error_category"] = _classify_exchange_error(exc)
                attempt["error"] = str(exc)
                tprint(
                    "Small cross-margin position cleanup skipped/failed: "
                    f"{symbol} {side} {attempt['error_category']}: {exc}"
                )

    def _maybe_convert_margin_dust_to_bnb(
        self,
        *,
        candidate_assets: List[str],
        report: Dict[str, Any],
    ) -> None:
        """Periodically convert positive cross-margin dust balances to BNB."""
        result: Dict[str, Any] = {
            "enabled": bool(
                self.config.get(
                    "cross_margin_dust_to_bnb_enabled",
                    _is_live_execution_mode(self.mode),
                )
            ),
            "candidate_assets": sorted(set(candidate_assets)),
            "converted": False,
        }
        report["dust_to_bnb"] = result
        if not result["enabled"]:
            result["reason"] = "disabled"
            return
        if _exchange_id(self.exchange) != "binance":
            result["reason"] = "unsupported_exchange"
            result["exchange"] = _exchange_id(self.exchange) or "unknown"
            return
        assets = [
            str(asset).upper()
            for asset in sorted(set(candidate_assets))
            if str(asset or "").strip()
        ]
        if not assets:
            result["reason"] = "no_candidate_assets"
            return
        state_path = Path(
            self.config.get(
                "cross_margin_dust_to_bnb_state_path",
                str(
                    Path(str(self.config.get("live_data_root", "data")))
                    / "live_state"
                    / "margin_dust_to_bnb_state.json"
                ),
            )
        )
        interval_hours = float(
            self.config.get("cross_margin_dust_to_bnb_interval_hours", 24.0)
        )
        failure_cooldown_minutes = float(
            self.config.get(
                "cross_margin_dust_to_bnb_failure_cooldown_minutes",
                10.0,
            )
        )
        now_ts = pd.Timestamp.now(tz="UTC")
        try:
            state = json.loads(state_path.read_text()) if state_path.exists() else {}
        except Exception:
            state = {}
        method_version_changed = (
            state.get("method_version") != DUST_TO_BNB_METHOD_VERSION
        )
        last_success_raw = state.get("last_success_ts")
        last_attempt_raw = state.get("last_attempt_ts")
        if last_success_raw and not method_version_changed:
            try:
                last_ts = pd.Timestamp(last_success_raw)
                if last_ts.tzinfo is None:
                    last_ts = last_ts.tz_localize("UTC")
                else:
                    last_ts = last_ts.tz_convert("UTC")
                if now_ts < last_ts + pd.Timedelta(hours=interval_hours):
                    result["reason"] = "cooldown_active"
                    result["last_success_ts"] = last_ts.isoformat()
                    return
            except Exception:
                pass
        if last_attempt_raw and not method_version_changed:
            try:
                last_attempt = pd.Timestamp(last_attempt_raw)
                if last_attempt.tzinfo is None:
                    last_attempt = last_attempt.tz_localize("UTC")
                else:
                    last_attempt = last_attempt.tz_convert("UTC")
                if now_ts < last_attempt + pd.Timedelta(
                    minutes=failure_cooldown_minutes
                ):
                    result["reason"] = "failure_cooldown_active"
                    result["last_attempt_ts"] = last_attempt.isoformat()
                    return
            except Exception:
                pass

        state_path.parent.mkdir(parents=True, exist_ok=True)
        state["last_attempt_ts"] = now_ts.isoformat()
        state["method_version"] = DUST_TO_BNB_METHOD_VERSION
        try:
            state_path.write_text(json.dumps(state, indent=2, sort_keys=True))
        except Exception:
            pass

        try:
            convertible = self._binance_sapi_post(
                "asset/dust-btc",
                {"accountType": "MARGIN"},
            )
            details = (
                convertible.get("details") if isinstance(convertible, dict) else []
            )
            allowed_assets = {
                str(row.get("asset") or "").upper()
                for row in details
                if isinstance(row, dict)
                and bool(row.get("convertible", True))
                and str(row.get("asset") or "").strip()
            }
            if allowed_assets:
                assets = [asset for asset in assets if asset in allowed_assets]
                result["convertible_assets"] = assets
            if not assets:
                result["reason"] = "no_convertible_candidate_assets"
                return
            try:
                response = self._binance_sapi_post_repeated_params(
                    "margin/dust",
                    {"asset": assets},
                )
            except Exception as margin_exc:
                result["margin_dust_error_category"] = _classify_exchange_error(
                    margin_exc
                )
                result["margin_dust_error"] = str(margin_exc)
                response = self._binance_sapi_post_repeated_params(
                    "asset/dust",
                    {"asset": assets, "accountType": "MARGIN"},
                )
            result["converted"] = True
            result["assets"] = assets
            result["response"] = response
            state["last_success_ts"] = now_ts.isoformat()
            state["last_assets"] = assets
            try:
                state_path.write_text(json.dumps(state, indent=2, sort_keys=True))
            except Exception:
                pass
            tprint(
                "Converted cross-margin dust to BNB: "
                f"assets={assets} accountType=MARGIN"
            )
        except Exception as exc:
            result["error_category"] = _classify_exchange_error(exc)
            result["error"] = str(exc)
            tprint(
                "Cross-margin dust-to-BNB conversion skipped/failed: "
                f"{result['error_category']}: {exc}"
            )

    def _maybe_exchange_small_margin_liabilities(
        self,
        *,
        candidate_assets: List[str],
        report: Dict[str, Any],
    ) -> None:
        """Periodically clear tiny cross-margin liabilities via Binance small exchange."""
        result: Dict[str, Any] = {
            "enabled": bool(
                self.config.get(
                    "cross_margin_small_liability_exchange_enabled",
                    _is_live_execution_mode(self.mode),
                )
            ),
            "candidate_assets": sorted(set(candidate_assets)),
            "exchanged": False,
        }
        report["small_liability_exchange"] = result
        if not result["enabled"]:
            result["reason"] = "disabled"
            return
        if _exchange_id(self.exchange) != "binance":
            result["reason"] = "unsupported_exchange"
            result["exchange"] = _exchange_id(self.exchange) or "unknown"
            return
        assets = [
            str(asset).upper()
            for asset in sorted(set(candidate_assets))
            if str(asset or "").strip()
        ][:10]
        if not assets:
            result["reason"] = "no_candidate_assets"
            return
        state_path = Path(
            self.config.get(
                "cross_margin_small_liability_exchange_state_path",
                str(
                    Path(str(self.config.get("live_data_root", "data")))
                    / "live_state"
                    / "margin_small_liability_exchange_state.json"
                ),
            )
        )
        interval_hours = float(
            self.config.get("cross_margin_small_liability_exchange_interval_hours", 6.0)
        )
        failure_cooldown_minutes = float(
            self.config.get(
                "cross_margin_small_liability_exchange_failure_cooldown_minutes",
                10.0,
            )
        )
        now_ts = pd.Timestamp.now(tz="UTC")
        try:
            state = json.loads(state_path.read_text()) if state_path.exists() else {}
        except Exception:
            state = {}
        method_version_changed = (
            state.get("method_version") != SMALL_LIABILITY_EXCHANGE_METHOD_VERSION
        )
        last_success_raw = state.get("last_success_ts")
        last_attempt_raw = state.get("last_attempt_ts")
        if last_success_raw and not method_version_changed:
            try:
                last_ts = pd.Timestamp(last_success_raw)
                if last_ts.tzinfo is None:
                    last_ts = last_ts.tz_localize("UTC")
                else:
                    last_ts = last_ts.tz_convert("UTC")
                if now_ts < last_ts + pd.Timedelta(hours=interval_hours):
                    result["reason"] = "cooldown_active"
                    result["last_success_ts"] = last_ts.isoformat()
                    return
            except Exception:
                pass
        if last_attempt_raw and not method_version_changed:
            try:
                last_attempt = pd.Timestamp(last_attempt_raw)
                if last_attempt.tzinfo is None:
                    last_attempt = last_attempt.tz_localize("UTC")
                else:
                    last_attempt = last_attempt.tz_convert("UTC")
                if now_ts < last_attempt + pd.Timedelta(
                    minutes=failure_cooldown_minutes
                ):
                    result["reason"] = "failure_cooldown_active"
                    result["last_attempt_ts"] = last_attempt.isoformat()
                    return
            except Exception:
                pass

        state_path.parent.mkdir(parents=True, exist_ok=True)
        state["last_attempt_ts"] = now_ts.isoformat()
        state["method_version"] = SMALL_LIABILITY_EXCHANGE_METHOD_VERSION
        try:
            state_path.write_text(json.dumps(state, indent=2, sort_keys=True))
        except Exception:
            pass

        exchangeable_assets: List[str] = []
        liability_target_assets: List[str] = []
        exchangeable_response: Any = None
        try:
            exchangeable_assets, liability_target_assets, exchangeable_response = (
                self._fetch_exchangeable_small_liability_assets()
            )
            result["exchangeable_assets"] = exchangeable_assets
            result["liability_target_assets"] = liability_target_assets
            supported_liability_targets = {
                str(asset).upper()
                for asset in self.config.get(
                    "cross_margin_small_liability_supported_targets",
                    ["USDT"],
                )
                if str(asset or "").strip()
            }
            unsupported_targets = sorted(
                set(liability_target_assets).difference(supported_liability_targets)
            )
            if unsupported_targets:
                result["reason"] = "unsupported_liability_target"
                result["unsupported_liability_targets"] = unsupported_targets
                result["supported_liability_targets"] = sorted(
                    supported_liability_targets
                )
                result["exchangeable_response"] = exchangeable_response
                state["last_unsupported_target_ts"] = now_ts.isoformat()
                state["last_unsupported_liability_targets"] = unsupported_targets
                try:
                    state_path.write_text(json.dumps(state, indent=2, sort_keys=True))
                except Exception:
                    pass
                tprint(
                    "Skipping small-liability exchange: Binance reports unsupported "
                    f"liability targets {unsupported_targets}; supported targets="
                    f"{sorted(supported_liability_targets)}"
                )
                return
            assets = [asset for asset in assets if asset in set(exchangeable_assets)]
            result["filtered_assets"] = assets
            submit_assets = list(assets)
            if not assets:
                result["reason"] = "no_exchangeable_candidate_assets"
                result["exchangeable_response"] = exchangeable_response
                return
        except Exception as exc:
            result["exchangeable_error_category"] = _classify_exchange_error(exc)
            result["exchangeable_error"] = str(exc)
            result["filtered_assets"] = assets
            submit_assets = assets
            tprint(
                "Small-liability exchange eligibility fetch failed; "
                f"trying candidate assets directly: {result['exchangeable_error_category']}: {exc}"
            )

        successes: List[Dict[str, Any]] = []
        failures: List[Dict[str, Any]] = []

        def _submit_asset_batch(batch_assets: List[str]) -> Tuple[Any, str]:
            """Submit small-liability assets using Binance-compatible encodings."""
            submission_variants: List[Tuple[str, Dict[str, Any]]] = [
                ("array_query", {"assetNames": list(batch_assets)}),
                ("comma_query", {"assetNames": ",".join(batch_assets)}),
                (
                    "json_array_query",
                    {
                        "assetNames": json.dumps(
                            list(batch_assets), separators=(",", ":")
                        )
                    },
                ),
            ]
            last_exc: Optional[Exception] = None
            variant_errors: List[str] = []
            for variant_name, params in submission_variants:
                try:
                    response = self._binance_sapi_post_repeated_params(
                        "margin/exchange-small-liability",
                        params,
                    )
                    return response, variant_name
                except Exception as exc:
                    last_exc = exc
                    variant_errors.append(
                        f"{variant_name}:{_classify_exchange_error(exc)}:{exc}"
                    )
            if last_exc is not None:
                setattr(
                    last_exc,
                    "_small_liability_variant_errors",
                    " | ".join(variant_errors),
                )
                raise last_exc
            raise RuntimeError("small-liability exchange submission did not run")

        try:
            response, submission_format = _submit_asset_batch(submit_assets)
            successes.append(
                {
                    "assets": assets,
                    "submitted_asset_names": submit_assets,
                    "submission_format": submission_format,
                    "response": response,
                }
            )
            result["exchanged"] = True
            result["assets"] = assets
            result["submitted_asset_names"] = submit_assets
            result["submission_format"] = submission_format
            result["response"] = response
            state["last_success_ts"] = now_ts.isoformat()
            state["last_assets"] = assets
            try:
                state_path.write_text(json.dumps(state, indent=2, sort_keys=True))
            except Exception:
                pass
            tprint(
                "Exchanged small cross-margin liabilities: "
                f"assets={assets} threshold_quote={self.config.get('cross_margin_small_liability_quote_threshold', 1.0)}"
            )
        except Exception as exc:
            batch_error_category = _classify_exchange_error(exc)
            batch_error = str(exc)
            result["batch_error_category"] = batch_error_category
            result["batch_error"] = batch_error
            variant_errors = getattr(exc, "_small_liability_variant_errors", None)
            if variant_errors:
                result["batch_variant_errors"] = variant_errors
            tprint(
                "Small-liability batch exchange failed; retrying per asset: "
                f"{batch_error_category}: {exc}"
            )
            retry_assets = assets
            for asset in retry_assets:
                try:
                    response, submission_format = _submit_asset_batch([asset])
                    successes.append(
                        {
                            "assets": [asset],
                            "submitted_asset_names": [asset],
                            "submission_format": submission_format,
                            "response": response,
                        }
                    )
                    tprint(
                        "Exchanged small cross-margin liability: "
                        f"asset={asset} threshold_quote={self.config.get('cross_margin_small_liability_quote_threshold', 1.0)}"
                    )
                except Exception as asset_exc:
                    failure = {
                        "asset": asset,
                        "error_category": _classify_exchange_error(asset_exc),
                        "error": str(asset_exc),
                    }
                    variant_errors = getattr(
                        asset_exc, "_small_liability_variant_errors", None
                    )
                    if variant_errors:
                        failure["variant_errors"] = variant_errors
                    failures.append(failure)
                    tprint(
                        "Small-liability exchange failed for asset: "
                        f"{asset} {failure['error_category']}: {asset_exc}"
                    )

        if successes:
            converted_assets: List[str] = []
            for row in successes:
                for asset in row.get("assets", []):
                    if asset not in converted_assets:
                        converted_assets.append(str(asset).upper())
            result["exchanged"] = True
            result["assets"] = converted_assets
            result["responses"] = successes
            if failures:
                result["failures"] = failures
            state["last_success_ts"] = now_ts.isoformat()
            state["last_assets"] = converted_assets
            if failures:
                state["last_partial_failure_ts"] = now_ts.isoformat()
                state["last_failures"] = failures
            try:
                state_path.write_text(json.dumps(state, indent=2, sort_keys=True))
            except Exception:
                pass
            return

        result["exchanged"] = False
        result["failures"] = failures
        result["error_category"] = (
            failures[-1]["error_category"] if failures else "exchange_error"
        )
        result["error"] = failures[-1]["error"] if failures else "no_asset_exchanged"
        state["last_error_category"] = result["error_category"]
        state["last_error"] = result["error"]
        try:
            state_path.write_text(json.dumps(state, indent=2, sort_keys=True))
        except Exception:
            pass
        tprint(
            "Cross-margin small-liability exchange skipped/failed: "
            f"{result['error_category']}: {result['error']}"
        )

    def _load_pending_entry_context(self, symbol: str) -> Dict[str, Any]:
        """Return the latest pending live entry row for a symbol, if available."""
        paths: List[Path] = []
        for key in ("trade_log_path", "inference_trade_log_path"):
            raw = (self.config or {}).get(key)
            if raw:
                paths.append(Path(str(raw)))
        paths.append(Path("inference_trades.csv"))
        seen = set()
        for path in paths:
            if path in seen:
                continue
            seen.add(path)
            if not path.exists():
                continue
            try:
                df = pd.read_csv(path)
            except Exception:
                continue
            if df.empty or "symbol" not in df.columns:
                continue
            mask = df["symbol"].astype(str).eq(str(symbol))
            if "lifecycle_event" in df.columns:
                mask &= df["lifecycle_event"].astype(str).eq("entry_placed")
            if "status" in df.columns:
                # Startup reconciliation may mark a live system-origin entry as
                # reconciled_absent before a later margin snapshot re-discovers
                # the position.  If the exchange confirms the position exists,
                # keep using that entry row as provenance for simple-policy
                # stop import instead of treating it as an unknown external.
                allowed_statuses = {"pending", "open", "active", "reconciled_absent"}
                mask &= df["status"].astype(str).isin(allowed_statuses)
            if "action" in df.columns:
                mask &= df["action"].astype(str).eq("enter")
            rows = df.loc[mask].copy()
            if rows.empty:
                continue
            if "timestamp" in rows.columns:
                rows["_ts"] = pd.to_datetime(rows["timestamp"], errors="coerce")
                rows = rows.sort_values("_ts")
            return {
                str(k): (None if pd.isna(v) else v)
                for k, v in rows.iloc[-1].to_dict().items()
            }
        return {}

    def _has_active_or_pending_trade_context(self, symbol: str) -> bool:
        """Return True when a symbol is an active strategy trade or pending entry."""
        symbol_s = str(symbol or "")
        if not symbol_s:
            return False
        with self._state_lock:
            if symbol_s in self.positions:
                return True
            active_oco = bool(
                self.oco_executor
                and symbol_s in self.oco_executor.get_active_positions()
            )
            if active_oco:
                return True
        return bool(self._load_pending_entry_context(symbol_s))

    def _fetch_reconciled_entry_fill(
        self,
        symbol: str,
        *,
        side: str,
        amount: float,
        before_ts: Optional[pd.Timestamp],
    ) -> Dict[str, Any]:
        """Best-effort private fill lookup for an already-open reconciled position."""
        fetch_my_trades = getattr(self.exchange, "fetch_my_trades", None)
        if not callable(fetch_my_trades):
            return {}
        side_l = str(side or "").lower()
        expected_side = "buy" if side_l == "long" else "sell"
        end_ts = pd.Timestamp(before_ts) if before_ts is not None else pd.Timestamp.now(tz="UTC")
        if end_ts.tzinfo is None:
            end_ts = end_ts.tz_localize("UTC")
        else:
            end_ts = end_ts.tz_convert("UTC")
        since = int((end_ts - pd.Timedelta(hours=36)).timestamp() * 1000)
        try:
            trades = fetch_my_trades(symbol, since=since, limit=100)
        except Exception as exc:
            return {
                "error": str(exc),
                "error_category": _classify_exchange_error(exc),
            }
        candidates: List[Dict[str, Any]] = []
        for trade in trades or []:
            if not isinstance(trade, dict):
                continue
            if str(trade.get("side") or "").lower() != expected_side:
                continue
            px = _safe_float(trade.get("price"), default=np.nan)
            qty = abs(_safe_float(trade.get("amount"), default=np.nan))
            ts_raw = trade.get("timestamp")
            try:
                trade_ts = pd.to_datetime(float(ts_raw), unit="ms", utc=True)
            except Exception:
                trade_ts = pd.to_datetime(trade.get("datetime"), utc=True, errors="coerce")
            if not (np.isfinite(px) and px > 0.0 and np.isfinite(qty) and qty > 0.0):
                continue
            if pd.isna(trade_ts):
                continue
            if trade_ts > end_ts + pd.Timedelta(minutes=15):
                continue
            amount_gap = abs(qty - abs(float(amount))) / max(abs(float(amount)), 1e-12)
            candidates.append(
                {
                    "price": float(px),
                    "amount": float(qty),
                    "timestamp": pd.Timestamp(trade_ts),
                    "amount_gap": float(amount_gap),
                    "id": trade.get("id"),
                }
            )
        if not candidates:
            return {"checked": True, "matched": False}
        candidates.sort(key=lambda row: (row["amount_gap"], abs((end_ts - row["timestamp"]).total_seconds())))
        best = candidates[0]
        if best["amount_gap"] > 0.25:
            return {"checked": True, "matched": False, "best_amount_gap": best["amount_gap"]}
        return {
            "checked": True,
            "matched": True,
            "price": best["price"],
            "amount": best["amount"],
            "timestamp": best["timestamp"].isoformat(),
            "id": best.get("id"),
            "source": "fetch_my_trades",
            "amount_gap": best["amount_gap"],
        }

    def _track_external_margin_position(
        self,
        *,
        symbol: str,
        side: str,
        amount: float,
        entry_price: float,
        quote_value: float,
        reason: str,
    ) -> bool:
        """Import a real external margin position for monitoring-only tracking."""
        if not np.isfinite(entry_price) or entry_price <= 0.0:
            return False
        pending_context = self._load_pending_entry_context(symbol)
        if not isinstance(pending_context, dict):
            pending_context = {}
        had_pending_context = bool(pending_context)
        pending_strategy = ""
        original_pending_strategy = ""
        if pending_context:
            pending_strategy = str(pending_context.get("strategy_id") or "")
            original_pending_strategy = pending_strategy
            pending_entry = _safe_float(
                pending_context.get("actual_entry_price")
                or pending_context.get("realized_entry_price")
                or pending_context.get("entry_price"),
                default=np.nan,
            )
            if np.isfinite(pending_entry) and pending_entry > 0.0:
                entry_price = float(pending_entry)
        realized_entry_price = float(entry_price)
        policy_entry_price = _first_finite_price(
            pending_context,
            (
                "policy_entry_price",
                "theoretical_entry_price",
                "ohlcv_entry_price",
                "signal_price",
                "expected_entry_price",
                "decision_mid",
            ),
        )
        policy_entry_price_source = "entry_price"
        if np.isfinite(policy_entry_price) and policy_entry_price > 0.0:
            for key in (
                "policy_entry_price",
                "theoretical_entry_price",
                "ohlcv_entry_price",
                "signal_price",
                "expected_entry_price",
                "decision_mid",
            ):
                if abs(
                    _safe_float(pending_context.get(key), default=np.nan)
                    - float(policy_entry_price)
                ) <= max(abs(float(policy_entry_price)) * 1e-12, 1e-12):
                    policy_entry_price_source = key
                    break
        else:
            policy_entry_price = float(entry_price)
        pending_strategy = self.resolve_simple_policy_strategy_id(
            pending_strategy, side
        )
        bucket_key = pending_strategy or "external_margin_reconciliation"
        artifact_params = self.get_simple_policy_stop_params(pending_strategy)
        fallback_strategy_used = bool(
            pending_strategy
            and (
                not original_pending_strategy
                or original_pending_strategy == "external_margin_reconciliation"
                or original_pending_strategy != pending_strategy
            )
        )
        if artifact_params and isinstance(pending_context, dict):
            for src_key, dst_key in (
                ("params_source", "stop_policy_params_source"),
                ("params_hash", "stop_policy_params_hash"),
                ("schema", "stop_policy_schema"),
            ):
                if not pending_context.get(dst_key) and artifact_params.get(src_key):
                    pending_context[dst_key] = artifact_params.get(src_key)
            if not pending_context.get("sl_mult") and artifact_params.get("sl_mult"):
                pending_context["sl_mult"] = artifact_params.get("sl_mult")
            artifact_barrier = _safe_float(
                artifact_params.get("barrier_frac")
                or artifact_params.get("barrier_pct"),
                default=np.nan,
            )
            context_barrier = _safe_float(
                pending_context.get("barrier_frac")
                or pending_context.get("barrier_pct"),
                default=np.nan,
            )
            if (
                not (np.isfinite(context_barrier) and context_barrier > 0.0)
                and np.isfinite(artifact_barrier)
                and artifact_barrier > 0.0
            ):
                pending_context["barrier_frac"] = float(artifact_barrier)
                pending_context["barrier_pct"] = float(artifact_barrier)
                pending_context["reconciliation_barrier_source"] = (
                    "artifact_simple_policy_stop_params"
                )
            pending_context["strategy_id"] = pending_strategy
            if fallback_strategy_used:
                pending_context["reconciliation_strategy_fallback_used"] = True
                pending_context["reconciliation_original_strategy_id"] = (
                    original_pending_strategy or "unknown"
                )
        fallback_barrier = _safe_float(
            self.config.get("external_reconciliation_fallback_barrier_frac")
            or self.config.get("external_margin_reconciliation_fallback_barrier_frac"),
            default=np.nan,
        )
        if not (np.isfinite(fallback_barrier) and fallback_barrier > 0.0):
            fallback_barrier = 0.02
        context_barrier = _safe_float(
            pending_context.get("barrier_frac")
            or pending_context.get("barrier_pct"),
            default=np.nan,
        )
        if (
            not (np.isfinite(context_barrier) and context_barrier > 0.0)
            and pending_strategy
            and artifact_params
            and np.isfinite(fallback_barrier)
            and fallback_barrier > 0.0
        ):
            pending_context["barrier_frac"] = float(fallback_barrier)
            pending_context["barrier_pct"] = float(fallback_barrier)
            pending_context["reconciliation_barrier_source"] = (
                "config_default_external_reconciliation_barrier"
            )
            tprint(
                f"Using reconciliation fallback barrier for external {symbol}: "
                f"strategy_id={pending_strategy} barrier_frac={fallback_barrier:.6g}"
            )
        sl_mult = _safe_float((pending_context or {}).get("sl_mult"), default=np.nan)
        pending_barrier = _safe_float(
            (pending_context or {}).get("barrier_frac")
            or (pending_context or {}).get("barrier_pct"),
            default=np.nan,
        )
        pending_stop = _safe_float(
            pending_context.get("stop_price") if pending_context else None,
            default=np.nan,
        )
        if not (np.isfinite(pending_stop) and pending_stop > 0.0):
            try:
                open_orders = _fetch_open_protective_stop_orders(
                    self.exchange,
                    symbol=symbol,
                    position_side=side,
                    config=self.config,
                )
                for order in open_orders:
                    stop_px = _order_stop_price(order)
                    if np.isfinite(stop_px) and stop_px > 0.0:
                        pending_stop = float(stop_px)
                        pending_context["stop_price"] = float(stop_px)
                        pending_context["reconciliation_stop_source"] = (
                            "existing_exchange_stop_loss"
                        )
                        order_ts = (
                            order.get("datetime")
                            or order.get("timestamp")
                            or (
                                order.get("info", {}).get("receivedTime")
                                if isinstance(order.get("info"), dict)
                                else None
                            )
                            or (
                                order.get("info", {}).get("lastUpdateTime")
                                if isinstance(order.get("info"), dict)
                                else None
                            )
                        )
                        if (
                            order_ts not in ("", None)
                            and not pending_context.get("entry_time")
                            and not pending_context.get("timestamp")
                        ):
                            pending_context["entry_time"] = order_ts
                            pending_context["reconciliation_entry_time_source"] = (
                                "existing_exchange_stop_loss_time"
                            )
                        break
            except Exception as exc:
                tprint(
                    f"Could not inspect existing STOP_LOSS for {symbol} during "
                    f"reconciliation fallback: {_classify_exchange_error(exc)}: {exc}"
                )
        if (
            (not np.isfinite(pending_barrier) or pending_barrier <= 0.0)
            and np.isfinite(pending_stop)
            and pending_stop > 0.0
            and np.isfinite(sl_mult)
            and sl_mult > 0.0
        ):
            stop_gap = (
                (1.0 - pending_stop / entry_price)
                if side == "long"
                else (pending_stop / entry_price - 1.0)
            )
            if np.isfinite(stop_gap) and stop_gap > 0.0:
                pending_barrier = float(stop_gap) / float(sl_mult)
                pending_context["barrier_frac"] = pending_barrier
                pending_context["barrier_pct"] = pending_barrier
                tprint(
                    f"Recovered simple_policy barrier for {symbol} from persisted "
                    f"entry/stop distance: strategy_id={pending_strategy} "
                    f"barrier_frac={pending_barrier:.6g}"
                )
        if fallback_strategy_used:
            tprint(
                f"Using fallback simple_policy strategy for reconciled {symbol}: "
                f"side={side} original_strategy={original_pending_strategy or 'unknown'} "
                f"fallback_strategy={pending_strategy}"
            )
        required_policy_context = (
            "stop_policy_params_source",
            "stop_policy_params_hash",
            "stop_policy_schema",
            "strategy_id",
            "barrier_frac",
            "sl_mult",
        )
        missing_policy_context = [
            key
            for key in required_policy_context
            if not (pending_context or {}).get(key)
        ]
        has_policy_context = bool(
            pending_context
            and not missing_policy_context
            and str((pending_context or {}).get("stop_policy_schema"))
            == "simple_policy_v1"
            and np.isfinite(pending_barrier)
            and pending_barrier > 0.0
            and pending_strategy
            and np.isfinite(sl_mult)
            and sl_mult > 0.0
        )
        if not has_policy_context:
            tprint(
                f"Skipping external margin stop attachment for {symbol}: "
                "missing simple_policy_optimiser provenance/barrier context: "
                + ",".join(missing_policy_context)
            )
            return False
        attach_barrier_frac = float(pending_barrier)
        if np.isfinite(pending_stop) and pending_stop > 0.0:
            stop_price = float(pending_stop)
            pending_stop_gap = (
                (1.0 - stop_price / entry_price)
                if side == "long"
                else (stop_price / entry_price - 1.0)
            )
            if np.isfinite(pending_stop_gap) and pending_stop_gap > 0.0:
                attach_barrier_frac = float(pending_stop_gap) / float(sl_mult)
        else:
            stop_price = (
                entry_price * (1.0 - sl_mult * attach_barrier_frac)
                if side == "long"
                else entry_price * (1.0 + sl_mult * attach_barrier_frac)
            )
        stop_reason = "original_stop_loss"
        stop_reason_detail = (
            f"original_stop_loss: sl_mult={sl_mult:.6g} "
            f"barrier_frac={attach_barrier_frac:.6g}"
        )
        now = pd.Timestamp.now(tz="UTC")
        entry_ts = now
        if isinstance(pending_context, dict):
            for ts_key in ("entry_time", "timestamp"):
                raw_ts = pending_context.get(ts_key)
                if raw_ts in ("", None):
                    continue
                try:
                    candidate_ts = pd.Timestamp(raw_ts)
                    if candidate_ts.tzinfo is None:
                        candidate_ts = candidate_ts.tz_localize("UTC")
                    else:
                        candidate_ts = candidate_ts.tz_convert("UTC")
                    entry_ts = candidate_ts
                    break
                except Exception:
                    continue
        fill_lookup = self._fetch_reconciled_entry_fill(
            symbol,
            side=side,
            amount=float(amount),
            before_ts=entry_ts,
        )
        if isinstance(fill_lookup, dict) and fill_lookup.get("matched"):
            fill_price = _safe_float(fill_lookup.get("price"), default=np.nan)
            if np.isfinite(fill_price) and fill_price > 0.0:
                entry_price = float(fill_price)
                realized_entry_price = float(fill_price)
                policy_entry_price = float(fill_price)
                policy_entry_price_source = "realized_entry_price"
                try:
                    fill_ts = pd.Timestamp(fill_lookup.get("timestamp"))
                    if fill_ts.tzinfo is None:
                        fill_ts = fill_ts.tz_localize("UTC")
                    else:
                        fill_ts = fill_ts.tz_convert("UTC")
                    entry_ts = fill_ts
                except Exception:
                    pass
                pending_context["reconciliation_entry_fill_source"] = fill_lookup.get(
                    "source", "fetch_my_trades"
                )
                pending_context["reconciliation_entry_fill_id"] = fill_lookup.get("id")
                pending_context["reconciliation_entry_fill_amount"] = fill_lookup.get(
                    "amount"
                )
        state = {
            "side": side,
            "size": float(amount),
            "quote_size": float(quote_value),
            "entry_price": float(entry_price),
            "realized_entry_price": float(realized_entry_price),
            "theoretical_entry_price": float(policy_entry_price),
            "policy_entry_price": float(policy_entry_price),
            "policy_entry_price_source": policy_entry_price_source,
            "timestamp": datetime.now(),
            "entry_time": entry_ts,
            "bucket_key": "external_margin_reconciliation",
            "stop_price": float(stop_price),
            "limit_price": None,
            "peak_price": float(entry_price),
            "mfe": 0.0,
            "last_update": now,
            "last_5m_eval_ts": None,
            "external_position": True,
            "monitoring_only": True,
            "reconciliation_reason": reason,
            "bucket_key": bucket_key,
            "strategy_id": pending_strategy or "external_margin_reconciliation",
            "barrier_frac": float(attach_barrier_frac),
            "barrier_pct": float(attach_barrier_frac),
            "sl_mult": float(sl_mult),
            "stop_reason": stop_reason,
            "stop_reason_detail": stop_reason_detail,
            "stop_policy_params_source": pending_context.get(
                "stop_policy_params_source"
            ),
            "stop_policy_params_hash": pending_context.get("stop_policy_params_hash"),
            "stop_policy_schema": pending_context.get("stop_policy_schema"),
            "reconciliation_strategy_fallback_used": bool(fallback_strategy_used),
            "reconciliation_original_strategy_id": (
                original_pending_strategy or "unknown"
            ),
            "reconciliation_barrier_source": pending_context.get(
                "reconciliation_barrier_source"
            ),
            "reconciliation_context_source": (
                "pending_trade_log"
                if had_pending_context
                else "artifact_fallback_external_position"
            ),
            "reconciliation_entry_fill_source": pending_context.get(
                "reconciliation_entry_fill_source"
            ),
            "reconciliation_entry_fill_id": pending_context.get(
                "reconciliation_entry_fill_id"
            ),
            "reconciliation_entry_fill_amount": pending_context.get(
                "reconciliation_entry_fill_amount"
            ),
        }
        if had_pending_context:
            state["recovered_from_pending_trade_log"] = True
        if pending_context:
            for key in tuple(EXECUTION_AUDIT_KEYS) + MODEL_AND_POLICY_CONTEXT_KEYS:
                if key in pending_context and pending_context.get(key) not in (
                    "",
                    None,
                ):
                    state[key] = pending_context.get(key)
        if self.oco_executor is not None:
            oco_state = state.copy()
            oco_state.update(
                {
                    "oco_order_id": None,
                    "stop_order_id": None,
                    "take_profit_order_id": None,
                    "atr": float(attach_barrier_frac),
                    "sl_mult": float(sl_mult),
                }
            )
            policy_stop_price = float(stop_price)
            exchange_stop_price = float(policy_stop_price)
            exchange_stop_meta: Dict[str, Any] = {}
            if (
                _execution_account(self.config) == "perps"
                and _exchange_id(self.exchange) == "krakenfutures"
            ):
                ticker: Dict[str, Any] = {}
                try:
                    ticker = self.exchange.fetch_ticker(symbol)
                except Exception as exc:
                    _append_position_event(
                        oco_state,
                        "reconciliation_exchange_stop_ticker_fetch_failed",
                        error_category=_classify_exchange_error(exc),
                        error=str(exc),
                    )
                exchange_candidate, exchange_stop_meta = (
                    _kraken_futures_last_stop_from_executable_stop(
                        ticker,
                        self.config,
                        position_side=side,
                        policy_stop_price=float(policy_stop_price),
                    )
                )
                if np.isfinite(exchange_candidate) and exchange_candidate > 0.0:
                    exchange_stop_price = _exchange_precision(
                        self.exchange,
                        symbol,
                        float(exchange_candidate),
                        kind="price",
                    )
            oco_state["stop_price"] = float(policy_stop_price)
            oco_state["policy_stop_price"] = float(policy_stop_price)
            oco_state["requested_policy_stop"] = float(policy_stop_price)
            oco_state["exchange_stop_price"] = float(exchange_stop_price)
            oco_state["exchange_stop_trigger_reference_source"] = "last"
            oco_state["exchange_stop_adjustment"] = exchange_stop_meta
            oco_state["final_placed_stop"] = float(exchange_stop_price)
            existing_oco_state: Dict[str, Any] = {}
            try:
                with self.oco_executor._positions_lock:
                    existing_raw = self.oco_executor.active_positions.get(symbol)
                    if isinstance(existing_raw, dict):
                        existing_oco_state = dict(existing_raw)
            except Exception:
                existing_oco_state = {}
            if existing_oco_state:
                existing_side = str(existing_oco_state.get("side") or "").lower()
                existing_entry = _safe_float(
                    existing_oco_state.get("entry_price"), default=np.nan
                )
                same_position = (
                    existing_side == str(side).lower()
                    and np.isfinite(existing_entry)
                    and np.isfinite(entry_price)
                    and abs(existing_entry - float(entry_price))
                    <= max(abs(float(entry_price)) * 1e-6, 1e-12)
                )
                if same_position:
                    for key in (
                        "peak_price",
                        "mfe",
                        "mae",
                        "policy_entry_price",
                        "policy_entry_price_source",
                        "theoretical_entry_price",
                        "realized_entry_price",
                        "stop_price",
                        "policy_stop_price",
                        "exchange_stop_price",
                        "exchange_stop_trigger_reference_source",
                        "exchange_stop_adjustment",
                        "stop_order_id",
                        "stop_order_ids",
                        "stop_reason",
                        "stop_reason_detail",
                        "requested_policy_stop",
                        "final_placed_stop",
                        "current_price",
                        "current_price_source",
                        "last_price",
                        "current_price_ts",
                        "bars_in_trade",
                        "last_5m_eval_ts",
                        "trade_recap_events",
                        "simple_policy_shadow",
                    ):
                        if key in existing_oco_state:
                            oco_state[key] = existing_oco_state[key]
                    tprint(
                        f"Preserved live price-action state while reconciling "
                        f"{symbol}: mfe={_safe_float(oco_state.get('mfe'), 0.0):.6g} "
                        f"peak_price={_safe_float(oco_state.get('peak_price'), np.nan):.12g} "
                        f"stop_price={_safe_float(oco_state.get('stop_price'), np.nan):.12g}"
                    )
                    preserved_stop = _safe_float(oco_state.get("stop_price"), np.nan)
                    if np.isfinite(preserved_stop) and preserved_stop > 0.0:
                        stop_price = float(preserved_stop)
                        policy_stop_price = float(preserved_stop)
                        oco_state["policy_stop_price"] = float(policy_stop_price)
                        oco_state["requested_policy_stop"] = float(policy_stop_price)
                    preserved_exchange_stop = _safe_float(
                        oco_state.get("exchange_stop_price"), default=np.nan
                    )
                    if (
                        np.isfinite(preserved_exchange_stop)
                        and preserved_exchange_stop > 0.0
                    ):
                        exchange_stop_price = float(preserved_exchange_stop)
            adopted_stop_orders: List[Dict[str, Any]] = []
            mismatched_stop_orders: List[Dict[str, Any]] = []
            try:
                open_orders = _fetch_open_protective_stop_orders(
                    self.exchange,
                    symbol=symbol,
                    position_side=side,
                    config=self.config,
                )
                for order in open_orders:
                    stop_px = _order_stop_price(order)
                    if not (np.isfinite(stop_px) and stop_px > 0.0):
                        continue
                    trigger_matches_policy = _protective_stop_trigger_matches_policy(
                        self.exchange, order, self.config, position_side=side
                    )
                    if (
                        not trigger_matches_policy
                        and not _stop_is_at_least_as_protective(
                            side,
                            float(stop_px),
                            float(stop_price),
                        )
                    ):
                        mismatched_stop_orders.append(order)
                        continue
                    if (
                        np.isfinite(exchange_stop_price)
                        and not _stop_is_at_least_as_protective(
                            side,
                            float(stop_px),
                            float(exchange_stop_price),
                        )
                    ):
                        mismatched_stop_orders.append(order)
                        continue
                    adopted_stop_orders.append(order)
            except Exception as exc:
                tprint(
                    f"External position stop-order adoption failed for {symbol}: "
                    f"{_classify_exchange_error(exc)}: {exc}"
                )
            if mismatched_stop_orders:
                cancel_order = getattr(self.exchange, "cancel_order", None)
                if callable(cancel_order):
                    for order in mismatched_stop_orders:
                        order_id = order.get("id")
                        if not order_id:
                            continue
                        cancelled = False
                        for cancel_params in _margin_fetch_param_variants(self.config):
                            try:
                                cancel_order(order_id, symbol, cancel_params)
                                cancelled = True
                                break
                            except Exception as exc:
                                category = _classify_exchange_error(exc)
                                if category in {
                                    "order_not_found",
                                    "already_closed_or_cancelled",
                                }:
                                    cancelled = True
                                    break
                        if cancelled:
                            tprint(
                                f"Cancelled mismatched recovered STOP_LOSS for "
                                f"{symbol}: order_id={order_id} "
                                f"expected_stop={float(policy_stop_price):.12g} "
                                f"expected_exchange_stop={float(exchange_stop_price):.12g} "
                                f"trigger_signal={_order_trigger_signal(order) or 'unknown'}"
                            )

            if adopted_stop_orders:
                stop_order_ids = [
                    order.get("id") for order in adopted_stop_orders if order.get("id")
                ]
                oco_state["stop_order_ids"] = stop_order_ids
                oco_state["stop_order_id"] = (
                    stop_order_ids[0] if stop_order_ids else None
                )
                adopted_stop_order = adopted_stop_orders[0]
                adopted_stop = _safe_float(
                    adopted_stop_order.get("stopPrice")
                    or adopted_stop_order.get("stop_price")
                    or (
                        adopted_stop_order.get("info", {}).get("stopPrice")
                        if isinstance(adopted_stop_order.get("info"), dict)
                        else None
                    )
                    or adopted_stop_order.get("price"),
                    default=np.nan,
                )
                if np.isfinite(adopted_stop) and adopted_stop > 0.0:
                    oco_state["exchange_stop_price"] = float(adopted_stop)
                    oco_state["final_placed_stop"] = float(adopted_stop)
                stop_coverage = 0.0
                for order in adopted_stop_orders:
                    remaining = _order_remaining_amount(order)
                    if np.isfinite(remaining):
                        stop_coverage += float(remaining)
                if stop_coverage > 0.0:
                    oco_state["stop_order_coverage"] = float(stop_coverage)
                with self.oco_executor._positions_lock:
                    self.oco_executor.active_positions[symbol] = oco_state
                with self._state_lock:
                    self.positions[symbol] = state.copy()
                tprint(
                    f"Imported external margin position with existing STOP_LOSS: "
                    f"{symbol} side={side} stop_order_ids={stop_order_ids} "
                    f"policy_stop={oco_state.get('stop_price')} "
                    f"exchange_stop={oco_state.get('exchange_stop_price')} "
                    f"coverage={oco_state.get('stop_order_coverage')}"
                )
                return True

            protect_external = bool(
                self.config.get("protect_external_margin_positions", True)
            )
            if protect_external and _is_live_execution_mode(self.mode):
                try:
                    stop_result = self.oco_executor.place_oco_order(
                        symbol=symbol,
                        side=side,
                        entry_price=float(entry_price),
                        size=float(amount),
                        bucket_key=bucket_key,
                        barrier_frac=float(attach_barrier_frac),
                    )
                    with self.oco_executor._positions_lock:
                        tracked = self.oco_executor.active_positions.get(symbol)
                        if isinstance(tracked, dict):
                            tracked.update(oco_state)
                            tracked["stop_order_id"] = stop_result.get("stop_order_id")
                            tracked["stop_price"] = stop_result.get(
                                "stop_price", tracked.get("stop_price")
                            )
                            tracked["policy_stop_price"] = stop_result.get(
                                "policy_stop_price", tracked.get("policy_stop_price")
                            )
                            tracked["requested_policy_stop"] = stop_result.get(
                                "requested_policy_stop",
                                tracked.get("requested_policy_stop"),
                            )
                            tracked["exchange_stop_price"] = stop_result.get(
                                "exchange_stop_price",
                                tracked.get("exchange_stop_price"),
                            )
                            tracked["exchange_stop_trigger_reference_source"] = (
                                stop_result.get(
                                    "exchange_stop_trigger_reference_source",
                                    tracked.get(
                                        "exchange_stop_trigger_reference_source"
                                    ),
                                )
                            )
                            tracked["exchange_stop_adjustment"] = stop_result.get(
                                "exchange_stop_adjustment",
                                tracked.get("exchange_stop_adjustment"),
                            )
                            tracked["final_placed_stop"] = stop_result.get(
                                "final_placed_stop", tracked.get("final_placed_stop")
                            )
                            tracked["stop_reason"] = state["stop_reason"]
                            tracked["stop_reason_detail"] = state["stop_reason_detail"]
                            tracked["external_stop_attached"] = bool(
                                stop_result.get("success")
                            )
                    if stop_result.get("success"):
                        tprint(
                            f"Imported external margin position and attached STOP_LOSS: "
                            f"{symbol} side={side} stop={stop_result.get('stop_price')} "
                            f"stop_order_id={stop_result.get('stop_order_id')}"
                        )
                        with self._state_lock:
                            self.positions[symbol] = state.copy()
                        return True
                    else:
                        with self.oco_executor._positions_lock:
                            tracked = self.oco_executor.active_positions.get(symbol)
                            if isinstance(tracked, dict):
                                tracked.update(oco_state)
                                tracked["external_stop_attached"] = False
                                tracked["stop_order_error"] = stop_result.get("error")
                                tracked["stop_order_error_category"] = stop_result.get(
                                    "error_category"
                                )
                                tracked["stop_order_reject_reason"] = stop_result.get(
                                    "stop_order_reject_reason"
                                )
                            else:
                                oco_state["external_stop_attached"] = False
                                oco_state["stop_order_error"] = stop_result.get("error")
                                oco_state["stop_order_error_category"] = (
                                    stop_result.get("error_category")
                                )
                                self.oco_executor.active_positions[symbol] = oco_state
                        with self._state_lock:
                            state["external_stop_attached"] = False
                            state["stop_order_error"] = stop_result.get("error")
                            state["stop_order_error_category"] = stop_result.get(
                                "error_category"
                            )
                            self.positions[symbol] = state.copy()
                        tprint(
                            f"Imported external margin position WITHOUT exchange "
                            f"STOP_LOSS after attach failure; software monitoring "
                            f"remains active: "
                            f"{symbol} category={stop_result.get('error_category')} "
                            f"error={stop_result.get('error')}"
                        )
                    return True
                except Exception as exc:
                    with self._state_lock:
                        self.positions[symbol] = state.copy()
                    oco_state["external_stop_attached"] = False
                    oco_state["stop_order_error"] = str(exc)
                    oco_state["stop_order_error_category"] = _classify_exchange_error(
                        exc
                    )
                    with self.oco_executor._positions_lock:
                        self.oco_executor.active_positions[symbol] = oco_state
                    tprint(
                        f"Imported external margin position WITHOUT exchange "
                        f"STOP_LOSS after attach exception; software monitoring "
                        f"remains active: {symbol} "
                        f"{_classify_exchange_error(exc)}: {exc}"
                    )
                    return True
            tprint(
                f"External margin position not imported for monitoring without "
                f"STOP_LOSS protection: {symbol} side={side}"
            )
            return False
        return False

    def reconcile_perps_account(self) -> Dict[str, Any]:
        """Import existing perp positions into the live protective-stop monitor."""
        report: Dict[str, Any] = {
            "timestamp": pd.Timestamp.now(tz="UTC").isoformat(),
            "mode": self.mode,
            "execution_account": _execution_account(self.config),
            "margin_mode": _margin_mode(self.config),
            "quote_currency": str(
                self.config.get("live_quote_currency")
                or self.config.get("quote_currency")
                or "USD"
            ).upper(),
            "items": [],
            "summary": {},
        }
        if (
            not _is_live_execution_mode(self.mode)
            or _execution_account(self.config) != "perps"
            or self.exchange is None
        ):
            report["summary"] = {"skipped": True, "reason": "not_perps_live"}
            return report

        try:
            fetch_positions = getattr(self.exchange, "fetch_positions", None)
            if not callable(fetch_positions):
                report["summary"] = {
                    "skipped": True,
                    "reason": "fetch_positions_unavailable",
                }
                return report
            try:
                raw_positions = fetch_positions([], _cancel_params(self.config))
            except TypeError:
                raw_positions = fetch_positions(_cancel_params(self.config))
        except Exception as exc:
            category = _classify_exchange_error(exc)
            tprint(f"Perps reconciliation failed: {category}: {exc}")
            report["summary"] = {
                "skipped": True,
                "reason": "fetch_positions_failed",
                "error_category": category,
                "error": str(exc),
            }
            return report

        counts: Dict[str, int] = {}
        import_positions = bool(self.config.get("import_external_perp_positions", True))
        exchange_symbols: set[str] = set()
        for position in raw_positions or []:
            if not isinstance(position, dict):
                continue
            signed_amount = _position_contracts(position)
            amount = abs(float(signed_amount))
            if amount <= 0.0:
                continue
            symbol = _position_symbol(position, self.exchange)
            if not symbol:
                continue
            exchange_symbols.add(str(symbol))
            side = _position_side(position, signed_amount)
            entry_price = _position_entry_price(position)
            quote_value = _position_quote_value(position, amount, entry_price)
            item = {
                "symbol": symbol,
                "kind": "perp_position",
                "side": side,
                "amount": float(amount),
                "entry_price": float(entry_price) if np.isfinite(entry_price) else None,
                "quote_value": float(quote_value) if np.isfinite(quote_value) else None,
                "classification": "external_perp_position",
                "imported_for_monitoring": False,
            }
            imported = False
            if import_positions and np.isfinite(entry_price) and entry_price > 0.0:
                imported = self._track_external_margin_position(
                    symbol=symbol,
                    side=side,
                    amount=float(amount),
                    entry_price=float(entry_price),
                    quote_value=(
                        float(quote_value)
                        if np.isfinite(quote_value)
                        else float(amount) * float(entry_price)
                    ),
                    reason="external_perp_position",
                )
            item["imported_for_monitoring"] = bool(imported)
            if not imported:
                if not import_positions:
                    item["monitoring_skip_reason"] = "perp_position_import_disabled"
                elif not (np.isfinite(entry_price) and entry_price > 0.0):
                    item["monitoring_skip_reason"] = "missing_entry_price"
                elif self._load_pending_entry_context(symbol):
                    item["monitoring_skip_reason"] = "pending_trade_context_present"
                else:
                    item["monitoring_skip_reason"] = "missing_stop_policy_provenance"
            report["items"].append(item)
            counts[str(item["classification"])] = (
                counts.get(str(item["classification"]), 0) + 1
            )

        stale_tracked: List[str] = []
        stale_tracked_details: List[Dict[str, Any]] = []
        if self.oco_executor is not None:
            with self.oco_executor._positions_lock:
                tracked_symbols = list(self.oco_executor.active_positions.keys())
                for tracked_symbol in tracked_symbols:
                    if str(tracked_symbol) in exchange_symbols:
                        continue
                    stale_state = self.oco_executor.active_positions.pop(
                        tracked_symbol, None
                    )
                    stale_tracked.append(str(tracked_symbol))
                    if isinstance(stale_state, dict):
                        close_mode = _position_absent_reconciliation_mode(stale_state)
                        stale_state["exit_reason"] = close_mode
                        stale_state["reconciliation_mode"] = close_mode
                        _append_position_event(
                            stale_state,
                            "removed_after_perps_reconcile_absent",
                            reason=close_mode,
                            reconciliation_mode=close_mode,
                        )
                        stale_tracked_details.append(
                            {
                                "symbol": str(tracked_symbol),
                                "reconciliation_mode": close_mode,
                                "side": stale_state.get("side"),
                                "amount": stale_state.get("amount"),
                                "entry_price": stale_state.get("entry_price"),
                                "stop_order_id": stale_state.get("stop_order_id"),
                                "last_order_status": stale_state.get("last_order_status"),
                            }
                        )
            if stale_tracked:
                with self._state_lock:
                    for tracked_symbol in stale_tracked:
                        self.positions.pop(tracked_symbol, None)
                report["stale_tracked_positions_removed"] = stale_tracked
                report["stale_tracked_positions_removed_details"] = stale_tracked_details
                tprint(
                    "Perps reconciliation removed locally tracked positions absent "
                    f"from exchange snapshot: {stale_tracked_details or stale_tracked}"
                )

        report["summary"] = {
            "skipped": False,
            "item_count": len(report["items"]),
            "counts": counts,
            "exchange_open_symbols": sorted(exchange_symbols),
            "stale_tracked_positions_removed": stale_tracked,
            "stale_tracked_positions_removed_details": stale_tracked_details,
            "active_positions_after_reconcile": len(self.get_active_positions()),
        }
        tprint(
            "Perps reconciliation complete: "
            f"items={len(report['items'])} counts={counts} "
            f"active_positions={report['summary']['active_positions_after_reconcile']}"
        )
        return report

    def reconcile_cross_margin_account(self) -> Dict[str, Any]:
        """Classify existing cross-margin balances/debts before live trading."""
        report: Dict[str, Any] = {
            "timestamp": pd.Timestamp.now(tz="UTC").isoformat(),
            "mode": self.mode,
            "execution_account": _execution_account(self.config),
            "margin_mode": _margin_mode(self.config),
            "quote_currency": str(
                self.config.get("live_quote_currency")
                or self.config.get("quote_currency")
                or "USDC"
            ).upper(),
            "dust_quote_threshold": float(
                self.config.get(
                    "cross_margin_dust_quote_threshold",
                    _default_cross_margin_dust_quote_threshold(self.mode),
                )
            ),
            "cross_margin_residual_assets": list(
                self.config.get("cross_margin_residual_assets", ["BNB"])
            ),
            "residual_cleanup_cap_quote": float(
                self.config.get("cross_margin_residual_cleanup_cap_quote", 5.0)
            ),
            "items": [],
            "summary": {},
        }
        if _execution_account(self.config) == "perps":
            return self.reconcile_perps_account()
        if (
            not _is_live_execution_mode(self.mode)
            or _execution_account(self.config) != "margin"
            or _margin_mode(self.config) != "cross"
            or self.exchange is None
        ):
            report["summary"] = {"skipped": True, "reason": "not_cross_margin_live"}
            return report

        try:
            balance = self._fetch_margin_balance()
        except Exception as exc:
            category = _classify_exchange_error(exc)
            tprint(f"Cross-margin reconciliation failed: {category}: {exc}")
            report["summary"] = {
                "skipped": True,
                "reason": "fetch_balance_failed",
                "error_category": category,
                "error": str(exc),
            }
            return report

        quote = str(report["quote_currency"])
        threshold = float(report["dust_quote_threshold"])
        residual_assets = {
            str(asset).upper()
            for asset in self.config.get("cross_margin_residual_assets", ["BNB"])
        }
        import_non_dust = bool(
            self.config.get("import_external_margin_positions", True)
        )
        assets = self._parse_margin_assets(balance)
        counts: Dict[str, int] = {}
        dust_to_bnb_candidates: List[str] = []
        dust_to_bnb_excluded_due_debt: List[str] = []
        small_liability_candidates: List[str] = []
        small_position_cleanup_candidates: List[Dict[str, Any]] = []
        small_liability_threshold = float(
            self.config.get("cross_margin_small_liability_quote_threshold", 1.0)
        )
        residual_cleanup_cap_quote = float(
            self.config.get("cross_margin_residual_cleanup_cap_quote", 5.0)
        )

        def _below_residual_cleanup_cap(value: float) -> bool:
            return (
                np.isfinite(value)
                and value > 0.0
                and value <= residual_cleanup_cap_quote
            )

        for asset, vals in sorted(assets.items()):
            if not asset:
                continue
            total = max(_safe_float(vals.get("total"), 0.0), 0.0)
            debt = max(_safe_float(vals.get("debt"), 0.0), 0.0)
            interest = max(_safe_float(vals.get("interest"), 0.0), 0.0)
            free = max(_safe_float(vals.get("free"), 0.0), 0.0)
            used = max(_safe_float(vals.get("used"), 0.0), 0.0)

            if asset != quote and total > 0.0 and debt > 0.0:
                balance_value, balance_px, balance_symbol = self._asset_quote_value(
                    asset, total, quote
                )
                debt_value, _, _ = self._asset_quote_value(asset, debt, quote)
                net_amount = total - debt
                net_quote_value = (
                    abs(float(net_amount)) * float(balance_px)
                    if balance_px is not None and np.isfinite(balance_px)
                    else np.nan
                )
                net_item = {
                    "asset": asset,
                    "kind": "net_exposure",
                    "side": "long" if net_amount >= 0.0 else "short",
                    "amount": abs(float(net_amount)),
                    "free": float(free),
                    "used": float(used),
                    "debt": float(debt),
                    "interest": float(interest),
                    "gross_balance_quote_value": (
                        float(balance_value) if np.isfinite(balance_value) else None
                    ),
                    "gross_debt_quote_value": (
                        float(debt_value) if np.isfinite(debt_value) else None
                    ),
                    "quote_value": (
                        float(net_quote_value) if np.isfinite(net_quote_value) else None
                    ),
                    "price": float(balance_px) if balance_px is not None else None,
                    "symbol": balance_symbol,
                    "classification": "",
                    "imported_for_monitoring": False,
                    "netted_balance_and_debt": True,
                }
                if not np.isfinite(net_quote_value) or net_quote_value <= 0.0:
                    net_item["classification"] = "netted_flat_or_unpriced"
                elif (
                    asset in residual_assets
                    and net_amount >= 0.0
                    and _below_residual_cleanup_cap(float(net_quote_value))
                ):
                    net_item["classification"] = "fee_token_or_collateral_residual"
                elif net_quote_value < threshold:
                    net_item["classification"] = (
                        "dust_residual" if net_amount >= 0.0 else "dust_to_repay"
                    )
                else:
                    net_item["classification"] = (
                        "external_long_position"
                        if net_amount >= 0.0
                        else "external_short_position"
                    )
                classification = str(net_item["classification"])
                has_trade_context = bool(
                    balance_symbol
                    and self._has_active_or_pending_trade_context(balance_symbol)
                )
                if has_trade_context:
                    net_item["cleanup_skip_reason"] = "active_or_pending_trade_present"
                elif classification == "fee_token_or_collateral_residual":
                    net_item["cleanup_skip_reason"] = (
                        "residual_asset_below_cleanup_cap_not_imported"
                    )
                elif classification == "dust_to_repay" and _below_residual_cleanup_cap(
                    float(net_quote_value)
                ):
                    small_liability_candidates.append(asset)
                    small_position_cleanup_candidates.append(
                        {
                            **net_item,
                            "reason": "net_dust_to_repay",
                        }
                    )
                    self._try_repay_available_margin_debt(
                        asset=asset,
                        free=free,
                        debt=debt,
                        quote_value=net_quote_value,
                        report=report,
                        reason="net_dust_to_repay",
                    )
                elif (
                    classification == "dust_residual"
                    and asset not in {quote, "BNB"}
                    and free > 0.0
                    and debt <= 0.0
                    and interest <= 0.0
                ):
                    dust_to_bnb_candidates.append(asset)
                    small_position_cleanup_candidates.append(
                        {
                            **net_item,
                            "reason": "net_dust_residual",
                        }
                    )
                elif classification == "dust_to_repay":
                    net_item["cleanup_skip_reason"] = "above_residual_cleanup_cap_quote"
                    net_item["residual_cleanup_cap_quote"] = float(
                        residual_cleanup_cap_quote
                    )
                elif classification == "dust_residual":
                    dust_to_bnb_excluded_due_debt.append(asset)
                if (
                    import_non_dust
                    and classification
                    in {"external_long_position", "external_short_position"}
                    and balance_symbol
                    and balance_px is not None
                ):
                    imported = self._track_external_margin_position(
                        symbol=balance_symbol,
                        side=str(net_item["side"]),
                        amount=abs(float(net_amount)),
                        entry_price=float(balance_px),
                        quote_value=float(net_quote_value),
                        reason=classification,
                    )
                    net_item["imported_for_monitoring"] = bool(imported)
                    if not imported:
                        net_item["monitoring_skip_reason"] = "stop_loss_not_attached"
                        pending_context = self._load_pending_entry_context(
                            balance_symbol
                        )
                        if pending_context:
                            net_item["monitoring_skip_reason"] = (
                                "pending_trade_log_entry_present"
                            )
                            net_item["cleanup_attempted"] = False
                            net_item["cleanup_skip_reason"] = (
                                "do_not_admin_close_regular_trade"
                            )
                        else:
                            cleanup = self._try_admin_close_external_margin_position(
                                symbol=balance_symbol,
                                side=str(net_item["side"]),
                                amount=abs(float(net_amount)),
                                price=float(balance_px),
                                quote_value=float(net_quote_value),
                                report=report,
                                reason="unimported_netted_external_position",
                            )
                            net_item["cleanup_attempted"] = bool(
                                cleanup.get("attempted")
                            )
                            net_item["cleanup_success"] = bool(cleanup.get("success"))
                            if cleanup.get("error_category"):
                                net_item["cleanup_error_category"] = cleanup.get(
                                    "error_category"
                                )
                report["items"].append(net_item)
                counts[classification] = counts.get(classification, 0) + 1
                continue

            for kind, amount, side in (
                ("debt", debt, "short"),
                ("balance", total, "long"),
            ):
                if amount <= 0.0:
                    continue
                quote_value, px, symbol = self._asset_quote_value(asset, amount, quote)
                if kind == "balance" and asset in residual_assets:
                    classification = "fee_token_or_collateral_residual"
                elif not np.isfinite(quote_value) or quote_value <= 0.0:
                    classification = f"unpriced_{kind}"
                elif quote_value < threshold:
                    classification = (
                        "dust_to_repay" if kind == "debt" else "dust_residual"
                    )
                elif asset == quote and kind == "balance":
                    classification = "quote_balance_available"
                elif asset == quote:
                    classification = "quote_debt_liability"
                else:
                    classification = f"external_{side}_position"

                item = {
                    "asset": asset,
                    "kind": kind,
                    "side": side,
                    "amount": float(amount),
                    "free": float(free),
                    "used": float(used),
                    "debt": float(debt),
                    "interest": float(interest),
                    "quote_value": (
                        float(quote_value) if np.isfinite(quote_value) else None
                    ),
                    "price": float(px) if px is not None else None,
                    "symbol": symbol,
                    "classification": classification,
                    "imported_for_monitoring": False,
                }
                has_trade_context = bool(
                    symbol and self._has_active_or_pending_trade_context(symbol)
                )
                if has_trade_context:
                    item["cleanup_skip_reason"] = "active_or_pending_trade_present"
                if (
                    classification == "dust_to_repay"
                    and kind == "debt"
                    and asset != quote
                    and np.isfinite(quote_value)
                    and 0.0 < quote_value < small_liability_threshold
                    and _below_residual_cleanup_cap(float(quote_value))
                    and not has_trade_context
                ):
                    small_liability_candidates.append(asset)
                    small_position_cleanup_candidates.append(
                        {
                            **item,
                            "reason": "dust_to_repay",
                        }
                    )
                    self._try_repay_available_margin_debt(
                        asset=asset,
                        free=free,
                        debt=debt,
                        quote_value=quote_value,
                        report=report,
                        reason="dust_to_repay",
                    )
                elif (
                    classification == "dust_to_repay"
                    and kind == "debt"
                    and asset != quote
                    and np.isfinite(quote_value)
                    and quote_value > residual_cleanup_cap_quote
                ):
                    item["cleanup_skip_reason"] = "above_residual_cleanup_cap_quote"
                    item["residual_cleanup_cap_quote"] = float(
                        residual_cleanup_cap_quote
                    )
                if (
                    classification == "dust_residual"
                    and kind == "balance"
                    and asset not in {quote, "BNB"}
                    and free > 0.0
                ):
                    if debt > 0.0 or interest > 0.0:
                        dust_to_bnb_excluded_due_debt.append(asset)
                    else:
                        dust_to_bnb_candidates.append(asset)
                        small_position_cleanup_candidates.append(
                            {
                                **item,
                                "reason": "dust_residual",
                            }
                        )
                if (
                    import_non_dust
                    and classification == f"external_{side}_position"
                    and symbol
                    and px is not None
                ):
                    imported = self._track_external_margin_position(
                        symbol=symbol,
                        side=side,
                        amount=float(amount),
                        entry_price=float(px),
                        quote_value=float(quote_value),
                        reason=classification,
                    )
                    item["imported_for_monitoring"] = bool(imported)
                    if not imported:
                        item["monitoring_skip_reason"] = "stop_loss_not_attached"
                        pending_context = self._load_pending_entry_context(symbol)
                        if pending_context:
                            item["monitoring_skip_reason"] = (
                                "pending_trade_log_entry_present"
                            )
                            item["cleanup_attempted"] = False
                            item["cleanup_skip_reason"] = (
                                "do_not_admin_close_regular_trade"
                            )
                        else:
                            cleanup = self._try_admin_close_external_margin_position(
                                symbol=symbol,
                                side=side,
                                amount=float(amount),
                                price=float(px),
                                quote_value=float(quote_value),
                                report=report,
                                reason="unimported_external_position",
                            )
                            item["cleanup_attempted"] = bool(cleanup.get("attempted"))
                            item["cleanup_success"] = bool(cleanup.get("success"))
                            if cleanup.get("error_category"):
                                item["cleanup_error_category"] = cleanup.get(
                                    "error_category"
                                )
                report["items"].append(item)
                counts[classification] = counts.get(classification, 0) + 1
        self._maybe_convert_margin_dust_to_bnb(
            candidate_assets=dust_to_bnb_candidates,
            report=report,
        )
        if dust_to_bnb_excluded_due_debt:
            report.setdefault("dust_to_bnb", {})["excluded_due_same_asset_debt"] = (
                sorted(set(dust_to_bnb_excluded_due_debt))
            )
        self._maybe_exchange_small_margin_liabilities(
            candidate_assets=small_liability_candidates,
            report=report,
        )
        self._maybe_cleanup_small_margin_positions(
            candidate_positions=small_position_cleanup_candidates,
            report=report,
        )

        report["summary"] = {
            "skipped": False,
            "item_count": len(report["items"]),
            "counts": counts,
            "active_positions_after_reconcile": len(self.get_active_positions()),
        }
        tprint(
            "Cross-margin reconciliation complete: "
            f"items={len(report['items'])} counts={counts} "
            f"active_positions={report['summary']['active_positions_after_reconcile']}"
        )
        return report

    def get_bucket_params(self, bucket_key: Optional[str] = None) -> Dict[str, Any]:
        """Return non-stop bucket params for compatibility callers.

        STOP_LOSS placement/replacement must use get_simple_policy_stop_params()
        so legacy/runtime bucket fields cannot influence protective stops.
        """
        params: Dict[str, Any] = {}
        params.update(_non_stop_bucket_fields(self.bucket_params))
        if bucket_key is None:
            return params
        raw_key = str(bucket_key or "")
        core_key = strategy_core_id(raw_key)
        key_lower = raw_key.lower()
        key_upper = raw_key.upper()
        bucket = (
            self.bucket_params.get(raw_key, {})
            or self.bucket_params.get(core_key, {})
            or self.bucket_params.get(key_lower, {})
            or self.bucket_params.get(key_upper, {})
        )
        if (
            not bucket
            and "buckets" in self.bucket_params
            and isinstance(self.bucket_params["buckets"], dict)
        ):
            bucket = (
                self.bucket_params["buckets"].get(key_upper, {})
                or self.bucket_params["buckets"].get(raw_key, {})
                or self.bucket_params["buckets"].get(key_lower, {})
            )
        params.update(_non_stop_bucket_fields(bucket))
        return params

    def get_simple_policy_stop_params(
        self, bucket_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """Return exact-strategy simple_policy_optimiser stop params for stop decisions."""
        if self.oco_executor is not None:
            return self.oco_executor.get_simple_policy_stop_params(bucket_key)
        if bucket_key is None:
            return {}
        params = self.simple_policy_stop_params_by_strategy.get(str(bucket_key))
        return dict(params) if isinstance(params, dict) else {}

    def resolve_simple_policy_strategy_id(
        self, strategy_id: Optional[str], side: Optional[str]
    ) -> str:
        if self.oco_executor is not None:
            resolver = getattr(
                self.oco_executor, "resolve_simple_policy_strategy_id", None
            )
            if callable(resolver):
                return str(resolver(strategy_id, side) or "")
        sid = str(strategy_id or "").strip()
        side_l = str(side or "").lower().strip()
        candidates: List[str] = []
        if sid:
            candidates.append(sid)
            core = strategy_core_id(sid)
            if core and side_l in {"long", "short"}:
                candidates.append(f"{side_l}_{core}")
        for candidate in candidates:
            if candidate in self.simple_policy_stop_params_by_strategy:
                return candidate
        return self._fallback_simple_policy_strategy_id(side) or sid

    def _fallback_simple_policy_strategy_id(self, side: Optional[str]) -> str:
        """Pick a deterministic deployed simple-policy strategy for unknown imports."""
        if self.oco_executor is not None:
            resolver = getattr(
                self.oco_executor, "_fallback_simple_policy_strategy_id", None
            )
            if callable(resolver):
                return str(resolver(side) or "")
        side_l = str(side or "").lower().strip()
        candidates: List[str] = []
        for sid, params in self.simple_policy_stop_params_by_strategy.items():
            sid_s = str(sid)
            param_side = str((params or {}).get("side") or "").lower().strip()
            if side_l in {"long", "short"} and (
                param_side == side_l or sid_s.startswith(f"{side_l}_")
            ):
                candidates.append(sid_s)
        if not candidates:
            candidates = [
                str(sid) for sid in self.simple_policy_stop_params_by_strategy
            ]
        return sorted(candidates)[0] if candidates else ""

    def get_cooldown_hours(self, bucket_key: Optional[str] = None) -> float:
        """Return zero: live inference cooldown is handled on losing closes only."""
        return 0.0

    def execute_trade(
        self,
        symbol: str,
        side: str,
        size: float,
        price: Optional[float] = None,
        bucket_key: Optional[str] = None,
        ohlcv_reference_price: Optional[float] = None,
        trade_context: Optional[Dict[str, Any]] = None,
        execution_snapshot: Optional[Dict[str, Any]] = None,
        signal_price: Optional[float] = None,
        decision_mid: Optional[float] = None,
        expected_entry_price: Optional[float] = None,
        expected_fill_slippage_bps: Optional[float] = None,
        max_chase_bps: Optional[float] = None,
        rank_score: Optional[float] = None,
        adjusted_rank_score: Optional[float] = None,
        final_threshold: Optional[float] = None,
        position_size_before_liquidity: Optional[float] = None,
        position_size_after_liquidity: Optional[float] = None,
        order_type: Optional[str] = None,
        limit_price: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Execute a trade with OCO order support.

        Args:
            symbol: Trading symbol (e.g., "BTC/USDT")
            side: "long" or "short"
            size: Position size in quote currency
            price: Limit price (None for market order)
            bucket_key: Bucket key for getting SL/TP parameters (required for OCO)

        Returns:
            Dictionary with execution results
        """
        # The live inference path passes quote-currency USDT after portfolio
        # gating. Keep legacy fractional behavior for direct callers/tests.
        size_f = abs(float(size))
        if size_f <= 1.0:
            position_value = size_f * self.capital * self.max_position_size
        else:
            position_value = size_f

        cooldown_hours = self.get_cooldown_hours(bucket_key)
        now_utc = pd.Timestamp.now(tz="UTC")
        with self._state_lock:
            active_oco = bool(
                self.oco_executor and symbol in self.oco_executor.get_active_positions()
            )
            if symbol in self.positions or active_oco:
                return {"success": False, "error": f"symbol {symbol} already active"}
            last_ts = self._last_trade_timestamps.get(symbol)
            if (
                last_ts is not None
                and cooldown_hours > 0.0
                and now_utc
                < (pd.Timestamp(last_ts) + pd.Timedelta(hours=float(cooldown_hours)))
            ):
                return {
                    "success": False,
                    "error": f"symbol {symbol} in cooldown for {cooldown_hours:.1f}h",
                }

        audit_context = dict(trade_context or {})
        if isinstance(execution_snapshot, dict):
            audit_context.update(execution_snapshot)
        audit_context.update(
            {
                k: v
                for k, v in {
                    "signal_price": signal_price,
                    "decision_mid": decision_mid,
                    "expected_entry_price": expected_entry_price,
                    "expected_fill_slippage_bps": expected_fill_slippage_bps,
                    "max_chase_bps": max_chase_bps,
                    "rank_score": rank_score,
                    "adjusted_rank_score": adjusted_rank_score,
                    "final_threshold": final_threshold,
                    "position_size_before_liquidity": position_size_before_liquidity,
                    "position_size_after_liquidity": position_size_after_liquidity,
                    "entry_order_type_requested": order_type,
                    "entry_limit_price": limit_price,
                }.items()
                if v is not None
            }
        )
        stale_reject = _stale_signal_entry_reject(
            trade_context=audit_context,
            now=now_utc,
            config=self.config,
            mode=self.mode,
        )
        if stale_reject is not None:
            tprint(
                "[STALE_SIGNAL_ENTRY_BLOCK] "
                f"{symbol} {side} signal_close_to_entry_seconds="
                f"{_safe_float(stale_reject.get('signal_close_to_entry_seconds'), default=np.nan):.1f} "
                f"limit_seconds="
                f"{_safe_float(stale_reject.get('max_signal_close_to_entry_seconds'), default=np.nan):.1f} "
                f"signal_bar_ts={stale_reject.get('signal_bar_ts')} "
                f"signal_bar_close_ts={stale_reject.get('signal_bar_close_ts')}"
            )
            return {
                **stale_reject,
                "symbol": symbol,
                "side": side,
                "size": position_value,
                "bucket_key": bucket_key,
                **audit_context,
                "stale_signal_age_gate_exceeded": True,
            }
        effective_price = float(limit_price) if limit_price is not None else price

        if _is_live_execution_mode(self.mode):
            return self._execute_live(
                symbol,
                side,
                position_value,
                effective_price,
                bucket_key,
                ohlcv_reference_price=ohlcv_reference_price,
                trade_context=audit_context,
                order_type=order_type,
            )
        else:
            return self._record_shadow_trade(
                symbol,
                side,
                position_value,
                effective_price,
                bucket_key=bucket_key,
                ohlcv_reference_price=ohlcv_reference_price,
                trade_context=audit_context,
            )

    def _execute_live(
        self,
        symbol: str,
        side: str,
        size: float,
        price: Optional[float] = None,
        bucket_key: Optional[str] = None,
        ohlcv_reference_price: Optional[float] = None,
        trade_context: Optional[Dict[str, Any]] = None,
        order_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute live trade with STOP_LOSS support.

        Args:
            symbol: Trading symbol
            side: "long" or "short"
            size: Position size in quote currency
            price: Limit price
            bucket_key: Bucket key for OCO parameters

        Returns:
            Execution result dictionary
        """
        order_side = "buy" if side == "long" else "sell"
        exchange_symbol = _exchange_symbol_for_config(
            self.exchange, self.config, symbol
        )
        market = _load_market(self.exchange, exchange_symbol)
        expected_entry_price = float(price) if price is not None else np.nan
        live_barrier_frac = _safe_float(
            (trade_context or {}).get("barrier_frac")
            or (trade_context or {}).get("barrier_pct"),
            default=np.nan,
        )

        try:
            if not bucket_key or not self.oco_executor:
                error = (
                    "missing simple_policy_optimiser strategy context before live entry; "
                    "refusing to place unprotected order"
                )
                tprint(f"Refusing live entry for {symbol}: {error}")
                return {
                    "success": False,
                    "error": error,
                    "error_category": "invalid_simple_policy_stop_params",
                    "symbol": symbol,
                    "side": side,
                    "size": float(size),
                }

            live_bucket_params = self.oco_executor.get_simple_policy_stop_params(
                bucket_key
            )
            try:
                validate_simple_policy_stop_params(
                    live_bucket_params,
                    state={
                        "strategy_id": bucket_key,
                        "barrier_frac": live_barrier_frac,
                    },
                    require_metadata=True,
                    require_barrier=True,
                )
            except SimplePolicyStopParamsError as exc:
                err_text = str(exc)
                err_category = (
                    "missing_policy_barrier_pct"
                    if "barrier_frac/barrier_pct" in err_text
                    else "invalid_simple_policy_stop_params"
                )
                error = (
                    f"invalid simple_policy_optimiser params before live entry: {exc}; "
                    "refusing to place unprotected order"
                )
                tprint(f"Refusing live entry for {symbol}: {error}")
                return {
                    "success": False,
                    "error": error,
                    "error_category": err_category,
                    "symbol": symbol,
                    "side": side,
                    "size": float(size),
                }

            if not np.isfinite(expected_entry_price):
                ticker = self.exchange.fetch_ticker(exchange_symbol)
                expected_entry_price = _safe_float(ticker.get("last"), default=np.nan)
            amount_base = self._quote_to_base_amount(
                exchange_symbol,
                quote_size=float(size),
                reference_price=expected_entry_price,
                market=market,
            )
            entry_leverage = _perp_entry_leverage_from_context(
                trade_context, self.config
            )
            leverage_set_result = _set_perp_leverage_best_effort(
                self.exchange,
                symbol=exchange_symbol,
                leverage=entry_leverage,
                config=self.config,
            )
            if _execution_account(self.config) == "perps":
                tprint(
                    f"Perps entry leverage: symbol={symbol} "
                    f"leverage={entry_leverage:.4g} "
                    f"set_attempted={leverage_set_result.get('attempted')} "
                    f"set_success={leverage_set_result.get('success')}"
                )
                if leverage_set_result.get("error"):
                    tprint(
                        f"Perps set_leverage result for {symbol}: "
                        f"{leverage_set_result.get('error')}; "
                        "continuing with leveraged order params/implicit margin"
                    )

            force_market_entries = bool(
                self.config.get("force_market_entry_orders", False)
            )
            requested_order_type = str(order_type or "").lower()
            if requested_order_type == "market":
                force_market_entries = True
            if force_market_entries and price is not None:
                tprint(
                    f"Live entry for {symbol}: forcing market order; "
                    f"caller reference price={float(price):.8g}"
                )

            if price is not None and not force_market_entries:
                entry_price_for_order = _exchange_precision(
                    self.exchange, exchange_symbol, float(price), kind="price"
                )
                order = self.exchange.create_order(
                    symbol=exchange_symbol,
                    type="limit",
                    side=order_side,
                    amount=amount_base,
                    price=entry_price_for_order,
                    params=_order_params(
                        self.config,
                        reduce_only=False,
                        side=order_side,
                        leverage=entry_leverage,
                    ),
                )
                entry_order_type = "limit"
            else:
                order = self.exchange.create_order(
                    symbol=exchange_symbol,
                    type="market",
                    side=order_side,
                    amount=amount_base,
                    params=_order_params(
                        self.config,
                        reduce_only=False,
                        side=order_side,
                        leverage=entry_leverage,
                    ),
                )
                entry_order_type = "market"
            fallback_price = float(expected_entry_price)
            entry_price, filled_amount, partial_fill = _extract_order_fill(
                order, fallback_price
            )
            (
                entry_fee_quote,
                entry_fee_cost,
                entry_fee_currency,
                entry_fee_source,
            ) = _fee_to_quote(
                exchange_symbol,
                order.get("fee"),
                price=entry_price,
            )
            ohlcv_entry_price = _safe_float(
                ohlcv_reference_price,
                default=_safe_float(price, default=np.nan),
            )
            theoretical_entry_price = _first_finite_price(
                trade_context,
                (
                    "theoretical_entry_price",
                    "policy_entry_price",
                    "ohlcv_entry_price",
                    "signal_price",
                    "expected_entry_price",
                    "decision_mid",
                ),
            )
            if not (np.isfinite(theoretical_entry_price) and theoretical_entry_price > 0.0):
                theoretical_entry_price = (
                    float(ohlcv_entry_price)
                    if np.isfinite(ohlcv_entry_price) and ohlcv_entry_price > 0.0
                    else float(expected_entry_price)
                )
            entry_price_delta = (
                float(entry_price) - float(ohlcv_entry_price)
                if np.isfinite(ohlcv_entry_price)
                else np.nan
            )
            entry_price_delta_pct = (
                entry_price_delta / max(abs(float(ohlcv_entry_price)), 1e-12)
                if np.isfinite(entry_price_delta) and np.isfinite(ohlcv_entry_price)
                else np.nan
            )
            entry_delay_adverse_bps = _directional_price_gap_bps(
                side=side,
                actual_price=float(entry_price),
                reference_price=float(theoretical_entry_price),
            )
            entry_delay_effect_bps = (
                -float(entry_delay_adverse_bps)
                if np.isfinite(entry_delay_adverse_bps)
                else np.nan
            )
            entry_ts = pd.Timestamp.now(tz="UTC")
            decision_to_entry_seconds = _timestamp_delta_seconds(
                (trade_context or {}).get("decision_ts"),
                entry_ts,
            )
            signal_to_entry_seconds = _timestamp_delta_seconds(
                (trade_context or {}).get("signal_bar_ts"),
                entry_ts,
            )
            decision_price = _first_finite_price(
                trade_context,
                (
                    "decision_mid",
                    "ticker_mid",
                    "mid",
                    "expected_entry_price",
                    "expected_fill_price",
                ),
            )
            if not (np.isfinite(decision_price) and decision_price > 0.0):
                decision_price = float(expected_entry_price)
            entry_attribution = _entry_price_attribution_fields(
                side=side,
                theoretical_entry_price=float(theoretical_entry_price),
                decision_price=float(decision_price),
                fill_price=float(entry_price),
                trade_context=trade_context,
                entry_fee_quote=entry_fee_quote,
                entry_notional_quote=float(entry_price) * float(filled_amount),
            )
            signal_to_entry_alert_seconds = _safe_float(
                (trade_context or {}).get("signal_to_entry_alert_seconds"),
                default=600.0,
            )
            if (
                np.isfinite(signal_to_entry_seconds)
                and np.isfinite(signal_to_entry_alert_seconds)
                and signal_to_entry_seconds > signal_to_entry_alert_seconds
            ):
                tprint(
                    "[STALE_SIGNAL_ENTRY_ALERT] "
                    f"{symbol} {side} signal_to_entry_seconds="
                    f"{signal_to_entry_seconds:.1f} "
                    f"alert_seconds={signal_to_entry_alert_seconds:.1f} "
                    f"signal_bar_ts={(trade_context or {}).get('signal_bar_ts')} "
                    f"signal_bar_close_ts={(trade_context or {}).get('signal_bar_close_ts')}"
                )
            base_fee = _filled_base_fee(order, exchange_symbol)
            stop_amount_source = filled_amount
            if (
                side == "long"
                and np.isfinite(stop_amount_source)
                and stop_amount_source > base_fee
            ):
                stop_amount_source = max((stop_amount_source - base_fee) * 0.999, 0.0)
            stop_amount = (
                _exchange_precision(
                    self.exchange, exchange_symbol, stop_amount_source, kind="amount"
                )
                if np.isfinite(stop_amount_source) and stop_amount_source > 0
                else amount_base
            )

            oco_result = None
            if bucket_key and self.oco_executor:
                oco_result = self.oco_executor.place_oco_order(
                    symbol=exchange_symbol,
                    side=side,
                    entry_price=entry_price,
                    size=stop_amount,
                    bucket_key=bucket_key,
                    barrier_frac=live_barrier_frac,
                )
                if isinstance(trade_context, dict):
                    with self.oco_executor._positions_lock:
                        state = self.oco_executor.active_positions.get(exchange_symbol)
                        if isinstance(state, dict):
                            canonical_policy_fields = {
                                "stop_price": state.get("stop_price"),
                                "barrier_frac": state.get("barrier_frac"),
                                "barrier_pct": state.get("barrier_pct"),
                                "sl_mult": state.get("sl_mult"),
                                "strategy_id": state.get("strategy_id"),
                                "stop_policy_params_source": state.get(
                                    "stop_policy_params_source"
                                ),
                                "stop_policy_params_hash": state.get(
                                    "stop_policy_params_hash"
                                ),
                                "stop_policy_schema": state.get("stop_policy_schema"),
                                "decision_module": state.get("decision_module"),
                            }
                            state.update(
                                {
                                    **trade_context,
                                    **canonical_policy_fields,
                                    "quote_size": float(size),
                                    "requested_base_amount": amount_base,
                                    "entry_notional_quote": entry_price * filled_amount,
                                    "base_fee_amount": base_fee,
                                    "entry_fee_quote": entry_fee_quote,
                                    "entry_fee_cost": entry_fee_cost,
                                    "entry_fee_currency": entry_fee_currency,
                                    "entry_fee_source": entry_fee_source,
                                    "entry_order_type": entry_order_type,
                                    "ohlcv_entry_price": ohlcv_entry_price,
                                    "theoretical_entry_price": theoretical_entry_price,
                                    "policy_entry_price": theoretical_entry_price,
                                    "realized_entry_price": entry_price,
                                    "entry_price_delta_vs_ohlcv": entry_price_delta,
                                    "entry_price_delta_vs_ohlcv_pct": entry_price_delta_pct,
                                    "entry_delay_adverse_bps": entry_delay_adverse_bps,
                                    "entry_delay_effect_bps": entry_delay_effect_bps,
                                    "entry_delay_abs_bps": (
                                        abs(float(entry_delay_adverse_bps))
                                        if np.isfinite(entry_delay_adverse_bps)
                                        else np.nan
                                    ),
                                    "decision_to_entry_seconds": decision_to_entry_seconds,
                                    "signal_to_entry_seconds": signal_to_entry_seconds,
                                    **entry_attribution,
                                    "expected_friction_drag_bps": trade_context.get(
                                        "expected_total_entry_friction_bps"
                                    ),
                                }
                            )

            context = {
                k: v
                for k, v in dict(trade_context or {}).items()
                if k not in CANONICAL_STOP_POSITION_FIELDS
            }
            audit_fields = _execution_audit_fields(context)
            canonical_stop_fields = _canonical_stop_fields_from_oco_result(oco_result)
            with self._state_lock:
                self.positions[symbol] = {
                    "side": side,
                    "size": stop_amount,
                    "quote_size": float(size),
                    "requested_base_amount": amount_base,
                    "entry_notional_quote": entry_price * filled_amount,
                    "base_fee_amount": base_fee,
                    "entry_fee_quote": entry_fee_quote,
                    "entry_fee_cost": entry_fee_cost,
                    "entry_fee_currency": entry_fee_currency,
                    "entry_fee_source": entry_fee_source,
                    "entry_price": entry_price,
                    "realized_entry_price": entry_price,
                    "theoretical_entry_price": theoretical_entry_price,
                    "policy_entry_price": theoretical_entry_price,
                    "ohlcv_entry_price": ohlcv_entry_price,
                    "entry_price_delta_vs_ohlcv": entry_price_delta,
                    "entry_price_delta_vs_ohlcv_pct": entry_price_delta_pct,
                    "entry_delay_adverse_bps": entry_delay_adverse_bps,
                    "entry_delay_effect_bps": entry_delay_effect_bps,
                    "entry_delay_abs_bps": (
                        abs(float(entry_delay_adverse_bps))
                        if np.isfinite(entry_delay_adverse_bps)
                        else np.nan
                    ),
                    "decision_to_entry_seconds": decision_to_entry_seconds,
                    "signal_to_entry_seconds": signal_to_entry_seconds,
                    **entry_attribution,
                    "expected_friction_drag_bps": (trade_context or {}).get(
                        "expected_total_entry_friction_bps"
                    ),
                    "timestamp": entry_ts,
                    "entry_time": entry_ts,
                    "bucket_key": bucket_key,
                    "partial_fill": partial_fill,
                    "entry_order_type": entry_order_type,
                    **context,
                    **canonical_stop_fields,
                }
                self._last_trade_timestamps[symbol] = pd.Timestamp.now(tz="UTC")

            if oco_result is not None and not oco_result.get("success", False):
                error_category = oco_result.get("error_category") or "stop_loss_failed"
                canonical_stop_fields = _canonical_stop_fields_from_oco_result(
                    oco_result
                )
                return {
                    "success": False,
                    "error": oco_result.get("error") or "stop_loss_not_placed",
                    "error_category": error_category,
                    "order": order,
                    "oco_result": oco_result,
                    "symbol": symbol,
                    "side": side,
                    "size": float(size),
                    "base_amount": stop_amount,
                    "base_fee_amount": base_fee,
                    "entry_fee_quote": entry_fee_quote,
                    "entry_fee_cost": entry_fee_cost,
                    "entry_fee_currency": entry_fee_currency,
                    "entry_fee_source": entry_fee_source,
                    "expected_entry_price": expected_entry_price,
                    "theoretical_entry_price": theoretical_entry_price,
                    "policy_entry_price": theoretical_entry_price,
                    "realized_entry_price": entry_price,
                    "ohlcv_entry_price": ohlcv_entry_price,
                    "entry_price_delta_vs_ohlcv": entry_price_delta,
                    "entry_price_delta_vs_ohlcv_pct": entry_price_delta_pct,
                    "entry_delay_adverse_bps": entry_delay_adverse_bps,
                    "entry_delay_effect_bps": entry_delay_effect_bps,
                    "entry_delay_abs_bps": (
                        abs(float(entry_delay_adverse_bps))
                        if np.isfinite(entry_delay_adverse_bps)
                        else np.nan
                    ),
                    "decision_to_entry_seconds": decision_to_entry_seconds,
                    "signal_to_entry_seconds": signal_to_entry_seconds,
                    **entry_attribution,
                    "expected_friction_drag_bps": (trade_context or {}).get(
                        "expected_total_entry_friction_bps"
                    ),
                    "partial_fill": partial_fill,
                    "entry_order_type": entry_order_type,
                    **canonical_stop_fields,
                    **audit_fields,
                }

            canonical_stop_fields = _canonical_stop_fields_from_oco_result(oco_result)
            return {
                "success": True,
                "order": order,
                "oco_result": oco_result,
                "symbol": symbol,
                "side": side,
                "size": float(size),
                "base_amount": stop_amount,
                "base_fee_amount": base_fee,
                "entry_fee_quote": entry_fee_quote,
                "entry_fee_cost": entry_fee_cost,
                "entry_fee_currency": entry_fee_currency,
                "entry_fee_source": entry_fee_source,
                "price": entry_price,
                "expected_entry_price": expected_entry_price,
                "theoretical_entry_price": theoretical_entry_price,
                "policy_entry_price": theoretical_entry_price,
                "realized_entry_price": entry_price,
                "ohlcv_entry_price": ohlcv_entry_price,
                "entry_price_delta_vs_ohlcv": entry_price_delta,
                "entry_price_delta_vs_ohlcv_pct": entry_price_delta_pct,
                "entry_delay_adverse_bps": entry_delay_adverse_bps,
                "entry_delay_effect_bps": entry_delay_effect_bps,
                "entry_delay_abs_bps": (
                    abs(float(entry_delay_adverse_bps))
                    if np.isfinite(entry_delay_adverse_bps)
                    else np.nan
                ),
                "decision_to_entry_seconds": decision_to_entry_seconds,
                "signal_to_entry_seconds": signal_to_entry_seconds,
                **entry_attribution,
                "expected_friction_drag_bps": (trade_context or {}).get(
                    "expected_total_entry_friction_bps"
                ),
                "stop_price": (
                    oco_result.get("stop_price")
                    if isinstance(oco_result, dict)
                    else None
                ),
                "stop_order_id": (
                    oco_result.get("stop_order_id")
                    if isinstance(oco_result, dict)
                    else None
                ),
                "barrier_frac": (
                    oco_result.get("barrier_frac")
                    if isinstance(oco_result, dict)
                    else None
                ),
                **canonical_stop_fields,
                "partial_fill": partial_fill,
                "entry_order_type": entry_order_type,
                "price_slippage_pct": (
                    (entry_price - expected_entry_price)
                    / max(abs(expected_entry_price), 1e-12)
                    if np.isfinite(expected_entry_price)
                    else 0.0
                ),
                **audit_fields,
            }

        except Exception as e:
            category = _classify_exchange_error(e)
            tprint(f"Error executing live trade for {symbol}: {category}: {e}")
            return {"success": False, "error": str(e), "error_category": category}

    def _quote_to_base_amount(
        self,
        symbol: str,
        *,
        quote_size: float,
        reference_price: float,
        market: Optional[Dict[str, Any]] = None,
    ) -> float:
        """Convert quote-currency notional into exchange-ready base quantity."""
        if not np.isfinite(quote_size) or quote_size <= 0:
            raise ValueError(f"invalid quote size for {symbol}: {quote_size}")
        if not np.isfinite(reference_price) or reference_price <= 0:
            raise ValueError(f"invalid reference price for {symbol}: {reference_price}")
        market_info = (
            market if market is not None else _load_market(self.exchange, symbol)
        )
        raw_amount = quote_size / reference_price
        amount = _exchange_precision(self.exchange, symbol, raw_amount, kind="amount")
        _validate_order_filters(
            symbol, market_info, amount=amount, price=float(reference_price)
        )
        return amount

    def _record_shadow_trade(
        self,
        symbol: str,
        side: str,
        size: float,
        price: Optional[float] = None,
        bucket_key: Optional[str] = None,
        ohlcv_reference_price: Optional[float] = None,
        trade_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Record shadow trade decision.

        Args:
            symbol: Trading symbol
            side: "long" or "short"
            size: Position size in quote currency
            price: Entry price

        Returns:
            Shadow trade record
        """
        params = self.get_simple_policy_stop_params(bucket_key)
        expected_entry_price = float(price) if price is not None else 0.0
        entry_price = expected_entry_price
        ohlcv_entry_price = _safe_float(
            ohlcv_reference_price,
            default=_safe_float(price, default=np.nan),
        )
        theoretical_entry_price = _first_finite_price(
            trade_context,
            (
                "theoretical_entry_price",
                "policy_entry_price",
                "ohlcv_entry_price",
                "signal_price",
                "expected_entry_price",
                "decision_mid",
            ),
        )
        if not (np.isfinite(theoretical_entry_price) and theoretical_entry_price > 0.0):
            theoretical_entry_price = (
                float(ohlcv_entry_price)
                if np.isfinite(ohlcv_entry_price) and ohlcv_entry_price > 0.0
                else float(entry_price)
            )
        entry_price_delta = (
            float(entry_price) - float(ohlcv_entry_price)
            if np.isfinite(ohlcv_entry_price)
            else np.nan
        )
        entry_price_delta_pct = (
            entry_price_delta / max(abs(float(ohlcv_entry_price)), 1e-12)
            if np.isfinite(entry_price_delta) and np.isfinite(ohlcv_entry_price)
            else np.nan
        )
        entry_delay_adverse_bps = _directional_price_gap_bps(
            side=side,
            actual_price=float(entry_price),
            reference_price=float(theoretical_entry_price),
        )
        entry_delay_effect_bps = (
            -float(entry_delay_adverse_bps)
            if np.isfinite(entry_delay_adverse_bps)
            else np.nan
        )
        now_ts = pd.Timestamp.now(tz="UTC")
        decision_to_entry_seconds = _timestamp_delta_seconds(
            (trade_context or {}).get("decision_ts"),
            now_ts,
        )
        signal_to_entry_seconds = _timestamp_delta_seconds(
            (trade_context or {}).get("signal_bar_ts"),
            now_ts,
        )
        decision_price = _first_finite_price(
            trade_context,
            (
                "decision_mid",
                "ticker_mid",
                "mid",
                "expected_entry_price",
                "expected_fill_price",
            ),
        )
        if not (np.isfinite(decision_price) and decision_price > 0.0):
            decision_price = float(expected_entry_price)
        entry_attribution = _entry_price_attribution_fields(
            side=side,
            theoretical_entry_price=float(theoretical_entry_price),
            decision_price=float(decision_price),
            fill_price=float(entry_price),
            trade_context=trade_context,
        )
        signal_to_entry_alert_seconds = _safe_float(
            (trade_context or {}).get("signal_to_entry_alert_seconds"),
            default=600.0,
        )
        if (
            np.isfinite(signal_to_entry_seconds)
            and np.isfinite(signal_to_entry_alert_seconds)
            and signal_to_entry_seconds > signal_to_entry_alert_seconds
        ):
            tprint(
                "[STALE_SIGNAL_ENTRY_ALERT] "
                f"{symbol} {side} signal_to_entry_seconds="
                f"{signal_to_entry_seconds:.1f} "
                f"alert_seconds={signal_to_entry_alert_seconds:.1f} "
                f"signal_bar_ts={(trade_context or {}).get('signal_bar_ts')} "
                f"signal_bar_close_ts={(trade_context or {}).get('signal_bar_close_ts')}"
            )
        try:
            live_barrier_frac = _safe_float(
                (trade_context or {}).get("barrier_frac")
                or (trade_context or {}).get("barrier_pct"),
                default=np.nan,
            )
            initial_stop_decision = compute_initial_simple_policy_stop_decision(
                entry_price=float(entry_price),
                policy_params=params,
                side=side,
                strategy_id=bucket_key,
                barrier_frac=live_barrier_frac,
                require_metadata=True,
            )
        except SimplePolicyStopParamsError as exc:
            err_text = str(exc)
            return {
                "timestamp": datetime.now().isoformat(),
                "mode": "shadow",
                "symbol": symbol,
                "side": side,
                "size": size,
                "price": price,
                "status": "failed",
                "success": False,
                "bucket_key": bucket_key,
                "error": str(exc),
                "error_category": (
                    "missing_policy_barrier_pct"
                    if "barrier_frac/barrier_pct" in err_text
                    else "invalid_simple_policy_stop_params"
                ),
                **dict(trade_context or {}),
            }
        barrier_frac = initial_stop_decision.barrier_frac
        sl_mult = initial_stop_decision.sl_mult
        stop_price = float(initial_stop_decision.stop_price)
        limit_price = None
        record = {
            "timestamp": datetime.now().isoformat(),
            "mode": "shadow",
            "symbol": symbol,
            "side": side,
            "size": size,
            "price": price,
            "expected_entry_price": expected_entry_price,
            "theoretical_entry_price": theoretical_entry_price,
            "policy_entry_price": theoretical_entry_price,
            "realized_entry_price": entry_price,
            "ohlcv_entry_price": ohlcv_entry_price,
            "entry_price_delta_vs_ohlcv": entry_price_delta,
            "entry_price_delta_vs_ohlcv_pct": entry_price_delta_pct,
            "price_slippage_pct": 0.0,
            "entry_delay_adverse_bps": entry_delay_adverse_bps,
            "entry_delay_effect_bps": entry_delay_effect_bps,
            "entry_delay_abs_bps": (
                abs(float(entry_delay_adverse_bps))
                if np.isfinite(entry_delay_adverse_bps)
                else np.nan
            ),
            "decision_to_entry_seconds": decision_to_entry_seconds,
            "signal_to_entry_seconds": signal_to_entry_seconds,
            **entry_attribution,
            "expected_friction_drag_bps": (trade_context or {}).get(
                "expected_total_entry_friction_bps"
            ),
            "spread_proxy_pct": 0.0,
            "status": "recorded",
            "bucket_key": bucket_key,
            **dict(trade_context or {}),
            "stop_price": stop_price,
            "barrier_frac": float(barrier_frac),
            "barrier_pct": float(barrier_frac),
            "sl_mult": float(sl_mult),
            "strategy_id": initial_stop_decision.strategy_id,
            "stop_policy_params_source": initial_stop_decision.params_source,
            "stop_policy_params_hash": initial_stop_decision.params_hash,
            "stop_policy_schema": initial_stop_decision.params_schema,
        }
        shadow_stop_state = {
            "side": side,
            "entry_time": now_ts,
            "timestamp": now_ts,
            "stop_price": stop_price,
            "strategy_id": initial_stop_decision.strategy_id,
            "bucket_key": bucket_key,
        }
        stop_order_id = _shadow_stop_order_id(symbol, shadow_stop_state)
        record["stop_order_id"] = stop_order_id
        record["protective_stop_mode"] = "shadow"
        record["protective_stop_attached"] = True

        # Update positions
        with self._state_lock:
            self.positions[symbol] = {
                "side": side,
                "size": size,
                "entry_price": entry_price,
                "realized_entry_price": entry_price,
                "theoretical_entry_price": theoretical_entry_price,
                "policy_entry_price": theoretical_entry_price,
                "ohlcv_entry_price": ohlcv_entry_price,
                "entry_price_delta_vs_ohlcv": entry_price_delta,
                "entry_price_delta_vs_ohlcv_pct": entry_price_delta_pct,
                "entry_delay_adverse_bps": entry_delay_adverse_bps,
                "entry_delay_effect_bps": entry_delay_effect_bps,
                "entry_delay_abs_bps": (
                    abs(float(entry_delay_adverse_bps))
                    if np.isfinite(entry_delay_adverse_bps)
                    else np.nan
                ),
                "decision_to_entry_seconds": decision_to_entry_seconds,
                "signal_to_entry_seconds": signal_to_entry_seconds,
                **entry_attribution,
                "expected_friction_drag_bps": (trade_context or {}).get(
                    "expected_total_entry_friction_bps"
                ),
                "timestamp": now_ts,
                "entry_time": now_ts,
                "bucket_key": bucket_key,
                "limit_price": limit_price,
                "stop_reason": "original_stop_loss",
                "stop_reason_detail": initial_stop_decision.reason_detail,
                "peak_price": entry_price,
                "mfe": 0.0,
                "mae": 0.0,
                **dict(trade_context or {}),
                "stop_order_id": stop_order_id,
                "protective_stop_mode": "shadow",
                "protective_stop_attached": True,
                "stop_price": stop_price,
                "initial_stop_price": stop_price,
                "barrier_frac": barrier_frac,
                "barrier_pct": barrier_frac,
                "sl_mult": sl_mult,
                "trailing_activation_mult": initial_stop_decision.trailing_activation_mult,
                "trailing_power": initial_stop_decision.trailing_power,
                "trailing_squash_divisor": initial_stop_decision.trailing_squash_divisor,
                "giveback_beta": initial_stop_decision.giveback_beta,
                "atr_power": initial_stop_decision.atr_power,
                "atr_multiplier": initial_stop_decision.atr_multiplier,
                "hard_tp_abs_pct": initial_stop_decision.hard_tp_abs_pct,
                "capital_protect_mfe_mult": initial_stop_decision.capital_protect_mfe_mult,
                "capital_protect_regression_frac": initial_stop_decision.capital_protect_regression_frac,
                "decision_module": initial_stop_decision.decision_module,
                "stop_policy_params_source": initial_stop_decision.params_source,
                "stop_policy_params_hash": initial_stop_decision.params_hash,
                "stop_policy_schema": initial_stop_decision.params_schema,
                "strategy_id": initial_stop_decision.strategy_id,
                "last_update": now_ts,
                "last_5m_eval_ts": None,
            }
            _append_position_event(
                self.positions[symbol],
                "entry_stop_created",
                entry_price=float(entry_price),
                stop_price=stop_price,
                stop_dev_pct=(
                    (float(stop_price) - float(entry_price))
                    / max(float(entry_price), 1e-12)
                    if stop_price is not None
                    else np.nan
                ),
                sl_mult=float(sl_mult),
                barrier_frac=float(barrier_frac),
                stop_reason="original_stop_loss",
                params_source=initial_stop_decision.params_source,
                params_hash=initial_stop_decision.params_hash,
                schema=initial_stop_decision.params_schema,
                decision_module=initial_stop_decision.decision_module,
                trailing_activation_mult=initial_stop_decision.trailing_activation_mult,
                trailing_power=initial_stop_decision.trailing_power,
                trailing_squash_divisor=initial_stop_decision.trailing_squash_divisor,
                giveback_beta=initial_stop_decision.giveback_beta,
                atr_power=initial_stop_decision.atr_power,
                atr_multiplier=initial_stop_decision.atr_multiplier,
                hard_tp_abs_pct=initial_stop_decision.hard_tp_abs_pct,
                capital_protect_mfe_mult=initial_stop_decision.capital_protect_mfe_mult,
                capital_protect_regression_frac=initial_stop_decision.capital_protect_regression_frac,
            )
            self._last_trade_timestamps[symbol] = now_ts

        return record

    def close_position(
        self,
        symbol: str,
        price: Optional[float] = None,
        reason: str = "manual_close",
    ) -> Dict[str, Any]:
        """Close an existing position.

        Args:
            symbol: Trading symbol
            price: Exit price (None for market)

        Returns:
            Close result
        """
        if symbol not in self.positions:
            return {"success": False, "error": "No position for symbol"}

        position = self.positions[symbol]
        side = position["side"]
        size = position["size"]

        if _is_live_execution_mode(self.mode):
            return self._close_live(symbol, side, size, price, reason=reason)
        else:
            return self._record_shadow_close(symbol, side, size, price, reason=reason)

    def _close_live(
        self,
        symbol: str,
        side: str,
        size: float,
        price: Optional[float] = None,
        reason: str = "manual_close",
    ) -> Dict[str, Any]:
        """Close live position."""
        # Cancel OCO orders first if using OCO executor
        oco_state: Optional[Dict[str, Any]] = None
        if self.oco_executor:
            with self.oco_executor._positions_lock:
                raw_state = self.oco_executor.active_positions.get(symbol)
                if isinstance(raw_state, dict):
                    oco_state = raw_state
        if self.oco_executor and oco_state is not None:
            try:
                current_price = _safe_float(price, default=np.nan)
                if not np.isfinite(current_price) or current_price <= 0.0:
                    ticker = self.exchange.fetch_ticker(symbol)
                    current_price = float(ticker["last"])
                state = oco_state
                self.oco_executor._close_position(symbol, state, current_price, reason)
                with self._state_lock:
                    self.positions.pop(symbol, None)
                close_metrics = state.get("last_close_metrics")
                return {
                    "success": True,
                    "symbol": symbol,
                    "side": "closed",
                    "size": size,
                    "reason": reason,
                    "closed_trade": close_metrics,
                }
            except Exception as e:
                tprint(
                    f"Error canceling OCO for {symbol}: "
                    f"{_classify_exchange_error(e)}: {e}"
                )

        try:
            order_side = "sell" if side == "long" else "buy"

            if price is not None:
                order = self.exchange.create_order(
                    symbol=symbol,
                    type="limit",
                    side=order_side,
                    amount=size,
                    price=price,
                    params=_order_params(self.config, reduce_only=True),
                )
            else:
                order = self.exchange.create_order(
                    symbol=symbol,
                    type="market",
                    side=order_side,
                    amount=size,
                    params=_order_params(self.config, reduce_only=True),
                )

            with self._state_lock:
                if symbol in self.positions:
                    del self.positions[symbol]

            return {
                "success": True,
                "order": order,
                "symbol": symbol,
                "side": "closed",
                "size": size,
                "reason": reason,
            }
        except Exception as e:
            category = _classify_exchange_error(e)
            tprint(f"Error closing live trade for {symbol}: {category}: {e}")
            return {"success": False, "error": str(e), "error_category": category}

    def _record_shadow_close(
        self,
        symbol: str,
        side: str,
        size: float,
        price: Optional[float] = None,
        reason: str = "manual_close",
    ) -> Dict[str, Any]:
        """Record shadow position close."""
        record = {
            "timestamp": datetime.now().isoformat(),
            "mode": "shadow",
            "symbol": symbol,
            "side": "closed",
            "original_side": side,
            "size": size,
            "price": price,
            "status": "closed",
            "reason": reason,
        }

        with self._state_lock:
            if symbol in self.positions:
                del self.positions[symbol]

        return record

    def get_positions(self) -> Dict[str, Dict[str, Any]]:
        """Get current positions."""
        with self._state_lock:
            return self.positions.copy()

    def get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get current position for a symbol."""
        if self.oco_executor is not None:
            with self.oco_executor._positions_lock:
                state = self.oco_executor.active_positions.get(symbol)
                if isinstance(state, dict):
                    return dict(state)
        with self._state_lock:
            pos = self.positions.get(symbol)
            return dict(pos) if isinstance(pos, dict) else None

    def get_active_positions(self) -> Dict[str, Dict[str, Any]]:
        """Return currently active positions for live or shadow mode."""
        if self.oco_executor is not None:
            return self.oco_executor.get_active_positions()
        with self._state_lock:
            return {
                sym: dict(state)
                for sym, state in self.positions.items()
                if isinstance(state, dict)
            }

    def retry_missing_protective_stop(
        self, symbol: str, position_state: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Retry attaching the policy-derived STOP_LOSS for a monitored position."""
        if self.oco_executor is None:
            with self._state_lock:
                state = self.positions.get(symbol)
                if not isinstance(state, dict):
                    state = position_state
                if not isinstance(state, dict):
                    return {
                        "success": False,
                        "error_category": "position_not_found",
                        "error": "cannot retry shadow protective stop for unknown position",
                    }
                missing_provenance = _validate_policy_stop_provenance(state)
                if missing_provenance:
                    return {
                        "success": False,
                        "error_category": "missing_policy_provenance",
                        "error": (
                            "cannot reattach shadow stop without policy provenance: "
                            + ",".join(missing_provenance)
                        ),
                    }
                before_id = state.get("stop_order_id")
                stop_order_id = str(before_id or _shadow_stop_order_id(symbol, state))
                state["stop_order_id"] = stop_order_id
                state["stop_order_ids"] = [stop_order_id]
                state["protective_stop_mode"] = "shadow"
                state["protective_stop_attached"] = True
                state["last_order_status"] = "open"
                _append_position_event(
                    state,
                    "shadow_stop_reattach_result",
                    previous_stop_order_id=before_id,
                    stop_order_id=stop_order_id,
                    stop_price=state.get("stop_price"),
                    success=True,
                )
                return {
                    "success": True,
                    "mode": "shadow",
                    "simulated": True,
                    "stop_order_id": stop_order_id,
                    "stop_price": state.get("stop_price"),
                }
        with self.oco_executor._positions_lock:
            state = self.oco_executor.active_positions.get(symbol)
            if not isinstance(state, dict):
                state = position_state
            if not isinstance(state, dict):
                return {
                    "success": False,
                    "error_category": "position_not_found",
                    "error": "cannot retry protective stop for unknown position",
                }
        result = self.oco_executor._reattach_protective_stop(
            symbol,
            state,
            previous_status="missing_stop_order",
        )
        with self._state_lock:
            if symbol in self.positions and isinstance(self.positions[symbol], dict):
                self.positions[symbol].update(state)
        return result

    def monitor_orders_once(self) -> Dict[str, Dict[str, Any]]:
        """Fetch live order statuses for active stop-loss orders once."""
        if self.oco_executor is not None:
            statuses = self.oco_executor.monitor_order_statuses()
            closed_symbols = [
                symbol
                for symbol, status in statuses.items()
                if isinstance(status, dict)
                and str(status.get("status", "")).lower() in {"closed", "filled"}
            ]
            if closed_symbols:
                with self._state_lock:
                    for symbol in closed_symbols:
                        self.positions.pop(symbol, None)
            return statuses
        statuses: Dict[str, Dict[str, Any]] = {}
        now_utc = pd.Timestamp.now(tz="UTC")
        with self._state_lock:
            items = list(self.positions.items())
            for symbol, state in items:
                if not isinstance(state, dict):
                    continue
                status = {
                    "status": "open",
                    "mode": "shadow",
                    "symbol": symbol,
                    "side": state.get("side"),
                    "size": state.get("size"),
                    "entry_price": state.get("entry_price"),
                    "stop_price": state.get("stop_price"),
                    "stop_order_id": state.get("stop_order_id"),
                    "protective_stop_mode": state.get("protective_stop_mode"),
                    "protective_stop_attached": state.get("protective_stop_attached"),
                    "bucket_key": state.get("bucket_key"),
                    "last_order_check_ts": now_utc,
                }
                state["last_order_status"] = "open"
                state["last_order_check_ts"] = now_utc
                statuses[symbol] = status
        return statuses

    def update_position_policy_state(
        self,
        symbol: str,
        *,
        policy_stop_decision: Optional[Any] = None,
        limit_price: Optional[float] = None,
        peak_price: Optional[float] = None,
        mfe: Optional[float] = None,
        mae: Optional[float] = None,
        bars_in_trade: Optional[int] = None,
        last_5m_eval_ts: Optional[pd.Timestamp] = None,
        current_price: Optional[float] = None,
        current_price_source: Optional[str] = None,
        policy_entry_price: Optional[float] = None,
        policy_entry_price_source: Optional[str] = None,
        current_price_ts: Optional[pd.Timestamp] = None,
        shadow_simple_policy_state: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Persist monitor state and apply policy-authorised stop decisions only."""

        decision = policy_stop_decision
        if self.oco_executor is not None:
            with self.oco_executor._positions_lock:
                state = self.oco_executor.active_positions.get(symbol)
                if state is None:
                    return None
                if decision is not None:
                    self.oco_executor._update_stop_loss_from_policy_decision(
                        symbol, state, decision
                    )
                    close_metrics = state.get("last_close_metrics")
                    if isinstance(close_metrics, dict):
                        with self._state_lock:
                            self.positions.pop(symbol, None)
                        return {"closed_trade": close_metrics}
                if limit_price is not None and np.isfinite(limit_price):
                    state["limit_price"] = float(limit_price)
                if peak_price is not None and np.isfinite(peak_price):
                    state["peak_price"] = float(peak_price)
                if mfe is not None and np.isfinite(mfe):
                    state["mfe"] = float(mfe)
                if mae is not None and np.isfinite(mae):
                    state["mae"] = float(mae)
                if current_price is not None and np.isfinite(current_price):
                    state["current_price"] = float(current_price)
                    state["last_price"] = float(current_price)
                if current_price_source:
                    state["current_price_source"] = str(current_price_source)
                if policy_entry_price is not None and np.isfinite(policy_entry_price):
                    state["policy_entry_price"] = float(policy_entry_price)
                    state.setdefault("theoretical_entry_price", float(policy_entry_price))
                if policy_entry_price_source:
                    state["policy_entry_price_source"] = str(policy_entry_price_source)
                if current_price_ts is not None:
                    state["current_price_ts"] = pd.Timestamp(current_price_ts)
                if isinstance(shadow_simple_policy_state, dict):
                    state["simple_policy_shadow"] = shadow_simple_policy_state
                if bars_in_trade is not None:
                    state["bars_in_trade"] = int(max(0, bars_in_trade))
                if last_5m_eval_ts is not None:
                    state["last_5m_eval_ts"] = pd.Timestamp(last_5m_eval_ts)
                    state["last_update"] = pd.Timestamp.now(tz="UTC")
            return None
        with self._state_lock:
            state = self.positions.get(symbol)
            if state is None:
                return None
            if decision is not None:
                artifact_params = self.get_simple_policy_stop_params(
                    str(state.get("strategy_id") or state.get("bucket_key") or "")
                )
                valid, reason = _validate_policy_stop_decision(
                    decision,
                    require_should_replace=True,
                    position_state=state,
                    artifact_params=artifact_params,
                )
                if not valid:
                    state["stop_update_error"] = reason
                    state["stop_update_error_category"] = "unauthorised_stop_update"
                    _append_position_event(
                        state,
                        "stop_replace_skipped",
                        reason="invalid_simple_policy_decision_metadata",
                        error=reason,
                    )
                elif (
                    isinstance(decision, SimplePolicyStopDecision)
                    and decision.should_replace
                ):
                    stop_price_value = _safe_float(decision.stop_price, default=np.nan)
                    current_stop = _safe_float(state.get("stop_price"), default=np.nan)
                    side = str(state.get("side", "long")).lower()
                    improved = (
                        (
                            stop_price_value > current_stop
                            if side == "long"
                            else stop_price_value < current_stop
                        )
                        if valid
                        else False
                    )
                    if valid and improved:
                        state["stop_price"] = stop_price_value
                        state["stop_reason"] = decision.reason
                        state["stop_reason_detail"] = decision.reason_detail
                        state["stop_policy_params_source"] = decision.params_source
                        state["stop_policy_params_hash"] = decision.params_hash
                        state["stop_policy_schema"] = decision.params_schema
                        state["strategy_id"] = decision.strategy_id
                        state["barrier_frac"] = decision.barrier_frac
                        state["barrier_pct"] = decision.barrier_frac
                        state["sl_mult"] = decision.sl_mult
                        state["requested_policy_stop"] = stop_price_value
                        _append_position_event(
                            state,
                            "stop_replaced",
                            stop_reason=decision.reason,
                            previous_stop=current_stop,
                            new_stop=stop_price_value,
                            requested_policy_stop=stop_price_value,
                            final_placed_stop=stop_price_value,
                            strategy_id=decision.strategy_id,
                            params_source=decision.params_source,
                            params_hash=decision.params_hash,
                        )
                    else:
                        state["stop_update_error"] = (
                            reason or "non-improving simple-policy stop decision"
                        )
                        state["stop_update_error_category"] = "unauthorised_stop_update"
                        _append_position_event(
                            state,
                            "stop_replace_skipped",
                            reason="invalid_simple_policy_decision",
                            candidate_stop=decision.stop_price,
                            strategy_id=decision.strategy_id,
                            params_source=decision.params_source,
                            params_hash=decision.params_hash,
                        )
            if limit_price is not None and np.isfinite(limit_price):
                state["limit_price"] = float(limit_price)
            if peak_price is not None and np.isfinite(peak_price):
                state["peak_price"] = float(peak_price)
            if mfe is not None and np.isfinite(mfe):
                state["mfe"] = float(mfe)
            if mae is not None and np.isfinite(mae):
                state["mae"] = float(mae)
            if current_price is not None and np.isfinite(current_price):
                state["current_price"] = float(current_price)
                state["last_price"] = float(current_price)
            if current_price_source:
                state["current_price_source"] = str(current_price_source)
            if policy_entry_price is not None and np.isfinite(policy_entry_price):
                state["policy_entry_price"] = float(policy_entry_price)
                state.setdefault("theoretical_entry_price", float(policy_entry_price))
            if policy_entry_price_source:
                state["policy_entry_price_source"] = str(policy_entry_price_source)
            if current_price_ts is not None:
                state["current_price_ts"] = pd.Timestamp(current_price_ts)
            if isinstance(shadow_simple_policy_state, dict):
                state["simple_policy_shadow"] = shadow_simple_policy_state
            if bars_in_trade is not None:
                state["bars_in_trade"] = int(max(0, bars_in_trade))
            if last_5m_eval_ts is not None:
                state["last_5m_eval_ts"] = pd.Timestamp(last_5m_eval_ts)
                state["last_update"] = pd.Timestamp.now(tz="UTC")
        return None

    def get_oco_positions(self) -> Dict[str, Dict[str, Any]]:
        """Get current OCO positions."""
        if self.oco_executor:
            return self.oco_executor.get_active_positions()
        return self.get_active_positions()

    def fetch_5m_ohlcv_for_positions(self) -> Dict[str, pd.DataFrame]:
        """Fetch 5m OHLCV for all active OCO positions.

        Uses hf_data_loader.fetch_ohlcv_5m() to get high-frequency
        data for trailing profit analysis.

        Returns:
            Dictionary mapping symbol to 5m OHLCV DataFrame
        """
        if self.oco_executor:
            return self.oco_executor.fetch_5m_ohlcv_for_positions()
        results: Dict[str, pd.DataFrame] = {}
        if self.exchange is None:
            with self._state_lock:
                items = list(self.positions.items())
            for symbol, state in items:
                if not isinstance(state, dict):
                    continue
                bars = state.get("ohlcv_5m_latest", state.get("ohlcv_5m"))
                if (
                    bars is not None
                    and isinstance(bars, (pd.DataFrame, pd.Series))
                    and not (hasattr(bars, "empty") and bars.empty)
                ):
                    results[symbol] = pd.DataFrame(bars)
            return results

        current_time = pd.Timestamp.now(tz="UTC")
        with self._state_lock:
            items = list(self.positions.items())
        for symbol, state in items:
            if not isinstance(state, dict):
                continue
            try:
                entry_time = pd.Timestamp(state.get("entry_time", current_time))
                bars = hf_data_loader.fetch_ohlcv_5m(
                    self.exchange,
                    symbol,
                    entry_time - pd.Timedelta(hours=1),
                    current_time + pd.Timedelta(hours=1),
                )
                if (
                    bars is not None
                    and isinstance(bars, (pd.DataFrame, pd.Series))
                    and not (hasattr(bars, "empty") and bars.empty)
                ):
                    frame = pd.DataFrame(bars)
                    results[symbol] = frame
                    with self._state_lock:
                        if symbol in self.positions:
                            self.positions[symbol]["ohlcv_5m"] = frame
            except Exception as exc:
                tprint(
                    f"Error fetching shadow 5m OHLCV for {symbol}: "
                    f"{_classify_exchange_error(exc)}: {exc}"
                )
        return results

    def shutdown(self):
        """Shutdown executor and cleanup resources."""
        close_on_shutdown = _config_bool(
            self.config.get(
                "close_positions_on_shutdown",
                os.environ.get("EPM_CLOSE_POSITIONS_ON_SHUTDOWN"),
            ),
            default=False,
        )
        if self.oco_executor and close_on_shutdown:
            self.oco_executor.close_all_positions()
        elif self.oco_executor:
            tprint(
                "TradeExecutor shutdown: preserving active positions; set "
                "close_positions_on_shutdown=true to flatten on shutdown"
            )
        tprint("TradeExecutor shutdown complete")


def execute_live_trade(
    executor: TradeExecutor,
    symbol: str,
    side: str,
    size: float,
    price: Optional[float] = None,
    bucket_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute a live trade.

    Args:
        executor: TradeExecutor instance
        symbol: Trading symbol
        side: "long" or "short"
        size: Position size
        price: Limit price
        bucket_key: Bucket key for OCO parameters

    Returns:
        Execution result
    """
    if not _is_live_execution_mode(executor.mode):
        return {"success": False, "error": "Not in live mode"}

    return executor.execute_trade(symbol, side, size, price, bucket_key)


def record_shadow_trade(
    executor: TradeExecutor,
    symbol: str,
    side: str,
    size: float,
    price: Optional[float] = None,
) -> Dict[str, Any]:
    """Record a shadow trade.

    Args:
        executor: TradeExecutor instance
        symbol: Trading symbol
        side: "long" or "short"
        size: Position size
        price: Entry price

    Returns:
        Shadow record
    """
    if executor.mode != "shadow":
        return {"success": False, "error": "Not in shadow mode"}

    return executor._record_shadow_trade(symbol, side, size, price)
