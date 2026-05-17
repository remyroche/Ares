"""Typed live portfolio policy contract for inference and offline replay."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from extreme_price_movements.path_utils import resolve_mode_file


@dataclass(frozen=True)
class PortfolioPolicyConfig:
    """Shared source of truth for live portfolio policy defaults."""

    schema_version: str = "portfolio_policy_v1"

    max_concurrent_positions: int = 8
    max_concurrent_per_side: Optional[int] = None
    max_concurrent_per_strategy: Optional[int] = None
    reserved_position_slots: Optional[int] = None

    max_total_wallet_allocation_pct: float = 0.95
    max_available_wallet_position_pct: float = 0.50
    max_position_wallet_pct: float = 0.20
    max_position_quote_notional: float = 5000.0
    book_notional_multiplier: float = 1.0
    leverage_wallet_multiplier: float = 1.0
    min_margin_level_after_entry: float = 2.50

    live_test_min_quote_notional: float = 5.0
    live_test_quote_notional: float = 10.0

    initial_rank_threshold: float = 0.90
    initial_rank_threshold_floor: float = 0.90
    dynamic_threshold_enabled: bool = True

    side_crowding_penalty_max: float = 0.03
    strategy_crowding_penalty_max: float = 0.03
    price_gap_penalty_max: float = 0.05

    rank_multiplier_min: float = 0.80
    rank_multiplier_max: float = 1.60
    rank_size_power: float = 1.10

    ticker_precheck_enabled: bool = True
    orderbook_precheck_enabled: bool = True
    max_orderbook_slippage_bps: float = 50.0
    max_entry_friction_bps: float = 60.0
    max_spread_bps: float = 25.0
    hard_max_spread_bps: float = 100.0
    min_liquidity_capacity_weight: float = 0.25
    max_ticker_age_seconds: float = 4.0

    max_signal_gap_bps_default: float = 150.0
    max_order_chase_bps: float = 30.0
    entry_order_timeout_seconds: float = 10.0
    entry_order_max_retries: int = 1

    top_prediction_ledger_pct: float = 0.15
    enable_symbol_underperformance_gates: bool = False

    def resolved_max_concurrent_per_side(self) -> int:
        if self.max_concurrent_per_side is not None:
            return int(self.max_concurrent_per_side)
        return int(0.75 * self.max_concurrent_positions)

    def resolved_max_concurrent_per_strategy(self) -> int:
        if self.max_concurrent_per_strategy is not None:
            return int(self.max_concurrent_per_strategy)
        return int(0.75 * self.max_concurrent_positions)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _coerce_value(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(default, bool):
        return bool(value)
    if isinstance(default, int) and not isinstance(default, bool):
        return int(value)
    if isinstance(default, float):
        return float(value)
    return value


def _load_artifact_payload(data_root: str, run_id: str) -> Dict[str, Any]:
    path = (
        Path(data_root)
        / "artifacts"
        / str(run_id)
        / "policy_params"
        / "portfolio_policy_config.json"
    )
    path = resolve_mode_file(path)
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def load_portfolio_policy_config(
    *,
    data_root: str,
    run_id: str,
    runtime_cfg: Optional[Dict[str, Any]] = None,
) -> PortfolioPolicyConfig:
    """Load portfolio policy using artifact -> runtime -> dataclass precedence."""
    defaults = PortfolioPolicyConfig()
    values = defaults.to_dict()
    valid = {f.name for f in fields(PortfolioPolicyConfig)}
    aliases = {
        "symbol_underperformance_gates_enabled": "enable_symbol_underperformance_gates",
    }
    nested_sections = {
        "rank_sizing": {
            "book_notional_multiplier",
            "leverage_wallet_multiplier",
            "min_margin_level_after_entry",
            "rank_multiplier_min",
            "rank_multiplier_max",
            "rank_size_power",
        },
        "liquidity": {
            "max_orderbook_slippage_bps",
            "max_entry_friction_bps",
            "max_spread_bps",
            "hard_max_spread_bps",
            "min_liquidity_capacity_weight",
        },
    }

    runtime_cfg = runtime_cfg or {}
    for source in (runtime_cfg, _load_artifact_payload(data_root, run_id)):
        for section, keys in nested_sections.items():
            nested = source.get(section)
            if isinstance(nested, dict):
                for key in keys:
                    if key in nested:
                        values[key] = _coerce_value(nested[key], values.get(key))
        for key, value in source.items():
            key = aliases.get(key, key)
            if key not in valid:
                continue
            values[key] = _coerce_value(value, values.get(key))

    values["max_concurrent_positions"] = max(1, int(values["max_concurrent_positions"]))
    if values.get("reserved_position_slots") is not None:
        values["reserved_position_slots"] = max(
            1, int(values["reserved_position_slots"])
        )
    if values.get("max_concurrent_per_side") is not None:
        values["max_concurrent_per_side"] = max(
            1, int(values["max_concurrent_per_side"])
        )
    if values.get("max_concurrent_per_strategy") is not None:
        values["max_concurrent_per_strategy"] = max(
            1, int(values["max_concurrent_per_strategy"])
        )
    values["book_notional_multiplier"] = max(
        0.0, float(values.get("book_notional_multiplier", 1.0))
    )
    values["leverage_wallet_multiplier"] = max(
        1.0, float(values.get("leverage_wallet_multiplier", 1.0))
    )
    values["min_margin_level_after_entry"] = max(
        1.0, float(values.get("min_margin_level_after_entry", 2.5))
    )
    return PortfolioPolicyConfig(**{k: values[k] for k in valid})


def compute_rank_based_position_size(
    *,
    wallet_value: float,
    open_notional: float,
    adjusted_rank_score: float,
    final_threshold: float,
    policy: PortfolioPolicyConfig,
    liquidity_capacity_weight: float = 1.0,
    live_test_mode: bool = False,
    rank_size_power: float | None = None,
    total_assets_quote: float | None = None,
    total_liabilities_quote: float | None = None,
    open_positions: int | None = None,
    market_mode: str = "spot",
    available_wallet_value: float | None = None,
    stop_loss_pct: float | None = None,
    rank_number: int | None = None,
    rank_x: int | None = None,
    orderbook_capacity_quote: float | None = None,
) -> Dict[str, Any]:
    """Compute rank-only position size with policy and liquidity caps."""
    wallet = max(float(wallet_value), 0.0)
    open_notional = max(float(open_notional), 0.0)
    mode_raw = str(market_mode or "spot").strip().lower()
    if mode_raw in {"perp", "perps", "future", "futures", "swap"}:
        available_wallet = (
            max(float(available_wallet_value), 0.0)
            if available_wallet_value is not None
            and np.isfinite(float(available_wallet_value))
            else max(wallet - open_notional, 0.0)
        )
        rank = max(1, int(rank_number or 1))
        rx = max(1, int(rank_x or rank))
        if rx <= 1:
            rank_leverage = 5.0
        else:
            rank_leverage = 5.0 + 20.0 * max(float(rx - rank), 0.0) / float(rx - 1)
        sl_raw = (
            float(stop_loss_pct)
            if stop_loss_pct is not None and np.isfinite(float(stop_loss_pct))
            else np.nan
        )
        sl_pct = sl_raw * 100.0 if np.isfinite(sl_raw) and sl_raw <= 1.0 else sl_raw
        risk_cap = 100.0 / (1.5 * sl_pct) if np.isfinite(sl_pct) and sl_pct > 0 else float("inf")
        leverage = min(rank_leverage, risk_cap)
        leverage = max(float(leverage), 0.0) if np.isfinite(leverage) else rank_leverage
        leverage_power = leverage**1.5
        full_wallet_size = wallet * leverage_power / 100.0
        available_wallet_size = available_wallet * 2.5 * leverage_power / 100.0
        size_before_liquidity = max(0.0, min(full_wallet_size, available_wallet_size))
        book_cap = (
            float(orderbook_capacity_quote)
            if orderbook_capacity_quote is not None
            and np.isfinite(float(orderbook_capacity_quote))
            and float(orderbook_capacity_quote) > 0.0
            else float("inf")
        )
        size_after_liquidity = min(size_before_liquidity, book_cap)
        if live_test_mode and size_after_liquidity > 0.0:
            live_test_min_notional = float(policy.live_test_min_quote_notional)
            live_test_max_notional = float(policy.live_test_quote_notional)
            if size_after_liquidity < live_test_min_notional:
                size_after_liquidity = 0.0
            else:
                size_after_liquidity = min(size_after_liquidity, live_test_max_notional)
        else:
            live_test_min_notional = None
        liq_weight = float(np.clip(liquidity_capacity_weight, 0.0, 1.0))
        return {
            "market_mode": "perps",
            "wallet_value": wallet,
            "book_notional_multiplier": 1.0,
            "leverage_wallet_multiplier": leverage,
            "min_margin_level_after_entry": float(policy.min_margin_level_after_entry),
            "total_assets_quote": total_assets_quote,
            "total_liabilities_quote": total_liabilities_quote,
            "current_margin_level": None,
            "max_total_equity_allocation": wallet,
            "configured_book_notional": wallet * leverage,
            "margin_surplus_notional": None,
            "safe_book_notional": wallet * leverage,
            "max_total_notional": wallet * leverage,
            "open_notional": open_notional,
            "open_equity_allocation": open_notional,
            "remaining_total_notional": max(wallet * leverage - open_notional, 0.0),
            "open_position_count": int(open_positions or 0),
            "reserved_position_slots": int(policy.reserved_position_slots or policy.max_concurrent_positions),
            "remaining_position_slots": None,
            "available_wallet": available_wallet,
            "available_wallet_position_cap": available_wallet_size,
            "rank_scaled_cap": full_wallet_size,
            "rank_base_notional": full_wallet_size,
            "target_slot_notional": full_wallet_size,
            "reserved_slot_notional": full_wallet_size,
            "remaining_slot_notional": available_wallet_size,
            "slot_cap_notional": size_before_liquidity,
            "rank_excess": None,
            "curved_rank_excess": None,
            "rank_size_power": rank_size_power,
            "rank_multiplier": leverage,
            "rank_slot_fraction": None,
            "provisional_size": size_before_liquidity,
            "position_cap": size_before_liquidity,
            "size_before_liquidity": size_before_liquidity,
            "liquidity_capacity_weight": liq_weight,
            "configured_live_test_min_notional": (
                policy.live_test_min_quote_notional if live_test_mode else None
            ),
            "effective_live_test_min_notional": live_test_min_notional,
            "size_after_liquidity": max(float(size_after_liquidity), 0.0),
            "live_test_mode": bool(live_test_mode),
            "perp_rank_number": rank,
            "perp_rank_x": rx,
            "perp_rank_leverage": rank_leverage,
            "perp_risk_cap_leverage": risk_cap if np.isfinite(risk_cap) else None,
            "perp_effective_leverage": leverage,
            "perp_stop_loss_pct": sl_pct if np.isfinite(sl_pct) else None,
            "perp_full_wallet": wallet,
            "perp_available_wallet": available_wallet,
            "orderbook_capacity_quote_within_slippage": (
                book_cap if np.isfinite(book_cap) else None
            ),
        }
    book_multiplier = max(float(policy.book_notional_multiplier), 0.0)
    legacy_leverage_multiplier = max(float(policy.leverage_wallet_multiplier), 1.0)
    max_total_equity_allocation = policy.max_total_wallet_allocation_pct * wallet
    configured_book_notional = max_total_equity_allocation * book_multiplier
    assets_q = (
        float(total_assets_quote)
        if total_assets_quote is not None and np.isfinite(float(total_assets_quote))
        else np.nan
    )
    liabilities_q = (
        float(total_liabilities_quote)
        if total_liabilities_quote is not None
        and np.isfinite(float(total_liabilities_quote))
        else np.nan
    )
    margin_surplus_notional = float("inf")
    margin_level = float("inf")
    if np.isfinite(assets_q) and np.isfinite(liabilities_q) and liabilities_q > 0.0:
        margin_level = assets_q / max(liabilities_q, 1e-12)
        margin_surplus_notional = max(
            assets_q - float(policy.min_margin_level_after_entry) * liabilities_q,
            0.0,
        )
    elif np.isfinite(assets_q) and liabilities_q == 0.0:
        margin_level = float("inf")
        margin_surplus_notional = assets_q
    safe_book_notional = min(configured_book_notional, margin_surplus_notional)
    if not np.isfinite(safe_book_notional):
        safe_book_notional = configured_book_notional
    safe_book_notional = max(safe_book_notional, 0.0)
    open_equity_allocation = open_notional / max(book_multiplier, 1e-9)
    available_wallet = max(wallet - open_equity_allocation, 0.0)
    available_wallet_position_cap = policy.max_available_wallet_position_pct * (
        available_wallet * book_multiplier
    )
    denom = max(1.0 - float(final_threshold), 1e-9)
    rank_excess = (float(adjusted_rank_score) - float(final_threshold)) / denom
    rank_excess = float(np.clip(rank_excess, 0.0, 1.0))
    size_power = (
        float(policy.rank_size_power)
        if rank_size_power is None
        else float(rank_size_power)
    )
    size_power = max(size_power, 1.000001)
    curved_rank_excess = float(rank_excess**size_power)
    rank_multiplier = float(
        policy.rank_multiplier_min
        + (policy.rank_multiplier_max - policy.rank_multiplier_min) * curved_rank_excess
    )
    position_cap = min(
        policy.max_position_wallet_pct * wallet * book_multiplier,
        float(policy.max_position_quote_notional) * book_multiplier,
    )
    reserved_slots = int(
        policy.reserved_position_slots or policy.max_concurrent_positions
    )
    reserved_slots = max(1, reserved_slots)
    target_slot_notional = safe_book_notional / float(reserved_slots)
    open_position_count = (
        int(open_positions)
        if open_positions is not None and np.isfinite(float(open_positions))
        else int(np.floor(open_notional / max(target_slot_notional, 1e-9)))
    )
    open_position_count = max(0, min(open_position_count, policy.max_concurrent_positions))
    remaining_position_slots = max(reserved_slots - open_position_count, 1)
    reserved_slot_notional = safe_book_notional / float(reserved_slots)
    remaining_slot_notional = (
        max(safe_book_notional - open_notional, 0.0)
        / float(remaining_position_slots)
    )
    slot_cap_notional = max(0.0, min(reserved_slot_notional, remaining_slot_notional))
    rank_slot_fraction = rank_multiplier / max(float(policy.rank_multiplier_max), 1e-9)
    rank_slot_fraction = float(np.clip(rank_slot_fraction, 0.0, 1.0))
    provisional_size = slot_cap_notional * rank_slot_fraction
    remaining_total = max(safe_book_notional - open_notional, 0.0)
    size_before_liquidity = min(
        provisional_size,
        available_wallet_position_cap,
        position_cap,
        slot_cap_notional,
        remaining_total,
    )
    liq_weight = float(np.clip(liquidity_capacity_weight, 0.0, 1.0))
    size_after_liquidity = size_before_liquidity * liq_weight
    if live_test_mode and size_after_liquidity > 0.0:
        configured_live_test_min_notional = (
            policy.live_test_min_quote_notional * book_multiplier
        )
        live_test_min_notional = (
            configured_live_test_min_notional
            if slot_cap_notional >= configured_live_test_min_notional
            else 0.0
        )
        live_test_max_notional = policy.live_test_quote_notional * book_multiplier
        if size_after_liquidity < live_test_min_notional:
            size_after_liquidity = 0.0
        else:
            size_after_liquidity = min(size_after_liquidity, live_test_max_notional)
    return {
        "wallet_value": wallet,
        "book_notional_multiplier": book_multiplier,
        "leverage_wallet_multiplier": legacy_leverage_multiplier,
        "min_margin_level_after_entry": float(policy.min_margin_level_after_entry),
        "total_assets_quote": assets_q if np.isfinite(assets_q) else None,
        "total_liabilities_quote": (
            liabilities_q if np.isfinite(liabilities_q) else None
        ),
        "current_margin_level": margin_level if np.isfinite(margin_level) else None,
        "max_total_equity_allocation": max_total_equity_allocation,
        "configured_book_notional": configured_book_notional,
        "margin_surplus_notional": (
            margin_surplus_notional if np.isfinite(margin_surplus_notional) else None
        ),
        "safe_book_notional": safe_book_notional,
        "max_total_notional": safe_book_notional,
        "open_notional": open_notional,
        "open_equity_allocation": open_equity_allocation,
        "remaining_total_notional": remaining_total,
        "open_position_count": open_position_count,
        "reserved_position_slots": reserved_slots,
        "remaining_position_slots": remaining_position_slots,
        "available_wallet": available_wallet,
        "available_wallet_position_cap": available_wallet_position_cap,
        "rank_scaled_cap": position_cap,
        "rank_base_notional": target_slot_notional,
        "target_slot_notional": target_slot_notional,
        "reserved_slot_notional": reserved_slot_notional,
        "remaining_slot_notional": remaining_slot_notional,
        "slot_cap_notional": slot_cap_notional,
        "rank_excess": rank_excess,
        "curved_rank_excess": curved_rank_excess,
        "rank_size_power": size_power,
        "rank_multiplier": rank_multiplier,
        "rank_slot_fraction": rank_slot_fraction,
        "provisional_size": provisional_size,
        "position_cap": position_cap,
        "size_before_liquidity": size_before_liquidity,
        "liquidity_capacity_weight": liq_weight,
        "configured_live_test_min_notional": (
            policy.live_test_min_quote_notional * book_multiplier
            if live_test_mode
            else None
        ),
        "effective_live_test_min_notional": (
            live_test_min_notional if live_test_mode else None
        ),
        "size_after_liquidity": max(float(size_after_liquidity), 0.0),
        "live_test_mode": bool(live_test_mode),
    }
