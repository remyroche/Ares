"""Typed live portfolio policy contract for inference and offline replay."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


@dataclass(frozen=True)
class PortfolioPolicyConfig:
    """Shared source of truth for live portfolio policy defaults."""

    schema_version: str = "portfolio_policy_v1"

    max_concurrent_positions: int = 8
    max_concurrent_per_side: Optional[int] = None
    max_concurrent_per_strategy: Optional[int] = None

    max_total_wallet_allocation_pct: float = 0.75
    max_available_wallet_position_pct: float = 0.50
    max_position_wallet_pct: float = 0.15
    max_position_quote_notional: float = 5000.0

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
    hard_max_spread_bps: float = 75.0
    min_liquidity_capacity_weight: float = 0.25
    max_ticker_age_seconds: float = 30.0

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
            "rank_multiplier_min",
            "rank_multiplier_max",
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
    if values.get("max_concurrent_per_side") is not None:
        values["max_concurrent_per_side"] = max(
            1, int(values["max_concurrent_per_side"])
        )
    if values.get("max_concurrent_per_strategy") is not None:
        values["max_concurrent_per_strategy"] = max(
            1, int(values["max_concurrent_per_strategy"])
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
) -> Dict[str, Any]:
    """Compute rank-only position size with policy and liquidity caps."""
    wallet = max(float(wallet_value), 0.0)
    open_notional = max(float(open_notional), 0.0)
    max_total_notional = policy.max_total_wallet_allocation_pct * wallet
    available_wallet = max(wallet - open_notional, 0.0)
    available_wallet_position_cap = (
        policy.max_available_wallet_position_pct * available_wallet
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
        policy.max_position_wallet_pct * wallet,
        float(policy.max_position_quote_notional),
    )
    rank_scaled_cap = min(available_wallet_position_cap, position_cap)
    rank_base_notional = rank_scaled_cap / max(float(policy.rank_multiplier_max), 1e-9)
    provisional_size = rank_base_notional * rank_multiplier
    remaining_total = max(max_total_notional - open_notional, 0.0)
    size_before_liquidity = min(
        provisional_size,
        rank_scaled_cap,
        available_wallet_position_cap,
        position_cap,
    )
    liq_weight = float(np.clip(liquidity_capacity_weight, 0.0, 1.0))
    size_after_liquidity = size_before_liquidity * liq_weight
    if live_test_mode and size_after_liquidity > 0.0:
        size_after_liquidity = min(
            max(size_after_liquidity, policy.live_test_min_quote_notional),
            policy.live_test_quote_notional,
        )
    return {
        "wallet_value": wallet,
        "max_total_notional": max_total_notional,
        "open_notional": open_notional,
        "remaining_total_notional": remaining_total,
        "available_wallet": available_wallet,
        "available_wallet_position_cap": available_wallet_position_cap,
        "rank_scaled_cap": rank_scaled_cap,
        "rank_base_notional": rank_base_notional,
        "target_slot_notional": rank_base_notional,
        "rank_excess": rank_excess,
        "curved_rank_excess": curved_rank_excess,
        "rank_size_power": size_power,
        "rank_multiplier": rank_multiplier,
        "provisional_size": provisional_size,
        "position_cap": position_cap,
        "size_before_liquidity": size_before_liquidity,
        "liquidity_capacity_weight": liq_weight,
        "size_after_liquidity": max(float(size_after_liquidity), 0.0),
        "live_test_mode": bool(live_test_mode),
    }
