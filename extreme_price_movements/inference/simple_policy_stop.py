"""Simple-policy stop-loss replacement decisions.

This module is the single place where live/shadow stop-loss replacement
candidates are computed. Exchange code may place or reject a decision, but it
must not invent a replacement stop price outside this module.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Optional

import numpy as np
import pandas as pd

SIMPLE_POLICY_SCHEMA = "simple_policy_v1"
SIMPLE_POLICY_GENERATOR = "simple_policy_optimiser"

REQUIRED_SIMPLE_POLICY_STOP_FIELDS = (
    "sl_mult",
    "trailing_activation_mult",
    "trailing_power",
    "trailing_squash_divisor",
    "giveback_beta",
    "capital_protect_mfe_mult",
    "capital_protect_regression_frac",
)

LEGACY_STOP_REPLACEMENT_FIELDS = (
    "trail_mult",
    "giveback_pct",
    "profit_lock_amount",
    "fixed_stop_loss_pct",
    "stop_loss_pct",
    "stop_loss_frac",
    "mfe_early_exit_threshold",
)

SIMPLE_POLICY_STOP_PARAM_KEYS = (
    "generated_by",
    "schema",
    "schema_version",
    "params_source",
    "params_hash",
    "strategy_id",
    "strategy_for_inference",
    "sl_mult",
    "barrier_pct",
    "barrier_frac",
    "enable_trailing",
    "trailing_activation_mult",
    "trailing_override_alpha",
    "trailing_power",
    "trailing_squash_divisor",
    "giveback_beta",
    "capital_protect_mfe_mult",
    "capital_protect_regression_frac",
    *LEGACY_STOP_REPLACEMENT_FIELDS,
)


def _strategy_core_id_local(strategy_id: Any) -> str:
    sid = str(strategy_id or "").strip()
    lower = sid.lower()
    for prefix in ("long_", "short_"):
        if lower.startswith(prefix):
            return sid[len(prefix):]
    return sid


def _normalise_simple_policy_stop_param_row(
    row: Any,
    *,
    strategy_id: Optional[str] = None,
    artifact_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Return only simple_policy_optimiser stop fields from an artifact row."""
    if not isinstance(row, dict):
        return {}
    metadata = artifact_metadata or {}
    out = {k: row[k] for k in SIMPLE_POLICY_STOP_PARAM_KEYS if k in row}
    if "schema" not in out and "schema_version" in out:
        out["schema"] = out["schema_version"]
    for meta_key in ("generated_by", "schema", "params_source", "params_hash"):
        if meta_key not in out and metadata.get(meta_key) is not None:
            out[meta_key] = metadata.get(meta_key)
    if "schema" not in out and metadata.get("schema_version") is not None:
        out["schema"] = metadata.get("schema_version")
    if "trailing_activation_mult" not in out and "trailing_override_alpha" in out:
        out["trailing_activation_mult"] = out["trailing_override_alpha"]
    if strategy_id and "strategy_id" not in out:
        out["strategy_id"] = strategy_id
    return dict(out)


def extract_simple_policy_stop_params_by_strategy(
    bucket_params: Optional[Mapping[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Extract dedicated stop-param rows without ridge/default/runtime fields."""
    if not isinstance(bucket_params, Mapping):
        return {}
    explicit = (
        bucket_params.get("simple_policy_stop_params_by_strategy")
        or bucket_params.get("simple_policy_params_by_strategy")
        or bucket_params.get("simple_policy_stop_params")
    )
    artifact_metadata = {
        key: bucket_params.get(key)
        for key in ("generated_by", "schema", "schema_version", "params_source", "params_hash")
        if bucket_params.get(key) is not None
    }
    out: Dict[str, Dict[str, Any]] = {}

    def add_row(key: Any, row: Any) -> None:
        raw_key = str(key or "").strip()
        if not raw_key or not isinstance(row, dict):
            return
        strategy = str(row.get("strategy_id") or row.get("strategy_for_inference") or raw_key)
        filtered = _normalise_simple_policy_stop_param_row(
            row, strategy_id=strategy, artifact_metadata=artifact_metadata
        )
        if filtered:
            for alias in {
                raw_key,
                strategy,
                _strategy_core_id_local(raw_key),
                _strategy_core_id_local(strategy),
                raw_key.lower(),
                raw_key.upper(),
                strategy.lower(),
                strategy.upper(),
            }:
                if alias:
                    out[str(alias)] = dict(filtered)

    if isinstance(explicit, Mapping):
        for key, row in explicit.items():
            add_row(key, row)

    containers = []
    for container_key in ("selected", "strategies", "buckets"):
        value = bucket_params.get(container_key)
        if isinstance(value, Mapping):
            containers.extend(value.items())
        elif isinstance(value, list):
            containers.extend((item.get("strategy_id"), item) for item in value if isinstance(item, dict))
    for key, row in bucket_params.items():
        if isinstance(row, dict) and key not in {"buckets", "selected", "strategies"}:
            containers.append((key, row))
    for key, row in containers:
        add_row(key, row)
    return out


@dataclass(frozen=True)
class ValidatedSimplePolicyParams:
    sl_mult: float
    barrier_frac: float
    trailing_activation_mult: float
    trailing_power: float
    trailing_squash_divisor: float
    giveback_beta: float
    capital_protect_mfe_mult: float
    capital_protect_regression_frac: float
    enable_trailing: bool
    strategy_id: str
    params_source: str
    params_hash: str
    schema: str = SIMPLE_POLICY_SCHEMA

    def to_policy_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SimplePolicyStopDecision:
    should_replace: bool
    stop_price: Optional[float]
    reason: str
    reason_detail: str
    strategy_id: str
    params_source: str
    params_hash: str
    barrier_frac: float
    sl_mult: float
    requested_policy_stop: Optional[float] = None
    peak_price: Optional[float] = None
    mfe: Optional[float] = None
    mae: Optional[float] = None
    last_eval_ts: Optional[str] = None
    params_schema: str = SIMPLE_POLICY_SCHEMA

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SimplePolicyStopParamsError(ValueError):
    """Raised when simple-policy stop params are incomplete or unsafe."""


def _safe_float(value: Any, default: float = np.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def _params_hash(params: Mapping[str, Any]) -> str:
    excluded = {"params_hash", "stop_policy_params_hash"}
    payload = {str(k): v for k, v in params.items() if str(k) not in excluded}
    text = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _first_positive(params: Mapping[str, Any], *keys: str) -> float:
    for key in keys:
        value = _safe_float(params.get(key), default=np.nan)
        if np.isfinite(value) and value > 0.0:
            return float(value)
    return np.nan


def validate_simple_policy_stop_params(
    params: Mapping[str, Any],
    *,
    state: Optional[Mapping[str, Any]] = None,
    require_metadata: bool = False,
    reject_legacy_fields: bool = True,
    allow_state_barrier_fallback: bool = False,
) -> ValidatedSimplePolicyParams:
    """Validate and normalise simple-policy stop runtime params.

    The validator fails closed unless all required numeric fields are present and
    finite. Live/live-test callers should pass ``require_metadata=True`` unless
    explicitly allowing unversioned policy params via config.
    """
    params = dict(params or {})
    state = dict(state or {})

    legacy_present = sorted(k for k in LEGACY_STOP_REPLACEMENT_FIELDS if k in params)
    if reject_legacy_fields and legacy_present:
        raise SimplePolicyStopParamsError(
            "legacy stop replacement fields are not allowed: "
            + ",".join(legacy_present)
        )

    missing = [
        key
        for key in REQUIRED_SIMPLE_POLICY_STOP_FIELDS
        if key not in params
        and not (key == "trailing_activation_mult" and "trailing_override_alpha" in params)
    ]
    if missing:
        raise SimplePolicyStopParamsError(
            "missing simple-policy stop fields: " + ",".join(missing)
        )

    activation = _first_positive(
        params, "trailing_activation_mult", "trailing_override_alpha"
    )
    if not np.isfinite(activation):
        raise SimplePolicyStopParamsError(
            "missing simple-policy trailing activation field"
        )

    barrier_frac = _first_positive(params, "barrier_frac", "barrier_pct")
    if allow_state_barrier_fallback and not np.isfinite(barrier_frac):
        barrier_frac = _first_positive(state, "barrier_frac", "barrier_pct")
    if not np.isfinite(barrier_frac):
        raise SimplePolicyStopParamsError("missing simple-policy barrier_frac/barrier_pct")

    values = {
        key: _safe_float(params.get(key), default=np.nan)
        for key in REQUIRED_SIMPLE_POLICY_STOP_FIELDS
        if key != "trailing_activation_mult"
    }
    bad = [key for key, value in values.items() if not np.isfinite(value)]
    if bad:
        raise SimplePolicyStopParamsError(
            "non-finite simple-policy stop fields: " + ",".join(bad)
        )
    if values["sl_mult"] <= 0.0:
        raise SimplePolicyStopParamsError("simple-policy sl_mult must be positive")
    if values["trailing_squash_divisor"] <= 0.0:
        raise SimplePolicyStopParamsError(
            "simple-policy trailing_squash_divisor must be positive"
        )

    schema = str(
        params.get("schema")
        or params.get("policy_schema")
        or params.get("stop_policy_schema")
        or ""
    )
    generated_by = str(params.get("generated_by") or "").strip()
    if require_metadata and generated_by != SIMPLE_POLICY_GENERATOR:
        raise SimplePolicyStopParamsError(
            "live stop replacement requires generated_by=simple_policy_optimiser"
        )
    if require_metadata and schema != SIMPLE_POLICY_SCHEMA:
        raise SimplePolicyStopParamsError(
            "live stop replacement requires schema=simple_policy_v1"
        )

    strategy_id = str(
        params.get("strategy_id")
        or params.get("strategy_for_inference")
        or state.get("strategy_id")
        or state.get("bucket_key")
        or ""
    ).strip()
    if not strategy_id:
        raise SimplePolicyStopParamsError("missing simple-policy strategy_id")

    params_source = str(params.get("params_source") or "").strip()
    if not params_source:
        if require_metadata:
            raise SimplePolicyStopParamsError("missing simple-policy params_source")
        params_source = "unversioned_simple_policy_params"

    explicit_params_hash = str(
        params.get("params_hash") or params.get("stop_policy_params_hash") or ""
    ).strip()
    if require_metadata and not explicit_params_hash:
        raise SimplePolicyStopParamsError("missing simple-policy params_hash")
    params_hash = explicit_params_hash or _params_hash(params)

    return ValidatedSimplePolicyParams(
        sl_mult=float(values["sl_mult"]),
        barrier_frac=float(barrier_frac),
        trailing_activation_mult=float(activation),
        trailing_power=float(values["trailing_power"]),
        trailing_squash_divisor=float(values["trailing_squash_divisor"]),
        giveback_beta=float(values["giveback_beta"]),
        capital_protect_mfe_mult=float(values["capital_protect_mfe_mult"]),
        capital_protect_regression_frac=float(values["capital_protect_regression_frac"]),
        enable_trailing=bool(params.get("enable_trailing", True)),
        strategy_id=strategy_id,
        params_source=params_source,
        params_hash=params_hash,
        schema=schema or SIMPLE_POLICY_SCHEMA,
    )


def _latest_bars(latest_market_state: Any) -> pd.DataFrame:
    if latest_market_state is None:
        return pd.DataFrame()
    if isinstance(latest_market_state, pd.DataFrame):
        bars = latest_market_state.copy()
    elif isinstance(latest_market_state, pd.Series):
        bars = latest_market_state.to_frame().T
    elif isinstance(latest_market_state, Mapping):
        if all(k in latest_market_state for k in ("open", "high", "low", "close")):
            bars = pd.DataFrame([latest_market_state])
        elif "bars" in latest_market_state:
            bars = pd.DataFrame(latest_market_state["bars"])
        else:
            bars = pd.DataFrame()
    else:
        bars = pd.DataFrame(latest_market_state)
    if bars.empty:
        return bars
    return bars.sort_index()


def compute_simple_policy_stop_decision(
    *,
    state: Mapping[str, Any],
    latest_market_state: Any,
    policy_params: Mapping[str, Any],
    side: str,
    require_metadata: bool = False,
    reject_legacy_fields: bool = True,
) -> SimplePolicyStopDecision:
    """Compute the canonical simple-policy stop replacement decision."""
    validated = validate_simple_policy_stop_params(
        policy_params,
        state=state,
        require_metadata=require_metadata,
        reject_legacy_fields=reject_legacy_fields,
        allow_state_barrier_fallback=False,
    )

    side_l = str(side or state.get("side") or "long").lower()
    entry_price = _safe_float(state.get("entry_price"), default=np.nan)
    current_stop = _safe_float(state.get("stop_price"), default=np.nan)
    peak_price = _safe_float(state.get("peak_price"), default=entry_price)
    mfe = max(_safe_float(state.get("mfe"), default=0.0), 0.0)
    mae = max(_safe_float(state.get("mae"), default=0.0), 0.0)
    if not np.isfinite(entry_price) or entry_price <= 0.0:
        raise SimplePolicyStopParamsError("missing finite entry_price")
    if not np.isfinite(current_stop) or current_stop <= 0.0:
        raise SimplePolicyStopParamsError("missing finite current stop_price")

    bars = _latest_bars(latest_market_state)
    if not bars.empty:
        high_col = bars.get("high")
        low_col = bars.get("low")
        highs = high_col.values if high_col is not None else np.full(len(bars), np.nan)
        lows = low_col.values if low_col is not None else np.full(len(bars), np.nan)

        for high, low in zip(highs, lows):
            high = _safe_float(high, default=np.nan)
            low = _safe_float(low, default=np.nan)
            if side_l == "long":
                if np.isfinite(high):
                    peak_price = max(peak_price, high)
                    mfe = max(mfe, (high - entry_price) / max(entry_price, 1e-12))
                if np.isfinite(low):
                    mae = max(mae, (entry_price - low) / max(entry_price, 1e-12))
            else:
                if np.isfinite(low):
                    peak_price = min(peak_price, low)
                    mfe = max(mfe, (entry_price - low) / max(entry_price, 1e-12))
                if np.isfinite(high):
                    mae = max(mae, (high - entry_price) / max(entry_price, 1e-12))

    candidate = float(current_stop)
    reason = "original_stop_loss"
    detail = "unchanged_original_stop_loss"

    cap_mfe_mult = validated.capital_protect_mfe_mult
    cap_reg_frac = validated.capital_protect_regression_frac
    if cap_mfe_mult > 0.0:
        sl_dist_ret = validated.sl_mult * validated.barrier_frac
        x_dist = cap_mfe_mult * validated.barrier_frac
        lock_dist = x_dist - cap_reg_frac * (x_dist + sl_dist_ret)
        if float(mfe) >= x_dist:
            cap_stop = (
                entry_price * (1.0 + lock_dist)
                if side_l == "long"
                else entry_price * (1.0 - lock_dist)
            )
            improved = cap_stop > candidate if side_l == "long" else cap_stop < candidate
            if improved:
                candidate = float(cap_stop)
                reason = "capital_preservation"
                detail = (
                    f"capital_preservation: mfe={float(mfe):.6g} "
                    f"trigger={x_dist:.6g} lock_dist={lock_dist:.6g}"
                )

    activation = validated.trailing_activation_mult * validated.barrier_frac
    if validated.enable_trailing and float(mfe) > activation:
        max_favorable_abs = max(float(mfe), 0.0) * entry_price
        barrier_price_dist = max(entry_price * validated.barrier_frac, 1e-12)
        dynamic_giveback = (
            max_favorable_abs
            / (barrier_price_dist * validated.trailing_squash_divisor)
        ) ** validated.trailing_power
        dynamic_giveback = float(np.clip(dynamic_giveback, 0.0, 1.0))
        trail_amount = max_favorable_abs * validated.giveback_beta * (1.0 - dynamic_giveback)
        locked_profit_abs = max_favorable_abs - trail_amount
        trail_stop = (
            entry_price + locked_profit_abs
            if side_l == "long"
            else entry_price - locked_profit_abs
        )
        improved = trail_stop > candidate if side_l == "long" else trail_stop < candidate
        if improved:
            candidate = float(trail_stop)
            reason = "trailing_profit"
            detail = (
                f"trailing_profit: mfe={float(mfe):.6g} activation={activation:.6g} "
                f"giveback_beta={validated.giveback_beta:.6g} "
                f"dynamic_giveback={dynamic_giveback:.6g}"
            )

    should_replace = candidate > current_stop if side_l == "long" else candidate < current_stop
    last_eval_ts = None
    if not bars.empty:
        try:
            last_eval_ts = pd.Timestamp(bars.index[-1]).isoformat()
        except Exception:
            last_eval_ts = None
    return SimplePolicyStopDecision(
        should_replace=bool(should_replace),
        stop_price=float(candidate) if should_replace else None,
        requested_policy_stop=float(candidate) if should_replace else None,
        reason=reason,
        reason_detail=detail,
        strategy_id=validated.strategy_id,
        params_source=validated.params_source,
        params_hash=validated.params_hash,
        barrier_frac=validated.barrier_frac,
        sl_mult=validated.sl_mult,
        peak_price=float(peak_price) if np.isfinite(peak_price) else None,
        mfe=float(mfe),
        mae=float(mae),
        last_eval_ts=last_eval_ts,
        params_schema=validated.schema,
    )
