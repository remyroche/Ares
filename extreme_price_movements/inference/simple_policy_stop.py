"""Simple-policy stop-loss replacement decisions.

This module is the single place where live/shadow stop-loss replacement
candidates are computed. Exchange code may place or reject a decision, but it
must not invent a replacement stop price outside this module.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
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

SIMPLE_POLICY_STOP_PARAM_KEYS = (
    "generated_by",
    "schema",
    "params_source",
    "params_hash",
    "strategy_id",
    "sl_mult",
    "barrier_pct",
    "barrier_frac",
    "enable_trailing",
    "trailing_activation_mult",
    "trailing_power",
    "trailing_squash_divisor",
    "giveback_beta",
    "capital_protect_mfe_mult",
    "capital_protect_regression_frac",
)

_OPTIONAL_SIMPLE_POLICY_STOP_FIELDS = {"barrier_pct", "barrier_frac", "enable_trailing"}
_ARTIFACT_AUDIT_FIELDS = {
    "_loaded_from_simple_policy_artifact",
    "_artifact_path",
    "_artifact_mtime_ns",
}
_ALLOWED_SIMPLE_POLICY_STOP_FIELDS = set(SIMPLE_POLICY_STOP_PARAM_KEYS) | _ARTIFACT_AUDIT_FIELDS
_GENERIC_PARAMS_SOURCES = {
    "",
    SIMPLE_POLICY_GENERATOR,
    "unversioned_simple_policy_params",
    "simple_policy_optimisation",
    "simple_policy_optimisation:",
}


def _artifact_file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _normalise_schema(row: Mapping[str, Any], metadata: Mapping[str, Any]) -> str:
    return str(
        row.get("schema")
        or row.get("schema_version")
        or metadata.get("schema")
        or metadata.get("schema_version")
        or ""
    )


def is_concrete_simple_policy_params_source(params_source: Any) -> bool:
    source = str(params_source or "").strip()
    if source in _GENERIC_PARAMS_SOURCES:
        return False
    if source.startswith("artifacts/") or "/artifacts/" in source:
        return True
    if source.endswith(".json") and ("/" in source or "\\" in source):
        return True
    return False


def _is_simple_policy_artifact_metadata(metadata: Mapping[str, Any]) -> bool:
    schema = str(metadata.get("schema") or metadata.get("schema_version") or "")
    return (
        str(metadata.get("generated_by") or "").strip() == SIMPLE_POLICY_GENERATOR
        and schema == SIMPLE_POLICY_SCHEMA
    )


def _normalise_simple_policy_stop_param_row(
    row: Any,
    *,
    strategy_id: str,
    artifact_metadata: Mapping[str, Any],
    params_source: str,
    params_hash: str,
) -> Dict[str, Any]:
    """Return canonical simple_policy_optimiser stop fields for one exact strategy."""
    if not isinstance(row, Mapping):
        return {}
    schema = _normalise_schema(row, artifact_metadata)
    generated_by = str(row.get("generated_by") or artifact_metadata.get("generated_by") or "").strip()
    row_strategy = str(row.get("strategy_id") or row.get("strategy_for_inference") or "").strip()
    if row_strategy != str(strategy_id):
        return {}
    out = {k: row[k] for k in SIMPLE_POLICY_STOP_PARAM_KEYS if k in row}
    for private_key in _ARTIFACT_AUDIT_FIELDS:
        if private_key in row:
            out[private_key] = row[private_key]
    out["generated_by"] = generated_by
    out["schema"] = schema
    out["strategy_id"] = row_strategy
    out["params_source"] = params_source
    out["params_hash"] = params_hash
    out.pop("schema_version", None)
    out.pop("strategy_for_inference", None)
    return dict(out)


def extract_simple_policy_stop_params_by_strategy(
    bucket_params: Optional[Mapping[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Extract explicit canonical stop params keyed by exact strategy_id only."""
    if not isinstance(bucket_params, Mapping):
        return {}
    explicit = bucket_params.get("simple_policy_stop_params_by_strategy")
    if not isinstance(explicit, Mapping):
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for key, row in explicit.items():
        if not isinstance(row, Mapping):
            continue
        strategy_id = str(row.get("strategy_id") or "").strip()
        if not strategy_id or strategy_id != str(key):
            continue
        unknown = sorted(
            set(row) - _ALLOWED_SIMPLE_POLICY_STOP_FIELDS - {"schema_version", "strategy_for_inference"}
        )
        if unknown:
            raise SimplePolicyStopParamsError(
                "unknown simple-policy stop fields: " + ",".join(unknown)
            )
        if row.get("_loaded_from_simple_policy_artifact") is not True:
            raise SimplePolicyStopParamsError(
                "simple-policy stop params must be loaded from a simple_policy_optimiser artifact"
            )
        params_source = str(row.get("params_source") or "").strip()
        params_hash = str(row.get("params_hash") or "").strip()
        normalised = _normalise_simple_policy_stop_param_row(
            row,
            strategy_id=strategy_id,
            artifact_metadata=row,
            params_source=params_source,
            params_hash=params_hash,
        )
        if normalised:
            validate_simple_policy_stop_params(normalised, state={"strategy_id": strategy_id})
            out[strategy_id] = normalised
    return out


def _iter_simple_policy_artifact_paths(data_root: str, run_id: Optional[str] = None):
    artifacts_root = Path(data_root) / "artifacts"
    if not artifacts_root.exists():
        return
    # Strict runtime loading accepts only the canonical optimiser namespace.
    run_dirs = sorted(artifacts_root.iterdir())
    for run_dir in run_dirs:
        if not run_dir.is_dir():
            continue
        opt_root = run_dir / "simple_policy_optimiser"
        if opt_root.is_dir():
            yield from opt_root.glob("*/best_policy_params.json")


def _artifact_params_source(path: Path, data_root: str) -> str:
    try:
        return path.relative_to(Path(data_root)).as_posix()
    except ValueError:
        return path.as_posix()


def _artifact_strategy_rows(payload: Any):
    if not isinstance(payload, Mapping):
        return []
    strategy_for_inference = payload.get("strategy_for_inference")
    if isinstance(strategy_for_inference, Mapping):
        rows = strategy_for_inference.get("strategies", [])
    elif isinstance(strategy_for_inference, list):
        rows = strategy_for_inference
    else:
        rows = payload.get("strategies", [])
    return rows if isinstance(rows, list) else []


def load_simple_policy_stop_params_by_strategy(
    data_root: str,
    run_id: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """Load latest valid simple_policy_optimiser stop params per exact strategy_id.

    The returned payload is intentionally narrow and suitable for
    ``bucket_params["simple_policy_stop_params_by_strategy"]``.
    """
    candidates: Dict[str, tuple] = {}
    for path in _iter_simple_policy_artifact_paths(data_root, run_id):
        if not path.exists() or not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text())
        except Exception:
            continue
        if not isinstance(payload, Mapping):
            continue
        payload_meta = {
            "generated_by": payload.get("generated_by"),
            "schema": payload.get("schema") or payload.get("schema_version"),
        }
        if not _is_simple_policy_artifact_metadata(payload_meta):
            continue
        params_source = _artifact_params_source(path, data_root)
        params_hash = _artifact_file_hash(path)
        for row in _artifact_strategy_rows(payload):
            if not isinstance(row, Mapping) or row.get("selected") is False:
                continue
            strategy_id = str(row.get("strategy_id") or row.get("strategy_for_inference") or "").strip()
            if not strategy_id:
                continue
            normalised = _normalise_simple_policy_stop_param_row(
                row,
                strategy_id=strategy_id,
                artifact_metadata=payload_meta,
                params_source=params_source,
                params_hash=params_hash,
            )
            normalised["_loaded_from_simple_policy_artifact"] = True
            normalised["_artifact_path"] = path.as_posix()
            normalised["_artifact_mtime_ns"] = int(path.stat().st_mtime_ns)
            if not normalised:
                continue
            try:
                validate_simple_policy_stop_params(
                    normalised, state={"strategy_id": strategy_id}
                )
            except SimplePolicyStopParamsError:
                continue
            rank_ts = int(
                payload.get("created_at_ns")
                or payload.get("artifact_mtime_ns")
                or path.stat().st_mtime_ns
            )
            rank = (rank_ts, path.as_posix())
            prev = candidates.get(strategy_id)
            if prev is None or rank > prev[0]:
                candidates[strategy_id] = (rank, normalised)
    return {strategy_id: params for strategy_id, (_, params) in candidates.items()}


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
    trailing_activation_mult: Optional[float] = None
    trailing_power: Optional[float] = None
    trailing_squash_divisor: Optional[float] = None
    giveback_beta: Optional[float] = None
    capital_protect_mfe_mult: Optional[float] = None
    capital_protect_regression_frac: Optional[float] = None
    decision_module: str = "simple_policy_stop.py"
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
    require_metadata: bool = True,
) -> ValidatedSimplePolicyParams:
    """Validate and normalise simple-policy stop runtime params.

    The validator fails closed unless all required numeric fields and optimiser
    provenance fields are present, finite, and identify a simple_policy_optimiser
    artifact.
    """
    params = dict(params or {})
    state = dict(state or {})

    unknown = sorted(set(params) - _ALLOWED_SIMPLE_POLICY_STOP_FIELDS)
    if unknown:
        raise SimplePolicyStopParamsError(
            "unknown simple-policy stop fields: " + ",".join(unknown)
        )

    missing = [
        key for key in REQUIRED_SIMPLE_POLICY_STOP_FIELDS if key not in params
    ]
    if missing:
        raise SimplePolicyStopParamsError(
            "missing simple-policy stop fields: " + ",".join(missing)
        )

    activation = _first_positive(params, "trailing_activation_mult")
    if not np.isfinite(activation):
        raise SimplePolicyStopParamsError(
            "missing simple-policy trailing activation field"
        )

    barrier_frac = _first_positive(params, "barrier_frac", "barrier_pct")
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
    if generated_by != SIMPLE_POLICY_GENERATOR:
        raise SimplePolicyStopParamsError(
            "stop decision requires generated_by=simple_policy_optimiser"
        )
    if schema != SIMPLE_POLICY_SCHEMA:
        raise SimplePolicyStopParamsError(
            "stop decision requires schema=simple_policy_v1"
        )

    strategy_id = str(params.get("strategy_id") or "").strip()
    if not strategy_id:
        raise SimplePolicyStopParamsError("missing simple-policy strategy_id")
    expected_strategy = str(state.get("strategy_id") or state.get("bucket_key") or "").strip()
    if expected_strategy and strategy_id != expected_strategy:
        raise SimplePolicyStopParamsError(
            f"simple-policy strategy_id mismatch: params={strategy_id} state={expected_strategy}"
        )

    params_source = str(params.get("params_source") or "").strip()
    if not is_concrete_simple_policy_params_source(params_source):
        raise SimplePolicyStopParamsError(
            "stop decision requires concrete simple_policy_optimiser artifact params_source"
        )

    params_hash = str(params.get("params_hash") or "").strip()
    if not params_hash:
        raise SimplePolicyStopParamsError("missing simple-policy params_hash")
    if params.get("_loaded_from_simple_policy_artifact") is not True:
        raise SimplePolicyStopParamsError(
            "simple-policy stop params were not loaded from a simple_policy_optimiser artifact"
        )
    artifact_path = Path(str(params.get("_artifact_path") or "").strip())
    if not str(artifact_path):
        raise SimplePolicyStopParamsError("missing simple-policy artifact_path")
    if not artifact_path.is_file():
        raise SimplePolicyStopParamsError("simple-policy artifact_path does not exist")
    artifact_path_text = artifact_path.as_posix()
    if not artifact_path_text.endswith(params_source):
        raise SimplePolicyStopParamsError(
            "simple-policy params_source does not match artifact_path"
        )
    if _artifact_file_hash(artifact_path) != params_hash:
        raise SimplePolicyStopParamsError(
            "simple-policy params_hash does not match artifact content"
        )

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


def compute_initial_simple_policy_stop_decision(
    *,
    entry_price: float,
    policy_params: Mapping[str, Any],
    side: str,
    strategy_id: Optional[str] = None,
    require_metadata: bool = True,
) -> SimplePolicyStopDecision:
    """Compute the canonical initial simple-policy STOP_LOSS order decision."""
    entry = _safe_float(entry_price, default=np.nan)
    if not np.isfinite(entry) or entry <= 0.0:
        raise SimplePolicyStopParamsError("missing finite entry_price")
    state = {"strategy_id": strategy_id or ""}
    validated = validate_simple_policy_stop_params(
        policy_params,
        state=state,
        require_metadata=require_metadata,
    )
    side_l = str(side or "long").lower()
    if side_l == "long":
        stop_price = entry * (1.0 - validated.sl_mult * validated.barrier_frac)
    elif side_l == "short":
        stop_price = entry * (1.0 + validated.sl_mult * validated.barrier_frac)
    else:
        raise SimplePolicyStopParamsError(f"unsupported side for simple-policy stop: {side}")
    if not np.isfinite(stop_price) or stop_price <= 0.0:
        raise SimplePolicyStopParamsError("initial simple-policy stop_price must be positive")
    if side_l == "long" and stop_price >= entry:
        raise SimplePolicyStopParamsError("long stop must be below entry_price")
    if side_l == "short" and stop_price <= entry:
        raise SimplePolicyStopParamsError("short stop must be above entry_price")
    detail = (
        "original_stop_loss: "
        f"sl_mult={validated.sl_mult:.6g} "
        f"barrier_frac={validated.barrier_frac:.6g} "
        f"params_source={validated.params_source} "
        f"params_hash={validated.params_hash}"
    )
    return SimplePolicyStopDecision(
        should_replace=True,
        stop_price=float(stop_price),
        requested_policy_stop=float(stop_price),
        reason="original_stop_loss",
        reason_detail=detail,
        strategy_id=validated.strategy_id,
        params_source=validated.params_source,
        params_hash=validated.params_hash,
        barrier_frac=validated.barrier_frac,
        sl_mult=validated.sl_mult,
        trailing_activation_mult=validated.trailing_activation_mult,
        trailing_power=validated.trailing_power,
        trailing_squash_divisor=validated.trailing_squash_divisor,
        giveback_beta=validated.giveback_beta,
        capital_protect_mfe_mult=validated.capital_protect_mfe_mult,
        capital_protect_regression_frac=validated.capital_protect_regression_frac,
        peak_price=float(entry),
        mfe=0.0,
        mae=0.0,
        params_schema=validated.schema,
    )


def compute_simple_policy_initial_stop_decision(*args: Any, **kwargs: Any) -> SimplePolicyStopDecision:
    """Backward-compatible alias for the canonical initial stop decision."""
    return compute_initial_simple_policy_stop_decision(*args, **kwargs)


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
    require_metadata: bool = True,
) -> SimplePolicyStopDecision:
    """Compute the canonical simple-policy stop replacement decision."""
    validated = validate_simple_policy_stop_params(
        policy_params,
        state=state,
        require_metadata=require_metadata,
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
        for _, row in bars.iterrows():
            high = _safe_float(row.get("high"), default=np.nan)
            low = _safe_float(row.get("low"), default=np.nan)
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
        trailing_activation_mult=validated.trailing_activation_mult,
        trailing_power=validated.trailing_power,
        trailing_squash_divisor=validated.trailing_squash_divisor,
        giveback_beta=validated.giveback_beta,
        capital_protect_mfe_mult=validated.capital_protect_mfe_mult,
        capital_protect_regression_frac=validated.capital_protect_regression_frac,
        peak_price=float(peak_price) if np.isfinite(peak_price) else None,
        mfe=float(mfe),
        mae=float(mae),
        last_eval_ts=last_eval_ts,
        params_schema=validated.schema,
    )
