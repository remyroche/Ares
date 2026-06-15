"""Simple-policy stop-loss replacement decisions.

This module is the single place where live/shadow stop-loss replacement
candidates are computed. Exchange code may place or reject a decision, but it
must not invent a replacement stop price outside this module.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Optional

import numpy as np
import pandas as pd

from extreme_price_movements.path_utils import mode_file_candidates

SIMPLE_POLICY_SCHEMA = "simple_policy_v1"
SIMPLE_POLICY_GENERATOR = "simple_policy_optimiser"
ADVERSE_EXIT_MAX_SL_FRACTION = 0.75
MIN_TRAILING_GIVEBACK_FRAC = 0.003

REQUIRED_SIMPLE_POLICY_STOP_FIELDS = (
    "sl_mult",
    "trailing_activation_mult",
    "trailing_power",
    "trailing_squash_divisor",
    "giveback_beta",
    "atr_power",
    "atr_multiplier",
    "hard_tp_abs_pct",
    "capital_protect_mfe_mult",
    "capital_protect_regression_frac",
)
ADVERSE_SIMPLE_POLICY_STOP_FIELDS = (
    "adverse_exit_enabled",
    "adverse_exit_alpha",
    "adverse_exit_beta",
    "adverse_exit_delta",
    "adverse_exit_theta_quantile",
    "adverse_exit_theta",
    "adverse_exit_fast_bars",
    "adverse_exit_min_mae_atr",
    "adverse_exit_min_speed",
    "adverse_exit_max_mfe_atr",
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
    "median_barrier_frac",
    "policy_median_barrier_frac",
    "median_atr_frac",
    "policy_median_atr_frac",
    "enable_trailing",
    "trailing_activation_mult",
    "trailing_override_alpha",
    "trailing_power",
    "trailing_squash_divisor",
    "giveback_beta",
    "atr_power",
    "atr_multiplier",
    "hard_tp_abs_pct",
    "exit_pressure_enabled",
    "exit_pressure_alpha",
    "exit_pressure_beta",
    "exit_pressure_delta",
    "exit_pressure_kappa",
    "exit_pressure_psi",
    "exit_pressure_omega",
    "exit_pressure_min_multiplier",
    "redeploy_scale_bps",
    "target_holding_hours",
    "churn_penalty_bps",
    "capital_protect_mfe_mult",
    "capital_protect_regression_frac",
    *ADVERSE_SIMPLE_POLICY_STOP_FIELDS,
)

_OPTIONAL_SIMPLE_POLICY_STOP_FIELDS = {
    "barrier_pct",
    "barrier_frac",
    "median_barrier_frac",
    "policy_median_barrier_frac",
    "median_atr_frac",
    "policy_median_atr_frac",
    "enable_trailing",
    "trailing_override_alpha",
    "atr_power",
    "atr_multiplier",
    "hard_tp_abs_pct",
    "exit_pressure_enabled",
    "exit_pressure_alpha",
    "exit_pressure_beta",
    "exit_pressure_delta",
    "exit_pressure_kappa",
    "exit_pressure_psi",
    "exit_pressure_omega",
    "exit_pressure_min_multiplier",
    "redeploy_scale_bps",
    "target_holding_hours",
    "churn_penalty_bps",
}
_ARTIFACT_AUDIT_FIELDS = {
    "_loaded_from_simple_policy_artifact",
    "_artifact_path",
    "_artifact_mtime_ns",
    "provisional_trailing_stage_sl_mult",
    "sl_mult_source",
    "stage2_selection_method",
    "adverse_exit_disabled_reason",
}


def _policy_export_invalid_marker(run_dir: Path) -> Optional[Path]:
    for marker_path in mode_file_candidates(
        run_dir / "simple_policy_optimiser" / "policy_export_invalid.json"
    ):
        if marker_path.exists():
            return marker_path
    return None
_ALLOWED_SIMPLE_POLICY_STOP_FIELDS = (
    set(SIMPLE_POLICY_STOP_PARAM_KEYS) | _ARTIFACT_AUDIT_FIELDS
)
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
    generated_by = str(
        row.get("generated_by") or artifact_metadata.get("generated_by") or ""
    ).strip()
    row_strategy = str(
        row.get("strategy_id") or row.get("strategy_for_inference") or ""
    ).strip()
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
            set(row)
            - _ALLOWED_SIMPLE_POLICY_STOP_FIELDS
            - {"schema_version", "strategy_for_inference"}
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
            out[strategy_id] = normalised
    return out


def _iter_simple_policy_artifact_paths(data_root: str, run_id: Optional[str] = None):
    override = str(os.environ.get("EPM_INFERENCE_POLICY_ARTIFACT_ROOT", "") or "").strip()
    if override:
        run_dirs = [Path(override)]
        artifacts_root = None
    else:
        run_dirs = []
        artifacts_root = Path(data_root) / "artifacts"
    if artifacts_root is not None and artifacts_root.exists():
        if run_id:
            run_dirs.append(artifacts_root / str(run_id))
        else:
            run_dirs.extend(sorted(artifacts_root.iterdir()))
    for run_dir in run_dirs:
        if not run_dir.is_dir():
            continue
        invalid_marker = _policy_export_invalid_marker(run_dir)
        if invalid_marker is not None:
            raise SimplePolicyStopParamsError(
                f"Refusing to load simple policy params for {run_dir.name}; "
                f"strict optimiser export is marked invalid at {invalid_marker}"
            )
        # Prefer the canonical optimiser namespace. Historical deployment
        # copies are accepted only after strict payload provenance validation.
        opt_root = run_dir / "simple_policy_optimiser"
        if opt_root.is_dir():
            seen: set[Path] = set()
            for deployment_dir in opt_root.iterdir():
                if not deployment_dir.is_dir():
                    continue
                canonical = deployment_dir / "best_policy_params.json"
                for candidate in mode_file_candidates(canonical):
                    if candidate.is_file() and candidate not in seen:
                        seen.add(candidate)
                        yield candidate
                for candidate in sorted(deployment_dir.glob("best_policy_params*.json")):
                    if candidate.is_file() and candidate not in seen:
                        seen.add(candidate)
                        yield candidate
        for rel in (
            "policy_params/best_policy_params.json",
            "best_policy_params.json",
        ):
            path = run_dir / rel
            for candidate in mode_file_candidates(path):
                if candidate.is_file():
                    yield candidate


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
            strategy_id = str(
                row.get("strategy_id") or row.get("strategy_for_inference") or ""
            ).strip()
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
                    normalised,
                    state={"strategy_id": strategy_id},
                    require_barrier=False,
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
    atr_power: float
    atr_multiplier: float
    hard_tp_abs_pct: float
    exit_pressure_enabled: bool
    exit_pressure_alpha: float
    exit_pressure_beta: float
    exit_pressure_delta: float
    exit_pressure_kappa: float
    exit_pressure_psi: float
    exit_pressure_omega: float
    exit_pressure_min_multiplier: float
    redeploy_scale_bps: float
    target_holding_hours: float
    churn_penalty_bps: float
    capital_protect_mfe_mult: float
    capital_protect_regression_frac: float
    adverse_exit_enabled: bool
    adverse_exit_alpha: float
    adverse_exit_beta: float
    adverse_exit_delta: float
    adverse_exit_theta_quantile: float
    adverse_exit_theta: float
    adverse_exit_fast_bars: int
    adverse_exit_min_mae_atr: float
    adverse_exit_min_speed: float
    adverse_exit_max_mfe_atr: float
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
    atr_power: Optional[float] = None
    atr_multiplier: Optional[float] = None
    hard_tp_abs_pct: Optional[float] = None
    exit_pressure: Optional[float] = None
    tightening_multiplier: Optional[float] = None
    effective_sl_mult: Optional[float] = None
    effective_trailing_activation_mult: Optional[float] = None
    effective_hard_tp_abs_pct: Optional[float] = None
    target_holding_hours: Optional[float] = None
    churn_penalty_bps: Optional[float] = None
    capital_protect_mfe_mult: Optional[float] = None
    capital_protect_regression_frac: Optional[float] = None
    adverse_exit_enabled: bool = False
    adverse_exit_theta: Optional[float] = None
    adverse_exit_theta_quantile: Optional[float] = None
    adverse_exit_min_mae_atr: Optional[float] = None
    adverse_exit_min_speed: Optional[float] = None
    adverse_exit_fast_bars: Optional[int] = None
    adverse_exit_max_mfe_atr: Optional[float] = None
    should_exit: bool = False
    exit_reason: Optional[str] = None
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


def _barrier_from_params_or_state(
    params: Mapping[str, Any],
    state: Mapping[str, Any],
) -> float:
    """Return row-specific policy barrier, preferring live row state."""
    barrier_frac = _first_positive(state, "barrier_frac", "barrier_pct", "policy_barrier_frac")
    if np.isfinite(barrier_frac):
        return float(barrier_frac)
    return _first_positive(params, "barrier_frac", "barrier_pct")


def _effective_barrier_frac(
    barrier_frac: float,
    params: Mapping[str, Any],
    state: Mapping[str, Any],
) -> float:
    """Apply policy ATR-power scaling when live state provides a median anchor."""

    barrier = float(barrier_frac)
    if not np.isfinite(barrier) or barrier <= 0.0:
        return barrier
    power = _safe_float(params.get("atr_power"), 1.0)
    if not np.isfinite(power):
        power = 1.0
    power = float(np.clip(power, 0.5, 1.2))
    multiplier = _safe_float(params.get("atr_multiplier"), 1.0)
    if not np.isfinite(multiplier):
        multiplier = 1.0
    multiplier = float(np.clip(multiplier, 0.5, 2.0))
    median = _first_positive(
        state,
        "median_barrier_frac",
        "policy_median_barrier_frac",
        "median_atr_frac",
        "policy_median_atr_frac",
    )
    if not np.isfinite(median) or median <= 0.0:
        median = _first_positive(
            params,
            "median_barrier_frac",
            "policy_median_barrier_frac",
            "median_atr_frac",
            "policy_median_atr_frac",
        )
    if not np.isfinite(median) or median <= 0.0:
        return barrier
    ratio = max(barrier / median, 1e-6)
    return float(max(multiplier * median * (ratio**power), 1e-6))


def validate_simple_policy_stop_params(
    params: Mapping[str, Any],
    *,
    state: Optional[Mapping[str, Any]] = None,
    require_metadata: bool = True,
    require_barrier: bool = True,
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

    missing = [key for key in REQUIRED_SIMPLE_POLICY_STOP_FIELDS if key not in params]
    if missing:
        raise SimplePolicyStopParamsError(
            "missing simple-policy stop fields: " + ",".join(missing)
        )

    activation = _first_positive(params, "trailing_activation_mult", "trailing_override_alpha")
    if not np.isfinite(activation):
        raise SimplePolicyStopParamsError(
            "missing simple-policy trailing activation field"
        )

    barrier_frac = _barrier_from_params_or_state(params, state)
    if not np.isfinite(barrier_frac):
        if require_barrier:
            raise SimplePolicyStopParamsError(
                "missing simple-policy barrier_frac/barrier_pct"
            )
        barrier_frac = np.nan

    values = {
        key: _safe_float(params.get(key), default=np.nan)
        for key in REQUIRED_SIMPLE_POLICY_STOP_FIELDS
        if key != "trailing_activation_mult"
    }
    values["atr_power"] = _safe_float(params.get("atr_power"), 1.0)
    values["atr_multiplier"] = _safe_float(params.get("atr_multiplier"), 1.0)
    values["hard_tp_abs_pct"] = _safe_float(params.get("hard_tp_abs_pct"), 0.0)
    values["exit_pressure_alpha"] = _safe_float(params.get("exit_pressure_alpha"), 1.0)
    values["exit_pressure_beta"] = _safe_float(params.get("exit_pressure_beta"), 0.0)
    values["exit_pressure_delta"] = _safe_float(params.get("exit_pressure_delta"), 1.0)
    values["exit_pressure_kappa"] = _safe_float(params.get("exit_pressure_kappa"), 0.0)
    values["exit_pressure_psi"] = _safe_float(params.get("exit_pressure_psi"), 0.7)
    values["exit_pressure_omega"] = _safe_float(params.get("exit_pressure_omega"), 1.0)
    values["exit_pressure_min_multiplier"] = _safe_float(
        params.get("exit_pressure_min_multiplier"), 1.0
    )
    values["redeploy_scale_bps"] = _safe_float(params.get("redeploy_scale_bps"), 100.0)
    values["target_holding_hours"] = _safe_float(params.get("target_holding_hours"), 0.0)
    values["churn_penalty_bps"] = _safe_float(params.get("churn_penalty_bps"), 100.0)
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
    values["atr_power"] = float(np.clip(values["atr_power"], 0.5, 1.2))
    values["atr_multiplier"] = float(np.clip(values["atr_multiplier"], 0.5, 2.0))
    if values["hard_tp_abs_pct"] < 0.0:
        raise SimplePolicyStopParamsError(
            "simple-policy hard_tp_abs_pct must be non-negative"
        )
    values["exit_pressure_alpha"] = float(np.clip(values["exit_pressure_alpha"], 0.25, 4.0))
    values["exit_pressure_beta"] = max(0.0, float(values["exit_pressure_beta"]))
    values["exit_pressure_delta"] = float(np.clip(values["exit_pressure_delta"], 0.25, 4.0))
    values["exit_pressure_kappa"] = max(0.0, float(values["exit_pressure_kappa"]))
    values["exit_pressure_psi"] = float(np.clip(values["exit_pressure_psi"], 0.0, 2.0))
    values["exit_pressure_omega"] = max(0.0, float(values["exit_pressure_omega"]))
    values["exit_pressure_min_multiplier"] = float(
        np.clip(values["exit_pressure_min_multiplier"], 0.01, 1.0)
    )
    values["redeploy_scale_bps"] = max(1e-6, float(values["redeploy_scale_bps"]))
    values["target_holding_hours"] = max(0.0, float(values["target_holding_hours"]))
    values["churn_penalty_bps"] = max(0.0, float(values["churn_penalty_bps"]))

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
    expected_strategy = str(
        state.get("strategy_id") or state.get("bucket_key") or ""
    ).strip()
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

    effective_barrier_frac = _effective_barrier_frac(barrier_frac, params, state)

    return ValidatedSimplePolicyParams(
        sl_mult=float(values["sl_mult"]),
        barrier_frac=float(effective_barrier_frac),
        trailing_activation_mult=float(activation),
        trailing_power=float(values["trailing_power"]),
        trailing_squash_divisor=float(values["trailing_squash_divisor"]),
        giveback_beta=float(values["giveback_beta"]),
        atr_power=float(values["atr_power"]),
        atr_multiplier=float(values["atr_multiplier"]),
        hard_tp_abs_pct=float(values["hard_tp_abs_pct"]),
        exit_pressure_enabled=bool(params.get("exit_pressure_enabled", False)),
        exit_pressure_alpha=float(values["exit_pressure_alpha"]),
        exit_pressure_beta=float(values["exit_pressure_beta"]),
        exit_pressure_delta=float(values["exit_pressure_delta"]),
        exit_pressure_kappa=float(values["exit_pressure_kappa"]),
        exit_pressure_psi=float(values["exit_pressure_psi"]),
        exit_pressure_omega=float(values["exit_pressure_omega"]),
        exit_pressure_min_multiplier=float(values["exit_pressure_min_multiplier"]),
        redeploy_scale_bps=float(values["redeploy_scale_bps"]),
        target_holding_hours=float(values["target_holding_hours"]),
        churn_penalty_bps=float(values["churn_penalty_bps"]),
        capital_protect_mfe_mult=float(values["capital_protect_mfe_mult"]),
        capital_protect_regression_frac=float(
            values["capital_protect_regression_frac"]
        ),
        adverse_exit_enabled=bool(params.get("adverse_exit_enabled", False)),
        adverse_exit_alpha=float(_safe_float(params.get("adverse_exit_alpha"), 1.0)),
        adverse_exit_beta=float(_safe_float(params.get("adverse_exit_beta"), 1.0)),
        adverse_exit_delta=float(_safe_float(params.get("adverse_exit_delta"), 1.0)),
        adverse_exit_theta_quantile=float(
            _safe_float(params.get("adverse_exit_theta_quantile"), 0.75)
        ),
        adverse_exit_theta=float(_safe_float(params.get("adverse_exit_theta"), np.nan)),
        adverse_exit_fast_bars=int(
            max(1, _safe_float(params.get("adverse_exit_fast_bars"), 4.0))
        ),
        adverse_exit_min_mae_atr=float(
            _safe_float(params.get("adverse_exit_min_mae_atr"), 1.0)
        ),
        adverse_exit_min_speed=float(
            _safe_float(params.get("adverse_exit_min_speed"), 0.3)
        ),
        adverse_exit_max_mfe_atr=float(
            _safe_float(params.get("adverse_exit_max_mfe_atr"), 0.25)
        ),
        enable_trailing=bool(params.get("enable_trailing", True)),
        strategy_id=strategy_id,
        params_source=params_source,
        params_hash=params_hash,
        schema=schema or SIMPLE_POLICY_SCHEMA,
    )


def _state_first_float(state: Mapping[str, Any], keys: tuple[str, ...], default: float = np.nan) -> float:
    for key in keys:
        val = _safe_float(state.get(key), default=np.nan)
        if np.isfinite(val):
            return float(val)
    return float(default)


def _runtime_exit_pressure(
    validated: ValidatedSimplePolicyParams,
    state: Mapping[str, Any],
    *,
    bars_in_trade: int,
    mfe: float,
) -> tuple[float, float]:
    if (
        not bool(validated.exit_pressure_enabled)
        or validated.exit_pressure_beta <= 0.0
        or validated.exit_pressure_min_multiplier >= 1.0
    ):
        return 0.0, 1.0
    hours_open = _state_first_float(
        state,
        ("hours_open", "holding_time_hours", "age_hours"),
        default=float(max(bars_in_trade, 0)) * 15.0 / 60.0,
    )
    target_hours = _state_first_float(
        state,
        ("target_holding_hours",),
        default=validated.target_holding_hours,
    )
    target_hours = max(float(target_hours), 15.0 / 60.0)
    explicit_capital_pressure = _state_first_float(
        state,
        ("capital_pressure",),
        default=np.nan,
    )
    if np.isfinite(explicit_capital_pressure):
        capital_pressure = float(np.clip(explicit_capital_pressure, 0.0, 1.0))
    else:
        capital_allocated = _state_first_float(
            state,
            ("capital_allocated", "Capital_allocated", "active_capital_allocated"),
            default=np.nan,
        )
        capital_allowed = _state_first_float(
            state,
            ("capital_allowed_total", "Capital_allowed_total", "allowed_capital_total"),
            default=np.nan,
        )
        if np.isfinite(capital_allocated) and np.isfinite(capital_allowed) and capital_allowed > 0.0:
            capital_pressure = float(
                np.clip(
                    max(0.0, capital_allocated / capital_allowed - validated.exit_pressure_psi)
                    * validated.exit_pressure_omega,
                    0.0,
                    1.0,
                )
            )
        else:
            capital_pressure = 0.0
    friction_bps = _state_first_float(
        state,
        ("expected_friction_bps",),
        default=np.nan,
    )
    if not np.isfinite(friction_bps):
        entry_spread = _state_first_float(
            state,
            ("expected_half_spread_bps", "spread_cost_bps"),
            default=0.0,
        )
        exit_spread = _state_first_float(
            state,
            ("exit_spread_cost_bps", "exit_quote_half_spread_bps"),
            default=0.0,
        )
        fee_bps = _state_first_float(
            state,
            ("round_trip_fee_bps", "fees_bps"),
            default=0.0,
        )
        friction_bps = max(0.0, entry_spread) + max(0.0, exit_spread) + max(0.0, fee_bps)
    current_ev = _state_first_float(
        state,
        (
            "current_trade_expected_EV_bps",
            "current_trade_ev_bps",
            "expected_EV_bps",
            "edge_bps",
        ),
        default=0.0,
    ) - max(0.0, friction_bps)
    candidate_ev = _state_first_float(
        state,
        (
            "best_available_cross_strategy_candidate_EV_bps",
            "candidate_EV_bps",
            "candidate_net_edge_bps",
        ),
        default=0.0,
    ) - max(0.0, friction_bps)
    unrealized_net_bps = _state_first_float(
        state,
        ("current_unrealized_net_bps", "unrealized_net_bps", "unrealized_pnl_bps"),
        default=float(max(float(mfe), 0.0)) * 10_000.0 - max(0.0, friction_bps),
    )
    expected_remaining_ev = max(current_ev - max(0.0, unrealized_net_bps), 0.0)
    candidate_net_edge = candidate_ev - validated.churn_penalty_bps
    redeploy_advantage = max(
        candidate_net_edge - expected_remaining_ev - validated.churn_penalty_bps,
        0.0,
    )
    redeploy_pressure = capital_pressure * float(
        np.clip(redeploy_advantage / max(validated.redeploy_scale_bps, 1e-6), 0.0, 2.0)
    )
    duration_pressure = validated.exit_pressure_kappa * (
        max(hours_open, 0.0) / max(target_hours, 1e-6)
    ) ** validated.exit_pressure_delta
    exit_pressure = float(np.clip(redeploy_pressure + duration_pressure, 0.0, 2.0))
    multiplier = float(
        np.clip(
            1.0 / (1.0 + validated.exit_pressure_beta * (exit_pressure ** validated.exit_pressure_alpha)),
            validated.exit_pressure_min_multiplier,
            1.0,
        )
    )
    return exit_pressure, multiplier


def compute_initial_simple_policy_stop_decision(
    *,
    entry_price: float,
    policy_params: Mapping[str, Any],
    side: str,
    strategy_id: Optional[str] = None,
    barrier_frac: Optional[float] = None,
    state: Optional[Mapping[str, Any]] = None,
    require_metadata: bool = True,
) -> SimplePolicyStopDecision:
    """Compute the canonical initial simple-policy STOP_LOSS order decision."""
    entry = _safe_float(entry_price, default=np.nan)
    if not np.isfinite(entry) or entry <= 0.0:
        raise SimplePolicyStopParamsError("missing finite entry_price")
    state = dict(state or {})
    state.setdefault("strategy_id", strategy_id or "")
    if barrier_frac is not None:
        state["barrier_frac"] = barrier_frac
    validated = validate_simple_policy_stop_params(
        policy_params,
        state=state,
        require_metadata=require_metadata,
        require_barrier=True,
    )
    side_l = str(side or "long").lower()
    if side_l == "long":
        stop_price = entry * (1.0 - validated.sl_mult * validated.barrier_frac)
    elif side_l == "short":
        stop_price = entry * (1.0 + validated.sl_mult * validated.barrier_frac)
    else:
        raise SimplePolicyStopParamsError(
            f"unsupported side for simple-policy stop: {side}"
        )
    if not np.isfinite(stop_price) or stop_price <= 0.0:
        raise SimplePolicyStopParamsError(
            "initial simple-policy stop_price must be positive"
        )
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
        atr_power=validated.atr_power,
        atr_multiplier=validated.atr_multiplier,
        hard_tp_abs_pct=validated.hard_tp_abs_pct,
        capital_protect_mfe_mult=validated.capital_protect_mfe_mult,
        capital_protect_regression_frac=validated.capital_protect_regression_frac,
        adverse_exit_enabled=validated.adverse_exit_enabled,
        adverse_exit_theta=validated.adverse_exit_theta,
        adverse_exit_theta_quantile=validated.adverse_exit_theta_quantile,
        adverse_exit_min_mae_atr=validated.adverse_exit_min_mae_atr,
        adverse_exit_min_speed=validated.adverse_exit_min_speed,
        adverse_exit_fast_bars=validated.adverse_exit_fast_bars,
        adverse_exit_max_mfe_atr=validated.adverse_exit_max_mfe_atr,
        peak_price=float(entry),
        mfe=0.0,
        mae=0.0,
        params_schema=validated.schema,
    )


def compute_simple_policy_initial_stop_decision(
    *args: Any, **kwargs: Any
) -> SimplePolicyStopDecision:
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
    bars_in_trade = int(max(0, _safe_float(state.get("bars_in_trade"), 0.0)))
    if not np.isfinite(entry_price) or entry_price <= 0.0:
        raise SimplePolicyStopParamsError("missing finite entry_price")
    if not np.isfinite(current_stop) or current_stop <= 0.0:
        raise SimplePolicyStopParamsError("missing finite current stop_price")
    exit_pressure = 0.0
    tightening_multiplier = 1.0
    effective_sl_mult = validated.sl_mult
    effective_trailing_activation_mult = validated.trailing_activation_mult
    effective_hard_tp_abs_pct = validated.hard_tp_abs_pct

    bars = _latest_bars(latest_market_state)
    if not bars.empty:
        for _, row in bars.iterrows():
            bars_in_trade += 1
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

            if validated.adverse_exit_enabled and np.isfinite(
                validated.adverse_exit_theta
            ):
                mae_atr = mae / max(validated.barrier_frac, 1e-12)
                mfe_atr = mfe / max(validated.barrier_frac, 1e-12)
                adverse_speed = mae_atr / max(bars_in_trade, 1)
                rank_source = _safe_float(
                    state.get(
                        "rank_percentile",
                        state.get(
                            "sizer_rank_percentile",
                            state.get("meta_train_rank_pct", np.nan),
                        ),
                    ),
                    default=np.nan,
                )
                ranked_confidence = float(np.clip(rank_source - 0.5, 0.0, 0.5))
                eligible = (
                    bars_in_trade <= int(validated.adverse_exit_fast_bars)
                    and mae_atr >= validated.adverse_exit_min_mae_atr
                    and mae_atr <= validated.sl_mult * ADVERSE_EXIT_MAX_SL_FRACTION
                    and adverse_speed >= validated.adverse_exit_min_speed
                    and mfe_atr <= validated.adverse_exit_max_mfe_atr
                )
                log_exit_score = (
                    np.log1p(validated.adverse_exit_alpha * (1.0 - ranked_confidence))
                    + np.log1p(validated.adverse_exit_beta * mae_atr)
                    + np.log1p(validated.adverse_exit_delta * adverse_speed)
                )
                if eligible and float(log_exit_score) > validated.adverse_exit_theta:
                    detail = (
                        "adverse_excursion_exit: "
                        f"bars={bars_in_trade} mae_atr={mae_atr:.6g} "
                        f"mfe_atr={mfe_atr:.6g} speed={adverse_speed:.6g} "
                        f"ranked_confidence_minus_0_5={ranked_confidence:.6g} "
                        f"score={float(log_exit_score):.6g} "
                        f"theta={validated.adverse_exit_theta:.6g}"
                    )
                    return SimplePolicyStopDecision(
                        should_replace=False,
                        stop_price=None,
                        requested_policy_stop=None,
                        reason="adverse_excursion_exit",
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
                        atr_power=validated.atr_power,
                        atr_multiplier=validated.atr_multiplier,
                        hard_tp_abs_pct=validated.hard_tp_abs_pct,
                        exit_pressure=exit_pressure,
                        tightening_multiplier=tightening_multiplier,
                        effective_sl_mult=effective_sl_mult,
                        effective_trailing_activation_mult=effective_trailing_activation_mult,
                        effective_hard_tp_abs_pct=effective_hard_tp_abs_pct,
                        target_holding_hours=validated.target_holding_hours,
                        churn_penalty_bps=validated.churn_penalty_bps,
                        capital_protect_mfe_mult=validated.capital_protect_mfe_mult,
                        capital_protect_regression_frac=validated.capital_protect_regression_frac,
                        adverse_exit_enabled=True,
                        adverse_exit_theta=validated.adverse_exit_theta,
                        adverse_exit_theta_quantile=validated.adverse_exit_theta_quantile,
                        adverse_exit_min_mae_atr=validated.adverse_exit_min_mae_atr,
                        adverse_exit_min_speed=validated.adverse_exit_min_speed,
                        adverse_exit_fast_bars=validated.adverse_exit_fast_bars,
                        adverse_exit_max_mfe_atr=validated.adverse_exit_max_mfe_atr,
                        should_exit=True,
                        exit_reason="adverse_excursion_exit",
                        peak_price=(
                            float(peak_price) if np.isfinite(peak_price) else None
                        ),
                        mfe=float(mfe),
                        mae=float(mae),
                        params_schema=validated.schema,
                    )

    candidate = float(current_stop)
    reason = "original_stop_loss"
    detail = "unchanged_original_stop_loss"
    exit_pressure, tightening_multiplier = _runtime_exit_pressure(
        validated,
        state,
        bars_in_trade=bars_in_trade,
        mfe=float(mfe),
    )
    effective_sl_mult = validated.sl_mult * tightening_multiplier
    effective_trailing_activation_mult = (
        validated.trailing_activation_mult * tightening_multiplier
    )
    effective_hard_tp_abs_pct = validated.hard_tp_abs_pct * tightening_multiplier
    pressure_stop = (
        entry_price * (1.0 - effective_sl_mult * validated.barrier_frac)
        if side_l == "long"
        else entry_price * (1.0 + effective_sl_mult * validated.barrier_frac)
    )
    pressure_improved = (
        pressure_stop > candidate if side_l == "long" else pressure_stop < candidate
    )
    if pressure_improved:
        candidate = float(pressure_stop)
        reason = "exit_pressure_stop_tightening"
        detail = (
            "exit_pressure_stop_tightening: "
            f"exit_pressure={exit_pressure:.6g} "
            f"multiplier={tightening_multiplier:.6g} "
            f"effective_sl_mult={effective_sl_mult:.6g}"
        )

    cap_mfe_mult = validated.capital_protect_mfe_mult
    cap_reg_frac = validated.capital_protect_regression_frac
    if cap_mfe_mult > 0.0:
        sl_dist_ret = effective_sl_mult * validated.barrier_frac
        x_dist = cap_mfe_mult * validated.barrier_frac
        lock_dist = x_dist - cap_reg_frac * (x_dist + sl_dist_ret)
        if float(mfe) >= x_dist:
            cap_stop = (
                entry_price * (1.0 + lock_dist)
                if side_l == "long"
                else entry_price * (1.0 - lock_dist)
            )
            improved = (
                cap_stop > candidate if side_l == "long" else cap_stop < candidate
            )
            if improved:
                candidate = float(cap_stop)
                reason = "capital_preservation"
                detail = (
                    f"capital_preservation: mfe={float(mfe):.6g} "
                    f"trigger={x_dist:.6g} lock_dist={lock_dist:.6g}"
                )

    if effective_hard_tp_abs_pct > 0.0 and float(mfe) >= effective_hard_tp_abs_pct:
        detail = (
            "hard_take_profit_exit: "
            f"mfe={float(mfe):.6g} "
            f"hard_tp_abs_pct={effective_hard_tp_abs_pct:.6g} "
            f"exit_pressure={exit_pressure:.6g} multiplier={tightening_multiplier:.6g}"
        )
        return SimplePolicyStopDecision(
            should_replace=False,
            stop_price=None,
            requested_policy_stop=None,
            reason="hard_take_profit_exit",
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
            atr_power=validated.atr_power,
            atr_multiplier=validated.atr_multiplier,
            hard_tp_abs_pct=validated.hard_tp_abs_pct,
            exit_pressure=exit_pressure,
            tightening_multiplier=tightening_multiplier,
            effective_sl_mult=effective_sl_mult,
            effective_trailing_activation_mult=effective_trailing_activation_mult,
            effective_hard_tp_abs_pct=effective_hard_tp_abs_pct,
            target_holding_hours=validated.target_holding_hours,
            churn_penalty_bps=validated.churn_penalty_bps,
            capital_protect_mfe_mult=validated.capital_protect_mfe_mult,
            capital_protect_regression_frac=validated.capital_protect_regression_frac,
            adverse_exit_enabled=validated.adverse_exit_enabled,
            adverse_exit_theta=validated.adverse_exit_theta,
            adverse_exit_theta_quantile=validated.adverse_exit_theta_quantile,
            adverse_exit_min_mae_atr=validated.adverse_exit_min_mae_atr,
            adverse_exit_min_speed=validated.adverse_exit_min_speed,
            adverse_exit_fast_bars=validated.adverse_exit_fast_bars,
            adverse_exit_max_mfe_atr=validated.adverse_exit_max_mfe_atr,
            should_exit=True,
            exit_reason="hard_take_profit_exit",
            peak_price=float(peak_price) if np.isfinite(peak_price) else None,
            mfe=float(mfe),
            mae=float(mae),
            params_schema=validated.schema,
        )

    activation = effective_trailing_activation_mult * validated.barrier_frac
    if validated.enable_trailing and float(mfe) > activation:
        profit_above_activation = max(float(mfe) - activation, 0.0)
        power_giveback = (
            profit_above_activation**validated.trailing_power
        ) / max(validated.trailing_squash_divisor, 1e-12)
        atr_giveback = validated.giveback_beta * validated.barrier_frac
        lock_ret = max(
            activation,
            min(float(mfe) - power_giveback, float(mfe) - atr_giveback),
        )
        trail_stop = (
            entry_price * (1.0 + lock_ret)
            if side_l == "long"
            else entry_price * (1.0 - lock_ret)
        )
        improved = (
            trail_stop > candidate if side_l == "long" else trail_stop < candidate
        )
        if improved:
            candidate = float(trail_stop)
            reason = "trailing_profit"
            detail = (
                f"trailing_profit: mfe={float(mfe):.6g} activation={activation:.6g} "
                f"exit_pressure={exit_pressure:.6g} "
                f"multiplier={tightening_multiplier:.6g} "
                f"giveback_beta={validated.giveback_beta:.6g} "
                f"power_giveback={power_giveback:.6g} "
                f"atr_giveback={atr_giveback:.6g} "
                f"lock_ret={lock_ret:.6g}"
            )

    should_replace = (
        candidate > current_stop if side_l == "long" else candidate < current_stop
    )
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
        atr_power=validated.atr_power,
        atr_multiplier=validated.atr_multiplier,
        hard_tp_abs_pct=validated.hard_tp_abs_pct,
        exit_pressure=exit_pressure,
        tightening_multiplier=tightening_multiplier,
        effective_sl_mult=effective_sl_mult,
        effective_trailing_activation_mult=effective_trailing_activation_mult,
        effective_hard_tp_abs_pct=effective_hard_tp_abs_pct,
        target_holding_hours=validated.target_holding_hours,
        churn_penalty_bps=validated.churn_penalty_bps,
        capital_protect_mfe_mult=validated.capital_protect_mfe_mult,
        capital_protect_regression_frac=validated.capital_protect_regression_frac,
        adverse_exit_enabled=validated.adverse_exit_enabled,
        adverse_exit_theta=validated.adverse_exit_theta,
        adverse_exit_theta_quantile=validated.adverse_exit_theta_quantile,
        adverse_exit_min_mae_atr=validated.adverse_exit_min_mae_atr,
        adverse_exit_min_speed=validated.adverse_exit_min_speed,
        adverse_exit_fast_bars=validated.adverse_exit_fast_bars,
        adverse_exit_max_mfe_atr=validated.adverse_exit_max_mfe_atr,
        peak_price=float(peak_price) if np.isfinite(peak_price) else None,
        mfe=float(mfe),
        mae=float(mae),
        last_eval_ts=last_eval_ts,
        params_schema=validated.schema,
    )
