"""Shared parity helpers for inference and inference_backtest paths."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

import pandas as pd

from extreme_price_movements.simple_position_sizer import (
    calibrate_score,
    load_calibration_contract,
)
from extreme_price_movements.utils import tprint

LIVE_UNAVAILABLE_FEATURES: Set[str] = {
    "reg_gate_target",
    "reg_train_target",
    "reg_target_positive",
    "reg_raw_vol_norm",
    "y_move",
    "y_move_soft",
    "move_threshold",
    "barrier_pct",
    "bars_to_mfe",
    "reg_weight",
}


def _bundle_payload(model_bundle: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(model_bundle, dict):
        return {}
    return model_bundle.get("bundle", model_bundle)


def _alpha_strategy_index(model_bundle: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Normalize flat and nested alpha model bundle layouts."""
    bundle = _bundle_payload(model_bundle)
    alpha_models = bundle.get("alpha_models", {}) if isinstance(bundle, dict) else {}
    out: Dict[str, Dict[str, Any]] = {}
    if not isinstance(alpha_models, dict):
        return out
    for key, value in alpha_models.items():
        if not isinstance(value, dict):
            continue
        if "model" in value or "feat_cols" in value:
            out[str(key)] = value
            continue
        for nested_key, model_info in value.items():
            if isinstance(model_info, dict):
                out[f"{key}_{nested_key}"] = model_info
    return out


def _meta_model_keys(model_bundle: Dict[str, Any]) -> Set[str]:
    bundle = _bundle_payload(model_bundle)
    meta_models = bundle.get("meta_models", {}) if isinstance(bundle, dict) else {}
    return (
        set(str(key) for key in meta_models.keys())
        if isinstance(meta_models, dict)
        else set()
    )


def _meta_models(model_bundle: Dict[str, Any]) -> Dict[str, Any]:
    bundle = _bundle_payload(model_bundle)
    meta_models = bundle.get("meta_models", {}) if isinstance(bundle, dict) else {}
    return meta_models if isinstance(meta_models, dict) else {}


def _candidate_meta_keys(strategy_id: str) -> Set[str]:
    sid = str(strategy_id or "")
    core = strategy_core_id(sid)
    side = strategy_side(sid)
    bases = {sid, core}
    if side and core:
        bases.add(f"{side}_{core}")
    out: Set[str] = set()
    for base in bases:
        if not base:
            continue
        out.add(base)
        for suffix in ("_clf", "_tbm_clf", "_reg", "_early_inval"):
            out.add(f"{base}{suffix}")
    return out


def _model_raw_selected_features(model: Any) -> List[str]:
    vals: List[str] = []
    for source in (model, getattr(model, "best_model", None)):
        if source is None:
            continue
        for attr in ("raw_selected_features", "selected_features"):
            raw = getattr(source, attr, None)
            if raw:
                vals.extend(str(v) for v in raw if str(v))
    return vals


def _model_meta_feature_columns(model: Any) -> List[str]:
    vals: List[str] = []
    for source in (model, getattr(model, "best_model", None)):
        if source is None:
            continue
        raw = getattr(source, "meta_feature_columns_", None)
        if raw:
            vals.extend(str(v) for v in raw if str(v))
    return list(dict.fromkeys(vals))


def _looks_positional_feature(feature: str) -> bool:
    return bool(re.fullmatch(r"f\d+", str(feature or "")))


def strategy_core_id(strategy_id: str) -> str:
    """Return the side-agnostic strategy id used by policy artifacts."""
    sid = str(strategy_id or "")
    for prefix in ("long_", "short_"):
        if sid.startswith(prefix):
            sid = sid[len(prefix) :]
            break
    return re.sub(r"_H\d+$", "", sid)


def strategy_side(strategy_id: str) -> str:
    """Infer strategy side from its identifier when available."""
    sid = str(strategy_id or "").lower()
    if sid.startswith("long_"):
        return "long"
    if sid.startswith("short_"):
        return "short"
    return ""


def strategy_id_matches(strategy_id: str, allowed: Optional[Set[str]]) -> bool:
    """Match either full side-prefixed ids or policy core ids."""
    if allowed is None:
        return True
    sid = str(strategy_id or "")
    sid_side = strategy_side(sid)
    aliases = _strategy_aliases(sid)
    for candidate in allowed:
        candidate_s = str(candidate)
        candidate_side = strategy_side(candidate_s)
        if (
            sid_side in {"long", "short"}
            and candidate_side in {"long", "short"}
            and sid_side != candidate_side
        ):
            continue
        if aliases & _strategy_aliases(candidate_s):
            return True
    return False


def _normalise_symbol(symbol: str) -> str:
    raw = str(symbol or "").strip().upper().replace("_", "/")
    if "/" in raw:
        return raw
    for quote in ("USDT", "USDC", "BUSD", "USD1", "FDUSD", "BTC", "ETH"):
        if raw.endswith(quote) and len(raw) > len(quote):
            return f"{raw[:-len(quote)]}/{quote}"
    return raw


def _strategy_ids_from_rows(rows: Any, *, side_aware: bool = False) -> Set[str]:
    out: Set[str] = set()
    if not isinstance(rows, list):
        return out
    for row in rows:
        if not isinstance(row, dict):
            continue
        if row.get("selected") is False:
            continue
        sid = (
            row.get("strategy_for_inference")
            or row.get("strategy_id")
            or row.get("strategy")
            or row.get("selected_strategy")
        )
        if sid:
            sid_s = str(sid)
            side = str(row.get("side") or strategy_side(sid_s)).lower()
            core = strategy_core_id(sid_s)
            if side_aware and side in {"long", "short"} and core:
                out.add(f"{side}_{core}")
            else:
                out.add(sid_s)
                if core not in {"mr", "tf"}:
                    out.add(core)
    return {s for s in out if s}


def _avg_pnl_per_trade_from_strategy_row(row: Dict[str, Any]) -> float:
    candidates = [
        row.get("avg_net_pnl_per_trade"),
        row.get("mean_net_trade"),
        row.get("avg_pnl_bankroll"),
    ]
    metrics = row.get("metrics")
    if isinstance(metrics, dict):
        for key in ("top_5", "final_fit_all", "cv_validation_average"):
            sub = metrics.get(key)
            if isinstance(sub, dict):
                candidates.extend(
                    [
                        sub.get("avg_pnl_bankroll"),
                        sub.get("mean_net_trade"),
                        sub.get("avg_net_pnl_per_trade"),
                    ]
                )
    for val in candidates:
        try:
            out = float(val)
        except Exception:
            continue
        if pd.notna(out):
            return out
    return float("-inf")


def _select_top_strategy_rows_by_side(rows: Any) -> List[Dict[str, Any]]:
    if not isinstance(rows, list):
        return []
    best_by_side: Dict[str, tuple[float, Dict[str, Any]]] = {}
    passthrough: List[Dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        if row.get("selected") is False:
            continue
        sid = (
            row.get("strategy_for_inference")
            or row.get("strategy_id")
            or row.get("strategy")
            or row.get("selected_strategy")
        )
        side = str(row.get("side") or strategy_side(str(sid or ""))).lower()
        if side not in {"long", "short"}:
            passthrough.append(row)
            continue
        score = _avg_pnl_per_trade_from_strategy_row(row)
        current = best_by_side.get(side)
        if current is None or score > current[0]:
            best_by_side[side] = (score, row)
    selected = [item[1] for _, item in sorted(best_by_side.items())]
    selected.extend(passthrough)
    return selected


def _strategy_aliases(strategy_id: str) -> Set[str]:
    sid = str(strategy_id or "")
    core = strategy_core_id(sid)
    side = strategy_side(sid)
    aliases = {sid}
    if core and core not in {"mr", "tf", "none"}:
        aliases.add(core)
    if side and core:
        aliases.add(f"{side}_{core}")
    return {alias for alias in aliases if alias}


def _with_strategy_aliases(strategy_ids: Set[str]) -> Set[str]:
    """Return strategy ids plus side/core aliases used by persisted artifacts."""
    out: Set[str] = set()
    for sid in strategy_ids:
        sid_s = str(sid or "")
        if not sid_s:
            continue
        out.add(sid_s)
        core = strategy_core_id(sid_s)
        side = strategy_side(sid_s)
        if core and core not in {"mr", "tf", "none"} and not side:
            out.add(core)
        if side and core:
            out.add(f"{side}_{core}")
    return out


def load_strategy_for_inference_filter(
    data_root: str, run_id: str
) -> Optional[Set[str]]:
    """Load the explicit deployment strategy set if present.

    ``policy_optimiser.py`` is the preferred source. ``holdout_strategy_eval.py``
    can still write the root artifact, but it is no longer mandatory for live
    inference readiness.
    """
    base = Path(data_root) / "artifacts" / run_id
    paths = [
        base / "policy_params" / "strategy_for_inference.json",
        base / "ridge_sizer" / "strategy_for_inference.json",
        base / "policy_params" / "strategy_for_inference.csv",
        base / "ridge_sizer" / "strategy_for_inference.csv",
        base / "strategy_for_inference.json",
        base / "strategy_for_inference.csv",
    ]

    for path in paths:
        if not path.exists():
            continue
        try:
            if path.suffix.lower() == ".csv":
                with path.open("r", newline="") as f:
                    rows = list(csv.DictReader(f))
                selected = _strategy_ids_from_rows(rows, side_aware=True)
            else:
                payload = json.loads(path.read_text())
                if isinstance(payload, dict):
                    rows = (
                        payload.get("strategies")
                        or payload.get("strategy_for_inference")
                        or payload.get("selected_strategies")
                        or []
                    )
                    if isinstance(rows, str):
                        rows = [{"strategy_id": rows}]
                    selected_rows = _select_top_strategy_rows_by_side(rows)
                    selected = _strategy_ids_from_rows(selected_rows, side_aware=True)
                    sid = payload.get("strategy_id") or payload.get("selected_strategy")
                    if sid:
                        selected.add(str(sid))
                        selected.add(strategy_core_id(str(sid)))
                else:
                    selected = _strategy_ids_from_rows(payload, side_aware=True)
            tprint(
                "[StrategyFilter] Loaded top deployment inference ids from "
                f"{path}: aliases={len(selected)} ids={sorted(selected)}"
            )
            return selected
        except Exception as exc:
            tprint(f"[StrategyFilter] Error loading {path}: {exc}")
    return None


def load_strategy_asset_exclusion_filter(
    data_root: str, run_id: str
) -> Dict[str, Set[str]]:
    """Load per-strategy symbols excluded by policy optimiser asset diagnostics."""
    base = Path(data_root) / "artifacts" / run_id
    paths = [
        base / "policy_params" / "strategy_for_inference.json",
        base / "ridge_sizer" / "strategy_for_inference.json",
        base / "strategy_for_inference.json",
    ]
    exclusions: Dict[str, Set[str]] = {}
    for path in paths:
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text())
            if not isinstance(payload, dict):
                continue
            raw_exclusions = payload.get("asset_exclusions", {})
            if isinstance(raw_exclusions, dict):
                for sid, symbols in raw_exclusions.items():
                    if not isinstance(symbols, list):
                        continue
                    cleaned = {
                        _normalise_symbol(str(sym))
                        for sym in symbols
                        if _normalise_symbol(str(sym))
                    }
                    for alias in _strategy_aliases(str(sid)):
                        exclusions.setdefault(alias, set()).update(cleaned)
            rows = (
                payload.get("strategies")
                or payload.get("strategy_for_inference")
                or payload.get("selected_strategies")
                or []
            )
            if isinstance(rows, dict):
                rows = [rows]
            if isinstance(rows, list):
                for row in rows:
                    if not isinstance(row, dict) or row.get("selected") is False:
                        continue
                    sid = (
                        row.get("strategy_for_inference")
                        or row.get("strategy_id")
                        or row.get("strategy")
                        or row.get("selected_strategy")
                    )
                    if not sid:
                        continue
                    symbols = (
                        row.get("excluded_symbols")
                        or row.get("blocked_symbols")
                        or row.get("asset_exclusions")
                        or []
                    )
                    if not isinstance(symbols, list):
                        continue
                    cleaned = {
                        _normalise_symbol(str(sym))
                        for sym in symbols
                        if _normalise_symbol(str(sym))
                    }
                    for alias in _strategy_aliases(str(sid)):
                        exclusions.setdefault(alias, set()).update(cleaned)
            if any(symbols for symbols in exclusions.values()):
                tprint(
                    f"[StrategyFilter] Loaded asset exclusions for "
                    f"{len(exclusions)} strategy aliases from {path}"
                )
                return exclusions
        except Exception as exc:
            tprint(f"[StrategyFilter] Error loading asset exclusions {path}: {exc}")
    return exclusions


def load_policy_params_by_strategy(
    data_root: str, run_id: str
) -> Dict[str, Dict[str, Any]]:
    """Load best policy optimiser params keyed by full and core strategy ids."""
    paths = [
        Path(data_root)
        / "artifacts"
        / run_id
        / "policy_params"
        / "best_policy_params.json",
        Path(data_root) / "artifacts" / run_id / "best_policy_params.json",
    ]
    out: Dict[str, Dict[str, Any]] = {}
    for path in paths:
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text())
            strategies = []
            if isinstance(payload, dict):
                strategy_for_inference = payload.get("strategy_for_inference")
                if isinstance(strategy_for_inference, dict):
                    strategies = strategy_for_inference.get("strategies", [])
                elif isinstance(strategy_for_inference, list):
                    strategies = strategy_for_inference
                else:
                    strategies = payload.get("strategies", [])
            for row in strategies:
                if not isinstance(row, dict) or not row.get("strategy_id"):
                    continue
                if row.get("selected") is False:
                    continue
                params = dict(row)
                sid = str(params["strategy_id"])
                core = strategy_core_id(sid)
                out[sid] = params
                out[core] = params
                side = strategy_side(sid)
                if side:
                    out[f"{side}_{core}"] = params
            tprint(
                f"[PolicyParams] Loaded {len(strategies)} selected policy optimiser rows from {path}"
            )
            return out
        except Exception as exc:
            tprint(f"[PolicyParams] Error loading {path}: {exc}")
    return out


def load_policy_strategy_filter(data_root: str, run_id: str) -> Optional[Set[str]]:
    """Load strategies that have policy optimiser params available for inference."""
    params = load_policy_params_by_strategy(data_root, run_id)
    if not params:
        return None
    selected = _with_strategy_aliases(set(params.keys()))
    tprint(f"[StrategyFilter] Loaded {len(selected)} policy-param strategy aliases")
    return selected


def load_profitable_sizer_strategy_filter(
    data_root: str,
    run_id: str,
    *,
    min_wallet_pnl: float = 0.0,
    min_net_pnl: float = 0.0,
) -> Optional[Set[str]]:
    """Load strategies tagged/proven profitable by simple_position_sizer.

    The current simple_position_sizer artifact lives under ``ridge_sizer`` and
    stores one row per strategy. Future reruns also write explicit downstream
    allow fields; older artifacts are interpreted using positive wallet and
    net PnL.
    """
    paths = [
        Path(data_root) / "artifacts" / run_id / "ridge_sizer" / "strategy_params.json",
        Path(data_root)
        / "artifacts"
        / run_id
        / "simple_sizer"
        / "strategy_params.json",
    ]
    for path in paths:
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text())
            rows = payload.get("strategies", []) if isinstance(payload, dict) else []
            selected: Set[str] = set()
            for row in rows:
                if not isinstance(row, dict):
                    continue
                sid = str(row.get("strategy_id") or row.get("strategy") or "")
                if not sid:
                    continue
                explicit_allowed = row.get("allow_downstream")
                if explicit_allowed is None:
                    explicit_allowed = row.get("profitable_for_downstream")
                if explicit_allowed is None:
                    wallet_pnl = float(row.get("wallet_pnl", float("nan")))
                    net_pnl = float(row.get("net_pnl", float("nan")))
                    is_allowed = (
                        pd.notna(wallet_pnl)
                        and pd.notna(net_pnl)
                        and wallet_pnl > float(min_wallet_pnl)
                        and net_pnl > float(min_net_pnl)
                    )
                else:
                    is_allowed = bool(explicit_allowed)
                if is_allowed:
                    selected.add(sid)
            selected = _with_strategy_aliases(selected)
            tprint(
                f"[StrategyFilter] Loaded {len(selected)} profitable sizer aliases from {path}"
            )
            return selected
        except Exception as exc:
            tprint(f"[StrategyFilter] Error loading {path}: {exc}")
    return None


def resolve_deployment_strategy_filter(
    data_root: str,
    run_id: str,
) -> Optional[Set[str]]:
    """Resolve the deployable inference strategy set.

    Deployment prefers strategies selected by ``policy_optimiser.py`` and then
    intersects them with simple_position_sizer profitability plus available
    policy optimiser params. ``holdout_strategy_eval.py`` output is optional:
    if present it is accepted as a compatible strategy_for_inference artifact,
    but it is not a prerequisite for production readiness.
    """
    accepted = load_strategy_for_inference_filter(data_root, run_id)
    explicit_strategy_for_inference = accepted is not None
    if accepted is not None and len(accepted) == 0:
        tprint(
            "[StrategyFilter] Explicit strategy_for_inference is empty; "
            "no strategies are deployable until policy_optimiser selects one"
        )
        return set()
    if accepted is None:
        accepted = load_strategy_acceptance_filter(data_root, run_id)
    profitable = load_profitable_sizer_strategy_filter(data_root, run_id)
    policy_ready = load_policy_strategy_filter(data_root, run_id)

    filters = {
        "accepted": accepted,
        "profitable": profitable,
        "policy_ready": policy_ready,
    }
    active_filters = {
        name: f for name, f in filters.items() if f is not None and len(f) > 0
    }
    if not active_filters:
        return None

    selected = set(
        active_filters.get("accepted") or next(iter(active_filters.values()))
    )
    policy_selected = selected
    if policy_ready is not None and len(policy_ready) > 0:
        policy_selected = {
            sid
            for sid in policy_selected
            if any(strategy_id_matches(sid, {candidate}) for candidate in policy_ready)
        }
        if policy_selected:
            selected = policy_selected

    if profitable is not None and len(profitable) > 0:
        profitable_selected = {
            sid
            for sid in selected
            if any(strategy_id_matches(sid, {candidate}) for candidate in profitable)
        }
        if profitable_selected and not explicit_strategy_for_inference:
            selected = profitable_selected
        elif profitable_selected and len(profitable_selected) < len(selected):
            tprint(
                "[StrategyFilter] Explicit strategy_for_inference selects "
                f"{len(selected)} aliases; profitable sizer allow-list would reduce "
                f"this to {len(profitable_selected)} aliases, so the explicit "
                "deployment contract is authoritative for this run"
            )
        elif policy_selected:
            tprint(
                "[StrategyFilter] Profitable sizer allow-list has no overlap with "
                "policy-selected deployment strategies; using policy optimiser "
                "strategy_for_inference as authoritative for this run"
            )
    selected = _with_strategy_aliases(selected)
    tprint(
        "[StrategyFilter] Deployment strategies resolved: " f"{len(selected)} aliases"
    )
    return selected


def load_strategy_acceptance_filter(data_root: str, run_id: str) -> Optional[Set[str]]:
    """Load accepted strategy identifiers from policy optimiser artifacts."""
    paths = [
        Path(data_root) / "artifacts" / run_id / "strategy_final_acceptation.json",
        Path(data_root)
        / "artifacts"
        / run_id
        / "policy_params"
        / "strategy_final_acceptation.json",
    ]

    for path in paths:
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text())
            strategies = payload.get("strategies", [])
            accepted = _strategy_ids_from_rows(strategies)
            tprint(
                f"[StrategyFilter] Loaded {len(accepted)} accepted strategies from {path}"
            )
            return accepted
        except Exception as exc:
            tprint(f"[StrategyFilter] Error loading {path}: {exc}")
    return None


def apply_strategy_acceptance_filter(
    df: pd.DataFrame,
    accepted_strategies: Optional[Set[str]],
    strategy_col: str = "strategy",
) -> pd.DataFrame:
    """Filter rows to strategies accepted by policy optimisation."""
    if accepted_strategies is None:
        return df
    n_before = len(df)
    out = df[
        df[strategy_col]
        .astype(str)
        .map(lambda sid: strategy_id_matches(sid, accepted_strategies))
    ].copy()
    tprint(f"[StrategyFilter] {n_before} -> {len(out)} rows after acceptance filtering")
    return out


def calibrated_score_and_threshold(
    raw_score: float,
    strategy_id: str,
    calibration_data: Dict[str, Dict[str, Any]],
    default_threshold: float = 1.0,
) -> tuple[float, float]:
    """Return calibrated score and p75 threshold for a strategy."""
    if not calibration_data:
        return float(raw_score), float(default_threshold)

    sid = str(strategy_id)
    calib = calibration_data.get(sid, {}) if isinstance(calibration_data, dict) else {}
    calibrated = float(calibrate_score(raw_score, sid, calibration_data))
    p75 = float(calib.get("p75_threshold", default_threshold) or default_threshold)
    return calibrated, p75


def passes_rank_filter(
    raw_score: float,
    strategy_id: str,
    calibration_data: Dict[str, Dict[str, Any]],
    default_threshold: float = 1.0,
) -> bool:
    """Check if a score passes strategy-specific confidence rank threshold."""
    calibrated, threshold = calibrated_score_and_threshold(
        raw_score=raw_score,
        strategy_id=strategy_id,
        calibration_data=calibration_data,
        default_threshold=default_threshold,
    )
    return bool(calibrated >= threshold)


def calibration_size_multiplier(
    raw_score: float,
    strategy_id: str,
    calibration_data: Dict[str, Dict[str, Any]],
    default_threshold: float = 1.0,
    max_mult: float = 2.0,
) -> float:
    """Convert calibrated rank strength into a bounded sizing multiplier."""
    calibrated, threshold = calibrated_score_and_threshold(
        raw_score=raw_score,
        strategy_id=strategy_id,
        calibration_data=calibration_data,
        default_threshold=default_threshold,
    )
    den = max(float(threshold), 1e-6)
    rel = max(0.0, float(calibrated) / den)
    return float(min(rel, float(max_mult)))


def validate_calibration_artifacts(
    data_root: str,
    run_id: str,
    calibration_data: Dict[str, Dict[str, Any]],
    *,
    strict: bool = True,
) -> bool:
    """Validate calibration artifact schema expected by inference runtime."""
    contract = load_calibration_contract(data_root, run_id)
    if not contract:
        if strict and calibration_data:
            raise ValueError(
                "Calibration data exists but confidence_calibration.contract.json is missing"
            )
        return False
    req = list(contract.get("required_strategy_fields", []) or [])
    for sid, row in (calibration_data or {}).items():
        missing = [k for k in req if k not in row]
        if missing:
            raise ValueError(
                f"Calibration artifact schema mismatch for strategy {sid}: missing={missing}"
            )
    return True


def load_meta_feature_contract(data_root: str, run_id: str) -> Dict[str, Any]:
    """Load the train_meta positional feature contract required by live meta heads."""
    path = (
        Path(data_root)
        / "artifacts"
        / str(run_id)
        / "meta_oof"
        / "meta_feature_contract.json"
    )
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text())
    except Exception as exc:
        raise ValueError(f"Unable to load meta feature contract {path}: {exc}") from exc
    return payload if isinstance(payload, dict) else {}


def validate_meta_feature_contract_artifact(
    data_root: str,
    run_id: str,
    model_bundle: Dict[str, Any],
    accepted_strategies: Optional[Set[str]] = None,
    *,
    strict: bool = True,
) -> bool:
    """Require train_meta's ``fN`` to raw-feature mapping before live inference.

    EBM-on-LGBM meta heads may persist selected features as positional names
    (``f0``, ``f1``, ...). Without the train-time mapping to raw meta feature
    columns, OOS/live scoring can silently evaluate a different feature order.
    """
    meta_models = _meta_models(model_bundle)
    if not meta_models:
        return True

    payload = load_meta_feature_contract(data_root, run_id)
    contract_models = (
        payload.get("meta_models", {}) if isinstance(payload, dict) else {}
    )
    if not isinstance(contract_models, dict) or not contract_models:
        message = (
            "meta_oof/meta_feature_contract.json is missing or empty. "
            "Rerun train_meta with the current code before inference or OOS "
            "meta-ranked metrics."
        )
        if strict:
            raise ValueError(message)
        tprint(f"[MetaFeatureContract] WARNING: {message}")
        return False

    required_keys: Set[str]
    if accepted_strategies:
        required_keys = set()
        for sid in accepted_strategies:
            required_keys.update(_candidate_meta_keys(str(sid)) & set(meta_models))
        if not required_keys:
            required_keys = {
                key
                for key in meta_models
                if strategy_id_matches(str(key), accepted_strategies)
            }
    else:
        required_keys = set(str(key) for key in meta_models)

    errors: List[str] = []
    for key in sorted(required_keys):
        row = contract_models.get(key)
        if row is None:
            alias_hits = [
                candidate
                for candidate in _candidate_meta_keys(key)
                if candidate in contract_models
            ]
            row = contract_models.get(alias_hits[0]) if alias_hits else None
        if not isinstance(row, dict):
            errors.append(f"{key}: missing meta feature contract row")
            continue

        feature_columns = [str(v) for v in row.get("feature_columns", []) if str(v)]
        mapping = row.get("positional_feature_mapping", {})
        mapping = mapping if isinstance(mapping, dict) else {}
        n_features = int(row.get("n_features", len(feature_columns)) or 0)
        if not feature_columns:
            errors.append(f"{key}: feature_columns empty")
        if not mapping:
            errors.append(f"{key}: positional_feature_mapping empty")
        if n_features != len(feature_columns):
            errors.append(
                f"{key}: n_features={n_features} differs from "
                f"len(feature_columns)={len(feature_columns)}"
            )
        expected_positional = {f"f{i}" for i in range(len(feature_columns))}
        missing_mapping = sorted(expected_positional - set(str(k) for k in mapping))
        if missing_mapping:
            errors.append(f"{key}: missing positional mappings {missing_mapping[:5]}")

        unavailable = sorted(set(feature_columns) & LIVE_UNAVAILABLE_FEATURES)
        if unavailable:
            errors.append(f"{key}: live-unavailable meta features {unavailable}")

        positional_required = [
            feat
            for feat in _model_raw_selected_features(meta_models.get(key))
            if _looks_positional_feature(feat)
        ]
        missing_required = sorted(
            set(positional_required) - set(str(k) for k in mapping)
        )
        if missing_required:
            errors.append(
                f"{key}: selected positional features lack mapping "
                f"{missing_required[:5]}"
            )

        model_contract_cols = _model_meta_feature_columns(meta_models.get(key))
        if model_contract_cols and model_contract_cols != feature_columns:
            errors.append(
                f"{key}: loaded model meta feature columns differ from artifact"
            )

    if errors:
        message = "Meta feature contract validation failed: " + "; ".join(errors)
        if strict:
            raise ValueError(message)
        tprint(f"[MetaFeatureContract] WARNING: {message}")
        return False
    return True


def validate_live_feature_contract(
    model_bundle: Dict[str, Any],
    *,
    strict: bool = True,
) -> bool:
    """Validate that active runtime models do not require unavailable targets.

    Training may persist diagnostic or tree-search feature lists that include
    target-derived fields. Those are acceptable as artifacts, but live
    inference must not open positions with an active model that directly
    consumes them.
    """
    active_features: Set[str] = set()
    ridge_sizer = (
        model_bundle.get("ridge_sizer") if isinstance(model_bundle, dict) else None
    )
    for attr in ("model_names_", "model_names_ridge_"):
        vals = getattr(ridge_sizer, attr, None)
        if not vals:
            continue
        vals_s = [str(v) for v in vals]
        unavailable = sorted(set(vals_s) & LIVE_UNAVAILABLE_FEATURES)
        if unavailable:
            tprint(
                "[FeatureContract] Ignoring legacy ridge_sizer feature list for "
                f"live contract because it contains target-derived fields: {unavailable}"
            )
            continue
        active_features.update(vals_s)

    bundle = (
        model_bundle.get("bundle", model_bundle)
        if isinstance(model_bundle, dict)
        else {}
    )
    alpha_models = bundle.get("alpha_models", {}) if isinstance(bundle, dict) else {}
    for value in alpha_models.values():
        if not isinstance(value, dict):
            continue
        if "feat_cols" in value:
            active_features.update(str(v) for v in value.get("feat_cols", []) or [])
            continue
        for model_info in value.values():
            if isinstance(model_info, dict):
                active_features.update(
                    str(v) for v in model_info.get("feat_cols", []) or []
                )

    for meta in _meta_models(model_bundle).values():
        meta_cols = _model_meta_feature_columns(meta)
        if meta_cols:
            active_features.update(meta_cols)

    ridge_weights = bundle.get("ridge_weights", {}) if isinstance(bundle, dict) else {}
    if isinstance(ridge_weights, dict):
        weight_map = ridge_weights.get("weights", {}) or {}
        if isinstance(weight_map, dict):
            for key in weight_map.keys():
                if not isinstance(key, str):
                    continue
                for prefix in (
                    "long_mr_",
                    "short_mr_",
                    "long_tf_",
                    "short_tf_",
                ):
                    if key.startswith(prefix):
                        active_features.add(key[len(prefix) :])
                        break
        params_per_bucket = ridge_weights.get("params_per_bucket", {}) or {}
        if isinstance(params_per_bucket, dict):
            for bucket_cfg in params_per_bucket.values():
                if isinstance(bucket_cfg, dict):
                    active_features.update(
                        str(v) for v in bucket_cfg.get("feature_names", []) or []
                    )

    unavailable = sorted(active_features & LIVE_UNAVAILABLE_FEATURES)
    if unavailable:
        message = (
            "Active live model artifacts include target-derived/unavailable fields: "
            f"{unavailable}. Retrain without these features or explicitly replace "
            "the active model before deployment."
        )
        if strict:
            raise ValueError(message)
        tprint(f"[FeatureContract] WARNING: {message}")
        return False
    return True


def validate_deployment_model_coverage(
    model_bundle: Dict[str, Any],
    accepted_strategies: Optional[Set[str]],
    *,
    strict: bool = True,
) -> bool:
    """Require every deployable strategy to have the full inference model chain.

    The runtime must not silently fall back from meta-model inference to
    alpha-only decisions for accepted deployment strategies. This validator
    checks the loaded alpha/base model, a matching meta model, and the position
    sizer/policy artifacts before the inference loop is allowed to run.
    """
    alpha_by_strategy = _alpha_strategy_index(model_bundle)
    selected_alpha = {
        sid: info
        for sid, info in alpha_by_strategy.items()
        if strategy_id_matches(sid, accepted_strategies)
    }
    meta_keys = _meta_model_keys(model_bundle)
    bundle = _bundle_payload(model_bundle)
    ridge_weights = bundle.get("ridge_weights", {}) if isinstance(bundle, dict) else {}
    ridge_sizer = (
        model_bundle.get("ridge_sizer") if isinstance(model_bundle, dict) else None
    )
    bucket_params = (
        model_bundle.get("bucket_params", {}) if isinstance(model_bundle, dict) else {}
    )
    ridge_params_per_bucket = (
        ridge_weights.get("params_per_bucket", {})
        if isinstance(ridge_weights, dict)
        else {}
    )

    errors: List[str] = []
    if accepted_strategies and not selected_alpha:
        errors.append(
            "no loaded alpha/base model matches the accepted deployment strategies"
        )
    if not alpha_by_strategy:
        errors.append("no alpha/base models loaded")
    if not meta_keys:
        errors.append("no meta models loaded")
    if ridge_sizer is None and not ridge_weights:
        errors.append("no position sizer or ridge weights loaded")
    if selected_alpha and not bucket_params and not ridge_params_per_bucket:
        errors.append("no policy/sizer bucket params loaded")

    for sid, info in selected_alpha.items():
        if info.get("model") is None:
            errors.append(f"{sid}: alpha/base model object missing")
        if not info.get("feat_cols"):
            errors.append(f"{sid}: alpha/base feature contract missing")
        if not (_candidate_meta_keys(sid) & meta_keys):
            errors.append(f"{sid}: matching meta model missing")

    if selected_alpha and (bucket_params or ridge_params_per_bucket):
        bucket_sources = []
        if isinstance(bucket_params, dict):
            bucket_sources.append(bucket_params.get("buckets", bucket_params))
        if isinstance(ridge_params_per_bucket, dict):
            bucket_sources.append(ridge_params_per_bucket)
        for sid in selected_alpha:
            core = strategy_core_id(sid)
            if not any(
                isinstance(buckets, dict)
                and (
                    core in buckets
                    or sid in buckets
                    or f"{strategy_side(sid)}_{core}" in buckets
                )
                for buckets in bucket_sources
            ):
                errors.append(f"{sid}: policy/sizer bucket params missing")

    if errors:
        message = "Deployment model coverage failed: " + "; ".join(errors)
        if strict:
            raise ValueError(message)
        tprint(f"[ModelCoverage] WARNING: {message}")
        return False
    return True


def validate_required_feature_frames(
    features: Dict[str, Any],
    required_feature_keys: Optional[Iterable[str]],
    *,
    symbols: Optional[Iterable[str]] = None,
    strict: bool = True,
) -> bool:
    """Require loaded/generated features to cover all active model contracts."""
    required = {str(key) for key in (required_feature_keys or set()) if str(key)}
    if not required:
        return True
    feature_map = features if isinstance(features, dict) else {}
    missing_keys = sorted(key for key in required if key not in feature_map)
    invalid_keys: List[str] = []
    missing_symbol_keys: Dict[str, List[str]] = {}
    symbol_list = [str(sym) for sym in (symbols or []) if str(sym)]

    for key in sorted(required - set(missing_keys)):
        value = feature_map.get(key)
        if isinstance(value, pd.Series):
            if value.dropna().empty:
                invalid_keys.append(key)
            continue
        if not isinstance(value, pd.DataFrame) or value.empty:
            invalid_keys.append(key)
            continue
        if symbol_list:
            missing_symbols = [sym for sym in symbol_list if sym not in value.columns]
            if missing_symbols:
                missing_symbol_keys[key] = missing_symbols[:10]

    if missing_keys or invalid_keys or missing_symbol_keys:
        parts = []
        if missing_keys:
            parts.append(f"missing_keys={missing_keys[:30]}")
        if invalid_keys:
            parts.append(f"invalid_or_empty={invalid_keys[:30]}")
        if missing_symbol_keys:
            sample = {key: vals for key, vals in list(missing_symbol_keys.items())[:10]}
            parts.append(f"missing_symbols={sample}")
        message = "Required inference features unavailable: " + "; ".join(parts)
        if strict:
            raise ValueError(message)
        tprint(f"[FeatureContract] WARNING: {message}")
        return False
    return True
