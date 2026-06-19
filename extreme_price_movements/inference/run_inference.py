"""
Main Entry Point for Inference.

This module provides the main entry point for running inference:
- Load models and config
- Run inference loop
- Support --live and --shadow modes
"""

# Fix for Numba threading on M1 Mac
import os

# Add Homebrew lib path for TBB on M1 Mac
homebrew_lib = "/opt/homebrew/lib"
if os.path.exists(homebrew_lib):
    os.environ["LIBRARY_PATH"] = homebrew_lib + ":" + os.environ.get("LIBRARY_PATH", "")


def _configure_numba_threading_layer() -> None:
    """Resolve a safe Numba threading layer at process startup, not import time."""
    _preferred_numba_layers = ("tbb", "omp", "workqueue")
    os.environ.pop("NUMBA_THREADING_LAYER", None)
    for _layer in _preferred_numba_layers:
        try:
            import importlib

            from numba import config as _numba_config

            os.environ["NUMBA_THREADING_LAYER"] = _layer
            _numba_config.THREADING_LAYER = _layer
            _threading_mod = importlib.import_module("numba.np.ufunc.parallel")
            _threading_mod._launch_threads()
            tprint(f"Using Numba threading layer: {_layer}")
            return
        except Exception as e:
            tprint(f"Threading layer {_layer} failed: {e}")
            os.environ.pop("NUMBA_THREADING_LAYER", None)
    os.environ["NUMBA_THREADING_LAYER"] = "workqueue"
    try:
        from numba import config as _numba_config

        _numba_config.THREADING_LAYER = "workqueue"
    except Exception:
        pass
    tprint("Falling back to Numba threading layer: workqueue")


import argparse
import hashlib
import inspect
import json
import os
import re
import resource
import sys
import threading
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from extreme_price_movements import hf_data_loader
from extreme_price_movements.feature_transforms import CausalFeatureTransformer
from extreme_price_movements.inference.candidate_selector import (
    build_latest_prepared_feature_frames,
    build_strategy_candidate_masks,
    select_candidates,
)
from extreme_price_movements.inference.config import (
    DEFAULT_EXECUTION_ACCOUNT,
    DEFAULT_LIVE_QUOTE_CURRENCY,
    DEFAULT_MARGIN_MODE,
    DEFAULT_MARKET_MODE,
    get_candidate_thresholds,
    get_inference_defaults,
    get_runtime_cfg,
    load_full_state,
    load_inference_config,
    resolve_inference_universes,
)
from extreme_price_movements.data_store import (
    _load_local_env_if_present,
    _resolve_perp_symbol,
)
from extreme_price_movements.inference.daily_reporter import DailyDeploymentReporter
from extreme_price_movements.inference.dynamic_strategy_performance import (
    StrategyPerformanceMonitor,
    meta_head_hash,
)
from extreme_price_movements.inference.data_fetcher import (
    DataFetcher,
    classify_api_error,
    fetch_and_build_panel,
    fetch_latest_ohlcv,
    make_exchange,
)
from extreme_price_movements.inference.feature_generator import (
    _compute_policy_barrier_pct,
    _coerce_feature_source_run_ids,
    _feature_runtime_cfg_hash,
    _hash_values,
    _is_live_source_derived_feature_key,
    _meta_model_derived_raw_dependencies,
    _merge_missing_feature_dicts,
    _required_tail_warmup_hours,
    compute_selector_features,
    generate_features,
    get_features_for_candidates,
    get_inference_required_feature_keys,
    get_market_data,
    is_model_derived_feature_key,
    live_model_feature_store_strict,
    load_or_compute_features,
    prewarm_selected_model_feature_cache_for_live,
    raw_required_feature_keys,
)
from extreme_price_movements.inference.feature_layer_debug import (
    dump_live_feature_layers,
    feature_layer_debug_enabled,
)
from extreme_price_movements.inference.feature_parity import (
    validate_required_source_panels,
)
from extreme_price_movements.inference.google_sheets_exporter import (
    GoogleSheetsTradeExporter,
)
from extreme_price_movements.inference.feature_layer_debug import (
    update_live_feature_layer_rank_summary,
)
from extreme_price_movements.inference.liquidity_precheck import (
    compute_price_gap_rank_penalty,
    evaluate_orderbook_liquidity,
    fetch_ticker_snapshot,
    marketable_limit_price,
)
from extreme_price_movements.inference.model_orchestrator import (
    DELETED_MODEL_FEATURE_KEYS,
    ModelOrchestrator,
    _effective_alpha_feature_contract,
    _effective_selected_feature_contract,
    _meta_live_unavailable_neutral_default,
)
from extreme_price_movements.inference.parity import (
    _policy_artifact_bases,
    calibrated_score_and_threshold,
    load_strategy_asset_exclusion_filter,
    resolve_deployment_strategy_filter,
    strategy_core_id,
    strategy_id_matches,
    strategy_side,
    validate_calibration_artifacts,
    validate_deployment_model_coverage,
    validate_live_feature_contract,
    validate_meta_feature_contract_artifact,
    validate_required_feature_frames,
)
from extreme_price_movements.inference.policy_rank_reference import (
    PolicyRankReferenceStore,
    apply_policy_rank_percentile_gate,
)
from extreme_price_movements.inference.portfolio_policy import (
    PortfolioPolicyConfig,
    compute_rank_based_position_size,
    load_portfolio_policy_config,
    validate_portfolio_strategy_contract,
)
from extreme_price_movements.inference.training_live_parity_contract import (
    load_training_live_parity_contract,
    validate_training_live_parity_contract,
)
from extreme_price_movements.inference.prediction_ledger import PredictionLedger
from extreme_price_movements.drift_monitoring import write_live_drift_recap
try:
    from extreme_price_movements.lgbm_pipeline import LGBM_INTERNAL_METRIC_FEATURE_NAMES
except Exception:  # pragma: no cover - keep live runner importable with old envs
    LGBM_INTERNAL_METRIC_FEATURE_NAMES = ()
from extreme_price_movements.inference.safety_switches import (
    MarketKillSwitch,
    StrategyKillSwitch,
)
from extreme_price_movements.inference.simple_policy_stop import (
    SimplePolicyStopParamsError,
    compute_simple_policy_stop_decision,
    load_simple_policy_stop_params_by_strategy,
)
from extreme_price_movements.inference.symbol_mapping import (
    normalise_symbol,
    symbol_base,
)
from extreme_price_movements.path_utils import mode_file_candidates
from extreme_price_movements.inference.trade_executor import TradeExecutor
from extreme_price_movements.inference.trade_logger import (
    TradeLogger,
    log_trade_decision,
)
from extreme_price_movements.portfolio_manager import PortfolioManager
from extreme_price_movements.utils import tprint


def _get_features_for_candidates_at_ts(
    feats: Dict[str, pd.DataFrame],
    candidates: List[str],
    *,
    ts: Any,
) -> pd.DataFrame:
    """Call the feature slicer with timestamp support when the implementation has it."""
    try:
        sig = inspect.signature(get_features_for_candidates)
        params = sig.parameters
        accepts_ts = "ts" in params or any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
        )
    except (TypeError, ValueError):
        accepts_ts = True
    if accepts_ts:
        return get_features_for_candidates(feats, candidates, ts=ts)
    return get_features_for_candidates(feats, candidates)


_FEATURE_COMPUTE_LOCK = threading.RLock()
_CANDIDATE_FEATURE_CYCLE_CACHE: Dict[str, Dict[str, Any]] = {}
BASE_TO_META_TOP_FRAC = 0.40
LIVE_TEST_RANK_THRESHOLD = 0.90
LIVE_TEST_THRESHOLD_RELAXATION = 0.06
AUCTION_EV_MIN_NET_RETURN = float(os.getenv("EPM_AUCTION_EV_MIN_NET_RETURN", "0.002"))
AUCTION_EV_MAX_NET_RETURN = float(os.getenv("EPM_AUCTION_EV_MAX_NET_RETURN", "0.006"))


def _raise_if_policy_export_invalid(data_root: str, run_id: str) -> None:
    base = Path(data_root) / "artifacts" / run_id
    for marker_path in mode_file_candidates(
        base / "simple_policy_optimiser" / "policy_export_invalid.json"
    ):
        if not marker_path.exists():
            continue
        reason = ""
        try:
            payload = json.loads(marker_path.read_text())
            if isinstance(payload, dict):
                reason = str(payload.get("reason") or "")
        except Exception:
            reason = ""
        suffix = f": {reason}" if reason else ""
        raise RuntimeError(
            f"Refusing to load policy artifacts for run_id={run_id}; "
            f"strict simple_policy_optimiser export is marked invalid at "
            f"{marker_path}{suffix}"
        )
AUCTION_EV_MIN_HIT_RATE = float(os.getenv("EPM_AUCTION_EV_MIN_HIT_RATE", "0.55"))
LOSING_TRADE_COOLDOWN_HOURS = 12.0
_HISTORICAL_SCORE_RANK_CACHE: Dict[tuple[str, str, str, str], np.ndarray] = {}
_META_HIT_RATE_CALIBRATION_CACHE: Dict[tuple[str, str], Dict[str, Any]] = {}
_STRATEGY_EV_CALIBRATION_CACHE: Dict[tuple[str, str], Dict[str, Any]] = {}


def _finite_positive_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if np.isfinite(out) and out > 0.0 else float("nan")


def _fetch_live_closeable_price(
    symbol: str,
    side: str,
    executor: TradeExecutor,
) -> Dict[str, Any]:
    """Return the live top-of-book price at which the position can be closed.

    This mirrors the executable stop reference for Kraken Futures: bid for long
    exits, ask for short exits.
    """
    exchange = getattr(executor, "exchange", None)
    side_l = str(side or "").lower()
    if exchange is None or side_l not in {"long", "short"}:
        return {"price": np.nan, "source": "unavailable"}

    ticker: Dict[str, Any] = {}
    try:
        ticker_raw = exchange.fetch_ticker(symbol)
        if isinstance(ticker_raw, dict):
            ticker = ticker_raw
    except Exception as exc:
        ticker = {"_ticker_error": f"{classify_api_error(exc)}: {exc}"}

    touch_key = "bid" if side_l == "long" else "ask"
    price = _finite_positive_float(ticker.get(touch_key))
    if np.isfinite(price):
        return {
            "price": float(price),
            "source": f"ticker_{touch_key}",
            "bid": _finite_positive_float(ticker.get("bid")),
            "ask": _finite_positive_float(ticker.get("ask")),
            "last": _finite_positive_float(ticker.get("last")),
            "timestamp": ticker.get("timestamp"),
        }

    try:
        orderbook = exchange.fetch_order_book(symbol)
        levels_key = "bids" if side_l == "long" else "asks"
        levels = orderbook.get(levels_key) if isinstance(orderbook, dict) else None
        if isinstance(levels, list) and levels:
            price = _finite_positive_float(levels[0][0])
            if np.isfinite(price):
                return {
                    "price": float(price),
                    "source": f"orderbook_best_{touch_key}",
                    "bid": (
                        float(price)
                        if touch_key == "bid"
                        else _finite_positive_float(ticker.get("bid"))
                    ),
                    "ask": (
                        float(price)
                        if touch_key == "ask"
                        else _finite_positive_float(ticker.get("ask"))
                    ),
                    "last": _finite_positive_float(ticker.get("last")),
                    "timestamp": orderbook.get("timestamp"),
                    "ticker_error": ticker.get("_ticker_error"),
                }
    except Exception as exc:
        return {
            "price": np.nan,
            "source": "unavailable",
            "ticker_error": ticker.get("_ticker_error"),
            "orderbook_error": f"{classify_api_error(exc)}: {exc}",
        }

    return {
        "price": np.nan,
        "source": "unavailable",
        "bid": _finite_positive_float(ticker.get("bid")),
        "ask": _finite_positive_float(ticker.get("ask")),
        "last": _finite_positive_float(ticker.get("last")),
        "timestamp": ticker.get("timestamp"),
        "ticker_error": ticker.get("_ticker_error"),
    }


def _executable_stop_breached(side: str, stop_price: float, price: float) -> bool:
    """Return whether the executable close-side price has crossed the stop."""
    stop = _finite_positive_float(stop_price)
    current = _finite_positive_float(price)
    if not (np.isfinite(stop) and np.isfinite(current)):
        return False
    side_l = str(side or "").strip().lower()
    if side_l == "long":
        return current <= stop
    if side_l == "short":
        return current >= stop
    return False


def _position_policy_entry_price(position_state: Mapping[str, Any]) -> tuple[float, str]:
    """Return the fill reference used for live policy MFE/MAE accounting."""
    for key in (
        "realized_entry_price",
        "actual_entry_price",
        "entry_price",
        "policy_entry_price",
        "theoretical_entry_price",
        "ohlcv_entry_price",
        "signal_price",
        "expected_entry_price",
        "decision_mid",
    ):
        value = _finite_positive_float(position_state.get(key))
        if np.isfinite(value):
            return float(value), key
    return float("nan"), "unavailable"


def _shadow_execution_realism_enabled() -> bool:
    raw = str(os.getenv("EPM_LIVE_EXECUTION_REALISM_SHADOW", "1")).strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _shadow_side_return(side: str, price: float, entry_price: float) -> float:
    if not (np.isfinite(price) and np.isfinite(entry_price) and entry_price > 0.0):
        return float("nan")
    if str(side).lower() == "short":
        return float((entry_price - price) / max(abs(entry_price), 1e-12))
    return float((price - entry_price) / max(abs(entry_price), 1e-12))


def _shadow_bps_delta(side: str, observed: Any, reference: Any) -> float:
    obs = _finite_positive_float(observed)
    ref = _finite_positive_float(reference)
    if not (np.isfinite(obs) and np.isfinite(ref) and ref > 0.0):
        return float("nan")
    sign = -1.0 if str(side).lower() == "short" else 1.0
    return float(sign * (obs / max(ref, 1e-12) - 1.0) * 10000.0)


def _ensure_simple_policy_shadow_state(
    position_state: Dict[str, Any],
    *,
    symbol: str,
    side: str,
    policy_entry_price: float,
    policy_entry_price_source: str,
    realized_entry_price: float,
    stop_price: float,
    stop_reason: str,
    params: Mapping[str, Any],
) -> Dict[str, Any]:
    shadow = position_state.get("simple_policy_shadow")
    if not isinstance(shadow, dict):
        shadow = {
            "schema": "simple_policy_execution_shadow_v1",
            "enabled": True,
            "status": "open",
            "symbol": symbol,
            "side": side,
            "strategy_id": position_state.get("strategy_id")
            or position_state.get("bucket_key"),
            "bucket_key": position_state.get("bucket_key"),
            "entry_time": _json_safe_audit_value(position_state.get("entry_time")),
            "policy_entry_price": (
                float(policy_entry_price) if np.isfinite(policy_entry_price) else None
            ),
            "policy_entry_price_source": policy_entry_price_source,
            "realized_entry_price": (
                float(realized_entry_price)
                if np.isfinite(realized_entry_price)
                else None
            ),
            "entry_gap_bps": _shadow_bps_delta(
                side, realized_entry_price, policy_entry_price
            ),
            "initial_shadow_stop_price": (
                float(stop_price) if np.isfinite(stop_price) else None
            ),
            "shadow_stop_price": (
                float(stop_price) if np.isfinite(stop_price) else None
            ),
            "shadow_stop_reason": stop_reason,
            "params_source": position_state.get("stop_policy_params_source")
            or params.get("params_source"),
            "params_hash": position_state.get("stop_policy_params_hash")
            or params.get("params_hash"),
            "params_schema": position_state.get("stop_policy_schema")
            or params.get("params_schema")
            or "simple_policy_v1",
            "events": [],
        }
        position_state["simple_policy_shadow"] = shadow
    return shadow


def _append_simple_policy_shadow_event(
    shadow: Dict[str, Any],
    event: str,
    **payload: Any,
) -> None:
    events = shadow.setdefault("events", [])
    if not isinstance(events, list):
        events = []
        shadow["events"] = events
    if len(events) >= 250:
        del events[: len(events) - 249]
    clean = {
        str(key): _json_safe_audit_value(value)
        for key, value in payload.items()
    }
    events.append(
        {
            "ts": pd.Timestamp.now(tz="UTC").isoformat(),
            "event": str(event),
            **clean,
        }
    )


def _strategy_mask_count_diagnostics(
    strategy_candidate_masks: Mapping[str, List[str]] | None,
    lgbm_strategy_mask_rows: Mapping[str, Mapping[str, Any]] | None,
    universe_symbols: List[str] | set[str] | tuple[str, ...] | None,
) -> Dict[str, Dict[str, Any]]:
    """Summarize pass/fail counts for each deployment strategy mask."""
    masks = strategy_candidate_masks or {}
    universe = {str(sym) for sym in (universe_symbols or []) if str(sym)}
    universe_count = len(universe)
    diagnostics: Dict[str, Dict[str, Any]] = {}
    for strategy_id, passed_symbols in masks.items():
        sid = str(strategy_id)
        passed = {str(sym) for sym in (passed_symbols or []) if str(sym)}
        row = (lgbm_strategy_mask_rows or {}).get(sid, {}) or {}
        side = str(row.get("trade_side") or row.get("side") or "").lower()
        if universe:
            pass_count = len(passed & universe)
            fail_count = max(universe_count - pass_count, 0)
        else:
            pass_count = len(passed)
            fail_count = None
        diagnostics[sid] = {
            "strategy_core_id": strategy_core_id(sid),
            "side": side or None,
            "universe_count": int(universe_count),
            "pass_count": int(pass_count),
            "fail_count": None if fail_count is None else int(fail_count),
            "pass_rate": (
                float(pass_count / universe_count) if universe_count > 0 else None
            ),
        }
    return diagnostics


class _StageTimer:
    """Log live-loop stage latencies without affecting trading decisions."""

    def __init__(self, label: str):
        self.label = label
        self.start = time.perf_counter()
        self.last = self.start

    def mark(self, stage: str) -> None:
        now = time.perf_counter()
        try:
            rss_raw = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            rss_mb = rss_raw / (1024.0 * 1024.0) if rss_raw > 10_000_000 else rss_raw / 1024.0
        except Exception:
            rss_mb = float("nan")
        tprint(
            f"[Timing] {self.label}.{stage}: "
            f"stage={now - self.last:.3f}s total={now - self.start:.3f}s "
            f"rss={rss_mb:.1f}MB"
        )
        self.last = now


def _is_live_test_mode(mode_or_executor: Any) -> bool:
    mode = getattr(mode_or_executor, "mode", mode_or_executor)
    return str(mode or "").strip().lower() in {"live-test", "live_test", "livetest"}


def _apply_live_test_threshold_relaxation(
    policy: PortfolioPolicyConfig,
    *,
    live_test_mode: bool,
) -> PortfolioPolicyConfig:
    """Relax rank gates in live-test only to speed end-to-end order-path tests."""
    if not live_test_mode:
        return policy
    delta = float(LIVE_TEST_THRESHOLD_RELAXATION)
    relaxed_initial = float(np.clip(policy.initial_rank_threshold - delta, 0.0, 1.0))
    relaxed_floor = float(
        np.clip(policy.initial_rank_threshold_floor - delta, 0.0, 1.0)
    )
    relaxed_viability_margin = float(
        max(float(policy.threshold_viability_margin) - delta, 0.0)
    )
    if (
        relaxed_initial == float(policy.initial_rank_threshold)
        and relaxed_floor == float(policy.initial_rank_threshold_floor)
        and relaxed_viability_margin == float(policy.threshold_viability_margin)
    ):
        return policy
    tprint(
        "LIVE-TEST threshold relaxation active: "
        f"initial_rank_threshold {policy.initial_rank_threshold:.4f}->{relaxed_initial:.4f}, "
        f"floor {policy.initial_rank_threshold_floor:.4f}->{relaxed_floor:.4f}, "
        f"viability_margin {policy.threshold_viability_margin:.4f}->{relaxed_viability_margin:.4f}"
    )
    return replace(
        policy,
        initial_rank_threshold=relaxed_initial,
        initial_rank_threshold_floor=relaxed_floor,
        threshold_viability_margin=relaxed_viability_margin,
    )


def _order_identifier(order_payload: Any) -> str:
    if isinstance(order_payload, dict):
        raw = order_payload.get("id") or order_payload.get("clientOrderId")
        return str(raw) if raw is not None else ""
    return ""


def _load_normalized_threshold_map(
    data_root: str, run_id: str
) -> Dict[str, Dict[str, Any]]:
    _raise_if_policy_export_invalid(data_root, run_id)
    rows_out: Dict[str, Dict[str, Any]] = {}
    strategy_paths = [
        path
        for base in _policy_artifact_bases(data_root, run_id)
        for path in (
            base / "simple_policy_optimiser" / "deployment" / "best_policy_params_perps.json",
            base / "simple_policy_optimiser" / "deployment" / "best_policy_params.json",
            base / "policy_params" / "strategy_for_inference.json",
            base / "strategy_for_inference.json",
        )
    ]
    strategy_paths = [
        candidate for path in strategy_paths for candidate in mode_file_candidates(path)
    ]
    for strategy_path in strategy_paths:
        if not strategy_path.exists():
            continue
        try:
            payload = json.loads(strategy_path.read_text())
            strategies = (
                payload.get("strategies", []) if isinstance(payload, dict) else []
            )
            if not isinstance(strategies, list):
                continue
            loaded = 0
            for row in strategies:
                if not isinstance(row, dict) or row.get("selected") is False:
                    continue
                sid = str(
                    row.get("strategy_for_inference")
                    or row.get("strategy_id")
                    or row.get("canonical_strategy_id")
                    or ""
                )
                if not sid:
                    continue
                threshold = _deployment_rank_threshold_from_strategy_row(row)
                nrow = {
                    "threshold_space": "rank_percentile",
                    "normalized_threshold": threshold,
                    "deployment_rank_threshold": threshold,
                    "viability_margin": 0.0,
                    "threshold_source": str(strategy_path),
                    "threshold_scope": "per_strategy_prediction_rank_only",
                    "avg_trades_per_day_at_top_1pct": float(
                        row.get("avg_trades_per_day_at_top_1pct", 0.0) or 0.0
                    ),
                    "avg_holding_time_hours": float(
                        row.get("avg_holding_time_hours", 0.0) or 0.0
                    ),
                }
                aliases = {
                    sid,
                    str(row.get("strategy_id", "") or ""),
                    str(row.get("canonical_strategy_id", "") or ""),
                    strategy_core_id(sid),
                }
                side = str(row.get("side", "") or "").lower()
                core = strategy_core_id(sid)
                if side in {"long", "short"} and core:
                    aliases.add(f"{side}_{core}")
                for alias in aliases:
                    if alias:
                        # The simple_policy_optimiser deployment export is the
                        # source of truth for deployable rank gates. The
                        # strategy_for_inference fallbacks remain only for
                        # pre-export artifacts that do not yet have a deployment
                        # policy file.
                        rows_out[str(alias)] = dict(nrow)
                loaded += 1
            tprint(f"Loaded {loaded} deployment rank thresholds from {strategy_path}")
            break
        except Exception as exc:
            tprint(
                f"Could not load deployment rank thresholds from {strategy_path}: {exc}"
            )
    return rows_out


def _load_policy_selection_rules(data_root: str, run_id: str) -> Dict[str, Any]:
    _raise_if_policy_export_invalid(data_root, run_id)
    for strategy_path in [
        candidate
        for base in _policy_artifact_bases(data_root, run_id)
        for path in (
            base / "policy_params" / "strategy_for_inference.json",
            base / "strategy_for_inference.json",
        )
        for candidate in mode_file_candidates(path)
    ]:
        if not strategy_path.exists():
            continue
        try:
            payload = json.loads(strategy_path.read_text())
        except Exception as exc:
            tprint(f"Could not load policy selection rules from {strategy_path}: {exc}")
            continue
        if not isinstance(payload, dict):
            continue
        rules = payload.get("selection_rules", {})
        if isinstance(rules, dict):
            return dict(rules)
    return {}


_LIVE_SPREAD_BASELINE_CACHE: Dict[str, Dict[str, Any]] = {}


def _live_spread_baseline_candidates(data_root: str) -> List[Path]:
    paths: List[Path] = []
    explicit = str(os.environ.get("EPM_SIMPLE_POLICY_SPREAD_BASELINE_PATH", "")).strip()
    if explicit:
        paths.append(Path(explicit))
    root = Path(data_root or "data")
    paths.extend(
        [
            root
            / "exchanges"
            / "krakenfutures"
            / "spread_model"
            / "per_asset_spread_baseline_latest.csv",
            root
            / "exchanges"
            / "krakenfutures"
            / "spread_model"
            / "per_asset_spread_baseline_latest.json",
            root / "spread_model" / "per_asset_spread_baseline_latest.csv",
            root / "spread_model" / "per_asset_spread_baseline_latest.json",
        ]
    )
    out: List[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        out.append(path)
    return out


def _spread_symbol_aliases(symbol: Any) -> set[str]:
    raw = str(symbol or "").strip()
    aliases = {raw}
    if raw:
        aliases.add(raw.upper())
        aliases.add(normalise_symbol(raw))
        aliases.add(normalise_symbol(raw.replace(":USD", "")))
        aliases.add(normalise_symbol(raw.replace(":USDT", "")))
    return {alias for alias in aliases if alias}


def _spread_fallback_average(
    values: Sequence[float],
    weights: Optional[Sequence[float]] = None,
) -> float:
    vals = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(vals) & (vals >= 0.0)
    vals = vals[valid]
    weight_arr: Optional[np.ndarray] = None
    if weights is not None:
        raw_weights = np.asarray(weights, dtype=np.float64)
        if raw_weights.shape == valid.shape:
            weight_arr = raw_weights[valid]
            weight_arr = np.where(
                np.isfinite(weight_arr) & (weight_arr > 0.0), weight_arr, 0.0
            )

    def _average(local_vals: np.ndarray, local_weights: Optional[np.ndarray]) -> float:
        if local_weights is not None and float(local_weights.sum()) > 0.0:
            return float(np.average(local_vals, weights=local_weights))
        return float(np.nanmean(local_vals))

    if vals.size == 0:
        return float("nan")
    if vals.size >= 20:
        cap = float(np.nanquantile(vals, 0.75))
        keep = vals <= cap
        trimmed = vals[keep]
        if trimmed.size >= 20:
            return _average(
                trimmed,
                weight_arr[keep] if weight_arr is not None else None,
            )
    return _average(vals, weight_arr)


def _load_live_spread_baseline(data_root: str) -> Dict[str, Any]:
    paths = _live_spread_baseline_candidates(data_root)
    cache_key = "|".join(str(path) for path in paths)
    cached = _LIVE_SPREAD_BASELINE_CACHE.get(cache_key)
    if cached is not None:
        return cached
    loaded: Dict[str, Any] = {}

    def _finish(
        *,
        path: Path,
        rows: Sequence[Mapping[str, Any]],
        payload_fallback: Any = None,
    ) -> Dict[str, Any]:
        by_symbol: Dict[str, float] = {}
        values: List[float] = []
        weights: List[float] = []
        for row in rows:
            symbol = str(row.get("symbol") or "").strip()
            if not symbol:
                continue
            value = row.get(
                "average_spread_bps",
                row.get("baseline_spread_bps", row.get("spread_bps")),
            )
            try:
                spread = float(value)
            except Exception:
                continue
            if not np.isfinite(spread) or spread < 0.0:
                continue
            values.append(spread)
            try:
                weight = max(0.0, float(row.get("rows", 0.0)))
            except Exception:
                weight = 0.0
            weights.append(weight)
            for alias in _spread_symbol_aliases(symbol):
                by_symbol[alias] = spread
        try:
            fallback = float(payload_fallback)
        except Exception:
            fallback = float("nan")
        if not np.isfinite(fallback) or fallback < 0.0:
            fallback = _spread_fallback_average(values, weights=weights)
        unique_symbols = {
            str(row.get("symbol") or "").strip()
            for row in rows
            if isinstance(row, Mapping) and str(row.get("symbol") or "").strip()
        }
        return {
            "source": str(path),
            "by_symbol": by_symbol,
            "effective_fallback_spread_bps": (
                max(0.0, float(fallback)) if np.isfinite(fallback) else None
            ),
            "symbol_count": int(len(unique_symbols)),
        }

    for path in paths:
        if not path.exists():
            continue
        try:
            if path.suffix.lower() == ".csv":
                frame = pd.read_csv(path)
                spread_col = next(
                    (
                        col
                        for col in (
                            "average_spread_bps",
                            "baseline_spread_bps",
                            "spread_bps",
                        )
                        if col in frame.columns
                    ),
                    None,
                )
                if "symbol" not in frame.columns or spread_col is None:
                    continue
                cols = ["symbol", spread_col]
                if "rows" in frame.columns:
                    cols.append("rows")
                rows = frame[cols].rename(
                    columns={spread_col: "average_spread_bps"}
                ).to_dict("records")
                loaded = _finish(path=path, rows=rows)
            else:
                payload = json.loads(path.read_text(encoding="utf-8"))
                rows = (
                    payload.get("per_asset_spread_baseline")
                    or payload.get("per_asset_average_spread")
                    or []
                )
                if not isinstance(rows, list):
                    continue
                loaded = _finish(
                    path=path,
                    rows=[row for row in rows if isinstance(row, Mapping)],
                    payload_fallback=payload.get(
                        "effective_fallback_spread_bps",
                        payload.get("global_average_spread_bps"),
                    ),
                )
            if loaded.get("by_symbol"):
                break
        except Exception as exc:
            tprint(f"Could not load live spread baseline from {path}: {exc}")
            continue
    _LIVE_SPREAD_BASELINE_CACHE[cache_key] = loaded
    return loaded


def _live_ev_haircut_spread_baseline_bps(
    *,
    symbol: Any,
    data_root: str,
    fallback_bps: Any,
) -> Tuple[float, str]:
    baseline = _load_live_spread_baseline(data_root)
    by_symbol = baseline.get("by_symbol") if isinstance(baseline, dict) else {}
    if isinstance(by_symbol, dict):
        for alias in _spread_symbol_aliases(symbol):
            value = by_symbol.get(alias)
            try:
                spread = float(value)
            except Exception:
                continue
            if np.isfinite(spread) and spread >= 0.0:
                return (
                    float(spread),
                    f"per_asset_spread_baseline.average_spread_bps:{baseline.get('source')}",
                )
    fallback = (
        baseline.get("effective_fallback_spread_bps")
        if isinstance(baseline, dict)
        else None
    )
    try:
        fallback_spread = float(fallback)
    except Exception:
        fallback_spread = float("nan")
    if np.isfinite(fallback_spread) and fallback_spread >= 0.0:
        return (
            float(fallback_spread),
            f"per_asset_spread_baseline.effective_fallback:{baseline.get('source')}",
        )
    try:
        policy_spread = float(fallback_bps)
    except Exception:
        policy_spread = 0.0
    policy_spread = (
        max(0.0, float(policy_spread)) if np.isfinite(policy_spread) else 0.0
    )
    return policy_spread, "portfolio_policy.ev_haircut_expected_spread_bps"


def _strategy_row_aliases(strategy_id: str, side: str = "") -> set[str]:
    sid = str(strategy_id or "")
    core = strategy_core_id(sid)
    aliases = {sid, core}
    side_l = str(side or "").lower()
    if side_l in {"long", "short"} and core:
        aliases.add(f"{side_l}_{core}")
    return {alias for alias in aliases if alias}


def _normalise_embedded_lgbm_mask_row(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    mask = row.get("lgbm_regime_mask")
    if not isinstance(mask, dict):
        mask = {
            key: row.get(key)
            for key in (
                "strategy_id",
                "trade_side",
                "side",
                "base_event_trigger",
                "canonical_key",
                "mask_params",
                "source_target",
                "source_horizon",
                "move_bucket",
                "candidate_bucket",
            )
            if row.get(key) is not None
        }
    if not isinstance(mask, dict) or not mask:
        return None

    sid = str(
        row.get("strategy_for_inference")
        or row.get("strategy_id")
        or row.get("canonical_strategy_id")
        or mask.get("strategy_id")
        or ""
    )
    if not sid:
        return None

    out = dict(mask)
    out["strategy_id"] = sid
    side = str(row.get("side") or out.get("trade_side") or out.get("side") or "")
    if side:
        out.setdefault("trade_side", side)
        out.setdefault("side", side)

    mask_params = dict(out.get("mask_params", {}) or {})
    canonical_key = str(
        out.get("base_event_trigger")
        or out.get("canonical_key")
        or mask_params.get("canonical_key")
        or ""
    )
    if not canonical_key:
        return None
    out["base_event_trigger"] = canonical_key
    out["canonical_key"] = canonical_key
    mask_params.setdefault("canonical_key", canonical_key)
    out["mask_params"] = mask_params
    return out


def _load_embedded_lgbm_strategy_mask_rows(
    data_root: str, run_id: str
) -> Dict[str, Dict[str, Any]]:
    _raise_if_policy_export_invalid(data_root, run_id)
    for strategy_path in [
        candidate
        for base in _policy_artifact_bases(data_root, run_id)
        for path in (
            base / "policy_params" / "strategy_for_inference.json",
            base / "strategy_for_inference.json",
        )
        for candidate in mode_file_candidates(path)
    ]:
        if not strategy_path.exists():
            continue
        try:
            payload = json.loads(strategy_path.read_text())
        except Exception as exc:
            tprint(
                f"Could not load embedded LGBM mask contracts from {strategy_path}: {exc}"
            )
            continue
        strategies = payload.get("strategies", []) if isinstance(payload, dict) else []
        if not isinstance(strategies, list):
            continue
        out: Dict[str, Dict[str, Any]] = {}
        for row in strategies:
            if not isinstance(row, dict) or row.get("selected") is False:
                continue
            mask_row = _normalise_embedded_lgbm_mask_row(row)
            if not mask_row:
                continue
            side = str(row.get("side") or mask_row.get("trade_side") or "")
            for alias in _strategy_row_aliases(str(mask_row["strategy_id"]), side):
                out[alias] = dict(mask_row)
        if out:
            _emit_structured_event(
                "LGBM_REGIME_MASK_HEALTH",
                {
                    "loaded_rows": int(len(out)),
                    "status": "loaded",
                    "source": str(strategy_path),
                    "source_type": "embedded_strategy_for_inference",
                },
            )
            return out
    return {}


def _load_selected_strategy_cores(data_root: str, run_id: str) -> set[str]:
    """Load selected deployment strategy cores from strategy_for_inference."""
    _raise_if_policy_export_invalid(data_root, run_id)
    selected: set[str] = set()
    for strategy_path in [
        candidate
        for base in _policy_artifact_bases(data_root, run_id)
        for path in (
            base / "policy_params" / "strategy_for_inference.json",
            base / "strategy_for_inference.json",
        )
        for candidate in mode_file_candidates(path)
    ]:
        if not strategy_path.exists():
            continue
        try:
            payload = json.loads(strategy_path.read_text())
        except Exception:
            continue
        strategies = payload.get("strategies", []) if isinstance(payload, dict) else []
        if not isinstance(strategies, list):
            continue
        for row in strategies:
            if not isinstance(row, dict) or row.get("selected") is False:
                continue
            sid = str(
                row.get("strategy_for_inference")
                or row.get("strategy_id")
                or row.get("canonical_strategy_id")
                or ""
            )
            core = strategy_core_id(sid)
            if core:
                selected.add(core)
        if selected:
            return selected
    return selected


def _resolve_portfolio_contract_strategy_filter(
    portfolio_policy: PortfolioPolicyConfig,
    fallback: Optional[set[str]],
) -> Optional[set[str]]:
    contracted = {str(sid) for sid in portfolio_policy.strategy_ids if str(sid)}
    if contracted:
        return contracted
    contracted_cores = {
        str(strategy_core_id(sid))
        for sid in portfolio_policy.strategy_cores
        if str(sid)
    }
    if contracted_cores:
        return contracted_cores
    return fallback


def _resolve_training_live_contract_strategy_filter(
    parity_contract: Optional[Dict[str, Any]],
    fallback: Optional[set[str]],
) -> Optional[set[str]]:
    contract = parity_contract if isinstance(parity_contract, dict) else {}
    strategy_contract = contract.get("strategy_contract") or {}
    contracted = {
        str(sid).strip()
        for sid in (strategy_contract.get("strategy_ids") or [])
        if str(sid).strip()
    }
    if contracted:
        return contracted
    contracted_cores = {
        str(strategy_core_id(sid)).strip()
        for sid in (strategy_contract.get("strategy_cores") or [])
        if str(sid).strip()
    }
    if contracted_cores:
        return contracted_cores
    return fallback


def _resolve_active_strategy_filter_for_policy(
    *,
    parity_contract: Optional[Dict[str, Any]],
    portfolio_policy: PortfolioPolicyConfig,
    policy_strategy_filter: Optional[set[str]],
    prefer_policy_contract: bool,
) -> Optional[set[str]]:
    """Resolve active deployment strategies for a model/policy artifact pair.

    A final-fit model artifact can be reused with a freshly optimized policy
    artifact. In that case the model-run training/live parity contract may
    still describe the previous deployed strategy subset, while the policy
    artifact intentionally declares the current portfolio strategy contract.
    Keep the portfolio contract strict, but do not let the stale parity
    strategy list downselect the policy artifact.
    """
    policy_filter = _resolve_portfolio_contract_strategy_filter(
        portfolio_policy,
        policy_strategy_filter,
    )
    if prefer_policy_contract and policy_filter:
        parity_filter = _resolve_training_live_contract_strategy_filter(
            parity_contract,
            None,
        )
        if parity_filter and set(parity_filter) != set(policy_filter):
            tprint(
                "Policy artifact strategy contract overrides training-live "
                "parity strategy filter: "
                f"policy={sorted(policy_filter)} parity={sorted(parity_filter)}"
            )
        return policy_filter
    return _resolve_training_live_contract_strategy_filter(
        parity_contract,
        policy_filter,
    )


def _load_lgbm_strategy_mask_rows(
    data_root: str, run_id: str, market_mode: str = "spot"
) -> Dict[str, Dict[str, Any]]:
    embedded = _load_embedded_lgbm_strategy_mask_rows(data_root, run_id)
    if embedded:
        tprint(
            "Loaded embedded LGBM strategy mask row(s) from strategy_for_inference: "
            f"aliases={len(embedded)}"
        )
        return embedded

    try:
        from extreme_price_movements.offline_optimisers.params_store import (
            load_inference_candidate_mask_params_per_bucket,
        )
    except Exception as exc:
        tprint(f"Could not import LGBM strategy mask loader: {exc}")
        return {}

    try:
        rows = load_inference_candidate_mask_params_per_bucket(
            top_n=99,
            ranking_metric="score_for_best_params",
            market_mode=market_mode,
        )
    except Exception as exc:
        tprint(f"Could not load LGBM strategy mask rows: {exc}")
        return {}

    selected_cores = _load_selected_strategy_cores(data_root, run_id)
    if selected_cores:
        before_count = len(rows)
        rows = [
            row
            for row in rows
            if isinstance(row, dict)
            and strategy_core_id(str(row.get("strategy_id", "") or ""))
            in selected_cores
        ]
        tprint(
            "Filtered fallback LGBM strategy mask rows to selected "
            f"strategy_for_inference cores: {before_count}->{len(rows)}"
        )

    out: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        sid = str(row.get("strategy_id", "") or "")
        if not sid:
            continue
        side = str(row.get("trade_side", row.get("side", "")) or "").lower()
        for alias in _strategy_row_aliases(sid, side):
            out[str(alias)] = dict(row)
    if not rows:
        _emit_structured_event(
            "LGBM_REGIME_MASK_HEALTH",
            {
                "loaded_rows": 0,
                "status": "missing_or_empty",
                "effect": "strategy_candidate_masks_disabled",
                "required_for": "pre_base_model_regime_gating",
                "source_type": "legacy_final_rule_registry",
                "market_mode": _normalise_market_mode(market_mode),
            },
        )
    else:
        _emit_structured_event(
            "LGBM_REGIME_MASK_HEALTH",
            {
                "loaded_rows": int(len(rows)),
                "alias_count": int(len(out)),
                "status": "loaded",
                "source_type": "legacy_final_rule_registry",
                "market_mode": _normalise_market_mode(market_mode),
            },
        )
    tprint(f"Loaded {len(rows)} LGBM strategy mask row(s) for inference gating")
    return out


def _validate_lgbm_strategy_mask_coverage(
    lgbm_strategy_mask_rows: Mapping[str, Mapping[str, Any]],
    accepted_strategies: Optional[set[str]],
    policy_selection_rules: Optional[Mapping[str, Any]] = None,
) -> None:
    """Fail closed when deployed LGBM strategies lack pre-base regime masks."""
    rules = policy_selection_rules or {}
    requires_mask_contract = bool(
        rules.get("requires_lgbm_regime_mask_contract", True)
    )
    selected = {
        strategy_core_id(str(sid))
        for sid in (accepted_strategies or set())
        if str(sid).strip()
    }
    if not selected:
        return
    rows = [
        row
        for row in (lgbm_strategy_mask_rows or {}).values()
        if isinstance(row, Mapping)
    ]
    row_cores = {
        strategy_core_id(str(row.get("strategy_id") or ""))
        for row in rows
        if str(row.get("strategy_id") or "").strip()
    }
    missing = sorted(core for core in selected if core not in row_cores)
    if missing:
        if not requires_mask_contract:
            tprint(
                "LGBM strategy regime masks absent for accepted strategies, "
                "but deployed policy marks requires_lgbm_regime_mask_contract=false; "
                f"continuing without pre-base strategy mask gating: {missing}"
            )
            return
        raise RuntimeError(
            "LGBM strategy regime masks missing for accepted strategies: "
            f"{missing}. Refusing to fall back to legacy candidate masks."
        )
    missing_triggers = sorted(
        strategy_core_id(str(row.get("strategy_id") or ""))
        for row in rows
        if strategy_core_id(str(row.get("strategy_id") or "")) in selected
        and not str(row.get("base_event_trigger") or "").strip()
    )
    if missing_triggers:
        raise RuntimeError(
            "LGBM strategy regime mask rows are missing base_event_trigger for "
            f"{missing_triggers}"
        )


def _strategy_mask_symbols(
    strategy_candidate_masks: Dict[str, List[str]],
    strategy_id: str,
) -> Optional[set[str]]:
    sid = str(strategy_id or "")
    aliases = [sid, strategy_core_id(sid)]
    side = sid.split("_", 1)[0] if "_" in sid else ""
    core = strategy_core_id(sid)
    if side in {"long", "short"} and core:
        aliases.append(f"{side}_{core}")
    for alias in aliases:
        if alias and alias in strategy_candidate_masks:
            return {str(symbol) for symbol in strategy_candidate_masks[alias]}
    return None


def _policy_int(
    rules: Dict[str, Any],
    key: str,
    default: int,
    *,
    minimum: int = 1,
) -> int:
    try:
        value = int(rules.get(key, default))
    except Exception:
        value = int(default)
    return max(int(minimum), value)


def _deployment_rank_threshold_from_strategy_row(row: Dict[str, Any]) -> float:
    """Compute the live per-strategy rank gate saved by policy_optimiser.py."""
    try:
        saved = float(row.get("deployment_rank_threshold", np.nan))
    except Exception:
        saved = np.nan
    if np.isfinite(saved):
        return float(np.clip(saved, 0.0, 1.0))

    try:
        avg_top1_trades_per_day = float(
            row.get("avg_trades_per_day_at_top_1pct")
            or row.get("top1_avg_trades_per_day")
            or row.get("opportunities_per_day")
            or 0.0
        )
    except Exception:
        avg_top1_trades_per_day = 0.0
    try:
        avg_holding_hours = float(
            row.get("avg_holding_time_hours")
            or row.get("avg_holding_hours_at_top_1pct")
            or row.get("top1_avg_holding_hours")
            or 1.0
        )
    except Exception:
        avg_holding_hours = 1.0
    avg_holding_hours = max(1.0, avg_holding_hours)
    threshold = (avg_top1_trades_per_day / 24.0) * 2.0 / avg_holding_hours * 0.95
    floor_cfg = float(
        get_runtime_cfg().get("inference_deployment_rank_threshold_floor", 0.95)
    )
    floor_cfg = float(np.clip(floor_cfg, 0.0, 1.0))
    return float(np.clip(max(floor_cfg, threshold), 0.0, 1.0))


def _attach_rank_percentile_scores(
    decision_rows: List[Dict[str, Any]],
    *,
    score_key: str = "rank_score",
    allow_live_batch_rank_fallback_for_debug: bool = False,
) -> None:
    """Attach diagnostic per-strategy percentiles.

    Production rank-percentile gates use policy-rank references instead. The
    local live-batch fallback is debug-only because its population is not the
    optimiser's policy slice.
    """
    if not decision_rows:
        return
    by_strategy: Dict[str, List[int]] = {}
    for i, row in enumerate(decision_rows):
        sid = str(row.get("strategy_id", ""))
        if sid:
            by_strategy.setdefault(sid, []).append(i)
    for indices in by_strategy.values():
        scores = np.asarray(
            [float(decision_rows[i].get(score_key, np.nan)) for i in indices],
            dtype=np.float64,
        )
        finite = np.isfinite(scores)
        if not finite.any():
            continue
        ranks = pd.Series(scores[finite]).rank(method="max", pct=True).to_numpy()
        rank_out = np.full(len(scores), np.nan, dtype=np.float64)
        rank_out[finite] = ranks
        for local_i, row_i in enumerate(indices):
            row = decision_rows[row_i]
            if str(
                row.get("rank_score_source", "")
            ) == "historical_meta_oof_percentile" and np.isfinite(scores[local_i]):
                decision_rows[row_i]["sizer_rank_percentile"] = float(
                    np.clip(scores[local_i], 0.0, 1.0)
                )
            elif allow_live_batch_rank_fallback_for_debug:
                decision_rows[row_i]["sizer_rank_percentile"] = float(rank_out[local_i])


def _should_log_prediction_candidate(
    decision: Dict[str, Any],
    *,
    policy: PortfolioPolicyConfig,
) -> bool:
    if _env_flag("EPM_LOG_ALL_PREDICTION_CANDIDATES", False):
        return True
    rank = _safe_float(decision.get("sizer_rank_percentile"))
    if not np.isfinite(rank):
        rank = _safe_float(decision.get("threshold_score"))
    if not np.isfinite(rank):
        return False
    if _env_flag("EPM_LOG_ALL_SCORED_PREDICTION_CANDIDATES", True):
        return True
    ledger_pct = max(float(policy.top_prediction_ledger_pct), 0.40)
    return rank >= float(1.0 - ledger_pct)


def _max_feature_timestamp(feats: Dict[str, pd.DataFrame]) -> Optional[pd.Timestamp]:
    max_ts: Optional[pd.Timestamp] = None
    for df in (feats or {}).values():
        if not isinstance(df, pd.DataFrame) or df.empty:
            continue
        idx = pd.to_datetime(df.index, utc=True, errors="coerce")
        idx = idx[pd.notna(idx)]
        if len(idx) == 0:
            continue
        ts = pd.Timestamp(idx.max())
        if max_ts is None or ts > max_ts:
            max_ts = ts
    return max_ts


def _diagnostic_timestamp(value: Any, fallback: Any = None) -> Any:
    raw = value if value is not None else fallback
    ts = pd.to_datetime(raw, utc=True, errors="coerce")
    if pd.isna(ts):
        return None
    return pd.Timestamp(ts).isoformat()


def _artifact_file_hash(path: Path) -> Optional[str]:
    try:
        if path.exists():
            return hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    except Exception:
        return None
    return None


def _artifact_file_mtime_iso(path: Path) -> Optional[str]:
    try:
        if path.exists():
            return pd.to_datetime(path.stat().st_mtime, unit="s", utc=True).isoformat()
    except Exception:
        return None
    return None


def _meta_feature_contract_hash(data_root: str, run_id: str) -> Optional[str]:
    return _artifact_file_hash(
        Path(data_root)
        / "artifacts"
        / str(run_id)
        / "meta_oof"
        / "meta_feature_contract.json"
    )


def _persist_source_parity_report(
    report: Mapping[str, Any],
    *,
    data_root: str,
    run_id: str,
    label: str,
) -> Optional[Path]:
    try:
        end_ts = pd.to_datetime(report.get("end_ts"), utc=True, errors="coerce")
        stamp = (
            pd.Timestamp(end_ts).strftime("%Y%m%dT%H%M%SZ")
            if pd.notna(end_ts)
            else datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        )
        out_dir = Path(data_root) / "artifacts" / str(run_id) / "live_source_parity"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{stamp}_{label}.json"
        out_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        return out_path
    except Exception as exc:
        tprint(f"Warning: failed to persist source parity report: {exc}")
        return None


def _load_meta_hit_rate_calibration(data_root: str, run_id: str) -> Dict[str, Any]:
    """Load historical meta-prediction reliability curves for entry hit-rate estimates."""
    key = (str(data_root), str(run_id))
    cached = _META_HIT_RATE_CALIBRATION_CACHE.get(key)
    if cached is not None:
        return cached
    path = (
        Path(data_root)
        / "artifacts"
        / str(run_id)
        / "meta_oof"
        / "meta_calibration_report.json"
    )
    if not path.exists():
        _META_HIT_RATE_CALIBRATION_CACHE[key] = {}
        return {}
    try:
        payload = json.loads(path.read_text())
    except Exception as exc:
        tprint(f"Warning: failed to load meta hit-rate calibration {path}: {exc}")
        payload = {}
    out: Dict[str, Any] = {}
    if isinstance(payload, dict):
        for raw_key, row in payload.items():
            if not isinstance(row, dict):
                continue
            curve = (
                row.get("move_calibration", {}).get("reliability_curve")
                if isinstance(row.get("move_calibration"), dict)
                else None
            )
            if not curve:
                continue
            for k in _meta_hit_rate_calibration_aliases(raw_key):
                out[k] = curve
    _META_HIT_RATE_CALIBRATION_CACHE[key] = out
    return out


def _meta_hit_rate_calibration_aliases(
    strategy_id: Any,
    *,
    side: Optional[Any] = None,
) -> List[str]:
    """Return strategy-key aliases used by meta calibration reports and live rows."""
    sid = str(strategy_id or "").strip()
    if not sid:
        return []
    stripped = sid
    had_model_suffix = False
    for suffix in ("_tbm_clf", "_clf", "_reg", "_early_inval"):
        if stripped.endswith(suffix):
            stripped = stripped[: -len(suffix)]
            had_model_suffix = True
            break
    inferred_side = str(side or strategy_side(stripped) or strategy_side(sid) or "").strip()
    core = strategy_core_id(stripped)
    bases: List[tuple[str, bool]] = [(sid, not had_model_suffix), (stripped, True)]
    if core:
        bases.append((core, True))
        if inferred_side:
            bases.append((f"{inferred_side}_{core}", True))
    out: List[str] = []
    for base, expand in bases:
        if not base:
            continue
        keys = (base, f"{base}_clf", f"{base}_tbm_clf") if expand else (base,)
        for key in keys:
            if key and key not in out:
                out.append(key)
    return out


def _load_strategy_ev_calibration(data_root: str, run_id: str) -> Dict[str, Any]:
    """Load per-strategy EV curves from the deployed simple-policy candidate table."""
    key = (str(data_root), str(run_id))
    cached = _STRATEGY_EV_CALIBRATION_CACHE.get(key)
    if cached is not None:
        return cached
    path = (
        Path(data_root)
        / "artifacts"
        / str(run_id)
        / "simple_policy_optimiser"
        / "simple_policy_candidates.parquet"
    )
    if not path.exists():
        _STRATEGY_EV_CALIBRATION_CACHE[key] = {}
        return {}
    try:
        cols = [
            "strategy_id",
            "calibrated_score",
            "normalized_rank_score",
            "gross_return",
            "net_return",
            "fees_bps",
            "slippage_bps",
        ]
        candidates = pd.read_parquet(path, columns=cols)
    except Exception as exc:
        tprint(f"Warning: failed to load strategy EV calibration {path}: {exc}")
        _STRATEGY_EV_CALIBRATION_CACHE[key] = {}
        return {}
    required_cols = {"strategy_id", "calibrated_score", "gross_return", "net_return"}
    if not required_cols.issubset(candidates.columns):
        _STRATEGY_EV_CALIBRATION_CACHE[key] = {}
        return {}
    candidates = candidates.copy()
    candidates["strategy_id"] = candidates["strategy_id"].astype(str)
    for col in [
        "calibrated_score",
        "normalized_rank_score",
        "gross_return",
        "net_return",
        "fees_bps",
        "slippage_bps",
    ]:
        if col in candidates.columns:
            candidates[col] = pd.to_numeric(candidates[col], errors="coerce")
    candidates = candidates[
        np.isfinite(candidates["calibrated_score"].to_numpy(dtype=float))
        & np.isfinite(candidates["gross_return"].to_numpy(dtype=float))
        & np.isfinite(candidates["net_return"].to_numpy(dtype=float))
    ]
    out: Dict[str, Any] = {}
    for strategy_id, group in candidates.groupby("strategy_id", sort=False):
        group = group.sort_values("calibrated_score")
        if len(group) < 20:
            continue
        bins = min(20, max(5, len(group) // 500))
        try:
            bucket = pd.qcut(
                group["calibrated_score"],
                q=bins,
                duplicates="drop",
            )
        except ValueError:
            continue
        rows: list[dict[str, Any]] = []
        for _, bucket_df in group.groupby(bucket, observed=True):
            if bucket_df.empty:
                continue
            net = bucket_df["net_return"].astype(float)
            gross = bucket_df["gross_return"].astype(float)
            costs_bps = (gross - net) * 10000.0
            if "fees_bps" in bucket_df.columns:
                fees = bucket_df["fees_bps"].astype(float)
            else:
                fees = pd.Series(np.nan, index=bucket_df.index)
            if "slippage_bps" in bucket_df.columns:
                slippage = bucket_df["slippage_bps"].astype(float)
            else:
                slippage = pd.Series(np.nan, index=bucket_df.index)
            rows.append(
                {
                    "mean_score": float(bucket_df["calibrated_score"].mean()),
                    "mean_rank": (
                        float(bucket_df["normalized_rank_score"].mean())
                        if "normalized_rank_score" in bucket_df.columns
                        else None
                    ),
                    "mean_gross_return": float(gross.mean()),
                    "mean_net_return": float(net.mean()),
                    "mean_cost_bps": float(costs_bps.mean()),
                    "mean_fees_bps": (
                        float(fees.mean()) if np.isfinite(fees).any() else None
                    ),
                    "mean_slippage_bps": (
                        float(slippage.mean())
                        if np.isfinite(slippage).any()
                        else None
                    ),
                    "hit_rate": float((net > 0.0).mean()),
                    "count": int(len(bucket_df)),
                }
            )
        if not rows:
            continue
        keys = {str(strategy_id)}
        core = strategy_core_id(str(strategy_id))
        if core:
            keys.add(core)
        for out_key in keys:
            out[out_key] = rows
    _STRATEGY_EV_CALIBRATION_CACHE[key] = out
    return out


def _weighted_isotonic_points(
    points: Sequence[tuple[float, float, int]],
) -> tuple[np.ndarray, np.ndarray]:
    """Return monotone historical hit-rate calibration points using PAV."""
    if not points:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    ordered = sorted(points, key=lambda item: item[0])
    blocks: list[dict[str, float]] = []
    for x, y, n in ordered:
        weight = float(max(int(n), 1))
        blocks.append(
            {
                "x_weighted_sum": float(x) * weight,
                "y_weighted_sum": float(y) * weight,
                "weight": weight,
            }
        )
        while len(blocks) >= 2:
            prev = blocks[-2]["y_weighted_sum"] / blocks[-2]["weight"]
            cur = blocks[-1]["y_weighted_sum"] / blocks[-1]["weight"]
            if prev <= cur:
                break
            merged = {
                "x_weighted_sum": blocks[-2]["x_weighted_sum"]
                + blocks[-1]["x_weighted_sum"],
                "y_weighted_sum": blocks[-2]["y_weighted_sum"]
                + blocks[-1]["y_weighted_sum"],
                "weight": blocks[-2]["weight"] + blocks[-1]["weight"],
            }
            blocks[-2:] = [merged]
    xs = np.asarray([b["x_weighted_sum"] / b["weight"] for b in blocks], dtype=float)
    ys = np.asarray([b["y_weighted_sum"] / b["weight"] for b in blocks], dtype=float)
    return xs, ys


def _estimated_hit_rate_from_meta_prediction(
    raw_meta_prediction: Any,
    strategy_id: str,
    calibration: Mapping[str, Any],
) -> Dict[str, Any]:
    """Map a live meta prediction to historical win probability when calibrated."""
    try:
        raw = float(raw_meta_prediction)
    except (TypeError, ValueError):
        raw = float("nan")
    if not np.isfinite(raw):
        return {
            "estimated_hit_rate": None,
            "estimated_hit_rate_source": "meta_prediction_non_finite",
            "estimated_hit_rate_calibration_n": 0,
        }
    candidates = _meta_hit_rate_calibration_aliases(strategy_id)
    curve = None
    matched_key = ""
    for key in candidates:
        if key and key in calibration:
            curve = calibration[key]
            matched_key = key
            break
    if not curve:
        return {
            "estimated_hit_rate": None,
            "estimated_hit_rate_source": "missing_meta_oof_reliability_curve",
            "estimated_hit_rate_calibration_n": 0,
        }
    points = []
    for item in curve:
        if not isinstance(item, Mapping):
            continue
        try:
            x = float(item.get("mean_pred"))
            y = float(item.get("mean_true"))
        except (TypeError, ValueError):
            continue
        if np.isfinite(x) and np.isfinite(y):
            points.append((x, y, int(item.get("count") or 0)))
    if not points:
        return {
            "estimated_hit_rate": None,
            "estimated_hit_rate_source": "empty_meta_oof_reliability_curve",
            "estimated_hit_rate_calibration_n": 0,
        }
    xs, ys = _weighted_isotonic_points(points)
    est = float(np.interp(raw, xs, ys))
    return {
        "estimated_hit_rate": float(np.clip(est, 0.0, 1.0)),
        "estimated_hit_rate_source": (
            "meta_oof_reliability_curve_isotonic_interp"
            + (f":{matched_key}" if matched_key else "")
        ),
        "estimated_hit_rate_calibration_n": int(sum(p[2] for p in points)),
    }


def _estimated_ev_from_strategy_prediction(
    calibrated_score: Any,
    strategy_id: str,
    calibration: Mapping[str, Any],
) -> Dict[str, Any]:
    """Map live strategy score to per-strategy historical gross/net EV."""
    try:
        score = float(calibrated_score)
    except (TypeError, ValueError):
        score = float("nan")
    if not np.isfinite(score):
        return {
            "estimated_ev_gross_return": None,
            "estimated_ev_net_return": None,
            "estimated_ev_cost_bps": None,
            "estimated_ev_hit_rate": None,
            "estimated_ev_source": "calibrated_score_non_finite",
            "estimated_ev_calibration_n": 0,
        }
    sid = str(strategy_id or "")
    candidates = [sid, strategy_core_id(sid)]
    curve = None
    for key in candidates:
        if key and key in calibration:
            curve = calibration[key]
            break
    if not curve:
        return {
            "estimated_ev_gross_return": None,
            "estimated_ev_net_return": None,
            "estimated_ev_cost_bps": None,
            "estimated_ev_hit_rate": None,
            "estimated_ev_source": "missing_strategy_ev_curve",
            "estimated_ev_calibration_n": 0,
        }

    def _points(metric: str) -> list[tuple[float, float, int]]:
        out = []
        for row in curve:
            if not isinstance(row, Mapping):
                continue
            try:
                x = float(row.get("mean_score"))
                y = float(row.get(metric))
                n = int(row.get("count") or 0)
            except (TypeError, ValueError):
                continue
            if np.isfinite(x) and np.isfinite(y):
                out.append((x, y, n))
        return out

    gross_points = _points("mean_gross_return")
    net_points = _points("mean_net_return")
    cost_points = _points("mean_cost_bps")
    hit_points = _points("hit_rate")
    if not net_points:
        return {
            "estimated_ev_gross_return": None,
            "estimated_ev_net_return": None,
            "estimated_ev_cost_bps": None,
            "estimated_ev_hit_rate": None,
            "estimated_ev_source": "empty_strategy_ev_curve",
            "estimated_ev_calibration_n": 0,
        }

    def _interp(points: list[tuple[float, float, int]]) -> float | None:
        if not points:
            return None
        xs, ys = _weighted_isotonic_points(points)
        if len(xs) == 0:
            return None
        return float(np.interp(score, xs, ys))

    gross = _interp(gross_points)
    net = _interp(net_points)
    cost = _interp(cost_points)
    hit = _interp(hit_points)
    return {
        "estimated_ev_gross_return": gross,
        "estimated_ev_net_return": net,
        "estimated_ev_cost_bps": cost,
        "estimated_ev_hit_rate": hit,
        "estimated_ev_source": "strategy_simple_policy_candidate_curve_isotonic_interp",
        "estimated_ev_calibration_n": int(sum(p[2] for p in net_points)),
    }


def _ev_adjusted_prediction_after_entry_friction(
    *,
    calibrated_score: Any,
    strategy_id: str,
    side: str,
    calibration: Mapping[str, Any],
    live_entry_friction_bps: Any,
    observed_spread_bps: Any = None,
    orderbook_slippage_bps: Any = None,
    adverse_signal_gap_bps: Any = None,
    spread_baseline_bps: float = 97.32886619027215,
    spread_baseline_source: str = "portfolio_policy.ev_haircut_expected_spread_bps",
    delay_slippage_baseline_bps: float = 40.0,
    policy_rank_reference_store: Optional[PolicyRankReferenceStore] = None,
) -> Dict[str, Any]:
    """Subtract only excess live execution drag from EV and remap to score/rank.

    The policy-OOS EV curve is already net of the optimiser's execution baseline.
    Live inference therefore haircuts only the part of spread/slippage/delay that
    exceeds that baseline, preserving the training/policy contract.
    """
    try:
        score = float(calibrated_score)
    except (TypeError, ValueError):
        score = float("nan")
    try:
        friction_bps = float(live_entry_friction_bps)
    except (TypeError, ValueError):
        friction_bps = float("nan")
    try:
        spread_bps = float(observed_spread_bps)
    except (TypeError, ValueError):
        spread_bps = float("nan")
    try:
        slippage_bps = float(orderbook_slippage_bps)
    except (TypeError, ValueError):
        slippage_bps = float("nan")
    try:
        adverse_gap_bps = float(adverse_signal_gap_bps)
    except (TypeError, ValueError):
        adverse_gap_bps = float("nan")
    if not np.isfinite(score):
        return {"ev_adjusted_source": "calibrated_score_non_finite"}
    if not np.isfinite(friction_bps) or friction_bps < 0.0:
        friction_bps = 0.0
    if not np.isfinite(spread_bps) or spread_bps < 0.0:
        spread_bps = 0.0
    if not np.isfinite(slippage_bps) or slippage_bps < 0.0:
        slippage_bps = 0.0
    if not np.isfinite(adverse_gap_bps) or adverse_gap_bps < 0.0:
        adverse_gap_bps = 0.0
    spread_baseline = max(0.0, float(spread_baseline_bps or 0.0))
    half_spread_baseline = spread_baseline / 2.0
    delay_slippage_baseline = max(0.0, float(delay_slippage_baseline_bps or 0.0))
    observed_half_spread_bps = spread_bps / 2.0
    spread_excess_bps = max(0.0, observed_half_spread_bps - half_spread_baseline)
    observed_delay_slippage_bps = max(0.0, adverse_gap_bps) + max(0.0, slippage_bps)
    delay_slippage_excess_bps = max(
        0.0, observed_delay_slippage_bps - delay_slippage_baseline
    )
    ev_haircut_bps = spread_excess_bps + delay_slippage_excess_bps
    contract_fields = {
        "ev_adjusted_entry_friction_bps": float(friction_bps),
        "ev_haircut_bps": float(ev_haircut_bps),
        "ev_haircut_raw_live_entry_friction_bps": float(friction_bps),
        "ev_haircut_observed_spread_bps": float(spread_bps),
        "ev_haircut_observed_half_spread_bps": float(observed_half_spread_bps),
        "ev_haircut_spread_baseline_bps": float(spread_baseline),
        "ev_haircut_spread_baseline_source": str(spread_baseline_source or ""),
        "ev_haircut_half_spread_baseline_bps": float(half_spread_baseline),
        "ev_haircut_spread_excess_bps": float(spread_excess_bps),
        "ev_haircut_orderbook_slippage_bps": float(slippage_bps),
        "ev_haircut_adverse_signal_gap_bps": float(adverse_gap_bps),
        "ev_haircut_observed_delay_slippage_bps": float(
            observed_delay_slippage_bps
        ),
        "ev_haircut_delay_slippage_baseline_bps": float(
            delay_slippage_baseline
        ),
        "ev_haircut_delay_slippage_excess_bps": float(
            delay_slippage_excess_bps
        ),
        "ev_haircut_contract": (
            "spread_excess=max(0, observed_spread_bps/2 - "
            "symbol_average_spread_bps/2); "
            "delay_slippage_excess=max(0, adverse_signal_gap_bps + "
            "orderbook_slippage_bps - delay_slippage_baseline_bps)"
        ),
    }
    sid = str(strategy_id or "")
    curve = None
    for key in (sid, strategy_core_id(sid)):
        if key and key in calibration:
            curve = calibration[key]
            break
    if not curve:
        return {
            **contract_fields,
            "ev_adjusted_source": "missing_strategy_ev_curve",
        }
    net_points: list[tuple[float, float, int]] = []
    for row in curve:
        if not isinstance(row, Mapping):
            continue
        try:
            x = float(row.get("mean_score"))
            y = float(row.get("mean_net_return"))
            n = int(row.get("count") or 0)
        except (TypeError, ValueError):
            continue
        if np.isfinite(x) and np.isfinite(y):
            net_points.append((x, y, n))
    if not net_points:
        return {
            **contract_fields,
            "ev_adjusted_source": "empty_strategy_ev_curve",
        }
    xs, ys = _weighted_isotonic_points(net_points)
    if len(xs) == 0 or len(ys) == 0:
        return {
            **contract_fields,
            "ev_adjusted_source": "empty_strategy_ev_curve",
        }
    current_net_ev = float(np.interp(score, xs, ys))
    adjusted_net_ev = current_net_ev - float(ev_haircut_bps) / 10000.0
    if np.nanmax(ys) <= np.nanmin(ys):
        adjusted_score = float(score)
    else:
        adjusted_score = float(np.interp(adjusted_net_ev, ys, xs))
    adjusted_score = float(np.clip(adjusted_score, float(np.nanmin(xs)), float(np.nanmax(xs))))
    adjusted_rank = float("nan")
    rank_n = 0
    rank_source = ""
    if policy_rank_reference_store is not None:
        lookup = policy_rank_reference_store.lookup(
            strategy_id=sid,
            calibrated_score=adjusted_score,
            side=side,
        )
        adjusted_rank = float(lookup.policy_rank_pct)
        rank_n = int(lookup.n_rows)
        rank_source = str(lookup.source)
    return {
        **contract_fields,
        "ev_adjusted_initial_calibrated_score": float(score),
        "ev_adjusted_net_return_before_friction": current_net_ev,
        "ev_adjusted_net_return_after_friction": adjusted_net_ev,
        "ev_adjusted_calibrated_score": adjusted_score,
        "ev_adjusted_rank_score": adjusted_rank if np.isfinite(adjusted_rank) else None,
        "ev_adjusted_rank_reference_n": rank_n,
        "ev_adjusted_rank_reference_source": rank_source,
        "ev_adjusted_source": "strategy_ev_curve_inverse_after_excess_live_entry_friction",
    }


def _adverse_signal_gap_bps(*, side: str, signal_price: Any, decision_mid: Any) -> float:
    signal = _safe_float(signal_price, np.nan)
    mid = _safe_float(decision_mid, np.nan)
    if not (np.isfinite(signal) and signal > 0.0 and np.isfinite(mid) and mid > 0.0):
        return 0.0
    side_s = str(side).lower()
    if side_s == "long":
        return float(max(mid / signal - 1.0, 0.0) * 10000.0)
    if side_s == "short":
        return float(max(signal / mid - 1.0, 0.0) * 10000.0)
    return 0.0


def _panel_symbol_series(
    panel: Mapping[str, Any],
    key: str,
    symbol: str,
) -> Optional[pd.Series]:
    frame = panel.get(key) if isinstance(panel, Mapping) else None
    if not isinstance(frame, pd.DataFrame) or symbol not in frame.columns:
        return None
    series = pd.to_numeric(frame[symbol], errors="coerce")
    try:
        series.index = pd.to_datetime(series.index, utc=True, errors="coerce")
    except Exception:
        pass
    return series.replace([np.inf, -np.inf], np.nan)


def _latest_panel_value(
    panel: Mapping[str, Any],
    key: str,
    symbol: str,
) -> tuple[float, Optional[pd.Timestamp]]:
    series = _panel_symbol_series(panel, key, symbol)
    if series is None:
        return np.nan, None
    non_null = series.dropna()
    if non_null.empty:
        return np.nan, None
    ts = pd.to_datetime(non_null.index[-1], utc=True, errors="coerce")
    return _safe_float(non_null.iloc[-1], np.nan), (
        pd.Timestamp(ts) if not pd.isna(ts) else None
    )


def _panel_value_at_or_before(
    panel: Mapping[str, Any],
    key: str,
    symbol: str,
    ts: Optional[pd.Timestamp],
) -> tuple[float, Optional[pd.Timestamp]]:
    series = _panel_symbol_series(panel, key, symbol)
    if series is None:
        return np.nan, None
    if ts is not None:
        try:
            series = series.loc[series.index <= pd.Timestamp(ts)]
        except Exception:
            pass
    non_null = series.dropna()
    if non_null.empty:
        return np.nan, None
    out_ts = pd.to_datetime(non_null.index[-1], utc=True, errors="coerce")
    return _safe_float(non_null.iloc[-1], np.nan), (
        pd.Timestamp(out_ts) if not pd.isna(out_ts) else None
    )


def _raw_close_reference_gap_bps(
    runtime_config: Mapping[str, Any],
    default: float,
) -> float:
    raw = os.environ.get(
        "EPM_RAW_CLOSE_REFERENCE_GAP_BPS",
        runtime_config.get("raw_close_reference_gap_bps", default),
    )
    try:
        out = float(raw)
    except (TypeError, ValueError):
        out = float(default)
    return out if np.isfinite(out) and out >= 0.0 else float(default)


def _raw_signal_close_reliability_snapshot(
    panel: Mapping[str, Any],
    symbol: str,
    *,
    max_reference_gap_bps: float = 150.0,
) -> Dict[str, Any]:
    """Audit whether the raw close used by live features is tradable enough.

    We intentionally do not replace raw close with mark/index here: doing so
    would make live execution diverge from the feature values already scored.
    If raw close is stale or inconsistent enough to be unreliable, the caller
    should skip the candidate.
    """
    close, close_ts = _latest_panel_value(panel, "close", symbol)
    snap: Dict[str, Any] = {
        "signal_price": close if np.isfinite(close) and close > 0.0 else None,
        "raw_signal_close": close if np.isfinite(close) else None,
        "raw_signal_close_ts": close_ts.isoformat() if close_ts is not None else None,
        "raw_signal_close_unreliable": False,
        "raw_signal_close_unreliable_reason": "",
        "raw_signal_close_reference_gap_bps": np.nan,
        "raw_signal_close_reference_price": np.nan,
        "raw_signal_close_reference_source": "",
        "raw_signal_close_reference_ts": None,
    }
    if not (np.isfinite(close) and close > 0.0):
        snap["raw_signal_close_unreliable"] = True
        snap["raw_signal_close_unreliable_reason"] = "missing_raw_close"
        return snap

    volume, volume_ts = _panel_value_at_or_before(panel, "volume", symbol, close_ts)
    snap["raw_signal_volume"] = volume if np.isfinite(volume) else None
    snap["raw_signal_volume_ts"] = (
        volume_ts.isoformat() if volume_ts is not None else None
    )

    best_source = ""
    best_price = np.nan
    best_ts: Optional[pd.Timestamp] = None
    best_gap = np.nan
    for ref_key in (
        "mark_close",
        "mark_price",
        "index_price",
        "index_close",
        "spot_close",
        "canonical_index",
    ):
        ref_price, ref_ts = _panel_value_at_or_before(
            panel, ref_key, symbol, close_ts
        )
        if not (np.isfinite(ref_price) and ref_price > 0.0):
            continue
        gap = abs(close / ref_price - 1.0) * 10000.0
        if not np.isfinite(best_gap) or gap < best_gap:
            best_gap = float(gap)
            best_price = float(ref_price)
            best_source = ref_key
            best_ts = ref_ts

    if np.isfinite(best_gap):
        snap["raw_signal_close_reference_gap_bps"] = float(best_gap)
        snap["raw_signal_close_reference_price"] = float(best_price)
        snap["raw_signal_close_reference_source"] = best_source
        snap["raw_signal_close_reference_ts"] = (
            best_ts.isoformat() if best_ts is not None else None
        )

    if np.isfinite(volume) and volume <= 0.0:
        snap["raw_signal_close_unreliable"] = True
        snap["raw_signal_close_unreliable_reason"] = "zero_volume_raw_close"
    elif (
        np.isfinite(best_gap)
        and np.isfinite(max_reference_gap_bps)
        and best_gap >= float(max_reference_gap_bps)
    ):
        snap["raw_signal_close_unreliable"] = True
        snap["raw_signal_close_unreliable_reason"] = (
            "raw_close_reference_gap_too_large"
        )
    return snap


def _signal_bar_close_ts(signal_bar_ts: Any, *, bar_hours: float = 1.0) -> pd.Timestamp | None:
    ts = pd.to_datetime(signal_bar_ts, utc=True, errors="coerce")
    if pd.isna(ts):
        return None
    return pd.Timestamp(ts) + pd.Timedelta(hours=float(bar_hours))


def _entry_timing_snapshot(
    *,
    decision: Mapping[str, Any],
    now: pd.Timestamp,
    signal_bar_ts: Any,
    max_signal_close_age_seconds: float,
) -> Dict[str, Any]:
    signal_ts = decision.get("signal_bar_ts") or signal_bar_ts
    close_ts = decision.get("signal_bar_close_ts") or _signal_bar_close_ts(signal_ts)
    close_ts_parsed = pd.to_datetime(close_ts, utc=True, errors="coerce")
    now_ts = pd.to_datetime(now, utc=True, errors="coerce")
    age_seconds = np.nan
    if not pd.isna(close_ts_parsed) and not pd.isna(now_ts):
        age_seconds = float((pd.Timestamp(now_ts) - pd.Timestamp(close_ts_parsed)).total_seconds())
    signal_ts_parsed = pd.to_datetime(signal_ts, utc=True, errors="coerce")
    signal_to_decision_seconds = np.nan
    if not pd.isna(signal_ts_parsed) and not pd.isna(now_ts):
        signal_to_decision_seconds = float(
            (pd.Timestamp(now_ts) - pd.Timestamp(signal_ts_parsed)).total_seconds()
        )
    limit_seconds = float(max_signal_close_age_seconds)
    return {
        "decision_ts": pd.Timestamp(now_ts).isoformat() if not pd.isna(now_ts) else None,
        "signal_bar_ts": (
            pd.Timestamp(signal_ts_parsed).isoformat()
            if not pd.isna(signal_ts_parsed)
            else None
        ),
        "signal_bar_close_ts": (
            pd.Timestamp(close_ts_parsed).isoformat()
            if not pd.isna(close_ts_parsed)
            else None
        ),
        "signal_close_to_decision_seconds": age_seconds,
        "signal_to_decision_seconds": signal_to_decision_seconds,
        "max_signal_close_to_entry_seconds": limit_seconds,
        "stale_signal_age_gate_enabled": bool(limit_seconds >= 0.0),
        "stale_signal_age_gate_exceeded": bool(
            np.isfinite(age_seconds) and limit_seconds >= 0.0 and age_seconds > limit_seconds
        ),
    }


def _max_signal_close_to_entry_seconds(runtime_config: Mapping[str, Any]) -> float:
    raw = os.environ.get(
        "EPM_MAX_SIGNAL_CLOSE_TO_ENTRY_SECONDS",
        runtime_config.get("max_signal_close_to_entry_seconds", 900.0),
    )
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 900.0


def _signal_to_entry_alert_seconds(runtime_config: Mapping[str, Any]) -> float:
    raw = os.environ.get(
        "EPM_SIGNAL_TO_ENTRY_ALERT_SECONDS",
        runtime_config.get("signal_to_entry_alert_seconds", 600.0),
    )
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 600.0


_LIVE_PRESCORE_MARKET_MASK_MODES = {
    "live",
    "live-test",
    "live_test",
    "livetest",
    "paper",
    "shadow_live",
}


def _runtime_flag(
    runtime_config: Mapping[str, Any],
    key: str,
    env_name: str,
    default: bool,
) -> bool:
    raw = os.environ.get(env_name, runtime_config.get(key, default))
    return str(raw).strip().lower() not in {"0", "false", "no", "off"}


def _runtime_float(
    runtime_config: Mapping[str, Any],
    key: str,
    env_name: str,
    default: float,
) -> float:
    raw = os.environ.get(env_name, runtime_config.get(key, default))
    try:
        out = float(raw)
    except (TypeError, ValueError):
        out = float(default)
    return out if np.isfinite(out) else float(default)


def _runtime_int(
    runtime_config: Mapping[str, Any],
    key: str,
    env_name: str,
    default: int,
) -> int:
    raw = os.environ.get(env_name, runtime_config.get(key, default))
    try:
        return int(raw)
    except (TypeError, ValueError):
        return int(default)


def _live_prescore_market_mask_enabled(
    runtime_config: Mapping[str, Any],
    executor_mode: str,
) -> bool:
    default = str(executor_mode or "").strip().lower() in _LIVE_PRESCORE_MARKET_MASK_MODES
    return _runtime_flag(
        runtime_config,
        "live_prescore_market_mask_enabled",
        "EPM_LIVE_PRESCORE_MARKET_MASK_ENABLED",
        default,
    )


def _live_prescore_orderbook_enabled(
    runtime_config: Mapping[str, Any],
    executor_mode: str,
    policy: PortfolioPolicyConfig,
) -> bool:
    default = (
        str(executor_mode or "").strip().lower() in _LIVE_PRESCORE_MARKET_MASK_MODES
        and bool(policy.orderbook_precheck_enabled)
    )
    return _runtime_flag(
        runtime_config,
        "live_prescore_orderbook_enabled",
        "EPM_LIVE_PRESCORE_ORDERBOOK_ENABLED",
        default,
    )


def _panel_metric_snapshot(
    panel: Mapping[str, Any],
    keys: Sequence[str],
    symbol: str,
    at_ts: Optional[pd.Timestamp],
) -> Dict[str, Any]:
    for key in keys:
        if at_ts is None:
            value, ts = _latest_panel_value(panel, key, symbol)
        else:
            value, ts = _panel_value_at_or_before(panel, key, symbol, at_ts)
        if np.isfinite(value):
            return {"key": key, "value": float(value), "ts": ts}
    return {"key": "", "value": np.nan, "ts": None}


def _pre_score_market_mask_snapshot(
    *,
    panel: Mapping[str, Any],
    symbol: str,
    side: str,
    strategy_id: str,
    executor: TradeExecutor,
    policy: PortfolioPolicyConfig,
    runtime_config: Mapping[str, Any],
    now: pd.Timestamp,
    signal_bar_ts: Any,
    raw_close_reference_gap_bps: float,
    max_signal_close_to_entry_seconds: float,
) -> Dict[str, Any]:
    """Fail fast on live market/data-quality inputs before model scoring.

    The final execution path still repeats ticker/orderbook checks after rank
    sizing because position size is only known later. This pre-score pass is a
    cheap mask that keeps stale, untradable, or clearly illiquid symbols out of
    the base/meta model chain.
    """
    snapshot: Dict[str, Any] = {
        "prescore_market_mask_enabled": True,
        "prescore_market_mask_allowed": True,
        "prescore_market_mask_reason": "",
        "prescore_strategy_id": str(strategy_id),
    }

    def _reject(reason: str) -> Dict[str, Any]:
        snapshot["prescore_market_mask_allowed"] = False
        snapshot["prescore_market_mask_reason"] = reason
        return snapshot

    close_snapshot = _raw_signal_close_reliability_snapshot(
        panel,
        symbol,
        max_reference_gap_bps=raw_close_reference_gap_bps,
    )
    snapshot.update(
        {
            "prescore_signal_price": close_snapshot.get("signal_price"),
            "prescore_raw_signal_close": close_snapshot.get("raw_signal_close"),
            "prescore_raw_signal_close_ts": close_snapshot.get("raw_signal_close_ts"),
            "prescore_raw_signal_volume": close_snapshot.get("raw_signal_volume"),
            "prescore_raw_signal_volume_ts": close_snapshot.get("raw_signal_volume_ts"),
            "prescore_raw_signal_close_reference_gap_bps": close_snapshot.get(
                "raw_signal_close_reference_gap_bps"
            ),
            "prescore_raw_signal_close_reference_source": close_snapshot.get(
                "raw_signal_close_reference_source"
            ),
        }
    )
    if bool(close_snapshot.get("raw_signal_close_unreliable")):
        return _reject(
            "unreliable_raw_signal_close:"
            + str(close_snapshot.get("raw_signal_close_unreliable_reason") or "")
        )

    timing_snapshot = _entry_timing_snapshot(
        decision={"signal_bar_ts": signal_bar_ts},
        now=now,
        signal_bar_ts=signal_bar_ts,
        max_signal_close_age_seconds=max_signal_close_to_entry_seconds,
    )
    snapshot.update(
        {
            "prescore_signal_bar_close_ts": timing_snapshot.get("signal_bar_close_ts"),
            "prescore_signal_close_to_decision_seconds": timing_snapshot.get(
                "signal_close_to_decision_seconds"
            ),
            "prescore_max_signal_close_to_entry_seconds": timing_snapshot.get(
                "max_signal_close_to_entry_seconds"
            ),
            "prescore_stale_signal_age_gate_exceeded": timing_snapshot.get(
                "stale_signal_age_gate_exceeded"
            ),
        }
    )
    if bool(timing_snapshot.get("stale_signal_age_gate_exceeded")):
        return _reject("stale_signal_age_exceeded")

    signal_close_ts_raw = pd.to_datetime(
        timing_snapshot.get("signal_bar_close_ts"), utc=True, errors="coerce"
    )
    signal_close_ts = None if pd.isna(signal_close_ts_raw) else pd.Timestamp(signal_close_ts_raw)
    oi_snapshot = _panel_metric_snapshot(
        panel,
        (
            "open_interest",
            "open_interest_value",
            "open_interest_usd",
            "oi",
            "oi_value",
            "openInterest",
        ),
        symbol,
        signal_close_ts,
    )
    oi_ts = oi_snapshot.get("ts")
    oi_age_hours = np.nan
    if signal_close_ts is not None and isinstance(oi_ts, pd.Timestamp):
        oi_age_hours = max(
            float((signal_close_ts - oi_ts).total_seconds()) / 3600.0,
            0.0,
        )
    snapshot.update(
        {
            "prescore_oi_key": oi_snapshot.get("key"),
            "prescore_oi_value": oi_snapshot.get("value"),
            "prescore_oi_ts": oi_ts.isoformat() if isinstance(oi_ts, pd.Timestamp) else None,
            "prescore_oi_age_hours": oi_age_hours,
        }
    )
    require_oi = _runtime_flag(
        runtime_config,
        "live_prescore_require_open_interest",
        "EPM_LIVE_PRESCORE_REQUIRE_OI",
        _is_perps_config(dict(runtime_config)),
    )
    max_oi_age_hours = _runtime_float(
        runtime_config,
        "live_prescore_max_open_interest_age_hours",
        "EPM_LIVE_PRESCORE_MAX_OI_AGE_HOURS",
        24.0,
    )
    oi_value = _safe_float(oi_snapshot.get("value"), np.nan)
    if require_oi:
        if not (np.isfinite(oi_value) and oi_value > 0.0):
            return _reject("missing_or_nonpositive_open_interest")
        if np.isfinite(oi_age_hours) and oi_age_hours > max_oi_age_hours:
            return _reject("stale_open_interest")

    exchange = getattr(executor, "exchange", None)
    require_ticker = _runtime_flag(
        runtime_config,
        "live_prescore_require_ticker",
        "EPM_LIVE_PRESCORE_REQUIRE_TICKER",
        True,
    )
    if exchange is None:
        return _reject("missing_exchange_for_prescore_ticker") if require_ticker else snapshot
    api_symbol = _live_exchange_symbol(exchange, dict(runtime_config), symbol)
    try:
        ticker_snapshot = fetch_ticker_snapshot(
            exchange=exchange,
            symbol=api_symbol,
            side=side,
            policy=policy,
            mode=str(getattr(executor, "mode", "")),
            now=now,
        )
    except Exception as exc:
        snapshot["prescore_ticker_error"] = f"{type(exc).__name__}: {exc}"
        return _reject("ticker_fetch_failed")
    ticker_dict = ticker_snapshot.to_dict()
    details = ticker_dict.get("details") if isinstance(ticker_dict.get("details"), dict) else {}
    snapshot.update(
        {
            "prescore_ticker_bid": ticker_dict.get("bid"),
            "prescore_ticker_ask": ticker_dict.get("ask"),
            "prescore_ticker_mid": ticker_dict.get("mid"),
            "prescore_ticker_last": ticker_dict.get("last"),
            "prescore_ticker_spread_bps": ticker_dict.get("spread_bps"),
            "prescore_ticker_spread_weight": ticker_dict.get("spread_weight"),
            "prescore_ticker_age_seconds": details.get("exchange_ticker_age_seconds"),
            "prescore_ticker_fetch_latency_seconds": details.get(
                "ticker_fetch_latency_seconds"
            ),
            "prescore_ticker_reject_reason": ticker_dict.get("reject_reason"),
        }
    )
    if bool(ticker_dict.get("hard_reject")):
        return _reject(str(ticker_dict.get("reject_reason") or "ticker_rejected"))
    max_spread_bps = _runtime_float(
        runtime_config,
        "live_prescore_max_spread_bps",
        "EPM_LIVE_PRESCORE_MAX_SPREAD_BPS",
        float(policy.max_spread_bps),
    )
    spread_bps = _safe_float(ticker_dict.get("spread_bps"), np.nan)
    if not np.isfinite(spread_bps):
        return _reject("missing_ticker_spread")
    if spread_bps > max_spread_bps:
        snapshot["prescore_max_spread_bps"] = float(max_spread_bps)
        return _reject("ticker_spread_above_prescore_max")
    snapshot["prescore_max_spread_bps"] = float(max_spread_bps)

    if _live_prescore_orderbook_enabled(
        runtime_config,
        str(getattr(executor, "mode", "")),
        policy,
    ):
        intended_quote_size = _runtime_float(
            runtime_config,
            "live_prescore_liquidity_probe_quote_notional",
            "EPM_LIVE_PRESCORE_LIQUIDITY_PROBE_QUOTE_NOTIONAL",
            max(
                float(policy.live_test_min_quote_notional),
                min(100.0, float(policy.max_position_quote_notional) * 0.02),
            ),
        )
        try:
            liquidity_snapshot = evaluate_orderbook_liquidity(
                exchange=exchange,
                symbol=api_symbol,
                side=side,
                intended_quote_size=float(intended_quote_size),
                ticker_snapshot=ticker_snapshot,
                policy=policy,
                mode=str(getattr(executor, "mode", "")),
            )
        except Exception as exc:
            snapshot["prescore_orderbook_error"] = f"{type(exc).__name__}: {exc}"
            return _reject("orderbook_fetch_failed")
        liq_dict = liquidity_snapshot.to_dict()
        snapshot.update(
            {
                "prescore_orderbook_side": liq_dict.get("orderbook_side"),
                "prescore_orderbook_capacity_quote_within_slippage": liq_dict.get(
                    "orderbook_capacity_quote_within_slippage"
                ),
                "prescore_orderbook_intended_quote_size": liq_dict.get(
                    "intended_quote_size"
                ),
                "prescore_orderbook_depth_weight": liq_dict.get("depth_weight"),
                "prescore_liquidity_capacity_weight": liq_dict.get(
                    "liquidity_capacity_weight"
                ),
                "prescore_orderbook_slippage_bps": liq_dict.get(
                    "expected_fill_slippage_bps"
                ),
                "prescore_orderbook_reject_reason": liq_dict.get("reject_reason"),
            }
        )
        if bool(liq_dict.get("hard_reject")):
            return _reject(str(liq_dict.get("reject_reason") or "orderbook_rejected"))

    return snapshot


def _apply_pre_score_market_masks(
    *,
    panel: Mapping[str, Any],
    candidates: Sequence[str],
    side: str,
    strategy_id: str,
    executor: TradeExecutor,
    policy: PortfolioPolicyConfig,
    runtime_config: Mapping[str, Any],
    now: pd.Timestamp,
    signal_bar_ts: Any,
    raw_close_reference_gap_bps: float,
    max_signal_close_to_entry_seconds: float,
    side_metrics: Dict[str, Any],
) -> tuple[List[str], Dict[str, Dict[str, Any]]]:
    kept: List[str] = []
    snapshots: Dict[str, Dict[str, Any]] = {}
    reason_counts: Dict[str, int] = {}
    for symbol in candidates:
        snap = _pre_score_market_mask_snapshot(
            panel=panel,
            symbol=str(symbol),
            side=side,
            strategy_id=strategy_id,
            executor=executor,
            policy=policy,
            runtime_config=runtime_config,
            now=now,
            signal_bar_ts=signal_bar_ts,
            raw_close_reference_gap_bps=raw_close_reference_gap_bps,
            max_signal_close_to_entry_seconds=max_signal_close_to_entry_seconds,
        )
        snapshots[str(symbol)] = snap
        side_metrics["prescore_market_mask_input"] += 1
        if bool(snap.get("prescore_market_mask_allowed")):
            kept.append(str(symbol))
            side_metrics["prescore_market_mask_pass"] += 1
            continue
        reason = str(snap.get("prescore_market_mask_reason") or "unknown")
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
        side_metrics["prescore_market_mask_block"] += 1
        side_metrics["non_fatal_issues"] += 1
    if candidates:
        side_metrics.setdefault("prescore_market_mask_reasons", {})
        merged_reasons = dict(side_metrics.get("prescore_market_mask_reasons") or {})
        for reason, count in reason_counts.items():
            merged_reasons[reason] = int(merged_reasons.get(reason, 0) or 0) + int(count)
        side_metrics["prescore_market_mask_reasons"] = merged_reasons
        _emit_structured_event(
            "LIVE_PRESCORE_MARKET_MASK",
            {
                "side": side,
                "strategy_id": strategy_core_id(str(strategy_id)),
                "input": len(candidates),
                "kept": len(kept),
                "blocked": int(len(candidates) - len(kept)),
                "reasons": reason_counts,
                "sample_blocked": [
                    {"symbol": sym, "reason": snapshots[sym].get("prescore_market_mask_reason")}
                    for sym in snapshots
                    if not bool(snapshots[sym].get("prescore_market_mask_allowed"))
                ][:10],
            },
        )
    return kept, snapshots


def _live_entry_adverse_hourly_close_gate(runtime_config: Dict[str, Any]) -> Tuple[bool, float]:
    enabled_raw = os.environ.get(
        "EPM_ADVERSE_HOURLY_CLOSE_ENTRY_GATE_ENABLED",
        runtime_config.get("adverse_hourly_close_entry_gate_enabled", True),
    )
    enabled = str(enabled_raw).strip().lower() not in {"0", "false", "no", "off"}
    try:
        gate_bps = float(
            os.environ.get(
                "EPM_ADVERSE_HOURLY_CLOSE_ENTRY_GATE_BPS",
                runtime_config.get("adverse_hourly_close_entry_gate_bps", 150.0),
            )
        )
    except (TypeError, ValueError):
        gate_bps = 150.0
    return bool(enabled and gate_bps >= 0.0), float(gate_bps)


def _validate_policy_rank_reference_startup(
    *,
    policy_rank_reference_store: Optional[PolicyRankReferenceStore],
    require_cross_strategy_auction_rank: bool,
) -> None:
    """Fail early when live thresholds require a missing auction rank reference."""
    if not require_cross_strategy_auction_rank:
        return
    if policy_rank_reference_store is None:
        raise RuntimeError(
            "Policy rank reference guard failed: missing PolicyRankReferenceStore "
            "while cross-strategy auction rank thresholds are required."
        )
    manifest = policy_rank_reference_store.manifest
    auction = manifest.get("auction") if isinstance(manifest, Mapping) else None
    if not isinstance(auction, Mapping):
        raise RuntimeError(
            "Policy rank reference guard failed: rank_reference/manifest.json "
            "does not contain a cross-strategy auction reference."
        )
    score_col = str(auction.get("score_col") or "")
    rank_col = str(auction.get("rank_col") or "")
    if score_col != "calibrated_score" or rank_col != "normalized_rank_score":
        raise RuntimeError(
            "Policy rank reference guard failed: cross-strategy auction reference "
            f"uses score_col={score_col!r}, rank_col={rank_col!r}; expected "
            "calibrated_score -> normalized_rank_score."
        )
    probe = policy_rank_reference_store.lookup_auction(calibrated_score=0.5)
    if int(probe.n_rows) <= 0 or not str(probe.source or ""):
        raise RuntimeError(
            "Policy rank reference guard failed: cross-strategy auction reference "
            "could not be loaded or has zero finite scores."
        )


def _assert_policy_rank_threshold_source(decision: Mapping[str, Any]) -> None:
    """Ensure rank thresholds are compared in the same rank space as deployment."""
    if str(decision.get("threshold_space") or "rank_percentile") != "rank_percentile":
        return
    chain = decision.get("chain_results")
    if not isinstance(chain, Mapping):
        chain = {}
    auction_rank = _safe_float(
        chain.get("auction_rank_pct", decision.get("auction_rank_pct")), np.nan
    )
    if not np.isfinite(auction_rank):
        return
    threshold_rank = _safe_float(
        chain.get("threshold_rank_score", decision.get("threshold_rank_score")),
        np.nan,
    )
    threshold_source = str(
        chain.get(
            "threshold_rank_score_source",
            decision.get("threshold_rank_score_source", ""),
        )
        or ""
    )
    allowed_threshold_sources = {
        "cross_strategy_auction_reference",
        "fullscope_score_distribution_auction_reference_in_sample",
    }
    if threshold_source not in allowed_threshold_sources:
        raise RuntimeError(
            "Policy rank threshold guard failed: auction rank is available but "
            f"threshold_rank_score_source={threshold_source!r}; expected "
            f"one of {sorted(allowed_threshold_sources)!r}."
        )
    if not np.isfinite(threshold_rank) or not np.isclose(
        threshold_rank, auction_rank, rtol=0.0, atol=1e-12
    ):
        raise RuntimeError(
            "Policy rank threshold guard failed: threshold_rank_score does not "
            f"match auction_rank_pct ({threshold_rank!r} vs {auction_rank!r})."
        )


def _prediction_ledger_row(
    decision: Dict[str, Any],
    *,
    timestamp: Any,
    side: str,
    portfolio_decision: str,
    portfolio_reject_reason: Optional[str] = None,
    liquidity_reject_reason: Optional[str] = None,
    execution_snapshot: Optional[Dict[str, Any]] = None,
    was_traded: bool = False,
    trade_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a durable top-candidate audit row without mutating inference state."""
    chain = dict(decision.get("chain_results") or {})
    snap = dict(execution_snapshot or {})
    snap_details = snap.get("details") if isinstance(snap.get("details"), dict) else {}
    sizing = dict(chain.get("portfolio_rank_sizing") or {})
    trade = dict(trade_result or {})
    normalized_rank = _safe_float(
        chain.get(
            "normalized_rank_score",
            decision.get(
                "normalized_rank_score",
                decision.get("threshold_score", np.nan),
            ),
        )
    )
    final_threshold = _safe_float(
        chain.get("effective_threshold", decision.get("effective_threshold", np.nan))
    )
    portfolio_gate_info = (
        chain.get("portfolio_gate_after_liquidity")
        if isinstance(chain.get("portfolio_gate_after_liquidity"), dict)
        else chain.get("portfolio_gate")
        if isinstance(chain.get("portfolio_gate"), dict)
        else {}
    )
    portfolio_gate_rank = _safe_float(
        portfolio_gate_info.get("rank_score")
        if isinstance(portfolio_gate_info, dict)
        else np.nan,
        np.nan,
    )
    portfolio_gate_threshold = _safe_float(
        portfolio_gate_info.get("final_threshold")
        if isinstance(portfolio_gate_info, dict)
        else np.nan,
        final_threshold,
    )
    final_gate_rank = (
        portfolio_gate_rank
        if np.isfinite(portfolio_gate_rank)
        else _safe_float(snap.get("adjusted_rank_score"), normalized_rank)
    )
    final_gate_threshold = (
        portfolio_gate_threshold
        if np.isfinite(portfolio_gate_threshold)
        else final_threshold
    )
    order = trade.get("order") if isinstance(trade.get("order"), dict) else {}
    entry_notional_quote = _safe_float(
        trade.get("entry_notional_quote", trade.get("size")), np.nan
    )
    entry_fee_quote = _safe_float(trade.get("entry_fee_quote"), np.nan)
    entry_fee_bps = (
        float(entry_fee_quote) / max(abs(float(entry_notional_quote)), 1e-12) * 10000.0
        if np.isfinite(entry_fee_quote)
        and np.isfinite(entry_notional_quote)
        and abs(float(entry_notional_quote)) > 0.0
        else np.nan
    )
    snapshot_fee_bps = _safe_float(
        trade.get(
            "fee_bps",
            snap.get(
                "fee_bps",
                snap_details.get("fee_bps") if isinstance(snap_details, dict) else np.nan,
            ),
        ),
        np.nan,
    )
    ledger_fee_bps = (
        float(entry_fee_bps)
        if np.isfinite(entry_fee_bps)
        else (float(snapshot_fee_bps) if np.isfinite(snapshot_fee_bps) else np.nan)
    )
    policy_rank_reference_source = chain.get(
        "policy_rank_reference_source",
        decision.get("policy_rank_reference_source"),
    )
    auction_rank_reference_source = chain.get(
        "auction_rank_reference_source",
        decision.get("auction_rank_reference_source"),
    )
    policy_rank_reference_path = (
        Path(str(policy_rank_reference_source))
        if policy_rank_reference_source
        else None
    )
    auction_rank_reference_path = (
        Path(str(auction_rank_reference_source))
        if auction_rank_reference_source
        else None
    )
    row = {
        "timestamp": timestamp,
        "decision_ts": _diagnostic_timestamp(decision.get("decision_ts"), timestamp),
        "signal_bar_ts": _diagnostic_timestamp(
            decision.get("signal_bar_ts"), timestamp
        ),
        "signal_bar_close_ts": _diagnostic_timestamp(
            snap.get("signal_bar_close_ts", decision.get("signal_bar_close_ts"))
        ),
        "signal_close_to_decision_seconds": snap.get(
            "signal_close_to_decision_seconds",
            decision.get("signal_close_to_decision_seconds"),
        ),
        "signal_to_decision_seconds": snap.get(
            "signal_to_decision_seconds",
            decision.get("signal_to_decision_seconds"),
        ),
        "max_signal_close_to_entry_seconds": snap.get(
            "max_signal_close_to_entry_seconds",
            decision.get("max_signal_close_to_entry_seconds"),
        ),
        "stale_signal_age_gate_enabled": snap.get(
            "stale_signal_age_gate_enabled",
            decision.get("stale_signal_age_gate_enabled"),
        ),
        "stale_signal_age_gate_exceeded": snap.get(
            "stale_signal_age_gate_exceeded",
            decision.get("stale_signal_age_gate_exceeded"),
        ),
        "feature_source_max_ts": _diagnostic_timestamp(
            decision.get("feature_source_max_ts")
        ),
        "feature_available_ts": _diagnostic_timestamp(
            decision.get("feature_available_ts")
        ),
        "feature_contract_hash": decision.get("feature_contract_hash"),
        "feature_transform_contract_hash": decision.get(
            "feature_transform_contract_hash"
        ),
        "model_artifact_run_id": decision.get("model_artifact_run_id"),
        "policy_artifact_run_id": decision.get("policy_artifact_run_id"),
        "model_feature_audit_schema": chain.get("model_feature_audit_schema"),
        "model_feature_snapshot_hash": chain.get("model_feature_snapshot_hash"),
        "base_model_key": chain.get("base_model_key"),
        "meta_model_feature_key": chain.get("meta_model_feature_key"),
        "base_model_feature_count": chain.get("base_model_feature_count"),
        "meta_model_feature_count": chain.get("meta_model_feature_count"),
        "base_model_features_json": chain.get("base_model_features_json"),
        "meta_model_features_json": chain.get("meta_model_features_json"),
        "base_model_feature_values_json": chain.get(
            "base_model_feature_values_json"
        ),
        "meta_model_feature_values_json": chain.get(
            "meta_model_feature_values_json"
        ),
        "model_feature_value_sources_json": chain.get(
            "model_feature_value_sources_json"
        ),
        "model_feature_missing_json": chain.get("model_feature_missing_json"),
        "symbol": decision.get("symbol"),
        "side": side,
        "strategy_id": decision.get("strategy_id"),
        "meta_model_key": chain.get("meta_model_key", decision.get("meta_model_key")),
        "meta_head_hash": chain.get("meta_head_hash", decision.get("meta_head_hash")),
        "raw_prediction_score": decision.get("raw_score"),
        "base_pred": chain.get("base_pred"),
        "meta_pred": chain.get("meta_pred", decision.get("raw_score")),
        "calibrated_score": chain.get(
            "calibrated_score", decision.get("calibrated_score")
        ),
        "estimated_hit_rate": chain.get(
            "estimated_hit_rate", decision.get("estimated_hit_rate")
        ),
        "estimated_hit_rate_source": chain.get(
            "estimated_hit_rate_source", decision.get("estimated_hit_rate_source")
        ),
        "estimated_hit_rate_calibration_n": chain.get(
            "estimated_hit_rate_calibration_n",
            decision.get("estimated_hit_rate_calibration_n"),
        ),
        "estimated_ev_gross_return": chain.get(
            "estimated_ev_gross_return", decision.get("estimated_ev_gross_return")
        ),
        "estimated_ev_net_return": chain.get(
            "estimated_ev_net_return", decision.get("estimated_ev_net_return")
        ),
        "estimated_ev_cost_bps": chain.get(
            "estimated_ev_cost_bps", decision.get("estimated_ev_cost_bps")
        ),
        "estimated_ev_hit_rate": chain.get(
            "estimated_ev_hit_rate", decision.get("estimated_ev_hit_rate")
        ),
        "estimated_ev_source": chain.get(
            "estimated_ev_source", decision.get("estimated_ev_source")
        ),
        "estimated_ev_calibration_n": chain.get(
            "estimated_ev_calibration_n", decision.get("estimated_ev_calibration_n")
        ),
        "base_train_rank_pct": chain.get("base_train_rank_pct"),
        "meta_train_rank_pct": chain.get("meta_train_rank_pct"),
        "rank_score_source": chain.get(
            "rank_score_source", decision.get("rank_score_source")
        ),
        "policy_rank_pct": chain.get(
            "policy_rank_pct", decision.get("policy_rank_pct")
        ),
        "policy_rank_reference_n": chain.get(
            "policy_rank_reference_n", decision.get("policy_rank_reference_n")
        ),
        "policy_rank_reference_source": chain.get(
            "policy_rank_reference_source",
            decision.get("policy_rank_reference_source"),
        ),
        "policy_rank_reference_hash": (
            _artifact_file_hash(policy_rank_reference_path)
            if policy_rank_reference_path is not None
            else None
        ),
        "policy_rank_reference_mtime": (
            _artifact_file_mtime_iso(policy_rank_reference_path)
            if policy_rank_reference_path is not None
            else None
        ),
        "auction_rank_pct": chain.get(
            "auction_rank_pct", decision.get("auction_rank_pct")
        ),
        "auction_rank_reference_n": chain.get(
            "auction_rank_reference_n", decision.get("auction_rank_reference_n")
        ),
        "auction_rank_reference_source": chain.get(
            "auction_rank_reference_source",
            decision.get("auction_rank_reference_source"),
        ),
        "auction_rank_reference_hash": (
            _artifact_file_hash(auction_rank_reference_path)
            if auction_rank_reference_path is not None
            else None
        ),
        "auction_rank_reference_mtime": (
            _artifact_file_mtime_iso(auction_rank_reference_path)
            if auction_rank_reference_path is not None
            else None
        ),
        "auction_rank_score_source": chain.get(
            "auction_rank_score_source",
            decision.get("auction_rank_score_source"),
        ),
        "threshold_rank_score": chain.get(
            "threshold_rank_score", decision.get("threshold_rank_score")
        ),
        "threshold_rank_score_source": chain.get(
            "threshold_rank_score_source",
            decision.get("threshold_rank_score_source"),
        ),
        "historical_rank_pct": chain.get("meta_train_rank_pct"),
        "batch_rank_pct": decision.get("sizer_rank_percentile"),
        "normalized_rank_score": normalized_rank,
        "side_crowding_penalty": snap.get("side_crowding_penalty", 0.0),
        "strategy_crowding_penalty": snap.get("strategy_crowding_penalty", 0.0),
        "price_gap_penalty": snap.get("price_gap_penalty", 0.0),
        "adjusted_rank_score": snap.get("adjusted_rank_score", normalized_rank),
        "initial_rank_threshold": decision.get("rank_threshold"),
        "final_threshold": final_threshold,
        "final_gate_rank_score": final_gate_rank,
        "final_gate_threshold": final_gate_threshold,
        "final_gate_rank_score_source": (
            "portfolio_gate"
            if np.isfinite(portfolio_gate_rank)
            else "execution_adjusted_rank_score"
            if np.isfinite(_safe_float(snap.get("adjusted_rank_score"), np.nan))
            else "normalized_rank_score"
        ),
        "portfolio_gate_rank_score": portfolio_gate_info.get("rank_score")
        if isinstance(portfolio_gate_info, dict)
        else None,
        "portfolio_gate_initial_threshold": portfolio_gate_info.get("initial_threshold")
        if isinstance(portfolio_gate_info, dict)
        else None,
        "portfolio_gate_final_threshold": portfolio_gate_info.get("final_threshold")
        if isinstance(portfolio_gate_info, dict)
        else None,
        "portfolio_gate_threshold_viability_margin": snap.get(
            "threshold_viability_margin"
        ),
        "dynamic_performance_multiplier": chain.get(
            "dynamic_performance_multiplier",
            decision.get("dynamic_performance_multiplier"),
        ),
        "dynamic_performance_reason": chain.get(
            "dynamic_performance_reason",
            decision.get("dynamic_performance_reason"),
        ),
        "dynamic_performance_expected_hit_rate": chain.get(
            "dynamic_performance_expected_hit_rate",
            decision.get("dynamic_performance_expected_hit_rate"),
        ),
        "dynamic_performance_recent_hit_rate": chain.get(
            "dynamic_performance_recent_hit_rate",
            decision.get("dynamic_performance_recent_hit_rate"),
        ),
        "dynamic_performance_recent_n": chain.get(
            "dynamic_performance_recent_n",
            decision.get("dynamic_performance_recent_n"),
        ),
        "inference_drift_score": chain.get(
            "inference_drift_score", decision.get("inference_drift_score")
        ),
        "uncertainty_score": chain.get(
            "uncertainty_score", decision.get("uncertainty_score")
        ),
        "passed_rank_gate": bool(
            np.isfinite(final_gate_rank)
            and np.isfinite(final_gate_threshold)
            and final_gate_rank >= final_gate_threshold
        ),
        "portfolio_priority": decision.get("portfolio_priority"),
        "portfolio_state_snapshot_json": decision.get(
            "portfolio_state_snapshot_json"
        ),
        "portfolio_state_snapshot_hash": decision.get(
            "portfolio_state_snapshot_hash"
        ),
        "portfolio_state_snapshot_error": decision.get(
            "portfolio_state_snapshot_error"
        ),
        "open_positions_before_json": decision.get("open_positions_before_json"),
        "active_positions_before_json": decision.get(
            "active_positions_before_json",
            decision.get("open_positions_before_json"),
        ),
        "cooldowns_before_json": decision.get("cooldowns_before_json"),
        "recent_losing_trade_cooldown_state_json": decision.get(
            "recent_losing_trade_cooldown_state_json",
            decision.get("cooldowns_before_json"),
        ),
        "wallet_before": decision.get("wallet_before", sizing.get("wallet_value")),
        "open_notional_before": decision.get(
            "open_notional_before", sizing.get("open_notional")
        ),
        "available_wallet_before": decision.get("available_wallet_before"),
        "open_positions_before": decision.get(
            "open_positions_before",
            (chain.get("portfolio_gate") or {}).get(
                "n_positions_before",
                (chain.get("portfolio_gate_after_liquidity") or {}).get(
                    "n_positions_before"
                ),
            )
            if isinstance(chain, dict)
            else None,
        ),
        "open_positions_before_count": decision.get(
            "open_positions_before_count",
            decision.get("open_positions_before"),
        ),
        "wallet_value": sizing.get("wallet_value"),
        "open_notional": sizing.get("open_notional"),
        "position_size_before_liquidity": sizing.get(
            "size_before_liquidity",
            snap.get("position_size_before_liquidity"),
        ),
        "position_size_after_liquidity": sizing.get(
            "size_after_liquidity",
            snap.get("position_size_after_liquidity"),
        ),
        "portfolio_decision": portfolio_decision,
        "portfolio_reject_reason": portfolio_reject_reason,
        "prescore_market_mask_enabled": snap.get(
            "prescore_market_mask_enabled", chain.get("prescore_market_mask_enabled")
        ),
        "prescore_market_mask_allowed": snap.get(
            "prescore_market_mask_allowed", chain.get("prescore_market_mask_allowed")
        ),
        "prescore_market_mask_reason": snap.get(
            "prescore_market_mask_reason", chain.get("prescore_market_mask_reason")
        ),
        "prescore_signal_price": snap.get(
            "prescore_signal_price", chain.get("prescore_signal_price")
        ),
        "prescore_raw_signal_close": snap.get(
            "prescore_raw_signal_close", chain.get("prescore_raw_signal_close")
        ),
        "prescore_raw_signal_close_ts": snap.get(
            "prescore_raw_signal_close_ts",
            chain.get("prescore_raw_signal_close_ts"),
        ),
        "prescore_raw_signal_volume": snap.get(
            "prescore_raw_signal_volume", chain.get("prescore_raw_signal_volume")
        ),
        "prescore_raw_signal_volume_ts": snap.get(
            "prescore_raw_signal_volume_ts",
            chain.get("prescore_raw_signal_volume_ts"),
        ),
        "prescore_raw_signal_close_reference_gap_bps": snap.get(
            "prescore_raw_signal_close_reference_gap_bps",
            chain.get("prescore_raw_signal_close_reference_gap_bps"),
        ),
        "prescore_raw_signal_close_reference_source": snap.get(
            "prescore_raw_signal_close_reference_source",
            chain.get("prescore_raw_signal_close_reference_source"),
        ),
        "prescore_signal_bar_close_ts": snap.get(
            "prescore_signal_bar_close_ts",
            chain.get("prescore_signal_bar_close_ts"),
        ),
        "prescore_signal_close_to_decision_seconds": snap.get(
            "prescore_signal_close_to_decision_seconds",
            chain.get("prescore_signal_close_to_decision_seconds"),
        ),
        "prescore_max_signal_close_to_entry_seconds": snap.get(
            "prescore_max_signal_close_to_entry_seconds",
            chain.get("prescore_max_signal_close_to_entry_seconds"),
        ),
        "prescore_stale_signal_age_gate_exceeded": snap.get(
            "prescore_stale_signal_age_gate_exceeded",
            chain.get("prescore_stale_signal_age_gate_exceeded"),
        ),
        "prescore_oi_key": snap.get("prescore_oi_key", chain.get("prescore_oi_key")),
        "prescore_oi_value": snap.get(
            "prescore_oi_value", chain.get("prescore_oi_value")
        ),
        "prescore_oi_ts": snap.get("prescore_oi_ts", chain.get("prescore_oi_ts")),
        "prescore_oi_age_hours": snap.get(
            "prescore_oi_age_hours", chain.get("prescore_oi_age_hours")
        ),
        "prescore_ticker_bid": snap.get(
            "prescore_ticker_bid", chain.get("prescore_ticker_bid")
        ),
        "prescore_ticker_ask": snap.get(
            "prescore_ticker_ask", chain.get("prescore_ticker_ask")
        ),
        "prescore_ticker_mid": snap.get(
            "prescore_ticker_mid", chain.get("prescore_ticker_mid")
        ),
        "prescore_ticker_last": snap.get(
            "prescore_ticker_last", chain.get("prescore_ticker_last")
        ),
        "prescore_ticker_spread_bps": snap.get(
            "prescore_ticker_spread_bps",
            chain.get("prescore_ticker_spread_bps"),
        ),
        "prescore_max_spread_bps": snap.get(
            "prescore_max_spread_bps", chain.get("prescore_max_spread_bps")
        ),
        "prescore_ticker_spread_weight": snap.get(
            "prescore_ticker_spread_weight",
            chain.get("prescore_ticker_spread_weight"),
        ),
        "prescore_ticker_age_seconds": snap.get(
            "prescore_ticker_age_seconds",
            chain.get("prescore_ticker_age_seconds"),
        ),
        "prescore_ticker_fetch_latency_seconds": snap.get(
            "prescore_ticker_fetch_latency_seconds",
            chain.get("prescore_ticker_fetch_latency_seconds"),
        ),
        "prescore_ticker_reject_reason": snap.get(
            "prescore_ticker_reject_reason",
            chain.get("prescore_ticker_reject_reason"),
        ),
        "prescore_orderbook_side": snap.get(
            "prescore_orderbook_side", chain.get("prescore_orderbook_side")
        ),
        "prescore_orderbook_capacity_quote_within_slippage": snap.get(
            "prescore_orderbook_capacity_quote_within_slippage",
            chain.get("prescore_orderbook_capacity_quote_within_slippage"),
        ),
        "prescore_orderbook_intended_quote_size": snap.get(
            "prescore_orderbook_intended_quote_size",
            chain.get("prescore_orderbook_intended_quote_size"),
        ),
        "prescore_orderbook_depth_weight": snap.get(
            "prescore_orderbook_depth_weight",
            chain.get("prescore_orderbook_depth_weight"),
        ),
        "prescore_liquidity_capacity_weight": snap.get(
            "prescore_liquidity_capacity_weight",
            chain.get("prescore_liquidity_capacity_weight"),
        ),
        "prescore_orderbook_slippage_bps": snap.get(
            "prescore_orderbook_slippage_bps",
            chain.get("prescore_orderbook_slippage_bps"),
        ),
        "prescore_orderbook_reject_reason": snap.get(
            "prescore_orderbook_reject_reason",
            chain.get("prescore_orderbook_reject_reason"),
        ),
        "ticker_bid": snap.get("ticker_bid", snap.get("bid")),
        "ticker_ask": snap.get("ticker_ask", snap.get("ask")),
        "ticker_mid": snap.get("ticker_mid", snap.get("mid")),
        "ticker_last": snap.get("ticker_last", snap.get("last")),
        "ticker_request_started_at": snap_details.get("ticker_request_started_at"),
        "ticker_received_at": snap_details.get("ticker_received_at"),
        "ticker_fetch_latency_seconds": snap_details.get(
            "ticker_fetch_latency_seconds"
        ),
        "exchange_ticker_timestamp": snap_details.get("exchange_ticker_timestamp"),
        "exchange_ticker_age_seconds": snap_details.get(
            "exchange_ticker_age_seconds"
        ),
        "spread_bps": snap.get("spread_bps", snap.get("ticker_spread_bps")),
        "ticker_spread_bps": snap.get("ticker_spread_bps", snap.get("spread_bps")),
        "orderbook_side": snap.get("orderbook_side"),
        "best_touch": snap.get("best_touch"),
        "max_walk_price": snap.get("max_walk_price"),
        "intended_quote_size": snap.get("intended_quote_size"),
        "orderbook_capacity_quote_within_slippage": snap.get(
            "orderbook_capacity_quote_within_slippage"
        ),
        "max_orderbook_slippage_bps": snap.get("max_orderbook_slippage_bps"),
        "spread_weight": snap.get("spread_weight"),
        "depth_weight": snap.get("depth_weight"),
        "half_spread_bps": snap_details.get("half_spread_bps"),
        "effective_orderbook_slippage_cap_bps": snap_details.get(
            "effective_orderbook_slippage_cap_bps"
        ),
        "max_entry_friction_bps": snap_details.get("max_entry_friction_bps"),
        "entry_friction_formula": snap_details.get("entry_friction_formula"),
        "entry_friction_gate": snap_details.get("entry_friction_gate"),
        "theoretical_entry_price": snap.get("theoretical_entry_price"),
        "policy_entry_price": snap.get("policy_entry_price"),
        "expected_entry_price": snap.get("expected_entry_price"),
        "expected_fill_price": snap.get("expected_fill_price"),
        "expected_fill_slippage_bps": snap.get("expected_fill_slippage_bps"),
        "orderbook_slippage_bps": snap.get(
            "orderbook_slippage_bps",
            snap.get("expected_fill_slippage_bps"),
        ),
        "slippage_bps": snap.get(
            "slippage_bps",
            snap.get("expected_fill_slippage_bps"),
        ),
        "entry_gap_bps": snap.get(
            "entry_gap_bps",
            snap.get("adverse_signal_gap_bps"),
        ),
        "entry_slippage_proxy_bps": snap.get(
            "entry_slippage_proxy_bps",
            snap.get("expected_fill_slippage_bps"),
        ),
        "hourly_close_to_latest_decision_price_bps": trade.get(
            "hourly_close_to_latest_decision_price_bps",
            snap.get("hourly_close_to_latest_decision_price_bps"),
        ),
        "decision_price_to_fill_bps": trade.get(
            "decision_price_to_fill_bps",
            snap.get("decision_price_to_fill_bps"),
        ),
        "latest_decision_price": trade.get(
            "latest_decision_price", snap.get("latest_decision_price")
        ),
        "entry_price_attribution_schema": trade.get(
            "entry_price_attribution_schema",
            snap.get("entry_price_attribution_schema"),
        ),
        "spread_proxy_bps": trade.get(
            "spread_proxy_bps",
            snap.get("spread_proxy_bps", snap.get("ticker_spread_bps")),
        ),
        "orderbook_live_slippage_bps": trade.get(
            "orderbook_live_slippage_bps",
            snap.get(
                "orderbook_live_slippage_bps",
                snap.get("orderbook_slippage_bps", snap.get("expected_fill_slippage_bps")),
            ),
        ),
        "adverse_signal_gap_bps": snap.get("adverse_signal_gap_bps"),
        "expected_total_entry_friction_bps": snap.get(
            "expected_total_entry_friction_bps"
        ),
        "expected_friction_drag_bps": snap.get(
            "expected_friction_drag_bps",
            snap.get("expected_total_entry_friction_bps"),
        ),
        "ev_haircut_bps": snap.get("ev_haircut_bps"),
        "ev_haircut_raw_live_entry_friction_bps": snap.get(
            "ev_haircut_raw_live_entry_friction_bps"
        ),
        "ev_haircut_observed_spread_bps": snap.get(
            "ev_haircut_observed_spread_bps"
        ),
        "ev_haircut_observed_half_spread_bps": snap.get(
            "ev_haircut_observed_half_spread_bps"
        ),
        "ev_haircut_spread_baseline_bps": snap.get(
            "ev_haircut_spread_baseline_bps"
        ),
        "ev_haircut_spread_baseline_source": snap.get(
            "ev_haircut_spread_baseline_source"
        ),
        "ev_haircut_half_spread_baseline_bps": snap.get(
            "ev_haircut_half_spread_baseline_bps"
        ),
        "ev_haircut_spread_excess_bps": snap.get(
            "ev_haircut_spread_excess_bps"
        ),
        "ev_haircut_orderbook_slippage_bps": snap.get(
            "ev_haircut_orderbook_slippage_bps"
        ),
        "ev_haircut_adverse_signal_gap_bps": snap.get(
            "ev_haircut_adverse_signal_gap_bps"
        ),
        "ev_haircut_observed_delay_slippage_bps": snap.get(
            "ev_haircut_observed_delay_slippage_bps"
        ),
        "ev_haircut_delay_slippage_baseline_bps": snap.get(
            "ev_haircut_delay_slippage_baseline_bps"
        ),
        "ev_haircut_delay_slippage_excess_bps": snap.get(
            "ev_haircut_delay_slippage_excess_bps"
        ),
        "ev_haircut_contract": snap.get("ev_haircut_contract"),
        "ev_adjusted_entry_friction_bps": snap.get(
            "ev_adjusted_entry_friction_bps"
        ),
        "ev_adjusted_net_return_before_friction": snap.get(
            "ev_adjusted_net_return_before_friction"
        ),
        "ev_adjusted_net_return_after_friction": snap.get(
            "ev_adjusted_net_return_after_friction"
        ),
        "ev_adjusted_calibrated_score": snap.get(
            "ev_adjusted_calibrated_score"
        ),
        "ev_adjusted_rank_score": snap.get("ev_adjusted_rank_score"),
        "ev_adjusted_source": snap.get("ev_adjusted_source"),
        "entry_delay_effect_bps": trade.get("entry_delay_effect_bps"),
        "entry_delay_adverse_bps": trade.get("entry_delay_adverse_bps"),
        "entry_delay_abs_bps": trade.get("entry_delay_abs_bps"),
        "decision_to_entry_seconds": trade.get("decision_to_entry_seconds"),
        "signal_to_entry_seconds": trade.get("signal_to_entry_seconds"),
        "gross_to_net_friction_drag_bps": trade.get(
            "gross_to_net_friction_drag_bps"
        ),
        "entry_notional_quote": entry_notional_quote,
        "base_amount": trade.get("base_amount"),
        "entry_fee_quote": trade.get("entry_fee_quote"),
        "entry_fee_cost": trade.get("entry_fee_cost"),
        "entry_fee_currency": trade.get("entry_fee_currency"),
        "entry_fee_source": trade.get("entry_fee_source"),
        "entry_fee_bps": ledger_fee_bps,
        "fee_bps": ledger_fee_bps,
        "liquidity_capacity_weight": snap.get("liquidity_capacity_weight"),
        "liquidity_reject_reason": liquidity_reject_reason,
        "signal_price": snap.get("signal_price"),
        "raw_signal_close": snap.get("raw_signal_close"),
        "raw_signal_close_ts": snap.get("raw_signal_close_ts"),
        "raw_signal_volume": snap.get("raw_signal_volume"),
        "raw_signal_volume_ts": snap.get("raw_signal_volume_ts"),
        "raw_signal_close_unreliable": snap.get("raw_signal_close_unreliable"),
        "raw_signal_close_unreliable_reason": snap.get(
            "raw_signal_close_unreliable_reason"
        ),
        "raw_signal_close_reference_gap_bps": snap.get(
            "raw_signal_close_reference_gap_bps"
        ),
        "raw_signal_close_reference_price": snap.get(
            "raw_signal_close_reference_price"
        ),
        "raw_signal_close_reference_source": snap.get(
            "raw_signal_close_reference_source"
        ),
        "raw_signal_close_reference_ts": snap.get("raw_signal_close_reference_ts"),
        "decision_mid": snap.get(
            "decision_mid", snap.get("ticker_mid", snap.get("mid"))
        ),
        "signal_gap_bps": snap.get("signal_gap_bps"),
        "max_chase_bps": snap.get("max_chase_bps"),
        "entry_limit_price": snap.get("entry_limit_price"),
        "limit_price": snap.get("limit_price"),
        "was_traded": bool(was_traded),
        "position_id": trade.get("position_id")
        or trade.get("order_id")
        or order.get("id"),
        "order_id": trade.get("order_id") or order.get("id"),
        "entry_price_expected": snap.get("expected_fill_price"),
        "entry_price_actual": trade.get("realized_entry_price"),
        "realized_entry_price": trade.get("realized_entry_price"),
        "realized_exit_price": trade.get("realized_exit_price"),
        "realized_fee_bps": trade.get("realized_fee_bps", ledger_fee_bps),
        "realized_funding_bps": trade.get("realized_funding_bps"),
        "realized_borrow_bps": trade.get("realized_borrow_bps"),
        "outcome_status": None,
        "tp_hit": None,
        "sl_hit": None,
        "ambiguous_both_hit": None,
        "resolved_at": None,
    }
    lgbm_diagnostics = chain.get("lgbm_diagnostics")
    if not isinstance(lgbm_diagnostics, dict):
        lgbm_diagnostics = {}
    meta_lgbm_diagnostics = chain.get("meta_lgbm_diagnostics")
    if not isinstance(meta_lgbm_diagnostics, dict):
        meta_lgbm_diagnostics = lgbm_diagnostics
    base_lgbm_diagnostics = chain.get("base_lgbm_diagnostics")
    if not isinstance(base_lgbm_diagnostics, dict):
        base_lgbm_diagnostics = {}
    diag_keys = tuple(
        dict.fromkeys(
            list(LGBM_INTERNAL_METRIC_FEATURE_NAMES)
            + [
                "feature_drift_psi_core",
                "feature_drift_ks_core",
                "feature_drift_cov_shift",
                "regime_centroid_similarity_train",
                "rare_leaf_fraction",
                "leaf_count_p10",
                "leaf_count_min",
                "leaf_weight_p10",
                "contrib_top1_abs_share",
                "contrib_top3_abs_share",
                "contrib_entropy",
                "contrib_balance",
                "num_material_contrib_features",
                "prob_uncertainty",
            ]
        )
    )
    for diag_key in diag_keys:
        if diag_key in chain or diag_key in meta_lgbm_diagnostics:
            row[diag_key] = chain.get(diag_key, meta_lgbm_diagnostics.get(diag_key))
        if diag_key in meta_lgbm_diagnostics:
            row[f"meta_lgbm_{diag_key}"] = meta_lgbm_diagnostics.get(diag_key)
        if diag_key in base_lgbm_diagnostics:
            row[f"base_lgbm_{diag_key}"] = base_lgbm_diagnostics.get(diag_key)
    return row


def _historical_score_paths(
    data_root: str,
    run_id: str,
    strategy_id: str,
    *,
    kind: str,
) -> List[Path]:
    """Return candidate OOF artifact paths for empirical rank normalization."""
    base = Path(data_root) / "artifacts" / run_id
    sid = str(strategy_id or "")
    core = strategy_core_id(sid)
    if kind == "meta":
        names = [
            f"meta_oof_{sid}_clf.parquet",
            f"meta_oof_{core}_clf.parquet",
        ]
        paths = [base / "meta_oof" / name for name in names if name]
        paths.extend(sorted((base / "meta_oof").glob(f"meta_oof_*{core}_clf.parquet")))
        return paths
    if kind == "base":
        return sorted((base / "oof").glob(f"oof_{core}_H*.parquet"))
    return []


def _load_historical_score_distribution(
    data_root: str,
    run_id: str,
    strategy_id: str,
    *,
    kind: str,
) -> np.ndarray:
    """Load sorted finite OOF scores used to map live predictions to pct ranks."""
    key = (str(data_root), str(run_id), str(strategy_id), str(kind))
    cached = _HISTORICAL_SCORE_RANK_CACHE.get(key)
    if cached is not None:
        return cached

    columns = (
        ["oof_meta_clf", "oof_pred", "oof_p_move"]
        if kind == "meta"
        else ["oof_prob_uncertainty_weighted", "oof_prob", "oof_prob_ebm_raw"]
    )
    for path in _historical_score_paths(data_root, run_id, strategy_id, kind=kind):
        if not path.exists():
            continue
        try:
            df = pd.read_parquet(path, columns=None)
            score_col = next((col for col in columns if col in df.columns), None)
            if score_col is None:
                continue
            values = pd.to_numeric(df[score_col], errors="coerce").to_numpy(
                dtype=np.float32,
                copy=False,
            )
            values = np.sort(values[np.isfinite(values)])
            if values.size:
                _HISTORICAL_SCORE_RANK_CACHE[key] = values
                tprint(
                    f"Loaded historical {kind} rank distribution for "
                    f"{strategy_core_id(strategy_id)}: n={values.size} "
                    f"col={score_col} path={path.name}"
                )
                return values
        except Exception as exc:
            tprint(f"Could not load historical {kind} scores from {path}: {exc}")

    empty = np.asarray([], dtype=np.float32)
    _HISTORICAL_SCORE_RANK_CACHE[key] = empty
    return empty


def _historical_prediction_rank_pct(
    score: Any,
    *,
    data_root: str,
    run_id: str,
    strategy_id: str,
    kind: str,
) -> float:
    """Map a live score to its empirical percentile among historical OOF scores."""
    try:
        value = float(score)
    except (TypeError, ValueError):
        return float("nan")
    if not np.isfinite(value):
        return float("nan")
    values = _load_historical_score_distribution(
        data_root,
        run_id,
        strategy_id,
        kind=kind,
    )
    if values.size == 0:
        return float("nan")
    return float(np.searchsorted(values, value, side="right") / float(values.size))


# Default symbols to trade
DEFAULT_SYMBOLS = [
    "BTC/USDC",
    "ETH/USDC",
    "BNB/USDC",
    "SOL/USDC",
    "XRP/USDC",
    "ADA/USDC",
    "DOGE/USDC",
    "AVAX/USDC",
    "DOT/USDC",
    "MATIC/USDC",
]


def _build_market_snapshot(
    symbol: str,
    panel: Dict[str, pd.DataFrame],
    feats: Dict[str, pd.DataFrame],
) -> Dict[str, Any]:
    """Build a compact market-data dict for logging."""
    snapshot: Dict[str, Any] = {}
    close = panel.get("close")
    volume = panel.get("volume")
    if isinstance(close, pd.DataFrame) and symbol in close.columns:
        s = close[symbol].dropna()
        if not s.empty:
            snapshot["close"] = float(s.iloc[-1])
    if isinstance(volume, pd.DataFrame) and symbol in volume.columns:
        s = volume[symbol].dropna()
        if not s.empty:
            snapshot["volume"] = float(s.iloc[-1])
    for feat_name in [
        "ret24h",
        "range_12h_pct",
        "volatility_zscore",
        "vol_zscore",
        "G_VOL",
        "G_TREND",
        "G_VOLUME",
        "vol_z",
        "trend_pct",
        "trend",
        "entropy",
        "vol_of_vol",
        "kurtosis",
        "jump_frequency",
        "funding_cost",
        "borrow_cost",
        "mkt_rv_ratio",
    ]:
        feat_df = feats.get(feat_name)
        if isinstance(feat_df, pd.DataFrame) and symbol in feat_df.columns:
            s = feat_df[symbol].dropna()
            if not s.empty:
                snapshot[feat_name] = float(s.iloc[-1])
    return snapshot


def _build_executor_bucket_params(config: Dict[str, Any]) -> Dict[str, Any]:
    """Build runtime params with stop policy isolated from blended buckets."""
    model_bundle = config.get("model_bundle", {}) or {}
    full_state = config.get("full_state", {}) or {}
    ridge_weights = model_bundle.get("ridge_weights", {}) or {}
    params_per_bucket = ridge_weights.get("params_per_bucket", {}) or {}
    bucket_params = (
        dict(params_per_bucket)
        if params_per_bucket
        else dict(full_state.get("bucket_params", {}) or {})
    )
    data_root = str(config.get("data_root", "data"))
    policy_run_id = str(
        config.get("policy_artifact_run_id") or config.get("run_id", "")
    )
    stop_params = load_simple_policy_stop_params_by_strategy(
        data_root,
        policy_run_id,
    )
    if not stop_params and policy_run_id != str(config.get("run_id", "")):
        stop_params = load_simple_policy_stop_params_by_strategy(
            data_root,
            str(config.get("run_id", "")),
        )
    bucket_params["simple_policy_stop_params_by_strategy"] = stop_params
    return bucket_params


def _attach_runtime_bucket_params(config: Dict[str, Any]) -> Dict[str, Any]:
    """Attach policy-optimiser params to runtime model config and return them."""
    bucket_params = _build_executor_bucket_params(config)
    model_bundle = config.setdefault("model_bundle", {})
    if isinstance(model_bundle, dict):
        model_bundle["bucket_params"] = bucket_params
    full_state = config.setdefault("full_state", {})
    if isinstance(full_state, dict):
        full_state["bucket_params"] = bucket_params
    return bucket_params


def _resolve_live_feature_source_run_id(config: Dict[str, Any]) -> Optional[str]:
    run_ids = _resolve_live_feature_source_run_ids(config)
    return run_ids[0] if run_ids else None


def _resolve_live_feature_source_run_ids(config: Dict[str, Any]) -> List[str]:
    """Return the selected-feature source run for live feature fallback.

    The live rolling transformed-feature cache is keyed by the active run and
    exact feature contract. Offline selected-feature lookup is only a fallback
    for missing historical rows, so it must follow explicit artifact/source
    provenance instead of stale run-specific defaults.
    """

    parity_contract = (
        config.get("training_live_parity_contract")
        if isinstance(config.get("training_live_parity_contract"), dict)
        else {}
    )
    feature_source = (
        parity_contract.get("feature_source")
        if isinstance(parity_contract.get("feature_source"), dict)
        else {}
    )
    values: List[str] = []
    for value in (
        config.get("live_feature_source_run_ids"),
        config.get("feature_source_run_ids"),
        config.get("live_feature_source_run_id"),
        config.get("feature_source_run_id"),
        config.get("artifact_source_run_id"),
        os.getenv("EPM_LIVE_FEATURE_SOURCE_RUN_ID"),
        os.getenv("EPM_FEATURE_SOURCE_RUN_ID"),
        os.getenv("EPM_ARTIFACT_SOURCE_RUN_ID"),
        os.getenv("EPM_SOURCE_RUN_ID"),
        parity_contract.get("feature_sources"),
        feature_source.get("run_id"),
    ):
        values.extend(_coerce_feature_source_run_ids(value))
    deduped: List[str] = []
    for value in values:
        if value and value not in deduped:
            deduped.append(value)
    return deduped


def _model_feature_offline_cache_enabled(config: Dict[str, Any]) -> bool:
    """Return whether live model features should read the selected-feature handoff.

    Model scoring should use the selected-feature store by default, because the
    deployed score is only parity-safe when missing/non-finite values match the
    training/OOS policy handoff. Operators can explicitly disable this for live
    recompute audits.
    """

    if not isinstance(config, dict):
        return True
    explicit = config.get("live_model_feature_offline_cache_enabled")
    if explicit is not None:
        return str(explicit).strip().lower() not in {"0", "false", "no", "off"}
    return True


def _force_shadow_entry_for_integration(executor: Any) -> bool:
    """Return True only for explicit shadow-mode integration smoke passes."""
    if getattr(executor, "mode", None) != "shadow":
        return False
    cfg = getattr(executor, "config", {}) or {}
    if bool(cfg.get("force_shadow_entry_for_integration", False)):
        return True
    return str(os.getenv("EPM_FORCE_SHADOW_ENTRY", "")).strip().lower() in {
        "1",
        "true",
        "yes",
    }


def _env_flag(name: str, default: bool = False) -> bool:
    """Return a bool from common environment flag spellings."""
    value = os.getenv(name)
    if value is None:
        return bool(default)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _ignore_market_kill_switch_for_reconciliation(config: Mapping[str, Any]) -> bool:
    """Allow shadow/live-test reconciliation to observe, not block, market halts."""
    mode = str((config or {}).get("mode", "")).strip().lower()
    if mode not in {"shadow", "live-test", "live_test", "livetest"}:
        return False
    return bool(
        (config or {}).get("shadow_ignore_market_kill_switch", False)
        or _env_flag("EPM_SHADOW_IGNORE_MARKET_KILL_SWITCH", False)
    )


def _allow_model_feature_tail_recompute_for_reconciliation(
    config: Mapping[str, Any],
) -> bool:
    """Allow shadow/live-test reconciliation to seed current model feature cache."""
    mode = str((config or {}).get("mode", "")).strip().lower()
    if mode not in {"shadow", "live-test", "live_test", "livetest"}:
        return False
    if "live_model_feature_tail_recompute_enabled" in (config or {}):
        return bool((config or {}).get("live_model_feature_tail_recompute_enabled"))
    return _env_flag("EPM_SHADOW_MODEL_FEATURE_TAIL_RECOMPUTE", True)


def _build_live_feature_runtime_cfg(
    *,
    config: Mapping[str, Any],
    accepted_strategies: Optional[Iterable[str]],
    policy_selection_rules: Mapping[str, Any],
    latest_closed_hour: pd.Timestamp,
    hourly_refresh_updates: int,
) -> Dict[str, Any]:
    """Build the feature runtime config shared by prewarm and scoring."""
    feature_runtime_cfg = {
        **dict(config.get("runtime_cfg") or get_runtime_cfg()),
        "data_root": str(config.get("live_data_root") or config["data_root"]),
        "artifact_data_root": str(config["data_root"]),
        "offline_feature_data_root": str(config["data_root"]),
        "live_data_root": str(config.get("live_data_root") or config["data_root"]),
        "accepted_strategy_ids": sorted(accepted_strategies or []),
        "policy_selection_rules": dict(policy_selection_rules or {}),
    }
    live_feature_state_dir = (
        Path(str(config["data_root"]))
        / "artifacts"
        / str(config["run_id"])
        / "live_state"
    )
    feature_runtime_cfg.setdefault(
        "live_feature_snapshot_cache_dir",
        str(live_feature_state_dir / "feature_cache"),
    )
    feature_runtime_cfg.setdefault("live_feature_snapshot_cache_enabled", True)
    feature_runtime_cfg.setdefault("live_feature_rolling_cache_enabled", True)
    feature_runtime_cfg.setdefault(
        "live_feature_rolling_cache_cross_key_fallback_enabled", True
    )
    feature_runtime_cfg.setdefault(
        "live_feature_rolling_cache_model_superset_for_mask_enabled", True
    )
    feature_runtime_cfg.setdefault(
        "live_feature_rolling_cache_latest_only_read_enabled", True
    )
    feature_runtime_cfg.setdefault("live_feature_memory_cache_enabled", True)
    feature_runtime_cfg.setdefault("live_raw_rolling_state_enabled", True)
    feature_runtime_cfg.setdefault(
        "live_raw_rolling_state_path",
        str(live_feature_state_dir / "raw_rolling_state.npz"),
    )
    feature_runtime_cfg.setdefault("live_causal_transform_state_enabled", True)
    feature_runtime_cfg.setdefault(
        "live_causal_transform_state_path",
        str(live_feature_state_dir / "causal_transform_state.npz"),
    )
    feature_runtime_cfg.setdefault(
        "live_model_feature_auto_sync_blocking",
        _env_flag("EPM_LIVE_MODEL_FEATURE_AUTO_SYNC_BLOCKING", False),
    )
    if _allow_model_feature_tail_recompute_for_reconciliation(config):
        feature_runtime_cfg["live_model_feature_tail_recompute_enabled"] = True
    live_feature_source_run_ids = _resolve_live_feature_source_run_ids(dict(config))
    live_feature_source_run_id = (
        live_feature_source_run_ids[0] if live_feature_source_run_ids else None
    )
    if live_feature_source_run_ids:
        feature_runtime_cfg["live_feature_source_run_ids"] = list(
            live_feature_source_run_ids
        )
    if live_feature_source_run_id:
        feature_runtime_cfg["live_feature_source_run_id"] = str(
            live_feature_source_run_id
        )
    if hourly_refresh_updates > 0:
        feature_runtime_cfg["live_feature_cache_raw_refresh_token"] = (
            f"{pd.Timestamp(latest_closed_hour).isoformat()}:{hourly_refresh_updates}"
        )
    return feature_runtime_cfg


def _live_warmup_state_fail_closed(config: Mapping[str, Any]) -> bool:
    mode = str((config or {}).get("mode", "")).strip().lower()
    default = mode in _LIVE_PRESCORE_MARKET_MASK_MODES
    return _runtime_flag(
        config,
        "live_warmup_state_fail_closed",
        "EPM_LIVE_WARMUP_STATE_FAIL_CLOSED",
        default,
    )


def _state_file_health(
    path_raw: Any, *, now: pd.Timestamp, max_age_hours: float
) -> Dict[str, Any]:
    path = Path(str(path_raw)) if path_raw else None
    out: Dict[str, Any] = {
        "path": str(path) if path is not None else "",
        "exists": False,
        "exact_exists": False,
        "mtime": None,
        "age_hours": np.nan,
        "hashed_count": 0,
        "latest_hashed_path": "",
        "latest_hashed_mtime": None,
        "latest_hashed_age_hours": np.nan,
        "ok": False,
        "reason": "missing_path",
    }
    if path is None:
        return out
    if not path.exists():
        hashed_candidates: List[Path] = []
        try:
            if path.parent.exists():
                if path.suffix:
                    pattern = f"{path.stem}.*{path.suffix}"
                else:
                    pattern = f"{path.name}.*"
                hashed_candidates = [
                    candidate
                    for candidate in path.parent.glob(pattern)
                    if candidate.is_file() and candidate.name != path.name
                ]
        except Exception as exc:
            out["reason"] = f"inventory_scan_failed:{type(exc).__name__}"
            return out
        out["hashed_count"] = len(hashed_candidates)
        if not hashed_candidates:
            out["reason"] = "missing_file"
            return out
        try:
            latest_path = max(
                hashed_candidates, key=lambda candidate: candidate.stat().st_mtime
            )
            mtime = pd.Timestamp(latest_path.stat().st_mtime, unit="s", tz="UTC")
            age_hours = max(
                float((pd.Timestamp(now) - mtime).total_seconds()) / 3600.0,
                0.0,
            )
            fresh = bool(np.isfinite(age_hours) and age_hours <= float(max_age_hours))
            out.update(
                {
                    "exists": True,
                    "mtime": mtime.isoformat(),
                    "age_hours": age_hours,
                    "latest_hashed_path": str(latest_path),
                    "latest_hashed_mtime": mtime.isoformat(),
                    "latest_hashed_age_hours": age_hours,
                    "ok": fresh,
                    "reason": (
                        "ok_hashed_state_inventory"
                        if fresh
                        else "stale_hashed_state_inventory"
                    ),
                }
            )
        except Exception as exc:
            out["reason"] = f"inventory_stat_failed:{type(exc).__name__}"
        return out
    try:
        mtime = pd.Timestamp(path.stat().st_mtime, unit="s", tz="UTC")
        age_hours = max(
            float((pd.Timestamp(now) - mtime).total_seconds()) / 3600.0,
            0.0,
        )
        out.update(
            {
                "exists": True,
                "exact_exists": True,
                "mtime": mtime.isoformat(),
                "age_hours": age_hours,
                "ok": bool(
                    np.isfinite(age_hours) and age_hours <= float(max_age_hours)
                ),
                "reason": (
                    "ok"
                    if np.isfinite(age_hours) and age_hours <= float(max_age_hours)
                    else "stale_file"
                ),
            }
        )
    except Exception as exc:
        out["reason"] = f"stat_failed:{type(exc).__name__}"
    return out


def _latest_rolling_meta_health(
    feature_runtime_cfg: Mapping[str, Any],
    *,
    latest_closed_hour: pd.Timestamp,
    now: pd.Timestamp,
    max_age_hours: float,
) -> Dict[str, Any]:
    root = Path(
        str(
            feature_runtime_cfg.get("live_feature_snapshot_cache_dir")
            or feature_runtime_cfg.get("live_feature_cache_dir")
            or ""
        )
    )
    out: Dict[str, Any] = {
        "root": str(root) if str(root) else "",
        "path": "",
        "exists": False,
        "end_ts": None,
        "mtime": None,
        "age_hours": np.nan,
        "ok": False,
        "reason": "missing_root",
    }
    if not str(root) or not root.exists():
        return out
    metas: List[tuple[pd.Timestamp, Path, Dict[str, Any]]] = []
    try:
        for meta_path in root.glob("*/rolling_meta.json"):
            try:
                payload = json.loads(meta_path.read_text(encoding="utf-8"))
                end_raw = pd.to_datetime(payload.get("end_ts"), utc=True, errors="coerce")
                end_ts = (
                    pd.Timestamp(end_raw)
                    if not pd.isna(end_raw)
                    else pd.Timestamp("1970-01-01", tz="UTC")
                )
                metas.append((end_ts, meta_path, payload))
            except Exception:
                continue
    except Exception as exc:
        out["reason"] = f"scan_failed:{type(exc).__name__}"
        return out
    if not metas:
        out["reason"] = "missing_rolling_meta"
        return out
    end_ts, meta_path, payload = max(metas, key=lambda item: item[0])
    try:
        mtime = pd.Timestamp(meta_path.stat().st_mtime, unit="s", tz="UTC")
        age_hours = max(float((pd.Timestamp(now) - mtime).total_seconds()) / 3600.0, 0.0)
    except Exception:
        mtime = None
        age_hours = np.nan
    end_ok = bool(pd.Timestamp(end_ts) >= pd.Timestamp(latest_closed_hour))
    age_ok = bool(np.isfinite(age_hours) and age_hours <= float(max_age_hours))
    out.update(
        {
            "path": str(meta_path),
            "exists": True,
            "end_ts": pd.Timestamp(end_ts).isoformat(),
            "mtime": mtime.isoformat() if mtime is not None else None,
            "age_hours": age_hours,
            "rows": payload.get("rows"),
            "features": len(payload.get("features") or []),
            "ok": bool(end_ok and age_ok),
            "reason": (
                "ok"
                if end_ok and age_ok
                else "stale_end_ts"
                if not end_ok
                else "stale_rolling_meta"
            ),
        }
    )
    return out


def _live_warmup_state_health_snapshot(
    *,
    panel: Mapping[str, Any],
    symbols: Sequence[str],
    lookback_hours: int,
    required_model_warmup_hours: int,
    latest_closed_hour: pd.Timestamp,
    feature_runtime_cfg: Mapping[str, Any],
    config: Mapping[str, Any],
    prewarm_result: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    now = pd.Timestamp.now(tz="UTC")
    close = panel.get("close") if isinstance(panel, Mapping) else None
    min_required_panel_hours = _runtime_float(
        config,
        "live_min_panel_warmup_hours",
        "EPM_LIVE_MIN_PANEL_WARMUP_HOURS",
        float(min(int(lookback_hours), 24 * 32)),
    )
    min_coverage = _runtime_float(
        config,
        "live_panel_warmup_min_coverage_ratio",
        "EPM_LIVE_PANEL_WARMUP_MIN_COVERAGE_RATIO",
        0.95,
    )
    max_panel_lag_hours = _runtime_float(
        config,
        "live_panel_max_latest_lag_hours",
        "EPM_LIVE_PANEL_MAX_LATEST_LAG_HOURS",
        1.25,
    )
    panel_ok = False
    panel_reason = "missing_close_panel"
    panel_rows = 0
    panel_symbols = 0
    panel_min_ts = None
    panel_max_ts = None
    panel_span_hours = np.nan
    panel_latest_lag_hours = np.nan
    if isinstance(close, pd.DataFrame) and not close.empty:
        cols = [c for c in close.columns if str(c) in {str(s) for s in symbols}]
        close_scoped = close.loc[:, cols] if cols else close
        non_empty = close_scoped.dropna(how="all")
        panel_rows = int(len(non_empty.index))
        panel_symbols = int(len(close_scoped.columns))
        if not non_empty.empty:
            panel_min_ts = pd.Timestamp(pd.to_datetime(non_empty.index.min(), utc=True))
            panel_max_ts = pd.Timestamp(pd.to_datetime(non_empty.index.max(), utc=True))
            panel_span_hours = max(
                float((panel_max_ts - panel_min_ts).total_seconds()) / 3600.0,
                0.0,
            )
            panel_latest_lag_hours = max(
                float((pd.Timestamp(latest_closed_hour) - panel_max_ts).total_seconds())
                / 3600.0,
                0.0,
            )
            required_span = float(min_required_panel_hours) * float(min_coverage)
            span_ok = bool(panel_span_hours >= required_span)
            lag_ok = bool(panel_latest_lag_hours <= float(max_panel_lag_hours))
            panel_ok = bool(span_ok and lag_ok)
            panel_reason = (
                "ok"
                if panel_ok
                else "insufficient_panel_warmup"
                if not span_ok
                else "stale_panel_latest_ts"
            )
    max_state_age_hours = _runtime_float(
        config,
        "live_rolling_state_max_age_hours",
        "EPM_LIVE_ROLLING_STATE_MAX_AGE_HOURS",
        26.0,
    )
    raw_state = (
        _state_file_health(
            feature_runtime_cfg.get("live_raw_rolling_state_path"),
            now=now,
            max_age_hours=max_state_age_hours,
        )
        if bool(feature_runtime_cfg.get("live_raw_rolling_state_enabled", True))
        else {"ok": True, "reason": "disabled"}
    )
    causal_state = (
        _state_file_health(
            feature_runtime_cfg.get("live_causal_transform_state_path"),
            now=now,
            max_age_hours=max_state_age_hours,
        )
        if bool(feature_runtime_cfg.get("live_causal_transform_state_enabled", True))
        else {"ok": True, "reason": "disabled"}
    )
    rolling_meta = _latest_rolling_meta_health(
        feature_runtime_cfg,
        latest_closed_hour=latest_closed_hour,
        now=now,
        max_age_hours=max_state_age_hours,
    )
    prewarm_status = (
        str((prewarm_result or {}).get("status") or "")
        if isinstance(prewarm_result, Mapping)
        else ""
    )
    prewarm_ok = prewarm_status in {
        "cache_hit",
        "sync_complete_verified",
        "sync_complete",
        "no_training_path_features",
        "no_required_features",
    }
    state_or_cache_ok = bool(
        rolling_meta.get("ok")
        or (raw_state.get("ok") and causal_state.get("ok"))
        or prewarm_ok
    )
    ok = bool(panel_ok and state_or_cache_ok)
    reason = (
        "ok"
        if ok
        else panel_reason
        if not panel_ok
        else "stale_or_missing_rolling_state"
    )
    return {
        "ok": ok,
        "reason": reason,
        "panel_ok": panel_ok,
        "panel_reason": panel_reason,
        "panel_rows": panel_rows,
        "panel_symbols": panel_symbols,
        "panel_min_ts": panel_min_ts.isoformat() if panel_min_ts is not None else None,
        "panel_max_ts": panel_max_ts.isoformat() if panel_max_ts is not None else None,
        "panel_span_hours": panel_span_hours,
        "panel_min_required_hours": float(min_required_panel_hours),
        "panel_latest_lag_hours": panel_latest_lag_hours,
        "required_model_warmup_hours": int(required_model_warmup_hours),
        "loaded_lookback_hours": int(lookback_hours),
        "raw_rolling_state": raw_state,
        "causal_transform_state": causal_state,
        "rolling_feature_cache": rolling_meta,
        "selected_feature_prewarm": dict(prewarm_result or {}),
    }


def _subset_panel(
    panel: Dict[str, pd.DataFrame],
    symbols: List[str],
) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    keep = [str(s) for s in symbols]
    for key, df in panel.items():
        if not isinstance(df, pd.DataFrame) or df.empty:
            continue
        cols = [c for c in keep if c in df.columns]
        if cols:
            out[key] = df.loc[:, cols]
    return out


def _latest_only_panel(
    panel: Dict[str, pd.DataFrame],
    symbols: List[str],
) -> Dict[str, pd.DataFrame]:
    """Return only the latest timestamp needed for live mask decisions."""
    out: Dict[str, pd.DataFrame] = {}
    close = panel.get("close")
    latest_ts = None
    if isinstance(close, pd.DataFrame) and not close.empty:
        latest_ts = close.index.max()
    keep = [str(s) for s in symbols]
    for key, df in panel.items():
        if not isinstance(df, pd.DataFrame) or df.empty:
            continue
        cols = [c for c in keep if c in df.columns]
        if not cols:
            continue
        if latest_ts is not None and latest_ts in df.index:
            out[key] = df.loc[[latest_ts], cols]
        else:
            out[key] = df.loc[:, cols].tail(1)
    return out


def _latest_only_features(
    feats: Dict[str, pd.DataFrame],
    *,
    latest_ts: Optional[pd.Timestamp],
    symbols: List[str],
) -> Dict[str, pd.DataFrame]:
    """Align live mask features to a single latest timestamp/symbol row."""
    out: Dict[str, pd.DataFrame] = {}
    keep = [str(s) for s in symbols]
    ts_index = None
    if latest_ts is not None:
        ts_index = pd.DatetimeIndex([pd.Timestamp(latest_ts)])
    if hasattr(feats, "latest_values_at"):
        ts_utc = None
        if latest_ts is not None:
            ts_utc = pd.Timestamp(latest_ts)
            if ts_utc.tzinfo is None:
                ts_utc = ts_utc.tz_localize("UTC")
            else:
                ts_utc = ts_utc.tz_convert("UTC")
        raw_payloads = getattr(feats, "_raw", {})
        assembled_payloads = getattr(feats, "_assembled", {})
        symbol_indices = getattr(feats, "_symbol_indices", {})
        latest_pos_by_symbol: dict[str, int] = {}
        if ts_utc is not None:
            for sym in keep:
                idx_vals = symbol_indices.get(sym)
                if idx_vals is None:
                    continue
                idx = pd.to_datetime(idx_vals, utc=True, errors="coerce")
                positions = np.flatnonzero(idx <= ts_utc)
                if positions.size:
                    latest_pos_by_symbol[sym] = int(positions[-1])
        for key in feats.keys():
            frame = assembled_payloads.get(key) if isinstance(assembled_payloads, dict) else None
            if isinstance(frame, pd.DataFrame) and not frame.empty:
                cols = [c for c in keep if c in frame.columns]
                if not cols:
                    continue
                idx = pd.to_datetime(frame.index, utc=True, errors="coerce")
                positions = (
                    np.flatnonzero(idx <= ts_utc)
                    if ts_utc is not None
                    else np.arange(len(frame.index))
                )
                if positions.size == 0:
                    continue
                pos = int(positions[-1])
                latest = frame.iloc[[pos]].loc[:, cols]
                if ts_index is not None:
                    latest = latest.copy()
                    latest.index = ts_index
                out[str(key)] = latest
                continue
            payload = raw_payloads.get(key) if isinstance(raw_payloads, dict) else None
            if isinstance(payload, dict) and ts_utc is not None:
                cols: list[str] = []
                vals: list[float] = []
                for sym in keep:
                    item = payload.get(sym)
                    if item is None:
                        continue
                    if isinstance(item, tuple) and len(item) == 2:
                        idx_vals, val_array = item
                        idx = pd.to_datetime(idx_vals, utc=True, errors="coerce")
                        positions = np.flatnonzero(idx <= ts_utc)
                        if positions.size == 0:
                            continue
                        pos = int(positions[-1])
                    else:
                        val_array = item
                        pos = latest_pos_by_symbol.get(sym)
                        if pos is None:
                            continue
                    arr = np.asarray(val_array)
                    if pos >= len(arr):
                        continue
                    cols.append(sym)
                    vals.append(float(arr[pos]))
                if cols:
                    out[str(key)] = pd.DataFrame([vals], columns=cols, index=ts_index)
                continue
            raw_symbols = set()
            if hasattr(feats, "raw_symbols_for_key"):
                try:
                    raw_symbols = set(str(sym) for sym in feats.raw_symbols_for_key(key))
                except Exception:
                    raw_symbols = set()
            cols = [sym for sym in keep if not raw_symbols or sym in raw_symbols]
            if not cols or latest_ts is None:
                continue
            try:
                values = feats.latest_values_at(key, cols, latest_ts)
            except Exception:
                continue
            if values is None or len(values) == 0:
                continue
            latest = pd.DataFrame(
                [pd.Series(values).reindex(cols).to_numpy()],
                columns=cols,
                index=ts_index,
            )
            out[str(key)] = latest
        return out
    for key in feats.keys():
        df = feats.get(key)
        if not isinstance(df, pd.DataFrame) or df.empty:
            continue
        cols = [c for c in keep if c in df.columns]
        if not cols:
            continue
        if latest_ts is not None and latest_ts in df.index:
            latest = df.loc[[latest_ts], cols]
        else:
            latest = df.loc[:, cols].tail(1)
            if ts_index is not None:
                latest = latest.copy()
                latest.index = ts_index
        out[str(key)] = latest
    return out


def _lgbm_mask_required_feature_keys(
    lgbm_strategy_mask_rows: Optional[Dict[str, Dict[str, Any]]],
) -> set[str]:
    """Extract raw feature names referenced by canonical LGBM mask rules."""
    if not lgbm_strategy_mask_rows:
        return set()
    out: set[str] = set()
    feature_cmp = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\b\s*(?:<=|>=|==|<|>)")
    ignore = {
        "and",
        "or",
        "not",
        "true",
        "false",
        "nan",
        "inf",
    }
    for row in lgbm_strategy_mask_rows.values():
        if not isinstance(row, dict):
            continue
        mask_params = (
            row.get("mask_params", {})
            if isinstance(row.get("mask_params"), dict)
            else {}
        )
        raw_rules = [
            row.get("base_event_trigger"),
            row.get("canonical_key"),
            mask_params.get("canonical_key"),
            mask_params.get("base_event_trigger"),
        ]
        for rule in raw_rules:
            text = str(rule or "")
            if not text:
                continue
            for match in feature_cmp.finditer(text):
                name = str(match.group(1) or "").strip()
                if name and name.lower() not in ignore:
                    out.add(name)
    return out


def _symbols_with_required_feature_coverage(
    feats: Dict[str, pd.DataFrame],
    required_feature_keys: set[str],
    symbols: List[str],
) -> tuple[List[str], Dict[str, List[str]]]:
    """Return candidate symbols that have all required feature columns available."""
    symbol_list = [str(sym) for sym in symbols if str(sym)]
    if not required_feature_keys or not symbol_list:
        return symbol_list, {}
    missing_by_symbol: Dict[str, List[str]] = {}
    if hasattr(feats, "raw_symbols_for_key"):
        for key in sorted(required_feature_keys):
            try:
                available = {str(sym) for sym in feats.raw_symbols_for_key(key)}
            except Exception:
                available = set()
            if not available:
                for sym in symbol_list:
                    missing_by_symbol.setdefault(sym, []).append(str(key))
                continue
            for sym in symbol_list:
                if sym not in available:
                    missing_by_symbol.setdefault(sym, []).append(str(key))
    elif hasattr(feats, "latest_values_at"):
        # Lazy/overlay live feature stores can represent sparse selected-cache
        # coverage by returning NaN cells for a requested symbol while still
        # carrying the trained feature column. Do not prune candidates on raw
        # symbol membership here; the model-matrix adapter applies the same
        # neutral-fill semantics used by training, and true missing keys remain
        # hard failures below.
        for key in sorted(required_feature_keys):
            try:
                has_key = str(key) in feats
            except Exception:
                has_key = False
            if not has_key:
                for sym in symbol_list:
                    missing_by_symbol.setdefault(sym, []).append(str(key))
        allowed = [sym for sym in symbol_list if sym not in missing_by_symbol]
        return allowed, missing_by_symbol
    else:
        for key in sorted(required_feature_keys):
            value = feats.get(key) if hasattr(feats, "get") else None
            if not isinstance(value, pd.DataFrame) or value.empty:
                for sym in symbol_list:
                    missing_by_symbol.setdefault(sym, []).append(str(key))
                continue
            columns = {str(col) for col in value.columns}
            for sym in symbol_list:
                if sym not in columns:
                    missing_by_symbol.setdefault(sym, []).append(str(key))
    allowed = [sym for sym in symbol_list if sym not in missing_by_symbol]
    return allowed, missing_by_symbol


def _effective_runtime_model_bundle(
    full_state: Dict[str, Any],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Mirror ModelOrchestrator's runtime bundle overlay for validation."""
    effective = dict(full_state or {})
    loaded_bundle = effective.get("bundle", {}) if isinstance(effective, dict) else {}
    runtime_bundle = config.get("model_bundle", {}) if isinstance(config, dict) else {}
    bundle = dict(loaded_bundle or {})
    if isinstance(runtime_bundle, dict):
        for key, value in runtime_bundle.items():
            if value:
                bundle[key] = value
    effective["bundle"] = bundle
    return effective


def _training_context_symbols_for_live_universe(
    universe_state: Mapping[str, Any],
) -> List[str]:
    """Map the trained artifact universe onto live symbols for feature context."""
    tradable = [str(sym) for sym in universe_state.get("tradable_symbols", [])]
    trained = [str(sym) for sym in universe_state.get("trained_symbols", [])]
    if not trained:
        return sorted(set(tradable))
    live_by_base: Dict[str, str] = {}
    for sym in universe_state.get("download_symbols", []):
        sym_s = str(sym)
        base = symbol_base(sym_s)
        if base and base not in live_by_base:
            live_by_base[base] = sym_s
    out: set[str] = set(tradable)
    for sym in trained:
        base = symbol_base(sym)
        if base:
            out.add(live_by_base.get(base, sym))
    return sorted(out)


def _candidate_feature_cycle_cache_key(
    *,
    panel: Dict[str, pd.DataFrame],
    symbols: List[str],
    run_id: str,
    data_root: str,
    cfg: Dict[str, Any],
    lookback_hours: int,
    raw_feature_keys: set[str],
    lgbm_strategy_mask_rows: Optional[Dict[str, Dict[str, Any]]],
    feature_context_symbols: Optional[List[str]],
    strategy_feature_contracts: Optional[Mapping[str, Sequence[str]]] = None,
    model_features_required: bool = True,
) -> Optional[str]:
    close = panel.get("close") if isinstance(panel, dict) else None
    if not isinstance(close, pd.DataFrame) or close.empty:
        return None
    cfg_for_hash = {
        str(k): v
        for k, v in (cfg or {}).items()
        if str(k) not in {"live_feature_cycle_cache_bypass"}
    }
    strategy_payload = []
    for sid, row in sorted((lgbm_strategy_mask_rows or {}).items()):
        if not isinstance(row, dict):
            continue
        strategy_payload.append(
            {
                "strategy_id": str(sid),
                "side": str(row.get("trade_side") or row.get("side") or ""),
                "canonical_key": str(
                    row.get("base_event_trigger") or row.get("canonical_key") or ""
                ),
                "mask_params": row.get("mask_params") or {},
            }
        )
    payload = {
        "run_id": str(run_id),
        "data_root": str(data_root),
        "lookback_hours": int(lookback_hours),
        "latest_ts": pd.Timestamp(close.index.max()).isoformat(),
        "symbols_hash": _hash_values(symbols),
        "feature_context_symbols_hash": _hash_values(feature_context_symbols or symbols),
        "raw_feature_keys_hash": _hash_values(raw_feature_keys),
        "strategy_feature_contracts_hash": _feature_runtime_cfg_hash(
            {
                str(k): sorted(str(vv) for vv in (v or []))
                for k, v in (strategy_feature_contracts or {}).items()
            }
        ),
        "model_features_required": bool(model_features_required),
        "cfg_hash": _feature_runtime_cfg_hash(cfg_for_hash),
        "strategies": strategy_payload,
    }
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:24]


def _strategy_feature_contracts_from_orchestrator(
    orchestrator: Any,
    lgbm_strategy_mask_rows: Optional[Mapping[str, Mapping[str, Any]]],
) -> Dict[str, List[str]]:
    """Resolve deployed strategy ids to exact base+meta decision feature contracts."""
    contracts: Dict[str, List[str]] = {}
    if not lgbm_strategy_mask_rows:
        return contracts
    for strategy_id, row in lgbm_strategy_mask_rows.items():
        side = str(
            (row or {}).get("trade_side")
            or (row or {}).get("side")
            or strategy_side(str(strategy_id))
            or ""
        ).lower()
        model_contracts = _model_feature_contracts_for_audit(
            orchestrator, side=side, strategy_id=str(strategy_id)
        )
        feat_cols = list(model_contracts.get("base_features") or [])
        raw_meta_cols = [str(c) for c in (model_contracts.get("meta_features") or []) if str(c)]
        feat_cols.extend(sorted(_meta_model_derived_raw_dependencies(raw_meta_cols)))
        feat_cols.extend(
            str(c)
            for c in raw_meta_cols
            if not is_model_derived_feature_key(str(c))
            and _meta_live_unavailable_neutral_default(str(c)) is None
        )
        if feat_cols:
            seen: set[str] = set()
            ordered: List[str] = []
            for col in feat_cols:
                col_s = str(col)
                if not col_s or col_s in seen:
                    continue
                seen.add(col_s)
                ordered.append(col_s)
            contracts[str(strategy_id)] = ordered
    return contracts


def _cfg_flag(
    cfg: Optional[Mapping[str, Any]],
    key: str,
    env_key: str,
    default: bool,
) -> bool:
    raw = (cfg or {}).get(key, os.environ.get(env_key, "1" if default else "0"))
    return str(raw).strip().lower() not in {"0", "false", "no", "off"}


def _is_train_tolerated_live_nonfinite_feature_key(key: str) -> bool:
    """Return True for selected inputs whose NaNs are handled by model scoring."""
    key_s = str(key or "")
    if _is_live_source_derived_feature_key(key_s):
        return True
    if key_s in {"G_VOL", "G_TREND"} or re.fullmatch(
        r"^.+_G_(?:VOL|TREND)_[01]$",
        key_s,
    ):
        return True
    return (
        key_s.startswith("volume_entropy_")
        or key_s.startswith("volume_autocorr_")
        or key_s.startswith("ker_")
        or key_s.startswith("loc_ema_stack_pos_")
        or key_s in {"impulse_reversal", "impulse_reversal_short"}
    )


def _filter_strategy_masks_by_finite_model_contract(
    feats: Dict[str, pd.DataFrame],
    strategy_candidate_masks: Dict[str, List[str]],
    strategy_feature_contracts: Mapping[str, Sequence[str]],
    *,
    latest_ts: Any,
    cfg: Optional[Mapping[str, Any]] = None,
) -> tuple[Dict[str, List[str]], Dict[str, Dict[str, Any]]]:
    """Keep symbols whose blocking deployed model inputs are finite."""
    if not strategy_candidate_masks or not strategy_feature_contracts:
        return strategy_candidate_masks, {}
    allow_train_tolerated_nonfinite = _cfg_flag(
        cfg,
        "live_model_contract_allow_train_tolerated_nonfinite",
        "EPM_LIVE_MODEL_CONTRACT_ALLOW_TRAIN_TOLERATED_NONFINITE",
        True,
    )
    filtered: Dict[str, List[str]] = {}
    diagnostics: Dict[str, Dict[str, Any]] = {}
    for strategy_id, symbols in strategy_candidate_masks.items():
        symbol_list = [str(sym) for sym in (symbols or []) if str(sym)]
        feat_cols = [str(col) for col in strategy_feature_contracts.get(str(strategy_id), []) if str(col)]
        if not symbol_list or not feat_cols:
            filtered[str(strategy_id)] = symbol_list
            continue
        matrix = _get_features_for_candidates_at_ts(feats, symbol_list, ts=latest_ts)
        missing_cols = [col for col in feat_cols if col not in matrix.columns]
        aligned = matrix.reindex(columns=feat_cols)
        allowed_nonfinite_cols = {
            col
            for col in feat_cols
            if (
                allow_train_tolerated_nonfinite
                and col not in missing_cols
                and _is_train_tolerated_live_nonfinite_feature_key(col)
            )
        }
        blocking_col_idx = [
            idx for idx, col in enumerate(feat_cols) if col not in allowed_nonfinite_cols
        ]
        try:
            values = aligned.astype(np.float32, copy=False).to_numpy(
                dtype=np.float32,
                copy=False,
            )
            finite = np.isfinite(values)
            if blocking_col_idx:
                finite_rows = finite[:, blocking_col_idx].all(axis=1)
            else:
                finite_rows = np.ones(len(symbol_list), dtype=bool)
        except Exception:
            finite = np.zeros((len(symbol_list), len(feat_cols)), dtype=bool)
            finite_rows = np.zeros(len(symbol_list), dtype=bool)
        kept = [sym for sym, ok in zip(symbol_list, finite_rows) if bool(ok)]
        filtered[str(strategy_id)] = kept
        if missing_cols or len(kept) != len(symbol_list):
            nonfinite_counts = (~finite).sum(axis=0) if finite.size else np.zeros(len(feat_cols), dtype=int)
            allowed_idx = [
                idx for idx, col in enumerate(feat_cols) if col in allowed_nonfinite_cols
            ]

            def _top_features(indices: Sequence[int], limit: int = 15) -> List[Dict[str, Any]]:
                ranked = sorted(
                    [int(i) for i in indices],
                    key=lambda i: int(nonfinite_counts[int(i)]),
                    reverse=True,
                )
                return [
                    {
                        "feature": str(feat_cols[int(i)]),
                        "rows": int(nonfinite_counts[int(i)]),
                        "pct": round(
                            float(nonfinite_counts[int(i)])
                            * 100.0
                            / max(1, len(symbol_list)),
                            2,
                        ),
                    }
                    for i in ranked[:limit]
                    if int(nonfinite_counts[int(i)]) > 0
                ]

            top_features = _top_features(range(len(feat_cols)))
            top_blocking = _top_features(blocking_col_idx)
            top_allowed = _top_features(allowed_idx)
            diagnostics[str(strategy_id)] = {
                "input": len(symbol_list),
                "kept": len(kept),
                "rejected": int(len(symbol_list) - len(kept)),
                "missing_cols": missing_cols[:20],
                "top_nonfinite_features": top_features,
                "top_blocking_nonfinite_features": top_blocking,
                "top_allowed_nonfinite_features": top_allowed,
                "allowed_nonfinite_feature_count": int(len(allowed_nonfinite_cols)),
                "sample_rejected_symbols": [
                    sym for sym, ok in zip(symbol_list, finite_rows) if not bool(ok)
                ][:10],
            }
    return filtered, diagnostics


def _strategy_decision_feature_contract(
    strategy_feature_contracts: Optional[Mapping[str, Sequence[str]]],
    *,
    side: str,
    strategy_id: str,
) -> List[str]:
    """Return the exact raw decision feature contract for one deployed strategy."""
    if not strategy_feature_contracts:
        return []
    strategy_id_s = str(strategy_id or "")
    side_s = str(side or "").lower()
    candidates = [
        strategy_id_s,
        strategy_core_id(strategy_id_s),
        f"{side_s}_{strategy_core_id(strategy_id_s)}" if side_s else "",
    ]
    for key in candidates:
        if not key:
            continue
        values = strategy_feature_contracts.get(key)
        if not values:
            continue
        seen: set[str] = set()
        ordered: List[str] = []
        for value in values:
            col = str(value)
            if not col or col in seen:
                continue
            seen.add(col)
            ordered.append(col)
        return ordered
    return []


def _copy_candidate_feature_cycle_entry(
    entry: Dict[str, Any],
) -> tuple[
    Dict[str, float],
    List[str],
    List[str],
    Dict[str, pd.DataFrame],
    Dict[str, List[str]],
]:
    return (
        dict(entry.get("thresholds") or {}),
        list(entry.get("long_cands") or []),
        list(entry.get("short_cands") or []),
        dict(entry.get("features") or {}),
        {
            str(k): list(v or [])
            for k, v in (entry.get("strategy_candidate_masks") or {}).items()
        },
    )


def _select_candidates_and_load_features(
    *,
    panel: Dict[str, pd.DataFrame],
    symbols: List[str],
    run_id: str,
    data_root: str,
    cfg: Dict[str, Any],
    lookback_hours: int,
    required_feature_keys: Optional[set[str]],
    lgbm_strategy_mask_rows: Optional[Dict[str, Dict[str, Any]]] = None,
    feature_context_symbols: Optional[List[str]] = None,
    strategy_feature_contracts: Optional[Mapping[str, Sequence[str]]] = None,
    model_features_required: bool = True,
) -> tuple[
    Dict[str, float],
    List[str],
    List[str],
    Dict[str, pd.DataFrame],
    Dict[str, List[str]],
]:
    with _FEATURE_COMPUTE_LOCK:
        timer = _StageTimer("candidate_feature_load")
        model_raw_feature_keys = raw_required_feature_keys(required_feature_keys)
        mask_raw_feature_keys = _lgbm_mask_required_feature_keys(
            lgbm_strategy_mask_rows
        )
        execution_policy_feature_keys = (
            {"barrier_pct"} if model_features_required else set()
        )
        raw_feature_keys = set(mask_raw_feature_keys)
        if model_features_required:
            raw_feature_keys |= set(model_raw_feature_keys)
            raw_feature_keys |= execution_policy_feature_keys
        cache_key = None
        if bool((cfg or {}).get("live_feature_cycle_cache_enabled", True)) and not bool(
            (cfg or {}).get("live_feature_cycle_cache_bypass", False)
        ):
            cache_key = _candidate_feature_cycle_cache_key(
                panel=panel,
                symbols=symbols,
                run_id=run_id,
                data_root=data_root,
                cfg=cfg,
                lookback_hours=lookback_hours,
                raw_feature_keys=raw_feature_keys,
                lgbm_strategy_mask_rows=lgbm_strategy_mask_rows,
                feature_context_symbols=feature_context_symbols,
                strategy_feature_contracts=strategy_feature_contracts,
                model_features_required=model_features_required,
            )
            if cache_key and cache_key in _CANDIDATE_FEATURE_CYCLE_CACHE:
                cached = _CANDIDATE_FEATURE_CYCLE_CACHE[cache_key]
                tprint(
                    "Loaded cached hourly candidate feature/mask matrix: "
                    f"key={cache_key} "
                    f"features={len(cached.get('features') or {})} "
                    f"strategies={len(cached.get('strategy_candidate_masks') or {})}"
                )
                timer.mark("hourly_feature_matrix_cache_hit")
                return _copy_candidate_feature_cycle_entry(cached)
        selector_feats = compute_selector_features(panel, symbols)
        timer.mark("selector_features")
        thresholds = get_candidate_thresholds(
            market_mode=(
                (cfg or {}).get("market_mode") if isinstance(cfg, dict) else None
            )
        )
        min_range_pct = thresholds.get("min_range_pct")
        if thresholds.get("min_move_12h_pct") is not None:
            min_range_pct = None
        _ = min_range_pct
        long_cands: List[str] = []
        short_cands: List[str] = []
        candidate_feats: Dict[str, pd.DataFrame] = dict(selector_feats)
        strategy_candidate_masks: Dict[str, List[str]] = {}
        pre_model_strategy_masks_authoritative = False
        if not lgbm_strategy_mask_rows:
            mode_cfg = dict((cfg or {}).get("candidate_mask_params_by_mode", {}) or {})
            policy_rules = dict((cfg or {}).get("policy_selection_rules", {}) or {})
            mask_contract_required = bool(
                policy_rules.get("requires_lgbm_regime_mask_contract", True)
            )
            accepted_contract_ids = [
                str(sid).strip()
                for sid in ((cfg or {}).get("accepted_strategy_ids") or [])
                if str(sid).strip()
            ]
            if not mode_cfg and not mask_contract_required and accepted_contract_ids:
                tradable_ordered = [str(sym) for sym in symbols if str(sym).strip()]
                long_strategy_ids = [
                    sid for sid in accepted_contract_ids if strategy_side(sid) == "long"
                ]
                short_strategy_ids = [
                    sid for sid in accepted_contract_ids if strategy_side(sid) == "short"
                ]
                if long_strategy_ids:
                    long_cands = list(tradable_ordered)
                if short_strategy_ids:
                    short_cands = list(tradable_ordered)
                strategy_candidate_masks = {
                    sid: list(tradable_ordered)
                    for sid in accepted_contract_ids
                    if strategy_side(sid) in {"long", "short"}
                }
                tprint(
                    "No deployed pre-model mask contract and no per-mode mask params; "
                    "using full tradable universe for model/policy rank gating "
                    f"strategies={len(strategy_candidate_masks)} "
                    f"long={len(long_cands)} short={len(short_cands)} "
                    f"symbols={len(tradable_ordered)}"
                )
            else:
                long_cands, short_cands = select_candidates(
                    panel=panel,
                    feats=selector_feats,
                    metric=str(thresholds.get("metric", "ret12h")),
                )
            timer.mark("selector_candidates")
        else:
            tprint(
                "Deployment LGBM masks are authoritative; "
                "skipping legacy per-mode candidate selector."
            )
        strategy_candidate_masks = dict(strategy_candidate_masks)
        tradable_symbol_set = {str(sym) for sym in symbols}
        context_symbols = [
            str(sym).strip()
            for sym in (feature_context_symbols or symbols)
            if str(sym).strip()
        ]
        model_feature_symbols = [
            str(sym)
            for sym in context_symbols
            if str(sym) in panel.get("close", pd.DataFrame()).columns
        ]
        if not model_feature_symbols:
            model_feature_symbols = list(context_symbols or symbols)
        if (
            lgbm_strategy_mask_rows
            and bool((cfg or {}).get("live_pre_mask_before_model_features", True))
            and mask_raw_feature_keys
        ):
            mask_panel = _latest_only_panel(panel, model_feature_symbols)
            mask_close = mask_panel.get("close")
            mask_latest_ts = (
                mask_close.index.max()
                if isinstance(mask_close, pd.DataFrame) and not mask_close.empty
                else None
            )
            if mask_raw_feature_keys.issubset(set(selector_feats)):
                pre_mask_feats = _latest_only_features(
                    selector_feats,
                    latest_ts=mask_latest_ts,
                    symbols=model_feature_symbols,
                )
                timer.mark("lgbm_pre_mask_features_selector")
            else:
                missing_pre_mask = sorted(mask_raw_feature_keys - set(selector_feats))
                tprint(
                    "LGBM pre-mask requires computed mask-only features: "
                    f"missing {len(missing_pre_mask)} keys: {missing_pre_mask[:12]}"
                )
                pre_mask_feats = load_or_compute_features(
                    panel=_subset_panel(panel, model_feature_symbols),
                    basket_syms=model_feature_symbols,
                    run_id=run_id,
                    data_root=data_root,
                    cfg={
                        **(cfg or {}),
                        "live_feature_return_latest_only": True,
                        "live_feature_cache_namespace": "mask",
                    },
                    lookback_hours=lookback_hours,
                    required_feature_keys=mask_raw_feature_keys,
                )
                pre_mask_feats = _latest_only_features(
                    pre_mask_feats,
                    latest_ts=mask_latest_ts,
                    symbols=model_feature_symbols,
                )
                pre_mask_feats.update(
                    _latest_only_features(
                        selector_feats,
                        latest_ts=mask_latest_ts,
                        symbols=model_feature_symbols,
                    )
                )
                timer.mark("lgbm_pre_mask_features_computed")
            candidate_feats = pre_mask_feats
            strategy_candidate_masks = build_strategy_candidate_masks(
                mask_panel,
                pre_mask_feats,
                lgbm_strategy_mask_rows.values(),
            )
            timer.mark("lgbm_pre_mask_eval")
            non_empty_masks = sum(
                1 for symbols_ in strategy_candidate_masks.values() if symbols_
            )
            tprint(
                "LGBM strategy masks latest pre-pass: "
                f"strategies={len(strategy_candidate_masks)} "
                f"non_empty={non_empty_masks}"
            )
            if strategy_candidate_masks:
                mask_diag = _strategy_mask_count_diagnostics(
                    strategy_candidate_masks,
                    lgbm_strategy_mask_rows,
                    model_feature_symbols,
                )
                tprint(
                    "LGBM strategy mask pass/fail counts: "
                    f"{ {strategy_core_id(k): {'pass': v.get('pass_count'), 'fail': v.get('fail_count'), 'universe': v.get('universe_count')} for k, v in mask_diag.items()} }"
                )
            if non_empty_masks == 0:
                result = (
                    thresholds,
                    long_cands,
                    short_cands,
                    pre_mask_feats,
                    strategy_candidate_masks,
                )
                if cache_key:
                    _CANDIDATE_FEATURE_CYCLE_CACHE.clear()
                    _CANDIDATE_FEATURE_CYCLE_CACHE[cache_key] = {
                        "thresholds": thresholds,
                        "long_cands": long_cands,
                        "short_cands": short_cands,
                        "features": pre_mask_feats,
                        "strategy_candidate_masks": strategy_candidate_masks,
                    }
                return result
            pre_model_strategy_masks_authoritative = True
            if not model_features_required:
                mask_long: set[str] = set()
                mask_short: set[str] = set()
                for strategy_id, passed_symbols in strategy_candidate_masks.items():
                    row = lgbm_strategy_mask_rows.get(strategy_id, {}) or {}
                    side = str(row.get("trade_side") or row.get("side") or "").lower()
                    if side == "long":
                        mask_long.update(str(sym) for sym in passed_symbols)
                    elif side == "short":
                        mask_short.update(str(sym) for sym in passed_symbols)
                before_long = len(long_cands)
                before_short = len(short_cands)
                long_cands = sorted(set(long_cands).union(mask_long))
                short_cands = sorted(set(short_cands).union(mask_short))
                if mask_long or mask_short:
                    tprint(
                        "Deployment LGBM masks expanded mask-only candidate universe: "
                        f"long {before_long}->{len(long_cands)} "
                        f"short {before_short}->{len(short_cands)} "
                        f"mask_pass_long={len(mask_long)} "
                        f"mask_pass_short={len(mask_short)}"
                    )
                result = (
                    thresholds,
                    long_cands,
                    short_cands,
                    candidate_feats,
                    strategy_candidate_masks,
                )
                if cache_key:
                    _CANDIDATE_FEATURE_CYCLE_CACHE.clear()
                    _CANDIDATE_FEATURE_CYCLE_CACHE[cache_key] = {
                        "thresholds": thresholds,
                        "long_cands": long_cands,
                        "short_cands": short_cands,
                        "features": candidate_feats,
                        "strategy_candidate_masks": strategy_candidate_masks,
                    }
                return result
        if not model_features_required:
            result = (
                thresholds,
                long_cands,
                short_cands,
                candidate_feats,
                strategy_candidate_masks,
            )
            if cache_key:
                _CANDIDATE_FEATURE_CYCLE_CACHE.clear()
                _CANDIDATE_FEATURE_CYCLE_CACHE[cache_key] = {
                    "thresholds": thresholds,
                    "long_cands": long_cands,
                    "short_cands": short_cands,
                    "features": candidate_feats,
                    "strategy_candidate_masks": strategy_candidate_masks,
                }
            return result
        stable_feature_universe = bool(
            (cfg or {}).get("live_feature_stable_model_universe_enabled", True)
        )
        source_allowed_model_symbols = list(model_feature_symbols)
        close_panel = panel.get("close") if isinstance(panel, dict) else None
        if isinstance(close_panel, pd.DataFrame) and not close_panel.empty:
            source_report = validate_required_source_panels(
                panel,
                model_feature_symbols,
                pd.Timestamp(close_panel.index.max()),
                model_raw_feature_keys,
                cfg=cfg,
                strict=True,
            )
            report_path = _persist_source_parity_report(
                source_report,
                data_root=data_root,
                run_id=run_id,
                label="model_sources",
            )
            accepted_source_symbols = [
                str(sym) for sym in source_report.get("accepted_symbols", [])
            ]
            source_allowed_model_symbols = accepted_source_symbols
            if len(accepted_source_symbols) != len(model_feature_symbols):
                source_summary = source_report.get("source_rejection_summary") or {}
                tprint(
                    "Live feature parity: source contract filtered model universe "
                    f"{len(model_feature_symbols)}->{len(accepted_source_symbols)} "
                    f"required_groups={list((source_report.get('required_source_groups') or {}).keys())} "
                    f"source_groups={source_summary.get('by_group', [])[:10]} "
                    f"missing_sources={source_summary.get('missing_source_keys', [])[:10]} "
                    f"stale_sources={source_summary.get('stale_source_keys', [])[:10]} "
                    f"report={report_path}"
                )
                if stable_feature_universe:
                    tprint(
                        "Live feature cache: keeping stable model feature universe "
                        f"for compute/cache ({len(model_feature_symbols)} symbols); "
                        f"source-eligible symbols for masks/orders={len(accepted_source_symbols)}"
                    )
                else:
                    model_feature_symbols = accepted_source_symbols
        if not source_allowed_model_symbols:
            source_allowed_model_symbols = list(model_feature_symbols)
        model_decision_feature_keys = set(model_raw_feature_keys)
        pre_mask_source_eligible_symbols = sorted(
            {
                str(sym)
                for syms in (strategy_candidate_masks or {}).values()
                for sym in (syms or [])
                if str(sym) in set(source_allowed_model_symbols)
                and str(sym) in tradable_symbol_set
            }
        )
        active_mask_strategy_ids = [
            str(sid)
            for sid, syms in (strategy_candidate_masks or {}).items()
            if any(
                str(sym) in set(source_allowed_model_symbols)
                and str(sym) in tradable_symbol_set
                for sym in (syms or [])
            )
        ]
        if active_mask_strategy_ids and strategy_feature_contracts:
            active_model_feature_keys: set[str] = set()
            for sid in active_mask_strategy_ids:
                candidate_keys = [
                    sid,
                    strategy_core_id(sid),
                    f"{strategy_side(sid)}_{strategy_core_id(sid)}"
                    if strategy_side(sid)
                    else "",
                ]
                for candidate_key in candidate_keys:
                    values = strategy_feature_contracts.get(candidate_key)
                    if values:
                        active_model_feature_keys.update(str(v) for v in values if str(v))
            if active_model_feature_keys:
                before_model_keys = len(model_raw_feature_keys)
                model_raw_feature_keys = raw_required_feature_keys(active_model_feature_keys)
                model_decision_feature_keys = set(active_model_feature_keys)
                raw_feature_keys = (
                    set(mask_raw_feature_keys)
                    .union(model_raw_feature_keys)
                    .union(execution_policy_feature_keys)
                )
                cache_key = None
                tprint(
                    "Live feature parity: narrowed model feature contract to "
                    "source-eligible mask-passing strategies "
                    f"strategies={len(active_mask_strategy_ids)} "
                    f"model_required={before_model_keys}->{len(model_raw_feature_keys)}"
                )
        model_feature_coverage_symbols = (
            pre_mask_source_eligible_symbols or list(source_allowed_model_symbols)
        )
        tprint(
            "Live feature parity: computing model feature frame on the full "
            f"tradable universe ({len(model_feature_symbols)} symbols), "
            f"model_required={len(model_raw_feature_keys)} "
            f"mask_required={len(mask_raw_feature_keys)} "
            f"coverage_symbols={len(model_feature_coverage_symbols)}."
        )
        model_feats = load_or_compute_features(
            panel=_subset_panel(panel, model_feature_symbols),
            basket_syms=model_feature_symbols,
            run_id=run_id,
            data_root=data_root,
            cfg={
                **(cfg or {}),
                "live_feature_cache_namespace": "model",
                # Source-parity rejected symbols remain in the stable cache
                # universe for transform consistency, but they must not make
                # the model-feature freshness guard fail the whole cycle.
                "live_feature_coverage_symbols": model_feature_coverage_symbols,
                # When the deployment contract names a selected-feature source
                # run, model scoring must use that handoff by default. Operators
                # can still explicitly disable it for live recompute audits via
                # live_model_feature_offline_cache_enabled=False.
                "live_feature_offline_cache_enabled": _model_feature_offline_cache_enabled(
                    cfg or {}
                ),
                "live_feature_prefer_offline_cache": _model_feature_offline_cache_enabled(
                    cfg or {}
                ),
                "live_model_feature_store_strict": live_model_feature_store_strict(
                    cfg or {}
                ),
                "live_feature_return_latest_only": True,
            },
            lookback_hours=lookback_hours,
            required_feature_keys=raw_feature_keys,
        )
        if (
            lgbm_strategy_mask_rows
            and model_decision_feature_keys
            and not live_model_feature_store_strict(cfg or {})
        ):
            try:
                first_mask_cfg = dict(next(iter(lgbm_strategy_mask_rows.values())) or {})
                first_mask_cfg.update(dict(first_mask_cfg.get("mask_params", {}) or {}))
                prepared_model_feats = build_latest_prepared_feature_frames(
                    _latest_only_panel(panel, model_feature_symbols),
                    {
                        **model_feats,
                        **_latest_only_features(
                            selector_feats,
                            latest_ts=(
                                panel.get("close").index.max()
                                if isinstance(panel.get("close"), pd.DataFrame)
                                and not panel.get("close").empty
                                else None
                            ),
                            symbols=model_feature_symbols,
                        ),
                    },
                    first_mask_cfg,
                    symbols=model_feature_symbols,
                    required_columns=model_decision_feature_keys,
                )
                if prepared_model_feats:
                    before_prepared = len(model_feats)
                    model_feats = _merge_missing_feature_dicts(
                        model_feats,
                        prepared_model_feats,
                    )
                    tprint(
                        "Live model feature parity: merged prepared "
                        "FeatureProcessor contract frames "
                        f"features={before_prepared}->{len(model_feats)}"
                    )
            except Exception as exc:
                tprint(
                    "Live model feature parity: failed to materialize prepared "
                    f"FeatureProcessor frames; scoring will fail closed if required: {exc}"
                )
        timer.mark("model_features")
        if lgbm_strategy_mask_rows:
            recompute_masks_after_model_features = bool(
                (cfg or {}).get("live_recompute_lgbm_masks_after_model_features", False)
            )
            if pre_model_strategy_masks_authoritative and not recompute_masks_after_model_features:
                timer.mark("lgbm_mask_eval_preserved_pre_model")
                tprint(
                    "LGBM strategy masks latest pass: preserving authoritative "
                    "pre-model mask evaluation after model feature load."
                )
                if strategy_candidate_masks:
                    mask_diag = _strategy_mask_count_diagnostics(
                        strategy_candidate_masks,
                        lgbm_strategy_mask_rows,
                        model_feature_symbols,
                    )
                    tprint(
                        "LGBM strategy mask pass/fail counts: "
                        f"{ {strategy_core_id(k): {'pass': v.get('pass_count'), 'fail': v.get('fail_count'), 'universe': v.get('universe_count')} for k, v in mask_diag.items()} }"
                    )
            else:
                mask_panel = _latest_only_panel(panel, model_feature_symbols)
                mask_close = mask_panel.get("close")
                mask_latest_ts = (
                    mask_close.index.max()
                    if isinstance(mask_close, pd.DataFrame) and not mask_close.empty
                    else None
                )
                mask_eval_feats = _latest_only_features(
                    model_feats,
                    latest_ts=mask_latest_ts,
                    symbols=model_feature_symbols,
                )
                mask_eval_feats.update(
                    _latest_only_features(
                        selector_feats,
                        latest_ts=mask_latest_ts,
                        symbols=model_feature_symbols,
                    )
                )
                strategy_candidate_masks = build_strategy_candidate_masks(
                    mask_panel,
                    mask_eval_feats,
                    lgbm_strategy_mask_rows.values(),
                )
                timer.mark("lgbm_mask_eval")
                non_empty_masks = sum(
                    1 for symbols_ in strategy_candidate_masks.values() if symbols_
                )
                tprint(
                    "LGBM strategy masks latest pass: "
                    f"strategies={len(strategy_candidate_masks)} "
                    f"non_empty={non_empty_masks}"
                )
                if strategy_candidate_masks:
                    mask_diag = _strategy_mask_count_diagnostics(
                        strategy_candidate_masks,
                        lgbm_strategy_mask_rows,
                        model_feature_symbols,
                    )
                    tprint(
                        "LGBM strategy mask pass/fail counts: "
                        f"{ {strategy_core_id(k): {'pass': v.get('pass_count'), 'fail': v.get('fail_count'), 'universe': v.get('universe_count')} for k, v in mask_diag.items()} }"
                    )
            mask_long: set[str] = set()
            mask_short: set[str] = set()
            for strategy_id, passed_symbols in strategy_candidate_masks.items():
                row = lgbm_strategy_mask_rows.get(strategy_id, {}) or {}
                side = str(row.get("trade_side") or row.get("side") or "").lower()
                if side == "long":
                    mask_long.update(str(sym) for sym in passed_symbols)
                elif side == "short":
                    mask_short.update(str(sym) for sym in passed_symbols)
            if mask_long or mask_short:
                before_long = len(long_cands)
                before_short = len(short_cands)
                long_cands = sorted(set(long_cands).union(mask_long))
                short_cands = sorted(set(short_cands).union(mask_short))
                tprint(
                    "Deployment LGBM masks expanded candidate universe: "
                    f"long {before_long}->{len(long_cands)} "
                    f"short {before_short}->{len(short_cands)} "
                    f"mask_pass_long={len(mask_long)} "
                    f"mask_pass_short={len(mask_short)}"
                )
        allowed_model_symbols = set(source_allowed_model_symbols)
        if allowed_model_symbols:
            before_long = len(long_cands)
            before_short = len(short_cands)
            long_cands = [
                sym
                for sym in long_cands
                if sym in allowed_model_symbols and sym in tradable_symbol_set
            ]
            short_cands = [
                sym
                for sym in short_cands
                if sym in allowed_model_symbols and sym in tradable_symbol_set
            ]
            if before_long != len(long_cands) or before_short != len(short_cands):
                tprint(
                    "Live feature parity: removed candidates without required "
                    "source panels or current tradability "
                    f"long {before_long}->{len(long_cands)} "
                    f"short {before_short}->{len(short_cands)}"
                )
            if strategy_candidate_masks:
                strategy_candidate_masks = {
                    sid: [
                        sym
                        for sym in syms
                        if sym in allowed_model_symbols and sym in tradable_symbol_set
                    ]
                    for sid, syms in strategy_candidate_masks.items()
                }
        if strategy_candidate_masks and strategy_feature_contracts:
            mask_close = panel.get("close") if isinstance(panel, dict) else None
            latest_ts = (
                mask_close.index.max()
                if isinstance(mask_close, pd.DataFrame) and not mask_close.empty
                else None
            )
            before_by_strategy = {
                str(sid): len(syms or [])
                for sid, syms in strategy_candidate_masks.items()
            }
            strategy_candidate_masks, contract_diagnostics = (
                _filter_strategy_masks_by_finite_model_contract(
                    model_feats,
                    strategy_candidate_masks,
                    strategy_feature_contracts,
                    latest_ts=latest_ts,
                    cfg=cfg,
                )
            )
            if contract_diagnostics:
                for sid, diag in contract_diagnostics.items():
                    tprint(
                        "Live model contract parity: rejected mask-passed "
                        f"candidates for {strategy_core_id(sid)} "
                        f"{diag.get('input')}->{diag.get('kept')} "
                        f"missing_cols={diag.get('missing_cols', [])[:5]} "
                        f"blocking_nonfinite={diag.get('top_blocking_nonfinite_features', [])[:10]} "
                        f"allowed_nonfinite={diag.get('top_allowed_nonfinite_features', [])[:10]} "
                        f"top_nonfinite={diag.get('top_nonfinite_features', [])[:10]} "
                        f"sample={diag.get('sample_rejected_symbols', [])[:8]}"
                    )
                mask_long: set[str] = set()
                mask_short: set[str] = set()
                for strategy_id, passed_symbols in strategy_candidate_masks.items():
                    row = (lgbm_strategy_mask_rows or {}).get(strategy_id, {}) or {}
                    side = str(row.get("trade_side") or row.get("side") or "").lower()
                    if side == "long":
                        mask_long.update(str(sym) for sym in passed_symbols)
                    elif side == "short":
                        mask_short.update(str(sym) for sym in passed_symbols)
                before_long = len(long_cands)
                before_short = len(short_cands)
                long_cands = sorted(mask_long)
                short_cands = sorted(mask_short)
                tprint(
                    "Live model contract parity: pruned candidate universe "
                    f"long {before_long}->{len(long_cands)} "
                    f"short {before_short}->{len(short_cands)} "
                    f"strategy_counts_before={before_by_strategy} "
                    f"strategy_counts_after="
                    f"{ {str(k): len(v or []) for k, v in strategy_candidate_masks.items()} }"
                )
        selected_symbols = sorted(set(long_cands + short_cands))
        if not selected_symbols:
            result = (
                thresholds,
                long_cands,
                short_cands,
                model_feats,
                strategy_candidate_masks,
            )
            if cache_key:
                _CANDIDATE_FEATURE_CYCLE_CACHE.clear()
                _CANDIDATE_FEATURE_CYCLE_CACHE[cache_key] = {
                    "thresholds": thresholds,
                    "long_cands": long_cands,
                    "short_cands": short_cands,
                    "features": model_feats,
                    "strategy_candidate_masks": strategy_candidate_masks,
                }
            return result
        feature_covered_symbols, missing_feature_by_symbol = (
            _symbols_with_required_feature_coverage(
                model_feats,
                model_raw_feature_keys,
                selected_symbols,
            )
        )
        if len(feature_covered_symbols) != len(selected_symbols):
            covered = set(feature_covered_symbols)
            before_long = len(long_cands)
            before_short = len(short_cands)
            long_cands = [sym for sym in long_cands if sym in covered]
            short_cands = [sym for sym in short_cands if sym in covered]
            if strategy_candidate_masks:
                strategy_candidate_masks = {
                    sid: [sym for sym in syms if sym in covered]
                    for sid, syms in strategy_candidate_masks.items()
                }
            sample_missing = {
                sym: keys[:5]
                for sym, keys in list(missing_feature_by_symbol.items())[:8]
            }
            tprint(
                "Live feature parity: removed candidates missing required "
                "model feature columns "
                f"long {before_long}->{len(long_cands)} "
                f"short {before_short}->{len(short_cands)} "
                f"rejected={len(missing_feature_by_symbol)} "
                f"sample={sample_missing}"
            )
            selected_symbols = sorted(set(long_cands + short_cands))
            if not selected_symbols:
                result = (
                    thresholds,
                    long_cands,
                    short_cands,
                    model_feats,
                    strategy_candidate_masks,
                )
                if cache_key:
                    _CANDIDATE_FEATURE_CYCLE_CACHE.clear()
                    _CANDIDATE_FEATURE_CYCLE_CACHE[cache_key] = {
                        "thresholds": thresholds,
                        "long_cands": long_cands,
                        "short_cands": short_cands,
                        "features": model_feats,
                        "strategy_candidate_masks": strategy_candidate_masks,
                    }
                return result
        validate_required_feature_frames(
            model_feats,
            model_raw_feature_keys,
            symbols=selected_symbols,
            strict=True,
        )
        timer.mark("validate_model_features")
        result = (
            thresholds,
            long_cands,
            short_cands,
            model_feats,
            strategy_candidate_masks,
        )
        if cache_key:
            _CANDIDATE_FEATURE_CYCLE_CACHE.clear()
            _CANDIDATE_FEATURE_CYCLE_CACHE[cache_key] = {
                "thresholds": thresholds,
                "long_cands": long_cands,
                "short_cands": short_cands,
                "features": model_feats,
                "strategy_candidate_masks": strategy_candidate_masks,
            }
            tprint(
                "Cached hourly candidate feature/mask matrix: "
                f"key={cache_key} features={len(model_feats)} "
                f"strategies={len(strategy_candidate_masks)}"
            )
        return result


def _targeted_recent_gap_backfill(
    data_fetcher: DataFetcher,
    symbols: Iterable[str],
    *,
    days: int,
    max_symbols: int,
    label: str,
) -> Dict[str, str]:
    """Run bounded recent 15m repair for a small symbol set only."""
    if days <= 0:
        return {}
    unique_symbols: List[str] = []
    seen: set[str] = set()
    for symbol in symbols:
        sym = str(symbol).strip()
        if not sym or sym in seen:
            continue
        seen.add(sym)
        unique_symbols.append(sym)
    if not unique_symbols:
        return {}
    limit = max(0, int(max_symbols))
    if limit > 0 and len(unique_symbols) > limit:
        tprint(
            f"Targeted recent-gap 15m backfill [{label}]: limiting "
            f"{len(unique_symbols)} symbols to {limit}"
        )
        unique_symbols = unique_symbols[:limit]

    results: Dict[str, str] = {}
    checked = 0
    skipped = 0
    backfilled = 0
    failed = 0
    start = time.monotonic()
    for symbol in unique_symbols:
        checked += 1
        try:
            if not data_fetcher.has_recent_gap(symbol, days=days):
                skipped += 1
                results[symbol] = "no_recent_gap"
                continue
            frame = data_fetcher.trigger_gap_backfill(symbol, days=days)
            if hasattr(data_fetcher, "_invalidate_symbol_cache"):
                data_fetcher._invalidate_symbol_cache(symbol, microdata=False)
            backfilled += 1
            if isinstance(frame, pd.DataFrame):
                results[symbol] = f"backfilled_rows={len(frame)}"
            else:
                results[symbol] = "backfilled"
        except Exception as exc:
            failed += 1
            results[symbol] = f"failed:{classify_api_error(exc)}"
            tprint(
                f"Targeted recent-gap 15m backfill [{label}] failed for "
                f"{symbol}: {classify_api_error(exc)}: {exc}"
            )
    elapsed = time.monotonic() - start
    tprint(
        f"Targeted recent-gap 15m backfill [{label}] complete: "
        f"checked={checked} skipped_no_gap={skipped} backfilled={backfilled} "
        f"failed={failed} days={days} elapsed={elapsed:.2f}s"
    )
    return results


def _is_symbol_cooldown_blocked(
    symbol: str,
    *,
    now: pd.Timestamp,
    logger: TradeLogger,
    executor: TradeExecutor,
    cooldown_hours: float,
) -> bool:
    """Return True if symbol is active or recently closed at negative PnL."""
    active = (
        executor.get_active_positions()
        if hasattr(executor, "get_active_positions")
        else {}
    )
    if symbol in active:
        return True
    if cooldown_hours <= 0:
        return False
    if hasattr(logger, "get_last_losing_trade_timestamp"):
        last_ts = logger.get_last_losing_trade_timestamp(symbol)
    else:
        last_ts = None
    if last_ts is None:
        return False
    return pd.Timestamp(now) < (
        pd.Timestamp(last_ts) + pd.Timedelta(hours=float(cooldown_hours))
    )


def _symbol_entry_block_reason(
    symbol: str,
    *,
    now: pd.Timestamp,
    logger: TradeLogger,
    executor: TradeExecutor,
    cooldown_hours: float,
) -> str:
    """Return symbol-level entry block reason, or empty string when allowed."""
    active = (
        executor.get_active_positions()
        if hasattr(executor, "get_active_positions")
        else {}
    )
    if symbol in active:
        return "symbol_already_active"
    if cooldown_hours <= 0:
        return ""
    if hasattr(logger, "get_last_losing_trade_timestamp"):
        last_ts = logger.get_last_losing_trade_timestamp(symbol)
    else:
        last_ts = None
    if last_ts is None:
        return ""
    blocked_until = pd.Timestamp(last_ts) + pd.Timedelta(hours=float(cooldown_hours))
    return "recent_losing_trade_cooldown" if pd.Timestamp(now) < blocked_until else ""


def _format_pct(value: Any) -> str:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return f"{100.0 * x:.4f}%" if np.isfinite(x) else "n/a"


def _format_float(value: Any, digits: int = 8) -> str:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return f"{x:.{digits}f}" if np.isfinite(x) else "n/a"


def _safe_float(value: Any, default: float = np.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def _candidate_rank_score(decision: Mapping[str, Any]) -> float:
    """Return the cross-strategy rank score used by the portfolio policy."""
    return _safe_float(
        decision.get(
            "normalized_rank_score",
            decision.get(
                "policy_rank_pct",
                decision.get(
                    "sizer_rank_percentile",
                    decision.get("threshold_score", decision.get("calibrated_score")),
                ),
            ),
        )
    )


def _candidate_portfolio_priority(decision: Mapping[str, Any]) -> float:
    """Portfolio optimiser priority: rank surplus above dynamic threshold minus costs."""
    rank_score = _candidate_rank_score(decision)
    dynamic_threshold = _safe_float(
        decision.get(
            "dynamic_threshold",
            decision.get("effective_threshold", decision.get("normalized_threshold")),
        )
    )
    if not (np.isfinite(rank_score) and np.isfinite(dynamic_threshold)):
        return -float("inf")
    denom = max(1.0 - float(dynamic_threshold), 1e-9)
    surplus = (float(rank_score) - float(dynamic_threshold)) / denom
    penalty = _safe_float(decision.get("price_gap_penalty"), 0.0)
    if not np.isfinite(penalty):
        penalty = 0.0
    expected_friction_bps = _safe_float(
        decision.get(
            "expected_friction_bps",
            decision.get(
                "expected_total_entry_friction_bps",
                decision.get("orderbook_slippage_bps", 0.0),
            ),
        ),
        0.0,
    )
    friction_penalty = max(float(expected_friction_bps), 0.0) / 10000.0
    return float(surplus - float(penalty) - friction_penalty)


def _auction_ev_target_for_occupancy(
    *,
    open_positions: int,
    policy: PortfolioPolicyConfig,
) -> float:
    """Progressively require more realised EV as portfolio capacity fills."""
    max_positions = max(int(getattr(policy, "max_concurrent_positions", 1) or 1), 1)
    occupancy = float(np.clip(int(open_positions) / max_positions, 0.0, 1.0))
    progress = occupancy
    lo = float(AUCTION_EV_MIN_NET_RETURN)
    hi = float(AUCTION_EV_MAX_NET_RETURN)
    if hi < lo:
        lo, hi = hi, lo
    return float(lo + (hi - lo) * progress)


def _candidate_expected_friction(decision: Mapping[str, Any]) -> float:
    return _safe_float(
        decision.get(
            "expected_friction_bps",
            decision.get(
                "expected_total_entry_friction_bps",
                decision.get("orderbook_slippage_bps", 0.0),
            ),
        ),
        0.0,
    )


def _json_safe_audit_value(value: Any) -> Any:
    """Return a JSON-stable scalar/container for trade decision audit blobs."""
    if value is None:
        return None
    if isinstance(value, (pd.Timestamp, datetime)):
        ts = pd.to_datetime(value, utc=True, errors="coerce")
        return None if pd.isna(ts) else pd.Timestamp(ts).isoformat()
    if isinstance(value, np.generic):
        return _json_safe_audit_value(value.item())
    if isinstance(value, float):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe_audit_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe_audit_value(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_json_safe_audit_value(v) for v in value.tolist()]
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    try:
        raw = float(value)
        return raw if np.isfinite(raw) else None
    except (TypeError, ValueError):
        return str(value)


def _latest_symbol_value(
    frame: Any,
    symbol: str,
    *,
    ts: Optional[Any] = None,
) -> Dict[str, Any]:
    """Return the latest non-null symbol value at or before ``ts`` from a panel frame."""
    if not isinstance(frame, pd.DataFrame) or symbol not in frame.columns:
        return {"value": None, "ts": None, "available": False}
    series = frame[symbol].dropna()
    if series.empty:
        return {"value": None, "ts": None, "available": False}
    if ts is not None:
        try:
            target = pd.to_datetime(ts, utc=True, errors="coerce")
            idx_ts = pd.to_datetime(series.index, utc=True, errors="coerce")
            if not pd.isna(target):
                mask = idx_ts <= target
                if np.any(mask):
                    series = series.loc[mask]
                else:
                    return {"value": None, "ts": None, "available": False}
        except Exception:
            pass
    if series.empty:
        return {"value": None, "ts": None, "available": False}
    return {
        "value": _json_safe_audit_value(series.iloc[-1]),
        "ts": _json_safe_audit_value(series.index[-1]),
        "available": True,
    }


def _resolve_alpha_model_info_for_audit(
    orchestrator: Any,
    *,
    side: str,
    strategy_id: str,
) -> tuple[str, Optional[Dict[str, Any]]]:
    alpha = getattr(orchestrator, "alpha_by_strategy", {}) or {}
    side_s = str(side or "").lower()
    candidates = [
        str(strategy_id),
        strategy_core_id(str(strategy_id)),
        f"{side_s}_{strategy_id}" if side_s else "",
        f"{side_s}_{strategy_core_id(str(strategy_id))}" if side_s else "",
    ]
    if not side_s:
        core = strategy_core_id(str(strategy_id))
        candidates.extend(
            [
                f"long_{strategy_id}",
                f"short_{strategy_id}",
                f"long_{core}" if core else "",
                f"short_{core}" if core else "",
            ]
        )
    for key in candidates:
        if not key:
            continue
        info = alpha.get(key)
        if isinstance(info, dict):
            return key, info
    return str(strategy_id), None


def _resolve_meta_model_for_audit(
    orchestrator: Any,
    *,
    side: str,
    strategy_id: str,
) -> tuple[str, Any]:
    meta_models = getattr(orchestrator, "meta_models", {}) or {}
    core = strategy_core_id(str(strategy_id))
    candidates = [
        str(strategy_id),
        core,
        f"{side}_{strategy_id}",
        f"{side}_{core}",
        f"{strategy_id}_clf",
        f"{core}_clf",
        f"{side}_{strategy_id}_clf",
        f"{side}_{core}_clf",
        f"{strategy_id}_tbm_clf",
        f"{core}_tbm_clf",
        f"{side}_{strategy_id}_tbm_clf",
        f"{side}_{core}_tbm_clf",
    ]
    for key in candidates:
        if key in meta_models:
            return key, meta_models.get(key)
    return str(strategy_id), None


def _model_feature_contracts_for_audit(
    orchestrator: Any,
    *,
    side: str,
    strategy_id: str,
) -> Dict[str, Any]:
    base_key, alpha_info = _resolve_alpha_model_info_for_audit(
        orchestrator, side=side, strategy_id=strategy_id
    )
    base_features = (
        _effective_alpha_feature_contract(alpha_info)
        if isinstance(alpha_info, dict)
        else []
    )
    meta_key, meta_model = _resolve_meta_model_for_audit(
        orchestrator, side=side, strategy_id=strategy_id
    )
    meta_features = _effective_selected_feature_contract(meta_model)
    if (
        not meta_features
        and meta_model is not None
        and hasattr(meta_model, "feature_columns")
    ):
        meta_features = [
            str(c) for c in (getattr(meta_model, "feature_columns", []) or [])
        ]
    base_features = [
        str(c)
        for c in (base_features or [])
        if str(c) not in DELETED_MODEL_FEATURE_KEYS
    ]
    meta_features = [
        str(c)
        for c in (meta_features or [])
        if str(c) not in DELETED_MODEL_FEATURE_KEYS
    ]
    return {
        "base_model_key": base_key,
        "meta_model_key": meta_key,
        "base_features": base_features,
        "meta_features": meta_features,
        "all_features": sorted(set(base_features).union(meta_features)),
    }


def _feature_audit_value(
    feature_name: str,
    *,
    symbol: str,
    candidate_features: Optional[pd.DataFrame],
    feats: Dict[str, pd.DataFrame],
    chain_results: Dict[str, Any],
    strategy_id: str,
    signal_bar_ts: Optional[Any],
) -> tuple[Any, str]:
    if (
        isinstance(candidate_features, pd.DataFrame)
        and symbol in candidate_features.index
        and feature_name in candidate_features.columns
    ):
        return (
            _json_safe_audit_value(candidate_features.at[symbol, feature_name]),
            "candidate_features",
        )
    if feature_name in feats:
        latest = _latest_symbol_value(feats.get(feature_name), symbol, ts=signal_bar_ts)
        if latest.get("available"):
            return latest.get("value"), f"feature_frame:{latest.get('ts')}"
    core = strategy_core_id(str(strategy_id))
    if feature_name in {str(strategy_id), core}:
        return _json_safe_audit_value(chain_results.get("base_pred")), "base_pred"
    if re.match(r"^pred(?:_.*)?_H\d+$", str(feature_name)):
        return _json_safe_audit_value(chain_results.get("base_pred")), "base_pred_alias"
    if re.match(r"^pred_logit(?:_H\d+)?$", str(feature_name)):
        base_pred = _safe_float(chain_results.get("base_pred"), np.nan)
        if np.isfinite(base_pred):
            base_prob = float(np.clip(base_pred, 1e-6, 1.0 - 1e-6))
            return _json_safe_audit_value(np.log(base_prob / (1.0 - base_prob))), "base_pred_logit"
    if feature_name in chain_results:
        return _json_safe_audit_value(chain_results.get(feature_name)), "chain_results"
    diagnostics = chain_results.get("lgbm_diagnostics")
    if isinstance(diagnostics, dict) and feature_name in diagnostics:
        return _json_safe_audit_value(diagnostics.get(feature_name)), "lgbm_diagnostics"
    return None, "missing"


def _materialized_meta_audit_features(
    *,
    orchestrator: Any,
    meta_model: Any,
    side: str,
    strategy_id: str,
    symbol: str,
    candidate_features: Optional[pd.DataFrame],
    chain_results: Dict[str, Any],
) -> Optional[pd.DataFrame]:
    if meta_model is None or not isinstance(candidate_features, pd.DataFrame):
        return candidate_features
    if symbol not in candidate_features.index:
        return candidate_features
    out = candidate_features.loc[[symbol]].copy()
    base_pred = chain_results.get("base_pred")
    aliases = {
        str(strategy_id),
        strategy_core_id(str(strategy_id)),
        f"{side}_{strategy_id}",
        f"{side}_{strategy_core_id(str(strategy_id))}",
    }
    for alias in aliases:
        if alias:
            out[str(alias)] = base_pred
    for method_name in (
        "_materialize_alpha_model_meta_features",
        "_materialize_meta_model_drift_features",
        "_materialize_meta_model_derived_features",
    ):
        method = getattr(orchestrator, method_name, None)
        if not callable(method):
            continue
        try:
            if method_name == "_materialize_alpha_model_meta_features":
                out = method(out, meta_model, side=side, kind=strategy_id)
            elif method_name == "_materialize_meta_model_drift_features":
                out = method(out, meta_model)
            else:
                out = method(out, meta_model, side=side, kind=strategy_id)
        except Exception:
            continue
    return out


def _model_feature_audit_for_trade(
    *,
    orchestrator: Any,
    side: str,
    strategy_id: str,
    symbol: str,
    candidate_features: Optional[pd.DataFrame],
    meta_model_input_features: Optional[pd.DataFrame] = None,
    feats: Dict[str, pd.DataFrame],
    chain_results: Dict[str, Any],
    signal_bar_ts: Optional[Any],
) -> Dict[str, Any]:
    contracts = _model_feature_contracts_for_audit(
        orchestrator, side=side, strategy_id=strategy_id
    )
    out: Dict[str, Any] = {
        "base_model_key": contracts.get("base_model_key"),
        "meta_model_key": contracts.get("meta_model_key"),
        "base": {},
        "meta": {},
        "sources": {},
        "missing": {"base": [], "meta": []},
    }
    _, meta_model = _resolve_meta_model_for_audit(
        orchestrator, side=side, strategy_id=strategy_id
    )
    meta_candidate_features = None
    if isinstance(meta_model_input_features, pd.DataFrame):
        if symbol in meta_model_input_features.index:
            meta_candidate_features = meta_model_input_features.loc[[symbol]]
    if meta_candidate_features is None:
        meta_candidate_features = _materialized_meta_audit_features(
            orchestrator=orchestrator,
            meta_model=meta_model,
            side=side,
            strategy_id=strategy_id,
            symbol=symbol,
            candidate_features=candidate_features,
            chain_results=chain_results,
        )
    for scope in ("base", "meta"):
        for feat_name in contracts.get(f"{scope}_features", []) or []:
            scope_candidate_features = (
                meta_candidate_features if scope == "meta" else candidate_features
            )
            value, source = _feature_audit_value(
                str(feat_name),
                symbol=symbol,
                candidate_features=scope_candidate_features,
                feats=feats,
                chain_results=chain_results,
                strategy_id=strategy_id,
                signal_bar_ts=signal_bar_ts,
            )
            out[scope][str(feat_name)] = value
            out["sources"][str(feat_name)] = source
            if source == "missing":
                out["missing"][scope].append(str(feat_name))
    return _json_safe_audit_value(out)


def _audit_json_dumps(value: Any) -> str:
    return json.dumps(
        _json_safe_audit_value(value),
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _attach_portfolio_replay_state_for_ledger(
    decision: Dict[str, Any],
    *,
    portfolio_mgr: Optional[PortfolioManager],
    capacity: Optional[Mapping[str, Any]] = None,
    now_utc: Optional[pd.Timestamp] = None,
) -> None:
    """Attach compact pre-decision portfolio state needed for exact replay."""
    if portfolio_mgr is None:
        return
    try:
        state = dict(portfolio_mgr.get_portfolio_state())
    except Exception as exc:
        decision["portfolio_state_snapshot_error"] = str(exc)
        return
    try:
        open_df = portfolio_mgr.get_open_positions_summary()
    except Exception:
        open_df = pd.DataFrame()

    if isinstance(open_df, pd.DataFrame) and not open_df.empty:
        open_positions = _json_safe_audit_value(
            open_df.sort_values(
                [col for col in ("symbol", "side", "strategy_id") if col in open_df.columns]
            ).to_dict(orient="records")
        )
    else:
        open_positions = []

    cooldowns = state.get("active_cooldowns", {})
    if not isinstance(cooldowns, Mapping):
        cooldowns = {}
    capacity_payload = {
        str(k): _json_safe_audit_value(v)
        for k, v in dict(capacity or {}).items()
        if k
        in {
            "wallet_value",
            "open_notional",
            "available_wallet_quote",
            "total_assets_quote",
            "total_liabilities_quote",
            "open_positions",
            "max_total_notional",
            "remaining_total_notional",
            "position_size_cap",
        }
    }
    snapshot = {
        "schema": "portfolio_replay_state_v1",
        "asof": _json_safe_audit_value(now_utc or pd.Timestamp.now(tz="UTC")),
        "open_positions": open_positions,
        "cooldowns": _json_safe_audit_value(cooldowns),
        "state_summary": {
            key: _json_safe_audit_value(state.get(key))
            for key in (
                "n_positions",
                "max_positions",
                "long_count",
                "short_count",
                "invested_pct",
                "remaining_pct",
                "strategy_counts",
            )
            if key in state
        },
        "capacity": capacity_payload,
    }
    snapshot_json = _audit_json_dumps(snapshot)
    decision["portfolio_state_snapshot_json"] = snapshot_json
    decision["portfolio_state_snapshot_hash"] = hashlib.sha256(
        snapshot_json.encode("utf-8")
    ).hexdigest()
    decision["open_positions_before_json"] = _audit_json_dumps(open_positions)
    decision["cooldowns_before_json"] = _audit_json_dumps(cooldowns)
    decision["open_positions_before"] = int(len(open_positions))
    decision["open_positions_before_count"] = int(len(open_positions))
    decision["wallet_before"] = capacity_payload.get(
        "wallet_value", state.get("portfolio_value")
    )
    decision["open_notional_before"] = capacity_payload.get("open_notional")
    decision["available_wallet_before"] = capacity_payload.get(
        "available_wallet_quote"
    )


def _model_feature_ledger_snapshot_for_decision(
    *,
    orchestrator: Any,
    side: str,
    strategy_id: str,
    symbol: str,
    candidate_features: Optional[pd.DataFrame],
    meta_model_input_features: Optional[pd.DataFrame] = None,
    feats: Dict[str, pd.DataFrame],
    chain_results: Dict[str, Any],
    signal_bar_ts: Optional[Any],
) -> Dict[str, Any]:
    """Return compact selected-model-feature snapshots for the prediction ledger."""
    audit = _model_feature_audit_for_trade(
        orchestrator=orchestrator,
        side=side,
        strategy_id=strategy_id,
        symbol=symbol,
        candidate_features=candidate_features,
        meta_model_input_features=meta_model_input_features,
        feats=feats,
        chain_results=chain_results,
        signal_bar_ts=signal_bar_ts,
    )
    base_values = audit.get("base") if isinstance(audit.get("base"), dict) else {}
    meta_values = audit.get("meta") if isinstance(audit.get("meta"), dict) else {}
    sources = audit.get("sources") if isinstance(audit.get("sources"), dict) else {}
    missing = audit.get("missing") if isinstance(audit.get("missing"), dict) else {}
    payload = {
        "schema": "selected_model_features_v1",
        "symbol": symbol,
        "side": side,
        "strategy_id": strategy_id,
        "base_model_key": audit.get("base_model_key"),
        "meta_model_key": audit.get("meta_model_key"),
        "base_features": list(base_values.keys()),
        "meta_features": list(meta_values.keys()),
        "base_values": base_values,
        "meta_values": meta_values,
        "sources": sources,
        "missing": missing,
    }
    snapshot_json = _audit_json_dumps(payload)
    return {
        "model_feature_audit_schema": "selected_model_features_v1",
        "model_feature_snapshot_hash": hashlib.sha256(
            snapshot_json.encode("utf-8")
        ).hexdigest(),
        "base_model_key": audit.get("base_model_key"),
        "meta_model_feature_key": audit.get("meta_model_key"),
        "base_model_feature_count": len(base_values),
        "meta_model_feature_count": len(meta_values),
        "base_model_features_json": _audit_json_dumps(list(base_values.keys())),
        "meta_model_features_json": _audit_json_dumps(list(meta_values.keys())),
        "base_model_feature_values_json": _audit_json_dumps(base_values),
        "meta_model_feature_values_json": _audit_json_dumps(meta_values),
        "model_feature_value_sources_json": _audit_json_dumps(sources),
        "model_feature_missing_json": _audit_json_dumps(missing),
    }


def _raw_data_audit_for_trade(
    *,
    panel: Dict[str, pd.DataFrame],
    symbol: str,
    signal_bar_ts: Optional[Any],
) -> Dict[str, Any]:
    raw: Dict[str, Any] = {}
    for key, frame in (panel or {}).items():
        latest = _latest_symbol_value(frame, symbol, ts=signal_bar_ts)
        if latest.get("available"):
            raw[str(key)] = {
                "value": latest.get("value"),
                "ts": latest.get("ts"),
            }
    return raw


def _prediction_audit_for_trade(
    *,
    decision: Dict[str, Any],
    chain_results: Dict[str, Any],
    execution_snapshot: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    chain = dict(chain_results or {})
    snap = dict(execution_snapshot or {})
    sizing = dict(chain.get("portfolio_rank_sizing") or {})
    audit = {
        "base": {
            "pred": chain.get("base_pred"),
            "batch_rank_pct": chain.get("base_rank_pct"),
            "train_rank_pct": chain.get("base_train_rank_pct"),
            "gate_top_frac": chain.get("base_gate_top_frac"),
        },
        "meta": {
            "pred": chain.get("meta_pred", decision.get("raw_score")),
            "raw_prediction_score": decision.get("raw_score"),
            "calibrated_score": chain.get(
                "calibrated_score", decision.get("calibrated_score")
            ),
            "train_rank_pct": chain.get("meta_train_rank_pct"),
        },
        "ranking": {
            "rank_score": decision.get("rank_score"),
            "threshold_score": decision.get("threshold_score"),
            "normalized_rank_score": chain.get(
                "normalized_rank_score", decision.get("normalized_rank_score")
            ),
            "policy_rank_pct": chain.get(
                "policy_rank_pct", decision.get("policy_rank_pct")
            ),
            "auction_rank_pct": chain.get(
                "auction_rank_pct", decision.get("auction_rank_pct")
            ),
            "sizer_rank_percentile": chain.get("sizer_rank_percentile"),
            "adjusted_rank_score": snap.get("adjusted_rank_score"),
            "rank_score_source": chain.get(
                "rank_score_source", decision.get("rank_score_source")
            ),
        },
        "thresholds": {
            "rank_threshold": decision.get("rank_threshold"),
            "normalized_threshold": decision.get("normalized_threshold"),
            "deployment_rank_threshold": decision.get("deployment_rank_threshold"),
            "effective_threshold": chain.get(
                "effective_threshold", decision.get("effective_threshold")
            ),
            "final_threshold": snap.get("final_threshold"),
        },
        "sizing": sizing,
    }
    return _json_safe_audit_value(audit)


def _build_trade_start_audit(
    *,
    orchestrator: Any,
    panel: Dict[str, pd.DataFrame],
    feats: Dict[str, pd.DataFrame],
    candidate_features: Optional[pd.DataFrame],
    meta_model_input_features: Optional[pd.DataFrame] = None,
    symbol: str,
    side: str,
    strategy_id: str,
    signal_bar_ts: Optional[Any],
    decision: Dict[str, Any],
    chain_results: Dict[str, Any],
    execution_snapshot: Optional[Dict[str, Any]] = None,
    parity_contract: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    contract = parity_contract if isinstance(parity_contract, dict) else {}
    return {
        "decision_audit_schema": "inference_trade_start_audit_v1",
        "training_live_parity_contract": _json_safe_audit_value(
            {
                "schema_version": contract.get("schema_version"),
                "path": contract.get("_contract_path"),
                "sha256": contract.get("_contract_sha256"),
                "artifact_hashes": contract.get("artifact_hashes") or {},
                "strategy_contract": contract.get("strategy_contract") or {},
                "rank_normalization": contract.get("rank_normalization") or {},
            }
        ),
        "model_prediction_audit": _prediction_audit_for_trade(
            decision=decision,
            chain_results=chain_results,
            execution_snapshot=execution_snapshot,
        ),
        "raw_data_audit": _raw_data_audit_for_trade(
            panel=panel, symbol=symbol, signal_bar_ts=signal_bar_ts
        ),
        "model_feature_audit": _model_feature_audit_for_trade(
            orchestrator=orchestrator,
            side=side,
            strategy_id=strategy_id,
            symbol=symbol,
            candidate_features=candidate_features,
            meta_model_input_features=meta_model_input_features,
            feats=feats,
            chain_results=chain_results,
            signal_bar_ts=signal_bar_ts,
        ),
    }


def _holding_time_hours(entry_time: Any, exit_time: Any) -> float:
    entry_ts = pd.to_datetime(entry_time, utc=True, errors="coerce")
    exit_ts = pd.to_datetime(exit_time, utc=True, errors="coerce")
    if pd.isna(entry_ts) or pd.isna(exit_ts):
        return np.nan
    return float(
        (pd.Timestamp(exit_ts) - pd.Timestamp(entry_ts)).total_seconds() / 3600.0
    )


_PERP_RANK_CONTEXT_CACHE: Dict[str, Dict[str, Any]] = {}


def _normalise_market_mode(mode: Any) -> str:
    raw = str(mode or "spot").strip().lower()
    return "perps" if raw in {"perp", "perps", "future", "futures", "swap"} else "spot"


def _is_perps_config(config: Dict[str, Any]) -> bool:
    return _normalise_market_mode(config.get("market_mode")) == "perps"


def _live_exchange_symbol(exchange: Any, config: Dict[str, Any], symbol: str) -> str:
    if not _is_perps_config(config) or ":" in str(symbol):
        return symbol
    return _resolve_perp_symbol(exchange, symbol) or symbol


def _perp_rank_context(
    *,
    data_root: str,
    run_id: str,
    side: str,
    strategy_id: str,
    score: float,
) -> Dict[str, Any]:
    """Map a live OOF-like score to rank and profitable rank cutoff for perps."""
    cache_key = f"{data_root}|{run_id}|{side}|{strategy_id}"
    cached = _PERP_RANK_CONTEXT_CACHE.get(cache_key)
    if cached is None:
        core = strategy_core_id(strategy_id)
        path = (
            Path(data_root)
            / "artifacts"
            / str(run_id)
            / "meta_oof"
            / f"meta_oof_{side}_{core}_clf.parquet"
        )
        cached = {"rank_x": 1, "scores": np.array([], dtype=float)}
        try:
            df = pd.read_parquet(path)
            pred_col = next(
                (
                    c
                    for c in ("oof_pred", "oof_meta_pred", "meta_pred", "pred")
                    if c in df.columns
                ),
                None,
            )
            target_col = next(
                (c for c in ("y_bin", "target", "label", "y") if c in df.columns),
                None,
            )
            if pred_col and target_col:
                scores = pd.to_numeric(df[pred_col], errors="coerce").to_numpy(
                    dtype=float
                )
                target = pd.to_numeric(df[target_col], errors="coerce").to_numpy(
                    dtype=float
                )
                mask = np.isfinite(scores) & np.isfinite(target)
                scores = scores[mask]
                target = target[mask]
                order = np.argsort(-scores)
                scores_sorted = scores[order]
                target_sorted = target[order]
                ranks = np.arange(1, len(scores_sorted) + 1, dtype=float)
                expected = None
                if len(scores_sorted) >= 20 and np.unique(target_sorted).size > 1:
                    try:
                        from sklearn.isotonic import IsotonicRegression

                        iso = IsotonicRegression(increasing=False, out_of_bounds="clip")
                        iso.fit(ranks, target_sorted)
                        expected = np.asarray(iso.predict(ranks), dtype=float)
                    except Exception:
                        expected = None
                if expected is None:
                    buckets = pd.qcut(
                        ranks, q=min(50, max(1, len(ranks) // 100)), duplicates="drop"
                    )
                    means = (
                        pd.Series(target_sorted)
                        .groupby(buckets, observed=True)
                        .transform("mean")
                    )
                    expected = means.to_numpy(dtype=float)
                profitable = np.flatnonzero(expected >= 0.50)
                profitable_rank_count = (
                    int(profitable[-1] + 1) if len(profitable) else 1
                )
                cached = {
                    "rank_x": max(1, profitable_rank_count // 2),
                    "profitable_rank_count": profitable_rank_count,
                    "scores": scores_sorted,
                }
        except Exception as exc:
            tprint(
                f"Warning: failed to load perp OOF rank context for {side}/{strategy_id}: {exc}"
            )
        _PERP_RANK_CONTEXT_CACHE[cache_key] = cached
    scores_sorted = cached.get("scores")
    rank = 1
    if isinstance(scores_sorted, np.ndarray) and scores_sorted.size:
        rank = int(np.searchsorted(-scores_sorted, -float(score), side="left") + 1)
    return {
        "rank_number": max(1, rank),
        "rank_x": int(cached.get("rank_x") or 1),
        "profitable_rank_count": int(cached.get("profitable_rank_count") or 1),
    }


def _send_trade_close_email(
    *,
    closed_trade: Dict[str, Any],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Send a per-trade close notification using the daily reporter SMTP config."""
    if not bool(config.get("trade_close_email_enabled", True)):
        return {"sent": False, "reason": "disabled"}
    recipient = str(
        config.get("trade_close_email_to")
        or config.get("daily_report_email_to")
        or os.environ.get("EPM_TRADE_EMAIL_TO")
        or os.environ.get("EPM_REPORT_EMAIL_TO")
        or ""
    ).strip()
    if not recipient:
        return {"sent": False, "reason": "missing_recipient"}

    symbol = str(closed_trade.get("symbol") or "unknown")
    side = str(closed_trade.get("side") or "unknown")
    reason = str(closed_trade.get("reason") or "closed")
    reason_detail = str(closed_trade.get("exit_reason_detail") or reason)
    subject = f"EPM trade closed: {symbol} {side} {reason}"
    gross_pnl_pct = closed_trade.get("gross_pnl_pct")
    net_pnl_pct = closed_trade.get("net_pnl_pct")
    net_pnl_amount = closed_trade.get("net_pnl_amount", closed_trade.get("net_pnl"))
    holding_time_hours = _safe_float(closed_trade.get("holding_time_hours"))
    if not np.isfinite(holding_time_hours):
        holding_time_hours = _holding_time_hours(
            closed_trade.get("entry_time"),
            closed_trade.get("exit_time"),
        )
    trade_recap = str(closed_trade.get("trade_recap") or "").strip()
    body = "\n".join(
        [
            "Extreme price movement trade close",
            "",
            "Trade",
            f"  market_mode: {config.get('market_mode', 'spot')}",
            f"  symbol: {symbol}",
            f"  side: {side}",
            f"  strategy_id: {closed_trade.get('strategy_id')}",
            f"  exit_reason: {reason}",
            f"  exit_reason_detail: {reason_detail}",
            f"entry_time: {closed_trade.get('entry_time')}",
            f"exit_time: {closed_trade.get('exit_time')}",
            f"holding_time_hours: {_format_float(holding_time_hours, digits=4)}",
            f"entry_price: {_format_float(closed_trade.get('entry_price'))}",
            f"entry_order_type: {closed_trade.get('entry_order_type')}",
            f"ohlcv_entry_price: {_format_float(closed_trade.get('ohlcv_entry_price'))}",
            "entry_price_delta_vs_ohlcv: "
            f"{_format_float(closed_trade.get('entry_price_delta_vs_ohlcv'))}",
            "entry_price_delta_vs_ohlcv_pct: "
            f"{_format_pct(closed_trade.get('entry_price_delta_vs_ohlcv_pct'))}",
            f"signal_price: {_format_float(closed_trade.get('signal_price'))}",
            f"decision_mid: {_format_float(closed_trade.get('decision_mid'))}",
            f"signal_gap_bps: {_format_float(closed_trade.get('signal_gap_bps'), digits=4)}",
            f"ticker_bid: {_format_float(closed_trade.get('ticker_bid'))}",
            f"ticker_ask: {_format_float(closed_trade.get('ticker_ask'))}",
            f"ticker_mid: {_format_float(closed_trade.get('ticker_mid'))}",
            f"ticker_spread_bps: {_format_float(closed_trade.get('ticker_spread_bps'), digits=4)}",
            "expected_fill_price: "
            f"{_format_float(closed_trade.get('expected_fill_price'))}",
            "expected_fill_slippage_bps: "
            f"{_format_float(closed_trade.get('expected_fill_slippage_bps'), digits=4)}",
            "expected_total_entry_friction_bps: "
            f"{_format_float(closed_trade.get('expected_total_entry_friction_bps'), digits=4)}",
            "orderbook_capacity_quote_within_slippage: "
            f"{_format_float(closed_trade.get('orderbook_capacity_quote_within_slippage'), digits=4)}",
            "liquidity_capacity_weight: "
            f"{_format_float(closed_trade.get('liquidity_capacity_weight'), digits=4)}",
            f"perp_effective_leverage: {_format_float(closed_trade.get('perp_effective_leverage'), digits=4)}",
            f"perp_rank_leverage: {_format_float(closed_trade.get('perp_rank_leverage'), digits=4)}",
            f"perp_risk_cap_leverage: {_format_float(closed_trade.get('perp_risk_cap_leverage'), digits=4)}",
            f"perp_rank_number: {closed_trade.get('perp_rank_number')}",
            f"perp_rank_x: {closed_trade.get('perp_rank_x')}",
            f"perp_stop_loss_pct: {_format_float(closed_trade.get('perp_stop_loss_pct'), digits=4)}",
            f"perp_full_wallet: {_format_float(closed_trade.get('perp_full_wallet'), digits=4)}",
            f"perp_available_wallet: {_format_float(closed_trade.get('perp_available_wallet'), digits=4)}",
            f"base_pred: {_format_float(closed_trade.get('base_pred'), digits=6)}",
            f"base_rank_pct: {_format_float(closed_trade.get('base_rank_pct'), digits=6)}",
            f"base_train_rank_pct: {_format_float(closed_trade.get('base_train_rank_pct'), digits=6)}",
            f"base_gate_top_frac: {closed_trade.get('base_gate_top_frac')}",
            f"meta_pred: {_format_float(closed_trade.get('meta_pred'), digits=6)}",
            f"meta_train_rank_pct: {_format_float(closed_trade.get('meta_train_rank_pct'), digits=6)}",
            f"rank_score_source: {closed_trade.get('rank_score_source')}",
            f"policy_rank_pct: {_format_float(closed_trade.get('policy_rank_pct'), digits=6)}",
            f"policy_rank_reference_n: {closed_trade.get('policy_rank_reference_n')}",
            "policy_rank_reference_source: "
            f"{closed_trade.get('policy_rank_reference_source')}",
            f"calibrated_score: {_format_float(closed_trade.get('calibrated_score'), digits=6)}",
            f"rank_percentile: {_format_float(closed_trade.get('rank_percentile'), digits=6)}",
            f"effective_threshold: {_format_float(closed_trade.get('effective_threshold'), digits=6)}",
            "deployment_rank_threshold: "
            f"{_format_float(closed_trade.get('deployment_rank_threshold'), digits=6)}",
            f"exit_price: {_format_float(closed_trade.get('exit_price'))}",
            f"filled: {_format_float(closed_trade.get('filled'))}",
            "entry_notional_quote: "
            f"{_format_float(closed_trade.get('entry_notional_quote'))}",
            "exit_notional_quote: "
            f"{_format_float(closed_trade.get('exit_notional_quote'))}",
            "wallet_value_at_entry: "
            f"{_format_float(closed_trade.get('wallet_value_at_entry'))}",
            "open_notional_at_entry: "
            f"{_format_float(closed_trade.get('open_notional_at_entry'))}",
            "leverage_wallet_multiplier: "
            f"{_format_float(closed_trade.get('leverage_wallet_multiplier'), digits=4)}x",
            "effective_position_leverage: "
            f"{_format_float(closed_trade.get('effective_position_leverage'), digits=4)}x",
            f"requested_quote_size: {_format_float(closed_trade.get('quote_size'))}",
            "requested_base_amount: "
            f"{_format_float(closed_trade.get('requested_base_amount'))}",
            "pnl_scope: position notional only; excludes whole-wallet equity, "
            "other positions, and borrow interest",
            f"net_pnl_quote_est_position: {_format_float(net_pnl_amount)}",
            f"net_pnl_pct_position_notional: {_format_pct(net_pnl_pct)}",
            "net_pnl_pct_wallet_leverage_adjusted: "
            f"{_format_pct(closed_trade.get('leverage_adjusted_net_pnl_pct'))}",
            f"gross_pnl_quote_est_position: {_format_float(closed_trade.get('gross_pnl'))}",
            f"gross_pnl_pct_position_notional: {_format_pct(gross_pnl_pct)}",
            "gross_pnl_pct_wallet_leverage_adjusted: "
            f"{_format_pct(closed_trade.get('leverage_adjusted_gross_pnl_pct'))}",
            "gross_to_net_cost_quote: "
            f"{_format_float(closed_trade.get('gross_to_net_cost_quote'))}",
            "gross_to_net_cost_pct_position_notional: "
            f"{_format_pct(closed_trade.get('gross_to_net_cost_pct'))}",
            f"entry_fee_quote: {_format_float(closed_trade.get('entry_fee_quote'))}",
            f"exit_fee_quote: {_format_float(closed_trade.get('exit_fee_quote'))}",
            f"entry_fee_source: {closed_trade.get('entry_fee_source')}",
            f"exit_fee_source: {closed_trade.get('exit_fee_source')}",
            f"fee_source: {closed_trade.get('fee_source')}",
            f"fees_verified: {closed_trade.get('fees_verified')}",
            f"mfe: {_format_pct(closed_trade.get('mfe'))}",
            f"mae: {_format_pct(closed_trade.get('mae'))}",
            "requested_policy_stop: "
            f"{_format_float(closed_trade.get('requested_policy_stop'))}",
            "final_placed_stop: "
            f"{_format_float(closed_trade.get('final_placed_stop'))}",
            "exit_vs_policy_stop_bps: "
            f"{_format_float(closed_trade.get('exit_vs_policy_stop_bps'), digits=4)}",
            "exit_vs_peak_giveback_pct: "
            f"{_format_pct(closed_trade.get('exit_vs_peak_giveback_pct'))}",
            f"policy_parity_ok: {closed_trade.get('policy_parity_ok')}",
            f"stop_price: {closed_trade.get('stop_price')}",
            f"stop_trigger_signal: {closed_trade.get('stop_trigger_signal')}",
            "stop_trigger_reference_source: "
            f"{closed_trade.get('stop_trigger_reference_source')}",
            f"decision_module: {closed_trade.get('decision_module')}",
            "stop_policy_params_source: "
            f"{closed_trade.get('stop_policy_params_source')}",
            "stop_policy_params_hash: "
            f"{closed_trade.get('stop_policy_params_hash')}",
            f"stop_policy_schema: {closed_trade.get('stop_policy_schema')}",
            f"stop_order_id: {closed_trade.get('stop_order_id')}",
            f"close_order_id: {closed_trade.get('close_order_id')}",
            f"close_order_status: {closed_trade.get('close_order_status')}",
            f"close_order_type: {closed_trade.get('close_order_type')}",
            f"close_order_cost: {closed_trade.get('close_order_cost')}",
            f"fee_cost: {closed_trade.get('fee_cost')}",
            f"fee_currency: {closed_trade.get('fee_currency')}",
            "",
            "Trade recap:",
            trade_recap or "  no stop/price recap events were recorded",
        ]
    )
    result = DailyDeploymentReporter()._send_email(
        subject=subject,
        body=body,
        recipient=recipient,
        config=config,
    )
    if result.get("success"):
        tprint(f"[TradeCloseEmail] Sent close email for {symbol} to {recipient}")
        return {"sent": True, "email_result": result}
    tprint(
        "[TradeCloseEmail] Close email failed: "
        f"{result.get('error_category')}: {result.get('error')}"
    )
    return {"sent": False, "reason": "email_failed", "email_result": result}


def _send_trade_open_email(
    *,
    symbol: str,
    side: str,
    strategy_id: str,
    size: float,
    decision: Dict[str, Any],
    trade_result: Dict[str, Any],
    predictions: Dict[str, Any],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Send a per-trade open notification using the daily reporter SMTP config."""
    if not bool(config.get("trade_open_email_enabled", False)):
        return {"sent": False, "reason": "disabled"}
    recipient = str(
        config.get("trade_open_email_to")
        or config.get("trade_close_email_to")
        or config.get("daily_report_email_to")
        or os.environ.get("EPM_TRADE_EMAIL_TO")
        or os.environ.get("EPM_REPORT_EMAIL_TO")
        or ""
    ).strip()
    if not recipient:
        return {"sent": False, "reason": "missing_recipient"}

    subject = f"EPM trade opened: {symbol} {side}"
    body = "\n".join(
        [
            "Extreme price movement trade open notification",
            "",
            f"market_mode: {config.get('market_mode', 'spot')}",
            f"symbol: {symbol}",
            f"side: {side}",
            f"strategy_id: {strategy_id}",
            f"opened_at_utc: {pd.Timestamp.now(tz='UTC')}",
            f"entry_order_type: {trade_result.get('entry_order_type')}",
            f"exchange_order_id: {_order_identifier(trade_result.get('order'))}",
            f"quote_size: {_format_float(abs(size), digits=4)}",
            f"base_amount: {_format_float(trade_result.get('base_amount'))}",
            f"expected_entry_price: {_format_float(trade_result.get('expected_entry_price'))}",
            f"realized_entry_price: {_format_float(trade_result.get('realized_entry_price'))}",
            f"ohlcv_entry_price: {_format_float(trade_result.get('ohlcv_entry_price'))}",
            "entry_price_delta_vs_ohlcv: "
            f"{_format_float(trade_result.get('entry_price_delta_vs_ohlcv'))}",
            "entry_price_delta_vs_ohlcv_pct: "
            f"{_format_pct(trade_result.get('entry_price_delta_vs_ohlcv_pct'))}",
            f"signal_price: {_format_float(trade_result.get('signal_price'))}",
            f"decision_mid: {_format_float(trade_result.get('decision_mid'))}",
            f"signal_gap_bps: {_format_float(trade_result.get('signal_gap_bps'), digits=4)}",
            f"ticker_bid: {_format_float(trade_result.get('ticker_bid'))}",
            f"ticker_ask: {_format_float(trade_result.get('ticker_ask'))}",
            f"ticker_mid: {_format_float(trade_result.get('ticker_mid'))}",
            f"ticker_spread_bps: {_format_float(trade_result.get('ticker_spread_bps'), digits=4)}",
            "expected_fill_price: "
            f"{_format_float(trade_result.get('expected_fill_price'))}",
            "expected_fill_slippage_bps: "
            f"{_format_float(trade_result.get('expected_fill_slippage_bps'), digits=4)}",
            "expected_total_entry_friction_bps: "
            f"{_format_float(trade_result.get('expected_total_entry_friction_bps'), digits=4)}",
            "orderbook_capacity_quote_within_slippage: "
            f"{_format_float(trade_result.get('orderbook_capacity_quote_within_slippage'), digits=4)}",
            "liquidity_capacity_weight: "
            f"{_format_float(trade_result.get('liquidity_capacity_weight'), digits=4)}",
            f"perp_effective_leverage: {_format_float(trade_result.get('perp_effective_leverage'), digits=4)}",
            f"perp_rank_leverage: {_format_float(trade_result.get('perp_rank_leverage'), digits=4)}",
            f"perp_risk_cap_leverage: {_format_float(trade_result.get('perp_risk_cap_leverage'), digits=4)}",
            f"perp_rank_number: {trade_result.get('perp_rank_number')}",
            f"perp_rank_x: {trade_result.get('perp_rank_x')}",
            f"perp_stop_loss_pct: {_format_float(trade_result.get('perp_stop_loss_pct'), digits=4)}",
            f"perp_full_wallet: {_format_float(trade_result.get('perp_full_wallet'), digits=4)}",
            f"perp_available_wallet: {_format_float(trade_result.get('perp_available_wallet'), digits=4)}",
            f"stop_price: {_format_float(trade_result.get('stop_price'))}",
            f"stop_order_id: {trade_result.get('stop_order_id')}",
            f"stop_trigger_signal: {trade_result.get('stop_trigger_signal')}",
            "stop_trigger_reference_source: "
            f"{trade_result.get('stop_trigger_reference_source')}",
            f"base_pred: {_format_float(predictions.get('base_pred'), digits=6)}",
            f"meta_pred: {_format_float(predictions.get('meta_pred'), digits=6)}",
            f"calibrated_score: {_format_float(decision.get('calibrated_score'), digits=6)}",
            f"rank_score_source: {decision.get('rank_score_source')}",
            f"policy_rank_pct: {_format_float(decision.get('policy_rank_pct'), digits=6)}",
            f"policy_rank_reference_n: {decision.get('policy_rank_reference_n')}",
            "policy_rank_reference_source: "
            f"{decision.get('policy_rank_reference_source')}",
            f"rank_percentile: {_format_float(decision.get('rank_percentile'), digits=6)}",
            f"rank_threshold: {_format_float(decision.get('rank_threshold'), digits=6)}",
            "deployment_rank_threshold: "
            f"{_format_float(decision.get('deployment_rank_threshold'), digits=6)}",
            "pnl_scope_for_future_close_email: position notional only; excludes "
            "whole-wallet equity, other positions, and borrow interest",
        ]
    )
    result = DailyDeploymentReporter()._send_email(
        subject=subject,
        body=body,
        recipient=recipient,
        config=config,
    )
    if result.get("success"):
        tprint(f"[TradeOpenEmail] Sent open email for {symbol} to {recipient}")
        return {"sent": True, "email_result": result}
    tprint(
        "[TradeOpenEmail] Open email failed: "
        f"{result.get('error_category')}: {result.get('error')}"
    )
    return {"sent": False, "reason": "email_failed", "email_result": result}


def _write_margin_reconciliation_report(
    config: Dict[str, Any],
    report: Dict[str, Any],
) -> None:
    """Persist the startup cross-margin reconciliation report for auditability."""
    try:
        data_root = Path(
            str(config.get("live_data_root") or config.get("data_root", "data"))
        )
        run_id = str(config.get("run_id", "latest"))
        out_dir = data_root / "live_state" / "reconciliation" / run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")
        payload = json.dumps(report, indent=2, sort_keys=True)
        for path in (
            out_dir / "cross_margin_reconciliation_latest.json",
            out_dir / f"cross_margin_reconciliation_{stamp}.json",
        ):
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_text(payload, encoding="utf-8")
            os.replace(tmp, path)
        tprint(
            "Saved cross-margin reconciliation report: "
            f"{out_dir / 'cross_margin_reconciliation_latest.json'}"
        )
    except Exception as exc:
        tprint(f"Warning: could not save cross-margin reconciliation report: {exc}")


def _safe_exchange_data_component(value: Any) -> str:
    raw = str(value or "unknown").strip().lower()
    safe = re.sub(r"[^a-z0-9_.=-]+", "_", raw)
    safe = safe.strip("._")
    return safe or "unknown"


def _resolve_live_data_root(
    *,
    artifact_data_root: str,
    exchange: Any,
    market_mode: str,
    explicit_live_data_root: Optional[str] = None,
) -> str:
    """Return the exchange-scoped root for live market cache and runtime state."""
    explicit = explicit_live_data_root or os.environ.get("EPM_LIVE_DATA_ROOT")
    if explicit:
        return str(Path(explicit))
    exchange_id = (
        getattr(exchange, "id", None)
        or os.environ.get("EPM_EXCHANGE")
        or ("perps" if market_mode == "perps" else "spot")
    )
    exchange_key = _safe_exchange_data_component(exchange_id)
    return str(Path(artifact_data_root) / "exchanges" / exchange_key)


def _resolve_prediction_ledger_path(
    *,
    live_data_root: str | Path,
    run_id: str,
    explicit_path: Optional[str | Path] = None,
    run_scoped: bool = False,
) -> Path:
    """Resolve the prediction ledger path, optionally namespaced by artifact run."""
    explicit = explicit_path or os.environ.get("EPM_PREDICTION_LEDGER_PATH")
    if explicit:
        return Path(explicit)
    state_dir = Path(live_data_root) / "live_state"
    if bool(run_scoped) or _env_flag("EPM_RUN_SCOPED_PREDICTION_LEDGER", True):
        return state_dir / "prediction_ledgers" / str(run_id) / "prediction_ledger.parquet"
    return state_dir / "prediction_ledger.parquet"


def _sync_reconciled_positions_to_portfolio_manager(
    executor: TradeExecutor,
    portfolio_mgr: PortfolioManager,
) -> None:
    """Mirror executor positions into portfolio risk state after exchange reconciliation."""
    try:
        active_positions = executor.get_active_positions()
    except Exception as exc:
        tprint(f"[PortfolioManager] Startup reconcile sync skipped: {exc}")
        return
    active_symbols = {str(symbol) for symbol in active_positions.keys()}
    stale_symbols = [
        str(symbol)
        for symbol, position in list(getattr(portfolio_mgr, "positions", {}).items())
        if getattr(position, "is_open", False) and str(symbol) not in active_symbols
    ]
    for symbol in stale_symbols:
        try:
            del portfolio_mgr.positions[symbol]
            tprint(
                f"[PortfolioManager] Removed stale local position absent from executor: {symbol}"
            )
        except Exception as exc:
            tprint(
                f"[PortfolioManager] Failed to remove stale local position {symbol}: {exc}"
            )
    for symbol, state in active_positions.items():
        if not isinstance(state, dict):
            continue
        if not bool(state.get("external_position")) or symbol in portfolio_mgr.positions:
            continue
        try:
            portfolio_mgr.record_position_open(
                symbol=str(symbol),
                side=str(state.get("side") or "long"),
                strategy_id=str(
                    state.get("strategy_id")
                    or state.get("bucket_key")
                    or "external_margin_reconciliation"
                ),
                position_size=float(
                    state.get("quote_size")
                    or abs(float(state.get("size", 0.0) or 0.0))
                    * float(state.get("entry_price", 0.0) or 0.0)
                ),
                entry_price=float(state.get("entry_price")),
                entry_time=pd.Timestamp(
                    state.get("entry_time") or pd.Timestamp.now(tz="UTC")
                ),
            )
        except Exception as exc:
            tprint(
                f"[PortfolioManager] Failed to sync reconciled position {symbol}: {exc}"
            )


def _apply_reconciliation_entry_gate(
    *,
    reconciliation_report: dict[str, Any],
    portfolio_mgr: PortfolioManager,
) -> list[dict[str, Any]]:
    """Block new entries if real external margin positions are not imported."""
    unimported_external_positions = [
        item
        for item in reconciliation_report.get("items", [])
        if str(item.get("classification", "")).startswith("external_")
        and not bool(item.get("imported_for_monitoring"))
    ]
    if not unimported_external_positions:
        return []
    symbols_blocking = sorted(
        {
            str(item.get("symbol") or item.get("asset") or "unknown")
            for item in unimported_external_positions
        }
    )
    portfolio_mgr.trip_hard_limit(
        "external_margin_reconciliation_incomplete: "
        f"{len(unimported_external_positions)} external position record(s) "
        f"not imported for monitoring ({','.join(symbols_blocking)})"
    )
    tprint(
        "New entries blocked because cross-margin reconciliation is incomplete; "
        "existing imported positions will continue to be monitored. "
        f"Unimported external positions: {symbols_blocking}"
    )
    return unimported_external_positions


def _margin_account_metrics_from_reconciliation(
    reconciliation_report: dict[str, Any],
) -> dict[str, float]:
    """Extract gross margin assets/liabilities for sizing-cap decisions."""

    total_assets_quote = 0.0
    total_liabilities_quote = 0.0
    for item in reconciliation_report.get("items", []):
        if not isinstance(item, dict):
            continue
        kind = str(item.get("kind") or "")
        if kind == "net_exposure":
            balance_value = item.get("gross_balance_quote_value")
            debt_value = item.get("gross_debt_quote_value")
            if balance_value is not None and np.isfinite(float(balance_value)):
                total_assets_quote += max(float(balance_value), 0.0)
            if debt_value is not None and np.isfinite(float(debt_value)):
                total_liabilities_quote += max(float(debt_value), 0.0)
            continue
        quote_value = item.get("quote_value")
        if quote_value is None or not np.isfinite(float(quote_value)):
            continue
        if kind == "balance":
            total_assets_quote += max(float(quote_value), 0.0)
        elif kind == "debt":
            total_liabilities_quote += max(float(quote_value), 0.0)

    equity_quote = max(total_assets_quote - total_liabilities_quote, 0.0)
    margin_level = (
        total_assets_quote / max(total_liabilities_quote, 1e-12)
        if total_liabilities_quote > 0.0
        else float("inf")
    )
    return {
        "total_assets_quote": total_assets_quote,
        "total_liabilities_quote": total_liabilities_quote,
        "equity_quote": equity_quote,
        "margin_level": margin_level,
    }


def _apply_margin_metrics_to_portfolio_manager(
    *,
    reconciliation_report: dict[str, Any],
    portfolio_mgr: PortfolioManager,
    exchange: Optional[Any] = None,
    config: Optional[dict[str, Any]] = None,
) -> dict[str, float]:
    """Sync margin-account gross assets/debt into PortfolioManager sizing state."""

    runtime_config = config or {}
    execution_account = str(runtime_config.get("execution_account") or "").lower()
    market_mode = _normalise_market_mode(runtime_config.get("market_mode"))
    is_perps = (
        execution_account in {"perp", "perps", "future", "futures", "swap"}
        or market_mode == "perps"
    )
    if is_perps and exchange is not None:
        quote_currency = str(runtime_config.get("live_quote_currency") or "USD").upper()
        margin_mode = str(runtime_config.get("margin_mode") or "cross").lower()
        snapshot = portfolio_mgr.fetch_exchange_snapshot(
            exchange,
            quote_currency=quote_currency,
            execution_account=execution_account or "perps",
            margin_mode=margin_mode,
        )
        total_assets = float(portfolio_mgr.margin_total_assets_quote or 0.0)
        total_liabilities = float(portfolio_mgr.margin_total_liabilities_quote or 0.0)
        equity_quote = max(total_assets - total_liabilities, 0.0)
        margin_level = (
            total_assets / max(total_liabilities, 1e-12)
            if total_liabilities > 0.0
            else float("inf")
        )
        metrics = {
            "total_assets_quote": total_assets,
            "total_liabilities_quote": total_liabilities,
            "equity_quote": equity_quote,
            "margin_level": margin_level,
        }
        if snapshot.get("errors"):
            tprint(
                "Perps wallet snapshot had errors while syncing sizing metrics: "
                f"{snapshot.get('errors')}"
            )
    else:
        metrics = _margin_account_metrics_from_reconciliation(reconciliation_report)
        portfolio_mgr.update_margin_account_metrics(
            total_assets_quote=metrics["total_assets_quote"],
            total_liabilities_quote=metrics["total_liabilities_quote"],
        )
    target_margin_level = float(portfolio_mgr.min_margin_level_after_entry)
    safe_surplus = max(
        metrics["total_assets_quote"]
        - target_margin_level * metrics["total_liabilities_quote"],
        0.0,
    )
    tprint(
        "Margin sizing metrics: "
        f"assets={metrics['total_assets_quote']:.4f} "
        f"liabilities={metrics['total_liabilities_quote']:.4f} "
        f"equity={metrics['equity_quote']:.4f} "
        f"margin_level={metrics['margin_level']:.4f} "
        f"safe_surplus@{target_margin_level:.2f}={safe_surplus:.4f}"
    )
    return metrics


def _maybe_send_daily_deployment_report(
    *,
    daily_reporter: DailyDeploymentReporter,
    exchange: Any,
    portfolio_mgr: PortfolioManager,
    trade_logger: TradeLogger,
    config: Dict[str, Any],
    force: bool = False,
) -> Dict[str, Any]:
    """Run the daily recap when due and keep failure modes visible."""
    try:
        result = daily_reporter.maybe_run(
            exchange=exchange,
            portfolio_mgr=portfolio_mgr,
            trade_logger=trade_logger,
            config=config,
            force=force,
        )
    except Exception as exc:
        tprint(f"Daily deployment report failed: {exc}")
        return {"sent": False, "reason": "exception", "error": str(exc)}

    if result.get("sent"):
        return result
    reason = str(result.get("reason") or "")
    if reason and reason != "not_due":
        tprint(f"Daily deployment report skipped: {reason}")
    return result


def _maybe_export_google_sheets(
    *,
    sheets_exporter: Optional[GoogleSheetsTradeExporter],
    trade_logger: TradeLogger,
    executor: TradeExecutor,
    force: bool = False,
) -> bool:
    """Export local trade diagnostics to Google Sheets when configured."""
    if sheets_exporter is None:
        return False
    try:
        active_positions = (
            executor.get_active_positions()
            if hasattr(executor, "get_active_positions")
            else {}
        )
        return sheets_exporter.export_trade_logger(
            trade_logger,
            active_positions=active_positions,
            force=force,
        )
    except Exception as exc:
        tprint(f"Google Sheets export wrapper failed: {type(exc).__name__}: {exc}")
        return False


def _is_symbol_blocked_for_strategy(
    symbol: str,
    strategy_id: str,
    strategy_asset_exclusions: Optional[Dict[str, set[str]]],
) -> bool:
    """Return True when policy optimiser excludes a symbol for this strategy."""
    if not strategy_asset_exclusions:
        return False
    symbol_norm = normalise_symbol(symbol)
    symbol_base_norm = symbol_base(symbol_norm)
    sid = str(strategy_id or "")
    core = strategy_core_id(sid)
    aliases = {sid, core}
    side = sid.split("_", 1)[0].lower() if "_" in sid else ""
    if side in {"long", "short"} and core:
        aliases.add(f"{side}_{core}")
    for alias in aliases:
        blocked = strategy_asset_exclusions.get(alias)
        if not blocked:
            continue
        blocked_norm = {normalise_symbol(str(sym)) for sym in blocked}
        blocked_bases = {symbol_base(sym) for sym in blocked_norm}
        if symbol_norm in blocked_norm or symbol_base_norm in blocked_bases:
            return True
    return False


def _record_trade_execution_health(
    portfolio_mgr: Optional[PortfolioManager],
    trade_result: Dict[str, Any],
) -> None:
    """Feed exchange execution failures into portfolio hard-gate counters."""
    if portfolio_mgr is None:
        return
    success = bool(
        trade_result.get("success", False) or trade_result.get("status") == "recorded"
    )
    if success:
        portfolio_mgr.record_order_result(True)
        return
    error_text = str(trade_result.get("error", "") or "")
    category = str(trade_result.get("error_category", "") or "")
    symbol_scoped_rejection_categories = {
        "asset_collateral_limit",
        "borrow_limit",
        "symbol_halted",
        "unsupported_liability_target",
    }
    if category in symbol_scoped_rejection_categories:
        tprint(
            "Order failure is symbol-scoped; not activating portfolio-wide "
            f"order-rejection backoff: category={category} "
            f"symbol={trade_result.get('symbol', '')}"
        )
        return
    rejection_categories = {
        "insufficient_balance",
        "invalid_precision_or_filter",
        "order_rejected",
        "duplicate_client_order_id",
        "cancel_failed",
        "stop_loss_failed",
    }
    api_failure_categories = {
        "network_timeout",
        "rate_limited",
        "auth_or_permission",
        "exchange_error",
    }
    is_rejected = category in rejection_categories or (
        "reject" in error_text.lower()
        or str(trade_result.get("status", "")).lower() == "rejected"
    )
    portfolio_mgr.record_order_result(
        False,
        rejected=is_rejected,
        error=f"{category}: {error_text}" if category else error_text,
    )
    if category in api_failure_categories:
        portfolio_mgr.record_api_call(
            False,
            error=f"order execution failed: {category}: {error_text}",
        )


def _is_expected_order_capacity_rejection(category: str) -> bool:
    """Return True for exchange-side capacity limits that are not code failures."""
    return str(category or "") in {
        "asset_collateral_limit",
        "borrow_limit",
    }


def _exchange_min_notional_for_symbol(exchange: Any, symbol: str) -> Optional[float]:
    """Return the exchange's min quote notional for a symbol when available."""
    if exchange is None:
        return None
    try:
        if hasattr(exchange, "load_markets") and not getattr(exchange, "markets", None):
            exchange.load_markets()
    except Exception as exc:
        tprint(
            f"Warning: could not load markets for min-notional check {symbol}: {exc}"
        )
    market: Dict[str, Any] = {}
    try:
        if hasattr(exchange, "market"):
            maybe_market = exchange.market(symbol)
            if isinstance(maybe_market, dict):
                market = maybe_market
    except Exception:
        market = {}
    if not market:
        markets = getattr(exchange, "markets", None)
        if isinstance(markets, dict) and isinstance(markets.get(symbol), dict):
            market = markets[symbol]
    limits = market.get("limits", {}) if isinstance(market, dict) else {}
    cost_limits = limits.get("cost", {}) if isinstance(limits, dict) else {}
    min_notional = _safe_float(cost_limits.get("min"), default=np.nan)
    return (
        float(min_notional) if np.isfinite(min_notional) and min_notional > 0 else None
    )


def _execute_trade_with_optional_context(
    executor: TradeExecutor,
    *,
    symbol: str,
    side: str,
    size: float,
    price: Optional[float],
    bucket_key: str,
    ohlcv_reference_price: Optional[float],
    trade_context: Optional[Dict[str, Any]] = None,
    execution_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Call modern executors with context, while supporting older test doubles."""
    execution_kwargs = execution_kwargs or {}
    try:
        return executor.execute_trade(
            symbol=symbol,
            side=side,
            size=size,
            price=price,
            bucket_key=bucket_key,
            ohlcv_reference_price=ohlcv_reference_price,
            trade_context=trade_context,
            **execution_kwargs,
        )
    except TypeError as exc:
        if "ohlcv_reference_price" not in str(exc) and "trade_context" not in str(exc):
            raise
        return executor.execute_trade(
            symbol=symbol,
            side=side,
            size=size,
            price=price,
            bucket_key=bucket_key,
        )


def _resolve_live_barrier_pct(
    symbol: str,
    features_log: Dict[str, Any],
    *,
    panel: Optional[Dict[str, pd.DataFrame]] = None,
    cfg: Optional[Dict[str, Any]] = None,
) -> Optional[float]:
    """Return the optimiser barrier fraction for live stop placement.

    This intentionally accepts only explicit policy barrier fields. Raw ATR%
    is materialized as ``barrier_pct`` in the inference feature generator; the
    transformed model features ``atr_pct``/``atr_pct_base`` must never be used
    as fallbacks for execution stops.
    """
    for key in ("barrier_pct", "barrier_frac"):
        value = features_log.get(key)
        try:
            barrier = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(barrier) and barrier > 0.0:
            return barrier
    if isinstance(panel, dict) and bool((cfg or {}).get("allow_raw_policy_barrier_recompute", False)):
        try:
            raw_barrier = _compute_policy_barrier_pct(panel, [symbol], cfg or {})
        except Exception as exc:
            tprint(f"Live barrier raw recompute failed for {symbol}: {exc}")
            raw_barrier = None
        if isinstance(raw_barrier, pd.DataFrame) and symbol in raw_barrier.columns:
            vals = raw_barrier[symbol].dropna()
            if not vals.empty:
                barrier = float(vals.iloc[-1])
                if np.isfinite(barrier) and barrier > 0.0:
                    features_log["barrier_pct"] = barrier
                    tprint(
                        f"Live barrier recomputed from raw OHLCV for {symbol}: "
                        f"barrier_pct={barrier:.6g}"
                    )
                    return barrier
    tprint(
        f"Live barrier unavailable for {symbol}: missing explicit raw barrier_pct; "
        "entry will be blocked rather than using transformed ATR fallbacks"
    )
    return None


def _sleep_until_hourly_ohlcv_window(
    now: pd.Timestamp,
    *,
    delay_seconds: float = 5.0,
) -> None:
    """Wait a few seconds after the hour so Binance has published the new kline."""
    current_hour = pd.Timestamp(now).floor("h")
    seconds_into_hour = (pd.Timestamp.now(tz="UTC") - current_hour).total_seconds()
    remaining = float(delay_seconds) - float(seconds_into_hour)
    if remaining > 0:
        tprint(f"Waiting {remaining:.1f}s for hourly OHLCV publication window")
        time.sleep(remaining)


def _latest_closed_candle_start(
    now: Optional[pd.Timestamp],
    *,
    timeframe_minutes: int,
    delay_seconds: float = 5.0,
) -> pd.Timestamp:
    """Return the start timestamp of the latest candle closed past delay."""
    now_ts = pd.Timestamp(now if now is not None else pd.Timestamp.now(tz="UTC"))
    if now_ts.tzinfo is None:
        now_ts = now_ts.tz_localize("UTC")
    else:
        now_ts = now_ts.tz_convert("UTC")
    freq = pd.Timedelta(minutes=int(timeframe_minutes))
    boundary = now_ts.floor(f"{int(timeframe_minutes)}min")
    if now_ts < boundary + pd.Timedelta(seconds=float(delay_seconds)):
        boundary -= freq
    return boundary - freq


def _closed_candle_age_seconds(
    now: Optional[pd.Timestamp],
    candle_start: pd.Timestamp,
    *,
    timeframe_minutes: int,
) -> float:
    """Age in seconds since the candle close time, not since candle start."""
    now_ts = pd.Timestamp(now if now is not None else pd.Timestamp.now(tz="UTC"))
    if now_ts.tzinfo is None:
        now_ts = now_ts.tz_localize("UTC")
    else:
        now_ts = now_ts.tz_convert("UTC")
    start_ts = pd.Timestamp(candle_start)
    if start_ts.tzinfo is None:
        start_ts = start_ts.tz_localize("UTC")
    else:
        start_ts = start_ts.tz_convert("UTC")
    close_ts = start_ts + pd.Timedelta(minutes=int(timeframe_minutes))
    return max(0.0, float((now_ts - close_ts).total_seconds()))


def _sleep_until_next_candle_close(
    *,
    timeframe_minutes: int = 15,
    delay_seconds: float = 5.0,
) -> None:
    """Sleep until the next candle close plus exchange publication delay."""
    now_ts = pd.Timestamp.now(tz="UTC")
    target = _next_candle_wake_target(
        now_ts,
        timeframe_minutes=timeframe_minutes,
        delay_seconds=delay_seconds,
    )
    while True:
        now_ts = pd.Timestamp.now(tz="UTC")
        sleep_seconds = (target - now_ts).total_seconds()
        if sleep_seconds <= 0:
            break
        time.sleep(float(min(sleep_seconds, 5.0)))


def _next_candle_wake_target(
    now_ts: pd.Timestamp,
    *,
    timeframe_minutes: int,
    delay_seconds: float,
) -> pd.Timestamp:
    """Return the next candle-close wake target for a timeframe."""
    now_ts = pd.Timestamp(now_ts)
    if now_ts.tzinfo is None:
        now_ts = now_ts.tz_localize("UTC")
    else:
        now_ts = now_ts.tz_convert("UTC")
    freq = pd.Timedelta(minutes=int(timeframe_minutes))
    boundary = now_ts.floor(f"{int(timeframe_minutes)}min")
    # Add a small guard buffer and re-check after sleeping. macOS can wake a
    # process a fraction early, which can make the hourly loop miss the newly
    # closed candle and then skip it as stale on the next hourly wake.
    guard_seconds = 1.5
    target = boundary + pd.Timedelta(seconds=float(delay_seconds) + guard_seconds)
    if now_ts >= target:
        target = (
            boundary + freq + pd.Timedelta(seconds=float(delay_seconds) + guard_seconds)
        )
    return target


def _select_top_base_prediction_symbols(
    orchestrator: ModelOrchestrator,
    candidate_features: pd.DataFrame,
    candidates: List[str],
    side: str,
    strategy_id: str,
    *,
    top_frac: float = BASE_TO_META_TOP_FRAC,
) -> Dict[str, Dict[str, float]]:
    """Rank base-model predictions and keep only the top fraction for meta."""

    def _log_nonfinite_contract_features(
        matrix_values: np.ndarray,
        feat_cols: Sequence[str],
        row_index: Sequence[Any],
        *,
        prefix: str,
    ) -> None:
        finite_mask = np.isfinite(matrix_values)
        nonfinite_counts = (~finite_mask).sum(axis=0)
        top_idx = np.argsort(nonfinite_counts)[::-1]
        top_features = [
            {
                "feature": str(feat_cols[int(i)]),
                "rows": int(nonfinite_counts[int(i)]),
                "pct": round(float(nonfinite_counts[int(i)]) * 100.0 / max(1, len(row_index)), 2),
            }
            for i in top_idx[:15]
            if int(nonfinite_counts[int(i)]) > 0
        ]
        bad_rows = np.flatnonzero(~finite_mask.all(axis=1))[:10]
        sample_symbols = [str(row_index[int(i)]) for i in bad_rows]
        tprint(
            f"{prefix}: top_nonfinite_features={top_features} "
            f"sample_symbols={sample_symbols}"
        )

    if not hasattr(orchestrator, "predict_alpha"):
        tprint(
            f"Base prediction gate unavailable for {side}/{strategy_id}; "
            "failing closed."
        )
        return {}
    model_info = getattr(orchestrator, "alpha_by_strategy", {}).get(strategy_id)
    if model_info is None:
        model_info = getattr(orchestrator, "alpha_by_strategy", {}).get(
            f"{side}_{strategy_id}"
        )
    if isinstance(model_info, dict):
        feat_cols = _effective_alpha_feature_contract(model_info)
        if feat_cols:
            available_cols = sum(
                1 for col in feat_cols if col in candidate_features.columns
            )
            try:
                aligned = orchestrator._align_alpha_feature_contract(  # noqa: SLF001
                    candidate_features,
                    feat_cols,
                )
                if aligned.empty:
                    missing_cols = [
                        col for col in feat_cols if col not in candidate_features.columns
                    ]
                    tprint(
                        f"Base feature contract block for {side}/{strategy_core_id(strategy_id)}: "
                        "alpha alignment returned no strict rows "
                        f"available={available_cols}/{len(feat_cols)} "
                        f"missing_sample={missing_cols[:20]}"
                    )
                    return {}
                if len(aligned.index) != len(candidate_features.index):
                    dropped_symbols = [
                        str(sym)
                        for sym in candidate_features.index
                        if str(sym) not in set(aligned.index.astype(str))
                    ]
                    tprint(
                        f"Base feature contract block for {side}/{strategy_core_id(strategy_id)}: "
                        f"dropping {len(dropped_symbols)}/{len(candidate_features.index)} "
                        f"rows during strict alpha alignment sample={dropped_symbols[:10]}"
                    )
                    candidate_features = candidate_features.loc[aligned.index].copy()
                    candidates = [
                        str(symbol)
                        for symbol in candidates
                        if str(symbol) in set(aligned.index.astype(str))
                    ]
                nonzero_cols = int((aligned.abs().sum(axis=0) > 0.0).sum())
                varying_cols = int((aligned.nunique(dropna=False) > 1).sum())
                row_fingerprint = int(aligned.drop_duplicates().shape[0])
            except Exception as exc:
                nonzero_cols = -1
                varying_cols = -1
                row_fingerprint = -1
                tprint(
                    f"Base feature contract health failed for "
                    f"{side}/{strategy_core_id(strategy_id)}: {exc}"
                )
            tprint(
                f"Base feature contract health {side}/{strategy_core_id(strategy_id)}: "
                f"rows={len(candidate_features)} contract_cols={len(feat_cols)} "
                f"available={available_cols} nonzero={nonzero_cols} "
                f"varying={varying_cols} unique_rows={row_fingerprint}"
            )
    try:
        preds = orchestrator.predict_alpha(candidate_features, side, strategy_id)
    except Exception as exc:
        tprint(
            f"Base prediction gate failed for {side}/{strategy_id}; "
            f"failing closed: {exc}"
        )
        return {}
    if not isinstance(preds, pd.Series) or preds.empty:
        return {}

    ranked = preds.reindex(candidates).replace([np.inf, -np.inf], np.nan).dropna()
    if ranked.empty:
        return {}
    top_n = max(1, int(np.ceil(len(ranked) * float(top_frac))))
    winners = ranked.sort_values(ascending=False).head(top_n)
    ranks = ranked.rank(method="first", pct=True, ascending=True)
    min_kept_score = float(winners.min()) if len(winners) else float("nan")
    tprint(
        f"Base gate {side}/{strategy_core_id(strategy_id)}: "
        f"{len(winners)}/{len(ranked)} candidates kept for meta "
        f"(min_base_pred={min_kept_score:.6g})"
    )
    return {
        str(symbol): {
            "base_pred": float(score),
            "base_rank_pct": float(ranks.loc[symbol]),
            "base_gate_top_frac": float(top_frac),
            "base_gate_min_kept_score": min_kept_score,
        }
        for symbol, score in winners.items()
    }


def _json_safe(value: Any) -> Any:
    """Return a compact JSON-safe representation for structured live logs."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


def _emit_structured_event(event_name: str, payload: Dict[str, Any]) -> None:
    try:
        tprint(
            f"{event_name} "
            + json.dumps(_json_safe(payload), sort_keys=True, default=str)
        )
    except Exception as exc:
        tprint(f"{event_name} emit_failed: {exc}; payload={payload}")


def _score_distribution_payload(values: list[float]) -> Dict[str, Any]:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return {"n": 0, "finite": 0}
    q90 = float(np.quantile(arr, 0.90))
    q95 = float(np.quantile(arr, 0.95))
    q99 = float(np.quantile(arr, 0.99))
    return {
        "n": int(arr.size),
        "finite": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "range": float(np.max(arr) - np.min(arr)),
        "q90": q90,
        "q95": q95,
        "q99": q99,
        "top1pct_count": int(np.sum(arr >= q99)),
        "top5pct_count": int(np.sum(arr >= q95)),
        "top10pct_count": int(np.sum(arr >= q90)),
    }


def _log_score_distribution(label: str, values: list[float]) -> Dict[str, Any]:
    payload = _score_distribution_payload(values)
    if int(payload.get("n", 0) or 0) == 0:
        tprint(f"{label}: no finite values")
        return payload
    tprint(
        f"{label}: n={payload['n']}, mean={payload['mean']:.6f}, "
        f"std={payload['std']:.6f}, min={payload['min']:.6f}, "
        f"max={payload['max']:.6f}, range={payload['range']:.6f}, "
        f"top1%={payload['top1pct_count']}, "
        f"top5%={payload['top5pct_count']}, "
        f"top10%={payload['top10pct_count']}"
    )
    return payload


def _log_generated_feature_frames(feats: Dict[str, pd.DataFrame]) -> None:
    if not isinstance(feats, dict) or not feats:
        tprint("Inference features generated: none")
        return
    n_frames = len(feats)
    row_counts = []
    symbol_union = set()
    empty = []
    for name, frame in feats.items():
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            empty.append(str(name))
            continue
        row_counts.append(int(len(frame)))
        symbol_union.update(str(c) for c in frame.columns)
    tprint(
        "Inference features generated: "
        f"frames={n_frames}, non_empty={n_frames - len(empty)}, "
        f"symbols={len(symbol_union)}, "
        f"row_range=[{min(row_counts) if row_counts else 0},{max(row_counts) if row_counts else 0}]"
    )
    if empty:
        tprint(f"Inference empty feature frames sample: {empty[:12]}")


def _log_feature_coverage(candidate_features: pd.DataFrame, side: str) -> None:
    if candidate_features is None or candidate_features.empty:
        tprint(f"Inference feature coverage [{side}]: empty")
        return
    n_rows = int(len(candidate_features))
    non_null_frac = candidate_features.notna().mean(axis=0)
    finite_frac = np.isfinite(
        candidate_features.select_dtypes(include=[np.number])
    ).mean(axis=0)
    low_coverage = [
        str(col) for col, frac in non_null_frac.items() if float(frac) < 0.80
    ][:10]
    low_finite = [str(col) for col, frac in finite_frac.items() if float(frac) < 0.80][
        :10
    ]
    tprint(
        f"Inference feature coverage [{side}]: rows={n_rows}, cols={candidate_features.shape[1]}, "
        f"row_nonnull_mean={float(candidate_features.notna().mean(axis=1).mean()):.4f}, "
        f"col_nonnull_mean={float(non_null_frac.mean()):.4f}, numeric_col_finite_mean={float(finite_frac.mean()) if len(finite_frac) else float('nan'):.4f}"
    )
    if low_coverage:
        tprint(
            f"Inference feature coverage [{side}] low-nonnull cols(<80%): {low_coverage}"
        )
    if low_finite:
        tprint(
            f"Inference feature coverage [{side}] low-finite numeric cols(<80%): {low_finite}"
        )


def _close_series_for_base(
    panel: Dict[str, pd.DataFrame],
    base_asset: str,
) -> pd.Series:
    close = panel.get("close")
    if not isinstance(close, pd.DataFrame) or close.empty:
        return pd.Series(dtype=np.float32)
    base = str(base_asset).upper()
    for col in close.columns:
        try:
            if symbol_base(str(col)).upper() == base:
                return pd.to_numeric(close[col], errors="coerce").dropna()
        except Exception:
            continue
    return pd.Series(dtype=np.float32)


def _policy_params_for_strategy(
    orchestrator: ModelOrchestrator,
    strategy_id: str,
) -> Dict[str, Any]:
    bucket_params = getattr(orchestrator, "bucket_params", {}) or {}
    if not isinstance(bucket_params, dict):
        return {}
    sid = str(strategy_id or "")
    core = strategy_core_id(sid)
    aliases = [sid, core]
    side = sid.split("_", 1)[0] if "_" in sid else ""
    if side in {"long", "short"} and core:
        aliases.append(f"{side}_{core}")
    for alias in aliases:
        params = bucket_params.get(alias)
        if isinstance(params, dict):
            return params
    buckets = bucket_params.get("buckets", {})
    if isinstance(buckets, dict):
        for alias in aliases:
            params = buckets.get(alias)
            if isinstance(params, dict):
                return params
    return {}


def _asset_policy_row(policy_params: Dict[str, Any], symbol: str) -> Dict[str, Any]:
    raw_rows = policy_params.get("asset_metrics", [])
    if not isinstance(raw_rows, list):
        return {}
    symbol_norm = normalise_symbol(str(symbol))
    for row in raw_rows:
        if not isinstance(row, dict):
            continue
        row_symbol = normalise_symbol(str(row.get("symbol", "")))
        if row_symbol == symbol_norm:
            return row
    return {}


def _meta_policy_position_size(
    *,
    calibrated_score: float,
    threshold: float,
    policy_params: Dict[str, Any],
    symbol: str,
) -> Dict[str, Any]:
    size_power = float(policy_params.get("best_size_power", 1.0) or 1.0)
    min_size = float(policy_params.get("min_position_size", 0.05) or 0.05)
    max_size = float(policy_params.get("max_position_size", 0.15) or 0.15)
    threshold = float(np.clip(threshold, 0.0, 0.999999))
    calibrated_score = float(np.clip(calibrated_score, 0.0, 1.0))
    scaled_rank = (calibrated_score - threshold) / max(1.0 - threshold, 1e-12)
    scaled_rank = float(np.clip(scaled_rank, 0.0, 1.0))
    base_size = min_size + (max_size - min_size) * (scaled_rank**size_power)
    asset_row = _asset_policy_row(policy_params, symbol)
    asset_decision = str(asset_row.get("asset_decision", "keep") or "keep")
    asset_multiplier = 1.0
    if asset_row and asset_decision not in {"", "keep"}:
        tprint(
            "Symbol underperformance artifact found but disabled by current "
            f"portfolio policy: symbol={symbol} asset_decision={asset_decision}"
        )
    final_size = float(np.clip(base_size, 0.0, max_size))
    return {
        "position_size": final_size,
        "base_position_size": float(base_size),
        "sizing_source": "meta_calibrated_score_policy_power_no_symbol_penalty",
        "size_power": float(size_power),
        "sizing_rank_between_threshold_and_one": scaled_rank,
        "asset_weight_multiplier": float(asset_multiplier),
        "asset_decision": asset_decision,
        "min_position_size": float(min_size),
        "max_position_size": float(max_size),
    }


def _log_concurrent_positions_snapshot(
    portfolio_mgr: Optional[PortfolioManager],
    *,
    label: str,
) -> None:
    if portfolio_mgr is None:
        return
    try:
        state = portfolio_mgr.get_portfolio_state()
        now_hour = pd.Timestamp.now(tz="UTC").floor("h")
        tprint(
            "Concurrent positions hourly snapshot "
            f"[{label}] hour={now_hour.isoformat()} "
            f"open={state.get('n_positions')} max={state.get('max_positions')} "
            f"long={state.get('long_count')} short={state.get('short_count')} "
            f"invested_pct={float(state.get('invested_pct', np.nan)):.4f} "
            f"strategy_counts={state.get('strategy_counts', {})}"
        )
    except Exception as exc:
        tprint(f"Concurrent positions hourly snapshot [{label}] failed: {exc}")


def run_inference_step(
    orchestrator: ModelOrchestrator,
    panel: Dict[str, pd.DataFrame],
    feats: Dict[str, pd.DataFrame],
    thresholds: Dict[str, float],
    executor: TradeExecutor,
    logger: TradeLogger,
    max_candidates: int = 10,
    *,
    accepted_strategies: Optional[set[str]] = None,
    calibration_data: Optional[Dict[str, Dict[str, Any]]] = None,
    normalized_thresholds: Optional[Dict[str, Dict[str, Any]]] = None,
    portfolio_mgr: Optional[PortfolioManager] = None,
    initial_rank_threshold: float = 1.0,
    strategy_asset_exclusions: Optional[Dict[str, set[str]]] = None,
    preselected_long_candidates: Optional[List[str]] = None,
    preselected_short_candidates: Optional[List[str]] = None,
    strategy_candidate_masks: Optional[Dict[str, List[str]]] = None,
    max_entries_per_side: int = 3,
    max_entries_total: int = 4,
    portfolio_policy: Optional[PortfolioPolicyConfig] = None,
    prediction_ledger: Optional[PredictionLedger] = None,
    dynamic_performance_monitor: Optional[StrategyPerformanceMonitor] = None,
    strategy_kill_switch: Optional[StrategyKillSwitch] = None,
    policy_rank_reference_store: Optional[PolicyRankReferenceStore] = None,
    strategy_feature_contracts: Optional[Mapping[str, Sequence[str]]] = None,
    stale_entry_context: bool = False,
    stale_entry_max_abs_signal_gap_bps: float | None = None,
) -> Dict[str, Any]:
    """Run a single inference step.

    Args:
        orchestrator: ModelOrchestrator instance
        panel: Price panel
        feats: Feature dictionary
        thresholds: Candidate thresholds
        executor: TradeExecutor instance
        logger: TradeLogger instance
        max_candidates: Maximum candidates per side

    Returns:
        Results dictionary
    """
    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "long_candidates": [],
        "short_candidates": [],
        "trades": [],
        "side_metrics": {},
        "score_distributions": {},
        "order_error_summary": {
            "order_errors": 0,
            "unexplained_order_errors": 0,
            "by_side": {},
        },
    }
    now_utc = pd.Timestamp.now(tz="UTC")
    feature_source_max_ts = _max_feature_timestamp(feats)
    feature_available_ts = now_utc
    signal_bar_ts = feature_source_max_ts or now_utc
    timer = _StageTimer("run_inference_step")
    total_entries_executed = 0
    calibration_data = calibration_data or {}
    normalized_thresholds = normalized_thresholds or {}
    strategy_candidate_masks = strategy_candidate_masks or {}
    runtime_config = dict(getattr(executor, "config", {}) or {})
    meta_hit_rate_calibration = _load_meta_hit_rate_calibration(
        str(runtime_config.get("data_root", "data")),
        str(runtime_config.get("run_id", "latest")),
    )
    strategy_ev_calibration = _load_strategy_ev_calibration(
        str(runtime_config.get("data_root", "data")),
        str(runtime_config.get("run_id", "latest")),
    )
    if policy_rank_reference_store is None:
        policy_rank_reference_store = PolicyRankReferenceStore(
            data_root=str(runtime_config.get("data_root", "data")),
            run_id=str(
                runtime_config.get("policy_artifact_run_id")
                or runtime_config.get("run_id")
                or "latest"
            ),
        )
    allow_live_batch_rank_fallback_for_debug = bool(
        runtime_config.get("allow_live_batch_rank_fallback_for_debug", False)
    )
    executor_mode = str(getattr(executor, "mode", "") or "").lower()
    if executor_mode in {"live", "live-test", "live_test", "paper", "shadow_live"}:
        runtime_config.setdefault("allow_raw_policy_barrier_recompute", True)
    require_cross_strategy_auction_rank = bool(
        runtime_config.get(
            "require_cross_strategy_auction_rank",
            executor_mode in {"live", "live-test", "live_test", "paper", "shadow_live"},
        )
    )
    _validate_policy_rank_reference_startup(
        policy_rank_reference_store=policy_rank_reference_store,
        require_cross_strategy_auction_rank=require_cross_strategy_auction_rank,
    )
    adverse_hourly_close_gate_enabled, adverse_hourly_close_gate_bps = (
        _live_entry_adverse_hourly_close_gate(runtime_config)
    )
    raw_close_reference_gap_bps = _raw_close_reference_gap_bps(
        runtime_config,
        adverse_hourly_close_gate_bps,
    )
    max_signal_close_to_entry_seconds = _max_signal_close_to_entry_seconds(runtime_config)
    hard_stale_gate_default = executor_mode in {
        "live",
        "live-test",
        "live_test",
        "paper",
        "shadow_live",
    }
    hard_stale_gate_raw = os.environ.get(
        "EPM_HARD_STALE_SIGNAL_ENTRY_GATE_ENABLED",
        runtime_config.get(
            "hard_stale_signal_entry_gate_enabled",
            hard_stale_gate_default,
        ),
    )
    hard_stale_gate_enabled = (
        str(hard_stale_gate_raw).strip().lower() not in {"0", "false", "no", "off"}
    )
    if not hard_stale_gate_enabled:
        max_signal_close_to_entry_seconds = -1.0
    signal_to_entry_alert_seconds = _signal_to_entry_alert_seconds(runtime_config)
    min_base_train_rank_cfg = runtime_config.get("inference_min_base_train_rank_pct")
    try:
        inference_min_base_train_rank_pct = (
            None if min_base_train_rank_cfg is None else float(min_base_train_rank_cfg)
        )
    except (TypeError, ValueError):
        inference_min_base_train_rank_pct = None
    live_test_mode = _is_live_test_mode(executor)
    try:
        starting_active_position_count = len(executor.get_active_positions())
    except Exception:
        starting_active_position_count = 0
    live_feature_layer_debug = feature_layer_debug_enabled(
        runtime_config,
        live_test_mode=live_test_mode,
    )
    portfolio_policy = portfolio_policy or PortfolioPolicyConfig()
    parity_contract = runtime_config.get("training_live_parity_contract")
    if isinstance(parity_contract, dict) and parity_contract:
        accepted_strategies = _resolve_active_strategy_filter_for_policy(
            parity_contract=parity_contract,
            portfolio_policy=portfolio_policy,
            policy_strategy_filter=accepted_strategies,
            prefer_policy_contract=bool(
                runtime_config.get("policy_strategy_contract_overrides_parity", False)
            ),
        )
    validate_portfolio_strategy_contract(
        portfolio_policy,
        sorted(accepted_strategies) if accepted_strategies is not None else None,
        strict=True,
    )
    if dynamic_performance_monitor is not None:
        try:
            dynamic_performance_monitor.refresh(now=now_utc)
        except Exception as exc:
            tprint(f"Dynamic strategy performance refresh failed: {exc}")
    if isinstance(parity_contract, dict) and parity_contract:
        validate_training_live_parity_contract(
            parity_contract,
            active_strategy_ids=(
                sorted(accepted_strategies)
                if accepted_strategies is not None
                else []
            ),
            data_root=str(runtime_config.get("data_root") or ""),
            run_id=str(
                runtime_config.get("model_artifact_run_id")
                or runtime_config.get("run_id")
                or ""
            ),
            strict=not bool(
                runtime_config.get("policy_strategy_contract_overrides_parity", False)
            ),
        )
    if portfolio_mgr is None:
        portfolio_mgr = PortfolioManager.from_policy_config(
            portfolio_policy,
            cooldown_hours=0.0,
        )
    else:
        portfolio_mgr.book_notional_multiplier = (
            portfolio_policy.book_notional_multiplier
        )
        portfolio_mgr.leverage_wallet_multiplier = (
            portfolio_policy.leverage_wallet_multiplier
        )
        portfolio_mgr.min_margin_level_after_entry = (
            portfolio_policy.min_margin_level_after_entry
        )
        portfolio_mgr.occupancy_threshold_alpha = (
            portfolio_policy.occupancy_threshold_alpha
        )
        portfolio_mgr.occupancy_threshold_power = (
            portfolio_policy.occupancy_threshold_power
        )
        portfolio_mgr.threshold_viability_margin = (
            portfolio_policy.threshold_viability_margin
        )
    prediction_ledger_rows: List[Dict[str, Any]] = []
    _log_generated_feature_frames(feats)
    _log_concurrent_positions_snapshot(portfolio_mgr, label="start")
    timer.mark("startup_logging")
    if live_test_mode:
        tprint(
            "LIVE-TEST mode active: production decision path with quote clamp="
            f"[{portfolio_policy.live_test_min_quote_notional:.2f}, "
            f"{portfolio_policy.live_test_quote_notional:.2f}] USDC notional "
            f"multiplied by book_notional_multiplier="
            f"{portfolio_policy.book_notional_multiplier:.2f}. "
            "Rank thresholds are loaded from the deployment policy artifacts."
        )
    if stale_entry_context:
        tprint(
            "Conditional stale-entry mode active: candidates may enter only after "
            "ticker precheck confirms absolute current-mid vs signal-price move "
            f"<= {float(stale_entry_max_abs_signal_gap_bps or 0.0):.2f} bps."
        )
    if adverse_hourly_close_gate_enabled:
        tprint(
            "Adverse hourly-close entry gate active: candidates are rejected when "
            "current mid has moved against the hourly close by "
            f">= {adverse_hourly_close_gate_bps:.2f} bps."
        )
    tprint(
        "Raw signal-close reliability gate active: candidates are rejected when "
        "the raw close used by features is missing, zero-volume, or diverges from "
        f"mark/index by >= {raw_close_reference_gap_bps:.2f} bps."
    )
    if max_signal_close_to_entry_seconds >= 0.0:
        signal_close_ts = _signal_bar_close_ts(signal_bar_ts)
        tprint(
            "Hard stale-signal entry gate active: new entries are rejected when "
            f"decision_ts - signal_bar_close_ts > "
            f"{max_signal_close_to_entry_seconds:.0f}s "
            f"(signal_bar_ts={signal_bar_ts}, signal_bar_close_ts={signal_close_ts})."
        )
    pre_score_market_mask_enabled = bool(
        max_entries_total > 0
        and _live_prescore_market_mask_enabled(runtime_config, executor_mode)
    )
    if pre_score_market_mask_enabled:
        tprint(
            "Live pre-score market mask active: spread, ticker freshness, "
            "raw signal close, open interest, and configured liquidity checks "
            "run before base/meta model scoring."
        )
    global_auction_enabled = str(
        getattr(portfolio_policy, "portfolio_policy_version", "")
    ).startswith("global_auction")
    global_auction_decisions: List[Dict[str, Any]] = []

    # Step 1: Select candidates. When the caller already selected candidates
    # on selector-only features before loading the full model feature set, keep
    # that exact set so the expensive model pass does not silently re-filter it.
    if preselected_long_candidates is None or preselected_short_candidates is None:
        long_cands, short_cands = select_candidates(
            panel=panel,
            feats=feats,
            metric=str(thresholds.get("metric", "ret12h")),
        )
    else:
        long_cands = list(preselected_long_candidates)
        short_cands = list(preselected_short_candidates)
    timer.mark("candidate_selection")

    # Limit candidates
    long_cands = long_cands[:max_candidates]
    short_cands = short_cands[:max_candidates]

    results["long_candidates"] = long_cands
    results["short_candidates"] = short_cands

    tprint(f"Candidates: {len(long_cands)} long, {len(short_cands)} short")
    tprint(
        "Candidate mask output: "
        f"long={len(long_cands)} sample={long_cands[:8]} "
        f"short={len(short_cands)} sample={short_cands[:8]}"
    )
    selection_scope = (
        "global-auction candidate pool, no per-side top-N truncation, "
        if global_auction_enabled
        else f"top {max(1, int(max_entries_per_side))} per side, "
    )
    tprint(
        "Order selection policy: "
        f"{selection_scope}"
        f"global entry cap={max(0, int(max_entries_total))}; "
        "candidates are sorted by portfolio_priority, rank, score, then friction"
    )

    # Step 2: Process long candidates
    for side, candidates in [("long", long_cands), ("short", short_cands)]:
        if not candidates:
            continue

        # Get features for candidates
        candidate_features = _get_features_for_candidates_at_ts(
            feats, candidates, ts=signal_bar_ts
        )
        timer.mark(f"{side}_candidate_feature_matrix")

        # Defensive check: ensure candidate_features is a DataFrame
        if not isinstance(candidate_features, pd.DataFrame):
            tprint(
                f"Warning: candidate_features is not a DataFrame: {type(candidate_features)}"
            )
            continue

        # Safely check for empty - handle case where it might be a string or other type
        try:
            is_empty = (
                candidate_features is None
                or not isinstance(candidate_features, (pd.DataFrame, pd.Series))
                or (hasattr(candidate_features, "empty") and candidate_features.empty)
            )
        except Exception as e:
            tprint(
                f"Error checking candidate_features.empty: {e}, type: {type(candidate_features)}"
            )
            continue

        if is_empty:
            continue

        _log_feature_coverage(candidate_features, side)
        side_metrics: Dict[str, Any] = {
            "input_candidates": int(len(candidates)),
            "eligible_candidates": 0,
            "lgbm_strategy_mask_pass": 0,
            "lgbm_strategy_mask_block": 0,
            "asset_exclusion_block": 0,
            "prescore_market_mask_input": 0,
            "prescore_market_mask_pass": 0,
            "prescore_market_mask_block": 0,
            "prescore_market_mask_reasons": {},
            "base_gate_pass": 0,
            "chain_enter": 0,
            "threshold_pass": 0,
            "cooldown_pass": 0,
            "portfolio_pass": 0,
            "executed": 0,
            "meta_missing": 0,
            "order_errors": 0,
            "unexplained_order_errors": 0,
            "non_fatal_issues": 0,
        }
        base_preds_all: list[float] = []
        meta_preds_all: list[float] = []
        rank_pct_all: list[float] = []

        # Run full inference chain
        try:
            decision_rows: List[Dict[str, Any]] = []
            strategy_ids = (
                orchestrator.available_strategies(side, accepted_strategies)
                if hasattr(orchestrator, "available_strategies")
                else [f"{side}_mr"]
            )
            if not strategy_ids:
                tprint(
                    f"No deployable {side} strategies available after deployment "
                    f"filter; skipping {len(candidates)} candidate(s)."
                )
            for selected_strategy in strategy_ids:
                mask_symbols = _strategy_mask_symbols(
                    strategy_candidate_masks,
                    str(selected_strategy),
                )
                eligible_candidates = []
                for symbol in candidates:
                    if mask_symbols is not None and str(symbol) not in mask_symbols:
                        side_metrics["lgbm_strategy_mask_block"] += 1
                        continue
                    if _is_symbol_blocked_for_strategy(
                        symbol, str(selected_strategy), strategy_asset_exclusions
                    ):
                        side_metrics["asset_exclusion_block"] += 1
                        continue
                    side_metrics["lgbm_strategy_mask_pass"] += 1
                    eligible_candidates.append(symbol)
                side_metrics["eligible_candidates"] += int(len(eligible_candidates))
                if not eligible_candidates:
                    if candidates:
                        if mask_symbols is not None:
                            tprint(
                                f"LGBM strategy mask block: all {side} candidates "
                                f"skipped before base/meta for {selected_strategy}"
                            )
                        else:
                            tprint(
                                f"Asset exclusion block: all {side} candidates skipped "
                                f"for {selected_strategy}"
                            )
                    continue
                pre_score_market_snapshots: Dict[str, Dict[str, Any]] = {}
                if pre_score_market_mask_enabled:
                    eligible_candidates, pre_score_market_snapshots = (
                        _apply_pre_score_market_masks(
                            panel=panel,
                            candidates=eligible_candidates,
                            side=side,
                            strategy_id=str(selected_strategy),
                            executor=executor,
                            policy=portfolio_policy,
                            runtime_config=runtime_config,
                            now=now_utc,
                            signal_bar_ts=signal_bar_ts,
                            raw_close_reference_gap_bps=raw_close_reference_gap_bps,
                            max_signal_close_to_entry_seconds=max_signal_close_to_entry_seconds,
                            side_metrics=side_metrics,
                        )
                    )
                    if not eligible_candidates:
                        tprint(
                            f"Pre-score market mask block: all {side} candidates "
                            f"skipped before base/meta for {selected_strategy}"
                        )
                        continue
                eligible_features = candidate_features.loc[
                    [
                        symbol
                        for symbol in eligible_candidates
                        if symbol in candidate_features.index
                    ]
                ]
                if eligible_features.empty:
                    tprint(
                        f"Feature block: no feature rows available after asset "
                        f"filter for {side}/{selected_strategy}"
                    )
                    continue
                decision_feature_contract = _strategy_decision_feature_contract(
                    strategy_feature_contracts,
                    side=side,
                    strategy_id=str(selected_strategy),
                )
                if decision_feature_contract:
                    available_contract = [
                        col
                        for col in decision_feature_contract
                        if col in eligible_features.columns
                    ]
                    missing_contract = [
                        col
                        for col in decision_feature_contract
                        if col not in eligible_features.columns
                    ]
                    if missing_contract:
                        tprint(
                            "Strategy feature contract missing columns before "
                            f"scoring {side}/{strategy_core_id(str(selected_strategy))}: "
                            f"missing={missing_contract[:20]}"
                        )
                    if available_contract:
                        before_cols = int(eligible_features.shape[1])
                        eligible_features = eligible_features.loc[
                            :, available_contract
                        ].copy()
                        if before_cols != len(available_contract):
                            tprint(
                                "Strategy-scoped model feature frame: "
                                f"{side}/{strategy_core_id(str(selected_strategy))} "
                                f"rows={len(eligible_features)} "
                                f"cols={before_cols}->{len(available_contract)}"
                            )
                if all(
                    _is_symbol_blocked_for_strategy(
                        symbol, str(selected_strategy), strategy_asset_exclusions
                    )
                    for symbol in eligible_candidates
                ):
                    tprint(
                        f"Asset exclusion block: all {side} candidates skipped for "
                        f"{selected_strategy}"
                    )
                    continue
                base_gate = _select_top_base_prediction_symbols(
                    orchestrator=orchestrator,
                    candidate_features=eligible_features,
                    candidates=eligible_candidates,
                    side=side,
                    strategy_id=str(selected_strategy),
                )
                timer.mark(
                    f"{side}_{strategy_core_id(str(selected_strategy))}_base_gate"
                )
                side_metrics["base_gate_pass"] += int(len(base_gate))
                batch_meta_preds = pd.Series(dtype=float)
                batch_meta_model_inputs: Optional[pd.DataFrame] = None
                if base_gate:
                    base_gate_symbols = [
                        symbol
                        for symbol in eligible_candidates
                        if symbol in base_gate and symbol in eligible_features.index
                    ]
                    if base_gate_symbols:
                        meta_batch_features = eligible_features.loc[
                            base_gate_symbols
                        ].copy()
                        meta_batch_features[str(selected_strategy)] = pd.Series(
                            {
                                symbol: vals.get("base_pred", np.nan)
                                for symbol, vals in base_gate.items()
                            },
                            dtype=float,
                        ).reindex(meta_batch_features.index)
                        try:
                            batch_meta_preds = orchestrator.predict_meta(
                                meta_batch_features,
                                side,
                                str(selected_strategy),
                            )
                            last_meta_input = getattr(
                                orchestrator, "_last_meta_model_input", None
                            )
                            if isinstance(last_meta_input, pd.DataFrame):
                                batch_meta_model_inputs = last_meta_input.copy()
                            timer.mark(
                                f"{side}_{strategy_core_id(str(selected_strategy))}_meta_pred"
                            )
                            if isinstance(batch_meta_preds, pd.Series):
                                finite_batch_meta = batch_meta_preds.replace(
                                    [np.inf, -np.inf], np.nan
                                ).dropna()
                                if not finite_batch_meta.empty:
                                    tprint(
                                        f"Batch meta predictions {side}/{strategy_core_id(str(selected_strategy))}: "
                                        f"n={len(finite_batch_meta)} "
                                        f"min={float(finite_batch_meta.min()):.6f} "
                                        f"max={float(finite_batch_meta.max()):.6f} "
                                        f"std={float(finite_batch_meta.std(ddof=0)):.6f}"
                                    )
                        except Exception as exc:
                            tprint(
                                f"Batch meta prediction failed for {side}/{strategy_core_id(str(selected_strategy))}: {exc}"
                            )
                            batch_meta_preds = pd.Series(dtype=float)
                for symbol in eligible_candidates:
                    if (
                        symbol not in candidate_features.index
                        or symbol not in base_gate
                    ):
                        continue
                    if _is_symbol_blocked_for_strategy(
                        symbol, str(selected_strategy), strategy_asset_exclusions
                    ):
                        tprint(
                            f"Asset exclusion block: {symbol} skipped for "
                            f"{selected_strategy}"
                        )
                        continue
                    try:
                        chain_results = orchestrator.run_full_chain(
                            symbol,
                            side,
                            candidate_features.loc[[symbol]],
                            panel=panel,
                            kind=selected_strategy,
                        )
                    except TypeError as exc:
                        if "kind" not in str(exc):
                            raise
                        chain_results = orchestrator.run_full_chain(
                            symbol, side, candidate_features.loc[[symbol]]
                        )
                    strategy_id = str(
                        chain_results.get("strategy_id")
                        or strategy_core_id(str(selected_strategy))
                    )
                    if (
                        isinstance(batch_meta_preds, pd.Series)
                        and symbol in batch_meta_preds.index
                    ):
                        batch_meta_val = float(batch_meta_preds.loc[symbol])
                        if np.isfinite(batch_meta_val):
                            chain_results["meta_pred"] = batch_meta_val
                            chain_results["meta_prediction_source"] = (
                                "batch_meta_after_base_gate"
                            )
                            chain_results["action"] = "enter"
                    if accepted_strategies is not None and not strategy_id_matches(
                        strategy_id, accepted_strategies
                    ):
                        continue
                    if _is_symbol_blocked_for_strategy(
                        symbol, strategy_id, strategy_asset_exclusions
                    ):
                        tprint(
                            f"Asset exclusion block: {symbol} skipped for {strategy_id}"
                        )
                        continue
                    chain_results.update(base_gate.get(symbol, {}))
                    prescore_snapshot = pre_score_market_snapshots.get(str(symbol))
                    if prescore_snapshot:
                        chain_results.update(prescore_snapshot)
                    if live_feature_layer_debug and np.isfinite(
                        _safe_float(chain_results.get("meta_pred"))
                    ):
                        try:
                            debug_dir = dump_live_feature_layers(
                                orchestrator=orchestrator,
                                feature_row=candidate_features.loc[[symbol]],
                                symbol=str(symbol),
                                side=side,
                                selected_strategy=str(selected_strategy),
                                chain_results=chain_results,
                                runtime_cfg=runtime_config,
                                timestamp=now_utc,
                                signal_bar_ts=signal_bar_ts,
                                feature_universe_symbols=list(
                                    panel.get("close", pd.DataFrame()).columns
                                ),
                            )
                            if debug_dir is not None:
                                chain_results["live_feature_layer_debug_dir"] = str(
                                    debug_dir
                                )
                                tprint(
                                    "Live feature-layer debug saved: "
                                    f"{debug_dir} "
                                    f"features={candidate_features.shape[1]} "
                                    f"feature_universe={len(panel.get('close', pd.DataFrame()).columns)}"
                                )
                        except Exception as exc:
                            tprint(
                                "Warning: failed to persist live feature-layer "
                                f"debug dump for {symbol} {side}/{strategy_id}: {exc}"
                            )
                    if chain_results.get("action") != "enter":
                        if chain_results.get("action") == "no_meta_prediction":
                            side_metrics["meta_missing"] += 1
                        if _force_shadow_entry_for_integration(executor):
                            tprint(
                                f"Shadow integration override: forcing entry for "
                                f"{symbol} {side}/{strategy_id} after "
                                f"action={chain_results.get('action')} "
                                f"reason={chain_results.get('reason', '')}"
                            )
                            chain_results["action"] = "enter"
                            chain_results["forced_shadow_entry"] = True
                        else:
                            tprint(
                                f"Entry policy block: {symbol} {side}/{strategy_id} "
                                f"action={chain_results.get('action')} "
                                f"reason={chain_results.get('reason', '')}"
                            )
                            continue
                    if chain_results.get("action") != "enter":
                        tprint(
                            f"Entry policy block: {symbol} {side}/{strategy_id} "
                            f"action={chain_results.get('action')} "
                            f"reason={chain_results.get('reason', '')}"
                        )
                        continue
                    side_metrics["chain_enter"] += 1
                    base_pred_val = float(chain_results.get("base_pred", np.nan))
                    if np.isfinite(base_pred_val):
                        base_preds_all.append(base_pred_val)
                    try:
                        raw_score = float(chain_results.get("meta_pred", np.nan))
                    except (TypeError, ValueError):
                        raw_score = float("nan")
                    if not np.isfinite(raw_score):
                        side_metrics["meta_missing"] += 1
                        tprint(
                            f"Meta prediction block: {symbol} {side}/{strategy_id} "
                            "has no finite meta prediction; no base fallback is used."
                        )
                        continue
                    if np.isfinite(raw_score):
                        meta_preds_all.append(raw_score)
                    estimated_hit_rate = _estimated_hit_rate_from_meta_prediction(
                        raw_score,
                        strategy_id,
                        meta_hit_rate_calibration,
                    )
                    calibrated_score, rank_threshold = calibrated_score_and_threshold(
                        raw_score=raw_score,
                        strategy_id=strategy_id,
                        calibration_data=calibration_data,
                        default_threshold=initial_rank_threshold,
                    )
                    estimated_ev = _estimated_ev_from_strategy_prediction(
                        calibrated_score,
                        strategy_id,
                        strategy_ev_calibration,
                    )
                    run_cfg = getattr(executor, "config", {}) or {}
                    artifact_data_root = str(run_cfg.get("data_root", "data"))
                    artifact_run_id = str(
                        run_cfg.get("model_artifact_run_id")
                        or run_cfg.get("run_id", "latest")
                    )
                    policy_artifact_run_id = str(
                        run_cfg.get("policy_artifact_run_id") or artifact_run_id
                    )
                    meta_hist_rank_pct = _historical_prediction_rank_pct(
                        raw_score,
                        data_root=artifact_data_root,
                        run_id=artifact_run_id,
                        strategy_id=strategy_id,
                        kind="meta",
                    )
                    base_hist_rank_pct = _historical_prediction_rank_pct(
                        chain_results.get("base_pred"),
                        data_root=artifact_data_root,
                        run_id=artifact_run_id,
                        strategy_id=strategy_id,
                        kind="base",
                    )
                    nrow = normalized_thresholds.get(strategy_id, {})
                    threshold_space = str(nrow.get("threshold_space", "") or "")
                    normalized_threshold = float(
                        nrow.get("normalized_threshold", rank_threshold)
                    )
                    viability_margin = float(nrow.get("viability_margin", 0.0))
                    policy_threshold_floor = float(
                        np.clip(
                            getattr(
                                portfolio_policy,
                                "initial_rank_threshold_floor",
                                getattr(
                                    portfolio_policy,
                                    "initial_rank_threshold",
                                    initial_rank_threshold,
                                ),
                            ),
                            0.0,
                            1.0,
                        )
                    )
                    auction_ev_open_positions = (
                        int(starting_active_position_count) + int(total_entries_executed)
                    )
                    auction_ev_target = _auction_ev_target_for_occupancy(
                        open_positions=auction_ev_open_positions,
                        policy=portfolio_policy,
                    )
                    auction_ev_gate = policy_rank_reference_store.auction_threshold_for_ev(
                        target_mean_net_return=auction_ev_target,
                        min_hit_rate=AUCTION_EV_MIN_HIT_RATE,
                        fallback_threshold=1.0,
                    )
                    strategy_ev_threshold = (
                        policy_rank_reference_store.strategy_threshold_for_ev(
                            strategy_id=strategy_id,
                            side=side,
                            target_mean_net_return=auction_ev_target,
                            min_hit_rate=AUCTION_EV_MIN_HIT_RATE,
                            fallback_threshold=1.0,
                        )
                    )
                    strategy_ev_gate = policy_rank_reference_store.strategy_ev_gate(
                        strategy_id=strategy_id,
                        side=side,
                        target_mean_net_return=auction_ev_target,
                        min_hit_rate=AUCTION_EV_MIN_HIT_RATE,
                    )
                    if not strategy_ev_threshold.enabled:
                        tprint(
                            f"Strategy EV threshold block: {symbol} {side}/{strategy_id} "
                            f"reason={strategy_ev_threshold.reason} "
                            f"best_avg_net={strategy_ev_threshold.mean_net_return:.6f} "
                            f"target={strategy_ev_threshold.target_mean_net_return:.6f} "
                            f"best_hit={strategy_ev_threshold.hit_rate:.6f} "
                            f"min_hit={strategy_ev_threshold.target_hit_rate:.6f}"
                        )
                        continue
                    meta_contracts = _model_feature_contracts_for_audit(
                        orchestrator,
                        side=side,
                        strategy_id=strategy_id,
                    )
                    meta_model_key = str(meta_contracts.get("meta_model_key") or "")
                    _, meta_model_for_hash = _resolve_meta_model_for_audit(
                        orchestrator,
                        side=side,
                        strategy_id=strategy_id,
                    )
                    meta_hash = meta_head_hash(
                        meta_model_key=meta_model_key,
                        meta_model=meta_model_for_hash,
                        feature_contract=list(meta_contracts.get("meta_features") or []),
                    )
                    dynamic_state = (
                        dynamic_performance_monitor.threshold_multiplier(
                            strategy_id, meta_hash
                        )
                        if dynamic_performance_monitor is not None
                        else None
                    )
                    dynamic_multiplier = float(
                        getattr(dynamic_state, "multiplier", 1.0) or 1.0
                    )
                    dynamic_multiplier = float(
                        np.clip(dynamic_multiplier, 0.8, 1.2)
                    )
                    raw_strategy_threshold = float(
                        np.clip(strategy_ev_threshold.threshold, 0.0, 1.0)
                    )
                    policy_threshold_floor = float(
                        np.clip(raw_strategy_threshold * dynamic_multiplier, 0.0, 1.0)
                    )
                    if dynamic_performance_monitor is not None:
                        tprint(
                            f"Dynamic performance threshold: {symbol} {side}/{strategy_id} "
                            f"meta_hash={meta_hash} raw={raw_strategy_threshold:.6f} "
                            f"mult={dynamic_multiplier:.3f} "
                            f"adjusted={policy_threshold_floor:.6f} "
                            f"recent_hit={_safe_float(getattr(dynamic_state, 'recent_hit_rate', np.nan)):.4f} "
                            f"expected_hit={_safe_float(getattr(dynamic_state, 'expected_hit_rate', np.nan)):.4f} "
                            f"n={int(getattr(dynamic_state, 'recent_n', 0) or 0)} "
                            f"reason={getattr(dynamic_state, 'reason', '')}"
                        )
                    if threshold_space == "calibrated_score":
                        effective_threshold = min(
                            1.0,
                            max(rank_threshold, normalized_threshold)
                            + viability_margin,
                        )
                        threshold_score = calibrated_score
                        if calibrated_score < effective_threshold:
                            tprint(
                                f"Calibration block: {symbol} {side}/{strategy_id} "
                                f"score={calibrated_score:.6f} "
                                f"threshold={effective_threshold:.6f}"
                            )
                            continue
                    else:
                        effective_threshold = policy_threshold_floor
                        threshold_score = np.nan
                    rank_score = (
                        float(meta_hist_rank_pct)
                        if np.isfinite(meta_hist_rank_pct)
                        else float(calibrated_score)
                    )
                    policy_params = _policy_params_for_strategy(
                        orchestrator,
                        strategy_id,
                    )
                    policy_size = _meta_policy_position_size(
                        calibrated_score=calibrated_score,
                        threshold=rank_threshold,
                        policy_params=policy_params,
                        symbol=symbol,
                    )
                    size = float(policy_size["position_size"])
                    chain_results["legacy_orchestrator_position_size"] = (
                        chain_results.get("orchestrator_position_size")
                        or chain_results.get("position_size")
                    )
                    chain_results.update(policy_size)
                    if abs(size) < 0.01:
                        tprint(
                            f"Size block: {symbol} {side}/{strategy_id} "
                            f"size={size:.6f} asset_decision={policy_size.get('asset_decision')}"
                        )
                        continue
                    chain_results["strategy_id"] = strategy_id
                    chain_results["meta_model_key"] = meta_model_key
                    chain_results["meta_head_hash"] = meta_hash
                    chain_results["calibrated_score"] = calibrated_score
                    chain_results["rank_threshold"] = rank_threshold
                    chain_results["normalized_threshold"] = normalized_threshold
                    chain_results["deployment_rank_threshold"] = normalized_threshold
                    chain_results["initial_rank_threshold_floor"] = (
                        policy_threshold_floor
                    )
                    chain_results["auction_ev_threshold"] = float(
                        auction_ev_gate.threshold
                    )
                    chain_results["auction_ev_threshold_enabled"] = bool(
                        auction_ev_gate.enabled
                    )
                    chain_results["auction_ev_threshold_reason"] = (
                        auction_ev_gate.reason
                    )
                    chain_results["auction_ev_target_mean_net_return"] = float(
                        auction_ev_gate.target_mean_net_return
                    )
                    chain_results["auction_ev_target_hit_rate"] = float(
                        auction_ev_gate.target_hit_rate
                    )
                    chain_results["auction_ev_threshold_mean_net_return"] = float(
                        auction_ev_gate.mean_net_return
                    )
                    chain_results["auction_ev_threshold_hit_rate"] = float(
                        auction_ev_gate.hit_rate
                    )
                    chain_results["auction_ev_threshold_n_trades"] = int(
                        auction_ev_gate.n_trades
                    )
                    chain_results["strategy_ev_threshold"] = float(
                        strategy_ev_threshold.threshold
                    )
                    chain_results["strategy_ev_threshold_before_dynamic"] = float(
                        raw_strategy_threshold
                    )
                    chain_results["dynamic_performance_multiplier"] = float(
                        dynamic_multiplier
                    )
                    chain_results["dynamic_performance_reason"] = (
                        getattr(dynamic_state, "reason", "disabled")
                        if dynamic_state is not None
                        else "disabled"
                    )
                    chain_results["dynamic_performance_expected_hit_rate"] = float(
                        getattr(dynamic_state, "expected_hit_rate", np.nan)
                        if dynamic_state is not None
                        else np.nan
                    )
                    chain_results["dynamic_performance_recent_hit_rate"] = float(
                        getattr(dynamic_state, "recent_hit_rate", np.nan)
                        if dynamic_state is not None
                        else np.nan
                    )
                    chain_results["dynamic_performance_recent_n"] = int(
                        getattr(dynamic_state, "recent_n", 0) or 0
                    )
                    chain_results["strategy_ev_threshold_enabled"] = bool(
                        strategy_ev_threshold.enabled
                    )
                    chain_results["strategy_ev_threshold_reason"] = (
                        strategy_ev_threshold.reason
                    )
                    chain_results["strategy_ev_threshold_mean_net_return"] = float(
                        strategy_ev_threshold.mean_net_return
                    )
                    chain_results["strategy_ev_threshold_hit_rate"] = float(
                        strategy_ev_threshold.hit_rate
                    )
                    chain_results["strategy_ev_threshold_n_trades"] = int(
                        strategy_ev_threshold.n_trades
                    )
                    chain_results["strategy_ev_gate_allowed"] = bool(
                        strategy_ev_gate.allowed
                    )
                    chain_results["strategy_ev_gate_reason"] = strategy_ev_gate.reason
                    chain_results["strategy_ev_avg_net_return"] = float(
                        strategy_ev_gate.mean_net_return
                    )
                    chain_results["strategy_ev_hit_rate"] = float(
                        strategy_ev_gate.hit_rate
                    )
                    chain_results["strategy_ev_target_mean_net_return"] = float(
                        strategy_ev_gate.target_mean_net_return
                    )
                    chain_results["strategy_ev_min_hit_rate"] = float(
                        strategy_ev_gate.min_hit_rate
                    )
                    chain_results["viability_margin"] = viability_margin
                    chain_results["effective_threshold"] = effective_threshold
                    chain_results["base_train_rank_pct"] = base_hist_rank_pct
                    chain_results["meta_train_rank_pct"] = meta_hist_rank_pct
                    chain_results["rank_score_source"] = (
                        "historical_meta_oof_percentile"
                        if np.isfinite(meta_hist_rank_pct)
                        else "missing_policy_rank_reference_percentile"
                    )
                    try:
                        exact_meta_model_input = None
                        if (
                            isinstance(batch_meta_model_inputs, pd.DataFrame)
                            and symbol in batch_meta_model_inputs.index
                        ):
                            exact_meta_model_input = batch_meta_model_inputs.loc[
                                [symbol]
                            ]
                        chain_results.update(
                            _model_feature_ledger_snapshot_for_decision(
                                orchestrator=orchestrator,
                                side=side,
                                strategy_id=strategy_id,
                                symbol=symbol,
                                candidate_features=candidate_features.loc[[symbol]],
                                meta_model_input_features=exact_meta_model_input,
                                feats=feats,
                                chain_results=chain_results,
                                signal_bar_ts=signal_bar_ts,
                            )
                        )
                        if exact_meta_model_input is not None:
                            chain_results["_meta_model_input_features"] = (
                                exact_meta_model_input
                            )
                    except Exception as exc:
                        tprint(
                            "Warning: failed to build selected model feature ledger "
                            f"snapshot for {symbol} {side}/{strategy_id}: {exc}"
                        )
                    chain_results.update(estimated_hit_rate)
                    chain_results.update(estimated_ev)
                    side_metrics["threshold_pass"] += 1
                    decision_rows.append(
                        {
                            "symbol": symbol,
                            "side": side,
                            "size": size,
                            "strategy_id": strategy_id,
                            "meta_model_key": meta_model_key,
                            "meta_head_hash": meta_hash,
                            "raw_score": raw_score,
                            "calibrated_score": calibrated_score,
                            "threshold_space": threshold_space or "rank_percentile",
                            "rank_score": rank_score,
                            "rank_score_source": chain_results["rank_score_source"],
                            "threshold_score": threshold_score,
                            "rank_threshold": rank_threshold,
                            **estimated_hit_rate,
                            **estimated_ev,
                            "normalized_threshold": normalized_threshold,
                            "deployment_rank_threshold": normalized_threshold,
                            "initial_rank_threshold_floor": policy_threshold_floor,
                            "auction_ev_threshold": float(auction_ev_gate.threshold),
                            "auction_ev_threshold_enabled": bool(
                                auction_ev_gate.enabled
                            ),
                            "auction_ev_threshold_reason": auction_ev_gate.reason,
                            "auction_ev_target_mean_net_return": float(
                                auction_ev_gate.target_mean_net_return
                            ),
                            "auction_ev_target_hit_rate": float(
                                auction_ev_gate.target_hit_rate
                            ),
                            "auction_ev_threshold_mean_net_return": float(
                                auction_ev_gate.mean_net_return
                            ),
                            "auction_ev_threshold_hit_rate": float(
                                auction_ev_gate.hit_rate
                            ),
                            "auction_ev_threshold_n_trades": int(
                                auction_ev_gate.n_trades
                            ),
                            "strategy_ev_threshold": float(
                                strategy_ev_threshold.threshold
                            ),
                            "strategy_ev_threshold_before_dynamic": float(
                                raw_strategy_threshold
                            ),
                            "dynamic_performance_multiplier": float(
                                dynamic_multiplier
                            ),
                            "dynamic_performance_reason": (
                                getattr(dynamic_state, "reason", "disabled")
                                if dynamic_state is not None
                                else "disabled"
                            ),
                            "dynamic_performance_expected_hit_rate": float(
                                getattr(dynamic_state, "expected_hit_rate", np.nan)
                                if dynamic_state is not None
                                else np.nan
                            ),
                            "dynamic_performance_recent_hit_rate": float(
                                getattr(dynamic_state, "recent_hit_rate", np.nan)
                                if dynamic_state is not None
                                else np.nan
                            ),
                            "dynamic_performance_recent_n": int(
                                getattr(dynamic_state, "recent_n", 0) or 0
                            ),
                            "strategy_ev_threshold_enabled": bool(
                                strategy_ev_threshold.enabled
                            ),
                            "strategy_ev_threshold_reason": (
                                strategy_ev_threshold.reason
                            ),
                            "strategy_ev_threshold_mean_net_return": float(
                                strategy_ev_threshold.mean_net_return
                            ),
                            "strategy_ev_threshold_hit_rate": float(
                                strategy_ev_threshold.hit_rate
                            ),
                            "strategy_ev_threshold_n_trades": int(
                                strategy_ev_threshold.n_trades
                            ),
                            "strategy_ev_gate_allowed": bool(
                                strategy_ev_gate.allowed
                            ),
                            "strategy_ev_gate_reason": strategy_ev_gate.reason,
                            "strategy_ev_avg_net_return": float(
                                strategy_ev_gate.mean_net_return
                            ),
                            "strategy_ev_hit_rate": float(strategy_ev_gate.hit_rate),
                            "strategy_ev_target_mean_net_return": float(
                                strategy_ev_gate.target_mean_net_return
                            ),
                            "strategy_ev_min_hit_rate": float(
                                strategy_ev_gate.min_hit_rate
                            ),
                            "effective_threshold": effective_threshold,
                            "policy_sizing": policy_size,
                            "chain_results": chain_results,
                            "decision_ts": now_utc.isoformat(),
                            "signal_bar_ts": signal_bar_ts.isoformat(),
                            "signal_bar_close_ts": (
                                _signal_bar_close_ts(signal_bar_ts).isoformat()
                                if _signal_bar_close_ts(signal_bar_ts) is not None
                                else None
                            ),
                            "feature_source_max_ts": (
                                feature_source_max_ts.isoformat()
                                if feature_source_max_ts is not None
                                else None
                            ),
                            "feature_available_ts": feature_available_ts.isoformat(),
                            "feature_contract_hash": runtime_config.get(
                                "feature_contract_hash"
                            ),
                            "feature_transform_contract_hash": runtime_config.get(
                                "feature_transform_contract_hash"
                            ),
                            "model_artifact_run_id": artifact_run_id,
                            "policy_artifact_run_id": policy_artifact_run_id,
                            "_meta_model_input_features": chain_results.get(
                                "_meta_model_input_features"
                            ),
                        }
                    )

            _attach_rank_percentile_scores(
                decision_rows,
                allow_live_batch_rank_fallback_for_debug=allow_live_batch_rank_fallback_for_debug,
            )
            filtered_decision_rows: List[Dict[str, Any]] = []
            for decision in decision_rows:
                if str(decision.get("threshold_space", "")) == "calibrated_score":
                    filtered_decision_rows.append(decision)
                    continue
                gate_allowed, gate_reason = apply_policy_rank_percentile_gate(
                    decision,
                    store=policy_rank_reference_store,
                    allow_live_batch_rank_fallback_for_debug=allow_live_batch_rank_fallback_for_debug,
                    inference_min_base_train_rank_pct=inference_min_base_train_rank_pct,
                    require_cross_strategy_auction_rank=require_cross_strategy_auction_rank,
                    use_auction_rank_for_threshold=True,
                )
                _assert_policy_rank_threshold_source(decision)
                rank_pct = float(
                    decision.get(
                        "threshold_rank_score",
                        decision.get(
                            "threshold_score",
                            decision.get(
                                "policy_rank_pct",
                                decision.get("sizer_rank_percentile", np.nan),
                            ),
                        ),
                    )
                )
                threshold = float(decision.get("effective_threshold", 1.0))
                if np.isfinite(rank_pct):
                    rank_pct_all.append(rank_pct)
                if not gate_allowed:
                    reject_reason = str(gate_reason or "rank_below_dynamic_threshold")
                    rejected_chain_results = dict(decision.get("chain_results") or {})
                    update_live_feature_layer_rank_summary(
                        rejected_chain_results.get("live_feature_layer_debug_dir"),
                        decision=decision,
                        chain_results=rejected_chain_results,
                        gate_allowed=False,
                        gate_reason=reject_reason,
                    )
                    if (
                        prediction_ledger is not None
                        and _should_log_prediction_candidate(
                            decision, policy=portfolio_policy
                        )
                    ):
                        prediction_ledger_rows.append(
                            _prediction_ledger_row(
                                decision,
                                timestamp=now_utc.isoformat(),
                                side=str(decision.get("side", "")),
                                portfolio_decision="rank_rejected",
                                portfolio_reject_reason=reject_reason,
                            )
                        )
                    tprint(
                        f"Rank-threshold block: {decision['symbol']} "
                        f"{decision['side']}/{decision['strategy_id']} "
                        f"reason={reject_reason} rank={rank_pct:.6f} "
                        f"threshold={threshold:.6f}"
                    )
                    continue
                decision["threshold_score"] = rank_pct
                chain_results = dict(decision["chain_results"])
                chain_results["sizer_rank_percentile"] = decision.get(
                    "sizer_rank_percentile"
                )
                chain_results["normalized_rank_score"] = decision.get(
                    "normalized_rank_score", rank_pct
                )
                chain_results["policy_rank_pct"] = decision.get("policy_rank_pct")
                chain_results["auction_rank_pct"] = decision.get("auction_rank_pct")
                chain_results["auction_rank_reference_n"] = decision.get(
                    "auction_rank_reference_n"
                )
                chain_results["auction_rank_reference_source"] = decision.get(
                    "auction_rank_reference_source"
                )
                chain_results["auction_rank_score_source"] = decision.get(
                    "auction_rank_score_source"
                )
                chain_results["threshold_rank_score"] = rank_pct
                chain_results["threshold_rank_score_source"] = decision.get(
                    "threshold_rank_score_source"
                )
                chain_results["policy_rank_reference_n"] = decision.get(
                    "policy_rank_reference_n"
                )
                chain_results["policy_rank_reference_source"] = decision.get(
                    "policy_rank_reference_source"
                )
                chain_results["effective_threshold"] = threshold
                update_live_feature_layer_rank_summary(
                    chain_results.get("live_feature_layer_debug_dir"),
                    decision=decision,
                    chain_results=chain_results,
                    gate_allowed=True,
                    gate_reason=None,
                )
                decision["chain_results"] = chain_results
                decision["threshold_score"] = rank_pct
                decision["normalized_rank_score"] = decision.get(
                    "normalized_rank_score", rank_pct
                )
                decision["portfolio_priority"] = _candidate_portfolio_priority(
                    decision
                )
                filtered_decision_rows.append(decision)
            decision_rows = filtered_decision_rows
            decision_rows.sort(
                key=lambda row: (
                    _safe_float(row.get("portfolio_priority"), -float("inf")),
                    _candidate_rank_score(row),
                    _safe_float(row.get("calibrated_score"), -float("inf")),
                    -_candidate_expected_friction(row),
                ),
                reverse=True,
            )
            if not global_auction_enabled:
                decision_rows = decision_rows[: max(1, int(max_entries_per_side))]
            if strategy_kill_switch is not None and decision_rows:
                allowed_decision_rows = []
                for row in decision_rows:
                    row_strategy_id = str(row["strategy_id"])
                    row_symbol = str(row["symbol"])
                    strategy_switch_decision = strategy_kill_switch.is_blocked(
                        row_strategy_id
                    )
                    if strategy_switch_decision.allow_new_entries:
                        allowed_decision_rows.append(row)
                        continue
                    if (
                        prediction_ledger is not None
                        and _should_log_prediction_candidate(
                            row, policy=portfolio_policy
                        )
                    ):
                        prediction_ledger_rows.append(
                            _prediction_ledger_row(
                                row,
                                timestamp=now_utc.isoformat(),
                                side=side,
                                portfolio_decision="strategy_kill_switch_rejected",
                                portfolio_reject_reason=strategy_switch_decision.reason,
                            )
                        )
                    tprint(
                        f"Strategy kill-switch block: {row_symbol} "
                        f"{side}/{row_strategy_id} "
                        f"reason={strategy_switch_decision.reason}"
                    )
                    side_metrics["non_fatal_issues"] += 1
                decision_rows = allowed_decision_rows
                if not decision_rows:
                    tprint(
                        f"All ranked decisions [{side}] were blocked by strategy "
                        "kill switches"
                    )
                    continue
            if global_auction_enabled:
                for row in decision_rows:
                    row["_auction_side"] = side
                    row["_auction_side_metrics"] = side_metrics
                global_auction_decisions.extend(decision_rows)
                decision_rows = []
            global_entry_cap = max(0, int(max_entries_total))
            tprint(
                f"Top-{max(1, int(max_entries_per_side))} selection [{side}]: "
                f"selected={len(decision_rows)} from ranked_decisions "
                f"(global_remaining={max(0, global_entry_cap - total_entries_executed)})"
            )
            for decision in decision_rows:
                if total_entries_executed >= global_entry_cap:
                    tprint(
                        f"Global entry cap reached ({total_entries_executed}/{global_entry_cap}); skipping remaining ranked decisions"
                    )
                    break
                symbol = str(decision["symbol"])
                strategy_id = str(decision["strategy_id"])
                chain_results = dict(decision["chain_results"])
                size = float(decision["size"])
                bucket_key = strategy_core_id(strategy_id)
                resolver = getattr(executor, "resolve_simple_policy_strategy_id", None)
                if callable(resolver):
                    resolved_bucket_key = resolver(bucket_key, side)
                    if resolved_bucket_key:
                        bucket_key = str(resolved_bucket_key)
                threshold_for_size = float(decision["effective_threshold"])
                rank_for_size = float(
                    decision.get(
                        "threshold_score",
                        decision.get("calibrated_score", 0.0),
                    )
                )
                cooldown_hours = LOSING_TRADE_COOLDOWN_HOURS
                symbol_block_reason = _symbol_entry_block_reason(
                    symbol,
                    now=now_utc,
                    logger=logger,
                    executor=executor,
                    cooldown_hours=cooldown_hours,
                )
                if symbol_block_reason:
                    if (
                        prediction_ledger is not None
                        and (
                            trade_success
                            or _should_log_prediction_candidate(
                                decision, policy=portfolio_policy
                            )
                        )
                    ):
                        prediction_ledger_rows.append(
                            _prediction_ledger_row(
                                decision,
                                timestamp=now_utc.isoformat(),
                                side=side,
                                portfolio_decision="portfolio_rejected",
                                portfolio_reject_reason=symbol_block_reason,
                            )
                        )
                    base_pred = _safe_float(chain_results.get("base_pred"))
                    meta_pred = _safe_float(chain_results.get("meta_pred"))
                    rank_pct = _safe_float(chain_results.get("sizer_rank_percentile"))
                    base_train_rank = _safe_float(
                        chain_results.get("base_train_rank_pct")
                    )
                    meta_train_rank = _safe_float(
                        chain_results.get("meta_train_rank_pct")
                    )
                    threshold = _safe_float(chain_results.get("effective_threshold"))
                    if symbol_block_reason == "symbol_already_active":
                        reason_text = "active-symbol one-position constraint"
                    else:
                        reason_text = f"{cooldown_hours:.1f}h losing-trade window"
                    tprint(
                        f"Symbol entry block: {symbol} {side}/{strategy_id} "
                        f"reason={symbol_block_reason} skipped for {reason_text} "
                        f"base={base_pred:.6f} meta={meta_pred:.6f} "
                        f"base_train_rank={base_train_rank:.6f} "
                        f"meta_train_rank={meta_train_rank:.6f} "
                        f"norm_rank={rank_pct:.6f} threshold={threshold:.6f}"
                    )
                    side_metrics["non_fatal_issues"] += 1
                    continue
                side_metrics["cooldown_pass"] += 1
                if portfolio_mgr is not None:
                    capacity = portfolio_mgr.get_portfolio_capacity(
                        side=side,
                        strategy_id=strategy_id,
                    )
                    _attach_portfolio_replay_state_for_ledger(
                        decision,
                        portfolio_mgr=portfolio_mgr,
                        capacity=capacity,
                        now_utc=now_utc,
                    )
                    perp_rank = (
                        _perp_rank_context(
                            data_root=runtime_config["data_root"],
                            run_id=runtime_config["run_id"],
                            side=side,
                            strategy_id=strategy_id,
                            score=float(
                                chain_results.get("meta_pred")
                                or decision.get("calibrated_score")
                                or rank_for_size
                            ),
                        )
                        if _is_perps_config(runtime_config)
                        else {}
                    )
                    sizing_audit = compute_rank_based_position_size(
                        wallet_value=float(capacity["wallet_value"]),
                        open_notional=float(capacity["open_notional"]),
                        adjusted_rank_score=rank_for_size,
                        final_threshold=threshold_for_size,
                        policy=portfolio_policy,
                        liquidity_capacity_weight=1.0,
                        live_test_mode=live_test_mode,
                        rank_size_power=float(policy_size.get("size_power", 1.1)),
                        total_assets_quote=capacity.get("total_assets_quote"),
                        total_liabilities_quote=capacity.get("total_liabilities_quote"),
                        open_positions=capacity.get("open_positions"),
                        market_mode=runtime_config.get("market_mode", "spot"),
                        available_wallet_value=capacity.get("available_wallet_quote"),
                        stop_loss_pct=live_barrier_pct,
                        rank_number=perp_rank.get("rank_number"),
                        rank_x=perp_rank.get("rank_x"),
                    )
                    requested_position_usdt = float(
                        sizing_audit["size_after_liquidity"]
                    )
                    chain_results["portfolio_rank_sizing"] = sizing_audit
                    can_enter, info = portfolio_mgr.can_enter_position(
                        symbol=symbol,
                        side=side,
                        strategy_id=strategy_id,
                        rank_score=rank_for_size,
                        initial_threshold=threshold_for_size,
                        current_time=now_utc,
                        requested_position_size=requested_position_usdt,
                    )
                    chain_results["portfolio_gate"] = info
                    if not can_enter:
                        if (
                            prediction_ledger is not None
                            and _should_log_prediction_candidate(
                                decision, policy=portfolio_policy
                            )
                        ):
                            prediction_ledger_rows.append(
                                _prediction_ledger_row(
                                    decision,
                                    timestamp=now_utc.isoformat(),
                                    side=side,
                                    portfolio_decision="portfolio_rejected",
                                    portfolio_reject_reason=str(
                                        info.get("reason") or "portfolio_rejected"
                                    ),
                                )
                            )
                        tprint(
                            f"Portfolio block: {symbol} {side}/{strategy_id} "
                            f"reason={info.get('reason') or info}"
                        )
                        side_metrics["non_fatal_issues"] += 1
                        continue
                    side_metrics["portfolio_pass"] += 1
                    size = min(
                        requested_position_usdt,
                        float(info.get("position_size_cap", requested_position_usdt)),
                    )
                    if live_test_mode and size > 0:
                        live_test_min_notional = float(
                            portfolio_policy.live_test_min_quote_notional
                        )
                        if size < live_test_min_notional:
                            reject_reason = "below_live_test_min_notional_after_caps"
                            if (
                                prediction_ledger is not None
                                and _should_log_prediction_candidate(
                                    decision, policy=portfolio_policy
                                )
                            ):
                                prediction_ledger_rows.append(
                                    _prediction_ledger_row(
                                        decision,
                                        timestamp=now_utc.isoformat(),
                                        side=side,
                                        portfolio_decision="portfolio_rejected",
                                        portfolio_reject_reason=reject_reason,
                                    )
                                )
                            tprint(
                                f"Portfolio block: {symbol} {side}/{strategy_id} "
                                f"reason={reject_reason} size={size:.8f} "
                                f"min_live_test_notional={live_test_min_notional:.8f}"
                            )
                            side_metrics["non_fatal_issues"] += 1
                            continue
                    signal_close_snapshot = _raw_signal_close_reliability_snapshot(
                        panel,
                        symbol,
                        max_reference_gap_bps=raw_close_reference_gap_bps,
                    )
                    price = signal_close_snapshot.get("signal_price")
                    predictions = {
                        "position_size": size,
                        "base_position_size": chain_results.get(
                            "base_position_size", ""
                        ),
                        "sizing_source": chain_results.get("sizing_source", ""),
                        "size_power": chain_results.get("size_power", ""),
                        "asset_weight_multiplier": chain_results.get(
                            "asset_weight_multiplier", ""
                        ),
                        "asset_decision": chain_results.get("asset_decision", ""),
                        "meta_pred": chain_results.get("meta_pred", ""),
                        "estimated_hit_rate": chain_results.get(
                            "estimated_hit_rate", ""
                        ),
                        "estimated_hit_rate_source": chain_results.get(
                            "estimated_hit_rate_source", ""
                        ),
                        "estimated_hit_rate_calibration_n": chain_results.get(
                            "estimated_hit_rate_calibration_n", ""
                        ),
                        "estimated_ev_gross_return": chain_results.get(
                            "estimated_ev_gross_return", ""
                        ),
                        "estimated_ev_net_return": chain_results.get(
                            "estimated_ev_net_return", ""
                        ),
                        "estimated_ev_cost_bps": chain_results.get(
                            "estimated_ev_cost_bps", ""
                        ),
                        "estimated_ev_hit_rate": chain_results.get(
                            "estimated_ev_hit_rate", ""
                        ),
                        "estimated_ev_source": chain_results.get(
                            "estimated_ev_source", ""
                        ),
                        "estimated_ev_calibration_n": chain_results.get(
                            "estimated_ev_calibration_n", ""
                        ),
                        "action": chain_results.get("action", ""),
                        "base_pred": chain_results.get("base_pred", ""),
                        "base_rank_pct": chain_results.get("base_rank_pct", ""),
                        "base_train_rank_pct": chain_results.get(
                            "base_train_rank_pct", ""
                        ),
                        "base_gate_top_frac": chain_results.get(
                            "base_gate_top_frac", ""
                        ),
                        "meta_train_rank_pct": chain_results.get(
                            "meta_train_rank_pct", ""
                        ),
                        "rank_score_source": chain_results.get("rank_score_source", ""),
                        "policy_rank_pct": chain_results.get("policy_rank_pct", ""),
                        "policy_rank_reference_n": chain_results.get(
                            "policy_rank_reference_n", ""
                        ),
                        "policy_rank_reference_source": chain_results.get(
                            "policy_rank_reference_source", ""
                        ),
                        "sizer_rank_percentile": chain_results.get(
                            "sizer_rank_percentile", ""
                        ),
                        "effective_threshold": chain_results.get(
                            "effective_threshold", ""
                        ),
                        "model_artifact_run_id": decision.get("model_artifact_run_id"),
                        "policy_artifact_run_id": decision.get("policy_artifact_run_id"),
                        "ridge_confidence": chain_results.get("ridge_confidence", ""),
                    }
                    features_log = {}
                    for feat_name in [
                        "ret24h",
                        "range_12h_pct",
                        "volatility_zscore",
                        "vol_zscore",
                        "volume",
                        "vol_z",
                        "trend_pct",
                        "trend",
                        "entropy",
                        "vol_of_vol",
                        "kurtosis",
                        "jump_frequency",
                        "funding_cost",
                        "borrow_cost",
                        "mkt_rv_ratio",
                        "G_VOL",
                        "G_TREND",
                        "G_VOLUME",
                        "atr_frac",
                        "atr_pct",
                        "atr_pct_base",
                        "barrier_pct",
                    ]:
                        if feat_name in feats:
                            feat_df = feats[feat_name]
                            if symbol in feat_df.columns:
                                vals = feat_df[symbol].dropna()
                                if not vals.empty:
                                    features_log[feat_name] = vals.iloc[-1]
                    live_barrier_pct = _resolve_live_barrier_pct(
                        symbol,
                        features_log,
                        panel=panel,
                        cfg=runtime_config,
                    )
                    if live_barrier_pct is not None:
                        features_log["barrier_pct"] = live_barrier_pct
                    execution_snapshot = {}
                    execution_snapshot.update(signal_close_snapshot)
                    execution_kwargs: Dict[str, Any] = {}
                    execution_limit_price = None
                    timing_snapshot = _entry_timing_snapshot(
                        decision=decision,
                        now=now_utc,
                        signal_bar_ts=signal_bar_ts,
                        max_signal_close_age_seconds=max_signal_close_to_entry_seconds,
                    )
                    decision.update(
                        {
                            key: value
                            for key, value in timing_snapshot.items()
                            if key
                            in {
                                "signal_bar_close_ts",
                                "signal_close_to_decision_seconds",
                                "signal_to_decision_seconds",
                                "max_signal_close_to_entry_seconds",
                                "stale_signal_age_gate_enabled",
                                "stale_signal_age_gate_exceeded",
                            }
                        }
                    )
                    execution_snapshot.update(timing_snapshot)
                    if bool(signal_close_snapshot.get("raw_signal_close_unreliable")):
                        reject_reason = (
                            "unreliable_raw_signal_close:"
                            f"{signal_close_snapshot.get('raw_signal_close_unreliable_reason')}"
                        )
                        if (
                            prediction_ledger is not None
                            and _should_log_prediction_candidate(
                                decision, policy=portfolio_policy
                            )
                        ):
                            prediction_ledger_rows.append(
                                _prediction_ledger_row(
                                    decision,
                                    timestamp=now_utc.isoformat(),
                                    side=side,
                                    portfolio_decision="data_quality_rejected",
                                    portfolio_reject_reason=reject_reason,
                                    execution_snapshot=execution_snapshot,
                                )
                            )
                        tprint(
                            f"Raw signal-close quality block: {symbol} "
                            f"{side}/{strategy_id} reason={reject_reason} "
                            f"raw_close={signal_close_snapshot.get('raw_signal_close')} "
                            f"volume={signal_close_snapshot.get('raw_signal_volume')} "
                            f"reference={signal_close_snapshot.get('raw_signal_close_reference_source')} "
                            f"reference_gap_bps="
                            f"{_safe_float(signal_close_snapshot.get('raw_signal_close_reference_gap_bps'), np.nan):.2f}"
                        )
                        side_metrics["non_fatal_issues"] += 1
                        continue
                    if bool(timing_snapshot.get("stale_signal_age_gate_exceeded")):
                        reject_reason = "stale_signal_age_exceeded"
                        if (
                            prediction_ledger is not None
                            and _should_log_prediction_candidate(
                                decision, policy=portfolio_policy
                            )
                        ):
                            prediction_ledger_rows.append(
                                _prediction_ledger_row(
                                    decision,
                                    timestamp=now_utc.isoformat(),
                                    side=side,
                                    portfolio_decision="stale_signal_rejected",
                                    portfolio_reject_reason=reject_reason,
                                    execution_snapshot=execution_snapshot,
                                )
                            )
                        tprint(
                            f"Stale-signal age block: {symbol} {side}/{strategy_id} "
                            f"signal_bar_close_ts={timing_snapshot.get('signal_bar_close_ts')} "
                            f"age={_safe_float(timing_snapshot.get('signal_close_to_decision_seconds'), np.nan):.0f}s "
                            f"limit={max_signal_close_to_entry_seconds:.0f}s"
                        )
                        side_metrics["non_fatal_issues"] += 1
                        continue
                    api_symbol = (
                        _live_exchange_symbol(executor.exchange, runtime_config, symbol)
                        if getattr(executor, "exchange", None) is not None
                        else symbol
                    )
                    if stale_entry_context and (
                        getattr(executor, "exchange", None) is None or price is None
                    ):
                        reject_reason = "stale_entry_requires_ticker_and_signal_price"
                        if (
                            prediction_ledger is not None
                            and _should_log_prediction_candidate(
                                decision, policy=portfolio_policy
                            )
                        ):
                            prediction_ledger_rows.append(
                                _prediction_ledger_row(
                                    decision,
                                    timestamp=now_utc.isoformat(),
                                    side=side,
                                    portfolio_decision="price_gap_rejected",
                                    portfolio_reject_reason=reject_reason,
                                    execution_snapshot={
                                        "stale_entry_context": True,
                                    },
                                )
                            )
                        tprint(
                            f"Stale-entry block: {symbol} {side}/{strategy_id} "
                            f"reason={reject_reason}"
                        )
                        side_metrics["non_fatal_issues"] += 1
                        continue
                    if (
                        getattr(executor, "exchange", None) is not None
                        and price is not None
                        and (
                            stale_entry_context
                            or portfolio_policy.ticker_precheck_enabled
                            or portfolio_policy.orderbook_precheck_enabled
                            or adverse_hourly_close_gate_enabled
                        )
                    ):
                        try:
                            execution_precheck_ts = pd.Timestamp.now(tz="UTC")
                            ticker_snapshot = fetch_ticker_snapshot(
                                exchange=executor.exchange,
                                symbol=api_symbol,
                                side=side,
                                policy=portfolio_policy,
                                mode=str(getattr(executor, "mode", "")),
                                now=execution_precheck_ts,
                            )
                            execution_snapshot.update(ticker_snapshot.to_dict())
                            if ticker_snapshot.hard_reject:
                                if (
                                    prediction_ledger is not None
                                    and _should_log_prediction_candidate(
                                        decision, policy=portfolio_policy
                                    )
                                ):
                                    prediction_ledger_rows.append(
                                        _prediction_ledger_row(
                                            decision,
                                            timestamp=now_utc.isoformat(),
                                            side=side,
                                            portfolio_decision="liquidity_rejected",
                                            liquidity_reject_reason=str(
                                                ticker_snapshot.reject_reason
                                                or "ticker_rejected"
                                            ),
                                            execution_snapshot=execution_snapshot,
                                        )
                                    )
                                tprint(
                                    f"Liquidity ticker block: {symbol} {side}/{strategy_id} "
                                    f"reason={ticker_snapshot.reject_reason}"
                                )
                                side_metrics["non_fatal_issues"] += 1
                                continue
                            decision_mid = float(ticker_snapshot.mid or 0.0)
                            if stale_entry_context:
                                max_abs_gap_bps = float(
                                    stale_entry_max_abs_signal_gap_bps
                                    if stale_entry_max_abs_signal_gap_bps is not None
                                    else 0.0
                                )
                                stale_abs_gap_bps = (
                                    abs(decision_mid / max(float(price), 1e-12) - 1.0)
                                    * 10000.0
                                )
                                execution_snapshot["stale_entry_context"] = True
                                execution_snapshot["stale_entry_abs_signal_gap_bps"] = (
                                    float(stale_abs_gap_bps)
                                )
                                execution_snapshot[
                                    "stale_entry_max_abs_signal_gap_bps"
                                ] = float(max_abs_gap_bps)
                                if (
                                    not np.isfinite(stale_abs_gap_bps)
                                    or stale_abs_gap_bps > max_abs_gap_bps
                                ):
                                    reject_reason = "stale_entry_price_moved_too_far"
                                    if (
                                        prediction_ledger is not None
                                        and _should_log_prediction_candidate(
                                            decision, policy=portfolio_policy
                                        )
                                    ):
                                        prediction_ledger_rows.append(
                                            _prediction_ledger_row(
                                                decision,
                                                timestamp=now_utc.isoformat(),
                                                side=side,
                                                portfolio_decision=(
                                                    "price_gap_rejected"
                                                ),
                                                portfolio_reject_reason=reject_reason,
                                                execution_snapshot=execution_snapshot,
                                            )
                                        )
                                    tprint(
                                        f"Stale-entry price block: {symbol} "
                                        f"{side}/{strategy_id} "
                                        f"abs_signal_gap_bps={stale_abs_gap_bps:.2f} "
                                        f"max_abs_signal_gap_bps={max_abs_gap_bps:.2f}"
                                    )
                                    side_metrics["non_fatal_issues"] += 1
                                    continue
                            gap_penalty, gap_info = compute_price_gap_rank_penalty(
                                strategy_id=strategy_id,
                                side=side,
                                signal_price=float(price),
                                decision_mid=decision_mid,
                                policy=portfolio_policy,
                            )
                            adjusted_rank = max(rank_for_size - float(gap_penalty), 0.0)
                            execution_snapshot.update(gap_info)
                            adverse_gap_bps = _adverse_signal_gap_bps(
                                side=side,
                                signal_price=price,
                                decision_mid=decision_mid,
                            )
                            execution_snapshot["adverse_signal_gap_bps"] = float(
                                adverse_gap_bps
                            )
                            execution_snapshot["adverse_hourly_close_gap_bps"] = float(
                                adverse_gap_bps
                            )
                            execution_snapshot[
                                "adverse_hourly_close_gate_bps"
                            ] = float(adverse_hourly_close_gate_bps)
                            if (
                                adverse_hourly_close_gate_enabled
                                and (
                                    not np.isfinite(adverse_gap_bps)
                                    or adverse_gap_bps >= adverse_hourly_close_gate_bps
                                )
                            ):
                                reject_reason = "adverse_hourly_close_gap_too_large"
                                if (
                                    prediction_ledger is not None
                                    and _should_log_prediction_candidate(
                                        decision, policy=portfolio_policy
                                    )
                                ):
                                    prediction_ledger_rows.append(
                                        _prediction_ledger_row(
                                            decision,
                                            timestamp=now_utc.isoformat(),
                                            side=side,
                                            portfolio_decision="price_gap_rejected",
                                            portfolio_reject_reason=reject_reason,
                                            execution_snapshot=execution_snapshot,
                                        )
                                    )
                                tprint(
                                    f"Adverse hourly-close block: {symbol} "
                                    f"{side}/{strategy_id} "
                                    f"adverse_gap_bps={adverse_gap_bps:.2f} "
                                    f"gate_bps={adverse_hourly_close_gate_bps:.2f} "
                                    f"hourly_close={float(price):.10g} "
                                    f"decision_mid={decision_mid:.10g}"
                                )
                                side_metrics["non_fatal_issues"] += 1
                                continue
                            chain_results["initial_calibrated_score"] = decision.get(
                                "calibrated_score"
                            )
                            execution_snapshot["price_gap_penalty"] = float(gap_penalty)
                            execution_snapshot["adjusted_rank_score"] = float(
                                adjusted_rank
                            )
                            if adjusted_rank < float(decision["effective_threshold"]):
                                if (
                                    prediction_ledger is not None
                                    and _should_log_prediction_candidate(
                                        decision, policy=portfolio_policy
                                    )
                                ):
                                    prediction_ledger_rows.append(
                                        _prediction_ledger_row(
                                            decision,
                                            timestamp=now_utc.isoformat(),
                                            side=side,
                                            portfolio_decision="price_gap_rejected",
                                            portfolio_reject_reason=(
                                                "rank_below_dynamic_threshold_after_price_gap"
                                            ),
                                            execution_snapshot=execution_snapshot,
                                        )
                                    )
                                tprint(
                                    f"Price-gap rank block: {symbol} {side}/{strategy_id} "
                                    f"adjusted_rank={adjusted_rank:.6f} "
                                    f"threshold={float(decision['effective_threshold']):.6f}"
                                )
                                side_metrics["non_fatal_issues"] += 1
                                continue
                            spread_baseline_bps, spread_baseline_source = (
                                _live_ev_haircut_spread_baseline_bps(
                                    symbol=symbol,
                                    data_root=str(
                                        runtime_config.get("data_root", "data")
                                    ),
                                    fallback_bps=(
                                        portfolio_policy.ev_haircut_expected_spread_bps
                                    ),
                                )
                            )
                            if not portfolio_policy.orderbook_precheck_enabled:
                                ev_adjusted = _ev_adjusted_prediction_after_entry_friction(
                                    calibrated_score=decision.get("calibrated_score"),
                                    strategy_id=strategy_id,
                                    side=side,
                                    calibration=strategy_ev_calibration,
                                    live_entry_friction_bps=float(adverse_gap_bps),
                                    observed_spread_bps=getattr(
                                        ticker_snapshot, "spread_bps", None
                                    ),
                                    orderbook_slippage_bps=0.0,
                                    adverse_signal_gap_bps=float(adverse_gap_bps),
                                    spread_baseline_bps=spread_baseline_bps,
                                    spread_baseline_source=spread_baseline_source,
                                    delay_slippage_baseline_bps=(
                                        portfolio_policy.ev_haircut_delay_slippage_baseline_bps
                                    ),
                                    policy_rank_reference_store=policy_rank_reference_store,
                                )
                                chain_results.update(ev_adjusted)
                                execution_snapshot.update(ev_adjusted)
                                chain_results["adjusted_calibrated_score"] = (
                                    ev_adjusted.get("ev_adjusted_calibrated_score")
                                )
                                execution_snapshot["adjusted_calibrated_score"] = (
                                    ev_adjusted.get("ev_adjusted_calibrated_score")
                                )
                                ev_rank = _safe_float(
                                    ev_adjusted.get("ev_adjusted_rank_score"), np.nan
                                )
                                if np.isfinite(ev_rank):
                                    adjusted_rank = min(float(adjusted_rank), float(ev_rank))
                                    execution_snapshot["adjusted_rank_score"] = float(
                                        adjusted_rank
                                    )
                                    chain_results["threshold_rank_score_after_friction_ev"] = (
                                        float(adjusted_rank)
                                    )
                                    if adjusted_rank < threshold_for_size:
                                        side_metrics["non_fatal_issues"] += 1
                                        continue
                            if portfolio_policy.orderbook_precheck_enabled:
                                book_snapshot = evaluate_orderbook_liquidity(
                                    exchange=executor.exchange,
                                    symbol=api_symbol,
                                    side=side,
                                    intended_quote_size=float(size),
                                    ticker_snapshot=ticker_snapshot,
                                    policy=portfolio_policy,
                                    mode=str(getattr(executor, "mode", "")),
                                )
                                execution_snapshot.update(book_snapshot.to_dict())
                                perps_capacity_cap = (
                                    _is_perps_config(runtime_config)
                                    and book_snapshot.reject_reason
                                    == "liquidity_capacity_weight_below_min"
                                    and _safe_float(
                                        book_snapshot.orderbook_capacity_quote_within_slippage,
                                        0.0,
                                    )
                                    > 0.0
                                )
                                if book_snapshot.hard_reject and not perps_capacity_cap:
                                    if (
                                        prediction_ledger is not None
                                        and _should_log_prediction_candidate(
                                            decision, policy=portfolio_policy
                                        )
                                    ):
                                        prediction_ledger_rows.append(
                                            _prediction_ledger_row(
                                                decision,
                                                timestamp=now_utc.isoformat(),
                                                side=side,
                                                portfolio_decision="liquidity_rejected",
                                                liquidity_reject_reason=str(
                                                    book_snapshot.reject_reason
                                                    or "orderbook_rejected"
                                                ),
                                                execution_snapshot=execution_snapshot,
                                            )
                                        )
                                    tprint(
                                        f"Liquidity orderbook block: {symbol} {side}/{strategy_id} "
                                        f"reason={book_snapshot.reject_reason}"
                                    )
                                    side_metrics["non_fatal_issues"] += 1
                                    continue
                                live_entry_friction_bps = _safe_float(
                                    book_snapshot.expected_total_entry_friction_bps,
                                    0.0,
                                ) + float(adverse_gap_bps)
                                ev_adjusted = (
                                    _ev_adjusted_prediction_after_entry_friction(
                                        calibrated_score=decision.get(
                                            "calibrated_score"
                                        ),
                                        strategy_id=strategy_id,
                                        side=side,
                                        calibration=strategy_ev_calibration,
                                        live_entry_friction_bps=live_entry_friction_bps,
                                        observed_spread_bps=book_snapshot.spread_bps,
                                        orderbook_slippage_bps=(
                                            book_snapshot.expected_fill_slippage_bps
                                        ),
                                        adverse_signal_gap_bps=float(adverse_gap_bps),
                                        spread_baseline_bps=spread_baseline_bps,
                                        spread_baseline_source=spread_baseline_source,
                                        delay_slippage_baseline_bps=(
                                            portfolio_policy.ev_haircut_delay_slippage_baseline_bps
                                        ),
                                        policy_rank_reference_store=(
                                            policy_rank_reference_store
                                        ),
                                    )
                                )
                                chain_results.update(ev_adjusted)
                                execution_snapshot.update(ev_adjusted)
                                chain_results["adjusted_calibrated_score"] = (
                                    ev_adjusted.get("ev_adjusted_calibrated_score")
                                )
                                execution_snapshot["adjusted_calibrated_score"] = (
                                    ev_adjusted.get("ev_adjusted_calibrated_score")
                                )
                                ev_rank = _safe_float(
                                    ev_adjusted.get("ev_adjusted_rank_score"), np.nan
                                )
                                if np.isfinite(ev_rank):
                                    adjusted_rank = min(
                                        float(adjusted_rank), float(ev_rank)
                                    )
                                    execution_snapshot["adjusted_rank_score"] = float(
                                        adjusted_rank
                                    )
                                    chain_results[
                                        "threshold_rank_score_after_friction_ev"
                                    ] = float(adjusted_rank)
                                    if adjusted_rank < float(
                                        decision["effective_threshold"]
                                    ):
                                        if (
                                            prediction_ledger is not None
                                            and _should_log_prediction_candidate(
                                                decision, policy=portfolio_policy
                                            )
                                        ):
                                            prediction_ledger_rows.append(
                                                _prediction_ledger_row(
                                                    decision,
                                                    timestamp=now_utc.isoformat(),
                                                    side=side,
                                                    portfolio_decision=(
                                                        "liquidity_rejected"
                                                    ),
                                                    liquidity_reject_reason=(
                                                        "rank_below_dynamic_threshold_after_live_friction_ev"
                                                    ),
                                                    execution_snapshot=(
                                                        execution_snapshot
                                                    ),
                                                )
                                            )
                                        tprint(
                                            f"Live-friction EV block: {symbol} "
                                            f"{side}/{strategy_id} "
                                            f"adjusted_rank={adjusted_rank:.6f} "
                                            f"threshold={float(decision['effective_threshold']):.6f} "
                                            f"entry_friction_bps={live_entry_friction_bps:.2f}"
                                        )
                                        side_metrics["non_fatal_issues"] += 1
                                        continue
                                if portfolio_mgr is not None:
                                    capacity = portfolio_mgr.get_portfolio_capacity(
                                        side=side,
                                        strategy_id=strategy_id,
                                    )
                                    _attach_portfolio_replay_state_for_ledger(
                                        decision,
                                        portfolio_mgr=portfolio_mgr,
                                        capacity=capacity,
                                        now_utc=now_utc,
                                    )
                                    perp_rank = (
                                        _perp_rank_context(
                                            data_root=runtime_config["data_root"],
                                            run_id=runtime_config["run_id"],
                                            side=side,
                                            strategy_id=strategy_id,
                                            score=float(
                                                chain_results.get("meta_pred")
                                                or decision.get("calibrated_score")
                                                or adjusted_rank
                                            ),
                                        )
                                        if _is_perps_config(runtime_config)
                                        else {}
                                    )
                                    sizing_audit = compute_rank_based_position_size(
                                        wallet_value=float(capacity["wallet_value"]),
                                        open_notional=float(capacity["open_notional"]),
                                        adjusted_rank_score=float(adjusted_rank),
                                        final_threshold=float(
                                            decision["effective_threshold"]
                                        ),
                                        policy=portfolio_policy,
                                        liquidity_capacity_weight=float(
                                            book_snapshot.liquidity_capacity_weight
                                        ),
                                        live_test_mode=live_test_mode,
                                        rank_size_power=float(
                                            chain_results.get("size_power", 1.1)
                                        ),
                                        total_assets_quote=capacity.get(
                                            "total_assets_quote"
                                        ),
                                        total_liabilities_quote=capacity.get(
                                            "total_liabilities_quote"
                                        ),
                                        open_positions=capacity.get("open_positions"),
                                        market_mode=runtime_config.get(
                                            "market_mode", "spot"
                                        ),
                                        available_wallet_value=capacity.get(
                                            "available_wallet_quote"
                                        ),
                                        stop_loss_pct=live_barrier_pct,
                                        rank_number=perp_rank.get("rank_number"),
                                        rank_x=perp_rank.get("rank_x"),
                                        orderbook_capacity_quote=book_snapshot.orderbook_capacity_quote_within_slippage,
                                    )
                                    size = float(sizing_audit["size_after_liquidity"])
                                    chain_results["portfolio_rank_sizing"] = (
                                        sizing_audit
                                    )
                                    can_enter, info = portfolio_mgr.can_enter_position(
                                        symbol=symbol,
                                        side=side,
                                        strategy_id=strategy_id,
                                        rank_score=float(adjusted_rank),
                                        initial_threshold=float(
                                            decision["effective_threshold"]
                                        ),
                                        current_time=now_utc,
                                        requested_position_size=size,
                                    )
                                    chain_results["portfolio_gate_after_liquidity"] = (
                                        info
                                    )
                                    if not can_enter:
                                        if (
                                            prediction_ledger is not None
                                            and _should_log_prediction_candidate(
                                                decision, policy=portfolio_policy
                                            )
                                        ):
                                            prediction_ledger_rows.append(
                                                _prediction_ledger_row(
                                                    decision,
                                                    timestamp=now_utc.isoformat(),
                                                    side=side,
                                                    portfolio_decision=(
                                                        "portfolio_rejected"
                                                    ),
                                                    portfolio_reject_reason=str(
                                                        info.get("reason")
                                                        or "post_liquidity_portfolio_rejected"
                                                    ),
                                                    execution_snapshot=execution_snapshot,
                                                )
                                            )
                                        tprint(
                                            f"Portfolio post-liquidity block: {symbol} "
                                            f"{side}/{strategy_id} reason={info.get('reason')}"
                                        )
                                        side_metrics["non_fatal_issues"] += 1
                                        continue
                                execution_snapshot["liquidity_capacity_weight"] = float(
                                    book_snapshot.liquidity_capacity_weight
                                )
                                execution_snapshot["expected_entry_price"] = (
                                    book_snapshot.expected_fill_price
                                )
                                execution_snapshot["expected_fill_slippage_bps"] = (
                                    book_snapshot.expected_fill_slippage_bps
                                )
                            if decision_mid > 0:
                                execution_limit_price = marketable_limit_price(
                                    side=side,
                                    decision_mid=decision_mid,
                                    policy=portfolio_policy,
                                )
                                execution_snapshot["max_chase_bps"] = (
                                    portfolio_policy.max_order_chase_bps
                                )
                                execution_snapshot["entry_limit_price"] = (
                                    execution_limit_price
                                )
                        except Exception as exc:
                            if (
                                prediction_ledger is not None
                                and _should_log_prediction_candidate(
                                    decision, policy=portfolio_policy
                                )
                            ):
                                prediction_ledger_rows.append(
                                    _prediction_ledger_row(
                                        decision,
                                        timestamp=now_utc.isoformat(),
                                        side=side,
                                        portfolio_decision="liquidity_rejected",
                                        liquidity_reject_reason=(
                                            f"{classify_api_error(exc)}: {exc}"
                                        ),
                                        execution_snapshot=execution_snapshot,
                                    )
                                )
                            tprint(
                                f"Liquidity precheck block: {symbol} {side}/{strategy_id} "
                                f"error={classify_api_error(exc)}: {exc}"
                            )
                            side_metrics["non_fatal_issues"] += 1
                            continue
                    execution_price = (
                        float(execution_limit_price)
                        if execution_limit_price is not None
                        else float(chain_results.get("entry_px") or price)
                    )
                    if execution_snapshot:
                        policy_reference_price = (
                            float(price) if price is not None else None
                        )
                        sizing_for_log = (
                            chain_results.get("portfolio_rank_sizing", {}) or {}
                        )
                        execution_snapshot["wallet_value_at_entry"] = (
                            sizing_for_log.get("wallet_value")
                        )
                        execution_snapshot["open_notional_at_entry"] = (
                            sizing_for_log.get("open_notional")
                        )
                        execution_snapshot["leverage_wallet_multiplier"] = (
                            sizing_for_log.get("leverage_wallet_multiplier")
                        )
                        execution_snapshot["book_notional_multiplier"] = (
                            sizing_for_log.get("book_notional_multiplier")
                        )
                        execution_snapshot["safe_book_notional"] = sizing_for_log.get(
                            "safe_book_notional"
                        )
                        execution_snapshot["target_slot_notional"] = sizing_for_log.get(
                            "target_slot_notional"
                        )
                        execution_snapshot["slot_cap_notional"] = sizing_for_log.get(
                            "slot_cap_notional"
                        )
                        execution_snapshot["rank_slot_fraction"] = sizing_for_log.get(
                            "rank_slot_fraction"
                        )
                        execution_snapshot["current_margin_level"] = sizing_for_log.get(
                            "current_margin_level"
                        )
                        execution_snapshot["signal_price"] = policy_reference_price
                        if policy_reference_price is not None:
                            execution_snapshot["theoretical_entry_price"] = (
                                policy_reference_price
                            )
                            execution_snapshot["policy_entry_price"] = (
                                policy_reference_price
                            )
                            execution_snapshot["policy_entry_price_source"] = (
                                "signal_bar_close"
                            )
                        execution_snapshot["final_threshold"] = float(
                            decision["effective_threshold"]
                        )
                        execution_snapshot["position_size_before_liquidity"] = (
                            sizing_for_log.get("size_before_liquidity")
                        )
                        execution_snapshot["position_size_after_liquidity"] = size
                        execution_snapshot["market_mode"] = runtime_config.get(
                            "market_mode", "spot"
                        )
                        for perp_key in (
                            "perp_rank_number",
                            "perp_rank_x",
                            "perp_rank_leverage",
                            "perp_risk_cap_leverage",
                            "perp_effective_leverage",
                            "perp_stop_loss_pct",
                            "perp_full_wallet",
                            "perp_available_wallet",
                        ):
                            if perp_key in sizing_for_log:
                                execution_snapshot[perp_key] = sizing_for_log.get(
                                    perp_key
                                )
                        execution_snapshot["max_chase_bps"] = (
                            portfolio_policy.max_order_chase_bps
                        )
                        execution_snapshot["entry_limit_price"] = execution_limit_price
                        execution_snapshot.setdefault(
                            "ticker_bid", execution_snapshot.get("bid")
                        )
                        execution_snapshot.setdefault(
                            "ticker_ask", execution_snapshot.get("ask")
                        )
                        execution_snapshot.setdefault(
                            "ticker_mid", execution_snapshot.get("mid")
                        )
                        execution_snapshot.setdefault(
                            "ticker_spread_bps", execution_snapshot.get("spread_bps")
                        )
                        execution_snapshot.setdefault(
                            "expected_fill_price",
                            execution_snapshot.get("expected_entry_price")
                            or execution_snapshot.get("expected_fill_price"),
                        )
                        execution_snapshot.setdefault(
                            "decision_mid", execution_snapshot.get("mid")
                        )
                        features_log.update(
                            {
                                key: value
                                for key, value in execution_snapshot.items()
                                if key
                                in {
                                    "signal_price",
                                    "theoretical_entry_price",
                                    "policy_entry_price",
                                    "policy_entry_price_source",
                                    "decision_mid",
                                    "signal_bar_close_ts",
                                    "signal_close_to_decision_seconds",
                                    "signal_to_decision_seconds",
                                    "max_signal_close_to_entry_seconds",
                                    "stale_signal_age_gate_enabled",
                                    "stale_signal_age_gate_exceeded",
                                    "spread_bps",
                                    "ticker_bid",
                                    "ticker_ask",
                                    "ticker_mid",
                                    "ticker_spread_bps",
                                    "expected_fill_price",
                                    "liquidity_capacity_weight",
                                    "expected_fill_slippage_bps",
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
                                    "signal_gap_bps",
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
                                    "slot_cap_notional",
                                    "rank_slot_fraction",
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
                                }
                            }
                        )
                    trade_audit = _build_trade_start_audit(
                        orchestrator=orchestrator,
                        panel=panel,
                        feats=feats,
                        candidate_features=candidate_features,
                        meta_model_input_features=decision.get(
                            "_meta_model_input_features"
                        ),
                        symbol=symbol,
                        side=side,
                        strategy_id=strategy_id,
                        signal_bar_ts=signal_bar_ts,
                        decision=decision,
                        chain_results=chain_results,
                        execution_snapshot=execution_snapshot,
                        parity_contract=runtime_config.get(
                            "training_live_parity_contract"
                        ),
                    )
                    features_log.update(trade_audit)
                    exchange_min_notional = _exchange_min_notional_for_symbol(
                        getattr(executor, "exchange", None),
                        api_symbol,
                    )
                    if (
                        exchange_min_notional is not None
                        and np.isfinite(size)
                        and float(size) < float(exchange_min_notional)
                    ):
                        reject_reason = "below_exchange_min_notional"
                        reject_error = (
                            f"final quote size {float(size):.8f} is below "
                            f"{symbol} exchange min notional "
                            f"{float(exchange_min_notional):.8f}"
                        )
                        execution_snapshot["exchange_min_notional"] = float(
                            exchange_min_notional
                        )
                        execution_snapshot["position_size_after_liquidity"] = float(
                            size
                        )
                        features_log["exchange_min_notional"] = float(
                            exchange_min_notional
                        )
                        logger.log_entry(
                            symbol=symbol,
                            side=side,
                            size=abs(size),
                            price=price,
                            predictions=predictions,
                            features=features_log,
                            mode=executor.mode,
                            strategy_id=strategy_id,
                            calibrated_score=float(decision["calibrated_score"]),
                            rank_threshold=float(decision["rank_threshold"]),
                            order_error_category=reject_reason,
                            lifecycle_event="entry_rejected",
                            status="failed",
                            error=reject_error,
                        )
                        if (
                            prediction_ledger is not None
                            and _should_log_prediction_candidate(
                                decision, policy=portfolio_policy
                            )
                        ):
                            prediction_ledger_rows.append(
                                _prediction_ledger_row(
                                    decision,
                                    timestamp=now_utc.isoformat(),
                                    side=side,
                                    portfolio_decision="exchange_filter_rejected",
                                    portfolio_reject_reason=reject_reason,
                                    execution_snapshot=execution_snapshot,
                                    was_traded=False,
                                )
                            )
                        tprint(
                            f"Exchange filter block: {symbol} {side}/{strategy_id} "
                            f"reason={reject_reason} size={float(size):.8f} "
                            f"min_notional={float(exchange_min_notional):.8f}"
                        )
                        side_metrics["non_fatal_issues"] += 1
                        continue
                    sizing_context = chain_results.get("portfolio_rank_sizing", {}) or {}
                    perp_sizing_context = {
                        key: sizing_context.get(key)
                        for key in (
                            "leverage_wallet_multiplier",
                            "book_notional_multiplier",
                            "perp_rank_number",
                            "perp_rank_x",
                            "perp_rank_leverage",
                            "perp_risk_cap_leverage",
                            "perp_effective_leverage",
                            "perp_stop_loss_pct",
                            "perp_full_wallet",
                            "perp_available_wallet",
                            "orderbook_capacity_quote_within_slippage",
                        )
                        if key in sizing_context
                    }
                    trade_result = _execute_trade_with_optional_context(
                        executor,
                        symbol=symbol,
                        side=side,
                        size=abs(size),
                        price=execution_price,
                        bucket_key=bucket_key,
                        ohlcv_reference_price=(
                            float(price) if price is not None else None
                        ),
                        trade_context={
                            "base_pred": chain_results.get("base_pred"),
                            "base_rank_pct": chain_results.get("base_rank_pct"),
                            "base_train_rank_pct": chain_results.get(
                                "base_train_rank_pct"
                            ),
                            "base_gate_top_frac": chain_results.get(
                                "base_gate_top_frac"
                            ),
                            "meta_pred": chain_results.get("meta_pred"),
                            "estimated_hit_rate": chain_results.get(
                                "estimated_hit_rate"
                            ),
                            "estimated_hit_rate_source": chain_results.get(
                                "estimated_hit_rate_source"
                            ),
                            "estimated_hit_rate_calibration_n": chain_results.get(
                                "estimated_hit_rate_calibration_n"
                            ),
                            "estimated_ev_gross_return": chain_results.get(
                                "estimated_ev_gross_return"
                            ),
                            "estimated_ev_net_return": chain_results.get(
                                "estimated_ev_net_return"
                            ),
                            "estimated_ev_cost_bps": chain_results.get(
                                "estimated_ev_cost_bps"
                            ),
                            "estimated_ev_hit_rate": chain_results.get(
                                "estimated_ev_hit_rate"
                            ),
                            "estimated_ev_source": chain_results.get(
                                "estimated_ev_source"
                            ),
                            "estimated_ev_calibration_n": chain_results.get(
                                "estimated_ev_calibration_n"
                            ),
                            "meta_train_rank_pct": chain_results.get(
                                "meta_train_rank_pct"
                            ),
                            "rank_score_source": chain_results.get("rank_score_source"),
                            "policy_rank_pct": chain_results.get("policy_rank_pct"),
                            "policy_rank_reference_n": chain_results.get(
                                "policy_rank_reference_n"
                            ),
                            "policy_rank_reference_source": chain_results.get(
                                "policy_rank_reference_source"
                            ),
                            "ev_haircut_bps": chain_results.get("ev_haircut_bps"),
                            "ev_haircut_raw_live_entry_friction_bps": chain_results.get(
                                "ev_haircut_raw_live_entry_friction_bps"
                            ),
                            "ev_haircut_observed_spread_bps": chain_results.get(
                                "ev_haircut_observed_spread_bps"
                            ),
                            "ev_haircut_observed_half_spread_bps": chain_results.get(
                                "ev_haircut_observed_half_spread_bps"
                            ),
                            "ev_haircut_spread_baseline_bps": chain_results.get(
                                "ev_haircut_spread_baseline_bps"
                            ),
                            "ev_haircut_spread_baseline_source": chain_results.get(
                                "ev_haircut_spread_baseline_source"
                            ),
                            "ev_haircut_half_spread_baseline_bps": chain_results.get(
                                "ev_haircut_half_spread_baseline_bps"
                            ),
                            "ev_haircut_spread_excess_bps": chain_results.get(
                                "ev_haircut_spread_excess_bps"
                            ),
                            "ev_haircut_orderbook_slippage_bps": chain_results.get(
                                "ev_haircut_orderbook_slippage_bps"
                            ),
                            "ev_haircut_adverse_signal_gap_bps": chain_results.get(
                                "ev_haircut_adverse_signal_gap_bps"
                            ),
                            "ev_haircut_observed_delay_slippage_bps": chain_results.get(
                                "ev_haircut_observed_delay_slippage_bps"
                            ),
                            "ev_haircut_delay_slippage_baseline_bps": chain_results.get(
                                "ev_haircut_delay_slippage_baseline_bps"
                            ),
                            "ev_haircut_delay_slippage_excess_bps": chain_results.get(
                                "ev_haircut_delay_slippage_excess_bps"
                            ),
                            "ev_haircut_contract": chain_results.get(
                                "ev_haircut_contract"
                            ),
                            "ev_adjusted_entry_friction_bps": chain_results.get(
                                "ev_adjusted_entry_friction_bps"
                            ),
                            "ev_adjusted_net_return_before_friction": chain_results.get(
                                "ev_adjusted_net_return_before_friction"
                            ),
                            "ev_adjusted_net_return_after_friction": chain_results.get(
                                "ev_adjusted_net_return_after_friction"
                            ),
                            "ev_adjusted_calibrated_score": chain_results.get(
                                "ev_adjusted_calibrated_score"
                            ),
                            "ev_adjusted_rank_score": chain_results.get(
                                "ev_adjusted_rank_score"
                            ),
                            "ev_adjusted_source": chain_results.get(
                                "ev_adjusted_source"
                            ),
                            "calibrated_score": decision.get("calibrated_score"),
                            "rank_percentile": chain_results.get(
                                "sizer_rank_percentile"
                            )
                            or decision.get("rank_percentile"),
                            "effective_threshold": chain_results.get(
                                "effective_threshold"
                            ),
                            "deployment_rank_threshold": decision.get(
                                "deployment_rank_threshold"
                            ),
                            "wallet_value_at_entry": (
                                chain_results.get("portfolio_rank_sizing", {}) or {}
                            ).get("wallet_value"),
                            "open_notional_at_entry": (
                                chain_results.get("portfolio_rank_sizing", {}) or {}
                            ).get("open_notional"),
                            "leverage_wallet_multiplier": (
                                chain_results.get("portfolio_rank_sizing", {}) or {}
                            ).get("leverage_wallet_multiplier"),
                            "book_notional_multiplier": (
                                chain_results.get("portfolio_rank_sizing", {}) or {}
                            ).get("book_notional_multiplier"),
                            "safe_book_notional": (
                                chain_results.get("portfolio_rank_sizing", {}) or {}
                            ).get("safe_book_notional"),
                            "target_slot_notional": (
                                chain_results.get("portfolio_rank_sizing", {}) or {}
                            ).get("target_slot_notional"),
                            "slot_cap_notional": (
                                chain_results.get("portfolio_rank_sizing", {}) or {}
                            ).get("slot_cap_notional"),
                            "rank_slot_fraction": (
                                chain_results.get("portfolio_rank_sizing", {}) or {}
                            ).get("rank_slot_fraction"),
                            "current_margin_level": (
                                chain_results.get("portfolio_rank_sizing", {}) or {}
                            ).get("current_margin_level"),
                            "barrier_pct": live_barrier_pct,
                            "barrier_frac": live_barrier_pct,
                            **trade_audit,
                            "decision_ts": decision.get("decision_ts"),
                            "signal_bar_ts": decision.get("signal_bar_ts"),
                            "signal_bar_close_ts": decision.get("signal_bar_close_ts"),
                            "signal_close_to_decision_seconds": decision.get(
                                "signal_close_to_decision_seconds"
                            ),
                            "signal_to_decision_seconds": decision.get(
                                "signal_to_decision_seconds"
                            ),
                            "max_signal_close_to_entry_seconds": decision.get(
                                "max_signal_close_to_entry_seconds"
                            ),
                            "signal_to_entry_alert_seconds": signal_to_entry_alert_seconds,
                            **perp_sizing_context,
                        },
                        execution_kwargs={
                            "execution_snapshot": execution_snapshot,
                            "signal_price": (
                                float(price) if price is not None else None
                            ),
                            "decision_mid": execution_snapshot.get("decision_mid"),
                            "expected_entry_price": execution_snapshot.get(
                                "expected_entry_price"
                            )
                            or execution_snapshot.get(
                                "expected_fill_price"
                            )
                            or execution_price,
                            "expected_fill_slippage_bps": execution_snapshot.get(
                                "expected_fill_slippage_bps"
                            ),
                            "max_chase_bps": portfolio_policy.max_order_chase_bps,
                            "rank_score": rank_for_size,
                            "adjusted_rank_score": execution_snapshot.get(
                                "adjusted_rank_score", rank_for_size
                            ),
                            "final_threshold": float(decision["effective_threshold"]),
                            "position_size_before_liquidity": (
                                chain_results.get("portfolio_rank_sizing", {}) or {}
                            ).get("size_before_liquidity"),
                            "position_size_after_liquidity": size,
                            "order_type": "limit" if execution_limit_price else None,
                            "limit_price": execution_limit_price,
                        },
                    )
                    _record_trade_execution_health(portfolio_mgr, trade_result)
                    trade_success = bool(
                        trade_result.get("success", False)
                        or trade_result.get("status") == "recorded"
                    )
                    order_error_category = str(
                        trade_result.get("error_category", "") or ""
                    )
                    if not trade_success:
                        expected_capacity_rejection = (
                            _is_expected_order_capacity_rejection(order_error_category)
                        )
                        if expected_capacity_rejection:
                            side_metrics["non_fatal_issues"] += 1
                        else:
                            side_metrics["order_errors"] += 1
                        if not order_error_category:
                            order_error_category = "unclassified_order_error"
                            side_metrics["unexplained_order_errors"] += 1
                        tprint(
                            (
                                "[ORDER_CAPACITY_REJECTION] "
                                if expected_capacity_rejection
                                else "[ORDER_ERROR] "
                            )
                            + f"{symbol} {side}/{strategy_id} "
                            f"category={order_error_category} "
                            f"error={trade_result.get('error', '')}"
                        )
                    if portfolio_mgr is not None and (trade_success):
                        portfolio_mgr.record_position_open(
                            symbol=symbol,
                            side=side,
                            strategy_id=strategy_id,
                            position_size=float(abs(size)),
                            entry_price=float(price if price is not None else 0.0),
                            entry_time=now_utc,
                        )
                    if trade_success:
                        tprint(
                            f"Trade entry accepted: {symbol} {side}/{strategy_id} "
                            f"estimated_hit_rate={_safe_float(chain_results.get('estimated_hit_rate')):.3f} "
                            f"estimated_net_ev={_safe_float(chain_results.get('estimated_ev_net_return')):.4f} "
                            f"estimated_gross_ev={_safe_float(chain_results.get('estimated_ev_gross_return')):.4f} "
                            f"estimated_cost_bps={_safe_float(chain_results.get('estimated_ev_cost_bps')):.1f} "
                            f"source={chain_results.get('estimated_hit_rate_source')}"
                        )
                        logger.log_entry(
                            symbol=symbol,
                            side=side,
                            size=abs(size),
                            price=trade_result.get("realized_entry_price", price),
                            predictions=predictions,
                            features=features_log,
                            mode=executor.mode,
                            strategy_id=strategy_id,
                            calibrated_score=float(decision["calibrated_score"]),
                            rank_threshold=float(decision["rank_threshold"]),
                            expected_entry_price=trade_result.get(
                                "expected_entry_price"
                            ),
                            realized_entry_price=trade_result.get(
                                "realized_entry_price"
                            ),
                            entry_order_type=trade_result.get("entry_order_type"),
                            price_slippage_pct=trade_result.get("price_slippage_pct"),
                            ohlcv_entry_price=trade_result.get("ohlcv_entry_price"),
                            entry_price_delta_vs_ohlcv=trade_result.get(
                                "entry_price_delta_vs_ohlcv"
                            ),
                            entry_price_delta_vs_ohlcv_pct=trade_result.get(
                                "entry_price_delta_vs_ohlcv_pct"
                            ),
                            entry_delay_effect_bps=trade_result.get(
                                "entry_delay_effect_bps"
                            ),
                            entry_delay_adverse_bps=trade_result.get(
                                "entry_delay_adverse_bps"
                            ),
                            entry_delay_abs_bps=trade_result.get(
                                "entry_delay_abs_bps"
                            ),
                            decision_to_entry_seconds=trade_result.get(
                                "decision_to_entry_seconds"
                            ),
                            signal_to_entry_seconds=trade_result.get(
                                "signal_to_entry_seconds"
                            ),
                            expected_friction_drag_bps=trade_result.get(
                                "expected_friction_drag_bps"
                            ),
                            spread_proxy_pct=trade_result.get("spread_proxy_pct"),
                            orderbook_snapshot=trade_result.get("orderbook_snapshot"),
                            stop_price=trade_result.get("stop_price"),
                            stop_order_id=trade_result.get("stop_order_id"),
                            exchange_order_id=_order_identifier(
                                trade_result.get("order")
                            ),
                            order_error_category=order_error_category,
                            actual_entry_price=trade_result.get("realized_entry_price"),
                            exit_reason=trade_result.get("exit_reason"),
                            net_pnl=trade_result.get("net_pnl"),
                            gross_pnl_pct=trade_result.get("gross_pnl_pct"),
                            net_pnl_pct=trade_result.get("net_pnl_pct"),
                            gross_pnl_amount=trade_result.get("gross_pnl_amount"),
                            net_pnl_amount=trade_result.get("net_pnl_amount"),
                            status="pending",
                            error=trade_result.get("error", ""),
                        )
                    else:
                        logger.log_entry(
                            symbol=symbol,
                            side=side,
                            size=abs(size),
                            price=price,
                            predictions=predictions,
                            features=features_log,
                            mode=executor.mode,
                            strategy_id=strategy_id,
                            calibrated_score=float(decision["calibrated_score"]),
                            rank_threshold=float(decision["rank_threshold"]),
                            order_error_category=order_error_category,
                            lifecycle_event="entry_rejected",
                            status="failed",
                            error=trade_result.get("error", ""),
                        )
                    if (
                        prediction_ledger is not None
                        and (
                            trade_success
                            or _should_log_prediction_candidate(
                                decision, policy=portfolio_policy
                            )
                        )
                    ):
                        prediction_ledger_rows.append(
                            _prediction_ledger_row(
                                decision,
                                timestamp=now_utc.isoformat(),
                                side=side,
                                portfolio_decision=(
                                    "traded" if trade_success else "order_rejected"
                                ),
                                portfolio_reject_reason=(
                                    None if trade_success else order_error_category
                                ),
                                execution_snapshot=execution_snapshot,
                                was_traded=trade_success,
                                trade_result=trade_result,
                            )
                        )
                    if trade_success:
                        side_metrics["executed"] += 1
                        total_entries_executed += 1
                        results["trades"].append(
                            {
                                "symbol": symbol,
                                "side": side,
                                "size": size,
                                "price": price,
                                "result": trade_result,
                                "strategy_id": strategy_id,
                                "calibrated_score": float(decision["calibrated_score"]),
                                "estimated_hit_rate": chain_results.get(
                                    "estimated_hit_rate"
                                ),
                                "estimated_ev_gross_return": chain_results.get(
                                    "estimated_ev_gross_return"
                                ),
                                "estimated_ev_net_return": chain_results.get(
                                    "estimated_ev_net_return"
                                ),
                                "estimated_ev_cost_bps": chain_results.get(
                                    "estimated_ev_cost_bps"
                                ),
                            }
                        )
            base_dist = _log_score_distribution(
                f"Base predictions [{side}]", base_preds_all
            )
            meta_dist = _log_score_distribution(
                f"Meta predictions [{side}]", meta_preds_all
            )
            rank_dist = _log_score_distribution(
                f"Norm-rank scores [{side}]", rank_pct_all
            )
            results["side_metrics"][side] = dict(side_metrics)
            results["score_distributions"][side] = {
                "base": base_dist,
                "meta": meta_dist,
                "norm_rank": rank_dist,
            }
            results["order_error_summary"]["by_side"][side] = {
                "order_errors": int(side_metrics["order_errors"]),
                "unexplained_order_errors": int(
                    side_metrics["unexplained_order_errors"]
                ),
            }
            results["order_error_summary"]["order_errors"] += int(
                side_metrics["order_errors"]
            )
            results["order_error_summary"]["unexplained_order_errors"] += int(
                side_metrics["unexplained_order_errors"]
            )
            tprint(
                f"Inference side metrics [{side}]: input={side_metrics['input_candidates']}, "
                f"eligible={side_metrics['eligible_candidates']}, "
                f"lgbm_strategy_mask_pass={side_metrics['lgbm_strategy_mask_pass']}, "
                f"lgbm_strategy_mask_block={side_metrics['lgbm_strategy_mask_block']}, "
                f"asset_exclusion_block={side_metrics['asset_exclusion_block']}, "
                f"base_gate_pass={side_metrics['base_gate_pass']}, "
                f"chain_enter={side_metrics['chain_enter']}, threshold_pass={side_metrics['threshold_pass']}, "
                f"cooldown_pass={side_metrics['cooldown_pass']}, portfolio_pass={side_metrics['portfolio_pass']}, "
                f"executed={side_metrics['executed']}, meta_missing={side_metrics['meta_missing']}, "
                f"order_errors={side_metrics['order_errors']}, "
                f"unexplained_order_errors={side_metrics['unexplained_order_errors']}, "
                f"non_fatal_issues={side_metrics['non_fatal_issues']}"
            )
            if side_metrics["unexplained_order_errors"] > 0:
                tprint(
                    f"CRITICAL: unexplained order placement errors for {side}: "
                    f"{side_metrics['unexplained_order_errors']}"
                )
            if (
                side_metrics["chain_enter"] > 0
                and side_metrics["meta_missing"] >= side_metrics["chain_enter"]
            ):
                tprint(
                    f"WARNING: systemic meta prediction failure for {side}: "
                    f"missing={side_metrics['meta_missing']} chain_enter={side_metrics['chain_enter']}"
                )
        except Exception as e:
            tprint(f"Error running inference chain for {side}: {e}")
            continue

    if global_auction_enabled and global_auction_decisions:
        global_entry_cap = max(0, int(max_entries_total))
        for row in global_auction_decisions:
            row["portfolio_priority"] = _candidate_portfolio_priority(row)
        global_auction_decisions.sort(
            key=lambda row: (
                _safe_float(row.get("portfolio_priority"), -float("inf")),
                _candidate_rank_score(row),
                _safe_float(row.get("calibrated_score"), -float("inf")),
                -_candidate_expected_friction(row),
            ),
            reverse=True,
        )
        tprint(
            "Global auction execution: "
            f"candidates={len(global_auction_decisions)} cap={global_entry_cap}"
        )

        def _log_global_auction_capacity_rejects(
            remaining_decisions: Sequence[Dict[str, Any]],
            *,
            stage: str,
            reason: str,
        ) -> None:
            if prediction_ledger is None:
                return
            for skipped_decision in remaining_decisions:
                if not _should_log_prediction_candidate(
                    skipped_decision, policy=portfolio_policy
                ):
                    continue
                skip_side = str(
                    skipped_decision.get("_auction_side")
                    or skipped_decision.get("side")
                    or ""
                )
                skip_chain = dict(skipped_decision.get("chain_results") or {})
                skip_chain["global_auction_skip_stage"] = stage
                skip_chain["global_auction_skip_reason"] = reason
                skipped_decision["chain_results"] = skip_chain
                prediction_ledger_rows.append(
                    _prediction_ledger_row(
                        skipped_decision,
                        timestamp=now_utc.isoformat(),
                        side=skip_side,
                        portfolio_decision="portfolio_rejected",
                        portfolio_reject_reason=f"global_auction_{stage}:{reason}",
                    )
                )

        entries_this_bar = 0
        for auction_i, decision in enumerate(global_auction_decisions):
            if total_entries_executed >= global_entry_cap:
                _log_global_auction_capacity_rejects(
                    global_auction_decisions[auction_i:],
                    stage="capacity",
                    reason="global_entry_cap_reached",
                )
                break
            if entries_this_bar >= int(portfolio_policy.max_new_entries_per_bar):
                _log_global_auction_capacity_rejects(
                    global_auction_decisions[auction_i:],
                    stage="capacity",
                    reason="max_new_entries_per_bar_reached",
                )
                break
            side = str(decision.get("_auction_side") or decision.get("side") or "")
            side_metrics = decision.get("_auction_side_metrics")
            if not isinstance(side_metrics, dict):
                side_metrics = {}

            def _commit_global_side_metrics() -> None:
                if side:
                    results["side_metrics"][side] = dict(side_metrics)

            symbol = str(decision["symbol"])
            strategy_id = str(decision["strategy_id"])
            chain_results = dict(decision.get("chain_results") or {})
            threshold_for_size = float(decision["effective_threshold"])
            rank_for_size = _safe_float(
                decision.get(
                    "normalized_rank_score",
                    decision.get("threshold_score", decision.get("calibrated_score")),
                )
            )
            barrier_features_log: Dict[str, Any] = {}
            if (
                isinstance(candidate_features, pd.DataFrame)
                and symbol in candidate_features.index
            ):
                for barrier_key in ("barrier_pct", "barrier_frac"):
                    if barrier_key in candidate_features.columns:
                        barrier_features_log[barrier_key] = candidate_features.at[
                            symbol, barrier_key
                        ]
            live_barrier_pct = _resolve_live_barrier_pct(
                symbol,
                barrier_features_log,
                panel=panel,
                cfg=runtime_config,
            )
            perp_rank = (
                _perp_rank_context(
                    data_root=str(runtime_config.get("data_root", "data")),
                    run_id=str(runtime_config.get("run_id", "latest")),
                    side=side,
                    strategy_id=strategy_id,
                    score=float(
                        chain_results.get("meta_pred")
                        or decision.get("calibrated_score")
                        or rank_for_size
                    ),
                )
                if _is_perps_config(runtime_config)
                else {}
            )

            def _log_global_auction_skip(
                stage: str,
                reason: str,
                *,
                extra: Optional[Dict[str, Any]] = None,
                execution_snapshot: Optional[Dict[str, Any]] = None,
            ) -> None:
                chain_results["global_auction_skip_stage"] = stage
                chain_results["global_auction_skip_reason"] = reason
                decision["chain_results"] = chain_results
                if (
                    prediction_ledger is not None
                    and _should_log_prediction_candidate(
                        decision, policy=portfolio_policy
                    )
                ):
                    prediction_ledger_rows.append(
                        _prediction_ledger_row(
                            decision,
                            timestamp=now_utc.isoformat(),
                            side=side,
                            portfolio_decision="portfolio_rejected",
                            portfolio_reject_reason=(
                                f"global_auction_{stage}:{reason}"
                            ),
                            execution_snapshot=execution_snapshot,
                        )
                    )
                details: Dict[str, Any] = {
                    "rank": rank_for_size,
                    "threshold": threshold_for_size,
                    "size": decision.get("size"),
                }
                if extra:
                    details.update(extra)
                compact = " ".join(
                    f"{k}={v}"
                    for k, v in details.items()
                    if v is not None and v != ""
                )
                tprint(
                    "Global auction skip: "
                    f"{symbol} {side}/{strategy_id} stage={stage} "
                    f"reason={reason}"
                    + (f" {compact}" if compact else "")
                )

            symbol_block_reason = _symbol_entry_block_reason(
                symbol,
                now=now_utc,
                logger=logger,
                executor=executor,
                cooldown_hours=LOSING_TRADE_COOLDOWN_HOURS,
            )
            if symbol_block_reason:
                side_metrics["non_fatal_issues"] = (
                    int(side_metrics.get("non_fatal_issues", 0)) + 1
                )
                _log_global_auction_skip("symbol_entry_block", symbol_block_reason)
                _commit_global_side_metrics()
                continue
            side_metrics["cooldown_pass"] = (
                int(side_metrics.get("cooldown_pass", 0)) + 1
            )
            size = float(decision.get("size") or 0.0)
            if portfolio_mgr is not None:
                decision_policy_sizing = dict(decision.get("policy_sizing") or {})
                capacity = portfolio_mgr.get_portfolio_capacity(
                    side=side,
                    strategy_id=strategy_id,
                )
                _attach_portfolio_replay_state_for_ledger(
                    decision,
                    portfolio_mgr=portfolio_mgr,
                    capacity=capacity,
                    now_utc=now_utc,
                )
                sizing_audit = compute_rank_based_position_size(
                    wallet_value=float(capacity["wallet_value"]),
                    open_notional=float(capacity["open_notional"]),
                    adjusted_rank_score=rank_for_size,
                    final_threshold=threshold_for_size,
                    policy=portfolio_policy,
                    liquidity_capacity_weight=1.0,
                    live_test_mode=live_test_mode,
                    rank_size_power=float(
                        decision_policy_sizing.get(
                            "size_power", chain_results.get("size_power", 1.1)
                        )
                    ),
                    total_assets_quote=capacity.get("total_assets_quote"),
                    total_liabilities_quote=capacity.get("total_liabilities_quote"),
                    open_positions=capacity.get("open_positions"),
                    market_mode=runtime_config.get("market_mode", "spot"),
                    available_wallet_value=capacity.get("available_wallet_quote"),
                    stop_loss_pct=live_barrier_pct,
                    rank_number=perp_rank.get("rank_number"),
                    rank_x=perp_rank.get("rank_x"),
                )
                requested_position_usdt = float(sizing_audit["size_after_liquidity"])
                chain_results["portfolio_rank_sizing"] = sizing_audit
                decision["chain_results"] = chain_results
                can_enter, info = portfolio_mgr.can_enter_position(
                    symbol=symbol,
                    side=side,
                    strategy_id=strategy_id,
                    rank_score=rank_for_size,
                    initial_threshold=threshold_for_size,
                    current_time=now_utc,
                    requested_position_size=requested_position_usdt,
                )
                chain_results["portfolio_gate"] = info
                decision["chain_results"] = chain_results
                if not can_enter:
                    side_metrics["non_fatal_issues"] = (
                        int(side_metrics.get("non_fatal_issues", 0)) + 1
                    )
                    _log_global_auction_skip(
                        "portfolio_pre_liquidity",
                        str(info.get("reason") or "portfolio_rejected"),
                        extra={
                            "requested_position_size": requested_position_usdt,
                            "portfolio_final_threshold": info.get("final_threshold"),
                            "portfolio_initial_threshold": info.get(
                                "initial_threshold"
                            ),
                            "portfolio_rank_score": info.get("rank_score"),
                            "threshold_viability_margin": getattr(
                                portfolio_mgr, "threshold_viability_margin", None
                            ),
                            "position_size_cap": info.get("position_size_cap"),
                            "n_positions_before": info.get("n_positions_before"),
                            "constraints": ",".join(
                                str(x) for x in info.get("constraints_checked", []) or []
                            ),
                        },
                    )
                    _commit_global_side_metrics()
                    continue
                side_metrics["portfolio_pass"] = (
                    int(side_metrics.get("portfolio_pass", 0)) + 1
                )
                size = min(
                    requested_position_usdt,
                    float(info.get("position_size_cap", requested_position_usdt)),
                )
            if not np.isfinite(size) or size <= 0.0:
                side_metrics["non_fatal_issues"] = (
                    int(side_metrics.get("non_fatal_issues", 0)) + 1
                )
                _log_global_auction_skip("sizing", "invalid_or_zero_size", extra={"computed_size": size})
                _commit_global_side_metrics()
                continue
            signal_close_snapshot = _raw_signal_close_reliability_snapshot(
                panel,
                symbol,
                max_reference_gap_bps=raw_close_reference_gap_bps,
            )
            price = signal_close_snapshot.get("signal_price")
            execution_snapshot: Dict[str, Any] = {}
            execution_snapshot.update(signal_close_snapshot)
            execution_limit_price = None
            execution_kwargs: Dict[str, Any] = {}
            adjusted_rank = rank_for_size
            timing_snapshot = _entry_timing_snapshot(
                decision=decision,
                now=now_utc,
                signal_bar_ts=signal_bar_ts,
                max_signal_close_age_seconds=max_signal_close_to_entry_seconds,
            )
            decision.update(
                {
                    key: value
                    for key, value in timing_snapshot.items()
                    if key
                    in {
                        "signal_bar_close_ts",
                        "signal_close_to_decision_seconds",
                        "signal_to_decision_seconds",
                        "max_signal_close_to_entry_seconds",
                        "stale_signal_age_gate_enabled",
                        "stale_signal_age_gate_exceeded",
                    }
                }
            )
            execution_snapshot.update(timing_snapshot)
            if bool(signal_close_snapshot.get("raw_signal_close_unreliable")):
                reason = (
                    "unreliable_raw_signal_close:"
                    f"{signal_close_snapshot.get('raw_signal_close_unreliable_reason')}"
                )
                side_metrics["non_fatal_issues"] = (
                    int(side_metrics.get("non_fatal_issues", 0)) + 1
                )
                _log_global_auction_skip(
                    "data_quality",
                    reason,
                    extra={
                        "raw_close": signal_close_snapshot.get("raw_signal_close"),
                        "volume": signal_close_snapshot.get("raw_signal_volume"),
                        "reference": signal_close_snapshot.get(
                            "raw_signal_close_reference_source"
                        ),
                        "reference_gap_bps": signal_close_snapshot.get(
                            "raw_signal_close_reference_gap_bps"
                        ),
                    },
                    execution_snapshot=execution_snapshot,
                )
                _commit_global_side_metrics()
                continue
            if bool(timing_snapshot.get("stale_signal_age_gate_exceeded")):
                if (
                    prediction_ledger is not None
                    and _should_log_prediction_candidate(
                        decision, policy=portfolio_policy
                    )
                ):
                    prediction_ledger_rows.append(
                        _prediction_ledger_row(
                            decision,
                            timestamp=now_utc.isoformat(),
                            side=side,
                            portfolio_decision="stale_signal_rejected",
                            portfolio_reject_reason="stale_signal_age_exceeded",
                            execution_snapshot=execution_snapshot,
                        )
                    )
                side_metrics["non_fatal_issues"] = (
                    int(side_metrics.get("non_fatal_issues", 0)) + 1
                )
                _log_global_auction_skip(
                    "stale_signal_age",
                    "stale_signal_age_exceeded",
                    extra={
                        "signal_bar_close_ts": timing_snapshot.get(
                            "signal_bar_close_ts"
                        ),
                        "signal_close_to_decision_seconds": timing_snapshot.get(
                            "signal_close_to_decision_seconds"
                        ),
                        "max_signal_close_to_entry_seconds": (
                            max_signal_close_to_entry_seconds
                        ),
                    },
                    execution_snapshot=execution_snapshot,
                )
                _commit_global_side_metrics()
                continue
            exchange = getattr(executor, "exchange", None)
            api_symbol = (
                _live_exchange_symbol(exchange, runtime_config, symbol)
                if exchange is not None
                else symbol
            )
            if stale_entry_context and (exchange is None or price is None):
                side_metrics["non_fatal_issues"] = (
                    int(side_metrics.get("non_fatal_issues", 0)) + 1
                )
                _log_global_auction_skip(
                    "stale_entry_precheck",
                    "missing_exchange_or_signal_price",
                    extra={
                        "has_exchange": exchange is not None,
                        "signal_price": price,
                    },
                    execution_snapshot=execution_snapshot,
                )
                _commit_global_side_metrics()
                continue
            if (
                exchange is not None
                and price is not None
                and (
                    stale_entry_context
                    or portfolio_policy.ticker_precheck_enabled
                    or portfolio_policy.orderbook_precheck_enabled
                    or adverse_hourly_close_gate_enabled
                )
            ):
                try:
                    execution_precheck_ts = pd.Timestamp.now(tz="UTC")
                    ticker_snapshot = fetch_ticker_snapshot(
                        exchange=exchange,
                        symbol=api_symbol,
                        side=side,
                        policy=portfolio_policy,
                        mode=str(getattr(executor, "mode", "")),
                        now=execution_precheck_ts,
                    )
                    execution_snapshot.update(ticker_snapshot.to_dict())
                    if ticker_snapshot.hard_reject:
                        if (
                            prediction_ledger is not None
                            and _should_log_prediction_candidate(
                                decision, policy=portfolio_policy
                            )
                        ):
                            prediction_ledger_rows.append(
                                _prediction_ledger_row(
                                    decision,
                                    timestamp=now_utc.isoformat(),
                                    side=side,
                                    portfolio_decision="liquidity_rejected",
                                    liquidity_reject_reason=str(
                                        ticker_snapshot.reject_reason
                                        or "ticker_rejected"
                                    ),
                                    execution_snapshot=execution_snapshot,
                                )
                            )
                        side_metrics["non_fatal_issues"] = (
                            int(side_metrics.get("non_fatal_issues", 0)) + 1
                        )
                        _log_global_auction_skip(
                            "ticker_precheck",
                            str(ticker_snapshot.reject_reason or "ticker_rejected"),
                            extra={
                                "spread_bps": ticker_snapshot.spread_bps,
                                "mid": ticker_snapshot.mid,
                            },
                            execution_snapshot=execution_snapshot,
                        )
                        _commit_global_side_metrics()
                        continue
                    decision_mid = float(ticker_snapshot.mid or 0.0)
                    if stale_entry_context:
                        max_abs_gap_bps = float(
                            stale_entry_max_abs_signal_gap_bps
                            if stale_entry_max_abs_signal_gap_bps is not None
                            else 0.0
                        )
                        stale_abs_gap_bps = (
                            abs(decision_mid / max(float(price), 1e-12) - 1.0) * 10000.0
                        )
                        execution_snapshot["stale_entry_context"] = True
                        execution_snapshot["stale_entry_abs_signal_gap_bps"] = float(
                            stale_abs_gap_bps
                        )
                        execution_snapshot["stale_entry_max_abs_signal_gap_bps"] = (
                            float(max_abs_gap_bps)
                        )
                        if (
                            not np.isfinite(stale_abs_gap_bps)
                            or stale_abs_gap_bps > max_abs_gap_bps
                        ):
                            if (
                                prediction_ledger is not None
                                and _should_log_prediction_candidate(
                                    decision, policy=portfolio_policy
                                )
                            ):
                                prediction_ledger_rows.append(
                                    _prediction_ledger_row(
                                        decision,
                                        timestamp=now_utc.isoformat(),
                                        side=side,
                                        portfolio_decision="price_gap_rejected",
                                        portfolio_reject_reason=(
                                            "stale_entry_price_moved_too_far"
                                        ),
                                        execution_snapshot=execution_snapshot,
                                    )
                                )
                            side_metrics["non_fatal_issues"] = (
                                int(side_metrics.get("non_fatal_issues", 0)) + 1
                            )
                            _log_global_auction_skip(
                                "stale_entry_price_gap",
                                "stale_entry_price_moved_too_far",
                                extra={
                                    "abs_gap_bps": stale_abs_gap_bps,
                                    "max_abs_gap_bps": max_abs_gap_bps,
                                    "signal_price": price,
                                    "decision_mid": decision_mid,
                                },
                                execution_snapshot=execution_snapshot,
                            )
                            _commit_global_side_metrics()
                            continue
                    gap_penalty, gap_info = compute_price_gap_rank_penalty(
                        strategy_id=strategy_id,
                        side=side,
                        signal_price=float(price),
                        decision_mid=decision_mid,
                        policy=portfolio_policy,
                    )
                    adjusted_rank = max(rank_for_size - float(gap_penalty), 0.0)
                    execution_snapshot.update(gap_info)
                    adverse_gap_bps = _adverse_signal_gap_bps(
                        side=side,
                        signal_price=price,
                        decision_mid=decision_mid,
                    )
                    execution_snapshot["adverse_signal_gap_bps"] = float(
                        adverse_gap_bps
                    )
                    execution_snapshot["adverse_hourly_close_gap_bps"] = float(
                        adverse_gap_bps
                    )
                    execution_snapshot["adverse_hourly_close_gate_bps"] = float(
                        adverse_hourly_close_gate_bps
                    )
                    if (
                        adverse_hourly_close_gate_enabled
                        and (
                            not np.isfinite(adverse_gap_bps)
                            or adverse_gap_bps >= adverse_hourly_close_gate_bps
                        )
                    ):
                        if (
                            prediction_ledger is not None
                            and _should_log_prediction_candidate(
                                decision, policy=portfolio_policy
                            )
                        ):
                            prediction_ledger_rows.append(
                                _prediction_ledger_row(
                                    decision,
                                    timestamp=now_utc.isoformat(),
                                    side=side,
                                    portfolio_decision="price_gap_rejected",
                                    portfolio_reject_reason=(
                                        "adverse_hourly_close_gap_too_large"
                                    ),
                                    execution_snapshot=execution_snapshot,
                                )
                            )
                        side_metrics["non_fatal_issues"] = (
                            int(side_metrics.get("non_fatal_issues", 0)) + 1
                        )
                        _log_global_auction_skip(
                            "adverse_hourly_close_gap",
                            "adverse_hourly_close_gap_too_large",
                            extra={
                                "adverse_gap_bps": adverse_gap_bps,
                                "gate_bps": adverse_hourly_close_gate_bps,
                                "hourly_close": price,
                                "decision_mid": decision_mid,
                            },
                            execution_snapshot=execution_snapshot,
                        )
                        _commit_global_side_metrics()
                        continue
                    chain_results["initial_calibrated_score"] = decision.get(
                        "calibrated_score"
                    )
                    execution_snapshot["price_gap_penalty"] = float(gap_penalty)
                    execution_snapshot["adjusted_rank_score"] = float(adjusted_rank)
                    if adjusted_rank < threshold_for_size:
                        if (
                            prediction_ledger is not None
                            and _should_log_prediction_candidate(
                                decision, policy=portfolio_policy
                            )
                        ):
                            prediction_ledger_rows.append(
                                _prediction_ledger_row(
                                    decision,
                                    timestamp=now_utc.isoformat(),
                                    side=side,
                                    portfolio_decision="price_gap_rejected",
                                    portfolio_reject_reason=(
                                        "rank_below_dynamic_threshold_after_price_gap"
                                    ),
                                    execution_snapshot=execution_snapshot,
                                )
                            )
                        side_metrics["non_fatal_issues"] = (
                            int(side_metrics.get("non_fatal_issues", 0)) + 1
                        )
                        _log_global_auction_skip(
                            "price_gap_penalty",
                            "rank_below_dynamic_threshold_after_price_gap",
                            extra={
                                "adjusted_rank": adjusted_rank,
                                "gap_penalty": gap_penalty,
                                "signal_price": price,
                                "decision_mid": decision_mid,
                            },
                            execution_snapshot=execution_snapshot,
                        )
                        _commit_global_side_metrics()
                        continue
                    spread_baseline_bps, spread_baseline_source = (
                        _live_ev_haircut_spread_baseline_bps(
                            symbol=symbol,
                            data_root=str(runtime_config.get("data_root", "data")),
                            fallback_bps=(
                                portfolio_policy.ev_haircut_expected_spread_bps
                            ),
                        )
                    )
                    if not portfolio_policy.orderbook_precheck_enabled:
                        ev_adjusted = _ev_adjusted_prediction_after_entry_friction(
                            calibrated_score=decision.get("calibrated_score"),
                            strategy_id=strategy_id,
                            side=side,
                            calibration=strategy_ev_calibration,
                            live_entry_friction_bps=float(adverse_gap_bps),
                            observed_spread_bps=getattr(
                                ticker_snapshot, "spread_bps", None
                            ),
                            orderbook_slippage_bps=0.0,
                            adverse_signal_gap_bps=float(adverse_gap_bps),
                            spread_baseline_bps=spread_baseline_bps,
                            spread_baseline_source=spread_baseline_source,
                            delay_slippage_baseline_bps=(
                                portfolio_policy.ev_haircut_delay_slippage_baseline_bps
                            ),
                            policy_rank_reference_store=policy_rank_reference_store,
                        )
                        chain_results.update(ev_adjusted)
                        execution_snapshot.update(ev_adjusted)
                        chain_results["adjusted_calibrated_score"] = ev_adjusted.get(
                            "ev_adjusted_calibrated_score"
                        )
                        execution_snapshot["adjusted_calibrated_score"] = (
                            ev_adjusted.get("ev_adjusted_calibrated_score")
                        )
                        ev_rank = _safe_float(
                            ev_adjusted.get("ev_adjusted_rank_score"), np.nan
                        )
                        if np.isfinite(ev_rank):
                            adjusted_rank = min(float(adjusted_rank), float(ev_rank))
                            execution_snapshot["adjusted_rank_score"] = float(
                                adjusted_rank
                            )
                            chain_results["threshold_rank_score_after_friction_ev"] = (
                                float(adjusted_rank)
                            )
                            if adjusted_rank < threshold_for_size:
                                side_metrics["non_fatal_issues"] = (
                                    int(side_metrics.get("non_fatal_issues", 0)) + 1
                                )
                                _commit_global_side_metrics()
                                continue
                    if portfolio_policy.orderbook_precheck_enabled:
                        book_snapshot = evaluate_orderbook_liquidity(
                            exchange=exchange,
                            symbol=api_symbol,
                            side=side,
                            intended_quote_size=float(size),
                            ticker_snapshot=ticker_snapshot,
                            policy=portfolio_policy,
                            mode=str(getattr(executor, "mode", "")),
                        )
                        execution_snapshot.update(book_snapshot.to_dict())
                        perps_capacity_cap = (
                            _is_perps_config(runtime_config)
                            and book_snapshot.reject_reason
                            == "liquidity_capacity_weight_below_min"
                            and _safe_float(
                                book_snapshot.orderbook_capacity_quote_within_slippage,
                                0.0,
                            )
                            > 0.0
                        )
                        if book_snapshot.hard_reject and not perps_capacity_cap:
                            if (
                                prediction_ledger is not None
                                and _should_log_prediction_candidate(
                                    decision, policy=portfolio_policy
                                )
                            ):
                                prediction_ledger_rows.append(
                                    _prediction_ledger_row(
                                        decision,
                                        timestamp=now_utc.isoformat(),
                                        side=side,
                                        portfolio_decision="liquidity_rejected",
                                        liquidity_reject_reason=str(
                                            book_snapshot.reject_reason
                                            or "orderbook_rejected"
                                        ),
                                        execution_snapshot=execution_snapshot,
                                    )
                                )
                            side_metrics["non_fatal_issues"] = (
                                int(side_metrics.get("non_fatal_issues", 0)) + 1
                            )
                            _log_global_auction_skip(
                                "orderbook_precheck",
                                str(book_snapshot.reject_reason or "orderbook_rejected"),
                                extra={
                                    "capacity_quote": book_snapshot.orderbook_capacity_quote_within_slippage,
                                    "capacity_weight": book_snapshot.liquidity_capacity_weight,
                                    "expected_slippage_bps": book_snapshot.expected_fill_slippage_bps,
                                },
                                execution_snapshot=execution_snapshot,
                            )
                            _commit_global_side_metrics()
                            continue
                        live_entry_friction_bps = _safe_float(
                            book_snapshot.expected_total_entry_friction_bps,
                            0.0,
                        ) + float(adverse_gap_bps)
                        ev_adjusted = _ev_adjusted_prediction_after_entry_friction(
                            calibrated_score=decision.get("calibrated_score"),
                            strategy_id=strategy_id,
                            side=side,
                            calibration=strategy_ev_calibration,
                            live_entry_friction_bps=live_entry_friction_bps,
                            observed_spread_bps=book_snapshot.spread_bps,
                            orderbook_slippage_bps=(
                                book_snapshot.expected_fill_slippage_bps
                            ),
                            adverse_signal_gap_bps=float(adverse_gap_bps),
                            spread_baseline_bps=spread_baseline_bps,
                            spread_baseline_source=spread_baseline_source,
                            delay_slippage_baseline_bps=(
                                portfolio_policy.ev_haircut_delay_slippage_baseline_bps
                            ),
                            policy_rank_reference_store=policy_rank_reference_store,
                        )
                        chain_results.update(ev_adjusted)
                        execution_snapshot.update(ev_adjusted)
                        chain_results["adjusted_calibrated_score"] = ev_adjusted.get(
                            "ev_adjusted_calibrated_score"
                        )
                        execution_snapshot["adjusted_calibrated_score"] = (
                            ev_adjusted.get("ev_adjusted_calibrated_score")
                        )
                        ev_rank = _safe_float(
                            ev_adjusted.get("ev_adjusted_rank_score"), np.nan
                        )
                        if np.isfinite(ev_rank):
                            adjusted_rank = min(float(adjusted_rank), float(ev_rank))
                            execution_snapshot["adjusted_rank_score"] = float(
                                adjusted_rank
                            )
                            chain_results["threshold_rank_score_after_friction_ev"] = (
                                float(adjusted_rank)
                            )
                            if adjusted_rank < threshold_for_size:
                                if (
                                    prediction_ledger is not None
                                    and _should_log_prediction_candidate(
                                        decision, policy=portfolio_policy
                                    )
                                ):
                                    prediction_ledger_rows.append(
                                        _prediction_ledger_row(
                                            decision,
                                            timestamp=now_utc.isoformat(),
                                            side=side,
                                            portfolio_decision="liquidity_rejected",
                                            liquidity_reject_reason=(
                                                "rank_below_dynamic_threshold_after_live_friction_ev"
                                            ),
                                            execution_snapshot=execution_snapshot,
                                        )
                                    )
                                side_metrics["non_fatal_issues"] = (
                                    int(side_metrics.get("non_fatal_issues", 0)) + 1
                                )
                                _log_global_auction_skip(
                                    "live_friction_ev",
                                    "rank_below_dynamic_threshold_after_live_friction_ev",
                                    extra={
                                        "adjusted_rank": adjusted_rank,
                                        "entry_friction_bps": live_entry_friction_bps,
                                        "ev_before": ev_adjusted.get(
                                            "ev_adjusted_net_return_before_friction"
                                        ),
                                        "ev_after": ev_adjusted.get(
                                            "ev_adjusted_net_return_after_friction"
                                        ),
                                        "ev_adjusted_score": ev_adjusted.get(
                                            "ev_adjusted_calibrated_score"
                                        ),
                                    },
                                    execution_snapshot=execution_snapshot,
                                )
                                _commit_global_side_metrics()
                                continue
                            if portfolio_mgr is not None:
                                capacity = portfolio_mgr.get_portfolio_capacity(
                                    side=side,
                                    strategy_id=strategy_id,
                                )
                                _attach_portfolio_replay_state_for_ledger(
                                    decision,
                                    portfolio_mgr=portfolio_mgr,
                                    capacity=capacity,
                                    now_utc=now_utc,
                                )
                                perp_rank = (
                                    _perp_rank_context(
                                        data_root=str(
                                        runtime_config.get("data_root", "data")
                                    ),
                                    run_id=str(runtime_config.get("run_id", "latest")),
                                    side=side,
                                    strategy_id=strategy_id,
                                    score=float(
                                        chain_results.get("meta_pred")
                                        or decision.get("calibrated_score")
                                        or adjusted_rank
                                    ),
                                )
                                if _is_perps_config(runtime_config)
                                else {}
                            )
                            sizing_audit = compute_rank_based_position_size(
                                wallet_value=float(capacity["wallet_value"]),
                                open_notional=float(capacity["open_notional"]),
                                adjusted_rank_score=float(adjusted_rank),
                                final_threshold=threshold_for_size,
                                policy=portfolio_policy,
                                liquidity_capacity_weight=float(
                                    book_snapshot.liquidity_capacity_weight
                                ),
                                live_test_mode=live_test_mode,
                                rank_size_power=float(
                                    decision_policy_sizing.get(
                                        "size_power",
                                        chain_results.get("size_power", 1.1),
                                    )
                                ),
                                total_assets_quote=capacity.get("total_assets_quote"),
                                total_liabilities_quote=capacity.get(
                                    "total_liabilities_quote"
                                ),
                                open_positions=capacity.get("open_positions"),
                                market_mode=runtime_config.get("market_mode", "spot"),
                                available_wallet_value=capacity.get(
                                    "available_wallet_quote"
                                ),
                                rank_number=perp_rank.get("rank_number"),
                                rank_x=perp_rank.get("rank_x"),
                                orderbook_capacity_quote=(
                                    book_snapshot.orderbook_capacity_quote_within_slippage
                                ),
                            )
                            size = float(sizing_audit["size_after_liquidity"])
                            chain_results["portfolio_rank_sizing"] = sizing_audit
                            execution_snapshot["threshold_viability_margin"] = getattr(
                                portfolio_mgr, "threshold_viability_margin", None
                            )
                            can_enter, info = portfolio_mgr.can_enter_position(
                                symbol=symbol,
                                side=side,
                                strategy_id=strategy_id,
                                rank_score=float(adjusted_rank),
                                initial_threshold=threshold_for_size,
                                current_time=now_utc,
                                requested_position_size=size,
                            )
                            chain_results["portfolio_gate_after_liquidity"] = info
                            decision["chain_results"] = chain_results
                            if not can_enter:
                                if (
                                    prediction_ledger is not None
                                    and _should_log_prediction_candidate(
                                        decision, policy=portfolio_policy
                                    )
                                ):
                                    prediction_ledger_rows.append(
                                        _prediction_ledger_row(
                                            decision,
                                            timestamp=now_utc.isoformat(),
                                            side=side,
                                            portfolio_decision="portfolio_rejected",
                                            portfolio_reject_reason=str(
                                                info.get("reason")
                                                or "post_liquidity_portfolio_rejected"
                                            ),
                                            execution_snapshot=execution_snapshot,
                                        )
                                    )
                                side_metrics["non_fatal_issues"] = (
                                    int(side_metrics.get("non_fatal_issues", 0)) + 1
                                )
                                _log_global_auction_skip(
                                    "portfolio_post_liquidity",
                                    str(
                                        info.get("reason")
                                        or "post_liquidity_portfolio_rejected"
                                    ),
                                    extra={
                                        "requested_position_size": size,
                                        "position_size_cap": info.get("position_size_cap"),
                                        "n_positions_before": info.get("n_positions_before"),
                                        "constraints": ",".join(
                                            str(x)
                                            for x in info.get("constraints_checked", []) or []
                                        ),
                                    },
                                    execution_snapshot=execution_snapshot,
                                )
                                _commit_global_side_metrics()
                                continue
                        execution_snapshot["liquidity_capacity_weight"] = float(
                            book_snapshot.liquidity_capacity_weight
                        )
                        execution_snapshot["expected_entry_price"] = (
                            book_snapshot.expected_fill_price
                        )
                        execution_snapshot["expected_fill_slippage_bps"] = (
                            book_snapshot.expected_fill_slippage_bps
                        )
                    if decision_mid > 0:
                        execution_limit_price = marketable_limit_price(
                            side=side,
                            decision_mid=decision_mid,
                            policy=portfolio_policy,
                        )
                        execution_snapshot["max_chase_bps"] = (
                            portfolio_policy.max_order_chase_bps
                        )
                        execution_snapshot["entry_limit_price"] = execution_limit_price
                except Exception as exc:
                    if (
                        prediction_ledger is not None
                        and _should_log_prediction_candidate(
                            decision, policy=portfolio_policy
                        )
                    ):
                        prediction_ledger_rows.append(
                            _prediction_ledger_row(
                                decision,
                                timestamp=now_utc.isoformat(),
                                side=side,
                                portfolio_decision="liquidity_rejected",
                                liquidity_reject_reason=(
                                    f"{classify_api_error(exc)}: {exc}"
                                ),
                                execution_snapshot=execution_snapshot,
                            )
                        )
                    side_metrics["non_fatal_issues"] = (
                        int(side_metrics.get("non_fatal_issues", 0)) + 1
                    )
                    _log_global_auction_skip(
                        "execution_precheck_exception",
                        f"{classify_api_error(exc)}: {exc}",
                        execution_snapshot=execution_snapshot,
                    )
                    _commit_global_side_metrics()
                    continue
            exchange_min_notional = _exchange_min_notional_for_symbol(
                exchange,
                api_symbol,
            )
            if (
                exchange_min_notional is not None
                and np.isfinite(size)
                and float(size) < float(exchange_min_notional)
            ):
                if prediction_ledger is not None and _should_log_prediction_candidate(
                    decision, policy=portfolio_policy
                ):
                    prediction_ledger_rows.append(
                        _prediction_ledger_row(
                            decision,
                            timestamp=now_utc.isoformat(),
                            side=side,
                            portfolio_decision="exchange_filter_rejected",
                            portfolio_reject_reason="below_exchange_min_notional",
                            execution_snapshot=execution_snapshot,
                            was_traded=False,
                        )
                    )
                side_metrics["non_fatal_issues"] = (
                    int(side_metrics.get("non_fatal_issues", 0)) + 1
                )
                _log_global_auction_skip(
                    "exchange_filter",
                    "below_exchange_min_notional",
                    extra={
                        "computed_size": size,
                        "exchange_min_notional": exchange_min_notional,
                    },
                    execution_snapshot=execution_snapshot,
                )
                _commit_global_side_metrics()
                continue
            execution_price = (
                float(execution_limit_price)
                if execution_limit_price is not None
                else float(chain_results.get("entry_px") or price)
            )
            if execution_snapshot:
                policy_reference_price = float(price) if price is not None else None
                if policy_reference_price is not None:
                    execution_snapshot["signal_price"] = policy_reference_price
                    execution_snapshot["theoretical_entry_price"] = (
                        policy_reference_price
                    )
                    execution_snapshot["policy_entry_price"] = policy_reference_price
                    execution_snapshot["policy_entry_price_source"] = (
                        "signal_bar_close"
                    )
                execution_kwargs = {
                    "execution_snapshot": execution_snapshot,
                    "signal_price": policy_reference_price,
                    "decision_mid": execution_snapshot.get("decision_mid"),
                    "expected_entry_price": execution_snapshot.get(
                        "expected_entry_price"
                    )
                    or execution_snapshot.get("expected_fill_price")
                    or execution_price,
                    "expected_fill_slippage_bps": execution_snapshot.get(
                        "expected_fill_slippage_bps"
                    ),
                    "max_chase_bps": portfolio_policy.max_order_chase_bps,
                    "rank_score": rank_for_size,
                    "adjusted_rank_score": adjusted_rank,
                    "final_threshold": threshold_for_size,
                    "position_size_before_liquidity": (
                        chain_results.get("portfolio_rank_sizing", {}) or {}
                    ).get("size_before_liquidity"),
                    "position_size_after_liquidity": size,
                    "order_type": "limit" if execution_limit_price else None,
                    "limit_price": execution_limit_price,
                }
            trade_audit = _build_trade_start_audit(
                orchestrator=orchestrator,
                panel=panel,
                feats=feats,
                candidate_features=candidate_features,
                meta_model_input_features=decision.get("_meta_model_input_features"),
                symbol=symbol,
                side=side,
                strategy_id=strategy_id,
                signal_bar_ts=signal_bar_ts,
                decision=decision,
                chain_results=chain_results,
                execution_snapshot=execution_snapshot,
                parity_contract=runtime_config.get("training_live_parity_contract"),
            )
            bucket_key = strategy_core_id(strategy_id)
            resolver = getattr(executor, "resolve_simple_policy_strategy_id", None)
            if callable(resolver):
                resolved_bucket_key = resolver(bucket_key, side)
                if resolved_bucket_key:
                    bucket_key = str(resolved_bucket_key)
            sizing_context = chain_results.get("portfolio_rank_sizing", {}) or {}
            perp_sizing_context = {
                key: sizing_context.get(key)
                for key in (
                    "leverage_wallet_multiplier",
                    "book_notional_multiplier",
                    "perp_rank_number",
                    "perp_rank_x",
                    "perp_rank_leverage",
                    "perp_risk_cap_leverage",
                    "perp_effective_leverage",
                    "perp_stop_loss_pct",
                    "perp_full_wallet",
                    "perp_available_wallet",
                    "orderbook_capacity_quote_within_slippage",
                )
                if key in sizing_context
            }
            trade_result = _execute_trade_with_optional_context(
                executor,
                symbol=symbol,
                side=side,
                size=abs(size),
                price=execution_price if price is not None else None,
                bucket_key=bucket_key,
                ohlcv_reference_price=float(price) if price is not None else None,
                trade_context={
                    "base_pred": chain_results.get("base_pred"),
                    "meta_pred": chain_results.get("meta_pred"),
                    "estimated_hit_rate": chain_results.get("estimated_hit_rate"),
                    "estimated_hit_rate_source": chain_results.get(
                        "estimated_hit_rate_source"
                    ),
                    "estimated_hit_rate_calibration_n": chain_results.get(
                        "estimated_hit_rate_calibration_n"
                    ),
                    "estimated_ev_gross_return": chain_results.get(
                        "estimated_ev_gross_return"
                    ),
                    "estimated_ev_net_return": chain_results.get(
                        "estimated_ev_net_return"
                    ),
                    "estimated_ev_cost_bps": chain_results.get(
                        "estimated_ev_cost_bps"
                    ),
                    "estimated_ev_hit_rate": chain_results.get("estimated_ev_hit_rate"),
                    "estimated_ev_source": chain_results.get("estimated_ev_source"),
                    "estimated_ev_calibration_n": chain_results.get(
                        "estimated_ev_calibration_n"
                    ),
                    "rank_score_source": chain_results.get("rank_score_source"),
                    "policy_rank_pct": chain_results.get("policy_rank_pct"),
                    "auction_rank_pct": chain_results.get("auction_rank_pct"),
                    "calibrated_score": decision.get("calibrated_score"),
                    "initial_calibrated_score": chain_results.get(
                        "initial_calibrated_score"
                    ),
                    "adjusted_calibrated_score": chain_results.get(
                        "adjusted_calibrated_score"
                    ),
                    "adverse_signal_gap_bps": execution_snapshot.get(
                        "adverse_signal_gap_bps"
                    ),
                    "ev_adjusted_entry_friction_bps": chain_results.get(
                        "ev_adjusted_entry_friction_bps"
                    ),
                    "ev_haircut_bps": chain_results.get("ev_haircut_bps"),
                    "ev_haircut_raw_live_entry_friction_bps": chain_results.get(
                        "ev_haircut_raw_live_entry_friction_bps"
                    ),
                    "ev_haircut_observed_spread_bps": chain_results.get(
                        "ev_haircut_observed_spread_bps"
                    ),
                    "ev_haircut_observed_half_spread_bps": chain_results.get(
                        "ev_haircut_observed_half_spread_bps"
                    ),
                    "ev_haircut_spread_baseline_bps": chain_results.get(
                        "ev_haircut_spread_baseline_bps"
                    ),
                    "ev_haircut_spread_baseline_source": chain_results.get(
                        "ev_haircut_spread_baseline_source"
                    ),
                    "ev_haircut_half_spread_baseline_bps": chain_results.get(
                        "ev_haircut_half_spread_baseline_bps"
                    ),
                    "ev_haircut_spread_excess_bps": chain_results.get(
                        "ev_haircut_spread_excess_bps"
                    ),
                    "ev_haircut_orderbook_slippage_bps": chain_results.get(
                        "ev_haircut_orderbook_slippage_bps"
                    ),
                    "ev_haircut_adverse_signal_gap_bps": chain_results.get(
                        "ev_haircut_adverse_signal_gap_bps"
                    ),
                    "ev_haircut_observed_delay_slippage_bps": chain_results.get(
                        "ev_haircut_observed_delay_slippage_bps"
                    ),
                    "ev_haircut_delay_slippage_baseline_bps": chain_results.get(
                        "ev_haircut_delay_slippage_baseline_bps"
                    ),
                    "ev_haircut_delay_slippage_excess_bps": chain_results.get(
                        "ev_haircut_delay_slippage_excess_bps"
                    ),
                    "ev_haircut_contract": chain_results.get("ev_haircut_contract"),
                    "ev_adjusted_net_return_before_friction": chain_results.get(
                        "ev_adjusted_net_return_before_friction"
                    ),
                    "ev_adjusted_net_return_after_friction": chain_results.get(
                        "ev_adjusted_net_return_after_friction"
                    ),
                    "ev_adjusted_calibrated_score": chain_results.get(
                        "ev_adjusted_calibrated_score"
                    ),
                    "ev_adjusted_rank_score": chain_results.get(
                        "ev_adjusted_rank_score"
                    ),
                    "ev_adjusted_source": chain_results.get("ev_adjusted_source"),
                    "rank_percentile": chain_results.get("sizer_rank_percentile"),
                    "effective_threshold": chain_results.get("effective_threshold"),
                    "barrier_pct": live_barrier_pct,
                    "barrier_frac": live_barrier_pct,
                    "decision_ts": decision.get("decision_ts"),
                    "signal_bar_ts": decision.get("signal_bar_ts"),
                    "signal_bar_close_ts": decision.get("signal_bar_close_ts"),
                    "signal_close_to_decision_seconds": decision.get(
                        "signal_close_to_decision_seconds"
                    ),
                    "signal_to_decision_seconds": decision.get(
                        "signal_to_decision_seconds"
                    ),
                    "max_signal_close_to_entry_seconds": decision.get(
                        "max_signal_close_to_entry_seconds"
                    ),
                    "signal_to_entry_alert_seconds": signal_to_entry_alert_seconds,
                    **trade_audit,
                    **perp_sizing_context,
                },
                execution_kwargs=execution_kwargs,
            )
            _record_trade_execution_health(portfolio_mgr, trade_result)
            trade_success = bool(
                trade_result.get("success", False)
                or trade_result.get("status") == "recorded"
            )
            order_error_category = str(trade_result.get("error_category", "") or "")
            predictions = {
                "base_pred": chain_results.get("base_pred"),
                "meta_pred": chain_results.get("meta_pred"),
                "estimated_hit_rate": chain_results.get("estimated_hit_rate"),
                "estimated_hit_rate_source": chain_results.get(
                    "estimated_hit_rate_source"
                ),
                "estimated_hit_rate_calibration_n": chain_results.get(
                    "estimated_hit_rate_calibration_n"
                ),
                "estimated_ev_gross_return": chain_results.get(
                    "estimated_ev_gross_return"
                ),
                "estimated_ev_net_return": chain_results.get("estimated_ev_net_return"),
                "estimated_ev_cost_bps": chain_results.get("estimated_ev_cost_bps"),
                "estimated_ev_hit_rate": chain_results.get("estimated_ev_hit_rate"),
                "estimated_ev_source": chain_results.get("estimated_ev_source"),
                "estimated_ev_calibration_n": chain_results.get(
                    "estimated_ev_calibration_n"
                ),
                "rank_score_source": chain_results.get("rank_score_source"),
                "policy_rank_pct": chain_results.get("policy_rank_pct"),
                "auction_rank_pct": chain_results.get("auction_rank_pct"),
                "calibrated_score": decision.get("calibrated_score"),
                "initial_calibrated_score": chain_results.get(
                    "initial_calibrated_score"
                ),
                "adjusted_calibrated_score": chain_results.get(
                    "adjusted_calibrated_score"
                ),
                "adverse_signal_gap_bps": execution_snapshot.get(
                    "adverse_signal_gap_bps"
                ),
                "ev_adjusted_entry_friction_bps": chain_results.get(
                    "ev_adjusted_entry_friction_bps"
                ),
                "rank_percentile": chain_results.get("sizer_rank_percentile"),
                "effective_threshold": chain_results.get("effective_threshold"),
                "model_artifact_run_id": decision.get("model_artifact_run_id"),
                "policy_artifact_run_id": decision.get("policy_artifact_run_id"),
            }
            features_log = {
                **trade_audit,
                **dict(execution_snapshot or {}),
                "signal_price": float(price) if price is not None else None,
                "adjusted_rank_score": adjusted_rank,
                "final_threshold": threshold_for_size,
                "position_size_after_liquidity": size,
                "barrier_pct": live_barrier_pct,
                "barrier_frac": live_barrier_pct,
                "decision_ts": decision.get("decision_ts"),
                "signal_bar_ts": decision.get("signal_bar_ts"),
            }
            if portfolio_mgr is not None and trade_success:
                portfolio_mgr.record_position_open(
                    symbol=symbol,
                    side=side,
                    strategy_id=strategy_id,
                    position_size=float(abs(size)),
                    entry_price=float(price if price is not None else 0.0),
                    entry_time=now_utc,
                )
            if trade_success and _shadow_execution_realism_enabled():
                try:
                    live_position_state = executor.get_position(symbol) or {}
                    params_for_shadow = (
                        executor.get_simple_policy_stop_params(strategy_id)
                        if hasattr(executor, "get_simple_policy_stop_params")
                        else {}
                    )
                    policy_entry_for_shadow, policy_entry_source = (
                        _position_policy_entry_price(
                            {
                                **live_position_state,
                                "realized_entry_price": trade_result.get(
                                    "realized_entry_price"
                                ),
                                "expected_entry_price": trade_result.get(
                                    "expected_entry_price"
                                ),
                                "signal_price": price,
                            }
                        )
                    )
                    realized_entry_for_shadow = _finite_positive_float(
                        trade_result.get("realized_entry_price")
                    )
                    if not np.isfinite(realized_entry_for_shadow):
                        realized_entry_for_shadow = _finite_positive_float(price)
                    shadow_state = _ensure_simple_policy_shadow_state(
                        live_position_state,
                        symbol=symbol,
                        side=side,
                        policy_entry_price=policy_entry_for_shadow,
                        policy_entry_price_source=policy_entry_source,
                        realized_entry_price=realized_entry_for_shadow,
                        stop_price=_finite_positive_float(
                            trade_result.get("stop_price")
                        ),
                        stop_reason=str(
                            live_position_state.get("stop_reason")
                            or "original_stop_loss"
                        ),
                        params=params_for_shadow,
                    )
                    _append_simple_policy_shadow_event(
                        shadow_state,
                        "shadow_entry_seeded",
                        policy_entry_price=policy_entry_for_shadow,
                        policy_entry_price_source=policy_entry_source,
                        realized_entry_price=realized_entry_for_shadow,
                        entry_gap_bps=shadow_state.get("entry_gap_bps"),
                        initial_shadow_stop_price=shadow_state.get(
                            "initial_shadow_stop_price"
                        ),
                        live_stop_price=trade_result.get("stop_price"),
                    )
                    executor.update_position_policy_state(
                        symbol,
                        shadow_simple_policy_state=shadow_state,
                    )
                    features_log["simple_policy_shadow"] = shadow_state
                    features_log.update(
                        {
                            "shadow_policy_schema": shadow_state.get("schema"),
                            "shadow_policy_params_source": shadow_state.get(
                                "params_source"
                            ),
                            "shadow_policy_params_hash": shadow_state.get(
                                "params_hash"
                            ),
                            "shadow_policy_entry_price": shadow_state.get(
                                "policy_entry_price"
                            ),
                            "shadow_realized_entry_price": shadow_state.get(
                                "realized_entry_price"
                            ),
                            "shadow_entry_gap_bps": shadow_state.get(
                                "entry_gap_bps"
                            ),
                            "shadow_initial_stop_price": shadow_state.get(
                                "initial_shadow_stop_price"
                            ),
                            "shadow_latest_stop_price": shadow_state.get(
                                "shadow_stop_price"
                            ),
                            "shadow_live_stop_price": trade_result.get("stop_price"),
                            "shadow_stop_gap_bps": shadow_state.get(
                                "latest_stop_gap_bps"
                            ),
                            "shadow_status": shadow_state.get("status"),
                        }
                    )
                except Exception as exc:
                    tprint(
                        f"Warning: failed to seed execution-realism shadow for "
                        f"{symbol} {side}/{strategy_id}: {exc}"
                    )
            if trade_success:
                tprint(
                    f"Trade entry accepted: {symbol} {side}/{strategy_id} "
                    f"estimated_hit_rate={_safe_float(chain_results.get('estimated_hit_rate')):.3f} "
                    f"estimated_net_ev={_safe_float(chain_results.get('estimated_ev_net_return')):.4f} "
                    f"estimated_gross_ev={_safe_float(chain_results.get('estimated_ev_gross_return')):.4f} "
                    f"estimated_cost_bps={_safe_float(chain_results.get('estimated_ev_cost_bps')):.1f} "
                    f"source={chain_results.get('estimated_hit_rate_source')}"
                )
                logger.log_entry(
                    symbol=symbol,
                    side=side,
                    size=abs(size),
                    price=trade_result.get("realized_entry_price", price),
                    predictions=predictions,
                    features=features_log,
                    mode=executor.mode,
                    strategy_id=strategy_id,
                    calibrated_score=float(decision["calibrated_score"]),
                    rank_threshold=float(decision["rank_threshold"]),
                    expected_entry_price=trade_result.get("expected_entry_price"),
                    realized_entry_price=trade_result.get("realized_entry_price"),
                    entry_order_type=trade_result.get("entry_order_type"),
                    price_slippage_pct=trade_result.get("price_slippage_pct"),
                    ohlcv_entry_price=trade_result.get("ohlcv_entry_price"),
                    entry_price_delta_vs_ohlcv=trade_result.get(
                        "entry_price_delta_vs_ohlcv"
                    ),
                    entry_price_delta_vs_ohlcv_pct=trade_result.get(
                        "entry_price_delta_vs_ohlcv_pct"
                    ),
                    entry_delay_effect_bps=trade_result.get("entry_delay_effect_bps"),
                    entry_delay_adverse_bps=trade_result.get(
                        "entry_delay_adverse_bps"
                    ),
                    entry_delay_abs_bps=trade_result.get("entry_delay_abs_bps"),
                    decision_to_entry_seconds=trade_result.get(
                        "decision_to_entry_seconds"
                    ),
                    signal_to_entry_seconds=trade_result.get(
                        "signal_to_entry_seconds"
                    ),
                    expected_friction_drag_bps=trade_result.get(
                        "expected_friction_drag_bps"
                    ),
                    spread_proxy_pct=trade_result.get("spread_proxy_pct"),
                    orderbook_snapshot=trade_result.get("orderbook_snapshot"),
                    stop_price=trade_result.get("stop_price"),
                    stop_order_id=trade_result.get("stop_order_id"),
                    exchange_order_id=_order_identifier(trade_result.get("order")),
                    order_error_category=order_error_category,
                    actual_entry_price=trade_result.get("realized_entry_price"),
                    status="pending",
                    error=trade_result.get("error", ""),
                )
            else:
                logger.log_entry(
                    symbol=symbol,
                    side=side,
                    size=abs(size),
                    price=price,
                    predictions=predictions,
                    features=features_log,
                    mode=executor.mode,
                    strategy_id=strategy_id,
                    calibrated_score=float(decision["calibrated_score"]),
                    rank_threshold=float(decision["rank_threshold"]),
                    order_error_category=order_error_category,
                    lifecycle_event="entry_rejected",
                    status="failed",
                    error=trade_result.get("error", ""),
                )
            if (
                prediction_ledger is not None
                and (
                    trade_success
                    or _should_log_prediction_candidate(decision, policy=portfolio_policy)
                )
            ):
                prediction_ledger_rows.append(
                    _prediction_ledger_row(
                        decision,
                        timestamp=now_utc.isoformat(),
                        side=side,
                        portfolio_decision=(
                            "traded" if trade_success else "order_rejected"
                        ),
                        portfolio_reject_reason=(
                            None if trade_success else order_error_category
                        ),
                        execution_snapshot=execution_snapshot,
                        was_traded=trade_success,
                        trade_result=trade_result,
                    )
                )
            if trade_success:
                side_metrics["executed"] = int(side_metrics.get("executed", 0)) + 1
                total_entries_executed += 1
                entries_this_bar += 1
                results["trades"].append(
                    {
                        "symbol": symbol,
                        "side": side,
                        "size": size,
                        "price": price,
                        "result": trade_result,
                        "strategy_id": strategy_id,
                        "calibrated_score": float(decision["calibrated_score"]),
                        "estimated_hit_rate": chain_results.get("estimated_hit_rate"),
                        "estimated_ev_gross_return": chain_results.get(
                            "estimated_ev_gross_return"
                        ),
                        "estimated_ev_net_return": chain_results.get(
                            "estimated_ev_net_return"
                        ),
                        "estimated_ev_cost_bps": chain_results.get(
                            "estimated_ev_cost_bps"
                        ),
                        "decision_audit": trade_audit,
                    }
                )
            else:
                side_metrics["order_errors"] = (
                    int(side_metrics.get("order_errors", 0)) + 1
                )
            if side:
                results["side_metrics"][side] = dict(side_metrics)

    if portfolio_mgr is not None:
        try:
            open_df = portfolio_mgr.get_open_positions_summary()
            concurrent = int(len(open_df)) if isinstance(open_df, pd.DataFrame) else 0
            max_pos = int(getattr(portfolio_mgr, "max_positions", 0))
            util = (float(concurrent) / float(max_pos)) if max_pos > 0 else float("nan")
            tprint(
                f"Concurrent positions snapshot: open={concurrent}, max={max_pos}, utilization={util:.3f}"
            )
            _log_concurrent_positions_snapshot(portfolio_mgr, label="end")
        except Exception as exc:
            tprint(f"Concurrent positions snapshot failed: {exc}")
    if prediction_ledger is not None and prediction_ledger_rows:
        try:
            prediction_ledger.append_rows(prediction_ledger_rows)
            tprint(
                "Prediction ledger appended: "
                f"rows={len(prediction_ledger_rows)} "
                f"path={prediction_ledger.path}"
            )
            try:
                live_root = Path(
                    runtime_config.get("live_data_root")
                    or runtime_config.get("data_root")
                    or "data"
                )
                artifact_root = Path(runtime_config.get("data_root") or live_root)
                run_id = str(runtime_config.get("run_id") or "")
                benchmark_dir = (
                    artifact_root / "artifacts" / run_id / "drift_benchmarks"
                    if run_id
                    else None
                )
                drift_recap = write_live_drift_recap(
                    ledger_path=prediction_ledger.path,
                    output_root=live_root / "live_state" / "drift_monitoring",
                    benchmark_dir=benchmark_dir,
                    asof_ts=pd.Timestamp.now(tz="UTC"),
                    model_run_id=run_id,
                    policy_run_id=run_id,
                )
                if drift_recap.get("reason") not in {None, ""}:
                    tprint(f"Live drift recap skipped: {drift_recap.get('reason')}")
                else:
                    tprint(
                        "Live drift recap updated: "
                        f"rows={drift_recap.get('scored_metric_rows')} "
                        f"regime_features={drift_recap.get('regime_feature_rows')}"
                    )
            except Exception as exc:
                tprint(f"Warning: live drift recap update failed: {exc}")
        except Exception as exc:
            tprint(f"Warning: prediction ledger append failed: {exc}")
    _emit_structured_event("ORDER_HEALTH", results["order_error_summary"])
    timer.mark("complete")
    return results


def run_inference_loop(
    config: Dict[str, Any],
    symbols: List[str],
    lookback_periods: int,
    inference_interval: int,
    max_candidates: int,
    capital: float,
    mode: str,
):
    """Run the inference loop.

    Args:
        config: Inference configuration
        symbols: Trading symbols
        lookback_periods: Lookback periods for features
        inference_interval: Interval between inferences
        max_candidates: Maximum candidates per side
        capital: Starting capital
        mode: "live" or "shadow"
    """
    tprint(f"Starting inference loop in {mode} mode")
    tprint(f"Symbols: {symbols}")
    tprint(f"Interval: {inference_interval}s")

    # Extract config
    model_bundle = config["model_bundle"]
    full_state = config["full_state"]
    thresholds = config["thresholds"]
    run_id = config["run_id"]
    data_root = str(config.get("data_root", "data"))
    model_artifact_run_id = str(config.get("model_artifact_run_id") or run_id)
    policy_artifact_run_id = str(config.get("policy_artifact_run_id") or run_id)
    portfolio_policy = load_portfolio_policy_config(
        data_root=data_root,
        run_id=policy_artifact_run_id,
        runtime_cfg=config,
        require_artifact=_is_live_test_mode(mode) or str(mode).lower() == "live",
    )
    portfolio_policy = _apply_live_test_threshold_relaxation(
        portfolio_policy,
        live_test_mode=_is_live_test_mode(mode),
    )
    parity_contract = config.get("training_live_parity_contract")
    if not isinstance(parity_contract, dict) or not parity_contract:
        parity_contract = load_training_live_parity_contract(
            data_root=data_root,
            run_id=model_artifact_run_id,
            require=_is_live_test_mode(mode) or str(mode).lower() == "live",
        )
        config["training_live_parity_contract"] = parity_contract
    policy_strategy_filter = resolve_deployment_strategy_filter(
        data_root,
        policy_artifact_run_id,
    )
    prefer_policy_contract = (
        policy_artifact_run_id != model_artifact_run_id
        and policy_strategy_filter is not None
    )
    accepted_strategies = _resolve_active_strategy_filter_for_policy(
        parity_contract=parity_contract,
        portfolio_policy=portfolio_policy,
        policy_strategy_filter=policy_strategy_filter,
        prefer_policy_contract=prefer_policy_contract,
    )
    config["policy_strategy_contract_overrides_parity"] = bool(
        prefer_policy_contract
    )
    validate_portfolio_strategy_contract(
        portfolio_policy,
        sorted(accepted_strategies) if accepted_strategies is not None else None,
        strict=True,
    )
    validate_training_live_parity_contract(
        parity_contract,
        active_strategy_ids=(
            sorted(accepted_strategies) if accepted_strategies is not None else []
        ),
        data_root=data_root,
        run_id=model_artifact_run_id,
        strict=not prefer_policy_contract,
    )
    strategy_asset_exclusions = load_strategy_asset_exclusion_filter(
        data_root,
        policy_artifact_run_id,
    )

    # Initialize orchestrator
    orchestrator = ModelOrchestrator(model_bundle, full_state)

    # Initialize exchange (for live mode)
    exchange = None
    if mode == "live":
        exchange = make_exchange(config.get("market_mode", "spot"))

    # Initialize executor
    executor = TradeExecutor(
        mode=mode,
        exchange=exchange,
        capital=capital,
        bucket_params=_build_executor_bucket_params(config),
        config=config,
    )

    # Initialize logger
    logger = TradeLogger(run_id=run_id)

    # Main loop
    iteration = 0
    while True:
        iteration += 1
        tprint(f"\n=== Iteration {iteration} ===")

        try:
            # Fetch data
            tprint("Fetching OHLCV data...")
            panel = fetch_and_build_panel(
                symbols=symbols,
                exchange=exchange,
                timeframe="1h",
                lookback_periods=lookback_periods,
            )

            # Safely check panel data
            panel_close = panel.get("close")
            try:
                has_close = (
                    panel_close is not None
                    and isinstance(panel_close, (pd.DataFrame, pd.Series))
                    and not (hasattr(panel_close, "empty") and panel_close.empty)
                )
            except Exception as e:
                tprint(
                    f"Error checking panel_close.empty: {e}, type: {type(panel_close)}"
                )
                has_close = False

            if not has_close:
                tprint("Warning: No data fetched, skipping iteration")
                time.sleep(inference_interval)
                continue

            # Generate features
            tprint("Generating features...")
            feats = generate_features(
                panel=panel,
                basket_syms=symbols,
            )

            if not feats:
                tprint("Warning: No features generated, skipping iteration")
                time.sleep(inference_interval)
                continue

            # Apply feature normalization
            tprint("Applying feature normalization...")
            import gc

            transformer = CausalFeatureTransformer(
                winsor_qt=0.02,
                roll_window=24 * 30,
                cache_dir="./cache/feature_transforms",
                enable_cache=False,
            )

            skip_transform_set = {
                "liq_state",
                "sin_hod",
                "cos_hod",
                "sin_dow",
                "cos_dow",
                "range_24h_pct",
                "range_12h_pct",
                "volatility_zscore",
                "breakout_24h",
                "draw_sym_10h",
                "draw_extreme_10h",
                "G_VOL_LIQ_GT1",
                "G_VOL_LIQ_GT2",
                "G_VOL_LIQ_GT3",
                "G_LIQ_GOOD",
                "G_LIQ_GREAT",
                "G_LIQ_EXCEL",
                "mtf_divergence",
                "meta_alignment",
                "rsi_z",
                "dist_ema_fast_z",
                "dist_vwap_norm_z",
                "flow_persistence_z",
                "excess_6h_z",
                "vol_z_z",
                "atr_expansion_z",
                "coherence_24_z",
                "overext_surprise",
                "blowoff_risk_surprise",
                "exh_qual_surprise",
                "dist_vwap_resid",
                "dist_ema_fast_resid",
                "trend_pct_resid",
            }

            # Add gated feature patterns
            gate_windows = [6, 12, 24, 48, 72, 120]
            for w in gate_windows:
                for prefix in [
                    "s",
                    "reject",
                    "tf_qual",
                    "mr_qual",
                    "vol_z",
                    "liquidity",
                ]:
                    for suffix in [
                        "mean",
                        "std",
                        "z",
                        "pct",
                        "bin3",
                        "gt25",
                        "gt50",
                        "gt66",
                        "gt75",
                    ]:
                        skip_transform_set.add(f"{prefix}_{suffix}_{w}")

            def _is_boolean_like_feature(arr_like) -> bool:
                arr = np.asarray(arr_like, dtype=np.float32)
                if arr.size == 0:
                    return False
                finite = arr[np.isfinite(arr)]
                if finite.size == 0:
                    return False
                if finite.min() < 0.0 or finite.max() > 1.0:
                    return False
                rounded = np.round(finite)
                return bool(np.all(np.abs(finite - rounded) <= 1e-6))

            feat_keys_list = list(feats.keys())
            for k in feat_keys_list:
                if (
                    k.startswith("cs_rank_")
                    or k.startswith("cs_rz_")
                    or k.startswith("ts_pct_")
                ):
                    skip_transform_set.add(k)
                else:
                    arr = np.asarray(feats[k], dtype=np.float32)
                    if _is_boolean_like_feature(arr):
                        skip_transform_set.add(k)

            tprint(
                f"CausalTransform workset: {len(feats) - len(skip_transform_set)} transform, {len(skip_transform_set)} skipped"
            )

            feats = transformer.transform_batch(
                feats, skip_keys=skip_transform_set, chunk_size=50
            )
            del transformer
            gc.collect()

            # Final check for Inf/NaN. Keep non-finite values visible to the
            # strict model-contract gate; do not neutral-fill trained inputs.
            for k in list(feats.keys()):
                arr = np.asarray(feats[k], dtype=np.float32)
                if not np.isfinite(arr).all():
                    n_bad = (~np.isfinite(arr)).sum()
                    tprint(
                        f"  WARNING: {k} has {n_bad} non-finite values; preserving "
                        "NaN for strict model-contract gating"
                    )
                    arr = np.where(np.isfinite(arr), arr, np.nan).astype(
                        np.float32,
                        copy=False,
                    )

                # Ensure feats[k] is a dataframe if it was originally
                if isinstance(feats[k], pd.DataFrame):
                    feats[k] = pd.DataFrame(
                        arr, index=feats[k].index, columns=feats[k].columns
                    )
                elif isinstance(feats[k], pd.Series):
                    feats[k] = pd.Series(arr, index=feats[k].index, name=feats[k].name)
                elif isinstance(feats[k], np.ndarray) and not isinstance(
                    feats.get(k), (pd.DataFrame, pd.Series)
                ):
                    # If transform_batch returned raw numpy array, convert back to DataFrame if possible
                    # We need index and columns from somewhere. Let's use close index/cols
                    panel_close = panel["close"]
                    try:
                        feats[k] = pd.DataFrame(
                            arr,
                            index=panel_close.index[-arr.shape[0] :],
                            columns=panel_close.columns,
                        )
                    except Exception as e:
                        tprint(f"Warning: could not cast {k} back to DataFrame: {e}")
                        feats[k] = arr
                else:
                    feats[k] = arr

            # Run inference step
            results = run_inference_step(
                orchestrator=orchestrator,
                panel=panel,
                feats=feats,
                thresholds=thresholds,
                executor=executor,
                logger=logger,
                max_candidates=max_candidates,
                accepted_strategies=accepted_strategies,
                strategy_asset_exclusions=strategy_asset_exclusions,
                portfolio_policy=portfolio_policy,
            )

            tprint(f"Executed {len(results['trades'])} trades")

        except KeyboardInterrupt:
            tprint("\nInterrupted by user")
            break
        except Exception as e:
            tprint(f"Error in inference loop: {e}")
            import traceback

            tprint(traceback.format_exc())

        # Wait for next iteration
        time.sleep(inference_interval)

    tprint(f"\nInference loop ended. Log file: {logger.get_log_path()}")


def _record_portfolio_close_from_trade(
    portfolio_mgr: Optional[PortfolioManager],
    *,
    closed_trade: Dict[str, Any],
) -> None:
    """Synchronize PortfolioManager state after an executor-managed close."""
    if portfolio_mgr is None or not isinstance(closed_trade, dict):
        return
    symbol = str(closed_trade.get("symbol") or "")
    if not symbol:
        return
    try:
        exit_price = float(closed_trade.get("exit_price"))
    except (TypeError, ValueError):
        return
    if not np.isfinite(exit_price) or exit_price <= 0:
        return
    try:
        exit_time_raw = closed_trade.get("exit_time")
        exit_time = (
            pd.Timestamp(exit_time_raw)
            if exit_time_raw is not None
            else pd.Timestamp.now(tz="UTC")
        )
        if exit_time.tzinfo is None:
            exit_time = exit_time.tz_localize("UTC")
        else:
            exit_time = exit_time.tz_convert("UTC")
        result = portfolio_mgr.record_position_close(
            symbol=symbol,
            exit_price=exit_price,
            exit_time=exit_time,
            exit_reason=str(closed_trade.get("reason") or "closed"),
        )
        if result is not None:
            tprint(f"[PortfolioManager] Synced closed position for {symbol}")
    except Exception as exc:
        tprint(
            f"Portfolio close sync failed for {symbol}: "
            f"{classify_api_error(exc)}: {exc}"
        )


def _log_closed_trade_event(
    trade_logger: Optional[TradeLogger],
    *,
    closed_trade: Dict[str, Any],
    config: Dict[str, Any],
) -> None:
    """Persist a filled/closed lifecycle event to CSV and SQLite."""
    if trade_logger is None:
        return
    try:
        side = str(closed_trade.get("side") or "")
        scalar_context = {
            key: value
            for key, value in closed_trade.items()
            if value is None
            or isinstance(value, (str, int, float, bool, np.integer, np.floating))
            or isinstance(value, (pd.Timestamp, datetime))
        }
        holding_time_hours = _safe_float(closed_trade.get("holding_time_hours"))
        if not np.isfinite(holding_time_hours):
            holding_time_hours = _holding_time_hours(
                closed_trade.get("entry_time"),
                closed_trade.get("exit_time"),
            )
        scalar_context.update(
            {
                "run_id": config.get("run_id"),
                "lifecycle_event": "exit_filled",
                "strategy_id": closed_trade.get("strategy_id"),
                "actual_entry_price": closed_trade.get("entry_price"),
                "actual_exit_price": closed_trade.get("exit_price"),
                "realized_exit_price": closed_trade.get("exit_price"),
                "exit_reason": closed_trade.get("reason"),
                "exchange_order_id": closed_trade.get("close_order_id"),
                "holding_time_hours": holding_time_hours,
            }
        )
        trade_logger.log_trade_legacy(
            symbol=str(closed_trade.get("symbol") or ""),
            side=side,
            action="exit",
            size=float(_safe_float(closed_trade.get("filled"), 0.0) or 0.0),
            price=_safe_float(closed_trade.get("entry_price"), np.nan),
            mode=str(config.get("mode", "live")),
            status="closed",
            context=scalar_context,
        )
    except Exception as exc:
        tprint(
            f"Trade close logging failed for {closed_trade.get('symbol')}: "
            f"{classify_api_error(exc)}: {exc}"
        )


def _monitor_active_position_price_action(
    executor: TradeExecutor,
    *,
    exchange: Optional[Any] = None,
    now: Optional[pd.Timestamp] = None,
    config: Optional[Dict[str, Any]] = None,
    portfolio_mgr: Optional[PortfolioManager] = None,
    trade_logger: Optional[TradeLogger] = None,
    sheets_exporter: Optional[GoogleSheetsTradeExporter] = None,
) -> Dict[str, Dict[str, Any]]:
    """Monitor active positions and apply closed-5m trailing/stop updates."""
    statuses: Dict[str, Dict[str, Any]] = {}
    active_positions = (
        executor.get_active_positions()
        if hasattr(executor, "get_active_positions")
        else {}
    )
    if not active_positions:
        _emit_structured_event(
            "INFERENCE_MONITOR_HEARTBEAT",
            {
                "timestamp": pd.Timestamp(
                    now if now is not None else pd.Timestamp.now(tz="UTC")
                ).isoformat(),
                "active_positions": 0,
                "symbols": [],
                "order_status_checks": 0,
                "price_action_checks": 0,
                "stop_replacements": 0,
                "closed_positions": 0,
                "errors": 0,
            },
        )
        return statuses

    if hasattr(executor, "monitor_orders_once"):
        try:
            statuses.update(executor.monitor_orders_once())
            cfg = dict(config or getattr(executor, "config", {}) or {})
            for status in statuses.values():
                closed_trade = (
                    status.get("closed_trade") if isinstance(status, dict) else None
                )
                if isinstance(closed_trade, dict):
                    _record_portfolio_close_from_trade(
                        portfolio_mgr, closed_trade=closed_trade
                    )
                    _log_closed_trade_event(
                        trade_logger,
                        closed_trade=closed_trade,
                        config=cfg,
                    )
                    status["trade_close_email"] = _send_trade_close_email(
                        closed_trade=closed_trade,
                        config=cfg,
                    )
                    if trade_logger is not None:
                        _maybe_export_google_sheets(
                            sheets_exporter=sheets_exporter,
                            trade_logger=trade_logger,
                            executor=executor,
                            force=True,
                        )
        except Exception as exc:
            tprint(
                f"  Error monitoring order statuses: {classify_api_error(exc)}: {exc}"
            )
    active_positions = (
        executor.get_active_positions()
        if hasattr(executor, "get_active_positions")
        else active_positions
    )

    now_ts = pd.Timestamp(now if now is not None else pd.Timestamp.now(tz="UTC"))
    if now_ts.tzinfo is None:
        now_ts = now_ts.tz_localize("UTC")
    else:
        now_ts = now_ts.tz_convert("UTC")

    cfg = dict(config or getattr(executor, "config", {}) or {})
    monitor_delay = float(
        cfg.get(
            "five_minute_ohlcv_delay_seconds",
            cfg.get("fifteen_minute_ohlcv_delay_seconds", 5.0),
        )
    )
    latest_closed_5m = _latest_closed_candle_start(
        now_ts,
        timeframe_minutes=5,
        delay_seconds=monitor_delay,
    )
    cached_5m: Dict[str, pd.DataFrame] = {}
    price_action_checks = 0
    stop_replacements = 0
    closed_positions = 0
    errors = 0
    if exchange is None and hasattr(executor, "fetch_5m_ohlcv_for_positions"):
        try:
            cached_5m = executor.fetch_5m_ohlcv_for_positions()
        except Exception as exc:
            errors += 1
            tprint(
                f"  Error reading cached shadow 5m data: "
                f"{classify_api_error(exc)}: {exc}"
            )

    tprint(f"Monitoring {len(active_positions)} active positions for price action...")
    for symbol, position_state in active_positions.items():
        try:
            status = statuses.setdefault(symbol, {})
            if str(
                status.get("status") or ""
            ).lower() == "missing_stop_order" or not position_state.get(
                "stop_order_id"
            ):
                retry_stop = getattr(executor, "retry_missing_protective_stop", None)
                if callable(retry_stop):
                    retry_result = retry_stop(symbol, position_state)
                    status["missing_stop_retry"] = retry_result
                    refreshed = (
                        executor.get_position(symbol)
                        if hasattr(executor, "get_position")
                        else None
                    )
                    if isinstance(refreshed, dict):
                        position_state = refreshed
                    if isinstance(retry_result, dict) and retry_result.get("success"):
                        status["status"] = "open"
            entry_time = position_state.get("entry_time") or position_state.get(
                "timestamp"
            )
            if entry_time is None:
                continue
            start_time = pd.Timestamp(entry_time)
            if start_time.tzinfo is None:
                start_time = start_time.tz_localize("UTC")
            else:
                start_time = start_time.tz_convert("UTC")
            last_eval_ts = position_state.get("last_5m_eval_ts")
            if last_eval_ts is not None:
                start_time = max(
                    start_time,
                    pd.Timestamp(last_eval_ts) - pd.Timedelta(minutes=15),
                )
            start_time = max(start_time, now_ts - pd.Timedelta(hours=8))
            end_time = latest_closed_5m
            if start_time >= end_time:
                continue

            ohlcv_5m: Any = None
            if exchange is not None:
                ohlcv_5m = hf_data_loader.fetch_specific_period(
                    exchange,
                    symbol,
                    "5m",
                    start_time,
                    end_time,
                    use_cache=True,
                )
            else:
                ohlcv_5m = cached_5m.get(symbol)

            if (
                ohlcv_5m is None
                or not isinstance(ohlcv_5m, (pd.DataFrame, pd.Series))
                or (hasattr(ohlcv_5m, "empty") and ohlcv_5m.empty)
            ):
                continue

            bars = pd.DataFrame(ohlcv_5m)
            bars = bars[bars.index <= latest_closed_5m]
            if bars.empty:
                continue
            before_stop = float(position_state.get("stop_price", np.nan))
            position_state["ohlcv_5m_latest"] = bars
            eval_result = _evaluate_oco_policy(symbol, position_state, bars, executor)
            after_position = (
                executor.get_position(symbol)
                if hasattr(executor, "get_position")
                else None
            )
            price_action_checks += 1
            status = statuses.setdefault(symbol, {})
            if after_position is None:
                closed_positions += 1
                status["price_action"] = {
                    "status": "closed",
                    "bars_evaluated": int(len(bars)),
                    "stop_price_before": before_stop,
                }
                closed_trade = (
                    eval_result.get("closed_trade")
                    if isinstance(eval_result, dict)
                    else None
                )
                if isinstance(closed_trade, dict):
                    status["closed_trade"] = closed_trade
                    _record_portfolio_close_from_trade(
                        portfolio_mgr, closed_trade=closed_trade
                    )
                    _log_closed_trade_event(
                        trade_logger,
                        closed_trade=closed_trade,
                        config=dict(config or getattr(executor, "config", {}) or {}),
                    )
                    status["trade_close_email"] = _send_trade_close_email(
                        closed_trade=closed_trade,
                        config=dict(config or getattr(executor, "config", {}) or {}),
                    )
                    if trade_logger is not None:
                        _maybe_export_google_sheets(
                            sheets_exporter=sheets_exporter,
                            trade_logger=trade_logger,
                            executor=executor,
                            force=True,
                        )
                continue
            after_stop = float(after_position.get("stop_price", np.nan))
            status["price_action"] = {
                "status": "updated",
                "bars_evaluated": int(len(bars)),
                "stop_price_before": before_stop,
                "stop_price_after": after_stop,
                "peak_price": after_position.get("peak_price"),
                "mfe": after_position.get("mfe"),
                "current_price": after_position.get("current_price"),
                "current_price_source": after_position.get("current_price_source"),
                "policy_entry_price": after_position.get("policy_entry_price"),
                "policy_entry_price_source": after_position.get(
                    "policy_entry_price_source"
                ),
                "realized_entry_price": after_position.get("entry_price"),
                "last_5m_eval_ts": after_position.get("last_5m_eval_ts"),
            }
            if np.isfinite(before_stop) and np.isfinite(after_stop):
                if abs(after_stop - before_stop) > 1e-12:
                    order_snapshot = status.get("order")
                    if isinstance(order_snapshot, dict):
                        order_snapshot["id"] = after_position.get(
                            "stop_order_id", order_snapshot.get("id")
                        )
                        order_snapshot["stopPrice"] = after_stop
                        order_snapshot["triggerPrice"] = after_stop
                        info = order_snapshot.get("info")
                        if isinstance(info, dict):
                            info["orderId"] = after_position.get(
                                "stop_order_id", info.get("orderId")
                            )
                            info["stopPrice"] = f"{after_stop:.12g}"
                    stop_replacements += 1
                    tprint(
                        f"  [STOP_LOSS] {symbol} stop updated "
                        f"{before_stop:.8f} -> {after_stop:.8f}"
                    )
        except Exception as exc:
            errors += 1
            tprint(
                f"  Error evaluating 5m price action for {symbol}: "
                f"{classify_api_error(exc)}: {exc}"
            )
            statuses.setdefault(symbol, {})["price_action_error"] = str(exc)
    order_status_errors = sum(
        1
        for status in statuses.values()
        if isinstance(status, dict)
        and (
            (
                status.get("fetch_order_error")
                and not bool(status.get("reconciled_after_error"))
            )
            or status.get("price_action_error")
            or status.get("error")
            or str(status.get("status", "")).startswith("unprotected_stop_")
        )
    )
    try:
        active_after = (
            executor.get_active_positions()
            if hasattr(executor, "get_active_positions")
            else active_positions
        )
    except Exception:
        active_after = active_positions
    if portfolio_mgr is not None:
        _sync_reconciled_positions_to_portfolio_manager(executor, portfolio_mgr)
    _emit_structured_event(
        "INFERENCE_MONITOR_HEARTBEAT",
        {
            "timestamp": now_ts.isoformat(),
            "active_positions": int(len(active_after)),
            "symbols": sorted(str(s) for s in active_after.keys()),
            "order_status_checks": int(len(statuses)),
            "price_action_checks": int(price_action_checks),
            "stop_replacements": int(stop_replacements),
            "closed_positions": int(closed_positions),
            "errors": int(errors + order_status_errors),
            "statuses": statuses,
        },
    )
    return statuses


def _emit_inference_heartbeat(
    *,
    current_time: pd.Timestamp,
    config: Dict[str, Any],
    download_symbols: List[str],
    tradable_symbols: List[str],
    long_candidates: List[str],
    short_candidates: List[str],
    features: Dict[str, pd.DataFrame],
    strategy_candidate_masks: Dict[str, List[str]],
    results: Dict[str, Any],
    data_fetcher: DataFetcher,
    portfolio_mgr: Optional[PortfolioManager],
    executor: TradeExecutor,
) -> None:
    """Emit one structured summary for the hourly inference decision loop."""
    feature_frames = 0
    feature_non_empty = 0
    feature_symbols: set[str] = set()
    feature_rows: list[int] = []
    for frame in (features or {}).values():
        feature_frames += 1
        if isinstance(frame, pd.DataFrame) and not frame.empty:
            feature_non_empty += 1
            feature_rows.append(int(len(frame)))
            feature_symbols.update(str(c) for c in frame.columns)

    active_positions = (
        executor.get_active_positions()
        if hasattr(executor, "get_active_positions")
        else {}
    )
    portfolio_state: Dict[str, Any] = {}
    if portfolio_mgr is not None:
        try:
            portfolio_state = dict(portfolio_mgr.get_portfolio_state())
        except Exception as exc:
            portfolio_state = {"error": str(exc)}

    payload = {
        "timestamp": pd.Timestamp(current_time).isoformat(),
        "run_id": config.get("run_id"),
        "mode": config.get("mode"),
        "download_symbols": int(len(download_symbols)),
        "tradable_symbols": int(len(tradable_symbols)),
        "candidate_counts": {
            "long": int(len(long_candidates)),
            "short": int(len(short_candidates)),
            "total": int(len(long_candidates) + len(short_candidates)),
        },
        "lgbm_strategy_masks": {
            "loaded": int(len(strategy_candidate_masks or {})),
            "non_empty": int(
                sum(1 for values in (strategy_candidate_masks or {}).values() if values)
            ),
            "passed_symbols_total": int(
                sum(len(values) for values in (strategy_candidate_masks or {}).values())
            ),
            "per_strategy": _strategy_mask_count_diagnostics(
                strategy_candidate_masks,
                None,
                tradable_symbols,
            ),
        },
        "features": {
            "frames": feature_frames,
            "non_empty_frames": feature_non_empty,
            "symbols": int(len(feature_symbols)),
            "row_min": int(min(feature_rows)) if feature_rows else 0,
            "row_max": int(max(feature_rows)) if feature_rows else 0,
        },
        "side_metrics": results.get("side_metrics", {}),
        "score_distributions": results.get("score_distributions", {}),
        "trades": int(len(results.get("trades", []) or [])),
        "order_error_summary": results.get("order_error_summary", {}),
        "data_fetch_errors": dict(getattr(data_fetcher, "api_error_counts", {}) or {}),
        "dead_letter_symbols": int(
            len(getattr(data_fetcher, "dead_letter_symbols", {}) or {})
        ),
        "active_positions": {
            "count": int(len(active_positions or {})),
            "symbols": sorted(str(s) for s in (active_positions or {}).keys()),
        },
        "portfolio_state": portfolio_state,
    }
    _emit_structured_event("INFERENCE_HEARTBEAT", payload)


def main():
    import argparse

    _load_local_env_if_present()
    _configure_numba_threading_layer()
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="Run live trading mode")
    parser.add_argument(
        "--live-test",
        action="store_true",
        help=(
            "Run live test mode: production decision path with 5-10 USDC "
            "quote-size clamp"
        ),
    )
    parser.add_argument(
        "--shadow",
        action="store_true",
        default=True,
        help="Run shadow trading mode (default)",
    )
    parser.add_argument("--symbols", nargs="+", default=None, help="Symbols to trade")
    parser.add_argument(
        "--inference-interval",
        type=int,
        default=900,
        help="Inference interval in seconds (default: 900 = 15 minutes)",
    )
    parser.add_argument(
        "--challenger-interval",
        type=int,
        default=120,
        help="Position monitor interval in seconds (default: 120 = 2 min)",
    )
    parser.add_argument(
        "--lookback-hours",
        type=int,
        default=24 * 60,
        help="Lookback hours for features",
    )
    parser.add_argument(
        "--execution-account",
        choices=["spot", "margin", "perps"],
        default=DEFAULT_EXECUTION_ACCOUNT,
        help=f"Execution account for live orders (default: {DEFAULT_EXECUTION_ACCOUNT})",
    )
    parser.add_argument(
        "--market-mode",
        choices=["spot", "perps"],
        default=DEFAULT_MARKET_MODE,
        help="Market mode for data, exchange calls, and wallet dispatch.",
    )
    parser.add_argument(
        "--perps",
        action="store_true",
        help="Shortcut for --market-mode perps --execution-account perps.",
    )
    parser.add_argument(
        "--margin-mode",
        choices=["cross", "isolated"],
        default=DEFAULT_MARGIN_MODE,
        help="Margin mode when --execution-account margin is used",
    )
    parser.add_argument(
        "--live-quote-currency",
        default=None,
        help=(
            "Quote currency to trade/download at inference time "
            f"(default: {DEFAULT_LIVE_QUOTE_CURRENCY})"
        ),
    )
    parser.add_argument(
        "--max-position-pct",
        type=float,
        default=0.15,
        help="Maximum fraction of portfolio equity per position (default: 0.15)",
    )
    parser.add_argument(
        "--data-root",
        default=None,
        help="Artifact data root containing trained models and policy outputs.",
    )
    parser.add_argument(
        "--live-data-root",
        default=None,
        help=(
            "Exchange-scoped root for live OHLCV, funding, orderbook, and runtime state. "
            "Defaults to <data-root>/exchanges/<exchange-id>."
        ),
    )
    parser.add_argument(
        "--prediction-ledger-path",
        default=None,
        help=(
            "Explicit prediction ledger parquet path. Overrides "
            "EPM_RUN_SCOPED_PREDICTION_LEDGER."
        ),
    )
    parser.add_argument(
        "--run-scoped-prediction-ledger",
        action="store_true",
        help=(
            "Write prediction ledger rows under "
            "<live-data-root>/live_state/prediction_ledgers/<run-id>/."
        ),
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Artifact run id to load. Defaults to latest when omitted.",
    )
    parser.add_argument(
        "--model-artifact-run-id",
        default=None,
        help=(
            "Model artifact run id to load. Defaults to EPM_MODEL_ARTIFACT_RUN_ID, "
            "then final_model_fit_manifest.json, then --run-id."
        ),
    )
    parser.add_argument(
        "--policy-artifact-run-id",
        default=None,
        help="Policy artifact run id to load. Defaults to EPM_POLICY_ARTIFACT_RUN_ID or --run-id.",
    )
    parser.add_argument(
        "--run-once",
        action="store_true",
        help="Run one inference batch and exit after optional daily reporting.",
    )
    parser.add_argument(
        "--allow-late-entries",
        action="store_true",
        help=(
            "Temporarily allow new entries even when the latest closed hourly "
            "context is older than the normal freshness window."
        ),
    )
    args = parser.parse_args()

    # Initialize components
    market_mode = "perps" if args.perps else _normalise_market_mode(args.market_mode)
    data_root = args.data_root or ("data_perp" if market_mode == "perps" else None)
    config = load_inference_config(
        data_root=data_root,
        run_id=args.run_id,
        market_mode=market_mode,
        model_artifact_run_id=args.model_artifact_run_id,
        policy_artifact_run_id=args.policy_artifact_run_id,
    )
    config["market_mode"] = market_mode
    config["execution_account"] = (
        "perps" if market_mode == "perps" else args.execution_account
    )
    config["margin_mode"] = args.margin_mode
    config["live_quote_currency"] = str(
        args.live_quote_currency
        or os.environ.get("EPM_LIVE_QUOTE_CURRENCY")
        or DEFAULT_LIVE_QUOTE_CURRENCY
        or "USDC"
    ).upper()
    config["max_position_pct"] = float(args.max_position_pct)
    if args.live_test:
        config["mode"] = "live-test"
    else:
        config["mode"] = "live" if args.live else "shadow"
    config["allow_late_entries"] = bool(
        args.allow_late_entries or _env_flag("EPM_ALLOW_LATE_ENTRIES")
    )
    if config["allow_late_entries"]:
        tprint(
            "WARNING: late-entry override enabled; entries may be placed with "
            "hourly/15m context outside the normal freshness window."
        )
    neutral_fill_nonfinite = _env_flag(
        "EPM_STRICT_FEATURE_PARITY_NEUTRAL_FILL_NONFINITE",
        bool(config.get("strict_feature_parity_neutral_fill_nonfinite", False)),
    )
    config["strict_feature_parity_neutral_fill_nonfinite"] = neutral_fill_nonfinite
    runtime_cfg = dict(config.get("runtime_cfg") or get_runtime_cfg())
    runtime_cfg["use_perps"] = config["market_mode"] == "perps"
    runtime_cfg["market_mode"] = config["market_mode"]
    runtime_cfg["data_root"] = config["data_root"]
    runtime_cfg.setdefault(
        "feature_portability_mode",
        "cross_asset_portable",
    )
    runtime_cfg.setdefault("feature_portability_strict", True)
    runtime_cfg["strict_feature_parity_neutral_fill_nonfinite"] = (
        neutral_fill_nonfinite
    )
    config["runtime_cfg"] = runtime_cfg
    config.setdefault(
        "cross_margin_dust_quote_threshold",
        2.5 if _is_live_test_mode(config["mode"]) else 5.0,
    )
    exchange = make_exchange(config["market_mode"])
    config["artifact_data_root"] = str(config["data_root"])
    config["live_data_root"] = _resolve_live_data_root(
        artifact_data_root=str(config["data_root"]),
        exchange=exchange,
        market_mode=config["market_mode"],
        explicit_live_data_root=args.live_data_root,
    )
    Path(config["live_data_root"]).mkdir(parents=True, exist_ok=True)
    tprint(
        "Live data root resolved: "
        f"artifact_data_root={config['artifact_data_root']} "
        f"live_data_root={config['live_data_root']}"
    )
    config["prediction_ledger_path"] = args.prediction_ledger_path
    config["run_scoped_prediction_ledger"] = bool(
        args.run_scoped_prediction_ledger
        or _env_flag("EPM_RUN_SCOPED_PREDICTION_LEDGER", True)
    )
    runtime_bucket_params = _attach_runtime_bucket_params(config)
    model_artifact_run_id = str(
        config.get("model_artifact_run_id") or config["run_id"]
    )
    policy_artifact_run_id = str(
        config.get("policy_artifact_run_id") or config["run_id"]
    )
    model_bundle = load_full_state(model_artifact_run_id, config["data_root"])
    if isinstance(model_bundle, dict):
        model_bundle["bucket_params"] = runtime_bucket_params
        loaded_bundle = (
            model_bundle.get("bundle", {})
            if isinstance(model_bundle.get("bundle"), dict)
            else {}
        )
        for key in (
            "feature_transform_contract",
            "feature_transform_contract_hash",
            "feature_transform_manifest",
        ):
            value = model_bundle.get(key)
            if value is None:
                value = loaded_bundle.get(key)
            if value is not None:
                config[key] = value
                runtime_cfg[key] = value
        runtime_cfg.setdefault("bundle", loaded_bundle)
        config["runtime_cfg"] = runtime_cfg
    effective_model_bundle = _effective_runtime_model_bundle(model_bundle, config)
    validate_live_feature_contract(effective_model_bundle, strict=True)
    portfolio_policy = load_portfolio_policy_config(
        data_root=config["data_root"],
        run_id=policy_artifact_run_id,
        runtime_cfg=config,
        require_artifact=_is_live_test_mode(config["mode"])
        or str(config["mode"]).lower() == "live",
    )
    portfolio_policy = _apply_live_test_threshold_relaxation(
        portfolio_policy,
        live_test_mode=_is_live_test_mode(config["mode"]),
    )
    parity_contract = load_training_live_parity_contract(
        data_root=config["data_root"],
        run_id=model_artifact_run_id,
        require=_is_live_test_mode(config["mode"])
        or str(config["mode"]).lower() == "live",
    )
    config["training_live_parity_contract"] = parity_contract
    runtime_cfg["training_live_parity_contract"] = parity_contract
    portfolio_policy_path = (
        Path(config["data_root"])
        / "artifacts"
        / policy_artifact_run_id
        / "policy_params"
        / "optimized_portfolio_policy_config.json"
    )
    portfolio_policy_hash = "missing"
    if portfolio_policy_path.exists():
        portfolio_policy_hash = hashlib.sha256(
            portfolio_policy_path.read_bytes()
        ).hexdigest()[:16]
    tprint(
        "Optimized portfolio policy loaded: "
        f"path={portfolio_policy_path} hash={portfolio_policy_hash} "
        f"version={portfolio_policy.portfolio_policy_version} "
        f"strategy_contract={len(portfolio_policy.strategy_ids) or len(portfolio_policy.strategy_cores)}"
    )
    policy_strategy_filter = resolve_deployment_strategy_filter(
        config["data_root"],
        policy_artifact_run_id,
    )
    prefer_policy_contract = (
        policy_artifact_run_id != model_artifact_run_id
        and policy_strategy_filter is not None
    )
    accepted_strategies = _resolve_active_strategy_filter_for_policy(
        parity_contract=parity_contract,
        portfolio_policy=portfolio_policy,
        policy_strategy_filter=policy_strategy_filter,
        prefer_policy_contract=prefer_policy_contract,
    )
    config["policy_strategy_contract_overrides_parity"] = bool(
        prefer_policy_contract
    )
    runtime_cfg["policy_strategy_contract_overrides_parity"] = bool(
        prefer_policy_contract
    )
    validate_portfolio_strategy_contract(
        portfolio_policy,
        sorted(accepted_strategies) if accepted_strategies is not None else None,
        strict=True,
    )
    validate_training_live_parity_contract(
        parity_contract,
        active_strategy_ids=(
            sorted(accepted_strategies) if accepted_strategies is not None else []
        ),
        data_root=str(config["data_root"]),
        run_id=model_artifact_run_id,
        strict=not prefer_policy_contract,
    )
    tprint(
        "Training-live parity contract loaded: "
        f"path={parity_contract.get('_contract_path')} "
        f"hash={str(parity_contract.get('_contract_sha256', ''))[:16]} "
        f"strategies={len((parity_contract.get('strategy_contract') or {}).get('strategy_ids') or [])}"
    )
    validate_meta_feature_contract_artifact(
        config["data_root"],
        model_artifact_run_id,
        effective_model_bundle,
        accepted_strategies,
        strict=True,
    )
    config["model_artifact_run_id"] = model_artifact_run_id
    config["policy_artifact_run_id"] = policy_artifact_run_id
    config["feature_contract_hash"] = _meta_feature_contract_hash(
        config["data_root"], model_artifact_run_id
    )
    required_feature_keys = get_inference_required_feature_keys(
        effective_model_bundle,
        accepted_strategies,
    )
    use_legacy_sizer_calibration = str(
        os.getenv("EPM_INFERENCE_USE_SIMPLE_POSITION_SIZER_CALIBRATION", "0")
        or ""
    ).strip().lower() in {"1", "true", "yes", "on"}
    if use_legacy_sizer_calibration:
        from extreme_price_movements.simple_position_sizer import load_calibration_curves

        calibration_data = load_calibration_curves(
            config["data_root"], model_artifact_run_id
        )
    else:
        calibration_data = {}
        tprint(
            "simple_position_sizer calibration disabled; using raw model scores "
            "with simple_policy_optimiser rank references"
        )
    normalized_thresholds = _load_normalized_threshold_map(
        config["data_root"], policy_artifact_run_id
    )
    policy_selection_rules = _load_policy_selection_rules(
        config["data_root"], policy_artifact_run_id
    )
    lgbm_strategy_mask_rows = _load_lgbm_strategy_mask_rows(
        config["data_root"], policy_artifact_run_id, market_mode=config["market_mode"]
    )
    deployment_mask_coverage_error: str | None = None
    try:
        _validate_lgbm_strategy_mask_coverage(
            lgbm_strategy_mask_rows,
            accepted_strategies,
            policy_selection_rules,
        )
    except Exception as exc:
        deployment_mask_coverage_error = str(exc)
        tprint(
            "CRITICAL: deployment LGBM strategy mask coverage failed; live "
            "inference will start in monitor-only fail-closed mode until mask "
            f"artifacts are fixed: {deployment_mask_coverage_error}"
        )
    validate_calibration_artifacts(
        config["data_root"], model_artifact_run_id, calibration_data, strict=False
    )
    strategy_asset_exclusions = load_strategy_asset_exclusion_filter(
        config["data_root"], policy_artifact_run_id
    )
    deployment_model_coverage_error: str | None = None
    try:
        validate_deployment_model_coverage(
            effective_model_bundle,
            accepted_strategies,
            strict=True,
        )
    except Exception as exc:
        deployment_model_coverage_error = str(exc)
        tprint(
            "CRITICAL: deployment model coverage failed; live inference will "
            "start in monitor-only fail-closed mode until artifacts are fixed: "
            f"{deployment_model_coverage_error}"
        )
    if deployment_mask_coverage_error:
        previous_coverage_error = deployment_model_coverage_error
        deployment_model_coverage_error = (
            f"{previous_coverage_error}; {deployment_mask_coverage_error}"
            if previous_coverage_error
            else deployment_mask_coverage_error
        )

    # Initialize data fetcher with incremental updates
    data_fetcher = DataFetcher(
        exchange, config["live_data_root"], market_mode=config["market_mode"]
    )
    inference_defaults = get_inference_defaults()
    panel_warmup_hours = _required_tail_warmup_hours(
        lookback_hours=int(args.lookback_hours),
        trend_sma_hours=int(inference_defaults["trend_sma_hours"]),
        gate_vol_lookback_hours=int(inference_defaults["gate_vol_lookback_hours"]),
    )
    panel_lookback_hours = max(int(args.lookback_hours), panel_warmup_hours)
    live_decision_panel_lookback_hours = panel_lookback_hours
    if str(config.get("mode", "")).strip().lower() in {
        "live",
        "live-test",
        "live_test",
        "livetest",
        "paper",
        "shadow_live",
        "shadow",
    }:
        try:
            decision_cap_hours = int(
                config.get(
                    "live_decision_panel_lookback_hours",
                    os.getenv("EPM_LIVE_DECISION_PANEL_LOOKBACK_HOURS", 24 * 45),
                )
            )
        except Exception:
            decision_cap_hours = 24 * 45
        decision_cap_hours = max(24 * 32, int(decision_cap_hours))
        live_decision_panel_lookback_hours = min(
            int(panel_lookback_hours),
            decision_cap_hours,
        )
        if live_decision_panel_lookback_hours < panel_lookback_hours:
            tprint(
                "Live decision panel lookback capped: "
                f"{panel_lookback_hours}->{live_decision_panel_lookback_hours}h. "
                "Model scoring uses selected-feature caches; raw panel history is "
                "bounded for latest masks, market checks, and deterministic repairs."
            )

    # Step 9 universe split:
    # - download_symbols: full live exchange quote/margin universe, refreshed daily
    # - symbols: tradable subset restricted to the active training universe
    universe_state = resolve_inference_universes(
        exchange,
        data_root=config["data_root"],
        run_id=config["run_id"],
        explicit_symbols=args.symbols,
        accepted_strategy_ids=accepted_strategies,
        live_quote_currency=config["live_quote_currency"],
        market_mode=config["market_mode"],
    )
    download_symbols = list(universe_state["download_symbols"])
    symbols = list(universe_state["tradable_symbols"])
    feature_context_symbols = _training_context_symbols_for_live_universe(
        universe_state
    )
    if feature_context_symbols:
        before_download_n = len(download_symbols)
        download_symbols = sorted(set(download_symbols).union(feature_context_symbols))
        tprint(
            "Feature context universe aligned to training artifacts: "
            f"context={len(feature_context_symbols)} tradable={len(symbols)} "
            f"download={before_download_n}->{len(download_symbols)}"
        )
    if not symbols:
        tprint(
            "Warning: tradable universe is empty after training-universe restriction"
        )

    # Initialize on startup with historical data only when model coverage is valid.
    # In fail-closed mode we only reconcile/monitor existing exchange positions.
    if deployment_model_coverage_error:
        tprint(
            "Skipping startup historical data initialization because deployment "
            "model coverage failed; monitor-only mode does not need model panels."
        )
    else:
        tprint("Initializing with historical data...")
        data_fetcher.initialize_with_historical_data(
            download_symbols, lookback_hours=args.lookback_hours
        )

    # Initialize other components
    orchestrator = ModelOrchestrator(model_bundle, config)
    strategy_feature_contracts = _strategy_feature_contracts_from_orchestrator(
        orchestrator,
        lgbm_strategy_mask_rows,
    )
    if strategy_feature_contracts:
        tprint(
            "Resolved deployed base+meta decision feature contracts: "
            f"strategies={len(strategy_feature_contracts)} "
            f"features_by_strategy="
            f"{ {strategy_core_id(k): len(v) for k, v in strategy_feature_contracts.items()} }"
        )
    executor = TradeExecutor(
        mode=config["mode"],
        exchange=exchange,
        bucket_params=runtime_bucket_params,
        config=config,
    )
    logger = TradeLogger(run_id=str(config.get("run_id") or "latest"))
    sheets_exporter = GoogleSheetsTradeExporter.from_config(config)
    reconciliation_report = executor.reconcile_cross_margin_account()
    _write_margin_reconciliation_report(config, reconciliation_report)
    unimported_external_positions: list[dict[str, Any]] = []
    try:
        unimported_external_positions = [
            item
            for item in reconciliation_report.get("items", [])
            if str(item.get("classification", "")).startswith("external_")
            and not bool(item.get("imported_for_monitoring"))
        ]
        if unimported_external_positions:
            tprint(
                "Skipping pending trade-log absent reconciliation because "
                f"{len(unimported_external_positions)} external margin position(s) "
                "were not imported for monitoring."
            )
        else:
            logger.reconcile_pending_entries_absent(
                executor.get_active_positions().keys(),
                reason="absent_after_cross_margin_startup_reconciliation",
            )
    except Exception as exc:
        tprint(f"Warning: pending trade-log reconciliation failed: {exc}")
    daily_reporter = DailyDeploymentReporter(
        state_path=str(
            config.get("daily_report_state_path")
            or "extreme_price_movements/logs/daily_report_state.json"
        )
    )
    config["book_notional_multiplier"] = float(
        portfolio_policy.book_notional_multiplier
    )
    config["leverage_wallet_multiplier"] = float(
        portfolio_policy.leverage_wallet_multiplier
    )
    if hasattr(executor, "config") and isinstance(executor.config, dict):
        executor.config["book_notional_multiplier"] = float(
            portfolio_policy.book_notional_multiplier
        )
        executor.config["leverage_wallet_multiplier"] = float(
            portfolio_policy.leverage_wallet_multiplier
        )
    prediction_ledger_path = _resolve_prediction_ledger_path(
        live_data_root=config["live_data_root"],
        run_id=str(config["run_id"]),
        explicit_path=config.get("prediction_ledger_path"),
        run_scoped=bool(config.get("run_scoped_prediction_ledger", False)),
    )
    tprint(f"Prediction ledger path resolved: {prediction_ledger_path}")
    prediction_ledger = PredictionLedger(prediction_ledger_path)
    dynamic_performance_monitor = StrategyPerformanceMonitor(
        data_root=str(config["data_root"]),
        run_id=str(config["run_id"]),
        live_data_root=str(config["live_data_root"]),
        ledger_path=prediction_ledger.path,
        lookback_days=int(config.get("dynamic_strategy_performance_lookback_days", 21)),
        top_fraction=float(config.get("dynamic_strategy_performance_top_fraction", 0.40)),
        min_resolved=int(config.get("dynamic_strategy_performance_min_resolved", 20)),
    )
    market_kill_switch = MarketKillSwitch(
        Path(config["live_data_root"]) / "live_state" / "market_kill_switch.json"
    )
    strategy_kill_switch = StrategyKillSwitch(
        Path(config["live_data_root"]) / "live_state" / "strategy_kill_switches.json",
        observe_only=bool(config.get("strategy_kill_switch_observe_only", True)),
    )
    max_concurrent_per_strategy = (
        portfolio_policy.resolved_max_concurrent_per_strategy()
    )
    max_concurrent_per_strategy = min(
        max_concurrent_per_strategy,
        portfolio_policy.max_concurrent_positions,
    )
    max_concurrent_per_side = portfolio_policy.resolved_max_concurrent_per_side()
    max_concurrent_per_side = min(
        max_concurrent_per_side,
        portfolio_policy.max_concurrent_positions,
    )
    tprint(
        "Deployment concurrency policy: "
        f"max_positions={portfolio_policy.max_concurrent_positions} "
        f"max_per_strategy={max_concurrent_per_strategy} "
        f"max_per_side={max_concurrent_per_side} "
        f"max_wallet_allocation={portfolio_policy.max_total_wallet_allocation_pct:.2f} "
        f"book_notional_multiplier={portfolio_policy.book_notional_multiplier:.2f} "
        f"min_margin_level_after_entry={portfolio_policy.min_margin_level_after_entry:.2f} "
        f"max_position_pct={portfolio_policy.max_position_wallet_pct:.2f} "
        f"max_position_quote={portfolio_policy.max_position_quote_notional:.2f}"
    )
    portfolio_mgr = PortfolioManager.from_policy_config(
        portfolio_policy,
        cooldown_hours=0.0,
        max_same_side=max_concurrent_per_side,
        max_same_strategy=max_concurrent_per_strategy,
    )
    _apply_margin_metrics_to_portfolio_manager(
        reconciliation_report=reconciliation_report,
        portfolio_mgr=portfolio_mgr,
        exchange=exchange,
        config=config,
    )
    _sync_reconciled_positions_to_portfolio_manager(executor, portfolio_mgr)
    if unimported_external_positions:
        _apply_reconciliation_entry_gate(
            reconciliation_report=reconciliation_report,
            portfolio_mgr=portfolio_mgr,
        )
    if deployment_model_coverage_error:
        portfolio_mgr.trip_hard_limit(
            "deployment_model_coverage_failed: " + deployment_model_coverage_error
        )
        tprint(
            "New entries blocked because deployment model coverage failed; "
            "existing imported positions will continue to be monitored."
        )

    # Setup scheduling
    if args.run_once and args.challenger_interval > 0:
        tprint("Run-once mode: background challenger monitor disabled")
    if args.challenger_interval > 0 and not args.run_once:
        # Start background challenger monitoring thread
        challenger_thread = threading.Thread(
            target=run_challenger_monitor,
            args=(
                symbols,
                data_fetcher,
                orchestrator,
                executor,
                logger,
                config,
                args.challenger_interval,
                panel_lookback_hours,
                required_feature_keys,
                accepted_strategies,
                calibration_data,
                strategy_asset_exclusions,
                portfolio_mgr,
                sheets_exporter,
            ),
            daemon=True,
        )
        challenger_thread.start()

    # Main entry loop - run only after closed hourly candles. Open-position
    # monitoring is handled by the challenger thread on its own cadence.
    last_hourly_sync = None
    last_margin_reconciliation = pd.Timestamp.now(tz="UTC")
    margin_reconciliation_interval = pd.Timedelta(
        minutes=float(config.get("margin_reconciliation_interval_minutes", 60.0))
    )
    last_universe_refresh_day = pd.Timestamp.utcnow().floor("D")
    while True:
        try:
            loop_now = pd.Timestamp.now(tz="UTC")
            hourly_delay = float(
                os.environ.get(
                    "EPM_HOURLY_OHLCV_DELAY_SECONDS",
                    config.get("hourly_ohlcv_delay_seconds", 30.0),
                )
            )
            latest_closed_hour = _latest_closed_candle_start(
                loop_now,
                timeframe_minutes=60,
                delay_seconds=hourly_delay,
            )
            current_time = latest_closed_hour
            latest_closed_hour_close = latest_closed_hour + pd.Timedelta(hours=1)
            current_close = latest_closed_hour_close
            hourly_age_seconds = _closed_candle_age_seconds(
                loop_now,
                latest_closed_hour,
                timeframe_minutes=60,
            )
            max_entry_hourly_age_seconds = float(
                config.get("entry_hourly_max_staleness_seconds", 15.0 * 60.0)
            )
            hourly_entry_fresh = hourly_age_seconds <= max(
                max_entry_hourly_age_seconds,
                hourly_delay,
            )
            entry_context_fresh = bool(hourly_entry_fresh)
            tprint(
                f"\n=== Running inference after closed hourly candle "
                f"start={latest_closed_hour} close={latest_closed_hour_close} "
                f"age={hourly_age_seconds:.0f}s "
                f"fresh_for_entries={entry_context_fresh} ==="
            )
            did_hourly_refresh = False
            loop_timer = _StageTimer("live_entry_loop")

            if loop_now >= last_margin_reconciliation + margin_reconciliation_interval:
                try:
                    reconciliation_report = executor.reconcile_cross_margin_account()
                    _write_margin_reconciliation_report(config, reconciliation_report)
                    _apply_margin_metrics_to_portfolio_manager(
                        reconciliation_report=reconciliation_report,
                        portfolio_mgr=portfolio_mgr,
                        exchange=exchange,
                        config=config,
                    )
                    _sync_reconciled_positions_to_portfolio_manager(
                        executor,
                        portfolio_mgr,
                    )
                    _apply_reconciliation_entry_gate(
                        reconciliation_report=reconciliation_report,
                        portfolio_mgr=portfolio_mgr,
                    )
                    last_margin_reconciliation = loop_now
                    loop_timer.mark("margin_reconciliation")
                except Exception as exc:
                    tprint(f"Periodic cross-margin reconciliation failed: {exc}")

            current_day = pd.Timestamp.utcnow().floor("D")
            if current_day > last_universe_refresh_day and not args.symbols:
                universe_state = resolve_inference_universes(
                    exchange,
                    data_root=config["data_root"],
                    run_id=config["run_id"],
                    accepted_strategy_ids=accepted_strategies,
                    live_quote_currency=config["live_quote_currency"],
                    market_mode=config["market_mode"],
                )
                download_symbols[:] = universe_state["download_symbols"]
                symbols[:] = universe_state["tradable_symbols"]
                last_universe_refresh_day = current_day
                tprint(
                    "Daily live universe refresh complete: "
                    f"download={len(download_symbols)} tradable={len(symbols)}"
                )

            # Fetch full universe only for closed hourly candles.
            if deployment_model_coverage_error:
                tprint(
                    "Monitor-only fail-closed mode active: skipping data refresh, "
                    "feature generation, model scoring, and new entries because "
                    f"deployment model coverage failed: {deployment_model_coverage_error}"
                )
                _monitor_active_position_price_action(
                    executor,
                    exchange=executor.exchange,
                    now=current_time,
                    config=config,
                    portfolio_mgr=portfolio_mgr,
                    trade_logger=logger,
                    sheets_exporter=sheets_exporter,
                )
                _maybe_send_daily_deployment_report(
                    daily_reporter=daily_reporter,
                    exchange=exchange,
                    portfolio_mgr=portfolio_mgr,
                    trade_logger=logger,
                    config=config,
                )
                _maybe_export_google_sheets(
                    sheets_exporter=sheets_exporter,
                    trade_logger=logger,
                    executor=executor,
                    force=bool(args.run_once),
                )
                if args.run_once:
                    executor.shutdown()
                    break
                _sleep_until_next_candle_close(
                    timeframe_minutes=60,
                    delay_seconds=hourly_delay,
                )
                continue

            hourly_sync_due = (last_hourly_sync is None) or (
                latest_closed_hour > last_hourly_sync
            )
            late_entries_override = bool(config.get("allow_late_entries", False))
            stale_entry_gap_enabled = bool(
                config.get("allow_stale_entries_if_price_gap_within_limit", True)
            )
            hard_signal_close_gate_seconds = _max_signal_close_to_entry_seconds(config)
            hard_signal_close_gate_exceeded = bool(
                hard_signal_close_gate_seconds >= 0.0
                and hourly_age_seconds > hard_signal_close_gate_seconds
            )
            try:
                stale_entry_max_abs_signal_gap_bps = float(
                    config.get("stale_entry_max_abs_signal_gap_bps", 50.0)
                )
            except (TypeError, ValueError):
                stale_entry_max_abs_signal_gap_bps = 50.0
            stale_entry_gap_allowed = bool(
                (not entry_context_fresh)
                and stale_entry_gap_enabled
                and stale_entry_max_abs_signal_gap_bps >= 0.0
                and not hard_signal_close_gate_exceeded
            )
            scoring_entries_allowed = bool(
                entry_context_fresh or late_entries_override or stale_entry_gap_allowed
            )
            if hard_signal_close_gate_exceeded:
                scoring_entries_allowed = False
            hourly_refresh_updates = 0
            if hourly_sync_due:
                if hard_signal_close_gate_exceeded:
                    tprint(
                        "Hourly context exceeds hard signal-close entry gate; "
                        "refreshing data and running model scoring for diagnostics/"
                        "parity only. New orders are blocked by "
                        "max_entries_total=0 "
                        f"(target_hour={latest_closed_hour}, "
                        f"closed_at={latest_closed_hour_close}, "
                        f"hour_age={hourly_age_seconds:.0f}s, "
                        f"max_signal_close_age="
                        f"{hard_signal_close_gate_seconds:.0f}s)."
                    )
                elif not entry_context_fresh and late_entries_override:
                    tprint(
                        "Late-entry override active: allowing new entries despite "
                        "stale entry context "
                        f"(target_hour={latest_closed_hour}, "
                        f"closed_at={latest_closed_hour_close}, "
                        f"hour_age={hourly_age_seconds:.0f}s, "
                        f"max_hour_age={max_entry_hourly_age_seconds:.0f}s)."
                    )
                elif stale_entry_gap_allowed:
                    tprint(
                        "Hourly context is stale, but conditional stale-entry mode "
                        "is active: data/model scoring and candidate selection will "
                        "run; each candidate must pass ticker precheck and absolute "
                        "current-mid vs signal-price movement "
                        f"<= {stale_entry_max_abs_signal_gap_bps:.2f} bps before "
                        "any order can be placed "
                        f"(target_hour={latest_closed_hour}, "
                        f"closed_at={latest_closed_hour_close}, "
                        f"hour_age={hourly_age_seconds:.0f}s, "
                        f"max_hour_age={max_entry_hourly_age_seconds:.0f}s)."
                    )
                elif not scoring_entries_allowed:
                    tprint(
                        "Hourly context is stale for entries; refreshing data and "
                        "running model scoring for diagnostics/parity only. New "
                        "orders will be blocked by max_entries_total=0 "
                        f"(target_hour={latest_closed_hour}, "
                        f"closed_at={latest_closed_hour_close}, "
                        f"hour_age={hourly_age_seconds:.0f}s, "
                        f"max_hour_age={max_entry_hourly_age_seconds:.0f}s)."
                    )
                refresh_microdata = bool(config.get("hourly_refresh_microdata", True))
                hourly_gap_backfill_days = int(
                    config.get("hourly_refresh_recent_gap_backfill_days", 0) or 0
                )
                if (
                    hourly_gap_backfill_days <= 0
                    and _allow_model_feature_tail_recompute_for_reconciliation(config)
                ):
                    shadow_gap_days = int(
                        config.get("shadow_hourly_refresh_recent_gap_backfill_days", 2)
                        or 0
                    )
                    if shadow_gap_days > 0:
                        tprint(
                            "Skipping pre-scoring hourly recent-gap repair despite "
                            "shadow reconciliation setting "
                            f"days={shadow_gap_days}; target-hour decisions should "
                            "not wait for stale-symbol repair. Active-position and "
                            "post-candidate targeted gap backfills remain enabled."
                        )
                    hourly_gap_backfill_days = 0
                if hourly_gap_backfill_days > 0:
                    tprint(
                        "Hourly refresh recent-gap 1h backfill is enabled: "
                        f"days={hourly_gap_backfill_days}. This can be slow for "
                        "large universes and should normally be reserved for "
                        "targeted repair jobs, not pre-scoring live refresh."
                    )
                try:
                    active_position_symbols_for_refresh = sorted(
                        str(sym)
                        for sym in (
                            executor.get_active_positions().keys()
                            if hasattr(executor, "get_active_positions")
                            else []
                        )
                    )
                except Exception:
                    active_position_symbols_for_refresh = []
                if bool(config.get("hourly_fetch_model_universe_only", True)):
                    hourly_fetch_symbols = sorted(
                        set(feature_context_symbols or symbols or download_symbols)
                        .union(symbols)
                        .union(active_position_symbols_for_refresh)
                    )
                    tprint(
                        "Hourly refresh scoped to model context: "
                        f"fetch={len(hourly_fetch_symbols)} "
                        f"download_universe={len(download_symbols)} "
                        f"active_positions={len(active_position_symbols_for_refresh)}"
                    )
                else:
                    hourly_fetch_symbols = list(download_symbols)
                hourly_refresh_result = data_fetcher.fetch_hourly_universe_once(
                    hourly_fetch_symbols,
                    max_workers=int(
                        os.environ.get(
                            "EPM_HOURLY_OHLCV_WORKERS",
                            config.get("hourly_ohlcv_workers", 32),
                        )
                    ),
                    microdata_max_workers=int(
                        os.environ.get(
                            "EPM_HOURLY_MICRODATA_WORKERS",
                            config.get("hourly_microdata_workers", 24),
                        )
                    ),
                    no_progress_timeout_seconds=float(
                        config.get("hourly_ohlcv_no_progress_timeout_seconds", 60.0)
                    ),
                    check_recent_gaps_days=hourly_gap_backfill_days,
                    refresh_microdata=refresh_microdata,
                    target_hour=latest_closed_hour,
                )
                hourly_refresh_updates = (
                    len(hourly_refresh_result)
                    if isinstance(hourly_refresh_result, dict)
                    else 0
                )
                loop_timer.mark("hourly_fetch")
                tprint(
                    "Hourly data refresh complete: "
                    f"ohlcv_symbols={len(hourly_fetch_symbols)} "
                    f"download_universe_symbols={len(download_symbols)} "
                    f"updated_symbols={hourly_refresh_updates} "
                    f"target_hour={latest_closed_hour} "
                    f"microdata_refresh_enabled={refresh_microdata}"
                )
                last_hourly_sync = latest_closed_hour
                did_hourly_refresh = True
                active_gap_days = int(
                    config.get("active_position_recent_gap_backfill_days", 1) or 0
                )
                if active_gap_days > 0:
                    try:
                        active_gap_symbols = sorted(
                            str(sym)
                            for sym in (
                                executor.get_active_positions().keys()
                                if hasattr(executor, "get_active_positions")
                                else []
                            )
                        )
                        if active_gap_symbols:
                            _targeted_recent_gap_backfill(
                                data_fetcher,
                                active_gap_symbols,
                                days=active_gap_days,
                                max_symbols=int(
                                    config.get(
                                        "active_position_recent_gap_backfill_max_symbols",
                                        8,
                                    )
                                    or 8
                                ),
                                label="active_positions",
                            )
                    except Exception as exc:
                        tprint(
                            "Active-position targeted 15m backfill failed: "
                            f"{classify_api_error(exc)}: {exc}"
                        )

            if not did_hourly_refresh:
                _monitor_active_position_price_action(
                    executor,
                    exchange=executor.exchange,
                    now=current_time,
                    config=config,
                    portfolio_mgr=portfolio_mgr,
                    trade_logger=logger,
                    sheets_exporter=sheets_exporter,
                )
                _maybe_send_daily_deployment_report(
                    daily_reporter=daily_reporter,
                    exchange=exchange,
                    portfolio_mgr=portfolio_mgr,
                    trade_logger=logger,
                    config=config,
                )
                _maybe_export_google_sheets(
                    sheets_exporter=sheets_exporter,
                    trade_logger=logger,
                    executor=executor,
                    force=bool(args.run_once),
                )
                if args.run_once:
                    executor.shutdown()
                    break
                _sleep_until_next_candle_close(
                    timeframe_minutes=60,
                    delay_seconds=hourly_delay,
                )
                continue

            feature_runtime_cfg = _build_live_feature_runtime_cfg(
                config=config,
                accepted_strategies=accepted_strategies,
                policy_selection_rules=policy_selection_rules,
                latest_closed_hour=latest_closed_hour,
                hourly_refresh_updates=hourly_refresh_updates,
            )
            prewarm_result: Dict[str, Any] = {}
            if (
                _model_feature_offline_cache_enabled(config)
                and live_model_feature_store_strict(feature_runtime_cfg)
            ):
                prewarm_symbols = sorted(
                    set(feature_context_symbols or symbols or download_symbols).union(
                        symbols
                    )
                )
                prewarm_keys = set(raw_required_feature_keys(required_feature_keys))
                prewarm_keys.update(_lgbm_mask_required_feature_keys(lgbm_strategy_mask_rows))
                prewarm_keys.add("barrier_pct")
                try:
                    prewarm_result = prewarm_selected_model_feature_cache_for_live(
                        run_id=str(config["run_id"]),
                        data_root=str(config["data_root"]),
                        symbols=prewarm_symbols,
                        end_ts=latest_closed_hour,
                        cfg=feature_runtime_cfg,
                        required_feature_keys=prewarm_keys,
                        source_run_ids=feature_runtime_cfg.get(
                            "live_feature_source_run_ids"
                        ),
                    )
                    tprint(
                        "Live selected model-feature prewarm result: "
                        f"{prewarm_result}"
                    )
                except Exception as exc:
                    tprint(
                        "Live selected model-feature prewarm failed; scoring path "
                        "will still perform its strict fail-closed sync if needed: "
                        f"{type(exc).__name__}: {exc}"
                    )
                loop_timer.mark("selected_feature_prewarm")

            if bool(config.get("hourly_fetch_model_universe_only", True)):
                panel_symbols = sorted(
                    set(feature_context_symbols or symbols or download_symbols)
                    .union(symbols)
                )
            else:
                panel_symbols = list(download_symbols)
            panel = data_fetcher.get_panel(
                panel_symbols, lookback_hours=live_decision_panel_lookback_hours
            )
            loop_timer.mark("panel_load")
            warmup_state_health = _live_warmup_state_health_snapshot(
                panel=panel,
                symbols=symbols,
                lookback_hours=live_decision_panel_lookback_hours,
                required_model_warmup_hours=panel_warmup_hours,
                latest_closed_hour=latest_closed_hour,
                feature_runtime_cfg=feature_runtime_cfg,
                config=config,
                prewarm_result=prewarm_result,
            )
            _emit_structured_event("LIVE_WARMUP_STATE_HEALTH", warmup_state_health)
            if not bool(warmup_state_health.get("ok")):
                message = (
                    "Live warmup/state health check failed before model scoring: "
                    f"reason={warmup_state_health.get('reason')} "
                    f"panel_reason={warmup_state_health.get('panel_reason')} "
                    f"rolling_cache_reason="
                    f"{(warmup_state_health.get('rolling_feature_cache') or {}).get('reason')} "
                    f"raw_state_reason="
                    f"{(warmup_state_health.get('raw_rolling_state') or {}).get('reason')} "
                    f"causal_state_reason="
                    f"{(warmup_state_health.get('causal_transform_state') or {}).get('reason')}"
                )
                if _live_warmup_state_fail_closed(config):
                    raise RuntimeError(message)
                tprint("Warning: " + message)
            loop_timer.mark("warmup_state_health")
            tradable_panel = _subset_panel(panel, symbols)
            loop_timer.mark("panel_subset")
            usdc_usdt_ticker = None
            try:
                if getattr(exchange, "fetch_ticker", None) is not None:
                    usdc_usdt_ticker = exchange.fetch_ticker("USDC/USDT")
            except Exception as exc:
                tprint(f"Market kill switch ticker fetch warning: {exc}")
            market_decision = market_kill_switch.evaluate(
                now=current_time,
                usdc_usdt_ticker=usdc_usdt_ticker,
                btc_close=_close_series_for_base(tradable_panel, "BTC"),
                eth_close=_close_series_for_base(tradable_panel, "ETH"),
                basket_close=tradable_panel.get("close", pd.DataFrame()),
            )
            loop_timer.mark("market_kill_switch")
            ignore_market_kill_switch = _ignore_market_kill_switch_for_reconciliation(
                config
            )
            if not market_decision.allow_new_entries and ignore_market_kill_switch:
                tprint(
                    "Market kill switch active but ignored for shadow reconciliation: "
                    f"reason={market_decision.reason}, "
                    f"details={market_decision.details}"
                )
            if not market_decision.allow_new_entries and not ignore_market_kill_switch:
                tprint(
                    "Market kill switch active: blocking new entries but continuing "
                    "model scoring for diagnostics/parity with max_entries_total=0, "
                    f"reason={market_decision.reason}, "
                    f"details={market_decision.details}"
                )
                scoring_entries_allowed = False
            (
                thresholds,
                long_cands,
                short_cands,
                features,
                strategy_candidate_masks,
            ) = _select_candidates_and_load_features(
                panel=panel,
                symbols=symbols,
                run_id=config["run_id"],
                data_root=str(config.get("live_data_root") or config["data_root"]),
                cfg=feature_runtime_cfg,
                lookback_hours=live_decision_panel_lookback_hours,
                required_feature_keys=required_feature_keys,
                lgbm_strategy_mask_rows=lgbm_strategy_mask_rows,
                feature_context_symbols=feature_context_symbols,
                strategy_feature_contracts=strategy_feature_contracts,
            )
            candidate_gap_days = int(
                config.get("candidate_recent_gap_backfill_days", 0) or 0
            )
            candidate_gap_results: Dict[str, str] = {}
            if candidate_gap_days > 0 and (long_cands or short_cands):
                candidate_gap_results = _targeted_recent_gap_backfill(
                    data_fetcher,
                    list(long_cands) + list(short_cands),
                    days=candidate_gap_days,
                    max_symbols=int(
                        config.get("candidate_recent_gap_backfill_max_symbols", 24)
                        or 24
                    ),
                    label="post_mask_candidates",
                )
                if any(
                    str(status).startswith("backfilled")
                    for status in candidate_gap_results.values()
                ) and bool(
                    config.get("candidate_recent_gap_backfill_rerun_on_update", True)
                ):
                    tprint(
                        "Targeted candidate 15m backfill changed local data; "
                        "reloading panel and recomputing candidate features for "
                        "this cycle."
                    )
                    panel = data_fetcher.get_panel(
                        panel_symbols,
                        lookback_hours=live_decision_panel_lookback_hours,
                    )
                    tradable_panel = _subset_panel(panel, symbols)
                    (
                        thresholds,
                        long_cands,
                        short_cands,
                        features,
                        strategy_candidate_masks,
                    ) = _select_candidates_and_load_features(
                        panel=panel,
                        symbols=symbols,
                        run_id=config["run_id"],
                        data_root=str(
                            config.get("live_data_root") or config["data_root"]
                        ),
                        cfg={
                            **feature_runtime_cfg,
                            "live_feature_cycle_cache_bypass": True,
                        },
                        lookback_hours=live_decision_panel_lookback_hours,
                        required_feature_keys=required_feature_keys,
                        lgbm_strategy_mask_rows=lgbm_strategy_mask_rows,
                        feature_context_symbols=feature_context_symbols,
                        strategy_feature_contracts=strategy_feature_contracts,
                    )
            loop_timer.mark("candidate_and_feature_load")

            pre_score_now = pd.Timestamp.now(tz="UTC")
            pre_score_hourly_age_seconds = _closed_candle_age_seconds(
                pre_score_now,
                latest_closed_hour,
                timeframe_minutes=60,
            )
            if (
                scoring_entries_allowed
                and hard_signal_close_gate_seconds >= 0.0
                and pre_score_hourly_age_seconds > hard_signal_close_gate_seconds
            ):
                scoring_entries_allowed = False
                tprint(
                    "Hourly signal became stale during live feature preparation; "
                    "running model scoring for diagnostics/parity only and "
                    "blocking new orders with max_entries_total=0 "
                    f"(target_hour={latest_closed_hour}, "
                    f"closed_at={latest_closed_hour_close}, "
                    f"hour_age={pre_score_hourly_age_seconds:.0f}s, "
                    f"max_signal_close_age={hard_signal_close_gate_seconds:.0f}s)."
                )

            results = run_inference_step(
                orchestrator=orchestrator,
                panel=tradable_panel,
                feats=features,
                thresholds=thresholds,
                executor=executor,
                logger=logger,
                max_candidates=max(len(long_cands) + len(short_cands), 1),
                accepted_strategies=accepted_strategies,
                calibration_data=calibration_data,
                normalized_thresholds=normalized_thresholds,
                portfolio_mgr=portfolio_mgr,
                initial_rank_threshold=float(portfolio_policy.initial_rank_threshold),
                strategy_asset_exclusions=strategy_asset_exclusions,
                preselected_long_candidates=long_cands,
                preselected_short_candidates=short_cands,
                strategy_candidate_masks=strategy_candidate_masks,
                portfolio_policy=portfolio_policy,
                prediction_ledger=prediction_ledger,
                dynamic_performance_monitor=dynamic_performance_monitor,
                strategy_kill_switch=strategy_kill_switch,
                strategy_feature_contracts=strategy_feature_contracts,
                max_entries_total=(
                    int(config.get("max_entries_total", 4))
                    if scoring_entries_allowed
                    else 0
                ),
                stale_entry_context=bool(
                    stale_entry_gap_allowed and not late_entries_override
                ),
                stale_entry_max_abs_signal_gap_bps=(stale_entry_max_abs_signal_gap_bps),
            )
            loop_timer.mark("model_scoring_and_orders")
            tprint(
                f"Inference batch complete: download_symbols={len(download_symbols)} "
                f"panel_symbols={len(panel_symbols)} "
                f"tradable_symbols={len(symbols)} "
                f"candidates={len(long_cands) + len(short_cands)} "
                f"trades={len(results['trades'])}"
            )
            _emit_inference_heartbeat(
                current_time=current_time,
                config=config,
                download_symbols=download_symbols,
                tradable_symbols=symbols,
                long_candidates=long_cands,
                short_candidates=short_cands,
                features=features,
                strategy_candidate_masks=strategy_candidate_masks,
                results=results,
                data_fetcher=data_fetcher,
                portfolio_mgr=portfolio_mgr,
                executor=executor,
            )
            _maybe_send_daily_deployment_report(
                daily_reporter=daily_reporter,
                exchange=exchange,
                portfolio_mgr=portfolio_mgr,
                trade_logger=logger,
                config=config,
            )
            _maybe_export_google_sheets(
                sheets_exporter=sheets_exporter,
                trade_logger=logger,
                executor=executor,
                force=bool(args.run_once),
            )

            if args.run_once:
                executor.shutdown()
                break

            _sleep_until_next_candle_close(
                timeframe_minutes=60,
                delay_seconds=hourly_delay,
            )

        except KeyboardInterrupt:
            tprint("Shutting down...")
            executor.shutdown()
            break
        except Exception as e:
            tprint(f"Error in inference loop: {e}")
            import traceback

            tprint(traceback.format_exc())
            if args.run_once:
                executor.shutdown()
                raise
            time.sleep(60)  # Wait 1 minute on error


def run_challenger_monitor(
    symbols,
    data_fetcher,
    orchestrator,
    executor,
    logger,
    config,
    interval,
    panel_lookback_hours,
    required_feature_keys=None,
    accepted_strategies=None,
    calibration_data=None,
    strategy_asset_exclusions=None,
    portfolio_mgr: Optional[PortfolioManager] = None,
    sheets_exporter: Optional[GoogleSheetsTradeExporter] = None,
):
    """
    calibration_data = calibration_data or {}
    Run position monitoring every minute by default.

    New entries are intentionally not evaluated here. The loop only monitors
    existing positions and applies the stop policy on closed 15m bars.

    Args:
        symbols: List of trading symbols
        data_fetcher: DataFetcher instance
        orchestrator: ModelOrchestrator instance
        executor: TradeExecutor instance
        logger: TradeLogger instance
        config: Configuration dictionary
        interval: Check interval in seconds (default 60 = 1 min)
    """
    while True:
        try:
            current_time = pd.Timestamp.now(tz="UTC")
            tprint(f"\n=== Challenger monitor at {current_time} ===")

            # The challenger loop is intentionally position-management only.
            # New entries are evaluated by the hourly data/feature/model path.
            _monitor_active_position_price_action(
                executor,
                exchange=executor.exchange,
                now=current_time,
                config=config,
                portfolio_mgr=portfolio_mgr,
                trade_logger=logger,
                sheets_exporter=sheets_exporter,
            )

            time.sleep(interval)

        except Exception as e:
            tprint(f"Error in challenger monitor: {e}")
            import traceback

            tprint(traceback.format_exc())
            time.sleep(interval)


def _evaluate_oco_policy(
    symbol: str,
    position_state: Dict[str, Any],
    ohlcv_5m: pd.DataFrame,
    executor: TradeExecutor,
):
    """Evaluate stop touches and delegate replacement to simple-policy decision."""
    if (
        ohlcv_5m is None
        or not isinstance(ohlcv_5m, (pd.DataFrame, pd.Series))
        or (hasattr(ohlcv_5m, "empty") and ohlcv_5m.empty)
    ):
        return

    try:
        bars = pd.DataFrame(ohlcv_5m).sort_index()
        required_cols = {"open", "high", "low", "close"}
        if not required_cols.issubset(bars.columns):
            return

        last_eval_ts = position_state.get("last_5m_eval_ts")
        if last_eval_ts is not None:
            last_eval_ts = pd.Timestamp(last_eval_ts)
            bars = bars[bars.index > last_eval_ts]
        if bars.empty:
            return

        side = str(position_state.get("side", "long")).lower()
        realized_entry_price = float(position_state.get("entry_price", 0.0) or 0.0)
        entry_price, entry_price_source = _position_policy_entry_price(position_state)
        if not (np.isfinite(entry_price) and entry_price > 0.0):
            entry_price = realized_entry_price
            entry_price_source = "entry_price"
        if np.isfinite(entry_price) and entry_price > 0.0:
            position_state["policy_entry_price"] = float(entry_price)
            position_state.setdefault("theoretical_entry_price", float(entry_price))
            position_state["policy_entry_price_source"] = entry_price_source
        bucket_key = position_state.get("bucket_key", "")
        if hasattr(executor, "get_simple_policy_stop_params"):
            params = dict(executor.get_simple_policy_stop_params(bucket_key) or {})
        else:
            params = {}

        stop_price = float(position_state.get("stop_price", np.nan))
        peak_price = float(position_state.get("peak_price", entry_price) or entry_price)
        mfe = float(position_state.get("mfe", 0.0) or 0.0)
        mae = float(position_state.get("mae", 0.0) or 0.0)
        stop_reason = str(position_state.get("stop_reason") or "original_stop_loss")
        shadow_state: Optional[Dict[str, Any]] = None
        if _shadow_execution_realism_enabled():
            shadow_state = _ensure_simple_policy_shadow_state(
                position_state,
                symbol=symbol,
                side=side,
                policy_entry_price=entry_price,
                policy_entry_price_source=entry_price_source,
                realized_entry_price=realized_entry_price,
                stop_price=stop_price,
                stop_reason=stop_reason,
                params=params,
            )
        last_bar_ts = bars.index[-1]
        live_mode = str(getattr(executor, "mode", "") or "").lower() in {
            "live",
            "live-test",
            "live_test",
        }
        if live_mode and not position_state.get("stop_order_id"):
            retry_stop = getattr(executor, "retry_missing_protective_stop", None)
            if callable(retry_stop):
                retry_result = retry_stop(symbol, position_state)
                position_state.setdefault("trade_recap_events", []).append(
                    {
                        "ts": pd.Timestamp.now(tz="UTC").isoformat(),
                        "event": "missing_exchange_stop_retry",
                        "success": bool(
                            isinstance(retry_result, dict)
                            and retry_result.get("success")
                        ),
                        "error_category": (
                            retry_result.get("error_category")
                            if isinstance(retry_result, dict)
                            else None
                        ),
                        "error": (
                            retry_result.get("error")
                            if isinstance(retry_result, dict)
                            else None
                        ),
                        "stop_price": float(stop_price),
                        "stop_reason": stop_reason,
                    }
                )
                refreshed = (
                    executor.get_position(symbol)
                    if hasattr(executor, "get_position")
                    else None
                )
                if isinstance(refreshed, dict):
                    position_state.update(refreshed)
                    stop_price = float(position_state.get("stop_price", stop_price))

        for bar_ts, row in bars.iterrows():
            bar_open = float(row["open"])
            bar_high = float(row["high"])
            bar_low = float(row["low"])
            bar_close = float(row["close"])
            price_dev_pct = (
                (bar_close - entry_price) / max(abs(entry_price), 1e-12)
                if side == "long"
                else (entry_price - bar_close) / max(abs(entry_price), 1e-12)
            )
            position_state["current_price"] = bar_close
            position_state["last_price"] = bar_close
            position_state["current_price_ts"] = pd.Timestamp(bar_ts)
            position_state.setdefault("trade_recap_events", []).append(
                {
                    "ts": pd.Timestamp(bar_ts).isoformat(),
                    "event": "price_bar_5m",
                    "open": bar_open,
                    "high": bar_high,
                    "low": bar_low,
                    "close": bar_close,
                    "price_dev_pct": float(price_dev_pct),
                    "policy_entry_price": float(entry_price),
                    "policy_entry_price_source": entry_price_source,
                    "realized_entry_price": (
                        float(realized_entry_price)
                        if np.isfinite(realized_entry_price)
                        else None
                    ),
                    "stop_price": float(stop_price),
                    "stop_reason": stop_reason,
                }
            )
            if len(position_state.get("trade_recap_events", [])) > 500:
                del position_state["trade_recap_events"][:-500]

            if (
                isinstance(shadow_state, dict)
                and str(shadow_state.get("status") or "open") == "open"
            ):
                shadow_stop = _finite_positive_float(
                    shadow_state.get("shadow_stop_price")
                )
                touched = (
                    np.isfinite(shadow_stop)
                    and (bar_low <= shadow_stop if side == "long" else bar_high >= shadow_stop)
                )
                if touched:
                    shadow_state.update(
                        {
                            "status": "shadow_exit_triggered",
                            "shadow_exit_time": pd.Timestamp(bar_ts).isoformat(),
                            "shadow_exit_price": float(shadow_stop),
                            "shadow_exit_reason": (
                                "shadow_stop_loss_filled:"
                                f"{shadow_state.get('shadow_stop_reason') or stop_reason}"
                            ),
                            "shadow_exit_return": _shadow_side_return(
                                side, float(shadow_stop), entry_price
                            ),
                            "shadow_exit_vs_live_stop_bps": _shadow_bps_delta(
                                side, float(shadow_stop), stop_price
                            ),
                        }
                    )
                    _append_simple_policy_shadow_event(
                        shadow_state,
                        "shadow_stop_touch",
                        bar_ts=pd.Timestamp(bar_ts).isoformat(),
                        bar_low=bar_low,
                        bar_high=bar_high,
                        shadow_stop_price=shadow_stop,
                        live_stop_price=stop_price,
                        live_stop_gap_bps=shadow_state.get(
                            "shadow_exit_vs_live_stop_bps"
                        ),
                    )

            if side == "long":
                if np.isfinite(stop_price) and bar_low <= stop_price:
                    exit_reason = f"stop_loss_filled:{stop_reason}"
                    has_exchange_stop = bool(position_state.get("stop_order_id"))
                    if live_mode and has_exchange_stop:
                        position_state.setdefault("trade_recap_events", []).append(
                            {
                                "ts": pd.Timestamp(bar_ts).isoformat(),
                                "event": "stop_touch_deferred_to_exchange_order",
                                "bar_low": bar_low,
                                "stop_price": float(stop_price),
                                "stop_reason": stop_reason,
                            }
                        )
                        continue
                    return executor.close_position(
                        symbol, price=float(stop_price), reason=exit_reason
                    )
            else:
                if np.isfinite(stop_price) and bar_high >= stop_price:
                    exit_reason = f"stop_loss_filled:{stop_reason}"
                    has_exchange_stop = bool(position_state.get("stop_order_id"))
                    if live_mode and has_exchange_stop:
                        position_state.setdefault("trade_recap_events", []).append(
                            {
                                "ts": pd.Timestamp(bar_ts).isoformat(),
                                "event": "stop_touch_deferred_to_exchange_order",
                                "bar_high": bar_high,
                                "stop_price": float(stop_price),
                                "stop_reason": stop_reason,
                            }
                        )
                        continue
                    return executor.close_position(
                        symbol, price=float(stop_price), reason=exit_reason
                    )

        live_closeable_snapshot: Dict[str, Any] = {}
        live_closeable_price = float("nan")
        if live_mode:
            live_closeable_snapshot = _fetch_live_closeable_price(
                symbol, side, executor
            )
            live_closeable_price = _finite_positive_float(
                live_closeable_snapshot.get("price")
            )
            if np.isfinite(live_closeable_price):
                if side == "long":
                    peak_price = max(peak_price, live_closeable_price)
                    mfe = max(
                        mfe,
                        (live_closeable_price - entry_price)
                        / max(abs(entry_price), 1e-12),
                    )
                    mae = max(
                        mae,
                        (entry_price - live_closeable_price)
                        / max(abs(entry_price), 1e-12),
                    )
                else:
                    peak_price = min(peak_price, live_closeable_price)
                    mfe = max(
                        mfe,
                        (entry_price - live_closeable_price)
                        / max(abs(entry_price), 1e-12),
                    )
                    mae = max(
                        mae,
                        (live_closeable_price - entry_price)
                        / max(abs(entry_price), 1e-12),
                    )
                position_state["current_price"] = float(live_closeable_price)
                position_state["last_price"] = float(live_closeable_price)
                position_state["current_price_source"] = str(
                    live_closeable_snapshot.get("source") or "live_closeable_touch"
                )
                position_state["current_price_ts"] = pd.Timestamp.now(tz="UTC")
                position_state.setdefault("trade_recap_events", []).append(
                    {
                        "ts": position_state["current_price_ts"].isoformat(),
                        "event": "live_closeable_price_sample",
                        "side": side,
                        "price": float(live_closeable_price),
                        "source": position_state["current_price_source"],
                        "policy_entry_price": float(entry_price),
                        "policy_entry_price_source": entry_price_source,
                        "realized_entry_price": (
                            float(realized_entry_price)
                            if np.isfinite(realized_entry_price)
                            else None
                        ),
                        "bid": (
                            float(live_closeable_snapshot.get("bid"))
                            if np.isfinite(
                                _finite_positive_float(
                                    live_closeable_snapshot.get("bid")
                                )
                            )
                            else None
                        ),
                        "ask": (
                            float(live_closeable_snapshot.get("ask"))
                            if np.isfinite(
                                _finite_positive_float(
                                    live_closeable_snapshot.get("ask")
                                )
                            )
                            else None
                        ),
                        "last": (
                            float(live_closeable_snapshot.get("last"))
                            if np.isfinite(
                                _finite_positive_float(
                                    live_closeable_snapshot.get("last")
                                )
                            )
                            else None
                        ),
                        "mfe": float(mfe),
                        "mae": float(mae),
                        "peak_price": float(peak_price),
                    }
                )
                if len(position_state.get("trade_recap_events", [])) > 500:
                    del position_state["trade_recap_events"][:-500]
                if _executable_stop_breached(
                    side, float(stop_price), float(live_closeable_price)
                ):
                    exit_reason = f"software_executable_stop_breach:{stop_reason}"
                    if (
                        isinstance(shadow_state, dict)
                        and str(shadow_state.get("status") or "open") == "open"
                    ):
                        shadow_state.update(
                            {
                                "status": "shadow_exit_triggered",
                                "shadow_exit_time": position_state[
                                    "current_price_ts"
                                ].isoformat(),
                                "shadow_exit_price": float(live_closeable_price),
                                "shadow_exit_reason": exit_reason,
                                "shadow_exit_return": _shadow_side_return(
                                    side, float(live_closeable_price), entry_price
                                ),
                                "shadow_exit_vs_live_stop_bps": _shadow_bps_delta(
                                    side, float(live_closeable_price), stop_price
                                ),
                            }
                        )
                        _append_simple_policy_shadow_event(
                            shadow_state,
                            "shadow_executable_stop_breach",
                            exit_price=float(live_closeable_price),
                            live_stop_price=float(stop_price),
                            stop_reason=stop_reason,
                            source=position_state["current_price_source"],
                        )
                    position_state["shadow_simple_policy_state"] = shadow_state
                    position_state.setdefault("trade_recap_events", []).append(
                        {
                            "ts": position_state["current_price_ts"].isoformat(),
                            "event": "software_executable_stop_breach",
                            "side": side,
                            "price": float(live_closeable_price),
                            "source": position_state["current_price_source"],
                            "stop_price": float(stop_price),
                            "stop_reason": stop_reason,
                            "exchange_stop_order_id": position_state.get(
                                "stop_order_id"
                            ),
                            "exchange_stop_trigger_signal": position_state.get(
                                "stop_trigger_signal"
                            ),
                        }
                    )
                    return executor.close_position(
                        symbol,
                        price=float(live_closeable_price),
                        reason=exit_reason,
                    )

        require_metadata = True
        decision = None
        try:
            decision = compute_simple_policy_stop_decision(
                state={
                    **position_state,
                    "entry_price": entry_price,
                    "realized_entry_price": realized_entry_price,
                    "peak_price": peak_price,
                    "mfe": mfe,
                    "mae": mae,
                },
                latest_market_state=bars,
                policy_params=params,
                side=side,
                require_metadata=require_metadata,
            )
            if getattr(decision, "should_exit", False):
                policy_bar_exit_price = float(bars.iloc[-1]["close"])
                exit_price = (
                    float(live_closeable_price)
                    if np.isfinite(live_closeable_price)
                    else policy_bar_exit_price
                )
                exit_ts = (
                    pd.Timestamp(position_state.get("current_price_ts"))
                    if np.isfinite(live_closeable_price)
                    and position_state.get("current_price_ts") is not None
                    else pd.Timestamp(last_bar_ts)
                )
                exit_price_source = (
                    str(
                        live_closeable_snapshot.get("source")
                        or "live_closeable_touch"
                    )
                    if np.isfinite(live_closeable_price)
                    else "trade_5m_close"
                )
                if (
                    isinstance(shadow_state, dict)
                    and str(shadow_state.get("status") or "open") == "open"
                ):
                    shadow_state.update(
                        {
                            "status": "shadow_exit_triggered",
                            "shadow_exit_time": exit_ts.isoformat(),
                            "shadow_exit_price": exit_price,
                            "shadow_exit_reason": str(
                                decision.exit_reason or decision.reason
                            ),
                            "shadow_exit_return": _shadow_side_return(
                                side, exit_price, entry_price
                            ),
                            "shadow_policy_bar_exit_time": pd.Timestamp(
                                last_bar_ts
                            ).isoformat(),
                            "shadow_policy_bar_exit_price": policy_bar_exit_price,
                            "shadow_exit_price_source": exit_price_source,
                            "shadow_policy_bar_vs_live_exit_bps": _shadow_bps_delta(
                                side, policy_bar_exit_price, exit_price
                            ),
                            "shadow_mfe": float(
                                decision.mfe if decision.mfe is not None else mfe
                            ),
                            "shadow_mae": float(
                                decision.mae if decision.mae is not None else mae
                            ),
                        }
                    )
                    _append_simple_policy_shadow_event(
                        shadow_state,
                        "shadow_policy_exit",
                        reason=decision.reason,
                        exit_reason=decision.exit_reason,
                        exit_price=exit_price,
                        exit_price_source=exit_price_source,
                        policy_bar_exit_price=policy_bar_exit_price,
                        policy_bar_ts=pd.Timestamp(last_bar_ts).isoformat(),
                        policy_bar_vs_live_exit_bps=shadow_state.get(
                            "shadow_policy_bar_vs_live_exit_bps"
                        ),
                        mfe=shadow_state.get("shadow_mfe"),
                        mae=shadow_state.get("shadow_mae"),
                    )
                position_state.setdefault("trade_recap_events", []).append(
                    {
                        "ts": exit_ts.isoformat(),
                        "event": "adverse_excursion_exit_triggered",
                        "reason": decision.reason,
                        "detail": decision.reason_detail,
                        "exit_price": float(exit_price),
                        "exit_price_source": exit_price_source,
                        "policy_bar_exit_price": policy_bar_exit_price,
                        "policy_bar_ts": pd.Timestamp(last_bar_ts).isoformat(),
                        "mfe": float(decision.mfe if decision.mfe is not None else mfe),
                        "mae": float(decision.mae if decision.mae is not None else mae),
                    }
                )
                return executor.close_position(
                    symbol,
                    price=exit_price,
                    reason=str(decision.exit_reason or decision.reason),
                )
        except SimplePolicyStopParamsError as exc:
            position_state.setdefault("trade_recap_events", []).append(
                {
                    "ts": pd.Timestamp(last_bar_ts).isoformat(),
                    "event": "stop_update_skipped",
                    "reason": "invalid_simple_policy_runtime_params",
                    "error": str(exc),
                }
            )

        decision_peak = (
            decision.peak_price
            if decision is not None and decision.peak_price is not None
            else peak_price
        )
        decision_mfe = (
            decision.mfe if decision is not None and decision.mfe is not None else mfe
        )
        decision_mae = (
            decision.mae if decision is not None and decision.mae is not None else mae
        )
        if isinstance(shadow_state, dict) and decision is not None:
            decision_stop = _finite_positive_float(
                getattr(decision, "stop_price", np.nan)
            )
            if np.isfinite(decision_stop):
                current_shadow_stop = _finite_positive_float(
                    shadow_state.get("shadow_stop_price")
                )
                improved = (
                    (decision_stop > current_shadow_stop)
                    if side == "long"
                    else (decision_stop < current_shadow_stop)
                ) if np.isfinite(current_shadow_stop) else True
                if improved:
                    shadow_state["shadow_stop_price"] = float(decision_stop)
                    shadow_state["shadow_stop_reason"] = str(decision.reason)
                    shadow_state["shadow_stop_reason_detail"] = str(
                        decision.reason_detail
                    )
                shadow_state["latest_policy_requested_stop_price"] = float(
                    decision_stop
                )
                shadow_state["latest_live_stop_price"] = (
                    float(stop_price) if np.isfinite(stop_price) else None
                )
                shadow_state["latest_stop_gap_bps"] = _shadow_bps_delta(
                    side, decision_stop, stop_price
                )
                shadow_state["shadow_mfe"] = float(decision_mfe)
                shadow_state["shadow_mae"] = float(decision_mae)
                _append_simple_policy_shadow_event(
                    shadow_state,
                    "shadow_stop_decision",
                    requested_stop_price=decision_stop,
                    live_stop_price=stop_price,
                    stop_gap_bps=shadow_state.get("latest_stop_gap_bps"),
                    reason=decision.reason,
                    should_replace=getattr(decision, "should_replace", None),
                    mfe=decision_mfe,
                    mae=decision_mae,
                )
        final_current_price = (
            float(live_closeable_price)
            if np.isfinite(live_closeable_price)
            else float(bars.iloc[-1]["close"])
        )
        final_current_price_ts = (
            pd.Timestamp(position_state.get("current_price_ts"))
            if np.isfinite(live_closeable_price)
            and position_state.get("current_price_ts") is not None
            else last_bar_ts
        )
        final_current_price_source = (
            str(live_closeable_snapshot.get("source") or "live_closeable_touch")
            if np.isfinite(live_closeable_price)
            else "trade_5m_close"
        )
        update_result = executor.update_position_policy_state(
            symbol,
            policy_stop_decision=decision,
            peak_price=decision_peak,
            mfe=decision_mfe,
            mae=decision_mae,
            bars_in_trade=int(position_state.get("bars_in_trade", 0) or 0)
            + int(len(bars)),
            last_5m_eval_ts=last_bar_ts,
            current_price=final_current_price,
            current_price_source=final_current_price_source,
            policy_entry_price=float(entry_price),
            policy_entry_price_source=entry_price_source,
            current_price_ts=final_current_price_ts,
            shadow_simple_policy_state=shadow_state,
        )
        if (
            isinstance(update_result, dict)
            and isinstance(update_result.get("closed_trade"), dict)
        ):
            return update_result
    except Exception as e:
        tprint(f"  [STOP_LOSS] Error evaluating stop policy for {symbol}: {e}")


if __name__ == "__main__":
    main()
