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
import json
import os
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from extreme_price_movements import hf_data_loader
from extreme_price_movements.feature_transforms import CausalFeatureTransformer
from extreme_price_movements.inference.candidate_selector import select_candidates
from extreme_price_movements.inference.config import (
    DEFAULT_EXECUTION_ACCOUNT,
    DEFAULT_LIVE_QUOTE_CURRENCY,
    DEFAULT_MARGIN_MODE,
    get_candidate_thresholds,
    get_inference_defaults,
    get_runtime_cfg,
    load_full_state,
    load_inference_config,
    resolve_inference_universes,
)
from extreme_price_movements.inference.daily_reporter import DailyDeploymentReporter
from extreme_price_movements.inference.data_fetcher import (
    DataFetcher,
    classify_api_error,
    fetch_and_build_panel,
    fetch_latest_ohlcv,
    make_exchange,
)
from extreme_price_movements.inference.feature_generator import (
    compute_selector_features,
    generate_features,
    get_features_for_candidates,
    get_inference_required_feature_keys,
    get_market_data,
    load_or_compute_features,
)
from extreme_price_movements.inference.model_orchestrator import ModelOrchestrator
from extreme_price_movements.inference.parity import (
    calibrated_score_and_threshold,
    calibration_size_multiplier,
    load_policy_params_by_strategy,
    load_strategy_asset_exclusion_filter,
    resolve_deployment_strategy_filter,
    strategy_core_id,
    strategy_id_matches,
    validate_calibration_artifacts,
    validate_deployment_model_coverage,
    validate_live_feature_contract,
    validate_required_feature_frames,
)
from extreme_price_movements.inference.symbol_mapping import (
    normalise_symbol,
    symbol_base,
)
from extreme_price_movements.inference.trade_executor import TradeExecutor
from extreme_price_movements.inference.trade_logger import (
    TradeLogger,
    log_trade_decision,
)
from extreme_price_movements.portfolio_manager import PortfolioManager
from extreme_price_movements.simple_position_sizer import load_calibration_curves
from extreme_price_movements.utils import tprint


def _load_normalized_threshold_map(
    data_root: str, run_id: str
) -> Dict[str, Dict[str, Any]]:
    rows_out: Dict[str, Dict[str, Any]] = {}
    path = (
        Path(data_root)
        / "artifacts"
        / run_id
        / "ridge_sizer"
        / "normalized_strategy_thresholds.json"
    )
    if path.exists():
        try:
            payload = json.loads(path.read_text())
            rows = payload.get("strategies", {}) if isinstance(payload, dict) else {}
            if isinstance(rows, dict):
                threshold_space = str(payload.get("threshold_space", "") or "")
                for sid, row in rows.items():
                    if isinstance(row, dict):
                        row = dict(row)
                        row.setdefault("threshold_space", threshold_space)
                        rows_out[str(sid)] = row
        except Exception as exc:
            tprint(f"Could not load normalized strategy thresholds: {exc}")

    base = Path(data_root) / "artifacts" / run_id
    strategy_paths = [
        base / "policy_params" / "strategy_for_inference.json",
        base / "strategy_for_inference.json",
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
                        rows_out[str(alias)] = dict(nrow)
                loaded += 1
            tprint(f"Loaded {loaded} deployment rank thresholds from {strategy_path}")
            break
        except Exception as exc:
            tprint(
                f"Could not load deployment rank thresholds from {strategy_path}: {exc}"
            )
    return rows_out


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
    return float(np.clip(threshold, 0.0, 1.0))


def _attach_rank_percentile_scores(
    decision_rows: List[Dict[str, Any]],
    *,
    score_key: str = "rank_score",
) -> None:
    """Attach per-strategy percentile ranks for live rank-threshold gates."""
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
        ranks = pd.Series(scores[finite]).rank(method="average", pct=True).to_numpy()
        rank_out = np.full(len(scores), np.nan, dtype=np.float64)
        rank_out[finite] = ranks
        for local_i, row_i in enumerate(indices):
            decision_rows[row_i]["sizer_rank_percentile"] = float(rank_out[local_i])


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
    """Build executor params, preferring optimized ridge-sizer bucket params."""
    model_bundle = config.get("model_bundle", {}) or {}
    full_state = config.get("full_state", {}) or {}
    ridge_weights = model_bundle.get("ridge_weights", {}) or {}
    params_per_bucket = ridge_weights.get("params_per_bucket", {}) or {}
    bucket_params = (
        dict(params_per_bucket)
        if params_per_bucket
        else dict(full_state.get("bucket_params", {}) or {})
    )
    ridge_sizer = full_state.get("ridge_sizer")
    if ridge_sizer is not None and getattr(ridge_sizer, "best_params_", None):
        bucket_params.setdefault(
            "cooldown_hours", float(ridge_sizer.best_params_.get("cooldown_hours", 0.0))
        )
    policy_params = load_policy_params_by_strategy(
        str(config.get("data_root", "data")), str(config.get("run_id", ""))
    )
    for strategy_id, params in policy_params.items():
        existing = bucket_params.get(strategy_id, {})
        merged = dict(existing) if isinstance(existing, dict) else {}
        merged.update(params)
        bucket_params[strategy_id] = merged
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


def _select_candidates_and_load_features(
    *,
    panel: Dict[str, pd.DataFrame],
    symbols: List[str],
    run_id: str,
    data_root: str,
    cfg: Dict[str, Any],
    lookback_hours: int,
    required_feature_keys: Optional[set[str]],
) -> tuple[Dict[str, float], List[str], List[str], Dict[str, pd.DataFrame]]:
    selector_feats = compute_selector_features(panel, symbols)
    thresholds = get_candidate_thresholds()
    min_range_pct = thresholds.get("min_range_pct")
    if thresholds.get("min_move_12h_pct") is not None:
        min_range_pct = None
    _ = min_range_pct
    long_cands, short_cands = select_candidates(
        panel=panel,
        feats=selector_feats,
        metric=str(thresholds.get("metric", "ret12h")),
    )
    selected_symbols = sorted(set(long_cands + short_cands))
    if not selected_symbols:
        return thresholds, long_cands, short_cands, selector_feats
    model_feats = load_or_compute_features(
        panel=_subset_panel(panel, selected_symbols),
        basket_syms=selected_symbols,
        run_id=run_id,
        data_root=data_root,
        cfg=cfg,
        lookback_hours=lookback_hours,
        required_feature_keys=required_feature_keys,
    )
    validate_required_feature_frames(
        model_feats,
        required_feature_keys,
        symbols=selected_symbols,
        strict=True,
    )
    return thresholds, long_cands, short_cands, model_feats


def _is_symbol_cooldown_blocked(
    symbol: str,
    *,
    now: pd.Timestamp,
    logger: TradeLogger,
    executor: TradeExecutor,
    cooldown_hours: float,
) -> bool:
    """Return True if symbol is blocked by active position or cooldown."""
    if cooldown_hours <= 0:
        active = (
            executor.get_active_positions()
            if hasattr(executor, "get_active_positions")
            else {}
        )
        return symbol in active
    active = (
        executor.get_active_positions()
        if hasattr(executor, "get_active_positions")
        else {}
    )
    if symbol in active:
        return True
    last_ts = logger.get_last_trade_timestamp(symbol)
    if last_ts is None:
        return False
    return pd.Timestamp(now) < (
        pd.Timestamp(last_ts) + pd.Timedelta(hours=float(cooldown_hours))
    )


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
    rejection_categories = {
        "insufficient_balance",
        "invalid_precision_or_filter",
        "symbol_halted",
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


def _select_top_base_prediction_symbols(
    orchestrator: ModelOrchestrator,
    candidate_features: pd.DataFrame,
    candidates: List[str],
    side: str,
    strategy_id: str,
    *,
    top_frac: float = 0.25,
) -> Dict[str, Dict[str, float]]:
    """Rank base-model predictions and keep only the top fraction for meta."""
    if not hasattr(orchestrator, "predict_alpha"):
        return {
            str(symbol): {"base_pred": float("nan"), "base_rank_pct": 1.0}
            for symbol in candidates
        }
    try:
        preds = orchestrator.predict_alpha(candidate_features, side, strategy_id)
    except Exception as exc:
        tprint(
            f"Base prediction gate failed for {side}/{strategy_id}; "
            f"falling back to all candidates: {exc}"
        )
        return {
            str(symbol): {"base_pred": float("nan"), "base_rank_pct": 1.0}
            for symbol in candidates
        }
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
    initial_rank_threshold: float = 0.5,
    strategy_asset_exclusions: Optional[Dict[str, set[str]]] = None,
    preselected_long_candidates: Optional[List[str]] = None,
    preselected_short_candidates: Optional[List[str]] = None,
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
    }
    now_utc = pd.Timestamp.now(tz="UTC")
    calibration_data = calibration_data or {}
    normalized_thresholds = normalized_thresholds or {}

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

    # Limit candidates
    long_cands = long_cands[:max_candidates]
    short_cands = short_cands[:max_candidates]

    results["long_candidates"] = long_cands
    results["short_candidates"] = short_cands

    tprint(f"Candidates: {len(long_cands)} long, {len(short_cands)} short")

    # Step 2: Process long candidates
    for side, candidates in [("long", long_cands), ("short", short_cands)]:
        if not candidates:
            continue

        # Get features for candidates
        candidate_features = get_features_for_candidates(feats, candidates)

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

        # Run full inference chain
        try:
            decision_rows: List[Dict[str, Any]] = []
            strategy_ids = (
                orchestrator.available_strategies(side, accepted_strategies)
                if hasattr(orchestrator, "available_strategies")
                else [f"{side}_mr"]
            )
            for selected_strategy in strategy_ids:
                eligible_candidates = [
                    symbol
                    for symbol in candidates
                    if not _is_symbol_blocked_for_strategy(
                        symbol, str(selected_strategy), strategy_asset_exclusions
                    )
                ]
                if not eligible_candidates:
                    if candidates:
                        tprint(
                            f"Asset exclusion block: all {side} candidates skipped for "
                            f"{selected_strategy}"
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
                    if chain_results.get("action") != "enter":
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
                    raw_score = float(chain_results.get("meta_pred", 0.0) or 0.0)
                    calibrated_score, rank_threshold = calibrated_score_and_threshold(
                        raw_score=raw_score,
                        strategy_id=strategy_id,
                        calibration_data=calibration_data,
                        default_threshold=initial_rank_threshold,
                    )
                    nrow = normalized_thresholds.get(strategy_id, {})
                    threshold_space = str(nrow.get("threshold_space", "") or "")
                    normalized_threshold = float(
                        nrow.get("normalized_threshold", rank_threshold)
                    )
                    viability_margin = float(nrow.get("viability_margin", 0.0))
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
                        effective_threshold = min(
                            1.0, normalized_threshold + viability_margin
                        )
                        threshold_score = np.nan
                    rank_score = float(chain_results.get("position_size", 0.0) or 0.0)
                    size = rank_score
                    size *= calibration_size_multiplier(
                        raw_score=raw_score,
                        strategy_id=strategy_id,
                        calibration_data=calibration_data,
                        default_threshold=initial_rank_threshold,
                    )
                    if abs(size) < 0.01:
                        tprint(
                            f"Size block: {symbol} {side}/{strategy_id} "
                            f"size={size:.6f}"
                        )
                        continue
                    chain_results["strategy_id"] = strategy_id
                    chain_results["calibrated_score"] = calibrated_score
                    chain_results["rank_threshold"] = rank_threshold
                    chain_results["normalized_threshold"] = normalized_threshold
                    chain_results["viability_margin"] = viability_margin
                    chain_results["effective_threshold"] = effective_threshold
                    decision_rows.append(
                        {
                            "symbol": symbol,
                            "side": side,
                            "size": size,
                            "strategy_id": strategy_id,
                            "raw_score": raw_score,
                            "calibrated_score": calibrated_score,
                            "threshold_space": threshold_space or "rank_percentile",
                            "rank_score": rank_score,
                            "threshold_score": threshold_score,
                            "rank_threshold": rank_threshold,
                            "normalized_threshold": normalized_threshold,
                            "effective_threshold": effective_threshold,
                            "chain_results": chain_results,
                        }
                    )

            _attach_rank_percentile_scores(decision_rows)
            filtered_decision_rows: List[Dict[str, Any]] = []
            for decision in decision_rows:
                if str(decision.get("threshold_space", "")) == "calibrated_score":
                    filtered_decision_rows.append(decision)
                    continue
                rank_pct = float(decision.get("sizer_rank_percentile", np.nan))
                threshold = float(decision.get("effective_threshold", 1.0))
                if not np.isfinite(rank_pct) or rank_pct < threshold:
                    tprint(
                        f"Rank-threshold block: {decision['symbol']} "
                        f"{decision['side']}/{decision['strategy_id']} "
                        f"rank={rank_pct:.6f} threshold={threshold:.6f}"
                    )
                    continue
                decision["threshold_score"] = rank_pct
                chain_results = dict(decision["chain_results"])
                chain_results["sizer_rank_percentile"] = rank_pct
                chain_results["effective_threshold"] = threshold
                decision["chain_results"] = chain_results
                filtered_decision_rows.append(decision)
            decision_rows = filtered_decision_rows
            decision_rows.sort(
                key=lambda row: float(row.get("threshold_score", 0.0)), reverse=True
            )
            for decision in decision_rows:
                symbol = str(decision["symbol"])
                strategy_id = str(decision["strategy_id"])
                chain_results = dict(decision["chain_results"])
                size = float(decision["size"])
                bucket_key = strategy_core_id(strategy_id)
                cooldown_hours = (
                    float(executor.get_cooldown_hours(bucket_key))
                    if hasattr(executor, "get_cooldown_hours")
                    else 0.0
                )
                if _is_symbol_cooldown_blocked(
                    symbol,
                    now=now_utc,
                    logger=logger,
                    executor=executor,
                    cooldown_hours=cooldown_hours,
                ):
                    tprint(
                        f"Cooldown block: {symbol} skipped for {cooldown_hours:.1f}h window"
                    )
                    continue
                if portfolio_mgr is not None:
                    requested_position_usdt = (
                        abs(float(size))
                        if abs(float(size)) > 1.0
                        else abs(float(size)) * float(portfolio_mgr.portfolio_value)
                    )
                    can_enter, info = portfolio_mgr.can_enter_position(
                        symbol=symbol,
                        side=side,
                        strategy_id=strategy_id,
                        confidence_score=float(decision["calibrated_score"]),
                        initial_threshold=float(decision["effective_threshold"]),
                        current_time=now_utc,
                        requested_position_size=requested_position_usdt,
                    )
                    chain_results["portfolio_gate"] = info
                    if not can_enter:
                        tprint(
                            f"Portfolio block: {symbol} {side}/{strategy_id} "
                            f"reason={info.get('reason') or info}"
                        )
                        continue
                    size = min(
                        requested_position_usdt,
                        float(info.get("position_size_cap", requested_position_usdt)),
                    )
                close = panel.get("close")
                price = None
                if close is not None and symbol in close.columns:
                    close_col = close[symbol]
                    dropped = close_col.dropna()
                    price = (
                        dropped.iloc[-1]
                        if isinstance(dropped, (pd.DataFrame, pd.Series))
                        and not dropped.empty
                        else None
                    )
                predictions = {
                    "position_size": size,
                    "meta_pred": chain_results.get("meta_pred", ""),
                    "action": chain_results.get("action", ""),
                    "base_pred": chain_results.get("base_pred", ""),
                    "base_rank_pct": chain_results.get("base_rank_pct", ""),
                    "base_gate_top_frac": chain_results.get("base_gate_top_frac", ""),
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
                ]:
                    if feat_name in feats:
                        feat_df = feats[feat_name]
                        if symbol in feat_df.columns:
                            vals = feat_df[symbol].dropna()
                            if not vals.empty:
                                features_log[feat_name] = vals.iloc[-1]
                trade_result = executor.execute_trade(
                    symbol=symbol,
                    side=side,
                    size=abs(size),
                    price=float(chain_results.get("entry_px") or price),
                    bucket_key=bucket_key,
                )
                _record_trade_execution_health(portfolio_mgr, trade_result)
                if portfolio_mgr is not None and (
                    trade_result.get("success", False)
                    or trade_result.get("status") == "recorded"
                ):
                    portfolio_mgr.record_position_open(
                        symbol=symbol,
                        side=side,
                        strategy_id=strategy_id,
                        position_size=float(abs(size)),
                        entry_price=float(price if price is not None else 0.0),
                        entry_time=now_utc,
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
                    price_slippage_pct=trade_result.get("price_slippage_pct"),
                    spread_proxy_pct=trade_result.get("spread_proxy_pct"),
                    orderbook_snapshot=trade_result.get("orderbook_snapshot"),
                    stop_price=chain_results.get("stop_px")
                    or trade_result.get("stop_price"),
                    actual_entry_price=trade_result.get("realized_entry_price"),
                    exit_reason=trade_result.get("exit_reason"),
                    net_pnl=trade_result.get("net_pnl"),
                )
                results["trades"].append(
                    {
                        "symbol": symbol,
                        "side": side,
                        "size": size,
                        "price": price,
                        "result": trade_result,
                        "strategy_id": strategy_id,
                        "calibrated_score": float(decision["calibrated_score"]),
                    }
                )
        except Exception as e:
            tprint(f"Error running inference chain for {side}: {e}")
            continue

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
    accepted_strategies = resolve_deployment_strategy_filter(data_root, run_id)
    strategy_asset_exclusions = load_strategy_asset_exclusion_filter(data_root, run_id)

    # Initialize orchestrator
    orchestrator = ModelOrchestrator(model_bundle, full_state)

    # Initialize exchange (for live mode)
    exchange = None
    if mode == "live":
        exchange = make_exchange()

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
                "vol_price_diverge",
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
                    "retest_accept",
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

            # Final check for Inf/NaN
            for k in list(feats.keys()):
                arr = np.asarray(feats[k], dtype=np.float32)
                if not np.isfinite(arr).all():
                    n_bad = (~np.isfinite(arr)).sum()
                    tprint(
                        f"  WARNING: {k} has {n_bad} non-finite values, replacing with 0"
                    )
                    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

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


def _monitor_active_position_price_action(
    executor: TradeExecutor,
    *,
    exchange: Optional[Any] = None,
    now: Optional[pd.Timestamp] = None,
) -> Dict[str, Dict[str, Any]]:
    """Monitor active positions and apply 5m trailing/stop policy updates."""
    statuses: Dict[str, Dict[str, Any]] = {}
    active_positions = (
        executor.get_active_positions()
        if hasattr(executor, "get_active_positions")
        else {}
    )
    if not active_positions:
        return statuses

    if hasattr(executor, "monitor_orders_once"):
        try:
            statuses.update(executor.monitor_orders_once())
        except Exception as exc:
            tprint(
                f"  Error monitoring order statuses: {classify_api_error(exc)}: {exc}"
            )

    now_ts = pd.Timestamp(now if now is not None else pd.Timestamp.now(tz="UTC"))
    if now_ts.tzinfo is None:
        now_ts = now_ts.tz_localize("UTC")
    else:
        now_ts = now_ts.tz_convert("UTC")

    cached_5m: Dict[str, pd.DataFrame] = {}
    if exchange is None and hasattr(executor, "fetch_5m_ohlcv_for_positions"):
        try:
            cached_5m = executor.fetch_5m_ohlcv_for_positions()
        except Exception as exc:
            tprint(
                f"  Error reading cached shadow 5m data: "
                f"{classify_api_error(exc)}: {exc}"
            )

    tprint(f"Monitoring {len(active_positions)} active positions for price action...")
    for symbol, position_state in active_positions.items():
        try:
            entry_time = position_state.get("entry_time")
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
                    pd.Timestamp(last_eval_ts) - pd.Timedelta(minutes=5),
                )
            end_time = min(start_time + pd.Timedelta(hours=8), now_ts)
            if start_time >= end_time:
                continue

            ohlcv_5m: Any = None
            if exchange is not None:
                ohlcv_5m = hf_data_loader.fetch_ohlcv_5m(
                    exchange, symbol, start_time, end_time
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
            before_stop = float(position_state.get("stop_price", np.nan))
            position_state["ohlcv_5m_latest"] = bars
            _evaluate_oco_policy(symbol, position_state, bars, executor)
            after_position = (
                executor.get_position(symbol)
                if hasattr(executor, "get_position")
                else None
            )
            status = statuses.setdefault(symbol, {})
            if after_position is None:
                status["price_action"] = {
                    "status": "closed",
                    "bars_evaluated": int(len(bars)),
                    "stop_price_before": before_stop,
                }
                continue
            after_stop = float(after_position.get("stop_price", np.nan))
            status["price_action"] = {
                "status": "updated",
                "bars_evaluated": int(len(bars)),
                "stop_price_before": before_stop,
                "stop_price_after": after_stop,
                "peak_price": after_position.get("peak_price"),
                "mfe": after_position.get("mfe"),
                "last_5m_eval_ts": after_position.get("last_5m_eval_ts"),
            }
            if np.isfinite(before_stop) and np.isfinite(after_stop):
                if abs(after_stop - before_stop) > 1e-12:
                    tprint(
                        f"  [STOP_LOSS] {symbol} stop updated "
                        f"{before_stop:.8f} -> {after_stop:.8f}"
                    )
        except Exception as exc:
            tprint(
                f"  Error evaluating 5m price action for {symbol}: "
                f"{classify_api_error(exc)}: {exc}"
            )
            statuses.setdefault(symbol, {})["price_action_error"] = str(exc)
    return statuses


def _infer_policy_barrier_frac(
    *,
    params: Dict[str, Any],
    entry_price: float,
    stop_price: float,
) -> float:
    """Infer the policy optimiser barrier fraction from params/active stop."""
    for key in ("barrier_frac", "barrier_pct", "atr", "tp_base_pct", "tp_abs_pct"):
        if key not in params:
            continue
        value = float(params.get(key, np.nan))
        if np.isfinite(value) and value > 0.0:
            return value
    sl_mult = max(float(params.get("sl_mult", 1.0) or 1.0), 1e-12)
    if (
        np.isfinite(entry_price)
        and entry_price > 0.0
        and np.isfinite(stop_price)
        and stop_price > 0.0
    ):
        return abs(entry_price - stop_price) / entry_price / sl_mult
    return 0.01


def _policy_optimiser_trailing_stop(
    *,
    side: str,
    entry_price: float,
    peak_price: float,
    mfe: float,
    stop_price: float,
    params: Dict[str, Any],
) -> float:
    """Return the policy-optimiser trailing threshold for the current MFE."""
    if not bool(params.get("enable_trailing", False)):
        return float(stop_price)
    barrier_frac = _infer_policy_barrier_frac(
        params=params,
        entry_price=entry_price,
        stop_price=stop_price,
    )
    activation = float(params.get("trailing_override_alpha", 0.0)) * barrier_frac
    if float(mfe) <= activation:
        return float(stop_price)

    trailing_power = float(params.get("trailing_power", 1.0))
    trailing_squash_divisor = max(
        float(params.get("trailing_squash_divisor", 1.0)), 1e-6
    )
    giveback_beta = float(params.get("giveback_beta", 0.5))
    profit_above_activation = max(float(mfe) - activation, 0.0)
    trail_dist_ret = (profit_above_activation**trailing_power) / (
        trailing_squash_divisor
    )
    trail_dist_abs = trail_dist_ret * float(entry_price)
    giveback_dist = giveback_beta * barrier_frac

    if str(side).lower() == "long":
        power_stop = float(peak_price) - trail_dist_abs
        giveback_stop = float(peak_price) * (1.0 - giveback_dist)
        candidate = min(power_stop, giveback_stop)
        return max(float(stop_price), float(candidate))

    power_stop = float(peak_price) + trail_dist_abs
    giveback_stop = float(peak_price) * (1.0 + giveback_dist)
    candidate = max(power_stop, giveback_stop)
    return min(float(stop_price), float(candidate))


def main():
    import argparse

    _configure_numba_threading_layer()
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="Run live trading mode")
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
        default=300,
        help="Challenger check interval in seconds (default: 300 = 5 min)",
    )
    parser.add_argument(
        "--lookback-hours",
        type=int,
        default=24 * 60,
        help="Lookback hours for features",
    )
    parser.add_argument(
        "--execution-account",
        choices=["spot", "margin"],
        default=DEFAULT_EXECUTION_ACCOUNT,
        help=f"Execution account for live orders (default: {DEFAULT_EXECUTION_ACCOUNT})",
    )
    parser.add_argument(
        "--margin-mode",
        choices=["cross", "isolated"],
        default=DEFAULT_MARGIN_MODE,
        help="Margin mode when --execution-account margin is used",
    )
    parser.add_argument(
        "--live-quote-currency",
        default=DEFAULT_LIVE_QUOTE_CURRENCY,
        help=(
            "Quote currency to trade/download at inference time "
            f"(default: {DEFAULT_LIVE_QUOTE_CURRENCY})"
        ),
    )
    args = parser.parse_args()

    # Initialize components
    config = load_inference_config()
    config["execution_account"] = args.execution_account
    config["margin_mode"] = args.margin_mode
    config["live_quote_currency"] = str(args.live_quote_currency or "USDC").upper()
    config["mode"] = "live" if args.live else "shadow"
    exchange = make_exchange()
    runtime_bucket_params = _attach_runtime_bucket_params(config)
    model_bundle = load_full_state(config["run_id"], config["data_root"])
    if isinstance(model_bundle, dict):
        model_bundle["bucket_params"] = runtime_bucket_params
    effective_model_bundle = _effective_runtime_model_bundle(model_bundle, config)
    validate_live_feature_contract(effective_model_bundle, strict=True)
    accepted_strategies = resolve_deployment_strategy_filter(
        config["data_root"], config["run_id"]
    )
    required_feature_keys = get_inference_required_feature_keys(
        effective_model_bundle,
        accepted_strategies,
    )
    calibration_data = load_calibration_curves(config["data_root"], config["run_id"])
    normalized_thresholds = _load_normalized_threshold_map(
        config["data_root"], config["run_id"]
    )
    validate_calibration_artifacts(
        config["data_root"], config["run_id"], calibration_data, strict=False
    )
    strategy_asset_exclusions = load_strategy_asset_exclusion_filter(
        config["data_root"], config["run_id"]
    )
    validate_deployment_model_coverage(
        effective_model_bundle,
        accepted_strategies,
        strict=True,
    )

    # Initialize data fetcher with incremental updates
    data_fetcher = DataFetcher(exchange, config["data_root"])
    inference_defaults = get_inference_defaults()
    panel_warmup_hours = (
        max(
            int(inference_defaults["trend_sma_hours"]),
            int(inference_defaults["gate_vol_lookback_hours"]),
        )
        + 72
    )
    panel_lookback_hours = max(int(args.lookback_hours), panel_warmup_hours)

    # Step 9 universe split:
    # - download_symbols: full live Binance quote/margin universe, refreshed daily
    # - symbols: tradable subset restricted to the active training universe
    universe_state = resolve_inference_universes(
        exchange,
        data_root=config["data_root"],
        run_id=config["run_id"],
        explicit_symbols=args.symbols,
        live_quote_currency=config["live_quote_currency"],
    )
    download_symbols = list(universe_state["download_symbols"])
    symbols = list(universe_state["tradable_symbols"])
    if not symbols:
        tprint(
            "Warning: tradable universe is empty after training-universe restriction"
        )

    # Initialize on startup with historical data
    tprint("Initializing with historical data...")
    data_fetcher.initialize_with_historical_data(
        download_symbols, lookback_hours=args.lookback_hours
    )

    # Initialize other components
    orchestrator = ModelOrchestrator(model_bundle, config)
    executor = TradeExecutor(
        mode="live" if args.live else "shadow",
        exchange=exchange,
        bucket_params=runtime_bucket_params,
        config=config,
    )
    logger = TradeLogger()
    daily_reporter = DailyDeploymentReporter(
        state_path=str(
            config.get("daily_report_state_path")
            or "extreme_price_movements/logs/daily_report_state.json"
        )
    )
    portfolio_mgr = PortfolioManager(
        max_positions=4,
        max_portfolio_pct=None,
        max_position_usdt=None,
        max_position_pct=0.15,
        cooldown_hours=24.0,
        max_same_side_pct=1.0,
        max_same_strategy_pct=0.50,
    )

    # Setup scheduling
    if args.challenger_interval > 0:
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
            ),
            daemon=True,
        )
        challenger_thread.start()

    # Main inference loop - run every 15m
    last_hourly_sync = None
    last_universe_refresh_day = pd.Timestamp.utcnow().floor("D")
    while True:
        try:
            current_time = pd.Timestamp.now(tz="UTC").floor("15min")
            tprint(f"\n=== Running 15m inference at {current_time} ===")

            current_day = pd.Timestamp.utcnow().floor("D")
            if current_day > last_universe_refresh_day and not args.symbols:
                universe_state = resolve_inference_universes(
                    exchange,
                    data_root=config["data_root"],
                    run_id=config["run_id"],
                    live_quote_currency=config["live_quote_currency"],
                )
                download_symbols[:] = universe_state["download_symbols"]
                symbols[:] = universe_state["tradable_symbols"]
                last_universe_refresh_day = current_day
                tprint(
                    "Daily Binance universe refresh complete: "
                    f"download={len(download_symbols)} tradable={len(symbols)}"
                )

            # Fetch full universe hourly, shortly after the hour boundary.
            current_hour = current_time.floor("h")
            if (last_hourly_sync is None) or (current_hour > last_hourly_sync):
                _sleep_until_hourly_ohlcv_window(
                    current_time,
                    delay_seconds=float(config.get("hourly_ohlcv_delay_seconds", 5.0)),
                )
                data_fetcher.fetch_hourly_universe_once(
                    download_symbols,
                    max_workers=int(config.get("hourly_ohlcv_workers", 16)),
                    no_progress_timeout_seconds=float(
                        config.get("hourly_ohlcv_no_progress_timeout_seconds", 60.0)
                    ),
                    check_recent_gaps_days=7,
                )
                last_hourly_sync = current_hour

            panel = data_fetcher.get_panel(
                download_symbols, lookback_hours=panel_lookback_hours
            )
            tradable_panel = _subset_panel(panel, symbols)
            (
                thresholds,
                long_cands,
                short_cands,
                features,
            ) = _select_candidates_and_load_features(
                panel=tradable_panel,
                symbols=symbols,
                run_id=config["run_id"],
                data_root=config["data_root"],
                cfg=get_runtime_cfg(),
                lookback_hours=args.lookback_hours,
                required_feature_keys=required_feature_keys,
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
                initial_rank_threshold=0.5,
                strategy_asset_exclusions=strategy_asset_exclusions,
                preselected_long_candidates=long_cands,
                preselected_short_candidates=short_cands,
            )
            tprint(
                f"Inference batch complete: download_symbols={len(download_symbols)} "
                f"tradable_symbols={len(symbols)} "
                f"candidates={len(long_cands) + len(short_cands)} "
                f"trades={len(results['trades'])}"
            )
            try:
                daily_reporter.maybe_run(
                    exchange=exchange,
                    portfolio_mgr=portfolio_mgr,
                    trade_logger=logger,
                    config=config,
                )
            except Exception as exc:
                tprint(f"Daily deployment report failed: {exc}")

            # Sleep until next 15-minute interval
            next_interval = current_time + pd.Timedelta(minutes=15)
            sleep_seconds = (next_interval - pd.Timestamp.now(tz="UTC")).total_seconds()
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)

        except KeyboardInterrupt:
            tprint("Shutting down...")
            executor.shutdown()
            break
        except Exception as e:
            tprint(f"Error in inference loop: {e}")
            import traceback

            tprint(traceback.format_exc())
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
):
    """
    calibration_data = calibration_data or {}
    Run challenger monitoring every 5 minutes.

    Check existing candidates for better opportunities and monitor OCO orders:
    1. Check for better opportunities (existing challenger logic)
    2. In live mode: fetch 5m OHLCV data for active positions and evaluate OCO orders

    Args:
        symbols: List of trading symbols
        data_fetcher: DataFetcher instance
        orchestrator: ModelOrchestrator instance
        executor: TradeExecutor instance
        logger: TradeLogger instance
        config: Configuration dictionary
        interval: Check interval in seconds (default 300 = 5 min)
    """
    while True:
        try:
            current_time = pd.Timestamp.now(tz="UTC")
            tprint(f"\n=== Challenger monitor at {current_time} ===")

            # Full-universe OHLCV ingestion is handled hourly by the main loop.
            # The 5m challenger reads the latest persisted panel and avoids
            # creating an extra all-symbol API burst.
            panel = data_fetcher.get_panel(symbols, lookback_hours=panel_lookback_hours)
            (
                thresholds,
                long_cands,
                short_cands,
                features,
            ) = _select_candidates_and_load_features(
                panel=panel,
                symbols=symbols,
                run_id=config["run_id"],
                data_root=config["data_root"],
                cfg=get_runtime_cfg(),
                lookback_hours=get_inference_defaults()["lookback_periods"],
                required_feature_keys=required_feature_keys,
            )

            # Check if any new better opportunities
            for symbol in long_cands + short_cands:
                side = "long" if symbol in long_cands else "short"
                bucket_key = f"{side.lower()}_mr"
                cooldown_hours = (
                    float(executor.get_cooldown_hours(bucket_key))
                    if hasattr(executor, "get_cooldown_hours")
                    else 0.0
                )
                if _is_symbol_cooldown_blocked(
                    symbol,
                    now=current_time,
                    logger=logger,
                    executor=executor,
                    cooldown_hours=cooldown_hours,
                ):
                    continue
                market_data = _build_market_snapshot(symbol, panel, features)

                result: Dict[str, Any] = {}
                strategy_ids = (
                    orchestrator.available_strategies(side, accepted_strategies)
                    if hasattr(orchestrator, "available_strategies")
                    else [f"{side}_mr"]
                )
                for selected_strategy in strategy_ids:
                    if _is_symbol_blocked_for_strategy(
                        symbol, str(selected_strategy), strategy_asset_exclusions
                    ):
                        continue
                    try:
                        candidate_result = orchestrator.run_full_chain(
                            symbol, side, features, panel, kind=selected_strategy
                        )
                    except TypeError as exc:
                        if "kind" not in str(exc):
                            raise
                        candidate_result = orchestrator.run_full_chain(
                            symbol, side, features, panel
                        )
                    strategy_id = str(
                        candidate_result.get("strategy_id")
                        or strategy_core_id(str(selected_strategy))
                    )
                    if _is_symbol_blocked_for_strategy(
                        symbol, strategy_id, strategy_asset_exclusions
                    ):
                        continue
                    raw_score = float(candidate_result.get("meta_pred", 0.0) or 0.0)
                    calibrated_score, rank_threshold = calibrated_score_and_threshold(
                        raw_score=raw_score,
                        strategy_id=strategy_id,
                        calibration_data=calibration_data,
                        default_threshold=0.5,
                    )
                    if (
                        candidate_result.get("action") == "enter"
                        and calibrated_score >= rank_threshold
                    ):
                        candidate_result["strategy_id"] = strategy_id
                        candidate_result["calibrated_score"] = calibrated_score
                        candidate_result["rank_threshold"] = rank_threshold
                        result = candidate_result
                        break
                if not result:
                    continue
                bucket_key = strategy_core_id(
                    str(result.get("strategy_id", bucket_key))
                )

                # Compare with existing positions
                current_pos = executor.get_position(symbol)
                if current_pos:
                    # Check if new signal is better
                    if (
                        result.get("meta_pred", 0)
                        > current_pos.get("meta_pred", 0) * 1.2
                    ):
                        # Replace position
                        executor.close_position(symbol)
                        executor.execute_trade(
                            symbol,
                            side,
                            result.get("position_size"),
                            bucket_key=bucket_key,
                        )
                        logger.log_trade(
                            result,
                            orchestrator.get_last_results(),
                            market_data,
                            {**config, "mode": executor.mode, **thresholds},
                        )

            active_positions = (
                executor.get_active_positions()
                if hasattr(executor, "get_active_positions")
                else {}
            )
            if active_positions:
                _monitor_active_position_price_action(
                    executor,
                    exchange=executor.exchange,
                    now=current_time,
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
    """
    Evaluate OCO policy and update orders if needed based on 5m OHLCV data.

    Args:
        symbol: Trading symbol
        position_state: Position state dictionary containing entry info and OCO params
        ohlcv_5m: 5m OHLCV DataFrame
        executor: TradeExecutor instance for placing/updating orders
    """
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
        entry_price = float(position_state.get("entry_price", 0.0) or 0.0)
        bucket_key = position_state.get("bucket_key", "")
        params = executor.get_bucket_params(bucket_key)
        stop_price = float(position_state.get("stop_price", np.nan))
        peak_price = float(position_state.get("peak_price", entry_price) or entry_price)
        mfe = float(position_state.get("mfe", 0.0) or 0.0)
        enable_trailing = bool(params.get("enable_trailing", True))
        policy_style_trailing = any(
            key in params
            for key in (
                "trailing_power",
                "trailing_squash_divisor",
                "trailing_override_alpha",
                "giveback_beta",
            )
        )
        giveback_pct = float(params.get("giveback_pct", 0.005))
        trail_mult = float(params.get("trail_mult", 0.25))
        profit_lock = (
            float(params["profit_lock_amount"])
            if "profit_lock_amount" in params
            else None
        )
        last_bar_ts = bars.index[-1]

        for bar_ts, row in bars.iterrows():
            bar_open = float(row["open"])
            bar_high = float(row["high"])
            bar_low = float(row["low"])
            bar_close = float(row["close"])

            if side == "long":
                mfe = max(mfe, (bar_high - entry_price) / max(entry_price, 1e-12))
                if np.isfinite(stop_price) and bar_low <= stop_price:
                    exit_reason = "stop_loss_5m"
                    exit_price = stop_price
                    executor.close_position(
                        symbol, price=float(exit_price), reason=exit_reason
                    )
                    return
                peak_price = max(peak_price, bar_high)
                new_stop = stop_price
                if enable_trailing and peak_price > entry_price:
                    if policy_style_trailing:
                        new_stop = _policy_optimiser_trailing_stop(
                            side=side,
                            entry_price=entry_price,
                            peak_price=peak_price,
                            mfe=mfe,
                            stop_price=stop_price,
                            params=params,
                        )
                    else:
                        giveback_stop = peak_price * (1.0 - giveback_pct)
                        trailing_stop = entry_price + trail_mult * (
                            peak_price - entry_price
                        )
                        new_stop = max(float(stop_price), giveback_stop, trailing_stop)
                    if profit_lock is not None:
                        locked_stop = entry_price * (1.0 + profit_lock)
                        new_stop = max(float(new_stop), locked_stop)
                if np.isfinite(new_stop) and new_stop > float(stop_price):
                    stop_price = float(new_stop)
            else:
                mfe = max(mfe, (entry_price - bar_low) / max(entry_price, 1e-12))
                if np.isfinite(stop_price) and bar_high >= stop_price:
                    exit_reason = "stop_loss_5m"
                    exit_price = stop_price
                    executor.close_position(
                        symbol, price=float(exit_price), reason=exit_reason
                    )
                    return
                peak_price = min(peak_price, bar_low)
                new_stop = stop_price
                if enable_trailing and peak_price < entry_price:
                    if policy_style_trailing:
                        new_stop = _policy_optimiser_trailing_stop(
                            side=side,
                            entry_price=entry_price,
                            peak_price=peak_price,
                            mfe=mfe,
                            stop_price=stop_price,
                            params=params,
                        )
                    else:
                        giveback_stop = peak_price * (1.0 + giveback_pct)
                        trailing_stop = entry_price - trail_mult * (
                            entry_price - peak_price
                        )
                        new_stop = min(float(stop_price), giveback_stop, trailing_stop)
                    if profit_lock is not None:
                        locked_stop = entry_price * (1.0 - profit_lock)
                        new_stop = min(float(new_stop), locked_stop)
                if np.isfinite(new_stop) and new_stop < float(stop_price):
                    stop_price = float(new_stop)

        executor.update_position_policy_state(
            symbol,
            stop_price=stop_price,
            peak_price=peak_price,
            mfe=mfe,
            last_5m_eval_ts=last_bar_ts,
        )
    except Exception as e:
        tprint(f"  [STOP_LOSS] Error evaluating stop policy for {symbol}: {e}")


if __name__ == "__main__":
    main()
