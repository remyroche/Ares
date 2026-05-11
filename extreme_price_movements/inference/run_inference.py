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
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from extreme_price_movements import hf_data_loader
from extreme_price_movements.feature_transforms import CausalFeatureTransformer
from extreme_price_movements.inference.candidate_selector import (
    build_strategy_candidate_masks,
    select_candidates,
)
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
    _compute_policy_barrier_pct,
    compute_selector_features,
    generate_features,
    get_features_for_candidates,
    get_inference_required_feature_keys,
    get_market_data,
    load_or_compute_features,
    raw_required_feature_keys,
)
from extreme_price_movements.inference.liquidity_precheck import (
    compute_price_gap_rank_penalty,
    evaluate_orderbook_liquidity,
    fetch_ticker_snapshot,
    marketable_limit_price,
)
from extreme_price_movements.inference.model_orchestrator import ModelOrchestrator
from extreme_price_movements.inference.parity import (
    calibrated_score_and_threshold,
    load_strategy_asset_exclusion_filter,
    resolve_deployment_strategy_filter,
    strategy_core_id,
    strategy_id_matches,
    validate_calibration_artifacts,
    validate_deployment_model_coverage,
    validate_live_feature_contract,
    validate_meta_feature_contract_artifact,
    validate_required_feature_frames,
)
from extreme_price_movements.inference.prediction_ledger import PredictionLedger
from extreme_price_movements.inference.simple_policy_stop import (
    load_simple_policy_stop_params_by_strategy,
)
from extreme_price_movements.inference.portfolio_policy import (
    PortfolioPolicyConfig,
    compute_rank_based_position_size,
    load_portfolio_policy_config,
)
from extreme_price_movements.inference.safety_switches import (
    MarketKillSwitch,
    StrategyKillSwitch,
)
from extreme_price_movements.inference.symbol_mapping import (
    normalise_symbol,
    symbol_base,
)
from extreme_price_movements.inference.simple_policy_stop import (
    SimplePolicyStopParamsError,
    compute_simple_policy_stop_decision,
)
from extreme_price_movements.inference.trade_executor import TradeExecutor
from extreme_price_movements.inference.trade_logger import (
    TradeLogger,
    log_trade_decision,
)
from extreme_price_movements.portfolio_manager import PortfolioManager
from extreme_price_movements.utils import tprint

_FEATURE_COMPUTE_LOCK = threading.RLock()
LIVE_TEST_RANK_THRESHOLD = 0.90
LOSING_TRADE_COOLDOWN_HOURS = 12.0
_HISTORICAL_SCORE_RANK_CACHE: Dict[tuple[str, str, str, str], np.ndarray] = {}


class _StageTimer:
    """Log live-loop stage latencies without affecting trading decisions."""

    def __init__(self, label: str):
        self.label = label
        self.start = time.perf_counter()
        self.last = self.start

    def mark(self, stage: str) -> None:
        now = time.perf_counter()
        tprint(
            f"[Timing] {self.label}.{stage}: "
            f"stage={now - self.last:.3f}s total={now - self.start:.3f}s"
        )
        self.last = now


def _is_live_test_mode(mode_or_executor: Any) -> bool:
    mode = getattr(mode_or_executor, "mode", mode_or_executor)
    return str(mode or "").strip().lower() in {"live-test", "live_test", "livetest"}


def _order_identifier(order_payload: Any) -> str:
    if isinstance(order_payload, dict):
        raw = order_payload.get("id") or order_payload.get("clientOrderId")
        return str(raw) if raw is not None else ""
    return ""


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


def _load_policy_selection_rules(data_root: str, run_id: str) -> Dict[str, Any]:
    base = Path(data_root) / "artifacts" / run_id
    for strategy_path in (
        base / "policy_params" / "strategy_for_inference.json",
        base / "strategy_for_inference.json",
    ):
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
    base = Path(data_root) / "artifacts" / run_id
    for strategy_path in (
        base / "policy_params" / "strategy_for_inference.json",
        base / "strategy_for_inference.json",
    ):
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


def _load_lgbm_strategy_mask_rows(
    data_root: str, run_id: str
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
        )
    except Exception as exc:
        tprint(f"Could not load LGBM strategy mask rows: {exc}")
        return {}

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
            },
        )
    tprint(f"Loaded {len(rows)} LGBM strategy mask row(s) for inference gating")
    return out


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
            else:
                decision_rows[row_i]["sizer_rank_percentile"] = float(rank_out[local_i])


def _should_log_prediction_candidate(
    decision: Dict[str, Any],
    *,
    policy: PortfolioPolicyConfig,
) -> bool:
    rank = _safe_float(decision.get("sizer_rank_percentile"))
    if not np.isfinite(rank):
        rank = _safe_float(decision.get("threshold_score"))
    if not np.isfinite(rank):
        return False
    return rank >= float(1.0 - policy.top_prediction_ledger_pct)


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
    sizing = dict(chain.get("portfolio_rank_sizing") or {})
    trade = dict(trade_result or {})
    normalized_rank = _safe_float(
        chain.get(
            "sizer_rank_percentile",
            decision.get(
                "sizer_rank_percentile", decision.get("threshold_score", np.nan)
            ),
        )
    )
    final_threshold = _safe_float(
        chain.get("effective_threshold", decision.get("effective_threshold", np.nan))
    )
    order = trade.get("order") if isinstance(trade.get("order"), dict) else {}
    return {
        "timestamp": timestamp,
        "symbol": decision.get("symbol"),
        "side": side,
        "strategy_id": decision.get("strategy_id"),
        "raw_prediction_score": decision.get("raw_score"),
        "rank_score_source": chain.get(
            "rank_score_source", decision.get("rank_score_source")
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
        "passed_rank_gate": bool(
            np.isfinite(normalized_rank)
            and np.isfinite(final_threshold)
            and normalized_rank >= final_threshold
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
        "ticker_bid": snap.get("bid"),
        "ticker_ask": snap.get("ask"),
        "ticker_mid": snap.get("mid"),
        "spread_bps": snap.get("spread_bps"),
        "orderbook_capacity_quote_within_slippage": snap.get(
            "orderbook_capacity_quote_within_slippage"
        ),
        "max_orderbook_slippage_bps": snap.get("max_orderbook_slippage_bps"),
        "expected_fill_price": snap.get("expected_fill_price"),
        "expected_fill_slippage_bps": snap.get("expected_fill_slippage_bps"),
        "expected_total_entry_friction_bps": snap.get(
            "expected_total_entry_friction_bps"
        ),
        "liquidity_capacity_weight": snap.get("liquidity_capacity_weight"),
        "liquidity_reject_reason": liquidity_reject_reason,
        "signal_price": snap.get("signal_price"),
        "decision_mid": snap.get("decision_mid", snap.get("mid")),
        "signal_gap_bps": snap.get("signal_gap_bps"),
        "was_traded": bool(was_traded),
        "order_id": trade.get("order_id") or order.get("id"),
        "entry_price_expected": snap.get("expected_fill_price"),
        "entry_price_actual": trade.get("realized_entry_price"),
        "outcome_status": None,
        "tp_hit": None,
        "sl_hit": None,
        "ambiguous_both_hit": None,
        "resolved_at": None,
    }


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
    ridge_sizer = full_state.get("ridge_sizer")
    if ridge_sizer is not None and getattr(ridge_sizer, "best_params_", None):
        bucket_params.setdefault(
            "cooldown_hours", float(ridge_sizer.best_params_.get("cooldown_hours", 0.0))
        )
    stop_params = load_simple_policy_stop_params_by_strategy(
        str(config.get("data_root", "data")), str(config.get("run_id", ""))
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
    lgbm_strategy_mask_rows: Optional[Dict[str, Dict[str, Any]]] = None,
) -> tuple[
    Dict[str, float],
    List[str],
    List[str],
    Dict[str, pd.DataFrame],
    Dict[str, List[str]],
]:
    with _FEATURE_COMPUTE_LOCK:
        timer = _StageTimer("candidate_feature_load")
        raw_feature_keys = raw_required_feature_keys(required_feature_keys)
        selector_feats = compute_selector_features(panel, symbols)
        timer.mark("selector_features")
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
        timer.mark("selector_candidates")
        strategy_candidate_masks: Dict[str, List[str]] = {}
        if lgbm_strategy_mask_rows:
            strategy_candidate_masks = build_strategy_candidate_masks(
                panel,
                selector_feats,
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
        selected_symbols = sorted(set(long_cands + short_cands))
        if not selected_symbols:
            return (
                thresholds,
                long_cands,
                short_cands,
                selector_feats,
                strategy_candidate_masks,
            )
        model_feats = load_or_compute_features(
            panel=_subset_panel(panel, selected_symbols),
            basket_syms=selected_symbols,
            run_id=run_id,
            data_root=data_root,
            cfg=cfg,
            lookback_hours=lookback_hours,
            required_feature_keys=raw_feature_keys,
        )
        timer.mark("model_features")
        validate_required_feature_frames(
            model_feats,
            raw_feature_keys,
            symbols=selected_symbols,
            strict=True,
        )
        timer.mark("validate_model_features")
        return (
            thresholds,
            long_cands,
            short_cands,
            model_feats,
            strategy_candidate_masks,
        )


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
    trade_recap = str(closed_trade.get("trade_recap") or "").strip()
    body = "\n".join(
        [
            "Extreme price movement trade close",
            "",
            "Trade",
            f"  symbol: {symbol}",
            f"  side: {side}",
            f"  strategy_id: {closed_trade.get('strategy_id')}",
            f"  exit_reason: {reason}",
            f"  exit_reason_detail: {reason_detail}",
            f"entry_time: {closed_trade.get('entry_time')}",
            f"exit_time: {closed_trade.get('exit_time')}",
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
            f"base_pred: {_format_float(closed_trade.get('base_pred'), digits=6)}",
            f"base_rank_pct: {_format_float(closed_trade.get('base_rank_pct'), digits=6)}",
            f"base_train_rank_pct: {_format_float(closed_trade.get('base_train_rank_pct'), digits=6)}",
            f"base_gate_top_frac: {closed_trade.get('base_gate_top_frac')}",
            f"meta_pred: {_format_float(closed_trade.get('meta_pred'), digits=6)}",
            f"meta_train_rank_pct: {_format_float(closed_trade.get('meta_train_rank_pct'), digits=6)}",
            f"rank_score_source: {closed_trade.get('rank_score_source')}",
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
            f"requested_quote_size: {_format_float(closed_trade.get('quote_size'))}",
            "requested_base_amount: "
            f"{_format_float(closed_trade.get('requested_base_amount'))}",
            "pnl_scope: position notional only; excludes whole-wallet equity, "
            "other positions, and borrow interest",
            f"net_pnl_quote_est_position: {_format_float(net_pnl_amount)}",
            f"net_pnl_pct_position_notional: {_format_pct(net_pnl_pct)}",
            f"gross_pnl_quote_est_position: {_format_float(closed_trade.get('gross_pnl'))}",
            f"gross_pnl_pct_position_notional: {_format_pct(gross_pnl_pct)}",
            "gross_to_net_cost_quote: "
            f"{_format_float(closed_trade.get('gross_to_net_cost_quote'))}",
            "gross_to_net_cost_pct_position_notional: "
            f"{_format_pct(closed_trade.get('gross_to_net_cost_pct'))}",
            f"entry_fee_quote: {_format_float(closed_trade.get('entry_fee_quote'))}",
            f"exit_fee_quote: {_format_float(closed_trade.get('exit_fee_quote'))}",
            f"mfe: {_format_pct(closed_trade.get('mfe'))}",
            f"mae: {_format_pct(closed_trade.get('mae'))}",
            f"stop_price: {closed_trade.get('stop_price')}",
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
            f"stop_price: {_format_float(trade_result.get('stop_price'))}",
            f"stop_order_id: {trade_result.get('stop_order_id')}",
            f"base_pred: {_format_float(predictions.get('base_pred'), digits=6)}",
            f"meta_pred: {_format_float(predictions.get('meta_pred'), digits=6)}",
            f"calibrated_score: {_format_float(decision.get('calibrated_score'), digits=6)}",
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
        data_root = Path(str(config.get("data_root", "data")))
        run_id = str(config.get("run_id", "latest"))
        out_dir = data_root / "artifacts" / run_id / "live_reconciliation"
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


def _sync_reconciled_positions_to_portfolio_manager(
    executor: TradeExecutor,
    portfolio_mgr: PortfolioManager,
) -> None:
    """Mirror startup-reconciled executor positions into portfolio risk state."""
    try:
        active_positions = executor.get_active_positions()
    except Exception as exc:
        tprint(f"[PortfolioManager] Startup reconcile sync skipped: {exc}")
        return
    for symbol, state in active_positions.items():
        if not isinstance(state, dict):
            continue
        if not bool(state.get("external_position")):
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
        "symbol_halted",
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
    if isinstance(panel, dict):
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
    freq = pd.Timedelta(minutes=int(timeframe_minutes))
    boundary = now_ts.floor(f"{int(timeframe_minutes)}min")
    # Add a small guard buffer and re-check after sleeping. macOS can wake a
    # process a fraction early, which can make the hourly loop miss the newly
    # closed candle and then skip it as stale on the next 15m tick.
    guard_seconds = 1.5
    target = boundary + pd.Timedelta(seconds=float(delay_seconds) + guard_seconds)
    if now_ts >= target:
        target = (
            boundary + freq + pd.Timedelta(seconds=float(delay_seconds) + guard_seconds)
        )
    while True:
        now_ts = pd.Timestamp.now(tz="UTC")
        sleep_seconds = (target - now_ts).total_seconds()
        if sleep_seconds <= 0:
            break
        time.sleep(float(min(sleep_seconds, 5.0)))


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
    model_info = getattr(orchestrator, "alpha_by_strategy", {}).get(strategy_id)
    if model_info is None:
        model_info = getattr(orchestrator, "alpha_by_strategy", {}).get(
            f"{side}_{strategy_id}"
        )
    if isinstance(model_info, dict):
        feat_cols = list(model_info.get("feat_cols") or [])
        if feat_cols:
            available_cols = sum(
                1 for col in feat_cols if col in candidate_features.columns
            )
            try:
                aligned = orchestrator._align_alpha_feature_contract(  # noqa: SLF001
                    candidate_features,
                    feat_cols,
                )
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
    strategy_kill_switch: Optional[StrategyKillSwitch] = None,
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
    timer = _StageTimer("run_inference_step")
    total_entries_executed = 0
    calibration_data = calibration_data or {}
    normalized_thresholds = normalized_thresholds or {}
    strategy_candidate_masks = strategy_candidate_masks or {}
    runtime_config = dict(getattr(executor, "config", {}) or {})
    live_test_mode = _is_live_test_mode(executor)
    portfolio_policy = portfolio_policy or PortfolioPolicyConfig()
    prediction_ledger_rows: List[Dict[str, Any]] = []
    _log_generated_feature_frames(feats)
    _log_concurrent_positions_snapshot(portfolio_mgr, label="start")
    timer.mark("startup_logging")
    if live_test_mode:
        tprint(
            "LIVE-TEST mode active: production decision path with quote clamp="
            f"[{portfolio_policy.live_test_min_quote_notional:.2f}, "
            f"{portfolio_policy.live_test_quote_notional:.2f}] USDC. "
            "Rank thresholds are loaded from the deployment policy artifacts."
        )

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
    tprint(
        "Order selection policy: "
        f"top {max(1, int(max_entries_per_side))} per side, "
        f"global entry cap={max(1, int(max_entries_total))}; "
        "within each side candidates are sorted by threshold_score then calibrated_score"
    )

    # Step 2: Process long candidates
    for side, candidates in [("long", long_cands), ("short", short_cands)]:
        if not candidates:
            continue

        # Get features for candidates
        candidate_features = get_features_for_candidates(feats, candidates)
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
                timer.mark(
                    f"{side}_{strategy_core_id(str(selected_strategy))}_base_gate"
                )
                side_metrics["base_gate_pass"] += int(len(base_gate))
                batch_meta_preds = pd.Series(dtype=float)
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
                    calibrated_score, rank_threshold = calibrated_score_and_threshold(
                        raw_score=raw_score,
                        strategy_id=strategy_id,
                        calibration_data=calibration_data,
                        default_threshold=initial_rank_threshold,
                    )
                    run_cfg = getattr(executor, "config", {}) or {}
                    artifact_data_root = str(run_cfg.get("data_root", "data"))
                    artifact_run_id = str(run_cfg.get("run_id", "latest"))
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
                    chain_results["calibrated_score"] = calibrated_score
                    chain_results["rank_threshold"] = rank_threshold
                    chain_results["normalized_threshold"] = normalized_threshold
                    chain_results["viability_margin"] = viability_margin
                    chain_results["effective_threshold"] = effective_threshold
                    chain_results["base_train_rank_pct"] = base_hist_rank_pct
                    chain_results["meta_train_rank_pct"] = meta_hist_rank_pct
                    chain_results["rank_score_source"] = (
                        "historical_meta_oof_percentile"
                        if np.isfinite(meta_hist_rank_pct)
                        else "live_batch_percentile_fallback"
                    )
                    side_metrics["threshold_pass"] += 1
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
                            "rank_score_source": chain_results["rank_score_source"],
                            "threshold_score": threshold_score,
                            "rank_threshold": rank_threshold,
                            "normalized_threshold": normalized_threshold,
                            "effective_threshold": effective_threshold,
                            "policy_sizing": policy_size,
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
                if np.isfinite(rank_pct):
                    rank_pct_all.append(rank_pct)
                if not np.isfinite(rank_pct) or rank_pct < threshold:
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
                                portfolio_reject_reason="rank_below_dynamic_threshold",
                            )
                        )
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
                key=lambda row: (
                    float(row.get("threshold_score", 0.0)),
                    float(row.get("calibrated_score", 0.0)),
                ),
                reverse=True,
            )
            decision_rows = decision_rows[: max(1, int(max_entries_per_side))]
            tprint(
                f"Top-{max(1, int(max_entries_per_side))} selection [{side}]: "
                f"selected={len(decision_rows)} from ranked_decisions "
                f"(global_remaining={max(0, int(max_entries_total) - total_entries_executed)})"
            )
            for decision in decision_rows:
                if total_entries_executed >= max(1, int(max_entries_total)):
                    tprint(
                        f"Global entry cap reached ({total_entries_executed}/{max_entries_total}); skipping remaining ranked decisions"
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
                if strategy_kill_switch is not None:
                    strategy_switch_decision = strategy_kill_switch.is_blocked(
                        strategy_id
                    )
                    if not strategy_switch_decision.allow_new_entries:
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
                                    portfolio_decision="strategy_kill_switch_rejected",
                                    portfolio_reject_reason=(
                                        strategy_switch_decision.reason
                                    ),
                                )
                            )
                        tprint(
                            f"Strategy kill-switch block: {symbol} "
                            f"{side}/{strategy_id} "
                            f"reason={strategy_switch_decision.reason}"
                        )
                        side_metrics["non_fatal_issues"] += 1
                        continue
            cooldown_hours = LOSING_TRADE_COOLDOWN_HOURS
            symbol_block_reason = _symbol_entry_block_reason(
                symbol,
                now=now_utc,
                logger=logger,
                executor=executor,
                cooldown_hours=cooldown_hours,
            )
            if symbol_block_reason:
                if prediction_ledger is not None and _should_log_prediction_candidate(
                    decision, policy=portfolio_policy
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
                base_train_rank = _safe_float(chain_results.get("base_train_rank_pct"))
                meta_train_rank = _safe_float(chain_results.get("meta_train_rank_pct"))
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
                sizing_audit = compute_rank_based_position_size(
                    wallet_value=float(capacity["wallet_value"]),
                    open_notional=float(capacity["open_notional"]),
                    adjusted_rank_score=rank_for_size,
                    final_threshold=threshold_for_size,
                    policy=portfolio_policy,
                    liquidity_capacity_weight=1.0,
                    live_test_mode=live_test_mode,
                    rank_size_power=float(policy_size.get("size_power", 1.1)),
                )
                requested_position_usdt = float(sizing_audit["size_after_liquidity"])
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
                    "base_position_size": chain_results.get("base_position_size", ""),
                    "sizing_source": chain_results.get("sizing_source", ""),
                    "size_power": chain_results.get("size_power", ""),
                    "asset_weight_multiplier": chain_results.get(
                        "asset_weight_multiplier", ""
                    ),
                    "asset_decision": chain_results.get("asset_decision", ""),
                    "meta_pred": chain_results.get("meta_pred", ""),
                    "action": chain_results.get("action", ""),
                    "base_pred": chain_results.get("base_pred", ""),
                    "base_rank_pct": chain_results.get("base_rank_pct", ""),
                    "base_train_rank_pct": chain_results.get("base_train_rank_pct", ""),
                    "base_gate_top_frac": chain_results.get("base_gate_top_frac", ""),
                    "meta_train_rank_pct": chain_results.get("meta_train_rank_pct", ""),
                    "rank_score_source": chain_results.get("rank_score_source", ""),
                    "sizer_rank_percentile": chain_results.get(
                        "sizer_rank_percentile", ""
                    ),
                    "effective_threshold": chain_results.get("effective_threshold", ""),
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
                execution_kwargs: Dict[str, Any] = {}
                execution_limit_price = None
                if (
                    getattr(executor, "exchange", None) is not None
                    and price is not None
                    and (
                        portfolio_policy.ticker_precheck_enabled
                        or portfolio_policy.orderbook_precheck_enabled
                    )
                ):
                    try:
                        ticker_snapshot = fetch_ticker_snapshot(
                            exchange=executor.exchange,
                            symbol=symbol,
                            side=side,
                            policy=portfolio_policy,
                            mode=str(getattr(executor, "mode", "")),
                            now=now_utc,
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
                        gap_penalty, gap_info = compute_price_gap_rank_penalty(
                            strategy_id=strategy_id,
                            side=side,
                            signal_price=float(price),
                            decision_mid=decision_mid,
                            policy=portfolio_policy,
                        )
                        adjusted_rank = max(rank_for_size - float(gap_penalty), 0.0)
                        execution_snapshot.update(gap_info)
                        execution_snapshot["price_gap_penalty"] = float(gap_penalty)
                        execution_snapshot["adjusted_rank_score"] = float(adjusted_rank)
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
                        if portfolio_policy.orderbook_precheck_enabled:
                            book_snapshot = evaluate_orderbook_liquidity(
                                exchange=executor.exchange,
                                symbol=symbol,
                                side=side,
                                intended_quote_size=float(size),
                                ticker_snapshot=ticker_snapshot,
                                policy=portfolio_policy,
                                mode=str(getattr(executor, "mode", "")),
                            )
                            execution_snapshot.update(book_snapshot.to_dict())
                            if book_snapshot.hard_reject:
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
                            if portfolio_mgr is not None:
                                capacity = portfolio_mgr.get_portfolio_capacity(
                                    side=side,
                                    strategy_id=strategy_id,
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
                                )
                                size = float(sizing_audit["size_after_liquidity"])
                                chain_results["portfolio_rank_sizing"] = sizing_audit
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
                                chain_results["portfolio_gate_after_liquidity"] = info
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
                    sizing_for_log = (
                        chain_results.get("portfolio_rank_sizing", {}) or {}
                    )
                    execution_snapshot["signal_price"] = (
                        float(price) if price is not None else None
                    )
                    execution_snapshot["final_threshold"] = float(
                        decision["effective_threshold"]
                    )
                    execution_snapshot["position_size_before_liquidity"] = (
                        sizing_for_log.get("size_before_liquidity")
                    )
                    execution_snapshot["position_size_after_liquidity"] = size
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
                                "decision_mid",
                                "spread_bps",
                                "ticker_bid",
                                "ticker_ask",
                                "ticker_mid",
                                "ticker_spread_bps",
                                "expected_fill_price",
                                "liquidity_capacity_weight",
                                "expected_fill_slippage_bps",
                                "expected_total_entry_friction_bps",
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
                                "max_chase_bps",
                                "entry_limit_price",
                            }
                        }
                    )
                trade_result = _execute_trade_with_optional_context(
                    executor,
                    symbol=symbol,
                    side=side,
                    size=abs(size),
                    price=execution_price,
                    bucket_key=bucket_key,
                    ohlcv_reference_price=float(price) if price is not None else None,
                    trade_context={
                        "base_pred": chain_results.get("base_pred"),
                        "base_rank_pct": chain_results.get("base_rank_pct"),
                        "base_train_rank_pct": chain_results.get("base_train_rank_pct"),
                        "base_gate_top_frac": chain_results.get("base_gate_top_frac"),
                        "meta_pred": chain_results.get("meta_pred"),
                        "meta_train_rank_pct": chain_results.get("meta_train_rank_pct"),
                        "rank_score_source": chain_results.get("rank_score_source"),
                        "calibrated_score": decision.get("calibrated_score"),
                        "rank_percentile": chain_results.get("sizer_rank_percentile")
                        or decision.get("rank_percentile"),
                        "effective_threshold": chain_results.get("effective_threshold"),
                        "deployment_rank_threshold": decision.get(
                            "deployment_rank_threshold"
                        ),
                        "barrier_pct": live_barrier_pct,
                        "barrier_frac": live_barrier_pct,
                    },
                    execution_kwargs={
                        "execution_snapshot": execution_snapshot,
                        "signal_price": float(price) if price is not None else None,
                        "decision_mid": execution_snapshot.get("decision_mid"),
                        "expected_entry_price": execution_snapshot.get(
                            "expected_fill_price"
                        ),
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
                order_error_category = str(trade_result.get("error_category", "") or "")
                if not trade_success:
                    side_metrics["order_errors"] += 1
                    if not order_error_category:
                        order_error_category = "unclassified_order_error"
                        side_metrics["unexplained_order_errors"] += 1
                    tprint(
                        "[ORDER_ERROR] "
                        f"{symbol} {side}/{strategy_id} "
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
                        spread_proxy_pct=trade_result.get("spread_proxy_pct"),
                        orderbook_snapshot=trade_result.get("orderbook_snapshot"),
                        stop_price=trade_result.get("stop_price"),
                        stop_order_id=trade_result.get("stop_order_id"),
                        exchange_order_id=_order_identifier(trade_result.get("order")),
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
                if prediction_ledger is not None and _should_log_prediction_candidate(
                    decision, policy=portfolio_policy
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
        trade_logger.log_trade_legacy(
            symbol=str(closed_trade.get("symbol") or ""),
            side=side,
            action="exit",
            size=float(_safe_float(closed_trade.get("filled"), 0.0) or 0.0),
            price=_safe_float(closed_trade.get("entry_price"), np.nan),
            mode=str(config.get("mode", "live")),
            status="closed",
            context={
                "run_id": config.get("run_id"),
                "lifecycle_event": "exit_filled",
                "strategy_id": closed_trade.get("strategy_id"),
                "entry_price": closed_trade.get("entry_price"),
                "actual_entry_price": closed_trade.get("entry_price"),
                "actual_exit_price": closed_trade.get("exit_price"),
                "realized_exit_price": closed_trade.get("exit_price"),
                "exit_reason": closed_trade.get("reason"),
                "exit_reason_detail": closed_trade.get("exit_reason_detail"),
                "stop_price": closed_trade.get("stop_price"),
                "stop_order_id": closed_trade.get("stop_order_id"),
                "exchange_order_id": closed_trade.get("close_order_id"),
                "gross_pnl_pct": closed_trade.get("gross_pnl_pct"),
                "net_pnl_pct": closed_trade.get("net_pnl_pct"),
                "gross_pnl_amount": closed_trade.get("gross_pnl_amount"),
                "net_pnl_amount": closed_trade.get("net_pnl_amount"),
                "net_pnl": closed_trade.get("net_pnl"),
                "fees_amount": closed_trade.get("fees_amount"),
                "mfe": closed_trade.get("mfe"),
                "mae": closed_trade.get("mae"),
                "trade_recap": closed_trade.get("trade_recap"),
                "meta_pred": closed_trade.get("meta_pred"),
                "base_pred": closed_trade.get("base_pred"),
                "calibrated_score": closed_trade.get("calibrated_score"),
                "rank_percentile": closed_trade.get("rank_percentile"),
                "effective_threshold": closed_trade.get("effective_threshold"),
                "deployment_rank_threshold": closed_trade.get(
                    "deployment_rank_threshold"
                ),
            },
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
) -> Dict[str, Dict[str, Any]]:
    """Monitor active positions and apply closed-15m trailing/stop updates."""
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
    monitor_delay = float(cfg.get("fifteen_minute_ohlcv_delay_seconds", 5.0))
    latest_closed_15m = _latest_closed_candle_start(
        now_ts,
        timeframe_minutes=15,
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
            end_time = latest_closed_15m
            if start_time >= end_time:
                continue

            ohlcv_5m: Any = None
            if exchange is not None:
                ohlcv_5m = hf_data_loader.fetch_specific_period(
                    exchange,
                    symbol,
                    "15m",
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
            bars = bars[bars.index <= latest_closed_15m]
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
                f"  Error evaluating 15m price action for {symbol}: "
                f"{classify_api_error(exc)}: {exc}"
            )
            statuses.setdefault(symbol, {})["price_action_error"] = str(exc)
    order_status_errors = sum(
        1
        for status in statuses.values()
        if isinstance(status, dict)
        and (
            status.get("fetch_order_error")
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
        default=900,
        help="Position monitor interval in seconds (default: 900 = 15 min)",
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
    parser.add_argument(
        "--max-position-pct",
        type=float,
        default=0.15,
        help="Maximum fraction of portfolio equity per position (default: 0.15)",
    )
    parser.add_argument(
        "--data-root",
        default=None,
        help="Data root containing artifacts and live market cache.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Artifact run id to load. Defaults to latest when omitted.",
    )
    parser.add_argument(
        "--run-once",
        action="store_true",
        help="Run one inference batch and exit after optional daily reporting.",
    )
    args = parser.parse_args()

    # Initialize components
    config = load_inference_config(data_root=args.data_root, run_id=args.run_id)
    config["execution_account"] = args.execution_account
    config["margin_mode"] = args.margin_mode
    config["live_quote_currency"] = str(args.live_quote_currency or "USDC").upper()
    config["max_position_pct"] = float(args.max_position_pct)
    if args.live_test:
        config["mode"] = "live-test"
    else:
        config["mode"] = "live" if args.live else "shadow"
    config.setdefault(
        "cross_margin_dust_quote_threshold",
        2.5 if _is_live_test_mode(config["mode"]) else 5.0,
    )
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
    validate_meta_feature_contract_artifact(
        config["data_root"],
        config["run_id"],
        effective_model_bundle,
        accepted_strategies,
        strict=True,
    )
    required_feature_keys = get_inference_required_feature_keys(
        effective_model_bundle,
        accepted_strategies,
    )
    from extreme_price_movements.simple_position_sizer import load_calibration_curves

    calibration_data = load_calibration_curves(config["data_root"], config["run_id"])
    normalized_thresholds = _load_normalized_threshold_map(
        config["data_root"], config["run_id"]
    )
    policy_selection_rules = _load_policy_selection_rules(
        config["data_root"], config["run_id"]
    )
    lgbm_strategy_mask_rows = _load_lgbm_strategy_mask_rows(
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
        mode=config["mode"],
        exchange=exchange,
        bucket_params=runtime_bucket_params,
        config=config,
    )
    logger = TradeLogger()
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
    portfolio_policy = load_portfolio_policy_config(
        data_root=config["data_root"],
        run_id=config["run_id"],
        runtime_cfg=config,
    )
    prediction_ledger = PredictionLedger(
        Path(config["data_root"]) / "live_state" / "prediction_ledger.parquet"
    )
    market_kill_switch = MarketKillSwitch(
        Path(config["data_root"]) / "live_state" / "market_kill_switch.json"
    )
    strategy_kill_switch = StrategyKillSwitch(
        Path(config["data_root"]) / "live_state" / "strategy_kill_switches.json",
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
        f"max_position_pct={portfolio_policy.max_position_wallet_pct:.2f} "
        f"max_position_quote={portfolio_policy.max_position_quote_notional:.2f}"
    )
    portfolio_mgr = PortfolioManager.from_policy_config(
        portfolio_policy,
        cooldown_hours=0.0,
        max_same_side=max_concurrent_per_side,
        max_same_strategy=max_concurrent_per_strategy,
    )
    _sync_reconciled_positions_to_portfolio_manager(executor, portfolio_mgr)
    if unimported_external_positions:
        _apply_reconciliation_entry_gate(
            reconciliation_report=reconciliation_report,
            portfolio_mgr=portfolio_mgr,
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
            ),
            daemon=True,
        )
        challenger_thread.start()

    # Main inference loop - run after closed 15m candles. Model entry decisions
    # only run after a closed hourly candle is available.
    last_hourly_sync = None
    last_margin_reconciliation = pd.Timestamp.now(tz="UTC")
    margin_reconciliation_interval = pd.Timedelta(
        minutes=float(config.get("margin_reconciliation_interval_minutes", 60.0))
    )
    last_universe_refresh_day = pd.Timestamp.utcnow().floor("D")
    while True:
        try:
            loop_now = pd.Timestamp.now(tz="UTC")
            fifteen_delay = float(config.get("fifteen_minute_ohlcv_delay_seconds", 5.0))
            hourly_delay = float(config.get("hourly_ohlcv_delay_seconds", 5.0))
            current_time = _latest_closed_candle_start(
                loop_now,
                timeframe_minutes=15,
                delay_seconds=fifteen_delay,
            )
            latest_closed_hour = _latest_closed_candle_start(
                loop_now,
                timeframe_minutes=60,
                delay_seconds=hourly_delay,
            )
            current_close = current_time + pd.Timedelta(minutes=15)
            latest_closed_hour_close = latest_closed_hour + pd.Timedelta(hours=1)
            fifteen_age_seconds = _closed_candle_age_seconds(
                loop_now,
                current_time,
                timeframe_minutes=15,
            )
            hourly_age_seconds = _closed_candle_age_seconds(
                loop_now,
                latest_closed_hour,
                timeframe_minutes=60,
            )
            max_entry_15m_age_seconds = float(
                config.get("entry_15m_max_staleness_seconds", 15.0 * 60.0)
            )
            max_entry_hourly_age_seconds = float(
                config.get("entry_hourly_max_staleness_seconds", 15.0 * 60.0)
            )
            fifteen_entry_fresh = fifteen_age_seconds <= max(
                max_entry_15m_age_seconds,
                fifteen_delay,
            )
            hourly_entry_fresh = hourly_age_seconds <= max(
                max_entry_hourly_age_seconds,
                hourly_delay,
            )
            entry_context_fresh = bool(fifteen_entry_fresh and hourly_entry_fresh)
            tprint(
                f"\n=== Running inference after closed 15m candle "
                f"start={current_time} close={current_close} "
                f"age={fifteen_age_seconds:.0f}s "
                f"fresh_15m_for_entries={fifteen_entry_fresh} "
                f"(latest_closed_hour_start={latest_closed_hour} "
                f"close={latest_closed_hour_close} "
                f"age={hourly_age_seconds:.0f}s "
                f"fresh_1h_for_entries={hourly_entry_fresh} "
                f"fresh_for_entries={entry_context_fresh}) ==="
            )
            did_hourly_refresh = False
            loop_timer = _StageTimer("live_entry_loop")

            if loop_now >= last_margin_reconciliation + margin_reconciliation_interval:
                try:
                    reconciliation_report = executor.reconcile_cross_margin_account()
                    _write_margin_reconciliation_report(config, reconciliation_report)
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
                    live_quote_currency=config["live_quote_currency"],
                )
                download_symbols[:] = universe_state["download_symbols"]
                symbols[:] = universe_state["tradable_symbols"]
                last_universe_refresh_day = current_day
                tprint(
                    "Daily Binance universe refresh complete: "
                    f"download={len(download_symbols)} tradable={len(symbols)}"
                )

            # Fetch full universe only for closed hourly candles.
            if (
                (last_hourly_sync is None) or (latest_closed_hour > last_hourly_sync)
            ) and entry_context_fresh:
                refresh_microdata = bool(config.get("hourly_refresh_microdata", True))
                data_fetcher.fetch_hourly_universe_once(
                    download_symbols,
                    max_workers=int(config.get("hourly_ohlcv_workers", 16)),
                    no_progress_timeout_seconds=float(
                        config.get("hourly_ohlcv_no_progress_timeout_seconds", 60.0)
                    ),
                    check_recent_gaps_days=7,
                    refresh_microdata=refresh_microdata,
                    target_hour=latest_closed_hour,
                )
                loop_timer.mark("hourly_fetch")
                tprint(
                    "Hourly data refresh complete: "
                    f"ohlcv_symbols={len(download_symbols)} "
                    f"target_hour={latest_closed_hour} "
                    f"microdata_refresh_enabled={refresh_microdata}"
                )
                last_hourly_sync = latest_closed_hour
                did_hourly_refresh = True
            elif (last_hourly_sync is None) or (latest_closed_hour > last_hourly_sync):
                tprint(
                    "Skipping hourly entry refresh: latest closed candle context is "
                    f"stale for new entries (latest_15m={current_time}, "
                    f"latest_15m_closed_at={current_close}, "
                    f"latest_15m_age={fifteen_age_seconds:.0f}s, "
                    f"max_15m_age={max_entry_15m_age_seconds:.0f}s, "
                    f"target_hour={latest_closed_hour}, "
                    f"closed_at={latest_closed_hour_close}, "
                    f"hour_age={hourly_age_seconds:.0f}s, "
                    f"max_hour_age={max_entry_hourly_age_seconds:.0f}s). "
                    "Monitoring existing positions only until a fresh hourly close."
                )

            if not did_hourly_refresh:
                _monitor_active_position_price_action(
                    executor,
                    exchange=executor.exchange,
                    now=current_time,
                    config=config,
                    portfolio_mgr=portfolio_mgr,
                    trade_logger=logger,
                )
                _maybe_send_daily_deployment_report(
                    daily_reporter=daily_reporter,
                    exchange=exchange,
                    portfolio_mgr=portfolio_mgr,
                    trade_logger=logger,
                    config=config,
                )
                if args.run_once:
                    executor.shutdown()
                    break
                _sleep_until_next_candle_close(
                    timeframe_minutes=15,
                    delay_seconds=fifteen_delay,
                )
                continue

            panel = data_fetcher.get_panel(
                download_symbols, lookback_hours=panel_lookback_hours
            )
            loop_timer.mark("panel_load")
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
            if not market_decision.allow_new_entries:
                tprint(
                    "Market kill switch active: blocking new entries, "
                    f"reason={market_decision.reason}, "
                    f"details={market_decision.details}"
                )
                _monitor_active_position_price_action(
                    executor,
                    exchange=executor.exchange,
                    now=current_time,
                    config=config,
                    portfolio_mgr=portfolio_mgr,
                    trade_logger=logger,
                )
                _maybe_send_daily_deployment_report(
                    daily_reporter=daily_reporter,
                    exchange=exchange,
                    portfolio_mgr=portfolio_mgr,
                    trade_logger=logger,
                    config=config,
                )
                if args.run_once:
                    executor.shutdown()
                    break
                _sleep_until_next_candle_close(
                    timeframe_minutes=15,
                    delay_seconds=fifteen_delay,
                )
                continue
            (
                thresholds,
                long_cands,
                short_cands,
                features,
                strategy_candidate_masks,
            ) = _select_candidates_and_load_features(
                panel=tradable_panel,
                symbols=symbols,
                run_id=config["run_id"],
                data_root=config["data_root"],
                cfg=get_runtime_cfg(),
                lookback_hours=args.lookback_hours,
                required_feature_keys=required_feature_keys,
                lgbm_strategy_mask_rows=lgbm_strategy_mask_rows,
            )
            loop_timer.mark("candidate_and_feature_load")

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
                strategy_kill_switch=strategy_kill_switch,
            )
            loop_timer.mark("model_scoring_and_orders")
            tprint(
                f"Inference batch complete: download_symbols={len(download_symbols)} "
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

            if args.run_once:
                executor.shutdown()
                break

            _sleep_until_next_candle_close(
                timeframe_minutes=15,
                delay_seconds=fifteen_delay,
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
):
    """
    calibration_data = calibration_data or {}
    Run position monitoring every 15 minutes by default.

    New entries are intentionally not evaluated here. The loop only monitors
    existing positions and applies the stop policy on closed 15m bars.

    Args:
        symbols: List of trading symbols
        data_fetcher: DataFetcher instance
        orchestrator: ModelOrchestrator instance
        executor: TradeExecutor instance
        logger: TradeLogger instance
        config: Configuration dictionary
        interval: Check interval in seconds (default 900 = 15 min)
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
        entry_price = float(position_state.get("entry_price", 0.0) or 0.0)
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
            position_state.setdefault("trade_recap_events", []).append(
                {
                    "ts": pd.Timestamp(bar_ts).isoformat(),
                    "event": "price_bar_15m",
                    "open": bar_open,
                    "high": bar_high,
                    "low": bar_low,
                    "close": bar_close,
                    "price_dev_pct": float(price_dev_pct),
                    "stop_price": float(stop_price),
                    "stop_reason": stop_reason,
                }
            )
            if len(position_state.get("trade_recap_events", [])) > 500:
                del position_state["trade_recap_events"][:-500]

            if side == "long":
                if np.isfinite(stop_price) and bar_low <= stop_price:
                    exit_reason = f"stop_loss_filled:{stop_reason}"
                    has_exchange_stop = bool(position_state.get("stop_order_id"))
                    if live_mode and has_exchange_stop:
                        position_state.setdefault("trade_recap_events", []).append(
                            {
                                "ts": pd.Timestamp(bar_ts).isoformat(),
                                "event": "stop_touch_waiting_exchange_fill",
                                "bar_low": bar_low,
                                "stop_price": float(stop_price),
                                "stop_reason": stop_reason,
                            }
                        )
                        return {
                            "status": "stop_touch_waiting_exchange_fill",
                            "reason": exit_reason,
                        }
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
                                "event": "stop_touch_waiting_exchange_fill",
                                "bar_high": bar_high,
                                "stop_price": float(stop_price),
                                "stop_reason": stop_reason,
                            }
                        )
                        return {
                            "status": "stop_touch_waiting_exchange_fill",
                            "reason": exit_reason,
                        }
                    return executor.close_position(
                        symbol, price=float(stop_price), reason=exit_reason
                    )

        require_metadata = True
        decision = None
        try:
            decision = compute_simple_policy_stop_decision(
                state={
                    **position_state,
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
                position_state.setdefault("trade_recap_events", []).append(
                    {
                        "ts": pd.Timestamp(last_bar_ts).isoformat(),
                        "event": "adverse_excursion_exit_triggered",
                        "reason": decision.reason,
                        "detail": decision.reason_detail,
                        "mfe": float(decision.mfe if decision.mfe is not None else mfe),
                        "mae": float(decision.mae if decision.mae is not None else mae),
                    }
                )
                exit_price = float(bars.iloc[-1]["close"])
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
        executor.update_position_policy_state(
            symbol,
            policy_stop_decision=decision,
            peak_price=decision_peak,
            mfe=decision_mfe,
            mae=decision_mae,
            bars_in_trade=int(position_state.get("bars_in_trade", 0) or 0)
            + int(len(bars)),
            last_5m_eval_ts=last_bar_ts,
        )
    except Exception as e:
        tprint(f"  [STOP_LOSS] Error evaluating stop policy for {symbol}: {e}")


if __name__ == "__main__":
    main()
