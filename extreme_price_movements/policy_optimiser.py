"""Policy optimisation stage for extreme_price_movements.

This module consumes the already-selected best strategy (from simple_position_sizer),
keeps TP/SL fixed, runs offset generation first, then sequentially optimises richer
exit-policy parameters. The resulting params are persisted for holdout OOS replay.
"""

from __future__ import annotations

import itertools
import json
import logging
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from extreme_price_movements.metrics import _stable_equity_and_drawdown
from extreme_price_movements.inference.parity import strategy_core_id
from extreme_price_movements.run_ridge_sizer import (
    load_base_oof_predictions,
    load_meta_oof_predictions,
    load_trade_outcomes,
)
from extreme_price_movements.simple_offset_generator import (
    build_policy_path_state_bundle,
    run_simple_offset_generator_from_sizer,
)
from extreme_price_movements.utils import tprint

EPS = 1e-9
MIN_DEPLOYMENT_AVG_NET_PNL_PER_TRADE = 0.002
MAX_DEPLOYMENT_STRATEGIES_PER_SIDE = 2
SIMPLE_POSITION_SIZER_ARTIFACT_DIR = "simple_position_sizer"
POLICY_OPTIMISER_CANDIDATE_FRAC = 0.15
POLICY_OPTIMISER_ASSESSMENT_FRAC = 0.05
POLICY_OPTIMISER_DIAGNOSTIC_FRAC = 0.01
POLICY_OPTIMISER_OFFSET_LIMITER_ENABLED = False
logger = logging.getLogger(__name__)


def _strategy_description(row: Dict[str, Any]) -> str:
    side = str(row.get("side", "") or "").strip().lower()
    source_target = str(row.get("source_target", "") or "").strip()
    source_horizon = row.get("source_horizon", np.nan)
    threshold_pct = float(row.get("threshold_pct", np.nan))
    selection_frac = float(row.get("selection_frac", np.nan))
    parts: List[str] = []
    if side:
        parts.append(side)
    if source_target:
        parts.append(source_target)
    if np.isfinite(source_horizon):
        parts.append(f"h{float(source_horizon):g}")
    if np.isfinite(threshold_pct):
        parts.append(f"thr={threshold_pct:.1f}%")
    if np.isfinite(selection_frac):
        parts.append(f"sel={selection_frac:.3f}")
    return " | ".join(parts) if parts else str(row.get("strategy_id", ""))


def _safe_log(x: float) -> float:
    return float(math.log(max(float(x), EPS)))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(out):
        return float(default)
    return out


def _strategy_side(row: Dict[str, Any]) -> str:
    side = str(row.get("side", "") or "").strip().lower()
    if side in {"long", "short"}:
        return side
    strategy_id = str(row.get("strategy_id", "") or "").lower()
    if strategy_id.startswith("long_"):
        return "long"
    if strategy_id.startswith("short_"):
        return "short"
    target = str(row.get("source_target", "") or "").lower()
    if target.startswith("long_") or target == "long":
        return "long"
    if target.startswith("short_") or target == "short":
        return "short"
    meta_key = str(
        row.get("meta_model_key")
        or row.get("alpha_model_key")
        or row.get("model_key")
        or ""
    ).lower()
    if meta_key.startswith("long_"):
        return "long"
    if meta_key.startswith("short_"):
        return "short"
    return "unknown"


def _load_lgbm_mask_contracts_for_deployment() -> Dict[str, Dict[str, Any]]:
    """Load LGBM rule-mask contracts so deployment is independent of registry discovery."""
    try:
        from extreme_price_movements.offline_optimisers.params_store import (
            load_inference_candidate_mask_params_per_bucket,
        )
    except Exception as exc:
        tprint(f"[deployment] could not import LGBM mask contract loader: {exc}")
        return {}

    try:
        rows = load_inference_candidate_mask_params_per_bucket(
            top_n=99,
            ranking_metric="score_for_best_params",
        )
    except Exception as exc:
        tprint(f"[deployment] could not load LGBM mask contracts: {exc}")
        return {}

    contracts: Dict[str, Dict[str, Any]] = {}
    keep_keys = {
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
        "ranking_metric",
        "ranking_score",
        "ranking_score_norm",
        "adjusted_ranking_score",
    }
    for row in rows:
        if not isinstance(row, dict):
            continue
        sid = str(row.get("strategy_id", "") or "")
        if not sid:
            continue
        contract = {k: row.get(k) for k in keep_keys if k in row}
        mask_params = dict(contract.get("mask_params", {}) or {})
        canonical_key = str(
            contract.get("base_event_trigger")
            or contract.get("canonical_key")
            or mask_params.get("canonical_key")
            or ""
        )
        if canonical_key:
            contract["base_event_trigger"] = canonical_key
            contract["canonical_key"] = canonical_key
            mask_params.setdefault("canonical_key", canonical_key)
        contract["mask_params"] = mask_params

        aliases = {sid, strategy_core_id(sid)}
        side = str(row.get("trade_side", row.get("side", "")) or "").lower()
        core = strategy_core_id(sid)
        if side in {"long", "short"} and core:
            aliases.add(f"{side}_{core}")
        for alias in aliases:
            if alias:
                contracts[str(alias)] = dict(contract)
    tprint(
        "[deployment] loaded LGBM mask contracts for strategy_for_inference: "
        f"rows={len(rows)} aliases={len(contracts)}"
    )
    return contracts


def _compute_strategy_D(row: Dict[str, Any]) -> float:
    pf = float(row.get("profit_factor", float("nan")))
    stability = float(row.get("stability", float("nan")))
    monthly_sortino = float(row.get("monthly_sortino", float("nan")))
    hit_rate = float(row.get("hit_rate", float("nan")))
    trades_per_day = float(row.get("trades_per_day", float("nan")))
    max_drawdown = float(row.get("max_drawdown", float("nan")))
    wallet_pnl = float(row.get("wallet_pnl", float("nan")))
    if not all(
        np.isfinite(v)
        for v in (
            pf,
            stability,
            monthly_sortino,
            hit_rate,
            trades_per_day,
            max_drawdown,
            wallet_pnl,
        )
    ):
        return float("nan")
    if wallet_pnl <= 0.0:
        return float("nan")
    return (
        0.35 * _safe_log(pf)
        + 0.20 * stability
        + 0.15 * monthly_sortino
        + 0.10 * hit_rate
        + 0.10 * _safe_log(1.0 + trades_per_day)
        + 0.10 * (1.0 - max_drawdown / wallet_pnl)
    )


def _load_simple_sizer_strategy_rows(
    data_root: str, run_id: str
) -> Dict[str, Dict[str, Any]]:
    path = (
        Path(data_root)
        / "artifacts"
        / run_id
        / SIMPLE_POSITION_SIZER_ARTIFACT_DIR
        / "strategy_params.json"
    )
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return {}
    rows = payload.get("strategies", []) if isinstance(payload, dict) else []
    if not isinstance(rows, list):
        return {}
    by_strategy_id: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        sid = str(row.get("strategy_id", "") or "")
        if not sid:
            continue
        by_strategy_id[sid] = dict(row)
    return by_strategy_id


def strategy_final_acceptation(
    data_root: str,
    run_id: str,
    strategy_rows: List[Dict[str, Any]],
    *,
    output_filename: str = "strategy_final_acceptation.json",
) -> Dict[str, Any]:
    """Filter policy candidates by hard gates and write the acceptance artifact."""
    simple_sizer_rows = _load_simple_sizer_strategy_rows(data_root, run_id)
    accepted: List[Dict[str, Any]] = []
    for row in strategy_rows:
        if not isinstance(row, dict):
            continue
        strategy_id = str(row.get("strategy_id", "") or "")
        source_row = simple_sizer_rows.get(strategy_id, row)
        accepted_row = dict(source_row)
        accepted_row["strategy_id"] = strategy_id
        accepted_row["strategy_description"] = _strategy_description(source_row)
        accepted_row["D"] = _compute_strategy_D(source_row)
        pf = float(source_row.get("profit_factor", float("nan")))
        stability = float(source_row.get("stability", float("nan")))
        monthly_sortino = float(source_row.get("monthly_sortino", float("nan")))
        max_drawdown = float(source_row.get("max_drawdown", float("nan")))
        wallet_pnl = float(source_row.get("wallet_pnl", float("nan")))
        if not all(
            np.isfinite(v)
            for v in (pf, stability, monthly_sortino, max_drawdown, wallet_pnl)
        ):
            continue
        if (
            pf > 1.3
            and stability > 0.7
            and monthly_sortino > 1.0
            and wallet_pnl > 0.0
            and max_drawdown < wallet_pnl
        ):
            accepted.append(accepted_row)

    accepted.sort(
        key=lambda r: (
            float(r.get("D", float("-inf"))),
            float(r.get("wallet_pnl", float("-inf"))),
            float(r.get("profit_factor", float("-inf"))),
        ),
        reverse=True,
    )

    payload = {
        "schema_version": "v1",
        "generated_by": "policy_optimiser",
        "run_id": run_id,
        "hard_gates": {
            "profit_factor_min": 1.3,
            "stability_min": 0.7,
            "monthly_sortino_min": 1.0,
            "max_drawdown_lt_wallet_pnl": True,
        },
        "strategies": [
            {
                "strategy_id": r.get("strategy_id", ""),
                "strategy_description": r.get("strategy_description", ""),
                "metrics": r,
                "n_trades": int(
                    r.get(
                        "trades_selected",
                        r.get("n_trades", r.get("trades_total", 0)),
                    )
                    or 0
                ),
            }
            for r in accepted
        ],
    }
    out_dir = Path(data_root) / "artifacts" / run_id / "policy_params"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / output_filename
    path.write_text(json.dumps(payload, indent=2, default=float))
    (Path(data_root) / "artifacts" / run_id / output_filename).write_text(
        json.dumps(payload, indent=2, default=float)
    )
    tprint(f"Wrote strategy acceptance gate output -> {path}")
    return payload


def _avg_net_pnl_per_trade(row: Dict[str, Any]) -> float:
    metrics = row.get("metrics_final")
    if not isinstance(metrics, dict):
        metrics = row.get("metrics_final_val")
    if not isinstance(metrics, dict):
        metrics = row
    net_pnl = _safe_float(
        metrics.get("net_pnl", row.get("final_net_pnl_full", row.get("final_net_pnl")))
    )
    n_trades = int(
        max(
            0,
            _safe_float(
                metrics.get(
                    "n_trades",
                    row.get("n_trades", row.get("trades_selected", 0)),
                )
            ),
        )
    )
    return net_pnl / max(1, n_trades)


def _deployment_rank(row: Dict[str, Any]) -> float:
    metrics = row.get("metrics_final")
    if not isinstance(metrics, dict):
        metrics = row.get("metrics_final_val")
    if not isinstance(metrics, dict):
        metrics = row
    avg_net_pnl = _avg_net_pnl_per_trade(row)
    avg_holding_time = max(1.0, _safe_float(row.get("avg_holding_time_hours"), 1.0))
    opportunities_per_day = max(0.0, _safe_float(row.get("opportunities_per_day"), 0.0))
    effective_ops_day = math.sqrt(
        max(0.0, min(36.0 / avg_holding_time, opportunities_per_day))
    )
    weekly_wallet_vol = max(0.0, _safe_float(metrics.get("weekly_wallet_vol"), 0.0))
    monthly_wallet_vol = max(0.0, _safe_float(metrics.get("monthly_wallet_vol"), 0.0))
    denominator = math.sqrt(avg_holding_time) * math.sqrt(
        weekly_wallet_vol + monthly_wallet_vol + EPS
    )
    return float(effective_ops_day * avg_net_pnl / max(denominator, EPS))


def _deployment_rank_components(row: Dict[str, Any]) -> Dict[str, float]:
    metrics = row.get("metrics_final")
    if not isinstance(metrics, dict):
        metrics = row.get("metrics_final_val")
    if not isinstance(metrics, dict):
        metrics = row
    avg_holding_time = max(1.0, _safe_float(row.get("avg_holding_time_hours"), 1.0))
    opportunities_per_day = max(0.0, _safe_float(row.get("opportunities_per_day"), 0.0))
    weekly_wallet_vol = max(0.0, _safe_float(metrics.get("weekly_wallet_vol"), 0.0))
    monthly_wallet_vol = max(0.0, _safe_float(metrics.get("monthly_wallet_vol"), 0.0))
    effective_ops_day = math.sqrt(
        max(0.0, min(36.0 / avg_holding_time, opportunities_per_day))
    )
    return {
        "avg_holding_time_hours": avg_holding_time,
        "opportunities_per_day": opportunities_per_day,
        "effective_ops_day": effective_ops_day,
        "weekly_wallet_vol": weekly_wallet_vol,
        "monthly_wallet_vol": monthly_wallet_vol,
    }


def _deployment_order_rank_threshold(row: Dict[str, Any]) -> float:
    """Return per-strategy prediction-rank threshold for live order placement."""
    avg_top1_trades_per_day = _safe_float(
        row.get(
            "avg_trades_per_day_at_top_1pct",
            row.get("top1_avg_trades_per_day", row.get("opportunities_per_day", 0.0)),
        ),
        0.0,
    )
    avg_holding_hours = max(1.0, _safe_float(row.get("avg_holding_time_hours"), 1.0))
    threshold = (avg_top1_trades_per_day / 24.0) * 2.0 / avg_holding_hours * 0.95
    return float(np.clip(max(0.95, threshold), 0.0, 1.0))


def _opportunities_per_day(timestamps: pd.Series) -> float:
    ts = pd.to_datetime(timestamps, utc=True, errors="coerce").dropna()
    if ts.empty:
        return 0.0
    elapsed_days = max(
        1.0,
        float((ts.max() - ts.min()).total_seconds()) / (24.0 * 3600.0),
    )
    return float(len(ts) / elapsed_days)


def _active_trades_per_day(timestamps: pd.Series) -> Tuple[float, int]:
    ts = pd.to_datetime(timestamps, utc=True, errors="coerce").dropna()
    if ts.empty:
        return 0.0, 0
    active_days = int(ts.dt.floor("D").nunique())
    if active_days <= 0:
        return 0.0, 0
    return float(len(ts) / active_days), active_days


def _average_holding_hours_from_rows(rows: pd.DataFrame) -> float:
    if rows.empty:
        return 0.0
    for col in ("exit_bar", "bars_since_entry", "holding_bars"):
        if col in rows.columns:
            vals = pd.to_numeric(rows[col], errors="coerce").to_numpy(dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            if vals.size:
                return float(np.nanmean(np.maximum(1.0, vals)))
    if {"timestamp", "exit_timestamp"}.issubset(rows.columns):
        entry = pd.to_datetime(rows["timestamp"], utc=True, errors="coerce")
        exit_ts = pd.to_datetime(rows["exit_timestamp"], utc=True, errors="coerce")
        hours = (exit_ts - entry).dt.total_seconds().to_numpy(dtype=np.float64) / 3600.0
        hours = hours[np.isfinite(hours)]
        if hours.size:
            return float(np.nanmean(np.maximum(1.0, hours)))
    return 0.0


def _top_slice_opportunity_diagnostics(
    *,
    outcomes: pd.DataFrame,
    selected_idx: np.ndarray,
    frac: float,
) -> Dict[str, Any]:
    selected_idx = np.asarray(selected_idx, dtype=np.int64)
    selected_idx = selected_idx[(selected_idx >= 0) & (selected_idx < len(outcomes))]
    rows = outcomes.iloc[selected_idx] if selected_idx.size else outcomes.iloc[:0]
    timestamps = rows.get("timestamp", pd.Series(dtype="datetime64[ns]"))
    active_trades_per_day, active_days = _active_trades_per_day(timestamps)
    return {
        "top_frac": float(frac),
        "n_trades": int(len(rows)),
        "active_days": int(active_days),
        "avg_trades_per_day": _opportunities_per_day(timestamps),
        "avg_trades_per_active_day": active_trades_per_day,
        "avg_trades_per_calendar_day": _opportunities_per_day(timestamps),
        "avg_holding_hours": _average_holding_hours_from_rows(rows),
    }


def _asset_profit_factor_exclusions(
    *,
    symbols: np.ndarray,
    net_rets: np.ndarray,
    avg_profit_factor: float,
) -> Dict[str, Any]:
    if len(symbols) != len(net_rets) or len(net_rets) == 0:
        return {"excluded_symbols": [], "asset_metrics": []}
    threshold = float((1.0 + max(0.0, avg_profit_factor)) / 2.0)
    frame = pd.DataFrame({"symbol": symbols.astype(str), "net_ret": net_rets})
    asset_rows: List[Dict[str, Any]] = []
    excluded: List[str] = []
    for symbol, group in frame.groupby("symbol", sort=True):
        vals = np.asarray(group["net_ret"], dtype=np.float64)
        gross_win = float(np.sum(vals[vals > 0.0]))
        gross_loss = float(np.abs(np.sum(vals[vals < 0.0])))
        if gross_loss > EPS:
            pf = min(gross_win / gross_loss, 100.0)
        elif gross_win > EPS:
            pf = 100.0
        else:
            pf = 0.0
        reject = bool(pf < threshold)
        if reject:
            excluded.append(str(symbol))
        asset_rows.append(
            {
                "symbol": str(symbol),
                "profit_factor": float(pf),
                "profit_factor_threshold": threshold,
                "n_trades": int(len(vals)),
                "net_pnl": float(np.sum(vals)),
                "avg_net_pnl_per_trade": float(np.mean(vals)) if len(vals) else 0.0,
                "excluded": reject,
            }
        )
    return {"excluded_symbols": excluded, "asset_metrics": asset_rows}


def build_strategy_for_inference_payload(
    strategy_rows: List[Dict[str, Any]],
    *,
    data_root: str,
    run_id: str,
    output_filename: str = "strategy_for_inference.json",
) -> Dict[str, Any]:
    """Select deployable strategies and persist the live inference contract."""
    enriched: List[Dict[str, Any]] = []
    lgbm_mask_contracts = _load_lgbm_mask_contracts_for_deployment()
    tolerance = 1e-6
    for row in strategy_rows:
        if not isinstance(row, dict):
            continue
        metrics = row.get("metrics_final")
        if not isinstance(metrics, dict):
            metrics = {}
        avg_net_pnl = _avg_net_pnl_per_trade(row)
        rank = _deployment_rank(row)
        components = _deployment_rank_components(row)
        side = _strategy_side(row)
        enriched_row = dict(row)
        enriched_row["side"] = side
        sid = str(row.get("strategy_id", "") or "")
        lgbm_mask_contract = (
            lgbm_mask_contracts.get(sid)
            or lgbm_mask_contracts.get(strategy_core_id(sid))
            or {}
        )
        enriched_row["lgbm_regime_mask"] = lgbm_mask_contract
        enriched_row["regime_mask_source"] = (
            "embedded_lgbm_final_rule_registry"
            if lgbm_mask_contract
            else "missing_lgbm_mask_contract"
        )
        enriched_row["avg_net_pnl_per_trade"] = float(avg_net_pnl)
        enriched_row["selection_rank"] = float(rank)
        enriched_row.update(components)
        enriched_row["deployment_rank_threshold"] = _deployment_order_rank_threshold(
            enriched_row
        )
        enriched_row["deployment_min_avg_net_pnl_per_trade"] = (
            MIN_DEPLOYMENT_AVG_NET_PNL_PER_TRADE
        )
        enriched_row["deployment_reject_reasons"] = []
        if avg_net_pnl < MIN_DEPLOYMENT_AVG_NET_PNL_PER_TRADE:
            enriched_row["deployment_reject_reasons"].append(
                "avg_net_pnl_per_trade_below_0_2pct"
            )
        if side not in {"long", "short"}:
            enriched_row["deployment_reject_reasons"].append("unknown_side")
        if not np.isfinite(rank):
            enriched_row["deployment_reject_reasons"].append("non_finite_rank")
        if row.get("meta_oof_available") is False:
            enriched_row["deployment_reject_reasons"].append("no_matching_meta_oof")
        if row.get("base_oof_available") is False:
            enriched_row["deployment_reject_reasons"].append("no_matching_base_oof")
        if bool(row.get("empty_trade_outcomes", False)):
            enriched_row["deployment_reject_reasons"].append("empty_trade_outcomes")
        if bool(row.get("no_finite_sizer_oof", False)):
            enriched_row["deployment_reject_reasons"].append("no_finite_sizer_oof")
        baseline_net_pnl = _safe_float(row.get("baseline_net_pnl"), np.nan)
        final_net_pnl = _safe_float(row.get("final_net_pnl"), np.nan)
        baseline_pf = _safe_float(row.get("baseline_profit_factor"), np.nan)
        final_pf = _safe_float(row.get("final_profit_factor"), np.nan)
        if (
            np.isfinite(baseline_net_pnl)
            and np.isfinite(final_net_pnl)
            and final_net_pnl < baseline_net_pnl - tolerance
        ):
            enriched_row["deployment_reject_reasons"].append(
                "policy_replay_worse_than_baseline"
            )
        if avg_net_pnl < 0.0:
            enriched_row["deployment_reject_reasons"].append(
                "final_avg_net_pnl_per_trade_negative"
            )
        if (
            np.isfinite(baseline_pf)
            and np.isfinite(final_pf)
            and final_pf < baseline_pf - tolerance
        ):
            enriched_row["deployment_reject_reasons"].append(
                "final_profit_factor_worse_than_baseline"
            )
        enriched.append(enriched_row)

    candidates = [row for row in enriched if not row.get("deployment_reject_reasons")]
    selected: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = [
        row for row in enriched if row.get("deployment_reject_reasons")
    ]
    for side in ("long", "short"):
        side_rows = [row for row in candidates if row.get("side") == side]
        dedup_by_core: Dict[str, Dict[str, Any]] = {}
        for row in side_rows:
            sid = str(row.get("strategy_id", "") or "")
            core = strategy_core_id(sid)
            prev = dedup_by_core.get(core)
            if prev is None or float(row.get("selection_rank", -np.inf)) > float(
                prev.get("selection_rank", -np.inf)
            ):
                dedup_by_core[core] = row
            else:
                dup_row = dict(row)
                dup_row["deployment_reject_reasons"] = [
                    "duplicate_core_strategy_variant"
                ]
                rejected.append(dup_row)
        side_rows = list(dedup_by_core.values())
        side_rows.sort(
            key=lambda row: float(row.get("selection_rank", float("-inf"))),
            reverse=True,
        )
        selected.extend(side_rows[:MAX_DEPLOYMENT_STRATEGIES_PER_SIDE])
        for row in side_rows[MAX_DEPLOYMENT_STRATEGIES_PER_SIDE:]:
            row = dict(row)
            row["deployment_reject_reasons"] = ["outside_top_1_per_side"]
            rejected.append(row)

    selected.sort(
        key=lambda row: (
            str(row.get("side", "")),
            -float(row.get("selection_rank", float("-inf"))),
        )
    )
    asset_exclusions = {
        str(row.get("strategy_id")): sorted(
            set(str(sym) for sym in row.get("excluded_symbols", []) if str(sym))
        )
        for row in selected
        if row.get("strategy_id")
    }
    payload = {
        "schema_version": "v2",
        "generated_by": "policy_optimiser",
        "run_id": run_id,
        "selection_rules": {
            "min_avg_net_pnl_per_trade": MIN_DEPLOYMENT_AVG_NET_PNL_PER_TRADE,
            "max_strategies_per_side": MAX_DEPLOYMENT_STRATEGIES_PER_SIDE,
            "runtime_rank_threshold_formula": (
                "avg_trades_per_calendar_day_at_top_1pct / 24 * 2 / "
                "avg_holding_time_hours * 0.95"
            ),
            "runtime_rank_threshold_scope": "per_strategy_prediction_rank_only",
            "rank_formula": (
                "effective_ops_day * avg_net_pnl_per_trade / "
                "(sqrt(avg_holding_time_hours) * "
                "sqrt(weekly_wallet_vol + monthly_wallet_vol + eps))"
            ),
            "effective_ops_day_formula": (
                "sqrt(min(36 / avg_holding_time_hours, opportunities_per_day))"
            ),
            "asset_pf_reject_formula": "PF(asset) < (1 + strategy_profit_factor) / 2",
        },
        "strategies": [
            {
                "strategy_id": row.get("strategy_id", ""),
                "canonical_strategy_id": strategy_core_id(
                    str(row.get("strategy_id", "") or "")
                ),
                "strategy_for_inference": row.get("strategy_id", ""),
                "lgbm_regime_mask": row.get("lgbm_regime_mask", {}),
                "regime_mask_source": row.get(
                    "regime_mask_source", "missing_lgbm_mask_contract"
                ),
                "meta_oof_available": bool(row.get("meta_oof_available", True)),
                "base_oof_available": bool(row.get("base_oof_available", True)),
                "policy_replay_source": str(
                    row.get("policy_replay_source", "deployment")
                ),
                "n_trades": int(
                    row.get(
                        "n_trades",
                        row.get("trades_selected", row.get("trades_total", 0)),
                    )
                    or 0
                ),
                "baseline_vs_policy_delta": {
                    "net_pnl_delta": _safe_float(row.get("net_pnl_delta"), 0.0),
                    "profit_factor_delta": _safe_float(
                        row.get("final_profit_factor"), np.nan
                    )
                    - _safe_float(row.get("baseline_profit_factor"), np.nan),
                },
                "side": row.get("side", ""),
                "selected": True,
                "selection_rank": row.get("selection_rank", 0.0),
                "deployment_rank_threshold": row.get("deployment_rank_threshold", 0.0),
                "avg_net_pnl_per_trade": row.get("avg_net_pnl_per_trade", 0.0),
                "effective_ops_day": row.get("effective_ops_day", 0.0),
                "opportunities_per_day": row.get("opportunities_per_day", 0.0),
                "avg_trades_per_day_at_top_1pct": row.get(
                    "avg_trades_per_day_at_top_1pct",
                    row.get("top1_avg_trades_per_day", 0.0),
                ),
                "top1_active_days": row.get("top1_active_days", 0),
                "top1_avg_trades_per_active_day": row.get(
                    "top1_avg_trades_per_active_day",
                    row.get("top1_avg_trades_per_day", 0.0),
                ),
                "top1_avg_trades_per_calendar_day": row.get(
                    "top1_avg_trades_per_calendar_day", 0.0
                ),
                "avg_holding_hours_at_top_1pct": row.get(
                    "avg_holding_hours_at_top_1pct",
                    row.get("top1_avg_holding_hours", 0.0),
                ),
                "policy_candidate_frac": row.get("policy_candidate_frac", 0.0),
                "policy_assessment_frac": row.get("policy_assessment_frac", 0.0),
                "policy_diagnostic_frac": row.get("policy_diagnostic_frac", 0.0),
                "top1_n_trades": row.get("top1_n_trades", 0),
                "top5_n_trades": row.get("top5_n_trades", 0),
                "top5_active_days": row.get("top5_active_days", 0),
                "top5_avg_trades_per_active_day": row.get(
                    "top5_avg_trades_per_active_day",
                    row.get("top5_avg_trades_per_day", 0.0),
                ),
                "top5_avg_trades_per_calendar_day": row.get(
                    "top5_avg_trades_per_calendar_day", 0.0
                ),
                "avg_holding_time_hours": row.get("avg_holding_time_hours", 0.0),
                "weekly_wallet_vol": row.get("weekly_wallet_vol", 0.0),
                "monthly_wallet_vol": row.get("monthly_wallet_vol", 0.0),
                "profit_factor": row.get("profit_factor", 0.0),
                "excluded_symbols": sorted(
                    set(str(sym) for sym in row.get("excluded_symbols", []))
                ),
                "asset_metrics": row.get("asset_metrics", []),
                "metrics": row,
            }
            for row in selected
        ],
        "reconciliation": {
            "sizer_oof_total": int(len(strategy_rows)),
            "sizer_downstream_pass": int(
                sum(
                    1
                    for row in strategy_rows
                    if bool(row.get("allow_downstream", True))
                    and bool(row.get("profitable_for_downstream", True))
                )
            ),
            "meta_oof_matched": int(
                sum(
                    1
                    for row in strategy_rows
                    if bool(row.get("meta_oof_available", True))
                )
            ),
            "policy_optimized": int(len(strategy_rows)),
            "deployment_selected": int(len(selected)),
            "deployment_rejected": int(len(rejected)),
            "deployment_skipped_with_reasons": {
                str(r.get("strategy_id", "")): list(
                    r.get("deployment_reject_reasons", [])
                )
                for r in rejected
            },
        },
        "rejected_strategies": [
            {
                "strategy_id": row.get("strategy_id", ""),
                "side": row.get("side", ""),
                "selected": False,
                "selection_rank": row.get("selection_rank", 0.0),
                "avg_net_pnl_per_trade": row.get("avg_net_pnl_per_trade", 0.0),
                "reject_reasons": row.get("deployment_reject_reasons", []),
                "metrics": row,
            }
            for row in rejected
        ],
        "asset_exclusions": asset_exclusions,
    }
    out_dir = Path(data_root) / "artifacts" / run_id / "policy_params"
    out_dir.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, default=float)
    path = out_dir / output_filename
    path.write_text(text)
    (Path(data_root) / "artifacts" / run_id / output_filename).write_text(text)
    tprint(
        f"Wrote strategy_for_inference selection -> {path} "
        f"(selected={len(selected)} rejected={len(rejected)})"
    )
    return payload


def _metric_score(
    rets: np.ndarray,
    cost_pct: float = 0.003,
    already_net: bool = False,
    already_sized: bool = False,
    executed_mask: Optional[np.ndarray] = None,
    sizes: Optional[np.ndarray] = None,
    ts_vals: Optional[np.ndarray] = None,
    scores: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    if len(rets) == 0:
        return {
            "net_pnl": 0.0,
            "gross_pnl": 0.0,
            "fee_pnl": 0.0,
            "avg_net_pnl_per_trade": 0.0,
            "avg_gross_pnl_per_trade": 0.0,
            "trades_per_calendar_day": 0.0,
            "sortino": 0.0,
            "hit_rate": 0.0,
            "profit_factor": 0.0,
            "max_drawdown": 0.0,
            "ulcer": 0.0,
            "tuw": 0.0,
            "pct_negative_trades": 0.0,
            "n_trades": 0,
            "robust_downside_semi_variance": 1e-12,
            "pnl_top_25pct_taken_trades": 0.0,
            "weekly_sortino": 0.0,
            "monthly_sortino": 0.0,
            "median_pnl_per_winning_trade": 0.0,
            "weekly_gtp": 0.0,
            "trade_return_skew": 0.0,
            "neg_months_pct": 0.0,
            "worst_week_dd_mag": 0.0,
            "weekly_wallet_vol": 0.0,
            "monthly_wallet_vol": 0.0,
        }
    if executed_mask is not None:
        executed_mask = np.asarray(executed_mask, dtype=bool)
        if executed_mask.shape[0] == len(rets):
            rets = rets[executed_mask]
            if sizes is not None and sizes.shape[0] == executed_mask.shape[0]:
                sizes = sizes[executed_mask]
            if ts_vals is not None and len(ts_vals) == executed_mask.shape[0]:
                ts_vals = np.asarray(ts_vals)[executed_mask]
            if scores is not None and len(scores) == executed_mask.shape[0]:
                scores = np.asarray(scores)[executed_mask]

    if len(rets) == 0:
        return {
            "net_pnl": 0.0,
            "gross_pnl": 0.0,
            "fee_pnl": 0.0,
            "avg_net_pnl_per_trade": 0.0,
            "avg_gross_pnl_per_trade": 0.0,
            "trades_per_calendar_day": 0.0,
            "sortino": 0.0,
            "hit_rate": 0.0,
            "profit_factor": 0.0,
            "max_drawdown": 0.0,
            "ulcer": 0.0,
            "tuw": 0.0,
            "pct_negative_trades": 0.0,
            "n_trades": 0,
            "robust_downside_semi_variance": 1e-12,
            "pnl_top_25pct_taken_trades": 0.0,
            "weekly_sortino": 0.0,
            "monthly_sortino": 0.0,
            "median_pnl_per_winning_trade": 0.0,
            "weekly_gtp": 0.0,
            "trade_return_skew": 0.0,
            "neg_months_pct": 0.0,
            "worst_week_dd_mag": 0.0,
            "weekly_wallet_vol": 0.0,
            "monthly_wallet_vol": 0.0,
        }

    sizes_were_provided = sizes is not None
    if sizes is None:
        sizes = np.ones_like(rets)

    # Apply sizing and exact fee calculation.
    # `already_sized=True` means `rets` already includes position size scaling.
    # In that case, only fees are applied again using `sizes`.
    # `already_net=True` means `rets` are already after fee; we only return them.
    if already_sized:
        gross_rets = rets.astype(np.float64)
    else:
        gross_rets = rets.astype(np.float64) * sizes

    if already_net:
        net_rets = gross_rets
        fee_rets = np.zeros_like(net_rets, dtype=np.float64)
    else:
        fee_rets = float(cost_pct) * sizes.astype(np.float64)
        net_rets = gross_rets - fee_rets

    downside = net_rets[net_rets < 0]
    full_std = float(np.std(net_rets)) if len(net_rets) > 1 else 1e-6
    ds_std = float(np.std(downside)) if len(downside) > 1 else 0.0
    ds_std = max(ds_std, full_std * 0.1, 1e-4)
    # Robust downside semi-variance: variance of downside returns (for Sortino denominator)
    robust_downside_semi_variance = (
        float(np.var(downside, ddof=0)) if len(downside) > 1 else ds_std**2
    )
    gross_win = float(np.sum(net_rets[net_rets > 0]))
    gross_loss = float(np.abs(np.sum(net_rets[net_rets < 0])))
    _, dd = _stable_equity_and_drawdown(net_rets)

    ulcer = float(np.sqrt(np.mean(np.square(dd * 100.0)))) if dd.size else 100.0
    tuw = float(np.mean(dd > 1e-12)) if dd.size else 1.0
    pct_negative_trades = float(np.mean(net_rets < 0)) if len(net_rets) > 0 else 0.0

    # Additional metrics for policy optimization
    winning_trades = net_rets[net_rets > 0]
    median_pnl_per_winning_trade = (
        float(np.median(winning_trades)) if len(winning_trades) > 0 else 0.0
    )
    trade_return_skew = (
        float(
            np.mean((net_rets - np.mean(net_rets)) ** 3)
            / (np.std(net_rets) ** 3 + 1e-12)
        )
        if len(net_rets) > 2
        else 0.0
    )

    # Top 25% trades PnL (by score if available, otherwise by return)
    if scores is not None and len(scores) == len(net_rets) and np.std(scores) > 1e-12:
        # Only use scores if they have meaningful variation
        score_threshold = np.percentile(scores, 75)
        top_25_mask = scores >= score_threshold
        pnl_top_25pct_taken_trades = (
            float(np.sum(net_rets[top_25_mask])) if np.any(top_25_mask) else 0.0
        )
    else:
        # Use top 25% by return as fallback
        ret_threshold = np.percentile(net_rets, 75)
        top_25_mask = net_rets >= ret_threshold
        # Ensure we only get top 25%, not all trades
        if np.mean(top_25_mask) > 0.4:  # If more than 40% selected, something is wrong
            # Fall back to taking highest 25% by raw return value
            n_top = max(1, int(len(net_rets) * 0.25))
            top_indices = np.argsort(net_rets)[-n_top:]
            top_25_mask = np.zeros(len(net_rets), dtype=bool)
            top_25_mask[top_indices] = True
        pnl_top_25pct_taken_trades = (
            float(np.sum(net_rets[top_25_mask])) if np.any(top_25_mask) else 0.0
        )

    # Weekly/monthly aggregations using actual timestamps when available
    n = len(net_rets)
    _has_ts = ts_vals is not None and len(ts_vals) == n and np.std(ts_vals) > 1e-6
    trades_per_calendar_day = 0.0
    if _has_ts:
        ts_arr = np.asarray(ts_vals, dtype=np.float64)
        ts_order = np.argsort(ts_arr)
        sorted_ts = ts_arr[ts_order]
        sorted_rets = net_rets[ts_order]
        elapsed_days = max(
            1.0,
            float(sorted_ts[-1] - sorted_ts[0]) / (24.0 * 3600.0 * 1000.0),
        )
        trades_per_calendar_day = float(n / elapsed_days)

        weekly_period = 7.0 * 24.0 * 3600.0 * 1000.0
        monthly_period = 30.0 * 24.0 * 3600.0 * 1000.0

        t_start = sorted_ts[0]
        # --- weekly ---
        weekly_pnls: List[float] = []
        w_start = t_start
        w_sum = 0.0
        for ti in range(len(sorted_rets)):
            if sorted_ts[ti] >= w_start + weekly_period:
                weekly_pnls.append(w_sum)
                w_start = sorted_ts[ti]
                w_sum = 0.0
            w_sum += float(sorted_rets[ti])
        if w_sum != 0.0 or not weekly_pnls:
            weekly_pnls.append(w_sum)

        weekly_gross_win = sum(p for p in weekly_pnls if p > 0)
        weekly_gross_loss = sum(abs(p) for p in weekly_pnls if p < 0)
        weekly_downside = [p for p in weekly_pnls if p < 0]
        weekly_downside_std = (
            float(np.std(weekly_downside)) if len(weekly_downside) > 1 else 0.0
        )
        weekly_downside_std = max(
            weekly_downside_std, abs(np.mean(weekly_pnls)) * 0.05, 1e-6
        )
        weekly_sortino = (
            float(np.mean(weekly_pnls) / weekly_downside_std)
            if weekly_downside_std > 0
            else 0.0
        )
        weekly_gtp = (
            weekly_gross_win / (weekly_gross_loss + 1e-12)
            if weekly_gross_win > 0
            else 0.0
        )
        neg_weeks = sum(1 for p in weekly_pnls if p < 0)
        worst_week_dd_mag = max(0.0, -min(weekly_pnls)) if weekly_pnls else 0.0
        weekly_wallet_vol = float(np.std(weekly_pnls)) if len(weekly_pnls) > 1 else 0.0

        # --- monthly ---
        monthly_pnls: List[float] = []
        m_start = t_start
        m_sum = 0.0
        for ti in range(len(sorted_rets)):
            if sorted_ts[ti] >= m_start + monthly_period:
                monthly_pnls.append(m_sum)
                m_start = sorted_ts[ti]
                m_sum = 0.0
            m_sum += float(sorted_rets[ti])
        if m_sum != 0.0 or not monthly_pnls:
            monthly_pnls.append(m_sum)

        monthly_downside = [p for p in monthly_pnls if p < 0]
        monthly_downside_std = (
            float(np.std(monthly_downside)) if len(monthly_downside) > 1 else 0.0
        )
        monthly_downside_std = max(
            monthly_downside_std, abs(np.mean(monthly_pnls)) * 0.05, 1e-6
        )
        monthly_sortino = (
            float(np.mean(monthly_pnls) / monthly_downside_std)
            if monthly_downside_std > 0
            else 0.0
        )
        neg_months = sum(1 for p in monthly_pnls if p < 0)
        neg_months_pct = float(neg_months / max(1, len(monthly_pnls)))
        monthly_wallet_vol = (
            float(np.std(monthly_pnls)) if len(monthly_pnls) > 1 else 0.0
        )
    else:
        # No timestamps: fall back to per-trade Sortino and chunk-based aggregations
        weekly_sortino = float(np.mean(net_rets) / ds_std)
        if gross_loss > 1e-12:
            weekly_gtp = gross_win / gross_loss
        elif gross_win > 1e-12:
            weekly_gtp = float("inf")
        else:
            weekly_gtp = 0.0
        neg_weeks = 0
        worst_week_dd_mag = float(np.max(dd)) if len(dd) else 0.0

        monthly_sortino = float(np.mean(net_rets) / ds_std)
        n_synthetic_months = max(1, min(12, n))
        chunk_size = max(1, n // n_synthetic_months)
        synthetic_pnls = [
            float(np.sum(net_rets[i * chunk_size : (i + 1) * chunk_size]))
            for i in range(n_synthetic_months)
            if i * chunk_size < n
        ]
        neg_months = sum(1 for p in synthetic_pnls if p < 0)
        neg_months_pct = float(neg_months / max(1, len(synthetic_pnls)))
        weekly_wallet_vol = float(np.std(net_rets)) if len(net_rets) > 1 else 0.0
        monthly_wallet_vol = (
            float(np.std(synthetic_pnls)) if len(synthetic_pnls) > 1 else 0.0
        )
        trades_per_calendar_day = float(n / 725.0)

    result = {
        "net_pnl": float(np.sum(net_rets)),
        "gross_pnl": float(np.sum(gross_rets)),
        "fee_pnl": float(np.sum(fee_rets)),
        "avg_net_pnl_per_trade": float(np.mean(net_rets)) if len(net_rets) else 0.0,
        "avg_gross_pnl_per_trade": (
            float(np.mean(gross_rets)) if len(gross_rets) else 0.0
        ),
        "trades_per_calendar_day": trades_per_calendar_day,
        "sortino": float(np.mean(net_rets)) / ds_std,
        "hit_rate": float(np.mean(net_rets > 0)),
        "profit_factor": (
            min(gross_win / gross_loss, 100.0)
            if gross_loss > EPS
            else (100.0 if gross_win > EPS else 0.0)
        ),
        "max_drawdown": float(np.max(dd)) if len(dd) else 0.0,
        "ulcer": ulcer,
        "tuw": tuw,
        "pct_negative_trades": pct_negative_trades,
        "n_trades": int(len(rets)),
        "robust_downside_semi_variance": robust_downside_semi_variance,
        "pnl_top_25pct_taken_trades": pnl_top_25pct_taken_trades,
        "weekly_sortino": weekly_sortino,
        "monthly_sortino": monthly_sortino,
        "median_pnl_per_winning_trade": median_pnl_per_winning_trade,
        "weekly_gtp": weekly_gtp,
        "trade_return_skew": trade_return_skew,
        "neg_months_pct": neg_months_pct,
        "worst_week_dd_mag": worst_week_dd_mag,
        "weekly_wallet_vol": weekly_wallet_vol,
        "monthly_wallet_vol": monthly_wallet_vol,
    }
    result["avg_net_pnl_per_trade_pct"] = float(result["avg_net_pnl_per_trade"]) * 100.0
    result["avg_gross_pnl_per_trade_pct"] = (
        float(result["avg_gross_pnl_per_trade"]) * 100.0
    )
    result["fee_model"] = "cost_pct_times_position_size"
    result["return_units"] = "decimal_return"
    result["pnl_basis"] = "sized" if already_sized or sizes_were_provided else "unsized"

    _pnl_direct = float(np.sum(net_rets))
    _pnl_parts = gross_win - gross_loss
    _pnl_check = abs(_pnl_direct - _pnl_parts)
    _pnl_tolerance = max(abs(_pnl_direct) * 1e-6, 1e-10)
    if _pnl_check > _pnl_tolerance:
        logger.warning(
            "PnL integrity check: sum(net_rets)=%.8f vs gross_win-gross_loss=%.8f "
            "(delta=%.2e, tol=%.2e, n=%d, gw=%.6f, gl=%.6f)",
            _pnl_direct,
            _pnl_parts,
            _pnl_check,
            _pnl_tolerance,
            n,
            gross_win,
            gross_loss,
        )

    if (
        result["n_trades"] > 0
        and result["net_pnl"] > 0
        and result["profit_factor"] < 1.0
    ):
        logger.warning(
            "Metric consistency: positive net_pnl=%.6f but PF=%.4f (gw=%.6f, gl=%.6f)",
            result["net_pnl"],
            result["profit_factor"],
            gross_win,
            gross_loss,
        )

    if result["max_drawdown"] > 1.0:
        logger.warning(
            "Metric consistency: max_drawdown=%.4f > 1.0 (fractional DD exceeds 100%%)",
            result["max_drawdown"],
        )

    logger.debug(
        "metric_score: n=%d, pnl=%.6f, sortino=%.4f, pf=%.4f, hr=%.4f, "
        "mdd=%.6f, ulcer=%.4f, tuw=%.4f, ds_std=%.6f, ts_based=%s",
        result["n_trades"],
        result["net_pnl"],
        result["sortino"],
        result["profit_factor"],
        result["hit_rate"],
        result["max_drawdown"],
        result["ulcer"],
        result["tuw"],
        ds_std,
        _has_ts,
    )

    return result


def _top_fraction_metrics(
    rets: np.ndarray,
    *,
    fractions: Tuple[float, ...] = (0.01, 0.05, 0.10, 0.20, 0.30),
    cost_pct: float = 0.003,
    already_net: bool = False,
    already_sized: bool = False,
    sizes: Optional[np.ndarray] = None,
    executed_mask: Optional[np.ndarray] = None,
    ts_vals: Optional[np.ndarray] = None,
    scores: Optional[np.ndarray] = None,
) -> Dict[str, Dict[str, float]]:
    n = len(rets)
    if n == 0 or scores is None or len(scores) != n:
        return {}
    score_arr = np.asarray(scores, dtype=np.float64)
    finite = np.isfinite(score_arr)
    if not finite.any():
        return {}
    finite_idx = np.flatnonzero(finite)
    ranked_idx = finite_idx[np.argsort(score_arr[finite], kind="stable")[::-1]]
    out: Dict[str, Dict[str, float]] = {}
    for frac in fractions:
        frac = min(max(float(frac), 0.0), 1.0)
        if frac <= 0.0:
            continue
        k = max(1, int(math.ceil(len(ranked_idx) * frac)))
        idx = ranked_idx[:k]
        out[f"top_{int(round(frac * 10000)):04d}bp"] = _metric_score(
            np.asarray(rets)[idx],
            cost_pct=cost_pct,
            already_net=already_net,
            already_sized=already_sized,
            sizes=(
                np.asarray(sizes)[idx]
                if sizes is not None and len(sizes) == n
                else None
            ),
            executed_mask=(
                np.asarray(executed_mask)[idx]
                if executed_mask is not None and len(executed_mask) == n
                else None
            ),
            ts_vals=(
                np.asarray(ts_vals)[idx]
                if ts_vals is not None and len(ts_vals) == n
                else None
            ),
            scores=score_arr[idx],
        )
    return out


def _resolve_selection_score(outcomes: pd.DataFrame) -> Tuple[np.ndarray, str]:
    """Return the sizer ranking score for trade selection."""
    candidate_cols = ["sizer_score"]
    for col in candidate_cols:
        if col not in outcomes.columns:
            continue
        arr = np.asarray(outcomes[col].values, dtype=np.float32)
        if np.any(np.isfinite(arr)) and float(np.nanstd(arr)) > 1e-12:
            return arr, col
    return np.zeros(len(outcomes), dtype=np.float32), "flat_zero"


def _robust_fit(arr: np.ndarray, mask: np.ndarray) -> Tuple[float, float, float, float]:
    x = np.asarray(arr, dtype=np.float64)
    tr = x[mask]
    if tr.size == 0:
        return 0.0, 1.0, -5.0, 5.0
    med = float(np.nanmedian(tr))
    q1 = float(np.nanpercentile(tr, 25))
    q3 = float(np.nanpercentile(tr, 75))
    iqr = max(q3 - q1, 1e-6)
    lo = float(np.nanpercentile((tr - med) / iqr, 1))
    hi = float(np.nanpercentile((tr - med) / iqr, 99))
    return med, iqr, lo, hi


def _robust_apply(
    arr: np.ndarray, params: Tuple[float, float, float, float]
) -> np.ndarray:
    med, iqr, lo, hi = params
    z = (np.asarray(arr, dtype=np.float64) - med) / max(iqr, 1e-6)
    return np.clip(z, lo, hi).astype(np.float32)


def _load_best_strategy(data_root: str, run_id: str) -> Dict[str, Any]:
    """Load the simple_position_sizer winner only."""
    params_path = (
        Path(data_root)
        / "artifacts"
        / run_id
        / SIMPLE_POSITION_SIZER_ARTIFACT_DIR
        / "strategy_params.json"
    )
    if not params_path.exists():
        return {}
    payload = json.loads(params_path.read_text())
    if str(payload.get("generated_by", "")) != "simple_position_sizer":
        return {}
    buckets = payload.get("buckets", {})
    sid = str(payload.get("best_strategy_id", ""))
    row = buckets.get(sid, {}) if isinstance(buckets, dict) else {}
    return {
        "strategy_id": sid,
        "threshold_pct": float(
            row.get("threshold_pct", payload.get("best_threshold_pct", 90.0))
        ),
        "model": "meta_clf_only",
        "tp_mult": float(row.get("tp_mult", 1.0)),
        "sl_mult": float(row.get("sl_mult", 1.0)),
    }


def _load_sizer_oof_strategy_metadata(
    data_root: str, run_id: str
) -> Dict[str, Dict[str, Any]]:
    path = (
        Path(data_root) / "artifacts" / run_id / "oof" / "simple_sizer_oof_all.parquet"
    )
    if not path.exists():
        return {}
    try:
        df = pd.read_parquet(
            path, columns=["strategy_id", "model", "winner_score_source"]
        )
    except Exception:
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for sid, g in df.groupby("strategy_id", sort=False):
        row: Dict[str, Any] = {}
        if "model" in g.columns:
            vals = g["model"].dropna().astype(str)
            if not vals.empty:
                row["model"] = vals.mode().iloc[0]
        if "winner_score_source" in g.columns:
            vals = g["winner_score_source"].dropna().astype(str)
            if not vals.empty:
                row["winner_score_source"] = vals.mode().iloc[0]
        out[str(sid)] = row
    return out


def _load_strategy_candidates(data_root: str, run_id: str) -> List[Dict[str, Any]]:
    """Load policy candidates from the latest simple_position_sizer artifact only."""

    def _passes_sizer_policy_gate(row: Dict[str, Any]) -> bool:
        allow_downstream = row.get("allow_downstream", None)
        profitable_for_downstream = row.get("profitable_for_downstream", None)
        if allow_downstream is False or profitable_for_downstream is False:
            return False
        wallet_pnl = _safe_float(row.get("wallet_pnl"), float("nan"))
        net_pnl = _safe_float(row.get("net_pnl"), float("nan"))
        pnl_per_trade_pct = _safe_float(
            row.get(
                "net_pnl_per_trade_pct",
                row.get("pnl_per_trade_pct", row.get("pnl_per_trade", float("nan"))),
            ),
            float("nan"),
        )
        return bool(
            math.isfinite(wallet_pnl)
            and math.isfinite(net_pnl)
            and math.isfinite(pnl_per_trade_pct)
            and wallet_pnl > 0.0
            and net_pnl > 0.0
            and pnl_per_trade_pct > 0.2
        )

    rel_path = f"{SIMPLE_POSITION_SIZER_ARTIFACT_DIR}/strategy_params.json"
    path = Path(data_root) / "artifacts" / run_id / rel_path
    if not path.exists():
        tprint(
            "Policy optimiser: no simple_position_sizer strategy_params artifact found."
        )
        return []
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return []
    if str(payload.get("generated_by", "")) != "simple_position_sizer":
        return []
    tprint(
        "Policy optimiser: loading candidates from latest simple_position_sizer "
        f"artifact {rel_path}."
    )

    oof_strategy_metadata = _load_sizer_oof_strategy_metadata(data_root, run_id)
    candidates: List[Dict[str, Any]] = []
    rows = payload.get("strategies", []) if isinstance(payload, dict) else []
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, dict):
                continue
            sid = str(row.get("strategy_id", "") or "")
            if not sid:
                continue
            cand = dict(row)
            cand["_passes_sizer_policy_gate"] = bool(_passes_sizer_policy_gate(row))
            cand["strategy_id"] = sid
            oof_meta = oof_strategy_metadata.get(sid, {})
            cand["model"] = str(oof_meta.get("model") or "meta_clf_only")
            if oof_meta.get("winner_score_source"):
                cand["winner_score_source"] = str(oof_meta["winner_score_source"])
            cand["source_artifact"] = SIMPLE_POSITION_SIZER_ARTIFACT_DIR
            candidates.append(cand)

    merged: Dict[str, Dict[str, Any]] = {}
    for cand in candidates:
        sid = str(cand.get("strategy_id", ""))
        if not sid:
            continue
        prev = merged.get(sid)
        if prev is None:
            merged[sid] = cand
            continue
        prev_pnl = float(prev.get("net_pnl", float("-inf")))
        cand_pnl = float(cand.get("net_pnl", float("-inf")))
        if cand_pnl > prev_pnl:
            merged[sid] = cand
    out = list(merged.values())
    out.sort(
        key=lambda r: (
            float(r.get("net_pnl", float("-inf"))),
            float(r.get("profit_factor", float("-inf"))),
            float(r.get("hit_rate", float("-inf"))),
        ),
        reverse=True,
    )
    return out


def _load_sizer_oof_scores(data_root: str, run_id: str) -> pd.DataFrame:
    """Load simple_position_sizer OOF scores only."""
    _art = Path(data_root) / "artifacts" / run_id
    simple_sizer_oof_path = _art / "oof" / "simple_sizer_oof_all.parquet"
    if simple_sizer_oof_path.exists():
        df = pd.read_parquet(simple_sizer_oof_path)
        if (
            "timestamp" in df.columns
            and "symbol" in df.columns
            and "strategy_id" in df.columns
        ):
            tprint(
                f"Using simple_position_sizer OOF scores from {simple_sizer_oof_path}"
            )
            score_col = (
                "deployment_score_oof"
                if "deployment_score_oof" in df.columns
                else "sizer_score_oof"
            )
            # Pivot to get scores per strategy
            pivot = df.pivot_table(
                index=["timestamp", "symbol"],
                columns="strategy_id",
                values=score_col,
            ).reset_index()
            out = pivot.copy()
            out["timestamp"] = pd.to_datetime(
                out["timestamp"], utc=True, errors="coerce"
            ).dt.tz_convert(None)
            out["symbol"] = out["symbol"].astype(str)
            return out

    raise FileNotFoundError(
        "Policy optimiser requires simple_position_sizer OOF predictions only. "
        f"Missing {simple_sizer_oof_path}"
    )


def _attach_sizer_score(
    outcomes: pd.DataFrame, sizer_scores: pd.DataFrame, strategy_id: str
) -> pd.DataFrame:
    """Attach a strategy-specific score column from the sizer OOF matrix."""
    if strategy_id not in sizer_scores.columns:
        return outcomes
    score_df = sizer_scores[["timestamp", "symbol", strategy_id]].rename(
        columns={strategy_id: "sizer_score"}
    )
    out = outcomes.copy()
    out["timestamp"] = pd.to_datetime(
        out["timestamp"], utc=True, errors="coerce"
    ).dt.tz_convert(None)
    out["symbol"] = out["symbol"].astype(str)
    merged = out.merge(
        score_df, on=["timestamp", "symbol"], how="left", validate="one_to_one"
    )
    return merged


def _resolve_policy_context_oof(
    strategy_id: str,
    meta_oofs: Dict[str, pd.DataFrame],
    base_oofs: Dict[str, pd.DataFrame],
) -> Tuple[str, pd.DataFrame, str]:
    """Resolve the OOF rows used for outcomes while scores come from the sizer.

    simple_position_sizer may persist both base-style ids and *_tbm variants.  The
    policy score is always attached from simple_sizer_oof_all.parquet; this helper
    only chooses the trade-context frame needed to replay exits.
    """
    sid = str(strategy_id or "")
    candidates = [sid]
    if sid.endswith("_tbm"):
        candidates.append(sid[: -len("_tbm")])
    else:
        candidates.append(f"{sid}_tbm")
    meta_contains = [
        str(key) for key in meta_oofs if sid and (sid in str(key) or str(key) in sid)
    ]
    base_contains = [
        str(key) for key in base_oofs if sid and (sid in str(key) or str(key) in sid)
    ]

    for source, oofs, keys in (
        ("meta", meta_oofs, candidates + meta_contains),
        ("base", base_oofs, candidates + base_contains),
    ):
        seen: set[str] = set()
        for key in keys:
            if key in seen:
                continue
            seen.add(key)
            found = oofs.get(key)
            if found is not None:
                return key, found, source

    raise KeyError(f"No matching base/meta OOF context for strategy_id={sid}")


def _filter_df_by_stage_view(
    df: pd.DataFrame, stage_view: Optional[Dict[str, Any]]
) -> pd.DataFrame:
    """Apply symbol/time restrictions from a materialized stage view."""
    if stage_view is None or df is None or df.empty:
        return df

    out = df.copy()
    symbol_col = next((c for c in ("symbol", "__symbol__") if c in out.columns), None)
    ts_col = next((c for c in ("timestamp", "__ts__", "t0") if c in out.columns), None)

    allowed_symbols = stage_view.get("symbols")
    if allowed_symbols is not None and symbol_col is not None:
        out = out[out[symbol_col].astype(str).isin([str(s) for s in allowed_symbols])]

    if ts_col is not None and not out.empty:
        ts = pd.to_datetime(out[ts_col], utc=True, errors="coerce")

        periods = stage_view.get("allowed_periods") or None
        if periods:
            mask = np.zeros(len(out), dtype=bool)
            for period in periods:
                if isinstance(period, dict):
                    p_start = period.get("start_ts") or period.get("start")
                    p_end = period.get("end_ts") or period.get("end")
                elif isinstance(period, (list, tuple)) and len(period) >= 2:
                    p_start, p_end = period[0], period[1]
                else:
                    continue
                p_start = pd.to_datetime(p_start, utc=True, errors="coerce")
                p_end = pd.to_datetime(p_end, utc=True, errors="coerce")
                if pd.isna(p_start) or pd.isna(p_end) or p_end <= p_start:
                    continue
                mask |= (ts >= p_start) & (ts < p_end)
            out = out.loc[mask]
            ts = pd.to_datetime(out[ts_col], utc=True, errors="coerce")

        start_ts = stage_view.get("allowed_start_ts")
        end_ts = stage_view.get("allowed_end_ts")
        if start_ts:
            out = out[ts >= pd.to_datetime(start_ts, utc=True, errors="coerce")]
            ts = pd.to_datetime(out[ts_col], utc=True, errors="coerce")
        if end_ts:
            out = out[ts <= pd.to_datetime(end_ts, utc=True, errors="coerce")]

    return out


def resolve_optimised_selection_frac(
    *, data_root: str, run_id: str, selected: Dict[str, Any]
) -> float:
    """Return the exact selection fraction chosen upstream by the sizer race.

    Contract:
    - Must come from upstream `selection_frac` (selected row or persisted strategy row).
    - No threshold-derived fallback is allowed.
    """
    explicit_frac = _safe_float(selected.get("selection_frac"), float("nan"))
    if np.isfinite(explicit_frac) and explicit_frac > 0.0:
        return float(max(0.01, min(1.0, explicit_frac)))

    params_path = (
        Path(data_root)
        / "artifacts"
        / run_id
        / SIMPLE_POSITION_SIZER_ARTIFACT_DIR
        / "strategy_params.json"
    )
    if params_path.exists():
        try:
            payload = json.loads(params_path.read_text())
            buckets = payload.get("buckets", {}) or {}
            sid = str(selected.get("strategy_id", ""))
            row = buckets.get(sid, {}) if isinstance(buckets, dict) else {}
            row_frac = _safe_float(row.get("selection_frac"), float("nan"))
            if np.isfinite(row_frac) and row_frac > 0.0:
                return float(max(0.01, min(1.0, row_frac)))
        except Exception:
            pass

    sid = str(selected.get("strategy_id", "") or "")
    raise ValueError(
        "Missing upstream selection_frac for policy optimisation "
        f"(strategy_id='{sid}'). selection_frac must be persisted by the "
        "simple_position_sizer artifact."
    )


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20.0, 20.0)))


def build_replay_context(
    *,
    returns: np.ndarray,
    mfe_ret: np.ndarray,
    mae_ret: np.ndarray,
    delta_mfe: Optional[np.ndarray] = None,
    delta_mae: Optional[np.ndarray] = None,
    bars_since_entry: np.ndarray,
    barrier_pct: np.ndarray,
    confidence: np.ndarray,
    trend: Optional[np.ndarray] = None,
    choppiness: Optional[np.ndarray] = None,
    asym: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """Build canonical replay context shared by optimiser and OOS evaluation."""
    rets = np.asarray(returns, dtype=np.float32)
    mfe = np.asarray(mfe_ret, dtype=np.float32)
    mae = np.asarray(mae_ret, dtype=np.float32)
    bars = np.maximum(1, np.asarray(bars_since_entry, dtype=np.int32))
    barr = np.maximum(np.asarray(barrier_pct, dtype=np.float32), 1e-4)
    conf = np.asarray(confidence, dtype=np.float32)
    delta_mfe_arr = (
        np.asarray(delta_mfe, dtype=np.float32)
        if delta_mfe is not None
        else np.maximum(mfe / np.maximum(bars, 1), 1e-6)
    )
    delta_mae_arr = (
        np.asarray(delta_mae, dtype=np.float32)
        if delta_mae is not None
        else np.maximum(mae / np.maximum(bars, 1), 0.0)
    )

    ae_vel = delta_mae_arr / np.maximum(bars, 1)
    pressure = np.clip(delta_mae_arr / (delta_mfe_arr + 1e-6), 0.0, 20.0)
    path_quality = delta_mfe_arr - delta_mae_arr
    progress_per_bar = np.maximum(rets, 0.0) / np.maximum(barr * bars, 1e-4)

    asym_raw = np.log(
        (np.maximum(delta_mfe_arr, 0.0) + 1e-6)
        / (np.maximum(delta_mae_arr, 0.0) + 1e-6)
    )
    if asym is None:
        asym_vals = asym_raw
    else:
        asym_vals = np.asarray(asym, dtype=np.float32)

    return {
        "mfe_ret": mfe,
        "mae_ret": mae,
        "bars_since_entry": bars,
        "barrier_pct": barr,
        "confidence": conf,
        "AE_vel": ae_vel.astype(np.float32),
        "pressure": pressure.astype(np.float32),
        "path_quality": path_quality.astype(np.float32),
        "progress_per_bar": progress_per_bar.astype(np.float32),
        "trend": np.asarray(
            np.zeros_like(rets) if trend is None else trend, dtype=np.float32
        ),
        "choppiness": np.asarray(
            np.zeros_like(rets) if choppiness is None else choppiness, dtype=np.float32
        ),
        "asym": np.asarray(asym_vals, dtype=np.float32),
        "asym_raw": asym_raw.astype(np.float32),
    }


def _pack_future_path_matrices(
    outcomes: pd.DataFrame, selected_idx: np.ndarray
) -> Dict[str, np.ndarray]:
    """Pack selected 15m futures into dense padded matrices for replay."""
    if len(selected_idx) == 0:
        return {}
    required = ("future_opens", "future_highs", "future_lows", "future_closes")
    if not all(col in outcomes.columns for col in required):
        return {}

    idx = np.asarray(selected_idx, dtype=np.int64)
    raw_paths = {col: outcomes[col].values[idx] for col in required}
    lengths = np.zeros(len(idx), dtype=np.int32)
    for i, arr in enumerate(raw_paths["future_opens"]):
        if arr is None:
            lengths[i] = 0
        else:
            lengths[i] = int(np.asarray(arr).shape[0])
    max_len = int(lengths.max()) if len(lengths) else 0
    if max_len <= 0:
        return {}

    packed: Dict[str, np.ndarray] = {}
    for col in required:
        mat = np.full((len(idx), max_len), np.nan, dtype=np.float32)
        for i, arr in enumerate(raw_paths[col]):
            if arr is None:
                continue
            arr_np = np.asarray(arr, dtype=np.float32).reshape(-1)
            if arr_np.size == 0:
                continue
            n_copy = min(arr_np.size, max_len)
            mat[i, :n_copy] = arr_np[:n_copy]
        packed[col] = mat

    if "entry_price" in outcomes.columns:
        entry_price = np.asarray(outcomes["entry_price"].values[idx], dtype=np.float32)
    else:
        entry_price = packed["future_opens"][:, 0].copy()
        entry_price = np.where(
            np.isfinite(entry_price) & (entry_price > 0.0), entry_price, 1.0
        ).astype(np.float32)

    if "is_long" in outcomes.columns:
        is_long = np.asarray(outcomes["is_long"].values[idx], dtype=bool)
    elif "side" in outcomes.columns:
        side = np.asarray(outcomes["side"].values[idx], dtype=str)
        is_long = np.char.lower(side.astype("U")) != "short"
    else:
        is_long = np.ones(len(idx), dtype=bool)

    packed["future_lengths"] = lengths
    packed["entry_price"] = entry_price.astype(np.float32)
    packed["is_long"] = np.asarray(is_long, dtype=bool)
    return packed


def _simulate_tpsl_from_paths_unified(
    context: Dict[str, np.ndarray],
    tp_mult: float = 1.0,
    sl_mult: float = 1.0,
    enable_trailing: bool = False,
    is_baseline: bool = True,  # If True, this is baseline simulation
) -> Optional[np.ndarray]:
    """UNIFIED simulation for both baseline and policy - ensures identical results.

    Exit Logic:
    1. SL: Hard stop at -sl_mult * barrier (capped, as in real trading)
    2. TP: Trailing profit mechanism (if enabled):
       - Track highest high (for longs) or lowest low (for shorts)
       - When new extreme is reached, update trailing threshold
       - Exit when price retraces trail_mult * distance from extreme to activation
       - If disabled: simple exit at TP level

    Args:
        context: Dictionary with future OHLC paths and entry prices
        tp_mult: Take-profit activation level (relative to barrier)
        sl_mult: Stop-loss level (relative to barrier)
        scores: Optional sizer confidence scores for position sizing (5-15% wallet allocation)
        wallet_range: (min_wallet%, max_wallet%) allocation per trade
        trail_mult: Fraction of profit to give back before trailing exit (0.5 = 50% giveback)
        enable_trailing: If True, use trailing profit; if False, use simple TP exit
        is_baseline: If True, use baseline-specific logic (simpler); if False, use policy logic
    """
    required = ("future_opens", "future_highs", "future_lows", "future_closes")
    if not all(col in context for col in required):
        return None

    future_opens = np.asarray(context["future_opens"], dtype=np.float32)
    future_highs = np.asarray(context["future_highs"], dtype=np.float32)
    future_lows = np.asarray(context["future_lows"], dtype=np.float32)
    future_closes = np.asarray(context["future_closes"], dtype=np.float32)
    future_lengths = np.asarray(
        context.get(
            "future_lengths", np.full(len(future_opens), future_opens.shape[1])
        ),
        dtype=np.int32,
    )

    if future_opens.ndim != 2:
        return None

    n_trades, max_bars = future_opens.shape
    entry = np.asarray(context.get("entry_price", future_opens[:, 0]), dtype=np.float32)
    is_long = np.asarray(
        context.get("is_long", np.ones(n_trades, dtype=bool)), dtype=bool
    )
    side = np.where(is_long, 1.0, -1.0).astype(np.float32)

    barrier = np.maximum(
        np.asarray(
            context.get("barrier_pct", np.full(n_trades, 0.02)), dtype=np.float32
        ),
        1e-4,
    )

    tp_dist = tp_mult * barrier
    sl_dist = sl_mult * barrier

    exited = np.zeros(n_trades, dtype=bool)
    exit_rets = np.zeros(n_trades, dtype=np.float32)

    # Trailing profit state tracking
    tp_activated = np.zeros(n_trades, dtype=bool)  # Has TP level been reached?
    extreme_price = entry.copy()  # Highest high (long) or lowest low (short) seen
    trailing_thresh = np.zeros(n_trades, dtype=np.float32)  # Trailing stop threshold

    # We will keep track of exactly what bar a trade exits to compute time-in-market

    for bar in range(max_bars):
        active = (~exited) & (bar < future_lengths)

        if not np.any(active):
            break

        idx = np.flatnonzero(active)
        ent = entry[idx]
        side_a = side[idx]
        bar_high = future_highs[idx, bar]
        bar_low = future_lows[idx, bar]
        bar_close = future_closes[idx, bar]

        valid = (
            np.isfinite(ent)
            & np.isfinite(bar_high)
            & np.isfinite(bar_low)
            & np.isfinite(bar_close)
            & (ent > 0.0)
        )
        if not np.any(valid):
            continue

        idx = idx[valid]
        ent = ent[valid]
        side_a = side_a[valid]
        bar_high = bar_high[valid]
        bar_low = bar_low[valid]
        bar_close = bar_close[valid]

        tp_dist_a = tp_dist[idx]
        sl_dist_a = sl_dist[idx]

        # Compute returns at each price level
        high_ret = side_a * (bar_high / ent - 1.0)
        low_ret = side_a * (bar_low / ent - 1.0)
        worst_ret = np.minimum(high_ret, low_ret)

        # --- STOP LOSS (hard capped) ---
        sl_hit = worst_ret <= -sl_dist_a

        # --- TAKE PROFIT (simple hit detection) ---
        best_ret = np.maximum(high_ret, low_ret)
        tp_hit = best_ret >= tp_dist_a

        # --- TRAILING PROFIT LOGIC ---
        # Update extreme price tracking
        for i_idx, trade_idx in enumerate(idx):
            if side_a[i_idx] > 0:  # Long
                if bar_high[i_idx] > extreme_price[trade_idx]:
                    extreme_price[trade_idx] = bar_high[i_idx]
            else:  # Short
                if bar_low[i_idx] < extreme_price[trade_idx]:
                    extreme_price[trade_idx] = bar_low[i_idx]

        # Check if TP activation level reached
        for i_idx, trade_idx in enumerate(idx):
            if not tp_activated[trade_idx]:
                extreme_ret = side_a[i_idx] * (
                    extreme_price[trade_idx] / ent[i_idx] - 1.0
                )
                if extreme_ret >= tp_dist_a[i_idx]:
                    tp_activated[trade_idx] = True

        # Update trailing threshold when TP is activated
        if enable_trailing:
            for i_idx, trade_idx in enumerate(idx):
                if tp_activated[trade_idx]:
                    if side_a[i_idx] > 0:
                        trailing_thresh[trade_idx] = max(
                            trailing_thresh[trade_idx],
                            ent[i_idx] * (1.0 + tp_dist_a[i_idx] * 0.5),
                        )
                    else:
                        trailing_thresh[trade_idx] = (
                            min(
                                trailing_thresh[trade_idx],
                                ent[i_idx] * (1.0 - tp_dist_a[i_idx] * 0.5),
                            )
                            if trailing_thresh[trade_idx] > 0
                            else ent[i_idx] * (1.0 - tp_dist_a[i_idx] * 0.5)
                        )

        # Check exit conditions
        bar_exit = np.full(len(idx), np.nan, dtype=np.float32)
        for i_idx, trade_idx in enumerate(idx):
            if sl_hit[i_idx]:
                # SL hit: use hard stop (capped)
                bar_exit[i_idx] = -sl_dist_a[i_idx]
            elif tp_activated[trade_idx]:
                if enable_trailing:
                    # Check if trailing stop hit
                    if side_a[i_idx] > 0:  # Long
                        if bar_low[i_idx] <= trailing_thresh[trade_idx]:
                            # Exit at trailing threshold (or better)
                            exit_price = max(bar_low[i_idx], trailing_thresh[trade_idx])
                            bar_exit[i_idx] = exit_price / ent[i_idx] - 1.0
                    else:  # Short
                        if bar_high[i_idx] >= trailing_thresh[trade_idx]:
                            exit_price = min(
                                bar_high[i_idx], trailing_thresh[trade_idx]
                            )
                            bar_exit[i_idx] = ent[i_idx] / exit_price - 1.0
                else:
                    # Simple TP exit (no trailing)
                    # Exit when we first hit the TP level
                    bar_exit[i_idx] = tp_dist_a[i_idx]

        # If no hit, use close for last bar
        is_last = bar >= future_lengths[idx] - 1
        close_ret = side_a * (bar_close / ent - 1.0)
        bar_exit = np.where(np.isnan(bar_exit) & is_last, close_ret, bar_exit)

        exit_now = np.isfinite(bar_exit)
        if np.any(exit_now):
            exit_idx = idx[exit_now]
            exit_rets[exit_idx] = bar_exit[exit_now]
            exited[exit_idx] = True

    # For any not exited, compute final return from last available bar
    not_exited = ~exited
    if np.any(not_exited):
        for i in np.flatnonzero(not_exited):
            last_bar = min(future_lengths[i] - 1, max_bars - 1)
            if last_bar >= 0 and np.isfinite(future_closes[i, last_bar]):
                exit_rets[i] = side[i] * (future_closes[i, last_bar] / entry[i] - 1.0)

    return exit_rets.astype(np.float32)


def _simulate_barwise_path_policy(
    returns: np.ndarray,
    context: Dict[str, np.ndarray],
    params: Dict[str, Any],
    sizes: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """Replay the policy using actual 15m OHLC paths and causal bar-by-bar logic."""
    # DELEGATE to baseline function for simple TP/SL (ensures identical results)
    enable_trailing = bool(params.get("enable_trailing", True))
    enable_early_exit = bool(params.get("enable_early_exit", True))
    enable_compression = bool(params.get("enable_compression", True))
    enable_barrier_conf = bool(params.get("enable_barrier_conf", True))

    # If only simple TP/SL (no advanced features), use baseline for identical results
    if not any(
        [enable_trailing, enable_early_exit, enable_compression, enable_barrier_conf]
    ):
        tp_mult = float(params.get("tp_mult", 1.0))
        sl_mult = float(params.get("sl_mult", 1.0))
        # Call the baseline simulation directly for identical results
        baseline_rets = _simulate_tpsl_from_paths_unified(
            context,
            tp_mult=tp_mult,
            sl_mult=sl_mult,
            enable_trailing=False,
            is_baseline=True,
        )
        if baseline_rets is not None and sizes is not None:
            size_arr = np.asarray(sizes, dtype=np.float32)
            if size_arr.shape[0] == baseline_rets.shape[0]:
                baseline_rets = baseline_rets * size_arr
        return baseline_rets

    required = ("future_opens", "future_highs", "future_lows", "future_closes")
    if not all(col in context for col in required):
        return None

    future_opens = np.asarray(context["future_opens"], dtype=np.float32)
    future_highs = np.asarray(context["future_highs"], dtype=np.float32)
    future_lows = np.asarray(context["future_lows"], dtype=np.float32)
    future_closes = np.asarray(context["future_closes"], dtype=np.float32)
    future_lengths = np.asarray(
        context.get("future_lengths", np.full(len(returns), future_opens.shape[1])),
        dtype=np.int32,
    )
    if future_opens.ndim != 2 or future_highs.shape != future_opens.shape:
        return None
    if (
        future_lows.shape != future_opens.shape
        or future_closes.shape != future_opens.shape
    ):
        return None

    n_trades, max_bars = future_opens.shape
    if n_trades != len(returns):
        return None

    entry = np.asarray(context.get("entry_price", future_opens[:, 0]), dtype=np.float32)
    is_long = np.asarray(
        context.get("is_long", np.ones(n_trades, dtype=bool)), dtype=bool
    )
    side = np.where(is_long, 1.0, -1.0).astype(np.float32)

    barrier = np.maximum(
        np.asarray(
            context.get("barrier_pct", np.full(n_trades, 0.02)), dtype=np.float32
        ),
        1e-4,
    )
    conf = np.asarray(context.get("confidence", np.zeros(n_trades)), dtype=np.float32)
    p_tp = np.asarray(context.get("p_tp", np.full(n_trades, np.nan)), dtype=np.float32)
    p_sl = np.asarray(context.get("p_sl", np.full(n_trades, np.nan)), dtype=np.float32)
    has_barrier_proba = np.any(np.isfinite(p_tp)) and np.any(p_tp > 0.0)

    trend = np.asarray(context.get("trend", np.zeros(n_trades)), dtype=np.float32)
    asym = np.asarray(context.get("asym", np.full(n_trades, 0.5)), dtype=np.float32)
    choppy = np.asarray(context.get("choppiness", np.zeros(n_trades)), dtype=np.float32)
    score_weight_trend = float(params.get("score_weight_trend", 1.0))
    score_weight_asym = float(params.get("score_weight_asym", 0.6))
    score_weight_choppiness = float(params.get("score_weight_choppiness", 0.9))
    s = (
        score_weight_trend * trend
        + score_weight_asym * asym
        - score_weight_choppiness * choppy
    )
    m_raw = 0.7 + 0.6 * _sigmoid(s)
    m = np.clip(
        m_raw,
        float(params.get("multiplier_band_min", 0.8)),
        float(params.get("multiplier_band_max", 1.2)),
    )

    tp_mult = float(params.get("tp_mult", 1.0))
    sl_mult = float(params.get("sl_mult", 1.0))
    tp_dist = tp_mult * barrier
    sl_dist = sl_mult * barrier
    a1 = float(params.get("a1", 0.5))
    a2 = float(params.get("a2", 1.0 - a1))
    b1 = float(params.get("b1", 0.5))
    b2 = float(params.get("b2", 1.0 - b1))
    theta_fail = float(params.get("theta_fail", 0.0))
    theta_path = float(params.get("theta_path", 0.0))
    d_path = int(params.get("d_path", 1))
    k_early = int(params.get("K_early", 3))
    tp_gate = float(params.get("tp_conf_threshold", 0.5))
    sl_gate = float(params.get("sl_early_exit_threshold", 0.4))
    progress_threshold = float(params.get("progress_threshold", 0.0))
    mfe_progress_threshold = float(params.get("mfe_progress_threshold", 0.25))

    trailing_power = float(params.get("trailing_power", 1.0))
    trailing_squash_divisor = float(params.get("trailing_squash_divisor", 1.0))
    trailing_override_alpha = float(params.get("trailing_override_alpha", 0.0))

    enable_early_exit = bool(params.get("enable_early_exit", True))
    enable_trailing = bool(params.get("enable_trailing", True))
    enable_barrier_conf = bool(params.get("enable_barrier_conf", True))
    enable_compression = bool(params.get("enable_compression", True))

    rets = np.asarray(returns, dtype=np.float32).copy()
    exited = np.zeros(n_trades, dtype=bool)
    exit_rets = rets.copy()
    mfe = np.zeros(n_trades, dtype=np.float32)
    mae = np.zeros(n_trades, dtype=np.float32)
    prev_mfe = np.zeros(n_trades, dtype=np.float32)
    prev_mae = np.zeros(n_trades, dtype=np.float32)

    tp_hit_ever = np.zeros(n_trades, dtype=bool)

    # TP-anchored trailing state (matching baseline)
    extreme_price = entry.copy()  # Track highest high / lowest low
    # Initialise trailing thresholds to safe sentinel values so degenerate
    # zero-threshold never triggers a spurious exit:
    #   Longs: threshold well below entry (will never be breached by bar_low)
    #   Shorts: threshold well above entry (will never be breached by bar_high)
    trailing_thresh = np.where(
        is_long,
        entry * 0.01,  # longs: 99% below entry — safe floor
        entry * 100.0,  # shorts: 100x above entry — safe ceiling
    ).astype(np.float32)

    import pandas as pd

    # Strict causal expanding median grouped by symbol to avoid cross-asset contamination
    if "symbol" in context:
        df_atr = pd.DataFrame({"barrier": barrier, "symbol": context["symbol"]})
        full_median_atr = (
            df_atr.groupby("symbol")["barrier"]
            .expanding(min_periods=1)
            .median()
            .reset_index(level=0, drop=True)
            .sort_index()
            .values.astype(np.float32)
        )
    else:
        full_median_atr = (
            pd.Series(barrier)
            .expanding(min_periods=1)
            .median()
            .values.astype(np.float32)
        )

    # We will keep track of exactly what bar a trade exits to compute time-in-market
    exit_lengths = np.full(n_trades, max_bars, dtype=np.int32)

    for bar in range(max_bars):
        active = (~exited) & (bar < future_lengths)

        if not np.any(active):
            break

        idx = np.flatnonzero(active)
        ent = entry[idx]
        side_a = side[idx]
        bar_open = future_opens[idx, bar]
        bar_high = future_highs[idx, bar]
        bar_low = future_lows[idx, bar]
        bar_close = future_closes[idx, bar]

        valid = (
            np.isfinite(ent)
            & np.isfinite(bar_open)
            & np.isfinite(bar_high)
            & np.isfinite(bar_low)
            & np.isfinite(bar_close)
            & (ent > 0.0)
        )
        if not np.any(valid):
            continue

        idx = idx[valid]
        ent = ent[valid]
        side_a = side_a[valid]
        bar_open = bar_open[valid]
        bar_high = bar_high[valid]
        bar_low = bar_low[valid]
        bar_close = bar_close[valid]
        barrier_a = barrier[idx]
        tp_dist_a = tp_dist[idx]
        sl_dist_a = sl_dist[idx]
        m_a = m[idx]

        open_ret = side_a * (bar_open / ent - 1.0)
        high_ret = side_a * (bar_high / ent - 1.0)
        low_ret = side_a * (bar_low / ent - 1.0)
        close_ret = side_a * (bar_close / ent - 1.0)
        best_ret = np.maximum(high_ret, low_ret)
        worst_ret = np.minimum(high_ret, low_ret)

        mfe[idx] = np.maximum(mfe[idx], np.maximum(best_ret, 0.0))
        mae[idx] = np.maximum(mae[idx], np.maximum(-worst_ret, 0.0))
        delta_mfe = np.maximum(mfe[idx] - prev_mfe[idx], 0.0)
        delta_mae = np.maximum(mae[idx] - prev_mae[idx], 0.0)
        prev_mfe[idx] = mfe[idx]
        prev_mae[idx] = mae[idx]

        ae_vel = delta_mae / float(bar + 1)
        pressure = np.clip(delta_mae / (delta_mfe + 1e-6), 0.0, 20.0)
        path_quality = delta_mfe - delta_mae
        progress = np.maximum(close_ret, 0.0) / np.maximum(
            barrier_a * float(bar + 1), 1e-4
        )
        fail_scale = 1.0 - 0.5 * (m_a - 1.0)
        s_fail = (a1 * ae_vel + a2 * pressure) * fail_scale
        s_path = b1 * path_quality + b2 * progress

        # TP hit logic
        tp_hit = best_ret >= tp_dist_a
        tp_hit_ever[idx] = tp_hit_ever[idx] | tp_hit

        # --- NEW TRAILING LOGIC ---
        # Update extreme price tracking
        for i_idx, trade_idx in enumerate(idx):
            if side_a[i_idx] > 0:  # Long
                if bar_high[i_idx] > extreme_price[trade_idx]:
                    extreme_price[trade_idx] = bar_high[i_idx]
            else:  # Short
                if bar_low[i_idx] < extreme_price[trade_idx]:
                    extreme_price[trade_idx] = bar_low[i_idx]

        # Calculate ATR-based median
        atr_ratio_a = barrier_a / np.maximum(full_median_atr[idx], 1e-6)
        trailing_override_band_a = trailing_override_alpha * barrier_a
        giveback_beta_param = float(params.get("giveback_beta", 0.5))

        # Check activation and set trailing threshold
        for i_idx, trade_idx in enumerate(idx):
            extreme_ret = side_a[i_idx] * (extreme_price[trade_idx] / ent[i_idx] - 1.0)

            # Activation condition based on unrealized PnL (extreme_ret) exceeding the band
            if extreme_ret > trailing_override_band_a[i_idx]:
                tp_hit_ever[trade_idx] = True

                activation_price = (
                    ent[i_idx] * (1.0 + trailing_override_band_a[i_idx])
                    if side_a[i_idx] > 0
                    else ent[i_idx] * (1.0 - trailing_override_band_a[i_idx])
                )
                profit_above_activation = abs(
                    extreme_price[trade_idx] - activation_price
                )

                # Apply power curve
                profit_above_activation_ret = profit_above_activation / ent[i_idx]
                trail_dist_ret = (
                    profit_above_activation_ret**trailing_power
                ) / trailing_squash_divisor
                trail_dist = trail_dist_ret * ent[i_idx]

                # Giveback constraint
                D = giveback_beta_param * barrier_a[i_idx]

                if side_a[i_idx] > 0:
                    power_stop = extreme_price[trade_idx] - trail_dist
                    giveback_stop = extreme_price[trade_idx] * (1.0 - D)
                    trailing_thresh[trade_idx] = min(power_stop, giveback_stop)
                else:
                    power_stop = extreme_price[trade_idx] + trail_dist
                    giveback_stop = extreme_price[trade_idx] * (1.0 + D)
                    trailing_thresh[trade_idx] = max(power_stop, giveback_stop)
            else:
                # Keep active flag as false if not breached
                pass

        trail_active = tp_hit_ever[idx]
        # Convert price threshold to return for exit logic
        # trail_floor_price = trailing_thresh[idx]
        # trail_floor_return = side_a * (trail_floor_price / ent - 1.0)
        trail_floor_ret = side_a * (trailing_thresh[idx] / ent - 1.0)

        sl_eff = sl_dist_a.copy()
        if enable_compression:
            mfe_norm = mfe[idx] / np.maximum(tp_dist_a, 1e-4)
            c_start = float(params.get("compression_start", 0.5))
            c_full_raw = float(params.get("compression_full", 1.0))
            c_full = max(c_start + 1e-6, c_full_raw)
            c_max = float(params.get("compression_max_fraction", 0.5))
            c_alpha = (
                np.clip((mfe_norm - c_start) / (c_full - c_start), 0.0, 1.0) * c_max
            )
            sl_eff = sl_dist_a * (1.0 - c_alpha)

        open_sl = open_ret <= -sl_eff

        sl_hit = worst_ret <= -sl_eff
        # Check trail breach symmetrically: Longs check Low, Shorts check High
        trail_hit_long = (side_a > 0) & (bar_low <= trailing_thresh[idx])
        trail_hit_short = (side_a < 0) & (bar_high >= trailing_thresh[idx])
        trail_hit = trail_active & (trail_hit_long | trail_hit_short) & enable_trailing

        # For simplicity, open_trail is checked on open_ret vs trail_floor_ret to be completely symmetric
        # regardless of long/short since return handles side inversion
        open_trail = trail_active & (open_ret <= trail_floor_ret) & enable_trailing

        # TP exit: when trailing is enabled and trade is trail-active, use trailing
        # instead; otherwise fall back to simple TP hit.
        simple_tp_hit = tp_hit & (~trail_active | ~enable_trailing)

        bar_exit = np.full(len(idx), np.nan, dtype=np.float32)
        bar_exit = np.where(open_sl, -sl_eff, bar_exit)
        bar_exit = np.where(open_trail & np.isnan(bar_exit), trail_floor_ret, bar_exit)
        bar_exit = np.where(simple_tp_hit & np.isnan(bar_exit), tp_dist_a, bar_exit)

        hard_hit = sl_hit | trail_hit | open_sl | open_trail | simple_tp_hit
        bar_exit = np.where(
            np.isnan(bar_exit),
            np.where(
                simple_tp_hit,
                tp_dist_a,
                np.where(
                    trail_hit,
                    trail_floor_ret,
                    np.where(sl_hit, -sl_eff, np.nan),
                ),
            ),
            bar_exit,
        )

        # MFE protection: don't trigger fail_exit if trade has made significant progress toward TP
        # This prevents cutting winning trades during temporary pullbacks
        mfe_progress = mfe[idx] / np.maximum(tp_dist_a, 1e-6)
        has_made_progress = mfe_progress >= mfe_progress_threshold

        fail_exit = np.zeros(len(idx), dtype=bool)
        path_exit = np.zeros(len(idx), dtype=bool)

        if enable_early_exit:
            # Determine if we have high continuation confidence to gate early exits
            high_cont = np.zeros(len(idx), dtype=bool)
            if enable_barrier_conf:
                if has_barrier_proba:
                    p_tp_safe = np.where(np.isfinite(p_tp[idx]), p_tp[idx], 0.5)
                    high_cont = p_tp_safe >= tp_gate
                else:
                    cont_thr = float(params.get("continuation_conf_threshold", 0.5))
                    high_cont = conf[idx] >= cont_thr

            # Never trigger fail_exit at bar 0 (first bar) - give trade time to develop
            if bar >= 1:
                if has_barrier_proba and enable_barrier_conf:
                    p_sl_safe = np.where(np.isfinite(p_sl[idx]), p_sl[idx], 0.5)
                    # Allow fail_exit only if trade hasn't made significant progress
                    fail_exit = (
                        (bar + 1 <= k_early)
                        & (p_sl_safe >= sl_gate)
                        & (~has_made_progress)
                        & (~high_cont)
                    )
                else:
                    # Scale theta_fail threshold by progress - harder to trigger as MFE increases
                    progress_factor = np.where(
                        has_made_progress, 2.0, 1.0
                    )  # 2x harder to trigger if progressed
                    fail_exit = (
                        (bar + 1 <= k_early)
                        & (s_fail > theta_fail * progress_factor)
                        & (~high_cont)
                    )

            path_exit = (
                (bar + 1 >= max(3, d_path))
                & (progress >= progress_threshold)
                & (s_path < theta_path)
                & (~high_cont)
            )

        discretionary_exit = ~(hard_hit) & (fail_exit | path_exit)
        if np.any(discretionary_exit):
            bar_exit = np.where(
                discretionary_exit,
                close_ret,
                bar_exit,
            )

        exit_now = np.isfinite(bar_exit)
        if np.any(exit_now):
            exit_idx = idx[exit_now]
            exit_rets[exit_idx] = bar_exit[exit_now]
            exited[exit_idx] = True
            exit_lengths[exit_idx] = bar + 1

    # Fallback: any trade that never exited gets its return from the last
    # available bar (same pattern as _simulate_tpsl_from_paths_unified).
    not_exited = ~exited
    if np.any(not_exited):
        for i in np.flatnonzero(not_exited):
            last_bar = min(int(future_lengths[i]) - 1, max_bars - 1)
            if last_bar >= 0 and np.isfinite(future_closes[i, last_bar]):
                exit_rets[i] = side[i] * (future_closes[i, last_bar] / entry[i] - 1.0)
                exit_lengths[i] = last_bar + 1

    # Inject lengths into context so downstream components can use them
    context["_cached_lengths_"] = exit_lengths

    if sizes is not None:
        size_arr = np.asarray(sizes, dtype=np.float32)
        if size_arr.shape[0] == exit_rets.shape[0]:
            exit_rets = exit_rets * size_arr

    return exit_rets.astype(np.float32)


def replay_exit_policy(
    returns: np.ndarray, context: Dict[str, np.ndarray], params: Dict[str, Any]
) -> np.ndarray:
    """Shared replay logic for optimisation and OOS evaluation."""
    rets = np.asarray(returns, dtype=np.float32).copy()
    barwise_rets = _simulate_barwise_path_policy(rets, context, params)
    if barwise_rets is not None:
        return barwise_rets
    mfe = np.asarray(context.get("mfe_ret", np.abs(rets)), dtype=np.float32)
    mae = np.asarray(
        context.get("mae_ret", np.abs(np.minimum(rets, 0.0))), dtype=np.float32
    )
    bars = np.maximum(
        1,
        np.asarray(
            context.get("bars_since_entry", np.full(len(rets), 4)), dtype=np.int32
        ),
    )
    np.asarray(context.get("confidence", np.zeros(len(rets))), dtype=np.float32)
    barrier = np.maximum(
        np.asarray(
            context.get("barrier_pct", np.full(len(rets), 0.02)), dtype=np.float32
        ),
        1e-4,
    )

    tp_mult = float(params.get("tp_mult", 1.0))
    sl_mult = float(params.get("sl_mult", 1.0))
    tp_dist = tp_mult * barrier
    sl_dist = sl_mult * barrier

    ae_vel = np.asarray(
        context.get("AE_vel", mae / np.maximum(bars, 1)), dtype=np.float32
    )
    # NOTE: pressure uses cumulative formula here for backward compatibility,
    # but the policy_optimiser overwrites this with delta-based pressure from path_bundle.
    # Ensure OOS replay uses path_bundle-based context to get consistent delta-pressure.
    pressure = np.asarray(context.get("pressure", mae / (mfe + EPS)), dtype=np.float32)
    path_quality = np.asarray(context.get("path_quality", mfe - mae), dtype=np.float32)
    progress = np.asarray(
        context.get(
            "progress_per_bar", np.maximum(rets, 0.0) / np.maximum(bars * barrier, 1e-4)
        ),
        dtype=np.float32,
    )

    a1 = float(params.get("a1", 0.5))
    a2 = float(params.get("a2", 1.0 - a1))
    b1 = float(params.get("b1", 0.5))
    b2 = float(params.get("b2", 1.0 - b1))
    theta_fail = float(params.get("theta_fail", 0.0))
    theta_path = float(params.get("theta_path", 0.0))
    d_path = int(params.get("d_path", 1))
    k_early = int(params.get("K_early", 3))

    conf = np.asarray(context.get("confidence", np.zeros(len(rets))), dtype=np.float32)
    p_tp = np.asarray(context.get("p_tp", np.full(len(rets), np.nan)), dtype=np.float32)
    p_sl = np.asarray(context.get("p_sl", np.full(len(rets), np.nan)), dtype=np.float32)
    has_barrier_proba = np.any(np.isfinite(p_tp)) and np.any(p_tp > 0)

    trend = np.asarray(context.get("trend", np.zeros(len(rets))), dtype=np.float32)
    asym = np.asarray(context.get("asym", np.full(len(rets), 0.5)), dtype=np.float32)
    choppy = np.asarray(
        context.get("choppiness", np.zeros(len(rets))), dtype=np.float32
    )
    score_weight_trend = float(params.get("score_weight_trend", 1.0))
    score_weight_asym = float(params.get("score_weight_asym", 0.6))
    score_weight_choppiness = float(params.get("score_weight_choppiness", 0.9))
    s = (
        score_weight_trend * trend
        + score_weight_asym * asym
        - score_weight_choppiness * choppy
    )
    m_raw = 0.7 + 0.6 * _sigmoid(s)
    m = np.clip(
        m_raw,
        float(params.get("multiplier_band_min", 0.8)),
        float(params.get("multiplier_band_max", 1.2)),
    )

    fail_scale = 1.0 - 0.5 * (m - 1.0)
    s_fail = (a1 * ae_vel + a2 * pressure) * fail_scale
    s_path = b1 * path_quality + b2 * progress

    tp_gate = float(params.get("tp_conf_threshold", 0.5))
    sl_gate = float(params.get("sl_early_exit_threshold", 0.4))

    mfe_progress_threshold = float(params.get("mfe_progress_threshold", 0.25))

    # MFE protection: don't trigger fail_exit if trade has made significant progress toward TP
    mfe_progress = mfe / np.maximum(tp_dist, 1e-6)
    has_made_progress = mfe_progress >= mfe_progress_threshold

    enable_early_exit = bool(params.get("enable_early_exit", True))
    enable_trailing = bool(params.get("enable_trailing", True))
    enable_barrier_conf = bool(params.get("enable_barrier_conf", True))
    enable_compression = bool(params.get("enable_compression", True))

    if enable_early_exit:
        # Determine if we have high continuation confidence to gate early exits
        high_cont = np.zeros(len(rets), dtype=bool)
        if enable_barrier_conf:
            if has_barrier_proba:
                p_tp_safe = np.where(np.isfinite(p_tp), p_tp, 0.5)
                high_cont = p_tp_safe >= tp_gate
            else:
                cont_thr = float(params.get("continuation_conf_threshold", 0.5))
                high_cont = conf >= cont_thr

        if has_barrier_proba and enable_barrier_conf:
            p_sl_safe = np.where(np.isfinite(p_sl), p_sl, 0.5)
            fail_exit = (
                (bars <= k_early)
                & (p_sl_safe >= sl_gate)
                & (~has_made_progress)
                & (~high_cont)
            )
        else:
            # Scale theta_fail threshold by progress - harder to trigger as MFE increases
            progress_factor = np.where(has_made_progress, 2.0, 1.0)
            fail_exit = (
                (bars <= k_early)
                & (s_fail > theta_fail * progress_factor)
                & (~high_cont)
            )

        path_exit = (
            (bars >= max(3, d_path))
            & (progress >= float(params.get("progress_threshold", 0.0)))
            & (s_path < theta_path)
            & (~high_cont)
        )
        # Note: in vectorize replay we don't have close_ret easily accessible,
        # but the fallback is just whatever return the trade had at that point
        rets = np.where(fail_exit | path_exit, rets, rets)

    if enable_compression:
        mfe_norm = mfe / np.maximum(tp_dist, 1e-4)
        c_start = float(params.get("compression_start", 0.5))
        c_full_raw = float(params.get("compression_full", 1.0))
        if c_full_raw <= c_start:
            import warnings

            warnings.warn(
                f"compression_full ({c_full_raw}) <= compression_start ({c_start}). "
                f"Adjusting to compression_start + 1e-6. Review your parameter grid.",
                UserWarning,
            )
        c_full = max(c_start + 1e-6, c_full_raw)
        c_max = float(params.get("compression_max_fraction", 0.5))
        c_alpha = np.clip((mfe_norm - c_start) / (c_full - c_start), 0.0, 1.0) * c_max
        sl_eff = sl_dist * (1.0 - c_alpha)
        rets = np.maximum(rets, -sl_eff)

    trailing_power = float(params.get("trailing_power", 1.0))
    trailing_squash_divisor = float(params.get("trailing_squash_divisor", 1.0))
    trailing_override_alpha = float(params.get("trailing_override_alpha", 0.0))
    giveback_beta_param = float(params.get("giveback_beta", 0.5))

    # Compute expanding median ATR strictly causally across the mask to prevent future leakage
    # Using a fast pandas rolling median grouped by symbol to prevent cross-asset leakage
    import pandas as pd

    if "symbol" in context:
        df_atr = pd.DataFrame({"barrier": barrier, "symbol": context["symbol"]})
        median_atr = (
            df_atr.groupby("symbol")["barrier"]
            .expanding(min_periods=1)
            .median()
            .reset_index(level=0, drop=True)
            .sort_index()
            .values.astype(np.float32)
        )
    else:
        median_atr = (
            pd.Series(barrier)
            .expanding(min_periods=1)
            .median()
            .values.astype(np.float32)
        )

    atr_ratio = barrier / np.maximum(median_atr, 1e-6)
    trailing_override_band = trailing_override_alpha * barrier

    trail_on = (mfe > trailing_override_band) & enable_trailing

    # Activation price return is exactly trailing_override_band
    # Profit above activation (in return terms)
    profit_above_activation_ret = np.maximum(0.0, mfe - trailing_override_band)

    # Apply power curve
    trail_dist_ret = (
        profit_above_activation_ret**trailing_power
    ) / trailing_squash_divisor

    # Giveback constraint
    D = giveback_beta_param * atr_ratio / 100.0

    # Trail floor is the max profit minus the giveback distance
    power_stop = mfe - trail_dist_ret
    giveback_stop = mfe - D
    trail_floor = np.minimum(power_stop, giveback_stop)

    # Apply trail floor if trailing is active
    rets = np.where(trail_on, np.maximum(rets, trail_floor), rets)

    if any(
        [enable_trailing, enable_early_exit, enable_compression, enable_barrier_conf]
    ):
        sl_scale = 1.0 + 0.4 * (m - 1.0)
    else:
        sl_scale = 1.0

    # We no longer hard-clip at TP. It's just an activation threshold.
    rets = np.maximum(rets, -sl_dist * sl_scale)
    return rets.astype(np.float32)


def _sequential_optimise(
    base_returns: np.ndarray,
    context: Dict[str, np.ndarray],
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    fixed: Dict[str, Any],
    cost_pct: float,
    executed_mask: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    params = dict(fixed)
    params["enable_early_exit"] = False
    params["enable_compression"] = False
    params["enable_trailing"] = False  # BASIC RUN: Disable trailing for simple TP/SL
    params["enable_barrier_conf"] = False

    # Group parameters into families for incremental testing
    # Each family is tested separately - if it degrades performance, it's disabled
    param_families: List[Tuple[str, List[Tuple[str, List[Any]]]]] = [
        (
            "trailing_stop",
            [
                ("trailing_power", [1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]),
                (
                    "trailing_squash_divisor",
                    [
                        1.0,
                        1.25,
                        1.5,
                        1.75,
                        2.0,
                        2.25,
                        2.5,
                        2.75,
                        3.0,
                        3.25,
                        3.5,
                        3.75,
                        4.0,
                    ],
                ),
                (
                    "trailing_override_alpha",
                    [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
                ),
                ("giveback_beta", [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
            ],
        ),
        (
            "position_sizing",
            [
                ("size_power", [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]),
            ],
        ),
        (
            "multiplier_band",
            [
                ("multiplier_band", [(0.70, 1.30), (0.80, 1.20), (0.85, 1.15)]),
            ],
        ),
    ]

    # Check if future paths are available for fair baseline computation
    has_future_paths_ctx = all(
        col in context
        for col in ("future_opens", "future_highs", "future_lows", "future_closes")
    )
    fixed_tp_mult = float(fixed.get("tp_mult", 1.0))
    fixed_sl_mult = float(fixed.get("sl_mult", 1.0))

    # Track best validation performance and corresponding params
    best_val_params = dict(params)
    param_history: List[Tuple[str, Any, float, float]] = (
        []
    )  # (param_name, value, train_pnl, val_pnl)

    # Use path-based simulation when future paths available - this is the FAIR comparison
    def _get_position_sizes(
        rets: np.ndarray, params_to_use: Dict[str, Any], mask_idx: np.ndarray
    ) -> np.ndarray:
        sizes = np.ones_like(rets)
        size_power = float(params_to_use.get("size_power", 1.0))
        if size_power == 1.0 and "size_power" not in params_to_use:
            return sizes
        if len(rets) == 0:
            return sizes

        scores = context.get("confidence")
        if scores is None or len(scores) != len(rets):
            return sizes

        subset_scores = scores[mask_idx]
        if len(subset_scores) > 0:
            # Rank percentile among approved trades
            ranks = np.argsort(np.argsort(subset_scores))
            approved_rank_pct = ranks / float(max(1, len(subset_scores) - 1))

            size_min = 0.05
            size_max = 0.15
            position_sizes = size_min + (size_max - size_min) * (
                approved_rank_pct**size_power
            )

            sizes[mask_idx] = position_sizes

        return sizes

    # Apply position sizing BEFORE simulation (not post-hoc)
    def _simulate_policy(
        params_to_use: Dict[str, Any], evaluate_mask: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Returns sized returns (sizing applied during simulation)."""
        # Compute sizes upfront based on confidence scores
        sizes = _get_position_sizes(base_returns, params_to_use, evaluate_mask)

        if has_future_paths_ctx:
            rets = _simulate_barwise_path_policy(
                base_returns, context, params_to_use, sizes=sizes
            )
        else:
            base_returns_sized = base_returns * sizes
            rets = replay_exit_policy(base_returns_sized, context, params_to_use)

        if rets is not None:
            # Apply concurrency logic
            if "timestamps_ms" in context and "symbol" in context:
                lengths = context.get("_cached_lengths_")
                if lengths is None:
                    lengths = np.full(len(rets), 24, dtype=np.float32)

                bar_ms = 3600000
                exit_ms = context["timestamps_ms"] + (
                    np.nan_to_num(lengths, nan=0.0) * bar_ms
                ).astype(np.int64)

                conc_mask = _fast_concurrency_mask(
                    context["timestamps_ms"],
                    exit_ms,
                    context["symbol"],
                    context.get("confidence", np.zeros(len(rets))),
                )

                rets = np.where(conc_mask, rets, np.nan)

        # Return sized returns and corresponding sizes used during simulation.
        return rets, sizes

    # Test each parameter family incrementally
    # If a family degrades performance, disable it and move to next
    baseline_rets, baseline_sizes = _simulate_policy(params, val_mask)
    if baseline_rets is not None:
        baseline_val_metrics = _metric_score(
            baseline_rets[val_mask],
            cost_pct=cost_pct,
            already_net=False,
            executed_mask=(
                executed_mask[val_mask] if executed_mask is not None else None
            ),
            already_sized=True,
            sizes=baseline_sizes[val_mask],
            ts_vals=(
                context.get("timestamps_ms", np.arange(len(baseline_rets)))[val_mask]
                if "timestamps_ms" in context
                else None
            ),
            scores=(
                context.get("confidence")[val_mask]
                if context.get("confidence") is not None
                else None
            ),
        )
        baseline_val_metrics["robust_downside_ratio"] = baseline_val_metrics[
            "net_pnl"
        ] / np.sqrt(baseline_val_metrics["robust_downside_semi_variance"])

        baseline_val_metric = 0.0
    else:
        baseline_val_metric = -1e18

    # Track best validation score and corresponding params across families
    best_overall_val_metric = baseline_val_metric
    best_overall_val_metrics = baseline_val_metrics if baseline_rets is not None else {}
    best_overall_params = dict(params)

    disabled_families: List[str] = []

    # Precompute baseline reference metrics on train_mask for custom score calculation
    # This must be done BEFORE the family loop so we can calculate proper custom scores
    if baseline_rets is not None:
        baseline_train_metrics = _metric_score(
            baseline_rets[train_mask],
            cost_pct=cost_pct,
            already_net=False,
            executed_mask=(
                executed_mask[train_mask] if executed_mask is not None else None
            ),
            already_sized=True,
            sizes=baseline_sizes[train_mask],
            ts_vals=(
                context.get("timestamps_ms", np.arange(len(baseline_rets)))[train_mask]
                if "timestamps_ms" in context
                else None
            ),
            scores=(
                context.get("confidence")[train_mask]
                if context.get("confidence") is not None
                else None
            ),
        )
        baseline_train_metrics["robust_downside_ratio"] = baseline_train_metrics[
            "net_pnl"
        ] / np.sqrt(baseline_train_metrics["robust_downside_semi_variance"])
    else:
        baseline_train_metrics = {}

    def _calculate_custom_score(
        metrics: Dict[str, float], _validation_history=None
    ) -> float:
        """Calculate custom composite score using baseline train metrics as reference."""

        def z(val: float, key: str) -> float:
            # We use the baseline metric as the median, and a fixed percentage (e.g. 10% of abs baseline or a small epsilon) as IQR
            # to keep normalization completely stable across Optuna trials.
            med = float(baseline_train_metrics.get(key, 0.0))
            iqr = max(abs(med) * 0.1, 1e-5)
            return (val - med) / iqr

        return (
            0.15 * z(metrics.get("net_pnl", 0.0), "net_pnl")
            + 0.10
            * z(
                metrics.get("net_pnl", 0.0)
                / np.sqrt(metrics.get("robust_downside_semi_variance", 1e-6)),
                "robust_downside_ratio",
            )
            + 0.10
            * z(
                metrics.get("pnl_top_25pct_taken_trades", 0.0),
                "pnl_top_25pct_taken_trades",
            )
            + 0.14 * z(metrics.get("weekly_sortino", 0.0), "weekly_sortino")
            + 0.11 * z(metrics.get("monthly_sortino", 0.0), "monthly_sortino")
            + 0.06
            * z(
                metrics.get("median_pnl_per_winning_trade", 0.0),
                "median_pnl_per_winning_trade",
            )
            + 0.05 * z(metrics.get("weekly_gtp", 0.0), "weekly_gtp")
            + 0.05 * z(metrics.get("trade_return_skew", 0.0), "trade_return_skew")
            - 0.16 * z(metrics.get("ulcer", 0.0), "ulcer")
            - 0.08 * z(metrics.get("tuw", 0.0), "tuw")
            - 0.08 * z(metrics.get("neg_months_pct", 0.0), "neg_months_pct")
            - 0.06 * z(metrics.get("worst_week_dd_mag", 0.0), "worst_week_dd_mag")
        )

    # Calculate actual custom score for baseline validation metrics
    if baseline_rets is not None:
        baseline_val_metric = _calculate_custom_score(baseline_val_metrics)
        best_overall_val_metric = baseline_val_metric
        n_trades = baseline_val_metrics.get("n_trades", 0)
        pnl_per_trade_pct = (
            baseline_val_metrics.get("net_pnl", 0) / max(1, n_trades)
        ) * 100  # Convert to %
        tprint(
            f"  Baseline: PnL/trade={pnl_per_trade_pct:.4f}%, "
            f"Custom Score={baseline_val_metric:.4f}, "
            f"Sortino(W/M)={baseline_val_metrics.get('weekly_sortino', 0):.2f}/{baseline_val_metrics.get('monthly_sortino', 0):.2f}, "
            f"PF={baseline_val_metrics.get('profit_factor', 0):.2f}, "
            f"Trades={n_trades}"
        )

    # Initialize progression tracking before family loop
    if "_progression_" not in best_overall_params:
        best_overall_params["_progression_"] = []
    if baseline_rets is not None:
        # Store params without _progression_ to avoid circular reference
        baseline_params_clean = {
            k: v for k, v in best_overall_params.items() if k != "_progression_"
        }
        best_overall_params["_progression_"].append(
            {
                "step_name": "Baseline",
                "params": baseline_params_clean,
                "metrics": dict(best_overall_val_metrics),
                "score": float(best_overall_val_metric),
            }
        )

    for family_name, family_params in param_families:
        # Skip families that were previously disabled
        if family_name in disabled_families:
            tprint(f"  Skipping disabled family: {family_name}")
            continue

        family_best_params = dict(best_overall_params)  # Start from best known
        _activated_flags: List[str] = []
        if family_name in ("tp_anchored_giveback_combo", "trailing_stop"):
            family_best_params["enable_trailing"] = True
            _activated_flags.append("enable_trailing=True")
        elif family_name == "early_exit":
            family_best_params["enable_early_exit"] = True
            _activated_flags.append("enable_early_exit=True")
        elif family_name == "barrier_conf":
            family_best_params["enable_barrier_conf"] = True
            _activated_flags.append("enable_barrier_conf=True")
        elif family_name == "compression":
            family_best_params["enable_compression"] = True
            _activated_flags.append("enable_compression=True")
        if _activated_flags:
            tprint(
                f"  [VERIFY] Family '{family_name}' activated: {', '.join(_activated_flags)}"
            )
        else:
            tprint(
                f"  [VERIFY] Family '{family_name}' has no enable flags (param-only family)"
            )

        n_trades_best = best_overall_val_metrics.get("n_trades", 0)
        pnl_per_trade_best_pct = (
            best_overall_val_metrics.get("net_pnl", 0) / max(1, n_trades_best)
        ) * 100
        tprint(
            f"  Testing family: {family_name} (current_best_score={best_overall_val_metric:.4f}, current_pnl/trade={pnl_per_trade_best_pct:.4f}%)"
        )

        import optuna

        # We evaluate scores on the current stage/family.
        # This will be useful to perform Z-score normalisation over all trials.

        family_train_scores_all_metrics = []
        family_trial_configs = []

        is_large_family = family_name in ("early_exit", "trailing_stop")

        if is_large_family:
            # Use Optuna for large families
            current_rets, current_sizes = _simulate_policy(
                family_best_params, train_mask
            )
            if current_rets is not None:
                current_metrics = _metric_score(
                    current_rets[train_mask],
                    cost_pct=cost_pct,
                    already_net=False,
                    executed_mask=(
                        executed_mask[train_mask] if executed_mask is not None else None
                    ),
                    already_sized=True,
                    sizes=current_sizes[train_mask],
                    ts_vals=(
                        context.get("timestamps_ms", np.arange(len(current_rets)))[
                            train_mask
                        ]
                        if "timestamps_ms" in context
                        else None
                    ),
                    scores=(
                        context.get("confidence")[train_mask]
                        if context.get("confidence") is not None
                        else None
                    ),
                )
                # Compute robust downside ratio
                current_metrics["robust_downside_ratio"] = current_metrics[
                    "net_pnl"
                ] / np.sqrt(current_metrics["robust_downside_semi_variance"])
                family_train_scores_all_metrics.append(current_metrics)
            else:
                current_metrics = {}

            tprint(f"  Using Optuna to optimize {family_name}...")

            def objective(trial: optuna.Trial) -> float:
                trial_params = dict(family_best_params)
                for name, grid in family_params:
                    # Convert to tuple for mutability safety, optuna handles numerical lists nicely
                    cand = trial.suggest_categorical(name, grid)
                    if name == "multiplier_band":
                        (
                            trial_params["multiplier_band_min"],
                            trial_params["multiplier_band_max"],
                        ) = cand
                    elif name == "a1":
                        trial_params["a1"] = float(cand)
                        trial_params["a2"] = 1.0 - float(cand)
                    elif name == "b1":
                        trial_params["b1"] = float(cand)
                        trial_params["b2"] = 1.0 - float(cand)
                    else:
                        trial_params[name] = cand

                # Split train mask into 5 folds
                valid_train_idx = np.where(train_mask)[0]
                fold_size = len(valid_train_idx) // 5

                fold_scores = []
                for i in range(5):
                    start = i * fold_size
                    end = (i + 1) * fold_size if i < 4 else len(valid_train_idx)
                    fold_mask = np.zeros(len(train_mask), dtype=bool)
                    fold_mask[valid_train_idx[start:end]] = True

                    f_rets, f_sizes = _simulate_policy(trial_params, fold_mask)
                    if f_rets is None:
                        trial.report(-1e18, i)
                        if trial.should_prune():
                            raise optuna.TrialPruned()
                        fold_scores.append(-1e18)
                        continue

                    f_metrics = _metric_score(
                        f_rets[fold_mask],
                        cost_pct=cost_pct,
                        already_net=False,
                        executed_mask=(
                            executed_mask[fold_mask]
                            if executed_mask is not None
                            else None
                        ),
                        already_sized=True,
                        sizes=f_sizes[fold_mask],
                        ts_vals=(
                            context.get("timestamps_ms", np.arange(len(f_rets)))[
                                fold_mask
                            ]
                            if "timestamps_ms" in context
                            else None
                        ),
                        scores=(
                            context.get("confidence")[fold_mask]
                            if context.get("confidence") is not None
                            else None
                        ),
                    )
                    f_metrics["robust_downside_ratio"] = f_metrics["net_pnl"] / np.sqrt(
                        f_metrics["robust_downside_semi_variance"]
                    )

                    # Store metrics for dynamic calculation
                    family_train_scores_all_metrics.append(f_metrics)
                    f_score = _calculate_custom_score(
                        f_metrics, family_train_scores_all_metrics
                    )

                    trial.report(f_score, i)
                    if trial.should_prune():
                        raise optuna.TrialPruned()

                    fold_scores.append(f_score)

                # Full train evaluation
                rets, objective_sizes = _simulate_policy(trial_params, train_mask)
                if rets is None:
                    return -1e18

                metrics = _metric_score(
                    rets[train_mask],
                    cost_pct=cost_pct,
                    already_net=False,
                    executed_mask=(
                        executed_mask[train_mask] if executed_mask is not None else None
                    ),
                    already_sized=True,
                    sizes=objective_sizes[train_mask],
                    ts_vals=(
                        context.get("timestamps_ms", np.arange(len(rets)))[train_mask]
                        if "timestamps_ms" in context
                        else None
                    ),
                    scores=(
                        context.get("confidence")[train_mask]
                        if context.get("confidence") is not None
                        else None
                    ),
                )
                metrics["robust_downside_ratio"] = metrics["net_pnl"] / np.sqrt(
                    metrics["robust_downside_semi_variance"]
                )

                family_train_scores_all_metrics.append(metrics)
                family_trial_configs.append(trial_params)

                return _calculate_custom_score(metrics, family_train_scores_all_metrics)

            pruner = optuna.pruners.MedianPruner(
                n_startup_trials=20, n_warmup_steps=2, interval_steps=1
            )
            sampler = optuna.samplers.TPESampler(seed=42)
            study = optuna.create_study(
                direction="maximize", pruner=pruner, sampler=sampler
            )

            # Enqueue the current baseline to ensure we don't regress
            baseline_trial = {}
            for name, grid in family_params:
                if name == "multiplier_band":
                    baseline_trial[name] = (
                        family_best_params.get("multiplier_band_min"),
                        family_best_params.get("multiplier_band_max"),
                    )
                else:
                    baseline_trial[name] = family_best_params.get(
                        name, grid[0]
                    )  # fallback to first grid val if missing
                # if baseline val not in grid, append it to the grid so optuna doesn't crash on categorical
                if baseline_trial[name] not in grid:
                    grid.append(baseline_trial[name])

            study.enqueue_trial(baseline_trial)
            optuna.logging.set_verbosity(optuna.logging.WARNING)

            last_best_value = -1e18

            def heartbeat_callback(
                study: optuna.Study, trial: optuna.trial.FrozenTrial
            ):
                """Print progress every 10 trials and when new best found."""
                nonlocal last_best_value
                trial_num = trial.number

                # Check if this is a new best
                is_new_best = False
                if trial.value is not None and trial.value > last_best_value + 1e-9:
                    is_new_best = True
                    last_best_value = trial.value

                # Print heartbeat every 10 trials OR on new best
                if (trial_num + 1) % 10 == 0 or is_new_best:
                    best_val = (
                        study.best_value if study.best_value is not None else -1e18
                    )
                    best_trial_num = (
                        study.best_trial.number if study.best_trial is not None else -1
                    )
                    status = "NEW BEST" if is_new_best else f"Trial {trial_num + 1}"
                    current_val_str = (
                        f"{trial.value:.4f}" if trial.value is not None else "N/A"
                    )
                    tprint(
                        f"      [{status}] Best Score: {best_val:.4f} (Trial #{best_trial_num}) | Current: {current_val_str}"
                    )

                    # If new best, print full params and key metrics
                    if is_new_best and study.best_trial is not None:
                        bt = study.best_trial
                        params_str = " | ".join(
                            [
                                f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                for k, v in bt.params.items()
                            ]
                        )
                        tprint(f"         -> Params: {params_str}")

            def early_stopping_callback(
                study: optuna.Study, trial: optuna.trial.FrozenTrial
            ):
                if len(study.trials) > 100:
                    best_value = study.best_value
                    # check if no improvement in last 50 trials
                    recent_trials = study.trials[-50:]
                    improved_recently = any(
                        t.value is not None and t.value > best_value - 1e-6
                        for t in recent_trials
                    )
                    if not improved_recently:
                        study.stop()
                        return

                    if len(study.trials) > 75:
                        older_best = max(
                            [t.value for t in study.trials[:-75] if t.value is not None]
                            + [-1e18]
                        )
                        if best_value > 0 and older_best > 0:
                            if (best_value - older_best) / older_best < 0.005:
                                study.stop()

            tprint(f"      Starting 400 trials for {family_name}...")
            study.optimize(
                objective,
                n_trials=400,
                callbacks=[heartbeat_callback, early_stopping_callback],
            )  # Use 400 trials for trailing stop family with TPE

            # Re-evaluate best with final custom score
            best_combo = study.best_params
            best_custom_score = -1e18

            # Select top 10 unique configs from trials
            valid_trials = [t for t in study.trials if t.value is not None]
            valid_trials.sort(key=lambda t: t.value, reverse=True)
            top_10_configs = []
            seen_configs = set()
            for t in valid_trials:
                config_str = str(sorted(t.params.items()))
                if config_str not in seen_configs:
                    seen_configs.add(config_str)
                    top_10_configs.append(t.params)
                if len(top_10_configs) == 10:
                    break

            # Re-evaluate top 10 across 5 folds (using chronologically partitioned train_mask blocks)
            # This requires actual chronological splitting of the train_mask
            valid_train_idx = np.where(train_mask)[0]
            fold_size = len(valid_train_idx) // 5
            folds = []
            for i in range(5):
                start = i * fold_size
                end = (i + 1) * fold_size if i < 4 else len(valid_train_idx)
                fold_mask = np.zeros(len(train_mask), dtype=bool)
                fold_mask[valid_train_idx[start:end]] = True
                folds.append(fold_mask)

            top_10_scores = []
            for config in top_10_configs:
                fold_scores = []
                for f_mask in folds:
                    f_rets, f_sizes = _simulate_policy(config, f_mask)
                    if f_rets is not None:
                        f_metrics = _metric_score(
                            f_rets[f_mask],
                            cost_pct=cost_pct,
                            already_net=False,
                            executed_mask=(
                                executed_mask[f_mask]
                                if executed_mask is not None
                                else None
                            ),
                            already_sized=True,
                            sizes=f_sizes[f_mask],
                            ts_vals=(
                                context.get("timestamps_ms", np.arange(len(f_rets)))[
                                    f_mask
                                ]
                                if "timestamps_ms" in context
                                else None
                            ),
                            scores=(
                                context.get("confidence")[f_mask]
                                if context.get("confidence") is not None
                                else None
                            ),
                        )
                        f_metrics["robust_downside_ratio"] = f_metrics[
                            "net_pnl"
                        ] / np.sqrt(f_metrics["robust_downside_semi_variance"])
                        f_score = _calculate_custom_score(
                            f_metrics, family_train_scores_all_metrics
                        )
                        fold_scores.append(f_score)
                    else:
                        fold_scores.append(-1e18)

                # selection_score = median_5fold(Score) - 0.4 * IQR_5fold(Score) + 0.1 * min_5fold(Score)
                f_arr = np.array(fold_scores)
                if len(f_arr) > 0 and np.all(f_arr > -1e17):
                    f_med = np.median(f_arr)
                    f_iqr = np.percentile(f_arr, 75) - np.percentile(f_arr, 25)
                    f_min = np.min(f_arr)
                    sel_score = f_med - 0.4 * f_iqr + 0.1 * f_min
                else:
                    sel_score = -1e18
                top_10_scores.append((sel_score, config, f_arr))

            top_10_scores.sort(key=lambda x: x[0], reverse=True)
            if top_10_scores:
                best_custom_score = top_10_scores[0][0]
                best_combo = top_10_scores[0][1]

                tprint("    Top 10 Finalist Selection:")
                for rank, (s_score, cfg, f_arr) in enumerate(top_10_scores):
                    # Re-evaluate global metrics for this specific config to log them
                    g_metrics = None
                    if rank < 10:
                        g_rets, g_sizes = _simulate_policy(cfg, train_mask)
                        if g_rets is not None:
                            g_metrics = _metric_score(
                                g_rets[train_mask],
                                cost_pct=cost_pct,
                                already_net=False,
                                executed_mask=(
                                    executed_mask[train_mask]
                                    if executed_mask is not None
                                    else None
                                ),
                                already_sized=True,
                                sizes=g_sizes[train_mask],
                                ts_vals=(
                                    context.get(
                                        "timestamps_ms", np.arange(len(g_rets))
                                    )[train_mask]
                                    if "timestamps_ms" in context
                                    else None
                                ),
                                scores=(
                                    context.get("confidence")[train_mask]
                                    if context.get("confidence") is not None
                                    else None
                                ),
                            )
                            g_metrics["robust_downside_ratio"] = g_metrics[
                                "net_pnl"
                            ] / np.sqrt(g_metrics["robust_downside_semi_variance"])

                    # Build detailed output
                    tprint(
                        f"      {rank+1}. Selection Score: {s_score:.4f} | Folds: {[f'{x:.4f}' for x in f_arr]}"
                    )
                    if g_metrics is not None:
                        g_n_trades = g_metrics["n_trades"]
                        g_pnl_per_trade_pct = (
                            g_metrics["net_pnl"] / max(1, g_n_trades)
                        ) * 100
                        tprint(
                            f"         PnL/trade: {g_pnl_per_trade_pct:.4f}% | "
                            f"Sortino(W/M): {g_metrics['weekly_sortino']:.2f}/{g_metrics['monthly_sortino']:.2f} | "
                            f"PF: {g_metrics['profit_factor']:.2f} | "
                            f"HR: {g_metrics['hit_rate']:.1%} | "
                            f"MDD: {g_metrics['max_drawdown']:.4f} | "
                            f"Ulcer: {g_metrics['ulcer']:.2f} | "
                            f"Trades: {g_n_trades}"
                        )
                        g_top25_per_trade_pct = (
                            g_metrics["pnl_top_25pct_taken_trades"] / max(1, g_n_trades)
                        ) * 100
                        g_medwin_per_trade_pct = (
                            g_metrics["median_pnl_per_winning_trade"]
                        ) * 100
                        tprint(
                            f"         Top25%/trade: {g_top25_per_trade_pct:.4f}% | "
                            f"MedWin/trade: {g_medwin_per_trade_pct:.4f}% | "
                            f"Skew: {g_metrics['trade_return_skew']:.2f} | "
                            f"WeeklyGtP: {g_metrics['weekly_gtp']:.2f}"
                        )

                    if rank == 0:
                        params_str = " | ".join(
                            [
                                f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                for k, v in cfg.items()
                            ]
                        )
                        tprint(f"         Winner Params: {params_str}")

            baseline_custom_score = _calculate_custom_score(
                family_train_scores_all_metrics[0], family_train_scores_all_metrics
            )

            if best_custom_score > baseline_custom_score + 1e-6:
                tprint(
                    f"    Optuna {family_name}: best_params={best_combo}, delta Score={best_custom_score - baseline_custom_score:+.4f}"
                )
                for name, _ in family_params:
                    cand = best_combo[name]
                    param_history.append(
                        (name, cand, float(best_custom_score), float("nan"))
                    )  # val appended later
                    if name == "multiplier_band":
                        (
                            family_best_params["multiplier_band_min"],
                            family_best_params["multiplier_band_max"],
                        ) = cand
                    elif name == "a1":
                        family_best_params["a1"] = float(cand)
                        family_best_params["a2"] = 1.0 - float(cand)
                    elif name == "b1":
                        family_best_params["b1"] = float(cand)
                        family_best_params["b2"] = 1.0 - float(cand)
                    else:
                        family_best_params[name] = cand
            else:
                tprint(f"    Optuna {family_name}: no improvement over baseline")
        else:
            # Grid Search (Cartesian product) for small families
            current_rets, current_sizes = _simulate_policy(
                family_best_params, train_mask
            )
            if current_rets is not None:
                current_train_metrics_grid = _metric_score(
                    current_rets[train_mask],
                    cost_pct=cost_pct,
                    already_net=False,
                    executed_mask=(
                        executed_mask[train_mask] if executed_mask is not None else None
                    ),
                    already_sized=True,
                    sizes=current_sizes[train_mask],
                    ts_vals=(
                        context.get("timestamps_ms", np.arange(len(current_rets)))[
                            train_mask
                        ]
                        if "timestamps_ms" in context
                        else None
                    ),
                    scores=(
                        context.get("confidence")[train_mask]
                        if context.get("confidence") is not None
                        else None
                    ),
                )
                current_train_metrics_grid["robust_downside_ratio"] = (
                    current_train_metrics_grid["net_pnl"]
                    / np.sqrt(
                        current_train_metrics_grid["robust_downside_semi_variance"]
                    )
                )
                current_train_score = _calculate_custom_score(
                    current_train_metrics_grid, family_train_scores_all_metrics
                )
            else:
                current_train_score = -1e18

            # Build grids and ensure baseline is included
            safe_grids = []
            for name, grid in family_params:
                if name == "multiplier_band":
                    current_val = (
                        family_best_params.get("multiplier_band_min"),
                        family_best_params.get("multiplier_band_max"),
                    )
                    if current_val[0] is None or current_val[1] is None:
                        current_val = None
                else:
                    current_val = family_best_params.get(name)

                test_grid = list(grid)
                if current_val is not None and current_val not in test_grid:
                    test_grid.append(current_val)
                safe_grids.append(test_grid)

            all_combos = list(itertools.product(*safe_grids))
            tprint(f"  Testing {len(all_combos)} combinations for {family_name}...")

            best_combo_score = current_train_score
            best_combo = None

            for combo in all_combos:
                trial = dict(family_best_params)
                for i, (name, _) in enumerate(family_params):
                    cand = combo[i]
                    if name == "multiplier_band":
                        (
                            trial["multiplier_band_min"],
                            trial["multiplier_band_max"],
                        ) = cand
                    elif name == "a1":
                        trial["a1"] = float(cand)
                        trial["a2"] = 1.0 - float(cand)
                    elif name == "b1":
                        trial["b1"] = float(cand)
                        trial["b2"] = 1.0 - float(cand)
                    else:
                        trial[name] = cand

                rets, trial_sizes = _simulate_policy(trial, train_mask)
                if rets is None:
                    continue
                combo_metrics = _metric_score(
                    rets[train_mask],
                    cost_pct=cost_pct,
                    already_net=False,
                    executed_mask=(
                        executed_mask[train_mask] if executed_mask is not None else None
                    ),
                    already_sized=True,
                    sizes=trial_sizes[train_mask],
                    ts_vals=(
                        context.get("timestamps_ms", np.arange(len(rets)))[train_mask]
                        if "timestamps_ms" in context
                        else None
                    ),
                    scores=(
                        context.get("confidence")[train_mask]
                        if context.get("confidence") is not None
                        else None
                    ),
                )
                combo_metrics["robust_downside_ratio"] = combo_metrics[
                    "net_pnl"
                ] / np.sqrt(combo_metrics["robust_downside_semi_variance"])
                family_train_scores_all_metrics.append(combo_metrics)
                train_score = _calculate_custom_score(
                    combo_metrics, family_train_scores_all_metrics
                )

                if train_score > best_combo_score + 1e-6:
                    best_combo_score = train_score
                    best_combo = combo

            if best_combo is not None:
                delta_score = best_combo_score - current_train_score
                tprint(
                    f"    Grid {family_name}: best_combo={best_combo}, delta Score={delta_score:+.4f}"
                )
                for i, (name, _) in enumerate(family_params):
                    cand = best_combo[i]
                    param_history.append(
                        (name, cand, float(best_combo_score), float("nan"))
                    )
                    if name == "multiplier_band":
                        (
                            family_best_params["multiplier_band_min"],
                            family_best_params["multiplier_band_max"],
                        ) = cand
                    elif name == "a1":
                        family_best_params["a1"] = float(cand)
                        family_best_params["a2"] = 1.0 - float(cand)
                    elif name == "b1":
                        family_best_params["b1"] = float(cand)
                        family_best_params["b2"] = 1.0 - float(cand)
                    else:
                        family_best_params[name] = cand
            else:
                tprint(f"    Grid {family_name}: no improvement over baseline")

        # Test if this family improved over baseline
        family_rets, family_sizes = _simulate_policy(family_best_params, val_mask)
        if family_rets is not None:
            family_val_metrics = _metric_score(
                family_rets[val_mask],
                cost_pct=cost_pct,
                already_net=False,
                executed_mask=(
                    executed_mask[val_mask] if executed_mask is not None else None
                ),
                already_sized=True,
                sizes=family_sizes[val_mask],
                ts_vals=(
                    context.get("timestamps_ms", np.arange(len(family_rets)))[val_mask]
                    if "timestamps_ms" in context
                    else None
                ),
                scores=(
                    context.get("confidence")[val_mask]
                    if context.get("confidence") is not None
                    else None
                ),
            )
            family_val_metrics["robust_downside_ratio"] = family_val_metrics[
                "net_pnl"
            ] / np.sqrt(family_val_metrics["robust_downside_semi_variance"])

            # Recalculate baseline val metrics for comparison
            baseline_val_rets, baseline_val_sizes = _simulate_policy(
                best_overall_params, val_mask
            )
            baseline_val_metrics = _metric_score(
                baseline_val_rets[val_mask],
                cost_pct=cost_pct,
                already_net=False,
                executed_mask=(
                    executed_mask[val_mask] if executed_mask is not None else None
                ),
                already_sized=True,
                sizes=baseline_val_sizes[val_mask],
                ts_vals=(
                    context.get("timestamps_ms", np.arange(len(baseline_val_rets)))[
                        val_mask
                    ]
                    if "timestamps_ms" in context
                    else None
                ),
                scores=(
                    context.get("confidence")[val_mask]
                    if context.get("confidence") is not None
                    else None
                ),
            )
            baseline_val_metrics["robust_downside_ratio"] = baseline_val_metrics[
                "net_pnl"
            ] / np.sqrt(baseline_val_metrics["robust_downside_semi_variance"])

            # Since baseline_metrics is used as a static ref, we can just call it on the dict
            validation_history = [baseline_val_metrics, family_val_metrics]
            family_val_metric = _calculate_custom_score(
                family_val_metrics, validation_history
            )
            best_overall_val_metric_current = _calculate_custom_score(
                baseline_val_metrics, validation_history
            )

            if (
                family_val_metric > best_overall_val_metric_current + 0.001
            ):  # Significant improvement threshold
                _b_pnl = baseline_val_metrics.get("net_pnl", 0.0)
                _f_pnl = family_val_metrics.get("net_pnl", 0.0)
                _b_pf = baseline_val_metrics.get("profit_factor", 0.0)
                _f_pf = family_val_metrics.get("profit_factor", 0.0)
                _b_hr = baseline_val_metrics.get("hit_rate", 0.0)
                _f_hr = family_val_metrics.get("hit_rate", 0.0)
                _b_ws = baseline_val_metrics.get("weekly_sortino", 0.0)
                _f_ws = family_val_metrics.get("weekly_sortino", 0.0)
                _b_ms = baseline_val_metrics.get("monthly_sortino", 0.0)
                _f_ms = family_val_metrics.get("monthly_sortino", 0.0)
                tprint(
                    f"  ✓ Family '{family_name}' improved validation Score: {best_overall_val_metric_current:.4f} -> {family_val_metric:.4f}"
                )
                tprint(
                    f"  [VERIFY] PnL: {_b_pnl:+.6f} -> {_f_pnl:+.6f} (Δ{_f_pnl - _b_pnl:+.6f})"
                )
                tprint(
                    f"  [VERIFY] PF: {_b_pf:.4f} -> {_f_pf:.4f} | "
                    f"HR: {_b_hr:.4f} -> {_f_hr:.4f} | "
                    f"WkSortino: {_b_ws:.4f} -> {_f_ws:.4f} | "
                    f"MoSortino: {_b_ms:.4f} -> {_f_ms:.4f}"
                )
                _active_keys = [
                    k
                    for k in (
                        "enable_trailing",
                        "enable_early_exit",
                        "enable_compression",
                        "enable_barrier_conf",
                    )
                    if family_best_params.get(k)
                ]
                tprint(
                    f"  [VERIFY] Active policy features: {_active_keys or 'none (param-only)'}"
                )

                # End of family tprint
                tprint(f"  [End of Family: {family_name} Metrics]")
                tprint(f"    PnL: {family_val_metrics['net_pnl']:.4f}")
                tprint(
                    f"    Robust Downside Ratio: {family_val_metrics['robust_downside_ratio']:.4f}"
                )
                tprint(
                    f"    PnL Top 25% Taken Trades: {family_val_metrics['pnl_top_25pct_taken_trades']:.4f}"
                )
                tprint(
                    f"    Weekly Sortino: {family_val_metrics['weekly_sortino']:.4f}"
                )
                tprint(
                    f"    Monthly Sortino: {family_val_metrics['monthly_sortino']:.4f}"
                )
                tprint(
                    f"    Median PnL / Win: {family_val_metrics['median_pnl_per_winning_trade']:.4f}"
                )
                tprint(f"    Weekly GtP: {family_val_metrics['weekly_gtp']:.4f}")
                tprint(
                    f"    Trade Return Skew: {family_val_metrics['trade_return_skew']:.4f}"
                )
                tprint(f"    Ulcer: {family_val_metrics['ulcer']:.4f}")
                tprint(f"    TUW: {family_val_metrics['tuw']:.4f}")
                tprint(f"    Neg Months %: {family_val_metrics['neg_months_pct']:.4%}")
                tprint(
                    f"    Abs Worst Week DD: {family_val_metrics['worst_week_dd_mag']:.4f}"
                )
                tprint(
                    f"    Trades/Day: {family_val_metrics['n_trades'] / 725.0:.2f}"
                )  # Approx
                if family_val_metrics["n_trades"] > 0:
                    tprint(
                        f"    Avg PnL/Trade (net): {family_val_metrics['net_pnl']/family_val_metrics['n_trades']:.6f}"
                    )

                best_overall_params = dict(family_best_params)
                best_overall_val_metric = family_val_metric
                best_overall_val_metrics = family_val_metrics

                # Store params without _progression_ to avoid circular reference
                family_params_clean = {
                    k: v for k, v in best_overall_params.items() if k != "_progression_"
                }
                best_overall_params["_progression_"].append(
                    {
                        "step_name": family_name,
                        "params": family_params_clean,
                        "metrics": dict(best_overall_val_metrics),
                        "score": float(best_overall_val_metric),
                    }
                )

                # Update param_history val scores
                for i in range(len(param_history) - 1, -1, -1):
                    if np.isnan(param_history[i][3]):
                        param_history[i] = (
                            param_history[i][0],
                            param_history[i][1],
                            param_history[i][2],
                            float(family_val_metric),
                        )
                    else:
                        break  # Stop at previous family
            else:
                _b_pnl_r = (
                    baseline_val_metrics.get("net_pnl", 0.0)
                    if family_val_metrics
                    else 0.0
                )
                _f_pnl_r = (
                    family_val_metrics.get("net_pnl", 0.0)
                    if family_val_metrics
                    else 0.0
                )
                tprint(
                    f"  ✗ Family '{family_name}' did not improve (or degraded): {best_overall_val_metric:.4f} -> {family_val_metric:.4f}"
                )
                tprint(
                    f"  [VERIFY] Rejected: PnL was {_b_pnl_r:+.6f} -> {_f_pnl_r:+.6f} (Δ{_f_pnl_r - _b_pnl_r:+.6f})"
                )
                disabled_families.append(family_name)

                # Remove rejected family params from param_history
                for i in range(len(param_history) - 1, -1, -1):
                    if np.isnan(param_history[i][3]):
                        param_history.pop()
                    else:
                        break

    # Return the best params found across all families
    best_val_params = dict(best_overall_params)

    _accepted_families = [f for f, _ in param_families if f not in disabled_families]
    _rejected_families = list(disabled_families)
    tprint(
        f"Optimization complete: best_overall_val_score={best_overall_val_metric:.4f}"
    )
    tprint(
        f"  [VERIFY] Families accepted={_accepted_families}, rejected={_rejected_families}"
    )
    _final_active = [
        k
        for k in (
            "enable_trailing",
            "enable_early_exit",
            "enable_compression",
            "enable_barrier_conf",
        )
        if best_val_params.get(k)
    ]
    tprint(f"  [VERIFY] Final active policy features: {_final_active or 'none'}")
    tprint(
        f"  [VERIFY] Final params: tp_mult={best_val_params.get('tp_mult')}, "
        f"sl_mult={best_val_params.get('sl_mult')}, "
        f"size_power={best_val_params.get('size_power', 'default')}, "
        f"trailing_power={best_val_params.get('trailing_power', 'default')}"
    )

    # Compute fair baseline from paths when available
    baseline_rets_raw = None
    if has_future_paths_ctx:
        baseline_rets_raw = _simulate_tpsl_from_paths_unified(
            context,
            tp_mult=fixed_tp_mult,
            sl_mult=fixed_sl_mult,
            enable_trailing=False,  # BASIC RUN: Simple TP/SL only, no trailing
            is_baseline=True,
        )

    if baseline_rets_raw is not None:
        params["metrics_baseline"] = _metric_score(
            baseline_rets_raw[val_mask],
            cost_pct=cost_pct,
            already_net=False,
            already_sized=False,
            sizes=baseline_sizes[val_mask],
            executed_mask=(
                executed_mask[val_mask] if executed_mask is not None else None
            ),
            ts_vals=(
                context.get("timestamps_ms", np.arange(len(baseline_rets_raw)))[
                    val_mask
                ]
                if "timestamps_ms" in context
                else None
            ),
            scores=(
                context.get("confidence")[val_mask]
                if context.get("confidence") is not None
                else None
            ),
        )
    else:
        params["metrics_baseline"] = _metric_score(
            base_returns[val_mask],
            cost_pct=cost_pct,
            already_net=False,
            already_sized=False,
            sizes=baseline_sizes[val_mask],
            executed_mask=(
                executed_mask[val_mask] if executed_mask is not None else None
            ),
            ts_vals=(
                context.get("timestamps_ms", np.arange(len(base_returns)))[val_mask]
                if "timestamps_ms" in context
                else None
            ),
            scores=(
                context.get("confidence")[val_mask]
                if context.get("confidence") is not None
                else None
            ),
        )
    # Use the params that achieved best validation performance, not just final accumulated
    # This prevents overfitting to training set during sequential optimization
    final_rets, final_sizes = _simulate_policy(best_val_params, val_mask)

    best_val_params["metrics_final"] = _metric_score(
        final_rets[val_mask] if final_rets is not None else np.zeros(len(val_mask)),
        cost_pct=cost_pct,
        already_net=False,
        executed_mask=(executed_mask[val_mask] if executed_mask is not None else None),
        already_sized=True,
        sizes=final_sizes[val_mask],
        ts_vals=(
            context.get("timestamps_ms", np.arange(len(val_mask)))[val_mask]
            if "timestamps_ms" in context
            else None
        ),
        scores=(
            context.get("confidence")[val_mask]
            if context.get("confidence") is not None
            else None
        ),
    )
    best_val_params["_param_history_"] = param_history  # For diagnostics
    return best_val_params


def _fast_concurrency_mask(
    entry_timestamps: np.ndarray,
    exit_timestamps: np.ndarray,
    symbols: np.ndarray,
    scores: np.ndarray,
    max_global_concurrent: int = 3,
) -> np.ndarray:
    """
    Generates a boolean mask indicating which trades should be executed,
    enforcing that no two trades on the same symbol overlap, and that
    a maximum of `max_global_concurrent` trades can be open globally at any time.
    Trades are processed chronologically. In case of exact timestamp ties, higher confidence wins.
    """
    n_trades = len(entry_timestamps)
    mask = np.zeros(n_trades, dtype=bool)

    if n_trades == 0:
        return mask

    # Sort trades chronologically, then by score (descending)
    sort_idx = np.lexsort((-scores, entry_timestamps))

    active_trades = []  # List to store tuples of (exit_time, symbol)

    for idx in sort_idx:
        entry_t = entry_timestamps[idx]
        exit_t = exit_timestamps[idx]
        sym = symbols[idx]

        # Remove trades that have already exited before or exactly at current entry time
        active_trades = [t for t in active_trades if t[0] > entry_t]

        # Check symbol constraint
        symbol_already_active = any(t[1] == sym for t in active_trades)

        # Check global constraint
        global_cap_reached = len(active_trades) >= max_global_concurrent

        if not symbol_already_active and not global_cap_reached:
            mask[idx] = True
            active_trades.append((exit_t, sym))

    return mask


def _select_topk_non_concurrent(
    scores: np.ndarray,
    k: int,
    entry_timestamps: np.ndarray,
    exit_timestamps: np.ndarray,
    symbols: np.ndarray,
    max_global_concurrent: int = 3,
) -> np.ndarray:
    n = len(scores)
    if n <= 0:
        return np.array([], dtype=np.int32)
    k = max(1, min(int(k), n))
    if len(entry_timestamps) != n or len(exit_timestamps) != n or len(symbols) != n:
        return np.argsort(np.asarray(scores, dtype=np.float64), kind="stable")[-k:]
    mask = _fast_concurrency_mask(
        np.asarray(entry_timestamps),
        np.asarray(exit_timestamps),
        np.asarray(symbols).astype(str),
        np.asarray(scores, dtype=np.float64),
        max_global_concurrent=max_global_concurrent,
    )
    selected = np.flatnonzero(mask)
    if selected.size == 0:
        return np.argsort(np.asarray(scores, dtype=np.float64), kind="stable")[-k:]
    selected_scores = np.asarray(scores, dtype=np.float64)[selected]
    order = np.argsort(selected_scores, kind="stable")[::-1]
    selected_sorted = selected[order]
    if selected_sorted.size >= k:
        return np.asarray(selected_sorted[:k], dtype=np.int32)
    chosen = set(int(i) for i in selected_sorted.tolist())
    out: List[int] = [int(i) for i in selected_sorted.tolist()]
    global_rank = np.argsort(np.asarray(scores, dtype=np.float64), kind="stable")[::-1]
    for idx in global_rank:
        i = int(idx)
        if i not in chosen:
            out.append(i)
            chosen.add(i)
        if len(out) >= k:
            break
    return np.asarray(out[:k], dtype=np.int32)


def run_policy_optimisation(
    data_root: str,
    run_id: str,
    sizer_results: Optional[Dict[str, Any]] = None,
    holdout_frac: float = 0.30,
    cost_pct: float = 0.003,
    use_offset_optimiser: bool = False,
    stage_view: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    tprint("POLICY OPTIMISER START")
    if use_offset_optimiser and not POLICY_OPTIMISER_OFFSET_LIMITER_ENABLED:
        tprint("Policy optimiser: offset limiter disabled for this run.")
    use_offset_optimiser = bool(
        use_offset_optimiser and POLICY_OPTIMISER_OFFSET_LIMITER_ENABLED
    )
    selected_candidates = _load_strategy_candidates(data_root, run_id)
    if not selected_candidates:
        tprint(
            "No simple_position_sizer candidates passed policy gates; "
            "skipping policy optimisation."
        )
        return {}
    max_candidates: Optional[int] = None
    max_candidates_env = os.getenv("EPM_POLICY_MAX_STRATEGY_IDS", "").strip()
    if max_candidates_env:
        try:
            max_candidates = max(1, int(max_candidates_env))
            tprint(
                "Policy optimiser: limiting successfully optimised candidates "
                f"to {max_candidates} via EPM_POLICY_MAX_STRATEGY_IDS."
            )
        except ValueError:
            tprint(
                "WARNING: invalid EPM_POLICY_MAX_STRATEGY_IDS="
                f"{max_candidates_env!r}; ignoring."
            )

    base_oofs = load_base_oof_predictions(data_root, run_id)
    meta_oofs = load_meta_oof_predictions(data_root, run_id)
    if stage_view:
        _unfiltered_meta_oofs = meta_oofs
        filtered_meta: Dict[str, pd.DataFrame] = {}
        for key, mdf in meta_oofs.items():
            mdf_f = _filter_df_by_stage_view(mdf, stage_view)
            if not mdf_f.empty:
                filtered_meta[key] = mdf_f
        meta_oofs = filtered_meta
        if not meta_oofs:
            tprint(
                "Policy optimiser stage filtering removed all meta OOF rows; "
                "falling back to unfiltered OOF rows for this run."
            )
            meta_oofs = _unfiltered_meta_oofs

    sizer_scores = _load_sizer_oof_scores(data_root, run_id)
    _unfiltered_sizer_scores = sizer_scores
    sizer_scores = _filter_df_by_stage_view(sizer_scores, stage_view)
    if sizer_scores.empty:
        tprint(
            "Policy optimiser stage filtering removed all sizer OOF rows; "
            "falling back to unfiltered sizer OOF rows for this run."
        )
        sizer_scores = _unfiltered_sizer_scores
    if sizer_scores.empty:
        tprint("No sizer OOF rows available for policy optimisation.")
        return {}
    strategy_rows: List[Dict[str, Any]] = []
    for selected in selected_candidates:
        if max_candidates is not None and len(strategy_rows) >= max_candidates:
            break
        try:
            key, context_oof, context_source = _resolve_policy_context_oof(
                str(selected["strategy_id"]), meta_oofs, base_oofs
            )
        except KeyError:
            tprint(
                f"Strategy {selected['strategy_id']} not found in base/meta OOF context; "
                "skipping policy optimisation for this candidate."
            )
            continue

        outcomes = load_trade_outcomes(data_root, run_id, context_oof)
        unfiltered_outcomes = outcomes
        outcomes = _filter_df_by_stage_view(outcomes, stage_view)
        if (
            outcomes.empty
            and unfiltered_outcomes is not None
            and not unfiltered_outcomes.empty
        ):
            tprint(
                f"Policy optimiser stage filtering removed all trade outcomes for "
                f"{selected['strategy_id']}; falling back to unfiltered outcomes."
            )
            outcomes = unfiltered_outcomes
        if outcomes.empty:
            continue
        outcomes = _attach_sizer_score(outcomes, sizer_scores, selected["strategy_id"])
        if "sizer_score" not in outcomes.columns:
            raise RuntimeError(
                f"Could not attach simple_position_sizer scores for {selected['strategy_id']}"
            )
        pre_sizer_rows = len(outcomes)
        sizer_score = pd.to_numeric(outcomes["sizer_score"], errors="coerce")
        finite_sizer_mask = np.isfinite(np.asarray(sizer_score, dtype=np.float64))
        outcomes = outcomes.loc[finite_sizer_mask].copy()
        if outcomes.empty:
            tprint(
                f"Strategy {selected['strategy_id']} has no finite simple-sizer "
                "OOF rows after timestamp/symbol alignment; skipping."
            )
            continue
        outcomes["sizer_score"] = np.asarray(
            sizer_score.loc[finite_sizer_mask], dtype=np.float32
        )
        selected_side = _strategy_side(selected)
        if selected_side == "unknown" and "is_long" in outcomes.columns:
            is_long_vals = np.asarray(outcomes["is_long"].values, dtype=np.float64)
            finite_long = np.isfinite(is_long_vals)
            if finite_long.any():
                selected_side = (
                    "long"
                    if float(np.nanmean(is_long_vals[finite_long])) >= 0.5
                    else "short"
                )
        if selected_side in {"long", "short"}:
            selected["side"] = selected_side
        tprint(
            f"Strategy {selected['strategy_id']}: using {len(outcomes)}/"
            f"{pre_sizer_rows} finite simple-sizer OOF rows for policy optimisation "
            f"(context={context_source}:{key})."
        )

        conf, conf_name = _resolve_selection_score(outcomes)
        upstream_sizer_frac = resolve_optimised_selection_frac(
            data_root=data_root,
            run_id=run_id,
            selected=selected,
        )
        frac = min(
            float(upstream_sizer_frac),
            float(POLICY_OPTIMISER_CANDIDATE_FRAC),
        )
        frac = max(1.0 / float(max(1, len(conf))), min(1.0, frac))
        k = max(1, int(len(conf) * frac))
        assessment_frac = min(frac, float(POLICY_OPTIMISER_ASSESSMENT_FRAC))
        assessment_frac = max(1.0 / float(max(1, len(conf))), min(1.0, assessment_frac))
        k_assess = max(1, int(len(conf) * assessment_frac))
        k_diag = max(1, int(len(conf) * POLICY_OPTIMISER_DIAGNOSTIC_FRAC))
        entry_ts = pd.to_datetime(
            outcomes.get("timestamp", pd.Series(np.arange(len(outcomes)))),
            utc=True,
            errors="coerce",
        ).view("int64")
        entry_ts = np.where(entry_ts > 0, entry_ts // 1_000_000, 0).astype(np.int64)
        exit_bars = np.maximum(
            1,
            np.asarray(
                outcomes.get("exit_bar", pd.Series(np.ones(len(outcomes)))),
                dtype=np.int64,
            ),
        )
        exit_ts = entry_ts + exit_bars * 3_600_000
        selected_idx = _select_topk_non_concurrent(
            scores=conf,
            k=k,
            entry_timestamps=entry_ts,
            exit_timestamps=exit_ts,
            symbols=np.asarray(
                outcomes.get("symbol", pd.Series([""] * len(outcomes))).astype(str)
            ),
            max_global_concurrent=3,
        )
        assessment_idx = _select_topk_non_concurrent(
            scores=conf,
            k=k_assess,
            entry_timestamps=entry_ts,
            exit_timestamps=exit_ts,
            symbols=np.asarray(
                outcomes.get("symbol", pd.Series([""] * len(outcomes))).astype(str)
            ),
            max_global_concurrent=3,
        )
        diagnostic_idx = _select_topk_non_concurrent(
            scores=conf,
            k=k_diag,
            entry_timestamps=entry_ts,
            exit_timestamps=exit_ts,
            symbols=np.asarray(
                outcomes.get("symbol", pd.Series([""] * len(outcomes))).astype(str)
            ),
            max_global_concurrent=3,
        )
        policy_idx = np.asarray(selected_idx, dtype=np.int64)
        top1_diagnostics = _top_slice_opportunity_diagnostics(
            outcomes=outcomes,
            selected_idx=diagnostic_idx,
            frac=POLICY_OPTIMISER_DIAGNOSTIC_FRAC,
        )
        top5_diagnostics = _top_slice_opportunity_diagnostics(
            outcomes=outcomes,
            selected_idx=assessment_idx,
            frac=assessment_frac,
        )
        tprint(
            f"Policy optimiser selection: upstream_sizer_frac={upstream_sizer_frac:.3f}, "
            f"candidate_frac={frac:.3f}, assessment_frac={assessment_frac:.3f}, "
            f"top1_trades_per_active_day={top1_diagnostics['avg_trades_per_active_day']:.2f}, "
            f"top1_trades_per_calendar_day={top1_diagnostics['avg_trades_per_calendar_day']:.2f}, "
            f"top1_avg_holding_hours={top1_diagnostics['avg_holding_hours']:.2f}"
        )

        sizer_stub = {
            "best_simple_score_": conf,
            "best_simple_score_name_": "Ridge_Head_Sizer",
            "ridge_profit_proxy_table_": pd.DataFrame(
                [
                    {
                        "selection_frac": frac,
                        "wallet_min": 0.05,
                        "wallet_max": 0.15,
                        "sizing_mode": "linear",
                        "is_optimal": True,
                    }
                ]
            ),
            "opt_rets_": np.asarray(
                outcomes.get("return", pd.Series(np.zeros(len(outcomes)))).values,
                dtype=np.float32,
            )[selected_idx],
            "opt_ts_": np.asarray(
                outcomes.get("timestamp", pd.Series(np.arange(len(outcomes)))).values
            )[selected_idx],
        }

        path_bundle = build_policy_path_state_bundle(
            outcomes, selected_idx=selected_idx
        )
        path_matrices = _pack_future_path_matrices(outcomes, selected_idx=selected_idx)

        _label_tp_sl = float(
            np.nanmedian(path_bundle.get("label_tp_sl_ratio", np.array([3.0])))
        )
        _has_label_policy = bool(path_bundle.get("has_label_policy", 0.0) > 0.5)

        base_rets = np.asarray(
            outcomes.get("return", pd.Series(np.zeros(len(outcomes)))).values,
            dtype=np.float32,
        )[selected_idx]
        executed_mask = np.ones(len(base_rets), dtype=bool)

        if use_offset_optimiser:
            offset_result = run_simple_offset_generator_from_sizer(
                sizer_results=sizer_stub, trade_outcomes=outcomes, cost_pct=cost_pct
            )
            policy_idx = np.asarray(
                offset_result.get("above_threshold_idx", selected_idx), dtype=np.int64
            )
            base_rets = np.asarray(
                outcomes.get("return", pd.Series(np.zeros(len(outcomes)))).values,
                dtype=np.float32,
            )[offset_result.get("above_threshold_idx")]
            executed_mask = np.asarray(
                offset_result.get("executed", np.ones(len(base_rets), dtype=bool)),
                dtype=bool,
            )

        ts = pd.to_datetime(
            np.asarray(path_bundle.get("timestamps", np.arange(len(base_rets)))),
            utc=True,
            errors="coerce",
        )
        order = np.argsort(ts.view("int64"))
        n = len(base_rets)
        assessment_set = set(int(i) for i in np.asarray(assessment_idx).tolist())
        assessment_mask = np.asarray(
            [int(i) in assessment_set for i in np.asarray(policy_idx).tolist()],
            dtype=bool,
        )
        if assessment_mask.shape[0] != n or not assessment_mask.any():
            assessment_mask = np.zeros(n, dtype=bool)
            local_scores = np.asarray(
                path_bundle.get("confidence", np.zeros(n)), dtype=np.float64
            )
            if local_scores.shape[0] != n:
                local_scores = np.zeros(n, dtype=np.float64)
            local_k = max(1, min(n, int(len(conf) * POLICY_OPTIMISER_ASSESSMENT_FRAC)))
            local_order = np.argsort(local_scores, kind="stable")[::-1]
            assessment_mask[local_order[:local_k]] = True
        tprint(
            f"Policy assessment slice: {int(np.sum(assessment_mask))}/{n} "
            f"selected trades ({assessment_frac:.1%} of source opportunities)."
        )

        # USE UPSTREAM TRAIN/VAL FROM GLOBAL PARTITIONER.
        # Policy optimisation is downstream of the sizer OOF handoff, so we do
        # not reserve an additional local holdout here. All available candidate
        # rows are eligible for train/validation splitting; actual trading and
        # reporting still use the configured top-k slices.
        n_train_val = int(n)
        n_train = int(n_train_val * 0.70)  # 70% of train_val for training params
        n_val = n_train_val - n_train

        n_train = max(1, n_train)
        n_val = max(1, n_val)

        train_mask = np.zeros(n, dtype=bool)
        train_mask[order[:n_train]] = True

        val_mask = np.zeros(n, dtype=bool)
        val_mask[order[n_train : n_train + n_val]] = True

        ae_fit = _robust_fit(path_bundle["AE_vel"], train_mask)
        pr_fit = _robust_fit(path_bundle["pressure"], train_mask)
        pq_fit = _robust_fit(path_bundle["path_quality"], train_mask)
        pg_fit = _robust_fit(path_bundle["progress_per_bar"], train_mask)
        tr_fit = _robust_fit(path_bundle["trend"], train_mask)
        as_fit = _robust_fit(path_bundle["asym_raw"], train_mask)
        ch_fit = _robust_fit(path_bundle["choppiness"], train_mask)

        asym_z = _robust_apply(path_bundle["asym_raw"], as_fit)
        asym_01 = (asym_z - float(np.nanmin(asym_z[train_mask]))) / (
            float(np.nanmax(asym_z[train_mask]) - np.nanmin(asym_z[train_mask])) + EPS
        )
        context = build_replay_context(
            returns=base_rets,
            mfe_ret=path_bundle["mfe_ret"],
            mae_ret=path_bundle["mae_ret"],
            delta_mfe=path_bundle.get("delta_mfe"),
            delta_mae=path_bundle.get("delta_mae"),
            bars_since_entry=path_bundle["bars_since_entry"],
            barrier_pct=path_bundle["barrier_pct"],
            confidence=path_bundle["confidence"],
            trend=_robust_apply(path_bundle["trend"], tr_fit),
            choppiness=_robust_apply(path_bundle["choppiness"], ch_fit),
            asym=np.clip(asym_01, 0.0, 1.0).astype(np.float32),
        )
        context["AE_vel"] = _robust_apply(path_bundle["AE_vel"], ae_fit)
        context["pressure"] = _robust_apply(path_bundle["pressure"], pr_fit)
        context["path_quality"] = _robust_apply(path_bundle["path_quality"], pq_fit)
        context["progress_per_bar"] = _robust_apply(
            path_bundle["progress_per_bar"], pg_fit
        )
        context["p_tp"] = path_bundle.get("p_tp", np.full(len(base_rets), np.nan))
        context["p_sl"] = path_bundle.get("p_sl", np.full(len(base_rets), np.nan))
        context_ts = pd.to_datetime(
            np.asarray(path_bundle.get("timestamps", np.arange(len(base_rets)))),
            utc=True,
            errors="coerce",
        )
        context["timestamps_ms"] = (
            np.asarray(context_ts.view("int64"), dtype=np.int64) // 1_000_000
        )
        context.update(path_matrices)

        # CRITICAL FIX: Prioritize sizer's profitable TP/SL geometry over label-derived values
        # The sizer already found profitable parameters - don't overwrite with potentially different label geometry
        sizer_tp_mult = float(selected.get("tp_mult", 1.0))
        sizer_sl_mult = float(selected.get("sl_mult", 1.0))
        has_sizer_tp = "tp_mult" in selected
        has_sizer_sl = "sl_mult" in selected
        label_tp_mult = float(_label_tp_sl) if _has_label_policy else sizer_tp_mult
        label_sl_mult = 1.0

        tp_mult_to_use = sizer_tp_mult if has_sizer_tp else label_tp_mult
        sl_mult_to_use = sizer_sl_mult if has_sizer_sl else label_sl_mult

        fixed = {
            "strategy_id": selected["strategy_id"],
            "tp_mult": tp_mult_to_use,
            "sl_mult": sl_mult_to_use,
            "k_recent": 3,
            "K_early": 3,
            "lambda_path": 1.0,
            "a1": 0.5,
            "a2": 0.5,
            "b1": 0.5,
            "b2": 0.5,
            "score_weight_trend": 1.0,
            "score_weight_asym": 0.6,
            "score_weight_choppiness": 0.9,
            # DISABLE all advanced features for basic run
            "enable_early_exit": False,
            "enable_compression": False,
            "enable_barrier_conf": False,
            "enable_trailing": False,  # BASIC RUN: Simple TP/SL only
        }

        best = _sequential_optimise(
            base_rets,
            context,
            train_mask,
            val_mask,
            fixed=fixed,
            cost_pct=cost_pct,
            executed_mask=executed_mask,
        )
        best["side"] = selected.get("side", selected_side)
        source_model = str(selected.get("model") or selected.get("pipeline") or "")
        if not source_model:
            score_source = str(selected.get("winner_score_source") or "")
            if "MetaClfOnly" in score_source or "meta_clf" in score_source:
                source_model = "meta_clf_only"
        best["source_model"] = source_model
        best["source_artifact"] = selected.get("source_artifact", "")
        best["selected_key"] = key
        best["selection_score"] = conf_name

        # Get sizer's reported wallet_pnl early for logging and skip logic
        sizer_wallet_pnl = float(selected.get("wallet_pnl", 0.0))

        # Compute baseline sizes once using the same sizing formula as the
        # policy optimiser. Diagnostics below keep unsized/simple-sizer and
        # wallet-sized bases separate.
        baseline_sizes = np.ones(len(base_rets), dtype=np.float32)
        baseline_conf = context.get("confidence")
        if baseline_conf is not None and len(baseline_conf) == len(base_rets):
            ranks = np.argsort(np.argsort(baseline_conf.astype(np.float64)))
            pcts = ranks / float(max(1, len(ranks) - 1))
            baseline_sizes = (0.05 + (0.15 - 0.05) * pcts).astype(np.float32)

        simple_sizer_equivalent_assess = _metric_score(
            base_rets,
            cost_pct=cost_pct,
            already_net=False,
            already_sized=False,
            sizes=np.ones(len(base_rets), dtype=np.float32),
            executed_mask=executed_mask,
            ts_vals=(
                context.get("timestamps_ms", np.arange(len(base_rets)))
                if "timestamps_ms" in context
                else None
            ),
            scores=baseline_conf if baseline_conf is not None else None,
        )
        simple_sizer_equivalent_assess["source"] = "base_rets_flat_fee_unsized"

        has_future_paths = all(
            col in context
            for col in ("future_opens", "future_highs", "future_lows", "future_closes")
        )
        baseline_rets_raw = None
        path_assess_sized: Optional[Dict[str, float]] = None
        path_assess_unsized: Optional[Dict[str, float]] = None
        path_delta_mean: Optional[float] = None
        if has_future_paths:
            baseline_rets_raw = _simulate_tpsl_from_paths_unified(
                context,
                tp_mult=tp_mult_to_use,
                sl_mult=sl_mult_to_use,
                enable_trailing=False,
                is_baseline=True,
            )
            if baseline_rets_raw is not None:
                path_assess_sized = _metric_score(
                    baseline_rets_raw,
                    cost_pct=cost_pct,
                    already_net=False,
                    already_sized=False,
                    sizes=baseline_sizes,
                    executed_mask=executed_mask,
                    ts_vals=(
                        context.get("timestamps_ms", np.arange(len(baseline_rets_raw)))
                        if "timestamps_ms" in context
                        else None
                    ),
                    scores=(
                        context.get("confidence")
                        if context.get("confidence") is not None
                        else None
                    ),
                )
                path_assess_unsized = _metric_score(
                    baseline_rets_raw,
                    cost_pct=cost_pct,
                    already_net=False,
                    already_sized=False,
                    sizes=np.ones(len(baseline_rets_raw), dtype=np.float32),
                    executed_mask=executed_mask,
                    ts_vals=(
                        context.get("timestamps_ms", np.arange(len(baseline_rets_raw)))
                        if "timestamps_ms" in context
                        else None
                    ),
                    scores=(
                        context.get("confidence")
                        if context.get("confidence") is not None
                        else None
                    ),
                )
                pre_capped_pnl = (
                    float(np.mean(base_rets) * len(base_rets))
                    if len(base_rets) > 0
                    else 0.0
                )
                path_delta_mean = float(
                    np.nanmean(np.asarray(baseline_rets_raw, dtype=np.float64))
                    - np.nanmean(np.asarray(base_rets, dtype=np.float64))
                )
                tprint(
                    "Path-sim baseline diagnostic: "
                    f"sized_net_pnl={path_assess_sized.get('net_pnl', 0.0):.6f}, "
                    f"unsized_net_pnl={path_assess_unsized.get('net_pnl', 0.0):.6f}, "
                    f"mean_return_delta_vs_sizer={path_delta_mean:.6f} "
                    f"(pre-capped={pre_capped_pnl:.6f}, sizer wallet_pnl={sizer_wallet_pnl:.6f})"
                )

        # Apply sizing BEFORE metrics (consistent with policy simulation)
        base_rets_sized = base_rets * baseline_sizes

        wallet_sized_baseline_assess = _metric_score(
            base_rets_sized,
            cost_pct=cost_pct,
            already_net=False,
            already_sized=True,
            sizes=baseline_sizes,
            executed_mask=executed_mask,
            ts_vals=(
                context.get("timestamps_ms", np.arange(len(base_rets)))
                if "timestamps_ms" in context
                else None
            ),
            scores=baseline_conf if baseline_conf is not None else None,
        )
        tprint(
            "Wallet-sized baseline diagnostic: "
            f"net_pnl={wallet_sized_baseline_assess.get('net_pnl', 0.0):.6f}, "
            f"sizer wallet_pnl={sizer_wallet_pnl:.6f}"
        )

        # Only optimize strategies with positive sizer-reported PnL
        if sizer_wallet_pnl <= 0:
            tprint(
                f"Strategy {selected['strategy_id'][:40]} has non-positive sizer PnL ({sizer_wallet_pnl:.4f}); skipping optimization."
            )
            continue

        # APPLY POSITION SIZING DURING SIMULATION (not post-hoc)
        # Compute sizes upfront so they affect trade-level PnL during simulation
        def _compute_position_sizes(
            scores_arr: np.ndarray, size_power: float = 1.0
        ) -> np.ndarray:
            if scores_arr is None or len(scores_arr) == 0:
                return np.ones(len(base_rets), dtype=np.float32)
            ranks = np.argsort(np.argsort(scores_arr.astype(np.float64)))
            pcts = ranks / float(max(1, len(scores_arr) - 1))
            return (0.05 + (0.15 - 0.05) * (pcts**size_power)).astype(np.float32)

        size_power = float(best.get("size_power", 1.0))
        precomputed_sizes = _compute_position_sizes(
            context.get("confidence"), size_power
        )

        # Run simulation with position sizing applied inside the path replay. This
        # keeps gross/net per-trade metrics on the same basis as the optimiser.
        if has_future_paths:
            final_rets_assess = _simulate_barwise_path_policy(
                base_rets, context, best, sizes=precomputed_sizes
            )
            if final_rets_assess is None:
                base_rets_sized = base_rets * precomputed_sizes
                final_rets_assess = replay_exit_policy(base_rets_sized, context, best)
        else:
            base_rets_sized = base_rets * precomputed_sizes
            final_rets_assess = replay_exit_policy(base_rets_sized, context, best)

        assessment_val_mask = val_mask & assessment_mask
        if not assessment_val_mask.any():
            assessment_val_mask = assessment_mask.copy()
        if not assessment_val_mask.any():
            assessment_val_mask = val_mask.copy()
        assessment_full_mask = assessment_mask.copy()
        if not assessment_full_mask.any():
            assessment_full_mask = np.ones(len(base_rets), dtype=bool)

        # Compute metrics on the top-5% assessment slice for fair comparison.
        if baseline_rets_raw is not None:
            baseline_val_assess = _metric_score(
                baseline_rets_raw[assessment_val_mask],
                cost_pct=cost_pct,
                already_net=False,
                already_sized=False,
                sizes=baseline_sizes[assessment_val_mask],
                executed_mask=(
                    executed_mask[assessment_val_mask]
                    if executed_mask is not None
                    else None
                ),
                ts_vals=(
                    context.get("timestamps_ms", np.arange(len(baseline_rets_raw)))[
                        assessment_val_mask
                    ]
                    if "timestamps_ms" in context
                    else None
                ),
                scores=(
                    context.get("confidence")[assessment_val_mask]
                    if context.get("confidence") is not None
                    else None
                ),
            )
        else:
            # Sizing already applied to base_rets during baseline computation
            baseline_val_assess = _metric_score(
                base_rets_sized[assessment_val_mask],
                cost_pct=cost_pct,
                already_net=False,
                already_sized=True,
                sizes=baseline_sizes[assessment_val_mask],
                executed_mask=(
                    executed_mask[assessment_val_mask]
                    if executed_mask is not None
                    else None
                ),
                ts_vals=(
                    context.get("timestamps_ms", np.arange(len(base_rets)))[
                        assessment_val_mask
                    ]
                    if "timestamps_ms" in context
                    else None
                ),
                scores=(
                    context.get("confidence")[assessment_val_mask]
                    if context.get("confidence") is not None
                    else None
                ),
            )

        final_val_assess = _metric_score(
            final_rets_assess[assessment_val_mask],
            cost_pct=cost_pct,
            already_net=False,
            already_sized=True,
            sizes=precomputed_sizes[assessment_val_mask],
            executed_mask=(
                executed_mask[assessment_val_mask]
                if executed_mask is not None
                else None
            ),
            ts_vals=(
                context.get("timestamps_ms", np.arange(len(final_rets_assess)))[
                    assessment_val_mask
                ]
                if "timestamps_ms" in context
                else None
            ),
            scores=(
                context.get("confidence")[assessment_val_mask]
                if context.get("confidence") is not None
                else None
            ),
        )

        # Also compute full candidate-slice and top-5 assessment-slice metrics.
        final_candidate_assess = _metric_score(
            final_rets_assess,
            cost_pct=cost_pct,
            already_net=False,
            already_sized=True,
            sizes=precomputed_sizes,
            executed_mask=executed_mask,
            ts_vals=(
                context.get("timestamps_ms", np.arange(len(final_rets_assess)))
                if "timestamps_ms" in context
                else None
            ),
            scores=(
                context.get("confidence")
                if context.get("confidence") is not None
                else None
            ),
        )
        if baseline_rets_raw is not None:
            baseline_candidate_assess = _metric_score(
                baseline_rets_raw,
                cost_pct=cost_pct,
                already_net=False,
                already_sized=False,
                sizes=baseline_sizes,
                executed_mask=executed_mask,
                ts_vals=(
                    context.get("timestamps_ms", np.arange(len(baseline_rets_raw)))
                    if "timestamps_ms" in context
                    else None
                ),
                scores=(
                    context.get("confidence")
                    if context.get("confidence") is not None
                    else None
                ),
            )
            baseline_assess = _metric_score(
                baseline_rets_raw[assessment_full_mask],
                cost_pct=cost_pct,
                already_net=False,
                already_sized=False,
                sizes=baseline_sizes[assessment_full_mask],
                executed_mask=(
                    executed_mask[assessment_full_mask]
                    if executed_mask is not None
                    else None
                ),
                ts_vals=(
                    context.get("timestamps_ms", np.arange(len(baseline_rets_raw)))[
                        assessment_full_mask
                    ]
                    if "timestamps_ms" in context
                    else None
                ),
                scores=(
                    context.get("confidence")[assessment_full_mask]
                    if context.get("confidence") is not None
                    else None
                ),
            )
        else:
            baseline_candidate_assess = dict(wallet_sized_baseline_assess)
        baseline_assess = _metric_score(
            base_rets_sized[assessment_full_mask],
            cost_pct=cost_pct,
            already_net=False,
            already_sized=True,
            sizes=baseline_sizes[assessment_full_mask],
            executed_mask=(
                executed_mask[assessment_full_mask]
                if executed_mask is not None
                else None
            ),
            ts_vals=(
                context.get("timestamps_ms", np.arange(len(base_rets_sized)))[
                    assessment_full_mask
                ]
                if "timestamps_ms" in context
                else None
            ),
            scores=(
                context.get("confidence")[assessment_full_mask]
                if context.get("confidence") is not None
                else None
            ),
        )
        if baseline_rets_raw is not None:
            baseline_assess = _metric_score(
                baseline_rets_raw[assessment_full_mask],
                cost_pct=cost_pct,
                already_net=False,
                already_sized=False,
                sizes=baseline_sizes[assessment_full_mask],
                executed_mask=(
                    executed_mask[assessment_full_mask]
                    if executed_mask is not None
                    else None
                ),
                ts_vals=(
                    context.get("timestamps_ms", np.arange(len(baseline_rets_raw)))[
                        assessment_full_mask
                    ]
                    if "timestamps_ms" in context
                    else None
                ),
                scores=(
                    context.get("confidence")[assessment_full_mask]
                    if context.get("confidence") is not None
                    else None
                ),
            )
        final_assess = _metric_score(
            final_rets_assess[assessment_full_mask],
            cost_pct=cost_pct,
            already_net=False,
            already_sized=True,
            sizes=precomputed_sizes[assessment_full_mask],
            executed_mask=(
                executed_mask[assessment_full_mask]
                if executed_mask is not None
                else None
            ),
            ts_vals=(
                context.get("timestamps_ms", np.arange(len(final_rets_assess)))[
                    assessment_full_mask
                ]
                if "timestamps_ms" in context
                else None
            ),
            scores=(
                context.get("confidence")[assessment_full_mask]
                if context.get("confidence") is not None
                else None
            ),
        )

        basic_policy_metrics = dict(baseline_assess)
        basic_policy_metrics_val = dict(baseline_val_assess)
        advanced_policy_metrics = dict(final_assess)
        advanced_policy_metrics_val = dict(final_val_assess)
        metric_scores = (
            context.get("confidence")
            if context.get("confidence") is not None
            else np.asarray(conf, dtype=np.float32)[policy_idx]
        )
        metric_ts = (
            context.get("timestamps_ms", np.arange(len(final_rets_assess)))
            if "timestamps_ms" in context
            else None
        )
        if baseline_rets_raw is not None:
            baseline_top_fraction_metrics = _top_fraction_metrics(
                baseline_rets_raw,
                cost_pct=cost_pct,
                already_net=False,
                already_sized=False,
                sizes=baseline_sizes,
                executed_mask=executed_mask,
                ts_vals=metric_ts,
                scores=metric_scores,
            )
        else:
            baseline_top_fraction_metrics = _top_fraction_metrics(
                base_rets_sized,
                cost_pct=cost_pct,
                already_net=False,
                already_sized=True,
                sizes=baseline_sizes,
                executed_mask=executed_mask,
                ts_vals=metric_ts,
                scores=metric_scores,
            )
        final_top_fraction_metrics = _top_fraction_metrics(
            final_rets_assess,
            cost_pct=cost_pct,
            already_net=False,
            already_sized=True,
            sizes=precomputed_sizes,
            executed_mask=executed_mask,
            ts_vals=metric_ts,
            scores=metric_scores,
        )
        best_candidate_metrics = {
            "baseline_candidate_slice": baseline_candidate_assess,
            "advanced_candidate_slice": final_candidate_assess,
            "candidate_frac": float(frac),
        }

        # Compare on VALIDATION SET (fair - same period used for optimization)
        baseline_val_pnl = float(baseline_val_assess.get("net_pnl", 0.0))
        final_val_pnl = float(final_val_assess.get("net_pnl", 0.0))

        tprint(
            f"Fair comparison (validation set): baseline={baseline_val_pnl:.4f}, policy={final_val_pnl:.4f}, "
            f"delta={final_val_pnl - baseline_val_pnl:+.4f}"
        )

        # Only revert if policy degraded performance on validation set
        if final_val_pnl < baseline_val_pnl - 0.001:  # Small tolerance for noise
            if has_future_paths:
                logger.warning(
                    "Policy optimisation degraded validation net_pnl "
                    f"({final_val_pnl:.6f} < {baseline_val_pnl:.6f}); "
                    "reverting to baseline params."
                )
                # Preserve param_history for diagnostics even when reverting
                param_history = best.get("_param_history_", [])
                best = dict(fixed)
                best["_param_history_"] = param_history  # Keep for analysis
                best["_reverted_to_baseline_"] = True  # Flag for diagnostics
                final_rets_assess = np.asarray(base_rets_sized, dtype=np.float32)
                final_assess = dict(baseline_assess)
                final_val_assess = dict(baseline_val_assess)
                final_candidate_assess = dict(baseline_candidate_assess)
                final_top_fraction_metrics = dict(baseline_top_fraction_metrics)
                advanced_policy_metrics = dict(final_assess)
                advanced_policy_metrics_val = dict(baseline_val_assess)
            else:
                # Without future paths, the comparison is unfair - log but don't revert
                logger.info(
                    "Policy optimisation result lower than pre-capped baseline "
                    f"({float(final_assess.get('net_pnl', 0.0)):.6f} < "
                    f"{float(baseline_assess.get('net_pnl', 0.0)):.6f}), "
                    "but future paths unavailable for fair comparison - keeping optimised params."
                )

        best_candidate_metrics = {
            "baseline_candidate_slice": baseline_candidate_assess,
            "advanced_candidate_slice": final_candidate_assess,
            "candidate_frac": float(frac),
        }

        best["metrics_baseline"] = baseline_assess
        best["metrics_baseline_val"] = baseline_val_assess
        best["metrics_final"] = final_assess
        best["metrics_final_val"] = final_val_assess
        best["metrics_top_fractions"] = {
            "baseline": baseline_top_fraction_metrics,
            "final": final_top_fraction_metrics,
        }
        best["metrics_wallet_sized_baseline_diagnostic"] = wallet_sized_baseline_assess
        best["metrics_simple_sizer_equivalent_baseline"] = (
            simple_sizer_equivalent_assess
        )
        if path_assess_sized is not None:
            best["metrics_path_sized_baseline_diagnostic"] = path_assess_sized
        if path_assess_unsized is not None:
            best["metrics_path_unsized_baseline_diagnostic"] = path_assess_unsized
        if path_delta_mean is not None:
            best["path_replay_mean_return_delta_vs_sizer"] = float(path_delta_mean)
        best["metrics_sizer_reference"] = {
            "source": "simple_position_sizer.strategy_params",
            "net_pnl": _safe_float(selected.get("net_pnl"), 0.0),
            "wallet_pnl": _safe_float(selected.get("wallet_pnl"), 0.0),
            "hit_rate": _safe_float(selected.get("hit_rate"), 0.0),
            "profit_factor": _safe_float(selected.get("profit_factor"), 0.0),
            "monthly_sortino": _safe_float(selected.get("monthly_sortino"), 0.0),
            "max_drawdown": _safe_float(selected.get("max_drawdown"), 0.0),
            "net_pnl_per_trade_pct": _safe_float(
                selected.get("net_pnl_per_trade_pct"), 0.0
            ),
            "selection_frac": _safe_float(selected.get("selection_frac"), 0.0),
            "threshold_pct": _safe_float(selected.get("threshold_pct"), 0.0),
        }
        best["metrics_candidate_top15"] = best_candidate_metrics
        best["policy_passes"] = {
            "basic_tpsl_only": {
                "metrics": basic_policy_metrics,
                "metrics_val": basic_policy_metrics_val,
                "flags": {
                    "tp_mult": float(tp_mult_to_use),
                    "sl_mult": float(sl_mult_to_use),
                    "enable_trailing": False,
                    "enable_early_exit": False,
                    "enable_compression": False,
                    "enable_barrier_conf": False,
                },
            },
            "advanced_policy_final": {
                "metrics": advanced_policy_metrics,
                "metrics_val": advanced_policy_metrics_val,
                "flags": {
                    "tp_mult": float(best.get("tp_mult", tp_mult_to_use)),
                    "sl_mult": float(best.get("sl_mult", sl_mult_to_use)),
                    "enable_trailing": bool(best.get("enable_trailing", False)),
                    "enable_early_exit": bool(best.get("enable_early_exit", False)),
                    "enable_compression": bool(best.get("enable_compression", False)),
                    "enable_barrier_conf": bool(best.get("enable_barrier_conf", False)),
                },
            },
        }
        # Use VALIDATION SET metrics for fair delta comparison
        best["baseline_net_pnl"] = float(baseline_val_assess.get("net_pnl", 0.0))
        best["final_net_pnl"] = float(final_val_assess.get("net_pnl", 0.0))
        best["net_pnl_delta"] = float(best["final_net_pnl"] - best["baseline_net_pnl"])
        # Also store full-set metrics for reference
        best["baseline_net_pnl_full"] = float(baseline_assess.get("net_pnl", 0.0))
        best["final_net_pnl_full"] = float(final_assess.get("net_pnl", 0.0))
        best["net_pnl_delta_full"] = float(
            best["final_net_pnl_full"] - best["baseline_net_pnl_full"]
        )
        # Store selection parameters for comparison with sizer
        best["selection_frac"] = frac
        best["upstream_sizer_selection_frac"] = upstream_sizer_frac
        best["policy_candidate_frac"] = frac
        best["policy_assessment_frac"] = min(frac, POLICY_OPTIMISER_ASSESSMENT_FRAC)
        best["policy_diagnostic_frac"] = POLICY_OPTIMISER_DIAGNOSTIC_FRAC
        best["upstream_threshold_pct"] = float(selected.get("threshold_pct", 90.0))
        best["threshold_pct"] = float(100.0 * (1.0 - frac))
        best["assessment_threshold_pct"] = float(
            100.0 * (1.0 - min(frac, POLICY_OPTIMISER_ASSESSMENT_FRAC))
        )
        best["top1_n_trades"] = int(top1_diagnostics.get("n_trades", 0))
        best["top1_active_days"] = int(top1_diagnostics.get("active_days", 0))
        best["top1_avg_trades_per_day"] = float(
            top1_diagnostics.get("avg_trades_per_day", 0.0)
        )
        best["top1_avg_trades_per_active_day"] = float(
            top1_diagnostics.get("avg_trades_per_active_day", 0.0)
        )
        best["top1_avg_trades_per_calendar_day"] = float(
            top1_diagnostics.get("avg_trades_per_calendar_day", 0.0)
        )
        best["top1_avg_holding_hours"] = float(
            top1_diagnostics.get("avg_holding_hours", 0.0)
        )
        best["avg_trades_per_day_at_top_1pct"] = best["top1_avg_trades_per_day"]
        best["avg_holding_hours_at_top_1pct"] = best["top1_avg_holding_hours"]
        best["top5_n_trades"] = int(top5_diagnostics.get("n_trades", 0))
        best["top5_active_days"] = int(top5_diagnostics.get("active_days", 0))
        best["top5_avg_trades_per_day"] = float(
            top5_diagnostics.get("avg_trades_per_day", 0.0)
        )
        best["top5_avg_trades_per_active_day"] = float(
            top5_diagnostics.get("avg_trades_per_active_day", 0.0)
        )
        best["top5_avg_trades_per_calendar_day"] = float(
            top5_diagnostics.get("avg_trades_per_calendar_day", 0.0)
        )
        best["top5_avg_holding_hours"] = float(
            top5_diagnostics.get("avg_holding_hours", 0.0)
        )
        best["cost_pct"] = cost_pct
        best["wallet_range"] = [0.05, 0.15]
        best["sizer_wallet_pnl"] = sizer_wallet_pnl
        best["sizer_net_pnl"] = float(selected.get("net_pnl", 0.0))
        final_metrics = best.get("metrics_final", {})
        final_profit_factor = (
            _safe_float(final_metrics.get("profit_factor"), 0.0)
            if isinstance(final_metrics, dict)
            else 0.0
        )
        lengths = np.asarray(
            context.get(
                "_cached_lengths_",
                context.get("bars_since_entry", np.ones(len(base_rets))),
            ),
            dtype=np.float64,
        )
        if lengths.shape[0] == len(base_rets):
            avg_holding_time_hours = float(np.nanmean(np.maximum(1.0, lengths)))
        else:
            avg_holding_time_hours = 1.0
        policy_idx_arr = np.asarray(policy_idx, dtype=np.int64)

        def _avg_holding_for_original_indices(original_idx: np.ndarray) -> float:
            if lengths.shape[0] != len(policy_idx_arr):
                return 0.0
            wanted = set(int(i) for i in np.asarray(original_idx, dtype=np.int64))
            if not wanted:
                return 0.0
            local_mask = np.asarray(
                [int(i) in wanted for i in policy_idx_arr], dtype=bool
            )
            if not local_mask.any():
                return 0.0
            vals = np.asarray(lengths[local_mask], dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                return 0.0
            return float(np.nanmean(np.maximum(1.0, vals)))

        top1_holding_from_paths = _avg_holding_for_original_indices(diagnostic_idx)
        top5_holding_from_paths = _avg_holding_for_original_indices(assessment_idx)
        if top1_holding_from_paths > 0.0:
            best["top1_avg_holding_hours"] = top1_holding_from_paths
            best["avg_holding_hours_at_top_1pct"] = top1_holding_from_paths
        if top5_holding_from_paths > 0.0:
            best["top5_avg_holding_hours"] = top5_holding_from_paths
        best["avg_holding_time_hours"] = avg_holding_time_hours
        best["opportunities_per_day"] = _opportunities_per_day(
            outcomes.get("timestamp", pd.Series(np.arange(len(outcomes))))
        )
        if isinstance(final_metrics, dict):
            best["weekly_wallet_vol"] = _safe_float(
                final_metrics.get("weekly_wallet_vol"), 0.0
            )
            best["monthly_wallet_vol"] = _safe_float(
                final_metrics.get("monthly_wallet_vol"), 0.0
            )
            best["profit_factor"] = final_profit_factor
        final_net_trade_rets = np.asarray(final_rets_assess, dtype=np.float64) - float(
            cost_pct
        )
        if executed_mask is not None and len(executed_mask) == len(
            final_net_trade_rets
        ):
            exec_mask = np.asarray(executed_mask, dtype=bool)
            final_net_trade_rets = final_net_trade_rets[exec_mask]
            policy_idx_for_assets = (
                policy_idx[exec_mask]
                if len(policy_idx) == len(exec_mask)
                else policy_idx
            )
        else:
            policy_idx_for_assets = policy_idx
        if "symbol" in outcomes.columns and len(policy_idx_for_assets) == len(
            final_net_trade_rets
        ):
            asset_selection = _asset_profit_factor_exclusions(
                symbols=np.asarray(outcomes["symbol"].astype(str).values)[
                    policy_idx_for_assets
                ],
                net_rets=final_net_trade_rets,
                avg_profit_factor=final_profit_factor,
            )
            best["excluded_symbols"] = asset_selection["excluded_symbols"]
            best["asset_metrics"] = asset_selection["asset_metrics"]
        else:
            best["excluded_symbols"] = []
            best["asset_metrics"] = []

        # Build progression table for this strategy
        if "_progression_" in best:
            try:
                import datetime

                dt_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                prog_filename = f"policy_progression_{dt_str}.md"
                prog_path = (
                    Path(data_root)
                    / "artifacts"
                    / run_id
                    / "policy_params"
                    / prog_filename
                )
                os.makedirs(prog_path.parent, exist_ok=True)

                md_content = [f"# Policy Progression for {selected['strategy_id']}", ""]
                md_content.append(f"Run ID: {run_id}")
                md_content.append("")

                headers = [
                    "Step",
                    "Score",
                    "PnL",
                    "Rb_DownRatio",
                    "PnL_25pct",
                    "WkSortino",
                    "MoSortino",
                    "MedPnL/Win",
                    "WkGtP",
                    "Skew",
                    "Ulcer",
                    "TUW",
                    "NegMo%",
                    "DD_Mag",
                    "Trades/Day",
                    "AvgNetPnL",
                    "Params",
                ]

                # We format a text table manually
                row_fmt = "{:<15} | {:>8} | {:>8} | {:>12} | {:>9} | {:>9} | {:>9} | {:>10} | {:>8} | {:>6} | {:>6} | {:>5} | {:>6} | {:>8} | {:>10} | {:>9} | {}"

                md_content.append(row_fmt.format(*headers))
                md_content.append("-" * 180)

                for step in best["_progression_"]:
                    step_name = step.get("step_name", "")
                    score = step.get("score", 0.0)
                    m = step.get("metrics", {})
                    p = step.get("params", {})

                    # keep only active parameters from param_families + standard tp/sl
                    active_params = {}
                    if "size_power" in p:
                        active_params["size_power"] = p["size_power"]
                    if "trailing_power" in p:
                        active_params["trailing_power"] = p["trailing_power"]
                    if "trailing_squash_divisor" in p:
                        active_params["trailing_squash_divisor"] = p[
                            "trailing_squash_divisor"
                        ]
                    if "trailing_override_alpha" in p:
                        active_params["trailing_override_alpha"] = p[
                            "trailing_override_alpha"
                        ]
                    if "giveback_beta" in p:
                        active_params["giveback_beta"] = p["giveback_beta"]
                    if "capital_protect_mfe_mult" in p:
                        active_params["cap_mfe_mult"] = p["capital_protect_mfe_mult"]
                    if "capital_protect_regression_frac" in p:
                        active_params["cap_reg_frac"] = p[
                            "capital_protect_regression_frac"
                        ]
                    if "multiplier_band_min" in p:
                        active_params["mult_min"] = p["multiplier_band_min"]
                    if "multiplier_band_max" in p:
                        active_params["mult_max"] = p["multiplier_band_max"]
                    if "tp_mult" in p:
                        active_params["tp"] = p["tp_mult"]
                    if "sl_mult" in p:
                        active_params["sl"] = p["sl_mult"]

                    def _f(val, fmt=":.4f"):
                        try:
                            return format(val, fmt)
                        except Exception:
                            return str(val)

                    n_trades = m.get("n_trades", 0)
                    trades_per_day = m.get("trades_per_calendar_day", n_trades / 725.0)
                    avg_pnl = (
                        m.get("net_pnl", 0.0) / max(1, n_trades)
                        if n_trades > 0
                        else 0.0
                    )

                    row_str = row_fmt.format(
                        step_name[:15],
                        f"{score:.4f}",
                        f"{m.get('net_pnl', 0.0):.4f}",
                        f"{m.get('robust_downside_ratio', 0.0):.4f}",
                        f"{m.get('pnl_top_25pct_taken_trades', 0.0):.4f}",
                        f"{m.get('weekly_sortino', 0.0):.4f}",
                        f"{m.get('monthly_sortino', 0.0):.4f}",
                        f"{m.get('median_pnl_per_winning_trade', 0.0):.4f}",
                        f"{m.get('weekly_gtp', 0.0):.4f}",
                        f"{m.get('trade_return_skew', 0.0):.4f}",
                        f"{m.get('ulcer', 0.0):.4f}",
                        f"{m.get('tuw', 0.0):.4f}",
                        f"{m.get('neg_months_pct', 0.0):.1%}",
                        f"{m.get('worst_week_dd_mag', 0.0):.4f}",
                        f"{trades_per_day:.2f}",
                        f"{avg_pnl:.5f}",
                        str(active_params),
                    )
                    md_content.append(row_str)

                md_text = "\n".join(md_content)
                tprint("\n" + "=" * 120)
                tprint(md_text)
                tprint("=" * 120 + "\n")

                with open(prog_path, "a") as f:
                    f.write(md_text + "\n\n")

                tprint(f"Appended progression table to {prog_path}")
            except Exception as e:
                logger.warning(f"Failed to generate progression table: {e}")

        strategy_rows.append(best)

    if not strategy_rows:
        tprint("No strategy candidates produced valid policy optimisation results.")
        return {}

    strategy_rows.sort(
        key=lambda row: (
            float(row.get("final_net_pnl", float("-inf"))),
            float(row.get("net_pnl_delta", float("-inf"))),
            float(row.get("metrics_final", {}).get("profit_factor", float("-inf"))),
        ),
        reverse=True,
    )
    best = strategy_rows[0]
    payload = {
        "schema_version": "v4",
        "generated_by": "policy_optimiser",
        "run_id": run_id,
        "cost_pct": float(cost_pct),
        "strategies": strategy_rows,
        "best_strategy_id": strategy_rows[0].get("strategy_id", ""),
        "best_source_model": strategy_rows[0].get("source_model", ""),
    }
    strategy_for_inference = build_strategy_for_inference_payload(
        strategy_rows,
        data_root=data_root,
        run_id=run_id,
    )
    payload["strategy_for_inference"] = strategy_for_inference
    out_dir = Path(data_root) / "artifacts" / run_id / "policy_params"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "best_policy_params.json"
    path.write_text(json.dumps(payload, indent=2, default=float))
    (Path(data_root) / "artifacts" / run_id / "best_policy_params.json").write_text(
        json.dumps(payload, indent=2, default=float)
    )
    strategy_final_acceptation(data_root, run_id, strategy_rows)
    tprint(f"POLICY OPTIMISER COMPLETE -> {path}")
    return payload
