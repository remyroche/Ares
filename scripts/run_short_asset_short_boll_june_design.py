#!/usr/bin/env python3
"""Evaluate the June short-asset defensive and short-boll repair design.

This is a diagnostic replay, not a production promotion script.  It reuses the
materialized A0 June ledger and native component scores, keeps the global floor
unchanged, and separates two interventions:

* post-selection period sizing for ``short_asset``;
* causal head x timestamp eligibility repair for ``short_boll``.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.portfolio_policy_replay import (  # noqa: E402
    INITIAL_WALLET,
    fit_hierarchical_ev_curves,
    normalise_candidate_table,
    portfolio_policy_params_from_live_config,
    replay_candidates,
)


DEFAULT_A0_CANDIDATES = Path(
    "data_perp/artifacts/reliability_blend_arm_A0_anchor_only_20260625_jun15_22"
    "/simple_policy_optimiser/simple_policy_candidates_broad.parquet"
)
DEFAULT_COMPONENT_SCORES = Path(
    "data_perp/reports/native_reliability_blend_scores_20260625_jun15_22_fullfit"
    "/native_reliability_blend_scores.parquet"
)
DEFAULT_TRAIN_CANDIDATES = Path(
    "data_perp/artifacts/reliability_blend_native_simple_policy_replay_20260624_floor070"
    "/simple_policy_optimiser/simple_policy_candidates.parquet"
)
DEFAULT_TRAIN_COMPONENT_SCORES = Path(
    "data_perp/reports/reliability_blend_optuna_20260623_native_lgbm_only_50k"
    "/reliability_blend_component_scores.parquet"
)
DEFAULT_POLICY_MANIFEST = Path(
    "data_perp/reports/reliability_blend_component_arm_portfolio_ablation_20260625"
    "/A0_anchor_only/portfolio_policy_ablation_manifest.json"
)
DEFAULT_OUTPUT_DIR = Path(
    "data_perp/reports/short_asset_short_boll_june_design_20260625"
)

KEY_COLS = ["timestamp", "symbol", "side", "strategy_id"]
HEADS = ("long_bars", "long_dist", "short_asset", "short_boll")
DEFAULT_DISABLED_HEADS = {"long_bars", "long_dist"}


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(float(value)) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _infer_head(strategy_id: Any) -> str:
    text = str(strategy_id)
    for head in HEADS:
        if text.startswith(head):
            return head
    return "unknown"


def _load_policy_params(path: Path, variant: str):
    payload = json.loads(path.read_text(encoding="utf-8"))
    variant_params = payload.get("variant_params", {}).get(variant)
    if not isinstance(variant_params, dict):
        raise KeyError(f"Missing variant_params[{variant!r}] in {path}")
    return portfolio_policy_params_from_live_config(variant_params), payload


def _canonicalise(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    if "head" not in out.columns:
        out["head"] = out["strategy_id"].map(_infer_head)
    for col in ("symbol", "side", "strategy_id", "head"):
        if col in out.columns:
            out[col] = out[col].astype(str)
    return out


def _load_candidates(path: Path, components_path: Path) -> pd.DataFrame:
    candidates = _canonicalise(pd.read_parquet(path))
    components = _canonicalise(pd.read_parquet(components_path))
    keep = [
        "timestamp",
        "symbol",
        "strategy_id",
        "anchor_score",
        "anchor_component_rank",
        "period_component_score",
        "period_component_rank",
        "qfail_component_score",
        "qfail_component_rank",
    ]
    keep = [col for col in keep if col in components.columns]
    components = components[keep].drop_duplicates(["timestamp", "symbol", "strategy_id"])
    out = candidates.merge(
        components,
        on=["timestamp", "symbol", "strategy_id"],
        how="left",
        validate="many_to_one",
    )
    if "deployment_rank_threshold" not in out.columns:
        out["deployment_rank_threshold"] = out.get("base_strategy_threshold", np.nan)
    return normalise_candidate_table(out)


def _repair_short_boll_timestamp_rank(candidates: pd.DataFrame) -> pd.DataFrame:
    """Use causal cross-sectional ranks for short_boll only.

    This does not lower the global floor.  It changes the eligibility contract
    from fullscope absolute score CDF to head x timestamp cross-sectional rank
    for short_boll, which is available at decision time.
    """

    out = candidates.copy()
    mask = out["head"].astype(str).eq("short_boll")
    if not mask.any():
        return out
    score = "anchor_score" if "anchor_score" in out.columns else "calibrated_score"
    repaired = (
        pd.to_numeric(out.loc[mask, score], errors="coerce")
        .groupby([out.loc[mask, "head"], out.loc[mask, "timestamp"]])
        .rank(method="average", pct=True)
    )
    for col in ("normalized_rank_score", "strategy_rank_pct", "policy_rank_pct", "rank_pct"):
        if col in out.columns:
            out.loc[mask, col] = repaired.to_numpy(dtype=np.float64)
    out.loc[mask, "short_boll_rank_contract"] = "head_timestamp_rank_anchor_score"
    return normalise_candidate_table(out)


def _disable_heads(candidates: pd.DataFrame, disabled: set[str]) -> pd.DataFrame:
    out = candidates.loc[~candidates["head"].isin(disabled)].copy()
    return normalise_candidate_table(out)


def _override_head_threshold(
    candidates: pd.DataFrame,
    *,
    head: str,
    threshold: float,
) -> pd.DataFrame:
    out = candidates.copy()
    mask = out["head"].astype(str).eq(str(head))
    if mask.any():
        out.loc[mask, "base_strategy_threshold"] = float(threshold)
        if "deployment_rank_threshold" in out.columns:
            out.loc[mask, "deployment_rank_threshold"] = float(threshold)
    return normalise_candidate_table(out)


def _accepted_trades(candidates: pd.DataFrame, decisions: pd.DataFrame, arm: str) -> pd.DataFrame:
    accepted = decisions.loc[decisions["accepted"].astype(bool)].copy()
    if accepted.empty:
        return pd.DataFrame()
    idx = pd.to_numeric(accepted["candidate_index"], errors="coerce").astype("Int64")
    accepted = accepted.loc[idx.notna()].copy()
    idx = idx.loc[idx.notna()].astype(int)
    cand = candidates.reset_index(drop=True).iloc[idx.to_numpy()].reset_index(drop=True)
    accepted = accepted.reset_index(drop=True)
    for col in [
        "head",
        "net_return",
        "gross_return",
        "simple_policy_exit_reason",
        "normalized_rank_score",
        "policy_rank_pct",
        "base_strategy_threshold",
        "deployment_rank_threshold",
        "calibrated_score",
        "anchor_score",
        "anchor_component_rank",
        "period_component_score",
        "period_component_rank",
        "qfail_component_score",
        "qfail_component_rank",
    ]:
        if col in cand.columns:
            accepted[col] = cand[col].to_numpy()
    accepted["arm"] = arm
    accepted["timestamp"] = pd.to_datetime(accepted["timestamp"], utc=True, errors="coerce")
    accepted["net_pnl"] = (
        pd.to_numeric(accepted["position_size"], errors="coerce").fillna(0.0)
        * pd.to_numeric(accepted["net_return"], errors="coerce").fillna(0.0)
    )
    accepted["gross_pnl"] = (
        pd.to_numeric(accepted["position_size"], errors="coerce").fillna(0.0)
        * pd.to_numeric(accepted.get("gross_return", accepted["net_return"]), errors="coerce").fillna(0.0)
    )
    return accepted


def _period_multiplier(
    accepted: pd.DataFrame,
    *,
    lambda_: float,
    q0: float,
    m_min: float,
    orientation: str,
) -> pd.Series:
    if accepted.empty:
        return pd.Series(dtype=float)
    q = pd.to_numeric(accepted.get("period_component_rank"), errors="coerce")
    if orientation == "low_is_bad":
        q = 1.0 - q
    elif orientation != "high_is_bad":
        raise ValueError(f"Unknown period orientation: {orientation}")
    stress = ((q - float(q0)).clip(lower=0.0) / max(1.0 - float(q0), 1e-9)).fillna(0.0)
    m = 1.0 - float(lambda_) * stress
    m = m.clip(lower=float(m_min), upper=1.0)
    return m


def _apply_short_asset_period_sizing(
    accepted: pd.DataFrame,
    *,
    lambda_: float,
    q0: float,
    m_min: float,
    orientation: str,
) -> pd.DataFrame:
    out = accepted.copy()
    out["period_size_multiplier"] = 1.0
    mask = out["head"].astype(str).eq("short_asset")
    if mask.any():
        out.loc[mask, "period_size_multiplier"] = _period_multiplier(
            out.loc[mask],
            lambda_=lambda_,
            q0=q0,
            m_min=m_min,
            orientation=orientation,
        ).to_numpy(dtype=float)
    out["position_size_original"] = pd.to_numeric(out["position_size"], errors="coerce").fillna(0.0)
    out["position_size"] = out["position_size_original"] * out["period_size_multiplier"]
    out["net_pnl"] = out["position_size"] * pd.to_numeric(out["net_return"], errors="coerce").fillna(0.0)
    out["gross_pnl"] = out["position_size"] * pd.to_numeric(
        out.get("gross_return", out["net_return"]), errors="coerce"
    ).fillna(0.0)
    return out


def _period_size_grid(value: str, default: tuple[float, ...]) -> tuple[float, ...]:
    if not value:
        return default
    vals: list[float] = []
    for item in value.split(","):
        item = item.strip()
        if item:
            vals.append(float(item))
    return tuple(vals) if vals else default


def _attach_train_period_rank(
    accepted: pd.DataFrame,
    component_scores: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Attach a pre-June head x timestamp period rank to accepted train trades."""

    out = accepted.copy()
    out["period_component_rank"] = np.nan
    if not component_scores.exists():
        return out, {
            "component_scores": str(component_scores),
            "exists": False,
            "coverage": 0.0,
            "source": "missing",
        }
    comp = _canonicalise(pd.read_parquet(component_scores))
    if "period_new_score" not in comp.columns:
        return out, {
            "component_scores": str(component_scores),
            "exists": True,
            "coverage": 0.0,
            "source": "missing_period_new_score",
        }
    comp["period_new_score"] = pd.to_numeric(comp["period_new_score"], errors="coerce")
    agg = (
        comp.dropna(subset=["period_new_score"])
        .groupby(["head", "timestamp"], sort=True)["period_new_score"]
        .mean()
        .rename("period_component_score")
        .reset_index()
    )
    if agg.empty:
        return out, {
            "component_scores": str(component_scores),
            "exists": True,
            "coverage": 0.0,
            "source": "empty_period_new_score",
        }
    agg["period_component_rank"] = (
        agg.groupby("head", sort=False)["period_component_score"]
        .rank(method="average", pct=True)
        .astype(float)
    )
    merged = out.merge(
        agg[["head", "timestamp", "period_component_score", "period_component_rank"]],
        on=["head", "timestamp"],
        how="left",
        validate="many_to_one",
        suffixes=("", "_train"),
    )
    if "period_component_rank_train" in merged.columns:
        merged["period_component_rank"] = pd.to_numeric(
            merged["period_component_rank_train"], errors="coerce"
        )
        merged = merged.drop(columns=["period_component_rank_train"])
    coverage = float(pd.to_numeric(merged["period_component_rank"], errors="coerce").notna().mean())
    by_head = (
        merged.groupby("head")["period_component_rank"]
        .apply(lambda s: float(pd.to_numeric(s, errors="coerce").notna().mean()))
        .to_dict()
    )
    return merged, {
        "component_scores": str(component_scores),
        "exists": True,
        "source": "period_new_score head_timestamp mean percentile rank",
        "coverage": coverage,
        "coverage_by_head": by_head,
        "component_timestamp_min": agg["timestamp"].min(),
        "component_timestamp_max": agg["timestamp"].max(),
        "component_head_timestamp_rows": int(len(agg)),
    }


def _build_period_selection_folds(
    accepted: pd.DataFrame,
    *,
    min_train_days: int,
    valid_days: int,
    embargo_hours: int,
) -> list[dict[str, pd.Timestamp]]:
    ts = (
        pd.to_datetime(accepted["timestamp"], utc=True, errors="coerce")
        .dropna()
        .drop_duplicates()
        .sort_values()
    )
    if ts.empty:
        return []
    folds: list[dict[str, pd.Timestamp]] = []
    valid_start = ts.min() + pd.Timedelta(days=int(min_train_days))
    last = ts.max()
    fold_id = 0
    while valid_start <= last:
        valid_end = min(valid_start + pd.Timedelta(days=int(valid_days)), last + pd.Timedelta(nanoseconds=1))
        train_end = valid_start - pd.Timedelta(hours=int(embargo_hours))
        train_ts = ts.loc[ts < train_end]
        valid_ts = ts.loc[(ts >= valid_start) & (ts < valid_end)]
        if len(train_ts) >= 24 and len(valid_ts) >= 3:
            fold_id += 1
            folds.append(
                {
                    "fold": fold_id,
                    "train_start": train_ts.min(),
                    "train_end": train_ts.max(),
                    "valid_start": valid_ts.min(),
                    "valid_end": valid_ts.max() + pd.Timedelta(nanoseconds=1),
                }
            )
        valid_start = valid_end
    return folds


def _select_period_multiplier_config(
    train_accepted: pd.DataFrame,
    component_scores: Path,
    *,
    lambdas: tuple[float, ...],
    q0s: tuple[float, ...],
    min_multipliers: tuple[float, ...],
    orientations: tuple[str, ...],
    min_train_days: int,
    valid_days: int,
    embargo_hours: int,
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any]]:
    with_period, coverage_report = _attach_train_period_rank(train_accepted, component_scores)
    folds = _build_period_selection_folds(
        with_period,
        min_train_days=min_train_days,
        valid_days=valid_days,
        embargo_hours=embargo_hours,
    )
    if not folds:
        fallback = {
            "lambda": 0.0,
            "q0": 0.50,
            "m_min": 1.0,
            "orientation": "high_is_bad",
            "selected_by": "fallback_no_valid_pre_june_folds",
        }
        return fallback, pd.DataFrame(), coverage_report

    rows: list[dict[str, Any]] = []
    ts = pd.to_datetime(with_period["timestamp"], utc=True, errors="coerce")
    for orientation in orientations:
        for lambda_ in lambdas:
            for q0 in q0s:
                for m_min in min_multipliers:
                    fold_rows: list[dict[str, float]] = []
                    for fold in folds:
                        mask = (ts >= fold["valid_start"]) & (ts < fold["valid_end"])
                        valid = with_period.loc[mask].copy()
                        if valid.empty:
                            continue
                        base = _metrics_from_accepted(valid)
                        sized = _apply_short_asset_period_sizing(
                            valid,
                            lambda_=float(lambda_),
                            q0=float(q0),
                            m_min=float(m_min),
                            orientation=orientation,
                        )
                        metrics = _metrics_from_accepted(sized)
                        fold_rows.append(
                            {
                                "fold": float(fold["fold"]),
                                "base_net_pnl": float(base["net_pnl"]),
                                "net_pnl": float(metrics["net_pnl"]),
                                "delta_net_pnl": float(metrics["net_pnl"] - base["net_pnl"]),
                                "base_full_sl_rate": float(base["full_sl_rate"]),
                                "full_sl_rate": float(metrics["full_sl_rate"]),
                                "mean_period_size_multiplier": float(metrics["mean_period_size_multiplier"]),
                                "trade_count": float(metrics["trade_count"]),
                            }
                        )
                    if not fold_rows:
                        continue
                    fold_df = pd.DataFrame(fold_rows)
                    rows.append(
                        {
                            "orientation": orientation,
                            "lambda": float(lambda_),
                            "q0": float(q0),
                            "m_min": float(m_min),
                            "folds": int(fold_df["fold"].nunique()),
                            "median_delta_net_pnl": float(fold_df["delta_net_pnl"].median()),
                            "mean_delta_net_pnl": float(fold_df["delta_net_pnl"].mean()),
                            "q25_delta_net_pnl": float(fold_df["delta_net_pnl"].quantile(0.25)),
                            "positive_delta_share": float((fold_df["delta_net_pnl"] > 0.0).mean()),
                            "mean_period_size_multiplier": float(fold_df["mean_period_size_multiplier"].mean()),
                            "median_full_sl_rate": float(fold_df["full_sl_rate"].median()),
                            "fold_net_pnls": json.dumps(_json_safe(fold_df["net_pnl"].tolist())),
                            "fold_delta_net_pnls": json.dumps(_json_safe(fold_df["delta_net_pnl"].tolist())),
                        }
                    )
    trials = pd.DataFrame(rows)
    if trials.empty:
        fallback = {
            "lambda": 0.0,
            "q0": 0.50,
            "m_min": 1.0,
            "orientation": "high_is_bad",
            "selected_by": "fallback_no_valid_pre_june_trials",
        }
        return fallback, trials, coverage_report

    sort_cols = [
        "median_delta_net_pnl",
        "q25_delta_net_pnl",
        "positive_delta_share",
        "mean_delta_net_pnl",
        "mean_period_size_multiplier",
    ]
    selected = trials.sort_values(sort_cols, ascending=[False, False, False, False, False]).iloc[0]
    config = {
        "lambda": float(selected["lambda"]),
        "q0": float(selected["q0"]),
        "m_min": float(selected["m_min"]),
        "orientation": str(selected["orientation"]),
        "selected_by": "pre_june_walkforward_median_delta_net_pnl",
        "median_delta_net_pnl": float(selected["median_delta_net_pnl"]),
        "q25_delta_net_pnl": float(selected["q25_delta_net_pnl"]),
        "positive_delta_share": float(selected["positive_delta_share"]),
        "mean_period_size_multiplier": float(selected["mean_period_size_multiplier"]),
        "fold_count": int(selected["folds"]),
    }
    return config, trials, coverage_report


def _select_short_boll_threshold_config(
    train_candidates: pd.DataFrame,
    params,
    *,
    thresholds: tuple[float, ...],
    market_mode: str,
    min_train_days: int,
    valid_days: int,
    embargo_hours: int,
    min_total_valid_trades: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Select a short_boll timestamp-rank threshold using pre-June folds only."""

    repaired = _disable_heads(
        _repair_short_boll_timestamp_rank(train_candidates),
        disabled={"long_bars", "long_dist", "short_asset"},
    )
    if repaired.empty:
        fallback = {
            "threshold": float(max(0.71, min(thresholds) if thresholds else 0.71)),
            "selected_by": "fallback_no_short_boll_train_candidates",
        }
        return fallback, pd.DataFrame()

    folds = _build_period_selection_folds(
        repaired,
        min_train_days=min_train_days,
        valid_days=valid_days,
        embargo_hours=embargo_hours,
    )
    if not folds:
        fallback = {
            "threshold": float(max(0.71, min(thresholds) if thresholds else 0.71)),
            "selected_by": "fallback_no_short_boll_pre_june_folds",
        }
        return fallback, pd.DataFrame()

    ts = pd.to_datetime(repaired["timestamp"], utc=True, errors="coerce")
    grid = sorted({float(max(0.71, t)) for t in thresholds if np.isfinite(float(t))})
    rows: list[dict[str, Any]] = []
    for threshold in grid:
        fold_rows: list[dict[str, float]] = []
        for fold in folds:
            train_mask = ts < fold["train_end"]
            valid_mask = (ts >= fold["valid_start"]) & (ts < fold["valid_end"])
            train_fold = repaired.loc[train_mask].copy()
            valid_fold = repaired.loc[valid_mask].copy()
            if len(train_fold) < 100 or len(valid_fold) < 10:
                continue
            train_fold = _override_head_threshold(train_fold, head="short_boll", threshold=threshold)
            valid_fold = _override_head_threshold(valid_fold, head="short_boll", threshold=threshold)
            ev_curve = fit_hierarchical_ev_curves(train_fold)
            decisions, _, replay_metrics = replay_candidates(
                valid_fold,
                params,
                mode="global_auction",
                ev_curve=ev_curve,
                market_mode=market_mode,
            )
            accepted = _accepted_trades(valid_fold, decisions, f"short_boll_threshold_{threshold:.3f}")
            metrics = _metrics_from_accepted(accepted)
            fold_rows.append(
                {
                    "fold": float(fold["fold"]),
                    "net_pnl": float(metrics["net_pnl"]),
                    "trade_count": float(metrics["trade_count"]),
                    "win_rate": float(metrics["win_rate"]) if np.isfinite(metrics["win_rate"]) else np.nan,
                    "full_sl_rate": float(metrics["full_sl_rate"]) if np.isfinite(metrics["full_sl_rate"]) else np.nan,
                    "max_drawdown": float(metrics["max_drawdown"]) if np.isfinite(metrics["max_drawdown"]) else np.nan,
                    "replay_net_pnl": float(replay_metrics.get("net_pnl", np.nan)),
                }
            )
        if not fold_rows:
            continue
        fold_df = pd.DataFrame(fold_rows)
        rows.append(
            {
                "threshold": float(threshold),
                "folds": int(fold_df["fold"].nunique()),
                "total_valid_trades": int(fold_df["trade_count"].sum()),
                "mean_valid_trades": float(fold_df["trade_count"].mean()),
                "median_net_pnl": float(fold_df["net_pnl"].median()),
                "mean_net_pnl": float(fold_df["net_pnl"].mean()),
                "q25_net_pnl": float(fold_df["net_pnl"].quantile(0.25)),
                "positive_pnl_share": float((fold_df["net_pnl"] > 0.0).mean()),
                "median_win_rate": float(fold_df["win_rate"].median()),
                "median_full_sl_rate": float(fold_df["full_sl_rate"].median()),
                "worst_fold_net_pnl": float(fold_df["net_pnl"].min()),
                "fold_net_pnls": json.dumps(_json_safe(fold_df["net_pnl"].tolist())),
                "fold_trade_counts": json.dumps(_json_safe(fold_df["trade_count"].tolist())),
            }
        )

    trials = pd.DataFrame(rows)
    if trials.empty:
        fallback = {
            "threshold": float(max(0.71, min(thresholds) if thresholds else 0.71)),
            "selected_by": "fallback_no_short_boll_threshold_trials",
        }
        return fallback, trials
    eligible = trials.loc[trials["total_valid_trades"] >= int(min_total_valid_trades)].copy()
    if eligible.empty:
        eligible = trials.copy()
        selected_by = "pre_june_walkforward_median_pnl_no_min_trade_pass"
    else:
        selected_by = "pre_june_walkforward_median_pnl"
    selected = eligible.sort_values(
        [
            "median_net_pnl",
            "q25_net_pnl",
            "positive_pnl_share",
            "mean_net_pnl",
            "total_valid_trades",
        ],
        ascending=[False, False, False, False, False],
    ).iloc[0]
    config = {
        "threshold": float(selected["threshold"]),
        "selected_by": selected_by,
        "fold_count": int(selected["folds"]),
        "total_valid_trades": int(selected["total_valid_trades"]),
        "median_net_pnl": float(selected["median_net_pnl"]),
        "q25_net_pnl": float(selected["q25_net_pnl"]),
        "positive_pnl_share": float(selected["positive_pnl_share"]),
        "global_floor_lower_bound": 0.71,
    }
    return config, trials


def _metrics_from_accepted(accepted: pd.DataFrame) -> dict[str, Any]:
    if accepted.empty:
        return {
            "trade_count": 0,
            "timestamp_count": 0,
            "symbol_count": 0,
            "head_count": 0,
            "win_rate": np.nan,
            "net_pnl": 0.0,
            "gross_pnl": 0.0,
            "cost_pnl": 0.0,
            "mean_net_return": np.nan,
            "q05_net_return": np.nan,
            "full_sl_rate": np.nan,
            "timeout_rate": np.nan,
            "max_drawdown": np.nan,
            "mean_period_size_multiplier": np.nan,
        }
    net_return = pd.to_numeric(accepted["net_return"], errors="coerce").fillna(0.0)
    net_pnl = pd.to_numeric(accepted["net_pnl"], errors="coerce").fillna(0.0)
    gross_pnl = pd.to_numeric(accepted["gross_pnl"], errors="coerce").fillna(0.0)
    reason = accepted.get("simple_policy_exit_reason", pd.Series("", index=accepted.index)).astype(str).str.lower()
    ordered = accepted.sort_values("timestamp", kind="mergesort")
    equity = INITIAL_WALLET + pd.to_numeric(ordered["net_pnl"], errors="coerce").fillna(0.0).cumsum()
    drawdown = equity / equity.cummax().clip(lower=1e-9) - 1.0
    if "period_size_multiplier" in accepted.columns:
        mult = pd.to_numeric(accepted["period_size_multiplier"], errors="coerce")
    else:
        mult = pd.Series(np.nan, index=accepted.index, dtype=float)
    return {
        "trade_count": int(len(accepted)),
        "timestamp_count": int(pd.to_datetime(accepted["timestamp"], utc=True, errors="coerce").nunique()),
        "symbol_count": int(accepted["symbol"].astype(str).nunique()) if "symbol" in accepted else 0,
        "head_count": int(accepted["head"].astype(str).nunique()) if "head" in accepted else 0,
        "win_rate": float((net_pnl > 0.0).mean()),
        "net_pnl": float(net_pnl.sum()),
        "gross_pnl": float(gross_pnl.sum()),
        "cost_pnl": float((gross_pnl - net_pnl).sum()),
        "mean_net_return": float(net_return.mean()),
        "q05_net_return": float(net_return.quantile(0.05)),
        "full_sl_rate": float(reason.isin(["sl", "full_sl", "stop", "stop_loss"]).mean()),
        "timeout_rate": float(reason.str.contains("timeout", regex=False).mean()),
        "max_drawdown": float(drawdown.min()) if len(drawdown) else np.nan,
        "mean_period_size_multiplier": float(mult.mean()) if mult.notna().any() else np.nan,
    }


def _rank_contract_audit(candidates: pd.DataFrame, arm: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for head, group in candidates.groupby("head", sort=True):
        rank = pd.to_numeric(group.get("normalized_rank_score"), errors="coerce")
        threshold = pd.to_numeric(group.get("base_strategy_threshold"), errors="coerce")
        score = pd.to_numeric(group.get("calibrated_score"), errors="coerce")
        rows.append(
            {
                "arm": arm,
                "head": head,
                "rows": int(len(group)),
                "timestamps": int(group["timestamp"].nunique()),
                "rank_min": float(rank.min()),
                "rank_median": float(rank.median()),
                "rank_max": float(rank.max()),
                "threshold_median": float(threshold.median()),
                "rank_ge_070": float((rank >= 0.70).mean()),
                "rank_ge_threshold": float((rank >= threshold).mean()),
                "score_min": float(score.min()),
                "score_median": float(score.median()),
                "score_max": float(score.max()),
            }
        )
    return rows


def _rejection_audit(decisions: pd.DataFrame) -> pd.DataFrame:
    if decisions.empty:
        return pd.DataFrame()
    work = decisions.copy()
    work["head"] = work["strategy_id"].map(_infer_head)
    rows: list[dict[str, Any]] = []
    for (arm, head), group in work.groupby(["arm", "head"], sort=True):
        counts = group["rejection_reason"].astype(str).value_counts(dropna=False)
        for reason, count in counts.items():
            rows.append(
                {
                    "arm": arm,
                    "head": head,
                    "rejection_reason": reason,
                    "count": int(count),
                    "share": float(count / max(len(group), 1)),
                }
            )
    return pd.DataFrame(rows)


def _short_boll_validation(candidates: pd.DataFrame, repaired: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for label, frame in (
        ("strict_fullscope_rank", candidates),
        ("timestamp_rank_repair", repaired),
    ):
        group = frame.loc[frame["head"].astype(str).eq("short_boll")].copy()
        if group.empty:
            continue
        rank = pd.to_numeric(group["normalized_rank_score"], errors="coerce")
        score = pd.to_numeric(group["calibrated_score"], errors="coerce")
        net = pd.to_numeric(group["net_return"], errors="coerce")
        threshold = pd.to_numeric(group["base_strategy_threshold"], errors="coerce")
        rows.append(
            {
                "contract": label,
                "slice": "all",
                "rows": int(len(group)),
                "timestamps": int(group["timestamp"].nunique()),
                "rank_min": float(rank.min()),
                "rank_median": float(rank.median()),
                "rank_max": float(rank.max()),
                "rank_ge_070": float((rank >= 0.70).mean()),
                "rank_ge_threshold": float((rank >= threshold).mean()),
                "score_rank_spearman": float(score.corr(net, method="spearman")),
                "rank_return_spearman": float(rank.corr(net, method="spearman")),
                "win_rate": float((net > 0).mean()),
                "mean_net": float(net.mean()),
                "sum_net": float(net.sum()),
                "q05_net": float(net.quantile(0.05)),
            }
        )
        for lo, hi in ((0.70, 0.80), (0.80, 0.90), (0.90, 1.01)):
            sub = group.loc[(rank >= lo) & (rank < hi)]
            sub_net = pd.to_numeric(sub["net_return"], errors="coerce")
            rows.append(
                {
                    "contract": label,
                    "slice": f"rank_{lo:.2f}_{min(hi, 1.0):.2f}",
                    "rows": int(len(sub)),
                    "timestamps": int(sub["timestamp"].nunique()) if len(sub) else 0,
                    "rank_min": float(rank.loc[sub.index].min()) if len(sub) else np.nan,
                    "rank_median": float(rank.loc[sub.index].median()) if len(sub) else np.nan,
                    "rank_max": float(rank.loc[sub.index].max()) if len(sub) else np.nan,
                    "rank_ge_070": float((rank.loc[sub.index] >= 0.70).mean()) if len(sub) else np.nan,
                    "rank_ge_threshold": float((rank.loc[sub.index] >= threshold.loc[sub.index]).mean()) if len(sub) else np.nan,
                    "score_rank_spearman": np.nan,
                    "rank_return_spearman": np.nan,
                    "win_rate": float((sub_net > 0).mean()) if len(sub) else np.nan,
                    "mean_net": float(sub_net.mean()) if len(sub) else np.nan,
                    "sum_net": float(sub_net.sum()) if len(sub) else 0.0,
                    "q05_net": float(sub_net.quantile(0.05)) if len(sub) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def _render_report(summary: pd.DataFrame, rank_audit: pd.DataFrame, manifest: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Short Asset / Short Boll June Design")
    lines.append("")
    lines.append(f"Generated: {manifest['generated_at_utc']}")
    lines.append("")
    lines.append("## Arms")
    lines.append("")
    lines.append("- D0: A0 anchor-only baseline on the active replay universe.")
    lines.append("- D1: D0 accepted trades with short_asset period size multiplier after selection.")
    lines.append("- D2: D0 plus short_boll head x timestamp rank eligibility repair.")
    lines.append("- D3: D2 plus the same short_asset period size multiplier.")
    lines.append("")
    lines.append(
        "Global floor remains unchanged; long_bars and long_dist are removed from the default active replay universe."
    )
    lines.append("")
    lines.append("## Portfolio Summary")
    lines.append("")
    lines.append(summary.to_markdown(index=False))
    lines.append("")
    lines.append("## Rank Contract Audit")
    lines.append("")
    lines.append(rank_audit.to_markdown(index=False))
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- The strict A0 rank contract excludes short_boll because its frozen fullscope rank never reaches 0.70.")
    lines.append("- The repaired short_boll arm uses only causal head x timestamp ordering, not June return-selected thresholds.")
    sb_cfg = manifest.get("short_boll_repair", {}).get("selected_threshold", {})
    if sb_cfg:
        lines.append(
            "- The short_boll repaired eligibility threshold is selected before June: "
            f"threshold={sb_cfg.get('threshold')}, selected_by={sb_cfg.get('selected_by')}."
        )
    cfg = manifest.get("period_modifier", {})
    if cfg:
        lines.append(
            "- The short_asset period multiplier configuration is selected on pre-June walk-forward folds: "
            f"orientation={cfg.get('orientation')}, lambda={cfg.get('lambda')}, "
            f"q0={cfg.get('q0')}, m_min={cfg.get('m_min')}."
        )
    lines.append("- The flipped orientation rows are diagnostic only and are not the frozen candidate.")
    lines.append(
        f"- Evaluation window: {manifest.get('eval_start')} to {manifest.get('eval_end')} UTC."
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a0-candidates", type=Path, default=DEFAULT_A0_CANDIDATES)
    parser.add_argument("--component-scores", type=Path, default=DEFAULT_COMPONENT_SCORES)
    parser.add_argument("--train-candidates", type=Path, default=DEFAULT_TRAIN_CANDIDATES)
    parser.add_argument("--train-component-scores", type=Path, default=DEFAULT_TRAIN_COMPONENT_SCORES)
    parser.add_argument("--policy-manifest", type=Path, default=DEFAULT_POLICY_MANIFEST)
    parser.add_argument("--policy-variant", default="refit_bar4_strategy_bar2")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--period-lambda", type=float, default=0.50)
    parser.add_argument("--period-q0", type=float, default=0.50)
    parser.add_argument("--period-min-multiplier", type=float, default=0.35)
    parser.add_argument("--period-lambda-grid", default="0.0,0.25,0.5,0.75")
    parser.add_argument("--period-q0-grid", default="0.5,0.6,0.7,0.8")
    parser.add_argument("--period-min-multiplier-grid", default="0.25,0.35,0.5,0.75")
    parser.add_argument("--period-selection-min-train-days", type=int, default=21)
    parser.add_argument("--period-selection-valid-days", type=int, default=7)
    parser.add_argument("--period-selection-embargo-hours", type=int, default=96)
    parser.add_argument("--skip-period-prejune-selection", action="store_true")
    parser.add_argument("--short-boll-threshold-grid", default="0.71,0.75,0.8,0.85,0.9")
    parser.add_argument("--short-boll-selection-min-train-days", type=int, default=21)
    parser.add_argument("--short-boll-selection-valid-days", type=int, default=7)
    parser.add_argument("--short-boll-selection-embargo-hours", type=int, default=96)
    parser.add_argument("--short-boll-min-valid-trades", type=int, default=25)
    parser.add_argument("--skip-short-boll-prejune-threshold-selection", action="store_true")
    parser.add_argument(
        "--include-long-heads-in-d0",
        action="store_true",
        help=(
            "Diagnostic only: keep long_bars/long_dist candidates in the D0 universe. "
            "Default removes them to match the current active-head decision."
        ),
    )
    parser.add_argument("--market-mode", default="perps")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    params, policy_payload = _load_policy_params(args.policy_manifest, args.policy_variant)
    train = normalise_candidate_table(pd.read_parquet(args.train_candidates))
    ev_curve = fit_hierarchical_ev_curves(train)
    train_decisions, _, _ = replay_candidates(
        train,
        params,
        mode="global_auction",
        ev_curve=ev_curve,
        market_mode=args.market_mode,
    )
    train_accepted = _accepted_trades(train, train_decisions, "pre_june_train")
    if args.skip_period_prejune_selection:
        selected_period = {
            "lambda": float(args.period_lambda),
            "q0": float(args.period_q0),
            "m_min": float(args.period_min_multiplier),
            "orientation": "high_is_bad",
            "selected_by": "manual_cli_defaults_skip_period_prejune_selection",
        }
        period_trials = pd.DataFrame()
        period_coverage = {"source": "not_evaluated"}
    else:
        selected_period, period_trials, period_coverage = _select_period_multiplier_config(
            train_accepted,
            args.train_component_scores,
            lambdas=_period_size_grid(args.period_lambda_grid, (0.0, 0.25, 0.5, 0.75)),
            q0s=_period_size_grid(args.period_q0_grid, (0.5, 0.6, 0.7, 0.8)),
            min_multipliers=_period_size_grid(args.period_min_multiplier_grid, (0.25, 0.35, 0.5, 0.75)),
            orientations=("high_is_bad", "low_is_bad"),
            min_train_days=int(args.period_selection_min_train_days),
            valid_days=int(args.period_selection_valid_days),
            embargo_hours=int(args.period_selection_embargo_hours),
        )
    if args.skip_short_boll_prejune_threshold_selection:
        selected_short_boll = {
            "threshold": 0.71,
            "selected_by": "manual_global_floor_skip_short_boll_prejune_selection",
            "global_floor_lower_bound": 0.71,
        }
        short_boll_threshold_trials = pd.DataFrame()
    else:
        selected_short_boll, short_boll_threshold_trials = _select_short_boll_threshold_config(
            train,
            params,
            thresholds=_period_size_grid(args.short_boll_threshold_grid, (0.71, 0.75, 0.8, 0.85, 0.9)),
            market_mode=args.market_mode,
            min_train_days=int(args.short_boll_selection_min_train_days),
            valid_days=int(args.short_boll_selection_valid_days),
            embargo_hours=int(args.short_boll_selection_embargo_hours),
            min_total_valid_trades=int(args.short_boll_min_valid_trades),
        )
    a0_raw = _load_candidates(args.a0_candidates, args.component_scores)
    active_disabled_heads = set() if args.include_long_heads_in_d0 else set(DEFAULT_DISABLED_HEADS)
    a0 = _disable_heads(a0_raw, active_disabled_heads)
    d2_candidates = _disable_heads(
        _override_head_threshold(
            _repair_short_boll_timestamp_rank(a0),
            head="short_boll",
            threshold=float(selected_short_boll["threshold"]),
        ),
        disabled=active_disabled_heads,
    )

    arms = {
        "D0_A0_anchor_only": a0,
        "D2_A0_plus_short_boll_timestamp_rank": d2_candidates,
        "short_asset_standalone": _disable_heads(a0, disabled=set(DEFAULT_DISABLED_HEADS) | {"short_boll"}),
        "short_boll_standalone_timestamp_rank": _disable_heads(
            d2_candidates,
            disabled=set(DEFAULT_DISABLED_HEADS) | {"short_asset"},
        ),
    }

    accepted_frames: list[pd.DataFrame] = []
    decision_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    rank_rows: list[dict[str, Any]] = []
    for arm, candidates in arms.items():
        decisions, equity, replay_metrics = replay_candidates(
            candidates,
            params,
            mode="global_auction",
            ev_curve=ev_curve,
            market_mode=args.market_mode,
        )
        accepted = _accepted_trades(candidates, decisions, arm)
        accepted_frames.append(accepted)
        decision = decisions.copy()
        decision["arm"] = arm
        decision_frames.append(decision)
        rec = {
            "arm": arm,
            "description": "raw replay",
            "candidate_rows": int(len(candidates)),
            **_metrics_from_accepted(accepted),
            "replay_metrics_net_pnl": replay_metrics.get("net_pnl"),
            "replay_metrics_trade_count": replay_metrics.get("trade_count"),
        }
        summary_rows.append(rec)
        rank_rows.extend(_rank_contract_audit(candidates, arm))

        if arm == "D0_A0_anchor_only":
            d1 = _apply_short_asset_period_sizing(
                accepted,
                lambda_=float(selected_period["lambda"]),
                q0=float(selected_period["q0"]),
                m_min=float(selected_period["m_min"]),
                orientation=str(selected_period["orientation"]),
            )
            d1["arm"] = "D1_A0_short_asset_period_size_frozen_prejune"
            accepted_frames.append(d1)
            summary_rows.append(
                {
                    "arm": "D1_A0_short_asset_period_size_frozen_prejune",
                    "description": "post-selection sizing; pre-June selected period config",
                    "candidate_rows": int(len(candidates)),
                    **_metrics_from_accepted(d1),
                    "replay_metrics_net_pnl": np.nan,
                    "replay_metrics_trade_count": np.nan,
                }
            )
            d1_flip = _apply_short_asset_period_sizing(
                accepted,
                lambda_=args.period_lambda,
                q0=args.period_q0,
                m_min=args.period_min_multiplier,
                orientation="low_is_bad",
            )
            d1_flip["arm"] = "D1_DIAGNOSTIC_flipped_period_orientation"
            accepted_frames.append(d1_flip)
            summary_rows.append(
                {
                    "arm": "D1_DIAGNOSTIC_flipped_period_orientation",
                    "description": "diagnostic only; q_period low is bad",
                    "candidate_rows": int(len(candidates)),
                    **_metrics_from_accepted(d1_flip),
                    "replay_metrics_net_pnl": np.nan,
                    "replay_metrics_trade_count": np.nan,
                }
            )

        if arm == "D2_A0_plus_short_boll_timestamp_rank":
            d3 = _apply_short_asset_period_sizing(
                accepted,
                lambda_=float(selected_period["lambda"]),
                q0=float(selected_period["q0"]),
                m_min=float(selected_period["m_min"]),
                orientation=str(selected_period["orientation"]),
            )
            d3["arm"] = "D3_D2_short_asset_period_size_frozen_prejune"
            accepted_frames.append(d3)
            summary_rows.append(
                {
                    "arm": "D3_D2_short_asset_period_size_frozen_prejune",
                    "description": "D2 plus post-selection sizing; pre-June selected period config",
                    "candidate_rows": int(len(candidates)),
                    **_metrics_from_accepted(d3),
                    "replay_metrics_net_pnl": np.nan,
                    "replay_metrics_trade_count": np.nan,
                }
            )
            d3_flip = _apply_short_asset_period_sizing(
                accepted,
                lambda_=args.period_lambda,
                q0=args.period_q0,
                m_min=args.period_min_multiplier,
                orientation="low_is_bad",
            )
            d3_flip["arm"] = "D3_DIAGNOSTIC_flipped_period_orientation"
            accepted_frames.append(d3_flip)
            summary_rows.append(
                {
                    "arm": "D3_DIAGNOSTIC_flipped_period_orientation",
                    "description": "diagnostic only; D2 plus q_period low is bad",
                    "candidate_rows": int(len(candidates)),
                    **_metrics_from_accepted(d3_flip),
                    "replay_metrics_net_pnl": np.nan,
                    "replay_metrics_trade_count": np.nan,
                }
            )

    summary = pd.DataFrame(summary_rows)
    accepted_all = pd.concat([df for df in accepted_frames if not df.empty], ignore_index=True)
    decisions_all = pd.concat(decision_frames, ignore_index=True)
    rank_audit = pd.DataFrame(rank_rows)
    rejections = _rejection_audit(decisions_all)
    short_boll_validation = _short_boll_validation(a0, d2_candidates)
    eval_ts = pd.to_datetime(a0["timestamp"], utc=True, errors="coerce").dropna()

    summary.to_csv(args.output_dir / "portfolio_summary.csv", index=False)
    period_trials.to_csv(args.output_dir / "period_multiplier_selection_trials.csv", index=False)
    short_boll_threshold_trials.to_csv(
        args.output_dir / "short_boll_threshold_selection_trials.csv",
        index=False,
    )
    accepted_all.to_parquet(args.output_dir / "accepted_trades.parquet", index=False)
    decisions_all.to_parquet(args.output_dir / "decisions.parquet", index=False)
    rank_audit.to_csv(args.output_dir / "rank_contract_audit.csv", index=False)
    rejections.to_csv(args.output_dir / "rejection_audit.csv", index=False)
    short_boll_validation.to_csv(args.output_dir / "short_boll_validation.csv", index=False)

    by_head_rows: list[dict[str, Any]] = []
    if not accepted_all.empty:
        for (arm, head), group in accepted_all.groupby(["arm", "head"], sort=True):
            rec = {"arm": arm, "head": head}
            rec.update(_metrics_from_accepted(group))
            by_head_rows.append(rec)
    by_head = pd.DataFrame(by_head_rows)
    by_head.to_csv(args.output_dir / "portfolio_summary_by_head.csv", index=False)

    manifest = {
        "generated_by": "run_short_asset_short_boll_june_design",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "a0_candidates": str(args.a0_candidates),
        "component_scores": str(args.component_scores),
        "eval_start": eval_ts.min().isoformat() if not eval_ts.empty else None,
        "eval_end": eval_ts.max().isoformat() if not eval_ts.empty else None,
        "eval_candidate_rows": int(len(a0)),
        "eval_disabled_heads": sorted(active_disabled_heads),
        "include_long_heads_in_d0": bool(args.include_long_heads_in_d0),
        "train_candidates": str(args.train_candidates),
        "train_component_scores": str(args.train_component_scores),
        "policy_manifest": str(args.policy_manifest),
        "policy_variant": str(args.policy_variant),
        "policy_params": asdict(params),
        "period_modifier": {
            **selected_period,
            "train_period_coverage": period_coverage,
            "selection_trials": str(args.output_dir / "period_multiplier_selection_trials.csv"),
            "flipped_orientation_is_diagnostic_only": True,
        },
        "short_boll_repair": {
            "enabled": True,
            "contract": "head_x_timestamp_rank(anchor_score), causal at decision timestamp",
            "global_floor_lowered": False,
            "long_heads_disabled": True,
            "selected_threshold": selected_short_boll,
            "selection_trials": str(args.output_dir / "short_boll_threshold_selection_trials.csv"),
        },
        "outputs": {
            "summary": str(args.output_dir / "portfolio_summary.csv"),
            "summary_by_head": str(args.output_dir / "portfolio_summary_by_head.csv"),
            "period_multiplier_selection_trials": str(args.output_dir / "period_multiplier_selection_trials.csv"),
            "short_boll_threshold_selection_trials": str(args.output_dir / "short_boll_threshold_selection_trials.csv"),
            "accepted_trades": str(args.output_dir / "accepted_trades.parquet"),
            "decisions": str(args.output_dir / "decisions.parquet"),
            "rank_contract_audit": str(args.output_dir / "rank_contract_audit.csv"),
            "rejection_audit": str(args.output_dir / "rejection_audit.csv"),
            "short_boll_validation": str(args.output_dir / "short_boll_validation.csv"),
            "report": str(args.output_dir / "short_asset_short_boll_june_design_report.md"),
        },
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(_json_safe(manifest), indent=2) + "\n")
    report = _render_report(summary, rank_audit, manifest)
    (args.output_dir / "short_asset_short_boll_june_design_report.md").write_text(report, encoding="utf-8")
    print(json.dumps(_json_safe(manifest), indent=2)[:6000])
    print(f"\nWrote {args.output_dir}")


if __name__ == "__main__":
    main()
