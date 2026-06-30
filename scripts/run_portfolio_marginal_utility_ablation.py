#!/usr/bin/env python3
"""Counterfactual marginal-utility ablation for portfolio sizing.

This script keeps scores, rank contracts, thresholds, and auction ordering fixed.
It compares:

N0: baseline replay with no multiplier.
N1: current global risk model benchmark (G4-style combined risk).
N2: long-format marginal utility model selecting one global multiplier.
N3: shared timestamp encoder with strategy-specific size multipliers.
N4: shared timestamp encoder with strategy-specific threshold uplifts.

The N2 target is not generic risk.  For each timestamp and candidate multiplier,
it estimates the realized forward utility difference versus m=1.0 using frozen
policy replays inside the training fold.
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
    fit_hierarchical_ev_curves,
    normalise_candidate_table,
    replay_candidates,
)
from scripts.run_global_portfolio_period_multiplier import (  # noqa: E402
    DEFAULT_POLICY_MANIFEST,
    DEFAULT_TRAIN_BROAD,
    DEFAULT_TRAIN_DEPLOYABLE,
    _accepted_trades,
    _add_open_position_concentration_features,
    _add_portfolio_state_features,
    _add_trailing_performance,
    _apply_multiplier,
    _feature_columns,
    _fit_models,
    _forward_labels,
    _json_safe,
    _load_candidates,
    _load_policy_params,
    _map_risk_to_multiplier,
    _metrics_row as _global_metrics_row,
    _period_proxy,
    _predict_models,
    _timestamp_feature_fill_values,
    _timestamp_features,
)
from scripts.run_global_portfolio_period_multiplier_walkforward import (  # noqa: E402
    _build_folds,
    _timestamp_mask,
)


DEFAULT_OUTPUT_DIR = Path("data_perp/reports/portfolio_marginal_utility_ablation_20260625")
MULTIPLIERS = (0.25, 0.50, 0.75, 1.00)
THRESHOLD_UPLIFTS = {1.0: 0.0, 0.75: 0.025, 0.50: 0.05, 0.25: 0.10}


def _schedule_for_timestamps(timestamps: pd.Series, multiplier: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(timestamps, utc=True, errors="coerce"),
            "multiplier": float(multiplier),
        }
    ).dropna(subset=["timestamp"]).drop_duplicates("timestamp")


def _replay_with_multiplier(
    candidates: pd.DataFrame,
    params: Any,
    ev_curve: dict[str, Any],
    *,
    multiplier: float,
    market_mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], pd.DataFrame]:
    schedule = _schedule_for_timestamps(candidates["timestamp"], multiplier)
    arm_candidates = _apply_multiplier(
        candidates,
        schedule,
        scale_entries=False,
        max_entries=int(params.max_new_entries_per_bar),
    )
    decisions, equity, metrics = replay_candidates(
        arm_candidates,
        params,
        mode="global_auction",
        ev_curve=ev_curve,
        market_mode=market_mode,
    )
    accepted = _accepted_trades(arm_candidates, decisions)
    return decisions, equity, metrics, accepted


def _forward_j_labels(
    timestamps: pd.Series,
    accepted: pd.DataFrame,
    *,
    horizon_hours: int,
    lambda_cost: float,
    lambda_dd: float,
) -> pd.DataFrame:
    labels = _forward_labels(timestamps, accepted, int(horizon_hours))
    if accepted.empty:
        labels["future_worst_trade_return"] = np.nan
        labels["J"] = np.nan
        return labels
    acc = accepted.copy()
    acc["timestamp"] = pd.to_datetime(acc["timestamp"], utc=True, errors="coerce")
    acc["position_size"] = pd.to_numeric(acc["position_size"], errors="coerce").fillna(0.0)
    acc["net_return"] = pd.to_numeric(acc["net_return"], errors="coerce").fillna(0.0)
    rows: list[dict[str, Any]] = []
    ts_values = pd.to_datetime(timestamps, utc=True, errors="coerce").dropna().drop_duplicates().sort_values()
    for ts in ts_values:
        end = ts + pd.Timedelta(hours=int(horizon_hours))
        window = acc.loc[(acc["timestamp"] > ts) & (acc["timestamp"] <= end)]
        rows.append(
            {
                "timestamp": ts,
                "future_worst_trade_return": float(window["net_return"].min()) if len(window) else np.nan,
            }
        )
    dd = pd.DataFrame(rows)
    out = labels.merge(dd, on="timestamp", how="left")
    utility = pd.to_numeric(out["future_utility"], errors="coerce")
    cost = pd.to_numeric(out["future_cost_to_gross"], errors="coerce").clip(lower=0.0).fillna(0.0)
    worst = pd.to_numeric(out["future_worst_trade_return"], errors="coerce").fillna(0.0)
    drawdown_proxy = (-worst).clip(lower=0.0)
    out["J"] = utility - float(lambda_cost) * cost - float(lambda_dd) * drawdown_proxy
    return out


def _counterfactual_label_panel(
    candidates: pd.DataFrame,
    params: Any,
    ev_curve: dict[str, Any],
    *,
    horizon_hours: int,
    lambda_cost: float,
    lambda_dd: float,
    market_mode: str,
) -> tuple[pd.DataFrame, dict[float, dict[str, Any]], dict[float, pd.DataFrame]]:
    timestamps = pd.to_datetime(candidates["timestamp"], utc=True, errors="coerce").dropna().drop_duplicates().sort_values()
    label_frames: list[pd.DataFrame] = []
    metrics_by_m: dict[float, dict[str, Any]] = {}
    accepted_by_m: dict[float, pd.DataFrame] = {}
    base_j: pd.Series | None = None
    for multiplier in MULTIPLIERS:
        _, _, metrics, accepted = _replay_with_multiplier(
            candidates,
            params,
            ev_curve,
            multiplier=float(multiplier),
            market_mode=market_mode,
        )
        metrics_by_m[float(multiplier)] = metrics
        accepted_by_m[float(multiplier)] = accepted
        labels = _forward_j_labels(
            timestamps,
            accepted,
            horizon_hours=int(horizon_hours),
            lambda_cost=float(lambda_cost),
            lambda_dd=float(lambda_dd),
        )
        labels["multiplier"] = float(multiplier)
        label_frames.append(labels)
        if abs(float(multiplier) - 1.0) < 1e-12:
            base_j = labels.set_index("timestamp")["J"]
    panel = pd.concat(label_frames, ignore_index=True)
    if base_j is None:
        raise RuntimeError("Missing baseline multiplier labels")
    panel["delta_J"] = panel["J"] - panel["timestamp"].map(base_j)
    return panel, metrics_by_m, accepted_by_m


def _build_timestamp_features(
    candidates: pd.DataFrame,
    accepted: pd.DataFrame,
    equity: pd.DataFrame,
    *,
    feature_cols_raw: list[str],
    max_feature_cols: int,
    fill_values: pd.Series | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    features = _timestamp_features(
        candidates,
        feature_cols=feature_cols_raw,
        max_cols=int(max_feature_cols),
        fill_values=fill_values,
    )
    if fill_values is None:
        fill_values = _timestamp_feature_fill_values(features)
    features = _add_trailing_performance(features, accepted)
    features = _add_portfolio_state_features(features, equity)
    features = _add_open_position_concentration_features(features, accepted)
    features["period_proxy"] = _period_proxy(features)
    return features, fill_values


def _long_format_training_frame(features: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    base = features.copy()
    base["timestamp"] = pd.to_datetime(base["timestamp"], utc=True, errors="coerce")
    lab = labels[["timestamp", "multiplier", "delta_J"]].copy()
    lab["timestamp"] = pd.to_datetime(lab["timestamp"], utc=True, errors="coerce")
    out = lab.merge(base, on="timestamp", how="left")
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["delta_J"])
    return out


def _fit_marginal_utility_model(train: pd.DataFrame, feature_cols: list[str]):
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if len(train) < 100:
        raise RuntimeError(f"Not enough marginal utility rows: {len(train)}")
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        GradientBoostingRegressor(
            random_state=41,
            max_depth=2,
            n_estimators=160,
            learning_rate=0.035,
            subsample=0.85,
        ),
    )
    model.fit(train[feature_cols], pd.to_numeric(train["delta_J"], errors="coerce").fillna(0.0))
    return model


def _predict_multiplier_schedule(
    model: Any,
    features: pd.DataFrame,
    feature_cols: list[str],
    *,
    min_positive_edge: float,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for multiplier in MULTIPLIERS:
        frame = features.copy()
        frame["multiplier"] = float(multiplier)
        for col in feature_cols:
            if col not in frame.columns:
                frame[col] = 0.0
        pred = np.asarray(model.predict(frame[feature_cols]), dtype=float)
        rows.append(
            pd.DataFrame(
                {
                    "timestamp": frame["timestamp"].to_numpy(),
                    "multiplier": float(multiplier),
                    "pred_delta_J": pred,
                }
            )
        )
    pred = pd.concat(rows, ignore_index=True)
    pred = pred.sort_values(["timestamp", "pred_delta_J", "multiplier"], ascending=[True, False, False])
    best = pred.drop_duplicates("timestamp", keep="first").copy()
    best.loc[pd.to_numeric(best["pred_delta_J"], errors="coerce") < float(min_positive_edge), "multiplier"] = 1.0
    return best[["timestamp", "multiplier", "pred_delta_J"]].sort_values("timestamp")


def _oracle_schedule(labels: pd.DataFrame) -> pd.DataFrame:
    pred = labels[["timestamp", "multiplier", "delta_J"]].dropna(subset=["delta_J"]).copy()
    pred = pred.sort_values(["timestamp", "delta_J", "multiplier"], ascending=[True, False, False])
    best = pred.drop_duplicates("timestamp", keep="first").rename(columns={"delta_J": "oracle_delta_J"})
    return best[["timestamp", "multiplier", "oracle_delta_J"]].sort_values("timestamp")


def _replay_schedule(
    candidates: pd.DataFrame,
    schedule: pd.DataFrame,
    params: Any,
    ev_curve: dict[str, Any],
    *,
    market_mode: str,
    scale_entries: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], pd.DataFrame]:
    arm_candidates = _apply_multiplier(
        candidates,
        schedule[["timestamp", "multiplier"]],
        scale_entries=bool(scale_entries),
        max_entries=int(params.max_new_entries_per_bar),
    )
    decisions, equity, metrics = replay_candidates(
        arm_candidates,
        params,
        mode="global_auction",
        ev_curve=ev_curve,
        market_mode=market_mode,
    )
    accepted = _accepted_trades(arm_candidates, decisions)
    return decisions, equity, metrics, accepted


def _strategy_ids(candidates: pd.DataFrame) -> list[str]:
    if "strategy_id" not in candidates.columns:
        return []
    return sorted(str(x) for x in candidates["strategy_id"].dropna().astype(str).unique())


def _apply_strategy_schedule(
    candidates: pd.DataFrame,
    schedule: pd.DataFrame,
    *,
    action: str,
) -> pd.DataFrame:
    work = candidates.copy()
    if "strategy_id" not in work.columns or schedule.empty:
        return work
    keys = ["timestamp", "strategy_id"]
    sched = schedule.copy()
    sched["timestamp"] = pd.to_datetime(sched["timestamp"], utc=True, errors="coerce")
    sched["strategy_id"] = sched["strategy_id"].astype(str)
    work["timestamp"] = pd.to_datetime(work["timestamp"], utc=True, errors="coerce")
    work["strategy_id"] = work["strategy_id"].astype(str)
    work = work.merge(sched[keys + ["multiplier"]], on=keys, how="left")
    multiplier = pd.to_numeric(work.pop("multiplier"), errors="coerce").fillna(1.0).clip(lower=0.0, upper=1.0)
    if action == "size":
        base = (
            pd.to_numeric(work.get("portfolio_size_multiplier"), errors="coerce").fillna(1.0)
            if "portfolio_size_multiplier" in work.columns
            else pd.Series(1.0, index=work.index)
        )
        work["portfolio_size_multiplier"] = (base * multiplier).clip(lower=0.0, upper=1.0)
    elif action == "threshold":
        base = pd.to_numeric(work["base_strategy_threshold"], errors="coerce").fillna(1.0)
        uplift = multiplier.map(THRESHOLD_UPLIFTS).fillna(0.0)
        work["base_strategy_threshold"] = np.maximum(base, (base + uplift).clip(upper=0.999))
    else:
        raise ValueError(f"Unknown strategy action: {action}")
    return work


def _replay_strategy_schedule(
    candidates: pd.DataFrame,
    schedule: pd.DataFrame,
    params: Any,
    ev_curve: dict[str, Any],
    *,
    action: str,
    market_mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], pd.DataFrame]:
    arm_candidates = _apply_strategy_schedule(candidates, schedule, action=action)
    decisions, equity, metrics = replay_candidates(
        arm_candidates,
        params,
        mode="global_auction",
        ev_curve=ev_curve,
        market_mode=market_mode,
    )
    accepted = _accepted_trades(arm_candidates, decisions)
    return decisions, equity, metrics, accepted


def _replay_strategy_threshold_with_global_cap(
    candidates: pd.DataFrame,
    strategy_schedule: pd.DataFrame,
    global_schedule: pd.DataFrame,
    params: Any,
    ev_curve: dict[str, Any],
    *,
    market_mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], pd.DataFrame]:
    arm_candidates = _apply_strategy_schedule(candidates, strategy_schedule, action="threshold")
    if not global_schedule.empty:
        arm_candidates = _apply_multiplier(
            arm_candidates,
            global_schedule[["timestamp", "multiplier"]],
            scale_entries=False,
            max_entries=int(params.max_new_entries_per_bar),
        )
    decisions, equity, metrics = replay_candidates(
        arm_candidates,
        params,
        mode="global_auction",
        ev_curve=ev_curve,
        market_mode=market_mode,
    )
    accepted = _accepted_trades(arm_candidates, decisions)
    return decisions, equity, metrics, accepted


def _strategy_feature_frame(candidates: pd.DataFrame, timestamp_features: pd.DataFrame) -> pd.DataFrame:
    if "strategy_id" not in candidates.columns:
        return pd.DataFrame(columns=["timestamp", "strategy_id"])
    work = candidates.copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"], utc=True, errors="coerce")
    work["strategy_id"] = work["strategy_id"].astype(str)
    rank_col = "strategy_rank_pct" if "strategy_rank_pct" in work.columns else "rank_pct"
    score_col = "calibrated_score" if "calibrated_score" in work.columns else "normalized_rank_score"
    for col in (rank_col, score_col, "base_strategy_threshold"):
        if col not in work.columns:
            work[col] = np.nan
        work[col] = pd.to_numeric(work[col], errors="coerce")
    grouped = work.groupby(["timestamp", "strategy_id"], observed=True)
    rows = grouped.agg(
        strategy_candidate_count=(rank_col, "size"),
        strategy_rank_mean=(rank_col, "mean"),
        strategy_rank_max=(rank_col, "max"),
        strategy_rank_q75=(rank_col, lambda s: float(np.nanquantile(s, 0.75)) if len(s) else np.nan),
        strategy_score_mean=(score_col, "mean"),
        strategy_score_max=(score_col, "max"),
        strategy_threshold_mean=("base_strategy_threshold", "mean"),
    ).reset_index()
    ts = timestamp_features.copy()
    ts["timestamp"] = pd.to_datetime(ts["timestamp"], utc=True, errors="coerce")
    out = rows.merge(ts, on="timestamp", how="left")
    codes, _ = pd.factorize(out["strategy_id"].astype(str), sort=True)
    out["strategy_code"] = codes.astype(float)
    return out.replace([np.inf, -np.inf], np.nan)


def _constant_strategy_schedule(
    timestamps: pd.Series,
    strategy_ids: list[str],
    strategy_id: str,
    multiplier: float,
) -> pd.DataFrame:
    ts = pd.to_datetime(timestamps, utc=True, errors="coerce").dropna().drop_duplicates()
    rows = pd.MultiIndex.from_product([ts, strategy_ids], names=["timestamp", "strategy_id"]).to_frame(index=False)
    rows["multiplier"] = 1.0
    rows.loc[rows["strategy_id"].astype(str).eq(str(strategy_id)), "multiplier"] = float(multiplier)
    return rows


def _strategy_counterfactual_label_panel(
    candidates: pd.DataFrame,
    params: Any,
    ev_curve: dict[str, Any],
    *,
    action: str,
    horizon_hours: int,
    lambda_cost: float,
    lambda_dd: float,
    market_mode: str,
) -> pd.DataFrame:
    timestamps = pd.to_datetime(candidates["timestamp"], utc=True, errors="coerce").dropna().drop_duplicates().sort_values()
    strategy_ids = _strategy_ids(candidates)
    if not strategy_ids:
        return pd.DataFrame()
    base_schedule = pd.MultiIndex.from_product([timestamps, strategy_ids], names=["timestamp", "strategy_id"]).to_frame(index=False)
    base_schedule["multiplier"] = 1.0
    _, _, _, base_accepted = _replay_strategy_schedule(
        candidates,
        base_schedule,
        params,
        ev_curve,
        action=action,
        market_mode=market_mode,
    )
    base_j = _forward_j_labels(
        timestamps,
        base_accepted,
        horizon_hours=int(horizon_hours),
        lambda_cost=float(lambda_cost),
        lambda_dd=float(lambda_dd),
    ).set_index("timestamp")["J"]
    frames: list[pd.DataFrame] = []
    for strategy_id in strategy_ids:
        for multiplier in MULTIPLIERS:
            schedule = _constant_strategy_schedule(timestamps, strategy_ids, strategy_id, float(multiplier))
            _, _, _, accepted = _replay_strategy_schedule(
                candidates,
                schedule,
                params,
                ev_curve,
                action=action,
                market_mode=market_mode,
            )
            labels = _forward_j_labels(
                timestamps,
                accepted,
                horizon_hours=int(horizon_hours),
                lambda_cost=float(lambda_cost),
                lambda_dd=float(lambda_dd),
            )
            labels["strategy_id"] = str(strategy_id)
            labels["multiplier"] = float(multiplier)
            labels["delta_J"] = labels["J"] - labels["timestamp"].map(base_j)
            frames.append(labels[["timestamp", "strategy_id", "multiplier", "delta_J"]])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _strategy_long_training_frame(strategy_features: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    if strategy_features.empty or labels.empty:
        return pd.DataFrame()
    base = strategy_features.copy()
    base["timestamp"] = pd.to_datetime(base["timestamp"], utc=True, errors="coerce")
    base["strategy_id"] = base["strategy_id"].astype(str)
    lab = labels.copy()
    lab["timestamp"] = pd.to_datetime(lab["timestamp"], utc=True, errors="coerce")
    lab["strategy_id"] = lab["strategy_id"].astype(str)
    out = lab.merge(base, on=["timestamp", "strategy_id"], how="left")
    return out.replace([np.inf, -np.inf], np.nan).dropna(subset=["delta_J"])


def _predict_strategy_schedule(
    model: Any,
    strategy_features: pd.DataFrame,
    feature_cols: list[str],
    *,
    min_positive_edge: float,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for multiplier in MULTIPLIERS:
        frame = strategy_features.copy()
        frame["multiplier"] = float(multiplier)
        for col in feature_cols:
            if col not in frame.columns:
                frame[col] = 0.0
        pred = np.asarray(model.predict(frame[feature_cols]), dtype=float)
        rows.append(
            pd.DataFrame(
                {
                    "timestamp": frame["timestamp"].to_numpy(),
                    "strategy_id": frame["strategy_id"].astype(str).to_numpy(),
                    "multiplier": float(multiplier),
                    "pred_delta_J": pred,
                }
            )
        )
    pred = pd.concat(rows, ignore_index=True)
    pred = pred.sort_values(["timestamp", "strategy_id", "pred_delta_J", "multiplier"], ascending=[True, True, False, False])
    best = pred.drop_duplicates(["timestamp", "strategy_id"], keep="first").copy()
    best.loc[pd.to_numeric(best["pred_delta_J"], errors="coerce") < float(min_positive_edge), "multiplier"] = 1.0
    return best[["timestamp", "strategy_id", "multiplier", "pred_delta_J"]].sort_values(["timestamp", "strategy_id"])


def _promotion_summary(summary: pd.DataFrame) -> pd.DataFrame:
    base = summary.loc[summary["arm"].eq("N0_baseline"), ["fold_id", "net_pnl", "cost_pnl", "max_drawdown", "worst_24h_net_pnl", "notional_turnover"]]
    base = base.rename(
        columns={
            "net_pnl": "base_net_pnl",
            "cost_pnl": "base_cost_pnl",
            "max_drawdown": "base_max_drawdown",
            "worst_24h_net_pnl": "base_worst_24h_net_pnl",
            "notional_turnover": "base_notional_turnover",
        }
    )
    work = summary.merge(base, on="fold_id", how="left")
    work["delta_net_pnl"] = work["net_pnl"] - work["base_net_pnl"]
    work["delta_cost_pnl"] = work["cost_pnl"] - work["base_cost_pnl"]
    work["delta_max_drawdown"] = work["max_drawdown"] - work["base_max_drawdown"]
    work["delta_worst_24h_net_pnl"] = work["worst_24h_net_pnl"] - work["base_worst_24h_net_pnl"]
    work["exposure_ratio"] = work["notional_turnover"] / work["base_notional_turnover"].replace(0.0, np.nan)
    rows: list[dict[str, Any]] = []
    for arm, g in work.groupby("arm", sort=True):
        rows.append(
            {
                "arm": arm,
                "folds": int(g["fold_id"].nunique()),
                "median_delta_net_pnl": float(g["delta_net_pnl"].median()),
                "q25_delta_net_pnl": float(g["delta_net_pnl"].quantile(0.25)),
                "mean_delta_net_pnl": float(g["delta_net_pnl"].mean()),
                "positive_delta_net_pnl_share": float((g["delta_net_pnl"] > 0).mean()),
                "median_delta_cost_pnl": float(g["delta_cost_pnl"].median()),
                "median_delta_max_drawdown": float(g["delta_max_drawdown"].median()),
                "median_delta_worst_24h_net_pnl": float(g["delta_worst_24h_net_pnl"].median()),
                "median_exposure_ratio": float(g["exposure_ratio"].median()),
                "median_multiplier": float(g["mean_multiplier"].median()),
            }
        )
    return pd.DataFrame(rows)


def _metric_row_for_arm(arm: str, fold_id: int, metrics: dict[str, Any], schedule: pd.DataFrame, accepted: pd.DataFrame) -> dict[str, Any]:
    row = _global_metrics_row(arm, metrics, schedule, accepted)
    row["fold_id"] = int(fold_id)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--broad-candidates", type=Path, default=DEFAULT_TRAIN_BROAD)
    parser.add_argument("--deployable-candidates", type=Path, default=DEFAULT_TRAIN_DEPLOYABLE)
    parser.add_argument("--policy-manifest", type=Path, default=DEFAULT_POLICY_MANIFEST)
    parser.add_argument("--policy-variant", default="refit_bar4_strategy_bar2")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--horizon-hours", type=int, default=72)
    parser.add_argument("--embargo-hours", type=int, default=96)
    parser.add_argument("--min-train-hours", type=int, default=336)
    parser.add_argument("--fold-hours", type=int, default=168)
    parser.add_argument("--max-folds", type=int, default=3)
    parser.add_argument("--max-feature-cols", type=int, default=96)
    parser.add_argument("--lambda-cost", type=float, default=0.001)
    parser.add_argument("--lambda-dd", type=float, default=0.25)
    parser.add_argument("--min-positive-edge", type=float, default=0.0)
    parser.add_argument("--market-mode", default="perps")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    params, policy_payload = _load_policy_params(args.policy_manifest, args.policy_variant)
    broad = _load_candidates(args.broad_candidates)
    deployable = _load_candidates(args.deployable_candidates)
    folds = _build_folds(
        broad["timestamp"],
        min_train_hours=int(args.min_train_hours),
        fold_hours=int(args.fold_hours),
        embargo_hours=int(args.embargo_hours),
        max_folds=args.max_folds,
    )
    if not folds:
        raise RuntimeError("No folds available")

    summary_rows: list[dict[str, Any]] = []
    schedule_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []
    oracle_frames: list[pd.DataFrame] = []
    fold_rows: list[dict[str, Any]] = []

    for fold in folds:
        fold_id = int(fold["fold_id"])
        train_end = pd.Timestamp(fold["train_end"])
        eval_start = pd.Timestamp(fold["eval_start"])
        eval_end = pd.Timestamp(fold["eval_end"]) + pd.Timedelta(nanoseconds=1)
        train_broad = broad.loc[_timestamp_mask(broad, end=train_end + pd.Timedelta(nanoseconds=1))].copy()
        eval_candidates = broad.loc[_timestamp_mask(broad, start=eval_start, end=eval_end)].copy()
        train_deployable = deployable.loc[_timestamp_mask(deployable, end=train_end + pd.Timedelta(nanoseconds=1))].copy()
        if len(train_broad) < 200 or len(train_deployable) < 50 or len(eval_candidates) < 20:
            continue

        ev_curve = fit_hierarchical_ev_curves(train_deployable)
        train_decisions, train_equity, _ = replay_candidates(
            train_broad,
            params,
            mode="global_auction",
            ev_curve=ev_curve,
            market_mode=args.market_mode,
        )
        train_accepted = _accepted_trades(train_broad, train_decisions)
        feature_cols_raw = _feature_columns(train_broad, max_cols=int(args.max_feature_cols))
        train_features, fill_values = _build_timestamp_features(
            train_broad,
            train_accepted,
            train_equity,
            feature_cols_raw=feature_cols_raw,
            max_feature_cols=int(args.max_feature_cols),
        )
        label_panel, train_metrics_by_m, _ = _counterfactual_label_panel(
            train_broad,
            params,
            ev_curve,
            horizon_hours=int(args.horizon_hours),
            lambda_cost=float(args.lambda_cost),
            lambda_dd=float(args.lambda_dd),
            market_mode=args.market_mode,
        )
        label_cutoff = train_end - pd.Timedelta(hours=int(args.horizon_hours))
        label_panel = label_panel.loc[pd.to_datetime(label_panel["timestamp"], utc=True, errors="coerce") <= label_cutoff].copy()
        train_long = _long_format_training_frame(train_features, label_panel)
        model_feature_cols = [
            col
            for col in train_long.columns
            if col not in {"timestamp", "delta_J", "J"}
            and pd.api.types.is_numeric_dtype(train_long[col])
        ]
        mu_model = _fit_marginal_utility_model(train_long, model_feature_cols)
        train_strategy_features = _strategy_feature_frame(train_broad, train_features)
        strategy_models: dict[str, Any] = {}
        strategy_feature_cols_by_action: dict[str, list[str]] = {}
        for strategy_action in ("size", "threshold"):
            strategy_labels = _strategy_counterfactual_label_panel(
                train_broad,
                params,
                ev_curve,
                action=strategy_action,
                horizon_hours=int(args.horizon_hours),
                lambda_cost=float(args.lambda_cost),
                lambda_dd=float(args.lambda_dd),
                market_mode=args.market_mode,
            )
            if not strategy_labels.empty:
                strategy_labels = strategy_labels.loc[
                    pd.to_datetime(strategy_labels["timestamp"], utc=True, errors="coerce") <= label_cutoff
                ].copy()
            strategy_long = _strategy_long_training_frame(train_strategy_features, strategy_labels)
            strategy_feature_cols = [
                col
                for col in strategy_long.columns
                if col not in {"timestamp", "strategy_id", "delta_J", "J"}
                and pd.api.types.is_numeric_dtype(strategy_long[col])
            ]
            if len(strategy_long) >= 100 and strategy_feature_cols:
                strategy_models[strategy_action] = _fit_marginal_utility_model(strategy_long, strategy_feature_cols)
                strategy_feature_cols_by_action[strategy_action] = strategy_feature_cols

        # Current global risk model benchmark for N1, fitted on the same training fold.
        risk_labels = _forward_labels(train_features["timestamp"], train_accepted, int(args.horizon_hours))
        risk_frame = train_features.merge(risk_labels, on="timestamp", how="left")
        risk_frame = risk_frame.loc[
            pd.to_datetime(risk_frame["timestamp"], utc=True, errors="coerce") <= label_cutoff
        ].copy()
        risk_feature_cols = [c for c in train_features.columns if c != "timestamp" and pd.api.types.is_numeric_dtype(train_features[c])]
        risk_models, risk_cutoffs, _ = _fit_models(risk_frame, risk_feature_cols)

        # Eval features use baseline replay through the fold end for causal current-state context.
        history_candidates = broad.loc[_timestamp_mask(broad, end=eval_end)].copy()
        history_decisions, history_equity, _ = replay_candidates(
            history_candidates,
            params,
            mode="global_auction",
            ev_curve=ev_curve,
            market_mode=args.market_mode,
        )
        history_accepted = _accepted_trades(history_candidates, history_decisions)
        eval_features, _ = _build_timestamp_features(
            eval_candidates,
            history_accepted,
            history_equity,
            feature_cols_raw=feature_cols_raw,
            max_feature_cols=int(args.max_feature_cols),
            fill_values=fill_values,
        )
        for col in model_feature_cols:
            if col not in eval_features.columns:
                eval_features[col] = 0.0
        for col in risk_feature_cols:
            if col not in eval_features.columns:
                eval_features[col] = 0.0
        eval_strategy_features = _strategy_feature_frame(eval_candidates, eval_features)

        n0_schedule = _schedule_for_timestamps(eval_candidates["timestamp"], 1.0)
        _, _, n0_metrics, n0_accepted = _replay_schedule(
            eval_candidates,
            n0_schedule,
            params,
            ev_curve,
            market_mode=args.market_mode,
        )
        summary_rows.append(_metric_row_for_arm("N0_baseline", fold_id, n0_metrics, n0_schedule, n0_accepted))

        risk_pred = _predict_models(risk_models, eval_features, risk_feature_cols)
        g4_raw = _map_risk_to_multiplier(risk_pred["combined_risk"], risk_cutoffs)
        g4_raw = g4_raw.where(pd.to_numeric(risk_pred["pred_utility_q10"], errors="coerce") >= 0.0, 0.25)
        n1_schedule = pd.DataFrame({"timestamp": eval_features["timestamp"], "multiplier": g4_raw.to_numpy(dtype=float)})
        _, _, n1_metrics, n1_accepted = _replay_schedule(
            eval_candidates,
            n1_schedule,
            params,
            ev_curve,
            market_mode=args.market_mode,
        )
        summary_rows.append(_metric_row_for_arm("N1_global_risk_G4", fold_id, n1_metrics, n1_schedule, n1_accepted))

        n2_schedule = _predict_multiplier_schedule(
            mu_model,
            eval_features,
            model_feature_cols,
            min_positive_edge=float(args.min_positive_edge),
        )
        _, _, n2_metrics, n2_accepted = _replay_schedule(
            eval_candidates,
            n2_schedule,
            params,
            ev_curve,
            market_mode=args.market_mode,
        )
        summary_rows.append(_metric_row_for_arm("N2_global_marginal_utility", fold_id, n2_metrics, n2_schedule, n2_accepted))

        if "size" in strategy_models:
            n3_schedule = _predict_strategy_schedule(
                strategy_models["size"],
                eval_strategy_features,
                strategy_feature_cols_by_action["size"],
                min_positive_edge=float(args.min_positive_edge),
            )
            _, _, n3_metrics, n3_accepted = _replay_strategy_schedule(
                eval_candidates,
                n3_schedule,
                params,
                ev_curve,
                action="size",
                market_mode=args.market_mode,
            )
            summary_rows.append(_metric_row_for_arm("N3_strategy_size_marginal_utility", fold_id, n3_metrics, n3_schedule, n3_accepted))
        else:
            n3_schedule = pd.DataFrame()

        if "threshold" in strategy_models:
            n4_schedule = _predict_strategy_schedule(
                strategy_models["threshold"],
                eval_strategy_features,
                strategy_feature_cols_by_action["threshold"],
                min_positive_edge=float(args.min_positive_edge),
            )
            _, _, n4_metrics, n4_accepted = _replay_strategy_schedule(
                eval_candidates,
                n4_schedule,
                params,
                ev_curve,
                action="threshold",
                market_mode=args.market_mode,
            )
            summary_rows.append(_metric_row_for_arm("N4_strategy_threshold_marginal_utility", fold_id, n4_metrics, n4_schedule, n4_accepted))
        else:
            n4_schedule = pd.DataFrame()

        if not n4_schedule.empty:
            _, _, n5_metrics, n5_accepted = _replay_strategy_threshold_with_global_cap(
                eval_candidates,
                n4_schedule,
                n1_schedule,
                params,
                ev_curve,
                market_mode=args.market_mode,
            )
            n5_schedule = n4_schedule.copy()
            n5_schedule = n5_schedule.merge(
                n1_schedule.rename(columns={"multiplier": "global_cap_multiplier"}),
                on="timestamp",
                how="left",
            )
            n5_schedule["multiplier"] = pd.to_numeric(n5_schedule["multiplier"], errors="coerce").fillna(1.0) * pd.to_numeric(
                n5_schedule["global_cap_multiplier"], errors="coerce"
            ).fillna(1.0)
            summary_rows.append(_metric_row_for_arm("N5_strategy_threshold_plus_emergency_cap", fold_id, n5_metrics, n5_schedule, n5_accepted))
        else:
            n5_schedule = pd.DataFrame()

        eval_label_panel, _, _ = _counterfactual_label_panel(
            eval_candidates,
            params,
            ev_curve,
            horizon_hours=int(args.horizon_hours),
            lambda_cost=float(args.lambda_cost),
            lambda_dd=float(args.lambda_dd),
            market_mode=args.market_mode,
        )
        oracle_schedule = _oracle_schedule(eval_label_panel)
        _, _, oracle_metrics, oracle_accepted = _replay_schedule(
            eval_candidates,
            oracle_schedule,
            params,
            ev_curve,
            market_mode=args.market_mode,
        )
        summary_rows.append(_metric_row_for_arm("N2_oracle_multiplier", fold_id, oracle_metrics, oracle_schedule, oracle_accepted))

        for arm, sched in (
            ("N0_baseline", n0_schedule),
            ("N1_global_risk_G4", n1_schedule),
            ("N2_global_marginal_utility", n2_schedule),
            ("N3_strategy_size_marginal_utility", n3_schedule),
            ("N4_strategy_threshold_marginal_utility", n4_schedule),
            ("N5_strategy_threshold_plus_emergency_cap", n5_schedule),
            ("N2_oracle_multiplier", oracle_schedule.rename(columns={"oracle_delta_J": "pred_delta_J"})),
        ):
            if sched.empty:
                continue
            tmp = sched.copy()
            tmp["fold_id"] = fold_id
            tmp["arm"] = arm
            schedule_frames.append(tmp)
        pred = n2_schedule.copy()
        pred["fold_id"] = fold_id
        prediction_frames.append(pred)
        oracle = oracle_schedule.copy()
        oracle["fold_id"] = fold_id
        oracle_frames.append(oracle)
        fold_rows.append(
            {
                **fold,
                "train_rows_long": int(len(train_long)),
                "model_feature_count": int(len(model_feature_cols)),
                "strategy_size_model_feature_count": int(len(strategy_feature_cols_by_action.get("size", []))),
                "strategy_threshold_model_feature_count": int(len(strategy_feature_cols_by_action.get("threshold", []))),
                "train_baseline_net_pnl_m1": float(train_metrics_by_m[1.0].get("net_pnl", 0.0)),
            }
        )

    summary = pd.DataFrame(summary_rows)
    if summary.empty:
        raise RuntimeError("No fold summaries generated")
    promotion = _promotion_summary(summary)
    summary.to_csv(args.output_dir / "marginal_utility_fold_summary.csv", index=False)
    promotion.to_csv(args.output_dir / "marginal_utility_promotion_summary.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(args.output_dir / "marginal_utility_folds.csv", index=False)
    if schedule_frames:
        pd.concat(schedule_frames, ignore_index=True).to_csv(args.output_dir / "marginal_utility_schedules.csv", index=False)
    if prediction_frames:
        pd.concat(prediction_frames, ignore_index=True).to_csv(args.output_dir / "marginal_utility_predictions.csv", index=False)
    if oracle_frames:
        pd.concat(oracle_frames, ignore_index=True).to_csv(args.output_dir / "marginal_utility_oracle.csv", index=False)

    manifest = {
        "generated_by": "run_portfolio_marginal_utility_ablation",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "broad_candidates": str(args.broad_candidates),
        "deployable_candidates": str(args.deployable_candidates),
        "policy_manifest": str(args.policy_manifest),
        "policy_variant": str(args.policy_variant),
        "policy_manifest_run_id": policy_payload.get("run_id"),
        "policy_params": asdict(params),
        "multipliers": list(MULTIPLIERS),
        "horizon_hours": int(args.horizon_hours),
        "embargo_hours": int(args.embargo_hours),
        "lambda_cost": float(args.lambda_cost),
        "lambda_dd": float(args.lambda_dd),
        "fold_count": int(summary["fold_id"].nunique()),
        "limitations": [
            "Counterfactual labels are generated by replaying each multiplier through the fold, not by cloning an intra-replay state object at every timestamp.",
            "Strategy-response labels replay one strategy action at a time through the fold, so they are architecture diagnostics rather than exact single-timestamp intervention labels.",
            "The oracle schedule is a headroom diagnostic and can still have replay path interactions when applied timestamp-by-timestamp.",
        ],
        "outputs": {
            "fold_summary": str(args.output_dir / "marginal_utility_fold_summary.csv"),
            "promotion_summary": str(args.output_dir / "marginal_utility_promotion_summary.csv"),
            "schedules": str(args.output_dir / "marginal_utility_schedules.csv"),
            "predictions": str(args.output_dir / "marginal_utility_predictions.csv"),
            "oracle": str(args.output_dir / "marginal_utility_oracle.csv"),
        },
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(_json_safe(manifest), indent=2) + "\n")
    print(json.dumps(_json_safe(manifest), indent=2)[:6000])
    print(f"\nWrote {args.output_dir}")


if __name__ == "__main__":
    main()
