#!/usr/bin/env python3
"""Walk-forward execution guard for GMM/meta handoff replay candidates.

This validates whether current meta handoff columns can identify executable
replay winners after costs. Each validation week is held out from model fitting,
guard threshold selection, and EV-curve fitting.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.portfolio_policy_replay import (  # noqa: E402
    PortfolioPolicyParams,
    fit_hierarchical_ev_curves,
    replay_candidates,
)
from extreme_price_movements.simple_policy_optimiser import _json_safe  # noqa: E402


DEFAULT_CANDIDATES = Path(
    "data_perp/reports/gmm_cluster_policy_smoke_20260702_wide_sidebalanced/"
    "meta_handoff_execution_horizon_sweep_rank75_bad45_to12h_v1/"
    "meta_handoff_replay_attribution_candidates.parquet"
)
DEFAULT_OUT_DIR = Path(
    "data_perp/reports/gmm_cluster_policy_smoke_20260702_wide_sidebalanced/"
    "meta_handoff_execution_guard_walkforward"
)

BASE_FEATURE_COLUMNS = (
    "rank_pct",
    "strategy_rank_pct",
    "normalized_rank_score",
    "calibrated_score",
    "barrier_pct",
    "policy_effective_barrier_pct",
    "policy_sl_return",
    "policy_trailing_activation_return",
    "policy_uncapped_trailing_activation_return",
    "oof_regime_centroid_similarity_train",
    "archetype_meta_bad_risk",
    "archetype_meta_timeout_risk",
    "archetype_joint_bad_risk",
    "archetype_joint_timeout_risk",
)
EXECUTION_KNOWN_FEATURE_COLUMNS = (
    "expected_spread_bps",
    "expected_half_spread_bps",
    "exit_spread_cost_bps",
    "expected_friction_bps",
    "entry_reanchor_bps",
    "entry_gap_bps",
    "entry_slippage_proxy_bps",
    "price_gap_bps",
    "liquidity_capacity_weight",
    "entry_delay_actual_minutes",
    "delay_window_range_bps",
    "delay_entry_ref_gap_bps",
    "delay_close_gap_bps",
    "delay_max_adverse_bps",
    "delay_max_favorable_bps",
)
METHODS = (
    "risk_composite_rule",
    "exec_net_regressor",
    "clean_exit_classifier",
    "bad_exit_veto",
)


@dataclass(frozen=True)
class FoldResult:
    scenario: str
    fold_id: int
    validation_week: str
    variant: str
    train_rows: int
    validation_rows: int
    keep_frac: float
    score_threshold: float
    filtered_validation_rows: int
    accepted_trades: int
    objective: float
    net_pnl: float
    gross_pnl: float
    compounded_return: float
    max_drawdown: float
    full_sl_rate: float
    timeout_rate: float
    hit_rate: float
    candidate_mean_net_return: float
    candidate_hit_rate: float
    train_selector_score: float


def _side_code(value: Any) -> float:
    text = str(value).strip().lower()
    if text in {"-1", "short", "sell"} or text.startswith("short"):
        return -1.0
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 1.0
    return -1.0 if numeric < 0.0 else 1.0


def _week_start(values: pd.Series) -> pd.Series:
    ts = pd.to_datetime(values, utc=True, errors="coerce")
    return ts.dt.floor("D") - pd.to_timedelta(ts.dt.weekday, unit="D")


def _available_features(frame: pd.DataFrame, *, feature_mode: str) -> list[str]:
    columns = list(BASE_FEATURE_COLUMNS)
    if feature_mode == "execution_known":
        columns.extend(EXECUTION_KNOWN_FEATURE_COLUMNS)
    return [col for col in columns if col in frame.columns]


def _numeric_column(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce")


def _prepare_frame(path: Path, *, feature_mode: str) -> tuple[pd.DataFrame, list[str]]:
    frame = pd.read_parquet(path)
    required = {"timestamp", "scenario", "symbol", "strategy_id", "net_return"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")
    out = frame.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out = out.dropna(subset=["timestamp", "scenario", "symbol", "strategy_id"]).copy()
    out["scenario"] = out["scenario"].astype(str)
    out["symbol"] = out["symbol"].astype(str)
    out["strategy_id"] = out["strategy_id"].astype(str)
    out["side_code"] = out.get("side", 1.0).map(_side_code).astype("float32")
    out["is_short"] = (out["side_code"] < 0.0).astype("float32")
    out["week_start"] = _week_start(out["timestamp"])
    out["net_return"] = pd.to_numeric(out["net_return"], errors="coerce")
    out["gross_return"] = pd.to_numeric(out.get("gross_return", np.nan), errors="coerce")
    reason = out.get("simple_policy_exit_reason", pd.Series("", index=out.index)).astype(str)
    out["is_full_sl"] = reason.eq("full_sl")
    out["is_timeout"] = reason.eq("timeout")
    out["is_timeout_loss"] = out["is_timeout"] & out["net_return"].lt(0.0)
    out["clean_executable"] = (
        out["net_return"].gt(0.0) & ~out["is_full_sl"] & ~out["is_timeout_loss"]
    )
    out["bad_executable"] = out["is_full_sl"] | out["is_timeout_loss"] | out["net_return"].lt(0.0)
    out["rank_minus_joint_bad"] = (
        _numeric_column(out, "rank_pct", 0.0).fillna(0.0)
        - _numeric_column(out, "archetype_joint_bad_risk", 0.0).fillna(0.0)
    ).astype("float32")
    out["rank_minus_joint_timeout"] = (
        _numeric_column(out, "rank_pct", 0.0).fillna(0.0)
        - _numeric_column(out, "archetype_joint_timeout_risk", 0.0).fillna(0.0)
    ).astype("float32")
    features = _available_features(out, feature_mode=feature_mode)
    features.extend(["side_code", "is_short", "rank_minus_joint_bad", "rank_minus_joint_timeout"])
    features = [col for col in dict.fromkeys(features) if col in out.columns]
    for col in features:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.sort_values(["scenario", "timestamp", "strategy_id", "symbol"]).reset_index(drop=True), features


def _finite_imputed_matrix(
    train: pd.DataFrame,
    eval_frame: pd.DataFrame,
    features: Iterable[str],
) -> tuple[np.ndarray, np.ndarray]:
    cols = list(features)
    x_train = train[cols].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=np.float64)
    x_eval = eval_frame[cols].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=np.float64)
    med = np.nanmedian(x_train, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    train_nan = ~np.isfinite(x_train)
    eval_nan = ~np.isfinite(x_eval)
    if train_nan.any():
        x_train[train_nan] = np.take(med, np.where(train_nan)[1])
    if eval_nan.any():
        x_eval[eval_nan] = np.take(med, np.where(eval_nan)[1])
    return x_train.astype(np.float32), x_eval.astype(np.float32)


def _constant_scores(train: pd.DataFrame, eval_frame: pd.DataFrame, value: float) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.full(len(train), float(value), dtype=np.float32),
        np.full(len(eval_frame), float(value), dtype=np.float32),
    )


def _fit_predict_scores(
    method: str,
    train: pd.DataFrame,
    eval_frame: pd.DataFrame,
    features: list[str],
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, str]:
    if method == "risk_composite_rule":
        rank = _numeric_column(train, "rank_pct", 0.0).fillna(0.0)
        bad = _numeric_column(train, "archetype_joint_bad_risk", 0.0).fillna(0.0)
        timeout = _numeric_column(train, "archetype_joint_timeout_risk", 0.0).fillna(0.0)
        train_score = rank - 0.75 * bad - 0.35 * timeout
        rank_v = _numeric_column(eval_frame, "rank_pct", 0.0).fillna(0.0)
        bad_v = _numeric_column(eval_frame, "archetype_joint_bad_risk", 0.0).fillna(0.0)
        timeout_v = _numeric_column(eval_frame, "archetype_joint_timeout_risk", 0.0).fillna(0.0)
        eval_score = rank_v - 0.75 * bad_v - 0.35 * timeout_v
        return (
            train_score.to_numpy(dtype=np.float32),
            eval_score.to_numpy(dtype=np.float32),
            "deterministic_score",
        )

    if len(train) < 20 or not features:
        return (*_constant_scores(train, eval_frame, 0.0), "constant_insufficient_rows")

    try:
        from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
    except Exception:
        return (*_constant_scores(train, eval_frame, 0.0), "constant_sklearn_unavailable")

    x_train, x_eval = _finite_imputed_matrix(train, eval_frame, features)
    if method == "exec_net_regressor":
        y = pd.to_numeric(train["net_return"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        if np.nanstd(y) < 1e-9:
            return (*_constant_scores(train, eval_frame, float(np.nanmean(y))), "constant_target")
        model = GradientBoostingRegressor(
            n_estimators=80,
            learning_rate=0.04,
            max_depth=2,
            min_samples_leaf=max(8, int(len(train) * 0.03)),
            random_state=int(seed),
        )
        model.fit(x_train, y)
        return (
            model.predict(x_train).astype(np.float32),
            model.predict(x_eval).astype(np.float32),
            "gradient_boosting_regressor",
        )

    target_col = "clean_executable" if method == "clean_exit_classifier" else "bad_executable"
    y_bool = train[target_col].astype(bool).to_numpy()
    if len(np.unique(y_bool)) < 2:
        value = float(y_bool.mean())
        score_value = value if method == "clean_exit_classifier" else -value
        return (*_constant_scores(train, eval_frame, score_value), "constant_one_class")
    model = GradientBoostingClassifier(
        n_estimators=80,
        learning_rate=0.04,
        max_depth=2,
        min_samples_leaf=max(8, int(len(train) * 0.03)),
        random_state=int(seed),
    )
    model.fit(x_train, y_bool.astype(np.int8))
    train_prob = model.predict_proba(x_train)[:, 1].astype(np.float32)
    eval_prob = model.predict_proba(x_eval)[:, 1].astype(np.float32)
    if method == "bad_exit_veto":
        return -train_prob, -eval_prob, "gradient_boosting_bad_exit_classifier"
    return train_prob, eval_prob, "gradient_boosting_clean_exit_classifier"


def _filter_by_train_quantile(
    frame: pd.DataFrame,
    scores: np.ndarray,
    *,
    threshold: float,
) -> pd.DataFrame:
    mask = np.asarray(scores, dtype=np.float64) >= float(threshold)
    return frame.loc[mask].reset_index(drop=True).copy()


def _threshold_for_keep_frac(train_scores: np.ndarray, keep_frac: float) -> float:
    values = np.asarray(train_scores, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("-inf")
    keep = min(max(float(keep_frac), 0.0), 1.0)
    if keep >= 1.0:
        return float("-inf")
    if keep <= 0.0:
        return float("inf")
    return float(np.quantile(values, 1.0 - keep))


def _replay_with_train_curve(
    *,
    train_candidates: pd.DataFrame,
    eval_candidates: pd.DataFrame,
    market_mode: str,
    global_threshold_floor: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if train_candidates.empty or eval_candidates.empty:
        return (
            pd.DataFrame(),
            pd.DataFrame(),
            {"objective": float("-inf"), "trade_count": 0, "net_pnl": 0.0},
        )
    ev_curve = fit_hierarchical_ev_curves(train_candidates)
    decisions, equity, metrics = replay_candidates(
        eval_candidates,
        PortfolioPolicyParams(global_threshold_floor=float(global_threshold_floor)),
        mode="global_auction",
        ev_curve=ev_curve,
        market_mode=market_mode,
    )
    return decisions, equity, dict(metrics)


def _accepted_hit_rate(decisions: pd.DataFrame) -> float:
    if decisions.empty or "accepted" not in decisions.columns:
        return float("nan")
    accepted = decisions.loc[decisions["accepted"].astype(bool)].copy()
    if accepted.empty:
        return float("nan")
    return float(
        pd.to_numeric(accepted["position_net_return"], errors="coerce")
        .fillna(0.0)
        .gt(0.0)
        .mean()
    )


def _candidate_stats(frame: pd.DataFrame) -> tuple[float, float]:
    if frame.empty:
        return float("nan"), float("nan")
    net = pd.to_numeric(frame["net_return"], errors="coerce")
    return float(net.mean()), float(net.gt(0.0).mean())


def _fold_result(
    *,
    scenario: str,
    fold_id: int,
    validation_week: str,
    variant: str,
    train_rows: int,
    validation_rows: int,
    keep_frac: float,
    score_threshold: float,
    filtered_validation: pd.DataFrame,
    decisions: pd.DataFrame,
    metrics: Mapping[str, Any],
    train_selector_score: float,
) -> FoldResult:
    candidate_mean, candidate_hit = _candidate_stats(filtered_validation)
    return FoldResult(
        scenario=scenario,
        fold_id=int(fold_id),
        validation_week=str(validation_week),
        variant=str(variant),
        train_rows=int(train_rows),
        validation_rows=int(validation_rows),
        keep_frac=float(keep_frac),
        score_threshold=float(score_threshold),
        filtered_validation_rows=int(len(filtered_validation)),
        accepted_trades=int(metrics.get("trade_count", 0) or 0),
        objective=float(metrics.get("objective", np.nan)),
        net_pnl=float(metrics.get("net_pnl", np.nan)),
        gross_pnl=float(metrics.get("gross_pnl", np.nan)),
        compounded_return=float(metrics.get("compounded_return", np.nan)),
        max_drawdown=float(metrics.get("max_drawdown", np.nan)),
        full_sl_rate=float(metrics.get("full_sl_rate", np.nan)),
        timeout_rate=float(metrics.get("timeout_rate", np.nan)),
        hit_rate=_accepted_hit_rate(decisions),
        candidate_mean_net_return=candidate_mean,
        candidate_hit_rate=candidate_hit,
        train_selector_score=float(train_selector_score),
    )


def _selector_score(metrics: Mapping[str, Any], *, min_trades: int) -> float:
    trades = int(metrics.get("trade_count", 0) or 0)
    if trades < int(min_trades):
        return -1.0e12 + trades
    objective = float(metrics.get("objective", -np.inf))
    net_pnl = float(metrics.get("net_pnl", 0.0))
    full_sl = float(metrics.get("full_sl_rate", 1.0))
    timeout = float(metrics.get("timeout_rate", 1.0))
    drawdown = abs(float(metrics.get("max_drawdown", 0.0) or 0.0))
    return float(objective + 0.00002 * net_pnl - 0.08 * full_sl - 0.03 * timeout - 0.25 * drawdown)


def _select_keep_fraction(
    train: pd.DataFrame,
    train_scores: np.ndarray,
    keep_fracs: Iterable[float],
    *,
    market_mode: str,
    global_threshold_floor: float,
    min_train_trades: int,
) -> tuple[float, float, float, dict[str, Any]]:
    best: tuple[float, float, float, dict[str, Any]] | None = None
    for keep_frac in keep_fracs:
        threshold = _threshold_for_keep_frac(train_scores, float(keep_frac))
        filtered = _filter_by_train_quantile(train, train_scores, threshold=threshold)
        decisions, _equity, metrics = _replay_with_train_curve(
            train_candidates=train,
            eval_candidates=filtered,
            market_mode=market_mode,
            global_threshold_floor=global_threshold_floor,
        )
        score = _selector_score(metrics, min_trades=int(min_train_trades))
        current = (float(score), float(keep_frac), float(threshold), dict(metrics))
        if best is None or current[0] > best[0]:
            best = current
    if best is None:
        return 1.0, float("-inf"), -1.0e12, {}
    score, keep_frac, threshold, metrics = best
    return keep_frac, threshold, score, metrics


def _scenario_folds(frame: pd.DataFrame, *, min_train_weeks: int) -> list[tuple[int, pd.Timestamp]]:
    weeks = sorted(pd.to_datetime(frame["week_start"], utc=True).dropna().unique())
    out: list[tuple[int, pd.Timestamp]] = []
    for fold_id, week in enumerate(weeks[int(min_train_weeks) :], start=0):
        out.append((fold_id, pd.Timestamp(week)))
    return out


def _summarise(folds: pd.DataFrame, *, thresholds: Mapping[str, float]) -> pd.DataFrame:
    if folds.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for (scenario, variant), group in folds.groupby(["scenario", "variant"], sort=True):
        accepted = pd.to_numeric(group["accepted_trades"], errors="coerce").fillna(0.0)
        net = pd.to_numeric(group["net_pnl"], errors="coerce").fillna(0.0)
        objective = pd.to_numeric(group["objective"], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
        total_trades = float(accepted.sum())
        full_sl = pd.to_numeric(group["full_sl_rate"], errors="coerce")
        timeout = pd.to_numeric(group["timeout_rate"], errors="coerce")
        weighted_full_sl = float((full_sl.fillna(0.0) * accepted).sum() / max(total_trades, 1.0))
        weighted_timeout = float((timeout.fillna(0.0) * accepted).sum() / max(total_trades, 1.0))
        no_trade_folds = int(accepted.eq(0).sum())
        positive_folds = int(net.gt(0.0).sum())
        folds_count = int(len(group))
        pass_gate = (
            float(net.sum()) > float(thresholds["min_net_pnl"])
            and float(objective.mean(skipna=True)) > float(thresholds["min_mean_objective"])
            and positive_folds / max(folds_count, 1) >= float(thresholds["min_positive_fold_share"])
            and weighted_full_sl <= float(thresholds["max_full_sl_rate"])
            and weighted_timeout <= float(thresholds["max_timeout_rate"])
            and float(net.min()) >= float(thresholds["min_worst_fold_net_pnl"])
            and no_trade_folds <= int(thresholds["max_no_trade_folds"])
        )
        rows.append(
            {
                "scenario": scenario,
                "variant": variant,
                "folds": folds_count,
                "sum_net_pnl": float(net.sum()),
                "mean_objective": float(objective.mean(skipna=True)),
                "worst_fold_net_pnl": float(net.min()) if len(net) else float("nan"),
                "positive_folds": positive_folds,
                "positive_fold_share": float(positive_folds / max(folds_count, 1)),
                "no_trade_folds": no_trade_folds,
                "accepted_trades": int(total_trades),
                "mean_accepted_trades": float(accepted.mean()),
                "mean_keep_frac": float(
                    pd.to_numeric(group["keep_frac"], errors="coerce").mean(skipna=True)
                ),
                "weighted_full_sl_rate": weighted_full_sl,
                "weighted_timeout_rate": weighted_timeout,
                "mean_hit_rate": float(pd.to_numeric(group["hit_rate"], errors="coerce").mean(skipna=True)),
                "worst_max_drawdown": float(pd.to_numeric(group["max_drawdown"], errors="coerce").min(skipna=True)),
                "mean_candidate_net_return": float(
                    pd.to_numeric(group["candidate_mean_net_return"], errors="coerce").mean(skipna=True)
                ),
                "pass_simple_policy_gate": bool(pass_gate),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(
            [
                "pass_simple_policy_gate",
                "sum_net_pnl",
                "positive_fold_share",
                "weighted_timeout_rate",
            ],
            ascending=[False, False, False, True],
        ).reset_index(drop=True)
    return out


def _fmt_table(frame: pd.DataFrame, cols: list[str], max_rows: int = 20) -> str:
    if frame.empty:
        return "_No rows._"
    view = frame[cols].head(int(max_rows)).copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{x:,.4f}")
    return view.to_markdown(index=False)


def _parse_csv(value: str) -> list[str]:
    return [part.strip() for part in str(value).split(",") if part.strip()]


def _parse_float_grid(value: str) -> list[float]:
    out = [float(part.strip()) for part in str(value).split(",") if part.strip()]
    return sorted({float(min(max(v, 0.0), 1.0)) for v in out})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--scenarios",
        default="h9_delay_1_barrier_x4,h10_delay_1_barrier_x3,h12_delay_1_barrier_x3",
    )
    parser.add_argument(
        "--feature-mode",
        choices=["base", "execution_known"],
        default="execution_known",
    )
    parser.add_argument("--market-mode", default="perps", choices=["spot", "perps"])
    parser.add_argument("--global-threshold-floor", type=float, default=0.0)
    parser.add_argument("--min-train-weeks", type=int, default=2)
    parser.add_argument("--min-train-trades", type=int, default=12)
    parser.add_argument("--seed", type=int, default=104729)
    parser.add_argument("--keep-fracs", default="0.25,0.35,0.50,0.65,0.80,1.00")
    parser.add_argument("--min-net-pnl", type=float, default=0.0)
    parser.add_argument("--min-mean-objective", type=float, default=0.0)
    parser.add_argument("--min-positive-fold-share", type=float, default=0.67)
    parser.add_argument("--max-full-sl-rate", type=float, default=0.22)
    parser.add_argument("--max-timeout-rate", type=float, default=0.55)
    parser.add_argument("--min-worst-fold-net-pnl", type=float, default=-250.0)
    parser.add_argument("--max-no-trade-folds", type=int, default=0)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    frame, features = _prepare_frame(args.candidates, feature_mode=str(args.feature_mode))
    scenarios = _parse_csv(args.scenarios)
    keep_fracs = _parse_float_grid(args.keep_fracs)
    thresholds = {
        "min_net_pnl": float(args.min_net_pnl),
        "min_mean_objective": float(args.min_mean_objective),
        "min_positive_fold_share": float(args.min_positive_fold_share),
        "max_full_sl_rate": float(args.max_full_sl_rate),
        "max_timeout_rate": float(args.max_timeout_rate),
        "min_worst_fold_net_pnl": float(args.min_worst_fold_net_pnl),
        "max_no_trade_folds": int(args.max_no_trade_folds),
    }

    fold_rows: list[FoldResult] = []
    prediction_frames: list[pd.DataFrame] = []
    train_selection_rows: list[dict[str, Any]] = []
    for scenario in scenarios:
        scenario_frame = frame.loc[frame["scenario"].eq(scenario)].copy().reset_index(drop=True)
        if scenario_frame.empty:
            continue
        folds = _scenario_folds(scenario_frame, min_train_weeks=int(args.min_train_weeks))
        for fold_id, validation_week in folds:
            validation_end = validation_week + pd.Timedelta(days=7)
            train = scenario_frame.loc[scenario_frame["timestamp"].lt(validation_week)].copy().reset_index(drop=True)
            validation = scenario_frame.loc[
                scenario_frame["timestamp"].ge(validation_week)
                & scenario_frame["timestamp"].lt(validation_end)
            ].copy().reset_index(drop=True)
            if train.empty or validation.empty:
                continue
            baseline_decisions, _baseline_equity, baseline_metrics = _replay_with_train_curve(
                train_candidates=train,
                eval_candidates=validation,
                market_mode=str(args.market_mode),
                global_threshold_floor=float(args.global_threshold_floor),
            )
            fold_rows.append(
                _fold_result(
                    scenario=scenario,
                    fold_id=fold_id,
                    validation_week=validation_week.date().isoformat(),
                    variant="baseline",
                    train_rows=len(train),
                    validation_rows=len(validation),
                    keep_frac=1.0,
                    score_threshold=float("-inf"),
                    filtered_validation=validation,
                    decisions=baseline_decisions,
                    metrics=baseline_metrics,
                    train_selector_score=float("nan"),
                )
            )
            prediction_frame = validation[
                [
                    "timestamp",
                    "symbol",
                    "side",
                    "strategy_id",
                    "scenario",
                    "week_start",
                    "net_return",
                    "simple_policy_exit_reason",
                ]
            ].copy()
            for method in METHODS:
                train_scores, validation_scores, model_name = _fit_predict_scores(
                    method,
                    train,
                    validation,
                    features,
                    seed=int(args.seed) + int(fold_id),
                )
                keep_frac, threshold, selector_score, train_metrics = _select_keep_fraction(
                    train,
                    train_scores,
                    keep_fracs,
                    market_mode=str(args.market_mode),
                    global_threshold_floor=float(args.global_threshold_floor),
                    min_train_trades=int(args.min_train_trades),
                )
                filtered_validation = _filter_by_train_quantile(
                    validation,
                    validation_scores,
                    threshold=threshold,
                )
                decisions, _equity, metrics = _replay_with_train_curve(
                    train_candidates=train,
                    eval_candidates=filtered_validation,
                    market_mode=str(args.market_mode),
                    global_threshold_floor=float(args.global_threshold_floor),
                )
                variant = method
                fold_rows.append(
                    _fold_result(
                        scenario=scenario,
                        fold_id=fold_id,
                        validation_week=validation_week.date().isoformat(),
                        variant=variant,
                        train_rows=len(train),
                        validation_rows=len(validation),
                        keep_frac=keep_frac,
                        score_threshold=threshold,
                        filtered_validation=filtered_validation,
                        decisions=decisions,
                        metrics=metrics,
                        train_selector_score=selector_score,
                    )
                )
                prediction_frame[f"{method}_score"] = validation_scores.astype(np.float32)
                prediction_frame[f"{method}_threshold"] = np.float32(threshold)
                prediction_frame[f"{method}_keep"] = (
                    validation_scores >= float(threshold)
                ).astype(np.float32)
                train_selection_rows.append(
                    {
                        "scenario": scenario,
                        "fold_id": int(fold_id),
                        "validation_week": validation_week.date().isoformat(),
                        "method": method,
                        "variant": variant,
                        "model_name": model_name,
                        "keep_frac": keep_frac,
                        "score_threshold": threshold,
                        "train_selector_score": selector_score,
                        "train_objective": float(train_metrics.get("objective", np.nan)),
                        "train_net_pnl": float(train_metrics.get("net_pnl", np.nan)),
                        "train_trade_count": int(train_metrics.get("trade_count", 0) or 0),
                        "train_full_sl_rate": float(train_metrics.get("full_sl_rate", np.nan)),
                        "train_timeout_rate": float(train_metrics.get("timeout_rate", np.nan)),
                    }
                )
            prediction_frames.append(prediction_frame)

    folds_df = pd.DataFrame([row.__dict__ for row in fold_rows])
    train_selection_df = pd.DataFrame(train_selection_rows)
    predictions_df = (
        pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    )
    summary_df = _summarise(folds_df, thresholds=thresholds)

    paths = {
        "folds": args.out_dir / "execution_guard_walkforward_folds.csv",
        "summary": args.out_dir / "execution_guard_walkforward_summary.csv",
        "train_selection": args.out_dir / "execution_guard_train_selection.csv",
        "predictions": args.out_dir / "execution_guard_validation_predictions.parquet",
        "manifest": args.out_dir / "manifest.json",
        "report": args.out_dir / "execution_guard_walkforward_report.md",
    }
    folds_df.to_csv(paths["folds"], index=False)
    summary_df.to_csv(paths["summary"], index=False)
    train_selection_df.to_csv(paths["train_selection"], index=False)
    predictions_df.to_parquet(paths["predictions"], index=False)
    manifest = {
        "generated_by": "validate_meta_handoff_execution_guard_walkforward",
        "candidates": str(args.candidates),
        "out_dir": str(args.out_dir),
        "scenarios": scenarios,
        "feature_mode": str(args.feature_mode),
        "feature_columns": features,
        "methods": list(METHODS),
        "keep_fracs": keep_fracs,
        "thresholds": thresholds,
        "market_mode": str(args.market_mode),
        "global_threshold_floor": float(args.global_threshold_floor),
        "min_train_weeks": int(args.min_train_weeks),
        "min_train_trades": int(args.min_train_trades),
        "seed": int(args.seed),
        "outputs": {key: str(value) for key, value in paths.items()},
    }
    paths["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    lines = [
        "# Meta Handoff Execution Guard Walk-Forward",
        "",
        "Weekly chronological validation. Each guard model and keep fraction is selected on prior weeks only; validation replay uses EV curves fitted on prior weeks only.",
        "",
        "## Gate Thresholds",
        "",
        _fmt_table(pd.DataFrame([thresholds]), list(thresholds)),
        "",
        "## Summary",
        "",
        _fmt_table(
            summary_df,
            [
                "scenario",
                "variant",
                "pass_simple_policy_gate",
                "sum_net_pnl",
                "mean_objective",
                "worst_fold_net_pnl",
                "positive_fold_share",
                "accepted_trades",
                "mean_keep_frac",
                "weighted_full_sl_rate",
                "weighted_timeout_rate",
            ],
            max_rows=40,
        ),
        "",
        "## Fold Detail",
        "",
        _fmt_table(
            folds_df,
            [
                "scenario",
                "validation_week",
                "variant",
                "net_pnl",
                "objective",
                "accepted_trades",
                "full_sl_rate",
                "timeout_rate",
                "hit_rate",
                "keep_frac",
            ],
            max_rows=80,
        ),
    ]
    paths["report"].write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            _json_safe(
                {
                    "best": summary_df.head(5).to_dict(orient="records"),
                    "outputs": {key: str(value) for key, value in paths.items()},
                }
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
