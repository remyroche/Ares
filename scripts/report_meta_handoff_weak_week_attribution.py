#!/usr/bin/env python3
"""Weak-week attribution for GMM/meta handoff execution replay.

The report replays each validation week with EV curves fitted on prior weeks,
then attributes accepted trades by exit mode, side, oracle overlap, path MFE/MAE,
and current meta/risk scores.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

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
    "meta_handoff_weak_week_attribution"
)


def _parse_csv(value: str) -> list[str]:
    return [part.strip() for part in str(value).split(",") if part.strip()]


def _side_name(value: Any) -> str:
    text = str(value).strip().lower()
    if text in {"-1", "short", "sell"} or text.startswith("short"):
        return "short"
    try:
        return "short" if float(value) < 0.0 else "long"
    except (TypeError, ValueError):
        return "long"


def _week_start(values: pd.Series) -> pd.Series:
    ts = pd.to_datetime(values, utc=True, errors="coerce")
    return ts.dt.floor("D") - pd.to_timedelta(ts.dt.weekday, unit="D")


def _mean(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame.columns:
        return float("nan")
    values = pd.to_numeric(frame[column], errors="coerce")
    return float(values.mean()) if values.notna().any() else float("nan")


def _rate(values: pd.Series | np.ndarray) -> float:
    arr = np.asarray(values)
    return float(np.nanmean(arr.astype(float))) if arr.size else float("nan")


def _path_min(value: Any) -> float:
    try:
        arr = np.asarray(value, dtype=np.float64)
    except Exception:
        return float("nan")
    arr = arr[np.isfinite(arr)]
    return float(np.min(arr)) if arr.size else float("nan")


def _path_max(value: Any) -> float:
    try:
        arr = np.asarray(value, dtype=np.float64)
    except Exception:
        return float("nan")
    arr = arr[np.isfinite(arr)]
    return float(np.max(arr)) if arr.size else float("nan")


def _prepare_candidates(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    required = {
        "timestamp",
        "scenario",
        "symbol",
        "strategy_id",
        "net_return",
        "gross_return",
        "simple_policy_exit_reason",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")
    out = frame.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out = out.dropna(subset=["timestamp", "scenario", "symbol", "strategy_id"]).copy()
    out["scenario"] = out["scenario"].astype(str)
    out["symbol"] = out["symbol"].astype(str)
    out["strategy_id"] = out["strategy_id"].astype(str)
    out["side_name"] = out.get("side", 1.0).map(_side_name)
    out["week_start"] = _week_start(out["timestamp"])
    out["net_return"] = pd.to_numeric(out["net_return"], errors="coerce")
    out["gross_return"] = pd.to_numeric(out["gross_return"], errors="coerce")
    if "mtm_path_gross_returns" in out.columns:
        out["path_adverse_excursion"] = out["mtm_path_gross_returns"].map(_path_min).astype("float32")
        out["path_favorable_excursion"] = out["mtm_path_gross_returns"].map(_path_max).astype("float32")
    else:
        out["path_adverse_excursion"] = np.nan
        out["path_favorable_excursion"] = np.nan
    threshold = pd.to_numeric(
        out.get("policy_trailing_activation_return", out.get("barrier_pct", np.nan)),
        errors="coerce",
    )
    out["near_miss_barrier"] = (
        out["simple_policy_exit_reason"].astype(str).eq("timeout")
        & out["path_favorable_excursion"].ge(0.75 * threshold)
    )
    out["oracle_top1"] = False
    out["oracle_top3"] = False
    for (_scenario, ts), idx in out.groupby(["scenario", "timestamp"], sort=False).groups.items():
        group = out.loc[idx]
        ordered = group["net_return"].rank(method="first", ascending=False)
        out.loc[idx, "oracle_top1"] = ordered.le(1).to_numpy()
        out.loc[idx, "oracle_top3"] = ordered.le(3).to_numpy()
    return out.sort_values(["scenario", "timestamp", "strategy_id", "symbol"]).reset_index(drop=True)


def _score_with_train_curve(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    *,
    market_mode: str,
    global_threshold_floor: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if train.empty or validation.empty:
        return (
            pd.DataFrame(),
            pd.DataFrame(),
            {"objective": float("-inf"), "trade_count": 0, "net_pnl": 0.0},
        )
    ev_curve = fit_hierarchical_ev_curves(train)
    decisions, equity, metrics = replay_candidates(
        validation,
        PortfolioPolicyParams(global_threshold_floor=float(global_threshold_floor)),
        mode="global_auction",
        ev_curve=ev_curve,
        market_mode=market_mode,
    )
    return decisions, equity, dict(metrics)


def _accepted_with_candidates(decisions: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    if decisions.empty or "accepted" not in decisions.columns:
        return pd.DataFrame()
    accepted = decisions.loc[decisions["accepted"].astype(bool)].copy()
    if accepted.empty:
        return pd.DataFrame()
    accepted["candidate_index"] = pd.to_numeric(
        accepted["candidate_index"], errors="coerce"
    ).astype("int64")
    payload_cols = [
        "side_name",
        "rank_pct",
        "strategy_rank_pct",
        "calibrated_score",
        "archetype_meta_bad_risk",
        "archetype_meta_timeout_risk",
        "archetype_joint_bad_risk",
        "archetype_joint_timeout_risk",
        "expected_friction_bps",
        "entry_gap_bps",
        "entry_slippage_proxy_bps",
        "price_gap_bps",
        "holding_bars",
        "path_adverse_excursion",
        "path_favorable_excursion",
        "near_miss_barrier",
        "oracle_top1",
        "oracle_top3",
    ]
    payload_cols = [col for col in payload_cols if col in candidates.columns]
    payload = candidates[payload_cols].reset_index(names="candidate_index")
    out = accepted.merge(payload, on="candidate_index", how="left", suffixes=("", "_candidate"))
    size = pd.to_numeric(out.get("position_size", 0.0), errors="coerce").fillna(0.0)
    net = pd.to_numeric(out.get("position_net_return", 0.0), errors="coerce").fillna(0.0)
    gross = pd.to_numeric(out.get("position_gross_return", 0.0), errors="coerce").fillna(0.0)
    out["net_pnl_amount"] = size * net
    out["gross_pnl_amount"] = size * gross
    out["cost_pnl_amount"] = out["gross_pnl_amount"] - out["net_pnl_amount"]
    reason = out.get("position_exit_reason", pd.Series("", index=out.index)).astype(str)
    out["is_timeout"] = reason.eq("timeout")
    out["is_stop_or_adverse"] = reason.isin(["full_sl", "adverse_exit"])
    out["is_profit_exit"] = net.gt(0.0)
    out["is_barrier_or_protect_exit"] = reason.isin(["trailing", "hard_tp", "capital_protect"])
    return out


def _fold_row(
    *,
    scenario: str,
    fold_id: int,
    week_start: pd.Timestamp,
    train_rows: int,
    validation_rows: int,
    decisions: pd.DataFrame,
    accepted: pd.DataFrame,
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    side = accepted.get("side_name", pd.Series(dtype=str)).astype(str)
    long_mask = side.eq("long")
    short_mask = side.eq("short")
    return {
        "scenario": scenario,
        "fold_id": int(fold_id),
        "week_start": week_start.date().isoformat(),
        "train_rows": int(train_rows),
        "validation_rows": int(validation_rows),
        "selected_trades": int(metrics.get("trade_count", len(accepted)) or 0),
        "net_pnl": float(metrics.get("net_pnl", np.nan)),
        "gross_pnl": float(metrics.get("gross_pnl", np.nan)),
        "cost_pnl": float(accepted.get("cost_pnl_amount", pd.Series(dtype=float)).sum())
        if not accepted.empty
        else 0.0,
        "objective": float(metrics.get("objective", np.nan)),
        "max_drawdown": float(metrics.get("max_drawdown", np.nan)),
        "timeout_rate": float(metrics.get("timeout_rate", np.nan)),
        "barrier_hit_rate": _rate(accepted.get("is_barrier_or_protect_exit", [])),
        "stop_hit_rate": float(metrics.get("full_sl_rate", np.nan)),
        "hit_rate": _rate(accepted.get("is_profit_exit", [])),
        "mean_holding_bars": _mean(accepted, "holding_bars"),
        "mean_adverse_excursion": _mean(accepted, "path_adverse_excursion"),
        "mean_favorable_excursion": _mean(accepted, "path_favorable_excursion"),
        "near_miss_barrier_rate": _rate(accepted.get("near_miss_barrier", [])),
        "long_share": float(long_mask.mean()) if len(side) else float("nan"),
        "short_share": float(short_mask.mean()) if len(side) else float("nan"),
        "long_net_pnl": float(accepted.loc[long_mask, "net_pnl_amount"].sum())
        if not accepted.empty
        else 0.0,
        "short_net_pnl": float(accepted.loc[short_mask, "net_pnl_amount"].sum())
        if not accepted.empty
        else 0.0,
        "oracle_top1_overlap_rate": _rate(accepted.get("oracle_top1", [])),
        "oracle_top3_overlap_rate": _rate(accepted.get("oracle_top3", [])),
        "meta_score_mean": _mean(accepted, "calibrated_score"),
        "base_rank_pct_mean": _mean(accepted, "rank_pct"),
        "strategy_rank_pct_mean": _mean(accepted, "strategy_rank_pct"),
        "bad_mae_pred_mean": _mean(accepted, "archetype_joint_bad_risk"),
        "timeout_pred_mean": _mean(accepted, "archetype_joint_timeout_risk"),
        "meta_bad_risk_mean": _mean(accepted, "archetype_meta_bad_risk"),
        "meta_timeout_risk_mean": _mean(accepted, "archetype_meta_timeout_risk"),
        "expected_friction_bps_mean": _mean(accepted, "expected_friction_bps"),
        "entry_gap_bps_mean": _mean(accepted, "entry_gap_bps"),
        "price_gap_bps_mean": _mean(accepted, "price_gap_bps"),
        "candidate_rows_rejected": int(len(decisions) - len(accepted)) if not decisions.empty else 0,
        "weak_fold": bool(float(metrics.get("net_pnl", 0.0) or 0.0) < 0.0),
    }


def _bucket_rows(
    *,
    scenario: str,
    fold_id: int,
    week_start: pd.Timestamp,
    accepted: pd.DataFrame,
) -> list[dict[str, Any]]:
    if accepted.empty:
        return []
    bucket_masks = {
        "all_accepted": pd.Series(True, index=accepted.index),
        "profitable_exits": accepted["is_profit_exit"].astype(bool),
        "timeout_exits": accepted["is_timeout"].astype(bool),
        "stop_adverse_exits": accepted["is_stop_or_adverse"].astype(bool),
        "near_miss_barrier_timeout": accepted["near_miss_barrier"].astype(bool)
        if "near_miss_barrier" in accepted.columns
        else pd.Series(False, index=accepted.index),
    }
    rows: list[dict[str, Any]] = []
    for bucket, mask in bucket_masks.items():
        group = accepted.loc[mask].copy()
        if group.empty:
            continue
        rows.append(
            {
                "scenario": scenario,
                "fold_id": int(fold_id),
                "week_start": week_start.date().isoformat(),
                "bucket": bucket,
                "rows": int(len(group)),
                "net_pnl": float(group["net_pnl_amount"].sum()),
                "gross_pnl": float(group["gross_pnl_amount"].sum()),
                "cost_pnl": float(group["cost_pnl_amount"].sum()),
                "mean_net_return": _mean(group, "position_net_return"),
                "mean_gross_return": _mean(group, "position_gross_return"),
                "hit_rate": _rate(group["is_profit_exit"]),
                "timeout_rate": _rate(group["is_timeout"]),
                "stop_adverse_rate": _rate(group["is_stop_or_adverse"]),
                "oracle_top1_overlap_rate": _rate(group.get("oracle_top1", [])),
                "oracle_top3_overlap_rate": _rate(group.get("oracle_top3", [])),
                "mean_holding_bars": _mean(group, "holding_bars"),
                "mean_adverse_excursion": _mean(group, "path_adverse_excursion"),
                "mean_favorable_excursion": _mean(group, "path_favorable_excursion"),
                "meta_score_mean": _mean(group, "calibrated_score"),
                "base_rank_pct_mean": _mean(group, "rank_pct"),
                "bad_mae_pred_mean": _mean(group, "archetype_joint_bad_risk"),
                "timeout_pred_mean": _mean(group, "archetype_joint_timeout_risk"),
                "expected_friction_bps_mean": _mean(group, "expected_friction_bps"),
            }
        )
    return rows


def _summarise_folds(folds: pd.DataFrame) -> pd.DataFrame:
    if folds.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for scenario, group in folds.groupby("scenario", sort=True):
        net = pd.to_numeric(group["net_pnl"], errors="coerce").fillna(0.0)
        trades = pd.to_numeric(group["selected_trades"], errors="coerce").fillna(0.0)
        total_trades = float(trades.sum())
        rows.append(
            {
                "scenario": scenario,
                "folds": int(len(group)),
                "sum_net_pnl": float(net.sum()),
                "mean_objective": float(pd.to_numeric(group["objective"], errors="coerce").mean()),
                "positive_folds": int(net.gt(0.0).sum()),
                "positive_fold_share": float(net.gt(0.0).mean()),
                "worst_fold_net_pnl": float(net.min()),
                "selected_trades": int(total_trades),
                "weighted_timeout_rate": float(
                    (pd.to_numeric(group["timeout_rate"], errors="coerce").fillna(0.0) * trades).sum()
                    / max(total_trades, 1.0)
                ),
                "weighted_stop_rate": float(
                    (pd.to_numeric(group["stop_hit_rate"], errors="coerce").fillna(0.0) * trades).sum()
                    / max(total_trades, 1.0)
                ),
                "mean_oracle_top1_overlap_rate": float(
                    pd.to_numeric(group["oracle_top1_overlap_rate"], errors="coerce").mean()
                ),
                "mean_oracle_top3_overlap_rate": float(
                    pd.to_numeric(group["oracle_top3_overlap_rate"], errors="coerce").mean()
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("sum_net_pnl", ascending=False).reset_index(drop=True)


def _fmt_table(frame: pd.DataFrame, cols: list[str], max_rows: int = 40) -> str:
    if frame.empty:
        return "_No rows._"
    view = frame[cols].head(int(max_rows)).copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{x:,.4f}")
    return view.to_markdown(index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--scenarios", default="h9_delay_1_barrier_x4,h12_delay_1_barrier_x3")
    parser.add_argument("--market-mode", default="perps", choices=["spot", "perps"])
    parser.add_argument("--global-threshold-floor", type=float, default=0.0)
    parser.add_argument("--min-train-weeks", type=int, default=2)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    candidates = _prepare_candidates(args.candidates)
    scenarios = _parse_csv(args.scenarios)
    fold_rows: list[dict[str, Any]] = []
    bucket_rows: list[dict[str, Any]] = []
    accepted_frames: list[pd.DataFrame] = []
    for scenario in scenarios:
        scenario_frame = candidates.loc[candidates["scenario"].eq(scenario)].copy().reset_index(drop=True)
        if scenario_frame.empty:
            continue
        weeks = sorted(pd.to_datetime(scenario_frame["week_start"], utc=True).dropna().unique())
        for fold_id, week_value in enumerate(weeks[int(args.min_train_weeks) :]):
            week_start = pd.Timestamp(week_value)
            week_end = week_start + pd.Timedelta(days=7)
            train = scenario_frame.loc[scenario_frame["timestamp"].lt(week_start)].copy().reset_index(drop=True)
            validation = scenario_frame.loc[
                scenario_frame["timestamp"].ge(week_start)
                & scenario_frame["timestamp"].lt(week_end)
            ].copy().reset_index(drop=True)
            decisions, _equity, metrics = _score_with_train_curve(
                train,
                validation,
                market_mode=str(args.market_mode),
                global_threshold_floor=float(args.global_threshold_floor),
            )
            accepted = _accepted_with_candidates(decisions, validation)
            fold_rows.append(
                _fold_row(
                    scenario=scenario,
                    fold_id=fold_id,
                    week_start=week_start,
                    train_rows=len(train),
                    validation_rows=len(validation),
                    decisions=decisions,
                    accepted=accepted,
                    metrics=metrics,
                )
            )
            current_buckets = _bucket_rows(
                scenario=scenario,
                fold_id=fold_id,
                week_start=week_start,
                accepted=accepted,
            )
            bucket_rows.extend(current_buckets)
            if not accepted.empty:
                accepted = accepted.copy()
                accepted["scenario"] = scenario
                accepted["fold_id"] = int(fold_id)
                accepted["week_start"] = week_start.date().isoformat()
                accepted_frames.append(accepted)

    folds_df = pd.DataFrame(fold_rows)
    bucket_df = pd.DataFrame(bucket_rows)
    summary_df = _summarise_folds(folds_df)
    accepted_df = (
        pd.concat(accepted_frames, ignore_index=True) if accepted_frames else pd.DataFrame()
    )
    weak_bucket_df = pd.DataFrame()
    if not folds_df.empty and not bucket_df.empty:
        weak_keys = folds_df.loc[
            folds_df["weak_fold"].astype(bool), ["scenario", "fold_id"]
        ].drop_duplicates()
        weak_bucket_df = bucket_df.merge(weak_keys, on=["scenario", "fold_id"], how="inner")

    paths = {
        "summary": args.out_dir / "weak_week_attribution_summary.csv",
        "folds": args.out_dir / "weak_week_attribution_by_fold.csv",
        "exit_buckets": args.out_dir / "weak_week_attribution_exit_buckets.csv",
        "weak_exit_buckets": args.out_dir / "weak_week_attribution_losing_fold_exit_buckets.csv",
        "accepted_rows": args.out_dir / "weak_week_attribution_accepted_rows.parquet",
        "manifest": args.out_dir / "manifest.json",
        "report": args.out_dir / "weak_week_attribution_report.md",
    }
    summary_df.to_csv(paths["summary"], index=False)
    folds_df.to_csv(paths["folds"], index=False)
    bucket_df.to_csv(paths["exit_buckets"], index=False)
    weak_bucket_df.to_csv(paths["weak_exit_buckets"], index=False)
    accepted_df.to_parquet(paths["accepted_rows"], index=False)
    manifest = {
        "generated_by": "report_meta_handoff_weak_week_attribution",
        "candidates": str(args.candidates),
        "out_dir": str(args.out_dir),
        "scenarios": scenarios,
        "market_mode": str(args.market_mode),
        "global_threshold_floor": float(args.global_threshold_floor),
        "min_train_weeks": int(args.min_train_weeks),
        "outputs": {key: str(value) for key, value in paths.items()},
    }
    paths["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    lines = [
        "# Meta Handoff Weak-Week Attribution",
        "",
        "Weekly replay uses only prior weeks for EV-curve fitting. Oracle overlap is computed inside the validation-week candidate universe for each timestamp.",
        "",
        "## Scenario Summary",
        "",
        _fmt_table(
            summary_df,
            [
                "scenario",
                "folds",
                "sum_net_pnl",
                "mean_objective",
                "positive_fold_share",
                "worst_fold_net_pnl",
                "selected_trades",
                "weighted_timeout_rate",
                "weighted_stop_rate",
                "mean_oracle_top1_overlap_rate",
                "mean_oracle_top3_overlap_rate",
            ],
        ),
        "",
        "## Fold Attribution",
        "",
        _fmt_table(
            folds_df,
            [
                "scenario",
                "week_start",
                "weak_fold",
                "selected_trades",
                "net_pnl",
                "gross_pnl",
                "cost_pnl",
                "timeout_rate",
                "barrier_hit_rate",
                "stop_hit_rate",
                "mean_holding_bars",
                "mean_adverse_excursion",
                "mean_favorable_excursion",
                "long_share",
                "long_net_pnl",
                "short_net_pnl",
                "oracle_top1_overlap_rate",
                "meta_score_mean",
                "bad_mae_pred_mean",
                "timeout_pred_mean",
            ],
            max_rows=80,
        ),
        "",
        "## Losing-Fold Exit Buckets",
        "",
        _fmt_table(
            weak_bucket_df,
            [
                "scenario",
                "week_start",
                "bucket",
                "rows",
                "net_pnl",
                "gross_pnl",
                "cost_pnl",
                "hit_rate",
                "timeout_rate",
                "stop_adverse_rate",
                "oracle_top1_overlap_rate",
                "mean_holding_bars",
                "mean_adverse_excursion",
                "mean_favorable_excursion",
                "meta_score_mean",
                "bad_mae_pred_mean",
                "timeout_pred_mean",
            ],
            max_rows=120,
        ),
    ]
    paths["report"].write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            _json_safe(
                {
                    "summary": summary_df.to_dict(orient="records"),
                    "outputs": {key: str(value) for key, value in paths.items()},
                }
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
