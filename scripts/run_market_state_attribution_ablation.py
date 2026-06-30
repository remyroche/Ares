#!/usr/bin/env python3
"""Fixed-universe attribution ablation for market-state threshold control.

This runner separates three effects that were previously entangled in the
June replay:

* legacy versus repaired short_boll rank/eligibility contract;
* observed-state threshold suppression on top of the repaired contract;
* incremental forecast and latent state features.

The script deliberately reuses the already validated market-state controller
helpers.  It writes the T0-T5 replay tables, a period-recognition audit, and
controller action cohort audits under a new report directory.
"""

from __future__ import annotations

import argparse
import json
import sys
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
    replay_candidates,
)
from scripts import materialize_market_state_controller_bundle as bundle_builder  # noqa: E402
from scripts import run_market_state_threshold_controller as mstc  # noqa: E402


DEFAULT_OUTPUT_DIR = Path(
    "data_perp/reports/market_state_attribution_ablation_20260625"
)


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


def _load_candidates_for_contract(
    path: Path,
    *,
    rank_contract: str,
    disabled_heads: set[str],
) -> pd.DataFrame:
    return mstc._disable_heads(
        mstc._apply_rank_contract(mstc._load_candidates(path), rank_contract),
        disabled_heads,
    )


def _state_subset(
    states: dict[str, tuple[pd.DataFrame, pd.DataFrame, list[str]]],
    level: str,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    if level == "observed":
        return states["observed"]
    if level == "forecast_only":
        train, eval_, _ = states["forecast"]
        cols = [c for c in train.columns if c.startswith("forecast_")]
        return train[["timestamp", *cols]].copy(), eval_[["timestamp", *cols]].copy(), cols
    if level == "latent_only":
        train, eval_, _ = states["latent"]
        cols = [c for c in train.columns if c.startswith("latent_")]
        return train[["timestamp", *cols]].copy(), eval_[["timestamp", *cols]].copy(), cols
    if level == "full_s3":
        return states["latent"]
    raise ValueError(f"Unknown state subset level: {level}")


def _fit_controller(
    *,
    train_candidates: pd.DataFrame,
    train_state: pd.DataFrame,
    eval_candidates: pd.DataFrame,
    eval_state: pd.DataFrame,
    state_cols: list[str],
    args: argparse.Namespace,
    per_strategy_residual: bool = False,
) -> tuple[dict[str, Any], list[str], pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    train_frame = mstc.build_response_frame(train_candidates, train_state)
    models, response_features, response_report = mstc.fit_response_models(
        train_frame,
        state_cols,
        per_strategy_residual=per_strategy_residual,
        max_rows=int(args.max_response_rows),
        max_keyword_cols=int(args.max_response_keyword_cols),
        response_frontier_weight_gamma=float(args.response_frontier_weight_gamma),
        response_frontier_weight_bandwidth=float(args.response_frontier_weight_bandwidth),
        response_balance_timestamps=bool(args.response_balance_timestamps),
        response_balance_strategies=bool(args.response_balance_strategies),
    )
    eval_frame = mstc.build_response_frame(eval_candidates, eval_state)
    predictions = mstc.predict_response(models, eval_frame, response_features, state_cols)
    schedule = mstc.threshold_schedule(
        eval_frame,
        predictions,
        models["curves"],
        delta_max=float(args.threshold_delta_max),
        max_down_step=float(args.max_threshold_up_step),
        relax_alpha=float(args.threshold_relax_alpha),
        controller_mode=str(args.controller_mode),
        min_lcb_utility=float(args.controller_min_lcb_utility),
        use_timeout_cap=bool(args.use_timeout_cap),
        min_action_edge=float(args.controller_min_action_edge),
        winner_sacrifice_multiplier=float(args.controller_winner_sacrifice_multiplier),
        enabled_heads=mstc._parse_enabled_heads(args.controller_enabled_heads),
        min_prediction_coverage=float(args.controller_min_prediction_coverage),
        min_usable_candidates=int(args.controller_min_usable_candidates),
        max_state_ood_score=args.controller_max_state_ood_score,
    )
    scored = mstc.apply_thresholds(eval_candidates, schedule)
    return models, response_features, eval_frame, predictions, schedule, {
        "response_report": response_report,
        "response_feature_count": int(len(response_features)),
        "state_feature_count": int(len(state_cols)),
        "state_feature_columns": list(state_cols),
        "response_feature_columns": list(response_features),
    } | {"scored_candidates": scored}


def _replay(
    *,
    arm: str,
    candidates: pd.DataFrame,
    params: Any,
    ev_curve: dict[str, Any],
    output_dir: Path,
    market_mode: str,
    schedule: pd.DataFrame | None = None,
    predictions: pd.DataFrame | None = None,
) -> dict[str, Any]:
    arm_dir = output_dir / arm
    arm_dir.mkdir(parents=True, exist_ok=True)
    decisions, equity, metrics = replay_candidates(
        candidates,
        params,
        mode="global_auction",
        ev_curve=ev_curve,
        market_mode=market_mode,
    )
    accepted = mstc._accepted_trades(candidates, decisions)
    summary = pd.DataFrame([mstc._metrics_row(arm, metrics, accepted, schedule)])
    by_head = mstc._by_head(arm, accepted)
    candidates.to_parquet(arm_dir / "candidates.parquet", index=False)
    decisions.to_parquet(arm_dir / "decisions.parquet", index=False)
    equity.to_parquet(arm_dir / "equity_curve.parquet", index=False)
    accepted.to_parquet(arm_dir / "accepted_trades.parquet", index=False)
    summary.to_csv(arm_dir / "summary.csv", index=False)
    by_head.to_csv(arm_dir / "by_head.csv", index=False)
    if schedule is not None:
        schedule.to_csv(arm_dir / "schedule.csv", index=False)
    if predictions is not None:
        predictions.to_parquet(arm_dir / "predictions.parquet", index=False)
    return {
        "arm": arm,
        "candidates": candidates,
        "decisions": decisions,
        "accepted": accepted,
        "summary": summary,
        "by_head": by_head,
        "schedule": schedule,
        "predictions": predictions,
    }


def _accepted_key_frame(accepted: pd.DataFrame) -> pd.DataFrame:
    if accepted.empty:
        return pd.DataFrame(columns=list(mstc.DECISION_KEY_COLS))
    out = accepted.loc[:, mstc.DECISION_KEY_COLS].copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    for col in ("symbol", "side", "strategy_id"):
        out[col] = out[col].astype(str)
    return out.drop_duplicates()


def _mark_accepted_cohort(source: pd.DataFrame, target: pd.DataFrame) -> pd.Series:
    if source.empty:
        return pd.Series(False, index=source.index, dtype=bool)
    keys = _accepted_key_frame(target)
    if keys.empty:
        return pd.Series(False, index=source.index, dtype=bool)
    marker = keys.assign(_in_target=True)
    merged = source.reset_index().merge(
        marker,
        on=list(mstc.DECISION_KEY_COLS),
        how="left",
        validate="many_to_one",
    )
    return merged.set_index("index")["_in_target"].fillna(False).reindex(source.index).astype(bool)


def _cohort_metrics(arm: str, cohort: str, rows: pd.DataFrame) -> dict[str, Any]:
    if rows.empty:
        return {
            "arm": arm,
            "cohort": cohort,
            "trade_count": 0,
            "net_pnl": 0.0,
            "gross_pnl": 0.0,
            "cost_pnl": 0.0,
            "win_rate": np.nan,
            "loss_avoided": 0.0,
            "winner_sacrificed": 0.0,
            "defensive_success": 0.0,
        }
    net = pd.to_numeric(rows["net_pnl"], errors="coerce").fillna(0.0)
    gross = pd.to_numeric(rows["gross_pnl"], errors="coerce").fillna(0.0)
    loss_avoided = float((-net.clip(upper=0.0)).sum()) if cohort == "suppressed_by_controller" else 0.0
    winner_sacrificed = float(net.clip(lower=0.0).sum()) if cohort == "suppressed_by_controller" else 0.0
    return {
        "arm": arm,
        "cohort": cohort,
        "trade_count": int(len(rows)),
        "net_pnl": float(net.sum()),
        "gross_pnl": float(gross.sum()),
        "cost_pnl": float((gross - net).sum()),
        "win_rate": float((net > 0.0).mean()),
        "loss_avoided": loss_avoided,
        "winner_sacrificed": winner_sacrificed,
        "defensive_success": float(loss_avoided - winner_sacrificed),
    }


def _cohort_audit(results: dict[str, dict[str, Any]], baseline_arm: str) -> pd.DataFrame:
    baseline = results[baseline_arm]["accepted"]
    rows: list[dict[str, Any]] = []
    for arm, result in results.items():
        if arm == baseline_arm or result.get("schedule") is None:
            continue
        accepted = result["accepted"]
        base_in_arm = _mark_accepted_cohort(baseline, accepted)
        arm_in_base = _mark_accepted_cohort(accepted, baseline)
        rows.append(_cohort_metrics(arm, "suppressed_by_controller", baseline.loc[~base_in_arm]))
        rows.append(_cohort_metrics(arm, "retained_unchanged", baseline.loc[base_in_arm]))
        rows.append(_cohort_metrics(arm, "accepted_later_capacity_reallocation", accepted.loc[~arm_in_base]))
    return pd.DataFrame(rows)


def _paired_day_bootstrap(
    *,
    baseline_arm: str,
    candidate_arm: str,
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    n_bootstrap: int = 5000,
    seed: int = 917,
) -> pd.DataFrame:
    """Bootstrap paired entry-day PnL deltas between two accepted-trade sets."""

    def by_day(frame: pd.DataFrame) -> pd.Series:
        if frame.empty:
            return pd.Series(dtype=float)
        work = frame[["timestamp", "net_pnl"]].copy()
        work["timestamp"] = pd.to_datetime(work["timestamp"], utc=True, errors="coerce")
        work["day"] = work["timestamp"].dt.floor("D")
        return pd.to_numeric(work["net_pnl"], errors="coerce").fillna(0.0).groupby(work["day"]).sum()

    base = by_day(baseline)
    cand = by_day(candidate)
    days = base.index.union(cand.index).sort_values()
    if len(days) == 0:
        return pd.DataFrame(
            [
                {
                    "comparison": f"{candidate_arm}_minus_{baseline_arm}",
                    "block": "entry_day",
                    "days": 0,
                    "point_delta_net_pnl": 0.0,
                    "bootstrap_mean_delta_net_pnl": 0.0,
                    "bootstrap_q05_delta_net_pnl": 0.0,
                    "bootstrap_q50_delta_net_pnl": 0.0,
                    "bootstrap_q95_delta_net_pnl": 0.0,
                    "bootstrap_positive_share": 0.0,
                }
            ]
        )
    delta = cand.reindex(days, fill_value=0.0) - base.reindex(days, fill_value=0.0)
    values = delta.to_numpy(dtype=float)
    rng = np.random.default_rng(int(seed))
    draws = rng.choice(values, size=(int(n_bootstrap), len(values)), replace=True).sum(axis=1)
    return pd.DataFrame(
        [
            {
                "comparison": f"{candidate_arm}_minus_{baseline_arm}",
                "block": "entry_day",
                "days": int(len(values)),
                "point_delta_net_pnl": float(values.sum()),
                "bootstrap_mean_delta_net_pnl": float(np.mean(draws)),
                "bootstrap_q05_delta_net_pnl": float(np.quantile(draws, 0.05)),
                "bootstrap_q50_delta_net_pnl": float(np.quantile(draws, 0.50)),
                "bootstrap_q95_delta_net_pnl": float(np.quantile(draws, 0.95)),
                "bootstrap_positive_share": float(np.mean(draws > 0.0)),
            }
        ]
    )


def _prediction_strategy_aggregates(predictions: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "pred_mean_utility",
        "pred_lcb_utility",
        "pred_full_sl",
        "pred_timeout",
        "pred_excess_full_sl",
        "pred_excess_timeout",
        "state_feature_coverage",
        "response_feature_coverage",
        "state_ood_score",
        "state_ood_flag",
    ]
    present = [c for c in cols if c in predictions.columns]
    agg = predictions.groupby(["timestamp", "strategy_id", "head"], as_index=False)[present].mean(numeric_only=True)
    rename = {
        "pred_mean_utility": "predicted_residual_utility",
        "pred_full_sl": "predicted_full_sl",
        "pred_timeout": "predicted_timeout",
        "pred_excess_full_sl": "predicted_excess_full_sl",
        "pred_excess_timeout": "predicted_excess_timeout",
    }
    return agg.rename(columns=rename)


def _decision_threshold_aggregates(decisions: pd.DataFrame) -> pd.DataFrame:
    if decisions.empty:
        return pd.DataFrame()
    work = decisions[["timestamp", "strategy_id", "base_threshold", "dynamic_threshold"]].copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"], utc=True, errors="coerce")
    out = work.groupby(["timestamp", "strategy_id"], as_index=False).agg(
        occupancy_adjusted_threshold_mean=("dynamic_threshold", "mean"),
        occupancy_adjusted_threshold_min=("dynamic_threshold", "min"),
        occupancy_adjusted_threshold_max=("dynamic_threshold", "max"),
        replay_base_threshold_mean=("base_threshold", "mean"),
    )
    return out


def _controller_action_audit(
    *,
    arm: str,
    result: dict[str, Any],
    observed_eval_state: pd.DataFrame,
) -> pd.DataFrame:
    schedule = result.get("schedule")
    predictions = result.get("predictions")
    if schedule is None or predictions is None or schedule.empty:
        return pd.DataFrame()
    audit = schedule.copy()
    audit["arm"] = arm
    audit = audit.rename(columns={"state_threshold": "state_adjusted_threshold"})
    pred_agg = _prediction_strategy_aggregates(predictions)
    audit = audit.merge(pred_agg, on=["timestamp", "strategy_id", "head"], how="left", validate="one_to_one")
    dyn = _decision_threshold_aggregates(result["decisions"])
    if not dyn.empty:
        audit = audit.merge(dyn, on=["timestamp", "strategy_id"], how="left", validate="one_to_one")
    state_cols = [c for c in observed_eval_state.columns if c != "timestamp"]
    state = observed_eval_state[["timestamp", *state_cols]].copy()
    audit = audit.merge(state, on="timestamp", how="left", validate="many_to_one")
    keep_front = [
        "arm",
        "timestamp",
        "head",
        "strategy_id",
        "base_threshold",
        "state_adjusted_threshold",
        "occupancy_adjusted_threshold_mean",
        "occupancy_adjusted_threshold_min",
        "occupancy_adjusted_threshold_max",
        "predicted_residual_utility",
        "predicted_excess_full_sl",
        "predicted_excess_timeout",
        "state_ood_share",
        "state_ood_score_mean",
        "state_ood_score_max",
        "controller_reason",
    ]
    ordered = [c for c in keep_front if c in audit.columns] + [
        c for c in audit.columns if c not in keep_front
    ]
    return audit[ordered]


def _period_windows() -> list[tuple[str, pd.Timestamp, pd.Timestamp]]:
    return [
        ("April 15-30", pd.Timestamp("2026-04-15T00:00:00Z"), pd.Timestamp("2026-04-30T23:59:59Z")),
        ("May 1-30", pd.Timestamp("2026-05-01T00:00:00Z"), pd.Timestamp("2026-05-30T23:59:59Z")),
        ("June 4-10", pd.Timestamp("2026-06-04T00:00:00Z"), pd.Timestamp("2026-06-10T23:59:59Z")),
        ("June 15-22", pd.Timestamp("2026-06-15T00:00:00Z"), pd.Timestamp("2026-06-22T23:59:59Z")),
    ]


def _period_metrics_from_accepted(accepted: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if accepted.empty:
        return pd.DataFrame()
    work = accepted.copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"], utc=True, errors="coerce")
    for label, start, end in _period_windows():
        for head, g in work.loc[work["timestamp"].between(start, end)].groupby("head"):
            rows.append(
                {
                    "period": label,
                    "head": head,
                    "trade_count": int(len(g)),
                    "realized_net_pnl": float(pd.to_numeric(g["net_pnl"], errors="coerce").fillna(0.0).sum()),
                    "realized_gross_pnl": float(pd.to_numeric(g["gross_pnl"], errors="coerce").fillna(0.0).sum()),
                    "realized_win_rate": float((pd.to_numeric(g["net_pnl"], errors="coerce").fillna(0.0) > 0.0).mean()),
                }
            )
    return pd.DataFrame(rows)


def _period_recognition_audit(
    *,
    sample: str,
    schedule: pd.DataFrame,
    predictions: pd.DataFrame,
    accepted_t1: pd.DataFrame,
    accepted_t2: pd.DataFrame,
) -> pd.DataFrame:
    if schedule.empty:
        return pd.DataFrame()
    sched = schedule.copy()
    sched["timestamp"] = pd.to_datetime(sched["timestamp"], utc=True, errors="coerce")
    pred = _prediction_strategy_aggregates(predictions)
    work = sched.merge(pred, on=["timestamp", "strategy_id", "head"], how="left", validate="one_to_one")
    period_rows = []
    t1_realized = _period_metrics_from_accepted(accepted_t1).rename(
        columns={
            "trade_count": "t1_trade_count",
            "realized_net_pnl": "t1_realized_net_pnl",
            "realized_gross_pnl": "t1_realized_gross_pnl",
            "realized_win_rate": "t1_realized_win_rate",
        }
    )
    t2_realized = _period_metrics_from_accepted(accepted_t2).rename(
        columns={
            "trade_count": "t2_trade_count",
            "realized_net_pnl": "t2_realized_net_pnl",
            "realized_gross_pnl": "t2_realized_gross_pnl",
            "realized_win_rate": "t2_realized_win_rate",
        }
    )
    for label, start, end in _period_windows():
        sub = work.loc[work["timestamp"].between(start, end)].copy()
        if sub.empty:
            continue
        for head, g in sub.groupby("head"):
            base = pd.to_numeric(g["base_threshold"], errors="coerce")
            adjusted = pd.to_numeric(g["state_threshold"], errors="coerce")
            period_rows.append(
                {
                    "sample": sample,
                    "period": label,
                    "head": head,
                    "strategy_count": int(g["strategy_id"].nunique()),
                    "timestamp_count": int(g["timestamp"].nunique()),
                    "mean_base_threshold": float(base.mean()),
                    "mean_adjusted_threshold": float(adjusted.mean()),
                    "fraction_raised": float((adjusted > base + 1e-9).mean()),
                    "predicted_residual_utility": float(pd.to_numeric(g.get("predicted_residual_utility"), errors="coerce").mean()),
                    "excess_sl_risk": float(pd.to_numeric(g.get("predicted_excess_full_sl"), errors="coerce").mean()),
                    "excess_timeout_risk": float(pd.to_numeric(g.get("predicted_excess_timeout"), errors="coerce").mean()),
                    "mean_state_ood_share": float(pd.to_numeric(g.get("state_ood_share"), errors="coerce").mean()),
                    "controller_reason_top": str(g["controller_reason"].mode().iloc[0]) if "controller_reason" in g and not g["controller_reason"].mode().empty else "",
                }
            )
    out = pd.DataFrame(period_rows)
    if out.empty:
        return out
    out = out.merge(t1_realized, on=["period", "head"], how="left")
    out = out.merge(t2_realized, on=["period", "head"], how="left")
    for col in ("t1_trade_count", "t2_trade_count"):
        if col in out:
            out[col] = out[col].fillna(0).astype(int)
    return out


def _render_report(
    *,
    summary: pd.DataFrame,
    by_head: pd.DataFrame,
    cohort: pd.DataFrame,
    period: pd.DataFrame,
    bootstrap: pd.DataFrame,
    output_dir: Path,
) -> str:
    lines = [
        "# Market-State Attribution Ablation",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Per-Trial Replay Metrics",
        "",
        summary.to_markdown(index=False, floatfmt=".6f") if not summary.empty else "_No summary rows._",
        "",
        "## Per-Head Replay Metrics",
        "",
        by_head.to_markdown(index=False, floatfmt=".6f") if not by_head.empty else "_No per-head rows._",
        "",
        "## Controller Cohort Audit",
        "",
        cohort.to_markdown(index=False, floatfmt=".6f") if not cohort.empty else "_No cohort rows._",
        "",
        "## Period Recognition Audit",
        "",
        period.to_markdown(index=False, floatfmt=".6f") if not period.empty else "_No period rows._",
        "",
        "## Paired Day Bootstrap",
        "",
        bootstrap.to_markdown(index=False, floatfmt=".6f") if not bootstrap.empty else "_No bootstrap rows._",
        "",
        "## Files",
        "",
        f"- Summary CSV: `{output_dir / 'attribution_summary.csv'}`",
        f"- Per-head CSV: `{output_dir / 'attribution_by_head.csv'}`",
        f"- Cohort audit CSV: `{output_dir / 'controller_cohort_audit.csv'}`",
        f"- Action audit CSV: `{output_dir / 'controller_action_audit.csv'}`",
        f"- Period audit CSV: `{output_dir / 'period_recognition_audit.csv'}`",
        f"- Bootstrap CSV: `{output_dir / 'paired_day_bootstrap.csv'}`",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-broad-candidates", type=Path, default=mstc.DEFAULT_TRAIN_BROAD)
    parser.add_argument("--train-deployable-candidates", type=Path, default=mstc.DEFAULT_TRAIN_DEPLOYABLE)
    parser.add_argument("--eval-candidates", type=Path, default=mstc.DEFAULT_EVAL_CANDIDATES)
    parser.add_argument("--train-feature-store-dir", type=Path, default=mstc.DEFAULT_TRAIN_FEATURE_STORE)
    parser.add_argument("--eval-feature-store-dir", type=Path, default=mstc.DEFAULT_EVAL_FEATURE_STORE)
    parser.add_argument("--policy-manifest", type=Path, default=mstc.DEFAULT_POLICY_MANIFEST)
    parser.add_argument("--policy-variant", default="refit_bar4_strategy_bar2")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--repaired-rank-contract", choices=("short_boll_timestamp_rank",), default="short_boll_timestamp_rank")
    parser.add_argument("--legacy-rank-contract", choices=("strict",), default="strict")
    parser.add_argument("--disable-heads", default="long_bars,long_dist")
    parser.add_argument("--controller-enabled-heads", default="short_boll")
    parser.add_argument("--max-feature-cols", type=int, default=128)
    parser.add_argument("--max-feature-store-cols", type=int, default=96)
    parser.add_argument("--feature-store-symbol-cap", type=int, default=220)
    parser.add_argument("--allow-candidate-state-fallback", action="store_true", default=False)
    parser.add_argument("--forecast-horizons-steps", default="6,24")
    parser.add_argument("--latent-states", type=int, default=4)
    parser.add_argument("--max-response-rows", type=int, default=6000)
    parser.add_argument("--max-response-keyword-cols", type=int, default=24)
    parser.add_argument("--response-frontier-weight-gamma", type=float, default=3.0)
    parser.add_argument("--response-frontier-weight-bandwidth", type=float, default=0.06)
    parser.add_argument("--response-balance-timestamps", action="store_true", default=True)
    parser.add_argument("--no-response-balance-timestamps", dest="response_balance_timestamps", action="store_false")
    parser.add_argument("--response-balance-strategies", action="store_true", default=True)
    parser.add_argument("--no-response-balance-strategies", dest="response_balance_strategies", action="store_false")
    parser.add_argument("--threshold-delta-max", type=float, default=0.10)
    parser.add_argument("--max-threshold-up-step", type=float, default=0.03)
    parser.add_argument("--threshold-relax-alpha", type=float, default=0.25)
    parser.add_argument(
        "--controller-mode",
        choices=("rank_grid", "action_aware_rank_grid", "frontier_rank_grid", "frontier_action_rank_grid", "severity"),
        default="rank_grid",
    )
    parser.add_argument("--controller-min-lcb-utility", type=float, default=0.0)
    parser.add_argument("--controller-min-prediction-coverage", type=float, default=0.80)
    parser.add_argument("--controller-min-usable-candidates", type=int, default=1)
    parser.add_argument("--controller-max-state-ood-score", type=float, default=None)
    parser.add_argument("--controller-min-action-edge", type=float, default=0.0)
    parser.add_argument("--controller-winner-sacrifice-multiplier", type=float, default=1.0)
    parser.add_argument("--enable-timeout-cap", dest="use_timeout_cap", action="store_true", default=False)
    parser.add_argument("--market-mode", default="perps")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    disabled_heads = mstc._parse_disabled_heads(args.disable_heads)
    params, policy_payload = mstc._load_policy_params(args.policy_manifest, args.policy_variant)

    t0_eval = _load_candidates_for_contract(
        args.eval_candidates,
        rank_contract=args.legacy_rank_contract,
        disabled_heads=disabled_heads,
    )
    t1_eval = _load_candidates_for_contract(
        args.eval_candidates,
        rank_contract=args.repaired_rank_contract,
        disabled_heads=disabled_heads,
    )
    train_broad = _load_candidates_for_contract(
        args.train_broad_candidates,
        rank_contract=args.repaired_rank_contract,
        disabled_heads=disabled_heads,
    )
    train_deployable = _load_candidates_for_contract(
        args.train_deployable_candidates,
        rank_contract=args.repaired_rank_contract,
        disabled_heads=disabled_heads,
    )
    ev_curve = fit_hierarchical_ev_curves(train_deployable)

    state_artifacts = bundle_builder._build_state_artifacts(
        train_broad,
        t1_eval,
        train_feature_store_dir=args.train_feature_store_dir,
        eval_feature_store_dir=args.eval_feature_store_dir,
        max_feature_cols=int(args.max_feature_cols),
        max_feature_store_cols=int(args.max_feature_store_cols),
        feature_store_symbol_cap=int(args.feature_store_symbol_cap),
        allow_candidate_state_fallback=bool(args.allow_candidate_state_fallback),
        forecast_horizons_steps=mstc._parse_int_grid(args.forecast_horizons_steps, (6, 24)),
        latent_states=int(args.latent_states),
    )

    results: dict[str, dict[str, Any]] = {}
    results["T0_legacy_contract_no_controller"] = _replay(
        arm="T0_legacy_contract_no_controller",
        candidates=t0_eval,
        params=params,
        ev_curve=ev_curve,
        output_dir=args.output_dir,
        market_mode=args.market_mode,
    )
    results["T1_repaired_contract_no_controller"] = _replay(
        arm="T1_repaired_contract_no_controller",
        candidates=t1_eval,
        params=params,
        ev_curve=ev_curve,
        output_dir=args.output_dir,
        market_mode=args.market_mode,
    )

    controller_specs = [
        ("T2_s1_observed_axes", "observed"),
        ("T3_forecast_only", "forecast_only"),
        ("T4_latent_only", "latent_only"),
        ("T5_full_s3", "full_s3"),
    ]
    fit_reports: dict[str, Any] = {}
    controller_models: dict[str, dict[str, Any]] = {}
    for arm, level in controller_specs:
        train_state, eval_state, state_cols = _state_subset(state_artifacts["states"], level)
        models, response_features, eval_frame, predictions, schedule, report = _fit_controller(
            train_candidates=train_broad,
            train_state=train_state,
            eval_candidates=t1_eval,
            eval_state=eval_state,
            state_cols=state_cols,
            args=args,
            per_strategy_residual=False,
        )
        scored = report.pop("scored_candidates")
        schedule = schedule.copy()
        schedule["arm"] = arm
        predictions = predictions.copy()
        predictions["arm"] = arm
        fit_reports[arm] = report | {"state_level": level}
        controller_models[arm] = {
            "models": models,
            "response_features": response_features,
            "train_state": train_state,
            "eval_state": eval_state,
            "state_cols": state_cols,
            "predictions": predictions,
            "schedule": schedule,
        }
        results[arm] = _replay(
            arm=arm,
            candidates=scored,
            params=params,
            ev_curve=ev_curve,
            output_dir=args.output_dir,
            market_mode=args.market_mode,
            schedule=schedule,
            predictions=predictions,
        )

    summary = pd.concat([r["summary"] for r in results.values()], ignore_index=True)
    by_head = pd.concat(
        [r["by_head"] for r in results.values() if not r["by_head"].empty],
        ignore_index=True,
    )
    cohort = _cohort_audit(results, "T1_repaired_contract_no_controller")
    bootstrap = pd.concat(
        [
            _paired_day_bootstrap(
                baseline_arm="T1_repaired_contract_no_controller",
                candidate_arm="T2_s1_observed_axes",
                baseline=results["T1_repaired_contract_no_controller"]["accepted"],
                candidate=results["T2_s1_observed_axes"]["accepted"],
            ),
            _paired_day_bootstrap(
                baseline_arm="T2_s1_observed_axes",
                candidate_arm="T5_full_s3",
                baseline=results["T2_s1_observed_axes"]["accepted"],
                candidate=results["T5_full_s3"]["accepted"],
            ),
        ],
        ignore_index=True,
    )

    observed_train, observed_eval, _ = state_artifacts["states"]["observed"]
    action_audits = []
    for arm in ["T2_s1_observed_axes", "T3_forecast_only", "T4_latent_only", "T5_full_s3"]:
        action = _controller_action_audit(
            arm=arm,
            result=results[arm],
            observed_eval_state=observed_eval,
        )
        if not action.empty:
            action_audits.append(action)
    action_audit = pd.concat(action_audits, ignore_index=True) if action_audits else pd.DataFrame()

    # Period recognition audit for the production candidate T2.  Historical
    # periods are in-sample diagnostics; June 15-22 is the eval replay.
    t2_model = controller_models["T2_s1_observed_axes"]
    train_frame = mstc.build_response_frame(train_broad, observed_train)
    train_predictions = mstc.predict_response(
        t2_model["models"],
        train_frame,
        t2_model["response_features"],
        t2_model["state_cols"],
    )
    train_schedule = mstc.threshold_schedule(
        train_frame,
        train_predictions,
        t2_model["models"]["curves"],
        delta_max=float(args.threshold_delta_max),
        max_down_step=float(args.max_threshold_up_step),
        relax_alpha=float(args.threshold_relax_alpha),
        controller_mode=str(args.controller_mode),
        min_lcb_utility=float(args.controller_min_lcb_utility),
        use_timeout_cap=bool(args.use_timeout_cap),
        min_action_edge=float(args.controller_min_action_edge),
        winner_sacrifice_multiplier=float(args.controller_winner_sacrifice_multiplier),
        enabled_heads=mstc._parse_enabled_heads(args.controller_enabled_heads),
        min_prediction_coverage=float(args.controller_min_prediction_coverage),
        min_usable_candidates=int(args.controller_min_usable_candidates),
        max_state_ood_score=args.controller_max_state_ood_score,
    )
    train_scored = mstc.apply_thresholds(train_broad, train_schedule)
    train_t1 = _replay(
        arm="diagnostic_train_T1_repaired_no_controller",
        candidates=train_broad,
        params=params,
        ev_curve=ev_curve,
        output_dir=args.output_dir,
        market_mode=args.market_mode,
    )
    train_t2 = _replay(
        arm="diagnostic_train_T2_observed_controller",
        candidates=train_scored,
        params=params,
        ev_curve=ev_curve,
        output_dir=args.output_dir,
        market_mode=args.market_mode,
        schedule=train_schedule,
        predictions=train_predictions,
    )
    period_train = _period_recognition_audit(
        sample="train_insample_diagnostic",
        schedule=train_schedule,
        predictions=train_predictions,
        accepted_t1=train_t1["accepted"],
        accepted_t2=train_t2["accepted"],
    )
    period_eval = _period_recognition_audit(
        sample="eval_june_oos_replay",
        schedule=results["T2_s1_observed_axes"]["schedule"],
        predictions=results["T2_s1_observed_axes"]["predictions"],
        accepted_t1=results["T1_repaired_contract_no_controller"]["accepted"],
        accepted_t2=results["T2_s1_observed_axes"]["accepted"],
    )
    period = pd.concat([period_train, period_eval], ignore_index=True)

    summary.to_csv(args.output_dir / "attribution_summary.csv", index=False)
    by_head.to_csv(args.output_dir / "attribution_by_head.csv", index=False)
    cohort.to_csv(args.output_dir / "controller_cohort_audit.csv", index=False)
    bootstrap.to_csv(args.output_dir / "paired_day_bootstrap.csv", index=False)
    action_audit.to_csv(args.output_dir / "controller_action_audit.csv", index=False)
    period.to_csv(args.output_dir / "period_recognition_audit.csv", index=False)
    observed_train.to_csv(args.output_dir / "train_observed_state_features.csv", index=False)
    observed_eval.to_csv(args.output_dir / "eval_observed_state_features.csv", index=False)

    deltas = []
    metric_map = summary.set_index("arm")
    for left, right, name in [
        ("T1_repaired_contract_no_controller", "T0_legacy_contract_no_controller", "T1_minus_T0_rank_eligibility_repair"),
        ("T2_s1_observed_axes", "T1_repaired_contract_no_controller", "T2_minus_T1_observed_controller"),
        ("T3_forecast_only", "T1_repaired_contract_no_controller", "T3_minus_T1_forecast_only"),
        ("T4_latent_only", "T1_repaired_contract_no_controller", "T4_minus_T1_latent_only"),
        ("T5_full_s3", "T1_repaired_contract_no_controller", "T5_minus_T1_full_s3"),
        ("T5_full_s3", "T2_s1_observed_axes", "T5_minus_T2_incremental_s3"),
    ]:
        if left in metric_map.index and right in metric_map.index:
            deltas.append(
                {
                    "comparison": name,
                    "delta_trade_count": float(metric_map.loc[left, "trade_count"] - metric_map.loc[right, "trade_count"]),
                    "delta_net_pnl": float(metric_map.loc[left, "net_pnl"] - metric_map.loc[right, "net_pnl"]),
                    "delta_gross_pnl": float(metric_map.loc[left, "gross_pnl"] - metric_map.loc[right, "gross_pnl"]),
                    "delta_full_sl_rate": float(metric_map.loc[left, "full_sl_rate"] - metric_map.loc[right, "full_sl_rate"]),
                    "delta_timeout_rate": float(metric_map.loc[left, "timeout_rate"] - metric_map.loc[right, "timeout_rate"]),
                }
            )
    deltas_df = pd.DataFrame(deltas)
    deltas_df.to_csv(args.output_dir / "attribution_deltas.csv", index=False)

    manifest = {
        "generated_by": "run_market_state_attribution_ablation",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "train_broad_candidates": str(args.train_broad_candidates),
        "train_deployable_candidates": str(args.train_deployable_candidates),
        "eval_candidates": str(args.eval_candidates),
        "policy_manifest": str(args.policy_manifest),
        "policy_variant": str(args.policy_variant),
        "policy_manifest_run_id": policy_payload.get("run_id"),
        "legacy_rank_contract": args.legacy_rank_contract,
        "repaired_rank_contract": args.repaired_rank_contract,
        "disabled_heads": sorted(disabled_heads),
        "controller_enabled_heads": sorted(mstc._parse_enabled_heads(args.controller_enabled_heads) or []),
        "controller_params": {
            "threshold_delta_max": float(args.threshold_delta_max),
            "max_threshold_up_step": float(args.max_threshold_up_step),
            "threshold_relax_alpha": float(args.threshold_relax_alpha),
            "controller_mode": str(args.controller_mode),
            "controller_min_lcb_utility": float(args.controller_min_lcb_utility),
            "controller_min_prediction_coverage": float(args.controller_min_prediction_coverage),
            "controller_min_usable_candidates": int(args.controller_min_usable_candidates),
            "controller_max_state_ood_score": args.controller_max_state_ood_score,
            "controller_min_action_edge": float(args.controller_min_action_edge),
            "controller_winner_sacrifice_multiplier": float(args.controller_winner_sacrifice_multiplier),
            "use_timeout_cap": bool(args.use_timeout_cap),
        },
        "state_artifact_reports": state_artifacts.get("reports", {}),
        "controller_fit_reports": fit_reports,
        "outputs": {
            "summary": str(args.output_dir / "attribution_summary.csv"),
            "deltas": str(args.output_dir / "attribution_deltas.csv"),
            "by_head": str(args.output_dir / "attribution_by_head.csv"),
            "cohort": str(args.output_dir / "controller_cohort_audit.csv"),
            "bootstrap": str(args.output_dir / "paired_day_bootstrap.csv"),
            "action_audit": str(args.output_dir / "controller_action_audit.csv"),
            "period_audit": str(args.output_dir / "period_recognition_audit.csv"),
            "report": str(args.output_dir / "market_state_attribution_ablation_report.md"),
        },
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    report = _render_report(
        summary=summary,
        by_head=by_head,
        cohort=cohort,
        period=period,
        bootstrap=bootstrap,
        output_dir=args.output_dir,
    )
    (args.output_dir / "market_state_attribution_ablation_report.md").write_text(report, encoding="utf-8")
    print(json.dumps(_json_safe({"output_dir": str(args.output_dir), "deltas": deltas}), indent=2))


if __name__ == "__main__":
    main()
