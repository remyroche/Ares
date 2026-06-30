#!/usr/bin/env python3
"""Walk-forward validation for the global portfolio-period multiplier.

This is the promotion-style companion to ``run_global_portfolio_period_multiplier``.
It keeps strategy scores, ranks, thresholds and auction ordering fixed, then
tests whether one timestamp-level new-risk multiplier improves portfolio
behavior across chronological folds.
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
    _add_trailing_performance,
    _add_open_position_concentration_features,
    _apply_multiplier,
    _feature_columns,
    _fit_models,
    _forward_labels,
    _json_safe,
    _load_candidates,
    _load_policy_params,
    _map_period_proxy_to_multiplier,
    _map_risk_to_multiplier,
    _metrics_row,
    _period_proxy,
    _add_portfolio_state_features,
    _predict_models,
    _smooth_multiplier,
    _timestamp_feature_fill_values,
    _timestamp_features,
)


DEFAULT_OUTPUT_DIR = Path("data_perp/reports/global_portfolio_period_multiplier_walkforward_20260625")


def _timestamp_mask(df: pd.DataFrame, start: pd.Timestamp | None = None, end: pd.Timestamp | None = None) -> pd.Series:
    ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    mask = ts.notna()
    if start is not None:
        mask &= ts >= start
    if end is not None:
        mask &= ts < end
    return mask


def _build_folds(
    timestamps: pd.Series,
    *,
    min_train_hours: int,
    fold_hours: int,
    embargo_hours: int,
    max_folds: int | None,
) -> list[dict[str, Any]]:
    ts = pd.to_datetime(timestamps, utc=True, errors="coerce").dropna().drop_duplicates().sort_values()
    if ts.empty:
        return []
    first = ts.min()
    last = ts.max()
    eval_start = first + pd.Timedelta(hours=int(min_train_hours))
    folds: list[dict[str, Any]] = []
    fold_id = 0
    while eval_start <= last:
        eval_end = min(eval_start + pd.Timedelta(hours=int(fold_hours)), last + pd.Timedelta(nanoseconds=1))
        train_end = eval_start - pd.Timedelta(hours=int(embargo_hours))
        train_ts = ts.loc[ts < train_end]
        eval_ts = ts.loc[(ts >= eval_start) & (ts < eval_end)]
        if len(train_ts) >= 120 and len(eval_ts) >= 12:
            folds.append(
                {
                    "fold_id": fold_id,
                    "train_start": train_ts.min(),
                    "train_end": train_ts.max(),
                    "embargo_start": train_ts.max(),
                    "eval_start": eval_ts.min(),
                    "eval_end": eval_ts.max(),
                    "train_timestamp_count": int(len(train_ts)),
                    "eval_timestamp_count": int(len(eval_ts)),
                }
            )
            fold_id += 1
            if max_folds is not None and fold_id >= max_folds:
                break
        eval_start = eval_end
    return folds


def _prepare_model_frame(
    candidates: pd.DataFrame,
    accepted: pd.DataFrame,
    equity: pd.DataFrame,
    *,
    feature_cols_raw: list[str],
    max_feature_cols: int,
    horizon_hours: int,
    label_cutoff: pd.Timestamp,
) -> tuple[pd.DataFrame, list[str], pd.Series]:
    features = _timestamp_features(candidates, feature_cols=feature_cols_raw, max_cols=max_feature_cols)
    fill_values = _timestamp_feature_fill_values(features)
    features = _add_trailing_performance(features, accepted)
    features = _add_portfolio_state_features(features, equity)
    features = _add_open_position_concentration_features(features, accepted)
    features["period_proxy"] = _period_proxy(features)
    labels = _forward_labels(features["timestamp"], accepted, int(horizon_hours))
    frame = features.merge(labels, on="timestamp", how="left")
    frame = frame.loc[pd.to_datetime(frame["timestamp"], utc=True, errors="coerce") <= label_cutoff].copy()
    model_feature_cols = [
        col
        for col in features.columns
        if col != "timestamp" and pd.api.types.is_numeric_dtype(features[col])
    ]
    return frame, model_feature_cols, fill_values


def _prepare_eval_features(
    candidates: pd.DataFrame,
    accepted_history: pd.DataFrame,
    equity_history: pd.DataFrame,
    *,
    feature_cols_raw: list[str],
    max_feature_cols: int,
    fill_values: pd.Series,
    required_cols: list[str],
) -> pd.DataFrame:
    features = _timestamp_features(
        candidates,
        feature_cols=feature_cols_raw,
        max_cols=max_feature_cols,
        fill_values=fill_values,
    )
    features = _add_trailing_performance(features, accepted_history)
    features = _add_portfolio_state_features(features, equity_history)
    features = _add_open_position_concentration_features(features, accepted_history)
    features["period_proxy"] = _period_proxy(features)
    for col in required_cols:
        if col not in features.columns:
            features[col] = 0.0
    return features[["timestamp"] + [col for col in required_cols if col != "timestamp"]]


def _make_schedules(eval_features: pd.DataFrame, pred: pd.DataFrame, cutoffs: dict[str, float]) -> dict[str, pd.DataFrame]:
    base_ts = eval_features[["timestamp"]].copy()
    schedules: dict[str, pd.DataFrame] = {}
    schedules["G0_no_modifier"] = base_ts.assign(multiplier=1.0)
    schedules["G1_existing_new_period_global"] = base_ts.assign(
        multiplier=_map_period_proxy_to_multiplier(pred["period_proxy"], cutoffs).to_numpy(dtype=float)
    )
    schedules["G2_utility_lcb"] = base_ts.assign(
        multiplier=np.where(pd.to_numeric(pred["pred_utility_q10"], errors="coerce") < 0.0, 0.25, 1.0)
    )
    schedules["G3_adverse_risk"] = base_ts.assign(
        multiplier=_map_risk_to_multiplier(pred["pred_adverse_risk"], cutoffs).to_numpy(dtype=float)
    )
    g4_raw = _map_risk_to_multiplier(pred["combined_risk"], cutoffs)
    g4_raw = g4_raw.where(pd.to_numeric(pred["pred_utility_q10"], errors="coerce") >= 0.0, 0.25)
    schedules["G4_combined"] = base_ts.assign(multiplier=g4_raw.to_numpy(dtype=float))
    schedules["G5_combined_asymmetric_smoothing"] = base_ts.assign(
        multiplier=_smooth_multiplier(base_ts["timestamp"], g4_raw).to_numpy(dtype=float)
    )
    schedules["G6_G5_plus_entry_cap_scaling"] = schedules["G5_combined_asymmetric_smoothing"].copy()
    return schedules


def _aggregate_promotion(summary: pd.DataFrame) -> pd.DataFrame:
    base = summary.loc[summary["arm"].eq("G0_no_modifier"), ["fold_id", "net_pnl", "cost_pnl", "max_drawdown", "worst_24h_net_pnl", "notional_turnover"]]
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
        median_delta_net_pnl = float(g["delta_net_pnl"].median())
        q25_delta_net_pnl = float(g["delta_net_pnl"].quantile(0.25))
        median_delta_cost_pnl = float(g["delta_cost_pnl"].median())
        median_delta_max_drawdown = float(g["delta_max_drawdown"].median())
        median_delta_worst_24h_net_pnl = float(g["delta_worst_24h_net_pnl"].median())
        median_exposure_ratio = float(g["exposure_ratio"].median())
        improves_median_pnl = median_delta_net_pnl > 0.0
        nonnegative_lower_quartile_pnl = q25_delta_net_pnl >= 0.0
        improves_drawdown = median_delta_max_drawdown > 0.0
        improves_worst_24h = median_delta_worst_24h_net_pnl > 0.0
        reduces_costs = median_delta_cost_pnl < 0.0
        preserves_exposure = median_exposure_ratio >= 0.75
        rows.append(
            {
                "arm": arm,
                "folds": int(g["fold_id"].nunique()),
                "median_delta_net_pnl": median_delta_net_pnl,
                "q25_delta_net_pnl": q25_delta_net_pnl,
                "mean_delta_net_pnl": float(g["delta_net_pnl"].mean()),
                "positive_delta_net_pnl_share": float((g["delta_net_pnl"] > 0).mean()),
                "median_delta_cost_pnl": median_delta_cost_pnl,
                "median_delta_max_drawdown": median_delta_max_drawdown,
                "median_delta_worst_24h_net_pnl": median_delta_worst_24h_net_pnl,
                "median_exposure_ratio": median_exposure_ratio,
                "min_exposure_ratio": float(g["exposure_ratio"].min()),
                "median_multiplier": float(g["mean_multiplier"].median()),
                "gate_improves_median_net_pnl": improves_median_pnl,
                "gate_q25_net_pnl_nonnegative": nonnegative_lower_quartile_pnl,
                "gate_improves_worst_24h": improves_worst_24h,
                "gate_improves_max_drawdown": improves_drawdown,
                "gate_reduces_costs": reduces_costs,
                "gate_preserves_exposure": preserves_exposure,
                "promotion_pass": bool(
                    arm != "G0_no_modifier"
                    and improves_median_pnl
                    and nonnegative_lower_quartile_pnl
                    and improves_worst_24h
                    and improves_drawdown
                    and reduces_costs
                    and preserves_exposure
                ),
            }
        )
    return pd.DataFrame(rows)


def _render_report(summary: pd.DataFrame, promotion: pd.DataFrame, folds: pd.DataFrame, manifest: dict[str, Any]) -> str:
    lines = [
        "# Global Portfolio Period Multiplier Walk-Forward",
        "",
        f"Generated: {manifest['generated_at_utc']}",
        "",
        "## Promotion Summary",
        "",
        promotion.to_markdown(index=False),
        "",
        "## Fold Summary",
        "",
        summary.to_markdown(index=False),
        "",
        "## Folds",
        "",
        folds.to_markdown(index=False),
        "",
        "## Contract",
        "",
        "- Complete timestamps are split chronologically.",
        "- Training timestamps end before validation by the configured embargo.",
        "- Feature fill values, EV curves, targets and period models are fitted on earlier timestamps only.",
        "- The validation replay keeps scores, ranks, thresholds and auction ordering fixed.",
        "- G6 additionally scales max_new_entries_per_bar; all other arms only scale the wallet cap.",
    ]
    return "\n".join(lines) + "\n"


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
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--max-feature-cols", type=int, default=96)
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
        raise RuntimeError("No walk-forward folds could be built with the requested settings")

    summary_rows: list[dict[str, Any]] = []
    pred_frames: list[pd.DataFrame] = []
    schedule_frames: list[pd.DataFrame] = []
    accepted_frames: list[pd.DataFrame] = []
    fold_meta_rows: list[dict[str, Any]] = []
    for fold in folds:
        fold_id = int(fold["fold_id"])
        train_end = pd.Timestamp(fold["train_end"])
        eval_start = pd.Timestamp(fold["eval_start"])
        eval_end = pd.Timestamp(fold["eval_end"]) + pd.Timedelta(nanoseconds=1)
        train_mask = _timestamp_mask(broad, end=train_end + pd.Timedelta(nanoseconds=1))
        eval_mask = _timestamp_mask(broad, start=eval_start, end=eval_end)
        deploy_train_mask = _timestamp_mask(deployable, end=train_end + pd.Timedelta(nanoseconds=1))
        train_broad = broad.loc[train_mask].copy()
        eval_candidates = broad.loc[eval_mask].copy()
        train_deployable = deployable.loc[deploy_train_mask].copy()
        if len(train_deployable) < 50 or len(train_broad) < 200 or len(eval_candidates) < 20:
            continue

        ev_curve = fit_hierarchical_ev_curves(train_deployable)
        train_decisions, train_equity, train_metrics = replay_candidates(
            train_broad,
            params,
            mode="global_auction",
            ev_curve=ev_curve,
            market_mode=args.market_mode,
        )
        train_accepted = _accepted_trades(train_broad, train_decisions)
        feature_cols_raw = _feature_columns(train_broad, max_cols=args.max_feature_cols)
        label_cutoff = train_end - pd.Timedelta(hours=int(args.horizon_hours))
        train_model_frame, model_feature_cols, fill_values = _prepare_model_frame(
            train_broad,
            train_accepted,
            train_equity,
            feature_cols_raw=feature_cols_raw,
            max_feature_cols=int(args.max_feature_cols),
            horizon_hours=int(args.horizon_hours),
            label_cutoff=label_cutoff,
        )
        try:
            models, cutoffs, train_fit_frame = _fit_models(train_model_frame, model_feature_cols)
        except RuntimeError as exc:
            fold_meta_rows.append({**fold, "skipped": True, "skip_reason": str(exc)})
            continue

        history_mask = _timestamp_mask(broad, end=eval_end)
        history_candidates = broad.loc[history_mask].copy()
        history_decisions, history_equity, _ = replay_candidates(
            history_candidates,
            params,
            mode="global_auction",
            ev_curve=ev_curve,
            market_mode=args.market_mode,
        )
        history_accepted = _accepted_trades(history_candidates, history_decisions)
        eval_features = _prepare_eval_features(
            eval_candidates,
            history_accepted,
            history_equity,
            feature_cols_raw=feature_cols_raw,
            max_feature_cols=int(args.max_feature_cols),
            fill_values=fill_values,
            required_cols=model_feature_cols,
        )
        eval_pred = _predict_models(models, eval_features, model_feature_cols)
        pred = eval_features[["timestamp", "period_proxy"]].merge(eval_pred, on="timestamp", how="left")
        pred["fold_id"] = fold_id
        pred_frames.append(pred)

        schedules = _make_schedules(eval_features, pred, cutoffs)
        for arm, schedule in schedules.items():
            schedule = schedule.copy()
            schedule["fold_id"] = fold_id
            schedule_frames.append(schedule.assign(arm=arm))
            candidate_arm = _apply_multiplier(
                eval_candidates,
                schedule[["timestamp", "multiplier"]],
                scale_entries=arm == "G6_G5_plus_entry_cap_scaling",
                max_entries=int(params.max_new_entries_per_bar),
            )
            decisions, _, metrics = replay_candidates(
                candidate_arm,
                params,
                mode="global_auction",
                ev_curve=ev_curve,
                market_mode=args.market_mode,
            )
            accepted = _accepted_trades(candidate_arm, decisions)
            accepted["fold_id"] = fold_id
            accepted["arm"] = arm
            accepted_frames.append(accepted)
            row = _metrics_row(arm, metrics, schedule, accepted)
            row.update(
                {
                    "fold_id": fold_id,
                    "eval_start": pd.Timestamp(fold["eval_start"]).isoformat(),
                    "eval_end": pd.Timestamp(fold["eval_end"]).isoformat(),
                    "train_timestamp_count": int(fold["train_timestamp_count"]),
                    "eval_timestamp_count": int(fold["eval_timestamp_count"]),
                }
            )
            summary_rows.append(row)

        fold_meta_rows.append(
            {
                **fold,
                "skipped": False,
                "train_candidate_rows": int(len(train_broad)),
                "eval_candidate_rows": int(len(eval_candidates)),
                "train_deployable_rows": int(len(train_deployable)),
                "train_baseline_net_pnl": float(train_metrics.get("net_pnl", 0.0) or 0.0),
                "train_labeled_rows": int(len(train_fit_frame)),
                "model_feature_count": int(len(model_feature_cols)),
            }
        )

    summary = pd.DataFrame(summary_rows)
    if summary.empty:
        raise RuntimeError("All folds were skipped")
    promotion = _aggregate_promotion(summary)
    folds_df = pd.DataFrame(fold_meta_rows)
    summary.to_csv(args.output_dir / "walkforward_fold_summary.csv", index=False)
    promotion.to_csv(args.output_dir / "walkforward_promotion_summary.csv", index=False)
    folds_df.to_csv(args.output_dir / "walkforward_folds.csv", index=False)
    if pred_frames:
        pd.concat(pred_frames, ignore_index=True).to_csv(args.output_dir / "walkforward_predictions.csv", index=False)
    if schedule_frames:
        pd.concat(schedule_frames, ignore_index=True).to_csv(args.output_dir / "walkforward_schedules.csv", index=False)
    if accepted_frames:
        pd.concat(accepted_frames, ignore_index=True).to_parquet(args.output_dir / "walkforward_accepted_trades.parquet", index=False)

    manifest = {
        "generated_by": "run_global_portfolio_period_multiplier_walkforward",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "broad_candidates": str(args.broad_candidates),
        "deployable_candidates": str(args.deployable_candidates),
        "policy_manifest": str(args.policy_manifest),
        "policy_variant": str(args.policy_variant),
        "policy_params": asdict(params),
        "policy_manifest_run_id": policy_payload.get("run_id"),
        "horizon_hours": int(args.horizon_hours),
        "embargo_hours": int(args.embargo_hours),
        "min_train_hours": int(args.min_train_hours),
        "fold_hours": int(args.fold_hours),
        "fold_count": int(summary["fold_id"].nunique()),
        "model_feature_count_min": int(folds_df.loc[~folds_df["skipped"].astype(bool), "model_feature_count"].min()) if "model_feature_count" in folds_df else None,
        "model_feature_count_max": int(folds_df.loc[~folds_df["skipped"].astype(bool), "model_feature_count"].max()) if "model_feature_count" in folds_df else None,
        "train_labeled_rows_min": int(folds_df.loc[~folds_df["skipped"].astype(bool), "train_labeled_rows"].min()) if "train_labeled_rows" in folds_df else None,
        "train_labeled_rows_max": int(folds_df.loc[~folds_df["skipped"].astype(bool), "train_labeled_rows"].max()) if "train_labeled_rows" in folds_df else None,
        "contract": {
            "one_global_timestamp_multiplier": True,
            "changes_ranking_or_thresholds": False,
            "changes_new_risk_wallet_cap": True,
            "G6_also_scales_max_new_entries_per_bar": True,
            "complete_timestamp_splits": True,
            "train_only_feature_fill_values": True,
            "train_only_ev_curve": True,
            "embargo_hours": int(args.embargo_hours),
        },
        "outputs": {
            "fold_summary": str(args.output_dir / "walkforward_fold_summary.csv"),
            "promotion_summary": str(args.output_dir / "walkforward_promotion_summary.csv"),
            "folds": str(args.output_dir / "walkforward_folds.csv"),
            "predictions": str(args.output_dir / "walkforward_predictions.csv"),
            "schedules": str(args.output_dir / "walkforward_schedules.csv"),
            "accepted_trades": str(args.output_dir / "walkforward_accepted_trades.parquet"),
            "report": str(args.output_dir / "global_portfolio_period_multiplier_walkforward_report.md"),
        },
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(_json_safe(manifest), indent=2) + "\n")
    (args.output_dir / "global_portfolio_period_multiplier_walkforward_report.md").write_text(
        _render_report(summary, promotion, folds_df, manifest),
        encoding="utf-8",
    )
    print(json.dumps(_json_safe(manifest), indent=2)[:6000])
    print(f"\nWrote {args.output_dir}")


if __name__ == "__main__":
    main()
