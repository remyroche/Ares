#!/usr/bin/env python3
"""Walk-forward defensive-success gate for size-action scorer interventions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from extreme_price_movements.portfolio_policy_replay import fit_hierarchical_ev_curves, normalise_candidate_table
from scripts.run_global_portfolio_period_multiplier import _load_policy_params
from scripts.run_size_action_live_scorer_replay import _head_from_strategy, _load_candidates, _replay, _summarise


SCORE_FEATURES = [
    "p_intervene",
    "action_selector_score",
    "positive_value_score",
    "pred_delta_J",
    "pred_immediate_J",
    "pred_capacity_J",
]
EXCLUDE_FEATURES = {
    "timestamp",
    "strategy_id",
    "head",
    "component_scope",
    "accepted",
    "reject_reason",
    "selected_multiplier",
    "multiplier",
    "baseline_group_net_pnl",
    "baseline_group_winner_pnl",
    "baseline_group_loser_loss",
    "baseline_group_trades",
    "defensive_success_value",
    "defensive_success_target",
    "fold_week_start",
    "action_binds",
    "delta_full_J",
    "delta_immediate_J",
    "delta_full_net_pnl",
    "delta_full_cost_pnl",
    "delta_full_turnover",
    "delta_full_J_per_notional",
    "delta_immediate_J_per_notional",
    "fold_id",
    "split",
    "best_multiplier",
    "best_gain",
    "best_margin",
    "best_gain_per_notional",
    "best_margin_per_notional",
    "group_affected_notional",
    "best_immediate_gain",
    "best_nonbaseline_gain",
    "worst_nonbaseline_gain",
    "best_nonbaseline_multiplier",
    "group_can_bind",
    "y_intervene",
}


def _week_start(ts: pd.Series) -> pd.Series:
    return pd.to_datetime(ts, utc=True, errors="coerce").dt.to_period("W-SUN").dt.start_time.dt.tz_localize("UTC")


def _selected_action_features(action_features: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    scores_key = scores[["timestamp", "strategy_id", "selected_multiplier"]].copy()
    scores_key["timestamp"] = pd.to_datetime(scores_key["timestamp"], utc=True, errors="coerce")
    scores_key["strategy_id"] = scores_key["strategy_id"].astype(str)
    scores_key["selected_multiplier_round"] = pd.to_numeric(scores_key["selected_multiplier"], errors="coerce").fillna(1.0).round(6)

    features = action_features.copy()
    features["timestamp"] = pd.to_datetime(features["timestamp"], utc=True, errors="coerce")
    features["strategy_id"] = features["strategy_id"].astype(str)
    features["selected_multiplier_round"] = pd.to_numeric(features["multiplier"], errors="coerce").fillna(1.0).round(6)
    merged = scores_key.merge(
        features,
        on=["timestamp", "strategy_id", "selected_multiplier_round"],
        how="left",
        suffixes=("", "_feature"),
    )
    return merged.drop(columns=["selected_multiplier_round"], errors="ignore")


def _feature_columns(frame: pd.DataFrame, max_features: int) -> list[str]:
    preferred = [
        *SCORE_FEATURES,
        "head_code",
        "timestamp_rank_max",
        "timestamp_rank_mean",
        "timestamp_rank_q90",
        "timestamp_above_threshold_share",
        "timestamp_symbol_count",
        "wallet",
        "open_positions",
        "remaining_slots",
        "remaining_capital_pct",
        "open_capital_pct",
        "strategy_open_count",
        "strategy_open_notional_share",
        "side_open_hhi",
        "unrealized_pnl_to_wallet",
        "positions_exiting_24h",
        "strategy_candidate_count",
        "strategy_rank_max",
        "strategy_rank_mean",
        "strategy_rank_q90",
        "strategy_rank_std",
        "strategy_score_max",
        "strategy_score_mean",
        "strategy_threshold_gap_max",
        "strategy_above_threshold_share",
        "strategy_expected_cost_mean",
        "strategy_requested_notional_above_threshold",
        "strategy_candidate_open_or_cooldown_symbol_share",
        "projected_removed_trade_count",
        "projected_action_strength",
        "projected_notional_removed_to_wallet",
        "projected_removed_trade_share_strategy",
        "projected_removed_trade_share_timestamp",
    ]
    cols: list[str] = []
    for col in preferred:
        if col in frame.columns and col not in EXCLUDE_FEATURES and pd.api.types.is_numeric_dtype(frame[col]) and col not in cols:
            vals = pd.to_numeric(frame[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
            if vals.notna().sum() > 0 and vals.nunique(dropna=True) > 1:
                cols.append(col)
    for col in frame.columns:
        if len(cols) >= int(max_features):
            break
        if col in cols or col in EXCLUDE_FEATURES or not pd.api.types.is_numeric_dtype(frame[col]):
            continue
        vals = pd.to_numeric(frame[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if vals.notna().sum() > 0 and vals.nunique(dropna=True) > 1:
            cols.append(col)
    return cols[: int(max_features)]


def _matrix(frame: pd.DataFrame, cols: list[str], medians: pd.Series | None = None) -> tuple[pd.DataFrame, pd.Series]:
    x = frame[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if medians is None:
        medians = x.median(numeric_only=True).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return x.fillna(medians).astype(np.float32), medians


def _fit_predict_gate(train: pd.DataFrame, eval_rows: pd.DataFrame, features: list[str], seed: int) -> tuple[np.ndarray, dict[str, Any]]:
    y = pd.to_numeric(train["defensive_success_target"], errors="coerce").fillna(0).astype(int)
    if len(train) < 40 or y.nunique() < 2 or not features:
        rate = float(y.mean()) if len(y) else 0.0
        return np.full(len(eval_rows), rate, dtype=float), {
            "constant": True,
            "train_rows": int(len(train)),
            "positive_rate": rate,
            "feature_count": int(len(features)),
        }
    from lightgbm import LGBMClassifier

    value = pd.to_numeric(train["defensive_success_value"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    scale = max(float(np.nanmedian(np.abs(value[value != 0]))) if np.any(value != 0) else 1.0, 1.0)
    weights = 1.0 + np.clip(np.abs(value) / scale, 0.0, 10.0)
    pos = max(int(y.sum()), 1)
    neg = max(int(len(y) - y.sum()), 1)
    x_train, medians = _matrix(train, features)
    x_eval, _ = _matrix(eval_rows, features, medians)
    model = LGBMClassifier(
        objective="binary",
        n_estimators=180,
        learning_rate=0.035,
        num_leaves=15,
        max_depth=4,
        min_child_samples=max(20, int(0.05 * len(train))),
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=2.0,
        scale_pos_weight=float(max(neg / pos, 1.0)),
        random_state=int(seed),
        deterministic=True,
        force_col_wise=True,
        verbose=-1,
    )
    model.fit(x_train, y.to_numpy(dtype=int), sample_weight=weights)
    pred = np.asarray(model.predict_proba(x_eval), dtype=float)[:, 1]
    return pred, {
        "constant": False,
        "train_rows": int(len(train)),
        "positive_rate": float(y.mean()),
        "feature_count": int(len(features)),
    }


def _load_extra_training_panel(path: Path, *, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    panel = pd.read_csv(path)
    if panel.empty:
        return panel
    panel["timestamp"] = pd.to_datetime(panel["timestamp"], utc=True, errors="coerce")
    panel = panel.loc[panel["timestamp"].notna() & panel["timestamp"].lt(end)].copy()
    panel["strategy_id"] = panel["strategy_id"].astype(str)
    panel["head"] = panel["strategy_id"].map(_head_from_strategy)
    panel["selected_multiplier"] = pd.to_numeric(panel.get("multiplier"), errors="coerce").fillna(1.0)
    panel["scorer_intervention"] = panel["selected_multiplier"].lt(1.0)
    if "action_binds" in panel.columns:
        panel = panel.loc[pd.to_numeric(panel["action_binds"], errors="coerce").fillna(0.0).gt(0.0)].copy()
    panel = panel.loc[panel["scorer_intervention"]].copy()
    panel["defensive_success_value"] = pd.to_numeric(panel.get("delta_full_J"), errors="coerce").fillna(0.0)
    panel["defensive_success_target"] = panel["defensive_success_value"].gt(0.0).astype(int)
    panel["fold_week_start"] = _week_start(panel["timestamp"])
    panel["head_code"] = panel["head"].astype("category").cat.codes.astype(float)
    panel["extra_training_source"] = "exact_state_panel"
    # Keep only rows strictly before the live gate window when this panel overlaps.
    panel = panel.loc[panel["timestamp"].lt(start)].copy()
    return panel


def _load_extra_live_training_rows(
    labels_path: Path,
    action_features_path: Path,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    labels = pd.read_csv(labels_path)
    if labels.empty:
        return labels
    labels["timestamp"] = pd.to_datetime(labels["timestamp"], utc=True, errors="coerce")
    labels = labels.loc[labels["timestamp"].notna() & labels["timestamp"].lt(start) & labels["timestamp"].lt(end)].copy()
    if labels.empty:
        return labels
    labels["strategy_id"] = labels["strategy_id"].astype(str)
    labels["head"] = labels["strategy_id"].map(_head_from_strategy)
    labels["selected_multiplier"] = pd.to_numeric(labels.get("selected_multiplier"), errors="coerce").fillna(1.0)
    labels["scorer_intervention"] = labels["selected_multiplier"].lt(1.0)
    if "accepted" in labels.columns:
        accepted = labels["accepted"]
        if accepted.dtype == object:
            accepted = accepted.astype(str).str.lower().isin({"1", "true", "yes"})
        labels["scorer_intervention"] = labels["scorer_intervention"] & accepted.astype(bool)
    labels = labels.loc[labels["scorer_intervention"]].copy()
    if labels.empty:
        return labels

    action_features = pd.read_parquet(action_features_path)
    selected_features = _selected_action_features(action_features, labels)
    panel = labels.merge(
        selected_features,
        on=["timestamp", "strategy_id", "selected_multiplier"],
        how="left",
        suffixes=("", "_feat"),
    )
    for col in ("baseline_group_net_pnl", "baseline_group_winner_pnl", "baseline_group_loser_loss", "baseline_group_trades"):
        panel[col] = pd.to_numeric(panel.get(col), errors="coerce").fillna(0.0)
    if "defensive_success_value" not in panel.columns:
        panel["defensive_success_value"] = -panel["baseline_group_net_pnl"]
    panel["defensive_success_value"] = pd.to_numeric(panel["defensive_success_value"], errors="coerce").fillna(0.0)
    if "defensive_success_target" not in panel.columns:
        panel["defensive_success_target"] = panel["defensive_success_value"].gt(0.0).astype(int)
    panel["defensive_success_target"] = pd.to_numeric(panel["defensive_success_target"], errors="coerce").fillna(0).astype(int)
    panel["fold_week_start"] = _week_start(panel["timestamp"])
    panel["head_code"] = panel["head"].astype("category").cat.codes.astype(float)
    panel["extra_training_source"] = "live_replay_panel"
    return panel


def _choose_threshold(train_pred: np.ndarray, train_value: np.ndarray, grid: list[float], min_keep: int) -> tuple[float, dict[str, Any]]:
    best_threshold = 1.01
    best_score = 0.0
    best_keep = 0
    for threshold in grid:
        keep = np.asarray(train_pred >= float(threshold))
        n_keep = int(keep.sum())
        if n_keep < int(min_keep):
            continue
        score = float(train_value[keep].sum())
        if score > best_score:
            best_threshold = float(threshold)
            best_score = score
            best_keep = n_keep
    return best_threshold, {"threshold": best_threshold, "train_value": best_score, "train_keep": best_keep}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--broad-candidates", type=Path, required=True)
    parser.add_argument("--deployable-candidates", type=Path, required=True)
    parser.add_argument("--action-features", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--extra-training-panel", type=Path, default=None)
    parser.add_argument("--extra-live-training-labels", type=Path, default=None)
    parser.add_argument("--extra-live-action-features", type=Path, default=None)
    parser.add_argument("--policy-manifest", type=Path, required=True)
    parser.add_argument("--policy-variant", default="refit_bar4_strategy_bar2")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--start", default="2026-05-29T00:00:00+00:00")
    parser.add_argument("--end", default="2026-06-26T00:00:00+00:00")
    parser.add_argument("--market-mode", default="perps")
    parser.add_argument(
        "--scope",
        choices=["shared", "short_asset_only", "per_head", "shared_per_head_threshold"],
        default="short_asset_only",
    )
    parser.add_argument("--min-train-rows", type=int, default=40)
    parser.add_argument("--head-threshold-min-train-rows", type=int, default=20)
    parser.add_argument("--head-threshold-shrinkage-rows", type=int, default=80)
    parser.add_argument("--max-features", type=int, default=96)
    parser.add_argument("--threshold-grid", default="0.25,0.35,0.45,0.55,0.65,0.75")
    parser.add_argument("--min-train-keep", type=int, default=3)
    parser.add_argument("--max-eval-keep-share", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()

    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)
    start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
    end = end.tz_localize("UTC") if end.tzinfo is None else end.tz_convert("UTC")
    grid = [float(x.strip()) for x in str(args.threshold_grid).split(",") if x.strip()]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    params, _payload = _load_policy_params(args.policy_manifest, args.policy_variant)
    candidates = _load_candidates(args.broad_candidates, start=start, end=end)
    deployable = normalise_candidate_table(pd.read_parquet(args.deployable_candidates))
    deployable_train = deployable.loc[deployable["timestamp"].lt(start)].copy()
    ev_curve = fit_hierarchical_ev_curves(deployable_train if not deployable_train.empty else deployable)
    baseline, _baseline_metrics = _replay(candidates, params, ev_curve, market_mode=args.market_mode, arm="C0_baseline")

    scores = pd.read_csv(args.scores)
    scores["timestamp"] = pd.to_datetime(scores["timestamp"], utc=True, errors="coerce")
    scores = scores.loc[scores["timestamp"].ge(start) & scores["timestamp"].lt(end)].copy()
    scores["strategy_id"] = scores["strategy_id"].astype(str)
    scores["head"] = scores["strategy_id"].map(_head_from_strategy)
    scores["selected_multiplier"] = pd.to_numeric(scores["selected_multiplier"], errors="coerce").fillna(1.0)
    scores["scorer_intervention"] = scores["selected_multiplier"].lt(1.0)

    action_features = pd.read_parquet(args.action_features)
    selected_features = _selected_action_features(action_features, scores)
    group_labels = baseline.groupby(["timestamp", "strategy_id"], dropna=False).agg(
        baseline_group_net_pnl=("net_pnl", "sum"),
        baseline_group_winner_pnl=("net_pnl", lambda s: float(s[s > 0].sum())),
        baseline_group_loser_loss=("net_pnl", lambda s: float(-s[s < 0].sum())),
        baseline_group_trades=("net_pnl", "size"),
    ).reset_index()
    frame = scores.merge(selected_features, on=["timestamp", "strategy_id", "selected_multiplier"], how="left", suffixes=("", "_feat"))
    frame = frame.merge(group_labels, on=["timestamp", "strategy_id"], how="left")
    for col in ("baseline_group_net_pnl", "baseline_group_winner_pnl", "baseline_group_loser_loss", "baseline_group_trades"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0.0)
    frame["defensive_success_value"] = -frame["baseline_group_net_pnl"]
    frame["defensive_success_target"] = frame["defensive_success_value"].gt(0.0).astype(int)
    frame["fold_week_start"] = _week_start(frame["timestamp"])
    frame["head_code"] = frame["head"].astype("category").cat.codes.astype(float)
    model_rows = frame.loc[frame["scorer_intervention"]].copy()
    if args.scope == "short_asset_only":
        model_rows = model_rows.loc[model_rows["head"].eq("short_asset")].copy()
    eval_model_rows = model_rows.copy()
    extra_rows = pd.DataFrame()
    exact_extra_rows = pd.DataFrame()
    live_extra_rows = pd.DataFrame()
    if args.extra_training_panel is not None:
        exact_extra_rows = _load_extra_training_panel(args.extra_training_panel, start=start, end=end)
    if (args.extra_live_training_labels is None) ^ (args.extra_live_action_features is None):
        raise ValueError("--extra-live-training-labels and --extra-live-action-features must be passed together")
    if args.extra_live_training_labels is not None and args.extra_live_action_features is not None:
        live_extra_rows = _load_extra_live_training_rows(
            args.extra_live_training_labels,
            args.extra_live_action_features,
            start=start,
            end=end,
        )
    extra_parts = [part for part in (exact_extra_rows, live_extra_rows) if not part.empty]
    if extra_parts:
        extra_rows = pd.concat(extra_parts, ignore_index=True, sort=False)
        if args.scope == "short_asset_only" and not extra_rows.empty:
            extra_rows = extra_rows.loc[extra_rows["head"].eq("short_asset")].copy()
        if not extra_rows.empty:
            for col in model_rows.columns:
                if col not in extra_rows.columns:
                    extra_rows[col] = np.nan
            for col in extra_rows.columns:
                if col not in model_rows.columns:
                    model_rows[col] = np.nan
            model_rows = pd.concat([extra_rows[model_rows.columns], model_rows], ignore_index=True, sort=False)
    features = _feature_columns(model_rows, int(args.max_features))
    features_by_scope: dict[str, list[str]] = {}

    schedule = scores[["timestamp", "strategy_id", "selected_multiplier", "head"]].copy()
    schedule["multiplier"] = 1.0
    fold_rows: list[dict[str, Any]] = []
    pred_records: list[pd.DataFrame] = []
    weeks = sorted(eval_model_rows["fold_week_start"].dropna().unique())
    for idx, week in enumerate(weeks):
        week_ts = pd.Timestamp(week)
        week_ts = week_ts.tz_localize("UTC") if week_ts.tzinfo is None else week_ts.tz_convert("UTC")
        cutoff = max(week_ts, start)
        train = model_rows.loc[model_rows["timestamp"].lt(cutoff)].copy()
        eval_rows = eval_model_rows.loc[eval_model_rows["fold_week_start"].eq(week)].copy()
        if eval_rows.empty:
            continue
        if args.scope == "per_head":
            for head, head_eval_rows in eval_rows.groupby("head", dropna=False, sort=False):
                head_name = str(head)
                head_train = train.loc[train["head"].astype(str).eq(head_name)].copy()
                if len(head_train) < int(args.min_train_rows):
                    fold_rows.append(
                        {
                            "week_start": str(week),
                            "head": head_name,
                            "train_rows": int(len(head_train)),
                            "eval_rows": int(len(head_eval_rows)),
                            "used_model": False,
                            "reason": "insufficient_train_rows",
                        }
                    )
                    continue
                head_features = _feature_columns(head_train, int(args.max_features))
                features_by_scope[head_name] = head_features
                train_pred, train_diag = _fit_predict_gate(head_train, head_train, head_features, seed=int(args.seed) + idx)
                threshold, threshold_diag = _choose_threshold(
                    train_pred,
                    pd.to_numeric(head_train["defensive_success_value"], errors="coerce").fillna(0.0).to_numpy(dtype=float),
                    grid,
                    int(args.min_train_keep),
                )
                eval_pred, eval_diag = _fit_predict_gate(head_train, head_eval_rows, head_features, seed=int(args.seed) + idx)
                keep = eval_pred >= float(threshold)
                max_keep = int(np.floor(float(args.max_eval_keep_share) * len(head_eval_rows)))
                if float(args.max_eval_keep_share) < 1.0:
                    max_keep = max(0, max_keep)
                    if keep.sum() > max_keep:
                        order = np.argsort(-eval_pred)
                        capped_keep = np.zeros(len(head_eval_rows), dtype=bool)
                        capped_keep[order[:max_keep]] = True
                        keep = keep & capped_keep
                eval_out = head_eval_rows[
                    ["timestamp", "strategy_id", "selected_multiplier", "head", "defensive_success_value", "defensive_success_target"]
                ].copy()
                eval_out["gate_probability"] = eval_pred
                eval_out["gate_threshold"] = float(threshold)
                eval_out["gate_keep"] = keep
                pred_records.append(eval_out)
                if keep.any():
                    keys = pd.MultiIndex.from_frame(eval_out.loc[keep, ["timestamp", "strategy_id"]])
                    sched_idx = pd.MultiIndex.from_frame(schedule[["timestamp", "strategy_id"]])
                    mask = sched_idx.isin(keys)
                    schedule.loc[mask, "multiplier"] = schedule.loc[mask, "selected_multiplier"].to_numpy(dtype=float)
                fold_rows.append(
                    {
                        "week_start": str(week),
                        "head": head_name,
                        "train_rows": int(len(head_train)),
                        "eval_rows": int(len(head_eval_rows)),
                        "used_model": True,
                        "kept_eval_rows": int(keep.sum()),
                        "max_eval_keep_share": float(args.max_eval_keep_share),
                        "max_eval_keep": int(max_keep) if float(args.max_eval_keep_share) < 1.0 else int(len(head_eval_rows)),
                        **{f"train_{k}": v for k, v in train_diag.items()},
                        **{f"threshold_{k}": v for k, v in threshold_diag.items()},
                        **{f"eval_{k}": v for k, v in eval_diag.items()},
                        "feature_count": int(len(head_features)),
                    }
                )
            continue
        if len(train) < int(args.min_train_rows):
            fold_rows.append({"week_start": str(week), "train_rows": int(len(train)), "eval_rows": int(len(eval_rows)), "used_model": False, "reason": "insufficient_train_rows"})
            continue
        train_pred, train_diag = _fit_predict_gate(train, train, features, seed=int(args.seed) + idx)
        threshold, threshold_diag = _choose_threshold(
            train_pred,
            pd.to_numeric(train["defensive_success_value"], errors="coerce").fillna(0.0).to_numpy(dtype=float),
            grid,
            int(args.min_train_keep),
        )
        eval_pred, eval_diag = _fit_predict_gate(train, eval_rows, features, seed=int(args.seed) + idx)
        threshold_values = np.full(len(eval_rows), float(threshold), dtype=float)
        head_threshold_rows: list[dict[str, Any]] = []
        if args.scope == "shared_per_head_threshold":
            train_values = pd.to_numeric(train["defensive_success_value"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            for head_name, head_eval_rows in eval_rows.groupby("head", dropna=False, sort=False):
                head_name = str(head_name)
                head_train_mask = train["head"].astype(str).eq(head_name).to_numpy()
                head_train_n = int(head_train_mask.sum())
                chosen_threshold = float(threshold)
                raw_head_threshold = np.nan
                used_head_threshold = False
                if head_train_n >= int(args.head_threshold_min_train_rows):
                    candidate_threshold, candidate_diag = _choose_threshold(
                        train_pred[head_train_mask],
                        train_values[head_train_mask],
                        grid,
                        int(args.min_train_keep),
                    )
                    raw_head_threshold = float(candidate_threshold)
                    if np.isfinite(candidate_threshold) and float(candidate_threshold) <= 1.0:
                        weight = min(1.0, head_train_n / max(float(args.head_threshold_shrinkage_rows), 1.0))
                        chosen_threshold = float(weight * float(candidate_threshold) + (1.0 - weight) * float(threshold))
                        used_head_threshold = True
                    else:
                        candidate_diag = {"threshold": float(candidate_threshold), "train_value": 0.0, "train_keep": 0}
                else:
                    candidate_diag = {"threshold": np.nan, "train_value": 0.0, "train_keep": 0}
                eval_mask = eval_rows["head"].astype(str).eq(head_name).to_numpy()
                threshold_values[eval_mask] = chosen_threshold
                head_threshold_rows.append(
                    {
                        "head": head_name,
                        "head_train_rows": head_train_n,
                        "global_threshold": float(threshold),
                        "raw_head_threshold": raw_head_threshold,
                        "chosen_threshold": float(chosen_threshold),
                        "used_head_threshold": bool(used_head_threshold),
                        "head_threshold_train_value": float(candidate_diag.get("train_value", 0.0)),
                        "head_threshold_train_keep": int(candidate_diag.get("train_keep", 0)),
                    }
                )
        keep = eval_pred >= threshold_values
        max_keep = int(np.floor(float(args.max_eval_keep_share) * len(eval_rows)))
        if float(args.max_eval_keep_share) < 1.0:
            max_keep = max(0, max_keep)
            if keep.sum() > max_keep:
                order = np.argsort(-eval_pred)
                capped_keep = np.zeros(len(eval_rows), dtype=bool)
                capped_keep[order[:max_keep]] = True
                keep = keep & capped_keep
        eval_out = eval_rows[["timestamp", "strategy_id", "selected_multiplier", "head", "defensive_success_value", "defensive_success_target"]].copy()
        eval_out["gate_probability"] = eval_pred
        eval_out["gate_threshold"] = threshold_values
        eval_out["gate_keep"] = keep
        pred_records.append(eval_out)
        if keep.any():
            keys = pd.MultiIndex.from_frame(eval_out.loc[keep, ["timestamp", "strategy_id"]])
            sched_idx = pd.MultiIndex.from_frame(schedule[["timestamp", "strategy_id"]])
            mask = sched_idx.isin(keys)
            schedule.loc[mask, "multiplier"] = schedule.loc[mask, "selected_multiplier"].to_numpy(dtype=float)
        fold_rows.append(
            {
                "week_start": str(week),
                "train_rows": int(len(train)),
                "eval_rows": int(len(eval_rows)),
                "used_model": True,
                "kept_eval_rows": int(keep.sum()),
                "max_eval_keep_share": float(args.max_eval_keep_share),
                "max_eval_keep": int(max_keep) if float(args.max_eval_keep_share) < 1.0 else int(len(eval_rows)),
                **{f"train_{k}": v for k, v in train_diag.items()},
                **{f"threshold_{k}": v for k, v in threshold_diag.items()},
                **{f"eval_{k}": v for k, v in eval_diag.items()},
                "head_thresholds": json.dumps(head_threshold_rows, sort_keys=True, default=str) if head_threshold_rows else "",
            }
        )

    gated, _gated_metrics = _replay(
        candidates,
        params,
        ev_curve,
        market_mode=args.market_mode,
        arm=f"defensive_success_gate_{args.scope}",
        schedule=schedule[["timestamp", "strategy_id", "multiplier"]],
    )
    raw_schedule = scores[["timestamp", "strategy_id", "selected_multiplier"]].rename(columns={"selected_multiplier": "multiplier"})
    raw, _raw_metrics = _replay(candidates, params, ev_curve, market_mode=args.market_mode, arm="raw_scorer", schedule=raw_schedule)
    accepted_all = pd.concat([baseline, raw, gated], ignore_index=True)
    accepted_all.to_csv(args.out_dir / "accepted_trades.csv", index=False)
    for keys, name in [
        (["arm"], "overall"),
        (["arm", "head"], "by_head"),
        (["arm", "week_start"], "weekly"),
        (["arm", "week_start", "head"], "weekly_by_head"),
        (["arm", "month"], "monthly"),
        (["arm", "month", "head"], "monthly_by_head"),
    ]:
        _summarise(accepted_all, keys).to_csv(args.out_dir / f"{name}.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(args.out_dir / "gate_folds.csv", index=False)
    if pred_records:
        pd.concat(pred_records, ignore_index=True).to_csv(args.out_dir / "gate_predictions.csv", index=False)
    frame.to_csv(args.out_dir / "training_frame.csv", index=False)
    schedule.to_csv(args.out_dir / "size_schedule.csv", index=False)
    manifest = {
        "generated_by": "run_size_action_defensive_success_gate",
        "start": start.isoformat(),
        "end": end.isoformat(),
        "scope": str(args.scope),
        "features": features,
        "features_by_scope": features_by_scope,
        "feature_count": int(len(features)),
        "model_rows": int(len(model_rows)),
        "extra_training_rows": int(len(extra_rows)) if not extra_rows.empty else 0,
        "exact_extra_training_rows": int(len(exact_extra_rows)) if not exact_extra_rows.empty else 0,
        "live_extra_training_rows": int(len(live_extra_rows)) if not live_extra_rows.empty else 0,
        "extra_training_panel": str(args.extra_training_panel) if args.extra_training_panel is not None else None,
        "extra_live_training_labels": str(args.extra_live_training_labels) if args.extra_live_training_labels is not None else None,
        "extra_live_action_features": str(args.extra_live_action_features) if args.extra_live_action_features is not None else None,
        "interventions_kept": int(pd.to_numeric(schedule["multiplier"], errors="coerce").fillna(1.0).lt(1.0).sum()),
        "threshold_grid": grid,
        "min_train_rows": int(args.min_train_rows),
        "head_threshold_min_train_rows": int(args.head_threshold_min_train_rows),
        "head_threshold_shrinkage_rows": int(args.head_threshold_shrinkage_rows),
        "min_train_keep": int(args.min_train_keep),
        "max_eval_keep_share": float(args.max_eval_keep_share),
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str))
    print(json.dumps(manifest, indent=2, sort_keys=True, default=str))
    print(_summarise(accepted_all, ["arm"]).to_string(index=False))


if __name__ == "__main__":
    main()
