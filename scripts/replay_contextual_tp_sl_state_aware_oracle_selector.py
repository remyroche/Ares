#!/usr/bin/env python3
"""State-aware weekly selector for sparse reliability interventions.

This is a research ablation, not a production policy. It builds weekly
live-available state summaries from the candidate ledgers, joins them to
full-replay one-week oracle labels, then performs chronological walk-forward
selection. The selector defaults to the baseline unless prior similar states
suggest a conservative positive full-path intervention value.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.replay_contextual_tp_sl_weekly_combo_switching import (  # noqa: E402
    CHAMPION_COMBO,
    OBJECTIVE_COL,
    _load_arm_tables,
)
from scripts.replay_contextual_tp_sl_weekly_intervention_oracle import (  # noqa: E402
    _candidate_id,
    _load_baseline_tables,
)
from scripts.replay_contextual_tp_sl_weekly_rule_selector import (  # noqa: E402
    BASELINE_ID,
    _build_candidate_cache,
    _run_replay,
    _score_windows,
)
from scripts.sweep_contextual_tp_sl_arm_combinations import (  # noqa: E402
    _accepted_period_tables,
    _json_safe,
)


FEATURE_COLUMNS = [
    "auction_rank_score",
    "rank_pct",
    "policy_rank_pct",
    "reliability_blend_score",
    "generated_weighted_hr_surprise_24",
    "generated_weighted_hr_surprise_96",
    "generated_hr_surprise_24",
    "generated_hr_surprise_96",
    "generated_score_abs_diff_24",
    "generated_score_abs_minus_prev24_mean",
    "generated_strategy_score_ood_abs_z",
    "generated_score_uncertainty_p1mp",
    "generated_score_entropy",
    "generated_loss_rate_24",
    "generated_loss_rate_96",
    "generated_matured_count_24",
    "generated_matured_count_96",
    "expected_spread_bps",
    "slippage_bps",
    "fees_bps",
    "barrier_pct",
]


def _week_start(values: pd.Series) -> pd.Series:
    ts = pd.to_datetime(values, utc=True, errors="coerce")
    naive = ts.dt.tz_convert(None)
    starts = naive.dt.to_period("W").dt.start_time
    return pd.to_datetime(starts, utc=True, errors="coerce")


def _load_oracle(path: Path, rules: Sequence[str]) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["week_start"] = pd.to_datetime(frame["intervention_week"], utc=True, errors="coerce")
    frame = frame.loc[frame["candidate_rule"].astype(str).isin({str(r) for r in rules})].copy()
    for col in (
        "delta_full_objective",
        "delta_full_net_pnl",
        "delta_full_weekly_q20_pnl",
        "delta_full_daily_q20_pnl",
        "delta_intervention_week_net_pnl",
    ):
        frame[col] = pd.to_numeric(frame.get(col), errors="coerce")
    frame["full_net_tail_positive"] = (
        frame["delta_full_net_pnl"].gt(0.0) & frame["delta_full_weekly_q20_pnl"].ge(0.0)
    )
    return frame.dropna(subset=["week_start"])


def _numeric(frame: pd.DataFrame, col: str) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[col], errors="coerce")


def _head_name(strategy_id: str) -> str:
    value = str(strategy_id)
    if "boll" in value:
        return "short_bollinger"
    if "asset" in value:
        return "short_asset"
    if "dist" in value:
        return "long_dist"
    if "bars" in value:
        return "long_bars"
    return value


def _summarize_week_rule(frame: pd.DataFrame, *, rule_id: str, week_start: pd.Timestamp) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "week_start": week_start,
        "candidate_rule": rule_id,
        "candidate_rows": int(len(frame)),
    }
    if frame.empty:
        return out
    bound = _numeric(frame, "conditional_filter_bound").fillna(0.0).gt(0.0)
    out["bound_rows"] = int(bound.sum())
    out["bound_share"] = float(bound.mean())
    heads = frame["strategy_id"].astype(str).map(_head_name) if "strategy_id" in frame.columns else pd.Series("", index=frame.index)
    for head in ("long_bars", "long_dist", "short_asset", "short_bollinger"):
        hmask = heads.eq(head)
        out[f"{head}_row_share"] = float(hmask.mean()) if len(hmask) else 0.0
        out[f"{head}_bound_share"] = float((bound & hmask).sum() / max(int(hmask.sum()), 1))
    rank = _numeric(frame, "auction_rank_score").fillna(_numeric(frame, "rank_pct"))
    if rank.notna().any():
        out["rank_mean"] = float(rank.mean())
        out["rank_q75"] = float(rank.quantile(0.75))
        out["rank_q90"] = float(rank.quantile(0.90))
        out["rank_max"] = float(rank.max())
        out["bound_rank_mean"] = float(rank.loc[bound].mean()) if bound.any() else 0.0
        out["bound_rank_q90"] = float(rank.loc[bound].quantile(0.90)) if bound.any() else 0.0
    for col in FEATURE_COLUMNS:
        values = _numeric(frame, col)
        if not values.notna().any():
            continue
        out[f"{col}_mean"] = float(values.mean())
        out[f"{col}_q75"] = float(values.quantile(0.75))
        if bound.any():
            bvals = values.loc[bound].dropna()
            out[f"bound_{col}_mean"] = float(bvals.mean()) if not bvals.empty else 0.0
            out[f"bound_{col}_q75"] = float(bvals.quantile(0.75)) if not bvals.empty else 0.0
    return out


def _build_feature_table(
    cache: Mapping[str, pd.DataFrame],
    *,
    candidate_combo: str,
    rules: Sequence[str],
    state_lag_weeks: int,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for rule in rules:
        cid = _candidate_id(candidate_combo, rule)
        frame = cache[cid].copy()
        if "_week_start" not in frame.columns:
            frame["_week_start"] = _week_start(frame["timestamp"])
            if isinstance(cache, dict):
                cache[cid] = frame
        for week, group in frame.groupby("_week_start", sort=True):
            rows.append(_summarize_week_rule(group, rule_id=rule, week_start=pd.Timestamp(week)))
    table = pd.DataFrame(rows)
    if not table.empty:
        table["state_week_start"] = pd.to_datetime(table["week_start"], utc=True, errors="coerce")
        table["week_start"] = table["state_week_start"] + pd.to_timedelta(int(state_lag_weeks), unit="W")
    return table


def _feature_columns(frame: pd.DataFrame) -> List[str]:
    excluded = {"week_start", "state_week_start", "candidate_rule"}
    cols = []
    for col in frame.columns:
        if col in excluded:
            continue
        if pd.api.types.is_numeric_dtype(frame[col]):
            cols.append(col)
    return cols


def _standardize_train_test(train: pd.DataFrame, test: pd.DataFrame, cols: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
    train_values = train.loc[:, cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    med = train_values.median(axis=0).fillna(0.0)
    train_values = train_values.fillna(med)
    test_values = test.loc[:, cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(med)
    std = train_values.std(axis=0).replace(0.0, np.nan).fillna(1.0)
    return ((train_values - med) / std).to_numpy(dtype=float), ((test_values - med) / std).to_numpy(dtype=float)


def _select_knn(
    train: pd.DataFrame,
    candidates: pd.DataFrame,
    *,
    feature_cols: Sequence[str],
    k: int,
    min_history: int,
    min_score: float,
    min_net: float,
    min_tail: float,
) -> Tuple[str, Dict[str, Any]]:
    if len(train) < int(min_history) or not feature_cols:
        return BASELINE_ID, {"selection_reason": "fallback_insufficient_history"}
    x_train, x_test = _standardize_train_test(train, candidates, feature_cols)
    labels_obj = pd.to_numeric(train["delta_full_objective"], errors="coerce").fillna(-np.inf).to_numpy(dtype=float)
    labels_net = pd.to_numeric(train["delta_full_net_pnl"], errors="coerce").fillna(-np.inf).to_numpy(dtype=float)
    labels_tail = pd.to_numeric(train["delta_full_weekly_q20_pnl"], errors="coerce").fillna(-np.inf).to_numpy(dtype=float)
    rows: List[Dict[str, Any]] = []
    k = max(1, int(k))
    for i, rule in enumerate(candidates["candidate_rule"].astype(str).to_numpy()):
        dist = np.sqrt(np.nanmean((x_train - x_test[i]) ** 2, axis=1))
        order = np.argsort(dist)[: min(k, len(dist))]
        score = float(np.mean(labels_obj[order])) if len(order) else -np.inf
        net = float(np.mean(labels_net[order])) if len(order) else -np.inf
        tail = float(np.mean(labels_tail[order])) if len(order) else -np.inf
        positive_share = float(np.mean((labels_net[order] > 0.0) & (labels_tail[order] >= 0.0))) if len(order) else 0.0
        rows.append(
            {
                "candidate_rule": rule,
                "pred_score": score,
                "pred_net": net,
                "pred_tail": tail,
                "pred_positive_share": positive_share,
                "nearest_distance": float(np.mean(dist[order])) if len(order) else np.inf,
            }
        )
    scored = pd.DataFrame(rows).sort_values(["pred_score", "pred_net"], ascending=False)
    best = scored.iloc[0].to_dict()
    eligible = (
        float(best["pred_score"]) >= float(min_score)
        and float(best["pred_net"]) >= float(min_net)
        and float(best["pred_tail"]) >= float(min_tail)
    )
    if not eligible:
        best["selection_reason"] = "fallback_knn_gate"
        return BASELINE_ID, best
    best["selection_reason"] = "selected_knn_state"
    return str(best["candidate_rule"]), best


def _ridge_fit_predict(train: pd.DataFrame, candidates: pd.DataFrame, feature_cols: Sequence[str], alpha: float) -> Tuple[np.ndarray, float]:
    if len(train) <= len(feature_cols) + 2 or not feature_cols:
        return np.full(len(candidates), -np.inf), np.inf
    x_train, x_test = _standardize_train_test(train, candidates, feature_cols)
    y = pd.to_numeric(train["delta_full_objective"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    x_aug = np.column_stack([np.ones(len(x_train)), x_train])
    xtx = x_aug.T @ x_aug
    penalty = np.eye(xtx.shape[0]) * float(alpha)
    penalty[0, 0] = 0.0
    coef = np.linalg.pinv(xtx + penalty) @ x_aug.T @ y
    pred_train = x_aug @ coef
    residual_std = float(np.std(y - pred_train)) if len(y) > 1 else np.inf
    pred_test = np.column_stack([np.ones(len(x_test)), x_test]) @ coef
    return pred_test, residual_std


def _select_ridge_lcb(
    train: pd.DataFrame,
    candidates: pd.DataFrame,
    *,
    feature_cols: Sequence[str],
    min_history: int,
    alpha: float,
    lcb_z: float,
    min_lcb: float,
) -> Tuple[str, Dict[str, Any]]:
    if len(train) < int(min_history) or not feature_cols:
        return BASELINE_ID, {"selection_reason": "fallback_insufficient_history"}
    preds, residual_std = _ridge_fit_predict(train, candidates, feature_cols, alpha)
    scored = candidates[["candidate_rule"]].copy()
    scored["pred_score"] = preds
    scored["pred_lcb"] = scored["pred_score"] - float(lcb_z) * residual_std
    scored = scored.sort_values(["pred_lcb", "pred_score"], ascending=False)
    best = scored.iloc[0].to_dict()
    if float(best["pred_lcb"]) < float(min_lcb):
        best["selection_reason"] = "fallback_ridge_lcb_gate"
        best["residual_std"] = residual_std
        return BASELINE_ID, best
    best["selection_reason"] = "selected_ridge_lcb_state"
    best["residual_std"] = residual_std
    return str(best["candidate_rule"]), best


def _walk_forward_decisions(
    joined: pd.DataFrame,
    *,
    rules: Sequence[str],
    mode: str,
    k: int,
    min_history: int,
    min_score: float,
    min_net: float,
    min_tail: float,
    alpha: float,
    lcb_z: float,
    min_lcb: float,
) -> pd.DataFrame:
    feature_cols = _feature_columns(joined)
    rows: List[Dict[str, Any]] = []
    for week in sorted(joined["week_start"].dropna().unique()):
        week = pd.Timestamp(week)
        train = joined.loc[joined["week_start"].lt(week)].copy()
        current = joined.loc[joined["week_start"].eq(week) & joined["candidate_rule"].astype(str).isin(rules)].copy()
        if current.empty:
            continue
        if mode == "knn":
            selected, meta = _select_knn(
                train,
                current,
                feature_cols=feature_cols,
                k=k,
                min_history=min_history,
                min_score=min_score,
                min_net=min_net,
                min_tail=min_tail,
            )
        elif mode == "ridge_lcb":
            selected, meta = _select_ridge_lcb(
                train,
                current,
                feature_cols=feature_cols,
                min_history=min_history,
                alpha=alpha,
                lcb_z=lcb_z,
                min_lcb=min_lcb,
            )
        else:
            raise ValueError(f"Unknown mode `{mode}`")
        label = joined.loc[joined["week_start"].eq(week) & joined["candidate_rule"].astype(str).eq(str(selected))]
        label_row = label.iloc[0].to_dict() if selected != BASELINE_ID and not label.empty else {}
        rows.append(
            {
                "week_start": week.isoformat(),
                "selected_rule": selected,
                "selection_mode": mode,
                "prior_rows": int(len(train)),
                **meta,
                "selected_label_delta_full_objective": label_row.get("delta_full_objective", 0.0),
                "selected_label_delta_full_net_pnl": label_row.get("delta_full_net_pnl", 0.0),
                "selected_label_delta_full_weekly_q20_pnl": label_row.get("delta_full_weekly_q20_pnl", 0.0),
            }
        )
    return pd.DataFrame(rows)


def _complete_baseline_decisions(cache: Mapping[str, pd.DataFrame], decisions: pd.DataFrame) -> pd.DataFrame:
    baseline = cache[BASELINE_ID].copy()
    if "_week_start" not in baseline.columns:
        baseline["_week_start"] = _week_start(baseline["timestamp"])
        if isinstance(cache, dict):
            cache[BASELINE_ID] = baseline
    all_weeks = sorted(pd.Timestamp(v) for v in baseline["_week_start"].dropna().unique())
    existing = set(pd.to_datetime(decisions.get("week_start", pd.Series(dtype=str)), utc=True, errors="coerce").dropna())
    missing = [
        {
            "week_start": week.isoformat(),
            "selected_rule": BASELINE_ID,
            "selection_mode": "baseline_fill",
            "prior_rows": 0,
            "selection_reason": "fallback_missing_state_features",
            "selected_label_delta_full_objective": 0.0,
            "selected_label_delta_full_net_pnl": 0.0,
            "selected_label_delta_full_weekly_q20_pnl": 0.0,
        }
        for week in all_weeks
        if week not in existing
    ]
    if missing:
        decisions = pd.concat([decisions, pd.DataFrame(missing)], ignore_index=True)
    if not decisions.empty:
        decisions["_sort_week"] = pd.to_datetime(decisions["week_start"], utc=True, errors="coerce")
        decisions = decisions.sort_values("_sort_week").drop(columns=["_sort_week"]).reset_index(drop=True)
    return decisions


def _build_selected_stream(
    cache: Mapping[str, pd.DataFrame],
    decisions: pd.DataFrame,
    *,
    candidate_combo: str,
) -> pd.DataFrame:
    baseline = cache[BASELINE_ID].copy()
    if "_week_start" not in baseline.columns:
        baseline["_week_start"] = _week_start(baseline["timestamp"])
        if isinstance(cache, dict):
            cache[BASELINE_ID] = baseline
    frames: List[pd.DataFrame] = []
    for _, row in decisions.iterrows():
        week = pd.Timestamp(row["week_start"])
        selected = str(row["selected_rule"])
        cid = BASELINE_ID if selected == BASELINE_ID else _candidate_id(candidate_combo, selected)
        source = cache[cid].copy()
        if "_week_start" not in source.columns:
            source["_week_start"] = _week_start(source["timestamp"])
            if isinstance(cache, dict):
                cache[cid] = source
        part = source.loc[source["_week_start"].eq(week)].copy()
        if part.empty:
            continue
        part["selected_candidate_id"] = cid
        part["selected_rule"] = selected
        frames.append(part)
    if not frames:
        return pd.DataFrame()
    return (
        pd.concat(frames, ignore_index=True)
        .drop(columns=["_week_start"], errors="ignore")
        .sort_values(["timestamp", "strategy_id", "symbol"])
        .reset_index(drop=True)
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--oracle-summary", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--market-mode", default="perps", choices=["spot", "perps"])
    parser.add_argument("--baseline-combo", default=CHAMPION_COMBO)
    parser.add_argument("--candidate-combo", default="long_bars:I_long_dist:R_short_asset:S_short_bollinger:R")
    parser.add_argument("--candidate-rule", action="append", required=True)
    parser.add_argument("--selection-mode", action="append", default=["knn", "ridge_lcb"])
    parser.add_argument("--knn-k", default="3,5")
    parser.add_argument("--min-history", type=int, default=28)
    parser.add_argument("--min-score", default="0,25")
    parser.add_argument("--min-net", default="0")
    parser.add_argument("--min-tail", default="0")
    parser.add_argument("--ridge-alpha", default="1,10,100")
    parser.add_argument("--lcb-z", default="0,0.5")
    parser.add_argument("--min-lcb", default="0")
    parser.add_argument("--threshold-mode", default="expanding", choices=["full_sample", "expanding"])
    parser.add_argument("--min-threshold-history", type=int, default=500)
    parser.add_argument(
        "--state-lag-weeks",
        type=int,
        default=1,
        help="Use state features from this many weeks before the intervention week. Default 1 avoids same-week lookahead.",
    )
    parser.add_argument("--validation-start", default="2026-05-01")
    parser.add_argument("--june-start", default="2026-06-01")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rules = [str(r) for r in args.candidate_rule]
    oracle = _load_oracle(args.oracle_summary, rules)
    tables = _load_arm_tables(args.source_dir, [args.baseline_combo, args.candidate_combo])
    cache = _build_candidate_cache(
        tables,
        baseline_combo=args.baseline_combo,
        candidate_combo=args.candidate_combo,
        candidate_rules=rules,
        threshold_mode=args.threshold_mode,
        min_threshold_history=args.min_threshold_history,
    )
    features = _build_feature_table(
        cache,
        candidate_combo=args.candidate_combo,
        rules=rules,
        state_lag_weeks=args.state_lag_weeks,
    )
    joined = features.merge(
        oracle,
        on=["week_start", "candidate_rule"],
        how="inner",
        validate="one_to_one",
    )
    joined.to_csv(args.out_dir / "state_oracle_feature_table.csv", index=False)
    baseline_daily, baseline_weekly = _load_baseline_tables(args.baseline_dir)

    all_scores: List[pd.DataFrame] = []
    all_decisions: List[pd.DataFrame] = []
    all_daily: List[pd.DataFrame] = []
    all_weekly: List[pd.DataFrame] = []
    modes = [str(v).strip() for v in args.selection_mode if str(v).strip()]
    ks = [int(v.strip()) for v in str(args.knn_k).split(",") if v.strip()]
    min_scores = [float(v.strip()) for v in str(args.min_score).split(",") if v.strip()]
    min_nets = [float(v.strip()) for v in str(args.min_net).split(",") if v.strip()]
    min_tails = [float(v.strip()) for v in str(args.min_tail).split(",") if v.strip()]
    alphas = [float(v.strip()) for v in str(args.ridge_alpha).split(",") if v.strip()]
    lcb_values = [float(v.strip()) for v in str(args.lcb_z).split(",") if v.strip()]
    min_lcbs = [float(v.strip()) for v in str(args.min_lcb).split(",") if v.strip()]

    configs: List[Dict[str, Any]] = []
    if "knn" in modes:
        for k in ks:
            for min_score in min_scores:
                for min_net in min_nets:
                    for min_tail in min_tails:
                        configs.append(
                            {
                                "selection_mode": "knn",
                                "knn_k": k,
                                "min_score": min_score,
                                "min_net": min_net,
                                "min_tail": min_tail,
                                "ridge_alpha": np.nan,
                                "lcb_z": np.nan,
                                "min_lcb": np.nan,
                            }
                        )
    if "ridge_lcb" in modes:
        for alpha in alphas:
            for lcb_z in lcb_values:
                for min_lcb in min_lcbs:
                    configs.append(
                        {
                            "selection_mode": "ridge_lcb",
                            "knn_k": np.nan,
                            "min_score": np.nan,
                            "min_net": np.nan,
                            "min_tail": np.nan,
                            "ridge_alpha": alpha,
                            "lcb_z": lcb_z,
                            "min_lcb": min_lcb,
                        }
                    )
    for config in configs:
        decisions = _walk_forward_decisions(
            joined,
            rules=rules,
            mode=str(config["selection_mode"]),
            k=int(config["knn_k"]) if pd.notna(config["knn_k"]) else 3,
            min_history=args.min_history,
            min_score=float(config["min_score"]) if pd.notna(config["min_score"]) else 0.0,
            min_net=float(config["min_net"]) if pd.notna(config["min_net"]) else 0.0,
            min_tail=float(config["min_tail"]) if pd.notna(config["min_tail"]) else 0.0,
            alpha=float(config["ridge_alpha"]) if pd.notna(config["ridge_alpha"]) else 1.0,
            lcb_z=float(config["lcb_z"]) if pd.notna(config["lcb_z"]) else 0.0,
            min_lcb=float(config["min_lcb"]) if pd.notna(config["min_lcb"]) else 0.0,
        )
        decisions = _complete_baseline_decisions(cache, decisions)
        stream = _build_selected_stream(cache, decisions, candidate_combo=args.candidate_combo)
        if stream.empty:
            continue
        replay_decisions, _equity, metrics = _run_replay(stream, args.market_mode)
        daily, weekly = _accepted_period_tables(replay_decisions)
        scores = _score_windows(
            daily,
            weekly,
            baseline_daily,
            baseline_weekly,
            validation_start=args.validation_start,
            june_start=args.june_start,
        )
        selected_share = float(decisions["selected_rule"].ne(BASELINE_ID).mean()) if not decisions.empty else 0.0
        meta = {
            **config,
            "selected_candidate_week_share": selected_share,
            "selected_weeks": int(decisions["selected_rule"].ne(BASELINE_ID).sum()) if not decisions.empty else 0,
            "trade_count": int(metrics.get("trade_count", 0) or 0),
            "net_pnl": float(metrics.get("net_pnl", 0.0)),
            "full_sl_rate": float(metrics.get("full_sl_rate", 0.0)),
        }
        for frame in (scores, decisions, daily, weekly):
            for key, value in meta.items():
                frame[key] = value
        all_scores.append(scores)
        all_decisions.append(decisions)
        all_daily.append(daily)
        all_weekly.append(weekly)

    scores_all = pd.concat(all_scores, ignore_index=True) if all_scores else pd.DataFrame()
    decisions_all = pd.concat(all_decisions, ignore_index=True) if all_decisions else pd.DataFrame()
    daily_all = pd.concat(all_daily, ignore_index=True) if all_daily else pd.DataFrame()
    weekly_all = pd.concat(all_weekly, ignore_index=True) if all_weekly else pd.DataFrame()
    scores_all.to_csv(args.out_dir / "state_selector_scores.csv", index=False)
    decisions_all.to_csv(args.out_dir / "state_selector_decisions.csv", index=False)
    daily_all.to_csv(args.out_dir / "state_selector_daily.csv", index=False)
    weekly_all.to_csv(args.out_dir / "state_selector_weekly.csv", index=False)

    show_cols = [
        "selection_mode",
        "knn_k",
        "min_score",
        "min_net",
        "min_tail",
        "ridge_alpha",
        "lcb_z",
        "min_lcb",
        "window",
        "selected_candidate_week_share",
        f"delta_{OBJECTIVE_COL}",
        "delta_net_pnl",
        "delta_weekly_q20_pnl",
        "delta_daily_q20_pnl",
        "pass_pnl_tail_gate",
    ]
    validation = scores_all.loc[scores_all["window"].eq("validation_may_june")].copy() if not scores_all.empty else pd.DataFrame()
    full = scores_all.loc[scores_all["window"].eq("full")].copy() if not scores_all.empty else pd.DataFrame()
    lines = [
        "# State-Aware Oracle Selector Replay",
        "",
        f"Feature table rows: `{len(joined)}`",
        f"State lag weeks: `{args.state_lag_weeks}`",
        f"Candidate rules: `{', '.join(rules)}`",
        "Selector is chronological and defaults to baseline unless gates pass. Costs are included.",
        "",
        "## Top Full-Period Selectors",
        "",
        full[[c for c in show_cols if c in full.columns]]
        .sort_values(f"delta_{OBJECTIVE_COL}", ascending=False)
        .head(20)
        .round(6)
        .to_markdown(index=False)
        if not full.empty
        else "_No rows._",
        "",
        "## Top May-June Selectors",
        "",
        validation[[c for c in show_cols if c in validation.columns]]
        .sort_values(f"delta_{OBJECTIVE_COL}", ascending=False)
        .head(20)
        .round(6)
        .to_markdown(index=False)
        if not validation.empty
        else "_No rows._",
    ]
    (args.out_dir / "state_selector_report.md").write_text("\n".join(lines) + "\n")
    payload = {
        "source_dir": str(args.source_dir),
        "baseline_dir": str(args.baseline_dir),
        "oracle_summary": str(args.oracle_summary),
        "out_dir": str(args.out_dir),
        "candidate_rules": rules,
        "state_lag_weeks": int(args.state_lag_weeks),
        "feature_rows": int(len(joined)),
        "score_rows": int(len(scores_all)),
    }
    (args.out_dir / "state_selector_summary.json").write_text(json.dumps(_json_safe(payload), indent=2))
    print(json.dumps(_json_safe(payload), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
