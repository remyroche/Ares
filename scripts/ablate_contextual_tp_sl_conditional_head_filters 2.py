#!/usr/bin/env python3
"""Replay contextual TP/SL combos with conditional weak-head filters.

The filter rules use already-materialized reliability diagnostics such as
recent hit-rate surprise, drift, OOD, and uncertainty. Use
``--threshold-mode expanding`` for a live-like replay where percentile
thresholds are computed from prior timestamps only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

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
from scripts.sweep_contextual_tp_sl_arm_combinations import (  # noqa: E402
    ARMS,
    _accepted_period_tables,
    _arm_combinations,
    _combo_id,
    _concat_nonempty,
    _head_name,
    _json_safe,
    _load_arm_tables,
    _load_requested_combo_ids,
    _period_metrics,
)


DEFAULT_RULES: Dict[str, Dict[str, Any]] = {
    "none": {},
    "weak_recent_hr_rank_m002": {
        "heads": ["short_asset", "long_bars"],
        "condition": "recent_hr_bad",
        "action": "rank",
        "value": -0.02,
    },
    "weak_recent_hr_priority_50": {
        "heads": ["short_asset", "long_bars"],
        "condition": "recent_hr_bad",
        "action": "priority",
        "value": 0.50,
    },
    "weak_recent_hr_size_50": {
        "heads": ["short_asset", "long_bars"],
        "condition": "recent_hr_bad",
        "action": "size",
        "value": 0.50,
    },
    "weak_recent_hr_ood_rank_m002": {
        "heads": ["short_asset", "long_bars"],
        "condition": "recent_hr_bad_or_ood_high",
        "action": "rank",
        "value": -0.02,
    },
    "weak_recent_hr_drift_rank_m002": {
        "heads": ["short_asset", "long_bars"],
        "condition": "recent_hr_bad_or_drift_high",
        "action": "rank",
        "value": -0.02,
    },
    "weak_all_bad_rank_m002": {
        "heads": ["short_asset", "long_bars"],
        "condition": "any_bad_reliability",
        "action": "rank",
        "value": -0.02,
    },
    "weak_recent_hr_two_signal_rank_m002": {
        "heads": ["short_asset", "long_bars"],
        "condition": "recent_hr_bad_and_any_ood_drift_uncertainty",
        "action": "rank",
        "value": -0.02,
    },
    "weak_two_of_four_rank_m002": {
        "heads": ["short_asset", "long_bars"],
        "condition": "two_of_four_bad_reliability",
        "action": "rank",
        "value": -0.02,
    },
    "weak_recent_hr_ood_uncertainty_size_50": {
        "heads": ["short_asset", "long_bars"],
        "condition": "recent_hr_bad_and_ood_or_uncertainty",
        "action": "size",
        "value": 0.50,
    },
    "weak_ood_uncertainty_rank_m002": {
        "heads": ["short_asset", "long_bars"],
        "condition": "ood_high_and_uncertainty_high",
        "action": "rank",
        "value": -0.02,
    },
    "weak_all_bad_priority_70": {
        "heads": ["short_asset", "long_bars"],
        "condition": "any_bad_reliability",
        "action": "priority",
        "value": 0.70,
    },
    "short_asset_recent_hr_rank_m002": {
        "heads": ["short_asset"],
        "condition": "recent_hr_bad",
        "action": "rank",
        "value": -0.02,
    },
    "long_bars_recent_hr_rank_m002": {
        "heads": ["long_bars"],
        "condition": "recent_hr_bad",
        "action": "rank",
        "value": -0.02,
    },
}

RECENT_HR_SURPRISE_COLS = (
    "generated_weighted_hr_surprise_24",
    "generated_weighted_hr_surprise_96",
    "generated_hr_surprise_24",
    "generated_hr_surprise_96",
    "dynamic_hr_surprise_z_eff",
)

OOD_COLS = (
    "generated_strategy_score_ood_abs_z",
    "generated_strategy_barrier_ood_abs_z",
    "generated_strategy_friction_ood_abs_z",
)

DRIFT_COLS = (
    "generated_score_abs_diff_1",
    "generated_score_abs_diff_4",
    "generated_score_abs_diff_24",
    "generated_score_abs_minus_prev24_mean",
    "generated_score_prev24_std",
    "generated_strategy_score_shift_abs_z",
    "feature_drift_psi_core",
    "feature_drift_ks_core",
    "feature_drift_cov_shift",
    "row_drift_v1_psi_core",
    "row_drift_v1_ks_core",
    "row_drift_v1_mahalanobis_mean_shift",
    "row_drift_v1_contribution_drift_score",
    "inference_drift_score",
    "meta_lgbm_feature_drift_psi_core",
    "meta_lgbm_feature_drift_ks_core",
    "meta_lgbm_feature_drift_cov_shift",
    "meta_lgbm_row_drift_v1_psi_core",
    "meta_lgbm_row_drift_v1_ks_core",
    "meta_lgbm_row_drift_v1_mahalanobis_mean_shift",
    "meta_lgbm_row_drift_v1_contribution_drift_score",
    "meta_lgbm_inference_drift_score",
    "oof_feature_drift_psi_core",
    "oof_feature_drift_ks_core",
    "oof_feature_drift_cov_shift",
    "oof_latent_mahalanobis_drift",
)

UNCERTAINTY_HIGH_COLS = (
    "generated_score_uncertainty_p1mp",
    "generated_score_entropy",
    "prob_uncertainty",
    "uncertainty_score",
    "row_drift_v1_uncertainty_score",
    "meta_lgbm_uncertainty_score",
    "meta_lgbm_prob_uncertainty",
    "meta_lgbm_row_drift_v1_uncertainty_score",
    "oof_prob_uncertainty",
    "base_lgbm_prob_uncertainty",
)

UNCERTAINTY_LOW_COLS = (
    "generated_score_abs_distance_from_half",
)


def _numeric(frame: pd.DataFrame, col: str, default: float = np.nan) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[col], errors="coerce")


def _q_by_head(values: pd.Series, head: pd.Series, q: float) -> pd.Series:
    out = pd.Series(np.nan, index=values.index, dtype=float)
    for _, idx in head.groupby(head, sort=False).groups.items():
        vals = values.loc[idx].dropna()
        if vals.empty:
            continue
        out.loc[idx] = float(vals.quantile(float(q)))
    return out


def _expanding_q_by_head_timestamp(
    values: pd.Series,
    head: pd.Series,
    timestamp: pd.Series,
    q: float,
    min_history: int,
) -> pd.Series:
    """Prior-timestamp expanding quantile by head.

    Every row at timestamp t receives a threshold fitted on rows from earlier
    timestamps for the same head. Rows at the same timestamp are never used to
    score one another.
    """

    out = pd.Series(np.nan, index=values.index, dtype=float)
    work = pd.DataFrame(
        {
            "value": pd.to_numeric(values, errors="coerce"),
            "head": head.astype(str),
            "timestamp": pd.to_datetime(timestamp, utc=True, errors="coerce"),
            "_order": np.arange(len(values), dtype=np.int64),
        },
        index=values.index,
    ).dropna(subset=["timestamp"])
    min_history = max(1, int(min_history))
    q = float(q)
    for _, head_frame in work.groupby("head", sort=False):
        ordered = head_frame.sort_values(["timestamp", "_order"], kind="mergesort")
        row_threshold = (
            ordered["value"]
            .expanding(min_periods=min_history)
            .quantile(q)
            .shift(1)
        )
        threshold_frame = pd.DataFrame(
            {"timestamp": ordered["timestamp"].to_numpy(), "threshold": row_threshold.to_numpy()},
            index=ordered.index,
        )
        timestamp_threshold = threshold_frame.groupby("timestamp", sort=False)["threshold"].transform("first")
        out.loc[ordered.index] = timestamp_threshold.to_numpy(dtype=float)
    return out


def _threshold_by_head(
    candidates: pd.DataFrame,
    values: pd.Series,
    head: pd.Series,
    q: float,
    threshold_mode: str,
    min_history: int,
) -> pd.Series:
    if threshold_mode == "full_sample":
        return _q_by_head(values, head, q)
    if threshold_mode == "expanding":
        if "timestamp" not in candidates.columns:
            return pd.Series(np.nan, index=values.index, dtype=float)
        return _expanding_q_by_head_timestamp(values, head, candidates["timestamp"], q, min_history)
    raise ValueError(f"Unknown threshold mode `{threshold_mode}`")


def _family_threshold_flag(
    candidates: pd.DataFrame,
    *,
    cols: Sequence[str],
    head: pd.Series,
    q: float,
    direction: str,
    threshold_mode: str,
    min_history: int,
) -> pd.Series:
    flag = pd.Series(False, index=candidates.index, dtype=bool)
    for col in cols:
        if col not in candidates.columns:
            continue
        values = _numeric(candidates, col)
        threshold = _threshold_by_head(candidates, values, head, q, threshold_mode, min_history)
        if direction == "high":
            flag |= values.ge(threshold).fillna(False)
        elif direction == "low":
            flag |= values.le(threshold).fillna(False)
        else:
            raise ValueError(f"Unknown flag direction `{direction}`")
    return flag


def _condition_mask(
    candidates: pd.DataFrame,
    condition: str,
    *,
    threshold_mode: str = "full_sample",
    min_history: int = 500,
) -> pd.Series:
    cached = {
        "recent_hr_bad": "_cond_recent_hr_bad",
        "ood_high": "_cond_ood_high",
        "drift_high": "_cond_drift_high",
        "uncertainty_high": "_cond_uncertainty_high",
    }
    if condition in cached and cached[condition] in candidates.columns:
        return candidates[cached[condition]].astype(bool)
    required_flags = ("_cond_recent_hr_bad", "_cond_ood_high", "_cond_drift_high", "_cond_uncertainty_high")
    if not all(c in candidates.columns for c in required_flags):
        candidates = _add_condition_flags(
            candidates,
            threshold_mode=threshold_mode,
            min_history=min_history,
        )
    if condition in cached and cached[condition] in candidates.columns:
        return candidates[cached[condition]].astype(bool)
    if condition == "recent_hr_bad_or_ood_high" and all(
        c in candidates.columns for c in ("_cond_recent_hr_bad", "_cond_ood_high")
    ):
        return (candidates["_cond_recent_hr_bad"].astype(bool) | candidates["_cond_ood_high"].astype(bool))
    if condition == "recent_hr_bad_or_drift_high" and all(
        c in candidates.columns for c in ("_cond_recent_hr_bad", "_cond_drift_high")
    ):
        return (candidates["_cond_recent_hr_bad"].astype(bool) | candidates["_cond_drift_high"].astype(bool))
    if condition == "any_bad_reliability" and all(
        c in candidates.columns
        for c in ("_cond_recent_hr_bad", "_cond_ood_high", "_cond_drift_high", "_cond_uncertainty_high")
    ):
        return (
            candidates["_cond_recent_hr_bad"].astype(bool)
            | candidates["_cond_ood_high"].astype(bool)
            | candidates["_cond_drift_high"].astype(bool)
            | candidates["_cond_uncertainty_high"].astype(bool)
        )
    if condition == "recent_hr_bad_and_any_ood_drift_uncertainty" and all(
        c in candidates.columns
        for c in ("_cond_recent_hr_bad", "_cond_ood_high", "_cond_drift_high", "_cond_uncertainty_high")
    ):
        return (
            candidates["_cond_recent_hr_bad"].astype(bool)
            & (
                candidates["_cond_ood_high"].astype(bool)
                | candidates["_cond_drift_high"].astype(bool)
                | candidates["_cond_uncertainty_high"].astype(bool)
            )
        )
    if condition == "recent_hr_bad_and_ood_or_uncertainty" and all(
        c in candidates.columns for c in ("_cond_recent_hr_bad", "_cond_ood_high", "_cond_uncertainty_high")
    ):
        return (
            candidates["_cond_recent_hr_bad"].astype(bool)
            & (
                candidates["_cond_ood_high"].astype(bool)
                | candidates["_cond_uncertainty_high"].astype(bool)
            )
        )
    if condition == "ood_high_and_uncertainty_high" and all(
        c in candidates.columns for c in ("_cond_ood_high", "_cond_uncertainty_high")
    ):
        return candidates["_cond_ood_high"].astype(bool) & candidates["_cond_uncertainty_high"].astype(bool)
    if condition == "two_of_four_bad_reliability" and all(
        c in candidates.columns
        for c in ("_cond_recent_hr_bad", "_cond_ood_high", "_cond_drift_high", "_cond_uncertainty_high")
    ):
        count = (
            candidates["_cond_recent_hr_bad"].astype(np.int8)
            + candidates["_cond_ood_high"].astype(np.int8)
            + candidates["_cond_drift_high"].astype(np.int8)
            + candidates["_cond_uncertainty_high"].astype(np.int8)
        )
        return count.ge(2)
    raise ValueError(f"Unknown condition `{condition}`")


def _add_condition_flags(
    candidates: pd.DataFrame,
    *,
    threshold_mode: str,
    min_history: int,
) -> pd.DataFrame:
    out = candidates.copy()
    out["_condition_head"] = out["strategy_id"].astype(str).map(_head_name)
    head = out["_condition_head"].astype(str)
    out["_cond_recent_hr_bad"] = _family_threshold_flag(
        out,
        cols=RECENT_HR_SURPRISE_COLS,
        head=head,
        q=0.25,
        direction="low",
        threshold_mode=threshold_mode,
        min_history=min_history,
    )
    out["_cond_ood_high"] = _family_threshold_flag(
        out,
        cols=OOD_COLS,
        head=head,
        q=0.80,
        direction="high",
        threshold_mode=threshold_mode,
        min_history=min_history,
    )
    out["_cond_drift_high"] = _family_threshold_flag(
        out,
        cols=DRIFT_COLS,
        head=head,
        q=0.80,
        direction="high",
        threshold_mode=threshold_mode,
        min_history=min_history,
    )
    uncertainty_high = _family_threshold_flag(
        out,
        cols=UNCERTAINTY_HIGH_COLS,
        head=head,
        q=0.80,
        direction="high",
        threshold_mode=threshold_mode,
        min_history=min_history,
    )
    uncertainty_low_distance = _family_threshold_flag(
        out,
        cols=UNCERTAINTY_LOW_COLS,
        head=head,
        q=0.20,
        direction="low",
        threshold_mode=threshold_mode,
        min_history=min_history,
    )
    out["_cond_uncertainty_high"] = uncertainty_high | uncertainty_low_distance
    return out


def _apply_rule(
    candidates: pd.DataFrame,
    rule: Mapping[str, Any],
    *,
    threshold_mode: str = "full_sample",
    min_history: int = 500,
) -> pd.DataFrame:
    if not rule:
        return candidates
    out = candidates.copy()
    if "_condition_head" not in out.columns:
        out["_condition_head"] = out["strategy_id"].astype(str).map(_head_name)
    head = out["_condition_head"].astype(str)
    target_heads = set(str(v) for v in rule.get("heads", []))
    if not target_heads:
        return out
    mask = head.isin(target_heads) & _condition_mask(
        out,
        str(rule.get("condition", "")),
        threshold_mode=threshold_mode,
        min_history=min_history,
    )
    action = str(rule.get("action", "")).strip()
    value = float(rule.get("value", 0.0))
    for col, default in (
        ("portfolio_size_multiplier", 1.0),
        ("portfolio_priority_multiplier", 1.0),
        ("portfolio_rank_adjustment", 0.0),
    ):
        if col not in out.columns:
            out[col] = default
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(default)
    if action == "size":
        out.loc[mask, "portfolio_size_multiplier"] = out.loc[mask, "portfolio_size_multiplier"] * value
    elif action == "priority":
        out.loc[mask, "portfolio_priority_multiplier"] = (
            out.loc[mask, "portfolio_priority_multiplier"] * value
        )
    elif action == "rank":
        out.loc[mask, "portfolio_rank_adjustment"] = out.loc[mask, "portfolio_rank_adjustment"] + value
    else:
        raise ValueError(f"Unknown action `{action}`")
    out["conditional_filter_bound"] = mask.astype(np.int8)
    return out


def _load_rules(rule_file: Path | None, include_default: bool) -> Dict[str, Dict[str, Any]]:
    rules = dict(DEFAULT_RULES) if include_default or rule_file is None else {}
    if rule_file is None:
        return rules
    payload = json.loads(rule_file.read_text())
    if not isinstance(payload, dict):
        raise ValueError("rule file must contain a JSON object")
    for name, rule in payload.items():
        if not isinstance(rule, dict):
            raise ValueError(f"rule `{name}` must be an object")
        rules[str(name)] = rule
    return rules


def _head_delta_table(accepted: pd.DataFrame, baseline_combo: str, baseline_rule: str) -> pd.DataFrame:
    if accepted.empty:
        return pd.DataFrame()
    work = accepted.copy()
    work["head"] = work["strategy_id"].astype(str).map(_head_name)
    size = pd.to_numeric(work["position_size"], errors="coerce").fillna(0.0)
    net = pd.to_numeric(work["position_net_return"], errors="coerce").fillna(0.0)
    gross = pd.to_numeric(work["position_gross_return"], errors="coerce").fillna(0.0)
    work["net_pnl"] = size * net
    work["gross_pnl"] = size * gross
    work["hit"] = net.gt(0.0)
    work["full_sl"] = work["position_exit_reason"].astype(str).eq("full_sl")
    grouped = (
        work.groupby(["combo_id", "rule_id", "head"], dropna=False)
        .agg(
            trades=("head", "size"),
            net_pnl=("net_pnl", "sum"),
            gross_pnl=("gross_pnl", "sum"),
            hit_rate=("hit", "mean"),
            full_sl_rate=("full_sl", "mean"),
        )
        .reset_index()
    )
    baseline = grouped.loc[
        grouped["combo_id"].eq(baseline_combo) & grouped["rule_id"].eq(baseline_rule)
    ].set_index("head")
    rows: List[Dict[str, Any]] = []
    for (combo_id, rule_id), group in grouped.groupby(["combo_id", "rule_id"], sort=False):
        current = group.set_index("head")
        for h in sorted(set(baseline.index).union(current.index)):
            rec = {"combo_id": combo_id, "rule_id": rule_id, "head": h}
            for col in ("trades", "net_pnl", "gross_pnl", "hit_rate", "full_sl_rate"):
                b = float(baseline.loc[h, col]) if h in baseline.index else 0.0
                c = float(current.loc[h, col]) if h in current.index else 0.0
                rec[f"baseline_{col}"] = b
                rec[f"candidate_{col}"] = c
                rec[f"delta_{col}"] = c - b
            rows.append(rec)
    return pd.DataFrame(rows)


def _load_flat_candidate_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing flat candidate table: {path}")
    frame = pd.read_parquet(path)
    if "strategy_id" not in frame.columns:
        raise ValueError(f"{path} must contain strategy_id")
    frame = frame.copy()
    frame["strategy_id"] = frame["strategy_id"].astype(str)
    mapped_head = frame["strategy_id"].map(_head_name)
    if "head" in frame.columns:
        existing = frame["head"].astype("string")
        frame["head"] = existing.where(existing.notna() & existing.ne(""), mapped_head)
    else:
        frame["head"] = mapped_head
    return frame


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=None)
    parser.add_argument(
        "--flat-candidate-table",
        type=Path,
        default=None,
        help=(
            "Optional single candidate parquet to replay directly. This is for "
            "already materialized flat candidate ledgers and bypasses arm table loading."
        ),
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--market-mode", default="perps", choices=["spot", "perps"])
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--combo-id", action="append", default=None)
    parser.add_argument("--combo-file", type=Path, default=None)
    parser.add_argument("--rule-file", type=Path, default=None)
    parser.add_argument("--include-default-rules", action="store_true")
    parser.add_argument("--save-accepted-decisions", action="store_true")
    parser.add_argument(
        "--threshold-mode",
        default="full_sample",
        choices=["full_sample", "expanding"],
        help="How reliability diagnostic percentile thresholds are fitted.",
    )
    parser.add_argument(
        "--min-threshold-history",
        type=int,
        default=500,
        help="Minimum prior rows per head before an expanding threshold can bind.",
    )
    args = parser.parse_args()

    arms = tuple(a.strip() for a in str(args.arms).split(",") if a.strip())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.flat_candidate_table is None and args.source_dir is None:
        raise ValueError("Provide either --source-dir or --flat-candidate-table")
    if args.flat_candidate_table is not None and args.source_dir is not None:
        raise ValueError("Use either --source-dir or --flat-candidate-table, not both")
    flat_table: pd.DataFrame | None = None
    if args.flat_candidate_table is not None:
        flat_table = _load_flat_candidate_table(args.flat_candidate_table)
        heads = sorted(flat_table["head"].dropna().astype(str).unique())
        source_label = str(args.flat_candidate_table)
    else:
        tables = _load_arm_tables(args.source_dir, arms)
        heads = sorted(tables[arms[0]]["head"].dropna().astype(str).unique())
        source_label = str(args.source_dir)
    requested_combo_ids = _load_requested_combo_ids(args.combo_id, args.combo_file)
    rules = _load_rules(args.rule_file, args.include_default_rules)
    params = PortfolioPolicyParams(global_threshold_floor=0.0)

    rows: List[Dict[str, Any]] = []
    daily_frames: List[pd.DataFrame] = []
    weekly_frames: List[pd.DataFrame] = []
    accepted_frames: List[pd.DataFrame] = []
    if flat_table is not None:
        replay_inputs = [
            (
                "flat_candidate_set",
                {head_name: "flat" for head_name in heads},
                flat_table.drop(columns=["head"], errors="ignore")
                .sort_values(["timestamp", "strategy_id", "symbol"])
                .reset_index(drop=True),
            )
        ]
    else:
        replay_inputs = []
        for mapping in _arm_combinations(heads, arms):
            combo_id = _combo_id(mapping)
            if requested_combo_ids and combo_id not in requested_combo_ids:
                continue
            frames = []
            for head_name, arm in mapping.items():
                source = tables[arm]
                frames.append(source.loc[source["head"].eq(head_name)].copy())
            base_candidates = (
                pd.concat(frames, ignore_index=True)
                .drop(columns=["head"], errors="ignore")
                .sort_values(["timestamp", "strategy_id", "symbol"])
                .reset_index(drop=True)
            )
            replay_inputs.append((combo_id, mapping, base_candidates))

    for combo_id, mapping, base_candidates in replay_inputs:
        base_candidates = _add_condition_flags(
            base_candidates,
            threshold_mode=args.threshold_mode,
            min_history=args.min_threshold_history,
        )
        for rule_id, rule in rules.items():
            candidates = _apply_rule(
                base_candidates,
                rule,
                threshold_mode=args.threshold_mode,
                min_history=args.min_threshold_history,
            )
            bound_count = int(candidates.get("conditional_filter_bound", pd.Series(0, index=candidates.index)).sum())
            ev_curve = fit_hierarchical_ev_curves(candidates)
            decisions, _equity, metrics = replay_candidates(
                candidates,
                params,
                mode="global_auction",
                ev_curve=ev_curve,
                market_mode=args.market_mode,
            )
            daily, weekly = _accepted_period_tables(decisions)
            for frame in (daily, weekly):
                if not frame.empty:
                    frame.insert(0, "combo_id", combo_id)
                    frame.insert(1, "rule_id", rule_id)
                    frame.insert(2, "rule_spec", json.dumps(rule, sort_keys=True))
                    for head_name, arm in mapping.items():
                        frame[f"{head_name}_arm"] = arm
            daily_frames.append(daily)
            weekly_frames.append(weekly)
            if args.save_accepted_decisions and "accepted" in decisions.columns:
                accepted = decisions.loc[decisions["accepted"].astype(bool)].copy()
                if not accepted.empty:
                    accepted.insert(0, "combo_id", combo_id)
                    accepted.insert(1, "rule_id", rule_id)
                    accepted.insert(2, "rule_spec", json.dumps(rule, sort_keys=True))
                    for head_name, arm in mapping.items():
                        accepted[f"{head_name}_arm"] = arm
                    accepted_frames.append(accepted)
            rec = {
                "combo_id": combo_id,
                "rule_id": rule_id,
                "rule_spec": json.dumps(rule, sort_keys=True),
                "threshold_mode": args.threshold_mode,
                "min_threshold_history": int(args.min_threshold_history),
                "bound_candidate_rows": bound_count,
                **{f"{head_name}_arm": arm for head_name, arm in mapping.items()},
                "candidate_rows": int(len(candidates)),
                "candidate_start": str(pd.to_datetime(candidates["timestamp"], utc=True).min()),
                "candidate_end": str(pd.to_datetime(candidates["timestamp"], utc=True).max()),
                "objective": float(metrics.get("objective", 0.0)),
                "net_pnl": float(metrics.get("net_pnl", 0.0)),
                "gross_pnl": float(metrics.get("gross_pnl", 0.0)),
                "trade_count": int(metrics.get("trade_count", 0) or 0),
                "mean_net_return": float(metrics.get("mean_net_return_per_trade", 0.0)),
                "full_sl_rate": float(metrics.get("full_sl_rate", 0.0)),
                "timeout_rate": float(metrics.get("timeout_rate", 0.0)),
                "max_drawdown": float(metrics.get("max_drawdown", 0.0)),
                "worst_week_return": float(metrics.get("worst_week", 0.0)),
            }
            rec.update(_period_metrics(daily.get("net_pnl", pd.Series(dtype=float)), "daily"))
            rec.update(_period_metrics(weekly.get("net_pnl", pd.Series(dtype=float)), "weekly"))
            rows.append(rec)

    summary = pd.DataFrame(rows)
    if not summary.empty:
        weekly_count = pd.to_numeric(summary["weekly_count"], errors="coerce").replace(0.0, np.nan)
        summary["avg_week_pnl"] = pd.to_numeric(summary["net_pnl"], errors="coerce") / weekly_count
        summary["objective_avgweek_0p7dayq35_0p3dayq20"] = (
            summary["avg_week_pnl"].fillna(0.0)
            + 0.7 * pd.to_numeric(summary["daily_q35_pnl"], errors="coerce").fillna(0.0)
            + 0.3 * pd.to_numeric(summary["daily_q20_pnl"], errors="coerce").fillna(0.0)
        )
        summary = summary.sort_values(
            "objective_avgweek_0p7dayq35_0p3dayq20",
            ascending=False,
        ).reset_index(drop=True)
    daily_all = _concat_nonempty(daily_frames)
    weekly_all = _concat_nonempty(weekly_frames)
    accepted_all = _concat_nonempty(accepted_frames)
    summary.to_csv(args.out_dir / "conditional_filter_summary.csv", index=False)
    daily_all.to_csv(args.out_dir / "conditional_filter_daily.csv", index=False)
    weekly_all.to_csv(args.out_dir / "conditional_filter_weekly.csv", index=False)
    if args.save_accepted_decisions:
        accepted_all.to_parquet(args.out_dir / "conditional_filter_accepted_decisions.parquet", index=False)
        baseline_combo = (
            "flat_candidate_set"
            if args.flat_candidate_table is not None
            else "long_bars:S_long_dist:S_short_asset:S_short_bollinger:S"
        )
        head_delta = _head_delta_table(
            accepted_all,
            baseline_combo,
            "none",
        )
        head_delta.to_csv(args.out_dir / "conditional_filter_per_head_delta_vs_static_none.csv", index=False)

    keep = [
        "combo_id",
        "rule_id",
        "objective_avgweek_0p7dayq35_0p3dayq20",
        "avg_week_pnl",
        "net_pnl",
        "trade_count",
        "bound_candidate_rows",
        "daily_q20_pnl",
        "daily_q35_pnl",
        "weekly_q05_pnl",
        "weekly_q10_pnl",
        "weekly_q20_pnl",
        "weekly_q35_pnl",
        "full_sl_rate",
        "max_drawdown",
        "rule_spec",
    ]
    lines = [
        "# Conditional Weak-Head Filter Ablation",
        "",
        f"Source: `{source_label}`",
        f"Source mode: `{'flat_candidate_table' if args.flat_candidate_table is not None else 'arm_tables'}`",
        f"Threshold mode: `{args.threshold_mode}`",
        f"Minimum threshold history: `{args.min_threshold_history}`",
        f"Rows: `{len(summary)}`",
        "Period: full source candidate table period. Costs included.",
        "",
        "## Top Requested Objective",
        "",
        summary[[c for c in keep if c in summary.columns]].head(30).round(6).to_markdown(index=False)
        if not summary.empty
        else "_No rows._",
    ]
    (args.out_dir / "conditional_filter_report.md").write_text("\n".join(lines) + "\n")
    payload = {
        "source": source_label,
        "source_mode": "flat_candidate_table" if args.flat_candidate_table is not None else "arm_tables",
        "out_dir": str(args.out_dir),
        "threshold_mode": args.threshold_mode,
        "min_threshold_history": int(args.min_threshold_history),
        "combo_ids": sorted(requested_combo_ids),
        "rules": rules,
        "rows": int(len(summary)),
        "top_requested_objective": summary.head(10).to_dict(orient="records"),
    }
    (args.out_dir / "conditional_filter_summary.json").write_text(json.dumps(_json_safe(payload), indent=2))
    print(json.dumps(_json_safe({"out_dir": str(args.out_dir), "rows": len(summary)}), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
