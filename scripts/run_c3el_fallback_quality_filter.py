#!/usr/bin/env python3
"""Evaluate lightweight quality filters for C3el fallback actions.

This is a no-replay diagnostic.  It joins selected fallback actions to exact
cloned-state labels and deployable action-feature rows, then tests simple
feature guards that would keep or reject fallback cuts.  The purpose is to
decide whether the fallback state is learnable enough to justify a future
head-native replay, without starting another memory-heavy portfolio run.
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


KEYS = ["timestamp", "strategy_id", "multiplier_round"]
ALLOWED_MODEL_OUTPUT_FEATURES = {"p_intervene", "pred_action_delta_J"}
LEAK_TERMS = (
    "delta_",
    "base_",
    "action_",
    "candidate_",
    "baseline_",
    "direct_",
    "net_pnl",
    "gross_pnl",
    "cost_pnl",
    "full_j",
    "immediate_j",
    "turnover",
    "trade_count",
    "full_sl",
    "timeout",
    "is_baseline_action",
    "exact_positive",
    "label",
    "target",
    "future",
    "pnl",
)


def _read_frame(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _safe_num(value: Any, default: float = 0.0) -> pd.Series:
    if isinstance(value, pd.Series):
        return pd.to_numeric(value, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(default)
    return pd.Series(dtype=float)


def _normalise_selected_actions(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out = out.loc[out["timestamp"].notna()].copy()
    out["strategy_id"] = out["strategy_id"].astype(str)
    if "multiplier" in out.columns:
        multiplier = pd.to_numeric(out["multiplier"], errors="coerce")
    elif "action_value" in out.columns:
        multiplier = pd.to_numeric(out["action_value"], errors="coerce")
    else:
        raise ValueError("selected fallback actions must contain multiplier or action_value")
    if "action_value" in out.columns:
        multiplier = multiplier.fillna(pd.to_numeric(out["action_value"], errors="coerce"))
    out["multiplier_round"] = multiplier.fillna(1.0).round(6)
    for col in ["delta_full_J", "delta_immediate_J", "direct_delta_net_pnl"]:
        if col in out.columns:
            out[col] = _safe_num(out[col])
        else:
            out[col] = 0.0
    out["exact_positive_e50"] = out["delta_full_J"].gt(50.0)
    return out


def _normalise_action_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out = out.loc[out["timestamp"].notna()].copy()
    out["strategy_id"] = out["strategy_id"].astype(str)
    if "multiplier" not in out.columns:
        raise ValueError("action feature frame must contain multiplier")
    out["multiplier_round"] = pd.to_numeric(out["multiplier"], errors="coerce").round(6)
    return out


def _is_deployable_feature(col: str) -> bool:
    if col in ALLOWED_MODEL_OUTPUT_FEATURES:
        return True
    lower = col.lower()
    if lower.endswith("_label"):
        return False
    return not any(term in lower for term in LEAK_TERMS)


def _feature_columns(frame: pd.DataFrame, *, min_non_null: int) -> list[str]:
    cols: list[str] = []
    excluded = {
        "timestamp",
        "strategy_id",
        "multiplier_round",
        "exact_positive_e50",
        "delta_full_J",
        "delta_immediate_J",
        "direct_delta_net_pnl",
    }
    for col in frame.columns:
        if col in excluded or not _is_deployable_feature(str(col)):
            continue
        if not pd.api.types.is_numeric_dtype(frame[col]):
            continue
        vals = pd.to_numeric(frame[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if vals.notna().sum() >= int(min_non_null) and vals.nunique(dropna=True) > 1:
            cols.append(str(col))
    return cols


def _join_labels_and_features(fallback_actions: Path, action_features: Path) -> pd.DataFrame:
    labels = _normalise_selected_actions(_read_frame(fallback_actions))
    features = _normalise_action_features(_read_frame(action_features))
    joined = labels.merge(features, on=KEYS, how="left", suffixes=("_label", ""))
    if "multiplier" in joined.columns:
        joined["feature_row_matched"] = joined["multiplier"].notna()
    else:
        joined["feature_row_matched"] = False
    return joined


def _rule_mask(frame: pd.DataFrame, feature: str, direction: str, threshold: float) -> pd.Series:
    vals = pd.to_numeric(frame.get(feature), errors="coerce")
    if direction == "ge":
        return vals.ge(float(threshold)).fillna(False)
    if direction == "le":
        return vals.le(float(threshold)).fillna(False)
    raise ValueError(f"Unknown direction: {direction}")


def _summarise_selection(frame: pd.DataFrame, selected: pd.Series, *, name: str, rule: dict[str, Any]) -> dict[str, Any]:
    selected = selected.fillna(False).astype(bool)
    kept = frame.loc[selected].copy()
    rejected = frame.loc[~selected].copy()
    delta = _safe_num(kept["delta_full_J"]) if not kept.empty else pd.Series(dtype=float)
    rejected_delta = _safe_num(rejected["delta_full_J"]) if not rejected.empty else pd.Series(dtype=float)
    positives = kept["exact_positive_e50"].astype(bool) if not kept.empty else pd.Series(dtype=bool)
    return {
        "rule_name": str(name),
        "rule": json.dumps(rule, sort_keys=True, separators=(",", ":"), default=str),
        "keep_count": int(selected.sum()),
        "reject_count": int((~selected).sum()),
        "keep_rate": float(selected.mean()) if len(selected) else 0.0,
        "positive_e50_count": int(positives.sum()) if len(positives) else 0,
        "positive_e50_rate": float(positives.mean()) if len(positives) else 0.0,
        "delta_full_J_sum": float(delta.sum()) if len(delta) else 0.0,
        "delta_full_J_mean": float(delta.mean()) if len(delta) else 0.0,
        "delta_full_J_median": float(delta.median()) if len(delta) else 0.0,
        "delta_full_J_worst": float(delta.min()) if len(delta) else 0.0,
        "delta_immediate_J_sum": float(_safe_num(kept["delta_immediate_J"]).sum()) if not kept.empty else 0.0,
        "direct_delta_net_pnl_sum": float(_safe_num(kept["direct_delta_net_pnl"]).sum()) if not kept.empty else 0.0,
        "rejected_delta_full_J_sum": float(rejected_delta.sum()) if len(rejected_delta) else 0.0,
        "rejected_negative_delta_full_J_sum": float(rejected_delta.clip(upper=0.0).sum()) if len(rejected_delta) else 0.0,
        "rejected_positive_delta_full_J_sum": float(rejected_delta.clip(lower=0.0).sum()) if len(rejected_delta) else 0.0,
    }


def _candidate_rules_for_feature(frame: pd.DataFrame, feature: str, *, quantiles: list[float]) -> list[dict[str, Any]]:
    vals = pd.to_numeric(frame[feature], errors="coerce").replace([np.inf, -np.inf], np.nan)
    thresholds = sorted({float(vals.quantile(q)) for q in quantiles if np.isfinite(vals.quantile(q))})
    rules: list[dict[str, Any]] = []
    for threshold in thresholds:
        rules.append({"feature": feature, "direction": "ge", "threshold": threshold})
        rules.append({"feature": feature, "direction": "le", "threshold": threshold})
    return rules


def _evaluate_rule(frame: pd.DataFrame, rule: dict[str, Any]) -> pd.Series:
    if "rules" in rule:
        masks = [
            _rule_mask(frame, str(part["feature"]), str(part["direction"]), float(part["threshold"]))
            for part in list(rule["rules"])
        ]
        if not masks:
            return pd.Series(False, index=frame.index)
        out = masks[0].copy()
        for mask in masks[1:]:
            out &= mask
        return out
    return _rule_mask(frame, str(rule["feature"]), str(rule["direction"]), float(rule["threshold"]))


def _objective(row: pd.Series, *, min_keep: int, min_positive_rate: float) -> float:
    if int(row.get("keep_count", 0)) < int(min_keep):
        return float("-inf")
    if float(row.get("positive_e50_rate", 0.0)) < float(min_positive_rate):
        return float("-inf")
    return (
        float(row.get("delta_full_J_sum", 0.0))
        + 0.25 * float(row.get("rejected_negative_delta_full_J_sum", 0.0)) * -1.0
        + 50.0 * float(row.get("positive_e50_rate", 0.0))
        + min(float(row.get("delta_full_J_worst", 0.0)), 0.0)
    )


def _evaluate_rules(
    frame: pd.DataFrame,
    rules: list[dict[str, Any]],
    *,
    min_keep: int,
    min_positive_rate: float,
) -> pd.DataFrame:
    rows = [_summarise_selection(frame, pd.Series(True, index=frame.index), name="no_filter", rule={"type": "no_filter"})]
    for idx, rule in enumerate(rules):
        mask = _evaluate_rule(frame, rule)
        row = _summarise_selection(frame, mask, name=f"rule_{idx}", rule=rule)
        rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["objective"] = out.apply(lambda row: _objective(row, min_keep=min_keep, min_positive_rate=min_positive_rate), axis=1)
    return out.sort_values(["objective", "delta_full_J_sum", "positive_e50_rate"], ascending=[False, False, False]).reset_index(drop=True)


def _leave_one_day(frame: pd.DataFrame, rules: list[dict[str, Any]], *, min_keep: int, min_positive_rate: float) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    days = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce").dt.floor("D")
    rows: list[dict[str, Any]] = []
    for day in sorted(days.dropna().unique()):
        train = frame.loc[days.ne(day)].copy()
        test = frame.loc[days.eq(day)].copy()
        if train.empty or test.empty:
            continue
        train_eval = _evaluate_rules(train, rules, min_keep=max(1, min(int(min_keep), len(train))), min_positive_rate=min_positive_rate)
        candidate = train_eval.loc[train_eval["rule_name"].ne("no_filter")].head(1)
        if candidate.empty:
            candidate = train_eval.head(1)
        rule = json.loads(str(candidate.iloc[0]["rule"]))
        selected = _evaluate_rule(test, rule) if rule.get("type") != "no_filter" else pd.Series(True, index=test.index)
        row = _summarise_selection(test, selected, name="loo_selected", rule=rule)
        row["heldout_day"] = str(pd.Timestamp(day).date())
        row["train_objective"] = float(candidate.iloc[0]["objective"])
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fallback-actions", type=Path, required=True)
    parser.add_argument("--action-features", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--min-keep", type=int, default=8)
    parser.add_argument("--min-positive-rate", type=float, default=0.65)
    parser.add_argument("--top-features-for-pairs", type=int, default=12)
    parser.add_argument("--quantiles", default="0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90")
    args = parser.parse_args()

    quantiles = [float(x.strip()) for x in str(args.quantiles).split(",") if x.strip()]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    frame = _join_labels_and_features(args.fallback_actions, args.action_features)
    frame.to_csv(args.out_dir / "fallback_quality_joined_rows.csv", index=False)

    features = _feature_columns(frame, min_non_null=max(5, min(len(frame), int(args.min_keep))))
    single_rules: list[dict[str, Any]] = []
    for feature in features:
        single_rules.extend(_candidate_rules_for_feature(frame, feature, quantiles=quantiles))
    single_eval = _evaluate_rules(
        frame,
        single_rules,
        min_keep=int(args.min_keep),
        min_positive_rate=float(args.min_positive_rate),
    )
    single_eval.to_csv(args.out_dir / "single_feature_filter_trials.csv", index=False)

    top_single = single_eval.loc[single_eval["rule_name"].ne("no_filter")].head(int(args.top_features_for_pairs)).copy()
    pair_rules: list[dict[str, Any]] = []
    parsed = [json.loads(str(rule)) for rule in top_single["rule"].tolist()]
    for left, right in combinations(parsed, 2):
        if left.get("feature") == right.get("feature"):
            continue
        pair_rules.append({"rules": [left, right]})
    pair_eval = _evaluate_rules(
        frame,
        pair_rules,
        min_keep=int(args.min_keep),
        min_positive_rate=float(args.min_positive_rate),
    )
    pair_eval.to_csv(args.out_dir / "pair_feature_filter_trials.csv", index=False)

    all_trials = pd.concat([single_eval.assign(rule_family="single"), pair_eval.assign(rule_family="pair")], ignore_index=True)
    all_trials = all_trials.sort_values(["objective", "delta_full_J_sum", "positive_e50_rate"], ascending=[False, False, False])
    all_trials.to_csv(args.out_dir / "all_filter_trials.csv", index=False)
    usable_rules = [json.loads(str(rule)) for rule in all_trials.loc[all_trials["rule_name"].ne("no_filter"), "rule"].head(50)]
    loo = _leave_one_day(
        frame,
        usable_rules,
        min_keep=max(2, min(int(args.min_keep), max(len(frame) - 1, 1))),
        min_positive_rate=float(args.min_positive_rate),
    )
    loo.to_csv(args.out_dir / "leave_one_day_filter_validation.csv", index=False)

    best = all_trials.iloc[0].to_dict() if not all_trials.empty else {}
    no_filter = all_trials.loc[all_trials["rule_name"].eq("no_filter")].head(1)
    no_filter_row = no_filter.iloc[0].to_dict() if not no_filter.empty else {}
    manifest = {
        "generated_by": "run_c3el_fallback_quality_filter",
        "fallback_actions": str(args.fallback_actions),
        "action_features": str(args.action_features),
        "rows": int(len(frame)),
        "feature_rows_matched": int(frame["feature_row_matched"].sum()) if "feature_row_matched" in frame.columns else 0,
        "features_evaluated": int(len(features)),
        "single_rules": int(len(single_rules)),
        "pair_rules": int(len(pair_rules)),
        "min_keep": int(args.min_keep),
        "min_positive_rate": float(args.min_positive_rate),
        "best_rule": best,
        "no_filter": no_filter_row,
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str))

    lines = [
        "# C3el fallback-quality filter ablation",
        "",
        "This no-replay ablation uses exact cloned-state labels for the selected fallback actions. It is a research diagnostic, not a production replay.",
        "",
        f"Rows: `{manifest['rows']}`",
        f"Feature rows matched: `{manifest['feature_rows_matched']}`",
        f"Deployable features evaluated: `{manifest['features_evaluated']}`",
        "",
        "## Baseline Fallback Set",
        "",
    ]
    if no_filter_row:
        lines.append(pd.DataFrame([no_filter_row])[
            [
                "keep_count",
                "positive_e50_count",
                "positive_e50_rate",
                "delta_full_J_sum",
                "delta_full_J_worst",
                "delta_immediate_J_sum",
                "direct_delta_net_pnl_sum",
            ]
        ].to_markdown(index=False, floatfmt=".4f"))
    lines.extend(["", "## Best Filters", ""])
    cols = [
        "rule_family",
        "rule",
        "keep_count",
        "positive_e50_rate",
        "delta_full_J_sum",
        "delta_full_J_worst",
        "rejected_negative_delta_full_J_sum",
        "rejected_positive_delta_full_J_sum",
        "objective",
    ]
    if all_trials.empty:
        lines.append("No trials.")
    else:
        lines.append(all_trials[cols].head(15).to_markdown(index=False, floatfmt=".4f"))
    lines.extend(["", "## Leave-One-Day Validation", ""])
    if loo.empty:
        lines.append("No leave-one-day rows.")
    else:
        loo_summary = {
            "heldout_days": int(len(loo)),
            "total_keep_count": int(pd.to_numeric(loo["keep_count"], errors="coerce").fillna(0).sum()),
            "total_delta_full_J": float(pd.to_numeric(loo["delta_full_J_sum"], errors="coerce").fillna(0.0).sum()),
            "positive_day_share": float(pd.to_numeric(loo["delta_full_J_sum"], errors="coerce").fillna(0.0).gt(0.0).mean()),
            "worst_day_delta_full_J": float(pd.to_numeric(loo["delta_full_J_sum"], errors="coerce").fillna(0.0).min()),
        }
        lines.append(pd.DataFrame([loo_summary]).to_markdown(index=False, floatfmt=".4f"))
        lines.append("")
        lines.append(loo[["heldout_day", "keep_count", "positive_e50_rate", "delta_full_J_sum", "delta_full_J_worst"]].to_markdown(index=False, floatfmt=".4f"))
    (args.out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print((args.out_dir / "summary.md").read_text())


if __name__ == "__main__":
    main()
