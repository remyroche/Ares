#!/usr/bin/env python3
"""Walk-forward timestamp/head allocation rules from candidate diagnostics.

This is a proxy ablation. It uses accepted replay decisions to estimate the PnL
contribution of each timestamp/head, then tests whether current-bar candidate
diagnostics can disable one head at selected timestamps. Rules are selected only
from prior weeks and evaluated on the next week.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


BASE_FEATURES = [
    "diagnostic_uncertainty_risk",
    "diagnostic_ood_risk",
    "diagnostic_recent_hr_surprise_risk",
    "diagnostic_composite_risk",
    "generated_score_uncertainty_p1mp",
    "generated_score_entropy",
    "generated_score_abs_distance_from_half",
    "generated_score_abs_diff_1",
    "generated_score_abs_diff_4",
    "generated_score_abs_diff_24",
    "generated_score_abs_minus_prev24_mean",
    "generated_score_prev24_std",
    "generated_strategy_score_shift_abs_z",
    "generated_strategy_score_ood_abs_z",
    "generated_strategy_barrier_ood_abs_z",
    "generated_strategy_friction_ood_abs_z",
    "generated_hr_surprise_24",
    "generated_hr_surprise_96",
    "generated_weighted_hr_surprise_24",
    "generated_weighted_hr_surprise_96",
    "generated_loss_rate_24",
    "generated_loss_rate_96",
    "generated_matured_count_24",
    "generated_matured_count_96",
    "auction_rank_score",
    "policy_rank_pct",
    "reliability_blend_score",
    "calibrated_score",
    "simple_policy_calibrated_good_trade_prob",
    "portfolio_size_multiplier",
]


def _head_from_strategy_id(strategy_id: str) -> str:
    value = str(strategy_id)
    if value.startswith("long_bars"):
        return "long_bars"
    if value.startswith("long_dist"):
        return "long_dist"
    if value.startswith("short_asset"):
        return "short_asset"
    if value.startswith("short_bollinger"):
        return "short_bollinger"
    return value.split("_")[0]


def _week(timestamp: pd.Series) -> pd.Series:
    return pd.to_datetime(timestamp, utc=True).dt.to_period("W-SUN").astype(str)


def _objective(values: np.ndarray, q35_weight: float, q20_weight: float) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("-inf")
    return float(np.mean(values) + q35_weight * np.quantile(values, 0.35) + q20_weight * np.quantile(values, 0.20))


def _summary(values: np.ndarray, q35_weight: float, q20_weight: float) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return {
        "weeks": int(values.size),
        "sum_net_pnl": float(np.sum(values)),
        "avg_week_net_pnl": float(np.mean(values)),
        "q15_week_net_pnl": float(np.quantile(values, 0.15)),
        "q20_week_net_pnl": float(np.quantile(values, 0.20)),
        "q25_week_net_pnl": float(np.quantile(values, 0.25)),
        "q35_week_net_pnl": float(np.quantile(values, 0.35)),
        "worst_week_net_pnl": float(np.min(values)),
        "positive_weeks": int(np.sum(values > 0)),
        "objective": _objective(values, q35_weight, q20_weight),
    }


def _load_decision_pnl(decisions_path: Path, source_name: str) -> pd.DataFrame:
    cols = [
        "timestamp",
        "strategy_id",
        "accepted",
        "position_size",
        "position_net_return",
        "position_gross_return",
    ]
    decisions = pd.read_parquet(decisions_path, columns=cols)
    decisions = decisions[decisions["accepted"].astype(bool)].copy()
    decisions["timestamp"] = pd.to_datetime(decisions["timestamp"], utc=True)
    decisions["week"] = _week(decisions["timestamp"])
    decisions["head"] = decisions["strategy_id"].map(_head_from_strategy_id)
    decisions["net_pnl"] = pd.to_numeric(decisions["position_size"], errors="coerce") * pd.to_numeric(decisions["position_net_return"], errors="coerce")
    decisions["gross_pnl"] = pd.to_numeric(decisions["position_size"], errors="coerce") * pd.to_numeric(decisions["position_gross_return"], errors="coerce")
    out = (
        decisions.groupby(["week", "timestamp", "head"], sort=True)
        .agg(net_pnl=("net_pnl", "sum"), gross_pnl=("gross_pnl", "sum"), trades=("strategy_id", "size"))
        .reset_index()
    )
    out["decision_source"] = source_name
    return out


def _stitch_decisions(default_decisions: Path, noop_decisions: Path | None, selections: Path | None) -> pd.DataFrame:
    default = _load_decision_pnl(default_decisions, "default")
    if noop_decisions is None or selections is None:
        return default
    noop = _load_decision_pnl(noop_decisions, "noop")
    sel = pd.read_csv(selections)
    noop_weeks = set(sel[(sel["triggered"].astype(bool)) & (sel["action_label"].astype(str) == "__noop__")]["eval_week"].astype(str))
    default = default[~default["week"].isin(noop_weeks)]
    noop = noop[noop["week"].isin(noop_weeks)]
    return pd.concat([default, noop], ignore_index=True).sort_values(["timestamp", "head"]).reset_index(drop=True)


def _load_candidate_features(candidates_path: Path) -> pd.DataFrame:
    available = pd.read_parquet(candidates_path, columns=None).columns
    cols = ["timestamp", "strategy_id", *[c for c in BASE_FEATURES if c in available]]
    candidates = pd.read_parquet(candidates_path, columns=cols)
    candidates["timestamp"] = pd.to_datetime(candidates["timestamp"], utc=True)
    candidates["week"] = _week(candidates["timestamp"])
    candidates["head"] = candidates["strategy_id"].map(_head_from_strategy_id)
    numeric = [c for c in cols if c not in ("timestamp", "strategy_id")]
    for col in numeric:
        candidates[col] = pd.to_numeric(candidates[col], errors="coerce")
    agg: dict[str, tuple[str, object]] = {"candidate_count": ("strategy_id", "size")}
    for col in numeric:
        agg[f"{col}_mean"] = (col, "mean")
        agg[f"{col}_p75"] = (col, lambda x: float(np.nanquantile(x, 0.75)) if np.isfinite(x).any() else np.nan)
        agg[f"{col}_p90"] = (col, lambda x: float(np.nanquantile(x, 0.90)) if np.isfinite(x).any() else np.nan)
        agg[f"{col}_max"] = (col, "max")
    if "auction_rank_score" in numeric:
        candidates["auction_rank_ge_070"] = candidates["auction_rank_score"] >= 0.70
        candidates["auction_rank_ge_080"] = candidates["auction_rank_score"] >= 0.80
        candidates["auction_rank_ge_090"] = candidates["auction_rank_score"] >= 0.90
        agg["auction_rank_ge_070_count"] = ("auction_rank_ge_070", "sum")
        agg["auction_rank_ge_080_count"] = ("auction_rank_ge_080", "sum")
        agg["auction_rank_ge_090_count"] = ("auction_rank_ge_090", "sum")
    return candidates.groupby(["week", "timestamp", "head"], sort=True).agg(**agg).reset_index()


def _build_panel(candidates_path: Path, default_decisions: Path, noop_decisions: Path | None, selections: Path | None) -> pd.DataFrame:
    features = _load_candidate_features(candidates_path)
    pnl = _stitch_decisions(default_decisions, noop_decisions, selections)
    panel = features.merge(pnl, on=["week", "timestamp", "head"], how="left")
    for col in ("net_pnl", "gross_pnl", "trades"):
        panel[col] = panel[col].fillna(0.0)
    panel["decision_source"] = panel["decision_source"].fillna("none")
    return panel.sort_values(["timestamp", "head"]).reset_index(drop=True)


def _weekly_from_panel(panel: pd.DataFrame, pnl_col: str = "net_pnl", trades_col: str = "trades") -> pd.DataFrame:
    return (
        panel.groupby("week", sort=True)
        .agg(net_pnl=(pnl_col, "sum"), gross_pnl=("gross_pnl", "sum"), trades=(trades_col, "sum"))
        .reset_index()
    )


def _apply_rule(panel: pd.DataFrame, head: str, feature: str, direction: str, threshold: float) -> pd.DataFrame:
    out = panel.copy()
    mask = out["head"].eq(head) & out[feature].notna()
    if direction == "high":
        mask &= out[feature] >= threshold
    else:
        mask &= out[feature] <= threshold
    out["disabled"] = mask
    out["rule_net_pnl"] = np.where(mask, 0.0, out["net_pnl"])
    out["rule_gross_pnl"] = np.where(mask, 0.0, out["gross_pnl"])
    out["rule_trades"] = np.where(mask, 0.0, out["trades"])
    return out


def _fit_rule(
    panel: pd.DataFrame,
    train_weeks: list[str],
    heads: list[str],
    features: list[str],
    quantiles: list[float],
    q35_weight: float,
    q20_weight: float,
    max_disabled_trade_share: float,
) -> tuple[str, str, str, float, float]:
    train = panel[panel["week"].isin(train_weeks)].copy()
    baseline_weekly = _weekly_from_panel(train)
    baseline = _objective(baseline_weekly["net_pnl"].to_numpy(), q35_weight, q20_weight)
    baseline_by_week = baseline_weekly.set_index("week")["net_pnl"]
    best = ("__none__", "__none__", "high", np.nan, baseline)
    total_trades = train["trades"].sum()
    for head in heads:
        head_train = train[train["head"].eq(head)]
        if head_train["trades"].sum() <= 0:
            continue
        for feature in features:
            values = head_train[feature].dropna().to_numpy(dtype=np.float64)
            if values.size < 8 or np.nanstd(values) <= 0:
                continue
            for direction in ("high", "low"):
                qs = quantiles if direction == "high" else [1.0 - q for q in quantiles]
                for q in qs:
                    threshold = float(np.nanquantile(values, q))
                    mask = train["head"].eq(head) & train[feature].notna()
                    if direction == "high":
                        mask &= train[feature] >= threshold
                    else:
                        mask &= train[feature] <= threshold
                    disabled = train.loc[mask, ["week", "net_pnl", "trades"]]
                    disabled_trades = float(disabled["trades"].sum())
                    if total_trades > 0 and disabled_trades / total_trades > max_disabled_trade_share:
                        continue
                    disabled_by_week = disabled.groupby("week")["net_pnl"].sum()
                    rule_weekly = (baseline_by_week - disabled_by_week).fillna(baseline_by_week)
                    score = _objective(rule_weekly.to_numpy(), q35_weight, q20_weight)
                    if score > best[4]:
                        best = (head, feature, direction, threshold, score)
    return best


def _walk_forward(
    panel: pd.DataFrame,
    min_train_weeks: int,
    quantiles: list[float],
    q35_weight: float,
    q20_weight: float,
    max_disabled_trade_share: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    weeks = list(panel[["week", "timestamp"]].groupby("week", sort=True)["timestamp"].min().sort_values().index)
    heads = sorted(panel["head"].unique())
    feature_cols_all = [
        c
        for c in panel.columns
        if c
        not in {
            "week",
            "timestamp",
            "head",
            "net_pnl",
            "gross_pnl",
            "trades",
            "decision_source",
        }
        and pd.api.types.is_numeric_dtype(panel[c])
    ]
    preferred_features = {
        "diagnostic_uncertainty_risk_p75",
        "diagnostic_uncertainty_risk_p90",
        "diagnostic_ood_risk_p75",
        "diagnostic_ood_risk_p90",
        "diagnostic_recent_hr_surprise_risk_p75",
        "diagnostic_recent_hr_surprise_risk_p90",
        "diagnostic_composite_risk_p75",
        "diagnostic_composite_risk_p90",
        "generated_weighted_hr_surprise_24_mean",
        "generated_weighted_hr_surprise_24_p75",
        "generated_weighted_hr_surprise_96_mean",
        "generated_weighted_hr_surprise_96_p75",
        "generated_score_uncertainty_p1mp_p75",
        "generated_score_entropy_p75",
        "generated_score_abs_diff_24_p75",
        "generated_strategy_score_ood_abs_z_p75",
        "generated_strategy_score_shift_abs_z_p75",
        "auction_rank_score_p75",
        "auction_rank_score_max",
        "policy_rank_pct_p75",
        "policy_rank_pct_max",
        "simple_policy_calibrated_good_trade_prob_p75",
        "candidate_count",
        "auction_rank_ge_070_count",
        "auction_rank_ge_080_count",
        "auction_rank_ge_090_count",
    }
    feature_cols = [c for c in feature_cols_all if c in preferred_features]
    rows = []
    eval_panels = []
    for pos, week in enumerate(weeks):
        if pos < min_train_weeks:
            continue
        train_weeks = weeks[:pos]
        head, feature, direction, threshold, train_score = _fit_rule(
            panel,
            train_weeks,
            heads,
            feature_cols,
            quantiles,
            q35_weight,
            q20_weight,
            max_disabled_trade_share,
        )
        current = panel[panel["week"].eq(week)].copy()
        if head == "__none__":
            replay = current.copy()
            replay["disabled"] = False
            replay["rule_net_pnl"] = replay["net_pnl"]
            replay["rule_gross_pnl"] = replay["gross_pnl"]
            replay["rule_trades"] = replay["trades"]
        else:
            replay = _apply_rule(current, head, feature, direction, threshold)
        replay["selected_head"] = head
        replay["selected_feature"] = feature
        replay["selected_direction"] = direction
        replay["selected_threshold"] = threshold
        eval_panels.append(replay)
        baseline_week = current["net_pnl"].sum()
        rule_week = replay["rule_net_pnl"].sum()
        rows.append(
            {
                "week": week,
                "selected_head": head,
                "selected_feature": feature,
                "selected_direction": direction,
                "selected_threshold": threshold,
                "train_objective": train_score,
                "baseline_net_pnl": baseline_week,
                "rule_net_pnl": rule_week,
                "delta_net_pnl_vs_baseline": rule_week - baseline_week,
                "baseline_trades": current["trades"].sum(),
                "rule_trades": replay["rule_trades"].sum(),
                "disabled_trades": replay.loc[replay["disabled"], "trades"].sum(),
                "disabled_pnl": replay.loc[replay["disabled"], "net_pnl"].sum(),
                "disabled_timestamp_heads": int(replay["disabled"].sum()),
            }
        )
    return pd.DataFrame(rows), pd.concat(eval_panels, ignore_index=True) if eval_panels else pd.DataFrame()


def _markdown_table(frame: pd.DataFrame, cols: list[str], max_rows: int | None = None) -> str:
    view = frame[cols].head(max_rows).copy() if max_rows else frame[cols].copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{x:,.3f}")
    return view.to_markdown(index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--default-decisions", type=Path, required=True)
    parser.add_argument("--noop-decisions", type=Path)
    parser.add_argument("--selections", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-train-weeks", type=int, default=8)
    parser.add_argument("--quantiles", default="0.70,0.75,0.80,0.85,0.90,0.95")
    parser.add_argument("--q35-weight", type=float, default=0.70)
    parser.add_argument("--q20-weight", type=float, default=0.30)
    parser.add_argument("--max-disabled-trade-share", type=float, default=0.25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    panel = _build_panel(args.candidates, args.default_decisions, args.noop_decisions, args.selections)
    quantiles = [float(x) for x in str(args.quantiles).split(",") if x]
    selections, eval_panel = _walk_forward(
        panel,
        args.min_train_weeks,
        quantiles,
        args.q35_weight,
        args.q20_weight,
        args.max_disabled_trade_share,
    )
    baseline_summary = _summary(selections["baseline_net_pnl"].to_numpy(), args.q35_weight, args.q20_weight)
    rule_summary = _summary(selections["rule_net_pnl"].to_numpy(), args.q35_weight, args.q20_weight)
    summary = pd.DataFrame(
        [
            {"policy": "timestamp_head_rule", **rule_summary},
            {"policy": "baseline_same_eval_weeks", **baseline_summary},
        ]
    )
    for col in ("sum_net_pnl", "avg_week_net_pnl", "q15_week_net_pnl", "q20_week_net_pnl", "q25_week_net_pnl", "q35_week_net_pnl", "worst_week_net_pnl", "objective"):
        summary[f"delta_{col}_vs_baseline"] = summary[col] - baseline_summary[col]

    panel.to_parquet(args.output_dir / "timestamp_head_panel.parquet", index=False)
    selections.to_csv(args.output_dir / "timestamp_head_rule_walk_forward.csv", index=False)
    eval_panel.to_parquet(args.output_dir / "timestamp_head_rule_eval_panel.parquet", index=False)
    summary.to_csv(args.output_dir / "timestamp_head_rule_summary.csv", index=False)

    triggered = selections[selections["disabled_trades"] > 0].copy()
    june = selections[selections["week"].astype(str).str.startswith("2026-06")].copy()
    lines = [
        "# Timestamp Head Allocation Rule Ablation",
        "",
        "Proxy only: disables head/timestamp contributions using current-bar candidate diagnostics. Rules are selected walk-forward from prior weeks.",
        "",
        "## Summary",
        "",
        _markdown_table(
            summary,
            [
                "policy",
                "weeks",
                "sum_net_pnl",
                "avg_week_net_pnl",
                "q15_week_net_pnl",
                "q20_week_net_pnl",
                "q35_week_net_pnl",
                "worst_week_net_pnl",
                "positive_weeks",
                "objective",
                "delta_sum_net_pnl_vs_baseline",
                "delta_objective_vs_baseline",
            ],
        ),
        "",
        "## Triggered Eval Weeks",
        "",
        _markdown_table(
            triggered,
            [
                "week",
                "selected_head",
                "selected_feature",
                "selected_direction",
                "selected_threshold",
                "baseline_net_pnl",
                "rule_net_pnl",
                "delta_net_pnl_vs_baseline",
                "disabled_trades",
                "disabled_pnl",
            ],
        )
        if not triggered.empty
        else "No triggered eval weeks.",
        "",
        "## June Eval Weeks",
        "",
        _markdown_table(
            june,
            [
                "week",
                "selected_head",
                "selected_feature",
                "selected_direction",
                "baseline_net_pnl",
                "rule_net_pnl",
                "delta_net_pnl_vs_baseline",
                "disabled_trades",
                "disabled_pnl",
            ],
        ),
        "",
        "## Readout Guidance",
        "",
        "- Positive results here mean immediate candidate diagnostics can support head allocation and deserve a real replay.",
        "- Negative results mean the useful June head mix is not captured by simple one-rule diagnostic thresholds.",
    ]
    (args.output_dir / "timestamp_head_rule_report.md").write_text("\n".join(lines) + "\n")
    print(args.output_dir / "timestamp_head_rule_report.md")


if __name__ == "__main__":
    main()
