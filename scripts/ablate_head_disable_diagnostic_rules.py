#!/usr/bin/env python3
"""Walk-forward diagnostic rules for head-level disable decisions.

This is a proxy ablation: it uses materialized per-head weekly replay metrics and
candidate diagnostic features. Diagnostic features are aggregated by head/week
and lagged one week before a rule may disable a head, so the rule does not use
same-week outcomes or future-week diagnostics.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "diagnostic_uncertainty_risk",
    "diagnostic_ood_risk",
    "diagnostic_recent_hr_surprise_risk",
    "diagnostic_composite_risk",
    "generated_score_abs_diff_1",
    "generated_score_abs_diff_4",
    "generated_score_abs_diff_24",
    "generated_score_abs_minus_prev24_mean",
    "generated_score_prev24_std",
    "generated_strategy_score_shift_abs_z",
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


def _load_weekly(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = pd.read_csv(path)
    frame["head"] = frame["head"].fillna("__global__")
    frame["week_start"] = pd.to_datetime(frame["week"].str.split("/").str[0])
    global_weekly = frame[frame["head"] == "__global__"].copy()
    heads = frame[frame["head"] != "__global__"].copy()
    for col in ("net_pnl", "gross_pnl", "trades", "hit_rate", "full_sl_rate", "timeout_rate"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
        global_weekly[col] = pd.to_numeric(global_weekly[col], errors="coerce")
        heads[col] = pd.to_numeric(heads[col], errors="coerce")
    return global_weekly.sort_values("week_start").reset_index(drop=True), heads.sort_values(["week_start", "head"]).reset_index(drop=True)


def _load_head_week_features(candidates_path: Path) -> pd.DataFrame:
    columns = ["timestamp", "strategy_id", *FEATURE_COLUMNS]
    candidates = pd.read_parquet(candidates_path, columns=columns)
    candidates["head"] = candidates["strategy_id"].map(_head_from_strategy_id)
    candidates["timestamp"] = pd.to_datetime(candidates["timestamp"], utc=True)
    candidates["week_start"] = candidates["timestamp"].dt.to_period("W-MON").dt.start_time.dt.tz_localize(None)
    # Match pandas week labels used by replay metrics: Monday/Sunday periods.
    candidates["week"] = candidates["timestamp"].dt.to_period("W-SUN").astype(str)
    agg = {}
    for col in FEATURE_COLUMNS:
        agg[f"{col}_mean"] = (col, "mean")
        agg[f"{col}_p75"] = (col, lambda x: float(np.nanquantile(x, 0.75)) if np.isfinite(x).any() else np.nan)
        agg[f"{col}_p90"] = (col, lambda x: float(np.nanquantile(x, 0.90)) if np.isfinite(x).any() else np.nan)
    agg["candidate_count"] = ("strategy_id", "size")
    out = candidates.groupby(["week", "head"], sort=True).agg(**agg).reset_index()
    return out


def _lag_features(features: pd.DataFrame) -> pd.DataFrame:
    features = features.copy()
    feature_cols = [c for c in features.columns if c not in ("week", "head")]
    features = features.sort_values(["head", "week"])
    for col in feature_cols:
        features[f"lag1_{col}"] = features.groupby("head")[col].shift(1)
    return features[["week", "head", *[f"lag1_{c}" for c in feature_cols]]]


def _apply_rule(
    global_weekly: pd.DataFrame,
    head_weekly: pd.DataFrame,
    lagged: pd.DataFrame,
    feature: str,
    threshold: float,
) -> pd.DataFrame:
    data = head_weekly.merge(lagged[["week", "head", feature]], on=["week", "head"], how="left")
    data["disabled"] = data[feature] >= threshold
    disabled = data[data["disabled"]].groupby("week", as_index=False).agg(
        disabled_net_pnl=("net_pnl", "sum"),
        disabled_gross_pnl=("gross_pnl", "sum"),
        disabled_trades=("trades", "sum"),
        disabled_heads=("head", lambda x: ",".join(sorted(set(map(str, x))))),
    )
    out = global_weekly.merge(disabled, on="week", how="left")
    for col in ("disabled_net_pnl", "disabled_gross_pnl", "disabled_trades"):
        out[col] = out[col].fillna(0.0)
    out["disabled_heads"] = out["disabled_heads"].fillna("")
    out["rule_net_pnl"] = out["net_pnl"] - out["disabled_net_pnl"]
    out["rule_gross_pnl"] = out["gross_pnl"] - out["disabled_gross_pnl"]
    out["rule_trades"] = out["trades"] - out["disabled_trades"]
    out["rule_feature"] = feature
    out["rule_threshold"] = threshold
    return out


def _fit_rule(
    global_weekly: pd.DataFrame,
    head_weekly: pd.DataFrame,
    lagged: pd.DataFrame,
    train_weeks: list[str],
    features: list[str],
    quantiles: list[float],
    q35_weight: float,
    q20_weight: float,
    max_trigger_share: float,
) -> tuple[str, float, float]:
    best_feature = "__none__"
    best_threshold = float("inf")
    best_score = _objective(global_weekly[global_weekly["week"].isin(train_weeks)]["net_pnl"].to_numpy(), q35_weight, q20_weight)
    for feature in features:
        values = lagged[lagged["week"].isin(train_weeks)][feature].dropna().to_numpy(dtype=np.float64)
        if values.size < 4 or np.nanstd(values) <= 0:
            continue
        for q in quantiles:
            threshold = float(np.nanquantile(values, q))
            replay = _apply_rule(global_weekly, head_weekly, lagged, feature, threshold)
            train = replay[replay["week"].isin(train_weeks)].copy()
            trigger_share = float((train["disabled_heads"] != "").mean())
            if trigger_share > max_trigger_share:
                continue
            score = _objective(train["rule_net_pnl"].to_numpy(), q35_weight, q20_weight)
            if score > best_score:
                best_feature = feature
                best_threshold = threshold
                best_score = score
    return best_feature, best_threshold, best_score


def _walk_forward(
    global_weekly: pd.DataFrame,
    head_weekly: pd.DataFrame,
    lagged: pd.DataFrame,
    features: list[str],
    min_train_weeks: int,
    quantiles: list[float],
    q35_weight: float,
    q20_weight: float,
    max_trigger_share: float,
) -> pd.DataFrame:
    weeks = list(global_weekly["week"])
    rows = []
    for pos, week in enumerate(weeks):
        if pos < min_train_weeks:
            continue
        train_weeks = weeks[:pos]
        feature, threshold, train_score = _fit_rule(
            global_weekly,
            head_weekly,
            lagged,
            train_weeks,
            features,
            quantiles,
            q35_weight,
            q20_weight,
            max_trigger_share,
        )
        if feature == "__none__":
            replay = global_weekly[global_weekly["week"] == week].copy()
            replay["rule_net_pnl"] = replay["net_pnl"]
            replay["rule_gross_pnl"] = replay["gross_pnl"]
            replay["rule_trades"] = replay["trades"]
            replay["disabled_heads"] = ""
        else:
            replay = _apply_rule(global_weekly, head_weekly, lagged, feature, threshold)
            replay = replay[replay["week"] == week].copy()
        row = replay.iloc[0].to_dict()
        row.update(
            {
                "selected_feature": feature,
                "selected_threshold": threshold,
                "train_objective": train_score,
                "delta_net_pnl_vs_baseline": float(row["rule_net_pnl"] - row["net_pnl"]),
                "delta_trades_vs_baseline": float(row["rule_trades"] - row["trades"]),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _markdown_table(frame: pd.DataFrame, cols: list[str], max_rows: int | None = None) -> str:
    view = frame[cols].head(max_rows).copy() if max_rows else frame[cols].copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{x:,.3f}")
    return view.to_markdown(index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weekly-metrics", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-train-weeks", type=int, default=8)
    parser.add_argument("--quantiles", default="0.70,0.75,0.80,0.85,0.90,0.95")
    parser.add_argument("--q35-weight", type=float, default=0.70)
    parser.add_argument("--q20-weight", type=float, default=0.30)
    parser.add_argument("--max-trigger-share", type=float, default=0.35)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    global_weekly, head_weekly = _load_weekly(args.weekly_metrics)
    features = _load_head_week_features(args.candidates)
    lagged = _lag_features(features)
    feature_cols = [c for c in lagged.columns if c.startswith("lag1_") and c != "lag1_candidate_count"]
    quantiles = [float(x) for x in str(args.quantiles).split(",") if x]

    wf = _walk_forward(
        global_weekly,
        head_weekly,
        lagged,
        feature_cols,
        args.min_train_weeks,
        quantiles,
        args.q35_weight,
        args.q20_weight,
        args.max_trigger_share,
    )
    eval_weeks = set(wf["week"])
    baseline = global_weekly[global_weekly["week"].isin(eval_weeks)].copy()

    wf_summary = _summary(wf["rule_net_pnl"].to_numpy(), args.q35_weight, args.q20_weight)
    baseline_summary = _summary(baseline["net_pnl"].to_numpy(), args.q35_weight, args.q20_weight)
    summary = pd.DataFrame(
        [
            {"policy": "diagnostic_disable_walk_forward", **wf_summary},
            {"policy": "baseline_same_eval_weeks", **baseline_summary},
        ]
    )
    for col in ("sum_net_pnl", "avg_week_net_pnl", "q15_week_net_pnl", "q20_week_net_pnl", "q25_week_net_pnl", "q35_week_net_pnl", "worst_week_net_pnl", "objective"):
        summary[f"delta_{col}_vs_baseline"] = summary[col] - baseline_summary[col]

    wf.to_csv(args.output_dir / "diagnostic_head_disable_walk_forward.csv", index=False)
    features.to_csv(args.output_dir / "diagnostic_head_week_features.csv", index=False)
    lagged.to_csv(args.output_dir / "diagnostic_head_week_lagged_features.csv", index=False)
    summary.to_csv(args.output_dir / "diagnostic_head_disable_summary.csv", index=False)

    triggered = wf[wf["disabled_heads"].astype(str) != ""].copy()
    june = wf[wf["week"].astype(str).str.startswith("2026-06")].copy()
    lines = [
        "# Diagnostic Head-Disable Rule Ablation",
        "",
        "Proxy only: disables weekly head contributions from additive replay metrics. Diagnostic features are lagged one week.",
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
                "selected_feature",
                "selected_threshold",
                "disabled_heads",
                "net_pnl",
                "rule_net_pnl",
                "delta_net_pnl_vs_baseline",
                "rule_trades",
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
                "selected_feature",
                "disabled_heads",
                "net_pnl",
                "rule_net_pnl",
                "delta_net_pnl_vs_baseline",
                "full_sl_rate",
            ],
        ),
        "",
        "## Readout Guidance",
        "",
        "- Improvement here means diagnostics contain useful head-allocation information, but it still needs a real portfolio replay.",
        "- No improvement means one-week-lagged diagnostic aggregates are too blunt for the June problem.",
    ]
    (args.output_dir / "diagnostic_head_disable_report.md").write_text("\n".join(lines) + "\n")
    print(args.output_dir / "diagnostic_head_disable_report.md")


if __name__ == "__main__":
    main()
