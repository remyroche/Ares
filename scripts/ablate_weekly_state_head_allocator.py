#!/usr/bin/env python3
"""Conservative weekly allocator over real head-subset replay variants.

The allocator defaults to the baseline variant. In each walk-forward week it may
switch to one defensive variant only when a prior-week/rolling state trigger is
active. This is a proxy over already completed full replays; it does not stitch
portfolio state inside a single replay, but it is useful for testing whether a
bounded state rule can recover defensive head-mix value.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


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
        "q10_week_net_pnl": float(np.quantile(values, 0.10)),
        "q15_week_net_pnl": float(np.quantile(values, 0.15)),
        "q20_week_net_pnl": float(np.quantile(values, 0.20)),
        "q25_week_net_pnl": float(np.quantile(values, 0.25)),
        "q35_week_net_pnl": float(np.quantile(values, 0.35)),
        "worst_week_net_pnl": float(np.min(values)),
        "positive_weeks": int(np.sum(values > 0)),
        "objective": _objective(values, q35_weight, q20_weight),
    }


def _load_variant_weekly(root: Path, labels: list[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for label in labels:
        path = root / label / "combo_replay_weekly_metrics.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        frame = frame[frame["period_type"].eq("week")].copy()
        frame["label"] = label
        frame["week_start"] = pd.to_datetime(frame["week"].str.split("/").str[0])
        for col in ("net_pnl", "gross_pnl", "trades", "hit_rate", "full_sl_rate", "timeout_rate"):
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        frames.append(frame)
    return pd.concat(frames, ignore_index=True).sort_values(["week_start", "label"]).reset_index(drop=True)


def _baseline_signals(weekly: pd.DataFrame, baseline_label: str) -> pd.DataFrame:
    base = weekly[weekly["label"].eq(baseline_label)].sort_values("week_start").copy()
    base["prev_net_pnl"] = base["net_pnl"].shift(1)
    base["prev_hit_rate"] = base["hit_rate"].shift(1)
    base["prev_full_sl_rate"] = base["full_sl_rate"].shift(1)
    base["prev_timeout_rate"] = base["timeout_rate"].shift(1)
    base["roll2_net_pnl"] = base["net_pnl"].shift(1).rolling(2, min_periods=2).mean()
    base["roll3_net_pnl"] = base["net_pnl"].shift(1).rolling(3, min_periods=2).mean()
    base["roll2_full_sl_rate"] = base["full_sl_rate"].shift(1).rolling(2, min_periods=2).mean()
    base["roll3_full_sl_rate"] = base["full_sl_rate"].shift(1).rolling(3, min_periods=2).mean()
    base["roll2_hit_rate"] = base["hit_rate"].shift(1).rolling(2, min_periods=2).mean()
    base["roll3_hit_rate"] = base["hit_rate"].shift(1).rolling(3, min_periods=2).mean()
    return base[
        [
            "week",
            "week_start",
            "prev_net_pnl",
            "prev_hit_rate",
            "prev_full_sl_rate",
            "prev_timeout_rate",
            "roll2_net_pnl",
            "roll3_net_pnl",
            "roll2_full_sl_rate",
            "roll3_full_sl_rate",
            "roll2_hit_rate",
            "roll3_hit_rate",
        ]
    ]


def _diagnostic_columns(columns: list[str]) -> dict[str, list[str]]:
    groups = {
        "uncertainty": [
            c
            for c in columns
            if "uncert" in c.lower()
            or "entropy" in c.lower()
            or "abs_distance_from_half" in c.lower()
        ],
        "ood": [c for c in columns if "ood" in c.lower()],
        "drift": [
            c
            for c in columns
            if "drift" in c.lower()
            or "score_abs_diff" in c.lower()
            or "score_abs_minus_prev24_mean" in c.lower()
            or "score_prev24_std" in c.lower()
            or "score_shift_abs_z" in c.lower()
        ],
        "recent_hr_surprise": [
            c
            for c in columns
            if "hr_surprise" in c.lower()
            or "weighted_hr_surprise" in c.lower()
            or "diagnostic_recent_hr_surprise" in c.lower()
            or "loss_rate" in c.lower()
        ],
    }
    excluded = {"mtm_path_gross_returns"}
    return {
        family: sorted(c for c in set(cols) if c not in excluded)
        for family, cols in groups.items()
        if cols
    }


def _candidate_diagnostic_signals(root: Path, baseline_label: str) -> pd.DataFrame:
    path = root / baseline_label / "combo_candidates.parquet"
    if not path.exists():
        return pd.DataFrame()
    sample = pd.read_parquet(path)
    if "timestamp" not in sample.columns:
        return pd.DataFrame()
    groups = _diagnostic_columns(list(sample.columns))
    if not groups:
        return pd.DataFrame()
    use_cols = ["timestamp", *sorted({c for cols in groups.values() for c in cols})]
    frame = sample[use_cols].copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame = frame[frame["timestamp"].notna()].copy()
    frame["week_start"] = frame["timestamp"].dt.to_period("W-SUN").dt.start_time
    frame["week"] = (
        frame["week_start"].dt.strftime("%Y-%m-%d")
        + "/"
        + (frame["week_start"] + pd.Timedelta(days=6)).dt.strftime("%Y-%m-%d")
    )
    weekly_parts: list[pd.DataFrame] = [frame[["week", "week_start"]].drop_duplicates()]
    for family, cols in groups.items():
        numeric_cols = []
        for col in cols:
            values = pd.to_numeric(frame[col], errors="coerce")
            if values.notna().sum() == 0 or float(values.std(skipna=True) or 0.0) == 0.0:
                continue
            frame[col] = values
            numeric_cols.append(col)
        if not numeric_cols:
            continue
        agg = frame.groupby(["week", "week_start"], observed=True)[numeric_cols].agg(["mean", "median", lambda x: x.quantile(0.75)])
        agg.columns = [
            f"diag_{family}_{col}_{'p75' if stat == '<lambda_0>' else stat}"
            for col, stat in agg.columns.to_flat_index()
        ]
        weekly_parts.append(agg.reset_index())
    if len(weekly_parts) == 1:
        return pd.DataFrame()
    out = weekly_parts[0]
    for part in weekly_parts[1:]:
        out = out.merge(part, on=["week", "week_start"], how="left")
    out = out.sort_values("week_start").reset_index(drop=True)
    value_cols = [c for c in out.columns if c not in {"week", "week_start"}]
    shifted: dict[str, pd.Series] = {}
    for col in value_cols:
        lagged = out[col].shift(1)
        shifted[f"prev_{col}"] = lagged
        shifted[f"roll2_{col}"] = lagged.rolling(2, min_periods=2).mean()
        shifted[f"roll3_{col}"] = lagged.rolling(3, min_periods=2).mean()
    shifted_frame = pd.DataFrame(shifted)
    return pd.concat([out[["week", "week_start"]], shifted_frame], axis=1)


def _rule_trigger(signals: pd.DataFrame, feature: str, direction: str, threshold: float) -> pd.Series:
    values = pd.to_numeric(signals[feature], errors="coerce")
    if direction == "low":
        return values.le(threshold).fillna(False)
    return values.ge(threshold).fillna(False)


def _feature_directions(feature: str) -> list[str]:
    name = feature.lower()
    if "net_pnl" in name or "hit_rate" in name:
        return ["low"]
    if "diagnostic_recent_hr_surprise_risk" in name:
        return ["high"]
    if "hr_surprise" in name or "weighted_hr_surprise" in name:
        return ["low"]
    if any(key in name for key in ("full_sl", "timeout", "loss_rate", "uncert", "entropy", "ood", "drift", "abs_diff", "abs_minus", "prev24_std", "shift_abs_z", "risk")):
        return ["high"]
    return ["low", "high"]


def _candidate_rules(
    train_signals: pd.DataFrame,
    actions: list[str],
    quantiles: list[float],
    max_trigger_share: float,
) -> list[tuple[str, str, str, float]]:
    fixed_features = [
        "prev_net_pnl",
        "prev_hit_rate",
        "prev_full_sl_rate",
        "roll2_net_pnl",
        "roll3_net_pnl",
        "roll2_full_sl_rate",
        "roll3_full_sl_rate",
        "roll2_hit_rate",
        "roll3_hit_rate",
    ]
    features = [
        c
        for c in train_signals.columns
        if c in fixed_features or c.startswith(("prev_diag_", "roll2_diag_", "roll3_diag_"))
    ]
    rules: list[tuple[str, str, str, float]] = []
    for feature in features:
        values = pd.to_numeric(train_signals[feature], errors="coerce").dropna().to_numpy(dtype=np.float64)
        if values.size < 4 or np.nanstd(values) <= 0:
            continue
        directions = _feature_directions(feature)
        for direction in directions:
            qs = quantiles if direction == "high" else [1.0 - q for q in quantiles]
            for q in qs:
                threshold = float(np.nanquantile(values, q))
                trigger = _rule_trigger(train_signals, feature, direction, threshold)
                share = float(trigger.mean()) if len(trigger) else 0.0
                if share <= 0.0 or share > max_trigger_share:
                    continue
                for action in actions:
                    rules.append((action, feature, direction, threshold))
    return rules


def _weekly_values_for_rule(
    wide: pd.DataFrame,
    signals: pd.DataFrame,
    baseline_label: str,
    action: str,
    feature: str,
    direction: str,
    threshold: float,
) -> pd.DataFrame:
    out = signals[["week", "week_start", feature]].copy()
    out["trigger"] = _rule_trigger(signals, feature, direction, threshold)
    out["selected_label"] = np.where(out["trigger"], action, baseline_label)
    values = []
    for row in out.itertuples(index=False):
        values.append(float(wide.loc[row.week, row.selected_label]))
    out["net_pnl"] = values
    out["baseline_net_pnl"] = [float(wide.loc[w, baseline_label]) for w in out["week"]]
    out["action_net_pnl"] = [float(wide.loc[w, action]) for w in out["week"]]
    out["delta_net_pnl_vs_baseline"] = out["net_pnl"] - out["baseline_net_pnl"]
    out["action_label"] = action
    out["rule_feature"] = feature
    out["rule_direction"] = direction
    out["rule_threshold"] = threshold
    return out


def _walk_forward(
    weekly: pd.DataFrame,
    signals: pd.DataFrame,
    baseline_label: str,
    action_labels: list[str],
    min_train_weeks: int,
    quantiles: list[float],
    max_trigger_share: float,
    q35_weight: float,
    q20_weight: float,
    min_train_objective_delta: float,
    min_train_triggers: int,
    min_trigger_mean_delta: float,
    min_trigger_q20_delta: float,
) -> pd.DataFrame:
    wide = weekly.pivot(index="week", columns="label", values="net_pnl")
    signals = signals.sort_values("week_start").reset_index(drop=True)
    rows = []
    for pos, row in signals.iterrows():
        if pos < min_train_weeks:
            continue
        train = signals.iloc[:pos].copy()
        train_weeks = list(train["week"])
        baseline_score = _objective(wide.loc[train_weeks, baseline_label].to_numpy(), q35_weight, q20_weight)
        best_rule = {
            "action": baseline_label,
            "feature": "__none__",
            "direction": "low",
            "threshold": np.nan,
            "train_score": baseline_score,
            "train_objective_delta": 0.0,
            "train_trigger_count": 0,
            "train_trigger_mean_delta": 0.0,
            "train_trigger_q20_delta": 0.0,
        }
        for action, feature, direction, threshold in _candidate_rules(train, action_labels, quantiles, max_trigger_share):
            replay = _weekly_values_for_rule(wide, train, baseline_label, action, feature, direction, threshold)
            score = _objective(replay["net_pnl"].to_numpy(), q35_weight, q20_weight)
            objective_delta = float(score - baseline_score)
            triggered = replay[replay["trigger"].astype(bool)].copy()
            trigger_count = int(len(triggered))
            if trigger_count:
                trigger_deltas = triggered["delta_net_pnl_vs_baseline"].to_numpy(dtype=np.float64)
                trigger_mean_delta = float(np.nanmean(trigger_deltas))
                trigger_q20_delta = float(np.nanquantile(trigger_deltas, 0.20))
            else:
                trigger_mean_delta = 0.0
                trigger_q20_delta = 0.0
            if objective_delta < min_train_objective_delta:
                continue
            if trigger_count < min_train_triggers:
                continue
            if trigger_mean_delta < min_trigger_mean_delta:
                continue
            if trigger_q20_delta < min_trigger_q20_delta:
                continue
            if score > best_rule["train_score"]:
                best_rule = {
                    "action": action,
                    "feature": feature,
                    "direction": direction,
                    "threshold": threshold,
                    "train_score": score,
                    "train_objective_delta": objective_delta,
                    "train_trigger_count": trigger_count,
                    "train_trigger_mean_delta": trigger_mean_delta,
                    "train_trigger_q20_delta": trigger_q20_delta,
                }
        action = str(best_rule["action"])
        feature = str(best_rule["feature"])
        direction = str(best_rule["direction"])
        threshold = float(best_rule["threshold"])
        if feature == "__none__":
            selected_label = baseline_label
            trigger = False
            selected_net = float(wide.loc[row["week"], baseline_label])
        else:
            trigger = bool(_rule_trigger(pd.DataFrame([row]), feature, direction, threshold).iloc[0])
            selected_label = action if trigger else baseline_label
            selected_net = float(wide.loc[row["week"], selected_label])
        baseline_net = float(wide.loc[row["week"], baseline_label])
        rows.append(
            {
                "week": row["week"],
                "week_start": row["week_start"],
                "selected_label": selected_label,
                "action_label": action,
                "trigger": trigger,
                "rule_feature": feature,
                "rule_direction": direction,
                "rule_threshold": threshold,
                "train_objective": float(best_rule["train_score"]),
                "train_objective_delta": float(best_rule["train_objective_delta"]),
                "train_trigger_count": int(best_rule["train_trigger_count"]),
                "train_trigger_mean_delta": float(best_rule["train_trigger_mean_delta"]),
                "train_trigger_q20_delta": float(best_rule["train_trigger_q20_delta"]),
                "net_pnl": selected_net,
                "baseline_net_pnl": baseline_net,
                "delta_net_pnl_vs_baseline": selected_net - baseline_net,
            }
        )
    return pd.DataFrame(rows)


def _markdown_table(frame: pd.DataFrame, cols: list[str], max_rows: int | None = None) -> str:
    view = frame[cols].head(max_rows).copy() if max_rows else frame[cols].copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{x:,.3f}")
    return view.to_markdown(index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-root", type=Path, required=True)
    parser.add_argument("--baseline-label", default="all_heads")
    parser.add_argument("--action-label", action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-train-weeks", type=int, default=8)
    parser.add_argument("--quantiles", default="0.70,0.80,0.90")
    parser.add_argument("--max-trigger-share", type=float, default=0.25)
    parser.add_argument("--q35-weight", type=float, default=0.70)
    parser.add_argument("--q20-weight", type=float, default=0.30)
    parser.add_argument("--min-train-objective-delta", type=float, default=0.0)
    parser.add_argument("--min-train-triggers", type=int, default=1)
    parser.add_argument("--min-trigger-mean-delta", type=float, default=float("-inf"))
    parser.add_argument("--min-trigger-q20-delta", type=float, default=float("-inf"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    labels = [args.baseline_label, *args.action_label]
    if not args.action_label:
        raise ValueError("Provide at least one --action-label")
    weekly = _load_variant_weekly(args.replay_root, labels)
    signals = _baseline_signals(weekly, args.baseline_label)
    diagnostic_signals = _candidate_diagnostic_signals(args.replay_root, args.baseline_label)
    if not diagnostic_signals.empty:
        signals = signals.merge(diagnostic_signals, on=["week", "week_start"], how="left")
    quantiles = [float(x) for x in str(args.quantiles).split(",") if x]
    wf = _walk_forward(
        weekly,
        signals,
        args.baseline_label,
        list(args.action_label),
        args.min_train_weeks,
        quantiles,
        args.max_trigger_share,
        args.q35_weight,
        args.q20_weight,
        args.min_train_objective_delta,
        args.min_train_triggers,
        args.min_trigger_mean_delta,
        args.min_trigger_q20_delta,
    )
    baseline_eval = signals[signals["week"].isin(set(wf["week"]))].merge(
        weekly[weekly["label"].eq(args.baseline_label)][["week", "net_pnl"]],
        on="week",
        how="left",
    )
    summary = pd.DataFrame(
        [
            {"policy": "weekly_state_allocator", **_summary(wf["net_pnl"].to_numpy(), args.q35_weight, args.q20_weight)},
            {"policy": "baseline_same_eval_weeks", **_summary(baseline_eval["net_pnl"].to_numpy(), args.q35_weight, args.q20_weight)},
        ]
    )
    base = summary[summary["policy"].eq("baseline_same_eval_weeks")].iloc[0]
    for col in ("sum_net_pnl", "avg_week_net_pnl", "q15_week_net_pnl", "q20_week_net_pnl", "q25_week_net_pnl", "q35_week_net_pnl", "worst_week_net_pnl", "objective"):
        summary[f"delta_{col}_vs_baseline"] = summary[col] - base[col]

    wf.to_csv(args.output_dir / "weekly_state_allocator_walk_forward.csv", index=False)
    summary.to_csv(args.output_dir / "weekly_state_allocator_summary.csv", index=False)
    weekly.to_csv(args.output_dir / "weekly_state_allocator_variant_weekly.csv", index=False)
    if not diagnostic_signals.empty:
        diagnostic_signals.to_csv(args.output_dir / "weekly_state_allocator_diagnostic_signals.csv", index=False)
    signal_audit = pd.DataFrame(
        [
            {
                "feature_family": "drift",
                "feature_count": int(sum("diag_drift_" in c for c in signals.columns)),
            },
            {
                "feature_family": "recent_hr_surprise",
                "feature_count": int(sum("diag_recent_hr_surprise_" in c for c in signals.columns)),
            },
            {
                "feature_family": "ood",
                "feature_count": int(sum("diag_ood_" in c for c in signals.columns)),
            },
            {
                "feature_family": "uncertainty",
                "feature_count": int(sum("diag_uncertainty_" in c for c in signals.columns)),
            },
        ]
    )
    signal_audit.to_csv(args.output_dir / "weekly_state_allocator_signal_family_audit.csv", index=False)
    triggered = wf[wf["trigger"].astype(bool)].copy()
    june = wf[wf["week"].astype(str).str.startswith("2026-06")].copy()
    lines = [
        "# Weekly State Head Allocator",
        "",
        "Proxy over real head-subset replays. Default is baseline/all-heads; a prior-week state rule may switch the whole week to one defensive variant.",
        "",
        "## Configuration",
        "",
        _markdown_table(
            pd.DataFrame(
                [
                    {
                        "min_train_weeks": args.min_train_weeks,
                        "quantiles": args.quantiles,
                        "max_trigger_share": args.max_trigger_share,
                        "min_train_objective_delta": args.min_train_objective_delta,
                        "min_train_triggers": args.min_train_triggers,
                        "min_trigger_mean_delta": args.min_trigger_mean_delta,
                        "min_trigger_q20_delta": args.min_trigger_q20_delta,
                    }
                ]
            ),
            [
                "min_train_weeks",
                "quantiles",
                "max_trigger_share",
                "min_train_objective_delta",
                "min_train_triggers",
                "min_trigger_mean_delta",
                "min_trigger_q20_delta",
            ],
        ),
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
        "## Diagnostic Signal Families",
        "",
        _markdown_table(signal_audit, ["feature_family", "feature_count"]),
        "",
        "## Triggered Weeks",
        "",
        _markdown_table(
            triggered,
            [
                "week",
                "selected_label",
                "action_label",
                "rule_feature",
                "rule_direction",
                "rule_threshold",
                "train_objective_delta",
                "train_trigger_count",
                "train_trigger_mean_delta",
                "train_trigger_q20_delta",
                "net_pnl",
                "baseline_net_pnl",
                "delta_net_pnl_vs_baseline",
            ],
        )
        if not triggered.empty
        else "No triggered weeks.",
        "",
        "## June Weeks",
        "",
        _markdown_table(
            june,
            [
                "week",
                "selected_label",
                "rule_feature",
                "net_pnl",
                "baseline_net_pnl",
                "delta_net_pnl_vs_baseline",
            ],
        ),
        "",
        "## Readout Guidance",
        "",
        "- Positive results here indicate a simple state switch may be worth a true sequential replay.",
        "- Negative results mean prior-week all-head state is insufficient to time defensive head mixes.",
    ]
    (args.output_dir / "weekly_state_allocator_report.md").write_text("\n".join(lines) + "\n")
    print(args.output_dir / "weekly_state_allocator_report.md")


if __name__ == "__main__":
    main()
