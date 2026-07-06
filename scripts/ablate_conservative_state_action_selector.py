#!/usr/bin/env python3
"""Conservative state-conditioned selector over weekly replay variants.

This is a light-weight action-value proxy over existing weekly replay artifacts.
It never refits the portfolio replay. For each evaluation week, it uses only
prior weeks, finds similar diagnostic states, estimates action deltas versus the
baseline, and switches away from the baseline only when the lower-tail evidence
is positive enough.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "all_diagnostics": ("prev_diag_", "roll2_diag_", "roll3_diag_"),
    "recent_hr_surprise": ("prev_diag_recent_hr_surprise_", "roll2_diag_recent_hr_surprise_", "roll3_diag_recent_hr_surprise_"),
    "uncertainty": ("prev_diag_uncertainty_", "roll2_diag_uncertainty_", "roll3_diag_uncertainty_"),
    "recent_hr_uncertainty": (
        "prev_diag_recent_hr_surprise_",
        "roll2_diag_recent_hr_surprise_",
        "roll3_diag_recent_hr_surprise_",
        "prev_diag_uncertainty_",
        "roll2_diag_uncertainty_",
        "roll3_diag_uncertainty_",
    ),
    "no_drift_ood": (
        "prev_diag_recent_hr_surprise_",
        "roll2_diag_recent_hr_surprise_",
        "roll3_diag_recent_hr_surprise_",
        "prev_diag_uncertainty_",
        "roll2_diag_uncertainty_",
        "roll3_diag_uncertainty_",
    ),
    "baseline_perf": (
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
    ),
    "baseline_plus_recent_uncert": (
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
        "prev_diag_recent_hr_surprise_",
        "roll2_diag_recent_hr_surprise_",
        "roll3_diag_recent_hr_surprise_",
        "prev_diag_uncertainty_",
        "roll2_diag_uncertainty_",
        "roll3_diag_uncertainty_",
    ),
}


@dataclass(frozen=True)
class Config:
    feature_set: str
    k_neighbors: int
    min_neighbor_count: int
    min_mean_delta: float
    min_q20_delta: float
    min_positive_share: float
    score_q20_weight: float


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
        "q20_week_net_pnl": float(np.quantile(values, 0.20)),
        "q35_week_net_pnl": float(np.quantile(values, 0.35)),
        "worst_week_net_pnl": float(np.min(values)),
        "positive_weeks": int(np.sum(values > 0)),
        "objective": _objective(values, q35_weight, q20_weight),
    }


def _load_inputs(weekly_path: Path, signals_path: Path, baseline_label: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    weekly = pd.read_csv(weekly_path)
    signals = pd.read_csv(signals_path)
    weekly["week_start"] = pd.to_datetime(weekly["week_start"], errors="coerce")
    signals["week_start"] = pd.to_datetime(signals["week_start"], errors="coerce")
    weekly = weekly[weekly["week_start"].notna()].copy()
    signals = signals[signals["week_start"].notna()].copy()
    wide = weekly.pivot(index="week", columns="label", values="net_pnl")
    if baseline_label not in wide.columns:
        raise ValueError(f"Missing baseline label {baseline_label!r}")
    common_weeks = [w for w in signals["week"].tolist() if w in wide.index]
    signals = signals[signals["week"].isin(common_weeks)].sort_values("week_start").reset_index(drop=True)
    base = weekly[weekly["label"].eq(baseline_label)].sort_values("week_start").copy()
    for col in ("net_pnl", "hit_rate", "full_sl_rate", "timeout_rate"):
        base[col] = pd.to_numeric(base[col], errors="coerce")
    base_signals = base[["week", "week_start"]].copy()
    base_signals["prev_net_pnl"] = base["net_pnl"].shift(1)
    base_signals["prev_hit_rate"] = base["hit_rate"].shift(1)
    base_signals["prev_full_sl_rate"] = base["full_sl_rate"].shift(1)
    base_signals["prev_timeout_rate"] = base["timeout_rate"].shift(1)
    base_signals["roll2_net_pnl"] = base["net_pnl"].shift(1).rolling(2, min_periods=2).mean()
    base_signals["roll3_net_pnl"] = base["net_pnl"].shift(1).rolling(3, min_periods=2).mean()
    base_signals["roll2_full_sl_rate"] = base["full_sl_rate"].shift(1).rolling(2, min_periods=2).mean()
    base_signals["roll3_full_sl_rate"] = base["full_sl_rate"].shift(1).rolling(3, min_periods=2).mean()
    base_signals["roll2_hit_rate"] = base["hit_rate"].shift(1).rolling(2, min_periods=2).mean()
    base_signals["roll3_hit_rate"] = base["hit_rate"].shift(1).rolling(3, min_periods=2).mean()
    signals = signals.merge(base_signals, on=["week", "week_start"], how="left")
    action_labels = [c for c in wide.columns if c != baseline_label]
    deltas = wide[action_labels].sub(wide[baseline_label], axis=0).reset_index()
    deltas = signals[["week", "week_start"]].merge(deltas, on="week", how="left")
    return weekly, signals, deltas


def _select_feature_columns(signals: pd.DataFrame, feature_set: str) -> list[str]:
    prefixes = FEATURE_SETS[feature_set]
    cols = []
    for col in signals.columns:
        if col in {"week", "week_start"}:
            continue
        if any(col.startswith(prefix) for prefix in prefixes):
            cols.append(col)
    return cols


def _standardized_distances(train_x: pd.DataFrame, current_x: pd.Series) -> np.ndarray:
    arr = train_x.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    cur = pd.to_numeric(current_x, errors="coerce").to_numpy(dtype=np.float64)
    finite_col = np.isfinite(arr).sum(axis=0) >= max(4, int(0.5 * arr.shape[0]))
    if not finite_col.any():
        return np.full(arr.shape[0], np.inf, dtype=np.float64)
    arr = arr[:, finite_col]
    cur = cur[finite_col]
    med = np.nanmedian(arr, axis=0)
    q75 = np.nanquantile(arr, 0.75, axis=0)
    q25 = np.nanquantile(arr, 0.25, axis=0)
    scale = q75 - q25
    std = np.nanstd(arr, axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 1e-9), scale, std)
    good = np.isfinite(scale) & (scale > 1e-9) & np.isfinite(cur)
    if not good.any():
        return np.full(arr.shape[0], np.inf, dtype=np.float64)
    z = (arr[:, good] - med[good]) / scale[good]
    cz = (cur[good] - med[good]) / scale[good]
    diff = z - cz[None, :]
    valid = np.isfinite(diff)
    counts = valid.sum(axis=1)
    sq = np.where(valid, diff * diff, 0.0).sum(axis=1)
    dist = np.full(arr.shape[0], np.inf, dtype=np.float64)
    np.divide(sq, counts, out=dist, where=counts > 0)
    return dist.astype(np.float64)


def _walk_forward(
    signals: pd.DataFrame,
    deltas: pd.DataFrame,
    wide: pd.DataFrame,
    baseline_label: str,
    config: Config,
    min_train_weeks: int,
) -> pd.DataFrame:
    feature_cols = _select_feature_columns(signals, config.feature_set)
    if not feature_cols:
        raise ValueError(f"No feature columns for {config.feature_set}")
    action_cols = [c for c in deltas.columns if c not in {"week", "week_start"}]
    rows: list[dict[str, object]] = []
    for pos, row in signals.iterrows():
        if pos < min_train_weeks:
            continue
        train_signals = signals.iloc[:pos].reset_index(drop=True)
        train_deltas = deltas.iloc[:pos].reset_index(drop=True)
        distances = _standardized_distances(train_signals[feature_cols], row[feature_cols])
        if not np.isfinite(distances).any():
            selected = baseline_label
            trigger = False
            best = {"action": baseline_label, "mean_delta": 0.0, "q20_delta": 0.0, "positive_share": 0.0, "score": 0.0, "neighbor_count": 0}
        else:
            order = np.argsort(distances)
            order = order[np.isfinite(distances[order])][: config.k_neighbors]
            best = {"action": baseline_label, "mean_delta": 0.0, "q20_delta": 0.0, "positive_share": 0.0, "score": 0.0, "neighbor_count": int(len(order))}
            selected = baseline_label
            trigger = False
            if len(order) >= config.min_neighbor_count:
                for action in action_cols:
                    vals = pd.to_numeric(train_deltas.loc[order, action], errors="coerce").to_numpy(dtype=np.float64)
                    vals = vals[np.isfinite(vals)]
                    if vals.size < config.min_neighbor_count:
                        continue
                    mean_delta = float(np.mean(vals))
                    q20_delta = float(np.quantile(vals, 0.20))
                    positive_share = float(np.mean(vals > 0.0))
                    score = mean_delta + config.score_q20_weight * q20_delta
                    passes = (
                        mean_delta >= config.min_mean_delta
                        and q20_delta >= config.min_q20_delta
                        and positive_share >= config.min_positive_share
                    )
                    if passes and score > float(best["score"]):
                        best = {
                            "action": action,
                            "mean_delta": mean_delta,
                            "q20_delta": q20_delta,
                            "positive_share": positive_share,
                            "score": score,
                            "neighbor_count": int(vals.size),
                        }
                        selected = action
                        trigger = True
        week = row["week"]
        baseline_net = float(wide.loc[week, baseline_label])
        selected_net = float(wide.loc[week, selected])
        rows.append(
            {
                "week": week,
                "week_start": row["week_start"],
                "selected_label": selected,
                "trigger": trigger,
                "feature_set": config.feature_set,
                "k_neighbors": config.k_neighbors,
                "min_mean_delta": config.min_mean_delta,
                "min_q20_delta": config.min_q20_delta,
                "min_positive_share": config.min_positive_share,
                "score_q20_weight": config.score_q20_weight,
                "estimated_mean_delta": float(best["mean_delta"]),
                "estimated_q20_delta": float(best["q20_delta"]),
                "estimated_positive_share": float(best["positive_share"]),
                "estimated_score": float(best["score"]),
                "neighbor_count": int(best["neighbor_count"]),
                "net_pnl": selected_net,
                "baseline_net_pnl": baseline_net,
                "delta_net_pnl_vs_baseline": selected_net - baseline_net,
            }
        )
    return pd.DataFrame(rows)


def _configs() -> list[Config]:
    configs: list[Config] = []
    feature_sets = (
        "recent_hr_surprise",
        "uncertainty",
        "recent_hr_uncertainty",
        "baseline_perf",
        "baseline_plus_recent_uncert",
        "all_diagnostics",
    )
    for feature_set in feature_sets:
        for k in (3, 5, 8):
            for min_mean in (500.0, 1000.0):
                for min_q20 in (0.0, 500.0):
                    for min_pos in (0.70, 0.80):
                        for q20_w in (0.50, 1.00):
                            configs.append(
                                Config(
                                    feature_set=feature_set,
                                    k_neighbors=k,
                                    min_neighbor_count=min(3, k),
                                    min_mean_delta=min_mean,
                                    min_q20_delta=min_q20,
                                    min_positive_share=min_pos,
                                    score_q20_weight=q20_w,
                                )
                            )
    return configs


def _markdown_table(frame: pd.DataFrame, cols: list[str], max_rows: int | None = None) -> str:
    view = frame[cols].head(max_rows).copy() if max_rows else frame[cols].copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{x:,.3f}")
    return view.to_markdown(index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant-weekly", type=Path, required=True)
    parser.add_argument("--diagnostic-signals", type=Path, required=True)
    parser.add_argument("--baseline-label", default="all_heads")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-train-weeks", type=int, default=8)
    parser.add_argument("--q35-weight", type=float, default=0.70)
    parser.add_argument("--q20-weight", type=float, default=0.30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    weekly, signals, deltas = _load_inputs(args.variant_weekly, args.diagnostic_signals, args.baseline_label)
    wide = weekly.pivot(index="week", columns="label", values="net_pnl")
    configs = _configs()
    runs: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    for idx, config in enumerate(configs):
        wf = _walk_forward(signals, deltas, wide, args.baseline_label, config, args.min_train_weeks)
        if wf.empty:
            continue
        baseline_values = wf["baseline_net_pnl"].to_numpy(dtype=np.float64)
        base_summary = _summary(baseline_values, args.q35_weight, args.q20_weight)
        run_summary = _summary(wf["net_pnl"].to_numpy(dtype=np.float64), args.q35_weight, args.q20_weight)
        row = {
            "run_id": idx,
            **config.__dict__,
            **run_summary,
            "trigger_count": int(wf["trigger"].sum()),
            "delta_sum_net_pnl": float(run_summary["sum_net_pnl"] - base_summary["sum_net_pnl"]),
            "delta_objective": float(run_summary["objective"] - base_summary["objective"]),
            "delta_worst_week_net_pnl": float(run_summary["worst_week_net_pnl"] - base_summary["worst_week_net_pnl"]),
            "baseline_objective": float(base_summary["objective"]),
            "baseline_sum_net_pnl": float(base_summary["sum_net_pnl"]),
        }
        summary_rows.append(row)
        if idx % 100 == 0:
            wf = wf.copy()
            wf["run_id"] = idx
        runs.append(wf.assign(run_id=idx))
    summary = pd.DataFrame(summary_rows)
    if summary.empty:
        raise RuntimeError("No selector runs produced output")
    summary = summary.sort_values(
        ["delta_objective", "delta_sum_net_pnl", "delta_worst_week_net_pnl"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    best_run_id = int(summary.iloc[0]["run_id"])
    all_wf = pd.concat(runs, ignore_index=True)
    best_wf = all_wf[all_wf["run_id"].eq(best_run_id)].copy()
    baseline_summary = _summary(best_wf["baseline_net_pnl"].to_numpy(dtype=np.float64), args.q35_weight, args.q20_weight)
    best_summary = _summary(best_wf["net_pnl"].to_numpy(dtype=np.float64), args.q35_weight, args.q20_weight)
    summary.to_csv(args.output_dir / "conservative_state_selector_grid.csv", index=False)
    best_wf.to_csv(args.output_dir / "conservative_state_selector_best_walk_forward.csv", index=False)
    pd.DataFrame([baseline_summary | {"policy": "baseline_same_eval_weeks"}, best_summary | {"policy": "conservative_state_selector"}]).to_csv(
        args.output_dir / "conservative_state_selector_summary.csv", index=False
    )
    (args.output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "variant_weekly": str(args.variant_weekly),
                "diagnostic_signals": str(args.diagnostic_signals),
                "baseline_label": args.baseline_label,
                "min_train_weeks": args.min_train_weeks,
                "q35_weight": args.q35_weight,
                "q20_weight": args.q20_weight,
                "grid_runs": int(len(summary)),
                "best_run_id": best_run_id,
            },
            indent=2,
        )
        + "\n"
    )
    best = summary.iloc[0]
    triggered = best_wf[best_wf["trigger"].astype(bool)].copy()
    lines = [
        "# Conservative State Action Selector",
        "",
        "Development proxy over existing weekly replay variants. This is not a new portfolio replay; it tests whether drift/recent-HR/OOD/uncertainty state can select a better weekly action using only prior weeks.",
        "",
        "## Best Configuration",
        "",
        _markdown_table(pd.DataFrame([best]), ["run_id", "feature_set", "k_neighbors", "min_mean_delta", "min_q20_delta", "min_positive_share", "score_q20_weight", "trigger_count", "delta_sum_net_pnl", "delta_objective", "delta_worst_week_net_pnl"]),
        "",
        "## Policy Summary",
        "",
        _markdown_table(
            pd.read_csv(args.output_dir / "conservative_state_selector_summary.csv"),
            ["policy", "weeks", "sum_net_pnl", "avg_week_net_pnl", "q10_week_net_pnl", "q20_week_net_pnl", "q35_week_net_pnl", "worst_week_net_pnl", "positive_weeks", "objective"],
        ),
        "",
        "## Triggered Weeks",
        "",
        _markdown_table(
            triggered,
            ["week", "selected_label", "estimated_mean_delta", "estimated_q20_delta", "estimated_positive_share", "neighbor_count", "net_pnl", "baseline_net_pnl", "delta_net_pnl_vs_baseline"],
        )
        if not triggered.empty
        else "No triggered weeks.",
        "",
        "## Top Grid Runs",
        "",
        _markdown_table(
            summary,
            ["run_id", "feature_set", "k_neighbors", "min_mean_delta", "min_q20_delta", "min_positive_share", "score_q20_weight", "trigger_count", "delta_sum_net_pnl", "delta_objective", "delta_worst_week_net_pnl"],
            max_rows=15,
        ),
        "",
        "## Readout",
        "",
        "- Promotion requires positive objective, positive net PnL, and no tail degradation on this development proxy before a true continuous replay.",
        "- A no-trigger result is acceptable as a fail-closed control but does not improve the policy.",
    ]
    (args.output_dir / "conservative_state_selector_report.md").write_text("\n".join(lines) + "\n")
    print(args.output_dir / "conservative_state_selector_report.md")


if __name__ == "__main__":
    main()
