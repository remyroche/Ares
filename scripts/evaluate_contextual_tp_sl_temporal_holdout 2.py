#!/usr/bin/env python3
"""Temporal holdout audit for contextual TP/SL combination sweeps.

The combo sweep optimises head-specific TP/SL arms over replay outputs.  This
script reuses those replay outputs to test whether a combo selected on an
earlier period survives a later period, without rerunning expensive portfolio
simulation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import numpy as np
import pandas as pd


STATIC_COMBO_ID = "long_bars:S_long_dist:S_short_asset:S_short_bollinger:S"


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
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _parse_week_start(series: pd.Series) -> pd.Series:
    starts = series.astype(str).str.split("/", n=1).str[0]
    return pd.to_datetime(starts, utc=True, errors="coerce")


def _max_drawdown(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    equity = np.cumsum(np.nan_to_num(values.astype(float), nan=0.0))
    running_max = np.maximum.accumulate(np.r_[0.0, equity])[:-1]
    drawdown = equity - running_max
    return float(np.nanmin(drawdown)) if drawdown.size else 0.0


def _q(arr: np.ndarray, pct: float) -> float:
    finite = arr[np.isfinite(arr)]
    return float(np.nanpercentile(finite, pct)) if finite.size else 0.0


def _summarise_period(
    daily: pd.DataFrame,
    weekly: pd.DataFrame,
    *,
    combo_cols: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for combo_id, dgroup in daily.groupby("combo_id", sort=False):
        wgroup = weekly.loc[weekly["combo_id"].eq(combo_id)]
        net_day = pd.to_numeric(dgroup["net_pnl"], errors="coerce").to_numpy(dtype=float)
        gross_day = pd.to_numeric(dgroup["gross_pnl"], errors="coerce").to_numpy(dtype=float)
        trades_day = pd.to_numeric(dgroup["trades"], errors="coerce").fillna(0.0)
        hit_day = pd.to_numeric(dgroup["hit_rate"], errors="coerce")
        net_week = pd.to_numeric(wgroup["net_pnl"], errors="coerce").to_numpy(dtype=float)
        gross_week = pd.to_numeric(wgroup["gross_pnl"], errors="coerce").to_numpy(dtype=float)
        trades = int(trades_day.sum())
        weighted_hit = float(
            np.average(hit_day.fillna(0.0).to_numpy(dtype=float), weights=trades_day.to_numpy(dtype=float))
        ) if trades > 0 else 0.0
        rec: Dict[str, Any] = {
            "combo_id": combo_id,
            "net_pnl": float(np.nansum(net_day)),
            "gross_pnl": float(np.nansum(gross_day)),
            "trade_count": trades,
            "mean_net_pnl_per_trade": float(np.nansum(net_day) / max(trades, 1)),
            "hit_rate": weighted_hit,
            "daily_count": int(net_day.size),
            "daily_q05_pnl": _q(net_day, 5),
            "daily_q10_pnl": _q(net_day, 10),
            "daily_q20_pnl": _q(net_day, 20),
            "daily_q35_pnl": _q(net_day, 35),
            "daily_min_pnl": float(np.nanmin(net_day)) if net_day.size else 0.0,
            "daily_positive_rate": float(np.nanmean(net_day > 0.0)) if net_day.size else 0.0,
            "weekly_count": int(net_week.size),
            "weekly_q05_pnl": _q(net_week, 5),
            "weekly_q10_pnl": _q(net_week, 10),
            "weekly_q20_pnl": _q(net_week, 20),
            "weekly_q35_pnl": _q(net_week, 35),
            "weekly_min_pnl": float(np.nanmin(net_week)) if net_week.size else 0.0,
            "weekly_positive_rate": float(np.nanmean(net_week > 0.0)) if net_week.size else 0.0,
            "avg_week_pnl": float(np.nanmean(net_week)) if net_week.size else 0.0,
            "max_drawdown_pnl": _max_drawdown(net_day),
        }
        rec["objective"] = (
            rec["avg_week_pnl"] + 0.7 * rec["daily_q35_pnl"] + 0.3 * rec["daily_q20_pnl"]
        )
        rows.append(rec)
    out = pd.DataFrame(rows)
    if combo_cols is not None and not combo_cols.empty and not out.empty:
        out = out.merge(combo_cols.drop_duplicates("combo_id"), on="combo_id", how="left")
    return _add_balanced_score(out)


def _normalise(series: pd.Series, *, high: bool = True) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce").astype(float)
    lo = float(vals.min())
    hi = float(vals.max())
    if not np.isfinite(lo) or not np.isfinite(hi) or abs(hi - lo) <= 1e-12:
        return pd.Series(0.5, index=series.index, dtype=float)
    score = (vals - lo) / (hi - lo)
    return score if high else 1.0 - score


def _add_balanced_score(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy()
    out["n_net_pnl"] = _normalise(out["net_pnl"], high=True)
    out["n_daily_q10_pnl"] = _normalise(out["daily_q10_pnl"], high=True)
    out["n_weekly_q10_pnl"] = _normalise(out["weekly_q10_pnl"], high=True)
    out["n_max_drawdown_pnl"] = _normalise(out["max_drawdown_pnl"], high=True)
    out["balanced_score"] = (
        0.45 * out["n_net_pnl"]
        + 0.25 * out["n_daily_q10_pnl"]
        + 0.15 * out["n_weekly_q10_pnl"]
        + 0.15 * out["n_max_drawdown_pnl"]
    )
    return out


def _add_period_baseline_deltas(
    metrics: pd.DataFrame,
    weekly: pd.DataFrame,
    *,
    baseline_combo_id: str,
) -> pd.DataFrame:
    if metrics.empty:
        return metrics
    baseline_rows = metrics.loc[metrics["combo_id"].eq(baseline_combo_id)]
    if baseline_rows.empty:
        raise KeyError(f"Missing baseline combo in period metrics: {baseline_combo_id}")
    baseline = baseline_rows.iloc[0]
    out = metrics.copy()
    for col in (
        "net_pnl",
        "gross_pnl",
        "trade_count",
        "hit_rate",
        "daily_q10_pnl",
        "daily_q20_pnl",
        "daily_q35_pnl",
        "weekly_q10_pnl",
        "weekly_min_pnl",
        "max_drawdown_pnl",
        "objective",
        "balanced_score",
    ):
        if col in out.columns:
            out[f"delta_{col}"] = pd.to_numeric(out[col], errors="coerce") - float(baseline[col])

    base_weekly = weekly.loc[weekly["combo_id"].eq(baseline_combo_id), ["week", "net_pnl"]].rename(
        columns={"net_pnl": "baseline_week_net_pnl"}
    )
    if not base_weekly.empty:
        merged = weekly.merge(base_weekly, on="week", how="left")
        merged["delta_week_net_pnl"] = pd.to_numeric(merged["net_pnl"], errors="coerce") - pd.to_numeric(
            merged["baseline_week_net_pnl"], errors="coerce"
        )
        rows: List[Dict[str, Any]] = []
        for combo_id, group in merged.groupby("combo_id", sort=False):
            values = pd.to_numeric(group["delta_week_net_pnl"], errors="coerce").dropna().to_numpy(dtype=float)
            rows.append(
                {
                    "combo_id": combo_id,
                    "delta_week_count": int(values.size),
                    "delta_week_sum_pnl": float(values.sum()) if values.size else np.nan,
                    "delta_week_q10_pnl": float(np.nanpercentile(values, 10)) if values.size else np.nan,
                    "delta_week_q20_pnl": float(np.nanpercentile(values, 20)) if values.size else np.nan,
                    "delta_worst_week_pnl": float(np.nanmin(values)) if values.size else np.nan,
                    "positive_week_delta_share": float(np.nanmean(values > 0.0)) if values.size else np.nan,
                }
            )
        out = out.merge(pd.DataFrame(rows), on="combo_id", how="left")
    return out


def _select_combo(frame: pd.DataFrame, metric: str) -> pd.Series:
    if frame.empty:
        raise ValueError("Cannot select from an empty period frame")
    if metric not in frame.columns:
        raise ValueError(f"Metric {metric!r} not in frame columns")
    return frame.sort_values(metric, ascending=False).iloc[0]


def _select_tail_gate_combo(frame: pd.DataFrame, args: argparse.Namespace) -> pd.Series:
    if frame.empty:
        raise ValueError("Cannot select from an empty period frame")
    required = [
        "delta_net_pnl",
        "delta_max_drawdown_pnl",
        "delta_week_q10_pnl",
        "delta_week_q20_pnl",
        "positive_week_delta_share",
    ]
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise ValueError(f"Missing tail-gate selection columns: {missing}")
    out = frame.copy()
    out["tail_gate_pass"] = (
        (pd.to_numeric(out["delta_net_pnl"], errors="coerce") >= float(args.min_train_delta_net_pnl))
        & (
            pd.to_numeric(out["delta_max_drawdown_pnl"], errors="coerce")
            >= float(args.min_train_delta_max_drawdown_pnl)
        )
        & (
            pd.to_numeric(out["delta_week_q10_pnl"], errors="coerce")
            >= float(args.min_train_delta_week_q10_pnl)
        )
        & (
            pd.to_numeric(out["delta_week_q20_pnl"], errors="coerce")
            >= float(args.min_train_delta_week_q20_pnl)
        )
        & (
            pd.to_numeric(out["positive_week_delta_share"], errors="coerce")
            >= float(args.min_train_positive_week_delta_share)
        )
    )
    out["tail_adjusted_train_score"] = (
        pd.to_numeric(out["delta_net_pnl"], errors="coerce").fillna(-1.0e18)
        + 0.7 * pd.to_numeric(out["delta_week_q20_pnl"], errors="coerce").fillna(-1.0e6)
        + 0.3 * pd.to_numeric(out["delta_week_q10_pnl"], errors="coerce").fillna(-1.0e6)
    )
    candidates = out.loc[out["tail_gate_pass"]].copy()
    if candidates.empty and bool(args.tail_gate_fallback):
        candidates = out.copy()
    if candidates.empty:
        raise ValueError("No combo passed tail-gate selection and fallback is disabled")
    return candidates.sort_values(["tail_adjusted_train_score", "delta_net_pnl"], ascending=False).iloc[0]


def _row_for_combo(frame: pd.DataFrame, combo_id: str) -> pd.Series:
    rows = frame.loc[frame["combo_id"].eq(combo_id)]
    if rows.empty:
        raise KeyError(f"Missing combo_id: {combo_id}")
    return rows.iloc[0]


def _comparison_rows(
    *,
    split_name: str,
    train_end: str,
    holdout_start: str,
    holdout_end: str,
    selection_metric: str,
    selected: pd.Series,
    full_window_best: pd.Series,
    benchmark_combos: Iterable[str],
    train_metrics: pd.DataFrame,
    holdout_metrics: pd.DataFrame,
) -> List[Dict[str, Any]]:
    selected_id = str(selected["combo_id"])
    static_holdout = _row_for_combo(holdout_metrics, STATIC_COMBO_ID)
    selected_holdout = _row_for_combo(holdout_metrics, selected_id)
    full_best_holdout = _row_for_combo(holdout_metrics, str(full_window_best["combo_id"]))
    rows: List[Dict[str, Any]] = []
    for label, row in (
        ("static", static_holdout),
        ("selected_on_train", selected_holdout),
        ("full_window_best", full_best_holdout),
    ):
        rec = {
            "split": split_name,
            "train_end": train_end,
            "holdout_start": holdout_start,
            "holdout_end": holdout_end,
            "selection_metric": selection_metric,
            "variant": label,
            "combo_id": row["combo_id"],
        }
        for col in (
            "net_pnl",
            "gross_pnl",
            "trade_count",
            "mean_net_pnl_per_trade",
            "hit_rate",
            "daily_q10_pnl",
            "daily_q20_pnl",
            "daily_q35_pnl",
            "weekly_q10_pnl",
            "weekly_min_pnl",
            "max_drawdown_pnl",
            "objective",
            "balanced_score",
        ):
            rec[col] = row.get(col, np.nan)
        rows.append(rec)
    for idx, combo_id in enumerate(benchmark_combos):
        row = _row_for_combo(holdout_metrics, str(combo_id))
        rec = {
            "split": split_name,
            "train_end": train_end,
            "holdout_start": holdout_start,
            "holdout_end": holdout_end,
            "selection_metric": selection_metric,
            "variant": f"benchmark_{idx + 1}",
            "combo_id": row["combo_id"],
        }
        for col in (
            "net_pnl",
            "gross_pnl",
            "trade_count",
            "mean_net_pnl_per_trade",
            "hit_rate",
            "daily_q10_pnl",
            "daily_q20_pnl",
            "daily_q35_pnl",
            "weekly_q10_pnl",
            "weekly_min_pnl",
            "max_drawdown_pnl",
            "objective",
            "balanced_score",
        ):
            rec[col] = row.get(col, np.nan)
        rows.append(rec)
    selected_rec = rows[1]
    static_rec = rows[0]
    delta = {
        "split": split_name,
        "train_end": train_end,
        "holdout_start": holdout_start,
        "holdout_end": holdout_end,
        "selection_metric": selection_metric,
        "variant": "delta_selected_minus_static",
        "combo_id": selected_id,
    }
    for col in (
        "net_pnl",
        "gross_pnl",
        "trade_count",
        "mean_net_pnl_per_trade",
        "hit_rate",
        "daily_q10_pnl",
        "daily_q20_pnl",
        "daily_q35_pnl",
        "weekly_q10_pnl",
        "weekly_min_pnl",
        "max_drawdown_pnl",
        "objective",
        "balanced_score",
    ):
        delta[col] = float(selected_rec[col]) - float(static_rec[col])
    rows.append(delta)
    return rows


def _split_dates(value: str) -> pd.Timestamp:
    return pd.Timestamp(value, tz="UTC")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--splits",
        default="pre_apr_holdout_apr_jun:2026-03-31:2026-04-01,"
        "pre_may_holdout_may_jun:2026-04-30:2026-05-01,"
        "pre_jun_holdout_jun:2026-05-31:2026-06-01",
        help=(
            "Comma-separated name:train_end:holdout_start entries. "
            "A fourth holdout_end field is optional."
        ),
    )
    parser.add_argument("--selection-metric", default="balanced_score")
    parser.add_argument(
        "--selection-policy",
        default="metric",
        choices=["metric", "tail_gate"],
        help="Use a single selection metric or train-period baseline-delta tail gates.",
    )
    parser.add_argument("--baseline-combo-id", default=STATIC_COMBO_ID)
    parser.add_argument("--min-train-delta-net-pnl", type=float, default=0.0)
    parser.add_argument("--min-train-delta-max-drawdown-pnl", type=float, default=0.0)
    parser.add_argument("--min-train-delta-week-q10-pnl", type=float, default=-1000.0)
    parser.add_argument("--min-train-delta-week-q20-pnl", type=float, default=0.0)
    parser.add_argument("--min-train-positive-week-delta-share", type=float, default=0.60)
    parser.add_argument("--tail-gate-fallback", action="store_true")
    parser.add_argument(
        "--benchmark-combo",
        action="append",
        default=[],
        help="Fixed combo_id to report on each holdout. Repeatable.",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    daily = pd.read_csv(args.source_dir / "head_arm_combination_daily.csv")
    weekly = pd.read_csv(args.source_dir / "head_arm_combination_weekly.csv")
    summary = pd.read_csv(args.source_dir / "head_arm_combination_summary.csv")
    combo_cols = summary[
        [
            "combo_id",
            "long_bars_arm",
            "long_dist_arm",
            "short_asset_arm",
            "short_bollinger_arm",
        ]
    ].copy()
    daily["day_ts"] = pd.to_datetime(daily["day"], utc=True, errors="coerce")
    weekly["week_start_ts"] = _parse_week_start(weekly["week"])
    full_window_best = summary.sort_values("balanced_score", ascending=False).iloc[0]

    split_rows: List[Dict[str, Any]] = []
    selected_rows: List[Dict[str, Any]] = []
    all_period_frames: List[pd.DataFrame] = []
    for raw in [part.strip() for part in str(args.splits).split(",") if part.strip()]:
        parts = raw.split(":")
        if len(parts) not in {3, 4}:
            raise ValueError(
                f"Invalid split {raw!r}; expected name:train_end:holdout_start[:holdout_end]"
            )
        name, train_end_s, holdout_start_s = parts[:3]
        holdout_end_s = parts[3] if len(parts) == 4 else ""
        train_end = _split_dates(train_end_s)
        holdout_start = _split_dates(holdout_start_s)
        holdout_end = _split_dates(holdout_end_s) if holdout_end_s else None
        train_daily = daily.loc[daily["day_ts"].le(train_end)].copy()
        holdout_daily = daily.loc[daily["day_ts"].ge(holdout_start)].copy()
        if holdout_end is not None:
            holdout_daily = holdout_daily.loc[holdout_daily["day_ts"].le(holdout_end)].copy()
        train_weekly = weekly.loc[weekly["week_start_ts"].le(train_end)].copy()
        holdout_weekly = weekly.loc[weekly["week_start_ts"].ge(holdout_start)].copy()
        if holdout_end is not None:
            holdout_weekly = holdout_weekly.loc[
                holdout_weekly["week_start_ts"].le(holdout_end)
            ].copy()
        train_metrics = _add_period_baseline_deltas(
            _summarise_period(train_daily, train_weekly, combo_cols=combo_cols),
            train_weekly,
            baseline_combo_id=str(args.baseline_combo_id),
        )
        holdout_metrics = _add_period_baseline_deltas(
            _summarise_period(holdout_daily, holdout_weekly, combo_cols=combo_cols),
            holdout_weekly,
            baseline_combo_id=str(args.baseline_combo_id),
        )
        if str(args.selection_policy) == "tail_gate":
            selected = _select_tail_gate_combo(train_metrics, args)
            selected_score = selected.get("tail_adjusted_train_score", np.nan)
        else:
            selected = _select_combo(train_metrics, str(args.selection_metric))
            selected_score = selected[str(args.selection_metric)]
        selected_rows.append(
            {
                "split": name,
                "train_end": train_end_s,
                "holdout_start": holdout_start_s,
                "holdout_end": holdout_end_s,
                "selection_policy": str(args.selection_policy),
                "selection_metric": str(args.selection_metric),
                "selected_combo_id": selected["combo_id"],
                "selected_train_score": selected_score,
                "selected_tail_gate_pass": selected.get("tail_gate_pass", np.nan),
                "selected_train_delta_net_pnl": selected.get("delta_net_pnl", np.nan),
                "selected_train_delta_week_q10_pnl": selected.get("delta_week_q10_pnl", np.nan),
                "selected_train_delta_week_q20_pnl": selected.get("delta_week_q20_pnl", np.nan),
                "selected_train_positive_week_delta_share": selected.get("positive_week_delta_share", np.nan),
                "long_bars_arm": selected.get("long_bars_arm"),
                "long_dist_arm": selected.get("long_dist_arm"),
                "short_asset_arm": selected.get("short_asset_arm"),
                "short_bollinger_arm": selected.get("short_bollinger_arm"),
            }
        )
        split_rows.extend(
            _comparison_rows(
                split_name=name,
                train_end=train_end_s,
                holdout_start=holdout_start_s,
                holdout_end=holdout_end_s,
                selection_metric=str(args.selection_metric),
                selected=selected,
                full_window_best=full_window_best,
                benchmark_combos=list(args.benchmark_combo),
                train_metrics=train_metrics,
                holdout_metrics=holdout_metrics,
            )
        )
        train_metrics.insert(0, "period_role", "train")
        holdout_metrics.insert(0, "period_role", "holdout")
        train_metrics.insert(0, "split", name)
        holdout_metrics.insert(0, "split", name)
        all_period_frames.extend([train_metrics, holdout_metrics])

    selected_frame = pd.DataFrame(selected_rows)
    comparison = pd.DataFrame(split_rows)
    period_metrics = pd.concat(all_period_frames, ignore_index=True) if all_period_frames else pd.DataFrame()
    selected_frame.to_csv(args.out_dir / "temporal_holdout_selected_combos.csv", index=False)
    comparison.to_csv(args.out_dir / "temporal_holdout_comparison.csv", index=False)
    period_metrics.to_csv(args.out_dir / "temporal_holdout_all_combo_metrics.csv", index=False)

    payload = {
        "source_dir": str(args.source_dir),
        "selection_policy": str(args.selection_policy),
        "selection_metric": str(args.selection_metric),
        "static_combo_id": str(args.baseline_combo_id),
        "full_window_best_combo_id": str(full_window_best["combo_id"]),
        "benchmark_combos": list(args.benchmark_combo),
        "selected": selected_rows,
        "comparison": comparison.to_dict(orient="records"),
    }
    (args.out_dir / "temporal_holdout_report.json").write_text(
        json.dumps(_json_safe(payload), indent=2),
        encoding="utf-8",
    )

    lines = [
        "# Contextual TP/SL Temporal Holdout Audit",
        "",
        f"Source: `{args.source_dir}`",
        f"Selection policy: `{args.selection_policy}`",
        f"Selection metric: `{args.selection_metric}`",
        f"Static combo: `{args.baseline_combo_id}`",
        f"Full-window best combo: `{full_window_best['combo_id']}`",
        "",
        "## Selected Combos",
        "",
        selected_frame.to_markdown(index=False),
        "",
        "## Holdout Comparison",
        "",
        comparison.to_markdown(index=False),
    ]
    (args.out_dir / "temporal_holdout_report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            _json_safe(
                {
                    "out_dir": str(args.out_dir),
                    "splits": len(selected_frame),
                    "selection_metric": str(args.selection_metric),
                }
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
