#!/usr/bin/env python3
"""Diagnose weekly observable states where a C3el run helps or hurts.

This is an artifact-only diagnostic.  It compares C3el-vs-baseline weekly
replay deltas to weekly aggregates of observable action/opportunity features.
It is meant to propose fixed conditioning hypotheses, not to validate them.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


HEADS = ("long_bars", "long_dist", "short_asset", "short_boll")
TARGET_COLS = {
    "delta_full_J",
    "delta_immediate_J",
    "delta_full_net_pnl",
    "delta_full_cost_pnl",
    "delta_full_turnover",
    "best_multiplier",
    "best_gain",
    "best_margin",
    "best_immediate_gain",
    "best_nonbaseline_gain",
    "worst_nonbaseline_gain",
    "y_intervene",
}
KEY_COLS = {"timestamp", "strategy_id", "multiplier", "week_start", "head"}


def _head_from_strategy(strategy_id: str) -> str:
    text = str(strategy_id)
    for head in HEADS:
        if text.startswith(head):
            return head
    return "unknown"


def _week_start(values: pd.Series) -> pd.Series:
    ts = pd.to_datetime(values, utc=True, errors="coerce")
    return ts.dt.normalize() - pd.to_timedelta(ts.dt.weekday, unit="D")


def _read_frame(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path, columns=columns)
    return pd.read_csv(path, usecols=columns)


def _header(path: Path) -> list[str]:
    if path.suffix.lower() in {".parquet", ".pq"}:
        try:
            import pyarrow.parquet as pq

            return list(pq.ParquetFile(path).schema.names)
        except Exception:
            return list(pd.read_parquet(path).columns)
    return list(pd.read_csv(path, nrows=0).columns)


def load_weekly_deltas(weekly_path: Path, *, arm: str) -> pd.DataFrame:
    weekly = pd.read_csv(weekly_path)
    required = {"arm", "week_start", "net_pnl", "trade_count", "net_hit_rate_pct", "full_sl_rate_pct", "timeout_rate_pct"}
    missing = sorted(required.difference(weekly.columns))
    if missing:
        raise ValueError(f"{weekly_path} missing columns: {missing}")
    weekly["week_start"] = pd.to_datetime(weekly["week_start"], utc=True, errors="coerce")
    base = weekly.loc[weekly["arm"].eq("C0_baseline")].copy()
    cand = weekly.loc[weekly["arm"].eq(arm)].copy()
    if base.empty:
        raise ValueError(f"{weekly_path} has no C0_baseline rows")
    if cand.empty:
        raise ValueError(f"{weekly_path} has no arm rows for {arm}")
    keep = ["week_start", "net_pnl", "trade_count", "net_hit_rate_pct", "full_sl_rate_pct", "timeout_rate_pct"]
    out = cand[keep].merge(base[keep], on="week_start", suffixes=("_candidate", "_baseline"), how="inner")
    for metric in ["net_pnl", "trade_count", "net_hit_rate_pct", "full_sl_rate_pct", "timeout_rate_pct"]:
        out[f"delta_{metric}"] = pd.to_numeric(out[f"{metric}_candidate"], errors="coerce") - pd.to_numeric(
            out[f"{metric}_baseline"], errors="coerce"
        )
    out["positive_delta"] = out["delta_net_pnl"].gt(0.0)
    return out.sort_values("week_start").reset_index(drop=True)


def load_weekly_features(feature_path: Path, *, head: str) -> pd.DataFrame:
    columns = _header(feature_path)
    read_cols = [col for col in columns if col in {"timestamp", "strategy_id", "multiplier", "action_binds"}]
    numeric_candidates = [
        col
        for col in columns
        if col not in KEY_COLS
        and col not in TARGET_COLS
        and col not in {"action_binds", "group_can_bind"}
    ]
    read_cols.extend(numeric_candidates)
    frame = _read_frame(feature_path, columns=sorted(set(read_cols)))
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame = frame.loc[frame["timestamp"].notna()].copy()
    frame["strategy_id"] = frame["strategy_id"].astype(str)
    frame["head"] = frame["strategy_id"].map(_head_from_strategy)
    frame = frame.loc[frame["head"].eq(head)].copy()
    if "multiplier" in frame.columns:
        frame["multiplier"] = pd.to_numeric(frame["multiplier"], errors="coerce").fillna(1.0)
        frame = frame.loc[frame["multiplier"].lt(1.0)].copy()
    if "action_binds" in frame.columns:
        frame["action_binds"] = pd.to_numeric(frame["action_binds"], errors="coerce").fillna(0.0)
        frame = frame.loc[frame["action_binds"].gt(0.0)].copy()
    frame["week_start"] = _week_start(frame["timestamp"])
    numeric_cols = []
    for col in numeric_candidates:
        if col not in frame.columns:
            continue
        vals = pd.to_numeric(frame[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if vals.notna().sum() >= 5 and vals.nunique(dropna=True) >= 2:
            frame[col] = vals
            numeric_cols.append(col)
    if not numeric_cols:
        return pd.DataFrame(columns=["week_start"])
    parts = []
    grouped = frame.groupby("week_start", dropna=False)
    for func_name, func in {
        "mean": "mean",
        "median": "median",
        "q75": lambda s: s.quantile(0.75),
        "q90": lambda s: s.quantile(0.90),
        "max": "max",
    }.items():
        agg = grouped[numeric_cols].agg(func).reset_index()
        agg = agg.rename(columns={col: f"{col}__{func_name}" for col in numeric_cols})
        parts.append(agg)
    out = parts[0]
    for part in parts[1:]:
        out = out.merge(part, on="week_start", how="outer")
    out["feature_row_count"] = grouped.size().reindex(out["week_start"]).to_numpy()
    return out.sort_values("week_start").reset_index(drop=True)


def score_weekly_conditions(
    joined: pd.DataFrame,
    *,
    min_positive_weeks: int = 1,
    min_abs_median_separation: float = 1e-9,
) -> pd.DataFrame:
    feature_cols = [
        col
        for col in joined.columns
        if col not in {
            "week_start",
            "positive_delta",
            "net_pnl_candidate",
            "net_pnl_baseline",
            "trade_count_candidate",
            "trade_count_baseline",
            "net_hit_rate_pct_candidate",
            "net_hit_rate_pct_baseline",
            "full_sl_rate_pct_candidate",
            "full_sl_rate_pct_baseline",
            "timeout_rate_pct_candidate",
            "timeout_rate_pct_baseline",
            "delta_net_pnl",
            "delta_trade_count",
            "delta_net_hit_rate_pct",
            "delta_full_sl_rate_pct",
            "delta_timeout_rate_pct",
        }
        and pd.api.types.is_numeric_dtype(joined[col])
    ]
    rows: list[dict[str, Any]] = []
    for feature in feature_cols:
        work = joined[["week_start", "positive_delta", "delta_net_pnl", feature]].dropna().copy()
        if len(work) < 3 or work[feature].nunique() < 2:
            continue
        pos = work.loc[work["positive_delta"]]
        neg = work.loc[~work["positive_delta"]]
        if len(pos) < min_positive_weeks or neg.empty:
            continue
        pos_median = float(pos[feature].median())
        neg_median = float(neg[feature].median())
        if abs(pos_median - neg_median) <= float(min_abs_median_separation):
            continue
        direction = "high" if pos_median >= neg_median else "low"
        threshold = float(0.5 * (pos_median + neg_median))
        if direction == "high":
            selected = work.loc[work[feature].ge(threshold)].copy()
        else:
            selected = work.loc[work[feature].le(threshold)].copy()
        if selected.empty:
            continue
        rows.append(
            {
                "feature": feature,
                "direction": direction,
                "threshold": threshold,
                "weeks": int(len(work)),
                "selected_weeks": int(len(selected)),
                "selected_positive_week_share": float(selected["positive_delta"].mean()),
                "selected_delta_net_pnl_sum": float(selected["delta_net_pnl"].sum()),
                "selected_delta_net_pnl_mean": float(selected["delta_net_pnl"].mean()),
                "selected_worst_delta_net_pnl": float(selected["delta_net_pnl"].min()),
                "positive_median": pos_median,
                "negative_median": neg_median,
                "median_separation": float(pos_median - neg_median),
            }
        )
    report = pd.DataFrame(rows)
    if report.empty:
        return report
    report["objective"] = (
        report["selected_delta_net_pnl_sum"]
        + 1000.0 * report["selected_positive_week_share"]
        + 0.25 * report["selected_worst_delta_net_pnl"]
    )
    return report.sort_values(
        ["selected_delta_net_pnl_sum", "selected_positive_week_share", "selected_worst_delta_net_pnl"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def _json_safe(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    return value


def write_report(
    *,
    weekly_deltas: pd.DataFrame,
    weekly_features: pd.DataFrame,
    conditions: pd.DataFrame,
    out_dir: Path,
    weekly_path: Path,
    feature_path: Path,
    head: str,
    arm: str,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    joined = weekly_deltas.merge(weekly_features, on="week_start", how="left")
    joined.insert(0, "head", head)
    conditions_out = conditions.copy()
    if not conditions_out.empty:
        conditions_out.insert(0, "head", head)
    joined.to_csv(out_dir / "weekly_conditioning_joined.csv", index=False)
    conditions_out.to_csv(out_dir / "weekly_condition_candidates.csv", index=False)
    top = conditions.head(20) if not conditions.empty else pd.DataFrame()
    display_cols = [
        "feature",
        "direction",
        "threshold",
        "selected_weeks",
        "selected_positive_week_share",
        "selected_delta_net_pnl_sum",
        "selected_delta_net_pnl_mean",
        "selected_worst_delta_net_pnl",
        "positive_median",
        "negative_median",
    ]
    lines = [
        f"# C3el {head} weekly-conditioning diagnostic",
        "",
        "This report compares weekly C3el-vs-baseline deltas with observable weekly action/opportunity features.",
        "",
        "## Weekly Deltas",
        "",
        weekly_deltas[
            [
                "week_start",
                "delta_net_pnl",
                "delta_net_hit_rate_pct",
                "delta_full_sl_rate_pct",
                "delta_trade_count",
                "positive_delta",
            ]
        ].to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Top Weekly Conditions",
        "",
        top[display_cols].to_markdown(index=False, floatfmt=".4f") if not top.empty else "No conditions met support requirements.",
        "",
        "## Readout",
        "",
    ]
    if conditions.empty:
        lines.append("No weekly observable state separated helpful from harmful C3el weeks.")
    else:
        best = conditions.iloc[0]
        lines.extend(
            [
                f"Best weekly hypothesis: `{best['feature']} {best['direction']} {best['threshold']:.6g}`.",
                f"It selects `{int(best['selected_weeks'])}` weeks with delta net PnL `{best['selected_delta_net_pnl_sum']:.2f}` and positive-week share `{best['selected_positive_week_share']:.2%}`.",
                "",
                "This is a five-week hypothesis-mining diagnostic. A valid ablation must fix the condition before replaying another interval.",
            ]
        )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    manifest = {
        "generated_by": "diagnose_c3el_weekly_conditioning",
        "weekly_path": str(weekly_path),
        "feature_path": str(feature_path),
        "head": head,
        "arm": arm,
        "weeks": int(len(weekly_deltas)),
        "outputs": {
            "summary": str(out_dir / "summary.md"),
            "joined": str(out_dir / "weekly_conditioning_joined.csv"),
            "conditions": str(out_dir / "weekly_condition_candidates.csv"),
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(_json_safe(manifest), indent=2, sort_keys=True))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weekly", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--head", choices=HEADS, required=True)
    parser.add_argument("--arm", default="C3el_head_native")
    parser.add_argument("--min-abs-median-separation", type=float, default=1e-9)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    weekly = load_weekly_deltas(args.weekly, arm=args.arm)
    features = load_weekly_features(args.features, head=args.head)
    joined = weekly.merge(features, on="week_start", how="left")
    conditions = score_weekly_conditions(joined, min_abs_median_separation=float(args.min_abs_median_separation))
    write_report(
        weekly_deltas=weekly,
        weekly_features=features,
        conditions=conditions,
        out_dir=args.out_dir,
        weekly_path=args.weekly,
        feature_path=args.features,
        head=args.head,
        arm=args.arm,
    )
    print((args.out_dir / "summary.md").read_text())


if __name__ == "__main__":
    main()
