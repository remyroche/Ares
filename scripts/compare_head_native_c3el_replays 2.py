#!/usr/bin/env python3
"""Compare head-native/C3el replay artifacts across existing run folders.

The C3el/head-native experiments use a lightweight replay schema with
`overall.csv`, `weekly.csv`, and optional per-head files. This script normalizes
those outputs and ranks candidate arms against their run-local C0 baseline.
It is intentionally read-only over existing artifacts; it does not rerun the
portfolio engine or rebuild action panels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _objective(values: pd.Series, q35_weight: float, q20_weight: float) -> float:
    arr = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    return float(np.mean(arr) + q35_weight * np.quantile(arr, 0.35) + q20_weight * np.quantile(arr, 0.20))


def _period_stats(weekly: pd.DataFrame, arm: str, q35_weight: float, q20_weight: float) -> dict[str, Any]:
    rows = weekly.loc[weekly["arm"].astype(str).eq(str(arm))].copy()
    if rows.empty:
        return {}
    rows["week_start"] = pd.to_datetime(rows["week_start"], utc=True, errors="coerce")
    rows["net_pnl"] = pd.to_numeric(rows["net_pnl"], errors="coerce")
    rows["trade_count"] = pd.to_numeric(rows.get("trade_count"), errors="coerce").fillna(0)
    pnl = rows["net_pnl"].replace([np.inf, -np.inf], np.nan).dropna()
    if pnl.empty:
        return {}
    return {
        "weeks": int(rows["week_start"].nunique()),
        "period_start": str(rows["week_start"].min()),
        "period_end": str(rows["week_start"].max()),
        "trade_count": int(rows["trade_count"].sum()),
        "net_pnl_sum": float(pnl.sum()),
        "avg_week_net_pnl": float(pnl.mean()),
        "q20_week_net_pnl": float(pnl.quantile(0.20)),
        "q35_week_net_pnl": float(pnl.quantile(0.35)),
        "worst_week_net_pnl": float(pnl.min()),
        "positive_weeks": int((pnl > 0).sum()),
        "objective": _objective(pnl, q35_weight, q20_weight),
    }


def _window_stats(weekly: pd.DataFrame, arm: str, start: str | None, q35_weight: float, q20_weight: float, prefix: str) -> dict[str, Any]:
    rows = weekly.copy()
    rows["week_start"] = pd.to_datetime(rows["week_start"], utc=True, errors="coerce")
    rows = rows.loc[rows["arm"].astype(str).eq(str(arm))].copy()
    if start is not None:
        rows = rows.loc[rows["week_start"].ge(pd.Timestamp(start, tz="UTC"))].copy()
    if rows.empty:
        return {
            f"{prefix}_weeks": 0,
            f"{prefix}_net_pnl_sum": np.nan,
            f"{prefix}_objective": np.nan,
            f"{prefix}_worst_week_net_pnl": np.nan,
        }
    stats = _period_stats(rows, arm, q35_weight, q20_weight)
    return {
        f"{prefix}_weeks": stats.get("weeks", 0),
        f"{prefix}_net_pnl_sum": stats.get("net_pnl_sum", np.nan),
        f"{prefix}_objective": stats.get("objective", np.nan),
        f"{prefix}_worst_week_net_pnl": stats.get("worst_week_net_pnl", np.nan),
    }


def _manifest(path: Path) -> dict[str, Any]:
    manifest_path = path / "manifest.json"
    if not manifest_path.exists():
        return {}
    try:
        return json.loads(manifest_path.read_text())
    except Exception:
        return {}


def _coverage_bucket(weeks: int) -> str:
    if weeks >= 16:
        return "long_16w_plus"
    if weeks >= 8:
        return "medium_8w_plus"
    if weeks >= 4:
        return "short_4w_plus"
    return "very_short"


def _summarize_run(path: Path, q35_weight: float, q20_weight: float) -> list[dict[str, Any]]:
    weekly_path = path / "weekly.csv"
    if not weekly_path.exists() or not (path / "overall.csv").exists():
        return []
    weekly = pd.read_csv(weekly_path)
    if "arm" not in weekly.columns or "net_pnl" not in weekly.columns or "week_start" not in weekly.columns:
        return []
    manifest = _manifest(path)
    arms = [str(a) for a in weekly["arm"].dropna().unique()]
    if "C0_baseline" not in arms:
        return []
    baseline = _period_stats(weekly, "C0_baseline", q35_weight, q20_weight)
    if not baseline:
        return []
    rows: list[dict[str, Any]] = []
    max_week = pd.to_datetime(weekly["week_start"], utc=True, errors="coerce").max()
    last4_start = (max_week - pd.Timedelta(weeks=3)).normalize() if pd.notna(max_week) else None
    for arm in arms:
        if arm == "C0_baseline":
            continue
        stats = _period_stats(weekly, arm, q35_weight, q20_weight)
        if not stats:
            continue
        row = {
            "run_dir": path.name,
            "arm": arm,
            "generated_by": manifest.get("generated_by", ""),
            "policy_variant": manifest.get("policy_variant", ""),
            "manifest_start": manifest.get("start", ""),
            "manifest_end": manifest.get("end", ""),
            "active_heads": ",".join(map(str, manifest.get("active_heads", []))) if isinstance(manifest.get("active_heads"), list) else "",
            **stats,
            "coverage_bucket": _coverage_bucket(int(stats["weeks"])),
            "baseline_net_pnl_sum": baseline["net_pnl_sum"],
            "baseline_objective": baseline["objective"],
            "baseline_worst_week_net_pnl": baseline["worst_week_net_pnl"],
            "delta_net_pnl_sum": stats["net_pnl_sum"] - baseline["net_pnl_sum"],
            "delta_objective": stats["objective"] - baseline["objective"],
            "delta_worst_week_net_pnl": stats["worst_week_net_pnl"] - baseline["worst_week_net_pnl"],
        }
        row.update(_window_stats(weekly, arm, "2026-05-01", q35_weight, q20_weight, "may_june"))
        row.update(_window_stats(weekly, "C0_baseline", "2026-05-01", q35_weight, q20_weight, "baseline_may_june"))
        row.update(_window_stats(weekly, arm, "2026-06-01", q35_weight, q20_weight, "june"))
        row.update(_window_stats(weekly, "C0_baseline", "2026-06-01", q35_weight, q20_weight, "baseline_june"))
        if last4_start is not None:
            row.update(_window_stats(weekly, arm, str(last4_start), q35_weight, q20_weight, "last4w"))
            row.update(_window_stats(weekly, "C0_baseline", str(last4_start), q35_weight, q20_weight, "baseline_last4w"))
        for prefix in ("may_june", "june", "last4w"):
            row[f"delta_{prefix}_net_pnl_sum"] = row.get(f"{prefix}_net_pnl_sum", np.nan) - row.get(
                f"baseline_{prefix}_net_pnl_sum", np.nan
            )
            row[f"delta_{prefix}_objective"] = row.get(f"{prefix}_objective", np.nan) - row.get(
                f"baseline_{prefix}_objective", np.nan
            )
            row[f"delta_{prefix}_worst_week_net_pnl"] = row.get(f"{prefix}_worst_week_net_pnl", np.nan) - row.get(
                f"baseline_{prefix}_worst_week_net_pnl", np.nan
            )
        rows.append(row)
    return rows


def _summarize_weekly_artifact(
    run_label: str,
    weekly: pd.DataFrame,
    manifest: dict[str, Any],
    q35_weight: float,
    q20_weight: float,
    *,
    source_windows: str = "",
) -> list[dict[str, Any]]:
    if "arm" not in weekly.columns or "net_pnl" not in weekly.columns or "week_start" not in weekly.columns:
        return []
    arms = [str(a) for a in weekly["arm"].dropna().unique()]
    if "C0_baseline" not in arms:
        return []
    baseline = _period_stats(weekly, "C0_baseline", q35_weight, q20_weight)
    if not baseline:
        return []
    rows: list[dict[str, Any]] = []
    max_week = pd.to_datetime(weekly["week_start"], utc=True, errors="coerce").max()
    last4_start = (max_week - pd.Timedelta(weeks=3)).normalize() if pd.notna(max_week) else None
    for arm in arms:
        if arm == "C0_baseline":
            continue
        stats = _period_stats(weekly, arm, q35_weight, q20_weight)
        if not stats:
            continue
        row = {
            "run_dir": run_label,
            "arm": arm,
            "generated_by": manifest.get("generated_by", ""),
            "policy_variant": manifest.get("policy_variant", ""),
            "manifest_start": manifest.get("start", ""),
            "manifest_end": manifest.get("end", ""),
            "source_windows": source_windows,
            "active_heads": ",".join(map(str, manifest.get("active_heads", []))) if isinstance(manifest.get("active_heads"), list) else "",
            **stats,
            "coverage_bucket": _coverage_bucket(int(stats["weeks"])),
            "baseline_net_pnl_sum": baseline["net_pnl_sum"],
            "baseline_objective": baseline["objective"],
            "baseline_worst_week_net_pnl": baseline["worst_week_net_pnl"],
            "delta_net_pnl_sum": stats["net_pnl_sum"] - baseline["net_pnl_sum"],
            "delta_objective": stats["objective"] - baseline["objective"],
            "delta_worst_week_net_pnl": stats["worst_week_net_pnl"] - baseline["worst_week_net_pnl"],
        }
        row.update(_window_stats(weekly, arm, "2026-05-01", q35_weight, q20_weight, "may_june"))
        row.update(_window_stats(weekly, "C0_baseline", "2026-05-01", q35_weight, q20_weight, "baseline_may_june"))
        row.update(_window_stats(weekly, arm, "2026-06-01", q35_weight, q20_weight, "june"))
        row.update(_window_stats(weekly, "C0_baseline", "2026-06-01", q35_weight, q20_weight, "baseline_june"))
        if last4_start is not None:
            row.update(_window_stats(weekly, arm, str(last4_start), q35_weight, q20_weight, "last4w"))
            row.update(_window_stats(weekly, "C0_baseline", str(last4_start), q35_weight, q20_weight, "baseline_last4w"))
        for prefix in ("may_june", "june", "last4w"):
            row[f"delta_{prefix}_net_pnl_sum"] = row.get(f"{prefix}_net_pnl_sum", np.nan) - row.get(
                f"baseline_{prefix}_net_pnl_sum", np.nan
            )
            row[f"delta_{prefix}_objective"] = row.get(f"{prefix}_objective", np.nan) - row.get(
                f"baseline_{prefix}_objective", np.nan
            )
            row[f"delta_{prefix}_worst_week_net_pnl"] = row.get(f"{prefix}_worst_week_net_pnl", np.nan) - row.get(
                f"baseline_{prefix}_worst_week_net_pnl", np.nan
            )
        rows.append(row)
    return rows


def _summarize_paired_may_last4(root: Path, q35_weight: float, q20_weight: float) -> list[dict[str, Any]]:
    early_prefix = "exact_state_size_action_learning_20260628_may06_may29_c3el_"
    late_prefix = "exact_state_size_action_learning_20260628_last4w_c3el_"
    early = {p.name.removeprefix(early_prefix): p for p in root.glob(f"{early_prefix}*") if (p / "weekly.csv").exists()}
    late = {p.name.removeprefix(late_prefix): p for p in root.glob(f"{late_prefix}*") if (p / "weekly.csv").exists()}
    rows: list[dict[str, Any]] = []
    for suffix in sorted(set(early).intersection(late)):
        early_weekly = pd.read_csv(early[suffix] / "weekly.csv")
        late_weekly = pd.read_csv(late[suffix] / "weekly.csv")
        weekly = pd.concat([early_weekly, late_weekly], ignore_index=True)
        weekly["week_start"] = pd.to_datetime(weekly["week_start"], utc=True, errors="coerce")
        weekly = weekly.sort_values(["arm", "week_start"]).drop_duplicates(["arm", "week_start"], keep="last")
        manifest = _manifest(late[suffix]) or _manifest(early[suffix])
        label = f"combined_may06_jun26_c3el_{suffix}"
        source = f"{early[suffix].name};{late[suffix].name}"
        rows.extend(_summarize_weekly_artifact(label, weekly, manifest, q35_weight, q20_weight, source_windows=source))
    return rows


def _markdown_table(frame: pd.DataFrame, cols: list[str], max_rows: int = 20) -> str:
    if frame.empty:
        return "No rows."
    view = frame[cols].head(max_rows).copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{x:,.3f}")
    return view.to_markdown(index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-root", type=Path, default=Path("data_perp/reports"))
    parser.add_argument("--pattern", action="append", default=["*c3el*"])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--q35-weight", type=float, default=0.70)
    parser.add_argument("--q20-weight", type=float, default=0.30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dirs: set[Path] = set()
    for pattern in args.pattern:
        dirs.update(p for p in args.reports_root.glob(pattern) if p.is_dir())
    rows: list[dict[str, Any]] = []
    for path in sorted(dirs):
        rows.extend(_summarize_run(path, args.q35_weight, args.q20_weight))
    rows.extend(_summarize_paired_may_last4(args.reports_root, args.q35_weight, args.q20_weight))
    summary = pd.DataFrame(rows)
    if summary.empty:
        raise SystemExit("No comparable C3el/head-native replay artifacts found")
    summary = summary.sort_values(["coverage_bucket", "delta_objective", "delta_net_pnl_sum"], ascending=[True, False, False])
    summary.to_csv(args.output_dir / "head_native_c3el_replay_comparison.csv", index=False)

    ranked_all = summary.sort_values(["delta_objective", "delta_net_pnl_sum"], ascending=False)
    ranked_longish = summary.loc[summary["weeks"].ge(4)].sort_values(["delta_objective", "delta_net_pnl_sum"], ascending=False)
    ranked_medium = summary.loc[summary["weeks"].ge(8)].sort_values(["delta_objective", "delta_net_pnl_sum"], ascending=False)
    ranked_non_oracle = summary.loc[
        summary["weeks"].ge(4) & ~summary["arm"].astype(str).str.contains("oracle", case=False, na=False)
    ].sort_values(["delta_objective", "delta_net_pnl_sum"], ascending=False)
    ranked_june = summary.sort_values(["delta_june_objective", "delta_june_net_pnl_sum"], ascending=False)
    ranked_may_june = summary.sort_values(["delta_may_june_objective", "delta_may_june_net_pnl_sum"], ascending=False)
    harmful = summary.sort_values(["delta_objective", "delta_net_pnl_sum"], ascending=True)

    cols = [
        "run_dir",
        "arm",
        "weeks",
        "coverage_bucket",
        "active_heads",
        "source_windows",
        "net_pnl_sum",
        "delta_net_pnl_sum",
        "objective",
        "delta_objective",
        "worst_week_net_pnl",
        "delta_worst_week_net_pnl",
        "delta_may_june_net_pnl_sum",
        "delta_june_net_pnl_sum",
    ]
    report = [
        "# Head-Native C3el Replay Comparison",
        "",
        "Read-only normalization over existing `overall.csv` / `weekly.csv` replay artifacts. Costs are included because the source replay metrics are net of the recorded cost columns.",
        "",
        f"Objective: `avg_week_net_pnl + {args.q35_weight:.2f} * q35_week_net_pnl + {args.q20_weight:.2f} * q20_week_net_pnl`.",
        "",
        "## Top Overall Delta",
        "",
        _markdown_table(ranked_all, cols, 20),
        "",
        "## Top Runs With At Least Four Weeks",
        "",
        _markdown_table(ranked_longish, cols, 20),
        "",
        "## Top Medium-Window Runs",
        "",
        _markdown_table(ranked_medium, cols, 20),
        "",
        "## Top Non-Oracle Runs With At Least Four Weeks",
        "",
        _markdown_table(ranked_non_oracle, cols, 20),
        "",
        "## Top May-June Delta",
        "",
        _markdown_table(ranked_may_june, cols, 20),
        "",
        "## Top June Delta",
        "",
        _markdown_table(ranked_june, cols, 20),
        "",
        "## Most Harmful Overall",
        "",
        _markdown_table(harmful, cols, 20),
        "",
        "## Readout",
        "",
        "- Compare rows primarily within the same period length; `coverage_bucket` is included to avoid over-reading very short runs.",
        "- A positive June delta from a one-week post-June artifact is diagnostic only, not enough for promotion.",
        "- Candidate promotion still requires a replay over a sufficiently long common window.",
    ]
    (args.output_dir / "head_native_c3el_replay_comparison.md").write_text("\n".join(report) + "\n")
    print(args.output_dir / "head_native_c3el_replay_comparison.md")


if __name__ == "__main__":
    main()
