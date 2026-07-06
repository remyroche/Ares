#!/usr/bin/env python3
"""Summarize gated no-training tail-label proxy ledgers.

This is a reporting helper only. It combines aggregate/monthly CSVs from
`export_label_proxy_gated_candidate_ledger.py` runs so label candidates can be
compared under the same Apr-May-Jun OOS proxy protocol.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_GRID_ROOT = Path("data_perp/reports/label_proxy_gated_tail_label_grid_v1")
DEFAULT_REFERENCE_DIRS = (
    Path("data_perp/reports/label_proxy_gated_candidate_ledger_s14_w12_v1"),
)


def _read_manifest(report_dir: Path) -> dict[str, Any]:
    path = report_dir / "manifest.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_report(report_dir: Path, source: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest = _read_manifest(report_dir)
    label_arm = str(manifest.get("label_arm") or report_dir.name.split("__", 1)[0])
    weight_arm = str(manifest.get("weight_arm") or report_dir.name.split("__", 1)[-1])
    aggregate = pd.read_csv(report_dir / "aggregate_summary.csv")
    monthly = pd.read_csv(report_dir / "monthly_summary.csv")
    for frame in (aggregate, monthly):
        frame["label_arm"] = label_arm
        frame["weight_arm"] = weight_arm
        frame["source"] = source
        frame["report_dir"] = str(report_dir)
    return aggregate, monthly


def _candidate_reports(grid_root: Path, reference_dirs: tuple[Path, ...]) -> list[tuple[Path, str]]:
    reports: list[tuple[Path, str]] = []
    for path in reference_dirs:
        if (path / "aggregate_summary.csv").exists():
            reports.append((path, "reference"))
    if grid_root.exists():
        for path in sorted(grid_root.iterdir()):
            if path.is_dir() and (path / "aggregate_summary.csv").exists():
                reports.append((path, "tail_grid"))
    return reports


def _fmt(value: Any) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _table(frame: pd.DataFrame, cols: list[str], limit: int | None = None) -> str:
    if frame.empty:
        return "No rows."
    view = frame[[col for col in cols if col in frame.columns]].copy()
    if limit is not None:
        view = view.head(limit)
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(_fmt)
    return view.to_markdown(index=False)


def summarize(
    *,
    grid_root: Path,
    reference_dirs: tuple[Path, ...],
    output_dir: Path,
) -> dict[str, str]:
    aggregate_frames: list[pd.DataFrame] = []
    monthly_frames: list[pd.DataFrame] = []
    for report_dir, source in _candidate_reports(grid_root, reference_dirs):
        aggregate, monthly = _load_report(report_dir, source)
        aggregate_frames.append(aggregate)
        monthly_frames.append(monthly)

    if not aggregate_frames:
        raise FileNotFoundError(f"No aggregate reports found under {grid_root}")

    aggregate = pd.concat(aggregate_frames, ignore_index=True)
    monthly = pd.concat(monthly_frames, ignore_index=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    aggregate_path = output_dir / "combined_aggregate_summary.csv"
    monthly_path = output_dir / "combined_monthly_summary.csv"
    markdown_path = output_dir / "label_proxy_gated_tail_grid_summary.md"
    aggregate.to_csv(aggregate_path, index=False)
    monthly.to_csv(monthly_path, index=False)

    cols = [
        "label_arm",
        "weight_arm",
        "risk_kind",
        "risk_keep_frac",
        "top_frac",
        "positive_months",
        "mean_u",
        "worst_month_mean_u",
        "selected_weeks",
        "positive_selected_weeks",
        "q25_week_mean_u",
        "worst_week_mean_u",
        "hit_u",
        "q10_u",
        "bad_mae_1r_rate",
        "wide_barrier_25bps_rate",
        "mean_selected_rows_month",
    ]
    best_mean = aggregate.sort_values(
        ["mean_u", "worst_month_mean_u", "q25_week_mean_u"],
        ascending=[False, False, False],
    )
    best_worst_month = aggregate.sort_values(
        ["worst_month_mean_u", "mean_u", "q25_week_mean_u"],
        ascending=[False, False, False],
    )
    positive_mean = aggregate[pd.to_numeric(aggregate["mean_u"], errors="coerce") > 0.0].copy()
    best_week_q25 = positive_mean.sort_values(
        ["q25_week_mean_u", "worst_month_mean_u", "mean_u"],
        ascending=[False, False, False],
    )
    all_month_positive = aggregate[
        pd.to_numeric(aggregate["positive_months"], errors="coerce") >= 3
    ].sort_values(["mean_u", "q25_week_mean_u"], ascending=[False, False])

    best_by_pair = (
        aggregate.sort_values(
            ["label_arm", "weight_arm", "mean_u", "worst_month_mean_u", "q25_week_mean_u"],
            ascending=[True, True, False, False, False],
        )
        .groupby(["label_arm", "weight_arm"], dropna=False, observed=True)
        .head(1)
        .sort_values(["mean_u", "worst_month_mean_u"], ascending=[False, False])
    )

    monthly_cols = [
        "label_arm",
        "weight_arm",
        "risk_kind",
        "risk_keep_frac",
        "top_frac",
        "period",
        "selected_rows",
        "mean_u",
        "delta_mean_u_vs_period",
        "hit_u",
        "q10_u",
        "bad_mae_1r_rate",
        "wide_barrier_25bps_rate",
    ]
    monthly_focus_keys = best_worst_month.head(5)[
        ["label_arm", "weight_arm", "risk_kind", "risk_keep_frac", "top_frac"]
    ].drop_duplicates()
    monthly_focus = monthly.merge(
        monthly_focus_keys,
        on=["label_arm", "weight_arm", "risk_kind", "risk_keep_frac", "top_frac"],
        how="inner",
    ).sort_values(["label_arm", "weight_arm", "risk_kind", "risk_keep_frac", "top_frac", "period"])

    lines = [
        "# Gated Tail-Label Proxy Grid Summary",
        "",
        "Scope: no model training. Each month is scored by feature/risk proxies learned only from prior months.",
        "",
        "## Best By Mean Utility",
        "",
        _table(best_mean, cols, limit=20),
        "",
        "## Best By Worst Month",
        "",
        _table(best_worst_month, cols, limit=20),
        "",
        "## Best Weekly Lower Tail With Positive Mean",
        "",
        _table(best_week_q25, cols, limit=20),
        "",
        "## All-Month-Positive Candidates",
        "",
        _table(all_month_positive, cols, limit=20),
        "",
        "## Best Per Label/Weight Pair",
        "",
        _table(best_by_pair, cols, limit=20),
        "",
        "## Monthly Breakdown For Best Worst-Month Candidates",
        "",
        _table(monthly_focus, monthly_cols, limit=None),
        "",
        "## Outputs",
        "",
        f"- Combined aggregate: `{aggregate_path}`",
        f"- Combined monthly: `{monthly_path}`",
    ]
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "aggregate": str(aggregate_path),
        "monthly": str(monthly_path),
        "markdown": str(markdown_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-root", type=Path, default=DEFAULT_GRID_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_GRID_ROOT)
    parser.add_argument(
        "--reference-dir",
        type=Path,
        action="append",
        default=list(DEFAULT_REFERENCE_DIRS),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    outputs = summarize(
        grid_root=args.grid_root,
        reference_dirs=tuple(args.reference_dir),
        output_dir=args.output_dir,
    )
    print(json.dumps(outputs, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
