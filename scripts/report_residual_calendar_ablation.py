#!/usr/bin/env python3
"""Report residual-state ablations on matched side x archetype calendar cells."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


KEYS = ["__ts__", "__symbol__", "side_name", "archetype_policy_key"]
OUTCOME_COLUMNS = [
    "ev_after_1pct",
    "clean_exec",
    "dirty_positive",
    "first_touch_bad_mae_1r",
    "full_path_bad_mae_1r",
    "timeout",
]
GROUP_METRICS = {
    "selected_rows": ("selected", "sum"),
    "mean_ev_after_1pct": ("selected_ev", "mean"),
    "sum_ev_after_1pct": ("selected_ev", "sum"),
    "positive_ev_rate": ("selected_positive_ev", "mean"),
    "clean_exec_precision": ("selected_clean_exec", "mean"),
    "dirty_positive_rate": ("selected_dirty_positive", "mean"),
    "first_touch_bad_mae_rate": ("selected_first_touch_bad_mae_1r", "mean"),
    "full_path_bad_mae_rate": ("selected_full_path_bad_mae_1r", "mean"),
    "timeout_rate": ("selected_timeout", "mean"),
}


def _parse_arm(value: str) -> tuple[str, Path, str]:
    parts = value.split("=", 2)
    if len(parts) != 3 or not all(parts):
        raise argparse.ArgumentTypeError("arm must be NAME=PARQUET=SELECTION_COLUMN")
    return parts[0], Path(parts[1]), parts[2]


def _load_arm(name: str, path: Path, selection_column: str) -> pd.DataFrame:
    columns = [*KEYS, selection_column]
    frame = pd.read_parquet(path, columns=columns)
    frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True, errors="raise")
    frame = frame.drop_duplicates(KEYS, keep="last")
    return frame.rename(columns={selection_column: f"selected__{name}"})


def _event_sign(calendar: pd.DataFrame) -> pd.Series:
    adverse = pd.to_numeric(calendar["adverse_event_rows"], errors="coerce").gt(0)
    favorable = pd.to_numeric(calendar["favorable_event_rows"], errors="coerce").gt(0)
    return pd.Series(
        np.select([adverse, favorable], ["adverse", "favorable"], default="other"),
        index=calendar.index,
        dtype="string",
    )


def _aggregate(selected: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    work = selected.copy()
    work["selected_ev"] = work["ev_after_1pct"].where(work["selected"])
    work["selected_positive_ev"] = work["ev_after_1pct"].gt(0).where(work["selected"])
    for column in OUTCOME_COLUMNS[1:]:
        work[f"selected_{column}"] = work[column].where(work["selected"])
    result = work.groupby(groups, observed=True, dropna=False).agg(**GROUP_METRICS)
    return result.reset_index()


def _add_deltas(frame: pd.DataFrame, baseline: str, keys: list[str]) -> pd.DataFrame:
    metrics = list(GROUP_METRICS)
    reference = frame.loc[frame["arm"].eq(baseline), [*keys, *metrics]].rename(
        columns={metric: f"baseline__{metric}" for metric in metrics}
    )
    result = frame.merge(reference, on=keys, how="left", validate="many_to_one")
    for metric in metrics:
        result[f"delta_vs_baseline__{metric}"] = (
            result[metric] - result[f"baseline__{metric}"]
        )
    return result


def run(args: argparse.Namespace) -> dict[str, object]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outcomes = pd.read_parquet(args.outcomes, columns=[*KEYS, *OUTCOME_COLUMNS])
    outcomes["__ts__"] = pd.to_datetime(outcomes["__ts__"], utc=True, errors="raise")
    outcomes = outcomes.drop_duplicates(KEYS, keep="last")
    outcomes["day"] = outcomes["__ts__"].dt.floor("D")

    arms: list[str] = []
    scored = outcomes
    for name, path, selection_column in args.arm:
        if name in arms:
            raise ValueError(f"Duplicate arm name: {name}")
        arms.append(name)
        scored = scored.merge(
            _load_arm(name, path, selection_column),
            on=KEYS,
            how="inner",
            validate="one_to_one",
        )
    if args.baseline not in arms:
        raise ValueError(f"Baseline {args.baseline!r} is not one of {arms}")

    calendar = pd.read_csv(args.calendar)
    calendar["day"] = pd.to_datetime(calendar["day"], utc=True, errors="raise").dt.floor("D")
    calendar["event_sign"] = _event_sign(calendar)
    calendar_keys = ["day", "side_name", "archetype_policy_key"]
    calendar = calendar.drop_duplicates(calendar_keys, keep="last")
    matched = scored.merge(
        calendar[[*calendar_keys, "event_sign", "material_extreme"]],
        on=calendar_keys,
        how="inner",
        validate="many_to_one",
    )

    long_parts: list[pd.DataFrame] = []
    for arm in arms:
        part = matched[[
            *KEYS,
            "day",
            "event_sign",
            "material_extreme",
            *OUTCOME_COLUMNS,
            f"selected__{arm}",
        ]].copy()
        part["arm"] = arm
        part["selected"] = part.pop(f"selected__{arm}").fillna(False).astype(bool)
        long_parts.append(part)
    long = pd.concat(long_parts, ignore_index=True, copy=False)

    definitions = {
        "calendar_cell": ["arm", "day", "event_sign", "side_name", "archetype_policy_key"],
        "calendar_date": ["arm", "day", "event_sign"],
        "side_archetype": ["arm", "event_sign", "side_name", "archetype_policy_key"],
        "side": ["arm", "event_sign", "side_name"],
        "overall": ["arm", "event_sign"],
    }
    reports: dict[str, pd.DataFrame] = {}
    for name, groups in definitions.items():
        report = _aggregate(long, groups)
        delta_keys = [column for column in groups if column != "arm"]
        report = _add_deltas(report, args.baseline, delta_keys)
        report.to_csv(args.output_dir / f"{name}_metrics.csv", index=False)
        reports[name] = report

    cell = reports["calendar_cell"]
    compare = cell.loc[cell["arm"].ne(args.baseline)].copy()
    both_active = compare["selected_rows"].gt(0) & compare["baseline__selected_rows"].gt(0)
    compare["ev_improved"] = both_active & compare["delta_vs_baseline__mean_ev_after_1pct"].gt(0)
    compare["ev_worsened"] = both_active & compare["delta_vs_baseline__mean_ev_after_1pct"].lt(0)
    compare["selection_changed"] = compare["delta_vs_baseline__selected_rows"].ne(0)
    compare.to_csv(args.output_dir / "calendar_cell_deltas_vs_baseline.csv", index=False)

    coverage_start = scored["__ts__"].min()
    coverage_end = scored["__ts__"].max()
    calendar_in_coverage = calendar.loc[calendar["day"].between(coverage_start.floor("D"), coverage_end.floor("D"))]
    summary = (
        compare.groupby(["arm", "event_sign"], observed=True, dropna=False)
        .agg(
            calendar_cells=("day", "size"),
            calendar_dates=("day", "nunique"),
            changed_cells=("selection_changed", "sum"),
            improved_cells=("ev_improved", "sum"),
            worsened_cells=("ev_worsened", "sum"),
            mean_cell_ev_delta=("delta_vs_baseline__mean_ev_after_1pct", "mean"),
            total_ev_delta=("delta_vs_baseline__sum_ev_after_1pct", "sum"),
            selected_rows_delta=("delta_vs_baseline__selected_rows", "sum"),
        )
        .reset_index()
    )
    summary.to_csv(args.output_dir / "ablation_summary.csv", index=False)
    manifest = {
        "schema": "residual_calendar_ablation_report_v1",
        "baseline": args.baseline,
        "arms": arms,
        "cost_contract": "ev_after_1pct includes the 1% round-trip cost exactly once",
        "outcome_rows": int(len(outcomes)),
        "matched_scored_rows": int(len(scored)),
        "matched_calendar_rows": int(len(matched)),
        "calendar_cells_total": int(len(calendar)),
        "calendar_cells_in_scored_date_range": int(len(calendar_in_coverage)),
        "calendar_cells_matched": int(len(matched[calendar_keys].drop_duplicates())),
        "calendar_dates_matched": int(matched["day"].nunique()),
        "evaluation_start": str(coverage_start),
        "evaluation_end": str(coverage_end),
        "comparison_contract": (
            "All arms use identical outcome rows and calendar cells. Selection masks "
            "are frozen before reporting; no calendar outcome enters arm scoring."
        ),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outcomes", type=Path, required=True)
    parser.add_argument("--calendar", type=Path, required=True)
    parser.add_argument("--arm", type=_parse_arm, action="append", required=True)
    parser.add_argument("--baseline", default="champion")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
