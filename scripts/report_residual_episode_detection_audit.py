#!/usr/bin/env python3
"""Consolidate residual-episode coverage, taxonomy, and detector evidence.

This is an audit of *event blocks*, not a trade-level quality report.  It
keeps two meanings of recognition separate:

* ``legacy_calendar_status``: whether the earlier composite calendar found a
  state for a calendar cell;
* ``mechanism_status``: whether the new causal taxonomy has a confident,
  observable onset description.

The latter is descriptive.  A mechanism becomes a detector candidate only
when its chronological detector has repeatable OOS lift and false-alert rate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


KEYS = ["day", "side_name", "archetype_policy_key"]
EVENT_KEYS = ["event_start", "side_name", "archetype_policy_key", "event_block"]


def _event_days(events: pd.DataFrame) -> pd.DataFrame:
    """Expand a compact event-block table to calendar days without outcomes."""

    rows: list[dict[str, object]] = []
    for event in events.itertuples(index=False):
        start = pd.Timestamp(event.event_start)
        end = pd.Timestamp(event.event_end)
        for day in pd.date_range(start, end, freq="D", tz="UTC"):
            rows.append(
                {
                    "day": day,
                    "event_start": start,
                    "side_name": event.side_name,
                    "archetype_policy_key": event.archetype_policy_key,
                    "event_block": event.event_block,
                }
            )
    return pd.DataFrame(rows)


def _detector_metrics(path: Path, phase: str) -> pd.DataFrame:
    summary = pd.read_csv(path)
    if summary.empty:
        return summary
    columns = [
        "side_name", "archetype_policy_key", "family", "folds",
        "top05_mean_lift", "top05_mean_fpr", "top05_mean_recall",
        "top05_hit_folds", "passes_top05_repetition_gate",
    ]
    result = summary.loc[:, [name for name in columns if name in summary]].copy()
    result["detector_phase"] = phase
    return result


def _cnn_early_warning(
    events: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    horizon_days: int,
    fraction: float = 0.05,
) -> pd.DataFrame:
    """Mark OOS events warned in their preceding causal top-five-percent days."""

    local = predictions.loc[predictions["model"].eq("causal_cnn")].copy()
    if local.empty:
        return pd.DataFrame(columns=[*EVENT_KEYS, "cnn_top05_early_warning"])
    local["day"] = pd.to_datetime(local["day"], utc=True).dt.floor("D")
    local["fold_start"] = pd.to_datetime(local["fold_start"], utc=True)
    local["cnn_top05_alert"] = False
    for _, index in local.groupby(
        ["fold_start", "side_name", "archetype_policy_key"], observed=True
    ).groups.items():
        count = max(1, int(np.ceil(len(index) * fraction)))
        rank = local.loc[index, "risk"].rank(method="first", ascending=False)
        local.loc[index, "cnn_top05_alert"] = rank.le(count).to_numpy(bool)
    rows: list[dict[str, object]] = []
    for event in events.itertuples(index=False):
        start = pd.Timestamp(event.event_start)
        candidates = local.loc[
            local["side_name"].eq(event.side_name)
            & local["archetype_policy_key"].eq(event.archetype_policy_key)
            & local["day"].ge(start - pd.Timedelta(days=horizon_days))
            & local["day"].lt(start)
        ]
        # A future event must be inside the same prediction fold.  We retain
        # no alert when there was no eligible OOS inference day.
        rows.append(
            {
                "event_start": start,
                "side_name": event.side_name,
                "archetype_policy_key": event.archetype_policy_key,
                "event_block": event.event_block,
                "cnn_top05_early_warning": bool(candidates["cnn_top05_alert"].any()),
                "cnn_oos_eligible": bool(len(candidates)),
            }
        )
    return pd.DataFrame(rows)


def build_audit(
    calendar: pd.DataFrame,
    events: pd.DataFrame,
    detector_summaries: list[tuple[str, pd.DataFrame]],
    cnn_warning: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build event, archetype, and detector tables from frozen reports."""

    calendar = calendar.copy()
    calendar["day"] = pd.to_datetime(calendar["day"], utc=True).dt.floor("D")
    calendar["recognized"] = calendar["recognized"].fillna(False).astype(bool)
    events = events.copy()
    for name in ("event_start", "event_end"):
        events[name] = pd.to_datetime(events[name], utc=True).dt.floor("D")
    days = _event_days(events)
    cells = days.merge(
        calendar.loc[:, [*KEYS, "recognized", "status", "matching_composites", "best_composite_score"]],
        on=KEYS,
        how="left",
        validate="one_to_one",
    )
    cells["recognized"] = cells["recognized"].fillna(False).astype(bool)
    event_metrics = (
        cells.groupby(EVENT_KEYS, observed=True, as_index=False)
        .agg(
            event_days=("day", "size"),
            legacy_recognized_cells=("recognized", "sum"),
            legacy_recognition_rate=("recognized", "mean"),
            matching_composites=(
                "matching_composites",
                lambda values: "|".join(sorted({str(value) for value in values.dropna() if str(value)})),
            ),
            max_composite_score=("best_composite_score", "max"),
        )
    )
    event_metrics = event_metrics.merge(
        events.drop(columns=["event_days"], errors="ignore"),
        on=EVENT_KEYS,
        how="left",
        validate="one_to_one",
    )
    event_metrics["legacy_calendar_status"] = np.select(
        [
            event_metrics["legacy_recognized_cells"].eq(0),
            event_metrics["legacy_recognition_rate"].eq(1.0),
        ],
        ["not_recognized", "fully_recognized"],
        default="partially_recognized",
    )
    event_metrics["mechanism_status"] = np.where(
        event_metrics["onset_mechanism_confident"].fillna(False),
        "confident_observable_mechanism",
        "ambiguous_observable_mechanism",
    )
    if cnn_warning is not None and not cnn_warning.empty:
        event_metrics = event_metrics.merge(
            cnn_warning, on=EVENT_KEYS, how="left", validate="one_to_one"
        )
        event_metrics["cnn_top05_early_warning"] = event_metrics["cnn_top05_early_warning"].fillna(False).astype(bool)
        event_metrics["cnn_oos_eligible"] = event_metrics["cnn_oos_eligible"].fillna(False).astype(bool)
    event_metrics = event_metrics.sort_values(EVENT_KEYS, kind="stable")

    aggregation: dict[str, tuple[str, object]] = {
        "event_blocks": ("event_block", "size"),
        "event_days": ("event_days", "sum"),
        "fully_recognized_blocks": (
            "legacy_calendar_status", lambda values: int((values == "fully_recognized").sum()),
        ),
        "partially_recognized_blocks": (
            "legacy_calendar_status", lambda values: int((values == "partially_recognized").sum()),
        ),
        "unexplained_blocks": (
            "legacy_calendar_status", lambda values: int((values == "not_recognized").sum()),
        ),
        "confident_mechanism_blocks": ("onset_mechanism_confident", "sum"),
        "mean_event_ev": ("calendar_mean_ev", "mean"),
        "mean_event_persistence": ("calendar_persistence_strength", "mean"),
    }
    if "cnn_top05_early_warning" in event_metrics:
        aggregation["cnn_top05_early_warning_blocks"] = ("cnn_top05_early_warning", "sum")
        aggregation["cnn_oos_eligible_blocks"] = ("cnn_oos_eligible", "sum")
    archetype = (
        event_metrics.groupby(["side_name", "archetype_policy_key"], observed=True, as_index=False)
        .agg(**aggregation)
    )
    archetype["legacy_block_coverage"] = (
        (archetype["fully_recognized_blocks"] + 0.5 * archetype["partially_recognized_blocks"])
        / archetype["event_blocks"].clip(lower=1)
    )
    archetype = archetype.sort_values(
        ["unexplained_blocks", "event_blocks"], ascending=[False, False], kind="stable"
    )

    detector = pd.concat(
        [frame.assign(detector_phase=phase) for phase, frame in detector_summaries if not frame.empty],
        ignore_index=True,
    ) if detector_summaries else pd.DataFrame()
    if not detector.empty:
        detector["detector_status"] = np.select(
            [
                detector["passes_top05_repetition_gate"].fillna(False),
                detector["top05_hit_folds"].fillna(0).ge(2)
                & detector["top05_mean_lift"].fillna(0).ge(1.5)
                & detector["top05_mean_fpr"].fillna(np.inf).le(0.15),
            ],
            ["repetition_gate_pass", "promising_insufficient_repetition"],
            default="not_replicated",
        )
        detector = detector.sort_values(
            ["detector_status", "top05_mean_lift", "top05_mean_fpr"],
            ascending=[True, False, True],
            kind="stable",
        )
    return event_metrics, archetype, detector


def run(args: argparse.Namespace) -> dict[str, object]:
    args.output.mkdir(parents=True, exist_ok=True)
    calendar = pd.read_csv(args.calendar)
    events = pd.read_csv(args.event_calendar)
    onset = _detector_metrics(args.onset_summary, "onset")
    active = _detector_metrics(args.active_summary, "active")
    cnn_warning = None
    cnn_summary = pd.DataFrame()
    if args.cnn_oof:
        cnn_predictions = pd.read_parquet(args.cnn_oof)
        cnn_warning = _cnn_early_warning(events, cnn_predictions, horizon_days=args.cnn_horizon_days)
    if args.cnn_summary:
        cnn_summary = pd.read_csv(args.cnn_summary)
        if not cnn_summary.empty:
            cnn_summary = cnn_summary.rename(columns={"model": "family"})
            cnn_summary["family"] = "hard_period_" + cnn_summary["family"].astype(str)
            cnn_summary["detector_phase"] = "early_warning"
            cnn_summary["detector_status"] = np.where(
                cnn_summary["passes_top05_repetition_gate"].fillna(False),
                "repetition_gate_pass", "not_replicated",
            )
            detector_columns = [
                "detector_phase", "side_name", "archetype_policy_key", "family", "folds",
                "top05_mean_lift", "top05_mean_fpr", "top05_mean_event_recall", "top05_hit_folds",
                "passes_top05_repetition_gate", "detector_status",
            ]
            cnn_summary = cnn_summary.loc[:, [name for name in detector_columns if name in cnn_summary]].rename(
                columns={"top05_mean_event_recall": "top05_mean_recall"}
            )
    event_metrics, archetype, detector = build_audit(
        calendar, events, [("onset", onset), ("active", active)], cnn_warning=cnn_warning,
    )
    if not cnn_summary.empty:
        detector = pd.concat([detector, cnn_summary], ignore_index=True, sort=False)
        detector = detector.sort_values(
            ["detector_status", "top05_mean_lift", "top05_mean_fpr"],
            ascending=[True, False, True], kind="stable",
        )
    event_metrics.to_csv(args.output / "episode_recognition_audit.csv", index=False)
    archetype.to_csv(args.output / "archetype_recognition_audit.csv", index=False)
    detector.to_csv(args.output / "mechanism_detector_audit.csv", index=False)
    summary = {
        "event_blocks": int(len(event_metrics)),
        "event_days": int(event_metrics["event_days"].sum()),
        "fully_recognized_blocks": int(event_metrics["legacy_calendar_status"].eq("fully_recognized").sum()),
        "partially_recognized_blocks": int(event_metrics["legacy_calendar_status"].eq("partially_recognized").sum()),
        "unrecognized_blocks": int(event_metrics["legacy_calendar_status"].eq("not_recognized").sum()),
        "confident_observable_mechanism_blocks": int(event_metrics["onset_mechanism_confident"].sum()),
        "detector_candidates": int(len(detector)),
        "detector_repetition_gate_passes": int(detector.get("passes_top05_repetition_gate", pd.Series(dtype=bool)).fillna(False).sum()),
    }
    (args.output / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = [
        "# Residual Episode Detection Audit",
        "",
        "This report distinguishes legacy composite-calendar recognition from causal mechanism classification and OOS detector evidence. A mechanism label is not a production action.",
        "",
        "## Coverage",
        "",
        f"- Event blocks: {summary['event_blocks']}; event days: {summary['event_days']}",
        f"- Fully / partially / not legacy-recognized: {summary['fully_recognized_blocks']} / {summary['partially_recognized_blocks']} / {summary['unrecognized_blocks']}",
        f"- Confident observable onset mechanisms: {summary['confident_observable_mechanism_blocks']}",
        f"- Detector candidates passing the strict three-fold top-5 gate: {summary['detector_repetition_gate_passes']}",
        "",
        "## Current Detector Candidates",
        "",
    ]
    if detector.empty:
        lines.append("No detector summaries were available.")
    else:
        display = detector.assign(
            _status_rank=detector["detector_status"].map(
                {"repetition_gate_pass": 0, "promising_insufficient_repetition": 1, "not_replicated": 2}
            ).fillna(3)
        ).sort_values(
            ["_status_rank", "top05_mean_lift", "top05_mean_fpr"],
            ascending=[True, False, True], kind="stable",
        )
        for row in display.head(12).itertuples(index=False):
            lines.append(
                f"- `{row.detector_phase}` {row.side_name} / {row.archetype_policy_key} / {row.family}: "
                f"top-5 lift {row.top05_mean_lift:.2f}x, FPR {100 * row.top05_mean_fpr:.2f}%, "
                f"recall {100 * row.top05_mean_recall:.1f}%, hit folds {row.top05_hit_folds}/{row.folds}; {row.detector_status}."
            )
    (args.output / "README.md").write_text("\n".join(lines) + "\n")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calendar", type=Path, required=True)
    parser.add_argument("--event-calendar", type=Path, required=True)
    parser.add_argument("--onset-summary", type=Path, required=True)
    parser.add_argument("--active-summary", type=Path, required=True)
    parser.add_argument("--cnn-oof", type=Path)
    parser.add_argument("--cnn-summary", type=Path)
    parser.add_argument("--cnn-horizon-days", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2))
