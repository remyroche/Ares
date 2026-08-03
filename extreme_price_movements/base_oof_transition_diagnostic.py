"""Diagnostic-only transition attribution for frozen base-model OOF scores.

The event calendar is joined only after the accepted base OOF has been fixed.
It is therefore suitable for descriptive health/active-risk evidence, but can
never select model rows, train a model, or alter a trading route.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd


IDENTITY = ("candidate_id", "side_name", "__symbol__", "__ts__")
REQUIRED_SCORES = {
    *IDENTITY,
    "base_oof_score",
    "__first_touch_target_soft__",
    "execution_net_ev_12h",
}
REQUIRED_WINDOWS = {
    "transition_event_id",
    "transition_window_start_utc",
    "transition_window_end_utc",
    "transition_active_hours",
}
REQUIRED_ACTIVE = {"source_utc", "target__event_id", "target__transition_active"}


def _spearman(frame: pd.DataFrame, right: str) -> float:
    values = frame.loc[:, ["base_oof_score", right]].apply(pd.to_numeric, errors="coerce").dropna()
    return float(values.corr(method="spearman").iloc[0, 1]) if len(values) >= 2 else float("nan")


def _validate_unique(frame: pd.DataFrame, required: set[str], *, name: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{name} lacks required columns: {missing}")


def prepare_scores(scores: pd.DataFrame) -> pd.DataFrame:
    """Make deterministic timestamp-side top-40% membership from frozen scores."""

    _validate_unique(scores, REQUIRED_SCORES, name="base OOF")
    if scores.duplicated(list(IDENTITY)).any():
        raise ValueError("base OOF contains duplicate candidate identities")
    work = scores.copy()
    work["__ts__"] = pd.to_datetime(work["__ts__"], utc=True, errors="raise")
    work["side_name"] = work["side_name"].astype(str).str.lower()
    if not work["side_name"].isin(("long", "short")).all():
        raise ValueError("base OOF contains non-canonical side names")
    if work["base_oof_score"].isna().any():
        raise ValueError("base OOF score coverage is incomplete")
    ordered = work.sort_values(
        ["__ts__", "side_name", "base_oof_score", "candidate_id"],
        ascending=[True, True, False, True],
        kind="stable",
    ).copy()
    group = ordered.groupby(["__ts__", "side_name"], sort=False, observed=True)
    ordered["base_rank_timestamp_side"] = group.cumcount() + 1
    ordered["base_candidate_group_rows"] = group["candidate_id"].transform("size")
    ordered["base_selected_top40_timestamp_side"] = ordered["base_rank_timestamp_side"] <= np.ceil(
        0.40 * ordered["base_candidate_group_rows"]
    )
    return ordered


def _event_phase_mask(scores: pd.DataFrame, *, start: pd.Timestamp, end: pd.Timestamp, phase: str) -> pd.Series:
    # The event source uses inclusive hourly endpoints.  Before and after are
    # matching 24-hour, non-overlapping source-time blocks.
    if phase == "before_24h":
        return scores["__ts__"].ge(start - pd.Timedelta(hours=24)) & scores["__ts__"].lt(start)
    if phase == "during_window":
        return scores["__ts__"].ge(start) & scores["__ts__"].le(end)
    if phase == "after_24h":
        return scores["__ts__"].gt(end) & scores["__ts__"].le(end + pd.Timedelta(hours=24))
    raise ValueError(f"unknown phase: {phase}")


def _metrics(frame: pd.DataFrame) -> dict[str, Any]:
    selected = frame.loc[frame["base_selected_top40_timestamp_side"].astype(bool)]
    return {
        "candidate_rows": int(len(frame)),
        "candidate_hours": int(frame["__ts__"].nunique()),
        "base_score_mean": float(frame["base_oof_score"].mean()),
        "target_soft_mean": float(frame["__first_touch_target_soft__"].mean()),
        "execution_net_ev_12h_mean": float(frame["execution_net_ev_12h"].mean()),
        "base_score_target_spearman": _spearman(frame, "__first_touch_target_soft__"),
        "base_score_execution_ev_spearman": _spearman(frame, "execution_net_ev_12h"),
        "top40_rows": int(len(selected)),
        "top40_fraction": float(len(selected) / len(frame)) if len(frame) else float("nan"),
        "top40_execution_net_ev_12h_mean": float(selected["execution_net_ev_12h"].mean()) if len(selected) else float("nan"),
    }


def build_transition_diagnostic(
    scores: pd.DataFrame,
    windows: pd.DataFrame,
    active_hours: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Return event coverage, phase/side metrics and a research-only summary."""

    base = prepare_scores(scores)
    _validate_unique(windows, REQUIRED_WINDOWS, name="transition windows")
    _validate_unique(active_hours, REQUIRED_ACTIVE, name="active-transition ledger")
    event_windows = windows.loc[:, sorted(REQUIRED_WINDOWS)].copy()
    if event_windows["transition_event_id"].duplicated().any():
        raise ValueError("transition windows must have one row per event")
    for column in ("transition_window_start_utc", "transition_window_end_utc"):
        event_windows[column] = pd.to_datetime(event_windows[column], utc=True, errors="raise")
    if (event_windows["transition_window_end_utc"] < event_windows["transition_window_start_utc"]).any():
        raise ValueError("transition window ends before it starts")
    active = active_hours.loc[:, sorted(REQUIRED_ACTIVE)].copy()
    active["source_utc"] = pd.to_datetime(active["source_utc"], utc=True, errors="raise")
    if active["source_utc"].duplicated().any():
        raise ValueError("active-transition ledger requires one row per source hour")
    active["target__transition_active"] = active["target__transition_active"].fillna(0).astype(bool)

    coverage_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    for event in event_windows.sort_values("transition_event_id", kind="stable").itertuples(index=False):
        event_id = str(event.transition_event_id)
        start = pd.Timestamp(event.transition_window_start_utc)
        end = pd.Timestamp(event.transition_window_end_utc)
        event_active = active.loc[active["target__event_id"].astype(str).eq(event_id)].copy()
        event_active = event_active.loc[event_active["target__transition_active"]]
        if len(event_active) != int(event.transition_active_hours):
            raise ValueError(f"{event_id} active-hour count disagrees with frozen window calendar")
        if not event_active["source_utc"].between(start, end).all():
            raise ValueError(f"{event_id} active hour lies outside its frozen window")
        during = base.loc[_event_phase_mask(base, start=start, end=end, phase="during_window")]
        active_score_rows = base.merge(
            event_active.loc[:, ["source_utc"]], left_on="__ts__", right_on="source_utc", how="inner", validate="many_to_one"
        )
        active_top40_hours = int(
            active_score_rows.loc[active_score_rows["base_selected_top40_timestamp_side"].astype(bool), "__ts__"].nunique()
        )
        expected_window_hours = int((end - start) / pd.Timedelta(hours=1)) + 1
        coverage_rows.append(
            {
                "transition_event_id": event_id,
                "transition_window_start_utc": start,
                "transition_window_end_utc": end,
                "expected_window_hours": expected_window_hours,
                "observed_window_hours": int(during["__ts__"].nunique()),
                "window_candidate_rows": int(len(during)),
                "window_complete": int(during["__ts__"].nunique()) == expected_window_hours,
                "active_expected_hours": int(len(event_active)),
                "active_observed_hours": int(active_score_rows["__ts__"].nunique()),
                "active_candidate_rows": int(len(active_score_rows)),
                "active_complete": int(active_score_rows["__ts__"].nunique()) == int(len(event_active)),
                "active_top40_hours": active_top40_hours,
                "active_top40_covered": active_top40_hours == int(len(event_active)),
                "active_top40_candidate_rows": int(active_score_rows["base_selected_top40_timestamp_side"].sum()),
            }
        )
        for phase in ("before_24h", "during_window", "after_24h"):
            phase_rows = base.loc[_event_phase_mask(base, start=start, end=end, phase=phase)]
            for side in ("all", "long", "short"):
                local = phase_rows if side == "all" else phase_rows.loc[phase_rows["side_name"].eq(side)]
                metric_rows.append({"transition_event_id": event_id, "phase": phase, "side_name": side, **_metrics(local)})
        for side in ("all", "long", "short"):
            local = active_score_rows if side == "all" else active_score_rows.loc[active_score_rows["side_name"].eq(side)]
            metric_rows.append({"transition_event_id": event_id, "phase": "active_hours", "side_name": side, **_metrics(local)})

    coverage = pd.DataFrame(coverage_rows)
    metrics = pd.DataFrame(metric_rows)
    all_windows_complete = bool(coverage["window_complete"].all())
    all_active_complete = bool(coverage["active_complete"].all())
    all_active_top40 = bool(coverage["active_top40_covered"].all())
    summary = {
        "schema": "febapr2025_base_oof_transition_diagnostic_v1",
        "research_only": True,
        "scope": "descriptive base-only transition diagnostic; no model, routing, or policy selection is changed",
        "event_count": int(len(coverage)),
        "coverage": {
            "events_with_complete_window_scores": int(coverage["window_complete"].sum()),
            "events_with_complete_active_scores": int(coverage["active_complete"].sum()),
            "events_with_top40_in_every_active_hour": int(coverage["active_top40_covered"].sum()),
            "active_hours": int(coverage["active_expected_hours"].sum()),
            "active_hours_with_top40": int(coverage["active_top40_hours"].sum()),
        },
        "later_health_active_risk_readiness": {
            "sufficient_base_score_coverage": all_windows_complete and all_active_complete,
            "sufficient_top40_observability": all_active_top40,
            "limitation": "13 labelled events are diagnostic support, not independent promotion evidence; active labels are ex-post and may not be used to alter scoring/training.",
        },
    }
    return coverage, metrics, summary
