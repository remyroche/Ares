"""Descriptive base-EV versus residual-EV transition diagnostic."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


IDENTITY = ("candidate_id", "side_name", "__symbol__", "__ts__")
REQUIRED = {
    *IDENTITY, "base_expected_ev", "residual_expected_ev", "residual_is_oof",
    "selected_top40", "__first_touch_capture_net__", "execution_net_ev_12h",
}
WINDOW_REQUIRED = {"transition_event_id", "transition_window_start_utc", "transition_window_end_utc", "transition_active_hours"}
ACTIVE_REQUIRED = {"source_utc", "target__event_id", "target__transition_active"}


def _ic(frame: pd.DataFrame, column: str) -> float:
    local = frame.loc[:, [column, "__first_touch_capture_net__"]].apply(pd.to_numeric, errors="coerce").dropna()
    return float(local.corr(method="spearman").iloc[0, 1]) if len(local) >= 2 else float("nan")


def _phase(frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, name: str) -> pd.DataFrame:
    if name == "before_24h":
        return frame.loc[frame.__ts__.ge(start - pd.Timedelta(hours=24)) & frame.__ts__.lt(start)]
    if name == "during_window":
        return frame.loc[frame.__ts__.ge(start) & frame.__ts__.le(end)]
    if name == "after_24h":
        return frame.loc[frame.__ts__.gt(end) & frame.__ts__.le(end + pd.Timedelta(hours=24))]
    raise ValueError(name)


def _metrics(frame: pd.DataFrame) -> dict[str, Any]:
    selected = frame.loc[frame.selected_top40.astype(bool)]
    result: dict[str, Any] = {
        "candidate_rows": int(len(frame)), "candidate_hours": int(frame.__ts__.nunique()),
        "selected_top40_rows": int(len(selected)), "selected_top40_hours": int(selected.__ts__.nunique()),
        "execution_net_ev_12h_mean": float(frame.execution_net_ev_12h.mean()),
        "selected_top40_execution_net_ev_12h_mean": float(selected.execution_net_ev_12h.mean()) if len(selected) else float("nan"),
    }
    for name in ("base_expected_ev", "residual_expected_ev"):
        result[f"{name}_mean"] = float(frame[name].mean())
        result[f"{name}_native_target_spearman"] = _ic(frame, name)
        result[f"{name}_selected_top40_execution_net_ev_12h_mean"] = float(selected.execution_net_ev_12h.mean()) if len(selected) else float("nan")
    return result


def build_residual_transition_diagnostic(
    residual_oof: pd.DataFrame, windows: pd.DataFrame, active_hours: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    missing = sorted(REQUIRED.difference(residual_oof.columns))
    if missing:
        raise ValueError(f"residual OOF lacks columns: {missing}")
    if residual_oof.duplicated(list(IDENTITY)).any():
        raise ValueError("residual OOF has duplicate identities")
    scores = residual_oof.loc[residual_oof.residual_is_oof.astype(bool)].copy()
    scores["__ts__"] = pd.to_datetime(scores.__ts__, utc=True, errors="raise")
    scores.side_name = scores.side_name.astype(str).str.lower()
    if scores[["base_expected_ev", "residual_expected_ev"]].isna().any().any():
        raise ValueError("strict residual OOF lacks an EV score")
    missing = sorted(WINDOW_REQUIRED.difference(windows.columns))
    if missing or windows.transition_event_id.duplicated().any():
        raise ValueError(f"invalid event windows: {missing}")
    event_windows = windows.loc[:, sorted(WINDOW_REQUIRED)].copy()
    for col in ("transition_window_start_utc", "transition_window_end_utc"):
        event_windows[col] = pd.to_datetime(event_windows[col], utc=True, errors="raise")
    # February cannot have strict residual scores, hence only the 11 fully
    # scored March-April event windows are valid for this comparison.
    event_windows = event_windows.loc[event_windows.transition_window_start_utc.ge(scores.__ts__.min())].copy()
    if len(event_windows) != 11:
        raise ValueError(f"expected exactly 11 March-April events, found {len(event_windows)}")
    missing = sorted(ACTIVE_REQUIRED.difference(active_hours.columns))
    if missing:
        raise ValueError(f"active ledger lacks columns: {missing}")
    active = active_hours.loc[:, sorted(ACTIVE_REQUIRED)].copy()
    active.source_utc = pd.to_datetime(active.source_utc, utc=True, errors="raise")
    if active.source_utc.duplicated().any():
        raise ValueError("active ledger needs one source row per hour")
    active.target__transition_active = active.target__transition_active.fillna(0).astype(bool)
    coverage: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for event in event_windows.sort_values("transition_event_id").itertuples(index=False):
        event_id, start, end = str(event.transition_event_id), pd.Timestamp(event.transition_window_start_utc), pd.Timestamp(event.transition_window_end_utc)
        expected = int((end - start) / pd.Timedelta(hours=1)) + 1
        window = _phase(scores, start, end, "during_window")
        active_event = active.loc[active.target__event_id.astype(str).eq(event_id) & active.target__transition_active]
        if len(active_event) != int(event.transition_active_hours):
            raise ValueError(f"{event_id} active-hour calendar mismatch")
        active_scores = scores.merge(active_event[["source_utc"]], left_on="__ts__", right_on="source_utc", how="inner", validate="many_to_one")
        coverage.append({
            "transition_event_id": event_id, "transition_window_start_utc": start, "transition_window_end_utc": end,
            "window_expected_hours": expected, "window_observed_hours": int(window.__ts__.nunique()),
            "window_complete": int(window.__ts__.nunique()) == expected,
            "active_expected_hours": int(len(active_event)), "active_observed_hours": int(active_scores.__ts__.nunique()),
            "active_complete": int(active_scores.__ts__.nunique()) == int(len(active_event)),
            "active_selected_top40_rows": int(active_scores.selected_top40.astype(bool).sum()),
            "active_selected_top40_hours": int(active_scores.loc[active_scores.selected_top40.astype(bool), "__ts__"].nunique()),
        })
        for phase in ("before_24h", "during_window", "after_24h"):
            phase_scores = _phase(scores, start, end, phase)
            for side in ("all", "long", "short"):
                local = phase_scores if side == "all" else phase_scores.loc[phase_scores.side_name.eq(side)]
                rows.append({"transition_event_id": event_id, "phase": phase, "side_name": side, **_metrics(local)})
        for side in ("all", "long", "short"):
            local = active_scores if side == "all" else active_scores.loc[active_scores.side_name.eq(side)]
            rows.append({"transition_event_id": event_id, "phase": "active_hours", "side_name": side, **_metrics(local)})
    coverage_frame, metric_frame = pd.DataFrame(coverage), pd.DataFrame(rows)
    summary = {
        "schema": "marapr2025_strict_residual_oof_transition_diagnostic_v1", "research_only": True,
        "scope": "identical event calendar comparison of calibrated base EV and strict residual EV; no routing/policy effect",
        "event_count": int(len(coverage_frame)), "strict_residual_oof_rows": int(len(scores)),
        "coverage": {"complete_windows": int(coverage_frame.window_complete.sum()), "complete_active_events": int(coverage_frame.active_complete.sum()), "active_hours": int(coverage_frame.active_expected_hours.sum()), "active_hours_with_selected_top40": int(coverage_frame.active_selected_top40_hours.sum())},
        "readiness": {"sufficient_for_descriptive_health": bool(coverage_frame.window_complete.all() and coverage_frame.active_complete.all()), "limitation": "ex-post active labels and 11 events make this diagnostic-only, not promotion evidence."},
    }
    return coverage_frame, metric_frame, summary
