#!/usr/bin/env python3
"""Attribute native model economics to canonical regime-transition events.

This is an event-study report over the pooled research transition labels.  It
does not create a causal model feature, claim walk-forward evidence, or fill in
model-health measures that the canonical source does not contain.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


DEFAULT_SOURCE = Path("data_perp/artifacts/regime_transition_research_20260726_v3")
DEFAULT_OUTPUT = Path("data_perp/artifacts/regime_transition_event_impact_20260727_v1")
EVENT_OFFSETS = tuple(range(-12, 13))


def _safe(value: Any) -> Any:
    if isinstance(value, (Path, pd.Timestamp, pd.Timedelta)):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(_safe(payload), indent=2, sort_keys=True) + "\n")


def _mean(frame: pd.DataFrame, column: str) -> float:
    return float(pd.to_numeric(frame[column], errors="coerce").mean())


def _sum(frame: pd.DataFrame, column: str) -> float:
    return float(pd.to_numeric(frame[column], errors="coerce").sum(min_count=1))


def _window(
    hourly: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    return hourly.loc[hourly["source_utc"].ge(start) & hourly["source_utc"].lt(end)]


def _window_metrics(frame: pd.DataFrame, prefix: str) -> dict[str, float | int]:
    admitted = pd.to_numeric(frame["admitted_rows"], errors="coerce")
    scored = frame.loc[frame["net_ev_mean"].notna()]
    return {
        f"{prefix}_hour_count": int(len(frame)),
        f"{prefix}_scored_hour_count": int(len(scored)),
        f"{prefix}_candidate_rows": _sum(frame, "candidate_rows"),
        f"{prefix}_admitted_rows": _sum(frame, "admitted_rows"),
        # ``mapped_score_mean`` is the available base-model mapped EV; it is
        # not a raw score and should not be interpreted as a calibration error.
        f"base_model_ev_{prefix}": _mean(frame, "mapped_score_mean"),
        f"realized_net_ev_{prefix}": _mean(frame, "net_ev_mean"),
        f"realized_gross_ev_{prefix}": _mean(frame, "gross_ev_mean"),
        f"economic_residual_{prefix}": _mean(frame, "economic_residual_mean"),
        f"positive_net_rate_{prefix}": _mean(frame, "positive_net_rate"),
        f"{prefix}_mean_admitted_per_hour": float(admitted.mean()),
    }


def _recovery_hours(
    after: pd.DataFrame,
    before_ev: float,
    transition_end: pd.Timestamp,
) -> float:
    """First scored hour after the transition that reaches pre-event EV."""

    if not np.isfinite(before_ev):
        return np.nan
    candidates = after.loc[
        after["net_ev_mean"].notna() & after["net_ev_mean"].ge(before_ev)
    ]
    if candidates.empty:
        return np.nan
    first = pd.Timestamp(candidates["source_utc"].min())
    return float((first - transition_end) / pd.Timedelta(hours=1))


def attribute_event_impacts(
    events: pd.DataFrame,
    hourly_economics: pd.DataFrame,
    *,
    severe_quantile: float = 0.75,
    after_hours: int = 12,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return one event/origin impact row and its selected event-centred rows.

    Windows use the canonical source timestamp: before ``[-12h, 0h)``, during
    ``[0h, transition_end)``, and after ``[transition_end, transition_end+12h)``.
    A transition can appear once for each native evaluation generation whose
    hourly ledger overlaps one of those windows.
    """

    events = events.copy()
    hourly = hourly_economics.copy()
    for column in ("anchor_source_utc", "transition_start_utc", "transition_end_utc"):
        events[column] = pd.to_datetime(events[column], utc=True, errors="coerce")
    hourly["source_utc"] = pd.to_datetime(hourly["source_utc"], utc=True, errors="coerce")
    events["market_transition_severity"] = pd.to_numeric(
        events["robust_pre_post_shift"], errors="coerce"
    )
    severity_threshold = float(events["market_transition_severity"].quantile(severe_quantile))
    events["is_severe_market_transition"] = events["market_transition_severity"].ge(
        severity_threshold
    )

    records: list[dict[str, object]] = []
    centered_parts: list[pd.DataFrame] = []
    for event in events.itertuples(index=False):
        anchor = pd.Timestamp(event.anchor_source_utc)
        end = pd.Timestamp(event.transition_end_utc)
        after_end = end + pd.Timedelta(hours=after_hours)
        candidate_hourly = hourly.loc[
            hourly["source_utc"].ge(anchor - pd.Timedelta(hours=12))
            & hourly["source_utc"].lt(after_end)
        ]
        for origin, local in candidate_hourly.groupby("evaluation_origin", sort=True, observed=True):
            before = _window(local, anchor - pd.Timedelta(hours=12), anchor)
            during = _window(local, anchor, end)
            after = _window(local, end, after_end)
            metrics: dict[str, object] = {
                "event_id": event.event_id,
                "evaluation_origin": str(origin),
                "source_state": int(event.source_state),
                "destination_state": int(event.destination_state),
                "transition_pair": str(event.transition_archetype),
                "anchor_source_utc": anchor,
                "transition_start_utc": pd.Timestamp(event.transition_start_utc),
                "transition_end_utc": end,
                "transition_duration_hours": float((end - anchor) / pd.Timedelta(hours=1)),
                "market_transition_severity": float(event.market_transition_severity),
                "is_severe_market_transition": bool(event.is_severe_market_transition),
                "economic_failure_event_within_6h": event.economic_failure_event_within_6h,
                "economic_failure_distance_hours": event.economic_failure_distance_hours,
            }
            metrics.update(_window_metrics(before, "before"))
            metrics.update(_window_metrics(during, "during"))
            metrics.update(_window_metrics(after, "after"))
            metrics["base_model_ev_damage_during_vs_before"] = (
                metrics["base_model_ev_before"] - metrics["base_model_ev_during"]
            )
            metrics["realized_ev_damage_during_vs_before"] = (
                metrics["realized_net_ev_before"] - metrics["realized_net_ev_during"]
            )
            metrics["base_model_ev_recovery_after_vs_during"] = (
                metrics["base_model_ev_after"] - metrics["base_model_ev_during"]
            )
            metrics["realized_ev_recovery_after_vs_during"] = (
                metrics["realized_net_ev_after"] - metrics["realized_net_ev_during"]
            )
            metrics["recovery_hours_to_pre_net_ev"] = _recovery_hours(
                after, float(metrics["realized_net_ev_before"]), end
            )
            metrics["is_economically_damaging"] = bool(
                np.isfinite(metrics["realized_ev_damage_during_vs_before"])
                and metrics["realized_ev_damage_during_vs_before"] > 0.0
                and metrics["realized_net_ev_during"] < 0.0
            )
            metrics["is_selected_event"] = bool(
                metrics["is_severe_market_transition"]
                or metrics["is_economically_damaging"]
                or pd.notna(metrics["economic_failure_event_within_6h"])
            )
            records.append(metrics)
            if metrics["is_selected_event"]:
                centered = local.loc[
                    local["source_utc"].between(
                        anchor - pd.Timedelta(hours=12),
                        anchor + pd.Timedelta(hours=12),
                        inclusive="both",
                    )
                ].copy()
                if not centered.empty:
                    centered.insert(0, "event_id", event.event_id)
                    # ``evaluation_origin`` already belongs to the native
                    # hourly ledger and is deliberately retained as-is.
                    centered.insert(1, "source_state", int(event.source_state))
                    centered.insert(2, "destination_state", int(event.destination_state))
                    centered.insert(3, "transition_pair", str(event.transition_archetype))
                    centered.insert(4, "anchor_source_utc", anchor)
                    centered.insert(5, "offset_hours", ((centered["source_utc"] - anchor) / pd.Timedelta(hours=1)).astype(int))
                    centered.insert(6, "is_severe_market_transition", bool(metrics["is_severe_market_transition"]))
                    centered.insert(7, "is_economically_damaging", bool(metrics["is_economically_damaging"]))
                    centered_parts.append(centered)
    impacts = pd.DataFrame.from_records(records)
    centered = pd.concat(centered_parts, ignore_index=True) if centered_parts else pd.DataFrame()
    return impacts, centered


def _aggregate(impacts: pd.DataFrame, by: Iterable[str]) -> pd.DataFrame:
    grouping = list(by)
    mean_columns = [
        column
        for column in impacts.columns
        if column.startswith(
            (
                "base_model_ev_",
                "realized_",
                "economic_residual_",
                "positive_net_rate_",
                "recovery_hours_",
                "market_transition_severity",
            )
        )
        or column == "transition_duration_hours"
    ]
    total_columns = [
        column
        for column in impacts.columns
        if column.endswith(("_hour_count", "_candidate_rows", "_admitted_rows"))
    ]
    grouped = impacts.groupby(grouping, observed=True, sort=True)
    summary = grouped.size().rename("event_origin_count").to_frame()
    summary["unique_event_count"] = grouped["event_id"].nunique()
    summary["severe_event_origin_count"] = grouped["is_severe_market_transition"].sum()
    summary["economically_damaging_event_origin_count"] = grouped["is_economically_damaging"].sum()
    summary["linked_economic_failure_event_origin_count"] = grouped["economic_failure_event_within_6h"].count()
    summary = summary.join(grouped[mean_columns].mean())
    summary = summary.join(grouped[total_columns].sum().add_prefix("total_"))
    return summary.reset_index()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--severe-quantile", type=float, default=0.75)
    parser.add_argument("--after-hours", type=int, default=12)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not 0.0 < float(args.severe_quantile) < 1.0:
        raise ValueError("--severe-quantile must be between zero and one")
    if int(args.after_hours) < 1:
        raise ValueError("--after-hours must be positive")
    source = Path(args.source_dir)
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    events = pd.read_parquet(source / "transition_events.parquet")
    hourly = pd.read_parquet(source / "native_hourly_economics.parquet")
    impacts, centered = attribute_event_impacts(
        events,
        hourly,
        severe_quantile=float(args.severe_quantile),
        after_hours=int(args.after_hours),
    )
    output.mkdir(parents=True)
    impacts.to_parquet(output / "event_impacts.parquet", index=False)
    _aggregate(impacts, ["source_state"]).to_csv(output / "origin_state_impacts.csv", index=False)
    _aggregate(impacts, ["destination_state"]).to_csv(output / "destination_state_impacts.csv", index=False)
    _aggregate(impacts, ["source_state", "destination_state", "transition_pair"]).to_csv(
        output / "transition_pair_impacts.csv", index=False
    )
    centered.to_parquet(output / "severe_or_damaging_event_centered_metrics.parquet", index=False)
    manifest = {
        "schema": "regime_transition_event_impact_attribution_v1",
        "research_only": True,
        "promotion_evidence": False,
        "source_artifact": str(source),
        "event_count": int(len(events)),
        "event_evaluation_origin_rows": int(len(impacts)),
        "selected_event_evaluation_origin_rows": int(impacts["is_selected_event"].sum()),
        "selected_event_centered_rows": int(len(centered)),
        "event_windows": {
            "before": "[anchor_source_utc-12h, anchor_source_utc)",
            "during": "[anchor_source_utc, transition_end_utc)",
            "after": f"[transition_end_utc, transition_end_utc+{int(args.after_hours)}h)",
            "event_centered": "[-12h, +12h] relative to anchor_source_utc",
        },
        "severity": {
            "market_metric": "robust_pre_post_shift",
            "severe_quantile": float(args.severe_quantile),
            "threshold": float(events["robust_pre_post_shift"].quantile(float(args.severe_quantile))),
            "economically_damaging": "realized net EV during is negative and lower than before",
        },
        "available_metrics": {
            "base_model_ev": "mapped_score_mean",
            "realized_economics": ["net_ev_mean", "gross_ev_mean", "economic_residual_mean", "positive_net_rate"],
            "duration": "transition_end_utc - anchor_source_utc",
            "recovery": "first after-window scored hour with net_ev_mean >= before-window mean",
        },
        "unavailable_metrics": {
            "stops": "The canonical economics ledger has no stop or exit-reason field.",
            "calibration": "No probability/EV calibration target or calibration error is present.",
            "rank_ic": "No row-level score/realized-return cross-sectional panel is present.",
            "shrinkage": "No shrinkage factor, prior, or post-processed score field is present.",
        },
        "caveats": [
            "The market-state geometry and events are pooled research labels, not walk-forward promotion evidence.",
            "Native economics begin later than market events and are only available where admitted candidates exist; nulls mean unavailable, not zero impact.",
            "Hourly means are unweighted across hours. Candidate and admitted-row totals are retained so a downstream analysis can choose an exposure weighting.",
            "Event/origin rows are generation-specific; do not pool evaluation origins as if they shared an identical model contract without an explicit comparison design.",
        ],
    }
    _write_json(output / "manifest.json", manifest)
    return manifest


def main() -> None:
    print(json.dumps(_safe(run(_parser().parse_args())), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
