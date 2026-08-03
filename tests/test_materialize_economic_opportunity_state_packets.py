from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


RUNNER = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "materialize_economic_opportunity_state_packets.py"
)
SPEC = importlib.util.spec_from_file_location("opportunity_packets", RUNNER)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _candidates(hours: int = 100) -> pd.DataFrame:
    rows = []
    start = pd.Timestamp("2025-01-01", tz="UTC")
    for hour in range(hours):
        stamp = start + pd.Timedelta(hours=hour)
        for index, side in enumerate(("long", "short")):
            gross = 0.02 if hour < 80 else (-0.01 if index else 0.005)
            cost = 0.01
            net = gross - cost
            rows.append(
                {
                    "candidate_id": f"{hour}-{side}",
                    "__ts__": stamp,
                    "__symbol__": f"asset-{index}",
                    "side_name": side,
                    "execution_decision_utc": stamp + pd.Timedelta(hours=1),
                    "execution_label_end_utc": stamp + pd.Timedelta(hours=13),
                    "execution_gross_ev_12h": gross,
                    "execution_net_ev_12h": net,
                    "execution_cost_return": cost,
                    "execution_exit_reason": "timeout" if hour >= 80 else "trailing",
                    "execution_mfe_return_12h": gross + (0.03 if hour >= 80 else 0.005),
                    "execution_mae_return_12h": -0.01,
                    "score_base": 0.5 + index,
                    "score_alpha_residual": 0.4 + index,
                    "score_direct_execution_ev": 0.3 + index,
                    "score_mapped_execution_ev": 0.2 + index,
                }
            )
    return pd.DataFrame(rows)


def test_hourly_components_are_candidate_weighted_and_economic() -> None:
    hourly = MODULE.build_hourly_components(_candidates(), "test")
    assert len(hourly) == 100
    assert hourly["selected_rows"].eq(2).all()
    assert hourly.iloc[0]["opportunity_rate"] == 1.0
    assert hourly.iloc[-1]["opportunity_rate"] == 0.0
    assert hourly.iloc[-1]["timeout_rate"] == 1.0
    assert hourly.iloc[-1]["exit_conversion_loss_mean"] == pytest.approx(0.03)


def test_identity_contract_rejects_non_12h_labels() -> None:
    frame = _candidates(2)
    frame.loc[0, "execution_label_end_utc"] += pd.Timedelta(hours=1)
    with pytest.raises(ValueError, match="exact 12h"):
        MODULE._validate_candidate_identity(frame, "test")


def test_multilabel_taxonomy_keeps_mixed_and_unclassified_separate() -> None:
    reference = pd.DataFrame(
        {
            "opportunity_rate": [0.8] * 100,
            "positive_net_contribution": [0.01] * 100,
            "favorable_payoff_mean": [0.02] * 100,
            "adverse_payoff_magnitude_mean": [0.01] * 100,
            "loss_net_contribution": [0.001] * 100,
            "timeout_rate": [0.1] * 100,
            "timeout_loss_contribution": [0.001] * 100,
            "timeout_conditional_net": [-0.001] * 100,
            "exit_conversion_loss_mean": [0.005] * 100,
            "cost_mean": [0.01] * 100,
            "net_mean": [0.01] * 100,
            "selected_asset_hhi": [0.2] * 100,
            "distinct_assets": [10.0] * 100,
        }
    )
    event = pd.Series(
        {
            "opportunity_rate": 0.6,
            "positive_net_contribution": 0.005,
            "favorable_payoff_mean": 0.0199,
            "adverse_payoff_magnitude_mean": 0.03,
            "loss_net_contribution": 0.01,
            "timeout_rate": 0.1,
            "timeout_loss_contribution": 0.001,
            "timeout_conditional_net": -0.001,
            "exit_conversion_loss_mean": 0.005,
            "cost_mean": 0.01,
            "net_mean": -0.01,
            "selected_asset_hhi": 0.2,
            "distinct_assets": 10.0,
        }
    )
    result = MODULE.classify_opportunity_state(event, reference)
    assert result["state__sparse_opportunity"] is True
    assert result["state__adverse_payoff_expansion"] is True
    assert result["state__mixed"] is True
    assert result["state__unclassified"] is False


def test_broad_and_strict_anchors_consolidate_within_lineage() -> None:
    anchor = pd.Timestamp("2025-01-10", tz="UTC")
    events = pd.DataFrame(
        {
            "economic_event_id": ["broad-1", "strict-1", "strict-2"],
            "failure_label": ["broad", "strict", "strict"],
            "anchor_source_utc": [
                anchor,
                anchor + pd.Timedelta(hours=2),
                anchor + pd.Timedelta(days=2),
            ],
            "target_available_utc": [
                anchor + pd.Timedelta(days=1),
                anchor + pd.Timedelta(days=1, hours=2),
                anchor + pd.Timedelta(days=3),
            ],
        }
    )
    incidents = MODULE.consolidate_failure_incidents(events, "lineage")
    assert len(incidents) == 2
    first = incidents.iloc[0]
    assert bool(first["incident_has_broad_failure"])
    assert bool(first["incident_has_strict_failure"])
    assert first["source_anchor_count"] == 2


def test_event_reference_uses_only_outcomes_resolved_before_anchor() -> None:
    hourly = MODULE.build_hourly_components(_candidates(), "test")
    active = pd.DataFrame(
        {"source_utc": hourly["source_utc"], "prediction": 0.2}
    )
    destination = pd.DataFrame(
        {
            "source_utc": hourly["source_utc"].iloc[::10],
            "p_destination__state_0": 0.7,
            "destination_confidence": 0.7,
            "destination_entropy": 0.3,
        }
    )
    health = pd.DataFrame(
        {"source_utc": hourly["source_utc"], "health__candidate_rows": 2.0}
    )
    hourly = MODULE.attach_context(hourly, health, active, destination)
    anchor = pd.Timestamp("2025-01-04 12:00", tz="UTC")
    events = pd.DataFrame(
        {
            "economic_event_id": ["strict-1"],
            "failure_label": ["strict"],
            "anchor_source_utc": [anchor],
            "target_available_utc": [anchor + pd.Timedelta(hours=24)],
        }
    )
    packets, trajectory = MODULE.build_event_packets(
        hourly, events, "test", reference_days=21, minimum_reference_hours=24
    )
    assert len(packets) == 1
    # Labels resolve 13 hours after source, so the 13 most recent source hours
    # are unavailable at the event anchor and cannot enter the reference.
    assert packets.iloc[0]["reference_end_utc"] < anchor - pd.Timedelta(hours=12)
    assert set(trajectory["packet_phase"]).issubset({"origin", "event", "recovery"})


def test_insufficient_reference_does_not_manufacture_a_state() -> None:
    hourly = MODULE.build_hourly_components(_candidates(20), "test")
    hourly["active_transition_probability"] = np.nan
    hourly["destination_confidence"] = np.nan
    hourly["destination_entropy"] = np.nan
    event = pd.DataFrame(
        {
            "economic_event_id": ["strict-1"],
            "failure_label": ["strict"],
            "anchor_source_utc": [pd.Timestamp("2025-01-01 18:00", tz="UTC")],
            "target_available_utc": [pd.Timestamp("2025-01-03", tz="UTC")],
        }
    )
    packets, _ = MODULE.build_event_packets(
        hourly, event, "test", minimum_reference_hours=72
    )
    assert packets.iloc[0]["reference_status"] == "INSUFFICIENT_CAUSAL_REFERENCE"
    assert bool(packets.iloc[0]["state__unclassified"])
    assert int(packets.iloc[0]["state_label_count"]) == 0
