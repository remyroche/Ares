from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "attribute_regime_transition_event_impact.py"
SPEC = spec_from_file_location("transition_event_impact", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_attribute_event_impacts_reports_windows_damage_and_recovery() -> None:
    anchor = pd.Timestamp("2025-01-02T12:00:00Z")
    events = pd.DataFrame(
        {
            "event_id": ["event_1"],
            "anchor_source_utc": [anchor],
            "transition_start_utc": [anchor],
            "transition_end_utc": [anchor + pd.Timedelta(hours=3)],
            "source_state": [0],
            "destination_state": [1],
            "transition_archetype": ["state_0_to_state_1"],
            "robust_pre_post_shift": [2.0],
            "economic_failure_event_within_6h": [None],
            "economic_failure_distance_hours": [np.nan],
        }
    )
    stamps = pd.date_range(anchor - pd.Timedelta(hours=12), periods=27, freq="h", tz="UTC")
    net = np.where(stamps < anchor, 1.0, np.where(stamps < anchor + pd.Timedelta(hours=3), -2.0, 1.5))
    hourly = pd.DataFrame(
        {
            "source_utc": stamps,
            "evaluation_origin": "outer_oof",
            "candidate_rows": 10,
            "admitted_rows": 2,
            "mapped_score_mean": net + 0.25,
            "net_ev_mean": net,
            "gross_ev_mean": net + 0.1,
            "economic_residual_mean": -0.25,
            "positive_net_rate": (net > 0).astype(float),
        }
    )

    impacts, centered = MODULE.attribute_event_impacts(events, hourly)

    assert len(impacts) == 1
    row = impacts.iloc[0]
    assert row["before_hour_count"] == 12
    assert row["during_hour_count"] == 3
    assert row["after_hour_count"] == 12
    assert row["realized_ev_damage_during_vs_before"] == 3.0
    assert row["realized_ev_recovery_after_vs_during"] == 3.5
    assert row["recovery_hours_to_pre_net_ev"] == 0.0
    assert bool(row["is_economically_damaging"])
    assert centered["offset_hours"].min() == -12
    assert centered["offset_hours"].max() == 12
