import numpy as np
import pandas as pd

from extreme_price_movements.regime_transition_research import (
    TransitionResearchConfig,
    add_causal_transition_features,
    discover_stabilized_transition_events,
    materialize_transition_labels,
)


def _panel() -> pd.DataFrame:
    timestamp = pd.date_range("2025-01-01", periods=60, freq="h", tz="UTC")
    state = np.r_[np.zeros(30), np.ones(30)].astype(np.int16)
    return pd.DataFrame(
        {
            "source_utc": timestamp,
            "execution_decision_utc": timestamp + pd.Timedelta(hours=1),
            "segment_id": 1,
            "level": state.astype(float),
            "target__pooled_state": state,
        }
    )


def test_symmetric_destination_and_phase_labels() -> None:
    panel = _panel()
    events = discover_stabilized_transition_events(
        panel,
        config=TransitionResearchConfig(
            minimum_origin_dominance=2 / 3,
            minimum_destination_dominance=2 / 3,
        ),
    )
    assert len(events) == 1
    event = events.iloc[0]
    assert event["source_state"] == 0
    assert event["destination_state"] == 1
    assert event["anchor_source_utc"] == pd.Timestamp(
        "2025-01-02 06:00:00+00:00"
    )
    labels = materialize_transition_labels(panel, events)
    lead = labels.loc[
        labels["source_utc"].between(
            event["anchor_source_utc"] - pd.Timedelta(hours=3),
            event["anchor_source_utc"] - pd.Timedelta(hours=1),
        )
    ]
    assert lead["target__onset_within_3h"].eq(1).all()
    assert lead["target__phase"].eq("immediate_lead").all()
    assert labels.loc[
        labels["source_utc"].eq(event["anchor_source_utc"]),
        "target__transition_active",
    ].eq(1).all()


def test_exact_lags_do_not_bridge_segments() -> None:
    timestamp = pd.to_datetime(
        [
            "2025-01-01T00:00:00Z",
            "2025-01-01T01:00:00Z",
            "2025-01-01T02:00:00Z",
            "2025-01-02T00:00:00Z",
            "2025-01-02T01:00:00Z",
            "2025-01-02T02:00:00Z",
            "2025-01-02T03:00:00Z",
        ]
    )
    panel = pd.DataFrame(
        {
            "source_utc": timestamp,
            "execution_decision_utc": timestamp + pd.Timedelta(hours=1),
            "segment_id": [1, 1, 1, 2, 2, 2, 2],
            "negative_breadth_pct": np.arange(7, dtype=float),
        }
    )
    enriched, _ = add_causal_transition_features(
        panel, stems=["negative_breadth_pct"]
    )
    column = "transition_new__negative_breadth_pct__delta_3h"
    assert np.isnan(enriched.loc[3, column])
    assert enriched.loc[6, column] == 3.0
