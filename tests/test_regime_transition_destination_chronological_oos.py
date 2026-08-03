from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

STAGING = Path(__file__).resolve().parents[1] / "scripts"
if str(STAGING) not in sys.path:
    sys.path.insert(0, str(STAGING))

from run_regime_transition_destination_chronological_oos import (  # noqa: E402
    abstention_curve,
    destination_frame,
    destination_month_folds,
)


def _frame() -> pd.DataFrame:
    timestamps = pd.date_range("2023-01-01", "2024-03-31 23:00", freq="h", tz="UTC")
    event = timestamps.tz_localize(None).to_period("M").astype(str).to_numpy()
    bridge = (timestamps >= pd.Timestamp("2023-12-31 23:00", tz="UTC")) & (
        timestamps <= pd.Timestamp("2024-01-01 01:00", tz="UTC")
    )
    event[bridge] = "bridge"
    return pd.DataFrame(
        {
            "source_utc": timestamps,
            "target__event_id": event,
            "target__destination_state": timestamps.month % 5,
            "target__available_utc": timestamps + pd.Timedelta(hours=12),
        }
    )


def test_destination_frame_materializes_state_labels() -> None:
    frame = _frame().iloc[:3].copy()
    work = destination_frame(frame)
    assert work["destination_label"].str.startswith("state_").all()


def test_destination_folds_purge_labels_and_evaluation_events() -> None:
    frame = destination_frame(_frame())
    folds = destination_month_folds(
        frame,
        first_evaluation_month="2024-01-01",
        last_evaluation_month="2024-03-01",
        minimum_train_months=12,
    )
    assert len(folds) == 3
    for start, train, evaluation in folds:
        train_events = set(frame.iloc[train]["target__event_id"])
        evaluation_events = set(frame.iloc[evaluation]["target__event_id"])
        assert train_events.isdisjoint(evaluation_events)
        assert (
            pd.to_datetime(
                frame.iloc[train]["target__available_utc"], utc=True
            ).max()
            < start
        )


def test_abstention_curve_improves_accuracy_when_low_confidence_is_wrong() -> None:
    prediction = pd.DataFrame(
        {
            "target__event_id": ["a", "b"],
            "destination_label": ["state_0", "state_1"],
            "p_destination__state_0": [0.9, 0.3],
            "p_destination__state_1": [0.025, 0.2],
            "p_destination__state_2": [0.025, 0.2],
            "p_destination__state_3": [0.025, 0.2],
            "p_destination__state_4": [0.025, 0.1],
        }
    )
    curve = abstention_curve(prediction, (0.0, 0.8))
    assert curve.loc[0, "coverage"] == 1.0
    assert curve.loc[0, "accuracy"] == 0.5
    assert curve.loc[1, "coverage"] == 0.5
    assert curve.loc[1, "accuracy"] == 1.0
