from __future__ import annotations

import numpy as np
import pandas as pd

from extreme_price_movements.regime_transition_hazard import (
    build_transition_hazard_labels,
    expand_at_risk_rows,
    predict_cumulative_hazard,
)


def _frame() -> pd.DataFrame:
    stamps = pd.date_range("2026-01-01", periods=17, freq="h", tz="UTC")
    # The second segment starts after a deliberate two-hour gap.  The first
    # event is at position 4, so positions 1..3 are its three pre-onset rows.
    return pd.DataFrame(
        {
            "source_utc": stamps,
            "segment_id": [1] * 8 + [2] * 9,
            "feature": np.arange(17, dtype=np.float32),
            "target__event_id": [None, "e1", "e1", "e1", "e1", "e1", None, None] + [None] * 9,
            "target__time_to_onset_hours": [np.nan, -3.0, -2.0, -1.0, 0.0, 1.0, np.nan, np.nan] + [np.nan] * 9,
        }
    )


def test_expansion_censors_at_segment_end_and_stops_after_event() -> None:
    frame = _frame()
    labels = build_transition_hazard_labels(frame)
    matrix = frame[["feature"]]
    expanded, target, owner, _ = expand_at_risk_rows(matrix, np.flatnonzero(labels.base_mask), labels)
    # The row one hour before e1 can only contribute a first-bin event row;
    # the final row in a segment has no fully observed interval at all.
    assert target[owner == 3].tolist() == [1.0]
    assert not np.any(owner == 7)
    assert set(expanded.columns).issuperset({"__hazard_interval_to_1h", "__hazard_interval_to_12h"})


def test_pre_onset_rows_share_event_group_and_post_onset_is_not_at_risk() -> None:
    labels = build_transition_hazard_labels(_frame())
    assert labels.base_mask[1:4].all()
    assert not labels.base_mask[4]
    assert not labels.base_mask[5]
    assert len(set(labels.group_ids[1:4])) == 1
    assert labels.event_kind[1] == "unknown"


def test_cumulative_hazard_is_monotone() -> None:
    class FixedHazard:
        def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
            hazard = np.full(len(x), 0.2, dtype=float)
            return np.column_stack([1.0 - hazard, hazard])

    frame = _frame()
    prediction = predict_cumulative_hazard(FixedHazard(), frame[["feature"]], [1, 2, 3])
    assert np.all(np.diff(prediction, axis=1) >= -1e-8)
    assert np.allclose(prediction[:, -1], 1.0 - 0.8**4)
