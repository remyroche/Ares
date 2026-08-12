from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.extend_r5_domain_probability_sweep import timestamp_fraction


def test_timestamp_fraction_is_local_deterministic_and_future_score_independent() -> None:
    frame = pd.DataFrame({
        "candidate_id": ["a", "b", "c", "d", "e", "f"],
        "__decision_ts__": pd.to_datetime([
            "2026-01-01T00:00Z", "2026-01-01T00:00Z", "2026-01-01T00:00Z",
            "2026-01-01T01:00Z", "2026-01-01T01:00Z", "2026-01-01T01:00Z",
        ], utc=True),
        "final_score": [3.0, 2.0, 1.0, 3.0, 2.0, 1.0],
    })
    first = timestamp_fraction(frame)
    mutated = frame.copy()
    mutated.loc[mutated["__decision_ts__"].dt.hour.eq(1), "final_score"] *= -100.0
    second = timestamp_fraction(mutated)
    np.testing.assert_allclose(first[:3], second[:3])
    np.testing.assert_allclose(first[:3], [0.0, 1 / 3, 2 / 3])
