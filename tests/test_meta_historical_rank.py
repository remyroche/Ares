from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from extreme_price_movements.meta_historical_rank import HistoricalScoreRankReference
from scripts.report_meta_residual_historical_rank import _true_monday_week_start


def test_historical_rank_is_side_aware_and_serializable(tmp_path) -> None:
    train = pd.DataFrame(
        {
            "__ts__": pd.date_range("2026-01-01", periods=6, freq="h", tz="UTC"),
            "side_name": ["long"] * 3 + ["short"] * 3,
            "score_alternative": [0.1, 0.2, 0.3, 0.6, 0.7, 0.8],
        }
    )
    state = HistoricalScoreRankReference().fit(train)
    query = pd.DataFrame(
        {
            "side_name": ["long", "short", "unknown"],
            "score_alternative": [0.25, 0.65, 0.45],
        }
    )
    expected = np.asarray([2 / 3, 1 / 3, 3 / 6], dtype=np.float32)
    assert np.allclose(state.transform(query).to_numpy(), expected)
    path = tmp_path / "rank.joblib"
    joblib.dump(state, path)
    restored = joblib.load(path)
    assert np.array_equal(
        state.transform(query).to_numpy(), restored.transform(query).to_numpy()
    )


def test_true_monday_week_start_is_index_safe() -> None:
    values = pd.Series(
        pd.to_datetime(["2026-04-01 12:00:00Z", "2026-04-05 23:00:00Z"]),
        index=[7, 11],
    )
    result = _true_monday_week_start(values)
    assert result.index.tolist() == [7, 11]
    assert result.astype(str).tolist() == [
        "2026-03-30 00:00:00+00:00",
        "2026-03-30 00:00:00+00:00",
    ]
