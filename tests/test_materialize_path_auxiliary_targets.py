from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.materialize_path_auxiliary_targets import (
    LABEL_RESOLUTION_COLUMN,
    SymbolBars,
    materialize_batch_targets,
)


def test_materializer_starts_path_at_decision_bar_and_preserves_side_symmetry() -> None:
    index = pd.date_range("2026-01-01", periods=16, freq="1h", tz="UTC")
    index_ns = index.view("i8")
    # The signal bar contains a large move that must not enter either target.
    open_ = np.full(16, 100.0)
    high = np.array([100, 130, 100, 100, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113], dtype=float)
    low = np.array([100, 70, 100, 100, 98, 97, 96, 95, 94, 93, 92, 91, 90, 89, 88, 87], dtype=float)
    bars = {"X/USD:USD": SymbolBars(index_ns, open_, high, low)}
    frame = pd.DataFrame(
        {
            "__ts__": [index[1], index[1]],
            "__symbol__": ["X/USD:USD", "X/USD:USD"],
            "side_name": ["long", "short"],
            "__path_auxiliary_atr_fraction__": [0.01, 0.01],
        }
    )

    targets = materialize_batch_targets(frame, bars)

    # Decision is index[2]; the 30% signal-bar excursion at index[1] is absent.
    np.testing.assert_allclose(targets["__peak_mfe_atr_12h__"], [10.0, 10.0])
    np.testing.assert_allclose(
        targets["__time_to_first_meaningful_mfe_hours_12h__"], [3.0, 3.0]
    )
    assert targets["__path_auxiliary_target_valid__"].tolist() == [1, 1]
    expected_resolution = np.datetime64(index[14].tz_localize(None), "ns")
    np.testing.assert_array_equal(
        targets[LABEL_RESOLUTION_COLUMN],
        np.array([expected_resolution, expected_resolution], dtype="datetime64[ns]"),
    )


def test_materializer_rejects_non_contiguous_future_path() -> None:
    index = pd.DatetimeIndex(
        [pd.Timestamp("2026-01-01", tz="UTC") + pd.Timedelta(hours=i) for i in range(14) if i != 5]
    )
    values = np.full(len(index), 100.0)
    frame = pd.DataFrame(
        {
            "__ts__": [index[0]],
            "__symbol__": ["X/USD:USD"],
            "side_name": ["long"],
            "__path_auxiliary_atr_fraction__": [0.01],
        }
    )
    targets = materialize_batch_targets(
        frame,
        {"X/USD:USD": SymbolBars(index.view("i8"), values, values, values)},
    )
    assert targets["__path_auxiliary_target_valid__"].tolist() == [0]
    assert np.isnan(targets["__peak_mfe_atr_12h__"][0])
    assert np.isnat(targets[LABEL_RESOLUTION_COLUMN][0])
