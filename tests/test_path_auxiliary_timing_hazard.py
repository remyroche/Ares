from __future__ import annotations

import numpy as np
import pandas as pd

from extreme_price_movements.path_auxiliary_timing_hazard import (
    _event_interval,
    _expand_at_risk,
)


def test_at_risk_expansion_stops_after_event_and_retains_censor() -> None:
    matrix = pd.DataFrame({"feature": [1.0, 2.0, 3.0]})
    timing = np.asarray([1.0, 7.0, 12.0], dtype=np.float32)
    hit = np.asarray([1.0, 1.0, 0.0], dtype=np.float32)
    interval = _event_interval(timing, hit)

    expanded, target, source_rows = _expand_at_risk(matrix, np.arange(3), interval)

    assert interval.tolist() == [0, 2, -1]
    assert source_rows.tolist() == [0, 1, 2, 1, 2, 1, 2, 2]
    assert target.tolist() == [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    assert expanded.filter(like="__hazard_bin_").sum(axis=1).eq(1.0).all()


def test_hazard_to_cdf_is_structurally_monotone() -> None:
    from extreme_price_movements.path_auxiliary_timing_hazard import _UPPER

    hazard = np.asarray([[0.8, 0.1, 0.9, 0.2], [0.0, 0.4, 0.0, 0.7]], dtype=np.float64)
    cdf = 1.0 - np.cumprod(1.0 - hazard, axis=1)

    assert len(_UPPER) == cdf.shape[1]
    assert np.all(np.diff(cdf, axis=1) >= 0.0)
    assert np.all((cdf >= 0.0) & (cdf <= 1.0))
