from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.run_full_universe_t2_event_payoff_mapping import _apply_event_map, _fit_event_map


def test_event_payoff_maps_are_fit_only_from_supplied_calibration_rows() -> None:
    # The deliberately extreme OOS-like row is never supplied to the fit and
    # therefore cannot alter either global or side-local payoffs.
    calibration = pd.DataFrame(
        {
            "side_name": ["long", "long", "long", "short", "short", "short"],
            "event": [0, 1, 2, 0, 1, 2],
            "net_bps": [100.0, -200.0, -20.0, 50.0, -300.0, -60.0],
        }
    )
    global_map, _ = _fit_event_map(calibration, side_local=False)
    side_map, _ = _fit_event_map(calibration, side_local=True)
    assert global_map["global"] == [75.0, -250.0, -40.0]
    assert side_map["long"] == [100.0, -200.0, -20.0]
    assert side_map["short"] == [50.0, -300.0, -60.0]


def test_side_local_application_uses_the_corresponding_event_payoff_vector() -> None:
    frame = pd.DataFrame(
        {
            "side_name": ["long", "short"],
            "p_upper": [1.0, 0.0],
            "p_lower": [0.0, 1.0],
            "p_timeout": [0.0, 0.0],
        }
    )
    details = {"global": [0.0, 0.0, 0.0], "long": [10.0, 20.0, 30.0], "short": [40.0, 50.0, 60.0]}
    assert np.array_equal(_apply_event_map(frame, details, side_local=True), np.array([10.0, 50.0]))
