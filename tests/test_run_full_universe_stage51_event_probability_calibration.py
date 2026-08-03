from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.run_full_universe_stage51_event_probability_calibration import (
    _apply_b2,
    _fit_calibration,
    _fit_payoffs,
)


def _history() -> pd.DataFrame:
    # Enough repeated supported classes for both calibrators, with deliberately
    # different side distributions so the side-shrunk branch is exercised.
    rows = []
    for side, shift in (("long", 0.03), ("short", -0.03)):
        for event in range(3):
            for _ in range(24):
                p = np.array([0.25 + shift, 0.45 - shift, 0.30])
                p[event] += 0.12
                p /= p.sum()
                rows.append({"side_name": side, "event": event, "p_upper": p[0], "p_lower": p[1], "p_timeout": p[2],
                             "gross_bps": (160.0, -290.0, -55.0)[event]})
    return pd.DataFrame(rows)


def test_calibrators_keep_the_three_event_probabilities_on_the_simplex() -> None:
    history = _history()
    probs = history[["p_upper", "p_lower", "p_timeout"]].to_numpy(float)
    sides = history.side_name.to_numpy(str)
    for method in ("temperature", "vector"):
        for shrunk in (False, True):
            calibrated = _fit_calibration(history, method, shrunk).predict(probs, sides)
            assert np.isfinite(calibrated).all()
            assert (calibrated > 0.0).all()
            assert np.allclose(calibrated.sum(axis=1), 1.0)


def test_b2_is_expected_gross_less_the_single_fixed_cost() -> None:
    history = _history()
    payoff = _fit_payoffs(history)
    probabilities = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    scores = _apply_b2(probabilities, np.array(["long", "short"]), payoff)
    assert np.allclose(scores, np.array([60.0, -390.0]))
