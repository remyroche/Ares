from __future__ import annotations

import numpy as np

from scripts.run_strict_forward_semimarkov_regime_challenger import (
    PERSISTENT_STATE_GATE,
    contiguous_segments,
    duration_model,
    gate,
    semimarkov_filter,
)


def test_semimarkov_filter_enforces_minimum_dwell_before_exit() -> None:
    states = np.array([0, 0, 0, 1, 1, 1, 0, 0, 0])
    segments = np.array(["a"] * len(states))
    duration = duration_model(states, segments, n_states=2, minimum_dwell=3, max_duration=8)
    assert np.allclose(duration["hazards"][:, :2], 0.0)
    emissions = np.tile(np.array([[.51, .49]]), (6, 1))
    posterior, ages, hazard, _, _ = semimarkov_filter(emissions, np.array(["b"] * 6), duration)
    assert posterior.shape == (6, 2)
    assert np.allclose(posterior.sum(axis=1), 1.0)
    assert ages[2] >= 2.0
    assert hazard[0] == 0.0


def test_gap_resets_hourly_filter_and_gate_is_structural() -> None:
    segments = contiguous_segments(
        __import__("pandas").Series(__import__("pandas").to_datetime(["2025-01-01T00:00:00Z", "2025-01-01T01:00:00Z", "2025-01-01T04:00:00Z"])),
        __import__("pandas").Series(["x", "x", "x"]),
    )
    assert segments[0] == segments[1]
    assert segments[2] != segments[1]
    good = {"median_dwell_hours": PERSISTENT_STATE_GATE["median_dwell_hours_min"], "temporal_switch_rate": PERSISTENT_STATE_GATE["temporal_switch_rate_max"], "minimum_state_occupancy": PERSISTENT_STATE_GATE["minimum_state_occupancy_min"], "mean_max_posterior": PERSISTENT_STATE_GATE["mean_max_posterior_min"]}
    assert gate(good)
    assert not gate({**good, "temporal_switch_rate": .100001})
