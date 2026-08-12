from __future__ import annotations

import numpy as np
import pandas as pd

from extreme_price_movements.strict_r3_a5_trust import (
    A5CausalCalibration,
    apply_a5_bounded_10pct,
)


def test_bounded_a5_arms_cannot_remove_a0_admission() -> None:
    a0 = np.asarray([True, False, True, False])
    a5 = np.asarray([False, True, True, False])
    union = a0 | a5
    assert np.all(union[a0])
    assert union.tolist() == [True, True, True, False]


def test_demotion_correction_never_promotes_score() -> None:
    a0 = np.asarray([40.0, 60.0, 100.0])
    a5 = np.asarray([80.0, 30.0, 120.0])
    corrected = a0 + 0.20 * np.minimum(a5 - a0, 0.0)
    assert np.all(corrected <= a0)


def test_a5_bounded_10pct_uses_fixed_a0_top15_admission() -> None:
    frame = pd.DataFrame({
        "candidate_id": [f"c{i}" for i in range(20)],
        "__decision_ts__": pd.Timestamp("2026-08-01T00:00:00Z"),
        "final_score": np.arange(20, 0, -1),
        "trust_posterior_expected_bps": [60.0] * 20,
        "a4_raw_expected_bps": np.arange(20, dtype=float) * 10.0,
        "a4_raw_predictive_sd_bps": [100.0] * 20,
    })
    calibration = A5CausalCalibration(
        cutoff=pd.Timestamp("2026-08-01T00:00:00Z"),
        slope=1.0, intercept=0.0, predictive_sd_scale=1.0,
        prior_oos_rows=10_000, status="test", source_hashes=(),
    )
    output = apply_a5_bounded_10pct(frame, calibration=calibration)
    assert output["a5_timestamp_top15"].sum() == 3
    assert output["a5_bounded10_admitted"].sum() == 3
    expected = 60.0 + 0.10 * (frame["a4_raw_expected_bps"] - 60.0)
    np.testing.assert_allclose(output["a5_bounded10_expected_bps"], expected)


def test_a5_cannot_add_below_threshold_even_when_a4_is_high() -> None:
    frame = pd.DataFrame({
        "candidate_id": ["a", "b"],
        "__decision_ts__": pd.Timestamp("2026-08-01T00:00:00Z"),
        "final_score": [1.0, 0.0],
        "trust_posterior_expected_bps": [49.9, 100.0],
        "a4_raw_expected_bps": [1_000.0, 100.0],
        "a4_raw_predictive_sd_bps": [100.0, 100.0],
    })
    calibration = A5CausalCalibration(
        cutoff=pd.Timestamp("2026-08-01T00:00:00Z"),
        slope=1.0, intercept=0.0, predictive_sd_scale=1.0,
        prior_oos_rows=10_000, status="test", source_hashes=(),
    )
    output = apply_a5_bounded_10pct(frame, calibration=calibration)
    assert not output.loc[0, "a5_bounded10_admitted"]
