from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.ablate_r5_posterior_contract import prequential_calibration


def _frame() -> pd.DataFrame:
    rng = np.random.default_rng(19)
    parts = []
    for month in ("2025-10", "2025-11", "2025-12"):
        ts = pd.date_range(f"{month}-01", periods=2_500, freq="5min", tz="UTC")
        prediction = rng.normal(75.0, 60.0, len(ts))
        parts.append(pd.DataFrame({
            "__decision_ts__": ts,
            "policy_label_available_ts": ts + pd.Timedelta(hours=12),
            "policy_path_valid": True,
            "policy_net_bps": 10.0 + 0.8 * prediction + rng.normal(0.0, 75.0, len(ts)),
            "posterior_expected_bps": prediction,
            "posterior_predictive_sd": 100.0,
            "month": month,
        }))
    return pd.concat(parts, ignore_index=True)


def test_prequential_calibration_does_not_consume_held_or_future_outcomes() -> None:
    frame = _frame()
    expected, probability, audit = prequential_calibration(frame)
    mutated = frame.copy()
    mutated.loc[mutated["month"].eq("2025-12"), "policy_net_bps"] += 1_000_000.0
    expected_mutated, probability_mutated, audit_mutated = prequential_calibration(mutated)
    np.testing.assert_allclose(expected, expected_mutated)
    np.testing.assert_allclose(probability, probability_mutated)
    pd.testing.assert_frame_equal(audit, audit_mutated)


def test_prequential_calibration_is_cold_start_then_prior_oos_only() -> None:
    _expected, _probability, audit = prequential_calibration(_frame())
    assert audit.loc[0, "status"] == "identity_cold_start"
    assert audit.loc[0, "prior_oos_rows"] == 0
    assert audit.loc[1, "status"] == "prior_oos_huber_and_80pct_scale"
    assert audit.loc[1, "prior_oos_rows"] >= 2_000
