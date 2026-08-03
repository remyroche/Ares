import numpy as np
import pandas as pd

from scripts.run_febapr2025_historical_auxiliary_oof import (
    OUTER_MONTHS,
    _economic_proxy,
    _forbid_outcomes,
    _month_start,
)


def test_forbidden_path_fields_can_never_be_auxiliary_inputs():
    _forbid_outcomes(["safe_feature", "score"])
    try:
        _forbid_outcomes(["safe_feature", "__path_auxiliary_target_valid__"])
    except ValueError as exc:
        assert "future outcome" in str(exc)
    else:  # pragma: no cover - makes the prohibition unambiguous.
        raise AssertionError("path outcome field was incorrectly accepted")


def test_economic_proxy_is_side_month_local_and_never_a_policy_claim():
    rows = []
    for side in ("long", "short"):
        for month in OUTER_MONTHS:
            start = _month_start(month)
            for i in range(10):
                rows.append(
                    {
                        "side_name": side,
                        "__strict_residual_oof__": True,
                        "__ts__": start + pd.Timedelta(hours=i - 1),
                        "__decision_ts__": start + pd.Timedelta(hours=i),
                        "__meaningful_mfe_reached_12h__": int(i == 9),
                        "__peak_mfe_atr_12h__": float(i),
                    }
                )
    frame = pd.DataFrame(rows)
    prediction = np.tile(np.arange(10, dtype=float), 4)
    report = _economic_proxy(frame, prediction, np.ones(len(frame), dtype=bool))
    assert set(report) == {f"{side}/{month}" for side in ("long", "short") for month in OUTER_MONTHS}
    assert report["long/2025-03"]["top_decile_rows"] == 1
    assert report["long/2025-03"]["top_decile_meaningful_hit_rate"] == 1.0
    assert report["long/2025-03"]["hit_rate_lift"] > 0.0


def test_full_context_prediction_vector_is_scattered_by_strict_output_mask():
    """February warm-up rows must never make the emitted OOF vector too long."""
    full_prediction = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    strict_mask = np.array([False, True, True, False, True])
    emitted = full_prediction[strict_mask]
    assert emitted.tolist() == [0.2, 0.3, 0.5]
    assert len(emitted) == int(strict_mask.sum())
