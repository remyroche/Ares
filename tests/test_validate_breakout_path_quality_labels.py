import pandas as pd

from scripts.validate_breakout_path_quality_labels import _derive_outcomes


def test_derive_breakout_outcomes_uses_realized_path_columns():
    rows = pd.DataFrame(
        {
            "__ts__": pd.date_range("2025-01-01", periods=2, freq="h", tz="UTC"),
            "side_name": ["short", "long"],
            "__archetype_policy_key__": ["short_breakout_precision", "long_breakout_diagnostic_candidate"],
            "__path_trailing_success__": [1.0, 0.0],
            "__first_touch_mfe_norm__": [2.0, 1.0],
            "__first_touch_full_path_mae_norm__": [1.0, 3.0],
            "__first_touch_mfe_to_tp__": [2.0, 0.25],
            "__path_post_mfe_drawdown_norm__": [0.5, 2.0],
        }
    )
    outcomes = _derive_outcomes(rows)
    assert outcomes["breakout_retention_outcome"].tolist() == [1.0, 0.0]
    assert outcomes["breakout_path_efficiency_outcome"].tolist()[0] > outcomes["breakout_path_efficiency_outcome"].tolist()[1]
    assert outcomes["breakout_reversal_magnitude_outcome"].tolist() == [0.5, 2.0]
