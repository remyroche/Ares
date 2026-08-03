from __future__ import annotations

import pandas as pd

from scripts.audit_febapr_historical_path_heads_readiness import _role_support


def test_role_support_requires_explicit_meaningful_event_not_atr_only_proxy() -> None:
    frame = pd.DataFrame(
        {
            "__path_auxiliary_target_valid__": [1, 1],
            "__meaningful_mfe_reached_12h__": [1, 0],
            "__time_to_first_meaningful_mfe_hours_12h__": [2.0, 12.0],
            "__peak_mfe_atr_12h__": [2.0, 0.0],
            "__mae_before_meaningful_mfe_atr_12h__": [0.1, 0.5],
            "__bars_before_price_stops_decreasing_12h__": [1.0, 3.0],
            "__bars_to_confirmed_adverse_trough__": [2.0, float("nan")],
            "__future_slope_atr_per_hour_12h__": [0.3, 0.0],
        }
    )
    report = _role_support(frame)
    assert report["peak_mfe_12h_atr.p_hit"]["positive_rows"] == 1
    assert report["mae_before_meaningful_mfe_atr.if_hit"]["train_rows"] == 1
    assert report["mae_before_meaningful_mfe_atr.if_no_hit"]["train_rows"] == 1
