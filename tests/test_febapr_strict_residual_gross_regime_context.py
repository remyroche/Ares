import pandas as pd

from scripts.materialize_febapr_strict_residual_gross_regime_context import (
    TRANSITION_STEMS,
    add_causal_transition_deltas,
    forbidden_feature_names,
    validate_panel,
)


def test_forbidden_feature_name_gate_rejects_post_entry_fields():
    assert forbidden_feature_names(["range_24h_pct", "__future_slope_atr_per_hour_12h__"]) == [
        "__future_slope_atr_per_hour_12h__"
    ]
    assert forbidden_feature_names(["spread_proxy_abs_return_bps_robust_z"]) == []


def test_validate_panel_rejects_pathological_spread_proxy_before_size_gate():
    panel = pd.DataFrame(
        {
            "candidate_id": ["x"],
            "side_name": ["long"],
            "__symbol__": ["BTC_USD:USD"],
            "__ts__": [pd.Timestamp("2025-03-01T00:00:00Z")],
            "__decision_ts__": [pd.Timestamp("2025-03-01T01:00:00Z")],
            "__signal_ts__": [pd.Timestamp("2025-03-01T00:00:00Z")],
            "spread_proxy_abs_return_bps_robust_z": [0.0],
        }
    )
    try:
        validate_panel(panel, ("spread_proxy_abs_return_bps_robust_z",))
    except ValueError as exc:
        assert "pathological spread proxy" in str(exc)


def test_transition_delta_requires_an_exact_past_hour_gap():
    ts = pd.to_datetime(
        ["2025-03-01T00:00:00Z", "2025-03-01T03:00:00Z", "2025-03-01T07:00:00Z"], utc=True
    )
    frame = pd.DataFrame(
        {
            "side_name": ["long"] * 3,
            "__symbol__": ["BTC/USD:USD"] * 3,
            "__ts__": ts,
            **{name: [1.0, 2.0, 5.0] for name in TRANSITION_STEMS},
        }
    )
    result, _ = add_causal_transition_deltas(frame)
    column = "preentry_transition__range_24h_pct__delta_3h"
    assert pd.isna(result.loc[0, column])
    assert result.loc[1, column] == 1.0
    assert pd.isna(result.loc[2, column])
