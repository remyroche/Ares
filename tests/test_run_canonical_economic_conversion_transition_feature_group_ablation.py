from __future__ import annotations

import pandas as pd

from scripts.run_canonical_economic_conversion_transition_feature_group_ablation import (
    _build_gate_table,
    _feature_groups,
)


FEATURES = (
    "context__base_oof_score__mean",
    "context__base_rank_pct_timestamp_side__mean",
    "context__range_24h_pct__mean",
    "context__meta_raw__volatility_zscore__mean",
    "context__trend_r2_24__mean",
    "context__jump_intensity__mean",
    "context__meta_raw__chop_score__mean",
    "context__preentry_transition__range_24h_pct__delta_3h__mean",
    "context__preentry_transition__regime_source_shock_impulse_score__delta_12h__mean",
    "context__regime_source_execution_quality_score__mean",
    "context__regime_source_execution_risk_score__mean",
    "context__side_sign",
    "context__frozen_base_score_decile",
)


def test_feature_groups_are_causal_fixed_subsets_and_full_is_exact() -> None:
    groups = _feature_groups(FEATURES)
    assert tuple(groups) == (
        "identity_only",
        "score_only",
        "market_only",
        "regime_level_only",
        "regime_transition_only",
        "market_and_regime",
        "score_and_regime",
        "full_context",
    )
    assert groups["full_context"] == FEATURES
    assert set(groups["score_only"]).issubset(FEATURES)
    assert "context__side_sign" in groups["regime_transition_only"]
    assert "context__frozen_base_score_decile" in groups["regime_level_only"]


def test_gate_requires_aggregate_and_latest_mae_brier_ic_and_auc() -> None:
    keys = {
        "feature_group": ["good", "latest_bad"],
        "horizon_hours": [12, 12],
        "target": ["direct_mean_net", "direct_mean_net"],
    }
    aggregate = pd.DataFrame(
        {
            **keys,
            "model_regression_mae": [0.8, 0.8],
            "constant_regression_mae": [1.0, 1.0],
            "model_regression_rank_ic": [0.2, 0.2],
            "model_sign_auc": [0.6, 0.6],
            "model_sign_ap": [0.6, 0.6],
            "model_sign_brier": [0.20, 0.20],
            "model_sign_calibration_ece_10": [0.03, 0.03],
            "constant_sign_brier": [0.25, 0.25],
        }
    )
    latest = pd.DataFrame(
        {
            **keys,
            "fold_id": [4, 4],
            "validation_start_utc": pd.to_datetime(
                ["2026-04-26T00:00:00Z", "2026-04-26T00:00:00Z"]
            ),
            "validation_end_utc": pd.to_datetime(
                ["2026-05-03T00:00:00Z", "2026-05-03T00:00:00Z"]
            ),
            "target_valid_rows": [100, 100],
            "model_regression_mae": [0.8, 1.1],
            "constant_regression_mae": [1.0, 1.0],
            "model_regression_rank_ic": [0.2, 0.2],
            "model_sign_auc": [0.6, 0.6],
            "model_sign_ap": [0.6, 0.6],
            "model_sign_brier": [0.20, 0.20],
            "model_sign_calibration_ece_10": [0.03, 0.03],
            "constant_sign_brier": [0.25, 0.25],
        }
    )
    gates = _build_gate_table(aggregate, latest).set_index("feature_group")
    assert bool(gates.loc["good", "passes_both_period_gates"])
    assert not bool(gates.loc["latest_bad", "passes_both_period_gates"])
    assert gates.loc["good", "diagnostic_rank_within_target"] == 1
