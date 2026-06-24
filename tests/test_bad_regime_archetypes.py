import numpy as np
import pandas as pd

from extreme_price_movements.unsupervised_regime_learning.bad_regime_archetypes import (
    BadRegimeArchetypeFeatureConfig,
    build_bad_regime_archetype_feature_frame,
)


def _synthetic_panel() -> pd.DataFrame:
    timestamps = pd.date_range("2026-01-01", periods=120, freq="h", tz="UTC")
    rows = []
    for symbol_offset, symbol in enumerate(["AAA/USD:USD", "BBB/USD:USD"]):
        x = np.linspace(0.0, 1.0, len(timestamps), dtype=np.float32) + symbol_offset
        y = np.sin(np.linspace(0.0, 8.0, len(timestamps))).astype(np.float32) + symbol_offset * 0.1
        for ts, left, right in zip(timestamps, x, y):
            rows.append(
                {
                    "timestamp": ts,
                    "symbol": symbol,
                    "asset_minus_mkt_oi_1d": left,
                    "fund_high_neg_mom_self_z_5h": right,
                }
            )
    return pd.DataFrame(rows)


def test_bad_regime_archetype_features_are_causal_against_future_changes():
    frame = _synthetic_panel()
    definitions = {
        "leverage_crowding_archetype": {
            "evidence_score": 1.0,
            "mechanism_channel": "oi_funding_crowding",
            "top_features": [
                "asset_minus_mkt_oi_1d",
                "fund_high_neg_mom_self_z_5h",
            ],
            "deployable_features": ["oi_funding_crowding_score"],
        }
    }
    config = BadRegimeArchetypeFeatureConfig(
        trailing_window=48,
        min_periods=12,
        min_resolved_features=2,
    )
    base, _ = build_bad_regime_archetype_feature_frame(frame, definitions, config=config)
    changed = frame.copy()
    cutoff = changed["timestamp"].sort_values().unique()[80]
    future_mask = changed["timestamp"] > cutoff
    changed.loc[future_mask, "asset_minus_mkt_oi_1d"] += 1000.0
    changed.loc[future_mask, "fund_high_neg_mom_self_z_5h"] -= 1000.0
    perturbed, _ = build_bad_regime_archetype_feature_frame(changed, definitions, config=config)
    compare_mask = frame["timestamp"] <= cutoff
    col = "badregime__leverage_crowding_archetype_score"
    np.testing.assert_allclose(
        base.loc[compare_mask, col].to_numpy(),
        perturbed.loc[compare_mask, col].to_numpy(),
        equal_nan=True,
    )


def test_bad_regime_archetype_min_resolved_features_neutralizes_weak_archetype():
    frame = _synthetic_panel()
    definitions = {
        "network_concentration_archetype": {
            "evidence_score": 1.0,
            "mechanism_channel": "covariance_network_concentration",
            "top_features": ["asset_minus_mkt_oi_1d", "missing_network_feature"],
            "deployable_features": ["network_rewiring_score"],
        }
    }
    config = BadRegimeArchetypeFeatureConfig(
        trailing_window=48,
        min_periods=12,
        min_resolved_features=2,
    )
    features, diagnostics = build_bad_regime_archetype_feature_frame(frame, definitions, config=config)
    assert diagnostics["archetypes"]["network_concentration_archetype"]["active"] is False
    assert float(features["badregime__network_concentration_archetype_score"].max()) == 0.0
    assert float(features["network_rewiring_score"].max()) == 0.0
