from pathlib import Path

import pandas as pd

from scripts.run_c3el_fallback_quality_filter import (
    _evaluate_rules,
    _feature_columns,
    _join_labels_and_features,
)


def test_fallback_quality_filter_excludes_counterfactual_columns(tmp_path: Path) -> None:
    ts = pd.Timestamp("2026-06-15 12:00:00", tz="UTC")
    labels = tmp_path / "labels.csv"
    features = tmp_path / "features.parquet"

    pd.DataFrame(
        {
            "timestamp": [ts, ts + pd.Timedelta(hours=1), ts + pd.Timedelta(hours=2)],
            "strategy_id": ["short_asset_a"] * 3,
            "action_value": [0.0, 0.0, 0.0],
            "delta_full_J": [100.0, -50.0, 75.0],
            "delta_immediate_J": [10.0, -5.0, 8.0],
            "direct_delta_net_pnl": [90.0, -60.0, 70.0],
        }
    ).to_csv(labels, index=False)
    pd.DataFrame(
        {
            "timestamp": [ts, ts + pd.Timedelta(hours=1), ts + pd.Timedelta(hours=2)],
            "strategy_id": ["short_asset_a"] * 3,
            "multiplier": [0.0, 0.0, 0.0],
            "deployable_state": [1.0, 3.0, 2.0],
            "delta_full_net_pnl": [999.0, -999.0, 999.0],
            "p_intervene": [0.9, 0.8, 0.95],
        }
    ).to_parquet(features, index=False)

    joined = _join_labels_and_features(labels, features)
    cols = _feature_columns(joined, min_non_null=2)

    assert "deployable_state" in cols
    assert "p_intervene" in cols
    assert "delta_full_net_pnl" not in cols


def test_evaluate_rules_prefers_positive_exact_state_filter() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-06-15", periods=4, freq="h", tz="UTC"),
            "strategy_id": ["short_asset_a"] * 4,
            "deployable_state": [1.0, 2.0, 3.0, 4.0],
            "delta_full_J": [-100.0, 75.0, 90.0, -50.0],
            "delta_immediate_J": [-50.0, 20.0, 25.0, -10.0],
            "direct_delta_net_pnl": [-100.0, 80.0, 95.0, -60.0],
            "exact_positive_e50": [False, True, True, False],
        }
    )

    out = _evaluate_rules(
        frame,
        [{"feature": "deployable_state", "direction": "ge", "threshold": 2.0}],
        min_keep=2,
        min_positive_rate=0.5,
    )

    best = out.iloc[0]
    assert best["keep_count"] == 3
    assert best["positive_e50_count"] == 2
    assert best["delta_full_J_sum"] == 115.0
