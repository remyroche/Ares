import numpy as np
import pandas as pd

from scripts import backfill_complete_july_meta_predictions as july_backfill
from scripts.backfill_complete_july_meta_predictions import (
    _capture_outcomes,
    _store_input_columns,
)


def test_store_input_columns_include_raw_sources_for_meta_aliases():
    columns = _store_input_columns(
        ["__meta_raw__chop_score", "ret1h_G_VOL_0"],
        ["impulse"],
    )

    assert "__meta_raw__chop_score" in columns
    assert "chop_score" in columns
    assert "ret1h_G_VOL_0" in columns
    assert "impulse" in columns


def test_capture_outcomes_does_not_double_count_round_trip_cost():
    capture = pd.DataFrame(
        {
            "capture_net": [0.0125],
            "capture_valid_path": [1.0],
            "first_touch_mae_norm": [0.25],
            "full_path_mae_norm": [0.50],
            "capture_timeout": [0.0],
            "mfe_1r_before_mae_1r": [1.0],
            "mae_1r_before_mfe_1r": [0.0],
        }
    )

    outcomes = _capture_outcomes(capture)

    assert outcomes.loc[0, "exec_margin"] == 0.0125
    assert outcomes.loc[0, "ev_after_1pct"] == 0.0125


def test_base_attributions_are_aggregated_by_economic_family():
    class FakeBooster:
        def predict(self, matrix, *, pred_contrib=False):
            assert pred_contrib is True
            assert list(matrix.columns) == [
                "gmm_entropy",
                "custom_funding_signal",
                "custom_ema_slope",
            ]
            return np.array(
                [
                    [0.2, -0.1, 0.3, 0.4],
                    [-0.4, 0.2, 0.1, 0.5],
                ],
                dtype=np.float32,
            )

    class FakeModel:
        booster_ = FakeBooster()

        def predict(self, matrix):
            return np.array([0.8, 0.6], dtype=np.float32)

    matrix = pd.DataFrame(
        {
            "gmm_entropy": [0.1, 0.2],
            "custom_funding_signal": [1.0, -1.0],
            "custom_ema_slope": [0.5, 0.2],
        }
    )

    scores, attributions = july_backfill._predict_with_family_attributions(
        FakeModel(), matrix
    )

    np.testing.assert_allclose(scores, [0.8, 0.6])
    np.testing.assert_allclose(
        attributions["base_attr_signed__latent_state"], [0.2, -0.4]
    )
    np.testing.assert_allclose(
        attributions["base_attr_signed__derivatives_positioning"], [-0.1, 0.2]
    )
    np.testing.assert_allclose(
        attributions["base_attr_signed__direction_trend"], [0.3, 0.1]
    )
    np.testing.assert_allclose(attributions["base_attr_bias"], [0.4, 0.5])
    np.testing.assert_allclose(
        attributions["base_attr_abs_concentration"], [0.5, 4.0 / 7.0]
    )


def test_exact_hour_matrix_values_can_override_incremental_store(monkeypatch, tmp_path):
    frame = pd.DataFrame(
        {
            "__ts__": [pd.Timestamp("2026-07-15 15:00", tz="UTC")],
            "__symbol__": ["AAA/USD:USD"],
            "side_name": ["long"],
            "feature_a": np.array([2.0], dtype=np.float32),
        }
    )

    def fake_append_store_features(stripped, _root, names):
        out = stripped.copy()
        out[names[0]] = np.float32(1.0)
        return out, {names[0]: 1.0}

    monkeypatch.setattr(
        july_backfill,
        "_append_store_features",
        fake_append_store_features,
    )

    preferred, _ = july_backfill._fill_store_features(
        frame,
        tmp_path,
        ["feature_a"],
        prefer_existing_finite=True,
    )
    store_first, _ = july_backfill._fill_store_features(
        frame,
        tmp_path,
        ["feature_a"],
        prefer_existing_finite=False,
    )

    assert preferred.loc[0, "feature_a"] == 2.0
    assert store_first.loc[0, "feature_a"] == 1.0


def test_hourly_backcast_proxy_is_cost_aware_and_explicitly_diagnostic(
    monkeypatch, tmp_path
):
    rows = pd.DataFrame(
        {
            "__ts__": [pd.Timestamp("2024-01-15 00:00", tz="UTC")],
            "__symbol__": ["BTC/USD:USD"],
            "side_name": ["long"],
            "archetype_policy_key": ["long_test"],
            "__barrier_pct__": [0.02],
        }
    )

    class FakeHourlyStore:
        def __init__(self, _root, *, timeframe):
            assert timeframe == "1h"

        def load(self, symbol, *, columns, start_ts, end_ts):
            assert symbol == "BTC/USD:USD"
            assert columns == ["close"]
            index = pd.date_range(start_ts, end_ts, freq="h")
            return pd.DataFrame(
                {"close": np.array([100.0, 103.0, 103.0], dtype=np.float32)},
                index=index,
            )

    monkeypatch.setattr(
        "extreme_price_movements.data_store.PartitionedOHLCVStore",
        FakeHourlyStore,
    )
    monkeypatch.setattr(
        "scripts.replay_live_signal_predictions._market_data_root",
        lambda *_args, **_kwargs: tmp_path,
    )
    manifest = {
        "overrides": [
            {
                "policy_key": "long_test",
                "tp_r": 10.0,
                "sl_r": 10.0,
                "trail_r": 0.0,
            }
        ]
    }

    outcomes, stats = july_backfill._hourly_close_proxy_outcomes(
        rows,
        feature_root=tmp_path,
        policy_manifest=manifest,
        horizon_hours=2,
        round_trip_cost=0.01,
    )

    assert np.isclose(outcomes.loc[0, "ev_after_1pct"], 0.02, atol=1e-6)
    assert outcomes.loc[0, "timeout"] == 1.0
    assert stats["round_trip_cost"] == 0.01
    assert stats["execution_parity_claim"] is False
