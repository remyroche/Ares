import numpy as np
import pandas as pd

from extreme_price_movements.diagnostics.concentration_market import (
    build_concentration_market_diagnostics,
    build_structural_break_diagnostics,
)


def _rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2025-01-01 01:00:00",
                    "2025-01-01 02:00:00",
                    "2025-01-01 03:00:00",
                    "2025-01-02 01:00:00",
                ],
                utc=True,
            ),
            "symbol": ["BTC", "BTC", "ETH", "SOL"],
            "side": ["long", "long", "short", "short"],
            "setup": ["breakout", "breakout", "fade", "fade"],
            "model": ["m1", "m1", "m2", "m2"],
            "prediction": [0.1, 0.2, 0.3, 0.4],
            "feature_a": [1.0, 1.0, 0.0, 1.0],
            "feature_b": [0.0, 0.0, 1.0, 0.0],
            "embedding_0": [1.0, 1.0, 0.0, 1.0],
            "embedding_1": [0.0, 0.0, 1.0, 0.0],
        }
    )


def _returns() -> pd.DataFrame:
    timestamps = pd.date_range("2025-01-01", periods=4, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": list(timestamps) * 3,
            "symbol": ["BTC"] * 4 + ["ETH"] * 4 + ["SOL"] * 4,
            "return": [0.01, 0.02, 0.03, 0.04] * 2 + [-0.01, -0.02, -0.03, -0.04],
        }
    )


def test_daily_concentration_and_market_summary() -> None:
    state = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2025-01-01 00:00:00", "2025-01-01 12:00:00", "2025-01-02 00:00:00"],
                utc=True,
            ),
            "atr_pct": [0.02, 0.04, 0.01],
            "rv": [0.10, 0.30, 0.20],
            "dispersion": [0.2, 0.4, 0.3],
            "avg_pairwise_corr": [0.5, 0.7, 0.6],
            "pc1_variance_share": [0.8, 0.6, 0.7],
            "trend_efficiency": [0.3, 0.5, 0.4],
            "wick_ratio": [0.1, 0.3, 0.2],
            "volume_z": [1.0, 3.0, 2.0],
            "btc_dominance": [0.5, 0.6, 0.7],
            "btc_return": [0.01, 0.03, 0.02],
            "eth_return": [0.02, 0.04, 0.03],
        }
    )
    result = build_concentration_market_diagnostics(
        _rows(),
        market_returns=_returns(),
        market_state=state,
        feature_columns=["feature_a", "feature_b"],
        embedding_columns=["embedding_0", "embedding_1"],
    )

    first = result.daily.iloc[0]
    assert len(result.daily) == 2
    assert first["n_rows"] == 3
    assert first["n_symbols"] == 2
    assert np.isclose(first["symbol_hhi"], 5.0 / 9.0)
    assert np.isclose(first["symbol_effective_count"], 9.0 / 5.0)
    assert np.isclose(first["symbol_top1_share"], 2.0 / 3.0)
    assert first["symbol_top3_share"] == 1.0
    assert np.isclose(first["same_side_share"], 2.0 / 3.0)
    assert np.isclose(first["same_setup_share"], 2.0 / 3.0)
    assert np.isclose(first["same_model_share"], 2.0 / 3.0)
    assert np.isclose(first["same_entry_hour_share"], 1.0 / 3.0)
    assert np.isclose(first["average_feature_cosine"], 1.0 / 3.0)
    assert np.isclose(first["average_embedding_cosine"], 1.0 / 3.0)
    assert np.isclose(first["average_pairwise_trade_return_correlation"], 1.0)
    assert first["n_asset_return_pairs"] == 1
    assert np.isclose(first["market_atr"], 0.03)
    assert np.isclose(first["market_rv"], 0.20)
    assert np.isclose(first["market_correlation"], 0.60)
    assert np.isclose(first["market_pc1_share"], 0.70)
    assert np.isclose(first["market_wick"], 0.20)
    assert np.isclose(first["market_btc_return"], 0.02)


def test_zero_and_one_row_days_leave_pairwise_metrics_undefined() -> None:
    empty = _rows().iloc[0:0]
    result = build_concentration_market_diagnostics(empty)
    assert result.daily.empty
    assert "symbol_hhi" in result.daily.columns

    one = _rows().iloc[[3]]
    result = build_concentration_market_diagnostics(
        one, feature_columns=["feature_a", "feature_b"], embedding_columns=["embedding_0", "embedding_1"]
    )
    daily = result.daily.iloc[0]
    assert daily["n_rows"] == 1
    assert daily["n_symbols"] == 1
    assert daily["symbol_hhi"] == 1.0
    assert daily["symbol_effective_count"] == 1.0
    assert daily["symbol_top1_share"] == 1.0
    assert np.isnan(daily["average_prediction_similarity"])
    assert np.isnan(daily["average_feature_cosine"])
    assert np.isnan(daily["average_embedding_cosine"])
    assert np.isnan(daily["average_pairwise_trade_return_correlation"])


def test_structural_breaks_use_pooled_median_iqr_and_month_comparisons() -> None:
    daily = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2025-01-01", "2025-01-02", "2025-02-01", "2025-02-02"], utc=True
            ),
            "symbol_hhi": [0.1, 0.1, 0.1, 0.9],
        }
    )
    breaks, months = build_structural_break_diagnostics(
        daily, metrics=["symbol_hhi"], robust_z_threshold=3.5
    )
    exceptional = breaks.loc[breaks["value"] == 0.9].iloc[0]
    assert exceptional["pooled_median"] == 0.1
    assert np.isclose(exceptional["pooled_iqr"], 0.2)
    assert exceptional["is_structural_break"]
    assert len(months) == 2
    february = months.loc[
        months["month"] == pd.Timestamp("2025-02-01", tz="UTC")
    ].iloc[0]
    assert february["previous_month_median"] == 0.1
    assert np.isclose(february["month_to_month_delta"], 0.4)
