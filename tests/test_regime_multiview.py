from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.regime_multiview import (
    MultiViewRegimeConfig,
    build_causal_multiview_regime_features,
)


def _hourly_frame(rows: int = 420) -> pd.DataFrame:
    index = np.arange(rows, dtype=float)
    return pd.DataFrame(
        {
            "source_utc": pd.date_range("2024-01-01", periods=rows, freq="h", tz="UTC"),
            "breadth": np.sin(index / 9.0) + index / 200.0,
            "funding": np.cos(index / 13.0),
            "volume": 1_000.0 + 50.0 * np.sin(index / 7.0),
            "amihud_proxy": 0.002 + 0.0002 * np.cos(index / 11.0),
        }
    )


def test_generates_requested_hourly_views_with_observable_dependence_and_liquidity() -> None:
    features, metadata = build_causal_multiview_regime_features(_hourly_frame())

    assert metadata["horizons"] == ["1h", "3h", "6h", "12h", "24h", "48h", "72h", "168h"]
    assert "mv__breadth__robust_z_24h" in features
    assert "mv__funding__acceleration_per_hour2_6h" in features
    assert "mv__volume__realized_vol_48h" in features
    assert "mv__liquidity__volume__stress_24h" in features
    assert "mv__liquidity__amihud_proxy__stress_24h" in features
    assert "mv__dependence__corr_frobenius_shift_168h" in features
    assert "mv__dependence__covariance_frobenius_shift_168h" in features
    assert features["mv__dependence__effective_rank_168h"].notna().any()
    assert features.dtypes.eq(np.dtype("float32")).all()


def test_future_mutation_cannot_change_prior_causal_outputs() -> None:
    base = _hourly_frame()
    changed = base.copy()
    changed.loc[300:, ["breadth", "funding", "volume", "amihud_proxy"]] *= -17.0

    before, _ = build_causal_multiview_regime_features(base)
    after, _ = build_causal_multiview_regime_features(changed)

    pd.testing.assert_frame_equal(before.loc[:299], after.loc[:299])


def test_gap_starts_a_new_segment_and_prevents_lag_bridge() -> None:
    frame = _hourly_frame(240).drop(index=120).reset_index(drop=True)
    features, metadata = build_causal_multiview_regime_features(frame)
    first_after_gap = frame.index[frame["source_utc"].eq(pd.Timestamp("2024-01-06 01:00:00+00:00"))][0]

    assert metadata["segment_count"] == 2
    assert pd.isna(features.loc[first_after_gap, "mv__breadth__delta_1h"])
    assert pd.isna(features.loc[first_after_gap, "mv__dependence__mean_abs_corr_1h"])


def test_adds_15m_only_when_input_cadence_supports_it() -> None:
    frame = _hourly_frame(1_400)
    frame["source_utc"] = pd.date_range("2024-01-01", periods=len(frame), freq="15min", tz="UTC")
    features, metadata = build_causal_multiview_regime_features(frame)

    assert metadata["horizons"][0] == "15m"
    assert "mv__breadth__delta_15m" in features
    assert "mv__dependence__eig1_share_15m" in features


def test_rejects_outcome_and_post_entry_inputs() -> None:
    frame = _hourly_frame(32)
    frame["target__future_net_ev"] = 1.0

    with pytest.raises(ValueError, match="forbidden"):
        build_causal_multiview_regime_features(
            frame,
            config=MultiViewRegimeConfig(feature_columns=("breadth", "target__future_net_ev")),
        )
