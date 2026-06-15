import numpy as np
import pandas as pd

from extreme_price_movements.config import (
    CFG,
    SPREAD_PROXY_FEATURE_KEYS,
    is_non_portable_feature_key,
)
from extreme_price_movements.feature_family_registry import (
    FeatureFamily,
    get_feature_family,
)
from extreme_price_movements.features import (
    add_regime_gates,
    compute_features_hourly,
    compute_market_features,
)
from extreme_price_movements.training_utils import (
    get_base_feature_keys,
    get_meta_feature_keys,
)


def _spread_proxy_panel(rows: int = 96):
    idx = pd.date_range("2026-01-01", periods=rows, freq="h", tz="UTC")
    cols = ["AAA/USD:USD", "BBB/USD:USD"]
    t = np.arange(rows, dtype=np.float32)[:, None]
    offsets = np.array([0.0, 3.0], dtype=np.float32)[None, :]
    close = 100.0 + offsets + 0.08 * t + np.sin(t / 5.0)
    open_ = close * (1.0 + 0.0007 * np.cos(t / 4.0 + offsets))
    high = np.maximum(open_, close) * (1.0 + 0.0015 + 0.0002 * np.sin(t / 3.0))
    low = np.minimum(open_, close) * (1.0 - 0.0012 - 0.0002 * np.cos(t / 6.0))
    volume = 1000.0 + 2.0 * t + offsets
    panel = {
        "open": pd.DataFrame(open_, index=idx, columns=cols),
        "high": pd.DataFrame(high, index=idx, columns=cols),
        "low": pd.DataFrame(low, index=idx, columns=cols),
        "close": pd.DataFrame(close, index=idx, columns=cols),
        "volume": pd.DataFrame(volume, index=idx, columns=cols),
    }
    gates = add_regime_gates(
        compute_market_features(panel, cols, trend_sma_hours=24),
        gate_vol_lookback_hours=24,
        gate_trend_thr=0.02,
    )
    return panel, gates


def test_spread_proxy_group_is_registered_for_base_and_meta():
    assert CFG["spread_proxy_features"] == SPREAD_PROXY_FEATURE_KEYS

    base_keys = set(get_base_feature_keys("long", CFG))
    meta_keys = set(get_meta_feature_keys("clf", CFG))

    for key in SPREAD_PROXY_FEATURE_KEYS:
        assert key in base_keys
        assert key in meta_keys
        assert not is_non_portable_feature_key(key)
        assert get_feature_family(key) == FeatureFamily.ALREADY_STANDARDIZED


def test_compute_features_hourly_emits_spread_proxy_robust_features():
    panel, gates = _spread_proxy_panel()
    cfg = dict(CFG)
    cfg.update(
        {
            "feature_portability_mode": "legacy",
            "feature_transform_cache_enabled": False,
            "feature_causal_transform_state_enabled": False,
            "live_causal_transform_state_enabled": False,
            "live_lgbm_mask_feature_fast_path_enabled": False,
        }
    )

    feats, idx, cols = compute_features_hourly(
        panel,
        gates,
        cfg,
        requested_feature_keys=SPREAD_PROXY_FEATURE_KEYS,
    )

    assert list(idx) == list(panel["close"].index)
    assert cols == list(panel["close"].columns)
    assert set(SPREAD_PROXY_FEATURE_KEYS).issubset(feats)

    for key in SPREAD_PROXY_FEATURE_KEYS:
        frame = feats[key]
        assert frame.shape == panel["close"].shape
        assert frame.dtypes.eq(np.float32).all()
        assert np.isfinite(frame.iloc[20:].to_numpy(dtype=np.float32)).all(), key
