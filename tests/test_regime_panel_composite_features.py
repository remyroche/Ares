import numpy as np
import pandas as pd

from extreme_price_movements.config import CFG
from extreme_price_movements.features import (
    _add_regime_panel_composite_features,
    _expand_regime_composite_dependencies,
)
from extreme_price_movements.training_utils import get_base_feature_keys, get_meta_feature_keys


def test_regime_panel_composites_are_present_finite_and_broadcast():
    idx = pd.date_range("2026-01-01", periods=12, freq="1h", tz="UTC")
    cols = pd.Index(["A", "B", "C", "D"])
    base = np.arange(48, dtype=np.float32).reshape(12, 4)
    feats = {
        "price_x_oi_1d": pd.DataFrame(base + 1.0, index=idx, columns=cols),
        "price_x_oi_3d": pd.DataFrame(base * 0.5 + 2.0, index=idx, columns=cols),
        "funding_per_hour": pd.DataFrame(np.sin(base), index=idx, columns=cols),
        "oi_1d_x_funding": pd.DataFrame(np.cos(base), index=idx, columns=cols),
        "oi_3d_x_funding": pd.DataFrame(base / 50.0, index=idx, columns=cols),
        "oi_7d_x_funding": pd.DataFrame(base / 100.0, index=idx, columns=cols),
    }
    requested = {
        "xs_dispersion__price_x_oi_1d",
        "xs_std__price_x_oi_3d",
        "q_tail_width__price_x_oi_1d",
        "q_tail_asym__price_x_oi_3d",
        "eig_effective_rank__open_interest",
        "xs_cov_effective_rank__xs_open_interest",
        "state_spectral_eig_lambda1_share",
        "state_spectral_pc1_z",
        "state_spectral_top3_mahalanobis",
    }

    expanded = _expand_regime_composite_dependencies(requested, CFG)
    added = _add_regime_panel_composite_features(feats, expanded, CFG, idx, cols)

    assert requested.issubset(added)
    for key in requested:
        frame = feats[key]
        assert frame.shape == (len(idx), len(cols))
        arr = frame.to_numpy(dtype=np.float32)
        assert np.isfinite(arr).all()
        assert np.allclose(arr, arr[:, :1])


def test_requested_model_regime_keys_route_to_meta_not_base():
    base = set(get_base_feature_keys("long", CFG)) | set(get_base_feature_keys("short", CFG))
    meta = set(get_meta_feature_keys("clf", CFG)) | set(get_meta_feature_keys("reg", CFG))

    for key in [
        "loc_range_pos_24",
        "trend_slope_48h",
        "trend_pct_mkt_resid",
        "bars_in_high_vol_state_log_norm",
        "price_x_oi_1d",
        "funding_per_hour",
        "range_expansion_ratio",
        "efficiency_ratio_20",
    ]:
        assert key in base

    for key in [
        "mkt_ret_eq_4h",
        "xs_dispersion__price_x_oi_3d",
        "q_tail_width__price_x_oi_1d",
        "eig_effective_rank__open_interest",
        "xs_cov_effective_rank__xs_open_interest",
        "state_spectral_eig_lambda1_share",
        "state_spectral_top3_mahalanobis",
        "xasset_mkt_spread_bps",
        "regime_liquidity_score",
    ]:
        assert key in meta
        assert key not in base

    assert "meta_en_x_efficiency" not in meta
