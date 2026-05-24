import numpy as np
import pandas as pd

from extreme_price_movements.config import is_non_portable_feature_key
from extreme_price_movements.feature_family_registry import get_feature_family
from extreme_price_movements.features_residual import (
    RESIDUAL_FEATURE_KEYS,
    add_residual_features,
)


def test_add_residual_features_emits_top_40_and_legacy_aliases():
    idx = pd.date_range("2024-01-01", periods=620, freq="h", tz="UTC")
    cols = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT"]
    rng = np.random.default_rng(123)
    base_keys = [
        "ret4h",
        "ret24h",
        "ret48h",
        "rv_24h",
        "vol_z",
        "rvol_z",
        "amihud_z",
        "liquidity_ratio",
        "dist_vwap_norm",
        "dist_ema_fast",
        "trend_pct",
        "rsi",
        "flow_persistence",
        "excess_6h",
        "atr_expansion",
        "coherence_24",
        "overext",
        "blowoff_risk",
        "exh_qual",
        "spike_score",
        "grind_score",
        "chop_score",
        "basis_pct_z",
        "funding_per_hour_z",
        "fund_abs_z_14d",
        "basis_fund_div_z",
        "oi_chg_8h",
        "oi_rel_vol_8h",
        "squeeze_prob",
        "ob_book_pressure_l10",
        "ob_spread_z_24h",
        "ob_depth_z_25bps",
        "ob_imb_10bps",
        "volume_price_corr_10h",
        "path_efficiency_24",
        "entry_quality_composite",
    ]
    feats = {
        key: pd.DataFrame(
            rng.normal(size=(len(idx), len(cols))).astype(np.float32),
            index=idx,
            columns=cols,
        )
        for key in base_keys
    }
    mkt_gates = pd.DataFrame(
        {
            "mkt_trend": rng.normal(size=len(idx)),
            "mkt_rv": rng.random(len(idx)) + 0.1,
        },
        index=idx,
    )

    skip = add_residual_features(
        feats,
        mkt_gates,
        {"primary_benchmark": "BTC/USDT", "market_basket": ["BTC/USDT", "ETH/USDT"]},
    )

    assert set(RESIDUAL_FEATURE_KEYS).issubset(feats)
    assert set(RESIDUAL_FEATURE_KEYS).issubset(skip)
    assert "rsi_z" in feats
    assert "dist_vwap_resid" in feats
    for key in RESIDUAL_FEATURE_KEYS:
        assert feats[key].shape == (len(idx), len(cols))
        assert np.isfinite(feats[key].tail(20).to_numpy()).all(), key
        assert get_feature_family(key).value == "already_standardized"
        assert not is_non_portable_feature_key(key)
