import numpy as np
import pandas as pd

from extreme_price_movements.config import is_non_portable_feature_key
from extreme_price_movements.config import CFG, enable_perp_feature_keys
from extreme_price_movements.perp_features import compute_features


def test_perp_features_use_fractional_funding_oi_and_soft_scores():
    idx = pd.date_range("2024-01-01", periods=800, freq="h", tz="UTC")
    rng = np.random.default_rng(1)
    perp = pd.Series(
        100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, len(idx)))), index=idx
    )
    spot = perp * (1.0 + rng.normal(0.0, 0.001, len(idx)))
    funding = pd.Series(
        np.repeat(rng.normal(0.0001, 0.0002, len(idx) // 8 + 1), 8)[: len(idx)],
        index=idx,
    )
    open_interest = pd.Series(rng.lognormal(10.0, 0.2, len(idx)), index=idx)
    volume = pd.Series(rng.lognormal(8.0, 0.3, len(idx)), index=idx)
    df = pd.DataFrame(
        {
            "funding_rate": funding / 8.0,
            "open_interest": open_interest,
            "perp_price": perp,
            "spot_price": spot,
            "mark_price": perp * 1.0001,
            "volume": volume,
            "quote_volume": volume * perp,
            "close": perp,
        }
    )

    out = compute_features(df)
    keys = [
        "basis",
        "basis_frac",
        "basis_frac_z_14d",
        "basis_frac_rank_30d",
        "funding_per_hour",
        "funding_rank_30d",
        "oi_chg_z_2h",
        "oi_chg_2h",
        "oi_vel_2h",
        "leverage_build",
        "leverage_build_score",
        "unwind",
        "unwind_score",
        "squeeze_prob",
    ]

    assert np.isfinite(out[keys].tail(50).to_numpy()).all()
    assert out["basis"].abs().max() < 0.01
    for key in (
        "leverage_build",
        "leverage_build_score",
        "unwind",
        "unwind_score",
        "squeeze_prob",
    ):
        assert out[key].dropna().between(0.0, 1.0).all()


def test_normalized_perp_keys_are_portable_and_raw_diagnostics_are_not():
    portable = [
        "asset_atr_level_pct",
        "asset_vol_level_pct",
        "asset_atr_level",
        "asset_vol_level",
        "vol_state",
        "funding_per_hour",
        "funding_phase_sin",
        "fund_hours_to_next",
        "oi_chg_z_2h",
        "oi_chg_2h_robust_z",
        "oi_chg_4h_robust_z",
        "oi_chg_8h_robust_z",
        "leverage_build_score",
        "innovation_z_x_zr_3h",
        "oi_rel_vol_2h",
        "oi_rel_vol_4h",
        "oi_rel_vol_8h",
        "dist_vwap_norm",
        "dist_vwap_12_atr",
        "dist_vwap_24_atr",
        "dist_vwap_96_atr",
        "trapped_longs_12",
        "vwap_zone_1d_atr",
        "dist_stack",
        "ob_depth_usd_l10",
        "ob_depth_z_10bps",
    ]
    raw_diagnostics = [
        "mark_price",
        "index_price",
        "mark_vs_index_bps",
        "basis",
        "basis_frac",
        "basis_per_atr",
        "basis_mom_4h",
        "basis_stretch",
        "basis_adjusted_trend_5h",
        "ob_depth_quote_l10",
    ]

    assert not any(is_non_portable_feature_key(key) for key in portable)
    assert all(is_non_portable_feature_key(key) for key in raw_diagnostics)


def test_perp_oi_features_are_finite_after_oi_available_when_basis_has_gaps():
    idx = pd.date_range("2024-01-01", periods=900, freq="h", tz="UTC")
    rng = np.random.default_rng(2)
    perp = pd.Series(
        100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.006, len(idx)))), index=idx
    )
    spot = perp * (1.0 + rng.normal(0.0, 0.001, len(idx)))
    spot.iloc[720:] = np.nan
    funding = pd.Series(rng.normal(0.00001, 0.0001, len(idx)), index=idx)
    open_interest = pd.Series(rng.lognormal(10.0, 0.15, len(idx)), index=idx)
    open_interest.iloc[:100] = np.nan
    open_interest.iloc[350:360] = np.nan
    volume = pd.Series(rng.lognormal(8.0, 0.25, len(idx)), index=idx)
    df = pd.DataFrame(
        {
            "funding_rate": funding,
            "open_interest": open_interest,
            "perp_price": perp,
            "spot_price": spot,
            "mark_price": perp * 1.0001,
            "volume": volume,
            "quote_volume": volume * perp,
            "close": perp,
        }
    )

    out = compute_features(df)
    tail = out.loc[idx[800]:, ["oi_rel_vol_2h", "oi_rel_vol_4h", "oi_rel_vol_8h", "oi_chg_2h_robust_z", "oi_chg_4h_robust_z", "oi_chg_8h_robust_z", "leverage_build_score"]]

    assert np.isfinite(tail.to_numpy()).all()
    assert out["oi_rel_vol_2h"].iloc[:100].isna().all()


def test_perps_config_does_not_inject_reference_basis_features():
    cfg = enable_perp_feature_keys(CFG)
    trainable_keys = set()
    for name in (
        "base_long_feature_keys",
        "base_short_feature_keys",
        "meta_shared_feature_keys",
        "meta_product_feature_keys",
        "meta_reg_feature_keys",
        "meta_clf_feature_keys",
        "meta_mfe_feature_keys",
        "meta_mae_feature_keys",
        "meta_asym_feature_keys",
    ):
        trainable_keys.update(str(k) for k in cfg.get(name, []))

    assert "basis" not in trainable_keys
    assert not any(k.startswith("basis_") for k in trainable_keys)
    assert not is_non_portable_feature_key("innovation_z_x_zr_3h")
    assert not is_non_portable_feature_key("oi_rel_vol_2h")
    assert not is_non_portable_feature_key("dist_stack")
    assert not is_non_portable_feature_key("dist_vwap_norm")
    assert not is_non_portable_feature_key("dist_vwap_24_atr")
    assert not is_non_portable_feature_key("trapped_longs_24")
    assert not is_non_portable_feature_key("vwap_zone_1d_atr")
