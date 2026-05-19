from __future__ import annotations

from enum import Enum


class FeatureFamily(str, Enum):
    RISK_NORMALIZED_CONTINUOUS = "risk_normalized_continuous"
    ALREADY_STANDARDIZED = "already_standardized"
    BOUNDED_GEOMETRY = "bounded_geometry"
    CATEGORICAL_OR_BUCKETED = "categorical_or_bucketed"


FEATURE_FAMILY_REGISTRY = {
    # 1. TREND: Directional bias and momentum
    "ema20_gt_ema50": "trend",
    "ema50_gt_ema200": "trend",
    "ema50_ema200_spread_atr": "trend",
    "price_lt_ema200": "trend",
    "ema20_slope": "trend",
    "ema20_slope_5h": "trend",
    "ema50_slope": "trend",
    "ema_slope_norm": "trend",
    "slope": "trend",
    "trend_strength_percentile": "trend",
    "ret1h_z": "trend",
    "ret2h": "trend",
    "ret4h": "trend",
    "ret8h": "trend",
    "ret16h": "trend",
    "ret24h": "trend",
    "breakout_t": "trend",
    "pct_breakout_t": "trend",
    # 2. VOLATILITY: Market noise and range intensity
    "realized_volatility_24h": "volatility",
    "rv_2h": "volatility",
    "rv_4h": "volatility",
    "rv_8h": "volatility",
    "rv_24h": "volatility",
    "vol_z": "volatility",
    "vol_z_4h": "volatility",
    "rvol_z": "volatility",
    "atr_change_rate": "volatility",
    "atr_pct_change": "volatility",
    "atr_slope": "volatility",
    "true_range_percentile": "volatility",
    "prior_volatility": "volatility",
    "vol_regime_z": "volatility",
    "volatility_autocorr_48": "volatility",
    "volatility_ratio_short_long": "volatility",
    "variance_ratio_10_48": "volatility",
    # 3. CURVATURE: Acceleration and non-linear changes
    "accel": "curvature",
    "accel_5h": "curvature",
    "momentum_accel": "curvature",
    "trend_acceleration": "curvature",
    # 4. LOCATION: Position relative to structural anchors (Distances and Positions)
    "dist_ema20_atr": "location",
    "dist_ema50_atr": "location",
    "dist_ema200_atr": "location",
    "dist_ma100_atr": "location",
    "dist_vwap_atr": "location",
    "dist_vwap_norm": "location",
    "dist_weekly_vwap": "location",
    "dist_prior_day_high": "location",
    "dist_prior_day_low": "location",
    "dist_rolling_7d_high": "location",
    "dist_local_swing": "location",
    "dist_range_mid_atr": "location",
    "distance_to_ema": "location",
    "pullback_depth": "location",
    "zscore_price_50": "location",
    "zscore_price_200": "location",
    "loc_ema_stack_pos_24": "location",
    "loc_ema_stack_pos_48": "location",
    "loc_vwap_dev_z_24": "location",
    "loc_vwap_dev_z_48": "location",
    "loc_range_pos_24": "location",
    "loc_range_pos_48": "location",
    "loc_prior_bar_pos_24": "location",
    "loc_prior_bar_pos_48": "location",
    "loc_swing_range_pos_24": "location",
    "loc_swing_range_pos_48": "location",
    "loc_session_pos_24": "location",
    "loc_session_pos_48": "location",
    "loc_initial_balance_pos_24": "location",
    "loc_initial_balance_pos_48": "location",
    "loc_prev_day_range_pos_24": "location",
    "loc_prev_day_range_pos_48": "location",
    "loc_prev_week_range_pos_24": "location",
    "loc_prev_week_range_pos_48": "location",
    "loc_bb_channel_pos_24": "location",
    "loc_bb_channel_pos_48": "location",
    "loc_pullback_depth_24": "location",
    "loc_pullback_depth_48": "location",
    "loc_pivot_ladder_pos_24": "location",
    "loc_pivot_ladder_pos_48": "location",
    "sin_hod": "location",
    "sin_dow": "location",
    # 5. COMPRESSION: Tightness of ranges and inefficiency
    "bollinger_band_width": "compression",
    "atr_compression_ratio": "compression",
    "compression_ratio": "compression",
    "compression_score": "compression",
    "rolling_range_20": "compression",
    "atr_percentile": "compression",
    "range_24h_pct": "compression",
    "range_expansion_ratio": "compression",
    "efficiency_ratio_20": "compression",
    "choppiness_index_20": "compression",
    "path_efficiency_12": "compression",
    "path_efficiency_24": "compression",
    # 6. VOL_OF_VOL: Stability of the volatility regime
    "volatility_of_volatility_48": "vol_of_vol",
    "vov_fast_slow_ratio": "vol_of_vol",
    "vov_mad_20": "vol_of_vol",
    "vov_ratio": "vol_of_vol",
    "regime_stability_24h": "vol_of_vol",
    # 7. PERSISTENCE/DWELL: Dwell times and state duration
    "bars_in_high_vol_state_log_norm": "persistence_dwell",
    "bars_outside_ema20_atr_band_log_norm": "persistence_dwell",
    "bars_since_ema20_ema50_cross_log_norm": "persistence_dwell",
    # 8. STRUCTURE: Complexity, entropy, and asymmetry
    "direction_entropy_20": "structure",
    "shannon_entropy_ret_16": "structure",
    "spectral_entropy_ret_24": "structure",
    "perm_entropy_ret_24": "structure",
    "return_autocorr_48": "structure",
    "up_down_semivol_ratio_tanh": "structure",
    "up_down_return_mass_ratio_tanh": "structure",
    "tail_asymmetry_q90_q10_atr_norm": "structure",
    "hurst_exponent_ret_24": "structure",
    "volume_percentile": "structure",
    "volume_trend_48": "structure",
    "volume_autocorr_48": "structure",
    "volume_zscore_48h": "structure",
    "volume_entropy_12": "structure",
    "volume_entropy_24": "structure",
    "prior_range": "structure",
    # 9. KALMAN / STATE-SPACE: mostly portable only after dynamic normalization
    "kalman_price": "volatility",
    "kf_score_mean": "volatility",
    "kf_score_rm24_mean": "volatility",
    "kf_atr_mean": "volatility",
    "kf_vol_ratio_mean": "volatility",
    "kf_ret1h_mean": "volatility",
    "kf_innov_var": "volatility",
    "kf_state_uncertainty": "volatility",
    "kf_snr_est": "volatility",
    "price_state_slope_1h": "trend",
    "price_state_slope_6h": "trend",
    "price_state_slope_ratio_1h_6h": "trend",
    "state_uncertainty_1h": "volatility",
    "vol_state_slope_1h": "volatility",
    "realized_vol_minus_vol_state": "volatility",
    "log_volume_state_1h": "volatility",
    "volume_state_slope_1h": "volatility",
    "vol_state_1h": "volatility",
    "short_vol_state_over_long_vol_state": "volatility",
    "rolling_std(price_innovation)": "volatility",
    "price_slope_x_volume_surprise": "volatility",
    "vol_state_x_volume_state": "volatility",
    "volume_surprise_vs_state": "structure",
    # 10. TREND STACK / Z-SPACE COMPOSITES
    "zr_3h": "structure",
    "zr_6h": "structure",
    "zr_12h": "structure",
    "zr_1h_minus_zr_6h": "structure",
    "zr_3h_minus_zr_12h": "structure",
    "zr_6h_minus_zr_24h": "structure",
    "trend_stack_3_6_12": "trend",
    "trend_stack_6_12_24": "trend",
    "trend_dispersion_1_3_6": "volatility",
    "trend_dispersion_3_6_12": "volatility",
    "innovation_z_x_zr_1h": "curvature",
    "innovation_z_x_zr_3h": "curvature",
    "zr_1h_x_volume_z_24h": "volatility",
    "zr_3h_x_volume_z_24h": "volatility",
    "zr_6h_x_volume_z_48h": "volatility",
    "zr_6h_x_range_z_24h": "volatility",
    "zr_12h_x_range_z_48h": "volatility",
    "trend_alignment_1_3_6": "categorical",
    "trend_alignment_3_6_12": "categorical",
    "trend_alignment_6_12_24": "categorical",
}


_EXPLICIT_FAMILY_PREFIXES = (
    "kf_",
    "zr_",
    "trend_stack_",
    "trend_dispersion_",
    "trend_alignment_",
    "innovation_z_x_",
    "vol_state_",
    "volume_state_",
)
_EXPLICIT_FAMILY_NAMES = {
    "kalman_price",
    "price_innovation_z",
    "price_minus_state_z",
    "rolling_std(price_innovation)",
    "kalman_gain_1h",
    "state_uncertainty_1h",
    "realized_vol_minus_vol_state",
    "log_volume_state_1h",
    "short_vol_state_over_long_vol_state",
    "volume_surprise_vs_state",
    "price_slope_x_volume_surprise",
    "vol_state_x_volume_state",
}


def get_feature_family(feature_name: str) -> FeatureFamily:
    name = str(feature_name or "")
    base_name = name
    if base_name in {"price_minus_state_z", "price_innovation_z"}:
        return FeatureFamily.ALREADY_STANDARDIZED
    if base_name == "kalman_gain_1h":
        return FeatureFamily.BOUNDED_GEOMETRY
    if "_ts_" in base_name:
        return FeatureFamily.CATEGORICAL_OR_BUCKETED
    if any(token in base_name for token in ("==", "top", "bot", "band")):
        return FeatureFamily.CATEGORICAL_OR_BUCKETED

    registry_family = FEATURE_FAMILY_REGISTRY.get(base_name)
    if registry_family in {
        "trend",
        "volatility",
        "curvature",
        "compression",
        "vol_of_vol",
        "persistence_dwell",
        "structure",
    }:
        return FeatureFamily.RISK_NORMALIZED_CONTINUOUS
    if registry_family == "location":
        return FeatureFamily.BOUNDED_GEOMETRY
    if registry_family == "categorical":
        return FeatureFamily.CATEGORICAL_OR_BUCKETED

    if base_name in _EXPLICIT_FAMILY_NAMES or base_name.startswith(
        _EXPLICIT_FAMILY_PREFIXES
    ):
        raise ValueError(
            f"Feature '{base_name}' must be explicitly registered in "
            "FEATURE_FAMILY_REGISTRY or handled in get_feature_family()."
        )

    lowered = base_name.lower()
    if "_self_z_" in lowered or lowered.endswith("_self_z"):
        return FeatureFamily.ALREADY_STANDARDIZED
    if lowered.startswith(("loc_", "dist_", "zscore_")):
        return FeatureFamily.BOUNDED_GEOMETRY
    if any(
        token in lowered
        for token in (
            "pct",
            "ratio",
            "score",
            "entropy",
            "vol",
            "atr",
            "slope",
            "ret",
            "range",
            "compression",
            "trend",
        )
    ):
        return FeatureFamily.RISK_NORMALIZED_CONTINUOUS
    return FeatureFamily.ALREADY_STANDARDIZED
