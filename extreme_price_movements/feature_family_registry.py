"""Feature Family Registry mapping features to their appropriate normalization policy."""

import logging
from typing import Dict, Set

from extreme_price_movements.intraday_crypto_library import (
    LOCATION_FILTER_COLUMNS,
    INTRADAY_TRIGGER_COLUMNS,
)

logger = logging.getLogger(__name__)

class FeatureFamily:
    ALREADY_STANDARDIZED = "already_standardized"
    RISK_NORMALIZED_CONTINUOUS = "risk_normalized_continuous"
    BOUNDED_GEOMETRY = "bounded_geometry"
    CATEGORICAL_OR_BUCKETED = "categorical_or_bucketed"

FEATURE_FAMILY_REGISTRY: Dict[str, Set[str]] = {
    FeatureFamily.ALREADY_STANDARDIZED: {
        "vol_z", "vol_z24_base", "volatility_zscore", "amihud_z", "gap_zscore",
        "vol_shock_z", "range_zscore", "vol_z24", "vol_z_base", "vol_z_30_calm", "vol_z_4h", "volu_z",
        "vol_regime_z", "vol_regime_z_4d", "trend_overextension_z", "trend_z_t", "convexity_z_t",
        "breakout_z", "z_r_12", "z_r_24", "z_vwap_12", "z_vwap_24", "bb_pos_12", "bb_pos_24",
        "rsi_z", "dist_ema_fast_z", "dist_vwap_norm_z", "flow_persistence_z", "excess_6h_z", "vol_z_z",
        "atr_expansion_z", "coherence_24_z", "accept_surprise", "overext_surprise",
        "blowoff_risk_surprise", "exh_qual_surprise", "trend_pct_base", "trend_pct_resid", "dist_vwap_resid", "dist_ema_fast_resid"
    },
    FeatureFamily.RISK_NORMALIZED_CONTINUOUS: {
        "dist_ema_fast", "dist_ema_slow", "impulse", "trend_strength_4d", "trend_strength_vs_reversion",
        "trend_regime", "trend_t", "convexity_t", "convexity_bis_t", "breakout_t", "impulse_speed", "impulse_acceleration",
        "dist_ema_fast_base", "dist_ema_slow_base", "dist_vwap_norm", "dist_from_high_12h", "dist_from_low_12h",
        "dist_from_high_24h", "dist_from_low_24h", "dist_from_high_48h", "dist_from_low_48h", "pullback_2", "pullback_4",
        "pullback_8", "pullback_12", "pullback_24", "pullback_48", "pullback_72", "pullback_120",
        "donch_dist_2", "donch_dist_4", "donch_dist_6", "donch_dist_8", "donch_dist_12", "donch_dist_24", "donch_dist_48", "donch_dist_72", "donch_dist_120",
        "atr_pct_base", "atr_pct", "rv_24h", "rv_12h", "rv_8h", "rv_6h", "rv_4h", "rv_2h",
        "ret1h", "ret2h", "ret3h", "ret4h", "ret5h", "ret6h", "ret8h", "ret10h", "ret12h", "ret16h", "ret20h", "ret24h", "ret28h", "ret48h", "ret72h", "ret120h",
        "ret_mean", "ret_max", "ret_min", "rv_mean", "rv_max", "rv_min", "ret_pct5_24h", "ret_pct95_24h", "tail_risk_score"
    },
    FeatureFamily.BOUNDED_GEOMETRY: {
        "close_location_in_bar", "wick_body_ratio", "wick_ratio", "wick_ratio_4h_max", "body_ratio_15m", "body_ratio",
        "clv", "clv_mean_2", "clv_mean_4", "clv_mean_24", "clv_t", "clv_collapse", "clv_pullback",
        "rejection_proxy", "vol_compression", "vol_compression_ratio", "bidirectional_range_ratio",
        "impulse_ratio_24", "impulse_ratio_12", "vol_expansion_ratio", "atr_expansion", "atr_spike_ratio", "atr_ratio_short_long", "range_decay", "micro_range_decay", "range_last_3bars_impulse_range",
        "rsi", "rsi_base", "rsi_lag1", "ker_10", "ker_16", "ker_24", "path_efficiency_12", "path_efficiency_24", "hurst_proxy_24",
        "volatility_asymmetry", "down_up_vol_ratio_8", "down_up_vol_ratio_24", "tail_against", "asym_ratio", "asym_ft",
        "choppiness_index_20"
    },
    FeatureFamily.CATEGORICAL_OR_BUCKETED: {
        "liq_state", "sin_hod", "cos_hod", "sin_dow", "cos_dow", "session_progress",
        "G_VOL_LIQ_GT1", "G_VOL_LIQ_GT2", "G_VOL_LIQ_GT3", "G_LIQ_GOOD", "G_LIQ_GREAT", "G_LIQ_EXCEL",
        "G_EXH_EFFORT", "G_EXH_GIVEBACK", "G_EXH_TAIL_FAIL", "G_MR_SPIKE", "G_TF_GRIND", "G_TF_TREND", "G_MR_TAIL",
        "G_META_EXH", "G_META_TF_QUAL", "G_META_MR_QUAL", "G_META_AMBIG",
        "is_trending", "is_ranging", "is_high_vol_regime", "is_low_vol_regime", "trend_bin3", "vol_regime_switch_12h", "trend_regime_switch_12h",
        "trend_age_hours", "trend_regime_duration_4d", "higher_highs_count_48h", "rejection_bar_count",
        "adx_7_gt25", "adx_10_gt25", "adx_14_gt25", "accept_bin3", "meta_alignment", "mtf_divergence", "vol_price_diverge",
        # Boolean location and trigger features - MUST NOT be transformed
        *set(LOCATION_FILTER_COLUMNS),
        *set(INTRADAY_TRIGGER_COLUMNS),
    }
}

# Add matching rules for dynamic features (patterns rather than exact names)
# If a feature name starts with or ends with these patterns, it's classified accordingly.
FEATURE_FAMILY_PATTERNS = {
    FeatureFamily.ALREADY_STANDARDIZED: [
        lambda x: x.endswith("_z"),
        lambda x: x.endswith("_zscore"),
        lambda x: x.endswith("_pct"),
        lambda x: x.endswith("_rank"),
        lambda x: x.startswith("cs_rank_"),
        lambda x: x.startswith("ts_pct_") or x.endswith("_ts_pct")
    ],
    FeatureFamily.CATEGORICAL_OR_BUCKETED: [
        lambda x: x.startswith("G_"),
        lambda x: "bin3" in x,
        lambda x: "state" in x,
        lambda x: "session" in x,
        lambda x: "hod" in x,
        lambda x: "dow" in x,
        lambda x: x.endswith("_bin")
    ],
    FeatureFamily.BOUNDED_GEOMETRY: [
        lambda x: "ratio" in x and "rv_ratio" not in x,
        lambda x: "clv" in x,
        lambda x: x.startswith("rsi"),
        lambda x: "entropy" in x,
        lambda x: x.startswith("ker_"),
        lambda x: x.startswith("path_efficiency"),
        lambda x: "choppiness" in x,
        lambda x: "index" in x and "regime" not in x
    ]
}

def get_feature_family(feature_name: str) -> str:
    """Determine the transform family for a given feature name.

    1. Check exact match in registry.
    2. Check pattern match.
    3. Fallback to risk_normalized_continuous.
    """
    for family, features in FEATURE_FAMILY_REGISTRY.items():
        if feature_name in features:
            return family

    # Pattern matching
    for family, patterns in FEATURE_FAMILY_PATTERNS.items():
        for pattern in patterns:
            # Need to be careful with 'pct'. Things like 'atr_pct' might be Risk Normalized,
            # while 'range_pct' might be risk normalized. Let's make sure _pct doesn't override registry.
            if pattern(feature_name):
                # Exceptions for _pct: atr_pct, range_pct, gap_pct, body_pct are risk normalized, not already standardized in [0, 1] necessarily
                if family == FeatureFamily.ALREADY_STANDARDIZED and feature_name.endswith("_pct"):
                    if any(x in feature_name for x in ["atr_pct", "range_pct", "gap_pct", "body_pct", "trend_pct"]):
                        return FeatureFamily.RISK_NORMALIZED_CONTINUOUS
                return family

    # If not found, default to Risk Normalized Continuous and warn
    # Only warn once per feature to avoid spam, we'll use a global set
    if not hasattr(get_feature_family, "_warned_features"):
        get_feature_family._warned_features = set()

    if feature_name not in get_feature_family._warned_features:
        logger.debug(f"Feature '{feature_name}' not in registry. Defaulting to {FeatureFamily.RISK_NORMALIZED_CONTINUOUS}.")
        get_feature_family._warned_features.add(feature_name)

    return FeatureFamily.RISK_NORMALIZED_CONTINUOUS
