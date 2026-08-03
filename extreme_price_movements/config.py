# Central config. Keep it deterministic and explicit.
from extreme_price_movements.features_gmm_ae import AE_GMM_FEATURE_COLUMNS
from extreme_price_movements.continuation_features import (
    CONTINUATION_COMPOSITE_FEATURE_KEYS,
    CONTINUATION_CROSS_SECTIONAL_FEATURE_KEYS,
    CONTINUATION_FEATURE_GROUPS,
    CONTINUATION_FUNDING_FEATURE_KEYS,
    CONTINUATION_OI_FEATURE_KEYS,
    CONTINUATION_PRICE_FEATURE_KEYS,
    CONTINUATION_REGIME_FEATURE_KEYS,
    CONTINUATION_SIDE_PRICE_FEATURE_KEYS,
    CONTINUATION_SIDE_VOLATILITY_FEATURE_KEYS,
    CONTINUATION_VOLATILITY_FEATURE_KEYS,
    CONTINUATION_VOLUME_FEATURE_KEYS,
)
from extreme_price_movements.market_microstructure_features import (
    NATIVE_L2_CONTINUATION_FEATURE_KEYS,
)
from extreme_price_movements.features_oi import (
    ASSET_MARKET_LIFECYCLE_RESIDUAL_KEYS,
    ASSET_OI_LIFECYCLE_FEATURE_KEYS,
    FUNDING_LIFECYCLE_FEATURE_KEYS,
    FUNDING_PRICE_OI_INTERACTION_FEATURE_KEYS,
    MARKET_FUNDING_REGIME_FEATURE_KEYS,
    MARKET_OI_LIFECYCLE_FEATURE_KEYS,
    MARKET_OI_REGIME_FEATURE_KEYS,
    MARKET_PRICE_OI_STATE_FEATURE_KEYS,
    PRICE_OI_QUADRANT_FEATURE_KEYS,
    PRICE_OI_RECOVERY_FEATURE_KEYS,
    PRICE_OI_STATE_FEATURE_KEYS,
    get_oi_feature_names,
    get_oi_normalized_feature_names,
    get_oi_trading_feature_names,
)
from extreme_price_movements.features_negative_residuals import (
    NEGATIVE_RESIDUAL_CAUSAL_WINDOW_HOURS,
    NEGATIVE_RESIDUAL_COMPOSITE_FEATURE_KEYS,
    NEGATIVE_RESIDUAL_FEATURE_SCHEMA_VERSION,
    NEGATIVE_RESIDUAL_META_FEATURE_KEYS,
    NEGATIVE_RESIDUAL_PROMOTED_META_FEATURE_KEYS,
    NEGATIVE_RESIDUAL_PRIMITIVE_FEATURE_KEYS,
    NEGATIVE_RESIDUAL_TEMPORAL_MECHANISM_FEATURE_KEYS,
    negative_residual_feature_contract,
)
from extreme_price_movements.features_residual import residual_feature_names
from extreme_price_movements.lgbm_archetype_features import (
    RAW_STATE_DIAGNOSTIC_FEATURE_NAMES,
    RAW_STATE_SVD_SUMMARY_FEATURE_NAMES,
)
from extreme_price_movements.market_regime_change_contract import (
    MARKET_REGIME_CHANGE_FEATURE_KEYS,
    MARKET_REGIME_CHANGE_SCHEMA_VERSION,
    MARKET_REGIME_CHANGE_SOURCES,
)
from extreme_price_movements.model_drift_features import MODEL_DRIFT_FEATURE_KEYS
from extreme_price_movements.perp_features import get_perp_feature_names
from extreme_price_movements.residual_event_archetypes import (
    residual_event_feature_names,
    residual_event_market_feature_names,
)
from extreme_price_movements.stage_ii_meta_archetypes import stage_ii_feature_names
from extreme_price_movements.unsupervised_regime_learning.feature_registry import (
    UNSUPERVISED_REGIME_LEARNING_DEFAULTS,
)

# =============================================================================
# CANONICAL Horizons & Buckets - Single Source of Truth
# =============================================================================
# Canonical TBM horizons for current optimization/inference stack.
# Legacy bucket naming (still used for grouping), but strategy_ids are canonical LGBM keys
CANON_BUCKETS = ["MR_long", "MR_short", "TF_long", "TF_short"]
CANON_HORIZONS = [3, 5, 7]  # hours
CANON_CELLS = [f"{b}_H{h}" for b in CANON_BUCKETS for h in CANON_HORIZONS]

# Side-Horizon cells (agnostic to MR/TF distinction)
# Used for TBM optimization and position sizing without structural bucket assumptions
CANON_SIDES = ["long", "short"]
CANON_SIDE_HORIZON_CELLS = [
    f"{side}_H{h}" for side in CANON_SIDES for h in CANON_HORIZONS
]


_PERP_COLLISION_RENAMES = {
    "ret1h": "ret1h_perp",
}
PERP_FEATURE_KEYS = [
    _PERP_COLLISION_RENAMES.get(k, k) for k in get_perp_feature_names()
]
PERP_PRICE_RELATION_FEATURE_KEYS = [
    "mark_price",
    "mark_vs_perp_bps",
    "mark_perp_dislocation",
]
KRAKEN_INDEX_PREMIUM_FEATURE_KEYS = [
    "index_price",
    "canonical_index",
    "premium_index",
    "premium_proxy",
    "mark_vs_index_bps",
    "perp_vs_index_bps",
    "premium_proxy_bps",
    "premium_proxy_z",
    "premium_proxy_mom_8h",
    "mark_index_basis",
    "mark_index_basis_z",
    "perp_index_basis",
    "perp_index_basis_z",
    "premium_mean_reversion_halflife_24h",
]
for _h in CANON_HORIZONS:
    KRAKEN_INDEX_PREMIUM_FEATURE_KEYS.extend(
        [
            f"premium_expansion_speed_{_h}h",
            f"mark_trigger_risk_{_h}h",
            f"mark_trigger_dislocation_{_h}h",
            f"mark_trigger_dislocation_self_z_{_h}h",
        ]
    )
SPOT_FOR_PERPS_BASE_FEATURE_KEYS = [
    "spot_ret1h",
    "spot_ret4h",
    "spot_ret24h",
    "spot_rv_24h",
    "spot_volume_z_24h",
    "spot_quote_volume_z_24h",
    "spot_range_pct_24h",
    "spot_breakout_up_24h",
    "spot_breakout_down_24h",
    "spot_liquidity_sweep_up",
    "spot_liquidity_sweep_down",
    "spot_perp_return_agreement_4h",
    "perp_minus_spot_ret1h",
    "perp_minus_spot_ret4h",
    "perp_minus_spot_ret24h",
]
SPOT_FOR_PERPS_META_FEATURE_KEYS = [
    "basis_pct",
    "basis_pct_z",
    "basis_frac",
    "basis_frac_z_14d",
    "basis_frac_rank_30d",
    "basis_per_atr",
    "basis_mom_4h",
    "basis_fund_div_z",
    "spot_leads_perp_1h",
    "spot_perp_vol_ratio_24h",
    "spot_perp_volume_ratio_24h",
    "spot_available",
]
PERP_EVENT_RISK_FEATURE_KEYS = [
    "fund_hours_to_next",
    "fund_hours_since_last",
    "funding_phase_sin",
    "funding_phase_cos",
    "fund_next_event_proximity_5h",
    "fund_next_event_proximity_10h",
    "liq_buffer_long_mark_frac",
    "liq_buffer_short_mark_frac",
    "liq_buffer_atr",
    "liq_stop_safety_long_atr",
    "liq_stop_safety_short_atr",
]
for _h in CANON_HORIZONS:
    PERP_EVENT_RISK_FEATURE_KEYS.extend(
        [
            f"fund_pre_drift_{_h}h",
            f"fund_post_reversal_{_h}h",
            f"fund_ret_cond_sign_{_h}h",
            f"fund_payment_pressure_{_h}h",
            f"mark_gap_vol_{_h}h",
            f"funding_crowded_mom_exhaustion_{_h}h",
            f"funding_crowded_mom_exhaustion_self_z_{_h}h",
            f"fund_high_neg_mom_{_h}h",
            f"fund_high_neg_mom_self_z_{_h}h",
            f"persistent_pos_funding_failed_breakout_{_h}h",
            f"persistent_neg_funding_failed_breakdown_{_h}h",
            f"fund_flip_x_vol_expansion_{_h}h",
        ]
    )
PERP_CARRY_ALPHA_FEATURE_KEYS = []
for _h in CANON_HORIZONS:
    PERP_CARRY_ALPHA_FEATURE_KEYS.extend(
        [
            f"carry_adj_ret_{_h}h",
            f"carry_adj_ret_self_z_{_h}h",
            f"carry_adj_short_ret_{_h}h",
            f"carry_adj_short_ret_self_z_{_h}h",
            f"basis_adjusted_trend_{_h}h",
            f"basis_adjusted_trend_self_z_{_h}h",
        ]
    )
OI_TRADING_FEATURE_KEYS = get_oi_trading_feature_names()
OI_NORMALIZED_FEATURE_KEYS = get_oi_normalized_feature_names()
OI_FEATURE_KEYS = get_oi_feature_names()
LONG_HORIZON_PERP_META_FEATURE_KEYS = [
    "funding_mean_7d_robust_z",
    "funding_mean_10d_robust_z",
    "funding_mean_15d_robust_z",
    "funding_vol_7d_robust_z",
    "funding_vol_10d_robust_z",
    "funding_vol_15d_robust_z",
    "oi_trend_7d_robust_z",
    "oi_trend_10d_robust_z",
    "oi_trend_15d_robust_z",
    "oi_vol_7d_robust_z",
    "oi_vol_10d_robust_z",
    "oi_vol_15d_robust_z",
    "price_trend_7d_vol_norm",
    "price_trend_10d_vol_norm",
    "price_trend_15d_vol_norm",
    "price_rv_7d_robust_z",
    "price_rv_10d_robust_z",
    "price_rv_15d_robust_z",
]
VOLUME_FREE_PERP_BASE_FEATURE_KEYS = [
    "dist_oiw_intensity_12h_atr",
    "dist_oiw_z_delta_12h_atr",
    "dist_oiw_signed_delta_12h_atr",
    "dist_oiw_abs_delta_12h_atr",
    "oi_expansion_compression_balance_24h",
    "dist_funding_pressure_price_12h_atr",
    "oiw_entry_zone_1d_atr",
    "donchian_zone_1d_atr",
    "abs_ret_per_oi_z_24h",
    "impact_per_oi_intensity_z_24h",
    "range_per_funding_abs_z_24h",
]
VOLUME_FREE_PERP_META_FEATURE_KEYS = [
    "dist_oiw_intensity_96h_atr",
    "dist_oiw_z_delta_96h_atr",
    "oi_expansion_compression_balance_96h",
    "dist_funding_pressure_price_96h_atr",
]
PERP_FEATURE_KEYS = list(
    dict.fromkeys(
        PERP_FEATURE_KEYS
        + PERP_PRICE_RELATION_FEATURE_KEYS
        + SPOT_FOR_PERPS_BASE_FEATURE_KEYS
        + SPOT_FOR_PERPS_META_FEATURE_KEYS
        + PERP_EVENT_RISK_FEATURE_KEYS
        + PERP_CARRY_ALPHA_FEATURE_KEYS
        + OI_TRADING_FEATURE_KEYS
        + LONG_HORIZON_PERP_META_FEATURE_KEYS
        + VOLUME_FREE_PERP_BASE_FEATURE_KEYS
        + VOLUME_FREE_PERP_META_FEATURE_KEYS
        + ["xasset_asset_minus_mkt_funding"]
    )
)
RESIDUAL_FEATURE_KEYS = residual_feature_names(include_legacy_aliases=True)
RESIDUAL_BASE_FEATURE_KEYS = [
    "ret4h_bench_resid",
    "ret24h_bench_resid",
    "ret4h_peer_resid",
    "ret24h_peer_resid",
    "dist_ema_fast_mkt_resid",
    "trend_pct_mkt_resid",
    "dist_ema_fast_ts_resid",
    "rsi_ts_resid",
    "flow_persistence_ts_resid",
    "excess_6h_ts_resid",
    "funding_per_hour_mkt_resid",
    "xasset_funding_ts_resid",
    "funding_1d_chg_peer_resid",
    "asset_minus_mkt_oi_1d_ts_resid",
    "ob_pressure_mkt_resid",
    "ob_imbalance_mkt_resid",
    "xasset_ob_pressure_ts_resid",
    "xasset_ob_liquidity_peer_resid",
    "volume_price_corr_ts_resid",
]
RESIDUAL_META_FEATURE_KEYS = [
    "ret48h_bench_resid",
    "rv_24h_peer_resid",
    "rvol_z_peer_resid",
    "amihud_z_peer_resid",
    "liquidity_ratio_peer_resid",
    "atr_expansion_ts_resid",
    "coherence_24_ts_resid",
    "overext_surprise",
    "blowoff_risk_surprise",
    "exh_qual_surprise",
    "spike_score_surprise",
    "grind_score_surprise",
    "chop_score_surprise",
    "fund_abs_z_mkt_resid",
    "xasset_funding_peer_resid",
    "funding_1d_chg_ts_resid",
    "asset_minus_mkt_oi_7d_ts_resid",
    "asset_minus_mkt_oi_1d_peer_resid",
    "asset_minus_mkt_oi_7d_peer_resid",
    "ob_spread_mkt_resid",
    "ob_depth_mkt_resid",
    "xasset_ob_pressure_peer_resid",
    "xasset_ob_liquidity_ts_resid",
    "path_efficiency_24_ts_resid",
]
# Research-only context for the sequential H12 T2 funnel.  A row-level cost
# estimate may be admitted only after it is separately materialised with an
# entry-time availability contract.  The historical ``execution_cost_return``
# target ledger is not such a feature source and must never be aliased into
# this list.
T2_FUNNEL_BASE_CONTEXT_FEATURE_KEYS = [
    "side_is_long",
]
# A residual fit may additionally consume stopped-gradient base outputs.  The
# runner creates these fields only after strict base OOF prediction generation.
T2_FUNNEL_META_CONTEXT_FEATURE_KEYS = [
    *T2_FUNNEL_BASE_CONTEXT_FEATURE_KEYS,
    "base_expected_net_bps",
    "base_p_upper",
    "base_p_lower",
    "base_p_timeout",
    "base_probability_width",
]
# Stage-C only.  These are deliberately separate from the production base,
# residual and general meta registries: they are admissible solely for the
# conditional P(retain | clear) research head until a frozen hierarchy test
# passes every predeclared gate.
RETENTION_CONTINUATION_F1_FEATURE_KEYS = CONTINUATION_PRICE_FEATURE_KEYS + CONTINUATION_SIDE_PRICE_FEATURE_KEYS
RETENTION_CONTINUATION_F2_FEATURE_KEYS = CONTINUATION_VOLUME_FEATURE_KEYS
RETENTION_CONTINUATION_F3_FEATURE_KEYS = CONTINUATION_VOLATILITY_FEATURE_KEYS + CONTINUATION_SIDE_VOLATILITY_FEATURE_KEYS
RETENTION_CONTINUATION_F4_FEATURE_KEYS = CONTINUATION_OI_FEATURE_KEYS
RETENTION_CONTINUATION_F5_FEATURE_KEYS = CONTINUATION_FUNDING_FEATURE_KEYS
RETENTION_CONTINUATION_F6_FEATURE_KEYS = CONTINUATION_CROSS_SECTIONAL_FEATURE_KEYS
RETENTION_CONTINUATION_F7_FEATURE_KEYS = CONTINUATION_REGIME_FEATURE_KEYS
RETENTION_CONTINUATION_F8_FEATURE_KEYS = CONTINUATION_COMPOSITE_FEATURE_KEYS
RETENTION_CONTINUATION_FEATURE_GROUPS = CONTINUATION_FEATURE_GROUPS
PERP_TRADEABILITY_FEATURE_KEYS = [
    "asset_funding_rate_mean_3d",
    "asset_funding_rate_mean_7d",
    "asset_funding_rate_mean_15d",
    "asset_funding_rate_abs_mean_7d",
    "asset_funding_z",
    "asset_funding_trend_alignment",
    "funding_rate_cross_asset_dispersion",
]
LGBM_PERP_FEATURE_KEYS = list(
    dict.fromkeys(
        PERP_PRICE_RELATION_FEATURE_KEYS
        + SPOT_FOR_PERPS_BASE_FEATURE_KEYS
        + SPOT_FOR_PERPS_META_FEATURE_KEYS
        + PERP_TRADEABILITY_FEATURE_KEYS
        + OI_TRADING_FEATURE_KEYS
        + LONG_HORIZON_PERP_META_FEATURE_KEYS
        + VOLUME_FREE_PERP_BASE_FEATURE_KEYS
        + VOLUME_FREE_PERP_META_FEATURE_KEYS
        + ["xasset_asset_minus_mkt_funding"]
    )
)
PERP_META_PRIMARY_FEATURE_KEYS = (
    [
        "mark_price",
        "funding_abs_z",
        "funding_persistence",
        "funding_z",
        "fund_rate_mom_8h",
        "leverage_build",
        "squeeze_prob",
        "unwind",
    ]
    + PERP_EVENT_RISK_FEATURE_KEYS
    + OI_TRADING_FEATURE_KEYS
)

ORDERBOOK_RAW_BASE_FEATURE_KEYS = [
    # Directional book pressure
    "ob_microprice_premium_bps",
    "ob_l1_imbalance",
    "ob_l10_imbalance",
    "ob_l20_imbalance",
    "ob_wimb_l10",
    "ob_wimb_l20",
    "ob_book_pressure_l10",
    # Book dynamics
    "ob_imb_chg_1",
    "ob_imb_accel_4",
    "ob_imb_near_far_delta",
    "ob_depth_decay_asym_l20",
    "ob_wall_imb_l20",
    # Trade flow
    "ob_trade_flow_imbalance_1h",
    "ob_vwap_mid_gap_bps",
    # New flow features
    "ob_flow_qty_imbalance_1h",
    "ob_flow_notional_imbalance_1h",
    "ob_buy_notional_z_24h",
    "ob_sell_notional_z_24h",
    "ob_flow_notional_skew_z_24h",
    # Flow/book disagreement and absorption
    "ob_flow_vs_book_l10",
    "ob_flow_vs_book_l20",
    "ob_book_absorption_score",
    # Cross-asset directional pressure
    "xasset_asset_minus_mkt_ob_pressure",
    "xasset_btc_ob_pressure",
    "xasset_eth_ob_pressure",
    "xasset_asset_minus_basket_ob_pressure",
    "xasset_leverage_build_score",
    # Directional interaction
    "ob_pressure_x_ret4h_sign",
]
ORDERBOOK_NORMALIZED_BASE_FEATURE_KEYS = [
    "ob_microprice_dev_bps_z_24h",
    "ob_depth_l10_to_qv_24h",
    "ob_depth_l20_to_qv_24h",
    "ob_depth_l20_to_qv_z_7d",
]
ORDERBOOK_BASE_FEATURE_KEYS = list(
    dict.fromkeys(ORDERBOOK_NORMALIZED_BASE_FEATURE_KEYS)
)
ORDERBOOK_RAW_META_FEATURE_KEYS = [
    # Availability / staleness / data trust
    "ob_available",
    "ob_snapshot_age_sec",
    "ob_update_gap_flag",
    "ob_stale_flag",
    # Spread / execution friction / dislocation
    "ob_spread_bps",
    "ob_spread_z_24h",
    "ob_microprice_premium_bps",
    "ob_mid_close_dislocation_bps",
    # Pressure also useful for trust
    "ob_book_pressure_l10",
    # Depth / liquidity / stress
    "ob_top_liquidity_usd",
    "ob_depth_usd_l10",
    "ob_depth_usd_l20",
    "ob_depth_usd_l10_z",
    "ob_depth_usd_l20_z",
    "ob_depth_usd_z_24h",
    "ob_liquidity_shock_z",
    # Liquidity structure
    "ob_depth_ratio_l1_l20",
    "ob_bid_depth_decay_l20",
    "ob_ask_depth_decay_l20",
    # Activity regime
    "ob_trade_count_z_24h",
    "ob_notional_z_24h",
    "ob_vwap_mid_gap_bps",
    "ob_mean_trade_qty_z_24h",
    # New flow regime / stress
    "ob_buy_notional_z_24h",
    "ob_sell_notional_z_24h",
    "ob_abs_flow_vs_book_l20",
    "ob_book_absorption_score",
    # Impact / toxicity
    "ob_notional_to_depth_l20",
    "ob_trade_size_to_l1_depth",
    "ob_kyle_lambda_1h",
    "ob_flow_toxicity_1h",
    # Cross-asset trust/regime
    "xasset_btc_ob_pressure",
    "xasset_eth_ob_pressure",
    "xasset_mkt_spread_bps",
    "xasset_mkt_depth_z",
    "xasset_ob_stress_basket",
    "xasset_asset_minus_basket_ob_pressure",
    "xasset_ob_liquidity_divergence",
    # Meta interactions
    "ob_spread_z_x_rv_24h",
    "ob_depth_z_x_rvol_z",
    "asset_spread_proxy_p90_24h",
    "asset_spread_proxy_p90_96h",
    "asset_spread_proxy_p90_7d",
    "asset_spread_proxy_p90_15d",
    "asset_volume_depth_risk_p90_24h",
    "asset_volume_depth_risk_p90_96h",
    "asset_volume_depth_risk_p90_7d",
    "asset_volume_depth_risk_p90_15d",
    "asset_orderbook_imbalance_abs_mean_24h",
    "asset_orderbook_imbalance_abs_mean_96h",
    "asset_orderbook_imbalance_abs_mean_7d",
    "asset_orderbook_imbalance_abs_mean_15d",
    "asset_liquidity_stress_score_7d",
    "global_liquidity_stress_score_7d",
]
ORDERBOOK_NORMALIZED_META_FEATURE_KEYS = [
    "ob_available",
    "ob_spread_bps_z_24h",
    "ob_spread_bps_z_7d",
    "ob_mid_close_dislocation_bps_z_24h",
    "ob_depth_l10_to_qv_24h",
    "ob_depth_l20_to_qv_24h",
    "ob_top_liquidity_to_qv_24h",
    "ob_depth_l20_to_qv_z_7d",
    "ob_depth_ratio_l1_l20",
    "ob_depth_decay_asym_l20_z_7d",
    "ob_abs_flow_vs_book_l20_z_24h",
    "ob_notional_to_depth_l20_z_24h",
    "ob_trade_size_to_l1_depth_z_24h",
    "ob_spread_z_x_rv_24h",
    "ob_depth_to_qv_z_x_rvol_z",
    "xasset_mkt_spread_bps_z_24h",
    "xasset_mkt_depth_to_qv_z",
    "xasset_mkt_ob_stress_z_24h",
    "xasset_ob_stress_basket_z_24h",
    "xasset_ob_liquidity_divergence_z_24h",
]
ORDERBOOK_META_FEATURE_KEYS = list(
    dict.fromkeys(ORDERBOOK_NORMALIZED_META_FEATURE_KEYS)
)
ORDERBOOK_DIAGNOSTIC_ONLY_FEATURE_KEYS = [
    "ob_slope_diff_l10",
    "ob_gap_up_bps_l10",
    "ob_gap_dn_bps_l10",
    "ob_gap_skew_l10",
]
ORDERBOOK_EXCLUDED_STALE_FEATURE_KEYS = [
    "ob_snapshot_age_min",
    "ob_mid_vs_close_bps",
    "ob_l5_imbalance",
    "ob_bid_depth_5bps",
    "ob_ask_depth_5bps",
    "ob_depth_skew_5bps",
    "ob_bid_depth_10bps",
    "ob_ask_depth_10bps",
    "ob_depth_skew_10bps",
    "ob_bid_depth_25bps",
    "ob_ask_depth_25bps",
    "ob_depth_skew_25bps",
    "ob_bid_depth_50bps",
    "ob_ask_depth_50bps",
    "ob_depth_skew_50bps",
    "ob_bid_depth_100bps",
    "ob_ask_depth_100bps",
    "ob_depth_skew_100bps",
    "ob_bid_slope_20",
    "ob_ask_slope_20",
    "ob_bid_wall_found",
    "ob_ask_wall_found",
    "ob_nearest_bid_wall_dist_bps",
    "ob_nearest_ask_wall_dist_bps",
    "ob_nearest_bid_wall_dist_atr",
    "ob_nearest_ask_wall_dist_atr",
    "ob_nearest_bid_wall_to_qv24",
    "ob_nearest_ask_wall_to_qv24",
    "ob_liquidity_void_up_bps",
    "ob_liquidity_void_down_bps",
    "ob_max_gap_up_bps",
    "ob_max_gap_down_bps",
]
ORDERBOOK_FEATURE_KEYS = sorted(
    set(ORDERBOOK_BASE_FEATURE_KEYS) | set(ORDERBOOK_META_FEATURE_KEYS)
)
CROSS_ASSET_FEATURE_KEYS = [
    "cs_rank_ret_24h",
    "mkt_ret_eq_1h",
    "mkt_ret_eq_4h",
    "mkt_ret_eq_24h",
    "btc_ret_1h",
    "btc_ret_4h",
    "btc_ret_24h",
    "eth_ret_1h",
    "eth_ret_4h",
    "eth_ret_24h",
    "eth_btc_ret_1h",
    "eth_btc_ret_4h",
    "eth_btc_ret_24h",
    "beta_btc_24h",
    "beta_eth_24h",
    "corr_btc_24h",
    "corr_eth_24h",
    "ret_resid_btc_1h",
    "ret_resid_btc_4h",
    "ret_resid_eth_1h",
    "ret_resid_eth_4h",
    "symbol_minus_mkt_ret_1h",
    "symbol_minus_mkt_ret_4h",
    "symbol_minus_mkt_ret_24h",
    "market_breadth_1h",
    "market_breadth_4h",
    "market_breadth_24h",
    "market_dispersion_1h",
    "market_dispersion_4h",
    "market_dispersion_24h",
]

OHLCV_BREADTH_FEATURE_KEYS = [
    "pct_assets_up_15m",
    "pct_assets_up_1h",
    "pct_assets_up_4h",
    "pct_assets_up_24h",
    "pct_assets_below_minus_1atr_1h",
    "pct_assets_below_minus_2atr_1h",
    "pct_assets_below_minus_2atr_4h",
    "pct_assets_below_minus_3atr_4h",
    "breadth_chg_15m",
    "breadth_chg_1h",
    "breadth_accel_1h",
    "breadth_min_6h",
    "breadth_recovery_from_6h_min",
    "pct_assets_new_low_24h",
    "pct_assets_new_low_7d",
    "pct_assets_recovering_from_intraday_low",
    "pct_assets_above_intraday_vwap",
]

OHLCV_CRASH_PHASE_FEATURE_KEYS = [
    "mkt_ret_15m",
    "mkt_ret_1h",
    "mkt_ret_4h",
    "mkt_ret_24h",
    "mkt_return_accel_1h",
    "mkt_rv_1h",
    "mkt_rv_4h",
    "mkt_rv_24h",
    "mkt_rv_ratio_1h_24h",
    "mkt_atr_expansion_1h",
    "mkt_atr_expansion_4h",
    "mkt_volume_z_24h",
    "mkt_quote_volume_z_24h",
    "pct_assets_volume_z_gt_2",
    "pct_assets_climax_volume",
    "mkt_drawdown_from_24h_high_atr",
    "mkt_drawdown_from_7d_high_atr",
    "mkt_recovery_from_24h_low_atr",
    "bars_since_mkt_price_trough",
    "mkt_lower_wick_ratio_1h",
    "mkt_close_location_1h",
    "mkt_range_expansion_1h",
    "pct_assets_large_lower_wick",
    "pct_assets_bullish_reversal_candle",
]

MARKET_SYNCHRONIZATION_FEATURE_KEYS = [
    "cross_asset_corr_1h",
    "cross_asset_corr_4h",
    "cross_asset_corr_chg_1h",
    "cross_asset_downside_corr_1h",
    "cross_asset_downside_corr_4h",
    "mkt_first_pc_variance_share_1h",
    "mkt_first_pc_variance_share_4h",
    "return_dispersion_1h",
    "return_dispersion_4h",
]

OHLCV_LIFECYCLE_FEATURE_KEYS = [
    "downside_deceleration_4h_rz",
    "downside_deceleration_8h_rz",
    "price_recovery_from_low_24h_atr",
    "price_recovery_from_low_72h_atr",
    "bars_since_price_low_24h_norm",
    "bars_since_price_low_72h_norm",
    "volume_climax_decay_4h",
    "range_climax_decay_4h",
    "wick_recovery_intensity",
]

MARKET_BREADTH_LIFECYCLE_FEATURE_KEYS = [
    "market_breadth_recovery_from_24h_min",
    "market_breadth_drawdown_from_6h_max",
    "market_pct_recovering_from_24h_low",
]

MARKET_SYNCHRONIZATION_ADDITION_KEYS = [
    "market_pc1_variance_share_12h",
    "market_pc1_variance_share_24h",
    "market_pc1_variance_share_chg_4h",
    "market_downside_pairwise_corr_24h",
    "market_downside_corr_minus_unconditional_corr_24h",
]

LIQUIDATION_STATE_SCORE_FEATURE_KEYS = [
    "liquidation_onset_score",
    "liquidation_climax_score",
    "post_liquidation_rebound_score",
]

CRASH_LIFECYCLE_ASSET_FEATURE_KEYS = [
    *ASSET_OI_LIFECYCLE_FEATURE_KEYS,
    *PRICE_OI_QUADRANT_FEATURE_KEYS,
    *PRICE_OI_RECOVERY_FEATURE_KEYS,
    *FUNDING_LIFECYCLE_FEATURE_KEYS,
    *OHLCV_LIFECYCLE_FEATURE_KEYS,
    *ASSET_MARKET_LIFECYCLE_RESIDUAL_KEYS,
    "asset_liquidation_phase_score",
    "asset_flush_exhaustion_score",
    "asset_short_covering_score",
    "asset_mkt_liquidation_phase_divergence",
    "asset_mkt_exhaustion_phase_divergence",
]

CRASH_LIFECYCLE_MARKET_FEATURE_KEYS = [
    *MARKET_OI_LIFECYCLE_FEATURE_KEYS,
    *MARKET_PRICE_OI_STATE_FEATURE_KEYS,
    *MARKET_BREADTH_LIFECYCLE_FEATURE_KEYS,
    *MARKET_SYNCHRONIZATION_ADDITION_KEYS,
    "mkt_systemic_deleveraging_score",
    "mkt_flush_exhaustion_score",
    "mkt_leverage_rebuild_score",
]

CRASH_LIFECYCLE_NEW_FEATURE_KEYS = list(
    dict.fromkeys(
        CRASH_LIFECYCLE_ASSET_FEATURE_KEYS + CRASH_LIFECYCLE_MARKET_FEATURE_KEYS
    )
)

CROSS_ASSET_FEATURE_KEYS = list(
    dict.fromkeys(
        CROSS_ASSET_FEATURE_KEYS
        + OHLCV_BREADTH_FEATURE_KEYS
        + OHLCV_CRASH_PHASE_FEATURE_KEYS
        + MARKET_SYNCHRONIZATION_FEATURE_KEYS
        + MARKET_BREADTH_LIFECYCLE_FEATURE_KEYS
        + MARKET_SYNCHRONIZATION_ADDITION_KEYS
        + LIQUIDATION_STATE_SCORE_FEATURE_KEYS
    )
)


FEATURE_KEYS_15M_OHLCV = [
    "clv_t",
    "body_ratio_15m",
    "rejection_proxy",
    "range_norm_12",
    "sv_imb_12",
    "press_12",
    "impact_12",
    "ts_12",
    "prog_eff_12",
    "pers_12",
    "hh_count_12",
    "ll_count_12",
    "skew_12",
    "climax_range_12",
    "climax_vol_12",
    "z_vwap_12",
    "z_r_12",
    "bb_pos_12",
    "range_norm_24",
    "sv_imb_24",
    "press_24",
    "impact_24",
    "ts_24",
    "prog_eff_24",
    "pers_24",
    "hh_count_24",
    "ll_count_24",
    "skew_24",
    "climax_range_24",
    "climax_vol_24",
    "z_vwap_24",
    "z_r_24",
    "bb_pos_24",
]

SPREAD_PROXY_RAW_FEATURE_KEYS = [
    "hl_range_bps",
    "abs_return_bps",
    "body_bps",
    "upper_wick_bps",
    "lower_wick_bps",
    "wick_to_range",
    "close_location",
    "gap_bps",
]
SPREAD_PROXY_FEATURE_KEYS = [
    f"spread_proxy_{name}_robust_z" for name in SPREAD_PROXY_RAW_FEATURE_KEYS
]

PRICE_MEMORY_FEATURE_KEYS = [
    "log_bars_since_above_1atr",
    "log_bars_since_below_1atr",
    "memory_asymmetry_1ATR",
    "log_bars_since_above_2atr",
    "log_bars_since_below_2atr",
    "memory_asymmetry_2ATR",
    "log_bars_since_above_3atr",
    "log_bars_since_below_3atr",
    "memory_asymmetry_3ATR",
]

OI_WEIGHTED_LOCATION_BASE_FEATURE_KEYS = [
    "oiw_pos_delta_entry_dist_1d_atr",
    "oiw_intensity_entry_dist_1d_atr",
    "oiw_z_delta_entry_dist_1d_atr",
    "dist_oiw_intensity_12h_atr",
    "dist_oiw_z_delta_12h_atr",
    "dist_oiw_signed_delta_12h_atr",
    "dist_oiw_abs_delta_12h_atr",
    "oi_expansion_compression_balance_24h",
    "dist_funding_pressure_price_12h_atr",
    "oiw_entry_zone_1d_atr",
    "donchian_zone_1d_atr",
]

OI_WEIGHTED_LOCATION_META_FEATURE_KEYS = [
    "oiw_pos_delta_entry_dist_7d_atr",
    "oiw_pos_delta_entry_dist_14d_atr",
    "oiw_intensity_entry_dist_7d_atr",
    "oiw_z_delta_entry_dist_7d_atr",
    "oiw_z_delta_entry_dist_14d_atr",
    "dist_oiw_intensity_96h_atr",
    "dist_oiw_z_delta_96h_atr",
    "oi_expansion_compression_balance_96h",
    "dist_funding_pressure_price_96h_atr",
]

DAILY_SR_BASE_FEATURE_KEYS = [
    "vwap_zone_1d_atr",
    *OI_WEIGHTED_LOCATION_BASE_FEATURE_KEYS,
    "distance_to_support_daily_vwap_atr",
    "distance_to_resistance_daily_vwap_atr",
    "distance_to_support_daily_donchian_atr",
    "distance_to_resistance_daily_donchian_atr",
    "down_barrier_pressure_daily_vwap",
    "up_barrier_pressure_daily_vwap",
    "bars_to_support_daily_vwap",
    "bars_to_resistance_daily_vwap",
    "down_barrier_pressure_daily_donchian",
    "up_barrier_pressure_daily_donchian",
    "bars_to_support_daily_donchian",
    "bars_to_resistance_daily_donchian",
    "distance_to_support_atr",
    "distance_to_resistance_atr",
]

WEEKLY_SR_META_FEATURE_KEYS = [
    "vwap_zone_7d_atr",
    *OI_WEIGHTED_LOCATION_META_FEATURE_KEYS,
    "distance_to_support_weekly_vwap_atr",
    "distance_to_resistance_weekly_vwap_atr",
    "distance_to_support_weekly_donchian_atr",
    "distance_to_resistance_weekly_donchian_atr",
    "down_barrier_pressure_weekly_vwap",
    "up_barrier_pressure_weekly_vwap",
    "bars_to_support_weekly_vwap",
    "bars_to_resistance_weekly_vwap",
    "down_barrier_pressure_weekly_donchian",
    "up_barrier_pressure_weekly_donchian",
    "bars_to_support_weekly_donchian",
    "bars_to_resistance_weekly_donchian",
]

neutral_feature_keys = [
    "rsi",
    "momentum_accel",
    "dist_stack",
    "exh_qual",
]

MODEL_FEATURES = [
    # Kalman Base Features
    "price_state_slope_1h",
    "price_state_slope_ratio_1h_6h",
    "price_minus_state_z",
    "trend_stack_3_6_12",
    "trend_stack_6_12_24",
    "zr_1h_minus_zr_6h",
    "zr_3h_minus_zr_12h",
    "zr_6h_minus_zr_24h",
    "trend_dispersion_1_3_6",
    "trend_dispersion_3_6_12",
    "innovation_z_x_zr_1h",
    "innovation_z_x_zr_3h",
    "zr_1h_x_volume_z_24h",
    "zr_3h_x_volume_z_24h",
    "zr_6h_x_volume_z_48h",
    "zr_6h_x_range_z_24h",
    "zr_12h_x_range_z_48h",
    "zr_3h",
    "zr_6h",
    "zr_12h",
    "trend_alignment_1_3_6",
    "trend_alignment_3_6_12",
    "trend_alignment_6_12_24",
    # Momentum / structure extensions
    "thrust_decay_4",
    "decel_4",
    "ft_drop",
    "thrust_decay_8",
    "decel_8",
    "ft_drop_8",
    "ext_excess",
    "ext_atrExp",
    "comp_to_exp",
    "stall_x_flow",
    "prog_def",
    "clv_collapse",
    "clv_pullback",
    "coh",
    "align",
    "retest_quality",
    "pb_accel",
    "excess_coh",
    "asym_ft",
    "tf_bias",
    "shock_rel",
    "resid_strength",
    "evr_slope",
    "stall_ext",
    # Earlier trend following / volatility-of-volatility signals
    "accel_5h",
    "signed_max_bar_ret_5h",
    "jump_rate_10h",
    "draw_sym_10h",
    "draw_extreme_10h",
    "down_up_vol_ratio_24",
    "vol_shock_asym_8_24",
    "vol_shock_asym_4_12",
    "vol_shock_asym_4_212",
    "breakout_24h",
    "meta_abs_net_x_breakout",
    "meta_abs_net_x_drawext",
    "meta_alignment",
    "meta_signal_x_accel",
    # Price Action
    "gap_pct",
    "range_pct",
    "roc_div",
    "body_pct",
    "wick_body_ratio",
    "vol_price_spread",
    "wick_ratio",
    "body_ratio",
    # New Risk/Exhaustion (Report 2026-02-10)
    "wick_ratio_4h_max",
    "vol_price_div",
    "rsi_lag1",
    "rsi_1h_slope",
    "clv_mean_24",
    "atr_pct_change",
    # FFD d-specific features (d=0.4,0.6)
    "ffd_rv_2h_04",
    "ffd_rv_6h_04",
    "ffd_rv_24h_04",
    "ffd_vol_price_corr_10h_04",
    "ffd_donch_dist_04_12",
    "ffd_donch_dist_04_24",
    "ffd_donch_dist_04_48",
    "ffd_dist_ema_fast_04",
    "ffd_dist_ema_slow_04",
    "ffd_rv_2h_06",
    "ffd_rv_6h_06",
    "ffd_rv_24h_06",
    "ffd_accel_06",
    "ffd_z_06",
    "ffd_vol_price_corr_10h_06",
    "ffd_donch_dist_06_12",
    "ffd_donch_dist_06_24",
    "ffd_donch_dist_06_48",
    "ffd_atr_expansion_06",
    # D-family strength indicators
    "ffd_strength_04",
    "ffd_strength_05",
    "ffd_strength_06",
    # Alpha Features (Report 2026-02-10)
    "breakout_min",
    "impulse_reversal",
    "impulse_reversal_short",
    "breakout_confirmed",
    "breakout_t",
    "pct_breakout_t",
    # 2h directional path-risk
    "dir_path_long_2h",
    "dir_path_short_2h",
    "dir_path_risk_long_2h",
    "dir_path_risk_short_2h",
    "dir_path_edge_2h",
    "dir_path_risk_skew_2h",
    # Volume/Flow
    "v_power",
    "flow_persistence",
    "flow_ratio",
    "churn",
    "climax_decay",
    "cumulative_delta_stall",
    "vol_expansion_ratio",
    "vol_compression",
    # Advanced
    "fvg",
    "slope",
    "atr_slope",
    "dist_vwap_norm",
    "rsi_slope",
    "funding_proxy",
    "dist_ema_fast",
    # Scores
    # New regime-transition and entropy features for improved PR-AUC and robustness
    # Regime features for fold robustness (Report 2026-02-11)
    "is_trending",
    "is_ranging",
    "trend_x_trending",
    # New Indicators (KER, Vortex, ADX, VWAP, HVN/LVN)
    "ker_10",
    "ker_16",
    "ker_24",
    "adx_7",
    "adx_10",
    "adx_14",
    "adx_di_plus_7",
    "adx_di_minus_7",
    "adx_di_plus_10",
    "adx_di_minus_10",
    "adx_di_plus_14",
    "adx_di_minus_14",
    "adx_7_gt25",
    "adx_10_gt25",
    "adx_14_gt25",
    "adx_7_slope",
    "adx_10_slope",
    "adx_14_slope",
    "dist_vwap_12_atr",
    "trapped_longs_12",
    "dist_vwap_24_atr",
    "trapped_longs_24",
    "dist_vwap_96_atr",
    "trapped_longs_96",
    "vp_dist_poc_atr",
    "vp_dist_hvn_above_atr",
    "vp_dist_hvn_below_atr",
    "vp_dist_lvn_above_atr",
    "vp_dist_lvn_below_atr",
    "vp_in_poc_zone",
    "vp_in_hvn_above_zone",
    "vp_in_hvn_below_zone",
    "vp_in_lvn_above_zone",
    "vp_in_lvn_below_zone",
    "vp_profile_concentration",
    "vp_lvn_depth_ratio",
    "vp_accept_poc_touchrate",
    "vp_accept_hvn_touchrate",
    "vp_accept_lvn_touchrate",
    "vp_air_pocket_score",
    "G_TF_TREND",
    # Ridge model features
    "ema20_gt_ema50",
    "ema50_gt_ema200",
    "ema50_ema200_spread_atr",
    "compression_ratio",
    "range_expansion_ratio",
    "atr_compression_ratio",
    "price_lt_ema200",
    "ema50_slope",
    "trend_strength_percentile",
    "atr_change_rate",
    "true_range_percentile",
    "bollinger_band_width",
    "rolling_range_20",
    "atr_percentile",
    "prior_range",
    "ema20_slope_5h",
    "ema_slope_norm",
    "compression_score",
    "return_autocorr_48",
    "trend_acceleration",
    "ret24h",
]

# Time-based features to exclude from LGBM mask generation
# These are circular/seasonal features that can create spurious patterns
TIME_FEATURE_KEYS = [
    "sin_hod",
    "cos_hod",
    "sin_dow",
    "cos_dow",
]

CONTINUOUS_REGIME_FEATURES = {
    # Trend features
    "ema20_gt_ema50": {"family": "trend", "type": "binary"},
    "ema50_gt_ema200": {"family": "trend", "type": "binary"},
    "ema50_ema200_spread_atr": {"family": "trend", "type": "continuous"},
    "price_lt_ema200": {"family": "trend", "type": "binary"},
    "ema50_slope": {"family": "trend", "type": "continuous"},
    "trend_strength_percentile": {"family": "trend", "type": "continuous"},
    "realized_volatility_24h": {"family": "volatility", "type": "continuous"},
    "atr_change_rate": {"family": "volatility", "type": "continuous"},
    "true_range_percentile": {"family": "volatility", "type": "continuous"},
    "bollinger_band_width": {"family": "compression", "type": "continuous"},
    "rolling_range_20": {"family": "compression", "type": "continuous"},
    "atr_percentile": {"family": "compression", "type": "continuous"},
    "prior_range": {"family": "context", "type": "continuous"},
    "prior_volatility": {"family": "context", "type": "continuous"},
    # Micro-regime updates
    "efficiency_ratio_20": {"family": "path_structure", "type": "continuous"},
    "choppiness_index_20": {"family": "path_structure", "type": "continuous"},
    "direction_entropy_20": {"family": "path_structure", "type": "continuous"},
    "compression_ratio": {"family": "volatility_term_structure", "type": "continuous"},
    "range_expansion_ratio": {
        "family": "volatility_term_structure",
        "type": "continuous",
    },
    "volatility_ratio_short_long": {"family": "volatility", "type": "continuous"},
    "volume_percentile": {"family": "liquidity", "type": "continuous"},
    # User-requested technical regimes (v17)
    "ema20_slope_5h": {"family": "trend", "type": "continuous"},
    "ema_slope_norm": {"family": "trend", "type": "continuous"},
    "atr_compression_ratio": {
        "family": "volatility_term_structure",
        "type": "continuous",
    },
    "volume_zscore_48h": {"family": "liquidity", "type": "continuous"},
    "compression_score": {"family": "volatility_term_structure", "type": "continuous"},
    "return_autocorr_48": {"family": "momentum", "type": "continuous"},
    "variance_ratio_10_48": {"family": "volatility", "type": "continuous"},
    "volume_trend_48": {"family": "liquidity", "type": "continuous"},
    "volume_autocorr_48": {"family": "liquidity", "type": "continuous"},
    "volatility_of_volatility_48": {"family": "volatility", "type": "continuous"},
    "trend_acceleration": {"family": "trend", "type": "continuous"},
    "volatility_autocorr_48": {"family": "volatility", "type": "continuous"},
    "bars_since_ema20_ema50_cross_log_norm": {"family": "trend", "type": "continuous"},
    "bars_in_high_vol_state_log_norm": {"family": "volatility", "type": "continuous"},
    "bars_outside_ema20_atr_band_log_norm": {
        "family": "volatility",
        "type": "continuous",
    },
    "up_down_semivol_ratio_tanh": {"family": "path_structure", "type": "continuous"},
    "up_down_return_mass_ratio_tanh": {
        "family": "path_structure",
        "type": "continuous",
    },
    "tail_asymmetry_q90_q10_atr_norm": {
        "family": "path_structure",
        "type": "continuous",
    },
    # Additional 24h+ / structural regime features
    "rv_24h": {"family": "volatility", "type": "continuous"},
    "dist_ema_fast": {"family": "trend", "type": "continuous"},
    "range_24h_pct": {"family": "compression", "type": "continuous"},
    "spectral_entropy_ret_24": {"family": "path_structure", "type": "continuous"},
    "volume_entropy_24": {"family": "liquidity", "type": "continuous"},
    "path_efficiency_24": {"family": "path_structure", "type": "continuous"},
    "amihud_illiq": {"family": "liquidity", "type": "continuous"},
    "amihud_z": {"family": "liquidity", "type": "continuous"},
    "coherence_24": {"family": "path_structure", "type": "continuous"},
    "mkt_ret_eq_24h": {"family": "cross_asset", "type": "continuous"},
    "cs_rank_ret_24h": {"family": "cross_sectional", "type": "continuous"},
    "market_breadth_4h": {"family": "cross_sectional", "type": "continuous"},
    "market_breadth_24h": {"family": "cross_sectional", "type": "continuous"},
    "market_dispersion_4h": {"family": "cross_sectional", "type": "continuous"},
    "market_dispersion_24h": {"family": "cross_sectional", "type": "continuous"},
    "symbol_minus_mkt_ret_24h": {"family": "cross_asset", "type": "continuous"},
    "funding_rate": {"family": "funding", "type": "continuous"},
    "fund_rate_z_14d": {"family": "funding", "type": "continuous"},
    "funding_z": {"family": "funding", "type": "continuous"},
    "funding_abs_z": {"family": "funding", "type": "continuous"},
    "funding_per_hour": {"family": "funding", "type": "continuous"},
    "funding_per_hour_z": {"family": "funding", "type": "continuous"},
    "funding_rank_30d": {"family": "funding", "type": "continuous"},
    "funding_persistence": {"family": "funding", "type": "continuous"},
    "funding_mom_2h": {"family": "funding", "type": "continuous"},
    "funding_mom_4h": {"family": "funding", "type": "continuous"},
    "funding_mom_8h": {"family": "funding", "type": "continuous"},
    "funding_mom_w": {"family": "funding", "type": "continuous"},
    "oi_z": {"family": "open_interest", "type": "continuous"},
    "oi_rank": {"family": "open_interest", "type": "continuous"},
    "oi_chg_2h": {"family": "open_interest", "type": "continuous"},
    "oi_chg_z_2h": {"family": "open_interest", "type": "continuous"},
    "oi_chg_4h": {"family": "open_interest", "type": "continuous"},
    "oi_chg_z_4h": {"family": "open_interest", "type": "continuous"},
    "oi_chg_8h": {"family": "open_interest", "type": "continuous"},
    "oi_chg_z_8h": {"family": "open_interest", "type": "continuous"},
    "oi_value_log_1d_robust_z": {"family": "open_interest", "type": "continuous"},
    "oi_value_log_7d_robust_z": {"family": "open_interest", "type": "continuous"},
    "oi_chg_2h_robust_z": {"family": "open_interest", "type": "continuous"},
    "oi_chg_4h_robust_z": {"family": "open_interest", "type": "continuous"},
    "oi_chg_8h_robust_z": {"family": "open_interest", "type": "continuous"},
    "oi_vel_2h": {"family": "open_interest", "type": "continuous"},
    "oi_vel_4h": {"family": "open_interest", "type": "continuous"},
    "oi_vel_8h": {"family": "open_interest", "type": "continuous"},
    "oi_rel_vol_2h": {"family": "open_interest", "type": "continuous"},
    "oi_rel_vol_4h": {"family": "open_interest", "type": "continuous"},
    "oi_rel_vol_8h": {"family": "open_interest", "type": "continuous"},
    "oi_chg_w": {"family": "open_interest", "type": "continuous"},
    "basis_pct": {"family": "basis", "type": "continuous"},
    "basis_pct_z": {"family": "basis", "type": "continuous"},
    "basis_frac": {"family": "basis", "type": "continuous"},
    "basis_frac_z_14d": {"family": "basis", "type": "continuous"},
    "basis_frac_rank_30d": {"family": "basis", "type": "continuous"},
    "basis_per_atr": {"family": "basis", "type": "continuous"},
    "basis_stretch": {"family": "basis", "type": "continuous"},
    "basis_vol": {"family": "basis", "type": "continuous"},
    "basis_mom_2h": {"family": "basis", "type": "continuous"},
    "basis_mom_4h": {"family": "basis", "type": "continuous"},
    "basis_mom_8h": {"family": "basis", "type": "continuous"},
    "basis_mom_w": {"family": "basis", "type": "continuous"},
    "basis_funding_div": {"family": "basis", "type": "continuous"},
    "basis_funding_div_2h": {"family": "basis", "type": "continuous"},
    "basis_funding_div_4h": {"family": "basis", "type": "continuous"},
    "basis_funding_div_8h": {"family": "basis", "type": "continuous"},
    "mark_index_basis": {"family": "mark_index", "type": "continuous"},
    "mark_index_basis_z": {"family": "mark_index", "type": "continuous"},
    "mark_perp_dislocation": {"family": "mark_index", "type": "continuous"},
    "canonical_index": {"family": "mark_index", "type": "continuous"},
    "mark_vs_perp_bps": {"family": "mark_index", "type": "continuous"},
    "mark_vs_index_bps": {"family": "mark_index", "type": "continuous"},
    "perp_vs_index_bps": {"family": "mark_index", "type": "continuous"},
    "perp_index_basis": {"family": "mark_index", "type": "continuous"},
    "perp_index_basis_z": {"family": "mark_index", "type": "continuous"},
    "premium_proxy": {"family": "mark_index", "type": "continuous"},
    "premium_proxy_bps": {"family": "mark_index", "type": "continuous"},
    "premium_proxy_z": {"family": "mark_index", "type": "continuous"},
    "premium_proxy_mom_8h": {"family": "mark_index", "type": "continuous"},
    "leverage_build": {"family": "crowding", "type": "continuous"},
    "leverage_build_score": {"family": "crowding", "type": "continuous"},
    "unwind": {"family": "crowding", "type": "continuous"},
    "unwind_score": {"family": "crowding", "type": "continuous"},
    "squeeze_prob": {"family": "crowding", "type": "continuous"},
    "asset_atr_level_pct": {"family": "asset_state", "type": "continuous"},
    "asset_vol_level_pct": {"family": "asset_state", "type": "continuous"},
    "ob_age_ratio": {"family": "orderbook", "type": "continuous"},
    "ob_coverage_24h": {"family": "orderbook", "type": "continuous"},
    "ob_depth_z_10bps": {"family": "orderbook", "type": "continuous"},
    "ob_depth_z_25bps": {"family": "orderbook", "type": "continuous"},
    "ob_spread_z_24h": {"family": "orderbook", "type": "continuous"},
    "ob_depth_usd_l20_z": {"family": "orderbook", "type": "continuous"},
    "xasset_mkt_spread_bps": {"family": "cross_asset_orderbook", "type": "continuous"},
    "xasset_mkt_depth_z": {"family": "cross_asset_orderbook", "type": "continuous"},
    "xasset_mkt_ob_stress_z_24h": {
        "family": "cross_asset_orderbook",
        "type": "continuous",
    },
}

CONTINUOUS_REGIME_FEATURES.update(
    {
        name: {"family": "perp_tradeability", "type": "continuous"}
        for name in LGBM_PERP_FEATURE_KEYS
    }
)
CONTINUOUS_REGIME_FEATURES.update(
    {
        name: {"family": "open_interest", "type": "continuous"}
        for name in OI_FEATURE_KEYS
    }
)
CONTINUOUS_REGIME_FEATURES.update(
    {
        name: {"family": "residual", "type": "continuous"}
        for name in RESIDUAL_FEATURE_KEYS
    }
)
CONTINUOUS_REGIME_FEATURES.update(
    {
        name: {"family": "spread_proxy", "type": "continuous"}
        for name in SPREAD_PROXY_FEATURE_KEYS
    }
)
CONTINUOUS_REGIME_FEATURES.update(
    {
        "symbol_minus_mkt_ret_1h": {
            "family": "asset_relative_return",
            "type": "continuous",
        },
        "symbol_minus_mkt_ret_4h": {
            "family": "asset_relative_return",
            "type": "continuous",
        },
        "symbol_minus_mkt_ret_24h": {
            "family": "asset_relative_return",
            "type": "continuous",
        },
        "asset_minus_universe_median_ret_4h": {
            "family": "asset_relative_return",
            "type": "continuous",
        },
        "asset_minus_universe_median_ret_24h": {
            "family": "asset_relative_return",
            "type": "continuous",
        },
        "asset_minus_universe_median_ret_48h": {
            "family": "asset_relative_return",
            "type": "continuous",
        },
        "asset_mom_minus_basket_mom_4h": {
            "family": "asset_relative_return",
            "type": "continuous",
        },
        "asset_mom_minus_basket_mom_24h": {
            "family": "asset_relative_return",
            "type": "continuous",
        },
        "asset_ret_vs_universe_4h": {
            "family": "asset_relative_return",
            "type": "continuous",
        },
        "asset_ret_vs_universe_24h": {
            "family": "asset_relative_return",
            "type": "continuous",
        },
        "asset_ret_vs_universe_48h": {
            "family": "asset_relative_return",
            "type": "continuous",
        },
        "xasset_asset_minus_mkt_funding": {
            "family": "asset_relative_funding",
            "type": "continuous",
        },
        "xasset_asset_minus_basket_fund_z": {
            "family": "asset_relative_funding",
            "type": "continuous",
        },
        "xasset_asset_minus_mkt_ob_pressure": {
            "family": "asset_relative_orderbook",
            "type": "continuous",
        },
        "xasset_asset_minus_basket_ob_pressure": {
            "family": "asset_relative_orderbook",
            "type": "continuous",
        },
        "xasset_asset_minus_mkt_ob_pressure_z_24h": {
            "family": "asset_relative_orderbook",
            "type": "continuous",
        },
    }
)
for _h in CANON_HORIZONS:
    CONTINUOUS_REGIME_FEATURES.update(
        {
            f"carry_adj_ret_self_z_{_h}h": {
                "family": "perp_carry",
                "type": "continuous",
            },
            f"carry_adj_short_ret_self_z_{_h}h": {
                "family": "perp_carry",
                "type": "continuous",
            },
            f"basis_adjusted_trend_self_z_{_h}h": {
                "family": "basis",
                "type": "continuous",
            },
            f"funding_crowded_mom_exhaustion_self_z_{_h}h": {
                "family": "funding",
                "type": "continuous",
            },
            f"fund_high_neg_mom_self_z_{_h}h": {
                "family": "funding",
                "type": "continuous",
            },
            f"mark_trigger_dislocation_{_h}h": {
                "family": "mark_index",
                "type": "continuous",
            },
            f"mark_trigger_dislocation_self_z_{_h}h": {
                "family": "mark_index",
                "type": "continuous",
            },
        }
    )

for _key in KRAKEN_INDEX_PREMIUM_FEATURE_KEYS:
    CONTINUOUS_REGIME_FEATURES.pop(_key, None)

RIDGE_FEATURE_COLS = list(CONTINUOUS_REGIME_FEATURES.keys())

CONTINUOUS_TRIGGER_COLS = [
    "range_atr",
    "body_ratio",
    "upper_wick",
    "lower_wick",
    "upper_wick_ratio",
    "lower_wick_ratio",
    "wick_to_range",
    "acceleration_of_move",
    "acceleration",
    "acceleration_norm",
    "volume_spike",
    "orderflow_imbalance",
]

CONTINUOUS_LOCATION_COLS = [
    "dist_ema20_atr",
    "dist_ema50_atr",
    "dist_ema200_atr",
    "dist_vwap_atr",
    "dist_weekly_vwap",
    "vwap_zone_1d_atr",
    "vwap_zone_7d_atr",
    *OI_WEIGHTED_LOCATION_BASE_FEATURE_KEYS,
    *OI_WEIGHTED_LOCATION_META_FEATURE_KEYS,
    "dist_prior_day_high",
    "dist_prior_day_low",
    "dist_rolling_7d_high",
    "dist_local_swing",
    "dist_range_mid_atr",
    "dist_ma100_atr",
    "distance_to_ema",
    "pullback_depth",
    "zscore_price_50",
    "zscore_price_200",
    "loc_ema_stack_pos_24",
    "loc_ema_stack_pos_48",
    "loc_vwap_dev_z_24",
    "loc_vwap_dev_z_48",
    "loc_range_pos_24",
    "loc_range_pos_48",
    "loc_prior_bar_pos_24",
    "loc_prior_bar_pos_48",
    "loc_swing_range_pos_24",
    "loc_swing_range_pos_48",
    "loc_session_pos_24",
    "loc_session_pos_48",
    "loc_initial_balance_pos_24",
    "loc_initial_balance_pos_48",
    "loc_prev_day_range_pos_24",
    "loc_prev_day_range_pos_48",
    "loc_prev_week_range_pos_24",
    "loc_prev_week_range_pos_48",
    "loc_bb_channel_pos_24",
    "loc_bb_channel_pos_48",
    "loc_pullback_depth_24",
    "loc_pullback_depth_48",
    "loc_pivot_ladder_pos_24",
    "loc_pivot_ladder_pos_48",
    "bars_since_ema20_ema50_cross_log_norm",
    "bars_in_high_vol_state_log_norm",
    "bars_outside_ema20_atr_band_log_norm",
    "up_down_semivol_ratio_tanh",
    "up_down_return_mass_ratio_tanh",
    "tail_asymmetry_q90_q10_atr_norm",
]


FEATURE_SELECTION_KEYS = [
    "base_shared_feature_keys",
    "meta_shared_feature_keys",
]


TRAINING_RESIDUALIZATION_FEATURE_KEYS = [
    "ema50_ema200_spread_continuous",
    "atr_change_rate_ts_continuous",
    "bars_in_high_vol_state_log_norm",
    "volatility_of_volatility_48",
    "trend_strength_percentile",
    "volatility_autocorr_48",
]

ROLLING_ALPHA_FEATURE_KEYS = [
    "ra_ret1h_robust_z",
    "ra_ret5h_robust_z",
    "ra_ret24h_robust_z",
    "ra_resid_ret_6h_robust_z",
    "ra_rv24h_robust_z",
    "ra_log_quote_volume_robust_z",
    "ra_amihud_robust_z",
    "ra_symbol_minus_mkt_ret_4h_cs_rank",
    "ra_symbol_minus_mkt_ret_24h_cs_rank",
    "ra_market_dispersion_24h_pct",
    "ra_market_breadth_24h_pct",
    "ra_market_beta_24h",
    "ra_market_resid_ret_6h",
]

ROLLING_ALPHA_TARGET_AUDIT_COLUMNS = [
    "target_gross_residual_alpha_5h",
    "raw_gross_residual_alpha_5h",
    "market_factor_component_5h",
    "cluster_factor_component_5h",
    "market_beta_5h",
    "cluster_beta_5h",
    "gross_residual_alpha_scale_5h",
    "kalman_gross_residual_alpha_5h",
]

CURRENT_REGIME_AE_FEATURE_KEYS = [
    "z_ae_1",
    "z_ae_2",
    "z_ae_3",
    "z_ae_4",
    "z_ae_5",
    "z_ae_6",
    "z_ae_7",
    "z_ae_8",
    "ae_reconstruction_error",
    "ae_reconstruction_error_percentile",
    "ae_latent_norm",
    "ae_latent_norm_percentile",
    "ae_latent_distance",
    "ae_latent_distance_percentile",
]


LOC_CONTINUOUS_FAMILY_MAP = {
    "loc_ema_stack_pos_24": "trend",
    "loc_ema_stack_pos_48": "trend",
    "loc_vwap_dev_z_24": "liquidity",
    "loc_vwap_dev_z_48": "liquidity",
    "loc_range_pos_24": "context",
    "loc_range_pos_48": "context",
    "loc_prior_bar_pos_24": "context",
    "loc_prior_bar_pos_48": "context",
    "loc_swing_range_pos_24": "context",
    "loc_swing_range_pos_48": "context",
    "loc_session_pos_24": "context",
    "loc_session_pos_48": "context",
    "loc_initial_balance_pos_24": "context",
    "loc_initial_balance_pos_48": "context",
    "loc_prev_day_range_pos_24": "context",
    "loc_prev_day_range_pos_48": "context",
    "loc_prev_week_range_pos_24": "context",
    "loc_prev_week_range_pos_48": "context",
    "loc_bb_channel_pos_24": "compression",
    "loc_bb_channel_pos_48": "compression",
    "loc_pullback_depth_24": "path_structure",
    "loc_pullback_depth_48": "path_structure",
    "loc_pivot_ladder_pos_24": "context",
    "loc_pivot_ladder_pos_48": "context",
    "vwap_zone_1d_atr": "liquidity",
    "vwap_zone_7d_atr": "liquidity",
    **{
        _key: "open_interest_location"
        for _key in OI_WEIGHTED_LOCATION_BASE_FEATURE_KEYS
    },
    **{
        _key: "open_interest_location"
        for _key in OI_WEIGHTED_LOCATION_META_FEATURE_KEYS
    },
}

REGIME_FEATURE_KEYS = [
    "bars_since_ema20_ema50_cross_log_norm",
    "bars_in_high_vol_state_log_norm",
    "bars_outside_ema20_atr_band_log_norm",
    "up_down_semivol_ratio_tanh",
    "up_down_return_mass_ratio_tanh",
    "tail_asymmetry_q90_q10_atr_norm",
]

# Helper/base features produced in features.py that should remain selectable by model heads.
# This increases candidate breadth before MDI pruning.
HELPER_BASE_FEATURES = [
    # Kalman Meta Features
    "kalman_price",
    "price_innovation_z",
    "rolling_std(price_innovation)",
    "kalman_gain_1h",
    "state_uncertainty_1h",
    "vol_state_slope_1h",
    "realized_vol_minus_vol_state",
    "log_volume_state_1h",
    "volume_state_slope_1h",
    "price_slope_x_volume_surprise",
    "vol_state_x_volume_state",
    "ret1h",
    "ret2h",
    "ret4h",
    "ret6h",
    "ret8h",
    "ret48h",
    "ret72h",
    "ret120h",
    "atr_pct_base",
    "rsi_base",
    "rsi_slope_base",
    "qv",
    "log_quote_volume",
    "quote_volume_z_30d",
    "dist_ema_fast_base",
    "dist_ema_slow_base",
    "trend_pct_base",
    "signed_vol",
    "up_vol",
    "dn_vol",
    "up_vol_6",
    "dn_vol_6",
    "clv",
    "clv_mean_2",
    "excess_12h",
    "speed",
    "atr_expansion",
    "stall_ext_corr",
    "G_EXH_EFFORT",
    "G_EXH_GIVEBACK",
    "G_EXH_TAIL_FAIL",
    "G_TF_TREND",
    # FFD d-specific helper features
    "ffd_diff_1_04",
    "ffd_diff_2_04",
    "ffd_diff_4_04",
    "ffd_diff_8_04",
    "ffd_diff_1_05",
    "ffd_diff_2_05",
    "ffd_diff_4_05",
    "ffd_diff_8_05",
    "ffd_diff_1_06",
    "ffd_diff_2_06",
    "ffd_diff_4_06",
    "ffd_diff_8_06",
    "ffd_ema_spread_04",
    "ffd_ema_spread_05",
    "ffd_ema_spread_06",
    "ffd_rv_12_04",
    "ffd_rv_12_05",
    "ffd_rv_12_06",
    "ffd_rv_24_04",
    "ffd_rv_24_05",
    "ffd_rv_24_06",
    "ffd_z_24_04",
    "ffd_z_24_05",
    "ffd_z_24_06",
    "ffd_range_24_04",
    "ffd_range_24_05",
    "ffd_range_24_06",
    "ffd_slope_04_12",
    "ffd_slope_04_24",
    "ffd_mr_z_04",
    "ffd_mr_z_05",
    "ffd_d1_05",
    "ffd_d4_05",
    "ffd_ctx_slope_04_12",
    "ffd_ctx_slope_04_24",
    # Range features for event scoring and candidate selection
    "range_pct",
    "range_12h_pct",
    "range_16h_pct",
    "range_24h_pct",
    # FFD d-specific advanced features
    "ffd_rv_2h_04",
    "ffd_rv_6h_04",
    "ffd_rv_24h_04",
    "ffd_vol_price_corr_10h_04",
    "ffd_donch_dist_04_12",
    "ffd_donch_dist_04_24",
    "ffd_donch_dist_04_48",
    "ffd_dist_ema_fast_04",
    "ffd_dist_ema_slow_04",
    "ffd_rv_2h_06",
    "ffd_rv_6h_06",
    "ffd_rv_24h_06",
    "ffd_accel_06",
    "ffd_z_06",
    "ffd_vol_price_corr_10h_06",
    "ffd_donch_dist_06_12",
    "ffd_donch_dist_06_24",
    "ffd_donch_dist_06_48",
    "ffd_atr_expansion_06",
    "ffd_strength_04",
    "ffd_strength_05",
    "ffd_strength_06",
]
HELPER_BASE_FEATURES = list(
    dict.fromkeys(
        HELPER_BASE_FEATURES + PRICE_MEMORY_FEATURE_KEYS + DAILY_SR_BASE_FEATURE_KEYS
    )
)

# Compact feature basket for learnability tests across symbol universes,
# TBM geometry settings, and sample-weight configurations.
# Emphasis: 2/4/8-bar behavior + longer-horizon regime context.
TEST_FEATURE_KEYS = [
    # Realized vol / ATR (multi-horizon)
    "rv_2h",
    "rv_4h",
    "rv_8h",
    "rv_24h",
    "atr_pct_change",
    # Returns + slope family (2/4/8 focus)
    "ret2h",
    "ret4h",
    "ret8h",
    "ret24h",
    "slope",
    "atr_slope",
    "rsi_slope",
    # Momentum acceleration
    "momentum_accel",
    "accel",
    "accel_5h",
    # Price distance / z-score style context (EMA / VWAP / breakout band proxies)
    "dist_ema_fast",
    "dist_vwap_norm",
    "breakout_t",
    "pct_breakout_t",
    "ret1h_z",
    # RVOL + volume acceleration
    "rvol_z",
    "vol_z",
    "vol_z_4h",
    "dlog_vol_5h",
    "volume_entropy_12",
    # Vol-of-vol
    "vov_ratio",
    "vov_fast_slow_ratio",
    "vov_mad_20",
    # Autocorrelation / Hurst-ish / path efficiency proxies
    "autocorr_6h",
    "autocorr_24h",
    # "hurst_proxy_24",
    "path_efficiency_12",
    "path_efficiency_24",
    # Liquidity + time-of-day
    # Liquidity
    "amihud_illiq",
    "amihud_z",
    # Mid/long lookback context for 8-bar horizon learnability (16-24h + slower)
    "ret16h",
    "coherence_24",
    "impulse_ratio_24",
    "range_24h_pct",
    # shannon_entropy_ret_16",
    # "perm_entropy_ret_24",
    "spectral_entropy_ret_24",
    "volume_entropy_24",
    # Longer-timeframe regime context
    "vol_regime_z",
    "regime_stability_24h",
    # Ridge model features
    "compression_ratio",
    "trend_strength_percentile",
    "bollinger_band_width",
    "direction_entropy_20",
    "volatility_ratio_short_long",
    "volume_percentile",
]
TEST_FEATURE_KEYS = list(dict.fromkeys(TEST_FEATURE_KEYS + LGBM_PERP_FEATURE_KEYS))

CFG = {
    # persistence / fetch
    "data_root": "data",
    "reports_root": "reports",
    "hf_data_dir": "15m_ohlcv",
    "use_perps": False,
    "exchange_scoped_data": True,
    # Feature portability policy. The default deployable feature contract is
    # asset-portable: exchange/source-specific, dataset-selected, and state-tuned
    # feature families are removed instead of neutral-filled. Cross-asset and
    # basket features remain eligible when computed over a broad live universe.
    "feature_portability_mode": "cross_asset_portable",
    "feature_portability_strict": True,
    "feature_portability_fixed_basket": False,
    "feature_portability_allow_volume_source_dependent": False,
    "feature_portability_allow_dataset_selected": False,
    "feature_portability_allow_state_tuned": False,
    # The model-facing ``ob_*`` family is a causal hourly OHLCV/kline
    # microstructure proxy. Actual live L2 is used only for executable-price,
    # slippage and capacity controls.
    "enable_orderbook_features": True,
    "orderbook_stale_hours": 2,
    "orderbook_levels": 20,
    "orderbook_depth_bps": [5, 10, 25, 50, 100],
    "orderbook_wall_qty_mult": 3.0,
    "orderbook_missing_age_sentinel_min": float("nan"),
    # Historical microdata is built from hourly kline summaries, not true L2
    # depth snapshots. Wall/blocker primitives require real book levels and are
    # neutralized by the runtime summary injector.
    "enable_orderbook_wall_features": False,
    "enable_cross_asset_features": True,
    "enable_cross_sectional_rank_features": False,
    "cross_asset_reference_symbols": ["BTC/USDT", "ETH/USDT"],
    "feature_refresh_microdata_before_compute": False,
    "timeframe": "1h",
    "fetch_years": 4,
    "fetch_symbols_M": 9999,
    # Download controls (run_pipeline.py download)
    # - order: volume | alpha_asc | alpha_desc
    # - stride: process every Nth symbol (2 ~= half runtime/symbols)
    # - max_symbols: 0 keeps all post-stride symbols
    "download_symbol_order": "alpha_desc",
    "download_symbol_stride": 2,
    "download_symbol_offset": 0,
    "download_max_symbols": 0,
    "download_partition_count": 1,
    "download_partition_id": 0,
    "download_force": False,
    "download_check_complete": True,
    "download_skip_if_missing_lt_days": 3.0,
    "download_15m_enabled": True,
    "download_15m_full_backfill": True,
    "download_microdata_enabled": True,
    "policy_optimiser_tail_months": 4,
    "policy_optimiser_holdout_start_months_ago": 16,
    "policy_optimiser_holdout_end_months_ago": 12,
    "policy_optimiser_recent_weeks_enable": True,
    "policy_optimiser_optimise_start_weeks_ago": 8,
    "policy_optimiser_optimise_end_weeks_ago": 0,
    "policy_optimiser_validation_start_weeks_ago": 13,
    "policy_optimiser_validation_end_weeks_ago": 9,
    "policy_optimiser_max_sample_fraction": 0.30,
    # Feature artifacts are keyed by run-id, but feature rows should extend near
    # the available data frontier rather than stopping at the historical run-id.
    "feature_generation_end_lag_days": 0,
    "offline_backtest_skip_universe_refresh": True,
    # feature transformation remediation
    "ffd_d_values": [0.4, 0.5, 0.6],
    "ffd_d_default": [0.4, 0.5, 0.6],
    "ffd_d_base": 0.4,
    # Family-level d priorities (primary first)
    # impulse/event momentum diffs -> fastest shock reaction
    "ffd_impulse_d_values": [0.6, 0.5],
    # carry/move continuation -> between impulse and context
    "ffd_carry_d_values": [0.5, 0.4],
    # context/trend under noise -> slowest of triad
    "ffd_context_d_values": [0.4],
    "ffd_thres": 1e-5,
    "ffd_mr_window": 24,
    "ffd_slope_windows": [12, 24],
    "atr_ln_floor": 1e-6,
    "safe_log_eps": 1e-9,
    # market basket
    "market_basket": [
        "BTC/USDT",
        "ETH/USDT",
        "BNB/USDT",
        "SOL/USDT",
        "XRP/USDT",
        "ADA/USDT",
        "DOGE/USDT",
        "TRX/USDT",
        "AVAX/USDT",
        "LINK/USDT",
        "LTC/USDT",
        "BCH/USDT",
        "DOT/USDT",
        "MATIC/USDT",
        "ATOM/USDT",
        "UNI/USDT",
        "ETC/USDT",
        "XLM/USDT",
        "FIL/USDT",
        "APT/USDT",
        "NEAR/USDT",
        "ARB/USDT",
        "OP/USDT",
        "INJ/USDT",
        "SUI/USDT",
        "SEI/USDT",
        "AAVE/USDT",
        "MKR/USDT",
        "PEPE/USDT",
        "WIF/USDT",
    ],
    # training horizons to compare
    # Canonical set is [1, 2, 4] hours. H1 added for entry timing; H8 removed.
    "label_horizons_hours": CANON_HORIZONS,
    "base_geometry_archetypes": ["tight", "balanced", "wide"],
    "base_geometry_train_variants": False,
    "base_skip_primary_variant": False,
    "base_geometry_grr_topk": 12,
    "base_geometry_learnability_weight": 0.75,
    "base_geometry_geometry_weight": 0.25,
    "label_horizons_use_shorter_grid": True,
    "label_tp_values_pct": [1.5, 2.0, 3.0, 4.0, 5.0, 6.0],
    "label_sl_values_pct": [0.5, 1.0, 2.0],
    "label_round_trip_fee_pct": 0.5,
    "policy_fee_rt": 0.003,
    "label_min_net_rr": 0.9,
    "label_min_tp_hit_rate": 0.02,
    "label_max_timeout_rate": 0.90,
    # Base label handling: exclude timeout (TO) from TP-vs-SL base classifier targets
    "base_exclude_timeout_from_classifier": False,
    # ATR normalization for barrier scaling
    "atr_norm_fast_hl_hours": 24,
    "atr_norm_slow_hl_hours": 24 * 5,
    "atr_norm_global_hl_hours": 24 * 5,
    "atr_norm_warmup_hours": 24 * 10,
    "atr_norm_clip_global": [0.7, 1.5],
    "atr_norm_clip_scale": [0.6, 2.5],
    # Consensus weight blending
    "consensus_amp": 0.25,
    "consensus_k": 2.0,
    "consensus_beta": 0.20,
    "train_lookback_hours": 24 * 365 * 4,  # 4 years
    "val_lookback_hours": 24 * 7,  # 7d validation (time-split, no leakage)
    "min_train_samples": 200,
    "base_min_samples_hard_floor": 3000,
    "mr_tf_masks": {
        "enabled": False,
        "optuna_enabled": False,
        "q_adx_tf": 0.65,
        "q_adx_mr": 0.45,
        "q_stretch_mr": 0.75,
        "q_persist_tf": 0.70,
        "q_persist_mr": 0.30,
        "q_tf_quality": 0.60,
        "q_mr_quality": 0.60,
        "N_tf": 3,
        "N_mr": 2,
        "ema_gap_min_tf": 0.0,
        "mom_min_tf": 0.0,
        "stretch_min_mr": 0.75,
        "reversal_min_mr": 0.0,
        "persistence_axis": "none",
        "tf_quality_axis": "none",
        "mr_quality_axis": "none",
        "min_train_samples": 400,
        "optuna_trials": 300,
        "optuna_patience": 40,
        "optuna_n_startup_trials": 8,
        "optuna_n_warmup_steps": 0,
        "optuna_use_numba": True,
        "optuna_numba_min_rows": 20000,
        "support_loss_hurdle": 0.0,
        "support_loss_hurdle_ratio": 0.0015,
        "support_loss_hurdle_floor": 5e-5,
        "support_loss_quadratic_multiplier": 2.0,
        "support_loss_hard_veto": False,
        "support_value_power": 1.25,
        "min_coverage": 0.10,
        "min_earned_quality_uplift": 0.0,
        "promotion_margin": 0.0,
        "promotion_top_frac": 0.30,
    },
    # Sample caps (symbol-balanced subsampling when exceeded)
    "base_fit_max_samples": 0,
    "base_selector_max_samples": 30000,
    "base_variant_fit_max_samples": 0,
    "base_variant_selector_max_samples": 30000,
    "meta_fit_max_samples": 0,
    # Broad global LGBM base runs rehydrate many feature-store columns per
    # side/horizon head. Keep this bounded before feature materialization.
    "lgbm_train_base_rehydrate_row_cap": 60000,
    # MFE/MAE-based sample weighting (Report 2026-02-12)
    # Weight samples by how "decisive" the price movement was relative to barriers
    # w = w_min + (1-w_min) * clip(max(MFE/TP, MAE/SL) / tau, 0, 1)
    # Timeout samples are capped at 0.7
    "mfe_mae_w_min": 0.5,  # Minimum weight floor
    "mfe_mae_tau": 1.0,  # Scaling factor (d/tau)
    "mfe_mae_cost_floor": 0.001,  # Cost floor for touch margin penalty
    # MR path-aware weighting (de-emphasize raw magnitude, emphasize efficient path)
    "mr_weight_magnitude_power": 0.35,
    "mr_weight_mfe_tau": 1.0,
    "mr_weight_mae_tau": 1.0,
    # MR utility target: velocity decay horizon for bars_to_mfe penalty
    "mr_utility_horizon_bars": 8,
    # Meta model sample weighting
    # Magnitude sigmoid: w = 1 + alpha * sigmoid((|ret| - q60) / std)
    # alpha=0.2 gives top-40% ~1.1-1.2x upweight (very slight emphasis)
    "meta_weight_sigmoid_alpha": 0.2,
    # MFE/MAE quality: w_exc = 0.5 + 0.5 * clip(max(MFE/barrier, MAE/barrier) / tau, 0, 1)
    "meta_mfe_mae_tau": 1.0,
    # Sample-weight optimization (base + meta)
    "sample_weight_opt_enable": False,
    "sample_weight_opt_min_samples": 400,
    "sample_weight_opt_trials": 16,
    "meta_sample_weight_opt_trials": 12,
    "meta_use_policy_value_target": False,
    "meta_clf_enabled": True,
    "meta_clf_type": "binary_move_softladder",
    "meta_clf_move_thresholds": [1.00, 1.50, 2.00],
    "meta_clf_move_weights": [0.50, 0.30, 0.20],
    "meta_clf_use_calibration": True,
    "meta_clf_use_class_weight_multiplier": True,
    "meta_clf_max_class_weight": 10.0,
    # Legacy compatibility knobs; the binary move head no longer consumes
    # engine-derived TP/TIME/SL multiclass labels by default.
    "meta_clf_use_engine_labels": False,
    # Policy-aligned downstream sizing requires the binary move probability
    # (oof_p_move) in meta_oof exports.
    "meta_race_include_classifiers": True,
    "meta_require_classifier_barrier_probs": True,
    "meta_train_regression_bucket_model": False,
    "meta_train_correctness_clf_head": False,
    "meta_train_tbm_clf_head": True,
    "meta_export_base_meta_corrected_head": False,
    "model_backend": "lgbm_pipeline",
    "base_model_backend": "lgbm_pipeline",
    "meta_model_backend": "lgbm_pipeline",
    "meta_base_gate_min_en_lift30": 1.15,
    "base_good_enough_min_en_lift30": 1.15,
    "legacy_en_uncertainty_combiner_enabled": False,
    "meta_training_pipeline_version": "legacy",
    "meta_train_save_legacy_setup": True,
    "meta_parallel_forest_disable_hpo": False,
    "meta_parallel_forest_num_parallel_tree": 20,
    "meta_parallel_forest_rounds": 100,
    "meta_parallel_forest_max_depth": 5,
    "meta_parallel_forest_learning_rate": 0.05,
    "meta_parallel_forest_reg_alpha": 2.0,
    "meta_parallel_forest_reg_lambda": 15.0,
    "meta_parallel_forest_min_child_weight": 20.0,
    "meta_parallel_forest_gamma": 1.5,
    "meta_parallel_forest_early_stopping_rounds": 40,
    "meta_map_tbm_geometries": [
        {"name": "tbm_500_250", "tp_pct": 0.05, "sl_pct": 0.025},
        {"name": "tbm_250_125", "tp_pct": 0.025, "sl_pct": 0.0125},
    ],
    "meta_map_tbm_horizons": [1, 2, 4],
    "meta_map_mae_horizons": [2, 4],
    "meta_map_mfe_horizons": [2, 4],
    "meta_map_weight_clip_lo": 0.5,
    "meta_map_weight_clip_hi": 1.5,
    # Meta classifier utility-based winner selection (logloss remains a gate)
    "meta_clf_max_logloss": 1.10,
    "meta_clf_u_tp": 1.0,
    "meta_clf_u_to": 0.0,
    "meta_clf_u_sl": -2.5,
    "meta_clf_top_frac": 0.15,
    "meta_move_top_frac": 0.15,
    "meta_trade_topx_values": [40],
    "enable_recent_effectiveness_features": True,
    "recent_effectiveness_top_frac": 0.15,
    "recent_effectiveness_min_samples": 100,
    "recent_effectiveness_min_top_samples": 25,
    "meta_product_feature_keys": [
        "trend_slope_24h",
        "trend_slope_48h",
        "trend_slope_72h",
        "vol_z",
        "vol_z24",
        "volatility_zscore",
        "efficiency_ratio_20",
        "compression_score",
        "dist_ema_fast",
        "dist_vwap_norm",
        "loc_vwap_dev_z_24",
        "loc_vwap_dev_z_48",
    ],
    "meta_clf_min_top_n": 50,
    "meta_clf_min_lift_vs_baseline": 0.0,
    "meta_clf_dynamic_utility_from_realized": True,
    "meta_clf_require_positive_oof_utility": True,
    # Smooth utility proxy computed deterministically from predicted MFE/MAE.
    "meta_utility_smooth_tp": 0.02,
    "meta_utility_smooth_sl": 0.01,
    "meta_utility_smooth_alpha": 6.0,
    "meta_utility_smooth_alpha_grid": [2.0, 4.0, 6.0, 8.0, 10.0, 12.0],
    "meta_utility_smooth_tp_quantile": 0.60,
    "meta_utility_smooth_sl_quantile": 0.60,
    "meta_utility_smooth_quantile_blend": 0.50,
    "meta_utility_smooth_tp_min": 0.003,
    "meta_utility_smooth_tp_max": 0.250,
    "meta_utility_smooth_sl_min": 0.002,
    "meta_utility_smooth_sl_max": 0.250,
    "meta_utility_smooth_use_zscore": True,
    "meta_utility_smooth_use_predicted_mfe_mae": True,
    "meta_utility_smooth_loss": "huber",
    "meta_utility_smooth_loss_weight": 1.0,
    # MAE auxiliary head target / weighting race
    "aux_mae_target_variants": ["rank_pct", "qbin_mid"],
    "aux_mae_weight_variants": [
        "none",
        "asymmetric_tail",
        "symmetric_tail",
        "top30_tail",
    ],
    "aux_mae_qbin_bins": 20,
    "aux_mfe_target_variants": ["rank_pct", "qbin_mid"],
    "aux_mfe_weight_variants": [
        "none",
        "asymmetric_tail",
        "symmetric_tail",
        "top30_tail",
    ],
    "aux_mfe_qbin_bins": 20,
    "aux_head_rank_tail_start": 0.70,
    "aux_head_rank_tail_amp": 0.50,
    # Aux-head selection objective (top-trade focused)
    "aux_head_select_top_frac": 0.30,
    "aux_head_select_w_ic_top": 0.70,
    "aux_head_select_w_ic_all": 0.10,
    "aux_head_select_w_mono": 0.10,
    "aux_head_select_w_stability": 0.15,
    "aux_head_select_w_stability_top30": 0.15,
    "aux_head_select_w_ece_top": 0.20,
    "aux_head_weight_lambda_grid": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "aux_head_weight_min_gain_vs_none": 1e-4,
    "aux_head_weight_topk_tolerance": 1e-4,
    # Stage-1 ablation uses one comparator model to avoid combinatorial explosion.
    "aux_head_ablation_model": "extratrees",
    "aux_head_ablation_et_estimators": 120,
    "aux_head_ablation_lgbm_estimators": 200,
    "aux_head_weight_optuna_trials": 12,
    "aux_head_weight_optuna_inner_splits": 3,
    "aux_head_ridge_alpha_min": 1e-3,
    "aux_head_ridge_alpha_max": 100.0,
    "aux_head_ridge_alpha_default": 1.0,
    # Stage-2 model race is run only on the target/weight winner from stage-1.
    "aux_head_run_model_race_on_winner": False,
    "aux_head_model_race_candidates": ["xgb_parallel_forest"],
    # Two-stage/three-head position sizer defaults.
    "position_sizer_enabled": True,
    "ev_decomposition_enabled": True,
    "ev_decomposition_train_in_meta": True,
    "position_sizer_backend": "ev_decomposition",  # ev_decomposition runtime bundle backend (offline sizer mode is ridge)
    "position_sizer_allow_fallback": False,
    "position_sizer_allow_unknown_bundle_version": False,
    "position_sizer_ev_threshold": 0.0,
    "position_sizer_costs_mode": "included_in_labels",
    "position_sizer_exp_win_quantile": 0.50,
    "position_sizer_risk_loss_quantile": 0.90,
    "position_sizer_calibration_method": "isotonic",
    "position_sizer_calibration_scope": "regime",
    "position_sizer_calibration_rolling_window": 2000,
    "position_sizer_p_min": 1e-3,
    # Soft pwin label from MFE/MAE smooth TP/SL proxy
    "position_sizer_pwin_soft_label_enabled": False,
    "position_sizer_pwin_soft_label_tp": 0.02,
    "position_sizer_pwin_soft_label_sl": 0.01,
    "position_sizer_pwin_soft_label_alpha": 15.0,
    "position_sizer_pwin_soft_label_target_mapping": "linear_to_01",
    "position_sizer_pwin_soft_label_loss": "bce",
    "position_sizer_pwin_soft_label_use_log_excursions": False,
    "position_sizer_pwin_soft_label_log_eps": 1e-12,
    # Position-sizer features (meta OOF prediction heads + market microstructure).
    # Used by Ridge sizer, ExtraTrees sizer, and barrier classifier.
    # Regime features removed — regime context is handled by the multiplier_band
    # mechanism in replay_exit_policy, not by inflating the feature space.
    "position_sizer_features": [
        # Keep core classifier and selected regime/liquidity features.
        "oof_p_tp",
        "oof_p_sl",
        "oof_p_time",
        "volatility_zscore",
        "clv_t",
        "body_ratio_15m",
        "rejection_proxy",
        "range_norm_12",
        "sv_imb_24",
        "press_24",
        "impact_24",
        "ts_24",
        "atr_12_15m",
        "clf_center",
        "clf_entropy",
        "base_clf_centered",
        "oof_base_meta_correctness_prob",
        "oof_ebm_raw",
        "oof_ebm_en",
        "oof_ebm_uncertainty_weighted",
        "meta_en_x_vol_z",
        "meta_en_x_trend",
        "meta_en_x_compression",
        "meta_en_x_efficiency",
        "meta_en_x_trend_x_vol",
        "meta_en_x_liquidity",
        "clf_prefix_std",
        "clf_leaf_support_q25",
        "clf_leaf_target_iqr_mean",
        # --- Conditionally available (regressor heads) ---
        "reg",
        "reg_mean",
        "reg_std",
        "reg_range",
        "reg_pred",
        "reg_prefix_std",
        "reg_leaf_support_q25",
        "reg_leaf_target_iqr_mean",
        "utility",
        "mae_q70",
        "mfe",
        "oof_u_hat",
        "oof_log_mae_q70_hat",
        "oof_log_mfe_hat",
        "oof_asym_hat",
        "Upside",
        "Downside",
        "EdgeSharpe",
        "risk_reward_ratio",
        "high_utility_pred",
        "risk_adjusted_pred",
        "utility_disagreement",
        "sign_agree",
        "joint_confidence",
        "conflict_score",
        "joint_instability",
        "edge_unc_pen",
        "edge_support_pen",
        "edge_noise_pen",
        # --- Market regime features ---
        "vol_of_vol",
        "volatility_of_volatility_48",
        "vov_ratio",
        "vov_fast_slow_ratio",
        "rv_24h",
        "rv_4h",
        "rv_2h",
        "realized_volatility_24h",
        "range_expansion_ratio",
        "range_24h_pct",
        "range_12h_pct",
        "atr_pct",
        "atr_pct_base",
        "asset_atr_level",
        "asset_atr_level_pct",
        "asset_vol_level",
        "asset_vol_level_pct",
        "adx_14",
        "adx_10",
        "adx_zscore",
        "trend_regime",
        "trend_slope_48h",
        "dist_ema_fast",
        "dist_ema_fast_z",
        "dist_ema_slow_base",
        "dist_vwap_norm",
        "dist_vwap_norm_z",
        "dist_vwap_24_atr",
        "loc_vwap_dev_z_24",
        "loc_vwap_dev_z_48",
        "dist_prior_day_low",
        "dist_prior_day_high",
        "loc_prev_day_range_pos_24",
        "loc_prev_day_range_pos_48",
        "loc_prev_week_range_pos_48",
        "spectral_entropy_ret_24",
        "perm_entropy_ret_24",
        "shannon_entropy_ret_16",
        "regime_transition_entropy_48h",
        "amihud_illiq",
        "amihud_z",
        "atr_pct_x_amihud_z",
        "rvol_z_x_range_expansion",
        # Meta-clf reliability diagnostics (causal rolling, top-decile scoped).
        "winrate_20",
        "winrate_50",
        "brier_50",
        "logloss_50",
        "rank_ic_50",
        "consecutive_losses",
        "edge_tbm",
        "edge_x_winrate_20",
        "edge_x_rank_ic_50",
        "edge_x_brier_50",
        "edge_x_volatility_zscore",
        "edge_x_amihud_z",
        "edge_x_vol_of_vol",
    ],
    # Backward-compatible aliases (point to the same list at runtime)
    "position_sizer_feature_priority": None,
    "limit_offset_sizer": None,
    # When running `run_pipeline.py sizer`, also run OOS backtest to emit
    # financial metrics (PnL/Sortino/etc.) with the freshly trained sizer.
    "sizer_run_oos_backtest": True,
    # Ranking-based allocation engine (capital allocation > prediction IC)
    "ranking_trade_percentile_threshold": 0.90,
    "ranking_rank_exponent": 2.0,
    "ranking_size_k": 1.0,
    "ranking_max_position_size": 1.0,
    "ranking_risk_epsilon": 1e-6,
    # Optional score sharpening (no retraining)
    "score_sharpening_alpha_power": 2.0,
    "score_sharpening_score_temperature": 0.7,
    # Turnover regularization hook (for future allocator objective)
    "turnover_control_turnover_lambda": 0.0,
    # TP/SL default selection for classifier-facing defaults.
    "tp_sl_search_enabled": False,
    "tp_sl_search_optimizer": "legacy",  # legacy | new
    "tp_sl_override": False,
    "tp_sl_search_k_tp_grid": [0.8, 1.0, 1.25, 1.5, 2.0],
    "tp_sl_search_k_sl_grid": [0.5, 0.75, 1.0, 1.25],
    "tp_sl_search_alpha_sigmoid": 15.0,
    "tp_sl_search_min_trades_per_fold": 200,
    "objective_mar": 0.0,
    "objective_eps_log": 1e-12,
    "objective_eps_sortino": 1e-12,
    "objective_composite_mode": "hard_gate",
    "objective_composite_q_top": 0.95,
    "objective_composite_selection": "min_std",
    "objective_scaling_elg_scale": 10000.0,
    "objective_scaling_mnpt_scale": 10000.0,
    "objective_clipping_elg_min": -1.0,
    "objective_clipping_elg_max": 1.0,
    "objective_clipping_sortino_min": -10.0,
    "objective_clipping_sortino_max": 10.0,
    "objective_clipping_mnpt_min": -1.0,
    "objective_clipping_mnpt_max": 1.0,
    # Ridge sizer target selection objective
    "sizer_select_metric": "topq_u_policy",
    "sizer_topq": 0.30,
    "sizer_require_positive_topq_u": True,
    "sizer_topq_min_samples": 50,
    "sizer_winsor_q_low": 0.01,
    "sizer_winsor_q_high": 0.99,
    # Passive limit offset optimizer constants
    "TICK_SIZE_BPS": 2.0,
    "K_MAX": 5,
    "HORIZON_15M_BARS": 4,
    "HORIZON_1H_BARS": 1,
    "UTILITY_LAMBDA": 0.0,
    "UTILITY_ETA": 0.0,
    "SOFTARGMAX_TAU": 1.0,
    "label_policy_optimizer_enabled": True,
    "label_policy_probe_alpha": 1.0,
    "label_policy_sortino_beta": 0.01,
    "label_policy_lambda": 0.5,
    "label_policy_max_timeout": 0.80,
    "label_policy_plateau_eps": 0.02,
    # Economic gate on base race: require positive realized return in top-k OOF slice
    "base_require_positive_oof_expectancy": True,
    "base_oof_expectancy_top_frac": 0.30,
    # Use ridge-sizer-aligned rollout labels when creating training rows
    "policy_rollout_labeling_enable": True,
    # Static policy geometry used upstream for policy-aligned labels/classifier
    # targets. This intentionally matches the ridge sizer's TP/SL/trailing
    # family, but stays fixed rather than using per-run optimized params.
    "policy_label_sl_atr_mult": 1.2,
    "policy_label_tp_sl_ratio": 2.0,
    "policy_label_trailing_pct": 0.35,
    "policy_label_max_hold_hours": 24,
    "META_EV_TP_SL_RATIO_SOURCE": "policy",
    "DEFAULT_TP_SL_RATIO": 2.0,
    "META_VALIDATE_CALIBRATION": True,
    "sample_weight_opt_n_splits": 5,
    "sample_weight_opt_embargo_bars": 10,
    "cv_embargo_bars": 12,
    "sample_weight_opt_min_n_eff_ratio": 0.30,
    "sample_weight_opt_max_top1pct": 0.05,
    "sample_weight_opt_model_family": "ExtraTrees",
    # Component controls
    "sample_weight_vol_direction": "downweight_high",
    "sample_weight_vol_power": 0.5,
    "sample_weight_vol_min_group_size": 20,
    "sample_weight_recency_half_life_bars": 24 * 30,
    "sample_weight_recency_min_era_neff_ratio": 0.2,
    "sample_weight_use_distance_component": True,
    "sample_weight_distance_form": "inverse",  # inverse | exp
    "sample_weight_distance_k": 0.5,
    "sample_weight_distance_min_dist": 0.5,
    # Native LGBM recency-weight HPO. This is intentionally disabled by
    # default; when enabled it reuses native LGBM preset features/params and
    # only searches the recency weighting contract.
    "recency_hpo_enabled": False,
    "recency_hpo_use_winner": True,
    "recency_hpo_persist_winner": True,
    "recency_hpo_train_years": 3,
    "recency_hpo_holdout_months": 2,
    "recency_hpo_base_half_life_months": [6.0, 9.0, 12.0],
    "recency_hpo_meta_half_life_months": [3.0, 4.5, 6.0],
    "recency_hpo_composite_weights": [0.3, 0.4, 0.5],
    "recency_hpo_confirm_with_distillation": True,
    "recency_hpo_confirmation_top_n": 3,
    "recency_hpo_require_distillation_confirmation": True,
    "recency_hpo_confirmation_score_tolerance": 1e-9,
    "recency_hpo_min_train_rows": 1000,
    "recency_hpo_min_oos_rows": 200,
    # Optional fixed-contract native-preset extension. When a source run's
    # native LGBM features/params are reused, this can append explicitly
    # approved extra features from a prior diagnostics report without rerunning
    # feature selection or HPO. Empty by default.
    "lgbm_native_preset_extra_feature_report": "",
    "lgbm_native_preset_extra_feature_report_slice": "top30",
    "lgbm_native_preset_extra_feature_allow_prefixes": [
        "mkt_",
        "xs_",
        "eig_",
        "q_",
        "xasset_mkt_spread_bps",
    ],
    "lgbm_native_preset_extra_feature_allow_exact": [
        "xasset_mkt_spread_bps",
        "xasset_mkt_spread_bps_z_24h",
        "regime_liquidity_score",
    ],
    "lgbm_native_preset_extra_feature_deny_exact": [
        "q_iqr__xasset_mkt_spread_bps",
        "q_tail_asym__xasset_mkt_spread_bps",
        "q_tail_width__xasset_mkt_spread_bps",
    ],
    "lgbm_native_preset_extra_allow_orderbook_features": False,
    # per-hour cross-sectional training selection
    "variance_filter_pct": 1.0,  # Keep all non-constant assets
    "variance_filter_stride": 100,
    "feature_use_local_store_universe": False,
    "label_variance_filter_enabled": False,
    "train_extreme_pct_hourly": 0.06,  # Keep top/bottom 6% as extreme candidates (reduced from 0.08)
    "train_extreme_min": 10,
    "train_extreme_max": 80,
    "train_min_range_pct": 0.07,
    "train_min_vol_zscore": 1.6,
    # Triple barrier geometry parameters (DEPRECATED - use unified barrier factory params below)
    # Old Pipeline 1 params:
    "train_z_max": 3.0,  # Max z-score clip (symmetric) - DEPRECATED
    "train_tp_k_lo": 0.5,  # TP lower bound = k_lo * base_ATR - DEPRECATED
    "train_tp_k_hi": 1.5,  # TP upper bound = k_hi * base_ATR - DEPRECATED
    "train_sl_mult_lo": 0.4,  # SL ratio in quiet markets - DEPRECATED
    "train_sl_mult_hi": 0.7,  # SL ratio in volatile markets - DEPRECATED
    # Unified barrier factory parameters (v3 - single source of truth for both pipelines)
    # Single barrier mode: k_tp and sl_base_mult are scalars
    "barrier_k_tp": 1.0,  # dimensionless k_tp for single geometry
    "barrier_sl_base_mult": 0.5,  # RR (e.g., 0.5 = 2:1 reward:risk)
    # TP bounds (match old scaled_atr_pct behavior: clamp to [tp_lo, tp_hi])
    "barrier_tp_lo": 0.02,  # Lower bound for TP (2%)
    "barrier_tp_hi": 0.06,  # Upper bound for TP (6%)
    # Multi-geometry mode: k_tp and sl_base_mult are grids
    "barrier_k_tp_grid": [0.8, 1.0, 1.25, 1.6, 2.0, 2.5],  # dimensionless k_tp grid
    "barrier_sl_base_grid": [
        0.5,
        1.0,
        1.5,
    ],  # RR grid (0.5 = 2:1, 1.0 = 1:1, 1.5 = 0.67:1)
    # Dispersion-based regime scaling
    "barrier_disp_floor": 0.1,  # MAD-based z-score floor (prevents division by near-zero)
    "barrier_z_max": 3.0,  # Max z-score clip (symmetric)
    "barrier_k_reg": 0.3,  # Regime tightness: lower = tighter TP in quiet markets
    "barrier_m_lo": 0.7,  # Multiplier low (quiet markets)
    "barrier_m_hi": 1.5,  # Multiplier high (volatile markets)
    "barrier_sl_lo": 0.4,  # SL ratio in quiet markets
    "barrier_sl_hi": 0.7,  # SL ratio in volatile markets
    "barrier_z_gate": 1.0,  # z-score threshold for regime transition
    # Horizon scaling
    "label_horizon_base": 4,  # base horizon for sqrt(H/H_base) scaling
    "label_min_net_rr": 0.9,  # min reward:risk ratio after fees
    # Optional rolling-alpha feature/target architecture. Disabled by default
    # because it changes the base regression objective and adds extra feature
    # computation.
    "rolling_alpha_features_enabled": False,
    "rolling_alpha_feature_norm_window": 336,
    "rolling_alpha_target_enabled": False,
    "rolling_alpha_target_mode": "legacy",
    "rolling_alpha_target_horizon_hours": 5,
    "rolling_alpha_target_transform": "asinh_scaled",
    "rolling_alpha_target_scale_window": 336,
    "rolling_alpha_target_scale_min_periods": 48,
    "rolling_alpha_target_scale_floor": 1e-4,
    "rolling_alpha_target_market_beta_window": 720,
    "rolling_alpha_target_cluster_beta_window": 720,
    "rolling_alpha_target_beta_min_periods": 168,
    "rolling_alpha_target_beta_var_floor": 1e-10,
    "rolling_alpha_target_default_market_beta": 1.0,
    "rolling_alpha_target_default_cluster_beta": 1.0,
    "rolling_alpha_target_cluster_columns": [
        "__asset_cluster__",
        "__cluster__",
        "asset_cluster",
        "cluster_label",
        "cluster_id",
        "liquidity_tier",
        "volatility_tier",
    ],
    "rolling_alpha_target_cluster_feature_columns": [
        "asset_vol_level_pct",
        "asset_atr_level_pct",
        "vol_z",
        "rvol_z",
        "trend_strength_percentile",
        "asset_volume_level_pct",
    ],
    "rolling_alpha_target_cluster_feature_max_columns": 3,
    "rolling_alpha_target_kalman_enabled": False,
    "rolling_alpha_target_kalman_blend": 0.35,
    "rolling_alpha_target_kalman_process_var": 1e-6,
    "rolling_alpha_target_kalman_obs_var": 1e-4,
    "rolling_alpha_target_clip_abs": 20.0,
    "simple_policy_regime_ae_enabled": True,
    "regime_ae_backend": "sklearn_mlp",
    "regime_ae_lookback_days": 92,
    "regime_ae_allow_full_history_fallback": False,
    "regime_ae_source_fail_closed": True,
    "regime_ae_candidate_generation": "walk_forward_prior_only",
    "regime_ae_oof_block_hours": 168,
    "regime_ae_walk_forward_min_prior_rows": 200,
    "regime_ae_min_rows": 200,
    "regime_ae_max_train_rows": 30000,
    "regime_ae_min_features": 8,
    "regime_ae_max_features": 180,
    "regime_ae_max_epochs": 50,
    "regime_ae_batch_size": 8192,
    "regime_ae_learning_rate": 1e-3,
    "regime_ae_min_learning_rate": 1e-5,
    "regime_ae_input_noise_std": 0.03,
    "regime_ae_alpha_loss_weight": 0.25,
    "regime_ae_rank_loss_weight": 0.25,
    "regime_ae_latent_l1": 1e-4,
    "regime_ae_latent_stability": 0.005,
    "regime_ae_weight_decay": 1e-4,
    "regime_ae_random_state": 42,
    # Current-regime specialist similarity for LGBM training. Disabled by
    # default because it is heavier and should be shadowed before activation.
    "lgbm_regime_specialist_enabled": False,
    "lgbm_regime_specialist_objectives": ["train_base", "train_meta"],
    "lgbm_regime_specialist_shadow_only": True,
    "lgbm_regime_specialist_apply_sample_weight": False,
    "lgbm_regime_specialist_apply_distillation_shrink": False,
    "lgbm_regime_specialist_distillation_power": 1.0,
    "lgbm_regime_specialist_current_window_days": 28.0,
    "lgbm_regime_specialist_candidate_window_days": 28.0,
    "lgbm_regime_specialist_day_window_days": 1.0,
    "lgbm_regime_specialist_embargo_days": 0.0,
    "lgbm_regime_specialist_label_horizon_hours": 0.0,
    "lgbm_regime_specialist_label_end_col": None,
    "lgbm_regime_specialist_recency_decay_per_week": 0.67,
    "lgbm_regime_specialist_drift_weight": 0.40,
    "lgbm_regime_specialist_covariance_weight": 0.35,
    "lgbm_regime_specialist_regime_weight": 0.15,
    "lgbm_regime_specialist_knn_weight": 0.10,
    "lgbm_regime_specialist_domain_classifier_weight": 0.10,
    "lgbm_regime_specialist_assessment_min_aligned_fraction": 0.80,
    "lgbm_regime_specialist_assessment_allow_timestamp_only_alignment": False,
    "lgbm_regime_specialist_global_assessment_enabled": True,
    "lgbm_regime_specialist_global_assessment_max_rows": 250000,
    "lgbm_regime_specialist_ae_weight": 0.10,
    "lgbm_regime_specialist_alpha": 1.5,
    "lgbm_regime_specialist_analogue_threshold": 0.55,
    "lgbm_regime_specialist_normal_threshold": 0.15,
    "lgbm_regime_specialist_knn_k": 25,
    "lgbm_regime_specialist_max_knn_current_rows": 2000,
    "lgbm_regime_specialist_max_knn_candidate_rows": 5000,
    "lgbm_regime_specialist_max_knn_historical_rows": 50000,
    "lgbm_regime_specialist_knn_fallback_chunk_pairs": 2_000_000,
    "lgbm_regime_specialist_max_fingerprint_rows_per_window": 20000,
    "lgbm_regime_specialist_max_day_fingerprint_rows": 10000,
    "lgbm_regime_specialist_max_covariance_features": 48,
    "lgbm_regime_specialist_max_asset_covariance_assets": 100,
    "lgbm_regime_specialist_max_asset_covariance_time_rows": 50000,
    "lgbm_regime_specialist_min_asset_observation_fraction": 0.60,
    "lgbm_regime_specialist_asset_covariance_shrinkage": 0.10,
    "lgbm_regime_specialist_cov_feature_cov_eig_weight": 0.15,
    "lgbm_regime_specialist_cov_feature_corr_eig_weight": 0.20,
    "lgbm_regime_specialist_cov_feature_concentration_weight": 0.20,
    "lgbm_regime_specialist_cov_asset_cov_eig_weight": 0.10,
    "lgbm_regime_specialist_cov_asset_corr_eig_weight": 0.20,
    "lgbm_regime_specialist_cov_asset_concentration_weight": 0.15,
    "lgbm_regime_specialist_drift_psi_scale": 0.25,
    "lgbm_regime_specialist_drift_psi_weight": 0.18,
    "lgbm_regime_specialist_drift_ks_weight": 0.14,
    "lgbm_regime_specialist_drift_wasserstein_weight": 0.14,
    "lgbm_regime_specialist_drift_mahalanobis_weight": 0.10,
    "lgbm_regime_specialist_drift_prediction_weight": 0.14,
    "lgbm_regime_specialist_drift_covariance_weight": 0.12,
    "lgbm_regime_specialist_drift_contribution_weight": 0.12,
    "lgbm_regime_specialist_drift_other_weight": 0.06,
    "lgbm_regime_specialist_max_window_diagnostics": 50,
    "lgbm_regime_specialist_top_eigenvalues": 5,
    "lgbm_regime_specialist_asset_return_col": None,
    "lgbm_regime_specialist_ae_enabled": False,
    "lgbm_regime_specialist_ae_min_windows": 50,
    "lgbm_regime_specialist_ae_max_windows": 5000,
    "lgbm_regime_specialist_ae_latent_dim": 4,
    "lgbm_regime_specialist_ae_max_iter": 50,
    "lgbm_regime_specialist_ae_input_noise": 0.02,
    "lgbm_regime_specialist_day_similarity_min_rows": 24,
    "lgbm_regime_specialist_day_similarity_strength": 0.50,
    "lgbm_regime_specialist_feature_engineering_enabled": False,
    "lgbm_regime_specialist_feature_engineering_max_final_features": 40,
    "lgbm_regime_specialist_feature_engineering_max_pair_candidates": 2500,
    "lgbm_regime_specialist_feature_engineering_univariate_subsample_per_class": 8000,
    "lgbm_regime_specialist_feature_engineering_lgbm_enabled": True,
    "lgbm_regime_specialist_feature_engineering_elasticnet_enabled": False,
    "lgbm_regime_specialist_feature_engineering_grouped_cv_folds": 5,
    "lgbm_regime_specialist_feature_engineering_grouped_cv_repeats": 3,
    "lgbm_regime_specialist_feature_engineering_permutation_repeats": 2,
    "lgbm_regime_specialist_feature_engineering_max_permutation_features": 80,
    "lgbm_regime_specialist_feature_engineering_max_permutation_rows": 4000,
    "lgbm_regime_specialist_feature_engineering_max_shap_rows": 4000,
    "lgbm_regime_specialist_feature_engineering_drift_window_days": 28.0,
    "lgbm_regime_specialist_feature_engineering_max_drift_raw_features": 80,
    "lgbm_regime_specialist_feature_engineering_drift_window_max_rows": 20000,
    "lgbm_regime_specialist_feature_engineering_drift_knn_max_rows": 4000,
    "lgbm_regime_specialist_feature_engineering_drift_knn_chunk_pairs": 2000000,
    "lgbm_regime_specialist_feature_engineering_domain_score_smoothing_enabled": True,
    "lgbm_regime_specialist_feature_engineering_domain_score_ewma_half_life_days": 1.0,
    "lgbm_regime_specialist_feature_engineering_domain_score_ewma_max_days": 4.0,
    "lgbm_regime_specialist_feature_engineering_diagnostics_enabled": True,
    "lgbm_regime_specialist_feature_engineering_diagnostics_final_only": True,
    "lgbm_regime_specialist_feature_engineering_run_validation_diagnostics": False,
    "lgbm_regime_specialist_min_candidate_rows": 24,
    "lgbm_regime_specialist_min_current_rows": 24,
    "lgbm_regime_specialist_weight_min": 0.05,
    "lgbm_regime_specialist_weight_max": 20.0,
    "lgbm_regime_specialist_recency_power": 0.50,
    "lgbm_regime_specialist_min_current_plus_analogue_mass": 0.50,
    "lgbm_regime_specialist_less_interesting_min_mass": 0.10,
    "lgbm_regime_specialist_less_interesting_max_mass": 0.50,
    "lgbm_regime_specialist_min_adaptive_reliability_to_train": 0.20,
    "lgbm_regime_specialist_weight_hpo_enabled": False,
    "lgbm_regime_specialist_weight_hpo_trials": 150,
    "lgbm_regime_specialist_weight_hpo_early_stop_patience": 30,
    "lgbm_regime_specialist_weight_hpo_random_state": 42,
    "lgbm_regime_specialist_weight_hpo_precision_2w_weight": 1.00,
    "lgbm_regime_specialist_weight_hpo_precision_4w_weight": 0.50,
    "lgbm_regime_specialist_weight_hpo_top30_return_4w_weight": 0.50,
    "lgbm_regime_specialist_weight_hpo_auc_4w_weight": 0.25,
    "lgbm_regime_specialist_weight_hpo_return_scale": 1.0,
    "lgbm_regime_specialist_weight_hpo_max_weight_p99_p50": 20.0,
    "lgbm_regime_specialist_weight_hpo_min_weighted_ess_frac": 0.03,
    "lgbm_regime_specialist_weight_hpo_concentration_penalty_weight": 0.02,
    "lgbm_regime_specialist_weight_hpo_low_ess_penalty_weight": 0.25,
    "lgbm_regime_specialist_weight_hpo_adaptive_floor_penalty_weight": 2.0,
    "lgbm_regime_specialist_weight_hpo_replay_cap_penalty_weight": 2.0,
    "lgbm_regime_specialist_weight_hpo_min_total_n_eff_reliability": 0.50,
    "lgbm_regime_specialist_weight_hpo_min_adaptive_n_eff_reliability": 0.25,
    "lgbm_regime_specialist_weight_hpo_min_current_weight_mass": 0.08,
    "lgbm_regime_specialist_weight_hpo_min_recent_4w_weight_mass": 0.10,
    "lgbm_regime_specialist_weight_hpo_n_eff_reliability_penalty_weight": 1.0,
    "lgbm_regime_specialist_weight_hpo_adaptive_n_eff_penalty_weight": 0.50,
    "lgbm_regime_specialist_weight_hpo_current_focus_penalty_weight": 0.50,
    "lgbm_regime_specialist_weight_hpo_recent_focus_penalty_weight": 0.50,
    "lgbm_regime_specialist_weight_hpo_analogue_gamma_low": 1.5,
    "lgbm_regime_specialist_weight_hpo_analogue_gamma_high": 3.0,
    "lgbm_regime_specialist_weight_hpo_replay_gamma_low": 1.5,
    "lgbm_regime_specialist_weight_hpo_replay_gamma_high": 3.0,
    "lgbm_regime_specialist_weight_hpo_tau_adaptive_low": 10000.0,
    "lgbm_regime_specialist_weight_hpo_tau_adaptive_high": 40000.0,
    "lgbm_regime_specialist_weight_hpo_tau_replay_low": 25000.0,
    "lgbm_regime_specialist_weight_hpo_tau_replay_high": 100000.0,
    "lgbm_regime_specialist_weight_hpo_min_current_plus_analogue_mass_low": 0.50,
    "lgbm_regime_specialist_weight_hpo_min_current_plus_analogue_mass_high": 0.60,
    "lgbm_regime_specialist_weight_hpo_less_interesting_max_mass_low": 0.30,
    "lgbm_regime_specialist_weight_hpo_less_interesting_max_mass_high": 0.50,
    "lgbm_regime_specialist_weight_hpo_current_gamma": 1.0,
    "lgbm_regime_specialist_weight_hpo_less_interesting_min_mass": 0.10,
    # Legacy / deprecated params
    "label_tp_mults": [0.5, 1.0, 1.5, 2.0],  # DEPRECATED: use barrier_k_tp_grid
    "label_sl_mults": [0.3, 0.5, 0.7, 1.0],  # DEPRECATED: use barrier_sl_base_grid
    "label_tp_values_pct": [1.5, 2.0, 3.0, 4.0, 5.0, 6.0],
    "label_sl_values_pct": [0.5, 1.0, 2.0],
    # hourly trading selection (top/bot deviations)
    "trade_extreme_pct": 0.07,
    "trade_extreme_min": 10,
    "trade_extreme_max": 80,
    "trade_deviation_metric": "dist_ema_fast",
    # Quantile label handling: keep union of samples, emphasize tails via weights
    # label_quantile_hi=0.65 gives 35% prevalence (samples >= 65th percentile are positive)
    "label_quantile_lo": 0.30,
    "label_quantile_hi": 0.65,
    "label_quantile_mode": "weighted_union",
    "label_quantile_weight_floor": 0.35,
    "label_quantile_weight_gamma": 1.5,
    # gates
    "gate_vol_lookback_hours": 24 * 14,
    "gate_trend_thr": 0.02,
    "accept_gate_window": 24,
    "accept_gate_percentile_mode": "approx",
    "enable_gated_features": False,
    "feature_backfill_symbol_chunk_size": 100,
    # By default compute all missing feature keys once per symbol chunk. Setting
    # EPM_FEATURE_BACKFILL_KEY_BATCH_SIZE can re-enable smaller save batches for
    # very memory-constrained repair runs.
    "feature_backfill_key_batch_size": 0,
    # base feature windows (used for base/fast/slow variants)
    "atr_n": 14,
    "rsi_n": 14,
    "rsi_slope_n": 6,
    "volz_n": 24 * 7,
    "trend_sma_n": 24 * 14,
    "ema_fast": 20,
    "ema_slow": 80,
    "rvol_days": 14,
    # adaptive window selection buckets (4)
    "rv_ratio_fast_thr": 1.20,
    "rv_ratio_slow_thr": 0.80,
    # MR/TF ElasticNet
    "alpha_mr": 5e-4,
    "l1_ratio_mr": 0.30,
    "alpha_tf": 5e-4,
    "l1_ratio_tf": 0.30,
    # RuleCleaner
    "ruleclean_corr_thr": 0.80,
    # stability gating (per trade, not per day)
    "coef_persist_window": 60,
    "min_feat_nonzero_rate": 0.30,
    "min_model_stability_to_trade": 0.15,
    # causal cols for interaction toggles
    # Added new features for TF/MR/Meta
    "drop_raw_causal": True,
    # Enable/disable 15m OHLCV-derived feature family across train/inference feature lists.
    "enable_15m_ohlcv_features": True,
    "spread_proxy_robust_window": 24 * 30,
    "causal_cols": [
        "dv_z",
        "rng_z",
        "impact_z",
        "liq_score",
        "liq_state",
        "ret24h",
        "rsi",
        "vol_z",
        "trend_pct",
        "rv_2h",
        "rv_4h",
        "rv_24h",
        "a_funding_proxy",
        "flow_ratio",
        "churn",
        "slope",
        "trend_snr",
        "vol_asym",
        "efficiency",
        "fvg",
        "rvol_z",
        "vol_range_shock",
        "climax_decay",
        "cumulative_delta_stall",
        "vol_expansion_ratio",
        "vol_compression",
        "atr_slope",
        "dist_vwap_norm",
        "momentum_accel",
        # New Risk/Exhaustion (Report 2026-02-10)
        "wick_ratio_4h_max",
        "vol_price_div",
        "rsi_lag1",
        "rsi_1h_slope",
        "cvar_5pct",
        "amihud_illiq",
        "clv_mean_24",
        "vol_z_4h",
        "atr_pct_change",
        # FFD d-specific features (d=0.4,0.6)
        "ffd_rv_2h_04",
        "ffd_rv_6h_04",
        "ffd_rv_24h_04",
        "ffd_vol_price_corr_10h_04",
        "ffd_donch_dist_04_12",
        "ffd_donch_dist_04_24",
        "ffd_donch_dist_04_48",
        "ffd_amihud_04",
        "ffd_vol_range_shock_04",
        "ffd_dist_ema_fast_04",
        "ffd_dist_ema_slow_04",
        "ffd_rv_2h_06",
        "ffd_rv_6h_06",
        "ffd_rv_24h_06",
        "ffd_accel_06",
        "ffd_z_06",
        "ffd_vol_price_corr_10h_06",
        "ffd_donch_dist_06_12",
        "ffd_donch_dist_06_24",
        "ffd_donch_dist_06_48",
        "ffd_atr_expansion_06",
        "ffd_cvar_5pct_06",
        "ffd_amihud_06",
        "ffd_vol_range_shock_06",
        # D-family strength indicators
        "ffd_strength_04",
        "ffd_strength_05",
        "ffd_strength_06",
        # New Feature Candidates
        "thrust_decay_4",
        "decel_4",
        "ft_drop",
        "ext_excess",
        "ext_atrExp",
        "comp_to_exp",
        "evr6_x_volz",
        "stall_x_flow",
        "prog_def",
        "clv_collapse",
        "clv_pullback",
        "coh",
        "align",
        "retest_quality",
        "pb_accel",
        "rv_ratio_6_24",
        "excess_coh",
        "asym_ft",
        "dist_stack",
        "tf_bias",
        "shock_rel",
        "resid_strength",
        "evr_slope",
        "stall_ext",
        "spike_score",
        "grind_score",
        "chop_score",
        "G_TF_TREND",
        "vol_z_x_trend_t",
        # Gates as continuous features
        "G_EXH_EFFORT",
        "G_EXH_GIVEBACK",
        "G_EXH_TAIL_FAIL",
        "G_MR_SPIKE",
        # New Model Features
        "overext",
        "overext_weak",
        "effort_gate",
        "tail_fail",
        "blowoff_risk",
        "S",
        "impulse_ratio_24",
        "impulse_ratio_12",
        "coherence_24",
        "accel",
        "tf_tape",
        "mr_tape",
        "retrace_12",
        "exh_qual",
        "mfe_2h",
        "mae_2h",
        "dir_path_long_2h",
        "dir_path_short_2h",
        "dir_path_risk_long_2h",
        "dir_path_risk_short_2h",
        "dir_path_edge_2h",
        "dir_path_risk_skew_2h",
        # OHLCV-based trend quality features (Report 2026-02-12)
        "trend_age_hours",
        "higher_highs_count_48h",
        "lower_lows_count_48h",
        "trend_retest_success_rate",
        "trend_overextension_z",
        "volume_trend_alignment",
        "trend_regime_stability",
        "trend_strength_vs_reversion",
        "support_quality_score",
        "dip_velocity",
        "dip_volume_profile",
        "reversion_target_distance",
        "vov_iqr_20",
        # "vov_mad_20",
        "vov_mad_60",
        # "vov_ratio",
        "vov_interaction",
        "vov_fast_slow_ratio",
        "accel_5h",
        "dlog_vol_5h",
        "signed_max_bar_ret_5h",
        "jump_rate_10h",
        "volu_z",
        "volume_price_corr_10h",
        "draw_sym_10h",
        "breakout_24h",
        "vol_z_30_calm",
        "meta_abs_net_x_breakout",
        "meta_abs_net_x_drawext",
        "meta_abs_net_x_vov_ratio",
        "meta_alignment",
        "meta_signal_x_accel",
        "kf_score_mean",
        "kf_score_rm24_mean",
        "kf_atr_mean",
        "kf_vol_ratio_mean",
        "kf_ret1h_mean",
        "kf_innov_var",
        "kf_snr_est",
        "kf_state_uncertainty",
        "vol_high",
        "vol_low",
        "cusum_strength_norm",
        "cusum_high",
        "liq_low",
        "p_vol_high",
        "p_cusum_high",
        "p_liq_low",
        # Orthogonal features (structurally independent dimensions)
        "mtf_divergence",
        "mtf_div_mag",
        "autocorr_6h",
        "autocorr_24h",
        "path_efficiency_12",
        "path_efficiency_24",
        "hurst_proxy_24",
        "vol_concentration_12",
        "shannon_entropy_ret_8",
        "shannon_entropy_ret_16",
        "perm_entropy_ret_12",
        "perm_entropy_ret_24",
        "spectral_entropy_ret_24",
        "spectral_entropy_ret_48",
        # "volume_entropy_12",
        "volume_entropy_24",
        "downside_semivariance_24",
        "upside_semivariance_8",
        "upside_semivariance_24",
        "down_up_vol_ratio_24",
        "vol_shock_asym_8_24",
        "vol_shock_asym_4_12",
        "vol_shock_asym_4_212",
        # Residualised features — relative surprise, not absolute magnitude
        "RESIDUAL_BASE_FEATURE_KEYS",
        "RESIDUAL_META_FEATURE_KEYS",
        "rsi_z",
        "dist_ema_fast_z",
        "dist_vwap_norm_z",
        "flow_persistence_z",
        "excess_6h_z",
        "vol_z_z",
        "atr_expansion_z",
        "coherence_24_z",
        "overext_surprise",
        "blowoff_risk_surprise",
        "exh_qual_surprise",
        "dist_vwap_resid",
        "dist_ema_fast_resid",
        "trend_pct_resid",
        "mkt_rv_pct",
        "abs_mkt_ret24h_z",
        # New Multi-Horizon Aggregated Features
        "ret_mean",
        "ret_max",
        "ret_min",
        "rv_mean",
        "rv_max",
        "rv_min",
        # New Tail-Risk Features
        "ret_pct5_24h",
        "ret_pct95_24h",
        "gap_zscore",
        "vol_shock_z",
        "range_zscore",
        "tail_risk_score",
        # Location Filter Columns (for mask optimizer)
        "LOC_01_AboveEMA",
        "LOC_02_BelowEMA",
        "LOC_03_BetweenFastMidEMA",
        "LOC_04_BetweenMidSlowEMA",
        "LOC_05_StackedAboveAllEMAs",
        "LOC_06_StackedBelowAllEMAs",
        "LOC_07_TouchFastEMA_Long",
        "LOC_08_TouchFastEMA_Short",
        "LOC_09_TouchMidEMA_Long",
        "LOC_10_TouchMidEMA_Short",
        "LOC_11_DeepPullbackToSlowEMA_Long",
        "LOC_12_DeepPullbackToSlowEMA_Short",
        "LOC_13_EMAValueZone_Long",
        "LOC_14_EMAValueZone_Short",
        "LOC_20_AboveVWAP",
        "LOC_21_BelowVWAP",
        "LOC_22_AtVWAP_Long",
        "LOC_23_AtVWAP_Short",
        "LOC_24_VWAPPlus1Dev",
        "LOC_25_VWAPMinus1Dev",
        "LOC_26_VWAPPlus2Dev",
        "LOC_27_VWAPMinus2Dev",
        "LOC_28_BetweenVWAPAndPlus1Dev",
        "LOC_29_BetweenVWAPAndMinus1Dev",
        "LOC_30_ReclaimVWAPZone_Long",
        "LOC_31_LoseVWAPZone_Short",
        "LOC_40_UpperQuartileOfRange",
        "LOC_41_LowerQuartileOfRange",
        "LOC_42_MidRange",
        "LOC_43_NearRangeHigh",
        "LOC_44_NearRangeLow",
        "LOC_45_AtRangeBreakoutZone_Long",
        "LOC_46_AtRangeBreakdownZone_Short",
        "LOC_50_AbovePriorHigh",
        "LOC_51_BelowPriorLow",
        "LOC_52_InsidePriorRange",
        "LOC_53_NearPriorHigh",
        "LOC_54_NearPriorLow",
        "LOC_55_AboveLastSwingHigh",
        "LOC_56_BelowLastSwingLow",
        "LOC_57_NearLastSwingHigh",
        "LOC_58_NearLastSwingLow",
        "LOC_59_BetweenLastSwingLowHigh",
        "LOC_70_AboveSessionOpen",
        "LOC_71_BelowSessionOpen",
        "LOC_72_AtSessionOpen_Long",
        "LOC_73_AtSessionOpen_Short",
        "LOC_74_AboveInitialBalanceMid",
        "LOC_75_BelowInitialBalanceMid",
        "LOC_76_NearInitialBalanceHigh",
        "LOC_77_NearInitialBalanceLow",
        "LOC_78_AtSessionHighZone",
        "LOC_79_AtSessionLowZone",
        "LOC_80_UpperHalfOfSessionRange",
        "LOC_81_LowerHalfOfSessionRange",
        "LOC_90_AbovePrevDayHigh",
        "LOC_91_BelowPrevDayLow",
        "LOC_92_InsidePrevDayRange",
        "LOC_93_NearPrevDayHigh",
        "LOC_94_NearPrevDayLow",
        "LOC_95_AbovePrevDayMid",
        "LOC_96_BelowPrevDayMid",
        "LOC_97_NearPrevWeekHigh",
        "LOC_98_NearPrevWeekLow",
        "LOC_99_InsidePrevWeekRange",
        "LOC_110_AboveBBMid",
        "LOC_111_BelowBBMid",
        "LOC_112_AtBBUpper",
        "LOC_113_AtBBLower",
        "LOC_114_OutsideBBUpper",
        "LOC_115_OutsideBBLower",
        "LOC_116_AtKCUpper",
        "LOC_117_AtKCLower",
        "LOC_118_BetweenBBMidAndUpper",
        "LOC_119_BetweenBBMidAndLower",
        "LOC_130_ShallowPullback_Long",
        "LOC_131_DeepPullback_Long",
        "LOC_132_ShallowPullback_Short",
        "LOC_133_DeepPullback_Short",
        "LOC_134_Fib382Zone_Long",
        "LOC_135_Fib50Zone_Long",
        "LOC_136_Fib618Zone_Long",
        "LOC_137_Fib382Zone_Short",
        "LOC_138_Fib50Zone_Short",
        "LOC_139_Fib618Zone_Short",
        "LOC_150_AtPivotResistance",
        "LOC_151_AtPivotSupport",
        "LOC_152_BetweenPivotAndR1",
        "LOC_153_BetweenPivotAndS1",
        "LOC_154_AtLiquidityPoolHigh",
        "LOC_155_AtLiquidityPoolLow",
        "LOC_156_AtUntestedBreakoutLevel",
        "LOC_157_AtUntestedBreakdownLevel",
        "LOC_170_NotTooExtendedAboveEMA",
        # Intraday Trigger Columns (for mask optimizer)
        "LONG_01_WideBullBody",
        "LONG_02_3CloseMomentum",
        "LONG_03_RollingHighBreakout",
        "LONG_04_EMATagCloseAbove",
        "SHORT_04_EMATagCloseBelow",
        "LONG_05_SmallBullContinuation",
        "SHORT_05_SmallBearContinuation",
        "LONG_10_2BarMomentum",
        "SHORT_10_2BarMomentum",
        "LONG_11_3BarPriceAcceleration",
        "SHORT_11_3BarPriceAcceleration",
        "LONG_12_HH_HL_Impulse",
        "SHORT_12_LL_LH_Impulse",
        "LONG_13_BullCloseNearHigh",
        "SHORT_13_BearCloseNearLow",
        "LONG_14_MomentumWithRelVol",
        "SHORT_14_MomentumWithRelVol",
        "LONG_15_MomoIgnition",
        "SHORT_15_MomoIgnition",
        "LONG_20_HighBreakClose",
        "SHORT_20_LowBreakClose",
        "LONG_21_DonchianBreak",
        "SHORT_21_DonchianBreak",
        "LONG_22_OpeningRangeBreak",
        "SHORT_22_OpeningRangeBreak",
        "LONG_23_InsideBarBreak",
        "SHORT_23_InsideBarBreak",
        "LONG_24_OutsideBarResolution",
        "SHORT_24_OutsideBarResolution",
        "LONG_25_NRBreakout",
        "SHORT_25_NRBreakout",
        "LONG_26_SqueezeRelease",
        "SHORT_26_SqueezeRelease",
        "LONG_27_PivotBreak",
        "SHORT_27_PivotBreak",
        "LONG_28_LevelBreakRetestHold",
        "SHORT_28_LevelBreakRetestHold",
        "LONG_30_EMA10_PullbackBounce",
        "SHORT_30_EMA10_PullbackReject",
        "LONG_31_EMA20_PullbackBounce",
        "SHORT_31_EMA20_PullbackReject",
        "LONG_32_EMAStackPullback",
        "SHORT_32_EMAStackPullback",
        "LONG_33_VWAPPullbackHold",
        "SHORT_33_VWAPPullbackReject",
        "LONG_34_BreakoutThenInsideContinuation",
        "SHORT_34_BreakdownThenInsideContinuation",
        "LONG_35_MicroPullbackHigherLow",
        "SHORT_35_MicroPullbackLowerHigh",
        "LONG_36_FlagBreak",
        "SHORT_36_FlagBreak",
        "LONG_37_HighTightFlag",
        "SHORT_37_LowTightFlag",
        "LONG_40_HammerReversal",
        "SHORT_40_ShootingStarReversal",
        "LONG_41_BullEngulf",
        "SHORT_41_BearEngulf",
        "LONG_42_FailedBreakdown",
        "SHORT_42_FailedBreakout",
        "LONG_43_Spring",
        "SHORT_43_Upthrust",
        "LONG_44_OutsideReversalUp",
        "SHORT_44_OutsideReversalDown",
        "LONG_45_3BarReversal",
        "SHORT_45_3BarReversal",
        "LONG_46_StopRunReclaim",
        "SHORT_46_StopRunReject",
        "LONG_50_BBLowerSnapback",
        "SHORT_50_BBUpperSnapback",
        "LONG_51_KCExtensionRevert",
        "SHORT_51_KCExtensionRevert",
        "LONG_52_VWAPStretchRevert",
        "SHORT_52_VWAPStretchRevert",
        "LONG_53_RSIRecovery",
        "SHORT_53_RSIReject",
        "LONG_54_StochCrossFromOS",
        "SHORT_54_StochCrossFromOB",
        "LONG_60_CloseCrossEMA",
        "SHORT_60_CloseCrossEMA",
        "LONG_61_FastCrossMid",
        "SHORT_61_FastCrossMid",
        "LONG_62_PriceReclaimsEMAStack",
        "SHORT_62_PriceLosesEMAStack",
        "LONG_63_EMACompressionExpansion",
        "SHORT_63_EMACompressionExpansion",
        "LONG_70_VWAPCrossHold",
        "SHORT_70_VWAPCrossReject",
        "LONG_71_VWAPReclaimAfterUndercut",
        "SHORT_71_VWAPRejectAfterOvershoot",
        "LONG_72_VWAPTrendContinuation",
        "SHORT_72_VWAPTrendContinuation",
        "LONG_80_RangeLowReversal",
        "SHORT_80_RangeHighReversal",
        "LONG_81_RangeEscape",
        "SHORT_81_RangeEscape",
        "LONG_82_IBHBreak",
        "SHORT_82_IBLBreak",
        "LONG_83_PreviousHighBreak",
        "SHORT_83_PreviousLowBreak",
        "LONG_84_PreviousLowSweepReclaim",
        "SHORT_84_PreviousHighSweepReject",
        "LONG_90_RangeExpansion",
        "SHORT_90_RangeExpansion",
        "LONG_91_TRExpansionBreak",
        "SHORT_91_TRExpansionBreak",
        "LONG_92_CompressionThenExpansion",
        "SHORT_92_CompressionThenExpansion",
        "LONG_93_NR7Expansion",
        "SHORT_93_NR7Expansion",
        "LONG_100_BOS_Up",
        "SHORT_100_BOS_Down",
        "LONG_101_CHOCH_Up",
        "SHORT_101_CHOCH_Down",
        "LONG_102_HigherLowContinuation",
        "SHORT_102_LowerHighContinuation",
        "LONG_103_FlipZoneLong",
        "SHORT_103_FlipZoneShort",
        "LONG_110_LongLowerWickAbsorption",
        "SHORT_110_LongUpperWickAbsorption",
        "LONG_111_BearTrapCandle",
        "SHORT_111_BullTrapCandle",
        "LONG_112_DojiResolveUp",
        "SHORT_112_DojiResolveDown",
        "LONG_113_PinBarBreakUp",
        "SHORT_113_PinBarBreakDown",
        "LONG_120_RSITrendPush",
        "SHORT_120_RSITrendPush",
        "LONG_121_ADX_DI_Long",
        "SHORT_121_ADX_DI_Short",
        "LONG_122_RSIMidlineReclaim",
        "SHORT_122_RSIMidlineLose",
        "LONG_130_DislocationUp",
        "SHORT_130_DislocationDown",
        "LONG_131_DislocationFillHold",
        "SHORT_131_DislocationFillReject",
        "LONG_140_ThreeWhiteSoldiersLite",
        "SHORT_140_ThreeBlackCrowsLite",
        "LONG_141_1_2_3_ReversalUp",
        "SHORT_141_1_2_3_ReversalDown",
        "LONG_142_PauseThenGo",
        "SHORT_142_PauseThenGo",
        "LONG_150_BreakoutQuality",
        "SHORT_150_BreakdownQuality",
        "LONG_151_PullbackQuality",
        "SHORT_151_PullbackQuality",
        "LONG_152_ReversalQuality",
        "SHORT_152_ReversalQuality",
        "LONG_153_SqueezeTrendRelease",
        "SHORT_153_SqueezeTrendRelease",
    ],
    # thresholds / picks
    "thr_long": 0.010,
    "thr_short": -0.010,
    "k_long": 10,
    "k_short": 10,
    "score_gate_q": 0.93,  # Global percentile gate: only trade signals in top x% (0.93 = top 7%) of global distribution
    # sizing / risk / costs
    "wallet_gross_cap": 0.25,
    "sizing_mode": "rank",  # "rank" (default), "equal", or "score" — rank uses percentile within batch
    "score_map": "tanh",
    "score_scale": 15.0,
    "tp": 0.05,
    "sl": 0.025,
    "hold_hours": 8,
    "fee_bps": 25.0,
    "borrow_apr": 0.20,
    "oos_holdout_days": 730,  # Enforce >= 2 years OOS holdout for robust signal evaluation
    # Trailing Profit Risk Params (used in backtest & live, all vol-scaled)
    # Target absolute: TP ~2%, SL ~0.7% (with median barrier_pct ~4%)
    "tp_mult": 0.50,  # Activation threshold = tp_mult * barrier_pct (~2%)
    "sl_mult": 0.18,  # Stop-loss = sl_mult * barrier_pct (~0.7%)
    "trail_mult": 0.25,  # Trailing deviation = trail_mult * barrier_pct
    # Hard constraints enforced in optimizer and defaults
    "min_tp_sl_ratio": 1.2,  # TP:SL ratio must be >= 1.2
    "min_tp_abs_pct": 0.02,  # TP must be >= 2% absolute
    # Regime throttle: reduce sizing during drawdowns
    "throttle_lookback_trades": 20,  # look at last N closed trades
    "throttle_dd_threshold": -0.02,  # cumPnL drawdown trigger
    "throttle_sizing_factor": 0.5,  # reduce sizing to 50% when triggered
    # Portfolio constraints
    "max_concurrent_trades": 5,
    "max_portfolio_weight": 0.25,
    # Daily risk budget: concentration controls
    "max_daily_per_specialist": 8,  # max trades/day per bucket (LONG_TF, SHORT_MR, etc.)
    "max_daily_total": 25,  # max total trades/day across all buckets
    # Legacy Risk Params (Trailing Stop fallback)
    "risk_k_sl": 2.0,  # stop distance in ATR multiples
    "risk_k_trail_start": 1.0,  # profit distance to start trailing
    "risk_k_trail_dist": 1.0,  # trailing distance
    # Spike / Regime Head
    "spike_feature_keys": [
        "S",
        "impulse_ratio_12",
        "impulse_ratio_24",
        "coherence_24",
        "accel",
        "mkt_rv_ratio",
        "wick_ratio",
        "body_ratio",
        "rvol_z",
        "retrace_12",
        "donch_dist_12",
    ],
    # TF Head (Specifics + Global) — includes trend maturity features
    # Kind-specific overlays for meta models (kept for backward compatibility)
    "base_shared_feature_keys": [
        "impulse_ratio_24",
        "clv_mean_4",
        "pullback_2",
        "pullback_4",
        "pullback_8",
        "ft_2",
        "ft_4",
        "ft_8",
        "flow_persistence",
        "flow_ratio",
        "progress",
        "evr_6",
        "delta_stall_6",
        "accel_5h",
        "breakout_24h",
        "vol_shock_asym_8_24",
        "vol_shock_asym_4_12",
        "vol_shock_asym_4_212",
        # Multi-horizon returns / momentum
        "ret3h",
        "ret5h",
        "ret10h",
        "ret12h",
        "ret20h",
        "ret28h",
        "impulse",
        "jump_intensity",
        "lr_1h",
        "lr_2h",
        "lr_4h",
        "lr_6h",
        "lr_12h",
        "lr_24h",
        "neutral_feature_keys",
        "MODEL_FEATURES",
        "HELPER_BASE_FEATURES",
        "CONTINUOUS_LOCATION_COLS",
        "breakout_confirmed",
        "breakout_min",
        "breakout_soft",
        "breakout_t",
        "climax",
        "climax_decay",
        "convexity_bis_t",
        "convexity_t",
        "convexity_z_t",
        "dip_velocity",
        "dip_volume_profile",
        "dir_path_edge_2h",
        "dir_path_long_2h",
        "dir_path_risk_long_2h",
        "dir_path_risk_short_2h",
        "dir_path_risk_skew_2h",
        "dir_path_short_2h",
        "dist_from_high_120h",
        "dist_from_high_48h",
        "dist_from_high_event_12h",
        "dist_from_high_vol",
        "dist_from_low_event_12h",
        "dist_from_low_vol",
        "donch_dist_120",
        "donch_dist_48",
        "donch_dist_72",
        "draw_extreme_10h",
        "impulse_reversal",
        "impulse_reversal_short",
        "is_trending",
        "meta_abs_net_x_breakout",
        "meta_signal_x_accel",
        "mr_climax",
        "mr_failure",
        "mr_pct",
        "mr_potential",
        "mr_soft",
        "mr_tape",
        "pct_breakout_t",
        "pct_extreme",
        "pullback_120",
        "pullback_48",
        "pullback_72",
        "ret120h",
        "ret1h",
        "ret2h",
        "ret48h",
        "ret4h",
        "ret6h",
        "reversion_target_distance",
        "second_leg_accel_1h",
        "second_leg_accel_2h",
        "second_leg_accel_vol_1h",
        "second_leg_accel_vol_2h",
        "shock_12h",
        "shock_decay",
        "shock_rel",
        "shock_vol_ratio",
        "stall",
        "tail_against",
        "tail_asymmetry_q90_q10_atr_norm",
        "tail_score",
        "tf_tape",
        "time_since_event_extreme_12h",
        "trap_quality",
        "trend_accel_120h",
        "trend_age_hours",
        "trend_retest_success_rate",
        "trend_overextension_z",
        "trend_slope_120h",
        "trend_slope_48h",
        "trend_strength_vs_reversion",
        "trend_t",
        "trend_z_t",
        "volume_trend_alignment",
        "vw_breakout",
    ],
    "base_long_feature_keys": [
        "breakout_confirmed",
        "tf_qual",
        "tf_bias",
        "tf_tape",
        "dir_path_long_2h",
        "dir_path_edge_2h",
        "donch_dist_48",
        "pullback_48",
        "dist_from_high_48h",
        "trend_age_hours",
        "trend_retest_success_rate",
        "trend_overextension_z",
        "ker_16",
        "adx_14",
        "adx_14_slope",
        "vp_air_pocket_score",
        "trapped_longs_96",
        "vp_dist_hvn_above_atr",
    ],
    "base_short_feature_keys": [
        "overext",
        "overext_weak",
        "mr_qual",
        "retrace_12",
        "mr_tape",
        "giveback",
        "blowoff_risk",
        "tail_fail",
        "tail_against",
        "dir_path_risk_long_2h",
        "dir_path_risk_short_2h",
        "donch_dist_12",
        "donch_dist_8",
        "excess_6h",
        "dist_from_low_48h",
        "dist_from_low_120h",
        "lower_lows_count_48h",
        "trend_strength_vs_reversion",
        "support_quality_score",
        "dip_velocity",
        "dip_volume_profile",
        "reversion_target_distance",
        "bounce_signal",
        "volume_capitulation",
        "trap_strength",
        "entry_quality_composite",
        "trap_quality",
        "mr_soft",
        "mr_potential",
        "climax",
        "mr_climax",
        "shock_decay",
        "pct_extreme",
        "mr_pct",
        "stall",
        "mr_failure",
        "impulse_reversal",
        "impulse_reversal_short",
        "breakout_min",
        "pct_breakout_t",
        "trapped_longs_12",
        "dist_vwap_12_atr",
        "vp_dist_poc_atr",
        "vp_in_poc_zone",
        "adx_7",
    ],
    "meta_shared_feature_keys": [
        # Kalman Meta Features
        "rolling_std(price_innovation)",
        "kalman_gain_1h",
        "state_uncertainty_1h",
        "vol_state_slope_1h",
        "realized_vol_minus_vol_state",
        "log_volume_state_1h",
        "volume_state_slope_1h",
        "price_slope_x_volume_surprise",
        "vol_state_x_volume_state",
        "ambig",
        "spike_score",
        "grind_score",
        "chop_score",
        "vol_z_30_calm",
        "kf_score_mean",
        "kf_score_rm24_mean",
        "kf_atr_mean",
        "kf_vol_ratio_mean",
        "kf_ret1h_mean",
        "kf_innov_var",
        "kf_snr_est",
        "kf_state_uncertainty",
        "vol_high",
        "vol_low",
        "cusum_strength_norm",
        "cusum_high",
        "liq_low",
        "p_vol_high",
        "p_cusum_high",
        "p_liq_low",
        "asset_atr_level",
        "asset_atr_level_pct",
        "asset_vol_level",
        "asset_vol_level_pct",
        "vol_state",
        "liq_score",
        "rng_z",
        "impact_z",
        "coherence_24",
        "time_since_peak_12h",
        "time_since_trough_12h",
        "vol_scale",
        "be_vol_units",
        "pl_vol_units",
        "trail_act_pct",
        "trail_act_vol_units",
        "giveback_vol_units",
        "t_be_proxy",
        "t_pl_proxy",
        "t_trail_proxy",
        "amihud_z",
        "amihud_illiq",
        "liq_regime",
        "rvol_hod_base",
        "vol_regime_z",
        "regime_stability_24h",
        "regime_transition_entropy_12h",
        "regime_transition_entropy_48h",
        "complexity_regime_24h",
        "vol_regime_switch_12h",
        "vol_concentration_12",
        "volume_entropy_12",
        "volume_entropy_24",
        "volatility_zscore",
        "clv_t",
        "body_ratio_15m",
        "rejection_proxy",
        "range_norm_12",
        "sv_imb_12",
        "press_12",
        "impact_12",
        "ts_12",
        "prog_eff_12",
        "pers_12",
        "hh_count_12",
        "ll_count_12",
        "skew_12",
        "z_vwap_12",
        "z_r_12",
        "bb_pos_12",
        "range_norm_24",
        "sv_imb_24",
        "press_24",
        "impact_24",
        "ts_24",
        "prog_eff_24",
        "pers_24",
        "hh_count_24",
        "ll_count_24",
        "skew_24",
        "z_vwap_24",
        "z_r_24",
        "bb_pos_24",
        "regime_stability_24h",
        "complexity_regime_24h",
        "bars_since_ema20_ema50_cross_log_norm",
        "bars_in_high_vol_state_log_norm",
        "bars_outside_ema20_atr_band_log_norm",
        "up_down_semivol_ratio_tanh",
        "up_down_return_mass_ratio_tanh",
        # Regime / state features
        "regime_vol_score",
        "regime_liquidity_score",
        "vol_regime_ratio",
        "vol_regime_shift",
        "vol_regime_transition",
        "stage_mr",
        "stage_tf",
        "stage_blowoff",
        # Position sizer V2 features
        "beta_24h",
        "hour_vol_ratio",
        "liquidity_ratio",
        "seasonality_strength",
        "session_progress",
        # Residual / surprise variants
        "RESIDUAL_META_FEATURE_KEYS",
        "volume_z_12",
        "volume_z_24",
        "vol_z24",
        "ffd_mr_z_06",
        "xs_rank_vol_z",
        "CONTINUOUS_LOCATION_COLS",
        "REGIME_FEATURE_KEYS",
        "FEATURE_SELECTION_KEYS",
        "TRAINING_RESIDUALIZATION_FEATURE_KEYS",
        # Structural Z-Normalization Features (Mask Optimiser Pre-calc)
        "z_hl_range",
        "z_intrabar_range_atr",
        "z_compression_expansion",
        "z_volume",
        "z_dist_ema_24",
        "z_dist_vwap_24",
        "z_atr_norm_ret_24",
        "z_path_efficiency_24",
        "G_LIQ_EXCEL",
        "G_LIQ_GOOD",
        "G_LIQ_GREAT",
        "G_MR_SPIKE",
        "G_VOL_LIQ_GT1",
        "G_VOL_LIQ_GT2",
        "G_VOL_LIQ_GT3",
        "choppiness_index_20",
        "cvar_5pct",
        "direction_entropy_20",
        "dlog_vol_5h",
        "downside_semivariance_24",
        "dv_z",
        "efficiency",
        "efficiency_ratio_20",
        "entropy_jump_24h",
        "evr6_x_volz",
        "ffd_amihud_04",
        "ffd_amihud_06",
        "ffd_cvar_5pct_06",
        "ffd_vol_range_shock_04",
        "ffd_vol_range_shock_06",
        "hurst_proxy_x_regime_trend_48h",
        "is_high_vol_regime",
        "is_low_vol_regime",
        "liq_state",
        "meta_abs_net_x_vov_ratio",
        "mkt_rv_ratio",
        "mtf_divergence_x_regime_vol_12h",
        "perm_entropy_ret_12",
        "perm_entropy_ret_24",
        "prior_volatility",
        "realized_volatility_24h",
        "ret1h_z",
        "rsi_x_high_vol",
        "rv_120h",
        "rv_12h",
        "rv_24h",
        "rv_2h",
        "rv_48h",
        "rv_4h",
        "rv_6h",
        "rv_8h",
        "rv_ratio_6_24",
        "rvol_z",
        "shannon_entropy_ret_16",
        "shannon_entropy_ret_8",
        "spectral_entropy_ret_24",
        "spectral_entropy_ret_48",
        "trend_regime_switch_12h",
        "trend_snr",
        "upside_semivariance_24",
        "upside_semivariance_8",
        "variance_ratio_10_48",
        "vol_asym",
        "vol_asym_6",
        "vol_range_shock",
        "vol_z",
        "vol_z24_base",
        "vol_z_4h",
        "vol_z_base",
        "vol_z_x_low_vol",
        "vol_z_x_regime_trend",
        "vol_z_x_trend_t",
        "volatility_autocorr_48",
        "volatility_of_volatility_48",
        "volatility_ratio_short_long",
        "volu_z",
        "volume_autocorr_48",
        "volume_percentile",
        "volume_price_corr_10h",
        "volume_trend_48",
        "volume_zscore_48h",
        "vov_fast_slow_ratio",
        "vov_interaction",
        "vov_iqr_20",
        "vov_mad_20",
        "vov_mad_60",
        "vov_ratio",
        "vp_bin_vol_share",
        "vp_profile_entropy",
        "RIDGE_FEATURE_COLS",
        "regime_trend_score",
        "trend_regime",
        "trend_regime_stability",
    ],
    "meta_reg_feature_keys": [
        "rv_2h",
        "rv_4h",
        "rv_6h",
        "rv_8h",
        "rv_24h",
        "meta_abs_net_x_drawext",
        "meta_abs_net_x_vov_ratio",
        "meta_alignment",
        "predicted_vol_6h",
        "resid_ret_6h",
    ],
    "meta_clf_feature_keys": [
        "rv_2h",
        "rv_4h",
        "rv_24h",
        "meta_alignment",
        "predicted_vol_6h",
    ],
    "meta_mfe_feature_keys": [
        "rv_ratio_24_120",
        "rv_48h",
        "rv_120h",
        "higher_highs_count_48h",
        "lower_lows_count_48h",
        "vol_expansion_ratio",
        "adx_zscore",
    ],
    "meta_mae_feature_keys": [
        "vol_exhaust",
        "vol_compression",
        "down_up_vol_ratio_24",
        "draw_sym_10h",
        "atr_pct_change",
        "move_magnitude_z",
    ],
    "meta_asym_feature_keys": [
        "vol_asym",
        "vol_asym_6",
        "asym_ratio",
        "asym_ft",
    ],
    # Selector v3 configs (top30-focused, per-head)
    "selector_feature_family_map": {},
    "base_selector_cfg": {
        "selector_focus_top_frac": 1.0,
        "selector_top_metric": "ic",
        "selector_frequency_hit_mode": "relative",
        "selector_frequency_hit_quantile": 0.80,
        "selector_frequency_hit_abs": 1e-6,
        "selector_interaction_mode": "tree_path_lift",
        "selector_interaction_topk_pairs": 48,
        "selector_interaction_max_pairs_per_feature": 4,
        "selector_interaction_corr_penalty": True,
        "selector_family_penalty": True,
        "selector_emit_report": True,
        "selector_hysteresis_margin": 0.05,
        "selector_min_overlap": 0.70,
        "analysis_n_estimators": 192,
        "analysis_max_samples": 3000,
        "min_samples_leaf_pct": 0.015,
        "selector_max_missing_frac": 0.15,
        "selector_near_constant_dominance": 0.999,
        "top30": 0.0,
        "global": 0.55,
        "stability": 0.25,
        "frequency": 0.15,
        "interaction": 0.05,
    },
    "meta_selector_cfg": {
        "selector_focus_top_frac": 1.0,
        "selector_top_metric": "ic",
        "selector_frequency_hit_mode": "relative",
        "selector_frequency_hit_quantile": 0.80,
        "selector_frequency_hit_abs": 1e-6,
        "selector_interaction_mode": "tree_path_lift",
        "selector_interaction_topk_pairs": 48,
        "selector_interaction_max_pairs_per_feature": 4,
        "selector_interaction_corr_penalty": True,
        "selector_family_penalty": True,
        "selector_emit_report": True,
        "selector_hysteresis_margin": 0.05,
        "selector_min_overlap": 0.70,
        "analysis_n_estimators": 192,
        "analysis_max_samples": 3000,
        "min_samples_leaf_pct": 0.015,
        "selector_max_missing_frac": 0.15,
        "selector_near_constant_dominance": 0.999,
        "top30": 0.0,
        "global": 0.55,
        "stability": 0.25,
        "frequency": 0.15,
        "interaction": 0.05,
    },
    "aux_mae_selector_cfg": {
        "selector_focus_top_frac": 1.0,
        "selector_top_metric": "ic",
        "selector_frequency_hit_mode": "relative",
        "selector_frequency_hit_quantile": 0.85,
        "selector_frequency_hit_abs": 1e-6,
        "selector_interaction_mode": "tree_path_lift",
        "selector_interaction_topk_pairs": 32,
        "selector_interaction_max_pairs_per_feature": 3,
        "selector_interaction_corr_penalty": True,
        "selector_family_penalty": True,
        "selector_emit_report": True,
        "selector_hysteresis_margin": 0.04,
        "selector_min_overlap": 0.75,
        "analysis_n_estimators": 160,
        "analysis_max_samples": 2500,
        "min_samples_leaf_pct": 0.02,
        "selector_max_missing_frac": 0.15,
        "selector_near_constant_dominance": 0.999,
        "top30": 0.0,
        "global": 0.55,
        "stability": 0.25,
        "frequency": 0.15,
        "interaction": 0.05,
    },
    "aux_mfe_selector_cfg": {
        "selector_focus_top_frac": 1.0,
        "selector_top_metric": "ic",
        "selector_frequency_hit_mode": "relative",
        "selector_frequency_hit_quantile": 0.80,
        "selector_frequency_hit_abs": 1e-6,
        "selector_interaction_mode": "tree_path_lift",
        "selector_interaction_topk_pairs": 40,
        "selector_interaction_max_pairs_per_feature": 4,
        "selector_interaction_corr_penalty": True,
        "selector_family_penalty": True,
        "selector_emit_report": True,
        "selector_hysteresis_margin": 0.05,
        "selector_min_overlap": 0.70,
        "analysis_n_estimators": 160,
        "analysis_max_samples": 2500,
        "min_samples_leaf_pct": 0.02,
        "selector_max_missing_frac": 0.15,
        "selector_near_constant_dominance": 0.999,
        "top30": 0.0,
        "global": 0.55,
        "stability": 0.25,
        "frequency": 0.15,
        "interaction": 0.05,
    },
    "aux_utility_selector_cfg": {
        "selector_focus_top_frac": 1.0,
        "selector_top_metric": "ic",
        "selector_emit_report": True,
        "top30": 0.0,
        "global": 0.45,
        "stability": 0.30,
        "frequency": 0.20,
        "interaction": 0.05,
    },
    # Backward-compatible selector aliases
    "base_mdi_selector_target": "classification",
    "base_mdi_selector_loss": "binary_logloss",
    "mda_config": {
        "enabled": True,
        "objective": "topk_opportunity_precision",
        "topk_fracs": [0.10, 0.20, 0.30],
        "topk_frac_weights": [0.50, 0.30, 0.20],
        "positive_label": 1,
        "use_sample_weight": True,
        "permutation_mode": "path_gated_lgbm",
        "permutation_style": "row_shuffle",
        "block_size": None,
        "min_repeats": 3,
        "max_repeats": 20,
        "repeat_batch_size": 2,
        "confidence_level": 0.95,
        "early_stop_strong_keep": True,
        "early_stop_null_drop": True,
        "min_effect_size": 0.0,
        "decision_default_for_borderline": "keep",
        "shadow_null_enabled": True,
        "shadow_max_features": 50,
        "shadow_sample_strategy": "variance_quantiles",
        "shadow_null_quantile": 0.95,
        "shadow_n_repeats": 5,
        "group_mda_enabled": True,
        "correlation_method": "spearman",
        "correlation_threshold": 0.85,
        "group_permutation_style": "joint_row_shuffle",
        "write_mda_report": True,
        "report_format": ["json", "csv"],
    },
    # Layer A Ablations and Config
    "model1_target_mode": "race",  # "race" | "fixed"
    "fixed_model1_target_name": "robust_utility_target",
    "score_blend_mode": "train_scaled_components",  # "legacy_raw" | "train_scaled_components"
    "use_model3_uncertainty": True,
    # Unified learnability-test feature basket used by research comparison scripts
    "test_feature_keys": TEST_FEATURE_KEYS,
    # Inference dynamic-basket controls
    "inference_event_window_hours": 12,
    "inference_event_threshold": 0.07,
    "inference_perf_pct": 0.10,
    "inference_draw_window_hours": 8,
    "inference_basket_ttl_hours": 24,
    # High-frequency simulation
    "use_15m_precision": True,  # Enable 15m OHLCV for trailing profit (requires CCXT exchange)
    "allow_15m_download": True,  # Allow Binance-backed backfill of missing 15m ranges during label/TBM refinement
    "allow_5m_download": True,  # Use 5m only for residual ambiguity solving after 15m refinement
    # Limit Order Simulation (Report 2026-02-22)
    "use_limit_orders": True,  # Enable limit orders per user request
    # ---------------------------------------------------------
    # LIMIT OFFSET SEMANTIC CONTRACT
    # ---------------------------------------------------------
    # Units: Basis points (bps). 10000 bps = 1.0 = 100%.
    # Sign convention: Positive means price improvement vs signal.
    # Bounds: Applied globally to valid limit offset targets and predictions.
    # Economics: Larger offset -> more price improvement, lower fill prob.
    "limit_offset_unit": "bps",
    "limit_offset_min": 5.0,
    "limit_offset_max": 50.0,
    # ML Offset Path
    "limit_offset_mode": "heuristic",  # "heuristic" | "ml" | "disabled"
    "limit_offset_target_mode": "undefined",  # requires definition before ML mode
    "allow_heuristic_fallback_if_ml_unavailable": True,
    "limit_offset_bps": 20.0,  # Default static entry offset (if fallback)
    "exit_limit_offset_bps": 20.0,  # Default static exit offset
    "signal_opt_debug": True,  # Emit detailed signal-optimization diagnostics
    "debug_signal_generation": True,  # Emit per-timestamp signal-generation stage counts
    "fee_bps": 35.0,  # Default fee (used when not using limit orders)
    # Fee Structure (Market vs Limit)
    "fee_bps_market": 25.0,  # 0.25% per side for market orders (50 bps RT)
    "fee_bps_limit_entry": 10.0,  # 0.10% per side for limit order entry (20 bps RT)
    "fee_bps_limit_exit": 10.0,  # 0.10% per side for limit order exit (20 bps RT)
    "fee_bps_market_exit": 25.0,  # 0.25% per side if using market order for exit
    # Limit Order Price Estimation (MAE/MFE-based heuristics)
    "use_mae_mfe_limit_offset": True,  # Use MAE/MFE predictions for limit offset
    "limit_offset_min_bps": 5.0,  # Legacy alias to limit_offset_min
    "limit_offset_max_bps": 50.0,  # Legacy alias to limit_offset_max
    "limit_fill_model_type": "heuristic",  # heuristic | learned
    "limit_fill_vol_regime_weight": 0.3,  # How much vol regime reduces fill prob
    "limit_fill_liquidity_bonus": 0.2,  # Liquidity adjustment to fill prob
    # Exit Limit Orders
    "use_exit_limit_orders": True,  # Enable limit orders for exits
    "exit_limit_offset_adaptive": True,  # Adapt exit offset based on profit locked
    # Risk logging
    "verbose_risk_logging": False,  # Enable detailed per-trade TP/SL logging
}


def _append_missing(existing, extra):
    out = list(existing or [])
    seen = set(out)
    for item in list(extra or []):
        if item in seen:
            continue
        out.append(item)
        seen.add(item)
    return out


TIME_CYCLICAL_FEATURE_KEYS = [
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
]

CFG["time_cyclical_feature_keys"] = TIME_CYCLICAL_FEATURE_KEYS
CFG["lgbm_time_feature_selector_bypass_enabled"] = True
CFG["lgbm_time_feature_selector_bypass_modes"] = ["train_base", "train_meta"]
CFG["lgbm_time_feature_selector_bypass_features"] = TIME_CYCLICAL_FEATURE_KEYS
for _time_feature_target in (
    "causal_cols",
    "base_long_feature_keys",
    "base_short_feature_keys",
    "meta_shared_feature_keys",
):
    CFG[_time_feature_target] = _append_missing(
        CFG.get(_time_feature_target, []),
        TIME_CYCLICAL_FEATURE_KEYS,
    )


def enable_perp_feature_keys(cfg: dict) -> dict:
    """
    Enable perp-specific features for runtime config.
    Spot pipeline remains unchanged unless this helper is called.
    """
    out = dict(cfg)

    def _is_perp_meta_primary_feature(name: str) -> bool:
        return (
            name in set(PERP_META_PRIMARY_FEATURE_KEYS)
            or name.startswith("funding_mom_")
            or name in set(SPOT_FOR_PERPS_META_FEATURE_KEYS)
        )

    base_perp_feature_keys = [
        k for k in PERP_FEATURE_KEYS if not _is_perp_meta_primary_feature(k)
    ]
    for k in (
        "base_long_feature_keys",
        "base_short_feature_keys",
    ):
        out[k] = _append_missing(
            out.get(k, []),
            base_perp_feature_keys + SPOT_FOR_PERPS_BASE_FEATURE_KEYS,
        )
    out["meta_shared_feature_keys"] = _append_missing(
        out.get("meta_shared_feature_keys", []),
        PERP_FEATURE_KEYS + SPOT_FOR_PERPS_META_FEATURE_KEYS,
    )
    out["spot_for_perps_base_feature_keys"] = SPOT_FOR_PERPS_BASE_FEATURE_KEYS
    out["spot_for_perps_meta_feature_keys"] = SPOT_FOR_PERPS_META_FEATURE_KEYS
    return out


def apply_15m_feature_toggle(cfg: dict) -> dict:
    """Apply 15m OHLCV feature family toggle to runtime feature key lists."""
    out = dict(cfg)
    target_lists = (
        "causal_cols",
        "base_long_feature_keys",
        "base_short_feature_keys",
        "meta_shared_feature_keys",
        "position_sizer_features",
        "limit_offset_sizer",
    )

    enabled = bool(out.get("enable_15m_ohlcv_features", True))
    for k in target_lists:
        existing = list(out.get(k, []) or [])
        if enabled:
            out[k] = _append_missing(existing, FEATURE_KEYS_15M_OHLCV)
        else:
            out[k] = [f for f in existing if f not in FEATURE_KEYS_15M_OHLCV]
    return out


CFG = apply_15m_feature_toggle(CFG)

# Resolve feature-list aliases to the canonical single source
CFG["position_sizer_feature_priority"] = CFG["position_sizer_features"]
CFG["limit_offset_sizer"] = CFG["position_sizer_features"]

# ============================================================
# Position Sizer V2 Feature Config
# ============================================================


POSITION_SIZER_V2_FEATURE_CONFIG = {
    "shared_feature_keys": [
        # Kalman Position Sizer Features
        "vol_state_1h",
        "short_vol_state_over_long_vol_state",
        "volume_surprise_vs_state",
        "oof_base_mean",
        "oof_base_std",
        "oof_base_min",
        "oof_base_max",
        "oof_base_range",
        "oof_meta_pred",
        "oof_meta_minus_base_mean",
        "oof_top2_gap",
        "oof_sign_agreement_frac",
        "oof_rank_among_candidates",
        "ret_1",
        "ret_3",
        "ret_6",
        "ret_12",
        "ret_24",
        "price_vs_ema_12_z",
        "price_vs_ema_24_z",
        "ema_12_minus_ema_24_z",
        "trend_slope_12_z",
        "trend_slope_24_z",
        "range_1_atr",
        "range_3_atr",
        "rv_6",
        "rv_12",
        "rv_24",
        "rv_ratio_6_24",
        "close_location_in_bar",
        "volume_z_12",
        "volume_z_24",
        "liquidity_shock_z",
        "regime_trend_score",
        "regime_vol_score",
        "regime_liquidity_score",
        "session_progress",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
    ],
    "model1_edge_feature_keys": [
        "oof_base_mean",
        "oof_base_std",
        "oof_meta_pred",
        "oof_top2_gap",
        "oof_sign_agreement_frac",
        "ret_3",
        "ret_6",
        "ret_12",
        "price_vs_ema_12_z",
        "ema_12_minus_ema_24_z",
        "rv_ratio_6_24",
        "regime_trend_score",
        "regime_vol_score",
        "session_progress",
        "range_last_3bars_impulse_range",
        "volatility_contraction_ratio",
        "micro_range_decay",
        "wick_ratio_last_bar",
        "close_position_in_range",
        "rejection_ratio",
        "impulse_participation_volume",
        "terminal_climax_volume",
        "post_impulse_persistence",
        "reversal_bar_strength",
        "bidirectional_range_ratio",
        "momentum_last_3bars_impulse_return",
        "drift_after_impulse",
        "slope_last_n_bars",
        "impulse_volume_ratio",
        "terminal_volume_ratio",
        "post_impulse_volume_persistence",
        "impulse_volume_slope",
        "impulse_vol_ratio",
        "impulse_range_atr_ratio",
        "vol_compression_ratio",
        "range_decay",
    ],
    "model2_downside_feature_keys": [
        "oof_base_mean",
        "oof_base_std",
        "oof_meta_pred",
        "ret_1",
        "ret_3",
        "close_location_in_bar",
        "range_1_atr",
        "rv_6",
        "rv_24",
        "downside_semivol_12",
        "regime_vol_score",
        "regime_liquidity_score",
        "session_progress",
        "impulse_speed",
        "impulse_acceleration",
        "acceleration_of_move",
        "wick_cluster_ratio",
        "rejection_bar_count",
        "ATR_spike_ratio",
        "distance_to_local_high",
        "distance_to_local_low",
        "distance_to_vwap",
        "climax_volume_ratio",
        "reversal_volume_ratio",
        "rejection_volume_ratio",
        "terminal_vol_ratio",
        "volatility_asymmetry",
    ],
    "model3_uncertainty_feature_keys": [
        "oof_base_std",
        "oof_base_range",
        "oof_meta_minus_base_mean",
        "oof_sign_agreement_frac",
        "edge_pred",
        "downside_pred",
        "edge_minus_downside",
        "abs_edge_pred",
        "oof_asym_hat",
        "rv_ratio_6_24",
        "liquidity_shock_z",
        "regime_vol_score",
        "regime_liquidity_score",
        "session_progress",
        "vol_regime_transition",
        "ATR_ratio_short_long",
        "bar_direction_entropy",
        "wick_entropy",
        "impulse_breakdown_score",
        "volume_volatility",
        "volume_regime_shift",
        "volume_entropy",
        "return_per_volume",
        "vol_of_vol",
        "range_cv",
        "return_vol_ratio",
    ],
}

POSITION_SIZER_V2_FEATURE_FLAGS = {
    "enable_model1_optional": False,
    "enable_model2_optional": False,
    "enable_model3_optional": False,
}

POSITION_SIZER_V2_BUCKETS = ["TF_up", "TF_down", "MR_up", "MR_down"]

POSITION_SIZER_V2_BUCKET_CONFIG = {
    "min_samples_total": 500,
    "min_samples_per_fold": 100,
    "min_active_trades_per_policy_eval": 30,
}

POSITION_SIZER_V2_FEATURE_SELECTION_CONFIG = {
    "enabled": True,
    "alpha_grid_small": "np.logspace(-3, 0, 8)",
    "alpha_grid_large": "np.logspace(-3, 0.5, 10)",
    "l1_ratio_grid_small": [0.10, 0.25, 0.50],
    "l1_ratio_grid_large": [0.15, 0.40, 0.70],
    "inner_n_splits_default": 3,
    "selection_rule": "one_std_stable_then_sparse",
    "selection_freq_threshold": 0.67,
    "sparsity_penalty": {
        "edge": 0.04,
        "downside": 0.035,
        "uncertainty": 0.025,
    },
    "max_features_cap": {
        "edge": 24,
        "downside": 18,
        "uncertainty": 14,
    },
    "min_features_floor": {
        "edge": 9,
        "downside": 7,
        "uncertainty": 7,
    },
    "enable_sign_consistency": False,
}


POSITION_SIZER_V2_LAYER0_CONFIG = {
    "enabled": True,
    # Primary families
    "families": [
        "std_threshold",
        "abs_move_threshold",
        "std_plus_abs",
    ],
    # Primary grid
    "z_hours_grid": [6, 8, 10, 12, 16],
    "x_std_grid": [1.4, 1.5, 1.6],
    "y_move_pct_grid": [4.0, 5.0, 6.0, 7.0],
    "duration_grid": [1, 2, 4, 6],
    # Screening horizon
    "phase1_forward_horizon_bars": 12,
    "phase1_ret_threshold": 0.0,  # can later be ATR-normalized if needed
    "mask_opt_max_rows": 10_000,
    "mask_opt_isolate_modes": True,
    "phase1_classifier_max_samples_per_class": 7_500,
    "phase2_metric_max_samples_per_class": 25_000,
    "phase1_classifier_n_splits": 2,
    "phase2_classifier_n_splits": 3,
    "phase2_metric_fold_splits": 3,
    "phase4_tbm_lgbm_n_splits": 3,
    "phase4_tbm_lgbm_max_subset": 100_000,
    "phase4_tbm_lgbm_min_regime_subset": 40,
    "phase4_tbm_lgbm_min_regime_class_count": 2,
    "mask_opt_min_slice_full_panel_fraction": 0.04,
    "mask_opt_min_cap_full_panel_fraction": 0.04,
    "phase1_min_full_panel_fraction": 0.04,
    "incremental_information_n_splits": 3,
    "stage1_symbol_fraction": 0.50,
    "stage1_history_fraction": 0.50,
    "top_k_for_learnability": 48,
    "layer1_mask_opt_max_rows": 5_000,
    "layer1_phase1_classifier_max_samples_per_class": 3_500,
    "layer1_phase2_metric_max_samples_per_class": 12_500,
    "layer1_phase1_classifier_n_splits": 2,
    "layer1_phase2_classifier_n_splits": 3,
    "layer1_phase2_metric_fold_splits": 3,
    "layer1_incremental_information_n_splits": 3,
    # Shortlist
    "shortlist_max_candidates": 5,
    "shortlist_max_per_family": 2,
    # Quantity
    "min_total_events": 300,
    "min_active_days_fraction": 0.20,
    "min_events_per_day": 1,
    "max_events_per_day": 50,
    # High / low viability
    "min_high_events": 100,
    "min_low_events": 100,
    # Distinctness / learnability
    "enable_regime_distinctness_check": True,
    "enable_learnability_check": True,
    "min_regime_distinctness_score": 1.1,
    "min_predictability_gain": 0.0,
    "phase1_min_representatives_per_feature": 2,
    "phase2_min_representatives_per_feature": 0,
    "phase2_prefilter_max_per_feature": 1,
    "phase2_max_candidates_per_family": 3,
    "phase1_prefilter_max_per_feature": 1,
    "phase1_min_representatives_per_family": 2,
    "phase1_min_fold_events": 12,
    "phase1_min_fold_symbols": 4,
    "phase1_overlap_prune_threshold": 0.92,
    "phase2_min_fold_events": 24,
    "phase2_min_fold_symbols": 6,
    "phase2_min_shrunk_edge_for_ridge": 5e-5,
    "phase2_min_positive_fold_fraction_for_ridge": 0.50,
    "phase2_min_candidates_after_sanity_gate": 2,
    "phase2_min_representatives_per_family": 2,
    "phase2_overlap_prune_threshold": 0.92,
    "ridge_phase3_abs_coef_keep_percentile": 50.0,
    "trigger_large_parent_count_threshold": 6,
    "trigger_large_parent_keep_fraction": 0.25,
    "trigger_prescreen_overlap_threshold": 0.92,
    "phase25_min_representatives_per_feature": 1,
    "phase1_min_distinct_symbols": 6,
    "phase1_max_top_symbol_share": 0.45,
    "phase1_min_fold_events": 20,
    "phase1_min_mean_fold_events": 40,
    "phase1_min_fold_symbols": 4,
    "phase1_min_span_days": 5.0,
    "phase2_min_distinct_symbols": 8,
    "phase2_max_top_symbol_share": 0.35,
    "phase2_min_fold_events": 25,
    "phase2_min_mean_fold_events": 75,
    "phase2_min_fold_symbols": 6,
    "phase2_min_span_days": 7.0,
    "phase3_min_distinct_symbols": 6,
    "phase3_max_top_symbol_share": 0.40,
    "phase3_min_fold_events": 15,
    "phase3_min_mean_fold_events": 40,
    "phase3_min_fold_symbols": 4,
    "phase3_min_span_days": 4.0,
    "max_candidates_per_family_per_stage": 3,
    # Conditioners
    "enable_secondary_conditioners": True,
    "enable_trigger_discovery_stage": True,
    "phase3_parent_mode": "regime_trigger",
    "fallback_to_base_regime_if_no_trigger_survives": False,
    "trigger_max_parent_regimes": 20,
    "top_k_triggers_per_regime": 3,
    "min_trigger_events": 150,
    "min_trigger_active_days_fraction": 0.15,
    "min_fold_events": 10,
    "min_trigger_support_ratio": 0.08,
    "trigger_min_distinct_symbols": 6,
    "trigger_max_top_symbol_share": 0.40,
    "trigger_timing_horizon_bars": 24,
    "trigger_score_threshold": 0.0,
    "enable_pullback_recovery": True,
    "enable_breakout": True,
    "enable_sweep_reversal": True,
    "enable_exhaustion": False,
    "enable_compression_release": False,
    "enable_ema_reclaim_touch": True,
    "enable_simple_close_breakout": True,
    "enable_expansion_bar_triggers": True,
    "enable_impulse_bar_triggers": True,
    "enable_relaxed_sweep_triggers": True,
    "enable_compression_release_triggers": False,
    "breakout_lookbacks": [5, 10, 20],
    "reclaim_ema_lens": [10, 20, 30],
    "wick_thresholds": [0.4, 0.6],
    "body_ratio_thresholds": [0.4, 0.6, 0.7],
    "close_location_thresholds": [0.7, 0.8, 0.9],
    "compression_ratio_thresholds": [0.5, 0.6, 0.7],
    "range_atr_thresholds": [1.2, 1.5],
    "distance_to_ema_thresholds": [1.0, 1.5, 2.0],
    "trigger_w_edge": 1.5,
    "trigger_w_stability": 0.8,
    "trigger_w_pred": 0.8,
    "trigger_w_timing": 1.3,
    "trigger_w_disp": 0.9,
    "trigger_w_parent": 0.8,
    "trigger_w_covloss": 0.7,
    "apply_non_dominance": True,
    "keep_family_diversity": True,
    "max_triggers_per_family_per_parent": 3,
    "conditioner_modes": [
        "none",
        "monotonicity_adjust",
        "volatility_adjust",
        "alternation_adjust",
        "liquidity_veto",
    ],
    # Dispersion cap
    "max_allowed_dispersion_quantile": 0.75,
    # Time scaling
    "bars_per_hour": 4,  # assume 15m bars default
    # --- Ridge candidate selection (global, all run modes) ---
    # Max rules sent to the first per-rule LGBM specialist assessment.
    "max_ridge_candidates_total": 200,
    # F1 overlap below this threshold is not penalised (diversity reward zone)
    "ridge_overlap_free_zone": 0.30,
    # Exponent applied to the cheap-rank score (higher = stronger preference for top-ranked)
    "ridge_cheap_rank_exponent": 1.3,
    # Exponent applied to the overlap-excess penalty
    "ridge_overlap_penalty_exponent": 1.8,
    # Minimum support ratio between two rules to qualify for overlap computation
    "ridge_support_ratio_min": 0.70,
    # Penalty/boost strength for rules away from TARGET_SUPPORT
    "ridge_support_penalty_strength": 1.0,
    "ridge_support_boost_strength": 1.0,
    # Cross-bucket overlap discount multipliers (lower = rules from different buckets
    # need much higher raw overlap before they suppress each other)
    "ridge_cross_side_overlap_mult": 0.50,  # different long/short
    "ridge_cross_horizon_overlap_mult": 0.70,  # same side, different horizon
    # Cascade dedup thresholds (descending): stop when ≤ max_ridge_candidates_total rules remain
    "ridge_dedup_thresholds": [
        0.95,
        0.925,
        0.90,
        0.875,
        0.85,
        0.825,
        0.80,
        0.75,
        0.70,
        0.65,
        0.60,
    ],
    # Per-bucket structural dedup target (top-N kept per bucket before global specialist cascade)
    "overlap_dedup_bucket_top_target": 100,
    "global_ridge_per_slice_basket_size": 100,
    "global_ridge_candidate_cap": 200,
    "stage1_lgbm_top_n_for_strong": 100,
    # --- Ridge validation criteria (research-grade vs production-grade) ---
    # Minimum gross PnL threshold to accept a rule (before fees)
    # Set to 0.0 to accept any positive gross PnL, higher values for stricter filtering
    "ridge_min_gross_pnl_threshold": 0.0,
    # Minimum Sharpe-like ratio (mean/std) of returns within the mask
    # Rules with insufficient signal quality in the mask are rejected even with gross profit
    # 0.3 = mild signal, 0.5 = moderate, 0.7 = strong (adjust based on your data)
    "ridge_min_mask_sharpe_threshold": 0.3,
    # Threshold selection policy for fallback when post-fee profit is not achieved
    # "best_net_pnl" = use best net PnL threshold (default, strict)
    # "best_gross_pnl" = use best gross PnL threshold when net fails (research mode)
    "ridge_threshold_selection_policy": "best_gross_pnl",
}


CFG["ORDERBOOK_BASE_FEATURE_KEYS"] = ORDERBOOK_BASE_FEATURE_KEYS
CFG["FUNDING_BASE_FEATURE_KEYS"] = []
CFG["ORDERBOOK_FEATURE_KEYS"] = ORDERBOOK_FEATURE_KEYS
CFG["ORDERBOOK_DIAGNOSTIC_ONLY_FEATURE_KEYS"] = ORDERBOOK_DIAGNOSTIC_ONLY_FEATURE_KEYS
CFG["ORDERBOOK_EXCLUDED_STALE_FEATURE_KEYS"] = ORDERBOOK_EXCLUDED_STALE_FEATURE_KEYS
CFG["CROSS_ASSET_FEATURE_KEYS"] = CROSS_ASSET_FEATURE_KEYS
CFG["MARKET_OI_REGIME_FEATURE_KEYS"] = MARKET_OI_REGIME_FEATURE_KEYS
CFG["PRICE_OI_STATE_FEATURE_KEYS"] = PRICE_OI_STATE_FEATURE_KEYS
CFG["MARKET_FUNDING_REGIME_FEATURE_KEYS"] = MARKET_FUNDING_REGIME_FEATURE_KEYS
CFG["FUNDING_PRICE_OI_INTERACTION_FEATURE_KEYS"] = (
    FUNDING_PRICE_OI_INTERACTION_FEATURE_KEYS
)
CFG["OHLCV_BREADTH_FEATURE_KEYS"] = OHLCV_BREADTH_FEATURE_KEYS
CFG["OHLCV_CRASH_PHASE_FEATURE_KEYS"] = OHLCV_CRASH_PHASE_FEATURE_KEYS
CFG["MARKET_SYNCHRONIZATION_FEATURE_KEYS"] = MARKET_SYNCHRONIZATION_FEATURE_KEYS
CFG["LIQUIDATION_STATE_SCORE_FEATURE_KEYS"] = LIQUIDATION_STATE_SCORE_FEATURE_KEYS
CFG["PERP_FEATURE_KEYS"] = PERP_FEATURE_KEYS
CFG["PERP_META_PRIMARY_FEATURE_KEYS"] = PERP_META_PRIMARY_FEATURE_KEYS
CFG["PERP_PRICE_RELATION_FEATURE_KEYS"] = PERP_PRICE_RELATION_FEATURE_KEYS
CFG["KRAKEN_INDEX_PREMIUM_FEATURE_KEYS"] = KRAKEN_INDEX_PREMIUM_FEATURE_KEYS
CFG["SPOT_FOR_PERPS_BASE_FEATURE_KEYS"] = SPOT_FOR_PERPS_BASE_FEATURE_KEYS
CFG["SPOT_FOR_PERPS_META_FEATURE_KEYS"] = SPOT_FOR_PERPS_META_FEATURE_KEYS
CFG["PERP_TRADEABILITY_FEATURE_KEYS"] = PERP_TRADEABILITY_FEATURE_KEYS
CFG["LGBM_PERP_FEATURE_KEYS"] = LGBM_PERP_FEATURE_KEYS
CFG["PERP_EVENT_RISK_FEATURE_KEYS"] = PERP_EVENT_RISK_FEATURE_KEYS
CFG["PERP_CARRY_ALPHA_FEATURE_KEYS"] = PERP_CARRY_ALPHA_FEATURE_KEYS
CFG["OI_FEATURE_KEYS"] = OI_FEATURE_KEYS
CFG["OI_TRADING_FEATURE_KEYS"] = OI_TRADING_FEATURE_KEYS
CFG["OI_NORMALIZED_FEATURE_KEYS"] = OI_NORMALIZED_FEATURE_KEYS
CFG["LONG_HORIZON_PERP_META_FEATURE_KEYS"] = LONG_HORIZON_PERP_META_FEATURE_KEYS
CFG["VOLUME_FREE_PERP_BASE_FEATURE_KEYS"] = VOLUME_FREE_PERP_BASE_FEATURE_KEYS
CFG["VOLUME_FREE_PERP_META_FEATURE_KEYS"] = VOLUME_FREE_PERP_META_FEATURE_KEYS
CFG["RESIDUAL_FEATURE_KEYS"] = RESIDUAL_FEATURE_KEYS
CFG["RESIDUAL_BASE_FEATURE_KEYS"] = RESIDUAL_BASE_FEATURE_KEYS
CFG["RESIDUAL_META_FEATURE_KEYS"] = RESIDUAL_META_FEATURE_KEYS
CFG["T2_FUNNEL_BASE_CONTEXT_FEATURE_KEYS"] = T2_FUNNEL_BASE_CONTEXT_FEATURE_KEYS
CFG["T2_FUNNEL_META_CONTEXT_FEATURE_KEYS"] = T2_FUNNEL_META_CONTEXT_FEATURE_KEYS
CFG["RETENTION_CONTINUATION_F1_FEATURE_KEYS"] = RETENTION_CONTINUATION_F1_FEATURE_KEYS
CFG["RETENTION_CONTINUATION_F2_FEATURE_KEYS"] = RETENTION_CONTINUATION_F2_FEATURE_KEYS
CFG["RETENTION_CONTINUATION_F3_FEATURE_KEYS"] = RETENTION_CONTINUATION_F3_FEATURE_KEYS
CFG["RETENTION_CONTINUATION_F4_FEATURE_KEYS"] = RETENTION_CONTINUATION_F4_FEATURE_KEYS
CFG["RETENTION_CONTINUATION_F5_FEATURE_KEYS"] = RETENTION_CONTINUATION_F5_FEATURE_KEYS
CFG["RETENTION_CONTINUATION_F6_FEATURE_KEYS"] = RETENTION_CONTINUATION_F6_FEATURE_KEYS
CFG["RETENTION_CONTINUATION_F7_FEATURE_KEYS"] = RETENTION_CONTINUATION_F7_FEATURE_KEYS
CFG["RETENTION_CONTINUATION_F8_FEATURE_KEYS"] = RETENTION_CONTINUATION_F8_FEATURE_KEYS
# Native L2 is intentionally registered as a research-only sidecar.  It is
# not appended to production base/meta feature lists until a complete
# candidate-level availability and OOF economics gate passes.
CFG["NATIVE_L2_CONTINUATION_FEATURE_KEYS"] = list(NATIVE_L2_CONTINUATION_FEATURE_KEYS)
CFG["NATIVE_L2_CONTINUATION_FEATURE_STATUS"] = "RESEARCH_ONLY_NATIVE_SOURCE_COHORT"
CFG["RETENTION_CONTINUATION_FEATURE_GROUPS"] = RETENTION_CONTINUATION_FEATURE_GROUPS
CFG["ROLLING_ALPHA_FEATURE_KEYS"] = ROLLING_ALPHA_FEATURE_KEYS
CFG["ROLLING_ALPHA_TARGET_AUDIT_COLUMNS"] = ROLLING_ALPHA_TARGET_AUDIT_COLUMNS
CFG["CURRENT_REGIME_AE_FEATURE_KEYS"] = CURRENT_REGIME_AE_FEATURE_KEYS
UNSUPERVISED_REGIME_LEARNING = UNSUPERVISED_REGIME_LEARNING_DEFAULTS
CFG["UNSUPERVISED_REGIME_LEARNING"] = UNSUPERVISED_REGIME_LEARNING
CFG["PRICE_MEMORY_FEATURE_KEYS"] = PRICE_MEMORY_FEATURE_KEYS
CFG["OI_WEIGHTED_LOCATION_BASE_FEATURE_KEYS"] = OI_WEIGHTED_LOCATION_BASE_FEATURE_KEYS
CFG["OI_WEIGHTED_LOCATION_META_FEATURE_KEYS"] = OI_WEIGHTED_LOCATION_META_FEATURE_KEYS
CFG["DAILY_SR_BASE_FEATURE_KEYS"] = DAILY_SR_BASE_FEATURE_KEYS
CFG["WEEKLY_SR_META_FEATURE_KEYS"] = WEEKLY_SR_META_FEATURE_KEYS
CFG["CROSS_ASSET_BASE_FEATURE_KEYS"] = []
CFG["ORDERBOOK_META_FEATURE_KEYS"] = ORDERBOOK_META_FEATURE_KEYS
CFG["FUNDING_META_FEATURE_KEYS"] = [
    "fund_rate",
    "fund_rate_ffill",
    "fund_rate_z_14d",
    "fund_rate_mom_8h",
    "fund_rate_mom_24h",
    "fund_abs_z",
    "fund_abs_z_14d",
    "fund_carry_24h",
    "funding_proxy",
    "a_funding_proxy",
    "fund_mom_8h",
    "fund_mom_24h",
    "fund_sign_persistence_3",
    "fund_sign_persistence_24h",
    "fund_extreme_duration_24h",
    "fund_rank_30d",
    "spot_perp_return_agreement_4h",
    "spot_leads_perp_1h",
    "spot_perp_vol_ratio_24h",
    "spot_available",
]
CFG["CROSS_ASSET_META_FEATURE_KEYS"] = [
    "xasset_fund_dispersion_basket",
    "xasset_fund_extreme_share_basket",
    "xasset_asset_minus_basket_fund_z",
    "xasset_asset_minus_mkt_funding",
    "xasset_btc_funding_z",
    "xasset_btc_fund_z",
]
CFG["INTERACTION_META_FEATURE_KEYS"] = [
    "fund_abs_z_x_ret24h_sign",
    "fund_abs_z_x_rv_24h",
    "fund_z_x_trend_strength",
]
CFG["CHANGE_POINT_REGIME_FEATURE_KEYS"] = [
    "log_realized_vol",
    "range_volatility",
    "range_per_volume",
    "trend_r2_24",
    "trend_r2_48",
    "path_entropy_12",
    "path_entropy_24",
    "binned_return_entropy_24",
    "directional_entropy_20",
    "log_realized_vol_cp_z_8_32_96",
    "log_realized_vol_cp_logstd_8_32",
    "log_realized_vol_cp_absratio_8_32",
    "range_volatility_cp_z_8_32_96",
    "range_volatility_cp_logstd_8_32",
    "range_volatility_cp_absratio_8_32",
    "vol_of_vol_cp_z_8_32_96",
    "vol_of_vol_cp_logstd_8_32",
    "vol_of_vol_cp_absratio_8_32",
    "volume_zscore_cp_z_8_32_96",
    "volume_zscore_cp_logstd_8_32",
    "volume_zscore_cp_absratio_8_32",
    "range_per_volume_cp_z_8_32_96",
    "range_per_volume_cp_logstd_8_32",
    "range_per_volume_cp_absratio_8_32",
    "return_autocorr_cp_z_8_32_96",
    "return_autocorr_cp_logstd_8_32",
    "return_autocorr_cp_absratio_8_32",
    "adx_cp_z_8_32_96",
    "adx_cp_logstd_8_32",
    "adx_cp_absratio_8_32",
    "choppiness_cp_z_8_32_96",
    "choppiness_cp_logstd_8_32",
    "choppiness_cp_absratio_8_32",
    "trend_r2_cp_z_8_32_96",
    "trend_r2_cp_logstd_8_32",
    "trend_r2_cp_absratio_8_32",
    "price_entropy_cp_z_8_32_96",
    "price_entropy_cp_logstd_8_32",
    "price_entropy_cp_absratio_8_32",
]
CFG["CHANGE_POINT_SIGNAL_FEATURE_KEYS"] = [
    "trend_r2_24",
    "trend_r2_48",
    "path_entropy_12",
    "path_entropy_24",
    "binned_return_entropy_24",
    "directional_entropy_20",
    "return_autocorr_cp_z_8_32_96",
    "return_autocorr_cp_logstd_8_32",
    "return_autocorr_cp_absratio_8_32",
    "adx_cp_z_8_32_96",
    "adx_cp_logstd_8_32",
    "adx_cp_absratio_8_32",
    "choppiness_cp_z_8_32_96",
    "choppiness_cp_logstd_8_32",
    "choppiness_cp_absratio_8_32",
    "trend_r2_cp_z_8_32_96",
    "trend_r2_cp_logstd_8_32",
    "trend_r2_cp_absratio_8_32",
    "price_entropy_cp_z_8_32_96",
    "price_entropy_cp_logstd_8_32",
    "price_entropy_cp_absratio_8_32",
]
CFG["CHANGE_POINT_CONTEXT_FEATURE_KEYS"] = [
    "log_realized_vol",
    "range_volatility",
    "range_per_volume",
    "log_realized_vol_cp_z_8_32_96",
    "log_realized_vol_cp_logstd_8_32",
    "log_realized_vol_cp_absratio_8_32",
    "range_volatility_cp_z_8_32_96",
    "range_volatility_cp_logstd_8_32",
    "range_volatility_cp_absratio_8_32",
    "vol_of_vol_cp_z_8_32_96",
    "vol_of_vol_cp_logstd_8_32",
    "vol_of_vol_cp_absratio_8_32",
    "volume_zscore_cp_z_8_32_96",
    "volume_zscore_cp_logstd_8_32",
    "volume_zscore_cp_absratio_8_32",
    "range_per_volume_cp_z_8_32_96",
    "range_per_volume_cp_logstd_8_32",
    "range_per_volume_cp_absratio_8_32",
]
CFG["spread_proxy_features"] = SPREAD_PROXY_FEATURE_KEYS
MODEL_DIRECT_BASE_FEATURE_KEYS = [
    # Asset-local directional/context features only. Cross-symbol regime
    # aggregates are deliberately routed to meta below.
    "loc_range_pos_24",
    "loc_range_pos_48",
    "loc_swing_range_pos_24",
    "trend_slope_48h",
    "trend_pct_mkt_resid",
    "bars_in_high_vol_state_log_norm",
    "price_x_oi_1d",
    "price_x_oi_3d",
    "price_x_oi_7d",
    "oi_value_1d_chg_z_90d",
    "oi_value_3d_chg_z_90d",
    "oi_value_7d_chg_z_90d",
    "oi_value_7d_chg_z_180d",
    "funding_per_hour",
    "range_expansion_ratio",
    "efficiency_ratio_20",
    *CRASH_LIFECYCLE_ASSET_FEATURE_KEYS,
]

MODEL_REGIME_CONTEXT_META_FEATURE_KEYS = [
    "mkt_oi_breadth_rising_24h",
    "mkt_oi_chg_z_24h",
    "mkt_oi_dispersion_24h",
    "mkt_oi_z_30d",
    "mkt_ret_eq_24h",
    "mkt_ret_eq_4h",
    "xasset_mkt_spread_bps",
    "regime_liquidity_score",
    *MARKET_OI_REGIME_FEATURE_KEYS,
    *PRICE_OI_STATE_FEATURE_KEYS,
    *MARKET_FUNDING_REGIME_FEATURE_KEYS,
    *FUNDING_PRICE_OI_INTERACTION_FEATURE_KEYS,
    *OHLCV_BREADTH_FEATURE_KEYS,
    *OHLCV_CRASH_PHASE_FEATURE_KEYS,
    *MARKET_SYNCHRONIZATION_FEATURE_KEYS,
    *CRASH_LIFECYCLE_MARKET_FEATURE_KEYS,
    *LIQUIDATION_STATE_SCORE_FEATURE_KEYS,
    *NEGATIVE_RESIDUAL_META_FEATURE_KEYS,
]

MODEL_REGIME_XS_META_FEATURE_KEYS = [
    "xs_mean__oi_1d_x_funding",
    "xs_mean__price_x_oi_1d",
    "xs_median__efficiency_ratio_20",
    "xs_std__oi_1d_x_funding",
    "xs_std__oi_3d_x_funding",
    "xs_std__oi_7d_x_funding",
    "xs_std__oi_to_volume_7d_z_180d",
    "xs_std__price_x_oi_1d",
    "xs_std__price_x_oi_3d",
    "xs_std__price_x_oi_7d",
    "xs_dispersion__amihud_illiq",
    "xs_dispersion__amihud_z",
    "xs_dispersion__amihud_z_peer_resid",
    "xs_dispersion__asset_minus_mkt_oi_1d",
    "xs_dispersion__bars_in_high_vol_state_log_norm",
    "xs_dispersion__efficiency_ratio_20",
    "xs_dispersion__ffd_amihud_04",
    "xs_dispersion__ffd_amihud_06",
    "xs_dispersion__funding_per_hour",
    "xs_dispersion__liquidity_ratio_peer_resid",
    "xs_dispersion__ob_depth_to_qv_z_x_rvol_z",
    "xs_dispersion__ob_spread_bps_z_24h",
    "xs_dispersion__ob_spread_bps_z_7d",
    "xs_dispersion__ob_spread_mkt_resid",
    "xs_dispersion__ob_spread_z_x_rv_24h",
    "xs_dispersion__oi_1d_x_funding",
    "xs_dispersion__oi_3d_x_funding",
    "xs_dispersion__oi_7d_x_funding",
    "xs_dispersion__oi_to_volume_7d_z_180d",
    "xs_dispersion__oi_value_1d_chg_z_90d",
    "xs_dispersion__oi_value_3d_chg_z_90d",
    "xs_dispersion__oi_value_7d_chg_z_180d",
    "xs_dispersion__oi_value_7d_chg_z_90d",
    "xs_dispersion__oi_value_z_90d",
    "xs_dispersion__price_x_oi_1d",
    "xs_dispersion__price_x_oi_3d",
    "xs_dispersion__price_x_oi_7d",
    "xs_dispersion__rvol_z",
    "xs_dispersion__rvol_z_peer_resid",
    "xs_dispersion__trend_pct_mkt_resid",
    "xs_dispersion__vol_z",
    "xs_dispersion__vol_z24",
    "xs_dispersion__vol_z_4h",
    "xs_dispersion__vol_z_peer_resid",
    "xs_dispersion__volatility_zscore",
    "xs_dispersion__volume_percentile",
    "xs_dispersion__volume_z_12",
    "xs_dispersion__volume_z_24",
    "xs_dispersion__volume_zscore_48h",
    "xs_dispersion__xasset_ob_liquidity_divergence_z_24h",
    "xs_dispersion__xasset_ob_liquidity_peer_resid",
    "xs_dispersion__xasset_ob_liquidity_ts_resid",
]

MODEL_REGIME_TAIL_META_FEATURE_KEYS = [
    *[
        f"q_lower_tail__{_p}"
        for _p in [
            "amihud_z_peer_resid",
            "liquidity_ratio_peer_resid",
            "ob_depth_usd_l20_z",
            "ob_spread_bps_z_24h",
            "ob_trade_size_to_l1_depth_z_24h",
            "oi_1d_x_funding",
            "oi_3d_x_funding",
            "oi_7d_x_funding",
            "oi_value_1d_chg_z_90d",
            "price_x_oi_1d",
            "ret48h_bench_resid",
            "vol_z_peer_resid",
            "volume_z_24",
            "xasset_mkt_spread_bps",
            "xasset_ob_liquidity_peer_resid",
            "xasset_ob_liquidity_ts_resid",
        ]
    ],
    *[
        f"q_upper_tail__{_p}"
        for _p in [
            "amihud_z_peer_resid",
            "bars_in_high_vol_state_log_norm",
            "liquidity_ratio_peer_resid",
            "ob_spread_bps_z_24h",
            "ob_trade_size_to_l1_depth_z_24h",
            "oi_1d_x_funding",
            "oi_3d_x_funding",
            "oi_7d_x_funding",
            "oi_to_volume_7d_z_180d",
            "oi_value_1d_chg_z_90d",
            "oi_value_3d_chg_z_90d",
            "price_x_oi_1d",
            "price_x_oi_3d",
            "price_x_oi_7d",
            "ret48h_bench_resid",
            "vol_z_peer_resid",
            "volume_z_24",
            "xasset_mkt_spread_bps",
            "xasset_ob_liquidity_peer_resid",
            "xasset_ob_liquidity_ts_resid",
        ]
    ],
    *[
        f"q_iqr__{_p}"
        for _p in [
            "amihud_z_peer_resid",
            "bars_in_high_vol_state_log_norm",
            "ob_trade_size_to_l1_depth_z_24h",
            "price_x_oi_1d",
            "ret48h_bench_resid",
            "vol_z24",
            "volume_z_12",
            "xasset_mkt_spread_bps",
        ]
    ],
    *[
        f"q_tail_width__{_p}"
        for _p in [
            "amihud_z_peer_resid",
            "bars_in_high_vol_state_log_norm",
            "liquidity_ratio_peer_resid",
            "loc_swing_range_pos_48",
            "ob_depth_l20_to_qv_24h",
            "ob_depth_l20_to_qv_z_7d",
            "ob_spread_z_x_rv_24h",
            "oi_1d_x_funding",
            "oi_3d_x_funding",
            "oi_7d_x_funding",
            "oi_to_volume_7d_z_180d",
            "oi_value_1d_chg_z_90d",
            "oi_value_3d_chg_z_90d",
            "oi_value_7d_chg_z_180d",
            "oi_value_7d_chg_z_90d",
            "oi_value_z_90d",
            "price_x_oi_1d",
            "price_x_oi_3d",
            "price_x_oi_7d",
            "ret48h_bench_resid",
            "rvol_z_peer_resid",
            "vol_z_peer_resid",
            "volatility_zscore",
            "volume_z_12",
            "volume_z_24",
            "xasset_mkt_spread_bps",
            "xasset_ob_liquidity_ts_resid",
        ]
    ],
    *[
        f"q_tail_asym__{_p}"
        for _p in [
            "amihud_z_peer_resid",
            "bars_in_high_vol_state_log_norm",
            "ob_depth_l20_to_qv_z_7d",
            "ob_depth_to_qv_z_x_rvol_z",
            "ob_depth_usd_l20_z",
            "ob_notional_to_depth_l20_z_24h",
            "ob_spread_bps_z_24h",
            "ob_spread_bps_z_7d",
            "ob_spread_mkt_resid",
            "ob_trade_size_to_l1_depth_z_24h",
            "oi_1d_x_funding",
            "oi_3d_x_funding",
            "oi_7d_x_funding",
            "oi_to_volume_7d_z_180d",
            "oi_value_1d_chg_z_90d",
            "oi_value_z_90d",
            "price_x_oi_1d",
            "price_x_oi_3d",
            "price_x_oi_7d",
            "ret48h_bench_resid",
            "vol_z_4h",
            "vol_z_peer_resid",
            "volume_z_12",
            "volume_z_24",
            "xasset_mkt_spread_bps",
            "xasset_ob_liquidity_peer_resid",
            "xasset_ob_liquidity_ts_resid",
        ]
    ],
]

MODEL_REGIME_EIGEN_META_FEATURE_KEYS = [
    "eig_effective_rank__breakout_all",
    "eig_effective_rank__open_interest",
    "eig_participation_ratio__breakout_all",
    "xs_cov_effective_rank__xs_asset_portable_all",
    "xs_cov_effective_rank__xs_open_interest",
]

MARKET_SPECTRAL_POSITION_META_FEATURE_KEYS = [
    "state_spectral_eig_lambda1_share",
    "state_spectral_eig_top3_share",
    "state_spectral_eig_effective_rank",
    "state_spectral_eig_entropy",
    "state_spectral_eig_gap_1_2",
    "state_spectral_eig_gap_ratio_1_2",
    "state_spectral_eig_condition",
    "state_spectral_pc1_score",
    "state_spectral_pc2_score",
    "state_spectral_pc3_score",
    "state_spectral_pc1_z",
    "state_spectral_pc2_z",
    "state_spectral_pc3_z",
    "state_spectral_abs_pc1_z",
    "state_spectral_abs_pc2_z",
    "state_spectral_abs_pc3_z",
    "state_spectral_sum_abs_top3_pc_z",
    "state_spectral_projection_norm_top3",
    "state_spectral_top3_reconstruction_error",
    "state_spectral_top3_reconstruction_ratio",
    "state_spectral_top3_mahalanobis",
]

MARKET_SPECTRAL_POSITION_SOURCE_FEATURE_KEYS = [
    "mkt_ret_eq_1h",
    "mkt_ret_eq_4h",
    "mkt_ret_eq_24h",
    "market_breadth_1h",
    "market_breadth_4h",
    "market_breadth_24h",
    "market_dispersion_1h",
    "market_dispersion_4h",
    "market_dispersion_24h",
    "symbol_minus_mkt_ret_1h",
    "symbol_minus_mkt_ret_4h",
    "symbol_minus_mkt_ret_24h",
    "rv_24h",
    "realized_volatility_24h",
    "volume_percentile",
    "volume_zscore_48h",
    "volume_z_12",
    "volume_z_24",
    "vol_z24",
    "amihud_z",
    "amihud_z_peer_resid",
    "liquidity_ratio_peer_resid",
    "funding_per_hour",
    "funding_z",
    "funding_abs_z",
    "oi_value_1d_chg_z_90d",
    "oi_value_3d_chg_z_90d",
    "oi_value_7d_chg_z_90d",
    "oi_value_z_90d",
    "oi_1d_x_funding",
    "oi_3d_x_funding",
    "oi_7d_x_funding",
    "oi_to_volume_7d_z_180d",
    "price_x_oi_1d",
    "price_x_oi_3d",
    "price_x_oi_7d",
    "xasset_mkt_spread_bps",
    "xasset_ob_liquidity_peer_resid",
    "xasset_ob_liquidity_ts_resid",
    "trend_pct_mkt_resid",
    "efficiency_ratio_20",
    "choppiness_index_20",
    "coherence_24",
]

MODEL_REGIME_COMPOSITE_EIGEN_GROUPS = {
    "breakout_all": [
        "bars_in_high_vol_state_log_norm",
        "efficiency_ratio_20",
        "loc_range_pos_24",
        "loc_range_pos_48",
        "loc_swing_range_pos_24",
        "range_expansion_ratio",
        "trend_pct_mkt_resid",
        "trend_slope_48h",
        "vol_z24",
        "volume_z_12",
        "volume_z_24",
        "xasset_mkt_spread_bps",
    ],
    "open_interest": [
        "funding_per_hour",
        "oi_1d_x_funding",
        "oi_3d_x_funding",
        "oi_7d_x_funding",
        "oi_to_volume_7d_z_180d",
        "oi_value_1d_chg_z_90d",
        "oi_value_3d_chg_z_90d",
        "oi_value_7d_chg_z_90d",
        "oi_value_7d_chg_z_180d",
        "oi_value_z_90d",
        "price_x_oi_1d",
        "price_x_oi_3d",
        "price_x_oi_7d",
    ],
    "xs_asset_portable_all": [
        "amihud_z",
        "amihud_z_peer_resid",
        "bars_in_high_vol_state_log_norm",
        "efficiency_ratio_20",
        "ffd_amihud_04",
        "ffd_amihud_06",
        "funding_per_hour",
        "liquidity_ratio_peer_resid",
        "oi_1d_x_funding",
        "oi_3d_x_funding",
        "oi_7d_x_funding",
        "oi_to_volume_7d_z_180d",
        "oi_value_1d_chg_z_90d",
        "oi_value_3d_chg_z_90d",
        "oi_value_7d_chg_z_90d",
        "oi_value_7d_chg_z_180d",
        "price_x_oi_1d",
        "price_x_oi_3d",
        "price_x_oi_7d",
        "trend_pct_mkt_resid",
        "vol_z24",
        "volume_z_12",
        "volume_z_24",
        "xasset_mkt_spread_bps",
    ],
    "xs_open_interest": [
        "funding_per_hour",
        "oi_1d_x_funding",
        "oi_3d_x_funding",
        "oi_7d_x_funding",
        "oi_to_volume_7d_z_180d",
        "oi_value_1d_chg_z_90d",
        "oi_value_3d_chg_z_90d",
        "oi_value_7d_chg_z_90d",
        "oi_value_7d_chg_z_180d",
        "oi_value_z_90d",
        "price_x_oi_1d",
        "price_x_oi_3d",
        "price_x_oi_7d",
    ],
}

MODEL_REGIME_COMPOSITE_META_FEATURE_KEYS = list(
    dict.fromkeys(
        MODEL_REGIME_CONTEXT_META_FEATURE_KEYS
        + MODEL_REGIME_XS_META_FEATURE_KEYS
        + MODEL_REGIME_TAIL_META_FEATURE_KEYS
        + MODEL_REGIME_EIGEN_META_FEATURE_KEYS
        + MARKET_SPECTRAL_POSITION_META_FEATURE_KEYS
    )
)

CFG["MODEL_DIRECT_BASE_FEATURE_KEYS"] = MODEL_DIRECT_BASE_FEATURE_KEYS
CFG["CRASH_LIFECYCLE_ASSET_FEATURE_KEYS"] = CRASH_LIFECYCLE_ASSET_FEATURE_KEYS
CFG["CRASH_LIFECYCLE_MARKET_FEATURE_KEYS"] = CRASH_LIFECYCLE_MARKET_FEATURE_KEYS
CFG["CRASH_LIFECYCLE_NEW_FEATURE_KEYS"] = CRASH_LIFECYCLE_NEW_FEATURE_KEYS
CFG["NEGATIVE_RESIDUAL_PRIMITIVE_FEATURE_KEYS"] = (
    NEGATIVE_RESIDUAL_PRIMITIVE_FEATURE_KEYS
)
CFG["NEGATIVE_RESIDUAL_COMPOSITE_FEATURE_KEYS"] = (
    NEGATIVE_RESIDUAL_COMPOSITE_FEATURE_KEYS
)
CFG["NEGATIVE_RESIDUAL_TEMPORAL_MECHANISM_FEATURE_KEYS"] = (
    NEGATIVE_RESIDUAL_TEMPORAL_MECHANISM_FEATURE_KEYS
)
CFG["NEGATIVE_RESIDUAL_ALL_FEATURE_KEYS"] = NEGATIVE_RESIDUAL_META_FEATURE_KEYS
CFG["NEGATIVE_RESIDUAL_META_FEATURE_KEYS"] = NEGATIVE_RESIDUAL_META_FEATURE_KEYS
CFG["NEGATIVE_RESIDUAL_PROMOTED_META_FEATURE_KEYS"] = (
    NEGATIVE_RESIDUAL_PROMOTED_META_FEATURE_KEYS
)
CFG["NEGATIVE_RESIDUAL_FEATURE_SCHEMA_VERSION"] = (
    NEGATIVE_RESIDUAL_FEATURE_SCHEMA_VERSION
)
CFG["NEGATIVE_RESIDUAL_FEATURE_CONTRACT"] = negative_residual_feature_contract()
CFG["MARKET_REGIME_CHANGE"] = {
    "schema_version": MARKET_REGIME_CHANGE_SCHEMA_VERSION,
    "source_features": dict(MARKET_REGIME_CHANGE_SOURCES),
    "feature_keys": list(MARKET_REGIME_CHANGE_FEATURE_KEYS),
    "scope": "meta_and_unsupervised_regime_learning",
}
CFG.setdefault("feature_required_lookback_hours_by_feature", {}).update(
    {
        key: NEGATIVE_RESIDUAL_CAUSAL_WINDOW_HOURS
        for key in NEGATIVE_RESIDUAL_META_FEATURE_KEYS
    }
)
CFG["OHLCV_LIFECYCLE_FEATURE_KEYS"] = OHLCV_LIFECYCLE_FEATURE_KEYS
CFG["MARKET_BREADTH_LIFECYCLE_FEATURE_KEYS"] = MARKET_BREADTH_LIFECYCLE_FEATURE_KEYS
CFG["MARKET_SYNCHRONIZATION_ADDITION_KEYS"] = MARKET_SYNCHRONIZATION_ADDITION_KEYS
CFG["MODEL_REGIME_CONTEXT_META_FEATURE_KEYS"] = MODEL_REGIME_CONTEXT_META_FEATURE_KEYS
CFG["MODEL_REGIME_XS_META_FEATURE_KEYS"] = MODEL_REGIME_XS_META_FEATURE_KEYS
CFG["MODEL_REGIME_TAIL_META_FEATURE_KEYS"] = MODEL_REGIME_TAIL_META_FEATURE_KEYS
CFG["MODEL_REGIME_EIGEN_META_FEATURE_KEYS"] = MODEL_REGIME_EIGEN_META_FEATURE_KEYS
CFG["MARKET_SPECTRAL_POSITION_META_FEATURE_KEYS"] = (
    MARKET_SPECTRAL_POSITION_META_FEATURE_KEYS
)
CFG["MARKET_SPECTRAL_POSITION_SOURCE_FEATURE_KEYS"] = (
    MARKET_SPECTRAL_POSITION_SOURCE_FEATURE_KEYS
)
CFG["market_spectral_position_lookback"] = 48
CFG["market_spectral_position_min_periods"] = 24
CFG["market_spectral_position_top_k"] = 3
CFG["market_spectral_position_max_source_features"] = 64
CFG["market_spectral_position_shrinkage"] = 0.10
CFG["MODEL_REGIME_COMPOSITE_EIGEN_GROUPS"] = MODEL_REGIME_COMPOSITE_EIGEN_GROUPS
CFG["MODEL_REGIME_COMPOSITE_META_FEATURE_KEYS"] = (
    MODEL_REGIME_COMPOSITE_META_FEATURE_KEYS
)
CFG["base_shared_feature_keys"] += [
    "MODEL_DIRECT_BASE_FEATURE_KEYS",
    "spread_proxy_features",
    "PRICE_MEMORY_FEATURE_KEYS",
    "DAILY_SR_BASE_FEATURE_KEYS",
    "VOLUME_FREE_PERP_BASE_FEATURE_KEYS",
    "RESIDUAL_BASE_FEATURE_KEYS",
    "ORDERBOOK_BASE_FEATURE_KEYS",
    "FUNDING_BASE_FEATURE_KEYS",
    "CROSS_ASSET_BASE_FEATURE_KEYS",
    "CHANGE_POINT_SIGNAL_FEATURE_KEYS",
]
CFG["meta_product_feature_keys"] = _append_missing(
    CFG.get("meta_product_feature_keys", []),
    ["spread_proxy_features"],
)
CFG["meta_shared_feature_keys"] += [
    "MODEL_REGIME_COMPOSITE_META_FEATURE_KEYS",
    "spread_proxy_features",
    "WEEKLY_SR_META_FEATURE_KEYS",
    "VOLUME_FREE_PERP_META_FEATURE_KEYS",
    "LONG_HORIZON_PERP_META_FEATURE_KEYS",
    "RESIDUAL_META_FEATURE_KEYS",
    "ORDERBOOK_META_FEATURE_KEYS",
    "FUNDING_META_FEATURE_KEYS",
    "CROSS_ASSET_META_FEATURE_KEYS",
    "INTERACTION_META_FEATURE_KEYS",
    "CHANGE_POINT_CONTEXT_FEATURE_KEYS",
]

# Stage-I grouped stability MDA is intentionally restricted to the declared
# base/meta feature-key universes.  The selector resolves these lists per
# side/head and refuses any un-inventoried head rather than falling back to all
# columns in an experiment panel.  Its runtime recipe lives with the reusable
# selector in ``stage_i_feature_selection.py``.
# Backward-compatible M6-named feature-pool key for the one active shared
# exact-net residual expert. No inactive path-auxiliary target pools may enter.
CFG["STAGE_I_M6_SHARED_UNION_META_FEATURE_KEYS"] = [
    "RESIDUAL_META_FEATURE_KEYS",
    "META_BASE_PERFORMANCE_FEATURE_KEYS",
    "META_MODEL_UNCERTAINTY_FEATURE_KEYS",
    "META_RECENT_EFFECTIVENESS_FEATURE_KEYS",
    "MODEL_REGIME_COMPOSITE_META_FEATURE_KEYS",
    "BASE_LGBM_META_UNCERTAINTY_FEATURE_KEYS",
    "BASE_LGBM_RAW_STATE_DRIFT_FEATURE_KEYS",
]
# Generated only after strict same-side chronological base OOF scoring.  These
# direct R3 outputs are deliberately separate from raw-store feature keys: the
# meta selector/HPO and strict-OOS residual refit must consume the same ordered
# handoff rather than replacing it with a mapped scalar or silently dropping it.
CFG["STAGE_I_REQUIRED_SAME_SIDE_BASE_OOF_HANDOFF_FEATURE_KEYS"] = [
    "r3_p_adverse",
    "r3_p_weak",
    "r3_p_clear",
    "r3_opportunity_score",
    "prequential_base_expected_net_bps",
]
CFG["STAGE_I_GROUPED_STABILITY_MDA"] = {
    "schema": "stage_i_grouped_stability_mda_v1",
    "layer_feature_key_groups": {
        "base": ["base_shared_feature_keys", "base_{side}_feature_keys"],
        "meta": [
            "meta_shared_feature_keys",
            "meta_product_feature_keys",
            "STAGE_I_M6_SHARED_UNION_META_FEATURE_KEYS",
            "STAGE_I_REQUIRED_SAME_SIDE_BASE_OOF_HANDOFF_FEATURE_KEYS",
        ],
    },
    "active_cells": [
        "base__long__R3_economic_simplex_b25",
        "base__short__R3_economic_simplex_b25",
        "meta__long__shared_exact_net_residual",
        "meta__short__shared_exact_net_residual",
    ],
    "correlation_method": "spearman",
    "correlation_threshold": 0.95,
    "topk_fraction": 0.10,
    "mda_objective": "signed_top10_trade_economics",
    "r3_selector_status": "multiclass3_pclear_minus_padverse_required",
    "label_availability_gate_hours": 13.0,
    "phantom_in_fit_threshold": True,
    "prefix_optimal_count": "smallest_within_one_se",
}


CFG["META_ORDERBOOK_WALL_FEATURE_KEYS"] = [
    "obw_wall_skew_book_r005",
    "obw_wall_skew_vol_r005",
    "obw_wall_concentration_skew_r005",
    "obw_wall_pressure_skew_r005",
    "obw_band_depth_skew_vol_r005",
    "obw_wall_skew_book_r010",
    "obw_wall_skew_vol_r010",
    "obw_wall_concentration_skew_r010",
    "obw_wall_pressure_skew_r010",
    "obw_band_depth_skew_vol_r010",
    "obw_wall_skew_book_r020",
    "obw_wall_skew_vol_r020",
    "obw_wall_concentration_skew_r020",
    "obw_wall_pressure_skew_r020",
    "obw_band_depth_skew_vol_r020",
    "obw_wall_skew_book_r030",
    "obw_wall_skew_vol_r030",
    "obw_wall_concentration_skew_r030",
    "obw_wall_pressure_skew_r030",
    "obw_band_depth_skew_vol_r030",
    "obw_wall_skew_book_a05",
    "obw_wall_skew_vol_a05",
    "obw_wall_pressure_skew_a05",
    "obw_band_depth_skew_vol_a05",
    "obw_wall_skew_book_a10",
    "obw_wall_skew_vol_a10",
    "obw_wall_pressure_skew_a10",
    "obw_band_depth_skew_vol_a10",
    "obw_wall_skew_book_a20",
    "obw_wall_skew_vol_a20",
    "obw_wall_pressure_skew_a20",
    "obw_band_depth_skew_vol_a20",
    "obw_wall_skew_book_a30",
    "obw_wall_skew_vol_a30",
    "obw_wall_pressure_skew_a30",
    "obw_band_depth_skew_vol_a30",
    "obw_nearest_bid_wall_to_vol",
    "obw_nearest_ask_wall_to_vol",
    "obw_nearest_wall_skew_vol",
    "obw_nearest_wall_distance_skew",
]
CFG["META_ORDERBOOK_BLOCKER_FEATURE_KEYS"] = [
    "obw_blocking_wall_to_vol_r005",
    "obw_support_wall_to_vol_r005",
    "obw_blocking_minus_support_wall_r005",
    "obw_blocking_wall_pressure_r005",
    "obw_blocking_wall_distance_r005",
    "obw_path_depth_to_target_r005",
    "obw_blocking_wall_to_vol_r010",
    "obw_support_wall_to_vol_r010",
    "obw_blocking_minus_support_wall_r010",
    "obw_blocking_wall_pressure_r010",
    "obw_blocking_wall_distance_r010",
    "obw_path_depth_to_target_r010",
    "obw_blocking_wall_to_vol_r020",
    "obw_support_wall_to_vol_r020",
    "obw_blocking_minus_support_wall_r020",
    "obw_blocking_wall_pressure_r020",
    "obw_blocking_wall_distance_r020",
    "obw_path_depth_to_target_r020",
    "obw_blocking_wall_to_vol_r030",
    "obw_support_wall_to_vol_r030",
    "obw_blocking_minus_support_wall_r030",
    "obw_blocking_wall_pressure_r030",
    "obw_blocking_wall_distance_r030",
    "obw_path_depth_to_target_r030",
    "obw_blocking_wall_to_vol_a05",
    "obw_support_wall_to_vol_a05",
    "obw_blocking_minus_support_wall_a05",
    "obw_blocking_wall_pressure_a05",
    "obw_blocking_wall_distance_a05",
    "obw_path_depth_to_target_a05",
    "obw_blocking_wall_to_vol_a10",
    "obw_support_wall_to_vol_a10",
    "obw_blocking_minus_support_wall_a10",
    "obw_blocking_wall_pressure_a10",
    "obw_blocking_wall_distance_a10",
    "obw_path_depth_to_target_a10",
    "obw_blocking_wall_to_vol_a20",
    "obw_support_wall_to_vol_a20",
    "obw_blocking_minus_support_wall_a20",
    "obw_blocking_wall_pressure_a20",
    "obw_blocking_wall_distance_a20",
    "obw_path_depth_to_target_a20",
    "obw_blocking_wall_to_vol_a30",
    "obw_support_wall_to_vol_a30",
    "obw_blocking_minus_support_wall_a30",
    "obw_blocking_wall_pressure_a30",
    "obw_blocking_wall_distance_a30",
    "obw_path_depth_to_target_a30",
]
ORDERBOOK_META_FEATURE_KEYS = list(dict.fromkeys(ORDERBOOK_META_FEATURE_KEYS))
ORDERBOOK_FEATURE_KEYS = sorted(
    set(ORDERBOOK_BASE_FEATURE_KEYS) | set(ORDERBOOK_META_FEATURE_KEYS)
)
CFG["ORDERBOOK_META_FEATURE_KEYS"] = ORDERBOOK_META_FEATURE_KEYS
CFG["ORDERBOOK_FEATURE_KEYS"] = ORDERBOOK_FEATURE_KEYS


CFG["META_CROSS_SECTIONAL_REGIME_KEYS"] = [
    "asset_minus_universe_median_ret_4h",
    "asset_minus_universe_median_ret_24h",
    "asset_minus_universe_median_ret_48h",
    "asset_mom_minus_basket_mom_4h",
    "resid_ret_vs_btceth_4h",
    "rv_rel_universe",
    "cs_dispersion_ret_4h",
    "cs_dispersion_ret_24h",
    "pct_assets_up_1h",
    "pct_assets_up_4h",
    "pct_assets_up_24h",
    "pct_assets_above_ema_fast",
    "avg_pair_corr_24h",
    "corr_concentration_24h",
    "btc_ret_4h_pct",
    "btc_ret_24h_pct",
    "btc_ret_48h_pct",
    "btc_rv_ratio_1h24h_pct",
    "btc_rv_ratio_4h24h_pct",
    "eth_rv_ratio_1h24h_pct",
    "eth_rv_ratio_4h24h_pct",
    "pct_assets_above_vwap",
    "cs_ret_dispersion_4h_pct",
    "cs_ret_dispersion_24h_pct",
    "asset_ret_vs_universe_4h",
    "asset_ret_vs_universe_24h",
    "asset_ret_vs_universe_48h",
    "median_rvol_z",
    "pct_assets_high_rvol",
    "median_spread_bps",
    "pct_assets_wide_spread",
    "median_volume_z",
    *NEGATIVE_RESIDUAL_META_FEATURE_KEYS,
]
CFG["meta_shared_feature_keys"] += ["META_CROSS_SECTIONAL_REGIME_KEYS"]


CFG["META_RECENT_EFFECTIVENESS_FEATURE_KEYS"] = [
    "recent_global_rolling_ic_2d",
    "recent_global_rolling_ic_5d",
    "recent_global_rolling_ic_15d",
    "recent_global_confidence_surprise_2d",
    "recent_global_confidence_surprise_5d",
    "recent_global_confidence_surprise_15d",
    "recent_global_model_ece_5d",
    "recent_global_model_ece_15d",
    "recent_global_top15_calibration_error_5d",
    "recent_global_abs_top15_calibration_error_5d",
    "recent_global_top15_hit_rate_5d",
    "recent_side_horizon_rolling_ic_5d",
    "recent_side_horizon_rolling_ic_15d",
    "recent_side_horizon_model_ece_5d",
    "recent_side_horizon_top15_hit_rate_5d",
    "recent_bucket_rolling_ic_5d",
    "recent_bucket_top15_hit_rate_5d",
    "recent_bucket_n_top15_5d",
    "recent_regime_rolling_ic_5d",
    "recent_regime_model_ece_5d",
    "recent_regime_top15_hit_rate_5d",
]
CFG["meta_shared_feature_keys"] += ["META_RECENT_EFFECTIVENESS_FEATURE_KEYS"]


CFG["META_BASE_PERFORMANCE_FEATURE_KEYS"] = [
    "base_model_score",
    "base_model_score_pct",
    "base_model_margin",
    "prob_error",
    "recent_prob_error_20",
    "recent_hit_rate_20",
    "base_model_abs_error_roll20",
    "signed_prediction_error",
    "negative_log_likelihood",
    "surprise_error_z",
    "wrong_confident",
]
CFG["meta_shared_feature_keys"] += ["META_BASE_PERFORMANCE_FEATURE_KEYS"]
CFG["REGIME_ADAPTOR_MODEL_ERROR_FEATURE_KEYS"] = [
    "signed_prediction_error",
    "negative_log_likelihood",
    "surprise_error_z",
    "wrong_confident",
]


CFG["META_SELF_FEATURE_KEYS"] = [
    "recent_meta_rank_ic_5d",
    "recent_meta_rank_ic_10d",
    "recent_meta_rank_ic_30d",
    "recent_meta_ece_5d",
    "recent_meta_ece_10d",
    "recent_meta_ece_30d",
    "recent_meta_brier_5d",
    "recent_meta_brier_10d",
    "recent_meta_brier_30d",
    "recent_meta_top15_cal_error_10d",
    "recent_meta_top15_cal_error_30d",
    "recent_meta_top15_cal_error_5d",
    "recent_meta_global_top15_hit_rate_30d",
    "recent_meta_global_top15_hit_rate_10d",
    "recent_meta_global_top15_hit_rate_5d",
]
# Meta-self features are generated after the meta OOF/policy-candidate layer
# exists. They are downstream inputs for regime_adaptor/reporting, not
# same-layer meta-model inputs.


CFG["META_RECENT_DISAGREEMENT_FEATURE_KEYS"] = [
    "recent_meta_brier_3d",
    "recent_meta_brier_7d",
    "recent_meta_brier_15d",
    "recent_base_meta_disagreement_sub_mean_3d",
    "recent_base_meta_disagreement_sub_mean_7d",
    "recent_base_meta_disagreement_sub_mean_15d",
    "recent_base_meta_disagreement_abs_sub_mean_3d",
    "recent_base_meta_disagreement_abs_sub_mean_7d",
    "recent_base_meta_disagreement_abs_sub_mean_15d",
    "recent_base_meta_disagreement_ratio_mean_3d",
    "recent_base_meta_disagreement_ratio_mean_7d",
    "recent_base_meta_disagreement_ratio_mean_15d",
    "recent_base_internal_disagreement_std_mean_3d",
    "recent_base_internal_disagreement_std_mean_7d",
    "recent_base_internal_disagreement_std_mean_15d",
    "recent_base_internal_disagreement_range_mean_3d",
    "recent_base_internal_disagreement_range_mean_7d",
    "recent_base_internal_disagreement_range_mean_15d",
    "recent_base_internal_disagreement_std_max_3d",
    "recent_base_internal_disagreement_std_max_7d",
    "recent_base_internal_disagreement_std_max_15d",
    "recent_base_internal_disagreement_range_max_3d",
    "recent_base_internal_disagreement_range_max_7d",
    "recent_base_internal_disagreement_range_max_15d",
]
# Recent base/meta disagreement also depends on meta predictions, so it is
# consumed downstream instead of being requested by train_meta itself.

CFG["META_MODEL_UNCERTAINTY_FEATURE_KEYS"] = [
    "feature_drift_psi_core",
    "feature_drift_ks_core",
    "regime_centroid_similarity_train",
    "regime_centroid_similarity_train_pc0",
    "regime_centroid_similarity_train_pc1",
    "regime_centroid_similarity_train_pc2",
    "regime_centroid_similarity_train_window_mean",
    "regime_centroid_similarity_train_window_p10",
    "feature_drift_psi_core_50",
    "feature_drift_psi_core_80",
    "feature_drift_psi_bin_mean",
    "feature_drift_psi_bin_max",
    "feature_drift_ks_bin_mean",
    "feature_drift_ks_bin_max",
    "mahalanobis_mean_shift",
    "frobenius_corr_shift",
    "feature_drift_cov_shift",
    "inference_drift_score",
    "uncertainty_score",
    "rare_leaf_low_support_score",
    "contribution_drift_score",
    "pred_std_norm",
    "pred_std_robust_norm",
    "leaf_support_mean_frac",
    "leaf_support_mean_log",
    "leaf_support_median_frac",
    "leaf_support_q25_frac",
    "leaf_train_freq_mean",
    "leaf_train_freq_p90",
    "leaf_train_freq_p10",
    "leaf_train_freq_min",
    "leaf_train_freq_max",
    "leaf_train_freq_std",
    "leaf_surprisal_mean",
    "leaf_surprisal_p90",
    "leaf_surprisal_max",
    "leaf_low_freq_fraction",
    "leaf_proximity_mean",
    "leaf_proximity_p90",
    "leaf_proximity_max",
    "leaf_model_space_distance_mean",
    "leaf_model_space_distance_p10",
    "leaf_target_mean_mean",
    "leaf_target_mean_std",
    "leaf_target_mean_min",
    "leaf_target_mean_max",
    "leaf_target_std_mean",
    "leaf_target_iqr_mean",
    "leaf_target_range_mean",
    "leaf_target_abs_mean",
    "leaf_target_positive_fraction",
    "leaf_hit_rate_avg",
    "leaf_target_dispersion",
    "support_gap",
    "leaf_pred_mean_mean",
    "leaf_error_mean_mean",
    "leaf_centroid_radius_mean",
    "leaf_centroid_dist_mean",
    "leaf_centroid_dist_median",
    "leaf_centroid_dist_std",
    "leaf_centroid_dist_p90",
    "leaf_centroid_dist_cv",
    "leaf_centroid_dist_rel_mean",
    "leaf_centroid_dist_rel_std",
    "leaf_centroid_dist_norm_mean",
    "leaf_centroid_dist_norm_p90",
    "leaf_centroid_dist_norm_max",
    "reg_pred_std_robust_norm",
    "reg_pred_range_q90_q10_norm",
    "reg_leaf_support_mean_frac",
    "reg_leaf_support_mean_log",
    "reg_leaf_support_q25_frac",
    "reg_rare_leaf_low_support_score",
    "shap_archetype_id",
    "shap_archetype_is_bad",
    "shap_archetype_is_good",
    "shap_archetype_is_neutral",
    "distance_to_archetype_centroid",
    "distance_to_nearest_bad_archetype",
    "archetype_oof_bad_rate_lift",
    "distance_to_bad_archetype",
    "distance_to_good_archetype",
    "base_error_archetype_id",
    "base_error_archetype_is_bad",
    "base_error_archetype_is_good",
    "base_error_archetype_is_neutral",
    "base_error_distance_to_archetype_centroid",
    "base_error_distance_to_nearest_bad_archetype",
    "base_error_archetype_oof_bad_rate_lift",
    "base_error_distance_to_bad_archetype",
    "base_error_distance_to_good_archetype",
]


def _meta_same_layer_safe_uncertainty_key(name: str) -> bool:
    """Return true for model-context fields known before the meta head is fit."""
    key = str(name)
    if key.startswith(
        (
            "leaf_",
            "reg_leaf_",
            "shap_archetype_",
            "base_error_",
            "distance_to_archetype_",
            "distance_to_bad_archetype",
            "distance_to_good_archetype",
            "distance_to_nearest_bad_archetype",
        )
    ):
        return False
    if key in {
        "archetype_oof_bad_rate_lift",
        "support_gap",
    }:
        return False
    return True


CFG["META_MODEL_UNCERTAINTY_META_INPUT_FEATURE_KEYS"] = [
    key
    for key in CFG["META_MODEL_UNCERTAINTY_FEATURE_KEYS"]
    if _meta_same_layer_safe_uncertainty_key(key)
]
CFG["meta_shared_feature_keys"] += ["META_MODEL_UNCERTAINTY_META_INPUT_FEATURE_KEYS"]


def _base_lgbm_meta_input_uncertainty_key(name: str) -> bool:
    """Return true for base LGBM diagnostics exported by the base OOF contract."""
    key = str(name)
    if key.startswith("shap_archetype_"):
        return False
    if key.startswith("distance_to_archetype_") or key in {
        "archetype_oof_bad_rate_lift",
        "distance_to_bad_archetype",
        "distance_to_good_archetype",
        "distance_to_nearest_bad_archetype",
    }:
        return False
    if key in {
        "leaf_centroid_radius_mean",
        "leaf_centroid_dist_std",
        "leaf_centroid_dist_p90",
        "leaf_centroid_dist_norm_mean",
        "leaf_centroid_dist_norm_p90",
        "leaf_centroid_dist_norm_max",
    }:
        return False
    return True


CFG["BASE_LGBM_META_UNCERTAINTY_FEATURE_KEYS"] = [
    f"base_lgbm_{key}"
    for key in CFG["META_MODEL_UNCERTAINTY_FEATURE_KEYS"]
    if _base_lgbm_meta_input_uncertainty_key(key)
]
CFG["meta_shared_feature_keys"] += ["BASE_LGBM_META_UNCERTAINTY_FEATURE_KEYS"]
CFG["BASE_LGBM_RAW_STATE_DRIFT_FEATURE_KEYS"] = [
    f"base_lgbm_{key}"
    for key in RAW_STATE_DIAGNOSTIC_FEATURE_NAMES
    if key not in RAW_STATE_SVD_SUMMARY_FEATURE_NAMES
]
CFG["meta_shared_feature_keys"] += ["BASE_LGBM_RAW_STATE_DRIFT_FEATURE_KEYS"]
CFG["BASE_LGBM_PREDICTIVE_ATLAS_FEATURE_KEYS"] = [
    "base_lgbm_predictive_atlas_ic",
    "base_lgbm_predictive_atlas_rank_ic",
    "base_lgbm_predictive_atlas_hit_rate",
    "base_lgbm_predictive_atlas_expected_hit_rate",
    "base_lgbm_predictive_atlas_hit_rate_surprise",
    "base_lgbm_predictive_atlas_hit_rate_surprise_z",
    "base_lgbm_predictive_atlas_support_n",
    "base_lgbm_predictive_atlas_effective_n",
    "base_lgbm_predictive_atlas_score_mean",
    "base_lgbm_predictive_atlas_score_std",
    "base_lgbm_predictive_atlas_support_quality",
]
CFG["meta_shared_feature_keys"] += ["BASE_LGBM_PREDICTIVE_ATLAS_FEATURE_KEYS"]
CFG["META_LGBM_PREDICTIVE_ATLAS_FEATURE_KEYS"] = [
    str(key).replace("base_lgbm_", "meta_lgbm_", 1)
    for key in CFG["BASE_LGBM_PREDICTIVE_ATLAS_FEATURE_KEYS"]
]
CFG["candidate_drift_denoising_ae_enabled"] = True
CFG["candidate_drift_denoising_ae_max_iter"] = 80
CFG["LGBM_AE_GMM_FEATURE_KEYS"] = list(AE_GMM_FEATURE_COLUMNS)
CFG["RESIDUAL_EVENT_AEGMM_META_FEATURE_KEYS"] = list(
    dict.fromkeys(
        [*residual_event_feature_names(), *residual_event_market_feature_names()]
    )
)
CFG["meta_shared_feature_keys"] += ["RESIDUAL_EVENT_AEGMM_META_FEATURE_KEYS"]
# Stage-II conversion-archetype values are generated only after the strict
# base-OOF handoff: side-local realised-path discovery -> causal soft
# memberships -> train-only residual prior.  They are meta-only context for a
# shared residual expert, never raw-store features and never base inputs.
CFG["STAGE_II_META_CONVERSION_ARCHETYPE_FEATURE_KEYS"] = stage_ii_feature_names(4)
CFG["meta_shared_feature_keys"] += ["STAGE_II_META_CONVERSION_ARCHETYPE_FEATURE_KEYS"]
CFG["BASE_LGBM_AE_GMM_FEATURE_KEYS"] = [
    f"base_lgbm_{key}" for key in CFG["LGBM_AE_GMM_FEATURE_KEYS"]
]
CFG["META_LGBM_AE_GMM_FEATURE_KEYS"] = [
    f"meta_lgbm_{key}" for key in CFG["LGBM_AE_GMM_FEATURE_KEYS"]
]
CFG["meta_shared_feature_keys"] += ["BASE_LGBM_AE_GMM_FEATURE_KEYS"]


# =============================================================================
# Rolling RegimeAdaptor config (next-few-days bad-regime detector)
# =============================================================================
REGIME_ADAPTOR_BASE_FEATURE_KEYS = [
    "rv_24h",
    "rv1_rv24",
    "rv4_rv24",
    "signed_adx",
    "dist_ema_fast",
    "dist_ema_slow",
    "dist_vwap",
    "prior_day_low",
    "prior_day_high",
    "rvol_z",
    "asset_volume_30d",
    "is_weekend",
    "asset_atr_30d",
    "ebm_unc_logodds_var",
    "ebm_unc_pi_width",
    "ebm_unc_entropy_mean",
    "ebm_unc_entropy_std",
    "ebm_unc_conflict_norm",
    "ebm_unc_proximity_min",
    "ebm_unc_support_mean",
    "ebm_unc_support_min",
    "ebm_unc_concentration",
    "ebm_unc_sign_ratio",
    "ebm_unc_interaction_share",
    "ebm_unc_gap50rel",
    "ebm_unc_support_adjusted_uncertainty",
    "ebm_unc_uncertainty_weight",
    "ebm_unc_friction_weight",
    "regime_centroid_similarity_train",
    "regime_centroid_similarity_train_pc0",
    "regime_centroid_similarity_train_pc1",
    "regime_centroid_similarity_train_pc2",
    "regime_centroid_similarity_train_window_mean",
    "regime_centroid_similarity_train_window_p10",
    "feature_drift_psi_core",
    "feature_drift_ks_core",
    "feature_drift_psi_core_50",
    "feature_drift_psi_core_80",
    "feature_drift_psi_bin_mean",
    "feature_drift_psi_bin_max",
    "feature_drift_ks_bin_mean",
    "feature_drift_ks_bin_max",
    "mahalanobis_mean_shift",
    "frobenius_corr_shift",
    "feature_drift_cov_shift",
    "inference_drift_score",
    "uncertainty_score",
    "rare_leaf_low_support_score",
    "contribution_drift_score",
]
REGIME_ADAPTOR_BASE_FEATURE_KEYS += ROLLING_ALPHA_FEATURE_KEYS
REGIME_ADAPTOR_BASE_FEATURE_KEYS += CURRENT_REGIME_AE_FEATURE_KEYS
REGIME_ADAPTOR_BASE_FEATURE_KEYS += CFG["REGIME_ADAPTOR_MODEL_ERROR_FEATURE_KEYS"]
REGIME_ADAPTOR_ARCHETYPE_FEATURE_KEYS = (
    [
        "contrib_abs_sum",
        "contrib_l2_norm",
        "contrib_entropy",
        "top_1_contrib_abs",
        "top_3_contrib_abs_sum",
        "positive_contrib_sum",
        "negative_contrib_sum",
    ]
    + [f"archetype_contrib_svd_{_i:02d}" for _i in range(16)]
    + [f"raw_state_svd_{_i:02d}" for _i in range(16)]
    + [
        "raw_state_svd_mean",
        "raw_state_svd_std",
        "raw_state_mahalanobis",
        "raw_state_knn_distance",
        "raw_state_min_cluster_distance",
        "raw_state_reconstruction_error",
        "raw_state_transition_norm",
        "raw_state_transition_mahalanobis",
        "state_log_likelihood",
        "state_tod_mahalanobis",
        "raw_state_psi_mean",
        "raw_state_psi_max",
        "raw_state_ks_mean",
        "raw_state_ks_max",
        "raw_state_svd_psi_mean",
        "raw_state_svd_psi_max",
        "raw_state_svd_ks_mean",
        "raw_state_svd_ks_max",
        "base_error_archetype_id",
        "base_error_archetype_is_bad",
        "base_error_archetype_is_good",
        "base_error_archetype_is_neutral",
        "base_error_distance_to_archetype_centroid",
        "base_error_distance_to_nearest_bad_archetype",
        "base_error_archetype_oof_bad_rate_lift",
        "base_error_distance_to_bad_archetype",
        "base_error_distance_to_good_archetype",
    ]
)
REGIME_ADAPTOR_BASE_FEATURE_KEYS += REGIME_ADAPTOR_ARCHETYPE_FEATURE_KEYS

REGIME_ADAPTOR_LGBM_INTERNAL_TOP25_METRIC_KEYS = (
    "lgbm_prob",
    "lgbm_raw_score",
    "prob_std",
    "raw_score_std",
    "margin_from_neutral",
    "prob_uncertainty",
    "entropy",
    "variance_proxy",
    "rank_pct",
    "score_margin_top10",
    "score_margin_top20",
    "score_margin_top30",
    "leaf_count_p10",
    "rare_leaf_fraction",
    "leaf_proximity_mean",
    "leaf_model_space_distance_mean",
    "leaf_train_freq_p10",
    "leaf_surprisal_mean",
    "leaf_low_freq_fraction",
    "leaf_hit_rate_avg",
    "leaf_target_dispersion",
    "support_gap",
    "leaf_target_std_mean",
    "leaf_target_iqr_mean",
    "leaf_centroid_dist_norm_mean",
    "leaf_centroid_dist_norm_p90",
    "contrib_entropy",
    "contrib_abs_sum",
    "score_path_std",
    "rank_bin_net_ret_oof",
)
REGIME_ADAPTOR_LGBM_INTERNAL_FEATURE_KEYS = [
    f"{prefix}_{key}"
    for prefix in ("base_lgbm", "meta_lgbm")
    for key in REGIME_ADAPTOR_LGBM_INTERNAL_TOP25_METRIC_KEYS
]
REGIME_ADAPTOR_BASE_FEATURE_KEYS += REGIME_ADAPTOR_LGBM_INTERNAL_FEATURE_KEYS

REGIME_ADAPTOR_META_LGBM_DRIFT_UNCERTAINTY_KEYS = tuple(
    dict.fromkeys(
        list(MODEL_DRIFT_FEATURE_KEYS)
        + [
            "feature_drift_psi_core",
            "feature_drift_ks_core",
        ]
    )
)
REGIME_ADAPTOR_META_LGBM_DRIFT_UNCERTAINTY_FEATURE_KEYS = [
    f"meta_lgbm_{key}" for key in REGIME_ADAPTOR_META_LGBM_DRIFT_UNCERTAINTY_KEYS
]
REGIME_ADAPTOR_BASE_FEATURE_KEYS += (
    REGIME_ADAPTOR_META_LGBM_DRIFT_UNCERTAINTY_FEATURE_KEYS
)
REGIME_ADAPTOR_META_LGBM_PREDICTIVE_ATLAS_FEATURE_KEYS = list(
    CFG["META_LGBM_PREDICTIVE_ATLAS_FEATURE_KEYS"]
)
REGIME_ADAPTOR_BASE_FEATURE_KEYS += (
    REGIME_ADAPTOR_META_LGBM_PREDICTIVE_ATLAS_FEATURE_KEYS
)
REGIME_ADAPTOR_META_LGBM_AE_GMM_FEATURE_KEYS = list(
    CFG["META_LGBM_AE_GMM_FEATURE_KEYS"]
)
REGIME_ADAPTOR_BASE_FEATURE_KEYS += REGIME_ADAPTOR_META_LGBM_AE_GMM_FEATURE_KEYS
REGIME_ADAPTOR_BASE_FEATURE_KEYS += CFG["META_SELF_FEATURE_KEYS"]
REGIME_ADAPTOR_BASE_FEATURE_KEYS += CFG["META_RECENT_DISAGREEMENT_FEATURE_KEYS"]

try:
    from extreme_price_movements.drift_monitoring import drift_regime_feature_names

    REGIME_ADAPTOR_DRIFT_MONITORING_FEATURE_KEYS = [
        name
        for name in drift_regime_feature_names()
        if "_all_score_" not in str(name) and not str(name).endswith(("_10d", "_14d"))
    ]
except Exception:
    REGIME_ADAPTOR_DRIFT_MONITORING_FEATURE_KEYS = []
REGIME_ADAPTOR_BASE_FEATURE_KEYS += REGIME_ADAPTOR_DRIFT_MONITORING_FEATURE_KEYS

try:
    from extreme_price_movements.candidate_drift_calibration import (
        CANDIDATE_DRIFT_FEATURE_COLUMNS,
    )
    from extreme_price_movements.candidate_drift_calibration import (
        DRIFT_SOURCE_COLUMNS as CANDIDATE_DRIFT_SOURCE_COLUMNS,
    )

    REGIME_ADAPTOR_CANDIDATE_DRIFT_FEATURE_KEYS = list(
        CANDIDATE_DRIFT_FEATURE_COLUMNS
    ) + [f"{key}_pct" for key in CANDIDATE_DRIFT_SOURCE_COLUMNS]
except Exception:
    REGIME_ADAPTOR_CANDIDATE_DRIFT_FEATURE_KEYS = []
REGIME_ADAPTOR_BASE_FEATURE_KEYS += REGIME_ADAPTOR_CANDIDATE_DRIFT_FEATURE_KEYS

REGIME_ADAPTOR_GLOBAL_FEATURE_KEYS = [
    "market_breadth_24h",
    "market_breadth_7d",
    "market_breadth_15d",
    "mkt_ret_eq_5d",
    "mkt_ret_eq_10d",
    "mkt_ret_eq_15d",
    "mkt_ret_eq_30d",
    "market_index_slope_5d",
    "market_index_slope_10d",
    "market_index_slope_15d",
    "market_index_slope_30d",
    "market_breadth_5d",
    "market_breadth_10d",
    "market_breadth_30d",
    "pct_assets_positive_return_5d",
    "pct_assets_positive_return_10d",
    "pct_assets_positive_return_15d",
    "pct_assets_positive_return_30d",
    "pct_assets_above_5d_ema",
    "pct_assets_above_10d_ema",
    "pct_assets_above_15d_ema",
    "pct_assets_above_30d_ema",
    "cross_asset_return_dispersion_24h",
    "cross_asset_return_dispersion_5d",
    "cross_asset_return_dispersion_10d",
    "cross_asset_return_dispersion_15d",
    "cross_asset_return_dispersion_30d",
    "cross_sectional_return_dispersion_5d",
    "cross_sectional_return_dispersion_10d",
    "cross_sectional_return_dispersion_15d",
    "cross_sectional_return_dispersion_30d",
    "cross_asset_return_dispersion_7d",
    "cross_asset_vol_dispersion_24h",
    "cross_asset_vol_dispersion_7d",
    "cross_asset_vol_dispersion_15d",
    "median_asset_rv_24h",
    "median_asset_rv_7d",
    "top_decile_asset_rv_24h",
    "top_decile_asset_rv_7d",
    "cross_asset_correlation_7d",
    "cross_asset_correlation_30d",
    "corr_asset_market_return_24h_20d",
    "beta_asset_market_20d",
    "avg_pairwise_corr_universe_20d",
    "btc_eth_trend_proxy",
    "btc_eth_vol_proxy",
    "funding_rate_cross_asset_dispersion",
    "global_ebm_unc_dispersion_mean_7d",
    "global_ebm_conflict_mean_7d",
    "global_ebm_support_risk_mean_7d",
    *MARKET_OI_REGIME_FEATURE_KEYS,
    *PRICE_OI_STATE_FEATURE_KEYS,
    *MARKET_FUNDING_REGIME_FEATURE_KEYS,
    *FUNDING_PRICE_OI_INTERACTION_FEATURE_KEYS,
    *OHLCV_BREADTH_FEATURE_KEYS,
    *OHLCV_CRASH_PHASE_FEATURE_KEYS,
    *MARKET_SYNCHRONIZATION_FEATURE_KEYS,
    *CRASH_LIFECYCLE_MARKET_FEATURE_KEYS,
    *LIQUIDATION_STATE_SCORE_FEATURE_KEYS,
    *NEGATIVE_RESIDUAL_META_FEATURE_KEYS,
]

REGIME_ADAPTOR_ASSET_FEATURE_KEYS = [
    "realized_vol_5d",
    "realized_vol_10d",
    "realized_vol_15d",
    "realized_vol_30d",
    "range_atr_regime_5d",
    "range_atr_regime_10d",
    "range_atr_regime_15d",
    "range_atr_regime_30d",
    "vol_zscore_5d",
    "vol_zscore_10d",
    "vol_zscore_15d",
    "vol_zscore_30d",
    "vol_of_vol_5d",
    "vol_of_vol_10d",
    "vol_of_vol_15d",
    "vol_of_vol_30d",
    "quote_volume_realized_vol_5d",
    "quote_volume_realized_vol_10d",
    "quote_volume_realized_vol_15d",
    "quote_volume_realized_vol_30d",
    "aggregate_relative_volume_5d",
    "aggregate_relative_volume_10d",
    "aggregate_relative_volume_15d",
    "aggregate_relative_volume_30d",
    "quote_volume_zscore_5d",
    "quote_volume_zscore_10d",
    "quote_volume_zscore_15d",
    "quote_volume_zscore_30d",
    "price_ema_gap_5d",
    "price_ema_gap_10d",
    "price_ema_gap_15d",
    "price_ema_gap_30d",
    "ema_slope_atr_5d",
    "ema_slope_atr_10d",
    "ema_slope_atr_15d",
    "ema_slope_atr_30d",
    "ema_stack_score_5d",
    "ema_stack_score_10d",
    "ema_stack_score_15d",
    "ema_stack_score_30d",
    "trend_consistency_5d",
    "trend_consistency_10d",
    "trend_consistency_15d",
    "trend_consistency_30d",
    "distance_to_5d_high",
    "distance_to_10d_high",
    "distance_to_15d_high",
    "distance_to_30d_high",
    "distance_to_5d_low",
    "distance_to_10d_low",
    "distance_to_15d_low",
    "distance_to_30d_low",
    "distance_to_5d_ema",
    "distance_to_10d_ema",
    "distance_to_15d_ema",
    "distance_to_30d_ema",
    "drawdown_from_5d_high_atr",
    "drawdown_from_10d_high_atr",
    "drawdown_from_15d_high_atr",
    "drawdown_from_30d_high_atr",
    "percentile_rank_in_recent_range_5d",
    "percentile_rank_in_recent_range_10d",
    "percentile_rank_in_recent_range_15d",
    "percentile_rank_in_recent_range_30d",
    "asset_rv_mean_24h",
    "asset_rv_mean_96h",
    "asset_rv_mean_7d",
    "asset_rv_mean_15d",
    "asset_rv_p90_7d",
    "asset_rv_trend_24h_to_7d",
    "asset_rvol_mean_24h",
    "asset_rvol_mean_7d",
    "asset_rvol_mean_15d",
    "asset_atr_30d",
    "asset_volume_30d",
    "asset_funding_rate_mean_3d",
    "asset_funding_rate_mean_7d",
    "asset_funding_rate_mean_15d",
    "asset_funding_rate_abs_mean_7d",
    "asset_funding_z",
    "asset_funding_side_alignment",
    "asset_funding_trend_alignment",
    "asset_ebm_unc_dispersion_mean_3d",
    "asset_ebm_unc_dispersion_mean_7d",
    "asset_ebm_unc_dispersion_mean_15d",
    "asset_ebm_conflict_mean_3d",
    "asset_ebm_conflict_mean_7d",
    "asset_ebm_conflict_mean_15d",
    "asset_ebm_support_risk_mean_3d",
    "asset_ebm_support_risk_mean_7d",
    "asset_ebm_support_risk_mean_15d",
    "asset_ebm_brittleness_mean_3d",
    "asset_ebm_brittleness_mean_7d",
    "asset_ebm_brittleness_mean_15d",
]

REGIME_ADAPTOR_STRATEGY_ASSET_FEATURE_KEYS = [
    "prior_1d_strategy_asset_pnl",
    "prior_3d_strategy_asset_pnl",
    "prior_5d_strategy_asset_pnl",
    "prior_7d_strategy_asset_pnl",
    "prior_15d_strategy_asset_pnl",
    "prior_30d_strategy_asset_pnl",
    "prior_3d_strategy_asset_maxDD",
    "prior_5d_strategy_asset_maxDD",
    "prior_7d_strategy_asset_maxDD",
    "prior_15d_strategy_asset_maxDD",
    "prior_30d_strategy_asset_maxDD",
    "prior_3d_strategy_asset_trade_count",
    "prior_5d_strategy_asset_trade_count",
    "prior_7d_strategy_asset_trade_count",
    "prior_15d_strategy_asset_trade_count",
    "prior_30d_strategy_asset_trade_count",
    "prior_3d_expected_hit_rate",
    "prior_5d_expected_hit_rate",
    "prior_7d_expected_hit_rate",
    "prior_15d_expected_hit_rate",
    "prior_30d_expected_hit_rate",
    "prior_3d_hit_rate_surprise_z",
    "prior_5d_hit_rate_surprise_z",
    "prior_7d_hit_rate_surprise_z",
    "prior_15d_hit_rate_surprise_z",
    "prior_30d_hit_rate_surprise_z",
]

REGIME_ADAPTOR_EBM_CONSOLIDATED_FEATURE_KEYS = [
    "global_ebm_unc_dispersion_mean_7d",
    "global_ebm_conflict_mean_7d",
    "global_ebm_support_risk_mean_7d",
    "asset_ebm_unc_dispersion_mean_3d",
    "asset_ebm_unc_dispersion_mean_7d",
    "asset_ebm_unc_dispersion_mean_15d",
    "asset_ebm_conflict_mean_3d",
    "asset_ebm_conflict_mean_7d",
    "asset_ebm_conflict_mean_15d",
    "asset_ebm_support_risk_mean_3d",
    "asset_ebm_support_risk_mean_7d",
    "asset_ebm_support_risk_mean_15d",
    "asset_ebm_brittleness_mean_3d",
    "asset_ebm_brittleness_mean_7d",
    "asset_ebm_brittleness_mean_15d",
]
_REGIME_ADAPTOR_REMOVED_EBM_CONSOLIDATED_FEATURE_KEYS = set(
    REGIME_ADAPTOR_EBM_CONSOLIDATED_FEATURE_KEYS
)
REGIME_ADAPTOR_GLOBAL_FEATURE_KEYS = [
    key
    for key in REGIME_ADAPTOR_GLOBAL_FEATURE_KEYS
    if key not in _REGIME_ADAPTOR_REMOVED_EBM_CONSOLIDATED_FEATURE_KEYS
]
REGIME_ADAPTOR_ASSET_FEATURE_KEYS = [
    key
    for key in REGIME_ADAPTOR_ASSET_FEATURE_KEYS
    if key not in _REGIME_ADAPTOR_REMOVED_EBM_CONSOLIDATED_FEATURE_KEYS
]
REGIME_ADAPTOR_EBM_CONSOLIDATED_FEATURE_KEYS = []

_REGIME_ADAPTOR_GLOBAL_FEATURE_ALLOWLIST = {
    "market_breadth_24h",
    "market_breadth_7d",
    "mkt_ret_eq_5d",
    "mkt_ret_eq_30d",
    "market_index_slope_5d",
    "market_index_slope_30d",
    "market_breadth_5d",
    "market_breadth_30d",
    "pct_assets_positive_return_5d",
    "pct_assets_positive_return_30d",
    "pct_assets_above_5d_ema",
    "pct_assets_above_30d_ema",
    "cross_asset_return_dispersion_24h",
    "cross_asset_return_dispersion_7d",
    "cross_asset_return_dispersion_30d",
    "cross_asset_vol_dispersion_24h",
    "cross_asset_vol_dispersion_7d",
    "median_asset_rv_24h",
    "median_asset_rv_7d",
    "top_decile_asset_rv_24h",
    "top_decile_asset_rv_7d",
    "cross_asset_correlation_7d",
    "cross_asset_correlation_30d",
    "corr_asset_market_return_24h_20d",
    "beta_asset_market_20d",
    "avg_pairwise_corr_universe_20d",
    "btc_eth_trend_proxy",
    "btc_eth_vol_proxy",
    "funding_rate_cross_asset_dispersion",
    *MARKET_OI_REGIME_FEATURE_KEYS,
    *PRICE_OI_STATE_FEATURE_KEYS,
    *MARKET_FUNDING_REGIME_FEATURE_KEYS,
    *FUNDING_PRICE_OI_INTERACTION_FEATURE_KEYS,
    *OHLCV_BREADTH_FEATURE_KEYS,
    *OHLCV_CRASH_PHASE_FEATURE_KEYS,
    *MARKET_SYNCHRONIZATION_FEATURE_KEYS,
    *CRASH_LIFECYCLE_MARKET_FEATURE_KEYS,
    *LIQUIDATION_STATE_SCORE_FEATURE_KEYS,
}
REGIME_ADAPTOR_GLOBAL_FEATURE_KEYS = [
    key
    for key in REGIME_ADAPTOR_GLOBAL_FEATURE_KEYS
    if key in _REGIME_ADAPTOR_GLOBAL_FEATURE_ALLOWLIST
]
REGIME_ADAPTOR_ASSET_FEATURE_KEYS = [
    key
    for key in REGIME_ADAPTOR_ASSET_FEATURE_KEYS
    if "_10d" not in key and "_15d" not in key
]


REGIME_ADAPTOR_CROSS_ASSET_FEATURE_KEYS = REGIME_ADAPTOR_GLOBAL_FEATURE_KEYS
REGIME_ADAPTOR_FUNDING_FEATURE_KEYS = [
    "asset_funding_rate_mean_3d",
    "asset_funding_rate_mean_7d",
    "asset_funding_rate_mean_15d",
    "asset_funding_rate_abs_mean_7d",
    "asset_funding_z",
    "asset_funding_side_alignment",
    "asset_funding_trend_alignment",
    "funding_rate_cross_asset_dispersion",
]
REGIME_ADAPTOR_ORDERBOOK_FEATURE_KEYS = [
    "asset_spread_decile",
    "asset_p75_spread_bps",
    "asset_p75_spread_decile",
    "spread_to_expected_move",
    "asset_spread_proxy_p90_24h",
    "asset_spread_proxy_p90_96h",
    "asset_spread_proxy_p90_7d",
    "asset_spread_proxy_p90_15d",
    "asset_volume_depth_risk_p90_24h",
    "asset_volume_depth_risk_p90_96h",
    "asset_volume_depth_risk_p90_7d",
    "asset_volume_depth_risk_p90_15d",
    "asset_orderbook_imbalance_abs_mean_24h",
    "asset_orderbook_imbalance_abs_mean_96h",
    "asset_orderbook_imbalance_abs_mean_7d",
    "asset_orderbook_imbalance_abs_mean_15d",
    "asset_liquidity_stress_score_7d",
    "global_liquidity_stress_score_7d",
]
REGIME_ADAPTOR_ROLLING_PRIOR_FEATURE_KEYS = REGIME_ADAPTOR_STRATEGY_ASSET_FEATURE_KEYS
REGIME_ADAPTOR_FEATURE_ORDER = (
    REGIME_ADAPTOR_BASE_FEATURE_KEYS
    + REGIME_ADAPTOR_GLOBAL_FEATURE_KEYS
    + REGIME_ADAPTOR_ASSET_FEATURE_KEYS
    + REGIME_ADAPTOR_ORDERBOOK_FEATURE_KEYS
    + REGIME_ADAPTOR_STRATEGY_ASSET_FEATURE_KEYS
    + REGIME_ADAPTOR_EBM_CONSOLIDATED_FEATURE_KEYS
)
CFG["REGIME_ADAPTOR_BASE_FEATURE_KEYS"] = REGIME_ADAPTOR_BASE_FEATURE_KEYS
CFG["REGIME_ADAPTOR_FEATURE_ORDER"] = REGIME_ADAPTOR_FEATURE_ORDER

REGIME_ADAPTOR_OBJECTIVE_WEIGHTS = {
    "pnl_ratio": 0.30,
    "sortino_ratio": 0.20,
    "dd_ratio": 0.30,
    "period_std_ratio": 0.10,
    "worst_loss_ratio": 0.10,
}
REGIME_ADAPTOR_RATIO_CLIPS = {
    "pnl_ratio": [0.70, 1.50],
    "sortino_ratio": [0.50, 2.00],
    "dd_ratio": [0.50, 2.00],
    "period_std_ratio": [0.50, 1.75],
    "worst_loss_ratio": [0.50, 1.75],
}


class RegimeAdaptorKey:
    """Canonical config keys for the meta-correctness RegimeAdaptor layer."""

    REGIME_ADAPTOR_ENABLED = "regime_adaptor.enabled"
    REGIME_ADAPTOR_GLOBAL_FEATURE_KEYS = "regime_adaptor.global_feature_keys"
    REGIME_ADAPTOR_ASSET_FEATURE_KEYS = "regime_adaptor.asset_feature_keys"
    REGIME_ADAPTOR_STRATEGY_ASSET_FEATURE_KEYS = (
        "regime_adaptor.strategy_asset_feature_keys"
    )
    REGIME_ADAPTOR_CROSS_ASSET_FEATURE_KEYS = "regime_adaptor.cross_asset_feature_keys"
    REGIME_ADAPTOR_FUNDING_FEATURE_KEYS = "regime_adaptor.funding_feature_keys"
    REGIME_ADAPTOR_ORDERBOOK_FEATURE_KEYS = "regime_adaptor.orderbook_feature_keys"
    REGIME_ADAPTOR_EBM_CONSOLIDATED_FEATURE_KEYS = (
        "regime_adaptor.ebm_consolidated_feature_keys"
    )
    REGIME_ADAPTOR_ROLLING_PRIOR_FEATURE_KEYS = (
        "regime_adaptor.rolling_prior_feature_keys"
    )
    REGIME_ADAPTOR_MODEL_TYPE = "regime_adaptor.model_type"
    REGIME_ADAPTOR_OBJECTIVE_WEIGHTS = "regime_adaptor.objective_weights"
    REGIME_ADAPTOR_RATIO_CLIPS = "regime_adaptor.ratio_clips"
    REGIME_ADAPTOR_INFERENCE_INTEGRATION_MODE = (
        "regime_adaptor.inference_integration_mode"
    )
    REGIME_ADAPTOR_ARTIFACT_PATH = "regime_adaptor.artifact_path"
    REGIME_ADAPTOR_DIAGNOSTICS_PATH = "regime_adaptor.diagnostics_path"


REGIME_ADAPTOR_DEFAULT_CONFIG = {
    RegimeAdaptorKey.REGIME_ADAPTOR_ENABLED: False,
    RegimeAdaptorKey.REGIME_ADAPTOR_GLOBAL_FEATURE_KEYS: REGIME_ADAPTOR_GLOBAL_FEATURE_KEYS,
    RegimeAdaptorKey.REGIME_ADAPTOR_ASSET_FEATURE_KEYS: REGIME_ADAPTOR_ASSET_FEATURE_KEYS,
    RegimeAdaptorKey.REGIME_ADAPTOR_STRATEGY_ASSET_FEATURE_KEYS: REGIME_ADAPTOR_STRATEGY_ASSET_FEATURE_KEYS,
    RegimeAdaptorKey.REGIME_ADAPTOR_CROSS_ASSET_FEATURE_KEYS: REGIME_ADAPTOR_CROSS_ASSET_FEATURE_KEYS,
    RegimeAdaptorKey.REGIME_ADAPTOR_FUNDING_FEATURE_KEYS: REGIME_ADAPTOR_FUNDING_FEATURE_KEYS,
    RegimeAdaptorKey.REGIME_ADAPTOR_ORDERBOOK_FEATURE_KEYS: REGIME_ADAPTOR_ORDERBOOK_FEATURE_KEYS,
    RegimeAdaptorKey.REGIME_ADAPTOR_EBM_CONSOLIDATED_FEATURE_KEYS: REGIME_ADAPTOR_EBM_CONSOLIDATED_FEATURE_KEYS,
    RegimeAdaptorKey.REGIME_ADAPTOR_ROLLING_PRIOR_FEATURE_KEYS: REGIME_ADAPTOR_ROLLING_PRIOR_FEATURE_KEYS,
    RegimeAdaptorKey.REGIME_ADAPTOR_MODEL_TYPE: "meta_correctness_lgbm",
    RegimeAdaptorKey.REGIME_ADAPTOR_OBJECTIVE_WEIGHTS: REGIME_ADAPTOR_OBJECTIVE_WEIGHTS,
    RegimeAdaptorKey.REGIME_ADAPTOR_RATIO_CLIPS: REGIME_ADAPTOR_RATIO_CLIPS,
    RegimeAdaptorKey.REGIME_ADAPTOR_INFERENCE_INTEGRATION_MODE: "disabled",
    RegimeAdaptorKey.REGIME_ADAPTOR_ARTIFACT_PATH: "ridge_sizer/regime_adaptors/{strategy_id}/regime_adaptor.json",
    RegimeAdaptorKey.REGIME_ADAPTOR_DIAGNOSTICS_PATH: "ridge_sizer/regime_adaptors/{strategy_id}/diagnostics",
}


# Expose RegimeAdaptor defaults through the runtime CFG dictionary as well as constants.
CFG.update(REGIME_ADAPTOR_DEFAULT_CONFIG)

# =============================================================================
# Portable Feature Contract
# =============================================================================
# Non-portable feature families are deleted from active contracts. Cross-asset,
# basket, and source-normalized perp/funding/orderbook features remain eligible
# when their source panels are present; source-specific raw fields fail closed.
PORTABLE_SOURCE_NORMALIZED_FEATURE_KEYS = {
    # Asset state: within-symbol percentiles/z-scores, never raw identity levels.
    "asset_atr_level_pct",
    "asset_vol_level_pct",
    "asset_atr_level",
    "asset_vol_level",
    "vol_state",
    # Funding: hourly/event-normalized local distribution features.
    "funding_per_hour",
    "funding_per_hour_z",
    "funding_rank_30d",
    "fund_rate",
    "funding_rate",
    "fund_rate_ffill",
    "fund_rate_z_14d",
    "fund_rate_mom_8h",
    "fund_rate_mom_24h",
    "fund_abs_z",
    "fund_abs_z_14d",
    "fund_carry_24h",
    "fund_mom_8h",
    "fund_mom_24h",
    "fund_sign_persistence_3",
    "fund_sign_persistence_24h",
    "fund_extreme_duration_24h",
    "fund_rank_30d",
    "fund_countdown_pressure",
    "fund_hours_to_next",
    "fund_hours_since_last",
    "funding_phase_sin",
    "funding_phase_cos",
    "fund_next_event_proximity_5h",
    "fund_next_event_proximity_10h",
    "fund_pre_drift_5h",
    "fund_pre_drift_10h",
    "fund_post_reversal_5h",
    "fund_post_reversal_10h",
    "fund_ret_cond_sign_5h",
    "fund_ret_cond_sign_10h",
    "fund_payment_pressure_5h",
    "fund_payment_pressure_10h",
    "funding_z",
    "funding_abs_z",
    "funding_persistence",
    "funding_mom_2h",
    "funding_mom_4h",
    "funding_mom_8h",
    "funding_mom_w",
    "funding_up_agree",
    "asset_funding_rate_mean_3d",
    "asset_funding_rate_mean_7d",
    "asset_funding_rate_mean_15d",
    "asset_funding_rate_abs_mean_7d",
    "asset_funding_z",
    "asset_funding_trend_alignment",
    "funding_rate_cross_asset_dispersion",
    "xasset_fund_dispersion_basket",
    "xasset_fund_extreme_share_basket",
    "xasset_asset_minus_basket_fund_z",
    "xasset_asset_minus_mkt_funding",
    "xasset_btc_funding_z",
    "xasset_btc_fund_z",
    "fund_abs_z_x_ret24h_sign",
    "fund_abs_z_x_rv_24h",
    "fund_z_x_trend_strength",
    # Perp basis / OI: percent, local z-score, rank, or volume-normalized forms.
    "basis",
    "basis_frac",
    "basis_frac_z_14d",
    "basis_frac_rank_30d",
    "basis_per_atr",
    "basis_pct",
    "basis_pct_z",
    "basis_stretch",
    "basis_vol",
    "basis_mom_2h",
    "basis_mom_4h",
    "basis_mom_8h",
    "basis_mom_w",
    "basis_fund_div_z",
    "basis_funding_div",
    "basis_funding_div_2h",
    "basis_funding_div_4h",
    "basis_funding_div_8h",
    "basis_up_agree",
    "perp_index_basis",
    "perp_index_basis_z",
    "mark_index_basis",
    "mark_index_basis_z",
    "mark_perp_dislocation",
    "mark_vs_perp_bps",
    "mark_vs_index_bps",
    "perp_vs_index_bps",
    "premium_proxy_bps",
    "premium_proxy_z",
    "premium_proxy_mom_8h",
    "premium_mean_reversion_halflife_24h",
    "oi_z",
    "oi_rank",
    "oi_chg_z_2h",
    "oi_chg_z_4h",
    "oi_chg_z_8h",
    "oi_chg_2h",
    "oi_chg_4h",
    "oi_chg_8h",
    "oi_value_log_1d_robust_z",
    "oi_value_log_7d_robust_z",
    "oi_chg_2h_robust_z",
    "oi_chg_4h_robust_z",
    "oi_chg_8h_robust_z",
    "oi_vel_2h",
    "oi_vel_4h",
    "oi_vel_8h",
    "oi_rel_vol_2h",
    "oi_rel_vol_4h",
    "oi_rel_vol_8h",
    "oi_up_agree",
    "leverage_build",
    "leverage_build_score",
    "unwind",
    "unwind_score",
    "squeeze_prob",
    "ret1h_perp",
    "mom_slow",
    "mom_slow_z",
    "carry_adj_ret_5h",
    "carry_adj_ret_10h",
    "carry_adj_ret_self_z_5h",
    "carry_adj_ret_self_z_10h",
    "carry_adj_short_ret_5h",
    "carry_adj_short_ret_10h",
    "carry_adj_short_ret_self_z_5h",
    "carry_adj_short_ret_self_z_10h",
    "basis_adjusted_trend_5h",
    "basis_adjusted_trend_10h",
    "basis_adjusted_trend_self_z_5h",
    "basis_adjusted_trend_self_z_10h",
    "funding_crowded_mom_exhaustion_5h",
    "funding_crowded_mom_exhaustion_10h",
    "funding_crowded_mom_exhaustion_self_z_5h",
    "funding_crowded_mom_exhaustion_self_z_10h",
    "fund_high_neg_mom_5h",
    "fund_high_neg_mom_10h",
    "fund_high_neg_mom_self_z_5h",
    "fund_high_neg_mom_self_z_10h",
    "mark_gap_vol_5h",
    "mark_gap_vol_10h",
    "premium_expansion_speed_5h",
    "premium_expansion_speed_10h",
    "mark_trigger_dislocation_5h",
    "mark_trigger_dislocation_10h",
    "mark_trigger_dislocation_self_z_5h",
    "mark_trigger_dislocation_self_z_10h",
    "mark_trigger_risk_5h",
    "mark_trigger_risk_10h",
    # Orderbook: bps, notional-normalized, z-scored, or availability masks.
    "ob_available",
    "ob_snapshot_age_min",
    "ob_snapshot_age_sec",
    "ob_update_gap_flag",
    "ob_stale_flag",
    "ob_age_ratio",
    "ob_coverage_24h",
    "ob_spread_bps",
    "ob_spread_z_24h",
    "ob_spread_bps_z_24h",
    "ob_spread_bps_z_7d",
    "ob_mid_close_dislocation_bps",
    "ob_mid_close_dislocation_bps_z_24h",
    "ob_microprice_dev_bps",
    "ob_microprice_dev_bps_z_24h",
    "ob_l1_imbalance",
    "ob_l1_imbalance_z_24h",
    "ob_imb_l1",
    "ob_trade_flow_imbalance_1h",
    "ob_flow_qty_imbalance_1h",
    "ob_flow_notional_imbalance_1h",
    "ob_buy_notional_z_24h",
    "ob_sell_notional_z_24h",
    "ob_flow_notional_skew_z_24h",
    "ob_trade_count_z_24h",
    "ob_notional_z_24h",
    "ob_vwap_mid_gap_bps",
    "ob_mean_trade_qty_z_24h",
    "ob_notional_to_depth_l20_z_24h",
    "ob_trade_size_to_l1_depth_z_24h",
    "ob_kyle_lambda_1h",
    "ob_flow_toxicity_1h",
    "ob_liquidity_shock_z",
    "ob_imb_10bps",
    "ob_imb_25bps",
    "ob_depth_to_qv_10bps",
    "ob_depth_to_qv_25bps",
    "ob_depth_z_10bps",
    "ob_depth_z_25bps",
    "ob_depth_usd_l10",
    "ob_depth_usd_l20",
    "ob_depth_usd_l10_z",
    "ob_depth_usd_l20_z",
    "ob_top_liquidity_usd",
    "ob_depth_l10_to_qv_24h",
    "ob_depth_l20_to_qv_24h",
    "ob_top_liquidity_to_qv_24h",
    "ob_depth_l20_to_qv_z_7d",
    "ob_spread_z_x_rv_24h",
    "ob_depth_z_x_rvol_z",
    "ob_depth_to_qv_z_x_rvol_z",
    "xasset_mkt_spread_bps_z_24h",
    "xasset_mkt_depth_to_qv_z",
    "xasset_mkt_ob_stress_z_24h",
    "xasset_ob_stress_basket_z_24h",
    "xasset_ob_liquidity_divergence_z_24h",
}
for _bps in (5, 10, 25, 50, 100):
    PORTABLE_SOURCE_NORMALIZED_FEATURE_KEYS.update(
        {
            f"ob_bid_depth_{_bps}bps",
            f"ob_ask_depth_{_bps}bps",
            f"ob_depth_skew_{_bps}bps",
        }
    )
PORTABLE_SOURCE_NORMALIZED_FEATURE_KEYS.update(RESIDUAL_FEATURE_KEYS)
PORTABLE_SOURCE_NORMALIZED_FEATURE_KEYS.update(SPREAD_PROXY_FEATURE_KEYS)
PORTABLE_SOURCE_NORMALIZED_FEATURE_KEYS.update(CROSS_ASSET_FEATURE_KEYS)
PORTABLE_SOURCE_NORMALIZED_FEATURE_KEYS.update(OI_TRADING_FEATURE_KEYS)
PORTABLE_SOURCE_NORMALIZED_FEATURE_KEYS.update(MARKET_OI_REGIME_FEATURE_KEYS)
PORTABLE_SOURCE_NORMALIZED_FEATURE_KEYS.update(PRICE_OI_STATE_FEATURE_KEYS)
PORTABLE_SOURCE_NORMALIZED_FEATURE_KEYS.update(MARKET_FUNDING_REGIME_FEATURE_KEYS)
PORTABLE_SOURCE_NORMALIZED_FEATURE_KEYS.update(
    FUNDING_PRICE_OI_INTERACTION_FEATURE_KEYS
)
PORTABLE_SOURCE_NORMALIZED_FEATURE_KEYS.update(OHLCV_BREADTH_FEATURE_KEYS)
PORTABLE_SOURCE_NORMALIZED_FEATURE_KEYS.update(OHLCV_CRASH_PHASE_FEATURE_KEYS)
PORTABLE_SOURCE_NORMALIZED_FEATURE_KEYS.update(MARKET_SYNCHRONIZATION_FEATURE_KEYS)
PORTABLE_SOURCE_NORMALIZED_FEATURE_KEYS.update(OHLCV_LIFECYCLE_FEATURE_KEYS)
PORTABLE_SOURCE_NORMALIZED_FEATURE_KEYS.update(MARKET_BREADTH_LIFECYCLE_FEATURE_KEYS)
PORTABLE_SOURCE_NORMALIZED_FEATURE_KEYS.update(MARKET_SYNCHRONIZATION_ADDITION_KEYS)
PORTABLE_SOURCE_NORMALIZED_FEATURE_KEYS.update(LIQUIDATION_STATE_SCORE_FEATURE_KEYS)
PORTABLE_SOURCE_NORMALIZED_FEATURE_KEYS.update(CRASH_LIFECYCLE_NEW_FEATURE_KEYS)
PORTABLE_SOURCE_NORMALIZED_FEATURE_KEYS.update(NEGATIVE_RESIDUAL_META_FEATURE_KEYS)
PORTABLE_SOURCE_NORMALIZED_FEATURE_KEYS.update(LONG_HORIZON_PERP_META_FEATURE_KEYS)
PORTABLE_SOURCE_NORMALIZED_FEATURE_KEYS.update(VOLUME_FREE_PERP_BASE_FEATURE_KEYS)
PORTABLE_SOURCE_NORMALIZED_FEATURE_KEYS.update(VOLUME_FREE_PERP_META_FEATURE_KEYS)
PORTABLE_SOURCE_NORMALIZED_FEATURE_KEYS.update(OI_WEIGHTED_LOCATION_BASE_FEATURE_KEYS)
PORTABLE_SOURCE_NORMALIZED_FEATURE_KEYS.update(OI_WEIGHTED_LOCATION_META_FEATURE_KEYS)
PORTABLE_SOURCE_NORMALIZED_FEATURE_KEYS.difference_update(
    KRAKEN_INDEX_PREMIUM_FEATURE_KEYS
)
REFERENCE_BASIS_FEATURE_KEYS = {
    key
    for key in PORTABLE_SOURCE_NORMALIZED_FEATURE_KEYS
    if key == "basis" or key.startswith("basis_")
}
REFERENCE_BASIS_FEATURE_KEYS.update(
    {
        key
        for key in (
            PERP_FEATURE_KEYS
            + SPOT_FOR_PERPS_META_FEATURE_KEYS
            + PERP_CARRY_ALPHA_FEATURE_KEYS
            + PERP_META_PRIMARY_FEATURE_KEYS
        )
        if str(key) == "basis" or str(key).startswith("basis_")
    }
)
PORTABLE_SOURCE_NORMALIZED_FEATURE_KEYS.difference_update(REFERENCE_BASIS_FEATURE_KEYS)
SPARSE_OR_STATE_DEPENDENT_FEATURE_KEYS = set()
PORTABLE_SOURCE_NORMALIZED_FEATURE_KEYS.difference_update(
    SPARSE_OR_STATE_DEPENDENT_FEATURE_KEYS
)
DELETED_CONTRACT_FEATURE_KEYS = {
    "liq_low",
    "liq_regime",
}
PORTABLE_SOURCE_NORMALIZED_FEATURE_KEYS.difference_update(DELETED_CONTRACT_FEATURE_KEYS)
for _key in DELETED_CONTRACT_FEATURE_KEYS:
    CONTINUOUS_REGIME_FEATURES.pop(_key, None)
RIDGE_FEATURE_COLS = [
    key for key in RIDGE_FEATURE_COLS if str(key) not in DELETED_CONTRACT_FEATURE_KEYS
]
CFG["RIDGE_FEATURE_COLS"] = [
    key
    for key in CFG.get("RIDGE_FEATURE_COLS", [])
    if str(key) not in DELETED_CONTRACT_FEATURE_KEYS
]
NON_PORTABLE_FEATURE_PREFIXES = (
    "ob_",
    "obw_",
    "_obw_",
    "fund_",
    "funding_",
    "asset_funding",
    "oi_",
    "basis_",
    "mark_",
    "premium_",
    "spot_",
    "perp_",
    "recent_",
    "kf_",
    "kalman_",
    "price_state_",
    "vol_state_",
    "volume_state_",
    "s_mean_",
    "s_std_",
    "s_z_",
    "s_pct_",
    "s_bin3_",
    "s_gt",
    "reject_",
    "retest_accept_",
    "tf_qual_",
    "mr_qual_",
    "vp_",
)
NON_PORTABLE_FEATURE_EXACT = set(
    ORDERBOOK_DIAGNOSTIC_ONLY_FEATURE_KEYS
    + list(SPARSE_OR_STATE_DEPENDENT_FEATURE_KEYS)
    + [
        "basis",
        "mark_price",
        "index_price",
        "canonical_index",
        "spot_available",
        "spot_leads_perp_1h",
        "spot_perp_return_agreement_4h",
        "spot_perp_vol_ratio_24h",
        "spot_perp_volume_ratio_24h",
        "fund_countdown_pressure",
        "fund_hours_to_next",
        "fund_hours_since_last",
        "fund_next_event_proximity_5h",
        "fund_next_event_proximity_10h",
        "fund_pre_drift_5h",
        "fund_pre_drift_10h",
        "fund_post_reversal_5h",
        "fund_post_reversal_10h",
        "fund_ret_cond_sign_5h",
        "fund_ret_cond_sign_10h",
        "fund_payment_pressure_5h",
        "fund_payment_pressure_10h",
        "fund_flip_x_vol_expansion_5h",
        "fund_flip_x_vol_expansion_10h",
        "persistent_pos_funding_failed_breakout_5h",
        "persistent_pos_funding_failed_breakout_10h",
        "persistent_neg_funding_failed_breakdown_5h",
        "persistent_neg_funding_failed_breakdown_10h",
        "oi_chg_2h",
        "oi_chg_4h",
        "oi_chg_8h",
        "oi_vel_2h",
        "oi_vel_4h",
        "oi_vel_8h",
        "oi_chg_w",
        "liq_buffer_long_mark_frac",
        "liq_buffer_short_mark_frac",
        "liq_buffer_atr",
        "liq_stop_safety_long_atr",
        "liq_stop_safety_short_atr",
        "ob_l5_imbalance",
        "ob_l10_imbalance",
        "ob_l20_imbalance",
        "ob_l10_imbalance_z_24h",
        "ob_l20_imbalance_z_24h",
        "ob_l10_abs_imbalance_z_7d",
        "ob_book_pressure_l10",
        "ob_book_pressure_l10_z_24h",
        "ob_book_pressure_l10_z_7d",
        "ob_depth_ratio_l1_l20",
        "ob_depth_decay_asym_l20_z_7d",
        "ob_imb_l10",
        "ob_imb_l20",
        "ob_wimb_l10",
        "ob_wimb_l20",
        "ob_imb_chg_1",
        "ob_imb_accel_4",
        "ob_imb_near_far_delta",
        "ob_depth_decay_asym_l20",
        "ob_wall_imb_l20",
        "ob_wall_skew_l20",
        "ob_depth_usd_l10",
        "ob_depth_usd_l20",
        "ob_depth_usd_l10_z",
        "ob_depth_usd_l20_z",
        "ob_depth_usd_z_24h",
        "ob_top_liquidity_usd",
        "ob_top_liquidity_usd_z",
        "ob_bid_depth_decay_l20",
        "ob_ask_depth_decay_l20",
        "ob_abs_flow_vs_book_l20",
        "ob_abs_flow_vs_book_l20_z_24h",
        "ob_flow_vs_book_l10",
        "ob_flow_vs_book_l20",
        "ob_book_absorption_score",
        "ob_pressure_ret4h_agreement",
        "ob_pressure_volume_agreement",
        "ob_pressure_x_ret4h_sign",
        "accept_gt66",
        "reject_like",
        "tf_qual",
        "mr_qual",
        "ambig",
        "stage_tf",
        "stage_blowoff",
        "stage_mr",
        "exh_qual",
        "tf_tape",
        "mr_tape",
        "tf_minus_mr",
        "G_VOL",
        "G_TREND",
        "G_LIQ_GOOD",
        "G_LIQ_GREAT",
        "G_LIQ_EXCEL",
        "G_MR_SPIKE",
        "G_VOL_LIQ_GT1",
        "G_VOL_LIQ_GT2",
        "G_VOL_LIQ_GT3",
        "rolling_std(price_innovation)",
        "kalman_gain_1h",
        "state_uncertainty_1h",
        "realized_vol_minus_vol_state",
        "log_volume_state_1h",
        "volume_state_slope_1h",
        "price_slope_x_volume_surprise",
        "vol_state_x_volume_state",
        "asset_atr_level",
        "asset_vol_level",
        "asset_volume_30d",
        "asset_atr_30d",
        "a_funding_proxy",
    ]
)
NON_PORTABLE_GROUP_KEYS = {
    "META_ORDERBOOK_WALL_FEATURE_KEYS",
    "META_ORDERBOOK_BLOCKER_FEATURE_KEYS",
    "META_RECENT_EFFECTIVENESS_FEATURE_KEYS",
    "META_BASE_PERFORMANCE_FEATURE_KEYS",
    "META_SELF_FEATURE_KEYS",
    "META_RECENT_DISAGREEMENT_FEATURE_KEYS",
}

MODEL_DERIVED_META_PERFORMANCE_GROUP_KEYS = {
    "META_RECENT_EFFECTIVENESS_FEATURE_KEYS",
    "META_BASE_PERFORMANCE_FEATURE_KEYS",
    "META_SELF_FEATURE_KEYS",
    "META_RECENT_DISAGREEMENT_FEATURE_KEYS",
    "META_MODEL_UNCERTAINTY_FEATURE_KEYS",
    "BASE_LGBM_META_UNCERTAINTY_FEATURE_KEYS",
    "BASE_LGBM_RAW_STATE_DRIFT_FEATURE_KEYS",
    "BASE_LGBM_PREDICTIVE_ATLAS_FEATURE_KEYS",
    "BASE_LGBM_AE_GMM_FEATURE_KEYS",
    "META_LGBM_AE_GMM_FEATURE_KEYS",
}
MODEL_DERIVED_META_PERFORMANCE_FEATURE_KEYS = set()
for _group_key in MODEL_DERIVED_META_PERFORMANCE_GROUP_KEYS:
    MODEL_DERIVED_META_PERFORMANCE_FEATURE_KEYS.update(
        str(_feature_key)
        for _feature_key in CFG.get(_group_key, [])
        if isinstance(_feature_key, str) and _feature_key
    )

# These are generated from base/meta OOF predictions during meta training, not
# raw exchange-specific panels. Keep them in portable configs even though their
# names intentionally use recent_* prefixes.
PORTABLE_SOURCE_NORMALIZED_FEATURE_KEYS.update(
    MODEL_DERIVED_META_PERFORMANCE_GROUP_KEYS
    | MODEL_DERIVED_META_PERFORMANCE_FEATURE_KEYS
)


def is_non_portable_feature_key(name: object) -> bool:
    key = str(name)
    if key in PORTABLE_SOURCE_NORMALIZED_FEATURE_KEYS:
        return False
    if key in NON_PORTABLE_GROUP_KEYS or key in NON_PORTABLE_FEATURE_EXACT:
        return True
    return key.startswith(NON_PORTABLE_FEATURE_PREFIXES)


def _portable_feature_list(values):
    return [
        k
        for k in list(values or [])
        if str(k) not in DELETED_CONTRACT_FEATURE_KEYS
        and not is_non_portable_feature_key(k)
    ]


for _name in (
    "PERP_FEATURE_KEYS",
    "PERP_PRICE_RELATION_FEATURE_KEYS",
    "SPOT_FOR_PERPS_BASE_FEATURE_KEYS",
    "SPOT_FOR_PERPS_META_FEATURE_KEYS",
    "PERP_EVENT_RISK_FEATURE_KEYS",
    "PERP_CARRY_ALPHA_FEATURE_KEYS",
    "OI_FEATURE_KEYS",
    "OI_TRADING_FEATURE_KEYS",
    "PERP_TRADEABILITY_FEATURE_KEYS",
    "LGBM_PERP_FEATURE_KEYS",
    "PERP_META_PRIMARY_FEATURE_KEYS",
    "ORDERBOOK_RAW_BASE_FEATURE_KEYS",
    "ORDERBOOK_NORMALIZED_BASE_FEATURE_KEYS",
    "ORDERBOOK_BASE_FEATURE_KEYS",
    "ORDERBOOK_RAW_META_FEATURE_KEYS",
    "ORDERBOOK_NORMALIZED_META_FEATURE_KEYS",
    "ORDERBOOK_META_FEATURE_KEYS",
    "ORDERBOOK_DIAGNOSTIC_ONLY_FEATURE_KEYS",
    "ORDERBOOK_EXCLUDED_STALE_FEATURE_KEYS",
    "ORDERBOOK_FEATURE_KEYS",
    "CROSS_ASSET_FEATURE_KEYS",
    "ROLLING_ALPHA_FEATURE_KEYS",
    "CURRENT_REGIME_AE_FEATURE_KEYS",
    "REGIME_ADAPTOR_BASE_FEATURE_KEYS",
    "REGIME_ADAPTOR_GLOBAL_FEATURE_KEYS",
    "REGIME_ADAPTOR_FUNDING_FEATURE_KEYS",
    "REGIME_ADAPTOR_ORDERBOOK_FEATURE_KEYS",
    "REGIME_ADAPTOR_CROSS_ASSET_FEATURE_KEYS",
    "MODEL_DIRECT_BASE_FEATURE_KEYS",
    "MODEL_REGIME_CONTEXT_META_FEATURE_KEYS",
    "MODEL_REGIME_XS_META_FEATURE_KEYS",
    "MODEL_REGIME_TAIL_META_FEATURE_KEYS",
    "MODEL_REGIME_EIGEN_META_FEATURE_KEYS",
    "MARKET_SPECTRAL_POSITION_META_FEATURE_KEYS",
    "MARKET_SPECTRAL_POSITION_SOURCE_FEATURE_KEYS",
    "MODEL_REGIME_COMPOSITE_META_FEATURE_KEYS",
    "NEGATIVE_RESIDUAL_PRIMITIVE_FEATURE_KEYS",
    "NEGATIVE_RESIDUAL_COMPOSITE_FEATURE_KEYS",
    "NEGATIVE_RESIDUAL_TEMPORAL_MECHANISM_FEATURE_KEYS",
    "NEGATIVE_RESIDUAL_META_FEATURE_KEYS",
):
    globals()[_name] = _portable_feature_list(globals().get(_name, []))
    CFG[_name] = globals()[_name]

CFG["NEGATIVE_RESIDUAL_ALL_FEATURE_KEYS"] = list(
    NEGATIVE_RESIDUAL_META_FEATURE_KEYS
)
CFG["NEGATIVE_RESIDUAL_META_FEATURE_KEYS"] = list(NEGATIVE_RESIDUAL_META_FEATURE_KEYS)

for _name in (
    "FUNDING_META_FEATURE_KEYS",
    "CROSS_ASSET_META_FEATURE_KEYS",
    "INTERACTION_META_FEATURE_KEYS",
    "META_ORDERBOOK_WALL_FEATURE_KEYS",
    "META_ORDERBOOK_BLOCKER_FEATURE_KEYS",
    "META_CROSS_SECTIONAL_REGIME_KEYS",
    "META_RECENT_EFFECTIVENESS_FEATURE_KEYS",
    "META_BASE_PERFORMANCE_FEATURE_KEYS",
    "META_SELF_FEATURE_KEYS",
    "META_RECENT_DISAGREEMENT_FEATURE_KEYS",
    "META_MODEL_UNCERTAINTY_FEATURE_KEYS",
    "BASE_LGBM_META_UNCERTAINTY_FEATURE_KEYS",
    "BASE_LGBM_RAW_STATE_DRIFT_FEATURE_KEYS",
    "BASE_LGBM_PREDICTIVE_ATLAS_FEATURE_KEYS",
    "BASE_LGBM_AE_GMM_FEATURE_KEYS",
    "META_LGBM_AE_GMM_FEATURE_KEYS",
):
    CFG[_name] = _portable_feature_list(CFG.get(_name, []))

for _name in (
    "base_shared_feature_keys",
    "base_long_feature_keys",
    "base_short_feature_keys",
    "meta_shared_feature_keys",
    "meta_product_feature_keys",
    "meta_reg_feature_keys",
    "meta_clf_feature_keys",
    "meta_mfe_feature_keys",
    "meta_mae_feature_keys",
    "meta_asym_feature_keys",
    "spike_feature_keys",
    "test_feature_keys",
):
    CFG[_name] = _portable_feature_list(CFG.get(_name, []))

for _name in (
    "shared_feature_keys",
    "model1_edge_feature_keys",
    "model2_downside_feature_keys",
    "model3_uncertainty_feature_keys",
):
    POSITION_SIZER_V2_FEATURE_CONFIG[_name] = _portable_feature_list(
        POSITION_SIZER_V2_FEATURE_CONFIG.get(_name, [])
    )
CFG["position_sizer_features"] = _portable_feature_list(
    CFG.get("position_sizer_features", [])
)

REGIME_ADAPTOR_FEATURE_ORDER = _portable_feature_list(REGIME_ADAPTOR_FEATURE_ORDER)
CFG["REGIME_ADAPTOR_BASE_FEATURE_KEYS"] = REGIME_ADAPTOR_BASE_FEATURE_KEYS
CFG["REGIME_ADAPTOR_FEATURE_ORDER"] = REGIME_ADAPTOR_FEATURE_ORDER
REGIME_ADAPTOR_DEFAULT_CONFIG[RegimeAdaptorKey.REGIME_ADAPTOR_GLOBAL_FEATURE_KEYS] = (
    REGIME_ADAPTOR_GLOBAL_FEATURE_KEYS
)
REGIME_ADAPTOR_DEFAULT_CONFIG[
    RegimeAdaptorKey.REGIME_ADAPTOR_CROSS_ASSET_FEATURE_KEYS
] = REGIME_ADAPTOR_CROSS_ASSET_FEATURE_KEYS
REGIME_ADAPTOR_DEFAULT_CONFIG[RegimeAdaptorKey.REGIME_ADAPTOR_FUNDING_FEATURE_KEYS] = (
    REGIME_ADAPTOR_FUNDING_FEATURE_KEYS
)
REGIME_ADAPTOR_DEFAULT_CONFIG[
    RegimeAdaptorKey.REGIME_ADAPTOR_ORDERBOOK_FEATURE_KEYS
] = REGIME_ADAPTOR_ORDERBOOK_FEATURE_KEYS
CFG.update(REGIME_ADAPTOR_DEFAULT_CONFIG)

for _cfg_key, _cfg_value in list(CFG.items()):
    if isinstance(_cfg_value, list):
        CFG[_cfg_key] = [
            item
            for item in _cfg_value
            if str(item) not in DELETED_CONTRACT_FEATURE_KEYS
        ]
    elif isinstance(_cfg_value, tuple):
        CFG[_cfg_key] = tuple(
            item
            for item in _cfg_value
            if str(item) not in DELETED_CONTRACT_FEATURE_KEYS
        )
