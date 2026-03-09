# Central config. Keep it deterministic and explicit.
from extreme_price_movements.perp_features import get_perp_feature_names

# =============================================================================
# CANONICAL Horizons & Buckets - Single Source of Truth
# =============================================================================
# H8 removed: poor signal (MR_long_H8: 0.017, TF_short_H8: 0.023)
CANON_BUCKETS = ["MR_long", "MR_short", "TF_long", "TF_short"]
CANON_HORIZONS = [1, 2, 4]  # hours - H8 removed due to poor signal
CANON_CELLS = [f"{b}_H{h}" for b in CANON_BUCKETS for h in CANON_HORIZONS]


_PERP_COLLISION_RENAMES = {
    "ret1h": "ret1h_perp",
}
PERP_FEATURE_KEYS = [
    _PERP_COLLISION_RENAMES.get(k, k)
    for k in get_perp_feature_names()
]


FEATURE_KEYS_15M_OHLCV = [
    "clv_t", "body_ratio_15m", "rejection_proxy",
    "range_norm_12", "sv_imb_12", "press_12", "impact_12", "ts_12",
    "prog_eff_12", "pers_12", "hh_count_12", "ll_count_12", "skew_12",
    "climax_range_12", "climax_vol_12", "z_vwap_12", "z_r_12", "bb_pos_12",
    "range_norm_24", "sv_imb_24", "press_24", "impact_24", "ts_24",
    "prog_eff_24", "pers_24", "hh_count_24", "ll_count_24", "skew_24",
    "climax_range_24", "climax_vol_24", "z_vwap_24", "z_r_24", "bb_pos_24",
]

neutral_feature_keys = [
    "rsi", "vol_z", "atr_pct", "mkt_rv_ratio", "skew", 
    "trend_snr", "efficiency", "vol_asym", "momentum_accel",
    "dist_stack", "stage_blowoff", "exh_qual", "volatility_zscore",
    "dv_z", "rng_z", "impact_z", "liq_score", "liq_state"
]

MODEL_FEATURES = [
    # Momentum / structure extensions
    "thrust_decay_4", "decel_4", "ft_drop",
    "thrust_decay_8", "decel_8", "ft_drop_8",
    "ext_excess", "ext_atrExp",
    "comp_to_exp", "evr6_x_volz", "stall_x_flow", "prog_def",
    "clv_collapse", "clv_pullback", "coh", "align", "retest_quality",
    "pb_accel", "rv_ratio_6_24", "excess_coh", "asym_ft",
    "tf_bias", "shock_rel", "resid_strength", "evr_slope", "stall_ext",
    # Earlier trend following / volatility-of-volatility signals
    "vov_iqr_20", "vov_mad_20", "vov_mad_60", "vov_ratio", "vov_interaction",
    "vov_fast_slow_ratio", "accel_5h", "dlog_vol_5h", "signed_max_bar_ret_5h",
    "jump_rate_10h", "volu_z", "volume_price_corr_10h", "draw_sym_10h", "draw_extreme_10h",
    "shannon_entropy_ret_8", "shannon_entropy_ret_16",
    "perm_entropy_ret_12", "perm_entropy_ret_24",
    "spectral_entropy_ret_24", "spectral_entropy_ret_48",
    "volume_entropy_12", "volume_entropy_24",
    "downside_semivariance_8", "downside_semivariance_24",
    "upside_semivariance_8", "upside_semivariance_24",
    "down_up_vol_ratio_8", "down_up_vol_ratio_24",
    "vol_shock_asym_8_24", "vol_shock_asym_4_12", "vol_shock_asym_4_212",
    "breakout_24h",
    "meta_abs_net_x_breakout", "meta_abs_net_x_drawext", "meta_abs_net_x_vov_ratio",
    "meta_alignment", "meta_signal_x_accel",
    "kf_score_mean", "kf_score_rm24_mean", "kf_atr_mean", "kf_vol_ratio_mean", "kf_ret1h_mean",
    "kf_innov_var", "kf_snr_est", "kf_state_uncertainty",
    # Price Action
    "gap_pct", "range_pct", "roc_div", "ret1h_z", "body_pct", "wick_body_ratio",
    "vol_price_spread", "wick_ratio", "body_ratio",
    # New Risk/Exhaustion (Report 2026-02-10)
    "wick_ratio_4h_max", "vol_price_div", "rsi_lag1", "rsi_1h_slope",
    "cvar_5pct", "amihud_illiq", "clv_mean_24", "vol_z_4h", "atr_pct_change",
    # FFD d-specific features (d=0.4,0.6)
    "ffd_rv_2h_04", "ffd_rv_6h_04", "ffd_rv_24h_04",
    "ffd_vol_price_corr_10h_04",
    "ffd_donch_dist_04_12", "ffd_donch_dist_04_24", "ffd_donch_dist_04_48",
    "ffd_amihud_04", "ffd_vol_range_shock_04",
    "ffd_dist_ema_fast_04", "ffd_dist_ema_slow_04",
    "ffd_rv_2h_06", "ffd_rv_6h_06", "ffd_rv_24h_06",
    "ffd_accel_06", "ffd_z_06",
    "ffd_vol_price_corr_10h_06",
    "ffd_donch_dist_06_12", "ffd_donch_dist_06_24", "ffd_donch_dist_06_48",
    "ffd_atr_expansion_06", "ffd_cvar_5pct_06",
    "ffd_amihud_06", "ffd_vol_range_shock_06",
    # D-family strength indicators
    "ffd_strength_04", "ffd_strength_05", "ffd_strength_06",
    # Alpha Features (Report 2026-02-10)
    "breakout_min", "impulse_reversal", "impulse_reversal_short",
    "breakout_confirmed", "breakout_t", "pct_breakout_t",
    # 2h directional path-risk
    "dir_path_long_2h", "dir_path_short_2h", "dir_path_risk_long_2h", "dir_path_risk_short_2h",
    "dir_path_edge_2h", "dir_path_risk_skew_2h",
    # Gate interactions (2h focus)
    "accept_x_dir_edge_2h", "reject_x_dir_edge_2h", "tfq_x_dir_edge_2h", "mrq_x_dir_edge_2h",
    "accept_dir2h_prod", "accept_dir2h_abs_prod", "accept_dir2h_signed_mag",
    "reject_dir2h_prod", "reject_dir2h_abs_prod", "reject_dir2h_signed_mag",
    "tfq_dir2h_prod", "tfq_dir2h_abs_prod", "tfq_dir2h_signed_mag",
    "mrq_dir2h_prod", "mrq_dir2h_abs_prod", "mrq_dir2h_signed_mag",
    # Volume/Flow
    "v_power", "flow_persistence", "flow_ratio", "churn",
    "vol_range_shock", "climax_decay", "cumulative_delta_stall",
    "vol_expansion_ratio", "vol_compression", "rvol_z",
    "G_VOL_LIQ_GT1", "G_VOL_LIQ_GT2", "G_VOL_LIQ_GT3",
    "amihud_z", "G_LIQ_GOOD", "G_LIQ_GREAT", "G_LIQ_EXCEL",
    # Advanced
    "fvg", "slope", "atr_slope", "dist_vwap_norm", "rsi_slope",
    "funding_proxy", "dist_ema_fast",
    # Scores
    "spike_score", "grind_score", "chop_score",
    # Time
    "sin_hod", "cos_hod", "sin_dow", "cos_dow",
    # New regime-transition and entropy features for improved PR-AUC and robustness
    "regime_transition_entropy_12h", "regime_transition_entropy_48h",
    "trend_regime_switch_12h", "vol_regime_switch_12h",
    "entropy_jump_24h", "complexity_regime_24h",
    "rsi_z_x_regime_vol", "vol_z_x_regime_trend",
    "mtf_divergence_x_regime_vol_12h", "hurst_proxy_x_regime_trend_48h",
    # Regime features for fold robustness (Report 2026-02-11)
    "vol_regime_z", "is_high_vol_regime", "is_low_vol_regime",
    "trend_regime", "is_trending", "is_ranging",
    "liq_regime", "regime_stability_24h",
    "rsi_x_high_vol", "trend_x_trending", "vol_z_x_low_vol",
    # New Indicators (KER, Vortex, ADX, VWAP, HVN/LVN)
    "ker_10", "ker_16", "ker_24",
    "vortex_diff_14", "vortex_diff_21", "vortex_diff_34",
    "adx_7", "adx_10", "adx_14",
    "adx_di_plus_7", "adx_di_minus_7",
    "adx_di_plus_10", "adx_di_minus_10",
    "adx_di_plus_14", "adx_di_minus_14",
    "adx_7_gt25", "adx_10_gt25", "adx_14_gt25",
    "adx_7_slope", "adx_10_slope", "adx_14_slope",
    "dist_vwap_12_atr", "trapped_longs_12",
    "dist_vwap_24_atr", "trapped_longs_24",
    "dist_vwap_96_atr", "trapped_longs_96",
    "vp_dist_poc_atr", "vp_dist_hvn_above_atr", "vp_dist_hvn_below_atr",
    "vp_dist_lvn_above_atr", "vp_dist_lvn_below_atr",
    "vp_in_poc_zone", "vp_in_hvn_above_zone", "vp_in_hvn_below_zone",
    "vp_in_lvn_above_zone", "vp_in_lvn_below_zone",
    "vp_bin_vol_share", "vp_profile_concentration", "vp_profile_entropy",
    "vp_lvn_depth_ratio", "vp_accept_poc_touchrate",
    "vp_accept_hvn_touchrate", "vp_accept_lvn_touchrate",
    "vp_air_pocket_score",
    "G_TF_TREND", "vol_z_x_trend_t",
]

# Helper/base features produced in features.py that should remain selectable by model heads.
# This increases candidate breadth before MDI pruning.
HELPER_BASE_FEATURES = [
    "ret1h", "ret2h", "ret4h", "ret6h", "ret8h", "ret48h", "ret72h", "ret120h",
    "atr_pct_base", "rsi_base", "rsi_slope_base",
    "rv_2h", "rv_4h", "rv_6h", "rv_8h", "rv_12h", "rv_24h", "rv_48h", "rv_120h",
    "qv", "vol_z24_base", "vol_z_base",
    "dist_ema_fast_base", "dist_ema_slow_base", "trend_pct_base",
    "rvol_hod_base", "signed_vol", "up_vol", "dn_vol", "up_vol_6", "dn_vol_6",
    "vol_asym_6", "clv", "clv_mean_2", "excess_12h", "speed",
    "atr_expansion", "stall_ext_corr",
    "asset_atr_level", "asset_vol_level", "atr_state", "vol_state",
    "G_EXH_EFFORT", "G_EXH_GIVEBACK", "G_EXH_TAIL_FAIL",
    "G_MR_SPIKE", "G_TF_GRIND", "G_TF_TREND", "G_MR_TAIL",
    "G_META_EXH", "G_META_TF_QUAL", "G_META_MR_QUAL", "G_META_AMBIG",
    # FFD d-specific helper features
    "ffd_diff_1_04", "ffd_diff_2_04", "ffd_diff_4_04", "ffd_diff_8_04",
    "ffd_diff_1_05", "ffd_diff_2_05", "ffd_diff_4_05", "ffd_diff_8_05",
    "ffd_diff_1_06", "ffd_diff_2_06", "ffd_diff_4_06", "ffd_diff_8_06",
    "ffd_ema_spread_04", "ffd_ema_spread_05", "ffd_ema_spread_06",
    "ffd_rv_12_04", "ffd_rv_12_05", "ffd_rv_12_06", "ffd_rv_24_04", "ffd_rv_24_05", "ffd_rv_24_06",
    "ffd_z_24_04", "ffd_z_24_05", "ffd_z_24_06",
    "ffd_range_24_04", "ffd_range_24_05", "ffd_range_24_06",
    "ffd_slope_04_12", "ffd_slope_04_24", "ffd_mr_z_04", "ffd_mr_z_05",
    "ffd_d1_05", "ffd_d4_05",
    "ffd_ctx_slope_04_12", "ffd_ctx_slope_04_24",
    # Range features for event scoring and candidate selection
    "range_pct", "range_12h_pct", "range_16h_pct", "range_24h_pct",
    # FFD d-specific advanced features
    "ffd_rv_2h_04", "ffd_rv_6h_04", "ffd_rv_24h_04",
    "ffd_vol_price_corr_10h_04",
    "ffd_donch_dist_04_12", "ffd_donch_dist_04_24", "ffd_donch_dist_04_48",
    "ffd_amihud_04", "ffd_vol_range_shock_04",
    "ffd_dist_ema_fast_04", "ffd_dist_ema_slow_04",
    "ffd_rv_2h_06", "ffd_rv_6h_06", "ffd_rv_24h_06",
    "ffd_accel_06", "ffd_z_06",
    "ffd_vol_price_corr_10h_06",
    "ffd_donch_dist_06_12", "ffd_donch_dist_06_24", "ffd_donch_dist_06_48",
    "ffd_atr_expansion_06", "ffd_cvar_5pct_06",
    "ffd_amihud_06", "ffd_vol_range_shock_06",
    "ffd_strength_04", "ffd_strength_05", "ffd_strength_06",
]

# Compact feature basket for learnability tests across symbol universes,
# TBM geometry settings, and sample-weight configurations.
# Emphasis: 2/4/8-bar behavior + longer-horizon regime context.
TEST_FEATURE_KEYS = [
    # Realized vol / ATR (multi-horizon)
    "rv_2h", "rv_4h", "rv_8h", "rv_24h", "atr_pct", "atr_pct_change",
    # Returns + slope family (2/4/8 focus)
    "ret2h", "ret4h", "ret8h", "ret24h", "slope", "atr_slope", "rsi_slope",
    # Momentum acceleration
    "momentum_accel", "accel", "accel_5h",
    # Price distance / z-score style context (EMA / VWAP / breakout band proxies)
    "dist_ema_fast", "dist_vwap_norm", "breakout_t", "pct_breakout_t", "ret1h_z",
    # RVOL + volume acceleration
    "rvol_z", "vol_z", "vol_z_4h", "dlog_vol_5h", "volume_entropy_12",
    # Vol-of-vol
    "vov_ratio", "vov_fast_slow_ratio", "vov_mad_20",
    # Autocorrelation / Hurst-ish / path efficiency proxies
    "autocorr_6h", "autocorr_24h", "hurst_proxy_24", "path_efficiency_12", "path_efficiency_24",
    # Liquidity + time-of-day
    "amihud_illiq", "amihud_z", "sin_hod", "cos_hod", "sin_dow", "cos_dow",
    # Mid/long lookback context for 8-bar horizon learnability (16-24h + slower)
    "ret16h", "coherence_24", "impulse_ratio_24", "range_24h_pct",
    "shannon_entropy_ret_16", "perm_entropy_ret_24", "spectral_entropy_ret_24", "volume_entropy_24",
    "ret48h", "ret120h", "rv_48h", "rv_120h", "spectral_entropy_ret_48",
    # Longer-timeframe regime context
    "trend_regime", "vol_regime_z", "regime_stability_24h", "trend_slope_48h", "trend_slope_120h",
    "rv_ratio_24_120", "donch_dist_48", "donch_dist_120", "dist_from_high_120h", "dist_from_low_120h",
]

CFG = {
    # persistence / fetch
    "data_root": "data",
    "reports_root": "reports",
    "hf_data_dir": "15m_ohlcv",
    "use_perps": False,
    "timeframe": "1h",
    "fetch_years": 4,
    "fetch_symbols_M": 600,
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
    "download_15m_full_backfill": True,

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
    "market_basket": ["BTC/USDT","ETH/USDT","AVAX/USDT","SOL/USDT","XRP/USDT"],

    # training horizons to compare
    # Canonical set is [1, 2, 4] hours. H1 added for entry timing; H8 removed.
    "label_horizons_hours": CANON_HORIZONS,
    "base_geometry_archetypes": ["tight", "balanced", "wide"],
    "base_geometry_train_variants": True,
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
    "train_lookback_hours": 24 * 365 * 4,   # 4 years
    "val_lookback_hours": 24 * 7,      # 7d validation (time-split, no leakage)
    "min_train_samples": 200,

    # MFE/MAE-based sample weighting (Report 2026-02-12)
    # Weight samples by how "decisive" the price movement was relative to barriers
    # w = w_min + (1-w_min) * clip(max(MFE/TP, MAE/SL) / tau, 0, 1)
    # Timeout samples are capped at 0.7
    "mfe_mae_w_min": 0.5,      # Minimum weight floor
    "mfe_mae_tau": 1.0,        # Scaling factor (d/tau)
    "mfe_mae_cost_floor": 0.001,  # Cost floor for touch margin penalty
    # MR path-aware weighting (de-emphasize raw magnitude, emphasize efficient path)
    "mr_weight_magnitude_power": 0.35,
    "mr_weight_mfe_tau": 1.0,
    "mr_weight_mae_tau": 1.0,
    # MR utility target: velocity decay horizon for bars_to_mfe penalty
    "mr_utility_horizon_bars": 8,

    # Meta model sample weighting
    # Magnitude sigmoid: w = 1 + alpha * sigmoid((|ret| - q70) / std)
    # alpha=0.5 gives top-30% ~1.25-1.5x upweight (moderate)
    "meta_weight_sigmoid_alpha": 0.5,
    # MFE/MAE quality: w_exc = 0.5 + 0.5 * clip(max(MFE/barrier, MAE/barrier) / tau, 0, 1)
    "meta_mfe_mae_tau": 1.0,

    # Sample-weight optimization (base + meta)
    "sample_weight_opt_enable": True,
    "sample_weight_opt_min_samples": 400,
    "sample_weight_opt_trials": 16,
    "meta_sample_weight_opt_trials": 12,
    "meta_use_policy_value_target": True,
    "meta_clf_use_engine_labels": True,
    # Policy-aligned downstream sizing requires classifier barrier probabilities
    # (oof_p_sl/oof_p_to/oof_p_tp) in meta_oof exports.
    "meta_race_include_classifiers": True,
    "meta_require_classifier_barrier_probs": True,
    "meta_train_regression_bucket_model": True,
    "meta_training_pipeline_version": "aligned_map_v2",
    "meta_train_save_legacy_setup": True,
    "meta_parallel_forest_disable_hpo": True,
    "meta_parallel_forest_num_parallel_tree": 160,
    "meta_parallel_forest_rounds": 8,
    "meta_parallel_forest_max_depth": 6,
    "meta_parallel_forest_learning_rate": 0.05,
    "meta_parallel_forest_reg_alpha": 2.0,
    "meta_parallel_forest_reg_lambda": 20.0,
    "meta_parallel_forest_min_child_weight": 48.0,
    "meta_parallel_forest_gamma": 2.5,
    "meta_map_tbm_geometries": [
        {"name": "tbm_50_25", "tp_pct": 0.0050, "sl_pct": 0.0025},
        {"name": "tbm_100_50", "tp_pct": 0.0100, "sl_pct": 0.0050},
    ],
    "meta_map_tbm_horizons": [1, 2, 4],
    "meta_map_mae_horizons": [2, 4],
    "meta_map_mfe_horizons": [2, 4],
    "meta_map_weight_clip_lo": 0.85,
    "meta_map_weight_clip_hi": 1.15,
    # Meta classifier utility-based winner selection (logloss remains a gate)
    "meta_clf_max_logloss": 1.10,
    "meta_clf_u_tp": 1.0,
    "meta_clf_u_to": 0.0,
    "meta_clf_u_sl": -3.0,
    "meta_clf_top_frac": 0.30,
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
    "aux_mae_weight_variants": ["none", "asymmetric_tail", "symmetric_tail", "top30_tail"],
    "aux_mae_qbin_bins": 20,
    "aux_mfe_target_variants": ["rank_pct", "qbin_mid"],
    "aux_mfe_weight_variants": ["none", "asymmetric_tail", "symmetric_tail", "top30_tail"],
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
    # Position-sizer features from meta OOF + regime context.
    "position_sizer_feature_priority": [
        "impact_z",
        "dv_z",
        "rng_z",
        "score",
        "reg",
        "reg_mean",
        "reg_std",
        "reg_range",
        "utility",
        "mae_q70",
        "mfe",
        "early_inval",
        "oof_u_hat",
        "oof_log_mae_q70_hat",
        "oof_log_mfe_hat",
        "mfe_mae_ratio_hat",
        "vol_regime_z",
        "regime_stability_24h",
        "complexity_regime_24h",
        "vol_regime_z_4d",
        "trend_strength_4d",
        "regime_stability_4d",
        "vol_persistence_4d",
        "trend_regime_duration_4d",
        "regime_transition_entropy_48h",
        "vol_concentration_12",
        "volume_entropy_12",
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
        "Upside",
        "Downside",
        "EdgeSharpe",
        "risk_reward_ratio",
        "high_utility_pred",
        "risk_adjusted_pred",
        "utility_disagreement",
    ],
    "limit_offset_sizer": [
        "impact_z",
        "dv_z",
        "rng_z",
        "score",
        "reg",
        "reg_mean",
        "reg_std",
        "reg_range",
        "utility",
        "mae_q70",
        "mfe",
        "early_inval",
        "oof_u_hat",
        "oof_log_mae_q70_hat",
        "oof_log_mfe_hat",
        "mfe_mae_ratio_hat",
        "vol_regime_z",
        "regime_stability_24h",
        "complexity_regime_24h",
        "vol_regime_z_4d",
        "trend_strength_4d",
        "regime_stability_4d",
        "vol_persistence_4d",
        "trend_regime_duration_4d",
        "regime_transition_entropy_48h",
        "vol_concentration_12",
        "volume_entropy_12",
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
        "Upside",
        "Downside",
        "EdgeSharpe",
        "risk_reward_ratio",
        "high_utility_pred",
        "risk_adjusted_pred",
        "utility_disagreement",
    ],
    "position_sizer_regime_feature_keys": [
        "vol_regime_z",
        "regime_stability_24h",
        "complexity_regime_24h",
        "vol_regime_z_4d",
        "trend_strength_4d",
        "regime_stability_4d",
        "vol_persistence_4d",
        "trend_regime_duration_4d",
    ],
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
    "sample_weight_opt_n_splits": 5,
    "sample_weight_opt_embargo_bars": 10,
    "cv_embargo_bars": 12,
    "sample_weight_opt_min_n_eff_ratio": 0.30,
    "sample_weight_opt_max_top1pct": 0.10,
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

    # per-hour cross-sectional training selection
    "variance_filter_pct": 1.0, # Keep all non-constant assets
    "train_extreme_pct_hourly": 0.06,  # Keep top/bottom 6% as extreme candidates (reduced from 0.08)
    "train_extreme_min": 10,
    "train_extreme_max": 80,
    "train_min_range_pct": 0.07,
    "train_min_vol_zscore": 1.6,

    # Triple barrier geometry parameters (DEPRECATED - use unified barrier factory params below)
    # Old Pipeline 1 params:
    "train_z_max": 3.0,            # Max z-score clip (symmetric) - DEPRECATED
    "train_tp_k_lo": 0.5,          # TP lower bound = k_lo * base_ATR - DEPRECATED
    "train_tp_k_hi": 1.5,          # TP upper bound = k_hi * base_ATR - DEPRECATED
    "train_sl_mult_lo": 0.4,       # SL ratio in quiet markets - DEPRECATED
    "train_sl_mult_hi": 0.7,       # SL ratio in volatile markets - DEPRECATED

    # Unified barrier factory parameters (v3 - single source of truth for both pipelines)
    # Single barrier mode: k_tp and sl_base_mult are scalars
    "barrier_k_tp": 1.0,                      # dimensionless k_tp for single geometry
    "barrier_sl_base_mult": 0.5,              # RR (e.g., 0.5 = 2:1 reward:risk)
    
    # TP bounds (match old scaled_atr_pct behavior: clamp to [tp_lo, tp_hi])
    "barrier_tp_lo": 0.02,     # Lower bound for TP (2%)
    "barrier_tp_hi": 0.06,     # Upper bound for TP (6%)
    
    # Multi-geometry mode: k_tp and sl_base_mult are grids
    "barrier_k_tp_grid": [0.8, 1.0, 1.25, 1.6, 2.0, 2.5],  # dimensionless k_tp grid
    "barrier_sl_base_grid": [0.5, 1.0, 1.5],  # RR grid (0.5 = 2:1, 1.0 = 1:1, 1.5 = 0.67:1)
    
    # Dispersion-based regime scaling
    "barrier_disp_floor": 0.1,     # MAD-based z-score floor (prevents division by near-zero)
    "barrier_z_max": 3.0,         # Max z-score clip (symmetric)
    "barrier_k_reg": 0.3,         # Regime tightness: lower = tighter TP in quiet markets
    "barrier_m_lo": 0.7,          # Multiplier low (quiet markets)
    "barrier_m_hi": 1.5,          # Multiplier high (volatile markets)
    "barrier_sl_lo": 0.4,         # SL ratio in quiet markets
    "barrier_sl_hi": 0.7,         # SL ratio in volatile markets
    "barrier_z_gate": 1.0,        # z-score threshold for regime transition
    
    # Horizon scaling
    "label_horizon_base": 4,       # base horizon for sqrt(H/H_base) scaling
    "label_min_net_rr": 0.9,      # min reward:risk ratio after fees
    
    # Legacy / deprecated params
    "label_tp_mults": [0.5, 1.0, 1.5, 2.0],   # DEPRECATED: use barrier_k_tp_grid
    "label_sl_mults": [0.3, 0.5, 0.7, 1.0],    # DEPRECATED: use barrier_sl_base_grid
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
    "causal_cols": [
        "dv_z", "rng_z", "impact_z", "liq_score", "liq_state",
        "ret24h", "rsi", "vol_z", "atr_pct", "trend_pct", "rv_2h", "rv_4h", "rv_24h",
        "p_exh_lag1",
        "a_funding_proxy",
        "flow_ratio", "churn", "slope", "trend_snr",
        "vol_asym", "skew", "efficiency", "fvg",
        "rvol_z", "vol_range_shock", "climax_decay",
        "cumulative_delta_stall", "vol_expansion_ratio", "vol_compression",
        "atr_slope", "dist_vwap_norm", "momentum_accel",
        # New Risk/Exhaustion (Report 2026-02-10)
        "wick_ratio_4h_max", "vol_price_div", "rsi_lag1", "rsi_1h_slope",
        "cvar_5pct", "amihud_illiq", "clv_mean_24", "vol_z_4h", "atr_pct_change",
        # FFD d-specific features (d=0.4,0.6)
        "ffd_rv_2h_04", "ffd_rv_6h_04", "ffd_rv_24h_04",
        "ffd_vol_price_corr_10h_04",
        "ffd_donch_dist_04_12", "ffd_donch_dist_04_24", "ffd_donch_dist_04_48",
        "ffd_amihud_04", "ffd_vol_range_shock_04",
        "ffd_dist_ema_fast_04", "ffd_dist_ema_slow_04",
        "ffd_rv_2h_06", "ffd_rv_6h_06", "ffd_rv_24h_06",
        "ffd_accel_06", "ffd_z_06",
        "ffd_vol_price_corr_10h_06",
        "ffd_donch_dist_06_12", "ffd_donch_dist_06_24", "ffd_donch_dist_06_48",
        "ffd_atr_expansion_06", "ffd_cvar_5pct_06",
        "ffd_amihud_06", "ffd_vol_range_shock_06",
        # D-family strength indicators
        "ffd_strength_04", "ffd_strength_05", "ffd_strength_06",
        # New Feature Candidates
        "thrust_decay_4", "decel_4", "ft_drop", "ext_excess", "ext_atrExp",
        "comp_to_exp", "evr6_x_volz", "stall_x_flow", "prog_def",
        "clv_collapse", "clv_pullback", "coh", "align", "retest_quality",
        "pb_accel", "rv_ratio_6_24", "excess_coh", "asym_ft", "dist_stack",
        "tf_bias", "shock_rel", "resid_strength", "evr_slope", "stall_ext",
        "spike_score", "grind_score", "chop_score", "G_TF_TREND", "vol_z_x_trend_t",
        # Gates as continuous features
        "G_EXH_EFFORT", "G_EXH_GIVEBACK",
        "G_EXH_TAIL_FAIL",
        "G_MR_SPIKE", "G_TF_GRIND", "G_TF_TREND", "G_MR_TAIL",
        "G_META_EXH", "G_META_TF_QUAL", "G_META_MR_QUAL", "G_META_AMBIG",
        # New Model Features
        "overext", "overext_weak", "effort_gate", "tail_fail", "blowoff_risk",
        "S", "impulse_ratio_24", "impulse_ratio_12", "coherence_24", "accel",
        "tf_tape", "mr_tape", "accept", "accept_bin3", "accept_gt66", "accept_gt85", "retest_accept", "tf_qual", "mr_qual",
        "retrace_12", "ambig", "stage_tf", "stage_blowoff", "stage_mr", "exh_qual"
        ,"mfe_2h", "mae_2h", "dir_path_long_2h", "dir_path_short_2h",
        "dir_path_risk_long_2h", "dir_path_risk_short_2h", "dir_path_edge_2h", "dir_path_risk_skew_2h",
        "accept_x_dir_edge_2h", "reject_x_dir_edge_2h", "tfq_x_dir_edge_2h", "mrq_x_dir_edge_2h",
        "accept_dir2h_prod", "accept_dir2h_abs_prod", "accept_dir2h_signed_mag",
        "reject_dir2h_prod", "reject_dir2h_abs_prod", "reject_dir2h_signed_mag",
        "tfq_dir2h_prod", "tfq_dir2h_abs_prod", "tfq_dir2h_signed_mag",
        "mrq_dir2h_prod", "mrq_dir2h_abs_prod", "mrq_dir2h_signed_mag"
        # OHLCV-based trend quality features (Report 2026-02-12)
        ,"trend_age_hours", "higher_highs_count_48h", "trend_retest_success_rate",
        "trend_overextension_z", "volume_trend_alignment", "trend_regime_stability",
        "trend_strength_vs_reversion", "support_quality_score", "dip_velocity",
        "dip_volume_profile", "reversion_target_distance"
        ,"vov_iqr_20", "vov_mad_20", "vov_mad_60", "vov_ratio", "vov_interaction",
        "vov_fast_slow_ratio", "accel_5h", "dlog_vol_5h", "signed_max_bar_ret_5h",
        "jump_rate_10h", "volu_z", "volume_price_corr_10h", "draw_sym_10h", "breakout_24h",
        "vol_z_30_calm",
        "meta_abs_net_x_breakout", "meta_abs_net_x_drawext", "meta_abs_net_x_vov_ratio",
        "meta_alignment", "meta_signal_x_accel",
        "kf_score_mean", "kf_score_rm24_mean", "kf_atr_mean", "kf_vol_ratio_mean", "kf_ret1h_mean",
        "kf_innov_var", "kf_snr_est", "kf_state_uncertainty",
        "vol_high", "vol_low", "cusum_strength_norm", "cusum_high", "liq_low",
        "p_vol_high", "p_cusum_high", "p_liq_low",
        "dir_path_long_2h", "dir_path_short_2h", "dir_path_risk_long_2h", "dir_path_risk_short_2h",
        "dir_path_edge_2h", "dir_path_risk_skew_2h",
        "accept_x_dir_edge_2h", "reject_x_dir_edge_2h", "tfq_x_dir_edge_2h", "mrq_x_dir_edge_2h",
        "accept_dir2h_prod", "accept_dir2h_abs_prod", "accept_dir2h_signed_mag",
        "reject_dir2h_prod", "reject_dir2h_abs_prod", "reject_dir2h_signed_mag",
        "tfq_dir2h_prod", "tfq_dir2h_abs_prod", "tfq_dir2h_signed_mag",
        "mrq_dir2h_prod", "mrq_dir2h_abs_prod", "mrq_dir2h_signed_mag",
        # Orthogonal features (structurally independent dimensions)
        "mtf_divergence", "mtf_div_mag",
        "autocorr_6h", "autocorr_24h",
        "path_efficiency_12", "path_efficiency_24",
        "hurst_proxy_24",
        "vol_concentration_12",
        "vol_price_diverge",
        "shannon_entropy_ret_8", "shannon_entropy_ret_16",
        "perm_entropy_ret_12", "perm_entropy_ret_24",
        "spectral_entropy_ret_24", "spectral_entropy_ret_48",
        "volume_entropy_12", "volume_entropy_24",
        "downside_semivariance_8", "downside_semivariance_24",
        "upside_semivariance_8", "upside_semivariance_24",
        "down_up_vol_ratio_8", "down_up_vol_ratio_24",
        "vol_shock_asym_8_24", "vol_shock_asym_4_12", "vol_shock_asym_4_212",
        # Residualised features — relative surprise, not absolute magnitude
        "rsi_z", "dist_ema_fast_z", "dist_vwap_norm_z", "flow_persistence_z",
        "excess_6h_z", "vol_z_z", "atr_expansion_z", "coherence_24_z",
        "accept_surprise", "overext_surprise",
        "blowoff_risk_surprise", "exh_qual_surprise",
        "dist_vwap_resid", "dist_ema_fast_resid", "trend_pct_resid",
        "mkt_rv_pct", "abs_mkt_ret24h_z", "trend_bin3",
        "s_z_64", "s_pct_64", "s_bin3_64",
        "s_gt25_64", "s_gt50_64", "s_gt66_64", "s_gt75_64", "s_gt85_64", "s_gt90_64",
        "s_z_8", "s_pct_8", "s_bin3_8",
        "s_gt25_8", "s_gt50_8", "s_gt66_8", "s_gt75_8", "s_gt85_8", "s_gt90_8",
        "reject_z_64", "reject_pct_64", "reject_bin3_64",
        "reject_gt25_64", "reject_gt50_64", "reject_gt66_64", "reject_gt75_64", "reject_gt85_64", "reject_gt90_64",
        "reject_z_8", "reject_pct_8", "reject_bin3_8",
        "reject_gt25_8", "reject_gt50_8", "reject_gt66_8", "reject_gt75_8", "reject_gt85_8", "reject_gt90_8",
        "retest_accept_z_64", "retest_accept_pct_64", "retest_accept_bin3_64",
        "retest_accept_gt25_64", "retest_accept_gt50_64", "retest_accept_gt66_64", "retest_accept_gt75_64", "retest_accept_gt85_64", "retest_accept_gt90_64",
        "retest_accept_z_8", "retest_accept_pct_8", "retest_accept_bin3_8",
        "retest_accept_gt25_8", "retest_accept_gt50_8", "retest_accept_gt66_8", "retest_accept_gt75_8", "retest_accept_gt85_8", "retest_accept_gt90_8",
        "tf_qual_z_64", "tf_qual_pct_64", "tf_qual_bin3_64",
        "tf_qual_gt25_64", "tf_qual_gt50_64", "tf_qual_gt66_64", "tf_qual_gt75_64", "tf_qual_gt85_64", "tf_qual_gt90_64",
        "tf_qual_z_8", "tf_qual_pct_8", "tf_qual_bin3_8",
        "tf_qual_gt25_8", "tf_qual_gt50_8", "tf_qual_gt66_8", "tf_qual_gt75_8", "tf_qual_gt85_8", "tf_qual_gt90_8",
        "mr_qual_z_64", "mr_qual_pct_64", "mr_qual_bin3_64",
        "mr_qual_gt25_64", "mr_qual_gt50_64", "mr_qual_gt66_64", "mr_qual_gt75_64", "mr_qual_gt85_64", "mr_qual_gt90_64",
        "mr_qual_z_8", "mr_qual_pct_8", "mr_qual_bin3_8",
        "mr_qual_gt25_8", "mr_qual_gt50_8", "mr_qual_gt66_8", "mr_qual_gt75_8", "mr_qual_gt85_8", "mr_qual_gt90_8",

        # New Multi-Horizon Aggregated Features
        "ret_mean", "ret_max", "ret_min",
        "rv_mean", "rv_max", "rv_min",
        
        # New Tail-Risk Features
        "ret_pct5_24h", "ret_pct95_24h", "gap_zscore", "vol_shock_z", 
        "range_zscore", "tail_risk_score",
        
    ],

    # thresholds / picks
    "thr_long":  0.010,
    "thr_short": -0.010,
    "k_long": 10,
    "k_short": 10,
    "score_gate_q": 0.93,              # Global percentile gate: only trade signals in top x% (0.93 = top 7%) of global distribution

    # sizing / risk / costs
    "wallet_gross_cap": 0.25,
    "sizing_mode": "rank",            # "rank" (default), "equal", or "score" — rank uses percentile within batch
    "score_map": "tanh",
    "score_scale": 15.0,
    "tp": 0.05,
    "sl": 0.025,
    "hold_hours": 8,
    "fee_bps": 25.0,
    "borrow_apr": 0.20,

    "oos_holdout_days": 730,   # Enforce >= 2 years OOS holdout for robust signal evaluation

    # Trailing Profit Risk Params (used in backtest & live, all vol-scaled)
    # Target absolute: TP ~2%, SL ~0.7% (with median barrier_pct ~4%)
    "tp_mult": 0.50,            # Activation threshold = tp_mult * barrier_pct (~2%)
    "sl_mult": 0.18,            # Stop-loss = sl_mult * barrier_pct (~0.7%)
    "trail_mult": 0.25,         # Trailing deviation = trail_mult * barrier_pct
    # Hard constraints enforced in optimizer and defaults
    "min_tp_sl_ratio": 1.2,     # TP:SL ratio must be >= 1.2
    "min_tp_abs_pct": 0.02,     # TP must be >= 2% absolute

    # Regime throttle: reduce sizing during drawdowns
    "throttle_lookback_trades": 20,     # look at last N closed trades
    "throttle_dd_threshold": -0.02,     # cumPnL drawdown trigger
    "throttle_sizing_factor": 0.5,      # reduce sizing to 50% when triggered

    # Portfolio constraints
    "max_concurrent_trades": 5,
    "max_portfolio_weight": 0.25,

    # Daily risk budget: concentration controls
    "max_daily_per_specialist": 8,   # max trades/day per bucket (LONG_TF, SHORT_MR, etc.)
    "max_daily_total": 25,           # max total trades/day across all buckets

    # Legacy Risk Params (Trailing Stop fallback)
    "risk_k_sl": 2.0,           # stop distance in ATR multiples
    "risk_k_trail_start": 1.0,  # profit distance to start trailing
    "risk_k_trail_dist": 1.0,   # trailing distance

    # Exhaustion model (hourly sensor)
    "exh_horizon_hours": 8,
    "exh_reversal_thr": 0.03,    # Relaxed from 0.04 to capture more reversals

    # Peak Targeting Labeling (Option A/B)
    "exh_label_type": "peak",    # "simple" or "peak"
    "exh_use_atr": True,         # Use ATR-based thresholds
    "exh_atr_rev_k": 1.5,        # Reversal size in ATRs (relaxed from 2.0)
    "exh_atr_near_k": 1.5,       # Proximity to peak in ATRs (relaxed from 1.0)

    # Clipping for Peak Targeting
    "exh_near_dist_cap_pct": 0.05, # Max proximity distance (5%) - relaxed for crypto volatility
    "exh_rev_dist_floor_pct": 0.01, # Min reversal distance (1%) - relaxed for crypto volatility

    "exh_near_thr": 0.03,       # Fallback % proximity
    "exh_rev_thr_pct": 0.02,     # Fallback % reversal (3%)

    # Soft Labels (Target Smoothing)
    "label_use_soft": True,
    "label_soft_alpha_max": 0.15, # Reverted to 0.15

    "exh_train_lookback_hours": 24 * 14,
    "min_exh_samples": 6000,
    "exh_C": 1.0,
    "exh_l1_ratio": 0.30,

    # which features go into exhaustion ML (plus sin/cos time features) (3)
    "exh_feature_keys": [
        "donch_dist_12", "excess_6h", "overext", "overext_weak", "effort_gate", "stall_ext", "tail_fail",
        "blowoff_risk",
        "clv_mean_4", "pullback_2", "pullback_4", "giveback", "evr_6", "progress",
        "delta_stall_6", "tail_against",
        # Context features (Volatility & Regime)
        "vol_z", "atr_pct", "rsi", "mkt_rv_ratio", "dist_vwap_norm", "accel",
        # Interaction Features
        "dist_ext_x_vol", "regime_x_vol", "rsi_x_vol"
    ],

    # Spike / Regime Head
    "spike_feature_keys": [
        "S", "impulse_ratio_12", "impulse_ratio_24", "coherence_24", "accel",
        "tf_minus_mr", "mkt_rv_ratio", "wick_ratio", "body_ratio", "rvol_z",
        "retrace_12", "donch_dist_12"
    ],

    # TF Head (Specifics + Global) — includes trend maturity features
    "tf_feature_keys": [
        "accept", "retest_accept", "tf_qual", "G_TF_TREND", "coherence_24", "impulse_ratio_24",
        "tf_tape", "clv_mean_4", "pullback_2", "pullback_4", "pullback_8", "ft_2", "ft_4", "ft_8",
        "vov_ratio", "vov_interaction", "vov_fast_slow_ratio", "accel_5h", "breakout_24h",
        "stage_tf", "tf_bias", "flow_persistence", "flow_ratio",
        "progress", "evr_6", "delta_stall_6", "rv_2h", "rv_4h",
        "dir_path_long_2h", "dir_path_short_2h", "dir_path_edge_2h",
        # Multi-day regime context (distinguish real trends from bear rallies)
        "donch_dist_48", "donch_dist_72", "donch_dist_120",
        "pullback_48", "pullback_72", "pullback_120",
        "dist_from_high_48h", "dist_from_high_120h",
        "trend_slope_48h", "trend_slope_120h", "trend_accel_120h",
        "rv_ratio_24_120", "ret48h", "ret120h",
        "accept_x_dir_edge_2h", "tfq_x_dir_edge_2h", "accept_dir2h_prod", "tfq_dir2h_prod",
        "downside_semivariance_8", "downside_semivariance_24",
        "upside_semivariance_8", "upside_semivariance_24",
        "down_up_vol_ratio_8", "down_up_vol_ratio_24",
        "vol_shock_asym_8_24", "vol_shock_asym_4_12", "vol_shock_asym_4_212",
        # OHLCV-based trend quality features (Report 2026-02-12)
        "trend_age_hours", "higher_highs_count_48h", "trend_retest_success_rate",
        "trend_overextension_z", "volume_trend_alignment", "trend_regime_stability",
        # New Indicators
        "ker_16", "adx_14", "adx_14_slope", "vortex_diff_21",
        "vp_air_pocket_score", "trapped_longs_96", "vp_dist_hvn_above_atr",
    ] + neutral_feature_keys + MODEL_FEATURES + HELPER_BASE_FEATURES,

    # MR Head (Specifics + Global) — includes exhaustion features
    "mr_feature_keys": [
        "accept", "accept_bin3", "overext", "overext_weak", "mr_qual", "retrace_12",
        "impulse_ratio_24", "coherence_24", "mr_tape",
        "clv_mean_4", "pullback_2", "pullback_4", "pullback_8", "ft_2", "ft_4", "ft_8",
        "giveback", "blowoff_risk", "exh_qual", "stage_blowoff", "stage_mr",
        "donch_dist_12", "donch_dist_8", "excess_6h", "tail_fail", "tail_against",
        "mfe_2h", "mae_2h", "mfe_4h", "mae_4h", "mfe_8h", "mae_8h",
        "rv_2h", "rv_4h", "dir_path_risk_long_2h", "dir_path_risk_short_2h",
        "reject_x_dir_edge_2h", "mrq_x_dir_edge_2h", "reject_dir2h_prod", "mrq_dir2h_prod",
        # Multi-day regime context (distinguish real dips from trend continuation)
        "donch_dist_48", "donch_dist_72", "donch_dist_120",
        "pullback_48", "pullback_72", "pullback_120",
        "dist_from_low_48h", "dist_from_low_120h",
        "trend_slope_48h", "trend_slope_120h", "trend_accel_120h",
        "rv_ratio_24_120", "ret48h", "ret120h",
        "downside_semivariance_8", "downside_semivariance_24",
        "upside_semivariance_8", "upside_semivariance_24",
        "down_up_vol_ratio_8", "down_up_vol_ratio_24",
        "vol_shock_asym_8_24", "vol_shock_asym_4_12", "vol_shock_asym_4_212",
        # OHLCV-based mean-reversion quality features (Report 2026-02-12)
        "trend_strength_vs_reversion", "support_quality_score", "dip_velocity",
        "dip_volume_profile", "reversion_target_distance",
        # Reversal mechanics and trap context
        "bounce_signal", "volume_capitulation", "trap_strength", "entry_quality_composite", "trap_quality",
        "mr_soft", "mr_potential", "mr_potential_exhaust", "climax", "vol_exhaust", "mr_climax",
        "shock_decay", "pct_extreme", "mr_pct", "stall", "mr_failure",
        "impulse_reversal", "impulse_reversal_short", "breakout_min", "pct_breakout_t",
        # New Indicators
        "trapped_longs_12", "dist_vwap_12_atr", "vp_dist_poc_atr", "vp_in_poc_zone",
        "vortex_diff_14", "adx_7",
    ] + neutral_feature_keys + MODEL_FEATURES + HELPER_BASE_FEATURES,

    # Meta Learner
    "meta_feature_keys": [
        "ambig", "stage_tf", "stage_blowoff", "stage_mr", "exh_qual",
        "accept", "accept_bin3", "accept_gt85", "rv_ratio_6_24",
        "G_TF_TREND", "vol_z_x_trend_t",
        "excess_6h", "donch_dist_12", "donch_dist_8", "clv_mean_4", "evr_6", "delta_stall_6",
        "ft_2", "asym_ratio", "mfe_2h", "mae_2h", "mfe_4h", "mae_4h", "mfe_8h", "mae_8h", "giveback",
        "ret1h", "ret2h", "ret4h", "ret6h", "rv_2h", "rv_4h", "rv_6h", "rv_8h", "rv_24h", "mkt_rv_ratio",
        "qv", "signed_vol", "vol_z", "atr_pct", "trend_pct",
        "spike_score", "grind_score", "chop_score",
        "G_META_EXH", "G_META_TF_QUAL", "G_META_MR_QUAL", "G_META_AMBIG",
        "vol_z_30_calm", "breakout_24h", "draw_extreme_10h",
        "meta_abs_net_x_breakout", "meta_abs_net_x_drawext", "meta_abs_net_x_vov_ratio",
        "meta_alignment", "meta_signal_x_accel",
        "kf_score_mean", "kf_score_rm24_mean", "kf_atr_mean", "kf_vol_ratio_mean", "kf_ret1h_mean",
        "kf_innov_var", "kf_snr_est", "kf_state_uncertainty",
        "vol_high", "vol_low", "cusum_strength_norm", "cusum_high", "liq_low",
        "p_vol_high", "p_cusum_high", "p_liq_low",
        "dir_path_long_2h", "dir_path_short_2h", "dir_path_risk_long_2h", "dir_path_risk_short_2h",
        "dir_path_edge_2h", "dir_path_risk_skew_2h",
        "accept_x_dir_edge_2h", "reject_x_dir_edge_2h", "tfq_x_dir_edge_2h", "mrq_x_dir_edge_2h",
        "accept_dir2h_prod", "accept_dir2h_abs_prod", "accept_dir2h_signed_mag",
        "reject_dir2h_prod", "reject_dir2h_abs_prod", "reject_dir2h_signed_mag",
        "tfq_dir2h_prod", "tfq_dir2h_abs_prod", "tfq_dir2h_signed_mag",
        "mrq_dir2h_prod", "mrq_dir2h_abs_prod", "mrq_dir2h_signed_mag",
        # Specialist features
        "trap_quality", "predicted_vol_6h",
        # Gated entry features
        "bounce_signal", "trap_strength", "volume_capitulation", "entry_quality_composite",
        # TF Meta Features (Report 2026-02-10)
        "trend_t", "trend_z_t", "convexity_t", "convexity_bis_t",
        "vw_breakout", "breakout_soft", "tail_score",
        # MR Meta Features (Report 2026-02-10)
        "mr_soft", "mr_potential", "mr_potential_exhaust",
        "climax", "vol_exhaust", "mr_climax", "shock_decay",
        "pct_extreme", "mr_pct", "stall", "mr_failure",
        # Multi-day regime context (meta learns regime-conditional weighting)
        "dist_from_high_48h", "dist_from_high_120h",
        "dist_from_low_48h", "dist_from_low_120h",
        "trend_slope_48h", "trend_slope_120h",
        "rv_ratio_24_120", "rv_48h", "rv_120h",
        "ret48h", "ret120h",
        "donch_dist_48", "donch_dist_120",
        # FFD d-specific features for meta learner
        "ffd_rv_2h_04", "ffd_rv_6h_04", "ffd_rv_24h_04",
        "ffd_vol_price_corr_10h_04",
        "ffd_donch_dist_04_12", "ffd_donch_dist_04_24", "ffd_donch_dist_04_48",
        "ffd_amihud_04", "ffd_vol_range_shock_04",
        "ffd_dist_ema_fast_04", "ffd_dist_ema_slow_04",
        "ffd_rv_2h_06", "ffd_rv_6h_06", "ffd_rv_24h_06",
        "ffd_accel_06", "ffd_z_06",
        "ffd_vol_price_corr_10h_06",
        "ffd_donch_dist_06_12", "ffd_donch_dist_06_24", "ffd_donch_dist_06_48",
        "ffd_atr_expansion_06", "ffd_cvar_5pct_06",
        "ffd_amihud_06", "ffd_vol_range_shock_06",
        # D-family strength indicators for meta
        "ffd_strength_04", "ffd_strength_05", "ffd_strength_06",
        # Asset identity features (raw-scale, not normalized)
        "asset_atr_level", "asset_vol_level", "atr_state", "vol_state",
        # New Indicators
        "vp_profile_concentration", "vp_profile_entropy", "vp_lvn_depth_ratio",
        "adx_14_slope", "trapped_longs_96",
        # New Liquidity features
        "liq_score", "rng_z", "impact_z",
        # Event-timing + policy-normalized stage features (entry-time only)
        "time_since_peak_12h", "time_since_trough_12h", "time_since_event_extreme_12h",
        "second_leg_accel_1h", "second_leg_accel_2h", "second_leg_accel_vol_1h", "second_leg_accel_vol_2h",
        "vol_scale", "be_vol_units", "pl_vol_units", "trail_act_pct", "trail_act_vol_units", "giveback_vol_units",
        "t_be_proxy", "t_pl_proxy", "t_trail_proxy",
        "shock_12h", "shock_vol_ratio", "dist_from_low_event_12h", "dist_from_high_event_12h",
        "dist_from_low_vol", "dist_from_high_vol",
        # Regime/liquidity/complexity features for meta utility/IC robustness
        "amihud_z", "amihud_illiq", "liq_regime", "rvol_hod_base", "mkt_rv_pct",
        "vol_regime_z", "regime_stability_24h", "regime_transition_entropy_12h",
        "regime_transition_entropy_48h", "complexity_regime_24h",
        "trend_regime_switch_12h", "vol_regime_switch_12h",
        "vol_concentration_12", "volume_entropy_12", "volume_entropy_24", "volatility_zscore",
        "clv_t", "body_ratio_15m", "rejection_proxy",
        "range_norm_12", "sv_imb_12", "press_12", "impact_12", "ts_12", "prog_eff_12", "pers_12", "hh_count_12", "ll_count_12", "skew_12", "climax_range_12", "climax_vol_12", "z_vwap_12", "z_r_12", "bb_pos_12",
        "range_norm_24", "sv_imb_24", "press_24", "impact_24", "ts_24", "prog_eff_24", "pers_24", "hh_count_24", "ll_count_24", "skew_24", "climax_range_24", "climax_vol_24", "z_vwap_24", "z_r_24", "bb_pos_24",
    ],
    # Kind-specific overlays for meta models (added on top of meta_feature_keys)
    "mr_meta_feature_keys": [
        "impulse_reversal", "impulse_reversal_short", "breakout_min", "pct_breakout_t",
        "dip_velocity", "dip_volume_profile", "support_quality_score", "reversion_target_distance",
        "trend_strength_vs_reversion", "mr_tape",
        # MR path quality and trap filtering
        "vol_compression", "climax_decay", "shock_rel", "vol_price_diverge",
        "rsi_z_x_regime_vol", "down_up_vol_ratio_8", "down_up_vol_ratio_24",
        "vol_shock_asym_4_12", "vol_shock_asym_8_24", "draw_sym_10h", "atr_pct_change",
    ],
    "tf_meta_feature_keys": [
        "tf_tape", "tf_minus_mr", "accept_gt66", "retest_accept", "tf_qual",
        "breakout_confirmed", "trend_retest_success_rate", "trend_regime_stability",
        "trend_age_hours", "higher_highs_count_48h",
        # TF continuation quality
        "trend_regime", "is_trending", "trend_snr", "trend_overextension_z",
        "volume_trend_alignment", "impulse_ratio_24", "breakout_t",
        "vol_expansion_ratio", "vol_z_x_regime_trend",
    ],
    # Selector v3 configs (top30-focused, per-head)
    "selector_feature_family_map": {},
    "base_selector_cfg": {
        "selector_focus_top_frac": 0.30,
        "selector_top_metric": None,
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
        "top30": 0.35,
        "global": 0.20,
        "stability": 0.25,
        "frequency": 0.15,
        "interaction": 0.05,
    },
    "meta_selector_cfg": {
        "selector_focus_top_frac": 0.30,
        "selector_top_metric": "ic_top",
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
        "top30": 0.35,
        "global": 0.20,
        "stability": 0.25,
        "frequency": 0.15,
        "interaction": 0.05,
    },
    "aux_mae_selector_cfg": {
        "selector_focus_top_frac": 0.40,
        "selector_top_metric": "ic_top",
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
        "top30": 0.20,
        "global": 0.35,
        "stability": 0.25,
        "frequency": 0.15,
        "interaction": 0.05,
    },
    "aux_mfe_selector_cfg": {
        "selector_focus_top_frac": 0.25,
        "selector_top_metric": "ic_top",
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
        "top30": 0.40,
        "global": 0.15,
        "stability": 0.25,
        "frequency": 0.15,
        "interaction": 0.05,
    },
    "aux_utility_selector_cfg": {
        "selector_focus_top_frac": 0.30,
        "selector_top_metric": "top30_mean_utility",
        "selector_emit_report": True,
        "top30": 0.30,
        "global": 0.15,
        "stability": 0.30,
        "frequency": 0.20,
        "interaction": 0.05,
    },
    # Backward-compatible selector aliases
    "base_mdi_selector_target": "classification",
    "base_mdi_selector_loss": "binary_logloss",

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
    "allow_5m_download": True,   # Use 5m only for residual ambiguity solving after 15m refinement
    
    # Limit Order Simulation (Report 2026-02-22)
    "use_limit_orders": True,    # Enable limit orders per user request
    "limit_offset_bps": 20.0,    # 0.2% entry offset (buy lower / sell higher)
    "exit_limit_offset_bps": 20.0,  # 0.2% exit offset for testing
    "signal_opt_debug": True,    # Emit detailed signal-optimization diagnostics
    "debug_signal_generation": True,  # Emit per-timestamp signal-generation stage counts
    "fee_bps": 35.0,  # Default fee (used when not using limit orders)
    
    # Fee Structure (Market vs Limit)
    "fee_bps_market": 25.0,      # 0.25% per side for market orders (50 bps RT)
    "fee_bps_limit_entry": 10.0, # 0.10% per side for limit order entry (20 bps RT)
    "fee_bps_limit_exit": 10.0,  # 0.10% per side for limit order exit (20 bps RT)
    "fee_bps_market_exit": 25.0, # 0.25% per side if using market order for exit
    
    # Limit Order Price Estimation (MAE/MFE-based)
    "use_mae_mfe_limit_offset": True,  # Use MAE/MFE predictions for limit offset
    "limit_offset_min_bps": 5.0,       # Minimum limit offset in bps
    "limit_offset_max_bps": 50.0,      # Maximum limit offset in bps
    "limit_fill_model_type": "heuristic",  # heuristic | learned
    "limit_fill_vol_regime_weight": 0.3,  # How much vol regime reduces fill prob
    "limit_fill_liquidity_bonus": 0.2,    # Liquidity adjustment to fill prob
    
    # Exit Limit Orders
    "use_exit_limit_orders": True,    # Enable limit orders for exits
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


def enable_perp_feature_keys(cfg: dict) -> dict:
    """
    Enable perp-specific features for runtime config.
    Spot pipeline remains unchanged unless this helper is called.
    """
    out = dict(cfg)
    for k in ("tf_feature_keys", "mr_feature_keys", "meta_feature_keys"):
        out[k] = _append_missing(out.get(k, []), PERP_FEATURE_KEYS)
    return out


def apply_15m_feature_toggle(cfg: dict) -> dict:
    """Apply 15m OHLCV feature family toggle to runtime feature key lists."""
    out = dict(cfg)
    target_lists = (
        "causal_cols",
        "tf_feature_keys",
        "mr_feature_keys",
        "meta_feature_keys",
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

# ============================================================
# Position Sizer V2 Feature Config
# ============================================================

POSITION_SIZER_V2_FEATURE_CONFIG = {
    "shared_feature_keys": [
        "oof_base_mean", "oof_base_std", "oof_base_min", "oof_base_max", "oof_base_range",
        "oof_meta_pred", "oof_meta_minus_base_mean", "oof_top2_gap", "oof_sign_agreement_frac",
        "oof_rank_among_candidates",
        "ret_1", "ret_3", "ret_6", "ret_12", "ret_24",
        "price_vs_ema_12_z", "price_vs_ema_24_z", "ema_12_minus_ema_24_z",
        "trend_slope_12_z", "trend_slope_24_z",
        "atr_pct", "range_1_atr", "range_3_atr", "rv_6", "rv_12", "rv_24", "rv_ratio_6_24",
        "close_location_in_bar",
        "volume_z_12", "volume_z_24", "dollar_vol_z_24", "spread_pct", "spread_to_atr",
        "cost_to_atr", "slippage_proxy", "liquidity_shock_z",
        "regime_trend_score", "regime_vol_score", "regime_liquidity_score",
        "session_progress", "hour_sin", "hour_cos", "dow_sin", "dow_cos"
    ],
    "model1_edge_feature_keys": [
        "oof_base_mean", "oof_base_std", "oof_meta_pred", "oof_top2_gap",
        "oof_sign_agreement_frac", "ret_3", "ret_6", "ret_12", "price_vs_ema_12_z",
        "ema_12_minus_ema_24_z", "atr_pct", "rv_ratio_6_24", "spread_to_atr",
        "cost_to_atr", "regime_trend_score", "regime_vol_score", "session_progress"
    ],
    "model2_downside_feature_keys": [
        "oof_base_mean", "oof_base_std", "oof_meta_pred", "ret_1", "ret_3",
        "close_location_in_bar", "range_1_atr", "atr_pct", "rv_6", "rv_24",
        "downside_semivol_12", "spread_to_atr", "slippage_proxy",
        "regime_vol_score", "regime_liquidity_score", "session_progress"
    ],
    "model3_uncertainty_feature_keys": [
        "oof_base_std", "oof_base_range", "oof_meta_minus_base_mean",
        "oof_sign_agreement_frac", "edge_pred", "downside_pred",
        "edge_minus_downside", "abs_edge_pred", "atr_pct", "rv_ratio_6_24",
        "spread_to_atr", "cost_to_atr", "liquidity_shock_z",
        "regime_vol_score", "regime_liquidity_score", "session_progress"
    ]
}

POSITION_SIZER_V2_FEATURE_FLAGS = {
    "enable_model1_optional": False,
    "enable_model2_optional": False,
    "enable_model3_optional": False,
}
