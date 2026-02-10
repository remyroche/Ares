# Central config. Keep it deterministic and explicit.

neutral_feature_keys = [
    "rsi", "vol_z", "atr_pct", "mkt_rv_ratio", "skew", 
    "trend_snr", "efficiency", "vol_asym", "momentum_accel",
    "dist_stack", "stage_blowoff", "exh_qual", "volatility_zscore"
]

MODEL_FEATURES = [
    # Momentum / structure extensions
    "thrust_decay_4", "decel_4", "ft_drop", "ext_excess", "ext_atrExp",
    "comp_to_exp", "evr6_x_volz", "stall_x_flow", "prog_def",
    "clv_collapse", "clv_pullback", "coh", "align", "retest_quality",
    "pb_accel", "rv_ratio_6_24", "excess_coh", "asym_ft",
    "tf_bias", "shock_rel", "resid_strength", "evr_slope", "stall_ext",
    # Earlier trend following / volatility-of-volatility signals
    "vov_iqr_20", "vov_mad_20", "vov_mad_60", "vov_ratio", "vov_interaction",
    "vov_fast_slow_ratio", "accel_5h", "dlog_vol_5h", "signed_max_bar_ret_5h",
    "jump_rate_10h", "volu_z", "volume_price_corr_10h", "draw_sym_10h", "draw_extreme_10h",
    "breakout_24h",
    "meta_abs_net_x_breakout", "meta_abs_net_x_drawext", "meta_abs_net_x_vov_ratio",
    "meta_alignment", "meta_signal_x_accel",
    # Price Action
    "gap_pct", "range_pct", "roc_div", "ret1h_z", "body_pct", "wick_body_ratio",
    "vol_price_spread", "wick_ratio", "body_ratio",
    # New Risk/Exhaustion (Report 2026-02-10)
    "wick_ratio_4h_max", "vol_price_div", "rsi_lag1", "rsi_1h_slope",
    "cvar_5pct", "amihud_illiq", "clv_mean_24", "vol_z_4h", "atr_pct_change",
    # Alpha Features (Report 2026-02-10)
    "breakout_min", "impulse_reversal", "impulse_reversal_short",
    "breakout_confirmed", "breakout_t", "pct_breakout_t",
    # Volume/Flow
    "v_power", "flow_persistence", "flow_ratio", "churn",
    "vol_range_shock", "climax_decay", "cumulative_delta_stall",
    "vol_expansion_ratio", "vol_compression", "rvol_z",
    # Advanced
    "fvg", "slope", "atr_slope", "dist_vwap_norm", "rsi_slope",
    "funding_proxy", "dist_ema_fast",
    # Scores
    "spike_score", "grind_score", "chop_score",
    # Time
    "sin_hod", "cos_hod", "sin_dow", "cos_dow",
]

# Helper/base features produced in features.py that should remain selectable by model heads.
# This increases candidate breadth before MDI pruning.
HELPER_BASE_FEATURES = [
    "ret1h", "ret6h", "atr_pct_base", "rsi_base", "rsi_slope_base",
    "rv_6h", "rv_12h", "rv_24h", "qv", "vol_z24_base", "vol_z_base",
    "dist_ema_fast_base", "dist_ema_slow_base", "trend_pct_base",
    "rvol_hod_base", "signed_vol", "up_vol", "dn_vol", "up_vol_6", "dn_vol_6",
    "vol_asym_6", "clv", "clv_mean_2", "excess_12h", "speed",
    "atr_expansion", "stall_ext_corr",
    "G_EXH_EFFORT", "G_EXH_GIVEBACK", "G_EXH_TAIL_FAIL",
    "G_MR_SPIKE", "G_TF_GRIND", "G_MR_TAIL",
    "G_META_EXH", "G_META_TF_QUAL", "G_META_MR_QUAL", "G_META_AMBIG",
]

CFG = {
    # persistence / fetch
    "data_root": "data",
    "timeframe": "1h",
    "fetch_years": 3,
    "fetch_symbols_M": 500,

    # market basket
    "market_basket": ["BTC/USDT","ETH/USDT","AVAX/USDT","SOL/USDT","XRP/USDT"],

    # training horizons to compare (1)
    "label_horizons_hours": [2, 4, 8],
    "label_tp_values_pct": [5.0, 3.5, 2.0],
    "label_sl_values_pct": [0.5, 1.0, 2.0],
    "label_round_trip_fee_pct": 0.5,
    "label_min_net_rr": 1.5,
    "label_min_tp_hit_rate": 0.02,
    "label_max_timeout_rate": 0.90,

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
    "train_lookback_hours": 24 * 365 * 3,   # 3 years
    "val_lookback_hours": 24 * 7,      # 7d validation (time-split, no leakage)
    "min_train_samples": 200,

    # per-hour cross-sectional training selection
    "variance_filter_pct": 1.0, # User requested to keep all non-constant features
    "train_extreme_pct_hourly": 0.05,
    "train_extreme_min": 10,
    "train_extreme_max": 80,

    # hourly trading selection (top/bot deviations)
    "trade_extreme_pct": 0.06,
    "trade_extreme_min": 10,
    "trade_extreme_max": 80,
    "trade_deviation_metric": "dist_ema_fast",

    # gates
    "gate_vol_lookback_hours": 24 * 14,
    "gate_trend_thr": 0.02,
    "accept_gate_window": 64,
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
    "min_feat_sign_consistency": 0.70,
    "min_model_stability_to_trade": 0.15,

    # causal cols for interaction toggles
    # Added new features for TF/MR/Meta
    "drop_raw_causal": True,
    "causal_cols": [
        "ret24h", "rsi", "vol_z", "atr_pct", "trend_pct", "rv_24h",
        "p_exh_lag1",
        "a_funding_proxy",
        "flow_ratio", "churn", "slope", "trend_snr",
        "vol_asym", "skew", "efficiency", "fvg",
        "rvol_z", "vol_range_shock", "climax_decay",
        "cumulative_delta_stall", "vol_expansion_ratio", "vol_compression",
        "atr_slope", "dist_vwap_norm", "momentum_accel",
        # New Exhaustion/Risk features (Report 2026-02-10)
        "wick_ratio_4h_max", "vol_price_div", "rsi_lag1", "rsi_1h_slope",
        "cvar_5pct", "amihud_illiq", "clv_mean_24", "vol_z_4h", "atr_pct_change",
        # New Feature Candidates
        "thrust_decay_4", "decel_4", "ft_drop", "ext_excess", "ext_atrExp",
        "comp_to_exp", "evr6_x_volz", "stall_x_flow", "prog_def",
        "clv_collapse", "clv_pullback", "coh", "align", "retest_quality",
        "pb_accel", "rv_ratio_6_24", "excess_coh", "asym_ft", "dist_stack",
        "tf_bias", "shock_rel", "resid_strength", "evr_slope", "stall_ext",
        "spike_score", "grind_score", "chop_score",
        # Gates as continuous features
        "G_EXH_EFFORT", "G_EXH_GIVEBACK",
        "G_EXH_TAIL_FAIL",
        "G_MR_SPIKE", "G_TF_GRIND", "G_MR_TAIL",
        "G_META_EXH", "G_META_TF_QUAL", "G_META_MR_QUAL", "G_META_AMBIG",
        # New Model Features
        "overext", "overext_weak", "effort_gate", "tail_fail", "blowoff_risk",
        "S", "impulse_ratio_24", "impulse_ratio_12", "coherence_24", "accel",
        "tf_tape", "mr_tape", "accept", "accept_bin3", "accept_gt66", "accept_gt85", "retest_accept", "tf_qual", "mr_qual",
        "retrace_12", "ambig", "stage_tf", "stage_blowoff", "stage_mr", "exh_qual"
        ,"vov_iqr_20", "vov_mad_20", "vov_mad_60", "vov_ratio", "vov_interaction",
        "vov_fast_slow_ratio", "accel_5h", "dlog_vol_5h", "signed_max_bar_ret_5h",
        "jump_rate_10h", "volu_z", "volume_price_corr_10h", "draw_sym_10h", "breakout_24h",
        "vol_z_30_calm",
        "meta_abs_net_x_breakout", "meta_abs_net_x_drawext", "meta_abs_net_x_vov_ratio",
        "meta_alignment", "meta_signal_x_accel",
        # Orthogonal features (structurally independent dimensions)
        "mtf_divergence", "mtf_div_mag",
        "autocorr_6h", "autocorr_24h",
        "path_efficiency_12", "path_efficiency_24",
        "hurst_proxy_24",
        "vol_concentration_12",
        "vol_price_diverge",
        # Residualised features — relative surprise, not absolute magnitude
        "rsi_z", "dist_ema_fast_z", "dist_vwap_norm_z", "flow_persistence_z",
        "excess_6h_z", "vol_z_z", "atr_expansion_z", "coherence_24_z",
        "accept_surprise", "overext_surprise",
        "blowoff_risk_surprise", "exh_qual_surprise",
        "dist_vwap_resid", "dist_ema_fast_resid", "trend_pct_resid",
        "mkt_rv_pct", "abs_mkt_ret24h_z", "trend_bin3",
        "s_z_64", "s_pct_64", "s_bin3_64", "s_gt66_64", "s_gt85_64",
        "s_z_8", "s_pct_8", "s_bin3_8", "s_gt66_8", "s_gt85_8",
        "reject_z_64", "reject_pct_64", "reject_bin3_64",
        "reject_z_8", "reject_pct_8", "reject_bin3_8",
        "retest_accept_z_64", "retest_accept_pct_64", "retest_accept_bin3_64",
        "retest_accept_z_8", "retest_accept_pct_8", "retest_accept_bin3_8",
        "tf_qual_z_64", "tf_qual_pct_64", "tf_qual_bin3_64",
        "tf_qual_z_8", "tf_qual_pct_8", "tf_qual_bin3_8",
        "mr_qual_z_64", "mr_qual_pct_64", "mr_qual_bin3_64",
        "mr_qual_z_8", "mr_qual_pct_8", "mr_qual_bin3_8",
        "accept_score_cs_trimmed_mean", "accept_score_cs_median", "accept_score_cs_p75", "accept_score_cs_p90", "accept_score_cs_iqr", "accept_score_cs_std",
        "reject_score_cs_trimmed_mean", "reject_score_cs_median", "reject_score_cs_p75", "reject_score_cs_p90", "reject_score_cs_iqr", "reject_score_cs_std",
        "tf_qual_score_cs_trimmed_mean", "tf_qual_score_cs_median", "tf_qual_score_cs_p75", "tf_qual_score_cs_p90", "tf_qual_score_cs_iqr", "tf_qual_score_cs_std",
        "mr_qual_score_cs_trimmed_mean", "mr_qual_score_cs_median", "mr_qual_score_cs_p75", "mr_qual_score_cs_p90", "mr_qual_score_cs_iqr", "mr_qual_score_cs_std"
    ],

    # thresholds / picks
    "thr_long":  0.010,
    "thr_short": -0.010,
    "k_long": 10,
    "k_short": 10,

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

    # OOS holdout for backtest
    "oos_holdout_days": 180,    # Exclude last 6 months from training for OOS backtest

    # Trailing Profit Risk Params (used in backtest & live, all vol-scaled)
    # Target absolute: TP ~2%, SL ~0.7% (with median barrier_pct ~4%)
    "tp_mult": 0.50,            # Activation threshold = tp_mult * barrier_pct (~2%)
    "sl_mult": 0.18,            # Stop-loss = sl_mult * barrier_pct (~0.7%)
    "trail_mult": 0.25,         # Trailing deviation = trail_mult * barrier_pct
    # Hard constraints enforced in optimizer and defaults
    "min_tp_sl_ratio": 1.5,     # TP:SL ratio must be >= 1.5
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
    "exh_near_dist_cap_pct": 0.02, # Max proximity distance (2%)
    "exh_rev_dist_floor_pct": 0.005, # Min reversal distance (0.5%) - User asked for 5% but using 0.5% as safe default

    "exh_near_thr": 0.015,       # Fallback % proximity
    "exh_rev_thr_pct": 0.03,     # Fallback % reversal (3%)

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
        "accept", "retest_accept", "tf_qual", "coherence_24", "impulse_ratio_24",
        "tf_tape", "clv_mean_4", "pullback_2", "pullback_4", "ft_2", "ft_4",
        "vov_ratio", "vov_interaction", "vov_fast_slow_ratio", "accel_5h", "breakout_24h",
        "stage_tf", "tf_bias", "flow_persistence", "flow_ratio",
        "progress", "evr_6", "delta_stall_6"
    ] + neutral_feature_keys + MODEL_FEATURES + HELPER_BASE_FEATURES,

    # MR Head (Specifics + Global) — includes exhaustion features
    "mr_feature_keys": [
        "accept", "accept_bin3", "overext", "overext_weak", "mr_qual", "retrace_12",
        "impulse_ratio_24", "coherence_24", "mr_tape",
        "clv_mean_4", "pullback_2", "pullback_4", "ft_2", "ft_4",
        "giveback", "blowoff_risk", "exh_qual", "stage_blowoff", "stage_mr",
        "donch_dist_12", "excess_6h", "tail_fail", "tail_against",
        "mfe_4h", "mae_4h"
    ] + neutral_feature_keys + MODEL_FEATURES + HELPER_BASE_FEATURES,

    # Meta Learner
    "meta_feature_keys": [
        "ambig", "stage_tf", "stage_blowoff", "stage_mr", "exh_qual",
        "accept", "accept_bin3", "accept_gt85", "rv_ratio_6_24",
        "excess_6h", "donch_dist_12", "clv_mean_4", "evr_6", "delta_stall_6",
        "ft_2", "asym_ratio", "mfe_4h", "mae_4h", "giveback",
        "ret1h", "ret6h", "rv_6h", "rv_24h", "mkt_rv_ratio",
        "qv", "signed_vol", "vol_z", "atr_pct", "trend_pct",
        "spike_score", "grind_score", "chop_score",
        "G_META_EXH", "G_META_TF_QUAL", "G_META_MR_QUAL", "G_META_AMBIG",
        "vol_z_30_calm", "breakout_24h", "draw_extreme_10h",
        "meta_abs_net_x_breakout", "meta_abs_net_x_drawext", "meta_abs_net_x_vov_ratio",
        "meta_alignment", "meta_signal_x_accel",
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
    ],

    # Inference dynamic-basket controls
    "inference_event_window_hours": 12,
    "inference_event_threshold": 0.07,
    "inference_perf_pct": 0.10,
    "inference_draw_window_hours": 8,
    "inference_sign_consistency_min": 0.80,
    "inference_basket_ttl_hours": 24,
    
    # High-frequency simulation
    "use_15m_precision": True,  # Enable 15m OHLCV for trailing profit (requires CCXT exchange)
    
    # Risk logging
    "verbose_risk_logging": False,  # Enable detailed per-trade TP/SL logging

}
