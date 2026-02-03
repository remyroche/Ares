# Central config. Keep it deterministic and explicit.
CFG = {
    # persistence / fetch
    "data_root": "data",
    "timeframe": "1h",
    "fetch_years": 4,
    "fetch_symbols_M": 300,

    # market basket
    "market_basket": ["BTC/USDT","ETH/USDT","AVAX/USDT","SOL/USDT","XRP/USDT"],

    # training horizons to compare (1)
    "label_horizons_hours": [12, 16, 20, 24, 28],
    "train_lookback_hours": 24 * 30,   # 30d
    "val_lookback_hours": 24 * 7,      # 7d validation (time-split, no leakage)
    "min_train_samples": 8000,

    # per-hour cross-sectional training selection
    "train_extreme_pct_hourly": 0.05,
    "train_extreme_min": 10,
    "train_extreme_max": 80,

    # hourly trading selection (top/bot deviations)
    "trade_extreme_pct": 0.05,
    "trade_extreme_min": 10,
    "trade_extreme_max": 80,
    "trade_deviation_metric": "dist_ema_fast",

    # gates
    "gate_vol_lookback_hours": 24 * 14,
    "gate_trend_thr": 0.02,

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
    # Added new meta features here to ensure they are carried over
    "drop_raw_causal": True,
    "causal_cols": [
        "a_ret24h","a_rsi","a_volz","a_atr","a_trend","a_rv24",
        "p_exh_lag1",
        "a_funding_proxy",
        "flow_ratio", "churn", "slope", "trend_snr",
        "vol_asym", "skew", "efficiency", "fvg",
        "rvol_z", "vol_range_shock", "climax_decay",
        "cumulative_delta_stall", "vol_expansion_ratio", "vol_compression",
        "atr_slope", "dist_vwap_norm", "momentum_accel"
    ],

    # thresholds / picks
    "thr_long":  0.010,
    "thr_short": -0.010,
    "k_long": 10,
    "k_short": 10,

    # sizing / risk / costs
    "wallet_gross_cap": 0.25,
    "score_map": "tanh",
    "score_scale": 15.0,
    "tp": 0.06,
    "sl": 0.04,
    "hold_hours": 48,
    "fee_bps": 10.0,
    "borrow_apr": 0.20,

    # New Risk Params (Trailing Stop)
    "risk_k_sl": 2.0,           # stop distance in ATR multiples
    "risk_k_trail_start": 1.0,  # profit distance to start trailing
    "risk_k_trail_dist": 1.0,   # trailing distance

    # Exhaustion model (hourly sensor)
    "exh_horizon_hours": 24,
    "exh_reversal_thr": 0.04,
    "exh_train_lookback_hours": 24 * 14,
    "min_exh_samples": 6000,
    "exh_C": 1.0,
    "exh_l1_ratio": 0.30,

    # which features go into exhaustion ML (plus sin/cos time features) (3)
    "exh_feature_keys": [
        "ret1h", "ret6h", "ret12h", "ret16h", "ret20h", "ret24h", "ret28h",
        "ret1h_z",
        "vol_z24", "rvol_hod_base",
        "range_pct", "atr_expansion",
        "wick_body_ratio", "body_pct",
        "dist_ema_fast", "dist_ema_slow",
        "roc_div",
        "vol_price_spread",
        "rsi", "rsi_slope",
        "a_funding_proxy",
        "sin_hod", "cos_hod", "sin_dow", "cos_dow",
        "efficiency", "v_power", "skew", "fvg",
        "rvol_z", "vol_range_shock", "climax_decay",
        "cumulative_delta_stall", "vol_expansion_ratio", "vol_compression"
    ],

    # Model Params (Lasso selection)
    "lasso_alpha": 0.001,
}

def update_config_for_mode(mode="standard"):
    if mode == "light":
        CFG["fetch_years"] = 0.5
        CFG["fetch_symbols_M"] = 50
    elif mode == "standard":
        CFG["fetch_years"] = 4
        CFG["fetch_symbols_M"] = 300
