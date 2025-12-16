# HPO Stage Report: layer2_trading\n\n- symbol: ETHUSDT\n- exchange: binance\n- timeframe: 15m\n- direction: long\n- run_timestamp: 20251215_223051\n\n## Best params\n```json\n{
  "profit_floor_tx_mult": 2.1236203565420873,
  "sl_atr_mult": 2.87678576602479,
  "risk_reward_ratio": 3.9279757672456204,
  "horizon_bars": 31,
  "min_event_spacing": 1,
  "trail_distance_atr_mult": 0.8899863008405067,
  "prob_threshold": 0.5116167224336399,
  "ev_margin": 0.2165440364437338,
  "volatility_penalty_lambda": 0.1502787529358022,
  "barrier_regime_strength": 0.7080725777960455,
  "barrier_regime_power": 0.5308767414437037,
  "sig_strength_sensitivity": 0.29097295564859826,
  "trail_trend_modulation": 1.6648852816008435,
  "barrier_trend_asymmetry": 0.31850866601741423,
  "horizon_volume_modulation": 0.36364993441420124,
  "barrier_vol_vol_exp": 0.2751067647801507,
  "moe_trend_dominance": 0.3042422429595377,
  "moe_scalp_dominance": 0.5247564316322378,
  "moe_vol_sensitivity": 0.43194501864211576,
  "moe_adx_trend_q": 0.6664916560792168,
  "moe_adx_chop_q": 0.2947411578889518,
  "moe_vol_spike_q": 0.740453219589092,
  "prob_stop_enable": 0,
  "prob_stop_threshold": 0.6965447373174767,
  "prob_stop_drift_window": 50
}\n```\n\n## Metrics\n```json\n{
  "best_score": -1.0
}\n```\n\n## Search space\n```json\n{
  "profit_floor_tx_mult": {
    "type": "float",
    "low": 1.0,
    "high": 4.0
  },
  "sl_atr_mult": {
    "type": "float",
    "low": 0.5,
    "high": 3.0
  },
  "risk_reward_ratio": {
    "type": "float",
    "low": 1.0,
    "high": 5.0
  },
  "horizon_bars": {
    "type": "int",
    "low": 6,
    "high": 48
  },
  "min_event_spacing": {
    "type": "int",
    "low": 0,
    "high": 6
  },
  "trail_distance_atr_mult": {
    "type": "float",
    "low": 0.5,
    "high": 3.0
  },
  "prob_threshold": {
    "type": "float",
    "low": 0.5,
    "high": 0.7
  },
  "ev_margin": {
    "type": "float",
    "low": 0.0,
    "high": 0.25
  },
  "volatility_penalty_lambda": {
    "type": "float",
    "low": 0.0,
    "high": 0.25
  },
  "barrier_regime_strength": {
    "type": "float",
    "low": 0.0,
    "high": 1.0
  },
  "barrier_regime_power": {
    "type": "float",
    "low": 0.5,
    "high": 2.0
  },
  "sig_strength_sensitivity": {
    "type": "float",
    "low": 0.0,
    "high": 0.3
  },
  "trail_trend_modulation": {
    "type": "float",
    "low": 0.0,
    "high": 2.0
  },
  "barrier_trend_asymmetry": {
    "type": "float",
    "low": 0.0,
    "high": 1.5
  },
  "horizon_volume_modulation": {
    "type": "float",
    "low": 0.0,
    "high": 2.0
  },
  "barrier_vol_vol_exp": {
    "type": "float",
    "low": 0.0,
    "high": 1.5
  },
  "moe_trend_dominance": {
    "type": "float",
    "low": 0.0,
    "high": 1.0
  },
  "moe_scalp_dominance": {
    "type": "float",
    "low": 0.0,
    "high": 1.0
  },
  "moe_vol_sensitivity": {
    "type": "float",
    "low": 0.0,
    "high": 1.0
  },
  "moe_adx_trend_q": {
    "type": "float",
    "low": 0.55,
    "high": 0.95
  },
  "moe_adx_chop_q": {
    "type": "float",
    "low": 0.05,
    "high": 0.45
  },
  "moe_vol_spike_q": {
    "type": "float",
    "low": 0.7,
    "high": 0.99
  },
  "prob_stop_enable": {
    "type": "int",
    "low": 0,
    "high": 1
  },
  "prob_stop_threshold": {
    "type": "float",
    "low": 0.55,
    "high": 0.95
  },
  "prob_stop_drift_window": {
    "type": "int",
    "low": 12,
    "high": 96
  }
}\n```\n\n## Trial artifacts\n- trials_csv: None\n- history_json: outcomes/hpo_layer2_history_ETHUSDT_15m_20251215_223051.json\n\n