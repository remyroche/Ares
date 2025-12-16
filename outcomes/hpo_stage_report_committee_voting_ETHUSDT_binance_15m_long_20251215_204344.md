# HPO Stage Report: committee_voting\n\n- symbol: ETHUSDT\n- exchange: binance\n- timeframe: 15m\n- direction: long\n- run_timestamp: 20251215_204344\n\n## Best params\n```json\n{
  "w_scalp": 1.0,
  "w_swing": 0.5,
  "w_trend": 0.5,
  "w_breakout": 1.0,
  "w_vwap_rev": 2.0,
  "w_vol_shock": 0.5,
  "consensus_quantile": 0.99,
  "consensus_threshold": 0.05,
  "abstain_margin": 0.25,
  "diversity_lambda": 0.0,
  "regime_threshold_sensitivity": 1.0
}\n```\n\n## Metrics\n```json\n{
  "loaded_from": null
}\n```\n\n## Search space\n```json\n{
  "w_scalp": {
    "type": "float",
    "low": 0.0,
    "high": 2.0
  },
  "w_swing": {
    "type": "float",
    "low": 0.0,
    "high": 2.0
  },
  "w_trend": {
    "type": "float",
    "low": 0.0,
    "high": 2.0
  },
  "w_breakout": {
    "type": "float",
    "low": 0.0,
    "high": 2.0
  },
  "w_vwap_rev": {
    "type": "float",
    "low": 0.0,
    "high": 2.0
  },
  "w_vol_shock": {
    "type": "float",
    "low": 0.0,
    "high": 2.0
  },
  "consensus_threshold": {
    "type": "float",
    "low": 0.05,
    "high": 0.95
  },
  "abstain_margin": {
    "type": "float",
    "low": 0.0,
    "high": 0.5
  },
  "diversity_lambda": {
    "type": "float",
    "low": 0.0,
    "high": 2.0
  },
  "regime_threshold_sensitivity": {
    "type": "float",
    "low": 0.0,
    "high": 2.0
  },
  "consensus_quantile": {
    "type": "float",
    "low": 0.5,
    "high": 0.99
  }
}\n```\n\n## Trial artifacts\n- trials_csv: None\n- history_json: None\n\n