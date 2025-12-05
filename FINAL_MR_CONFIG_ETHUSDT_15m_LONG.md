# Final MR ML Config – ETHUSDT 15m Long

- **Metrics source file**: `ml_mean_reversion_metrics_ETHUSDT_15m_long_20251204_001526.csv`
- **Instrument / timeframe / side**: ETHUSDT, 15m, long
- **Regime prevalence**: ~4–5% MR regime

## Model quality

- **Teacher**: balanced regime classifier (used as reference)
- **Student (production MR classifier)**:
  - AUC ≈ 0.57
  - Accuracy ≈ 0.56 (calibrated)
  - MR-regime recall: strong (prioritised over precision given ~4–5% base rate)
  - 1–3h directional accuracy ≈ 0.54–0.57
  - Forward-return structure: stable and monotone – more signal over longer horizons

## Trading configuration (selected grid)

- **Grid config**: `tp=2.200%, sl=0.300%, conf_top=50%, max_hold=24`
- **Net return**: ≈ 11% over evaluation window
- **Trades**: ≈ 47 trades
- **Activity regime**: healthy – neither too sparse nor hyper-active
- **Risk-adjusted performance**: high Sharpe (see metrics file for exact value)

## Decision

- Adopt this MR ML configuration as the **default ETHUSDT 15m long mean-reversion step**.
- Use the grid above as the starting production config; future sweeps can refine TP/SL and confidence cutoffs if regime dynamics change.
