# Final Volume Force ML Config  ETHUSDT 15m Long

- **Metrics source files**: `outcomes/volume_force_sweep_ETHUSDT_20251204_210106.csv`, `outcomes/volume_force_sweep_ETHUSDT_20251204_210106_analysis.json`
- **Instrument / timeframe / side**: ETHUSDT, 15m, long

## Model quality (breakout head)

- **Selected config ID**: 14
- **Signature**: `target_threshold_atr=1.2|lookahead=8|normalization_window=192|volatility_percentile=75|trend_beta=0.75|xgb_max_depth=6`
- **Breakout log loss**: ≈ 0.489 (OOF)
- **Breakout accuracy**: ≈ 0.813 (OOF)

This configuration remains the global best by breakout log loss in the latest hierarchical sweep, after introducing move-based weighting for stronger future moves.

## Trading evaluation (simple breakout effectiveness)

Evaluation uses `vol_force_breakout` probabilities versus `future_return_H` (H-bar forward return) to form simple long-only breakout proxies.

For config 14, on the latest sweep:

- **Top 5% quantile (highest-probability breakouts)**
  - Coverage: ≈ 9.5% of samples
  - Mean `future_return_H`: ≈ +1.7 per 100 units of return (0.0169)
  - Sharpe (simple trade-count–scaled): ≈ 2.9

- **Top 10% quantile**
  - Coverage: ≈ 14.2% of samples
  - Mean `future_return_H`: ≈ +1.1 per 100 units of return (0.0108)
  - Sharpe: ≈ 2.7

- **Top 30% quantile (focus region)**
  - Coverage: ≈ 57.3% of samples
  - Mean `future_return_H`: ≈ +0.76 per 100 units of return (0.0076)
  - Sharpe: ≈ 4.6

These figures ignore transaction costs, slippage, and position sizing, but they show that the signal is not only concentrated in the very top 5–10% of scores: a broad top-30% slice still exhibits a positive and economically meaningful forward-return structure with high risk-adjusted performance.

## Comparison: configs with highest top-30% Sharpe

Among all sweep runs, the top 5 configurations by `breakout_trade_top30_sharpe` are:

- Config 99: ATR=1.2, lookahead=16, norm_window=96, vol_pct=75, trend_beta=0.50, depth=6, lr=0.03, n_estimators=800
- Config 72: ATR=1.0, lookahead=16, norm_window=192, vol_pct=80, trend_beta=0.75, depth=6, lr=0.03, n_estimators=800
- Config 103: ATR=1.2, lookahead=16, norm_window=192, vol_pct=70, trend_beta=0.50, depth=6, lr=0.03, n_estimators=800
- Config 62: ATR=1.0, lookahead=16, norm_window=96, vol_pct=70, trend_beta=0.75, depth=6, lr=0.03, n_estimators=800
- Config 63: ATR=1.0, lookahead=16, norm_window=96, vol_pct=75, trend_beta=0.50, depth=6, lr=0.03, n_estimators=800

For these configs, the top-30% bucket has:

- Coverage ≈ 52.8% of samples
- Mean `future_return_H`: ≈ +1.71 per 100 units of return (0.0171)
- Sharpe: ≈ 6.8

This indicates that slightly longer lookahead (16 bars) and nearby structural settings can further increase mid-tail economic strength, at the cost of higher breakout log loss (~0.675–0.69 vs 0.489 for config 14).

## Decision

- **Production default**: keep config 14 as the global default Volume Force breakout configuration, based on its superior OOF log loss and strong, broad-based economic signal (especially in the top-30% bucket).
- **Trading-oriented alternates**: treat configs 99 / 72 / 103 / 62 / 63 as high-top30-Sharpe variants (longer lookahead, stronger mid-tail payoffs) that can be revisited for trading-specific strategies or future experiments if we decide to trade more explicitly on mid-horizon breakouts.
