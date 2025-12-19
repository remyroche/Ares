# SNR Label-Quality Diagnostics

**Symbol**: ETHUSDT
**Exchange**: binance
**Timeframe**: 15m

## Summary
- Total samples: 3484
- Labeled samples: 3484 (coverage=100.0%)
- Positive labels: 0 (0.0%)
- Negative labels: 0

## Retention
- Pre-filter events (realized_return not NaN): 3484
- Pre-filter pos/neg (raw econ > cost): 1233 / 2251
- Post-filter labeled events: 3484
- Post-filter pos/neg (binary_label): 0 / 0
- Total retention: 100.0%
- Positive retention: 0.0%
- Negative retention: 0.0%

## Economic Separation and SNR
- Pre-filter mean return (label=1/0): 0.85% / -0.96%
- Post-filter mean return (label=1/0): 0.00% / 0.00%
- Pre-filter Cohen's d: 4.016
- Post-filter Cohen's d: nan
- Pre-filter SNR (label=1): 2.274
- Post-filter SNR (label=1): 0.000

## Label Overlap and Cost Metrics
- Label overlap (mis-signed P&L share): 0.0%
- Transaction cost (approx per event): 0.300%
- Unconditional mean event return: -0.32%
- Mean return (label=1) minus cost: -0.30%
- Fraction of labeled events with |return| < cost: 7.6%
- Aleatoric uncertainty fraction (|return| < cost): 7.6%

## High-Probability Buckets (by meta_probability, isotonic expected returns)

## Enhanced Volatility Buckets (by volatility_1d)
- Vol low: n=1161, pos_rate=0.0%, mean_ret=-0.28%, Sharpe=-0.42, vol_range=[0.0027, 0.0037]
- Vol mid: n=1161, pos_rate=0.0%, mean_ret=-0.28%, Sharpe=-0.31, vol_range=[0.0027, 0.0037]
- Vol high: n=1162, pos_rate=0.0%, mean_ret=-0.41%, Sharpe=-0.32, vol_range=[0.0027, 0.0037]

## Interpretation Hints
- Coverage (100.0%): High coverage (>20%): many labeled events; check for redundancy or label noise.
- Post-filter effect size (Cohen's d=nan): Effect size not available (insufficient data).
- Post-filter SNR (label=1): 0.000 → Low SNR: positive-label returns are noisy relative to their mean.
- Retention (total=100.0%): Filters keep a substantial share of events; label density is relatively high.

## Overall Label-Quality Score
- Score (0-1): 0.400
- Rating: Pass
- Summary: Mixed label quality; some usable signal but economic separation or coverage may be modest.
