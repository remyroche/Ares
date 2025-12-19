# SNR Label-Quality Diagnostics

**Symbol**: ETHUSDT
**Exchange**: binance
**Timeframe**: 15m

## Summary
- Total samples: 536
- Labeled samples: 536 (coverage=100.0%)
- Positive labels: 205 (38.2%)
- Negative labels: 331

## Retention
- Pre-filter events (realized_return not NaN): 536
- Pre-filter pos/neg (raw econ > cost): 205 / 331
- Post-filter labeled events: 536
- Post-filter pos/neg (binary_label): 205 / 331
- Total retention: 100.0%
- Positive retention: 100.0%
- Negative retention: 100.0%

## Economic Separation and SNR
- Pre-filter mean return (label=1/0): 1.56% / -1.49%
- Post-filter mean return (label=1/0): 1.56% / -1.49%
- Pre-filter Cohen's d: 8.703
- Post-filter Cohen's d: 8.703
- Pre-filter SNR (label=1): 3.557
- Post-filter SNR (label=1): 3.557

## Label Overlap and Cost Metrics
- Label overlap (mis-signed P&L share): 0.0%
- Transaction cost (approx per event): 0.300%
- Unconditional mean event return: -0.33%
- Mean return (label=1) minus cost: 1.26%
- Fraction of labeled events with |return| < cost: 0.0%
- Aleatoric uncertainty fraction (|return| < cost): 0.0%

## High-Probability Buckets (by meta_probability, isotonic expected returns)

## Enhanced Volatility Buckets (by volatility_1d)
- Vol low: n=179, pos_rate=56.4%, mean_ret=0.24%, Sharpe=0.17, vol_range=[0.0040, 0.0048]
- Vol mid: n=178, pos_rate=32.6%, mean_ret=-0.48%, Sharpe=-0.35, vol_range=[0.0040, 0.0048]
- Vol high: n=179, pos_rate=25.7%, mean_ret=-0.74%, Sharpe=-0.45, vol_range=[0.0040, 0.0048]

## Interpretation Hints
- Coverage (100.0%): High coverage (>20%): many labeled events; check for redundancy or label noise.
- Post-filter effect size (Cohen's d=8.703): Very large separation; labels are strongly aligned with economic outcomes.
- Post-filter SNR (label=1): 3.557 → High SNR: positive-label returns are well separated from noise.
- Retention (total=100.0%): Filters keep a substantial share of events; label density is relatively high.

## Overall Label-Quality Score
- Score (0-1): 0.926
- Rating: Great
- Summary: Strong label quality with good coverage, separation and economic margins.
