# SNR Label-Quality Diagnostics

**Symbol**: ETHUSDT
**Exchange**: binance
**Timeframe**: 15m

## Summary
- Total samples: 536
- Labeled samples: 536 (coverage=100.0%)
- Positive labels: 208 (38.8%)
- Negative labels: 328

## Retention
- Pre-filter events (realized_return not NaN): 536
- Pre-filter pos/neg (raw econ > cost): 208 / 328
- Post-filter labeled events: 536
- Post-filter pos/neg (binary_label): 208 / 328
- Total retention: 100.0%
- Positive retention: 100.0%
- Negative retention: 100.0%

## Economic Separation and SNR
- Pre-filter mean return (label=1/0): 1.55% / -1.50%
- Post-filter mean return (label=1/0): 1.55% / -1.50%
- Pre-filter Cohen's d: 8.705
- Post-filter Cohen's d: 8.705
- Pre-filter SNR (label=1): 3.568
- Post-filter SNR (label=1): 3.568

## Label Overlap and Cost Metrics
- Label overlap (mis-signed P&L share): 0.0%
- Transaction cost (approx per event): 0.300%
- Unconditional mean event return: -0.31%
- Mean return (label=1) minus cost: 1.25%
- Fraction of labeled events with |return| < cost: 0.0%
- Aleatoric uncertainty fraction (|return| < cost): 0.0%

## High-Probability Buckets (by meta_probability, isotonic expected returns)

## Enhanced Volatility Buckets (by volatility_1d)
- Vol low: n=179, pos_rate=59.2%, mean_ret=0.31%, Sharpe=0.23, vol_range=[0.0040, 0.0048]
- Vol mid: n=178, pos_rate=31.5%, mean_ret=-0.51%, Sharpe=-0.37, vol_range=[0.0040, 0.0048]
- Vol high: n=179, pos_rate=25.7%, mean_ret=-0.74%, Sharpe=-0.45, vol_range=[0.0040, 0.0048]

## Interpretation Hints
- Coverage (100.0%): High coverage (>20%): many labeled events; check for redundancy or label noise.
- Post-filter effect size (Cohen's d=8.705): Very large separation; labels are strongly aligned with economic outcomes.
- Post-filter SNR (label=1): 3.568 → High SNR: positive-label returns are well separated from noise.
- Retention (total=100.0%): Filters keep a substantial share of events; label density is relatively high.

## Overall Label-Quality Score
- Score (0-1): 0.925
- Rating: Great
- Summary: Strong label quality with good coverage, separation and economic margins.
