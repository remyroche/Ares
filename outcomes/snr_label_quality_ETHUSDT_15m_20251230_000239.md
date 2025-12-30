# SNR Label-Quality Diagnostics

**Symbol**: ETHUSDT
**Exchange**: binance
**Timeframe**: 15m

## Summary
- Total samples: 7421
- Labeled samples: 7421 (coverage=100.0%)
- Positive labels: 3545 (47.8%)
- Negative labels: 3876

## Retention
- Pre-filter events (realized_return not NaN): 0
- Pre-filter pos/neg (raw econ > cost): 0 / 0
- Post-filter labeled events: 7421
- Post-filter pos/neg (binary_label): 3545 / 3876
- Total retention: 742100.0%
- Positive retention: 0.0%
- Negative retention: 0.0%

## Economic Separation and SNR
- Pre-filter mean return (label=1/0): 0.00% / 0.00%
- Post-filter mean return (label=1/0): 0.00% / 0.00%
- Pre-filter Cohen's d: nan
- Post-filter Cohen's d: nan
- Pre-filter SNR (label=1): 0.000
- Post-filter SNR (label=1): 0.000

## Label Overlap and Cost Metrics
- Label overlap (mis-signed P&L share): 0.0%
- Transaction cost (approx per event): 0.300%
- Unconditional mean event return: 0.00%
- Mean return (label=1) minus cost: -0.30%
- Fraction of labeled events with |return| < cost: 0.0%
- Aleatoric uncertainty fraction (|return| < cost): 0.0%

## High-Probability Buckets (by meta_probability, isotonic expected returns)
- meta_probability not available or insufficient data for bucket diagnostics.

## Enhanced Volatility Buckets (by volatility_1d)
- volatility_1d not available or insufficient data for volatility buckets.

## Interpretation Hints
- Coverage (100.0%): High coverage (>20%): many labeled events; check for redundancy or label noise.
- Post-filter effect size (Cohen's d=nan): Effect size not available (insufficient data).
- Post-filter SNR (label=1): 0.000 → Low SNR: positive-label returns are noisy relative to their mean.
- Retention (total=742100.0%): Filters keep a substantial share of events; label density is relatively high.

## Overall Label-Quality Score
- Score (0-1): 0.400
- Rating: Pass
- Summary: Mixed label quality; some usable signal but economic separation or coverage may be modest.
