# SNR Label-Quality Diagnostics

**Symbol**: ETHUSDT
**Exchange**: binance
**Timeframe**: 15m

## Summary
- Total samples: 34561
- Labeled samples: 2445 (coverage=7.1%)
- Positive labels: 670 (27.4%)
- Negative labels: 1775

## Retention
- Pre-filter events (realized_return not NaN): 6022
- Pre-filter pos/neg (raw econ > cost): 1691 / 4331
- Post-filter labeled events: 2445
- Post-filter pos/neg (binary_label): 670 / 1775
- Total retention: 40.6%
- Positive retention: 39.6%
- Negative retention: 41.0%

## Economic Separation and SNR
- Pre-filter mean return (label=1/0): 0.42% / -0.40%
- Post-filter mean return (label=1/0): 0.43% / -0.42%
- Pre-filter Cohen's d: 4.083
- Post-filter Cohen's d: 4.552
- Pre-filter SNR (label=1): 1.619
- Post-filter SNR (label=1): 1.622

## Label Overlap and Cost Metrics
- Label overlap (mis-signed P&L share): 0.2%
- Transaction cost (approx per event): 0.100%
- Unconditional mean event return: -0.19%
- Mean return (label=1) minus cost: 0.33%
- Fraction of labeled events with |return| < cost: 0.0%
- Aleatoric uncertainty fraction (|return| < cost): 0.0%

## High-Probability Buckets (by meta_probability, isotonic expected returns)
- meta_probability not available or insufficient data for bucket diagnostics.

## Enhanced Volatility Buckets (by volatility_1d)
- volatility_1d not available or insufficient data for volatility buckets.

## Interpretation Hints
- Coverage (7.1%): Moderate coverage (5–20%): typical for event-driven labeling.
- Post-filter effect size (Cohen's d=4.552): Very large separation; labels are strongly aligned with economic outcomes.
- Post-filter SNR (label=1): 1.622 → High SNR: positive-label returns are well separated from noise.
- Retention (total=40.6%): Filters keep a substantial share of events; label density is relatively high.

## Overall Label-Quality Score
- Score (0-1): 0.661
- Rating: Pass
- Summary: Mixed label quality; some usable signal but economic separation or coverage may be modest.