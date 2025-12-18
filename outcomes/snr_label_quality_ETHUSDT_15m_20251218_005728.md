# SNR Label-Quality Diagnostics

**Symbol**: ETHUSDT
**Exchange**: binance
**Timeframe**: 15m

## Summary
- Total samples: 1634
- Labeled samples: 1634 (coverage=100.0%)
- Positive labels: 434 (26.6%)
- Negative labels: 1200

## Retention
- Pre-filter events (realized_return not NaN): 1634
- Pre-filter pos/neg (raw econ > cost): 434 / 1200
- Post-filter labeled events: 1634
- Post-filter pos/neg (binary_label): 434 / 1200
- Total retention: 100.0%
- Positive retention: 100.0%
- Negative retention: 100.0%

## Economic Separation and SNR
- Pre-filter mean return (label=1/0): 1.02% / -0.90%
- Post-filter mean return (label=1/0): 1.02% / -0.90%
- Pre-filter Cohen's d: 9.794
- Post-filter Cohen's d: 9.794
- Pre-filter SNR (label=1): 3.378
- Post-filter SNR (label=1): 3.378

## Label Overlap and Cost Metrics
- Label overlap (mis-signed P&L share): 0.0%
- Transaction cost (approx per event): 0.300%
- Unconditional mean event return: -0.39%
- Mean return (label=1) minus cost: 0.72%
- Fraction of labeled events with |return| < cost: 0.0%
- Aleatoric uncertainty fraction (|return| < cost): 0.0%

## High-Probability Buckets (by meta_probability, isotonic expected returns)

## Enhanced Volatility Buckets (by volatility_1d)
- Vol low: n=545, pos_rate=28.4%, mean_ret=-0.34%, Sharpe=-0.45, vol_range=[0.0051, 0.0061]
- Vol mid: n=544, pos_rate=23.9%, mean_ret=-0.42%, Sharpe=-0.55, vol_range=[0.0051, 0.0061]
- Vol high: n=545, pos_rate=27.3%, mean_ret=-0.41%, Sharpe=-0.39, vol_range=[0.0051, 0.0061]

## Interpretation Hints
- Coverage (100.0%): High coverage (>20%): many labeled events; check for redundancy or label noise.
- Post-filter effect size (Cohen's d=9.794): Very large separation; labels are strongly aligned with economic outcomes.
- Post-filter SNR (label=1): 3.378 → High SNR: positive-label returns are well separated from noise.
- Retention (total=100.0%): Filters keep a substantial share of events; label density is relatively high.

## Overall Label-Quality Score
- Score (0-1): 0.872
- Rating: Great
- Summary: Strong label quality with good coverage, separation and economic margins.
