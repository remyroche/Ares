# SNR Label-Quality Diagnostics

**Symbol**: ETHUSDT
**Exchange**: binance
**Timeframe**: 15m

## Summary
- Total samples: 135775
- Labeled samples: 52696 (coverage=38.8%)
- Positive labels: 12481 (23.7%)
- Negative labels: 40215

## Retention
- Pre-filter events (realized_return not NaN): 66522
- Pre-filter pos/neg (raw econ > cost): 21569 / 44953
- Post-filter labeled events: 52696
- Post-filter pos/neg (binary_label): 12481 / 40215
- Total retention: 79.2%
- Positive retention: 57.9%
- Negative retention: 89.5%

## Economic Separation and SNR
- Pre-filter mean return (label=1/0): 1.01% / -0.66%
- Post-filter mean return (label=1/0): 1.12% / -0.73%
- Pre-filter Cohen's d: 5.761
- Post-filter Cohen's d: 10.526
- Pre-filter SNR (label=1): 2.982
- Post-filter SNR (label=1): 6.103

## Label Overlap and Cost Metrics
- Label overlap (mis-signed P&L share): 0.0%
- Transaction cost (approx per event): 0.150%
- Unconditional mean event return: -0.29%
- Mean return (label=1) minus cost: 0.97%
- Fraction of labeled events with |return| < cost: 0.0%
- Aleatoric uncertainty fraction (|return| < cost): 0.0%

## High-Probability Buckets (by meta_probability, isotonic expected returns)
- Top  5%: n=2635, win_rate=41.0%, mean_exp_ret=0.00%, Sharpe_exp=0.20
- Top 10%: n=5270, win_rate=38.5%, mean_exp_ret=0.00%, Sharpe_exp=0.14
- Top 20%: n=10540, win_rate=35.3%, mean_exp_ret=0.00%, Sharpe_exp=0.10
- Top 30%: n=15809, win_rate=32.9%, mean_exp_ret=0.00%, Sharpe_exp=0.08
- Top 40%: n=26725, win_rate=30.7%, mean_exp_ret=0.00%, Sharpe_exp=0.06

## Volatility Buckets (by volatility_1d)
- Vol low: n=17558, pos_rate=19.2%, mean_ret=-0.33%, Sharpe=-0.48
- Vol mid: n=17557, pos_rate=23.4%, mean_ret=-0.31%, Sharpe=-0.39
- Vol high: n=17558, pos_rate=28.5%, mean_ret=-0.23%, Sharpe=-0.25

## Interpretation Hints
- Coverage (38.8%): High coverage (>20%): many labeled events; check for redundancy or label noise.
- Post-filter effect size (Cohen's d=10.526): Very large separation; labels are strongly aligned with economic outcomes.
- Post-filter SNR (label=1): 6.103 → High SNR: positive-label returns are well separated from noise.
- Retention (total=79.2%): Filters keep a substantial share of events; label density is relatively high.

## Overall Label-Quality Score
- Score (0-1): 0.897
- Rating: Great
- Summary: Strong label quality with good coverage, separation and economic margins.