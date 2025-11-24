# SNR Label-Quality Diagnostics

**Symbol**: ETHUSDT
**Exchange**: binance
**Timeframe**: 15m

## Summary
- Total samples: 173439
- Labeled samples: 31972 (coverage=18.4%)
- Positive labels: 15986 (50.0%)
- Negative labels: 15986

## Retention
- Pre-filter events (realized_return not NaN): 56502
- Pre-filter pos/neg (raw econ > cost): 15307 / 41195
- Post-filter labeled events: 31972
- Post-filter pos/neg (binary_label): 15986 / 15986
- Total retention: 56.6%
- Positive retention: 104.4%
- Negative retention: 38.8%

## Economic Separation and SNR
- Pre-filter mean return (label=1/0): 1.63% / -0.76%
- Post-filter mean return (label=1/0): 1.48% / -0.80%
- Pre-filter Cohen's d: 6.048
- Post-filter Cohen's d: 3.409
- Pre-filter SNR (label=1): 2.419
- Post-filter SNR (label=1): 1.577

## Label Overlap and Cost Metrics
- Label overlap (mis-signed P&L share): 4.7%
- Transaction cost (approx per event): 0.150%
- Unconditional mean event return: 0.34%
- Mean return (label=1) minus cost: 1.33%
- Fraction of labeled events with |return| < cost: 0.0%
- Aleatoric uncertainty fraction (|return| < cost): 0.0%

## High-Probability Buckets (by meta_probability, isotonic expected returns)
- Top  5%: n=1599, win_rate=100.0%, mean_exp_ret=0.65%, Sharpe_exp=1.15
- Top 10%: n=3198, win_rate=100.0%, mean_exp_ret=0.64%, Sharpe_exp=1.13
- Top 20%: n=6395, win_rate=100.0%, mean_exp_ret=0.59%, Sharpe_exp=1.03
- Top 30%: n=9592, win_rate=88.5%, mean_exp_ret=0.49%, Sharpe_exp=0.90
- Top 40%: n=12789, win_rate=75.6%, mean_exp_ret=0.37%, Sharpe_exp=0.71

## Volatility Buckets (by volatility_1d)
- Vol low: n=10657, pos_rate=26.8%, mean_ret=-0.13%, Sharpe=-0.13
- Vol mid: n=10657, pos_rate=36.2%, mean_ret=0.08%, Sharpe=0.06
- Vol high: n=10658, pos_rate=87.1%, mean_ret=1.07%, Sharpe=0.83

## Interpretation Hints
- Coverage (18.4%): Moderate coverage (5–20%): typical for event-driven labeling.
- Post-filter effect size (Cohen's d=3.409): Very large separation; labels are strongly aligned with economic outcomes.
- Post-filter SNR (label=1): 1.577 → High SNR: positive-label returns are well separated from noise.
- Retention (total=56.6%): Filters keep a substantial share of events; label density is relatively high.

## Overall Label-Quality Score
- Score (0-1): 0.912
- Rating: Great
- Summary: Strong label quality with good coverage, separation and economic margins.