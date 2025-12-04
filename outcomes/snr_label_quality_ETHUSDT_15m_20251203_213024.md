# SNR Label-Quality Diagnostics

**Symbol**: ETHUSDT
**Exchange**: binance
**Timeframe**: 15m

## Summary
- Total samples: 34135
- Labeled samples: 4187 (coverage=12.3%)
- Positive labels: 1051 (25.1%)
- Negative labels: 3136

## Retention
- Pre-filter events (realized_return not NaN): 8304
- Pre-filter pos/neg (raw econ > cost): 2670 / 5634
- Post-filter labeled events: 4187
- Post-filter pos/neg (binary_label): 1051 / 3136
- Total retention: 50.4%
- Positive retention: 39.4%
- Negative retention: 55.7%

## Economic Separation and SNR
- Pre-filter mean return (label=1/0): 0.82% / -0.57%
- Post-filter mean return (label=1/0): 0.87% / -0.48%
- Pre-filter Cohen's d: 7.056
- Post-filter Cohen's d: 4.217
- Pre-filter SNR (label=1): 4.059
- Post-filter SNR (label=1): 9.354

## Label Overlap and Cost Metrics
- Label overlap (mis-signed P&L share): 8.6%
- Transaction cost (approx per event): 0.150%
- Unconditional mean event return: -0.14%
- Mean return (label=1) minus cost: 0.72%
- Fraction of labeled events with |return| < cost: 5.7%
- Aleatoric uncertainty fraction (|return| < cost): 5.7%

## High-Probability Buckets (by meta_probability, isotonic expected returns)
- Top  5%: n=210, win_rate=94.3%, mean_exp_ret=0.58%, Sharpe_exp=640.71
- Top 10%: n=419, win_rate=83.1%, mean_exp_ret=0.48%, Sharpe_exp=3.41
- Top 20%: n=838, win_rate=64.3%, mean_exp_ret=0.22%, Sharpe_exp=0.77
- Top 30%: n=3497, win_rate=24.3%, mean_exp_ret=-0.09%, Sharpe_exp=-0.39
- Top 40%: n=3497, win_rate=24.3%, mean_exp_ret=-0.09%, Sharpe_exp=-0.39

## Volatility Buckets (by volatility_1d)
- Vol low: n=1396, pos_rate=20.0%, mean_ret=-0.16%, Sharpe=-0.25
- Vol mid: n=1395, pos_rate=27.4%, mean_ret=-0.12%, Sharpe=-0.18
- Vol high: n=1396, pos_rate=27.9%, mean_ret=-0.13%, Sharpe=-0.19

## Interpretation Hints
- Coverage (12.3%): Moderate coverage (5–20%): typical for event-driven labeling.
- Post-filter effect size (Cohen's d=4.217): Very large separation; labels are strongly aligned with economic outcomes.
- Post-filter SNR (label=1): 9.354 → High SNR: positive-label returns are well separated from noise.
- Retention (total=50.4%): Filters keep a substantial share of events; label density is relatively high.

## Overall Label-Quality Score
- Score (0-1): 0.769
- Rating: Great
- Summary: Strong label quality with good coverage, separation and economic margins.