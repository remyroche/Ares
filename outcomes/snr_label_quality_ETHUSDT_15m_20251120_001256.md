# SNR Label-Quality Diagnostics

**Symbol**: ETHUSDT
**Exchange**: binance
**Timeframe**: 15m

## Summary
- Total samples: 135775
- Labeled samples: 22658 (coverage=16.7%)
- Positive labels: 11329 (50.0%)
- Negative labels: 11329

## Retention
- Pre-filter events (realized_return not NaN): 44589
- Pre-filter pos/neg (raw econ > cost): 14255 / 30334
- Post-filter labeled events: 22658
- Post-filter pos/neg (binary_label): 11329 / 11329
- Total retention: 50.8%
- Positive retention: 79.5%
- Negative retention: 37.3%

## Economic Separation and SNR
- Pre-filter mean return (label=1/0): 1.05% / -0.68%
- Post-filter mean return (label=1/0): 1.21% / -0.82%
- Pre-filter Cohen's d: 4.870
- Post-filter Cohen's d: 9.543
- Pre-filter SNR (label=1): 2.473
- Post-filter SNR (label=1): 4.370

## Label Overlap and Cost Metrics
- Label overlap (mis-signed P&L share): 0.0%
- Transaction cost (approx per event): 0.150%
- Unconditional mean event return: 0.19%
- Mean return (label=1) minus cost: 1.06%
- Fraction of labeled events with |return| < cost: 0.0%
- Aleatoric uncertainty fraction (|return| < cost): 0.0%

## High-Probability Buckets (by meta_probability, isotonic expected returns)
- Top  5%: n=1133, win_rate=100.0%, mean_exp_ret=0.50%, Sharpe_exp=0.92
- Top 10%: n=2266, win_rate=99.9%, mean_exp_ret=0.54%, Sharpe_exp=0.99
- Top 20%: n=4532, win_rate=89.1%, mean_exp_ret=0.42%, Sharpe_exp=0.85
- Top 30%: n=6798, win_rate=74.7%, mean_exp_ret=0.28%, Sharpe_exp=0.62
- Top 40%: n=11500, win_rate=67.4%, mean_exp_ret=0.17%, Sharpe_exp=0.44

## Volatility Buckets (by volatility_1d)
- Vol low: n=7553, pos_rate=33.7%, mean_ret=-0.15%, Sharpe=-0.17
- Vol mid: n=7552, pos_rate=35.7%, mean_ret=-0.11%, Sharpe=-0.12
- Vol high: n=7553, pos_rate=80.6%, mean_ret=0.84%, Sharpe=0.94

## Interpretation Hints
- Coverage (16.7%): Moderate coverage (5–20%): typical for event-driven labeling.
- Post-filter effect size (Cohen's d=9.543): Very large separation; labels are strongly aligned with economic outcomes.
- Post-filter SNR (label=1): 4.370 → High SNR: positive-label returns are well separated from noise.
- Retention (total=50.8%): Filters keep a substantial share of events; label density is relatively high.

## Overall Label-Quality Score
- Score (0-1): 0.862
- Rating: Great
- Summary: Strong label quality with good coverage, separation and economic margins.