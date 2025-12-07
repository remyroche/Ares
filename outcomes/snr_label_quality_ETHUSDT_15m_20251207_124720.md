# SNR Label-Quality Diagnostics

**Symbol**: ETHUSDT
**Exchange**: binance
**Timeframe**: 15m

## Summary
- Total samples: 140354
- Labeled samples: 21177 (coverage=15.1%)
- Positive labels: 3718 (17.6%)
- Negative labels: 17459

## Retention
- Pre-filter events (realized_return not NaN): 43471
- Pre-filter pos/neg (raw econ > cost): 11370 / 32101
- Post-filter labeled events: 21177
- Post-filter pos/neg (binary_label): 3718 / 17459
- Total retention: 48.7%
- Positive retention: 32.7%
- Negative retention: 54.4%

## Economic Separation and SNR
- Pre-filter mean return (label=1/0): 1.32% / -0.85%
- Post-filter mean return (label=1/0): 1.37% / -0.65%
- Pre-filter Cohen's d: 4.884
- Post-filter Cohen's d: 2.872
- Pre-filter SNR (label=1): 2.675
- Post-filter SNR (label=1): 3.118

## Label Overlap and Cost Metrics
- Label overlap (mis-signed P&L share): 13.9%
- Transaction cost (approx per event): 0.300%
- Unconditional mean event return: -0.29%
- Mean return (label=1) minus cost: 1.07%
- Fraction of labeled events with |return| < cost: 12.3%
- Aleatoric uncertainty fraction (|return| < cost): 12.3%

## High-Probability Buckets (by meta_probability, isotonic expected returns)
- Top  5%: n=1059, win_rate=48.6%, mean_exp_ret=0.61%, Sharpe_exp=11.59
- Top 10%: n=19458, win_rate=13.6%, mean_exp_ret=-0.36%, Sharpe_exp=-1.37
- Top 20%: n=19458, win_rate=13.6%, mean_exp_ret=-0.36%, Sharpe_exp=-1.37
- Top 30%: n=19458, win_rate=13.6%, mean_exp_ret=-0.36%, Sharpe_exp=-1.37
- Top 40%: n=19458, win_rate=13.6%, mean_exp_ret=-0.36%, Sharpe_exp=-1.37

## Volatility Buckets (by volatility_1d)
- Vol low: n=7055, pos_rate=12.1%, mean_ret=-0.28%, Sharpe=-0.29
- Vol mid: n=7054, pos_rate=17.5%, mean_ret=-0.31%, Sharpe=-0.29
- Vol high: n=7055, pos_rate=23.0%, mean_ret=-0.30%, Sharpe=-0.28

## Interpretation Hints
- Coverage (15.1%): Moderate coverage (5–20%): typical for event-driven labeling.
- Post-filter effect size (Cohen's d=2.872): Very large separation; labels are strongly aligned with economic outcomes.
- Post-filter SNR (label=1): 3.118 → High SNR: positive-label returns are well separated from noise.
- Retention (total=48.7%): Filters keep a substantial share of events; label density is relatively high.

## Overall Label-Quality Score
- Score (0-1): 0.841
- Rating: Great
- Summary: Strong label quality with good coverage, separation and economic margins.