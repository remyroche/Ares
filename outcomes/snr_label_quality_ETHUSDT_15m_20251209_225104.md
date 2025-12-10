# SNR Label-Quality Diagnostics

**Symbol**: ETHUSDT
**Exchange**: binance
**Timeframe**: 15m

## Summary
- Total samples: 140354
- Labeled samples: 21176 (coverage=15.1%)
- Positive labels: 3873 (18.3%)
- Negative labels: 17303

## Retention
- Pre-filter events (realized_return not NaN): 43470
- Pre-filter pos/neg (raw econ > cost): 11800 / 31670
- Post-filter labeled events: 21176
- Post-filter pos/neg (binary_label): 3873 / 17303
- Total retention: 48.7%
- Positive retention: 32.8%
- Negative retention: 54.6%

## Economic Separation and SNR
- Pre-filter mean return (label=1/0): 1.46% / -0.93%
- Post-filter mean return (label=1/0): 1.46% / -0.68%
- Pre-filter Cohen's d: 4.786
- Post-filter Cohen's d: 2.656
- Pre-filter SNR (label=1): 2.506
- Post-filter SNR (label=1): 2.859

## Label Overlap and Cost Metrics
- Label overlap (mis-signed P&L share): 13.8%
- Transaction cost (approx per event): 0.300%
- Unconditional mean event return: -0.29%
- Mean return (label=1) minus cost: 1.16%
- Fraction of labeled events with |return| < cost: 11.5%
- Aleatoric uncertainty fraction (|return| < cost): 11.5%

## High-Probability Buckets (by meta_probability, isotonic expected returns)
- Top  5%: n=20060, win_rate=17.7%, mean_exp_ret=-0.15%, Sharpe_exp=-0.68
- Top 10%: n=20060, win_rate=17.7%, mean_exp_ret=-0.15%, Sharpe_exp=-0.68
- Top 20%: n=20060, win_rate=17.7%, mean_exp_ret=-0.15%, Sharpe_exp=-0.68
- Top 30%: n=20060, win_rate=17.7%, mean_exp_ret=-0.15%, Sharpe_exp=-0.68
- Top 40%: n=20060, win_rate=17.7%, mean_exp_ret=-0.15%, Sharpe_exp=-0.68

## Volatility Buckets (by volatility_1d)
- Vol low: n=7054, pos_rate=13.5%, mean_ret=-0.28%, Sharpe=-0.26
- Vol mid: n=7054, pos_rate=17.5%, mean_ret=-0.30%, Sharpe=-0.26
- Vol high: n=7055, pos_rate=23.9%, mean_ret=-0.29%, Sharpe=-0.24

## Interpretation Hints
- Coverage (15.1%): Moderate coverage (5–20%): typical for event-driven labeling.
- Post-filter effect size (Cohen's d=2.656): Very large separation; labels are strongly aligned with economic outcomes.
- Post-filter SNR (label=1): 2.859 → High SNR: positive-label returns are well separated from noise.
- Retention (total=48.7%): Filters keep a substantial share of events; label density is relatively high.

## Overall Label-Quality Score
- Score (0-1): 0.851
- Rating: Great
- Summary: Strong label quality with good coverage, separation and economic margins.