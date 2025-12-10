# SNR Label-Quality Diagnostics

**Symbol**: ETHUSDT
**Exchange**: binance
**Timeframe**: 15m

## Summary
- Total samples: 140354
- Labeled samples: 21177 (coverage=15.1%)
- Positive labels: 4228 (20.0%)
- Negative labels: 16949

## Retention
- Pre-filter events (realized_return not NaN): 43471
- Pre-filter pos/neg (raw econ > cost): 10614 / 32857
- Post-filter labeled events: 21177
- Post-filter pos/neg (binary_label): 4228 / 16949
- Total retention: 48.7%
- Positive retention: 39.8%
- Negative retention: 51.6%

## Economic Separation and SNR
- Pre-filter mean return (label=1/0): 1.34% / -0.81%
- Post-filter mean return (label=1/0): 1.43% / -0.72%
- Pre-filter Cohen's d: 5.276
- Post-filter Cohen's d: 4.276
- Pre-filter SNR (label=1): 2.668
- Post-filter SNR (label=1): 3.399

## Label Overlap and Cost Metrics
- Label overlap (mis-signed P&L share): 9.4%
- Transaction cost (approx per event): 0.300%
- Unconditional mean event return: -0.29%
- Mean return (label=1) minus cost: 1.13%
- Fraction of labeled events with |return| < cost: 11.2%
- Aleatoric uncertainty fraction (|return| < cost): 11.2%

## High-Probability Buckets (by meta_probability, isotonic expected returns)
- Top  5%: n=1059, win_rate=24.3%, mean_exp_ret=2.09%, Sharpe_exp=10.87
- Top 10%: n=2118, win_rate=24.1%, mean_exp_ret=2.09%, Sharpe_exp=10.67
- Top 20%: n=4236, win_rate=23.9%, mean_exp_ret=2.11%, Sharpe_exp=11.04
- Top 30%: n=6353, win_rate=24.0%, mean_exp_ret=2.11%, Sharpe_exp=11.14
- Top 40%: n=8471, win_rate=24.1%, mean_exp_ret=2.12%, Sharpe_exp=11.22

## Volatility Buckets (by volatility_1d)
- Vol low: n=7055, pos_rate=16.1%, mean_ret=-0.28%, Sharpe=-0.30
- Vol mid: n=7054, pos_rate=21.2%, mean_ret=-0.30%, Sharpe=-0.29
- Vol high: n=7055, pos_rate=22.6%, mean_ret=-0.30%, Sharpe=-0.29

## Interpretation Hints
- Coverage (15.1%): Moderate coverage (5–20%): typical for event-driven labeling.
- Post-filter effect size (Cohen's d=4.276): Very large separation; labels are strongly aligned with economic outcomes.
- Post-filter SNR (label=1): 3.399 → High SNR: positive-label returns are well separated from noise.
- Retention (total=48.7%): Filters keep a substantial share of events; label density is relatively high.

## Overall Label-Quality Score
- Score (0-1): 0.848
- Rating: Great
- Summary: Strong label quality with good coverage, separation and economic margins.