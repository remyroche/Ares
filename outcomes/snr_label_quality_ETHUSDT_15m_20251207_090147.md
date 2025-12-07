# SNR Label-Quality Diagnostics

**Symbol**: ETHUSDT
**Exchange**: binance
**Timeframe**: 15m

## Summary
- Total samples: 140354
- Labeled samples: 21177 (coverage=15.1%)
- Positive labels: 2164 (10.2%)
- Negative labels: 19013

## Retention
- Pre-filter events (realized_return not NaN): 43472
- Pre-filter pos/neg (raw econ > cost): 11342 / 32130
- Post-filter labeled events: 21177
- Post-filter pos/neg (binary_label): 2164 / 19013
- Total retention: 48.7%
- Positive retention: 19.1%
- Negative retention: 59.2%

## Economic Separation and SNR
- Pre-filter mean return (label=1/0): 1.31% / -0.84%
- Post-filter mean return (label=1/0): 1.74% / -0.52%
- Pre-filter Cohen's d: 4.391
- Post-filter Cohen's d: 2.801
- Pre-filter SNR (label=1): 2.392
- Post-filter SNR (label=1): 9.398

## Label Overlap and Cost Metrics
- Label overlap (mis-signed P&L share): 22.2%
- Transaction cost (approx per event): 0.300%
- Unconditional mean event return: -0.29%
- Mean return (label=1) minus cost: 1.44%
- Fraction of labeled events with |return| < cost: 14.2%
- Aleatoric uncertainty fraction (|return| < cost): 14.2%

## High-Probability Buckets (by meta_probability, isotonic expected returns)
- Top  5%: n=20267, win_rate=10.0%, mean_exp_ret=-0.31%, Sharpe_exp=-1.65
- Top 10%: n=20267, win_rate=10.0%, mean_exp_ret=-0.31%, Sharpe_exp=-1.65
- Top 20%: n=20267, win_rate=10.0%, mean_exp_ret=-0.31%, Sharpe_exp=-1.65
- Top 30%: n=20267, win_rate=10.0%, mean_exp_ret=-0.31%, Sharpe_exp=-1.65
- Top 40%: n=20267, win_rate=10.0%, mean_exp_ret=-0.31%, Sharpe_exp=-1.65

## Volatility Buckets (by volatility_1d)
- Vol low: n=7055, pos_rate=5.7%, mean_ret=-0.28%, Sharpe=-0.28
- Vol mid: n=7054, pos_rate=10.3%, mean_ret=-0.30%, Sharpe=-0.28
- Vol high: n=7055, pos_rate=14.7%, mean_ret=-0.29%, Sharpe=-0.27

## Interpretation Hints
- Coverage (15.1%): Moderate coverage (5–20%): typical for event-driven labeling.
- Post-filter effect size (Cohen's d=2.801): Very large separation; labels are strongly aligned with economic outcomes.
- Post-filter SNR (label=1): 9.398 → High SNR: positive-label returns are well separated from noise.
- Retention (total=48.7%): Filters keep a substantial share of events; label density is relatively high.

## Overall Label-Quality Score
- Score (0-1): 0.878
- Rating: Great
- Summary: Strong label quality with good coverage, separation and economic margins.