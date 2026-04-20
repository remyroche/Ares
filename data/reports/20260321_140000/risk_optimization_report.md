# Risk Optimization Report — 20260321_140000
Generated: 2026-04-20 18:53 UTC

## Configuration
- **15m precision**: True
- **Fee BPS**: 15.0
- **Max concurrent trades**: 5
- **Max portfolio weight**: 0.25
- **Daily cap per specialist**: 8
- **Daily cap total**: 25

## Optimized Risk Parameters
| Bucket | TP | SL | Trail | BE% | Lock% | LockAmt% | MaxLoss% | Vol Lo | Vol Hi | Z Max | Hold (h) |
|--------|----|----|-------|-----|-------|----------|----------|--------|--------|-------|----------|
| risk_(price_down_mr==1)|(*)|(*)_worst | 0.50 | 0.18 | 0.25 | 0.0 | 0.0 | 0.0 | 0.0 | 0.030 | 0.060 | 3.0 | 12 |
| risk_(price_down_tf==1)|(*)|(*)_worst | 0.50 | 0.18 | 0.25 | 0.0 | 0.0 | 0.0 | 0.0 | 0.030 | 0.060 | 3.0 | 24 |
| risk_(price_up_mr==1)|(*)|(*)_best | 0.50 | 0.18 | 0.25 | 0.0 | 0.0 | 0.0 | 0.0 | 0.030 | 0.060 | 3.0 | 12 |
| risk_(price_up_tf==1)|(*)|(*)_best | 0.50 | 0.18 | 0.25 | 0.0 | 0.0 | 0.0 | 0.0 | 0.030 | 0.060 | 3.0 | 24 |
| risk_long_(price_down_mr==1)|(*)|(*) | 0.50 | 0.18 | 0.25 | 0.0 | 0.0 | 0.0 | 0.0 | 0.030 | 0.060 | 3.0 | 12 |
| risk_long_(price_up_tf==1)|(*)|(*) | 0.50 | 0.18 | 0.25 | 0.0 | 0.0 | 0.0 | 0.0 | 0.030 | 0.060 | 3.0 | 24 |
| risk_short_(price_down_tf==1)|(*)|(*) | 0.50 | 0.18 | 0.25 | 0.0 | 0.0 | 0.0 | 0.0 | 0.030 | 0.060 | 3.0 | 24 |
| risk_short_(price_up_mr==1)|(*)|(*) | 0.50 | 0.18 | 0.25 | 0.0 | 0.0 | 0.0 | 0.0 | 0.030 | 0.060 | 3.0 | 12 |
