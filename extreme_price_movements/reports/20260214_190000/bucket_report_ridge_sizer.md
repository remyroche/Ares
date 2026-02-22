# Ridge Position Sizer Report — 20260214_190000
Generated: 2026-02-22 15:31 UTC

## Direction: LONG
| Bucket | IC (train) | IC (val) | Sharpe | N weights | Top weight |
|---|---|---|---|---|---|
| long_tf | — | — | — | 6 | reg_H4=0.3126 |
| long_mr | — | — | — | 6 | reg_range=0.3548 |

### Weight Breakdown — LONG
**long_tf**: `reg_H2`=0.0956, `reg_H4`=0.3126, `reg_mean`=0.1344, `reg_range`=0.1860, `reg_sign_agree`=0.0855, `reg_std`=0.1860
**long_mr**: `reg_H2`=0.0000, `reg_H4`=0.2905, `reg_mean`=0.0000, `reg_range`=0.3548, `reg_sign_agree`=0.0000, `reg_std`=0.3548

## Direction: SHORT
| Bucket | IC (train) | IC (val) | Sharpe | N weights | Top weight |
|---|---|---|---|---|---|
| short_mr | — | — | — | 6 | reg_H4=0.5002 |
| short_tf | — | — | — | 4 | reg_std=0.6886 |

### Weight Breakdown — SHORT
**short_mr**: `reg_H2`=0.0000, `reg_H4`=0.5002, `reg_mean`=0.0000, `reg_range`=0.2499, `reg_sign_agree`=0.0000, `reg_std`=0.2499
**short_tf**: `clf`=0.0281, `reg_H8`=0.1416, `reg_mean`=0.1416, `reg_std`=0.6886
