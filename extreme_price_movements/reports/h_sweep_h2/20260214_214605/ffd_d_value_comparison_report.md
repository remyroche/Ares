# FFD d-value Comparison Report

## Weight Window Sizes K(d)

| d        | K    | warmup_bars | compute_cost |
| -------- | ---- | ----------- | ------------ |
| 0.400000 | 1458 | 1457        | O(N x 1458)  |
| 0.500000 | 927  | 926         | O(N x 927)   |
| 0.600000 | 590  | 589         | O(N x 590)   |
| 0.700000 | 372  | 371         | O(N x 372)   |

## Cross-Asset d Ranking (Event Regime Priority)

| d        | ic_event_mean | ic_event_med | ic_event_std_cross_asset | ic_ir_event_mean | ic_overall_mean | ic_nonevent_mean | assets    |
| -------- | ------------- | ------------ | ------------------------ | ---------------- | --------------- | ---------------- | --------- |
| 0.400000 | 0.019074      | 0.019171     | 0.049966                 | 0.331669         | 0.010588        | 0.008020         | 20.000000 |
| 0.500000 | 0.017669      | 0.014735     | 0.046610                 | 0.315254         | 0.009248        | 0.006754         | 20.000000 |
| 0.600000 | 0.014669      | 0.006859     | 0.043533                 | 0.260000         | 0.006691        | 0.004378         | 20.000000 |
| 0.700000 | 0.011201      | 0.003691     | 0.043947                 | 0.209243         | 0.003470        | 0.001085         | 20.000000 |

## Best d per Asset (by Event IC IR)

| asset           | d        | ic_event_mean | ic_ir_event | ic_overall_mean |
| --------------- | -------- | ------------- | ----------- | --------------- |
| 1INCH/USDT      | 0.400000 | 0.034413      | 2.078446    | 0.026841        |
| ACH/USDC        | 0.400000 | 0.073301      | 1.629954    | 0.043601        |
| ACT/USDC        | 0.400000 | 0.125616      | 1.587041    | 0.040977        |
| ACM/USDT        | 0.700000 | 0.060267      | 1.290761    | 0.050298        |
| 1000CHEEMS/USDT | 0.400000 | 0.069225      | 0.902809    | 0.046935        |
| ACX/USDT        | 0.400000 | 0.037531      | 0.875315    | -0.000821       |
| 1000CAT/USDC    | 0.700000 | 0.028989      | 0.864651    | 0.058439        |
| ACT/USDT        | 0.400000 | 0.085513      | 0.635530    | 0.012716        |
| ADA/USDC        | 0.400000 | 0.021605      | 0.479338    | 0.026605        |
| ADA/USDT        | 0.400000 | 0.021676      | 0.477962    | 0.027330        |
| 1000CAT/USDT    | 0.600000 | 0.021702      | 0.374407    | 0.027587        |
| AAVE/USDT       | 0.400000 | 0.009399      | 0.242761    | 0.010919        |
| ACE/USDT        | 0.600000 | 0.000582      | 0.010493    | 0.007972        |
| ACH/USDT        | 0.400000 | -0.000853     | -0.017369   | -0.013559       |
| 1000SATS/USDT   | 0.400000 | -0.002293     | -0.040555   | -0.013289       |
| AAVE/USDC       | 0.400000 | -0.004630     | -0.104885   | 0.002502        |
| A2Z/USDT        | 0.700000 | -0.034211     | -0.258953   | -0.049492       |
| A/USDT          | 0.700000 | -0.020825     | -0.464443   | -0.013461       |
| 1MBABYDOGE/USDT | 0.400000 | -0.045327     | -0.668214   | -0.029333       |
| ADX/USDT        | 0.700000 | -0.023901     | -0.887878   | -0.021713       |

## Interpretation

- Higher `ic_event_mean` and `ic_ir_event_mean` suggest better discrimination in high-vol/high-range regimes.
- Compare `ic_event_mean` vs `ic_nonevent_mean` to decide whether a d is event-specialized or broad.
- Larger K(d) means longer memory and higher compute/warmup costs; include this in production trade-offs.
