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
| 0.600000 | 0.005901      | 0.005550     | 0.035402                 | 0.207888         | 0.004359        | 0.001697         | 20.000000 |
| 0.500000 | 0.005702      | 0.004371     | 0.036907                 | 0.189074         | 0.004700        | 0.002557         | 20.000000 |
| 0.400000 | 0.006931      | 0.002720     | 0.040936                 | 0.159293         | 0.003291        | 0.001395         | 20.000000 |
| 0.700000 | 0.005217      | 0.007235     | 0.032372                 | 0.124116         | 0.003892        | 0.001675         | 20.000000 |

## Best d per Asset (by Event IC IR)

| asset           | d        | ic_event_mean | ic_ir_event | ic_overall_mean |
| --------------- | -------- | ------------- | ----------- | --------------- |
| ADA/USDC        | 0.400000 | 0.026808      | 1.751107    | 0.019110        |
| ADA/USDT        | 0.600000 | 0.029937      | 1.739547    | 0.023188        |
| ACT/USDT        | 0.400000 | 0.063110      | 1.519217    | 0.016570        |
| ACM/USDT        | 0.700000 | 0.067982      | 1.327708    | 0.050327        |
| ACT/USDC        | 0.700000 | 0.054032      | 1.117364    | 0.018912        |
| 1INCH/USDT      | 0.700000 | 0.011964      | 0.549708    | 0.011870        |
| 1000CAT/USDT    | 0.500000 | 0.047543      | 0.480067    | 0.025981        |
| 1000CAT/USDC    | 0.700000 | 0.016425      | 0.435452    | 0.033002        |
| 1000CHEEMS/USDT | 0.700000 | 0.017523      | 0.410651    | 0.005771        |
| A2Z/USDT        | 0.400000 | 0.064120      | 0.355750    | 0.011561        |
| AAVE/USDC       | 0.600000 | 0.014330      | 0.181497    | -0.000773       |
| ACX/USDT        | 0.400000 | 0.004798      | 0.181434    | -0.000361       |
| 1MBABYDOGE/USDT | 0.700000 | 0.004249      | 0.089045    | -0.029613       |
| 1000SATS/USDT   | 0.500000 | -0.000550     | -0.008922   | -0.008826       |
| AAVE/USDT       | 0.700000 | -0.006732     | -0.162154   | -0.000705       |
| ACH/USDT        | 0.400000 | -0.006993     | -0.354940   | 0.000012        |
| ACE/USDT        | 0.400000 | -0.023706     | -0.482821   | 0.005687        |
| ADX/USDT        | 0.400000 | -0.025603     | -0.519172   | -0.034680       |
| A/USDT          | 0.700000 | -0.047894     | -0.598240   | -0.000277       |
| ACH/USDC        | 0.400000 | -0.045054     | -0.660777   | -0.005318       |

## Interpretation

- Higher `ic_event_mean` and `ic_ir_event_mean` suggest better discrimination in high-vol/high-range regimes.
- Compare `ic_event_mean` vs `ic_nonevent_mean` to decide whether a d is event-specialized or broad.
- Larger K(d) means longer memory and higher compute/warmup costs; include this in production trade-offs.
