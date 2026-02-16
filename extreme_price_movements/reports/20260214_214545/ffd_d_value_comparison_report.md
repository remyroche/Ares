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
| 0.400000 | 0.007742      | 0.015218     | 0.042222                 | -0.005299        | 0.006201        | 0.006182         | 20.000000 |
| 0.600000 | 0.001040      | 0.005763     | 0.042684                 | -0.034671        | 0.003070        | 0.005231         | 20.000000 |
| 0.500000 | 0.004175      | 0.008080     | 0.043038                 | -0.046583        | 0.004706        | 0.006010         | 20.000000 |
| 0.700000 | -0.000941     | 0.000302     | 0.040331                 | -0.080873        | 0.003280        | 0.006094         | 20.000000 |

## Best d per Asset (by Event IC IR)

| asset           | d        | ic_event_mean | ic_ir_event | ic_overall_mean |
| --------------- | -------- | ------------- | ----------- | --------------- |
| ACH/USDT        | 0.600000 | 0.029193      | 1.139343    | 0.011194        |
| ACM/USDT        | 0.400000 | 0.077443      | 1.075525    | 0.076673        |
| ACT/USDC        | 0.400000 | 0.055991      | 0.971591    | 0.014528        |
| ACT/USDT        | 0.500000 | 0.053087      | 0.828880    | 0.008428        |
| ADA/USDT        | 0.700000 | 0.032598      | 0.815713    | 0.017663        |
| ADA/USDC        | 0.700000 | 0.030835      | 0.762028    | 0.016998        |
| ACE/USDT        | 0.400000 | 0.027690      | 0.569367    | 0.003901        |
| 1000CAT/USDT    | 0.400000 | 0.043961      | 0.491706    | 0.034297        |
| ACH/USDC        | 0.400000 | 0.024982      | 0.297318    | 0.047904        |
| 1000CAT/USDC    | 0.400000 | 0.018817      | 0.290152    | 0.034907        |
| ACX/USDT        | 0.400000 | 0.007705      | 0.288716    | 0.006955        |
| 1000SATS/USDT   | 0.700000 | 0.015022      | 0.207760    | -0.001737       |
| A2Z/USDT        | 0.400000 | 0.017349      | 0.119587    | 0.000378        |
| 1MBABYDOGE/USDT | 0.700000 | 0.000748      | 0.028225    | -0.022634       |
| 1INCH/USDT      | 0.700000 | -0.005187     | -0.265573   | -0.000471       |
| ADX/USDT        | 0.400000 | -0.015564     | -0.310260   | -0.027196       |
| 1000CHEEMS/USDT | 0.700000 | -0.017364     | -0.331070   | -0.006511       |
| AAVE/USDT       | 0.700000 | -0.016051     | -0.357426   | -0.002200       |
| A/USDT          | 0.700000 | -0.102493     | -1.288594   | -0.046017       |
| AAVE/USDC       | 0.500000 | -0.056531     | -2.191078   | -0.023770       |

## Interpretation

- Higher `ic_event_mean` and `ic_ir_event_mean` suggest better discrimination in high-vol/high-range regimes.
- Compare `ic_event_mean` vs `ic_nonevent_mean` to decide whether a d is event-specialized or broad.
- Larger K(d) means longer memory and higher compute/warmup costs; include this in production trade-offs.
