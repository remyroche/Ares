# FFD d-value Comparison Report

## Weight Window Sizes K(d)

| d        | K    | warmup_bars | compute_cost |
| -------- | ---- | ----------- | ------------ |
| 0.200000 | 3382 | 3381        | O(N x 3382)  |
| 0.300000 | 2275 | 2274        | O(N x 2275)  |
| 0.400000 | 1458 | 1457        | O(N x 1458)  |
| 0.500000 | 927  | 926         | O(N x 927)   |
| 0.600000 | 590  | 589         | O(N x 590)   |

## Cross-Asset d Ranking (Event Regime Priority)

| d        | ic_event_mean | ic_event_med | ic_event_std_cross_asset | ic_ir_event_mean | ic_overall_mean | ic_nonevent_mean | assets    |
| -------- | ------------- | ------------ | ------------------------ | ---------------- | --------------- | ---------------- | --------- |
| 0.400000 | 0.038294      | 0.045355     | 0.053034                 | 1.255756         | 0.012761        | -0.006004        | 18.000000 |
| 0.300000 | 0.044122      | 0.049295     | 0.059443                 | 1.236837         | 0.014438        | -0.006686        | 18.000000 |
| 0.200000 | 0.051650      | 0.055193     | 0.066833                 | 1.184381         | 0.017055        | -0.006041        | 18.000000 |
| 0.500000 | 0.032497      | 0.037807     | 0.047158                 | 1.141731         | 0.011818        | -0.004342        | 18.000000 |
| 0.600000 | 0.025316      | 0.029078     | 0.044376                 | 0.977630         | 0.010397        | -0.003445        | 18.000000 |

## Best d per Asset (by Event IC IR)

| asset           | d        | ic_event_mean | ic_ir_event | ic_overall_mean |
| --------------- | -------- | ------------- | ----------- | --------------- |
| 1INCH/USDT      | 0.400000 | 0.071536      | 5.770270    | 0.012943        |
| ACM/USDT        | 0.600000 | 0.115953      | 4.678867    | 0.079538        |
| ADX/USDT        | 0.600000 | 0.102853      | 3.203569    | 0.027519        |
| ADA/USDT        | 0.200000 | 0.090582      | 2.507946    | 0.021590        |
| ADA/USDC        | 0.300000 | 0.074497      | 2.328690    | 0.017146        |
| ACH/USDC        | 0.200000 | 0.146347      | 2.315800    | 0.090800        |
| 1MBABYDOGE/USDT | 0.400000 | 0.056031      | 2.003806    | -0.015201       |
| ACE/USDT        | 0.200000 | 0.110214      | 1.489108    | 0.029829        |
| AAVE/USDC       | 0.400000 | 0.050630      | 1.365195    | 0.000523        |
| 1000SATS/USDT   | 0.600000 | 0.042925      | 0.840284    | -0.008676       |
| AAVE/USDT       | 0.200000 | 0.049159      | 0.752996    | 0.012650        |
| ACH/USDT        | 0.200000 | 0.046722      | 0.550431    | 0.021229        |
| 1000CHEEMS/USDT | 0.200000 | 0.023046      | 0.467304    | -0.011721       |
| ACT/USDC        | 0.600000 | 0.016533      | 0.454356    | 0.002224        |
| ACT/USDT        | 0.200000 | 0.003987      | 0.049146    | 0.024073        |
| ACX/USDT        | 0.600000 | -0.007124     | -0.116504   | 0.010823        |
| 1000CAT/USDT    | 0.600000 | -0.023393     | -0.263570   | 0.021665        |
| 1000CAT/USDC    | 0.300000 | -0.066641     | -0.759345   | -0.033203       |

## Interpretation

- Higher `ic_event_mean` and `ic_ir_event_mean` suggest better discrimination in high-vol/high-range regimes.
- Compare `ic_event_mean` vs `ic_nonevent_mean` to decide whether a d is event-specialized or broad.
- Larger K(d) means longer memory and higher compute/warmup costs; include this in production trade-offs.
