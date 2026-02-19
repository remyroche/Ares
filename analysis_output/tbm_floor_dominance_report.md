# TBM Floor-Dominance Diagnostics

## Static callsite checks
- `make_effective_tp` callsites: compare `build_barriers`, training `compute_barrier_factory` (raw → scale → clip).
- `sl_method == tp_pct` derives SL from effective TP in compare path.

## Config CFG180A2E0408
- tp_floor_bind_prod_agg=0.000, max_cell_tp_floor_bind_prod=0.000
- real_tp_floor=max(tp_abs_lo_pct=0.0050, tp_min_abs_pct=0.0050, tp_min_bps_floor=0.0050)=0.0050

Worst cells by tp_floor_bind_prod:
| cell        |     n |   tp_floor_bind_prod |   tp_ceil_bind |   tp_raw_q50 |   tp_scaled_q50 |   tp_eff_q50 |   atr_q50 |   net_tp_le_0 |   net_tp_le_20bps |   corr_tp_eff_atr | atr_scaling_inactive   |
|:------------|------:|---------------------:|---------------:|-------------:|----------------:|-------------:|----------:|--------------:|------------------:|------------------:|:-----------------------|
| MR_short_H8 | 25657 |                    0 |       0.328721 |    0.0213027 |       0.0301266 |    0.0301266 |   0.75273 |             0 |                 0 |          0.591495 | False                  |
| TF_short_H8 | 19626 |                    0 |       0.280495 |    0.021     |       0.0296985 |    0.0296985 |   0.4883  |             0 |                 0 |          0.610234 | False                  |

## Config CFG566AEFED85
- tp_floor_bind_prod_agg=0.000, max_cell_tp_floor_bind_prod=0.000
- real_tp_floor=max(tp_abs_lo_pct=0.0050, tp_min_abs_pct=0.0050, tp_min_bps_floor=0.0050)=0.0050

Worst cells by tp_floor_bind_prod:
| cell        |     n |   tp_floor_bind_prod |   tp_ceil_bind |   tp_raw_q50 |   tp_scaled_q50 |   tp_eff_q50 |   atr_q50 |   net_tp_le_0 |   net_tp_le_20bps |   corr_tp_eff_atr | atr_scaling_inactive   |
|:------------|------:|---------------------:|---------------:|-------------:|----------------:|-------------:|----------:|--------------:|------------------:|------------------:|:-----------------------|
| MR_short_H8 | 27636 |                    0 |       0.331271 |      0.03125 |       0.0441941 |    0.0441941 |  0.620801 |             0 |                 0 |          0.431167 | False                  |
| TF_short_H8 | 21427 |                    0 |       0.276334 |      0.03125 |       0.0441941 |    0.0441941 |  0.396414 |             0 |                 0 |          0.418419 | False                  |

## Config CFG5B61E0AD47
- tp_floor_bind_prod_agg=0.000, max_cell_tp_floor_bind_prod=0.000
- real_tp_floor=max(tp_abs_lo_pct=0.0050, tp_min_abs_pct=0.0050, tp_min_bps_floor=0.0050)=0.0050

Worst cells by tp_floor_bind_prod:
| cell        |     n |   tp_floor_bind_prod |   tp_ceil_bind |   tp_raw_q50 |   tp_scaled_q50 |   tp_eff_q50 |   atr_q50 |   net_tp_le_0 |   net_tp_le_20bps |   corr_tp_eff_atr | atr_scaling_inactive   |
|:------------|------:|---------------------:|---------------:|-------------:|----------------:|-------------:|----------:|--------------:|------------------:|------------------:|:-----------------------|
| MR_short_H8 | 24294 |                    0 |       0.298222 |    0.0167116 |       0.0236337 |    0.0236337 |  0.716784 |             0 |                 0 |          0.55439  | False                  |
| TF_short_H8 | 18498 |                    0 |       0.25792  |    0.015     |       0.0212132 |    0.0212132 |  0.340289 |             0 |                 0 |          0.590721 | False                  |

## Config CFG5C32D45F34
- tp_floor_bind_prod_agg=0.000, max_cell_tp_floor_bind_prod=0.000
- real_tp_floor=max(tp_abs_lo_pct=0.0050, tp_min_abs_pct=0.0050, tp_min_bps_floor=0.0050)=0.0050

Worst cells by tp_floor_bind_prod:
| cell        |     n |   tp_floor_bind_prod |   tp_ceil_bind |   tp_raw_q50 |   tp_scaled_q50 |   tp_eff_q50 |   atr_q50 |   net_tp_le_0 |   net_tp_le_20bps |   corr_tp_eff_atr | atr_scaling_inactive   |
|:------------|------:|---------------------:|---------------:|-------------:|----------------:|-------------:|----------:|--------------:|------------------:|------------------:|:-----------------------|
| MR_short_H8 | 24294 |                    0 |       0.298222 |    0.0167116 |       0.0236337 |    0.0236337 |  0.716784 |             0 |                 0 |          0.55439  | False                  |
| TF_short_H8 | 18498 |                    0 |       0.25792  |    0.015     |       0.0212132 |    0.0212132 |  0.340289 |             0 |                 0 |          0.590721 | False                  |

## Config CFG2B5B7B6021
- tp_floor_bind_prod_agg=0.000, max_cell_tp_floor_bind_prod=0.000
- real_tp_floor=max(tp_abs_lo_pct=0.0050, tp_min_abs_pct=0.0050, tp_min_bps_floor=0.0050)=0.0050

Worst cells by tp_floor_bind_prod:
| cell        |     n |   tp_floor_bind_prod |   tp_ceil_bind |   tp_raw_q50 |   tp_scaled_q50 |   tp_eff_q50 |   atr_q50 |   net_tp_le_0 |   net_tp_le_20bps |   corr_tp_eff_atr | atr_scaling_inactive   |
|:------------|------:|---------------------:|---------------:|-------------:|----------------:|-------------:|----------:|--------------:|------------------:|------------------:|:-----------------------|
| MR_short_H8 | 28316 |                    0 |       0.285598 |        0.032 |       0.0452548 |    0.0452548 |  0.565907 |             0 |                 0 |          0.313328 | False                  |
| TF_short_H8 | 22182 |                    0 |       0.236904 |        0.032 |       0.0452548 |    0.0452548 |  0.299305 |             0 |                 0 |          0.287274 | False                  |

## Ranking change (no re-run; from existing results dataframe)
| config_id     |   stage2_score |   floor_penalty |   stage2_score_floor_adj |   rank_old |   rank_new |
|:--------------|---------------:|----------------:|-------------------------:|-----------:|-----------:|
| CFG7C320E7A53 |      0.068004  |       0.0241063 |               0.0438977  |          2 |          1 |
| CFG180A2E0408 |      0.0803855 |       0.0381184 |               0.042267   |          1 |          2 |
| CFGCD7C08989F |      0.0585107 |       0.0162983 |               0.0422124  |          4 |          3 |
| CFG026B1458BD |      0.0599487 |       0.0291054 |               0.0308433  |          3 |          4 |
| CFGE18FE2BE9C |      0.0546632 |       0.0270594 |               0.0276039  |          5 |          5 |
| CFG9AA24A0A48 |      0.0364204 |       0.0164108 |               0.0200095  |         19 |          6 |
| CFG73FAF636B2 |      0.0328315 |       0.0147778 |               0.0180536  |         22 |          7 |
| CFGEB53742DCC |      0.0490722 |       0.0314298 |               0.0176425  |          9 |          8 |
| CFGD83FC832AD |      0.0499207 |       0.0327957 |               0.017125   |          8 |          9 |
| CFG566AEFED85 |      0.0543292 |       0.0395928 |               0.0147364  |          6 |         10 |
| CFG9036C8BDFD |      0.0514745 |       0.0381768 |               0.0132977  |          7 |         11 |
| CFG413D9F9C10 |      0.0278034 |       0.0150154 |               0.012788   |         29 |         12 |
| CFG7A6F4EA7E0 |      0.0252355 |       0.0134431 |               0.0117924  |         35 |         13 |
| CFGBF6B8F8D17 |      0.0310947 |       0.0214361 |               0.00965855 |         24 |         14 |
| CFG25BC692CA6 |      0.0185889 |       0.0096752 |               0.00891367 |         48 |         15 |