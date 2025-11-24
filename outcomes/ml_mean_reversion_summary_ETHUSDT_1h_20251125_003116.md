# ML Mean-Reversion Regime Summary for ETHUSDT (1h)

## Teacher (OU/Hurst GMM)

- Components: 3
- Mean-reversion cluster: 0
- Cluster counts: {1: 9581, 0: 9363, 2: 4823, -1: 1701}

## Student (XGB Regressor)

- train: R2=0.9569, RMSE=0.002829, ACC=0.5000, F1=0.0000
- val: R2=0.9448, RMSE=0.003760, ACC=0.6924, F1=0.0000
- test: R2=0.9478, RMSE=0.004270, ACC=0.7179, F1=0.0000

## Calibration

- mu_long=0.017124, sigma_long=0.013820, min_std=0.008292

## Walk-Forward Stability

- folds=5
- R2 mean=0.9293, std=0.0304
- RMSE mean=0.003770, std=0.000463
- ACC mean=1.0000, std=0.0000
- F1 mean=0.0000, std=0.0000

## Feature WCoV (weighted by teacher mean-reversion labels)


## Forward-Return Diagnostics

### Horizon 5 bars

- n_samples=25423
- mean_fwd_return=0.000183
- std_fwd_return=0.014895
- corr_prob_fwd=0.0009

### Horizon 10 bars

- n_samples=25418
- mean_fwd_return=0.000363
- std_fwd_return=0.021315
- corr_prob_fwd=-0.0074

### Horizon 20 bars

- n_samples=25408
- mean_fwd_return=0.000726
- std_fwd_return=0.030632
- corr_prob_fwd=-0.0063

