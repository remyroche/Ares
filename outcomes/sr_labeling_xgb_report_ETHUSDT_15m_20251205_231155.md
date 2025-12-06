# sr_labeling_xgb XGBoost Report

**Symbol:** ETHUSDT | **Exchange:** binance | **Timeframe:** 15m

**Samples:** 409

## Performance

| Variant | AUC |
|---------|-----|
| Baseline | 0.7951 |
| Tuned | 0.7951 |

**AUC Improvement:** 0.0000

## Baseline Parameters

```json
{
  "n_estimators": 300,
  "max_depth": 6,
  "learning_rate": 0.05,
  "min_child_weight": 20.0,
  "subsample": 0.8,
  "colsample_bytree": 0.8,
  "gamma": 1.0,
  "reg_lambda": 3.0,
  "reg_alpha": 0.5
}
```

## Tuned Parameters (sr_labeling_xgb)

```json
{
  "max_depth": 6,
  "learning_rate": 0.03574712922600244,
  "gamma": 2.49816047538945,
  "min_child_weight": 39.01428612819832,
  "lambda": 3.9279757672456204,
  "alpha": 0.6789267873576292,
  "colsample_bytree": 0.6312037280884872,
  "subsample": 0.7311989040672405
}
```

