# ML MODEL TRAINING Report

**Generated:** 2025-11-01 15:53:55
**Symbol:** ETHUSDT
**Exchange:** binance
**Timeframe:** 15m
**Direction:** long
**Mode:** light

---

## Execution Summary

- **Status:** ✅ Success
- **Duration:** 192.14 seconds
- **Step:** ml_model_training

## Metrics

```json
{
  "cv_scores": [
    {
      "fold": 0,
      "train_samples": 646,
      "val_samples": 646,
      "train_rmse": 0.1973227424948091,
      "val_rmse": 0.23574112726420382,
      "train_r2": 0.3113084768479838,
      "val_r2": 0.09004051007726455,
      "train_mae": 0.17101992876909128,
      "val_mae": 0.21283513072240462,
      "num_boost_rounds": 98
    },
    {
      "fold": 1,
      "train_samples": 1292,
      "val_samples": 646,
      "train_rmse": 0.19438876643757147,
      "val_rmse": 0.22551613255547312,
      "train_r2": 0.35857580396814115,
      "val_r2": 0.14431867441903534,
      "train_mae": 0.1695540070479609,
      "val_mae": 0.19584762530488184,
      "num_boost_rounds": 140
    },
    {
      "fold": 2,
      "train_samples": 1938,
      "val_samples": 646,
      "train_rmse": 0.2062220337739724,
      "val_rmse": 0.22333778787184205,
      "train_r2": 0.2806012045760937,
      "val_r2": 0.14764351097264905,
      "train_mae": 0.18136098010994406,
      "val_mae": 0.19653256867463795,
      "num_boost_rounds": 89
    },
    {
      "fold": 3,
      "train_samples": 2584,
      "val_samples": 646,
      "train_rmse": 0.20864068475242878,
      "val_rmse": 0.2396421902999061,
      "train_r2": 0.2659530127996329,
      "val_r2": 0.17422733459974582,
      "train_mae": 0.18196825087946325,
      "val_mae": 0.20574260767269803,
      "num_boost_rounds": 80
    },
    {
      "fold": 4,
      "train_samples": 3230,
      "val_samples": 646,
      "train_rmse": 0.19605968226611195,
      "val_rmse": 0.22262206009877417,
      "train_r2": 0.3737259394917255,
      "val_r2": 0.19059932484841335,
      "train_mae": 0.16729552787783517,
      "val_mae": 0.19211344588194762,
      "num_boost_rounds": 210
    }
  ],
  "best_fold": "4",
  "avg_metrics": {
    "avg_val_rmse": 0.22937185961803985,
    "avg_val_r2": 0.14936587098342163,
    "avg_val_mae": 0.200614275651314,
    "std_val_rmse": 0.006969701704275524,
    "std_val_r2": 0.034252676551166816
  },
  "config": {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "num_leaves": 15,
    "max_depth": 4,
    "lambda_l1": 1.0,
    "lambda_l2": 1.0,
    "min_data_in_leaf": 50,
    "learning_rate": 0.03,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.7,
    "bagging_freq": 5,
    "verbose": -1,
    "seed": 42,
    "force_col_wise": true
  }
}
```

## Artifacts Created

- **ml_training:** {'training_data_path': 'data_cache/sr_ml_training/sr_quality_training_data.parquet', 'model_path': 'models/sr_quality_model.lgb', 'metrics': {'cv_scores': [{'fold': 0, 'train_samples': 646, 'val_samples': 646, 'train_rmse': 0.1973227424948091, 'val_rmse': 0.23574112726420382, 'train_r2': 0.3113084768479838, 'val_r2': 0.09004051007726455, 'train_mae': 0.17101992876909128, 'val_mae': 0.21283513072240462, 'num_boost_rounds': 98}, {'fold': 1, 'train_samples': 1292, 'val_samples': 646, 'train_rmse': 0.19438876643757147, 'val_rmse': 0.22551613255547312, 'train_r2': 0.35857580396814115, 'val_r2': 0.14431867441903534, 'train_mae': 0.1695540070479609, 'val_mae': 0.19584762530488184, 'num_boost_rounds': 140}, {'fold': 2, 'train_samples': 1938, 'val_samples': 646, 'train_rmse': 0.2062220337739724, 'val_rmse': 0.22333778787184205, 'train_r2': 0.2806012045760937, 'val_r2': 0.14764351097264905, 'train_mae': 0.18136098010994406, 'val_mae': 0.19653256867463795, 'num_boost_rounds': 89}, {'fold': 3, 'train_samples': 2584, 'val_samples': 646, 'train_rmse': 0.20864068475242878, 'val_rmse': 0.2396421902999061, 'train_r2': 0.2659530127996329, 'val_r2': 0.17422733459974582, 'train_mae': 0.18196825087946325, 'val_mae': 0.20574260767269803, 'num_boost_rounds': 80}, {'fold': 4, 'train_samples': 3230, 'val_samples': 646, 'train_rmse': 0.19605968226611195, 'val_rmse': 0.22262206009877417, 'train_r2': 0.3737259394917255, 'val_r2': 0.19059932484841335, 'train_mae': 0.16729552787783517, 'val_mae': 0.19211344588194762, 'num_boost_rounds': 210}], 'best_fold': 4, 'avg_metrics': {'avg_val_rmse': 0.22937185961803985, 'avg_val_r2': 0.14936587098342163, 'avg_val_mae': 0.200614275651314, 'std_val_rmse': 0.006969701704275524, 'std_val_r2': 0.034252676551166816}, 'config': {'objective': 'regression', 'metric': 'rmse', 'boosting_type': 'gbdt', 'num_leaves': 15, 'max_depth': 4, 'lambda_l1': 1.0, 'lambda_l2': 1.0, 'min_data_in_leaf': 50, 'learning_rate': 0.03, 'feature_fraction': 0.7, 'bagging_fraction': 0.7, 'bagging_freq': 5, 'verbose': -1, 'seed': 42, 'force_col_wise': True}}, 'shap_report': None}

## ML Model Training Details

- **Training Data Path:** data_cache/sr_ml_training/sr_quality_training_data.parquet
- **Model Path:** models/sr_quality_model.lgb
- **SHAP Report:** None

### Cross-Validation Metrics

```json
{
  "cv_scores": [
    {
      "fold": 0,
      "train_samples": 646,
      "val_samples": 646,
      "train_rmse": 0.1973227424948091,
      "val_rmse": 0.23574112726420382,
      "train_r2": 0.3113084768479838,
      "val_r2": 0.09004051007726455,
      "train_mae": 0.17101992876909128,
      "val_mae": 0.21283513072240462,
      "num_boost_rounds": 98
    },
    {
      "fold": 1,
      "train_samples": 1292,
      "val_samples": 646,
      "train_rmse": 0.19438876643757147,
      "val_rmse": 0.22551613255547312,
      "train_r2": 0.35857580396814115,
      "val_r2": 0.14431867441903534,
      "train_mae": 0.1695540070479609,
      "val_mae": 0.19584762530488184,
      "num_boost_rounds": 140
    },
    {
      "fold": 2,
      "train_samples": 1938,
      "val_samples": 646,
      "train_rmse": 0.2062220337739724,
      "val_rmse": 0.22333778787184205,
      "train_r2": 0.2806012045760937,
      "val_r2": 0.14764351097264905,
      "train_mae": 0.18136098010994406,
      "val_mae": 0.19653256867463795,
      "num_boost_rounds": 89
    },
    {
      "fold": 3,
      "train_samples": 2584,
      "val_samples": 646,
      "train_rmse": 0.20864068475242878,
      "val_rmse": 0.2396421902999061,
      "train_r2": 0.2659530127996329,
      "val_r2": 0.17422733459974582,
      "train_mae": 0.18196825087946325,
      "val_mae": 0.20574260767269803,
      "num_boost_rounds": 80
    },
    {
      "fold": 4,
      "train_samples": 3230,
      "val_samples": 646,
      "train_rmse": 0.19605968226611195,
      "val_rmse": 0.22262206009877417,
      "train_r2": 0.3737259394917255,
      "val_r2": 0.19059932484841335,
      "train_mae": 0.16729552787783517,
      "val_mae": 0.19211344588194762,
      "num_boost_rounds": 210
    }
  ],
  "best_fold": "4",
  "avg_metrics": {
    "avg_val_rmse": 0.22937185961803985,
    "avg_val_r2": 0.14936587098342163,
    "avg_val_mae": 0.200614275651314,
    "std_val_rmse": 0.006969701704275524,
    "std_val_r2": 0.034252676551166816
  },
  "config": {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "num_leaves": 15,
    "max_depth": 4,
    "lambda_l1": 1.0,
    "lambda_l2": 1.0,
    "min_data_in_leaf": 50,
    "learning_rate": 0.03,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.7,
    "bagging_freq": 5,
    "verbose": -1,
    "seed": 42,
    "force_col_wise": true
  }
}
```
