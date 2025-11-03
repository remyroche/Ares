# ML MODEL TRAINING Report

**Generated:** 2025-11-01 13:21:20
**Symbol:** ETHUSDT
**Exchange:** binance
**Timeframe:** 15m
**Direction:** long
**Mode:** light

---

## Execution Summary

- **Status:** ✅ Success
- **Duration:** 24.48 seconds
- **Step:** ml_model_training

## Metrics

```json
{
  "cv_scores": [
    {
      "fold": 0,
      "train_samples": 161,
      "val_samples": 157,
      "train_rmse": 0.2178812260042305,
      "val_rmse": 0.25415628490197284,
      "train_r2": 0.3410028575247901,
      "val_r2": 0.02999648460082127,
      "train_mae": 0.17034037592103815,
      "val_mae": 0.1919975264524635,
      "num_boost_rounds": 10
    },
    {
      "fold": 1,
      "train_samples": 318,
      "val_samples": 157,
      "train_rmse": 0.14238257302662333,
      "val_rmse": 0.23719476640883114,
      "train_r2": 0.7090713690000523,
      "val_r2": 0.15520530004866584,
      "train_mae": 0.10217913801554364,
      "val_mae": 0.17547970046429132,
      "num_boost_rounds": 74
    },
    {
      "fold": 2,
      "train_samples": 475,
      "val_samples": 157,
      "train_rmse": 0.12602352302164332,
      "val_rmse": 0.20633020090426782,
      "train_r2": 0.7687210502019072,
      "val_r2": 0.3855395158243683,
      "train_mae": 0.08996669010883561,
      "val_mae": 0.1566652307658949,
      "num_boost_rounds": 102
    },
    {
      "fold": 3,
      "train_samples": 632,
      "val_samples": 157,
      "train_rmse": 0.20985928113565397,
      "val_rmse": 0.24982557370228542,
      "train_r2": 0.36304316762222233,
      "val_r2": 0.08474972618142385,
      "train_mae": 0.1656139403373342,
      "val_mae": 0.19600385868850537,
      "num_boost_rounds": 13
    },
    {
      "fold": 4,
      "train_samples": 789,
      "val_samples": 157,
      "train_rmse": 0.24351929456173768,
      "val_rmse": 0.24155083240590436,
      "train_r2": 0.14007257984123034,
      "val_r2": -0.016866130352635622,
      "train_mae": 0.19468032257118204,
      "val_mae": 0.20128657022518043,
      "num_boost_rounds": 4
    }
  ],
  "best_fold": "2",
  "avg_metrics": {
    "avg_val_rmse": 0.23781153166465235,
    "avg_val_r2": 0.12772497926052873,
    "avg_val_mae": 0.1842865773192671,
    "std_val_rmse": 0.0168340436041372,
    "std_val_r2": 0.14108708184598473
  },
  "config": {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "max_depth": 6,
    "min_data_in_leaf": 20,
    "lambda_l1": 0.1,
    "lambda_l2": 0.1,
    "verbose": -1,
    "seed": 42,
    "force_col_wise": true
  }
}
```

## Artifacts Created

- **ml_training:** {'training_data_path': 'data_cache/sr_ml_training/sr_quality_training_data.parquet', 'model_path': 'models/sr_quality_model.lgb', 'metrics': {'cv_scores': [{'fold': 0, 'train_samples': 161, 'val_samples': 157, 'train_rmse': 0.2178812260042305, 'val_rmse': 0.25415628490197284, 'train_r2': 0.3410028575247901, 'val_r2': 0.02999648460082127, 'train_mae': 0.17034037592103815, 'val_mae': 0.1919975264524635, 'num_boost_rounds': 10}, {'fold': 1, 'train_samples': 318, 'val_samples': 157, 'train_rmse': 0.14238257302662333, 'val_rmse': 0.23719476640883114, 'train_r2': 0.7090713690000523, 'val_r2': 0.15520530004866584, 'train_mae': 0.10217913801554364, 'val_mae': 0.17547970046429132, 'num_boost_rounds': 74}, {'fold': 2, 'train_samples': 475, 'val_samples': 157, 'train_rmse': 0.12602352302164332, 'val_rmse': 0.20633020090426782, 'train_r2': 0.7687210502019072, 'val_r2': 0.3855395158243683, 'train_mae': 0.08996669010883561, 'val_mae': 0.1566652307658949, 'num_boost_rounds': 102}, {'fold': 3, 'train_samples': 632, 'val_samples': 157, 'train_rmse': 0.20985928113565397, 'val_rmse': 0.24982557370228542, 'train_r2': 0.36304316762222233, 'val_r2': 0.08474972618142385, 'train_mae': 0.1656139403373342, 'val_mae': 0.19600385868850537, 'num_boost_rounds': 13}, {'fold': 4, 'train_samples': 789, 'val_samples': 157, 'train_rmse': 0.24351929456173768, 'val_rmse': 0.24155083240590436, 'train_r2': 0.14007257984123034, 'val_r2': -0.016866130352635622, 'train_mae': 0.19468032257118204, 'val_mae': 0.20128657022518043, 'num_boost_rounds': 4}], 'best_fold': 2, 'avg_metrics': {'avg_val_rmse': 0.23781153166465235, 'avg_val_r2': 0.12772497926052873, 'avg_val_mae': 0.1842865773192671, 'std_val_rmse': 0.0168340436041372, 'std_val_r2': 0.14108708184598473}, 'config': {'objective': 'regression', 'metric': 'rmse', 'boosting_type': 'gbdt', 'num_leaves': 31, 'learning_rate': 0.05, 'feature_fraction': 0.9, 'bagging_fraction': 0.8, 'bagging_freq': 5, 'max_depth': 6, 'min_data_in_leaf': 20, 'lambda_l1': 0.1, 'lambda_l2': 0.1, 'verbose': -1, 'seed': 42, 'force_col_wise': True}}, 'shap_report': None}

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
      "train_samples": 161,
      "val_samples": 157,
      "train_rmse": 0.2178812260042305,
      "val_rmse": 0.25415628490197284,
      "train_r2": 0.3410028575247901,
      "val_r2": 0.02999648460082127,
      "train_mae": 0.17034037592103815,
      "val_mae": 0.1919975264524635,
      "num_boost_rounds": 10
    },
    {
      "fold": 1,
      "train_samples": 318,
      "val_samples": 157,
      "train_rmse": 0.14238257302662333,
      "val_rmse": 0.23719476640883114,
      "train_r2": 0.7090713690000523,
      "val_r2": 0.15520530004866584,
      "train_mae": 0.10217913801554364,
      "val_mae": 0.17547970046429132,
      "num_boost_rounds": 74
    },
    {
      "fold": 2,
      "train_samples": 475,
      "val_samples": 157,
      "train_rmse": 0.12602352302164332,
      "val_rmse": 0.20633020090426782,
      "train_r2": 0.7687210502019072,
      "val_r2": 0.3855395158243683,
      "train_mae": 0.08996669010883561,
      "val_mae": 0.1566652307658949,
      "num_boost_rounds": 102
    },
    {
      "fold": 3,
      "train_samples": 632,
      "val_samples": 157,
      "train_rmse": 0.20985928113565397,
      "val_rmse": 0.24982557370228542,
      "train_r2": 0.36304316762222233,
      "val_r2": 0.08474972618142385,
      "train_mae": 0.1656139403373342,
      "val_mae": 0.19600385868850537,
      "num_boost_rounds": 13
    },
    {
      "fold": 4,
      "train_samples": 789,
      "val_samples": 157,
      "train_rmse": 0.24351929456173768,
      "val_rmse": 0.24155083240590436,
      "train_r2": 0.14007257984123034,
      "val_r2": -0.016866130352635622,
      "train_mae": 0.19468032257118204,
      "val_mae": 0.20128657022518043,
      "num_boost_rounds": 4
    }
  ],
  "best_fold": "2",
  "avg_metrics": {
    "avg_val_rmse": 0.23781153166465235,
    "avg_val_r2": 0.12772497926052873,
    "avg_val_mae": 0.1842865773192671,
    "std_val_rmse": 0.0168340436041372,
    "std_val_r2": 0.14108708184598473
  },
  "config": {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "max_depth": 6,
    "min_data_in_leaf": 20,
    "lambda_l1": 0.1,
    "lambda_l2": 0.1,
    "verbose": -1,
    "seed": 42,
    "force_col_wise": true
  }
}
```
