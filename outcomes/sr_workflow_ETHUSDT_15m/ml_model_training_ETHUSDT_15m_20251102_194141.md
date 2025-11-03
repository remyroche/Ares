# ML MODEL TRAINING Report

**Generated:** 2025-11-02 19:41:59
**Symbol:** ETHUSDT
**Exchange:** binance
**Timeframe:** 15m
**Direction:** long
**Mode:** light

---

## Execution Summary

- **Status:** ✅ Success
- **Duration:** 17.86 seconds
- **Step:** ml_model_training

## Metrics

```json
{
  "cv_scores": [
    {
      "fold": 0,
      "train_samples": 67,
      "val_samples": 67,
      "train_rmse": 0.17836138202741603,
      "val_rmse": 0.14260087506290825,
      "train_r2": -0.1820146521328807,
      "val_r2": -0.01912624216427905,
      "train_mae": 0.1536263824612734,
      "val_mae": 0.12028435836619499,
      "num_boost_rounds": 1
    },
    {
      "fold": 1,
      "train_samples": 134,
      "val_samples": 67,
      "train_rmse": 0.17135075093245516,
      "val_rmse": 0.17373473355890626,
      "train_r2": -0.15424603613240984,
      "val_r2": -0.07260592523195464,
      "train_mae": 0.14038133293225455,
      "val_mae": 0.1409449454277341,
      "num_boost_rounds": 1
    },
    {
      "fold": 2,
      "train_samples": 201,
      "val_samples": 67,
      "train_rmse": 0.17427723580086749,
      "val_rmse": 0.16888728776800818,
      "train_r2": -0.15021125384644662,
      "val_r2": -0.21452654545689454,
      "train_mae": 0.14037947360192435,
      "val_mae": 0.13352148478661488,
      "num_boost_rounds": 3
    }
  ],
  "best_fold": "0",
  "avg_metrics": {
    "avg_val_rmse": 0.16174096546327424,
    "avg_val_r2": -0.10208623761770941,
    "avg_val_mae": 0.13158359619351465,
    "std_val_rmse": 0.013678004924563142,
    "std_val_r2": 0.08245053565818637
  },
  "config": {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "max_depth": 6,
    "lambda_l1": 1.0,
    "lambda_l2": 1.0,
    "min_data_in_leaf": 47,
    "min_gain_to_split": 0.3,
    "learning_rate": 0.03,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.7,
    "bagging_freq": 5,
    "verbose": -1,
    "seed": 42,
    "force_col_wise": true,
    "log_lambda_l1": 1.200052379578691,
    "log_lambda_l2": 2.565655557052992,
    "raw_learning_rate": -4.9748901608844935,
    "raw_min_gain_to_split": 0.47350331850331084,
    "raw_feature_fraction": -2.68754191758673,
    "raw_bagging_fraction": -2.1694384984984407
  },
  "feature_importance": [
    {
      "feature": "feature_strength",
      "importance": 0.0,
      "importance_pct": NaN
    },
    {
      "feature": "feature_touch_count",
      "importance": 0.0,
      "importance_pct": NaN
    },
    {
      "feature": "feature_age_bars",
      "importance": 0.0,
      "importance_pct": NaN
    },
    {
      "feature": "feature_consistency",
      "importance": 0.0,
      "importance_pct": NaN
    },
    {
      "feature": "feature_avg_bounce_ratio",
      "importance": 0.0,
      "importance_pct": NaN
    },
    {
      "feature": "feature_max_bounce_ratio",
      "importance": 0.0,
      "importance_pct": NaN
    },
    {
      "feature": "feature_volume_confirmation",
      "importance": 0.0,
      "importance_pct": NaN
    },
    {
      "feature": "feature_bounce_consistency",
      "importance": 0.0,
      "importance_pct": NaN
    },
    {
      "feature": "feature_recency_weighted_strength",
      "importance": 0.0,
      "importance_pct": NaN
    },
    {
      "feature": "feature_touch_quality_score",
      "importance": 0.0,
      "importance_pct": NaN
    },
    {
      "feature": "feature_price_zscore",
      "importance": 0.0,
      "importance_pct": NaN
    },
    {
      "feature": "feature_distance_to_current_pct",
      "importance": 0.0,
      "importance_pct": NaN
    },
    {
      "feature": "feature_is_support",
      "importance": 0.0,
      "importance_pct": NaN
    },
    {
      "feature": "feature_market_trend",
      "importance": 0.0,
      "importance_pct": NaN
    },
    {
      "feature": "feature_is_uptrend",
      "importance": 0.0,
      "importance_pct": NaN
    },
    {
      "feature": "feature_quality_tier",
      "importance": 0.0,
      "importance_pct": NaN
    }
  ],
  "hpo_results": {
    "best_params": {
      "num_leaves": 31,
      "max_depth": 6,
      "log_lambda_l1": 1.200052379578691,
      "log_lambda_l2": 2.565655557052992,
      "min_data_in_leaf": 47,
      "raw_learning_rate": -4.9748901608844935,
      "raw_min_gain_to_split": 0.47350331850331084,
      "raw_feature_fraction": -2.68754191758673,
      "raw_bagging_fraction": -2.1694384984984407,
      "bagging_freq": 5
    },
    "best_score": -0.032026863194349615,
    "n_trials": 5,
    "optimization_curve": [
      -0.032026863194349615,
      -0.032026863194349615,
      -0.032026863194349615,
      -0.032026863194349615,
      -0.032026863194349615
    ],
    "parameter_importance": {}
  },
  "hpo_best_params": {
    "num_leaves": 31,
    "max_depth": 6,
    "log_lambda_l1": 1.200052379578691,
    "log_lambda_l2": 2.565655557052992,
    "min_data_in_leaf": 47,
    "raw_learning_rate": -4.9748901608844935,
    "raw_min_gain_to_split": 0.47350331850331084,
    "raw_feature_fraction": -2.68754191758673,
    "raw_bagging_fraction": -2.1694384984984407,
    "bagging_freq": 5
  }
}
```

## Artifacts Created

- **ml_training:** {'training_data_path': 'data_cache/sr_ml_training/sr_quality_training_data.parquet', 'model_path': 'models/sr_quality_model.lgb', 'metrics': {'cv_scores': [{'fold': 0, 'train_samples': 67, 'val_samples': 67, 'train_rmse': 0.17836138202741603, 'val_rmse': 0.14260087506290825, 'train_r2': -0.1820146521328807, 'val_r2': -0.01912624216427905, 'train_mae': 0.1536263824612734, 'val_mae': 0.12028435836619499, 'num_boost_rounds': 1}, {'fold': 1, 'train_samples': 134, 'val_samples': 67, 'train_rmse': 0.17135075093245516, 'val_rmse': 0.17373473355890626, 'train_r2': -0.15424603613240984, 'val_r2': -0.07260592523195464, 'train_mae': 0.14038133293225455, 'val_mae': 0.1409449454277341, 'num_boost_rounds': 1}, {'fold': 2, 'train_samples': 201, 'val_samples': 67, 'train_rmse': 0.17427723580086749, 'val_rmse': 0.16888728776800818, 'train_r2': -0.15021125384644662, 'val_r2': -0.21452654545689454, 'train_mae': 0.14037947360192435, 'val_mae': 0.13352148478661488, 'num_boost_rounds': 3}], 'best_fold': 0, 'avg_metrics': {'avg_val_rmse': 0.16174096546327424, 'avg_val_r2': -0.10208623761770941, 'avg_val_mae': 0.13158359619351465, 'std_val_rmse': 0.013678004924563142, 'std_val_r2': 0.08245053565818637}, 'config': {'objective': 'regression', 'metric': 'rmse', 'boosting_type': 'gbdt', 'num_leaves': 31, 'max_depth': 6, 'lambda_l1': 1.0, 'lambda_l2': 1.0, 'min_data_in_leaf': 47, 'min_gain_to_split': 0.3, 'learning_rate': 0.03, 'feature_fraction': 0.7, 'bagging_fraction': 0.7, 'bagging_freq': 5, 'verbose': -1, 'seed': 42, 'force_col_wise': True, 'log_lambda_l1': 1.200052379578691, 'log_lambda_l2': 2.565655557052992, 'raw_learning_rate': -4.9748901608844935, 'raw_min_gain_to_split': 0.47350331850331084, 'raw_feature_fraction': -2.68754191758673, 'raw_bagging_fraction': -2.1694384984984407}, 'feature_importance': [{'feature': 'feature_strength', 'importance': 0.0, 'importance_pct': nan}, {'feature': 'feature_touch_count', 'importance': 0.0, 'importance_pct': nan}, {'feature': 'feature_age_bars', 'importance': 0.0, 'importance_pct': nan}, {'feature': 'feature_consistency', 'importance': 0.0, 'importance_pct': nan}, {'feature': 'feature_avg_bounce_ratio', 'importance': 0.0, 'importance_pct': nan}, {'feature': 'feature_max_bounce_ratio', 'importance': 0.0, 'importance_pct': nan}, {'feature': 'feature_volume_confirmation', 'importance': 0.0, 'importance_pct': nan}, {'feature': 'feature_bounce_consistency', 'importance': 0.0, 'importance_pct': nan}, {'feature': 'feature_recency_weighted_strength', 'importance': 0.0, 'importance_pct': nan}, {'feature': 'feature_touch_quality_score', 'importance': 0.0, 'importance_pct': nan}, {'feature': 'feature_price_zscore', 'importance': 0.0, 'importance_pct': nan}, {'feature': 'feature_distance_to_current_pct', 'importance': 0.0, 'importance_pct': nan}, {'feature': 'feature_is_support', 'importance': 0.0, 'importance_pct': nan}, {'feature': 'feature_market_trend', 'importance': 0.0, 'importance_pct': nan}, {'feature': 'feature_is_uptrend', 'importance': 0.0, 'importance_pct': nan}, {'feature': 'feature_quality_tier', 'importance': 0.0, 'importance_pct': nan}], 'hpo_results': {'best_params': {'num_leaves': 31, 'max_depth': 6, 'log_lambda_l1': 1.200052379578691, 'log_lambda_l2': 2.565655557052992, 'min_data_in_leaf': 47, 'raw_learning_rate': -4.9748901608844935, 'raw_min_gain_to_split': 0.47350331850331084, 'raw_feature_fraction': -2.68754191758673, 'raw_bagging_fraction': -2.1694384984984407, 'bagging_freq': 5}, 'best_score': -0.032026863194349615, 'n_trials': 5, 'optimization_curve': [-0.032026863194349615, -0.032026863194349615, -0.032026863194349615, -0.032026863194349615, -0.032026863194349615], 'parameter_importance': {}}, 'hpo_best_params': {'num_leaves': 31, 'max_depth': 6, 'log_lambda_l1': 1.200052379578691, 'log_lambda_l2': 2.565655557052992, 'min_data_in_leaf': 47, 'raw_learning_rate': -4.9748901608844935, 'raw_min_gain_to_split': 0.47350331850331084, 'raw_feature_fraction': -2.68754191758673, 'raw_bagging_fraction': -2.1694384984984407, 'bagging_freq': 5}}, 'shap_report': None}

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
      "train_samples": 67,
      "val_samples": 67,
      "train_rmse": 0.17836138202741603,
      "val_rmse": 0.14260087506290825,
      "train_r2": -0.1820146521328807,
      "val_r2": -0.01912624216427905,
      "train_mae": 0.1536263824612734,
      "val_mae": 0.12028435836619499,
      "num_boost_rounds": 1
    },
    {
      "fold": 1,
      "train_samples": 134,
      "val_samples": 67,
      "train_rmse": 0.17135075093245516,
      "val_rmse": 0.17373473355890626,
      "train_r2": -0.15424603613240984,
      "val_r2": -0.07260592523195464,
      "train_mae": 0.14038133293225455,
      "val_mae": 0.1409449454277341,
      "num_boost_rounds": 1
    },
    {
      "fold": 2,
      "train_samples": 201,
      "val_samples": 67,
      "train_rmse": 0.17427723580086749,
      "val_rmse": 0.16888728776800818,
      "train_r2": -0.15021125384644662,
      "val_r2": -0.21452654545689454,
      "train_mae": 0.14037947360192435,
      "val_mae": 0.13352148478661488,
      "num_boost_rounds": 3
    }
  ],
  "best_fold": "0",
  "avg_metrics": {
    "avg_val_rmse": 0.16174096546327424,
    "avg_val_r2": -0.10208623761770941,
    "avg_val_mae": 0.13158359619351465,
    "std_val_rmse": 0.013678004924563142,
    "std_val_r2": 0.08245053565818637
  },
  "config": {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "max_depth": 6,
    "lambda_l1": 1.0,
    "lambda_l2": 1.0,
    "min_data_in_leaf": 47,
    "min_gain_to_split": 0.3,
    "learning_rate": 0.03,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.7,
    "bagging_freq": 5,
    "verbose": -1,
    "seed": 42,
    "force_col_wise": true,
    "log_lambda_l1": 1.200052379578691,
    "log_lambda_l2": 2.565655557052992,
    "raw_learning_rate": -4.9748901608844935,
    "raw_min_gain_to_split": 0.47350331850331084,
    "raw_feature_fraction": -2.68754191758673,
    "raw_bagging_fraction": -2.1694384984984407
  },
  "feature_importance": [
    {
      "feature": "feature_strength",
      "importance": 0.0,
      "importance_pct": NaN
    },
    {
      "feature": "feature_touch_count",
      "importance": 0.0,
      "importance_pct": NaN
    },
    {
      "feature": "feature_age_bars",
      "importance": 0.0,
      "importance_pct": NaN
    },
    {
      "feature": "feature_consistency",
      "importance": 0.0,
      "importance_pct": NaN
    },
    {
      "feature": "feature_avg_bounce_ratio",
      "importance": 0.0,
      "importance_pct": NaN
    },
    {
      "feature": "feature_max_bounce_ratio",
      "importance": 0.0,
      "importance_pct": NaN
    },
    {
      "feature": "feature_volume_confirmation",
      "importance": 0.0,
      "importance_pct": NaN
    },
    {
      "feature": "feature_bounce_consistency",
      "importance": 0.0,
      "importance_pct": NaN
    },
    {
      "feature": "feature_recency_weighted_strength",
      "importance": 0.0,
      "importance_pct": NaN
    },
    {
      "feature": "feature_touch_quality_score",
      "importance": 0.0,
      "importance_pct": NaN
    },
    {
      "feature": "feature_price_zscore",
      "importance": 0.0,
      "importance_pct": NaN
    },
    {
      "feature": "feature_distance_to_current_pct",
      "importance": 0.0,
      "importance_pct": NaN
    },
    {
      "feature": "feature_is_support",
      "importance": 0.0,
      "importance_pct": NaN
    },
    {
      "feature": "feature_market_trend",
      "importance": 0.0,
      "importance_pct": NaN
    },
    {
      "feature": "feature_is_uptrend",
      "importance": 0.0,
      "importance_pct": NaN
    },
    {
      "feature": "feature_quality_tier",
      "importance": 0.0,
      "importance_pct": NaN
    }
  ],
  "hpo_results": {
    "best_params": {
      "num_leaves": 31,
      "max_depth": 6,
      "log_lambda_l1": 1.200052379578691,
      "log_lambda_l2": 2.565655557052992,
      "min_data_in_leaf": 47,
      "raw_learning_rate": -4.9748901608844935,
      "raw_min_gain_to_split": 0.47350331850331084,
      "raw_feature_fraction": -2.68754191758673,
      "raw_bagging_fraction": -2.1694384984984407,
      "bagging_freq": 5
    },
    "best_score": -0.032026863194349615,
    "n_trials": 5,
    "optimization_curve": [
      -0.032026863194349615,
      -0.032026863194349615,
      -0.032026863194349615,
      -0.032026863194349615,
      -0.032026863194349615
    ],
    "parameter_importance": {}
  },
  "hpo_best_params": {
    "num_leaves": 31,
    "max_depth": 6,
    "log_lambda_l1": 1.200052379578691,
    "log_lambda_l2": 2.565655557052992,
    "min_data_in_leaf": 47,
    "raw_learning_rate": -4.9748901608844935,
    "raw_min_gain_to_split": 0.47350331850331084,
    "raw_feature_fraction": -2.68754191758673,
    "raw_bagging_fraction": -2.1694384984984407,
    "bagging_freq": 5
  }
}
```
