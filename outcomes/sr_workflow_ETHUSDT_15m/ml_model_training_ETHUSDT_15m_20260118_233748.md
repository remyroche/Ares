# ML MODEL TRAINING Report

**Generated:** 2026-01-18 23:40:16
**Symbol:** ETHUSDT
**Exchange:** binance
**Timeframe:** 15m
**Direction:** long
**Mode:** light

---

## Execution Summary

- **Status:** ✅ Success
- **Duration:** 147.32 seconds
- **Step:** ml_model_training

## Metrics

```json
{
  "cv_scores": [
    {
      "fold": 0,
      "train_samples": 356,
      "val_samples": 353,
      "train_rmse": 0.14524043918221527,
      "val_rmse": 0.17915150848500655,
      "train_r2": -0.11738862545390538,
      "val_r2": -0.07031046593190582,
      "train_mae": 0.11995352266133695,
      "val_mae": 0.14569437417247258,
      "num_boost_rounds": 4
    },
    {
      "fold": 1,
      "train_samples": 709,
      "val_samples": 353,
      "train_rmse": 0.15957633559477308,
      "val_rmse": 0.20678440708221676,
      "train_r2": 0.053367964718161964,
      "val_r2": -0.015195839795876465,
      "train_mae": 0.1306632915526181,
      "val_mae": 0.17865683055222734,
      "num_boost_rounds": 54
    },
    {
      "fold": 2,
      "train_samples": 1062,
      "val_samples": 353,
      "train_rmse": 0.16612861077256047,
      "val_rmse": 0.22815915515326066,
      "train_r2": 0.2587797134717783,
      "val_r2": -0.15290960261688746,
      "train_mae": 0.13249797118325524,
      "val_mae": 0.18636894828133432,
      "num_boost_rounds": 72
    }
  ],
  "best_fold": "0",
  "avg_metrics": {
    "avg_val_rmse": 0.204698356906828,
    "avg_val_r2": -0.07947196944822325,
    "avg_val_mae": 0.17024005100201142,
    "std_val_rmse": 0.020061589534743404,
    "std_val_r2": 0.056593403488333874
  },
  "config": {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "num_leaves": 26,
    "max_depth": 4,
    "lambda_l1": 1.0,
    "lambda_l2": 1.0,
    "min_data_in_leaf": 45,
    "min_gain_to_split": 0.3,
    "learning_rate": 0.03,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.7,
    "bagging_freq": 5,
    "verbose": -1,
    "seed": 42,
    "force_col_wise": true,
    "log_lambda_l1": 0.6480125731710232,
    "log_lambda_l2": 2.124539528426067,
    "raw_learning_rate": -2.494264217577382,
    "raw_min_gain_to_split": 0.3663618432936917,
    "raw_feature_fraction": -0.5271601893955689,
    "raw_bagging_fraction": 3.4221115367161623
  },
  "feature_importance": [
    {
      "feature": "feature_distance_to_current_pct",
      "importance": 0.9880509972572327,
      "importance_pct": 100.0
    },
    {
      "feature": "feature_strength",
      "importance": 0.0,
      "importance_pct": 0.0
    },
    {
      "feature": "feature_touch_quality_score",
      "importance": 0.0,
      "importance_pct": 0.0
    },
    {
      "feature": "feature_is_uptrend",
      "importance": 0.0,
      "importance_pct": 0.0
    },
    {
      "feature": "feature_hour_of_day",
      "importance": 0.0,
      "importance_pct": 0.0
    },
    {
      "feature": "feature_market_trend",
      "importance": 0.0,
      "importance_pct": 0.0
    },
    {
      "feature": "feature_is_support",
      "importance": 0.0,
      "importance_pct": 0.0
    },
    {
      "feature": "feature_price_zscore",
      "importance": 0.0,
      "importance_pct": 0.0
    },
    {
      "feature": "feature_recency_weighted_strength",
      "importance": 0.0,
      "importance_pct": 0.0
    },
    {
      "feature": "feature_touch_count",
      "importance": 0.0,
      "importance_pct": 0.0
    },
    {
      "feature": "feature_bounce_consistency",
      "importance": 0.0,
      "importance_pct": 0.0
    },
    {
      "feature": "feature_volume_confirmation",
      "importance": 0.0,
      "importance_pct": 0.0
    },
    {
      "feature": "feature_max_bounce_ratio",
      "importance": 0.0,
      "importance_pct": 0.0
    },
    {
      "feature": "feature_avg_bounce_ratio",
      "importance": 0.0,
      "importance_pct": 0.0
    },
    {
      "feature": "feature_consistency",
      "importance": 0.0,
      "importance_pct": 0.0
    },
    {
      "feature": "feature_age_bars",
      "importance": 0.0,
      "importance_pct": 0.0
    },
    {
      "feature": "feature_quality_tier",
      "importance": 0.0,
      "importance_pct": 0.0
    }
  ],
  "hpo_results": {
    "best_params": {
      "num_leaves": 26,
      "max_depth": 4,
      "log_lambda_l1": 0.6480125731710232,
      "log_lambda_l2": 2.124539528426067,
      "min_data_in_leaf": 45,
      "raw_learning_rate": -2.494264217577382,
      "raw_min_gain_to_split": 0.3663618432936917,
      "raw_feature_fraction": -0.5271601893955689,
      "raw_bagging_fraction": 3.4221115367161623,
      "bagging_freq": 3
    },
    "best_score": -0.061560324602238224,
    "n_trials": 5,
    "optimization_curve": [
      -0.061563358066214637,
      -0.06157532421114711,
      -0.061560324602238224,
      -0.061572736628149584,
      -0.06157532421114711
    ],
    "parameter_importance": {
      "log_lambda_l2": 0.3618295255824648,
      "raw_feature_fraction": 0.1713849232060852,
      "raw_learning_rate": 0.10851954740788056,
      "num_leaves": 0.09478409589109418,
      "log_lambda_l1": 0.06452810887099507,
      "raw_min_gain_to_split": 0.06368379123947024,
      "min_data_in_leaf": 0.05368276147831973,
      "max_depth": 0.03250385338647517,
      "raw_bagging_fraction": 0.03243268753444235,
      "bagging_freq": 0.016650705402772723
    }
  },
  "hpo_best_params": {
    "num_leaves": 26,
    "max_depth": 4,
    "log_lambda_l1": 0.6480125731710232,
    "log_lambda_l2": 2.124539528426067,
    "min_data_in_leaf": 45,
    "raw_learning_rate": -2.494264217577382,
    "raw_min_gain_to_split": 0.3663618432936917,
    "raw_feature_fraction": -0.5271601893955689,
    "raw_bagging_fraction": 3.4221115367161623,
    "bagging_freq": 3
  }
}
```

## Artifacts Created

- **ml_training:** {'training_data_path': 'data_cache/sr_ml_training/sr_quality_training_data.parquet', 'model_path': 'models/sr_quality_model.lgb', 'metrics': {'cv_scores': [{'fold': 0, 'train_samples': 356, 'val_samples': 353, 'train_rmse': 0.14524043918221527, 'val_rmse': 0.17915150848500655, 'train_r2': -0.11738862545390538, 'val_r2': -0.07031046593190582, 'train_mae': 0.11995352266133695, 'val_mae': 0.14569437417247258, 'num_boost_rounds': 4}, {'fold': 1, 'train_samples': 709, 'val_samples': 353, 'train_rmse': 0.15957633559477308, 'val_rmse': 0.20678440708221676, 'train_r2': 0.053367964718161964, 'val_r2': -0.015195839795876465, 'train_mae': 0.1306632915526181, 'val_mae': 0.17865683055222734, 'num_boost_rounds': 54}, {'fold': 2, 'train_samples': 1062, 'val_samples': 353, 'train_rmse': 0.16612861077256047, 'val_rmse': 0.22815915515326066, 'train_r2': 0.2587797134717783, 'val_r2': -0.15290960261688746, 'train_mae': 0.13249797118325524, 'val_mae': 0.18636894828133432, 'num_boost_rounds': 72}], 'best_fold': 0, 'avg_metrics': {'avg_val_rmse': 0.204698356906828, 'avg_val_r2': -0.07947196944822325, 'avg_val_mae': 0.17024005100201142, 'std_val_rmse': 0.020061589534743404, 'std_val_r2': 0.056593403488333874}, 'config': {'objective': 'regression', 'metric': 'rmse', 'boosting_type': 'gbdt', 'num_leaves': 26, 'max_depth': 4, 'lambda_l1': 1.0, 'lambda_l2': 1.0, 'min_data_in_leaf': 45, 'min_gain_to_split': 0.3, 'learning_rate': 0.03, 'feature_fraction': 0.7, 'bagging_fraction': 0.7, 'bagging_freq': 5, 'verbose': -1, 'seed': 42, 'force_col_wise': True, 'log_lambda_l1': 0.6480125731710232, 'log_lambda_l2': 2.124539528426067, 'raw_learning_rate': -2.494264217577382, 'raw_min_gain_to_split': 0.3663618432936917, 'raw_feature_fraction': -0.5271601893955689, 'raw_bagging_fraction': 3.4221115367161623}, 'feature_importance': [{'feature': 'feature_distance_to_current_pct', 'importance': 0.9880509972572327, 'importance_pct': 100.0}, {'feature': 'feature_strength', 'importance': 0.0, 'importance_pct': 0.0}, {'feature': 'feature_touch_quality_score', 'importance': 0.0, 'importance_pct': 0.0}, {'feature': 'feature_is_uptrend', 'importance': 0.0, 'importance_pct': 0.0}, {'feature': 'feature_hour_of_day', 'importance': 0.0, 'importance_pct': 0.0}, {'feature': 'feature_market_trend', 'importance': 0.0, 'importance_pct': 0.0}, {'feature': 'feature_is_support', 'importance': 0.0, 'importance_pct': 0.0}, {'feature': 'feature_price_zscore', 'importance': 0.0, 'importance_pct': 0.0}, {'feature': 'feature_recency_weighted_strength', 'importance': 0.0, 'importance_pct': 0.0}, {'feature': 'feature_touch_count', 'importance': 0.0, 'importance_pct': 0.0}, {'feature': 'feature_bounce_consistency', 'importance': 0.0, 'importance_pct': 0.0}, {'feature': 'feature_volume_confirmation', 'importance': 0.0, 'importance_pct': 0.0}, {'feature': 'feature_max_bounce_ratio', 'importance': 0.0, 'importance_pct': 0.0}, {'feature': 'feature_avg_bounce_ratio', 'importance': 0.0, 'importance_pct': 0.0}, {'feature': 'feature_consistency', 'importance': 0.0, 'importance_pct': 0.0}, {'feature': 'feature_age_bars', 'importance': 0.0, 'importance_pct': 0.0}, {'feature': 'feature_quality_tier', 'importance': 0.0, 'importance_pct': 0.0}], 'hpo_results': {'best_params': {'num_leaves': 26, 'max_depth': 4, 'log_lambda_l1': 0.6480125731710232, 'log_lambda_l2': 2.124539528426067, 'min_data_in_leaf': 45, 'raw_learning_rate': -2.494264217577382, 'raw_min_gain_to_split': 0.3663618432936917, 'raw_feature_fraction': -0.5271601893955689, 'raw_bagging_fraction': 3.4221115367161623, 'bagging_freq': 3}, 'best_score': -0.061560324602238224, 'n_trials': 5, 'optimization_curve': [-0.061563358066214637, -0.06157532421114711, -0.061560324602238224, -0.061572736628149584, -0.06157532421114711], 'parameter_importance': {'log_lambda_l2': 0.3618295255824648, 'raw_feature_fraction': 0.1713849232060852, 'raw_learning_rate': 0.10851954740788056, 'num_leaves': 0.09478409589109418, 'log_lambda_l1': 0.06452810887099507, 'raw_min_gain_to_split': 0.06368379123947024, 'min_data_in_leaf': 0.05368276147831973, 'max_depth': 0.03250385338647517, 'raw_bagging_fraction': 0.03243268753444235, 'bagging_freq': 0.016650705402772723}}, 'hpo_best_params': {'num_leaves': 26, 'max_depth': 4, 'log_lambda_l1': 0.6480125731710232, 'log_lambda_l2': 2.124539528426067, 'min_data_in_leaf': 45, 'raw_learning_rate': -2.494264217577382, 'raw_min_gain_to_split': 0.3663618432936917, 'raw_feature_fraction': -0.5271601893955689, 'raw_bagging_fraction': 3.4221115367161623, 'bagging_freq': 3}}, 'shap_report': None}

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
      "train_samples": 356,
      "val_samples": 353,
      "train_rmse": 0.14524043918221527,
      "val_rmse": 0.17915150848500655,
      "train_r2": -0.11738862545390538,
      "val_r2": -0.07031046593190582,
      "train_mae": 0.11995352266133695,
      "val_mae": 0.14569437417247258,
      "num_boost_rounds": 4
    },
    {
      "fold": 1,
      "train_samples": 709,
      "val_samples": 353,
      "train_rmse": 0.15957633559477308,
      "val_rmse": 0.20678440708221676,
      "train_r2": 0.053367964718161964,
      "val_r2": -0.015195839795876465,
      "train_mae": 0.1306632915526181,
      "val_mae": 0.17865683055222734,
      "num_boost_rounds": 54
    },
    {
      "fold": 2,
      "train_samples": 1062,
      "val_samples": 353,
      "train_rmse": 0.16612861077256047,
      "val_rmse": 0.22815915515326066,
      "train_r2": 0.2587797134717783,
      "val_r2": -0.15290960261688746,
      "train_mae": 0.13249797118325524,
      "val_mae": 0.18636894828133432,
      "num_boost_rounds": 72
    }
  ],
  "best_fold": "0",
  "avg_metrics": {
    "avg_val_rmse": 0.204698356906828,
    "avg_val_r2": -0.07947196944822325,
    "avg_val_mae": 0.17024005100201142,
    "std_val_rmse": 0.020061589534743404,
    "std_val_r2": 0.056593403488333874
  },
  "config": {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "num_leaves": 26,
    "max_depth": 4,
    "lambda_l1": 1.0,
    "lambda_l2": 1.0,
    "min_data_in_leaf": 45,
    "min_gain_to_split": 0.3,
    "learning_rate": 0.03,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.7,
    "bagging_freq": 5,
    "verbose": -1,
    "seed": 42,
    "force_col_wise": true,
    "log_lambda_l1": 0.6480125731710232,
    "log_lambda_l2": 2.124539528426067,
    "raw_learning_rate": -2.494264217577382,
    "raw_min_gain_to_split": 0.3663618432936917,
    "raw_feature_fraction": -0.5271601893955689,
    "raw_bagging_fraction": 3.4221115367161623
  },
  "feature_importance": [
    {
      "feature": "feature_distance_to_current_pct",
      "importance": 0.9880509972572327,
      "importance_pct": 100.0
    },
    {
      "feature": "feature_strength",
      "importance": 0.0,
      "importance_pct": 0.0
    },
    {
      "feature": "feature_touch_quality_score",
      "importance": 0.0,
      "importance_pct": 0.0
    },
    {
      "feature": "feature_is_uptrend",
      "importance": 0.0,
      "importance_pct": 0.0
    },
    {
      "feature": "feature_hour_of_day",
      "importance": 0.0,
      "importance_pct": 0.0
    },
    {
      "feature": "feature_market_trend",
      "importance": 0.0,
      "importance_pct": 0.0
    },
    {
      "feature": "feature_is_support",
      "importance": 0.0,
      "importance_pct": 0.0
    },
    {
      "feature": "feature_price_zscore",
      "importance": 0.0,
      "importance_pct": 0.0
    },
    {
      "feature": "feature_recency_weighted_strength",
      "importance": 0.0,
      "importance_pct": 0.0
    },
    {
      "feature": "feature_touch_count",
      "importance": 0.0,
      "importance_pct": 0.0
    },
    {
      "feature": "feature_bounce_consistency",
      "importance": 0.0,
      "importance_pct": 0.0
    },
    {
      "feature": "feature_volume_confirmation",
      "importance": 0.0,
      "importance_pct": 0.0
    },
    {
      "feature": "feature_max_bounce_ratio",
      "importance": 0.0,
      "importance_pct": 0.0
    },
    {
      "feature": "feature_avg_bounce_ratio",
      "importance": 0.0,
      "importance_pct": 0.0
    },
    {
      "feature": "feature_consistency",
      "importance": 0.0,
      "importance_pct": 0.0
    },
    {
      "feature": "feature_age_bars",
      "importance": 0.0,
      "importance_pct": 0.0
    },
    {
      "feature": "feature_quality_tier",
      "importance": 0.0,
      "importance_pct": 0.0
    }
  ],
  "hpo_results": {
    "best_params": {
      "num_leaves": 26,
      "max_depth": 4,
      "log_lambda_l1": 0.6480125731710232,
      "log_lambda_l2": 2.124539528426067,
      "min_data_in_leaf": 45,
      "raw_learning_rate": -2.494264217577382,
      "raw_min_gain_to_split": 0.3663618432936917,
      "raw_feature_fraction": -0.5271601893955689,
      "raw_bagging_fraction": 3.4221115367161623,
      "bagging_freq": 3
    },
    "best_score": -0.061560324602238224,
    "n_trials": 5,
    "optimization_curve": [
      -0.061563358066214637,
      -0.06157532421114711,
      -0.061560324602238224,
      -0.061572736628149584,
      -0.06157532421114711
    ],
    "parameter_importance": {
      "log_lambda_l2": 0.3618295255824648,
      "raw_feature_fraction": 0.1713849232060852,
      "raw_learning_rate": 0.10851954740788056,
      "num_leaves": 0.09478409589109418,
      "log_lambda_l1": 0.06452810887099507,
      "raw_min_gain_to_split": 0.06368379123947024,
      "min_data_in_leaf": 0.05368276147831973,
      "max_depth": 0.03250385338647517,
      "raw_bagging_fraction": 0.03243268753444235,
      "bagging_freq": 0.016650705402772723
    }
  },
  "hpo_best_params": {
    "num_leaves": 26,
    "max_depth": 4,
    "log_lambda_l1": 0.6480125731710232,
    "log_lambda_l2": 2.124539528426067,
    "min_data_in_leaf": 45,
    "raw_learning_rate": -2.494264217577382,
    "raw_min_gain_to_split": 0.3663618432936917,
    "raw_feature_fraction": -0.5271601893955689,
    "raw_bagging_fraction": 3.4221115367161623,
    "bagging_freq": 3
  }
}
```
