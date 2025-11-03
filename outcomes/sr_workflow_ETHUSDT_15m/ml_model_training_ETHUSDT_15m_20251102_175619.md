# ML MODEL TRAINING Report

**Generated:** 2025-11-02 17:56:41
**Symbol:** ETHUSDT
**Exchange:** binance
**Timeframe:** 15m
**Direction:** long
**Mode:** light

---

## Execution Summary

- **Status:** ✅ Success
- **Duration:** 22.58 seconds
- **Step:** ml_model_training

## Metrics

```json
{
  "cv_scores": [
    {
      "fold": 0,
      "train_samples": 43,
      "val_samples": 40,
      "train_rmse": 0.1492316829382799,
      "val_rmse": 0.19216190572986233,
      "train_r2": -0.10885068818370214,
      "val_r2": -0.45595082490512784,
      "train_mae": 0.11904021410825134,
      "val_mae": 0.1471251270150068,
      "num_boost_rounds": 1
    },
    {
      "fold": 1,
      "train_samples": 83,
      "val_samples": 40,
      "train_rmse": 0.16250574614107804,
      "val_rmse": 0.16116937535831072,
      "train_r2": -0.12136196402511557,
      "val_r2": -0.024108047212835215,
      "train_mae": 0.13002088187930846,
      "val_mae": 0.1302413063646622,
      "num_boost_rounds": 1
    },
    {
      "fold": 2,
      "train_samples": 123,
      "val_samples": 40,
      "train_rmse": 0.16545369843298108,
      "val_rmse": 0.1504379855317096,
      "train_r2": -0.1255608020660215,
      "val_r2": -0.035625222889660524,
      "train_mae": 0.13107008629482397,
      "val_mae": 0.11205010198623196,
      "num_boost_rounds": 1
    },
    {
      "fold": 3,
      "train_samples": 163,
      "val_samples": 40,
      "train_rmse": 0.16332000461044316,
      "val_rmse": 0.1809932584486758,
      "train_r2": -0.11819078745634881,
      "val_r2": -0.46584843312596225,
      "train_mae": 0.12659801455539382,
      "val_mae": 0.13020768058166493,
      "num_boost_rounds": 1
    },
    {
      "fold": 4,
      "train_samples": 203,
      "val_samples": 40,
      "train_rmse": 0.16329926005753806,
      "val_rmse": 0.1783738764838958,
      "train_r2": -0.1140730647236472,
      "val_r2": -0.15965227136237248,
      "train_mae": 0.12615189797123416,
      "val_mae": 0.12293760496699715,
      "num_boost_rounds": 1
    }
  ],
  "best_fold": "2",
  "avg_metrics": {
    "avg_val_rmse": 0.17262728031049085,
    "avg_val_r2": -0.22823695989919165,
    "avg_val_mae": 0.12851236418291262,
    "std_val_rmse": 0.014888001455781626,
    "std_val_r2": 0.19584921253605972
  },
  "config": {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "max_depth": 6,
    "lambda_l1": 1.0,
    "lambda_l2": 1.0,
    "min_data_in_leaf": 92,
    "min_gain_to_split": 0.3,
    "learning_rate": 0.03,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.7,
    "bagging_freq": 5,
    "verbose": -1,
    "seed": 42,
    "force_col_wise": true,
    "raw_lambda_l1": 0.8838136419028563,
    "raw_lambda_l2": 0.5358933201225126,
    "raw_learning_rate": 1.5178533030450705,
    "raw_min_gain_to_split": -2.7636966675530217,
    "raw_feature_fraction": -1.6116228212588446,
    "raw_bagging_fraction": 3.625943614523706
  },
  "hpo_results": {
    "best_params": {
      "num_leaves": 31,
      "max_depth": 6,
      "raw_lambda_l1": 0.8838136419028563,
      "raw_lambda_l2": 0.5358933201225126,
      "min_data_in_leaf": 92,
      "raw_learning_rate": 1.5178533030450705,
      "raw_min_gain_to_split": -2.7636966675530217,
      "raw_feature_fraction": -1.6116228212588446,
      "raw_bagging_fraction": 3.625943614523706
    },
    "best_score": -0.03421177056805032,
    "n_trials": 5,
    "optimization_curve": [
      -0.03421177056805032,
      -0.03421177056805032,
      -0.03421177056805032,
      -0.03421177056805032,
      -0.03421177056805032
    ],
    "parameter_importance": {}
  },
  "hpo_best_params": {
    "num_leaves": 31,
    "max_depth": 6,
    "raw_lambda_l1": 0.8838136419028563,
    "raw_lambda_l2": 0.5358933201225126,
    "min_data_in_leaf": 92,
    "raw_learning_rate": 1.5178533030450705,
    "raw_min_gain_to_split": -2.7636966675530217,
    "raw_feature_fraction": -1.6116228212588446,
    "raw_bagging_fraction": 3.625943614523706
  }
}
```

## Artifacts Created

- **ml_training:** {'training_data_path': 'data_cache/sr_ml_training/sr_quality_training_data.parquet', 'model_path': 'models/sr_quality_model.lgb', 'metrics': {'cv_scores': [{'fold': 0, 'train_samples': 43, 'val_samples': 40, 'train_rmse': 0.1492316829382799, 'val_rmse': 0.19216190572986233, 'train_r2': -0.10885068818370214, 'val_r2': -0.45595082490512784, 'train_mae': 0.11904021410825134, 'val_mae': 0.1471251270150068, 'num_boost_rounds': 1}, {'fold': 1, 'train_samples': 83, 'val_samples': 40, 'train_rmse': 0.16250574614107804, 'val_rmse': 0.16116937535831072, 'train_r2': -0.12136196402511557, 'val_r2': -0.024108047212835215, 'train_mae': 0.13002088187930846, 'val_mae': 0.1302413063646622, 'num_boost_rounds': 1}, {'fold': 2, 'train_samples': 123, 'val_samples': 40, 'train_rmse': 0.16545369843298108, 'val_rmse': 0.1504379855317096, 'train_r2': -0.1255608020660215, 'val_r2': -0.035625222889660524, 'train_mae': 0.13107008629482397, 'val_mae': 0.11205010198623196, 'num_boost_rounds': 1}, {'fold': 3, 'train_samples': 163, 'val_samples': 40, 'train_rmse': 0.16332000461044316, 'val_rmse': 0.1809932584486758, 'train_r2': -0.11819078745634881, 'val_r2': -0.46584843312596225, 'train_mae': 0.12659801455539382, 'val_mae': 0.13020768058166493, 'num_boost_rounds': 1}, {'fold': 4, 'train_samples': 203, 'val_samples': 40, 'train_rmse': 0.16329926005753806, 'val_rmse': 0.1783738764838958, 'train_r2': -0.1140730647236472, 'val_r2': -0.15965227136237248, 'train_mae': 0.12615189797123416, 'val_mae': 0.12293760496699715, 'num_boost_rounds': 1}], 'best_fold': 2, 'avg_metrics': {'avg_val_rmse': 0.17262728031049085, 'avg_val_r2': -0.22823695989919165, 'avg_val_mae': 0.12851236418291262, 'std_val_rmse': 0.014888001455781626, 'std_val_r2': 0.19584921253605972}, 'config': {'objective': 'regression', 'metric': 'rmse', 'boosting_type': 'gbdt', 'num_leaves': 31, 'max_depth': 6, 'lambda_l1': 1.0, 'lambda_l2': 1.0, 'min_data_in_leaf': 92, 'min_gain_to_split': 0.3, 'learning_rate': 0.03, 'feature_fraction': 0.7, 'bagging_fraction': 0.7, 'bagging_freq': 5, 'verbose': -1, 'seed': 42, 'force_col_wise': True, 'raw_lambda_l1': 0.8838136419028563, 'raw_lambda_l2': 0.5358933201225126, 'raw_learning_rate': 1.5178533030450705, 'raw_min_gain_to_split': -2.7636966675530217, 'raw_feature_fraction': -1.6116228212588446, 'raw_bagging_fraction': 3.625943614523706}, 'hpo_results': {'best_params': {'num_leaves': 31, 'max_depth': 6, 'raw_lambda_l1': 0.8838136419028563, 'raw_lambda_l2': 0.5358933201225126, 'min_data_in_leaf': 92, 'raw_learning_rate': 1.5178533030450705, 'raw_min_gain_to_split': -2.7636966675530217, 'raw_feature_fraction': -1.6116228212588446, 'raw_bagging_fraction': 3.625943614523706}, 'best_score': -0.03421177056805032, 'n_trials': 5, 'optimization_curve': [-0.03421177056805032, -0.03421177056805032, -0.03421177056805032, -0.03421177056805032, -0.03421177056805032], 'parameter_importance': {}}, 'hpo_best_params': {'num_leaves': 31, 'max_depth': 6, 'raw_lambda_l1': 0.8838136419028563, 'raw_lambda_l2': 0.5358933201225126, 'min_data_in_leaf': 92, 'raw_learning_rate': 1.5178533030450705, 'raw_min_gain_to_split': -2.7636966675530217, 'raw_feature_fraction': -1.6116228212588446, 'raw_bagging_fraction': 3.625943614523706}}, 'shap_report': None}

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
      "train_samples": 43,
      "val_samples": 40,
      "train_rmse": 0.1492316829382799,
      "val_rmse": 0.19216190572986233,
      "train_r2": -0.10885068818370214,
      "val_r2": -0.45595082490512784,
      "train_mae": 0.11904021410825134,
      "val_mae": 0.1471251270150068,
      "num_boost_rounds": 1
    },
    {
      "fold": 1,
      "train_samples": 83,
      "val_samples": 40,
      "train_rmse": 0.16250574614107804,
      "val_rmse": 0.16116937535831072,
      "train_r2": -0.12136196402511557,
      "val_r2": -0.024108047212835215,
      "train_mae": 0.13002088187930846,
      "val_mae": 0.1302413063646622,
      "num_boost_rounds": 1
    },
    {
      "fold": 2,
      "train_samples": 123,
      "val_samples": 40,
      "train_rmse": 0.16545369843298108,
      "val_rmse": 0.1504379855317096,
      "train_r2": -0.1255608020660215,
      "val_r2": -0.035625222889660524,
      "train_mae": 0.13107008629482397,
      "val_mae": 0.11205010198623196,
      "num_boost_rounds": 1
    },
    {
      "fold": 3,
      "train_samples": 163,
      "val_samples": 40,
      "train_rmse": 0.16332000461044316,
      "val_rmse": 0.1809932584486758,
      "train_r2": -0.11819078745634881,
      "val_r2": -0.46584843312596225,
      "train_mae": 0.12659801455539382,
      "val_mae": 0.13020768058166493,
      "num_boost_rounds": 1
    },
    {
      "fold": 4,
      "train_samples": 203,
      "val_samples": 40,
      "train_rmse": 0.16329926005753806,
      "val_rmse": 0.1783738764838958,
      "train_r2": -0.1140730647236472,
      "val_r2": -0.15965227136237248,
      "train_mae": 0.12615189797123416,
      "val_mae": 0.12293760496699715,
      "num_boost_rounds": 1
    }
  ],
  "best_fold": "2",
  "avg_metrics": {
    "avg_val_rmse": 0.17262728031049085,
    "avg_val_r2": -0.22823695989919165,
    "avg_val_mae": 0.12851236418291262,
    "std_val_rmse": 0.014888001455781626,
    "std_val_r2": 0.19584921253605972
  },
  "config": {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "max_depth": 6,
    "lambda_l1": 1.0,
    "lambda_l2": 1.0,
    "min_data_in_leaf": 92,
    "min_gain_to_split": 0.3,
    "learning_rate": 0.03,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.7,
    "bagging_freq": 5,
    "verbose": -1,
    "seed": 42,
    "force_col_wise": true,
    "raw_lambda_l1": 0.8838136419028563,
    "raw_lambda_l2": 0.5358933201225126,
    "raw_learning_rate": 1.5178533030450705,
    "raw_min_gain_to_split": -2.7636966675530217,
    "raw_feature_fraction": -1.6116228212588446,
    "raw_bagging_fraction": 3.625943614523706
  },
  "hpo_results": {
    "best_params": {
      "num_leaves": 31,
      "max_depth": 6,
      "raw_lambda_l1": 0.8838136419028563,
      "raw_lambda_l2": 0.5358933201225126,
      "min_data_in_leaf": 92,
      "raw_learning_rate": 1.5178533030450705,
      "raw_min_gain_to_split": -2.7636966675530217,
      "raw_feature_fraction": -1.6116228212588446,
      "raw_bagging_fraction": 3.625943614523706
    },
    "best_score": -0.03421177056805032,
    "n_trials": 5,
    "optimization_curve": [
      -0.03421177056805032,
      -0.03421177056805032,
      -0.03421177056805032,
      -0.03421177056805032,
      -0.03421177056805032
    ],
    "parameter_importance": {}
  },
  "hpo_best_params": {
    "num_leaves": 31,
    "max_depth": 6,
    "raw_lambda_l1": 0.8838136419028563,
    "raw_lambda_l2": 0.5358933201225126,
    "min_data_in_leaf": 92,
    "raw_learning_rate": 1.5178533030450705,
    "raw_min_gain_to_split": -2.7636966675530217,
    "raw_feature_fraction": -1.6116228212588446,
    "raw_bagging_fraction": 3.625943614523706
  }
}
```
