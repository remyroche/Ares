# ML MODEL TRAINING Report

**Generated:** 2025-11-01 17:26:53
**Symbol:** ETHUSDT
**Exchange:** binance
**Timeframe:** 15m
**Direction:** long
**Mode:** light

---

## Execution Summary

- **Status:** ✅ Success
- **Duration:** 46.83 seconds
- **Step:** ml_model_training

## Metrics

```json
{
  "cv_scores": [
    {
      "fold": 0,
      "train_samples": 93,
      "val_samples": 93,
      "train_rmse": 0.19228273261036857,
      "val_rmse": 0.24653097811801838,
      "train_r2": -0.3001044036100755,
      "val_r2": -0.4068518928983602,
      "train_mae": 0.1493622049440145,
      "val_mae": 0.20392416813625247,
      "num_boost_rounds": 1
    },
    {
      "fold": 1,
      "train_samples": 186,
      "val_samples": 93,
      "train_rmse": 0.22209851767436478,
      "val_rmse": 0.2194608565416961,
      "train_r2": -0.36175949805787644,
      "val_r2": -0.23616754513565397,
      "train_mae": 0.1775224980912524,
      "val_mae": 0.17054840017704395,
      "num_boost_rounds": 1
    },
    {
      "fold": 2,
      "train_samples": 279,
      "val_samples": 93,
      "train_rmse": 0.22250362676412885,
      "val_rmse": 0.22866248158055383,
      "train_r2": -0.33040271656530606,
      "val_r2": -0.27171150486089846,
      "train_mae": 0.17492579944519226,
      "val_mae": 0.1884316065796343,
      "num_boost_rounds": 55
    },
    {
      "fold": 3,
      "train_samples": 372,
      "val_samples": 93,
      "train_rmse": 0.22280583932244583,
      "val_rmse": 0.27297748375336534,
      "train_r2": -0.299861470067984,
      "val_r2": -0.42211821884608414,
      "train_mae": 0.17570521280017512,
      "val_mae": 0.2225447245304473,
      "num_boost_rounds": 90
    },
    {
      "fold": 4,
      "train_samples": 465,
      "val_samples": 93,
      "train_rmse": 0.23888863434547258,
      "val_rmse": 0.22980757799752025,
      "train_r2": -0.3859739212415094,
      "val_r2": -0.2502528998531015,
      "train_mae": 0.19185420766831465,
      "val_mae": 0.18070742812569482,
      "num_boost_rounds": 1
    }
  ],
  "best_fold": "1",
  "avg_metrics": {
    "avg_val_rmse": 0.23948787559823076,
    "avg_val_r2": -0.3174204123188197,
    "avg_val_mae": 0.19323126550981456,
    "std_val_rmse": 0.01888458607995301,
    "std_val_r2": 0.08020277774190071
  },
  "config": {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "num_leaves": 13,
    "max_depth": 5,
    "lambda_l1": 1.0,
    "lambda_l2": 1.0,
    "min_data_in_leaf": 88,
    "learning_rate": 0.03,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.7,
    "bagging_freq": 5,
    "verbose": -1,
    "seed": 42,
    "force_col_wise": true,
    "raw_lambda_l1": 0.900975778653713,
    "raw_lambda_l2": 0.8466875627849348,
    "raw_learning_rate": -3.8333138351448435,
    "raw_feature_fraction": -4.534687687066004,
    "raw_bagging_fraction": -4.328348464651514
  },
  "hpo_results": {
    "best_params": {
      "num_leaves": 13,
      "max_depth": 5,
      "raw_lambda_l1": 0.900975778653713,
      "raw_lambda_l2": 0.8466875627849348,
      "min_data_in_leaf": 88,
      "raw_learning_rate": -3.8333138351448435,
      "raw_feature_fraction": -4.534687687066004,
      "raw_bagging_fraction": -4.328348464651514
    },
    "best_score": -0.04427985723974884,
    "n_trials": 26,
    "optimization_curve": [
      -0.04433375677019095,
      -0.04450419823731677,
      -0.04455385845889107,
      -0.044359658331777865,
      -0.044940282434997016,
      -0.04462457587649706,
      -0.04439474282948022,
      -0.04444703198679152,
      -0.04436514283243008,
      -0.0444328755303786,
      -0.04440304149030877,
      -0.04428743290364129,
      -0.04431669966252475,
      -0.04430442033729161,
      -0.04429765278866375,
      -0.044289643905964314,
      -0.04432783397911999,
      -0.04428707736125584,
      -0.044302192726886916,
      -0.0443157723143366,
      -0.044345828129666705,
      -0.04427985723974884,
      -0.044292713341098675,
      -0.04430261896661562,
      -0.044291812140534015,
      -0.04430167497001431
    ],
    "parameter_importance": {
      "raw_lambda_l1": 0.3245027908564638,
      "min_data_in_leaf": 0.2654434655431052,
      "raw_bagging_fraction": 0.20119075199777728,
      "raw_learning_rate": 0.12780571251274733,
      "raw_lambda_l2": 0.03460930759601574,
      "raw_feature_fraction": 0.025045819882476974,
      "num_leaves": 0.019341362560262823,
      "max_depth": 0.002060789051150886
    }
  },
  "hpo_best_params": {
    "num_leaves": 13,
    "max_depth": 5,
    "raw_lambda_l1": 0.900975778653713,
    "raw_lambda_l2": 0.8466875627849348,
    "min_data_in_leaf": 88,
    "raw_learning_rate": -3.8333138351448435,
    "raw_feature_fraction": -4.534687687066004,
    "raw_bagging_fraction": -4.328348464651514
  },
  "ranking_metrics": {
    "precision_at_k": 0.0,
    "spearman_rho": NaN,
    "spearman_p_value": NaN,
    "ndcg_at_k": 0.3041341971367416,
    "r2_score": -2.87428040066209,
    "rmse": 0.5037770379750011,
    "k": 10,
    "quality_threshold": 0.7,
    "total_samples": 1821
  }
}
```

## Artifacts Created

- **ml_training:** {'training_data_path': 'data_cache/sr_ml_training/sr_quality_training_data.parquet', 'model_path': 'models/sr_quality_model.lgb', 'metrics': {'cv_scores': [{'fold': 0, 'train_samples': 93, 'val_samples': 93, 'train_rmse': 0.19228273261036857, 'val_rmse': 0.24653097811801838, 'train_r2': -0.3001044036100755, 'val_r2': -0.4068518928983602, 'train_mae': 0.1493622049440145, 'val_mae': 0.20392416813625247, 'num_boost_rounds': 1}, {'fold': 1, 'train_samples': 186, 'val_samples': 93, 'train_rmse': 0.22209851767436478, 'val_rmse': 0.2194608565416961, 'train_r2': -0.36175949805787644, 'val_r2': -0.23616754513565397, 'train_mae': 0.1775224980912524, 'val_mae': 0.17054840017704395, 'num_boost_rounds': 1}, {'fold': 2, 'train_samples': 279, 'val_samples': 93, 'train_rmse': 0.22250362676412885, 'val_rmse': 0.22866248158055383, 'train_r2': -0.33040271656530606, 'val_r2': -0.27171150486089846, 'train_mae': 0.17492579944519226, 'val_mae': 0.1884316065796343, 'num_boost_rounds': 55}, {'fold': 3, 'train_samples': 372, 'val_samples': 93, 'train_rmse': 0.22280583932244583, 'val_rmse': 0.27297748375336534, 'train_r2': -0.299861470067984, 'val_r2': -0.42211821884608414, 'train_mae': 0.17570521280017512, 'val_mae': 0.2225447245304473, 'num_boost_rounds': 90}, {'fold': 4, 'train_samples': 465, 'val_samples': 93, 'train_rmse': 0.23888863434547258, 'val_rmse': 0.22980757799752025, 'train_r2': -0.3859739212415094, 'val_r2': -0.2502528998531015, 'train_mae': 0.19185420766831465, 'val_mae': 0.18070742812569482, 'num_boost_rounds': 1}], 'best_fold': 1, 'avg_metrics': {'avg_val_rmse': 0.23948787559823076, 'avg_val_r2': -0.3174204123188197, 'avg_val_mae': 0.19323126550981456, 'std_val_rmse': 0.01888458607995301, 'std_val_r2': 0.08020277774190071}, 'config': {'objective': 'regression', 'metric': 'rmse', 'boosting_type': 'gbdt', 'num_leaves': 13, 'max_depth': 5, 'lambda_l1': 1.0, 'lambda_l2': 1.0, 'min_data_in_leaf': 88, 'learning_rate': 0.03, 'feature_fraction': 0.7, 'bagging_fraction': 0.7, 'bagging_freq': 5, 'verbose': -1, 'seed': 42, 'force_col_wise': True, 'raw_lambda_l1': 0.900975778653713, 'raw_lambda_l2': 0.8466875627849348, 'raw_learning_rate': -3.8333138351448435, 'raw_feature_fraction': -4.534687687066004, 'raw_bagging_fraction': -4.328348464651514}, 'hpo_results': {'best_params': {'num_leaves': 13, 'max_depth': 5, 'raw_lambda_l1': 0.900975778653713, 'raw_lambda_l2': 0.8466875627849348, 'min_data_in_leaf': 88, 'raw_learning_rate': -3.8333138351448435, 'raw_feature_fraction': -4.534687687066004, 'raw_bagging_fraction': -4.328348464651514}, 'best_score': -0.04427985723974884, 'n_trials': 26, 'optimization_curve': [-0.04433375677019095, -0.04450419823731677, -0.04455385845889107, -0.044359658331777865, -0.044940282434997016, -0.04462457587649706, -0.04439474282948022, -0.04444703198679152, -0.04436514283243008, -0.0444328755303786, -0.04440304149030877, -0.04428743290364129, -0.04431669966252475, -0.04430442033729161, -0.04429765278866375, -0.044289643905964314, -0.04432783397911999, -0.04428707736125584, -0.044302192726886916, -0.0443157723143366, -0.044345828129666705, -0.04427985723974884, -0.044292713341098675, -0.04430261896661562, -0.044291812140534015, -0.04430167497001431], 'parameter_importance': {'raw_lambda_l1': 0.3245027908564638, 'min_data_in_leaf': 0.2654434655431052, 'raw_bagging_fraction': 0.20119075199777728, 'raw_learning_rate': 0.12780571251274733, 'raw_lambda_l2': 0.03460930759601574, 'raw_feature_fraction': 0.025045819882476974, 'num_leaves': 0.019341362560262823, 'max_depth': 0.002060789051150886}}, 'hpo_best_params': {'num_leaves': 13, 'max_depth': 5, 'raw_lambda_l1': 0.900975778653713, 'raw_lambda_l2': 0.8466875627849348, 'min_data_in_leaf': 88, 'raw_learning_rate': -3.8333138351448435, 'raw_feature_fraction': -4.534687687066004, 'raw_bagging_fraction': -4.328348464651514}, 'ranking_metrics': {'precision_at_k': 0.0, 'spearman_rho': nan, 'spearman_p_value': nan, 'ndcg_at_k': 0.3041341971367416, 'r2_score': -2.87428040066209, 'rmse': 0.5037770379750011, 'k': 10, 'quality_threshold': 0.7, 'total_samples': 1821}}, 'shap_report': None}

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
      "train_samples": 93,
      "val_samples": 93,
      "train_rmse": 0.19228273261036857,
      "val_rmse": 0.24653097811801838,
      "train_r2": -0.3001044036100755,
      "val_r2": -0.4068518928983602,
      "train_mae": 0.1493622049440145,
      "val_mae": 0.20392416813625247,
      "num_boost_rounds": 1
    },
    {
      "fold": 1,
      "train_samples": 186,
      "val_samples": 93,
      "train_rmse": 0.22209851767436478,
      "val_rmse": 0.2194608565416961,
      "train_r2": -0.36175949805787644,
      "val_r2": -0.23616754513565397,
      "train_mae": 0.1775224980912524,
      "val_mae": 0.17054840017704395,
      "num_boost_rounds": 1
    },
    {
      "fold": 2,
      "train_samples": 279,
      "val_samples": 93,
      "train_rmse": 0.22250362676412885,
      "val_rmse": 0.22866248158055383,
      "train_r2": -0.33040271656530606,
      "val_r2": -0.27171150486089846,
      "train_mae": 0.17492579944519226,
      "val_mae": 0.1884316065796343,
      "num_boost_rounds": 55
    },
    {
      "fold": 3,
      "train_samples": 372,
      "val_samples": 93,
      "train_rmse": 0.22280583932244583,
      "val_rmse": 0.27297748375336534,
      "train_r2": -0.299861470067984,
      "val_r2": -0.42211821884608414,
      "train_mae": 0.17570521280017512,
      "val_mae": 0.2225447245304473,
      "num_boost_rounds": 90
    },
    {
      "fold": 4,
      "train_samples": 465,
      "val_samples": 93,
      "train_rmse": 0.23888863434547258,
      "val_rmse": 0.22980757799752025,
      "train_r2": -0.3859739212415094,
      "val_r2": -0.2502528998531015,
      "train_mae": 0.19185420766831465,
      "val_mae": 0.18070742812569482,
      "num_boost_rounds": 1
    }
  ],
  "best_fold": "1",
  "avg_metrics": {
    "avg_val_rmse": 0.23948787559823076,
    "avg_val_r2": -0.3174204123188197,
    "avg_val_mae": 0.19323126550981456,
    "std_val_rmse": 0.01888458607995301,
    "std_val_r2": 0.08020277774190071
  },
  "config": {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "num_leaves": 13,
    "max_depth": 5,
    "lambda_l1": 1.0,
    "lambda_l2": 1.0,
    "min_data_in_leaf": 88,
    "learning_rate": 0.03,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.7,
    "bagging_freq": 5,
    "verbose": -1,
    "seed": 42,
    "force_col_wise": true,
    "raw_lambda_l1": 0.900975778653713,
    "raw_lambda_l2": 0.8466875627849348,
    "raw_learning_rate": -3.8333138351448435,
    "raw_feature_fraction": -4.534687687066004,
    "raw_bagging_fraction": -4.328348464651514
  },
  "hpo_results": {
    "best_params": {
      "num_leaves": 13,
      "max_depth": 5,
      "raw_lambda_l1": 0.900975778653713,
      "raw_lambda_l2": 0.8466875627849348,
      "min_data_in_leaf": 88,
      "raw_learning_rate": -3.8333138351448435,
      "raw_feature_fraction": -4.534687687066004,
      "raw_bagging_fraction": -4.328348464651514
    },
    "best_score": -0.04427985723974884,
    "n_trials": 26,
    "optimization_curve": [
      -0.04433375677019095,
      -0.04450419823731677,
      -0.04455385845889107,
      -0.044359658331777865,
      -0.044940282434997016,
      -0.04462457587649706,
      -0.04439474282948022,
      -0.04444703198679152,
      -0.04436514283243008,
      -0.0444328755303786,
      -0.04440304149030877,
      -0.04428743290364129,
      -0.04431669966252475,
      -0.04430442033729161,
      -0.04429765278866375,
      -0.044289643905964314,
      -0.04432783397911999,
      -0.04428707736125584,
      -0.044302192726886916,
      -0.0443157723143366,
      -0.044345828129666705,
      -0.04427985723974884,
      -0.044292713341098675,
      -0.04430261896661562,
      -0.044291812140534015,
      -0.04430167497001431
    ],
    "parameter_importance": {
      "raw_lambda_l1": 0.3245027908564638,
      "min_data_in_leaf": 0.2654434655431052,
      "raw_bagging_fraction": 0.20119075199777728,
      "raw_learning_rate": 0.12780571251274733,
      "raw_lambda_l2": 0.03460930759601574,
      "raw_feature_fraction": 0.025045819882476974,
      "num_leaves": 0.019341362560262823,
      "max_depth": 0.002060789051150886
    }
  },
  "hpo_best_params": {
    "num_leaves": 13,
    "max_depth": 5,
    "raw_lambda_l1": 0.900975778653713,
    "raw_lambda_l2": 0.8466875627849348,
    "min_data_in_leaf": 88,
    "raw_learning_rate": -3.8333138351448435,
    "raw_feature_fraction": -4.534687687066004,
    "raw_bagging_fraction": -4.328348464651514
  },
  "ranking_metrics": {
    "precision_at_k": 0.0,
    "spearman_rho": NaN,
    "spearman_p_value": NaN,
    "ndcg_at_k": 0.3041341971367416,
    "r2_score": -2.87428040066209,
    "rmse": 0.5037770379750011,
    "k": 10,
    "quality_threshold": 0.7,
    "total_samples": 1821
  }
}
```
