# src/training/enhanced_coarse_optimizer.py

import pandas as pd
import pandas_ta as ta
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
import optuna
from optuna.pruners import SuccessiveHalvingPruner
import shap
import time
from typing import Dict, List, Tuple, Any

from src.database.sqlite_manager import SQLiteManager
from src.analyst.feature_engineering import FeatureEngineeringEngine
from src.analyst.ml_target_generator import MLTargetGenerator
from src.utils.logger import system_logger
from src.config import CONFIG


class EnhancedCoarseOptimizer:
    """
    Enhanced coarse optimization with multi-model approach, advanced feature pruning,
    and wider hyperparameter search.
    """

    def __init__(
        self,
        db_manager: SQLiteManager,
        symbol: str,
        timeframe: str,
        optimal_target_params: dict,
        klines_data: pd.DataFrame,
        agg_trades_data: pd.DataFrame,
        futures_data: pd.DataFrame,
    ):
        """
        Initializes the Enhanced Coarse Optimizer.
        """
        self.db_manager = db_manager
        self.symbol = symbol
        self.timeframe = timeframe
        self.optimal_target_params = optimal_target_params
        self.logger = system_logger.getChild("EnhancedCoarseOptimizer")

        # Store the passed dataframes directly
        self.klines_data = klines_data
        self.agg_trades_data = agg_trades_data
        self.futures_data = futures_data

        self.data_with_targets = None
        self._prepare_data()

    def _prepare_data(self):
        """Prepares the data for optimization."""
        self.logger.info("🔧 Preparing data for enhanced coarse optimization...")
        
        # Initialize feature engineering
        feature_engine = FeatureEngineeringEngine(
            self.klines_data, self.agg_trades_data, self.futures_data
        )
        
        # Generate features
        engineered_features = feature_engine.generate_all_features()
        
        # Initialize target generator
        target_generator = MLTargetGenerator(
            engineered_features, self.optimal_target_params
        )
        
        # Generate targets
        self.data_with_targets = target_generator.generate_targets()
        
        self.logger.info(f"✅ Data prepared: {self.data_with_targets.shape}")

    def enhanced_prune_features(self, top_n_percent: float = 0.5) -> List[str]:
        """
        Enhanced feature pruning with multiple strategies:
        1. Variance-based pruning
        2. Correlation-based pruning  
        3. Mutual information pruning
        4. SHAP-based pruning
        """
        pruning_start_time = time.time()
        
        self.logger.info("🔍 ENHANCED FEATURE PRUNING PROCESS STARTED")
        self.logger.info(f"   Target: Keep top {top_n_percent * 100}% of features")
        self.logger.info(f"   Start time: {time.strftime('%H:%M:%S')}")

        # Step 1: Identify features for pruning
        self.logger.info("📋 Step 1: Identifying features for pruning...")
        features = [
            col for col in self.data_with_targets.columns
            if col not in ["target", "reward", "risk", "target_sr", "Market_Regime_Label"]
        ]
        X = self.data_with_targets[features]
        y = self.data_with_targets["target"]

        self.logger.info(f"   Total features identified: {len(features)}")
        self.logger.info(f"   Data shape: X={X.shape}, y={y.shape}")
        self.logger.info(f"   Target distribution: {y.value_counts().to_dict()}")

        if X.empty:
            self.logger.error("❌ Feature set is empty for pruning.")
            return features

        # Step 2: Variance-based pruning
        self.logger.info("📊 Step 2: Variance-based pruning...")
        variance_start = time.time()
        
        variance_selector = VarianceThreshold(threshold=0.01)  # Remove near-constant features
        X_variance_filtered = variance_selector.fit_transform(X)
        variance_selected_features = X.columns[variance_selector.get_support()].tolist()
        
        variance_duration = time.time() - variance_start
        self.logger.info(f"   ✅ Variance pruning completed in {variance_duration:.2f} seconds")
        self.logger.info(f"   Features after variance pruning: {len(variance_selected_features)} (removed {len(features) - len(variance_selected_features)})")

        # Step 3: Correlation-based pruning
        self.logger.info("🔗 Step 3: Correlation-based pruning...")
        correlation_start = time.time()
        
        X_variance = X[variance_selected_features]
        correlation_matrix = X_variance.corr().abs()
        
        # Remove highly correlated features (correlation > 0.95)
        upper_triangle = correlation_matrix.where(
            np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
        )
        high_corr_features = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.95)]
        correlation_selected_features = [f for f in variance_selected_features if f not in high_corr_features]
        
        correlation_duration = time.time() - correlation_start
        self.logger.info(f"   ✅ Correlation pruning completed in {correlation_duration:.2f} seconds")
        self.logger.info(f"   Features after correlation pruning: {len(correlation_selected_features)} (removed {len(variance_selected_features) - len(correlation_selected_features)})")

        # Step 4: Mutual information pruning
        self.logger.info("📈 Step 4: Mutual information pruning...")
        mi_start = time.time()
        
        X_correlation = X[correlation_selected_features]
        mi_scores = mutual_info_classif(X_correlation, y, random_state=42)
        mi_threshold = np.percentile(mi_scores, 10)  # Keep top 90% by MI
        mi_selected_features = [f for f, score in zip(correlation_selected_features, mi_scores) if score > mi_threshold]
        
        mi_duration = time.time() - mi_start
        self.logger.info(f"   ✅ Mutual information pruning completed in {mi_duration:.2f} seconds")
        self.logger.info(f"   Features after MI pruning: {len(mi_selected_features)} (removed {len(correlation_selected_features) - len(mi_selected_features)})")

        # Step 5: Multi-model SHAP pruning
        self.logger.info("🤖 Step 5: Multi-model SHAP pruning...")
        shap_start = time.time()
        
        # Test multiple models to find the best one
        models_to_test = [
            ("lightgbm", lgb.LGBMClassifier(random_state=42, verbose=-1)),
            ("xgboost", xgb.XGBClassifier(random_state=42, verbosity=0)),
            ("random_forest", RandomForestClassifier(random_state=42, n_jobs=-1)),
            ("catboost", CatBoostClassifier(random_state=42, verbose=False))
        ]
        
        best_model_name = None
        best_model = None
        best_score = 0
        
        X_mi = X[mi_selected_features]
        
        for model_name, model in models_to_test:
            try:
                # Quick cross-validation to find best model
                cv_scores = cross_val_score(model, X_mi, y, cv=3, scoring='accuracy')
                avg_score = cv_scores.mean()
                
                self.logger.info(f"   {model_name}: {avg_score:.4f} accuracy")
                
                if avg_score > best_score:
                    best_score = avg_score
                    best_model_name = model_name
                    best_model = model
                    
            except Exception as e:
                self.logger.warning(f"   {model_name} failed: {e}")
        
        self.logger.info(f"   Best model: {best_model_name} with {best_score:.4f} accuracy")
        
        # Use best model for SHAP analysis
        best_model.fit(X_mi, y)
        
        # SHAP analysis with sampling for performance
        sample_size = min(5000, len(X_mi))
        if len(X_mi) > sample_size:
            sample_indices = np.random.choice(len(X_mi), sample_size, replace=False)
            X_sample = X_mi.iloc[sample_indices]
        else:
            X_sample = X_mi
        
        explainer = shap.TreeExplainer(best_model)
        shap_values = explainer.shap_values(X_sample)
        
        # Process SHAP values
        if isinstance(shap_values, list):
            global_shap_values = np.mean([np.abs(s).mean(axis=0) for s in shap_values], axis=0)
        else:
            global_shap_values = np.abs(shap_values).mean(axis=0)
        
        # Select top features based on SHAP importance
        feature_importance = pd.DataFrame({
            'feature': mi_selected_features,
            'importance': global_shap_values
        }).sort_values('importance', ascending=False)
        
        n_features_to_keep = int(len(mi_selected_features) * top_n_percent)
        final_selected_features = feature_importance.head(n_features_to_keep)['feature'].tolist()
        
        shap_duration = time.time() - shap_start
        self.logger.info(f"   ✅ SHAP pruning completed in {shap_duration:.2f} seconds")
        self.logger.info(f"   Final selected features: {len(final_selected_features)}")
        
        # Log feature importance summary
        self.logger.info("   Top 10 most important features:")
        for i, (_, row) in enumerate(feature_importance.head(10).iterrows()):
            self.logger.info(f"     {i+1:2d}. {row['feature']:<30} {row['importance']:.6f}")
        
        total_duration = time.time() - pruning_start_time
        self.logger.info("🎉 ENHANCED FEATURE PRUNING COMPLETED")
        self.logger.info(f"   Total duration: {total_duration:.2f} seconds")
        self.logger.info(f"   Features kept: {len(final_selected_features)} out of {len(features)} ({len(final_selected_features)/len(features)*100:.1f}%)")
        self.logger.info(f"   Best model: {best_model_name}")
        self.logger.info(f"   Best CV score: {best_score:.4f}")
        
        return final_selected_features

    def find_enhanced_hyperparameter_ranges(
        self, pruned_features: list, n_trials: int = 50
    ) -> dict:
        """
        Enhanced hyperparameter search with wider ranges and multiple models.
        """
        self.logger.info("🔍 Finding enhanced hyperparameter ranges...")
        
        X = self.data_with_targets[pruned_features]
        y = self.data_with_targets["target"]
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        n_classes = len(y.unique())
        self.logger.info(f"   Target has {n_classes} unique classes: {sorted(y.unique())}")
        
        def objective(trial):
            # Test multiple model types
            model_type = trial.suggest_categorical("model_type", ["lightgbm", "xgboost", "random_forest", "catboost"])
            
            if model_type == "lightgbm":
                param = {
                    "objective": "multiclass" if n_classes > 2 else "binary",
                    "metric": "multi_logloss" if n_classes > 2 else "binary_logloss",
                    "verbosity": -1,
                    "boosting_type": "gbdt",
                    "n_estimators": trial.suggest_int("n_estimators", 50, 2000),
                    "learning_rate": trial.suggest_float("learning_rate", 1e-4, 0.3, log=True),
                    "num_leaves": trial.suggest_int("num_leaves", 10, 500),
                    "max_depth": trial.suggest_int("max_depth", 2, 20),
                    "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                    "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                    "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                }
                if n_classes > 2:
                    param["num_class"] = n_classes
                model = lgb.LGBMClassifier(**param)
                
            elif model_type == "xgboost":
                param = {
                    "objective": "multi:softmax" if n_classes > 2 else "binary:logistic",
                    "eval_metric": "mlogloss" if n_classes > 2 else "logloss",
                    "verbosity": 0,
                    "n_estimators": trial.suggest_int("n_estimators", 50, 2000),
                    "learning_rate": trial.suggest_float("learning_rate", 1e-4, 0.3, log=True),
                    "max_depth": trial.suggest_int("max_depth", 2, 20),
                    "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                    "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                    "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                }
                if n_classes > 2:
                    param["num_class"] = n_classes
                model = xgb.XGBClassifier(**param)
                
            elif model_type == "random_forest":
                param = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                    "max_depth": trial.suggest_int("max_depth", 2, 20),
                    "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                    "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                    "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
                    "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
                }
                model = RandomForestClassifier(**param, random_state=42, n_jobs=-1)
                
            else:  # catboost
                param = {
                    "iterations": trial.suggest_int("iterations", 50, 2000),
                    "learning_rate": trial.suggest_float("learning_rate", 1e-4, 0.3, log=True),
                    "depth": trial.suggest_int("depth", 2, 12),
                    "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-8, 10.0, log=True),
                    "border_count": trial.suggest_int("border_count", 32, 255),
                    "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
                }
                if n_classes > 2:
                    param["loss_function"] = "MultiClass"
                else:
                    param["loss_function"] = "Logloss"
                model = CatBoostClassifier(**param, random_state=42, verbose=False)
            
            # Train model with early stopping
            try:
                if hasattr(model, 'fit'):
                    model.fit(X_train, y_train)
                    preds = model.predict(X_val)
                    accuracy = accuracy_score(y_val, preds)
                    return accuracy
                else:
                    return 0.0
            except Exception as e:
                self.logger.warning(f"Model training failed: {e}")
                return 0.0

        # Use enhanced pruner
        study = optuna.create_study(
            direction="maximize", 
            pruner=SuccessiveHalvingPruner(min_resource=1, reduction_factor=3)
        )
        study.optimize(objective, n_trials=n_trials)

        # Analyze top trials to define ranges
        top_trials = sorted(study.trials, key=lambda t: t.value, reverse=True)[
            : max(5, int(n_trials * 0.1))
        ]

        ranges = {}
        for param_name in top_trials[0].params.keys():
            values = [t.params[param_name] for t in top_trials]
            ranges[param_name] = {
                "low": min(values),
                "high": max(values),
                "type": "float" if isinstance(values[0], float) else "int",
            }
            if isinstance(values[0], float):
                ranges[param_name]["step"] = (max(values) - min(values)) / 10.0
            else:
                ranges[param_name]["step"] = 1

        self.logger.info(f"✅ Enhanced hyperparameter search complete. Found ranges: {ranges}")
        return ranges

    def run(self) -> Tuple[List[str], dict]:
        """
        Orchestrates the enhanced coarse optimization process.
        """
        self.logger.info("--- Starting Enhanced Stage 2: Coarse Optimization & Pruning ---")
        
        # Enhanced feature pruning
        pruning_config = CONFIG.get("MODEL_TRAINING", {}).get("feature_pruning", {})
        pruned_features = self.enhanced_prune_features(
            top_n_percent=pruning_config.get("top_n_percent", 0.5)
        )

        # Enhanced hyperparameter optimization
        hpo_config = CONFIG.get("MODEL_TRAINING", {}).get("coarse_hpo", {})
        narrowed_ranges = self.find_enhanced_hyperparameter_ranges(
            pruned_features, n_trials=hpo_config.get("n_trials", 50)
        )
        
        self.logger.info("--- Enhanced Stage 2 Complete ---")
        return pruned_features, narrowed_ranges 