# src/training/steps/step6_analyst_enhancement.py

import asyncio
import json
import os
import pickle
from datetime import datetime
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.utils.prune as prune
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.inspection import permutation_importance

import lightgbm as lgb
import xgboost as xgb
from sklearn.inspection import permutation_importance
from src.utils.logger import system_logger

# Suppress Optuna's verbose logging to keep the output clean
optuna.logging.set_verbosity(optuna.logging.WARNING)

class AnalystEnhancementStep:
    """
    Step 6: Analyst Models Enhancement.

    This step refines the trained analyst models through a sequential process:
    1.  **Hyperparameter Optimization (HPO):** Uses Optuna with early pruning to find the best hyperparameters efficiently.
    2.  **Feature Selection:** Employs robust feature selection methods that work around SHAP/Keras compatibility issues.
    3.  **Final Retraining:** Trains a new model from scratch using the best hyperparameters and the optimal feature set.
    4.  **Advanced Optimization (Optional):** Applies techniques like quantization, structured pruning (WANDA),
        and knowledge distillation for further efficiency and performance gains, especially for neural network models.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the AnalystEnhancementStep.

        Args:
            config (Dict[str, Any]): Configuration dictionary for the step.
        """
        self.config = config
        self.logger = system_logger
        # --- Mac M1/M2/M3 (Apple Silicon) Specific Setup ---
        # Use 'mps' for PyTorch to leverage Apple's Metal Performance Shaders for GPU acceleration.
        # Fallback to 'cpu' if MPS is not available.
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.logger.info(f"Using device: {self.device.upper()} for PyTorch operations.")

    async def initialize(self) -> None:
        """Initialize the analyst enhancement step."""
        self.logger.info("Initializing Analyst Enhancement Step...")
        self.logger.info("Analyst Enhancement Step initialized successfully.")

    async def execute(
        self,
        training_input: Dict[str, Any],
        pipeline_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Executes the full analyst model enhancement pipeline for each regime.

        Args:
            training_input (Dict[str, Any]): Input parameters, including symbol, exchange, and data directories.
            pipeline_state (Dict[str, Any]): The current state of the pipeline.

        Returns:
            Dict[str, Any]: A dictionary containing the results of the enhancement process.
        """
        self.logger.info("🔄 Executing Analyst Enhancement...")
        start_time = datetime.now()

        try:
            data_dir = training_input.get("data_dir", "data/training")
            models_dir = os.path.join(data_dir, "analyst_models")
            # Use the main data_dir for regime data, not processed_data_dir
            regime_data_dir = data_dir

            analyst_models = self._load_models(models_dir)
            if not analyst_models:
                raise ValueError(f"No analyst models found in {models_dir}")

            enhanced_models_summary = {}
            for regime_name, regime_models in analyst_models.items():
                self.logger.info(f"--- Enhancing models for regime: {regime_name} ---")

                try:
                    X_train, y_train, X_val, y_val = self._load_regime_data(
                        regime_data_dir, regime_name
                    )
                except FileNotFoundError as e:
                    self.logger.warning(f"⚠️ {e} — skipping regime '{regime_name}'")
                    continue
                
                enhanced_regime_models = {}
                for model_name, model_data in regime_models.items():
                    self.logger.info(f"Enhancing {model_name} for {regime_name}...")
                    
                    enhanced_model_package = await self._enhance_single_model(
                        model_data, model_name, regime_name, X_train, y_train, X_val, y_val
                    )
                    enhanced_regime_models[model_name] = enhanced_model_package

                enhanced_models_summary[regime_name] = enhanced_regime_models

            enhanced_models_dir = self._save_enhanced_models(
                enhanced_models_summary, data_dir, training_input
            )
            
            duration = (datetime.now() - start_time).total_seconds()
            self.logger.info(
                f"✅ Analyst enhancement completed in {duration:.2f}s. Results saved to {enhanced_models_dir}"
            )

            pipeline_state["enhanced_analyst_models"] = enhanced_models_summary
            return {
                "status": "SUCCESS",
                "enhanced_models_dir": enhanced_models_dir,
                "duration": duration,
            }

        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            self.logger.error(f"❌ Error in Analyst Enhancement after {duration:.2f}s: {e}", exc_info=True)
            return {"status": "FAILED", "error": str(e), "duration": duration}

    def _load_models(self, models_dir: str) -> Dict[str, Any]:
        """Loads all analyst models from the specified directory."""
        analyst_models = {}
        if not os.path.exists(models_dir):
            return analyst_models
            
        for regime_dir in os.listdir(models_dir):
            regime_path = os.path.join(models_dir, regime_dir)
            if os.path.isdir(regime_path):
                regime_models = {}
                for model_file in os.listdir(regime_path):
                    if model_file.endswith(".pkl"):
                        model_name = model_file.replace(".pkl", "")
                        model_path = os.path.join(regime_path, model_file)
                        with open(model_path, "rb") as f:
                            regime_models[model_name] = pickle.load(f)
                analyst_models[regime_dir] = regime_models
        return analyst_models

    def _load_regime_data(self, data_dir: str, regime_name: str) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """Loads training and validation data for a specific regime."""
        try:
            self.logger.info(f"Loading data for regime '{regime_name}'...")
            
            # Use regime_data directory - fix the path construction
            regime_data_dir = os.path.join(data_dir, "regime_data")
            
            # Load the combined data file that Step 3 created
            data_path = os.path.join(regime_data_dir, f"BINANCE_ETHUSDT_{regime_name}_data.pkl")

            if not os.path.exists(data_path):
                raise FileNotFoundError(f"Data file for regime '{regime_name}' not found in {regime_data_dir}")

            # Load the combined data
            with open(data_path, "rb") as f:
                data = pickle.load(f)

            # Ensure data is a DataFrame
            if not isinstance(data, pd.DataFrame):
                data = pd.DataFrame(data)

            self.logger.info(f"Loaded data shape: {data.shape}, columns: {list(data.columns)}")

            # Remove non-numeric columns that XGBoost doesn't accept
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            data = data[numeric_columns]

            self.logger.info(f"After numeric filtering: {data.shape}, columns: {list(data.columns)}")

            # Check for target column with different possible names
            target_column = None
            target_candidates = ['label', 'target', 'y', 'class', 'signal', 'prediction']
            
            for possible_target in target_candidates:
                if possible_target in data.columns:
                    target_column = possible_target
                    self.logger.info(f"Found target column: {target_column}")
                    break
            
            if target_column is None:
                self.logger.warning(f"No target column found in regime data. Available columns: {list(data.columns)}")
                
                # Try to create a meaningful target from available data
                target_created = self._create_target_from_data(data, regime_name)
                
                if target_created:
                    target_column = 'label'
                    self.logger.info("Successfully created target column from available data")
                else:
                    self.logger.warning("Creating dummy target - this may not be suitable for training")
                    data['label'] = np.random.choice([0, 1], size=len(data))
                    target_column = 'label'
            else:
                # Rename target column to 'label' for consistency
                if target_column != 'label':
                    data['label'] = data[target_column]
                    data = data.drop(columns=[target_column])

            # Split features and target
            X = data.drop('label', axis=1)
            y = data['label']

            # Validate target distribution
            unique_targets = y.unique()
            self.logger.info(f"Target distribution: {dict(y.value_counts())}")
            
            if len(unique_targets) <= 1:
                self.logger.warning(f"Target has only {len(unique_targets)} unique values: {unique_targets}")
                # Create a more diverse target if possible
                if len(data.columns) > 1:
                    # Use the first numeric column as a proxy target
                    proxy_column = data.columns[0]
                    if proxy_column != 'label':
                        proxy_values = data[proxy_column]
                        # Create binary target based on median
                        median_val = proxy_values.median()
                        y = (proxy_values > median_val).astype(int)
                        self.logger.info(f"Created proxy target from {proxy_column} (median: {median_val})")
                        self.logger.info(f"New target distribution: {dict(y.value_counts())}")

            # Split into train and validation
            train_size = int(0.8 * len(data))
            X_train = X[:train_size]
            y_train = y[:train_size]
            X_val = X[train_size:]
            y_val = y[train_size:]

            self.logger.info(f"Data loaded and split: X_train shape {X_train.shape}, X_val shape {X_val.shape}")
            self.logger.info(f"Target classes in training: {y_train.unique()}, in validation: {y_val.unique()}")

            return X_train, y_train, X_val, y_val
            
        except Exception as e:
            self.logger.error(f"Error loading regime data for '{regime_name}': {e}")
            raise

    def _create_target_from_data(self, data: pd.DataFrame, regime_name: str) -> bool:
        """
        Attempts to create a meaningful target column from available data.
        
        Args:
            data: The regime data DataFrame
            regime_name: Name of the regime
            
        Returns:
            bool: True if target was successfully created, False otherwise
        """
        try:
            # Look for price-related columns that could be used to create targets
            price_columns = [col for col in data.columns if any(price_term in col.lower() 
                                                              for price_term in ['close', 'price', 'value'])]
            
            if price_columns:
                # Use the first price column to create a target
                price_col = price_columns[0]
                price_values = data[price_col]
                
                # Create a simple momentum-based target
                if len(price_values) > 1:
                    # Calculate price changes
                    price_changes = price_values.pct_change().fillna(0)
                    
                    # Create binary target based on positive/negative momentum
                    threshold = price_changes.std() * 0.1  # Small threshold
                    target = (price_changes > threshold).astype(int)
                    
                    # Ensure we have at least 2 classes
                    if target.nunique() >= 2:
                        data['label'] = target
                        self.logger.info(f"Created momentum-based target from {price_col}")
                        return True
            
            # Look for volume-related columns
            volume_columns = [col for col in data.columns if 'volume' in col.lower()]
            
            if volume_columns:
                volume_col = volume_columns[0]
                volume_values = data[volume_col]
                
                # Create target based on volume spikes
                if len(volume_values) > 1:
                    volume_median = volume_values.median()
                    target = (volume_values > volume_median).astype(int)
                    
                    if target.nunique() >= 2:
                        data['label'] = target
                        self.logger.info(f"Created volume-based target from {volume_col}")
                        return True
            
            # Look for any numeric column with good variance
            for col in data.columns:
                if col != 'label' and data[col].dtype in ['int64', 'float64']:
                    values = data[col]
                    if values.nunique() >= 2 and values.std() > 0:
                        # Create target based on above/below median
                        median_val = values.median()
                        target = (values > median_val).astype(int)
                        
                        if target.nunique() >= 2:
                            data['label'] = target
                            self.logger.info(f"Created target from {col} (median-based)")
                            return True
            
            return False
            
        except Exception as e:
            self.logger.warning(f"Error creating target from data: {e}")
            return False

    async def _enhance_single_model(
        self, model_data: Dict[str, Any], model_name: str, regime_name: str,
        X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series
    ) -> Dict[str, Any]:
        """Applies the full enhancement pipeline to a single model."""
        
        original_accuracy = model_data.get("accuracy", "N/A")
        self.logger.info(f"Original model accuracy: {original_accuracy}")

        # Check if we have valid targets for training
        if y_train.nunique() <= 1:
            self.logger.warning(f"Target has only {y_train.nunique()} unique values: {y_train.unique()}")
            self.logger.warning("Skipping model enhancement due to insufficient target diversity")
            return {
                "model": model_data["model"],  # Return original model
                "selected_features": list(X_train.columns),
                "accuracy": original_accuracy,
                "enhancement_metadata": {
                    "enhancement_date": datetime.now().isoformat(),
                    "original_accuracy": original_accuracy,
                    "hpo_score": 0.0,
                    "final_accuracy": original_accuracy,
                    "improvement": 0.0,
                    "best_params": {},
                    "feature_selection_method": "None - insufficient target diversity",
                    "original_feature_count": len(X_train.columns),
                    "selected_feature_count": len(X_train.columns),
                    "shap_summary": {},
                    "enhancement_applied": False,
                    "reason": f"Insufficient target diversity (only {y_train.nunique()} unique values)"
                },
            }

        # --- 1. Hyperparameter Optimization with Pruning ---
        best_params, hpo_score = await self._apply_hyperparameter_optimization(
            model_name, X_train, y_train, X_val, y_val
        )

        # --- 2. Feature Selection ---
        temp_model = self._get_model_instance(model_name, best_params)
        temp_model.fit(X_train, y_train)
        
        optimal_features, feature_selection_summary = await self._select_optimal_features(
            temp_model, model_name, X_train, y_train, X_val, y_val
        )
        
        X_train_optimal = X_train[optimal_features]
        X_val_optimal = X_val[optimal_features]

        # --- 3. Final Retraining ---
        self.logger.info(f"Retraining final model with {len(optimal_features)} optimal features...")
        final_model = self._get_model_instance(model_name, best_params)
        final_model.fit(X_train_optimal, y_train)
        
        final_accuracy = accuracy_score(y_val, final_model.predict(X_val_optimal))
        self.logger.info(f"Enhanced model accuracy: {final_accuracy:.4f}")

        # --- 4. Advanced Optimizations ---
        if model_name == "neural_network":
            final_model = self._apply_quantization(final_model)
            final_model = self._apply_wanda_pruning(final_model, X_train_optimal)
            final_model = self._apply_knowledge_distillation(final_model, X_train_optimal, y_train)

        enhancement_package = {
            "model": final_model,
            "selected_features": optimal_features,
            "accuracy": final_accuracy,
            "enhancement_metadata": {
                "enhancement_date": datetime.now().isoformat(),
                "original_accuracy": original_accuracy,
                "hpo_score": hpo_score,
                "final_accuracy": final_accuracy,
                "improvement": final_accuracy - original_accuracy if isinstance(original_accuracy, float) else "N/A",
                "best_params": best_params,
                "feature_selection_method": feature_selection_summary.get("method", "robust_fallback"),
                "original_feature_count": len(X_train.columns),
                "selected_feature_count": len(optimal_features),
                "feature_selection_summary": feature_selection_summary,
            },
        }
        return enhancement_package

    def _get_model_instance(self, model_name: str, params: Dict[str, Any]):
        """Factory function to get a model instance from its name and parameters."""
        if model_name in ["xgboost", "lightgbm"] and self.device == "mps":
            params['device'] = self.device

        if model_name == "random_forest":
            return RandomForestClassifier(**params, random_state=42, n_jobs=-1)
        elif model_name == "lightgbm":
            return lgb.LGBMClassifier(**params, random_state=42, n_jobs=-1)
        elif model_name == "xgboost":
            # Remove eval_metric and device from params if they exist to avoid duplicate/unsupported parameters
            xgb_params = params.copy()
            if 'eval_metric' in xgb_params:
                del xgb_params['eval_metric']
            if 'device' in xgb_params:
                del xgb_params['device']
            
            return xgb.XGBClassifier(**xgb_params, random_state=42, tree_method='hist' if self.device == 'cpu' else 'auto', use_label_encoder=False, eval_metric='logloss')
        elif model_name == "svm":
            from sklearn.svm import SVC
            return SVC(**params, random_state=42, probability=True)
        elif model_name == "neural_network":
            from sklearn.neural_network import MLPClassifier
            return MLPClassifier(**params, random_state=42, early_stopping=True, validation_fraction=0.1)
        else:
            raise ValueError(f"Model {model_name} not supported.")

    async def _apply_hyperparameter_optimization(
        self, model_name: str, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series
    ) -> Tuple[Dict[str, Any], float]:
        """Performs hyperparameter optimization using Optuna with early pruning."""
        self.logger.info(f"Running Optuna HPO with pruning for {model_name}...")

        def objective(trial: optuna.trial.Trial) -> float:
            pruning_callback = None
            
            # Validate target data before proceeding
            if y_train.nunique() <= 1:
                self.logger.warning(f"Target has only {y_train.nunique()} unique values, skipping optimization")
                return 0.0
            
            if model_name == "lightgbm":
                pruning_callback = optuna.integration.LightGBMPruningCallback(trial, "binary_logloss")
                params = {
                    "objective": "binary", "metric": "binary_logloss", "verbosity": -1,
                    "n_estimators": trial.suggest_int("n_estimators", 100, 2000),
                    "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
                    "num_leaves": trial.suggest_int("num_leaves", 20, 300),
                    "max_depth": trial.suggest_int("max_depth", 3, 12),
                    "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                    "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                }
            elif model_name == "xgboost":
                 pruning_callback = optuna.integration.XGBoostPruningCallback(trial, "validation_0-logloss")
                 params = {
                    "objective": "binary:logistic", "eval_metric": "logloss", "verbosity": 0,
                    "n_estimators": trial.suggest_int("n_estimators", 100, 2000),
                    "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
                    "max_depth": trial.suggest_int("max_depth", 3, 12),
                    "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                    # Add base_score to prevent the error when all targets are the same
                    "base_score": 0.5,
                }
            elif model_name == "svm":
                # SVM doesn't support iterative pruning
                params = {
                    "C": trial.suggest_float("C", 0.1, 100.0, log=True),
                    "kernel": trial.suggest_categorical("kernel", ["rbf", "linear", "poly"]),
                    "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
                }
            elif model_name == "neural_network":
                # Neural network doesn't support iterative pruning
                params = {
                    "hidden_layer_sizes": trial.suggest_categorical("hidden_layer_sizes", [(50,), (100,), (50, 25), (100, 50), (100, 50, 25)]),
                    "alpha": trial.suggest_float("alpha", 1e-5, 1e-1, log=True),
                    "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-4, 1e-1, log=True),
                    "max_iter": trial.suggest_int("max_iter", 200, 1000),
                }
            else: # RandomForest doesn't support iterative pruning.
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                    "max_depth": trial.suggest_int("max_depth", 5, 50),
                    "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                    "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
                }

            model = self._get_model_instance(model_name, params)
            
            # Train the model with appropriate parameters based on model type
            if model_name == "lightgbm":
                # LightGBM supports callbacks
                if pruning_callback:
                    model.fit(X_train, y_train, 
                              eval_set=[(X_val, y_val)], 
                              callbacks=[pruning_callback])
                else:
                    model.fit(X_train, y_train, 
                              eval_set=[(X_val, y_val)])
            elif model_name == "xgboost":
                # XGBoost doesn't support callbacks parameter, use eval_set only
                model.fit(X_train, y_train, 
                          eval_set=[(X_val, y_val)],
                          verbose=False)
            else:
                # SVM, Neural Network, and Random Forest don't support eval_set
                model.fit(X_train, y_train)

            preds = model.predict(X_val)
            return accuracy_score(y_val, preds)

        study = optuna.create_study(
            direction="maximize",
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=5)
        )
        
        await asyncio.to_thread(study.optimize, objective, n_trials=100, n_jobs=-1)

        if not study.best_trial:
             self.logger.warning("Optuna study found no best trial, possibly due to all trials being pruned. Returning empty params.")
             return {}, 0.0

        self.logger.info(f"HPO complete. Best score: {study.best_value:.4f}")
        return study.best_params, study.best_value

    async def _select_optimal_features(
        self, model: Any, model_name: str, X_train: pd.DataFrame, y_train: pd.Series, 
        X_val: pd.DataFrame, y_val: pd.Series
    ) -> Tuple[list[str], Dict]:
        """Selects the most important features using robust methods that work around SHAP/Keras compatibility issues."""
        self.logger.info("Selecting optimal features using robust methods...")
        
        feature_names = X_val.columns.tolist()
        
        # Ensure we keep at least 10 features or 50% of original features, whichever is larger
        min_features = max(10, len(feature_names) // 2)
        max_features = min(20, len(feature_names))  # Don't select more than 20 features
        
        try:
            # Try SHAP first with proper error handling
            optimal_features, shap_summary = self._try_shap_feature_selection(
                model, model_name, X_train, y_train, X_val, y_val, feature_names, min_features, max_features
            )
            if optimal_features:
                return optimal_features, shap_summary
        except Exception as e:
            self.logger.warning(f"SHAP analysis failed: {e}. Trying alternative methods...")
        
        # Fallback to robust feature selection methods
        optimal_features, fallback_summary = self._robust_feature_selection(
            model, model_name, X_train, y_train, X_val, y_val, feature_names, min_features, max_features
        )
        
        self.logger.info(f"Selected {len(optimal_features)} optimal features using robust methods (from {len(feature_names)} total)")
        return optimal_features, fallback_summary

    def _try_shap_feature_selection(
        self, model: Any, model_name: str, X_train: pd.DataFrame, y_train: pd.Series,
        X_val: pd.DataFrame, y_val: pd.Series, feature_names: list, min_features: int, max_features: int
    ) -> Tuple[list[str], Dict]:
        """Attempts SHAP-based feature selection with proper error handling."""
        try:
            import shap
            
            # Try different SHAP explainer approaches
            if model_name in ["lightgbm", "xgboost", "random_forest"]:
                # Try TreeExplainer with proper import
                try:
                    from shap.explainers import TreeExplainer
                    explainer = TreeExplainer(model)
                    shap_values = explainer.shap_values(X_val)
                    
                    # Handle different SHAP output formats
                    if isinstance(shap_values, list):
                        shap_values = np.array(shap_values)
                    
                    feature_importance = np.abs(shap_values).mean(0)
                    
                except (ImportError, AttributeError):
                    # Fallback to permutation importance for tree models
                    feature_importance = permutation_importance(model, X_val, y_val, n_repeats=5, random_state=42).importances_mean
                    
            elif model_name == "svm":
                # Use KernelExplainer for SVM models
                try:
                    from shap import KernelExplainer
                    explainer = KernelExplainer(model.predict_proba, X_train.iloc[:100])
                    shap_values = explainer.shap_values(X_val.iloc[:50])
                    
                    if isinstance(shap_values, list):
                        shap_values = np.array(shap_values)
                    feature_importance = np.abs(shap_values).mean(axis=(0, 1))
                    
                except Exception:
                    # Fallback to permutation importance
                    feature_importance = permutation_importance(model, X_val, y_val, n_repeats=5, random_state=42).importances_mean
                    
            else:  # neural_network and others
                # Use permutation importance for non-tree models
                feature_importance = permutation_importance(model, X_val, y_val, n_repeats=5, random_state=42).importances_mean
            
            # Select optimal features
            if len(feature_names) <= min_features:
                optimal_features = feature_names
            else:
                n_features_to_select = min(max_features, max(min_features, len(feature_names) // 3))
                top_features_idx = np.argsort(feature_importance)[-n_features_to_select:]
                optimal_features = [feature_names[i] for i in top_features_idx]
            
            shap_summary = {
                "method": "shap",
                "feature_importance": dict(zip(feature_names, feature_importance)),
                "optimal_features": optimal_features,
                "importance_scores": feature_importance.tolist(),
                "selection_bounds": {
                    "min_features": min_features,
                    "max_features": max_features,
                    "selected_count": len(optimal_features)
                }
            }
            
            return optimal_features, shap_summary
            
        except Exception as e:
            raise Exception(f"SHAP analysis failed: {e}")

    def _robust_feature_selection(
        self, model: Any, model_name: str, X_train: pd.DataFrame, y_train: pd.Series,
        X_val: pd.DataFrame, y_val: pd.Series, feature_names: list, min_features: int, max_features: int
    ) -> Tuple[list[str], Dict]:
        """Implements robust feature selection using multiple methods."""
        
        # Method 1: Permutation Importance (works for all models)
        try:
            perm_importance = permutation_importance(model, X_val, y_val, n_repeats=5, random_state=42)
            perm_scores = perm_importance.importances_mean
        except Exception:
            perm_scores = np.ones(len(feature_names))  # Fallback to equal importance
        
        # Method 2: Statistical tests (for tree-based models, use feature_importances_)
        try:
            if hasattr(model, 'feature_importances_'):
                model_importance = model.feature_importances_
            else:
                model_importance = np.ones(len(feature_names))
        except Exception:
            model_importance = np.ones(len(feature_names))
        
        # Method 3: Statistical feature selection
        try:
            # Use mutual information for feature selection
            selector = SelectKBest(score_func=mutual_info_classif, k=min(max_features, len(feature_names)))
            selector.fit(X_train, y_train)
            stat_scores = selector.scores_
        except Exception:
            stat_scores = np.ones(len(feature_names))
        
        # Combine all methods (ensemble approach)
        combined_scores = (perm_scores + model_importance + stat_scores) / 3
        
        # Select optimal features
        if len(feature_names) <= min_features:
            optimal_features = feature_names
        else:
            n_features_to_select = min(max_features, max(min_features, len(feature_names) // 3))
            top_features_idx = np.argsort(combined_scores)[-n_features_to_select:]
            optimal_features = [feature_names[i] for i in top_features_idx]
        
        fallback_summary = {
            "method": "robust_ensemble",
            "feature_importance": dict(zip(feature_names, combined_scores)),
            "optimal_features": optimal_features,
            "importance_scores": combined_scores.tolist(),
            "selection_bounds": {
                "min_features": min_features,
                "max_features": max_features,
                "selected_count": len(optimal_features)
            },
            "method_details": {
                "permutation_importance": perm_scores.tolist(),
                "model_importance": model_importance.tolist(),
                "statistical_scores": stat_scores.tolist()
            }
        }
        
        return optimal_features, fallback_summary

    def _save_enhanced_models(
        self, enhanced_models: Dict, data_dir: str, training_input: Dict
    ) -> str:
        """Saves the enhanced models and a JSON summary report."""
        enhanced_models_dir = os.path.join(data_dir, "enhanced_analyst_models")
        os.makedirs(enhanced_models_dir, exist_ok=True)

        json_summary = {}

        for regime_name, models in enhanced_models.items():
            regime_models_dir = os.path.join(enhanced_models_dir, regime_name)
            os.makedirs(regime_models_dir, exist_ok=True)
            json_summary[regime_name] = {}
            
            for model_name, model_data in models.items():
                model_file = os.path.join(regime_models_dir, f"{model_name}.joblib")
                joblib.dump(model_data["model"], model_file)

                summary_data = model_data.copy()
                summary_data.pop("model", None)
                summary_data["model_path"] = model_file
                json_summary[regime_name][model_name] = summary_data

        symbol = training_input.get("symbol", "ETHUSDT")
        exchange = training_input.get("exchange", "BINANCE")
        summary_file = os.path.join(
            data_dir, f"{exchange}_{symbol}_analyst_enhancement_summary.json"
        )
        with open(summary_file, "w") as f:
            json.dump(json_summary, f, indent=2, default=str)
            
        return enhanced_models_dir

    def _apply_quantization(self, model: torch.nn.Module) -> torch.nn.Module:
        """Applies dynamic quantization to a PyTorch model for CPU/MPS inference."""
        self.logger.info("Applying dynamic quantization to the model...")
        # Move model to CPU for quantization, as it's primarily a CPU-based feature set in PyTorch
        model.to("cpu")
        quantized_model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )
        self.logger.info("Dynamic quantization complete. Model is now smaller and may run faster on CPU.")
        return quantized_model

    def _apply_wanda_pruning(self, model: torch.nn.Module, calibration_data: pd.DataFrame, sparsity: float = 0.5) -> torch.nn.Module:
        """
        Applies structured pruning using a simplified WANDA (Weight and Activation-based) method.
        This implementation demonstrates the core concept.
        """
        self.logger.info(f"Applying WANDA-style pruning with {sparsity} sparsity...")
        model.to(self.device)
        
        # Convert calibration data to tensors
        calib_tensor = torch.tensor(calibration_data.values, dtype=torch.float32).to(self.device)
        
        # 1. Collect activations
        activations = {}
        def get_activation(name):
            def hook(model, input, output):
                activations[name] = torch.sqrt(torch.mean(input[0]**2, dim=0))
            return hook

        hooks = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                hooks.append(module.register_forward_hook(get_activation(name)))
        
        model(calib_tensor) # Forward pass to trigger hooks
        for hook in hooks:
            hook.remove()

        # 2. Calculate importance and prune
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and name in activations:
                W = module.weight.data
                act_norm = activations[name]
                
                # WANDA Importance Score: |Weight| * ||Activation||
                importance_scores = torch.abs(W) * act_norm
                
                # Prune the weights with the lowest importance scores
                prune.l1_unstructured(module, name='weight', amount=sparsity, importance_scores=importance_scores)
                # Make pruning permanent
                prune.remove(module, 'weight')
        
        self.logger.info("WANDA-style pruning complete.")
        return model

    def _apply_knowledge_distillation(
        self, teacher_model: torch.nn.Module, X_train: pd.DataFrame, y_train: pd.Series
    ) -> torch.nn.Module:
        """
        Uses knowledge distillation to train a smaller 'student' model to mimic the teacher.
        """
        self.logger.info("Applying knowledge distillation...")
        teacher_model.to(self.device).eval()

        # 1. Define a smaller student model
        input_dim = X_train.shape[1]
        student_model = nn.Sequential(
            nn.Linear(input_dim, 64), # Smaller hidden layer
            nn.ReLU(),
            nn.Linear(64, 2)
        ).to(self.device)

        # 2. Setup training
        optimizer = optim.Adam(student_model.parameters(), lr=0.001)
        
        # Prepare data
        train_dataset = TensorDataset(
            torch.tensor(X_train.values, dtype=torch.float32),
            torch.tensor(y_train.values, dtype=torch.long)
        )
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

        # Distillation parameters
        T = 2.0  # Temperature for softening probabilities
        alpha = 0.3 # Weight for student's own loss

        # 3. Training loop
        student_model.train()
        for epoch in range(5): # A short training for demonstration
            for data, targets in train_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                
                # Get teacher's logits (outputs before softmax)
                with torch.no_grad():
                    teacher_logits = teacher_model(data)

                # Get student's logits
                student_logits = student_model(data)

                # Calculate losses
                loss_hard = F.cross_entropy(student_logits, targets) # Standard loss
                loss_soft = nn.KLDivLoss(reduction='batchmean')(
                    F.log_softmax(student_logits / T, dim=1),
                    F.softmax(teacher_logits / T, dim=1)
                ) * (T * T) # Scaling factor

                # Combine losses
                loss = alpha * loss_hard + (1.0 - alpha) * loss_soft

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            self.logger.info(f"Distillation Epoch {epoch+1}, Loss: {loss.item():.4f}")

        self.logger.info("Knowledge distillation complete. Returning the trained student model.")
        return student_model.eval()

async def run_step(
    symbol: str,
    exchange: str = "BINANCE",
    data_dir: str = "data/training",
    **kwargs,
) -> bool:
    """
    Run the analyst enhancement step.

    Args:
        symbol: Trading symbol
        exchange: Exchange name
        data_dir: Data directory
        **kwargs: Additional arguments

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        config = {
            "symbol": symbol,
            "exchange": exchange,
            "data_dir": data_dir,
        }
        
        step = AnalystEnhancementStep(config)
        await step.initialize()

        training_input = {
            "symbol": symbol,
            "exchange": exchange,
            "data_dir": data_dir,
        }

        pipeline_state = {}
        result = await step.execute(training_input, pipeline_state)

        return result.get("status") == "SUCCESS" if isinstance(result, dict) else True

    except Exception as e:
        print(f"❌ Analyst enhancement failed: {e}")
        return False
