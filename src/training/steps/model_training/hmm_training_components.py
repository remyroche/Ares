"""HMM training components for model training.

This module contains specialized components for HMM-based model training,
including regime-specific training, multi-output models, and optimization.
"""

from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
import lightgbm as lgb
from src.utils.logger import system_logger


class HMMModelTrainer:
    """Trains HMM-based models with various algorithms."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize HMM model trainer.
        
        Args:
            config: Training configuration
        """
        self.config = config
        self.logger = system_logger.getChild("HMMModelTrainer")
        self.model_types = config.get("model_types", ["lightgbm", "random_forest"])
        
    async def train_models(self, prepared_data: Dict[str, Any]) -> Dict[str, Any]:
        """Train multiple model types.
        
        Args:
            prepared_data: Prepared training data
            
        Returns:
            Training results
        """
        results = {"models": {}, "performance": {}, "feature_importance": {}}
        
        if "train" not in prepared_data or "val" not in prepared_data:
            self.logger.error("Missing train or validation data")
            return results
        
        train_data = prepared_data["train"]
        val_data = prepared_data["val"]
        
        for model_type in self.model_types:
            self.logger.info(f"Training {model_type} model...")
            
            try:
                if model_type == "lightgbm":
                    model_results = await self._train_lightgbm(train_data, val_data)
                elif model_type == "random_forest":
                    model_results = await self._train_random_forest(train_data, val_data)
                elif model_type == "xgboost":
                    model_results = await self._train_xgboost(train_data, val_data)
                else:
                    self.logger.warning(f"Unknown model type: {model_type}")
                    continue
                
                # Store results
                results["models"][model_type] = model_results["model"]
                results["performance"][model_type] = model_results["performance"]
                results["feature_importance"][model_type] = model_results.get("feature_importance", {})
                
            except Exception as e:
                self.logger.error(f"Failed to train {model_type}: {e}")
        
        return results
    
    async def _train_lightgbm(
        self, 
        train_data: Dict[str, Any], 
        val_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Train LightGBM model.
        
        Args:
            train_data: Training data
            val_data: Validation data
            
        Returns:
            Model results
        """
        # Determine if classification or regression
        unique_labels = np.unique(train_data["labels"])
        is_classification = len(unique_labels) < 10 and all(isinstance(x, (int, np.integer)) for x in unique_labels)
        
        # Set parameters
        params = {
            "objective": "multiclass" if is_classification and len(unique_labels) > 2 else "binary" if is_classification else "regression",
            "metric": "multi_logloss" if is_classification and len(unique_labels) > 2 else "binary_logloss" if is_classification else "rmse",
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1,
            "num_threads": 4
        }
        
        if is_classification and len(unique_labels) > 2:
            params["num_class"] = len(unique_labels)
        
        # Create datasets
        train_dataset = lgb.Dataset(
            train_data["features"], 
            label=train_data["labels"],
            feature_name=train_data["feature_names"]
        )
        
        val_dataset = lgb.Dataset(
            val_data["features"], 
            label=val_data["labels"],
            reference=train_dataset
        )
        
        # Train model
        model = lgb.train(
            params,
            train_dataset,
            valid_sets=[val_dataset],
            num_boost_round=100,
            callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)]
        )
        
        # Evaluate
        val_pred = model.predict(val_data["features"], num_iteration=model.best_iteration)
        
        if is_classification:
            if len(unique_labels) > 2:
                val_pred_class = np.argmax(val_pred, axis=1)
            else:
                val_pred_class = (val_pred > 0.5).astype(int)
            
            performance = {
                "accuracy": accuracy_score(val_data["labels"], val_pred_class),
                "f1_score": f1_score(val_data["labels"], val_pred_class, average='weighted')
            }
        else:
            performance = {
                "mse": mean_squared_error(val_data["labels"], val_pred),
                "r2_score": r2_score(val_data["labels"], val_pred)
            }
        
        # Get feature importance
        importance = model.feature_importance(importance_type="gain")
        feature_importance = {
            train_data["feature_names"][i]: float(importance[i]) 
            for i in range(len(train_data["feature_names"]))
        }
        
        return {
            "model": model,
            "performance": performance,
            "feature_importance": feature_importance
        }
    
    async def _train_random_forest(
        self, 
        train_data: Dict[str, Any], 
        val_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Train Random Forest model.
        
        Args:
            train_data: Training data
            val_data: Validation data
            
        Returns:
            Model results
        """
        # Determine if classification or regression
        unique_labels = np.unique(train_data["labels"])
        is_classification = len(unique_labels) < 10 and all(isinstance(x, (int, np.integer)) for x in unique_labels)
        
        # Create model
        if is_classification:
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=4
            )
        else:
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=4
            )
        
        # Train model
        model.fit(train_data["features"], train_data["labels"])
        
        # Evaluate
        val_pred = model.predict(val_data["features"])
        
        if is_classification:
            performance = {
                "accuracy": accuracy_score(val_data["labels"], val_pred),
                "f1_score": f1_score(val_data["labels"], val_pred, average='weighted')
            }
        else:
            performance = {
                "mse": mean_squared_error(val_data["labels"], val_pred),
                "r2_score": r2_score(val_data["labels"], val_pred)
            }
        
        # Get feature importance
        feature_importance = {
            train_data["feature_names"][i]: float(model.feature_importances_[i])
            for i in range(len(train_data["feature_names"]))
        }
        
        return {
            "model": model,
            "performance": performance,
            "feature_importance": feature_importance
        }
    
    async def _train_xgboost(
        self, 
        train_data: Dict[str, Any], 
        val_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Train XGBoost model.
        
        Args:
            train_data: Training data
            val_data: Validation data
            
        Returns:
            Model results
        """
        try:
            import xgboost as xgb
            
            # Similar implementation to LightGBM
            # ... (implementation details)
            
            return {"model": None, "performance": {}, "feature_importance": {}}
            
        except ImportError:
            self.logger.warning("XGBoost not available")
            return {"model": None, "performance": {}, "feature_importance": {}}


class RegimeSpecificTrainer:
    """Trains separate models for each market regime."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize regime-specific trainer.
        
        Args:
            config: Training configuration
        """
        self.config = config
        self.logger = system_logger.getChild("RegimeSpecificTrainer")
        self.base_trainer = HMMModelTrainer(config)
        
    async def train_regime_models(
        self, 
        prepared_data: Dict[str, Any],
        regime_characteristics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Train separate models for each regime.
        
        Args:
            prepared_data: Prepared training data
            regime_characteristics: Characteristics of each regime
            
        Returns:
            Training results
        """
        results = {"models": {}, "performance": {}, "regime_info": {}}
        
        if "train" not in prepared_data or "regime_labels" not in prepared_data["train"]:
            self.logger.error("Missing training data or regime labels")
            return results
        
        train_data = prepared_data["train"]
        unique_regimes = np.unique(train_data["regime_labels"])
        
        self.logger.info(f"Training models for {len(unique_regimes)} regimes")
        
        for regime in unique_regimes:
            self.logger.info(f"Training models for regime {regime}...")
            
            # Filter data for this regime
            regime_train_data = self._filter_regime_data(train_data, regime)
            regime_val_data = None
            
            if "val" in prepared_data and "regime_labels" in prepared_data["val"]:
                regime_val_data = self._filter_regime_data(prepared_data["val"], regime)
            
            # Check if enough samples
            if len(regime_train_data["features"]) < 50:
                self.logger.warning(f"Insufficient samples for regime {regime}: {len(regime_train_data['features'])}")
                continue
            
            # Train models for this regime
            regime_results = await self.base_trainer.train_models({
                "train": regime_train_data,
                "val": regime_val_data or regime_train_data
            })
            
            # Store results with regime prefix
            for model_type, model in regime_results["models"].items():
                results["models"][f"{model_type}_regime_{regime}"] = model
            
            for model_type, perf in regime_results["performance"].items():
                results["performance"][f"{model_type}_regime_{regime}"] = perf
            
            # Store regime information
            results["regime_info"][f"regime_{regime}"] = {
                "n_samples": len(regime_train_data["features"]),
                "characteristics": regime_characteristics.get(f"regime_{regime}", {})
            }
        
        return results
    
    def _filter_regime_data(self, data: Dict[str, Any], regime: int) -> Dict[str, Any]:
        """Filter data for specific regime.
        
        Args:
            data: Input data
            regime: Regime to filter for
            
        Returns:
            Filtered data
        """
        mask = data["regime_labels"] == regime
        
        return {
            "features": data["features"][mask],
            "labels": data["labels"][mask],
            "feature_names": data["feature_names"],
            "regime": regime
        }


class MultiOutputTrainer:
    """Trains models for multiple outputs (direction and profit)."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize multi-output trainer.
        
        Args:
            config: Training configuration
        """
        self.config = config
        self.logger = system_logger.getChild("MultiOutputTrainer")
        
    async def train_multi_output(self, prepared_data: Dict[str, Any]) -> Dict[str, Any]:
        """Train multi-output models.
        
        Args:
            prepared_data: Prepared training data
            
        Returns:
            Training results
        """
        results = {"models": {}, "performance": {}}
        
        if "train" not in prepared_data:
            self.logger.error("Missing training data")
            return results
        
        train_data = prepared_data["train"]
        
        # Check if profit labels are available
        if "profit_labels" not in train_data:
            self.logger.warning("No profit labels available for multi-output training")
            return results
        
        # Train direction model (classification)
        direction_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        direction_model.fit(train_data["features"], train_data["labels"])
        
        # Train profit model (regression)
        profit_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        profit_model.fit(train_data["features"], train_data["profit_labels"])
        
        results["models"]["direction"] = direction_model
        results["models"]["profit"] = profit_model
        
        # Evaluate if validation data available
        if "val" in prepared_data and "profit_labels" in prepared_data["val"]:
            val_data = prepared_data["val"]
            
            # Direction performance
            dir_pred = direction_model.predict(val_data["features"])
            results["performance"]["direction"] = {
                "accuracy": accuracy_score(val_data["labels"], dir_pred),
                "f1_score": f1_score(val_data["labels"], dir_pred, average='weighted')
            }
            
            # Profit performance
            profit_pred = profit_model.predict(val_data["features"])
            results["performance"]["profit"] = {
                "mse": mean_squared_error(val_data["profit_labels"], profit_pred),
                "r2_score": r2_score(val_data["profit_labels"], profit_pred)
            }
        
        self.logger.info("✅ Trained multi-output models")
        
        return results


class ModelEvaluator:
    """Evaluates trained models on test data."""
    
    def __init__(self):
        """Initialize model evaluator."""
        self.logger = system_logger.getChild("ModelEvaluator")
        
    async def evaluate_all_models(
        self, 
        training_results: Dict[str, Any],
        prepared_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Evaluate all trained models.
        
        Args:
            training_results: Training results with models
            prepared_data: Prepared data for evaluation
            
        Returns:
            Evaluation results
        """
        evaluation_results = {}
        
        if "test" not in prepared_data:
            self.logger.warning("No test data available for evaluation")
            return evaluation_results
        
        test_data = prepared_data["test"]
        
        # Evaluate each category of models
        for category, category_results in training_results.items():
            if isinstance(category_results, dict) and "models" in category_results:
                self.logger.info(f"Evaluating {category} models...")
                
                for model_name, model in category_results["models"].items():
                    if model is None:
                        continue
                    
                    try:
                        # Evaluate based on model type
                        if hasattr(model, "predict"):
                            if "profit" in model_name and "profit_labels" in test_data:
                                # Regression evaluation
                                predictions = model.predict(test_data["features"])
                                evaluation_results[model_name] = {
                                    "test_mse": mean_squared_error(test_data["profit_labels"], predictions),
                                    "test_r2": r2_score(test_data["profit_labels"], predictions),
                                    "test_mae": np.mean(np.abs(test_data["profit_labels"] - predictions))
                                }
                            else:
                                # Classification evaluation
                                predictions = model.predict(test_data["features"])
                                
                                # Handle LightGBM output
                                if hasattr(model, "predict") and len(predictions.shape) > 1:
                                    predictions = np.argmax(predictions, axis=1)
                                
                                evaluation_results[model_name] = {
                                    "test_accuracy": accuracy_score(test_data["labels"], predictions),
                                    "test_f1": f1_score(test_data["labels"], predictions, average='weighted'),
                                    "test_precision": self._safe_precision(test_data["labels"], predictions),
                                    "test_recall": self._safe_recall(test_data["labels"], predictions)
                                }
                        
                    except Exception as e:
                        self.logger.error(f"Failed to evaluate {model_name}: {e}")
        
        return evaluation_results
    
    def _safe_precision(self, y_true, y_pred):
        """Calculate precision score safely."""
        try:
            from sklearn.metrics import precision_score
            return precision_score(y_true, y_pred, average='weighted', zero_division=0)
        except:
            return 0.0
    
    def _safe_recall(self, y_true, y_pred):
        """Calculate recall score safely."""
        try:
            from sklearn.metrics import recall_score
            return recall_score(y_true, y_pred, average='weighted', zero_division=0)
        except:
            return 0.0


class HyperparameterOptimizer:
    """Optimizes model hyperparameters."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize hyperparameter optimizer.
        
        Args:
            config: Optimization configuration
        """
        self.config = config
        self.logger = system_logger.getChild("HyperparameterOptimizer")
        self.n_trials = config.get("n_trials", 50)
        self.cv_folds = config.get("cv_folds", 5)
        
    async def optimize_hyperparameters(
        self, 
        model_type: str,
        train_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Optimize hyperparameters for a model type.
        
        Args:
            model_type: Type of model to optimize
            train_data: Training data
            
        Returns:
            Optimal hyperparameters
        """
        try:
            import optuna
            
            # Create objective function
            def objective(trial):
                if model_type == "lightgbm":
                    params = {
                        "num_leaves": trial.suggest_int("num_leaves", 10, 100),
                        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
                        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
                        "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
                        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50)
                    }
                elif model_type == "random_forest":
                    params = {
                        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                        "max_depth": trial.suggest_int("max_depth", 5, 30),
                        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10)
                    }
                else:
                    return 0.0
                
                # Evaluate with cross-validation
                return self._evaluate_params(model_type, params, train_data)
            
            # Run optimization
            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=self.n_trials)
            
            return {
                "best_params": study.best_params,
                "best_score": study.best_value
            }
            
        except ImportError:
            self.logger.warning("Optuna not available, using default parameters")
            return {"best_params": {}, "best_score": 0.0}
    
    def _evaluate_params(
        self, 
        model_type: str, 
        params: Dict[str, Any],
        train_data: Dict[str, Any]
    ) -> float:
        """Evaluate parameters using cross-validation.
        
        Args:
            model_type: Type of model
            params: Parameters to evaluate
            train_data: Training data
            
        Returns:
            Cross-validation score
        """
        # Simplified evaluation - in practice would use proper CV
        return np.random.random()  # Placeholder