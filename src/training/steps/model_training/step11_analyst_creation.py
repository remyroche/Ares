"""Step 11: Analyst Creation - Refactored to use BaseStep.

This step creates base analyst models for each regime using regime-specific
data and features. It focuses on creating robust base models that will be
enhanced in subsequent steps.
"""

import json
import os
import pickle
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import KFold, train_test_split
from torch.utils.data import DataLoader, TensorDataset

from src.training.base_step import BaseStep
from src.core.decorators import handles_errors, traced, validates
from src.utils.logger import system_logger
from src.utils.pipeline_standards import PipelineStandards

# Suppress Optuna's verbose logging
optuna.logging.set_verbosity(optuna.logging.WARNING)


class AnalystModelBuilder:
    """Builds and trains analyst models for each regime."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the model builder.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = system_logger.getChild("AnalystModelBuilder")
        
        # Model configurations
        self.model_types = config.get("model_types", ["lightgbm", "xgboost", "random_forest"])
        self.optimization_trials = config.get("optimization_trials", 50)
        self.cv_folds = config.get("cv_folds", 5)
        
    def build_regime_analyst(
        self,
        regime_id: int,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        """Build and train analyst model for a specific regime.
        
        Args:
            regime_id: Regime identifier
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            
        Returns:
            Dictionary containing model and metadata
        """
        self.logger.info(f"Building analyst for regime {regime_id}")
        
        results = {
            "regime_id": regime_id,
            "models": {},
            "best_model": None,
            "best_score": -np.inf,
            "feature_importance": {},
            "training_metrics": {}
        }
        
        # Train each model type
        for model_type in self.model_types:
            try:
                if model_type == "lightgbm":
                    model_result = self._train_lightgbm(X_train, y_train, X_val, y_val)
                elif model_type == "xgboost":
                    model_result = self._train_xgboost(X_train, y_train, X_val, y_val)
                elif model_type == "random_forest":
                    model_result = self._train_random_forest(X_train, y_train, X_val, y_val)
                else:
                    self.logger.warning(f"Unknown model type: {model_type}")
                    continue
                
                results["models"][model_type] = model_result
                
                # Track best model
                if model_result["validation_score"] > results["best_score"]:
                    results["best_score"] = model_result["validation_score"]
                    results["best_model"] = model_type
                    
            except Exception as e:
                self.logger.error(f"Error training {model_type}: {e}")
                
        # Extract feature importance from best model
        if results["best_model"]:
            best_model_result = results["models"][results["best_model"]]
            results["feature_importance"] = best_model_result.get("feature_importance", {})
            
        return results
    
    def _train_lightgbm(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame],
        y_val: Optional[pd.Series]
    ) -> Dict[str, Any]:
        """Train LightGBM model with optimization."""
        self.logger.info("Training LightGBM model...")
        
        # Create dataset
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_val, label=y_val) if X_val is not None else None
        
        # Optimize hyperparameters
        def objective(trial):
            params = {
                'objective': 'multiclass' if len(np.unique(y_train)) > 2 else 'binary',
                'num_class': len(np.unique(y_train)) if len(np.unique(y_train)) > 2 else 1,
                'metric': 'multi_logloss' if len(np.unique(y_train)) > 2 else 'binary_logloss',
                'boosting_type': 'gbdt',
                'num_leaves': trial.suggest_int('num_leaves', 10, 100),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
                'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
                'verbosity': -1
            }
            
            # Cross-validation
            if valid_data:
                model = lgb.train(
                    params,
                    train_data,
                    valid_sets=[valid_data],
                    num_boost_round=100,
                    callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)]
                )
                score = accuracy_score(y_val, model.predict(X_val, num_iteration=model.best_iteration).argmax(axis=1))
            else:
                # Use cross-validation
                cv_results = lgb.cv(
                    params,
                    train_data,
                    num_boost_round=100,
                    nfold=self.cv_folds,
                    callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)]
                )
                score = -min(cv_results[params['metric'] + '-mean'])
                
            return score
        
        # Run optimization
        study = optuna.create_study(direction='maximize', study_name='lightgbm_opt')
        study.optimize(objective, n_trials=self.optimization_trials)
        
        # Train final model with best parameters
        best_params = study.best_params
        best_params.update({
            'objective': 'multiclass' if len(np.unique(y_train)) > 2 else 'binary',
            'num_class': len(np.unique(y_train)) if len(np.unique(y_train)) > 2 else 1,
            'metric': 'multi_logloss' if len(np.unique(y_train)) > 2 else 'binary_logloss',
            'verbosity': -1
        })
        
        model = lgb.train(
            best_params,
            train_data,
            valid_sets=[valid_data] if valid_data else None,
            num_boost_round=200,
            callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)] if valid_data else []
        )
        
        # Calculate validation score
        if X_val is not None:
            predictions = model.predict(X_val, num_iteration=model.best_iteration)
            if len(predictions.shape) > 1:
                predictions = predictions.argmax(axis=1)
            validation_score = accuracy_score(y_val, predictions)
        else:
            validation_score = study.best_value
            
        # Get feature importance
        importance = model.feature_importance(importance_type='gain')
        feature_importance = dict(zip(X_train.columns, importance))
        
        return {
            "model": model,
            "best_params": best_params,
            "validation_score": validation_score,
            "feature_importance": feature_importance,
            "optimization_history": study.trials_dataframe()
        }
    
    def _train_xgboost(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame],
        y_val: Optional[pd.Series]
    ) -> Dict[str, Any]:
        """Train XGBoost model with optimization."""
        self.logger.info("Training XGBoost model...")
        
        # Optimize hyperparameters
        def objective(trial):
            params = {
                'objective': 'multi:softprob' if len(np.unique(y_train)) > 2 else 'binary:logistic',
                'num_class': len(np.unique(y_train)) if len(np.unique(y_train)) > 2 else None,
                'eval_metric': 'mlogloss' if len(np.unique(y_train)) > 2 else 'logloss',
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'verbosity': 0
            }
            
            # Remove num_class if binary
            if params['num_class'] is None:
                del params['num_class']
                
            model = xgb.XGBClassifier(**params)
            
            if X_val is not None:
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    early_stopping_rounds=10,
                    verbose=False
                )
                score = accuracy_score(y_val, model.predict(X_val))
            else:
                # Use cross-validation
                scores = []
                kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
                for train_idx, val_idx in kf.split(X_train):
                    X_fold_train = X_train.iloc[train_idx]
                    y_fold_train = y_train.iloc[train_idx]
                    X_fold_val = X_train.iloc[val_idx]
                    y_fold_val = y_train.iloc[val_idx]
                    
                    model.fit(X_fold_train, y_fold_train)
                    scores.append(accuracy_score(y_fold_val, model.predict(X_fold_val)))
                    
                score = np.mean(scores)
                
            return score
        
        # Run optimization
        study = optuna.create_study(direction='maximize', study_name='xgboost_opt')
        study.optimize(objective, n_trials=self.optimization_trials)
        
        # Train final model
        best_params = study.best_params
        best_params.update({
            'objective': 'multi:softprob' if len(np.unique(y_train)) > 2 else 'binary:logistic',
            'num_class': len(np.unique(y_train)) if len(np.unique(y_train)) > 2 else None,
            'eval_metric': 'mlogloss' if len(np.unique(y_train)) > 2 else 'logloss',
            'verbosity': 0
        })
        
        if best_params['num_class'] is None:
            del best_params['num_class']
            
        model = xgb.XGBClassifier(**best_params)
        
        if X_val is not None:
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=20,
                verbose=False
            )
            validation_score = accuracy_score(y_val, model.predict(X_val))
        else:
            model.fit(X_train, y_train)
            validation_score = study.best_value
            
        # Get feature importance
        importance = model.feature_importances_
        feature_importance = dict(zip(X_train.columns, importance))
        
        return {
            "model": model,
            "best_params": best_params,
            "validation_score": validation_score,
            "feature_importance": feature_importance,
            "optimization_history": study.trials_dataframe()
        }
    
    def _train_random_forest(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame],
        y_val: Optional[pd.Series]
    ) -> Dict[str, Any]:
        """Train Random Forest model with optimization."""
        self.logger.info("Training Random Forest model...")
        
        # Optimize hyperparameters
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                'random_state': 42,
                'n_jobs': -1
            }
            
            model = RandomForestClassifier(**params)
            
            if X_val is not None:
                model.fit(X_train, y_train)
                score = accuracy_score(y_val, model.predict(X_val))
            else:
                # Use cross-validation
                scores = []
                kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
                for train_idx, val_idx in kf.split(X_train):
                    X_fold_train = X_train.iloc[train_idx]
                    y_fold_train = y_train.iloc[train_idx]
                    X_fold_val = X_train.iloc[val_idx]
                    y_fold_val = y_train.iloc[val_idx]
                    
                    model.fit(X_fold_train, y_fold_train)
                    scores.append(accuracy_score(y_fold_val, model.predict(X_fold_val)))
                    
                score = np.mean(scores)
                
            return score
        
        # Run optimization
        study = optuna.create_study(direction='maximize', study_name='rf_opt')
        study.optimize(objective, n_trials=self.optimization_trials)
        
        # Train final model
        best_params = study.best_params
        best_params.update({'random_state': 42, 'n_jobs': -1})
        
        model = RandomForestClassifier(**best_params)
        model.fit(X_train, y_train)
        
        if X_val is not None:
            validation_score = accuracy_score(y_val, model.predict(X_val))
        else:
            validation_score = study.best_value
            
        # Get feature importance
        importance = model.feature_importances_
        feature_importance = dict(zip(X_train.columns, importance))
        
        return {
            "model": model,
            "best_params": best_params,
            "validation_score": validation_score,
            "feature_importance": feature_importance,
            "optimization_history": study.trials_dataframe()
        }


class MultiOutputAnalystBuilder:
    """Builds analyst models that support multiple output types."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the multi-output builder.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = system_logger.getChild("MultiOutputAnalystBuilder")
        self.base_builder = AnalystModelBuilder(config)
        
    def build_multi_output_analyst(
        self,
        regime_id: int,
        X_train: pd.DataFrame,
        y_train_dict: Dict[str, pd.Series],
        X_val: Optional[pd.DataFrame] = None,
        y_val_dict: Optional[Dict[str, pd.Series]] = None
    ) -> Dict[str, Any]:
        """Build analyst models for multiple outputs.
        
        Args:
            regime_id: Regime identifier
            X_train: Training features
            y_train_dict: Dictionary of training labels for each output
            X_val: Validation features (optional)
            y_val_dict: Dictionary of validation labels (optional)
            
        Returns:
            Dictionary containing models for each output
        """
        self.logger.info(f"Building multi-output analyst for regime {regime_id}")
        
        results = {
            "regime_id": regime_id,
            "output_models": {},
            "aggregated_metrics": {},
            "feature_importance": {}
        }
        
        # Build model for each output
        for output_name, y_train in y_train_dict.items():
            y_val = y_val_dict.get(output_name) if y_val_dict else None
            
            output_result = self.base_builder.build_regime_analyst(
                regime_id, X_train, y_train, X_val, y_val
            )
            
            results["output_models"][output_name] = output_result
            
            # Aggregate feature importance
            if output_result["feature_importance"]:
                for feature, importance in output_result["feature_importance"].items():
                    if feature not in results["feature_importance"]:
                        results["feature_importance"][feature] = {}
                    results["feature_importance"][feature][output_name] = importance
        
        # Calculate aggregated metrics
        results["aggregated_metrics"] = self._calculate_aggregated_metrics(
            results["output_models"]
        )
        
        return results
    
    def _calculate_aggregated_metrics(
        self,
        output_models: Dict[str, Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calculate aggregated metrics across all outputs."""
        metrics = {
            "avg_validation_score": 0.0,
            "min_validation_score": float('inf'),
            "max_validation_score": -float('inf'),
            "output_scores": {}
        }
        
        scores = []
        for output_name, model_result in output_models.items():
            score = model_result["best_score"]
            scores.append(score)
            metrics["output_scores"][output_name] = score
            metrics["min_validation_score"] = min(metrics["min_validation_score"], score)
            metrics["max_validation_score"] = max(metrics["max_validation_score"], score)
            
        if scores:
            metrics["avg_validation_score"] = np.mean(scores)
            
        return metrics


class AnalystCreationStep(BaseStep):
    """Step 11: Analyst Creation - Creates base analyst models for each regime."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the step."""
        super().__init__(config, "11", "analyst_creation")
        
    def _initialize_step(self) -> None:
        """Initialize step-specific components."""
        # Initialize builders
        self.model_builder = AnalystModelBuilder(self.config)
        self.multi_output_builder = MultiOutputAnalystBuilder(self.config)
        
        # Configuration
        self.use_multi_output = self.config.get("use_multi_output", False)
        self.validation_split = self.config.get("validation_split", 0.2)
        self.random_state = self.config.get("random_state", 42)
        
    def get_required_inputs(self) -> List[str]:
        """Get required inputs for this step."""
        return [
            "regime_features",
            "regime_labels",
            "num_regimes"
        ]
    
    def get_produced_outputs(self) -> List[str]:
        """Get outputs produced by this step."""
        return [
            "regime_analysts",
            "analyst_metadata",
            "feature_importance",
            "analyst_performance"
        ]
    
    def get_dependencies(self) -> List[str]:
        """Get step dependencies."""
        return ["step10_unified_regime_intelligence"]
    
    @validates(
        input_schema={
            "training_input": dict,
            "pipeline_state": dict
        }
    )
    def validate_inputs(
        self,
        training_input: Dict[str, Any],
        pipeline_state: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        """Validate step inputs."""
        errors = []
        
        # Check required inputs
        for key in self.get_required_inputs():
            if key not in pipeline_state:
                errors.append(f"Missing required input: {key}")
        
        # Validate regime features
        if "regime_features" in pipeline_state:
            features = pipeline_state["regime_features"]
            if not isinstance(features, pd.DataFrame):
                errors.append("regime_features must be a pandas DataFrame")
            elif features.empty:
                errors.append("regime_features cannot be empty")
        
        # Validate regime labels
        if "regime_labels" in pipeline_state:
            labels = pipeline_state["regime_labels"]
            if self.use_multi_output:
                if not isinstance(labels, dict):
                    errors.append("regime_labels must be a dictionary for multi-output mode")
            else:
                if not isinstance(labels, (pd.Series, np.ndarray)):
                    errors.append("regime_labels must be a Series or array for single-output mode")
        
        # Validate num_regimes
        if "num_regimes" in pipeline_state:
            num_regimes = pipeline_state["num_regimes"]
            if not isinstance(num_regimes, int) or num_regimes <= 0:
                errors.append("num_regimes must be a positive integer")
        
        return len(errors) == 0, errors
    
    @traced
    @handles_errors(
        exceptions=(Exception,),
        default_return={},
        context="analyst creation execution"
    )
    async def execute_logic(
        self,
        training_input: Dict[str, Any],
        pipeline_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute the analyst creation logic."""
        self.logger.info("Starting analyst creation...")
        
        # Extract inputs
        regime_features = pipeline_state["regime_features"]
        regime_labels = pipeline_state["regime_labels"]
        num_regimes = pipeline_state["num_regimes"]
        
        # Initialize results storage
        regime_analysts = {}
        analyst_metadata = {}
        feature_importance = {}
        analyst_performance = {}
        
        # Split data by regime
        self.logger.info(f"Creating analysts for {num_regimes} regimes...")
        
        for regime_id in range(num_regimes):
            self.logger.info(f"Processing regime {regime_id}...")
            
            # Get regime-specific data
            if isinstance(regime_labels, dict):
                # Multi-output case - use first output to determine regime membership
                first_output = list(regime_labels.keys())[0]
                regime_mask = regime_labels[first_output] == regime_id
            else:
                # Single output case
                regime_mask = regime_labels == regime_id
            
            regime_X = regime_features[regime_mask]
            
            if len(regime_X) < 10:  # Skip if too few samples
                self.logger.warning(f"Regime {regime_id} has only {len(regime_X)} samples, skipping...")
                continue
            
            # Split into train/validation
            train_idx, val_idx = train_test_split(
                np.arange(len(regime_X)),
                test_size=self.validation_split,
                random_state=self.random_state
            )
            
            X_train = regime_X.iloc[train_idx]
            X_val = regime_X.iloc[val_idx]
            
            # Build analyst based on mode
            if self.use_multi_output and isinstance(regime_labels, dict):
                # Multi-output mode
                y_train_dict = {
                    output: labels[regime_mask].iloc[train_idx]
                    for output, labels in regime_labels.items()
                }
                y_val_dict = {
                    output: labels[regime_mask].iloc[val_idx]
                    for output, labels in regime_labels.items()
                }
                
                analyst_result = self.multi_output_builder.build_multi_output_analyst(
                    regime_id, X_train, y_train_dict, X_val, y_val_dict
                )
            else:
                # Single output mode
                if isinstance(regime_labels, dict):
                    # Use first output if dict provided in single-output mode
                    first_output = list(regime_labels.keys())[0]
                    y_train = regime_labels[first_output][regime_mask].iloc[train_idx]
                    y_val = regime_labels[first_output][regime_mask].iloc[val_idx]
                else:
                    y_train = regime_labels[regime_mask].iloc[train_idx]
                    y_val = regime_labels[regime_mask].iloc[val_idx]
                
                analyst_result = self.model_builder.build_regime_analyst(
                    regime_id, X_train, y_train, X_val, y_val
                )
            
            # Store results
            regime_analysts[f"regime_{regime_id}"] = analyst_result
            
            # Extract metadata
            analyst_metadata[f"regime_{regime_id}"] = {
                "training_samples": len(X_train),
                "validation_samples": len(X_val),
                "best_model_type": analyst_result.get("best_model"),
                "best_score": analyst_result.get("best_score", 0.0)
            }
            
            # Aggregate feature importance
            if analyst_result.get("feature_importance"):
                feature_importance[f"regime_{regime_id}"] = analyst_result["feature_importance"]
            
            # Store performance metrics
            if self.use_multi_output:
                analyst_performance[f"regime_{regime_id}"] = analyst_result.get("aggregated_metrics", {})
            else:
                analyst_performance[f"regime_{regime_id}"] = {
                    "validation_score": analyst_result.get("best_score", 0.0),
                    "model_type": analyst_result.get("best_model")
                }
        
        # Calculate overall metrics
        overall_metrics = self._calculate_overall_metrics(analyst_performance)
        analyst_performance["overall"] = overall_metrics
        
        # Update pipeline state
        result = pipeline_state.copy()
        result.update({
            "regime_analysts": regime_analysts,
            "analyst_metadata": analyst_metadata,
            "feature_importance": feature_importance,
            "analyst_performance": analyst_performance
        })
        
        # Save artifacts
        await self._save_artifacts(result)
        
        self.logger.info(f"Analyst creation completed. Created {len(regime_analysts)} regime analysts.")
        
        return result
    
    def _calculate_overall_metrics(
        self,
        analyst_performance: Dict[str, Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calculate overall performance metrics across all regimes."""
        scores = []
        for regime_id, performance in analyst_performance.items():
            if isinstance(performance, dict):
                if "validation_score" in performance:
                    scores.append(performance["validation_score"])
                elif "avg_validation_score" in performance:
                    scores.append(performance["avg_validation_score"])
        
        if scores:
            return {
                "mean_score": np.mean(scores),
                "std_score": np.std(scores),
                "min_score": np.min(scores),
                "max_score": np.max(scores),
                "num_regimes": len(scores)
            }
        else:
            return {"num_regimes": 0}
    
    async def _save_artifacts(self, result: Dict[str, Any]) -> None:
        """Save step artifacts."""
        artifacts_dir = Path(self.config.get("artifacts_dir", "artifacts")) / self.full_step_name
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        # Save regime analysts
        if "regime_analysts" in result:
            analysts_dir = artifacts_dir / "regime_analysts"
            analysts_dir.mkdir(exist_ok=True)
            
            for regime_id, analyst_data in result["regime_analysts"].items():
                regime_dir = analysts_dir / regime_id
                regime_dir.mkdir(exist_ok=True)
                
                # Save models
                if "models" in analyst_data:
                    for model_type, model_result in analyst_data["models"].items():
                        if "model" in model_result:
                            model_path = regime_dir / f"{model_type}_model.pkl"
                            joblib.dump(model_result["model"], model_path)
                
                # Save metadata
                metadata = {
                    "regime_id": analyst_data.get("regime_id"),
                    "best_model": analyst_data.get("best_model"),
                    "best_score": analyst_data.get("best_score"),
                    "feature_importance": analyst_data.get("feature_importance", {})
                }
                with open(regime_dir / "metadata.json", "w") as f:
                    json.dump(metadata, f, indent=2)
        
        # Save overall metadata
        if "analyst_metadata" in result:
            with open(artifacts_dir / "analyst_metadata.json", "w") as f:
                json.dump(result["analyst_metadata"], f, indent=2)
        
        # Save performance metrics
        if "analyst_performance" in result:
            with open(artifacts_dir / "analyst_performance.json", "w") as f:
                json.dump(result["analyst_performance"], f, indent=2)
        
        self.logger.info(f"Artifacts saved to {artifacts_dir}")
    
    def validate_outputs(
        self,
        pipeline_state: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        """Validate step outputs."""
        errors = []
        
        # Check required outputs
        required_outputs = ["regime_analysts", "analyst_metadata", "analyst_performance"]
        for output in required_outputs:
            if output not in pipeline_state:
                errors.append(f"Missing required output: {output}")
            elif pipeline_state[output] is None:
                errors.append(f"Output {output} is None")
        
        # Validate regime analysts structure
        if "regime_analysts" in pipeline_state:
            analysts = pipeline_state["regime_analysts"]
            if not isinstance(analysts, dict):
                errors.append("regime_analysts must be a dictionary")
            elif not analysts:
                errors.append("No regime analysts were created")
            else:
                # Check each analyst
                for regime_id, analyst_data in analysts.items():
                    if not isinstance(analyst_data, dict):
                        errors.append(f"Analyst data for {regime_id} must be a dictionary")
                    elif "models" not in analyst_data:
                        errors.append(f"No models found for {regime_id}")
        
        # Validate performance metrics
        if "analyst_performance" in pipeline_state:
            performance = pipeline_state["analyst_performance"]
            if "overall" not in performance:
                errors.append("Missing overall performance metrics")
        
        return len(errors) == 0, errors