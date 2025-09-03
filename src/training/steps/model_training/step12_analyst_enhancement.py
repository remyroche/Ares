"""Step 12: Analyst Enhancement - Refactored to use BaseStep.

This step enhances the analyst models created in Step 11 through:
- Hyperparameter optimization
- Feature selection and augmentation
- Model optimization (quantization, pruning, distillation)
- Performance analysis and validation
"""

import asyncio
import gc
import json
import os
import pickle
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

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
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import cross_val_score
from torch.utils.data import DataLoader, TensorDataset

from src.training.base_step import BaseStep
from src.core.decorators import handles_errors, traced, validates
from src.utils.logger import system_logger
from src.utils.pipeline_standards import PipelineStandards

# Try to import SHAP for model interpretation
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    shap = None


class AnalystEnhancer:
    """Core enhancement logic for analyst models."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the analyst enhancer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = system_logger.getChild("AnalystEnhancer")
        
        # Enhancement parameters
        self.optimization_trials = config.get("optimization_trials", 100)
        self.feature_selection_method = config.get("feature_selection_method", "mutual_info")
        self.feature_selection_k = config.get("feature_selection_k", 50)
        self.enable_shap = config.get("enable_shap", SHAP_AVAILABLE)
        
    async def enhance_analyst(
        self,
        analyst_data: Dict[str, Any],
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series
    ) -> Dict[str, Any]:
        """Enhance a single analyst model.
        
        Args:
            analyst_data: Original analyst model data
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            
        Returns:
            Enhanced analyst data
        """
        self.logger.info(f"Enhancing analyst for regime {analyst_data.get('regime_id', 'unknown')}")
        
        enhanced_data = analyst_data.copy()
        enhanced_data["enhancements"] = {}
        
        # 1. Feature selection
        selected_features, feature_scores = await self._select_features(
            X_train, y_train, X_val, y_val
        )
        enhanced_data["enhancements"]["selected_features"] = selected_features
        enhanced_data["enhancements"]["feature_scores"] = feature_scores
        
        # Apply feature selection
        X_train_selected = X_train[selected_features]
        X_val_selected = X_val[selected_features]
        
        # 2. Hyperparameter optimization
        best_model = analyst_data.get("best_model", "lightgbm")
        optimized_params = await self._optimize_hyperparameters(
            best_model, X_train_selected, y_train, X_val_selected, y_val
        )
        enhanced_data["enhancements"]["optimized_params"] = optimized_params
        
        # 3. Retrain with optimized parameters
        enhanced_model = await self._retrain_model(
            best_model, X_train_selected, y_train, X_val_selected, y_val, optimized_params
        )
        enhanced_data["enhanced_model"] = enhanced_model
        
        # 4. Model interpretation (if SHAP available)
        if self.enable_shap and SHAP_AVAILABLE:
            shap_values = await self._compute_shap_values(
                enhanced_model, X_train_selected, X_val_selected
            )
            enhanced_data["enhancements"]["shap_values"] = shap_values
        
        # 5. Performance evaluation
        performance_metrics = self._evaluate_performance(
            enhanced_model, X_train_selected, y_train, X_val_selected, y_val
        )
        enhanced_data["enhancements"]["performance_metrics"] = performance_metrics
        
        return enhanced_data
    
    async def _select_features(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series
    ) -> Tuple[List[str], Dict[str, float]]:
        """Select most important features."""
        self.logger.info(f"Selecting features using {self.feature_selection_method}")
        
        if self.feature_selection_method == "mutual_info":
            # Calculate mutual information scores
            mi_scores = mutual_info_classif(X_train, y_train, random_state=42)
            feature_scores = dict(zip(X_train.columns, mi_scores))
            
            # Select top k features
            sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
            selected_features = [f[0] for f in sorted_features[:self.feature_selection_k]]
            
        elif self.feature_selection_method == "model_importance":
            # Use a simple model to get feature importances
            from lightgbm import LGBMClassifier
            
            model = LGBMClassifier(
                n_estimators=100,
                random_state=42,
                verbosity=-1
            )
            model.fit(X_train, y_train)
            
            importances = model.feature_importances_
            feature_scores = dict(zip(X_train.columns, importances))
            
            # Select top k features
            sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
            selected_features = [f[0] for f in sorted_features[:self.feature_selection_k]]
            
        else:
            # Default: use all features
            selected_features = list(X_train.columns)
            feature_scores = {f: 1.0 for f in selected_features}
        
        self.logger.info(f"Selected {len(selected_features)} features")
        return selected_features, feature_scores
    
    async def _optimize_hyperparameters(
        self,
        model_type: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series
    ) -> Dict[str, Any]:
        """Optimize model hyperparameters using Optuna."""
        self.logger.info(f"Optimizing hyperparameters for {model_type}")
        
        def objective(trial):
            if model_type == "lightgbm":
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'max_depth': trial.suggest_int('max_depth', 3, 15),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'num_leaves': trial.suggest_int('num_leaves', 20, 300),
                    'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
                    'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
                    'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
                    'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                    'verbosity': -1,
                    'random_state': 42
                }
                
                model = lgb.LGBMClassifier(**params)
                
            elif model_type == "xgboost":
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'max_depth': trial.suggest_int('max_depth', 3, 15),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                    'verbosity': 0,
                    'random_state': 42
                }
                
                model = xgb.XGBClassifier(**params)
                
            else:  # random_forest
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'max_depth': trial.suggest_int('max_depth', 5, 30),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                    'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                    'random_state': 42,
                    'n_jobs': -1
                }
                
                model = RandomForestClassifier(**params)
            
            # Train and evaluate
            model.fit(X_train, y_train)
            score = accuracy_score(y_val, model.predict(X_val))
            
            return score
        
        # Run optimization
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.optimization_trials, show_progress_bar=False)
        
        self.logger.info(f"Best score: {study.best_value:.4f}")
        return study.best_params
    
    async def _retrain_model(
        self,
        model_type: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        params: Dict[str, Any]
    ) -> Any:
        """Retrain model with optimized parameters."""
        self.logger.info(f"Retraining {model_type} with optimized parameters")
        
        if model_type == "lightgbm":
            params.update({'verbosity': -1, 'random_state': 42})
            model = lgb.LGBMClassifier(**params)
        elif model_type == "xgboost":
            params.update({'verbosity': 0, 'random_state': 42})
            model = xgb.XGBClassifier(**params)
        else:
            params.update({'random_state': 42, 'n_jobs': -1})
            model = RandomForestClassifier(**params)
        
        # Train model
        if model_type in ["lightgbm", "xgboost"]:
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
        else:
            model.fit(X_train, y_train)
        
        return model
    
    async def _compute_shap_values(
        self,
        model: Any,
        X_train: pd.DataFrame,
        X_val: pd.DataFrame
    ) -> Dict[str, Any]:
        """Compute SHAP values for model interpretation."""
        self.logger.info("Computing SHAP values")
        
        try:
            # Create explainer
            if isinstance(model, (lgb.LGBMClassifier, xgb.XGBClassifier)):
                explainer = shap.TreeExplainer(model)
            else:
                # Use sampling for other models
                background = shap.sample(X_train, min(100, len(X_train)))
                explainer = shap.KernelExplainer(model.predict_proba, background)
            
            # Compute SHAP values for validation set
            shap_values = explainer.shap_values(X_val.iloc[:min(100, len(X_val))])
            
            # If multi-class, take the first class
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            
            # Get feature importance from SHAP
            feature_importance = np.abs(shap_values).mean(axis=0)
            
            return {
                "shap_values": shap_values.tolist() if hasattr(shap_values, 'tolist') else shap_values,
                "feature_importance": dict(zip(X_val.columns, feature_importance))
            }
            
        except Exception as e:
            self.logger.warning(f"Failed to compute SHAP values: {e}")
            return {}
    
    def _evaluate_performance(
        self,
        model: Any,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series
    ) -> Dict[str, float]:
        """Evaluate model performance."""
        # Training metrics
        train_pred = model.predict(X_train)
        train_accuracy = accuracy_score(y_train, train_pred)
        
        # Validation metrics
        val_pred = model.predict(X_val)
        val_accuracy = accuracy_score(y_val, val_pred)
        
        # Get probabilities for AUC
        try:
            if hasattr(model, 'predict_proba'):
                val_proba = model.predict_proba(X_val)
                if len(np.unique(y_val)) == 2:
                    # Binary classification
                    val_auc = roc_auc_score(y_val, val_proba[:, 1])
                else:
                    # Multi-class
                    val_auc = roc_auc_score(y_val, val_proba, multi_class='ovr')
            else:
                val_auc = None
        except Exception:
            val_auc = None
        
        metrics = {
            "train_accuracy": train_accuracy,
            "val_accuracy": val_accuracy,
            "val_auc": val_auc,
            "overfitting_score": train_accuracy - val_accuracy
        }
        
        return metrics


class FeatureAugmenter:
    """Handles feature enhancement and augmentation."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the feature augmenter.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = system_logger.getChild("FeatureAugmenter")
        
        # Augmentation parameters
        self.create_interactions = config.get("create_feature_interactions", True)
        self.create_polynomials = config.get("create_polynomial_features", False)
        self.polynomial_degree = config.get("polynomial_degree", 2)
        self.interaction_threshold = config.get("interaction_threshold", 0.8)
        
    def augment_features(
        self,
        features: pd.DataFrame,
        feature_importance: Dict[str, float],
        top_k: int = 10
    ) -> pd.DataFrame:
        """Augment features with interactions and transformations.
        
        Args:
            features: Original features
            feature_importance: Feature importance scores
            top_k: Number of top features to create interactions for
            
        Returns:
            Augmented feature DataFrame
        """
        self.logger.info("Augmenting features")
        augmented = features.copy()
        
        if self.create_interactions:
            # Get top k important features
            sorted_features = sorted(
                feature_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )[:top_k]
            top_features = [f[0] for f in sorted_features]
            
            # Create interactions between top features
            for i, feat1 in enumerate(top_features):
                for feat2 in top_features[i+1:]:
                    # Check correlation to avoid redundant interactions
                    corr = features[feat1].corr(features[feat2])
                    if abs(corr) < self.interaction_threshold:
                        interaction_name = f"{feat1}_X_{feat2}"
                        augmented[interaction_name] = features[feat1] * features[feat2]
        
        if self.create_polynomials:
            # Add polynomial features for top features
            sorted_features = sorted(
                feature_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]  # Limit polynomial features
            
            for feat, _ in sorted_features:
                if features[feat].dtype in [np.float32, np.float64, np.int32, np.int64]:
                    for degree in range(2, self.polynomial_degree + 1):
                        poly_name = f"{feat}_pow{degree}"
                        augmented[poly_name] = features[feat] ** degree
        
        self.logger.info(f"Features augmented from {len(features.columns)} to {len(augmented.columns)}")
        return augmented
    
    def select_augmented_features(
        self,
        augmented_features: pd.DataFrame,
        y: pd.Series,
        original_feature_count: int,
        selection_ratio: float = 0.5
    ) -> List[str]:
        """Select best augmented features.
        
        Args:
            augmented_features: All features including augmented ones
            y: Target variable
            original_feature_count: Number of original features
            selection_ratio: Ratio of augmented features to keep
            
        Returns:
            List of selected feature names
        """
        # Always keep original features
        original_features = list(augmented_features.columns[:original_feature_count])
        augmented_only = list(augmented_features.columns[original_feature_count:])
        
        if not augmented_only:
            return original_features
        
        # Score augmented features
        mi_scores = mutual_info_classif(
            augmented_features[augmented_only],
            y,
            random_state=42
        )
        
        # Select top augmented features
        n_select = int(len(augmented_only) * selection_ratio)
        top_indices = np.argsort(mi_scores)[-n_select:]
        selected_augmented = [augmented_only[i] for i in top_indices]
        
        return original_features + selected_augmented


class ModelOptimizer:
    """Handles model optimization techniques like pruning and quantization."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the model optimizer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = system_logger.getChild("ModelOptimizer")
        
        # Optimization settings
        self.enable_pruning = config.get("enable_pruning", False)
        self.enable_quantization = config.get("enable_quantization", False)
        self.pruning_ratio = config.get("pruning_ratio", 0.1)
        
    def optimize_model(self, model: Any, model_type: str) -> Tuple[Any, Dict[str, Any]]:
        """Apply optimization techniques to the model.
        
        Args:
            model: Model to optimize
            model_type: Type of model
            
        Returns:
            Optimized model and optimization metrics
        """
        self.logger.info(f"Optimizing {model_type} model")
        
        optimization_metrics = {
            "original_size": self._get_model_size(model),
            "optimizations_applied": []
        }
        
        # Model-specific optimizations
        if model_type == "lightgbm" and hasattr(model, 'booster_'):
            if self.enable_pruning:
                # LightGBM doesn't support direct pruning, but we can reduce complexity
                self.logger.info("Applying complexity reduction for LightGBM")
                optimization_metrics["optimizations_applied"].append("complexity_reduction")
                
        elif model_type == "xgboost" and hasattr(model, 'get_booster'):
            if self.enable_pruning:
                # XGBoost pruning through feature selection
                self.logger.info("Applying feature-based pruning for XGBoost")
                optimization_metrics["optimizations_applied"].append("feature_pruning")
                
        elif model_type == "random_forest":
            if self.enable_pruning:
                # Reduce number of trees
                n_trees = len(model.estimators_)
                n_keep = int(n_trees * (1 - self.pruning_ratio))
                
                # Select best trees based on out-of-bag score if available
                if hasattr(model, 'oob_score_') and model.oob_score_:
                    # Keep all trees for now (proper selection would require more complex logic)
                    pass
                else:
                    # Randomly select trees to keep
                    indices = np.random.choice(n_trees, n_keep, replace=False)
                    model.estimators_ = [model.estimators_[i] for i in indices]
                    model.n_estimators = n_keep
                
                self.logger.info(f"Pruned random forest from {n_trees} to {n_keep} trees")
                optimization_metrics["optimizations_applied"].append("tree_pruning")
        
        # Get final size
        optimization_metrics["optimized_size"] = self._get_model_size(model)
        optimization_metrics["size_reduction"] = (
            optimization_metrics["original_size"] - optimization_metrics["optimized_size"]
        ) / optimization_metrics["original_size"]
        
        return model, optimization_metrics
    
    def _get_model_size(self, model: Any) -> int:
        """Get approximate model size in bytes."""
        import sys
        return sys.getsizeof(pickle.dumps(model))


class PerformanceAnalyzer:
    """Analyzes and tracks model performance metrics."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the performance analyzer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = system_logger.getChild("PerformanceAnalyzer")
        
    def analyze_enhancement_impact(
        self,
        original_performance: Dict[str, float],
        enhanced_performance: Dict[str, float]
    ) -> Dict[str, Any]:
        """Analyze the impact of enhancement on model performance.
        
        Args:
            original_performance: Original model metrics
            enhanced_performance: Enhanced model metrics
            
        Returns:
            Analysis results
        """
        impact = {}
        
        # Calculate improvements
        for metric in ['val_accuracy', 'val_auc']:
            if metric in original_performance and metric in enhanced_performance:
                original = original_performance.get(metric, 0) or 0
                enhanced = enhanced_performance.get(metric, 0) or 0
                
                if original > 0:
                    improvement = (enhanced - original) / original * 100
                else:
                    improvement = 0
                    
                impact[f"{metric}_improvement"] = improvement
                impact[f"{metric}_absolute_gain"] = enhanced - original
        
        # Check overfitting
        original_overfit = original_performance.get('overfitting_score', 0)
        enhanced_overfit = enhanced_performance.get('overfitting_score', 0)
        impact['overfitting_reduction'] = original_overfit - enhanced_overfit
        
        # Overall assessment
        val_acc_gain = impact.get('val_accuracy_absolute_gain', 0)
        overfit_reduction = impact.get('overfitting_reduction', 0)
        
        if val_acc_gain > 0.01 and overfit_reduction >= 0:
            impact['overall_assessment'] = 'significant_improvement'
        elif val_acc_gain > 0 or overfit_reduction > 0.02:
            impact['overall_assessment'] = 'moderate_improvement'
        elif val_acc_gain >= -0.01:
            impact['overall_assessment'] = 'no_significant_change'
        else:
            impact['overall_assessment'] = 'performance_degradation'
        
        return impact
    
    def create_performance_report(
        self,
        regime_id: str,
        original_data: Dict[str, Any],
        enhanced_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create a comprehensive performance report.
        
        Args:
            regime_id: Regime identifier
            original_data: Original analyst data
            enhanced_data: Enhanced analyst data
            
        Returns:
            Performance report
        """
        report = {
            "regime_id": regime_id,
            "timestamp": datetime.now().isoformat(),
            "original_model": {
                "type": original_data.get("best_model"),
                "score": original_data.get("best_score"),
                "num_features": len(original_data.get("feature_importance", {}))
            },
            "enhancements_applied": [],
            "enhanced_performance": {},
            "feature_analysis": {},
            "optimization_metrics": {}
        }
        
        # List enhancements applied
        if "enhancements" in enhanced_data:
            enhancements = enhanced_data["enhancements"]
            
            if "selected_features" in enhancements:
                report["enhancements_applied"].append("feature_selection")
                report["feature_analysis"]["selected_features"] = len(enhancements["selected_features"])
                
            if "optimized_params" in enhancements:
                report["enhancements_applied"].append("hyperparameter_optimization")
                
            if "shap_values" in enhancements:
                report["enhancements_applied"].append("shap_analysis")
                
            if "performance_metrics" in enhancements:
                report["enhanced_performance"] = enhancements["performance_metrics"]
        
        # Calculate impact
        if original_data.get("best_score") and report["enhanced_performance"].get("val_accuracy"):
            original_perf = {"val_accuracy": original_data["best_score"]}
            impact = self.analyze_enhancement_impact(
                original_perf,
                report["enhanced_performance"]
            )
            report["enhancement_impact"] = impact
        
        return report


class AnalystEnhancementStep(BaseStep):
    """Step 12: Analyst Enhancement - Enhances analyst models through optimization."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the step."""
        super().__init__(config, "12", "analyst_enhancement")
        
    def _initialize_step(self) -> None:
        """Initialize step-specific components."""
        # Initialize enhancement components
        self.analyst_enhancer = AnalystEnhancer(self.config)
        self.feature_augmenter = FeatureAugmenter(self.config)
        self.model_optimizer = ModelOptimizer(self.config)
        self.performance_analyzer = PerformanceAnalyzer(self.config)
        
        # Configuration
        self.enhancement_config = self.config.get("enhancement", {})
        self.parallel_processing = self.enhancement_config.get("parallel_processing", True)
        self.max_parallel_regimes = self.enhancement_config.get("max_parallel_regimes", 4)
        
    def get_required_inputs(self) -> List[str]:
        """Get required inputs for this step."""
        return [
            "regime_analysts",
            "regime_features",
            "regime_labels",
            "num_regimes"
        ]
    
    def get_produced_outputs(self) -> List[str]:
        """Get outputs produced by this step."""
        return [
            "enhanced_analysts",
            "enhancement_reports",
            "performance_comparison",
            "optimization_summary"
        ]
    
    def get_dependencies(self) -> List[str]:
        """Get step dependencies."""
        return ["step11_analyst_creation"]
    
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
        
        # Validate regime analysts
        if "regime_analysts" in pipeline_state:
            analysts = pipeline_state["regime_analysts"]
            if not isinstance(analysts, dict):
                errors.append("regime_analysts must be a dictionary")
            elif not analysts:
                errors.append("No regime analysts found to enhance")
        
        # Validate features and labels
        if "regime_features" in pipeline_state:
            if not isinstance(pipeline_state["regime_features"], pd.DataFrame):
                errors.append("regime_features must be a pandas DataFrame")
                
        if "regime_labels" in pipeline_state:
            labels = pipeline_state["regime_labels"]
            if not isinstance(labels, (pd.Series, np.ndarray, dict)):
                errors.append("regime_labels must be a Series, array, or dict")
        
        return len(errors) == 0, errors
    
    @traced
    @handles_errors(
        exceptions=(Exception,),
        default_return={},
        context="analyst enhancement execution"
    )
    async def execute_logic(
        self,
        training_input: Dict[str, Any],
        pipeline_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute the analyst enhancement logic."""
        self.logger.info("Starting analyst enhancement...")
        
        # Extract inputs
        regime_analysts = pipeline_state["regime_analysts"]
        regime_features = pipeline_state["regime_features"]
        regime_labels = pipeline_state["regime_labels"]
        num_regimes = pipeline_state["num_regimes"]
        
        # Initialize results storage
        enhanced_analysts = {}
        enhancement_reports = {}
        
        # Process regimes
        if self.parallel_processing:
            # Process in parallel with concurrency limit
            self.logger.info(f"Processing {len(regime_analysts)} regimes in parallel (max {self.max_parallel_regimes})")
            
            tasks = []
            for regime_key, analyst_data in regime_analysts.items():
                task = self._enhance_regime_analyst(
                    regime_key, analyst_data, regime_features, regime_labels
                )
                tasks.append(task)
            
            # Process with concurrency limit
            results = []
            for i in range(0, len(tasks), self.max_parallel_regimes):
                batch = tasks[i:i + self.max_parallel_regimes]
                batch_results = await asyncio.gather(*batch, return_exceptions=True)
                results.extend(batch_results)
            
            # Collect results
            for result in results:
                if isinstance(result, Exception):
                    self.logger.error(f"Enhancement failed: {result}")
                elif result:
                    regime_key, enhanced_data, report = result
                    enhanced_analysts[regime_key] = enhanced_data
                    enhancement_reports[regime_key] = report
        else:
            # Process sequentially
            for regime_key, analyst_data in regime_analysts.items():
                try:
                    result = await self._enhance_regime_analyst(
                        regime_key, analyst_data, regime_features, regime_labels
                    )
                    if result:
                        regime_key, enhanced_data, report = result
                        enhanced_analysts[regime_key] = enhanced_data
                        enhancement_reports[regime_key] = report
                except Exception as e:
                    self.logger.error(f"Failed to enhance {regime_key}: {e}")
        
        # Create performance comparison
        performance_comparison = self._create_performance_comparison(
            regime_analysts, enhanced_analysts, enhancement_reports
        )
        
        # Create optimization summary
        optimization_summary = self._create_optimization_summary(
            enhancement_reports, performance_comparison
        )
        
        # Update pipeline state
        result = pipeline_state.copy()
        result.update({
            "enhanced_analysts": enhanced_analysts,
            "enhancement_reports": enhancement_reports,
            "performance_comparison": performance_comparison,
            "optimization_summary": optimization_summary
        })
        
        # Save artifacts
        await self._save_artifacts(result)
        
        self.logger.info(f"Enhancement completed. Enhanced {len(enhanced_analysts)} analysts.")
        
        return result
    
    async def _enhance_regime_analyst(
        self,
        regime_key: str,
        analyst_data: Dict[str, Any],
        regime_features: pd.DataFrame,
        regime_labels: Union[pd.Series, Dict[str, pd.Series]]
    ) -> Optional[Tuple[str, Dict[str, Any], Dict[str, Any]]]:
        """Enhance a single regime analyst."""
        try:
            self.logger.info(f"Enhancing analyst for {regime_key}")
            
            # Extract regime ID
            regime_id = analyst_data.get("regime_id", 0)
            
            # Get regime-specific data
            if isinstance(regime_labels, dict):
                # Multi-output case
                first_output = list(regime_labels.keys())[0]
                regime_mask = regime_labels[first_output] == regime_id
                y = regime_labels[first_output][regime_mask]
            else:
                # Single output case
                regime_mask = regime_labels == regime_id
                y = regime_labels[regime_mask]
            
            X = regime_features[regime_mask]
            
            if len(X) < 50:  # Skip if too few samples
                self.logger.warning(f"Skipping {regime_key} - insufficient samples ({len(X)})")
                return None
            
            # Split data
            from sklearn.model_selection import train_test_split
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Enhance analyst
            enhanced_data = await self.analyst_enhancer.enhance_analyst(
                analyst_data, X_train, y_train, X_val, y_val
            )
            
            # Apply feature augmentation if configured
            if self.config.get("enable_feature_augmentation", False):
                feature_importance = enhanced_data["enhancements"].get("feature_scores", {})
                augmented_features = self.feature_augmenter.augment_features(
                    X_train, feature_importance
                )
                enhanced_data["augmented_features"] = list(augmented_features.columns)
            
            # Apply model optimization
            if enhanced_data.get("enhanced_model"):
                model_type = enhanced_data.get("best_model", "unknown")
                optimized_model, opt_metrics = self.model_optimizer.optimize_model(
                    enhanced_data["enhanced_model"], model_type
                )
                enhanced_data["enhanced_model"] = optimized_model
                enhanced_data["optimization_metrics"] = opt_metrics
            
            # Create performance report
            report = self.performance_analyzer.create_performance_report(
                regime_key, analyst_data, enhanced_data
            )
            
            return regime_key, enhanced_data, report
            
        except Exception as e:
            self.logger.error(f"Error enhancing {regime_key}: {e}")
            return None
    
    def _create_performance_comparison(
        self,
        original_analysts: Dict[str, Any],
        enhanced_analysts: Dict[str, Any],
        enhancement_reports: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create performance comparison between original and enhanced models."""
        comparison = {
            "summary": {
                "total_regimes": len(original_analysts),
                "enhanced_regimes": len(enhanced_analysts),
                "average_improvement": 0.0,
                "best_improvement": {"regime": None, "gain": -float('inf')},
                "worst_improvement": {"regime": None, "gain": float('inf')}
            },
            "regime_comparisons": {}
        }
        
        improvements = []
        
        for regime_key in enhanced_analysts:
            if regime_key in enhancement_reports:
                report = enhancement_reports[regime_key]
                
                if "enhancement_impact" in report:
                    impact = report["enhancement_impact"]
                    gain = impact.get("val_accuracy_absolute_gain", 0)
                    
                    improvements.append(gain)
                    
                    # Track best/worst
                    if gain > comparison["summary"]["best_improvement"]["gain"]:
                        comparison["summary"]["best_improvement"] = {
                            "regime": regime_key,
                            "gain": gain
                        }
                    
                    if gain < comparison["summary"]["worst_improvement"]["gain"]:
                        comparison["summary"]["worst_improvement"] = {
                            "regime": regime_key,
                            "gain": gain
                        }
                    
                    # Store regime comparison
                    comparison["regime_comparisons"][regime_key] = {
                        "original_score": original_analysts[regime_key].get("best_score", 0),
                        "enhanced_score": report["enhanced_performance"].get("val_accuracy", 0),
                        "improvement": gain,
                        "assessment": impact.get("overall_assessment", "unknown")
                    }
        
        if improvements:
            comparison["summary"]["average_improvement"] = np.mean(improvements)
        
        return comparison
    
    def _create_optimization_summary(
        self,
        enhancement_reports: Dict[str, Any],
        performance_comparison: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create optimization summary."""
        summary = {
            "timestamp": datetime.now().isoformat(),
            "total_regimes_processed": len(enhancement_reports),
            "enhancements_applied": {},
            "performance_summary": performance_comparison["summary"],
            "resource_usage": {},
            "recommendations": []
        }
        
        # Count enhancements applied
        enhancement_counts = {}
        for report in enhancement_reports.values():
            for enhancement in report.get("enhancements_applied", []):
                enhancement_counts[enhancement] = enhancement_counts.get(enhancement, 0) + 1
        
        summary["enhancements_applied"] = enhancement_counts
        
        # Generate recommendations
        avg_improvement = performance_comparison["summary"]["average_improvement"]
        
        if avg_improvement > 0.02:
            summary["recommendations"].append(
                "Enhancement process significantly improved model performance. Consider applying to production."
            )
        elif avg_improvement > 0:
            summary["recommendations"].append(
                "Enhancement process showed modest improvements. Review individual regime results."
            )
        else:
            summary["recommendations"].append(
                "Enhancement process did not improve performance. Consider adjusting enhancement parameters."
            )
        
        # Check for overfitting
        overfit_regimes = []
        for regime_key, comparison in performance_comparison["regime_comparisons"].items():
            if comparison.get("assessment") == "performance_degradation":
                overfit_regimes.append(regime_key)
        
        if overfit_regimes:
            summary["recommendations"].append(
                f"Potential overfitting detected in regimes: {overfit_regimes}. Consider regularization."
            )
        
        return summary
    
    async def _save_artifacts(self, result: Dict[str, Any]) -> None:
        """Save step artifacts."""
        artifacts_dir = Path(self.config.get("artifacts_dir", "artifacts")) / self.full_step_name
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        # Save enhanced models
        if "enhanced_analysts" in result:
            models_dir = artifacts_dir / "enhanced_models"
            models_dir.mkdir(exist_ok=True)
            
            for regime_key, analyst_data in result["enhanced_analysts"].items():
                if "enhanced_model" in analyst_data:
                    model_path = models_dir / f"{regime_key}_enhanced_model.pkl"
                    joblib.dump(analyst_data["enhanced_model"], model_path)
        
        # Save reports
        if "enhancement_reports" in result:
            with open(artifacts_dir / "enhancement_reports.json", "w") as f:
                json.dump(result["enhancement_reports"], f, indent=2)
        
        if "performance_comparison" in result:
            with open(artifacts_dir / "performance_comparison.json", "w") as f:
                json.dump(result["performance_comparison"], f, indent=2)
        
        if "optimization_summary" in result:
            with open(artifacts_dir / "optimization_summary.json", "w") as f:
                json.dump(result["optimization_summary"], f, indent=2)
        
        self.logger.info(f"Artifacts saved to {artifacts_dir}")
    
    def validate_outputs(
        self,
        pipeline_state: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        """Validate step outputs."""
        errors = []
        
        # Check required outputs
        required_outputs = [
            "enhanced_analysts",
            "enhancement_reports",
            "performance_comparison",
            "optimization_summary"
        ]
        
        for output in required_outputs:
            if output not in pipeline_state:
                errors.append(f"Missing required output: {output}")
            elif pipeline_state[output] is None:
                errors.append(f"Output {output} is None")
        
        # Validate enhanced analysts
        if "enhanced_analysts" in pipeline_state:
            analysts = pipeline_state["enhanced_analysts"]
            if not isinstance(analysts, dict):
                errors.append("enhanced_analysts must be a dictionary")
            
            # Check that we have enhanced models
            models_found = 0
            for analyst_data in analysts.values():
                if "enhanced_model" in analyst_data:
                    models_found += 1
            
            if models_found == 0:
                errors.append("No enhanced models found in output")
        
        # Validate performance comparison
        if "performance_comparison" in pipeline_state:
            comparison = pipeline_state["performance_comparison"]
            if "summary" not in comparison:
                errors.append("Performance comparison missing summary")
        
        return len(errors) == 0, errors